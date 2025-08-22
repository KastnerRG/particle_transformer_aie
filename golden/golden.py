import numpy as np
import os, glob, shutil, subprocess
from utils.np_mha_linear import NumpyMHALinear, residual_add_int8


def tile_matrix(matrix, row_tiles, col_tiles): # (R,C) -> (R/r, C/c, r, c).flatten()
    rows, cols = matrix.shape
    assert rows % row_tiles == 0 and cols % col_tiles == 0, "Matrix must be divisible by block sizes"
    reshaped = matrix.reshape(rows // row_tiles, row_tiles, cols // col_tiles, col_tiles)
    transposed = reshaped.transpose(0, 2, 1, 3) # (R/r, C/c, r, c)
    return transposed.flatten()

def process_layer(idx, layer, m, k, n, iterations, graph_cpp_content=None, class_members=None, layer_connections=None):
    """Process a layer: generate graph.cpp code and write weights/data files.
    
    Args:
        idx: Layer index
        layer: Layer dictionary with type, x, a, config, etc.
        m, k, n: Tiling parameters
        iterations: Number of iterations
        graph_cpp_content: Optional list to append graph.cpp code to
        class_members: Optional list to append class member declarations
        layer_connections: Optional list to append layer connection code
    """
    layer_type = layer['type']
    
    if layer_type == 'dense':
        # Extract config parameters
        config = layer['config']
        k_matrix = config['k']
        shift = config['shift']
        is_relu = config['is_relu']
        
        # Generate weights & intermediate input/output matrices
        k_tiled = tile_matrix(k_matrix, k, n)
        np.savetxt(f"data/k{idx}.txt", k_matrix, fmt="%d")
        array_str = ', '.join(str(x) for x in k_tiled)
        with open("aie/weights.h", 'a') as f:
            f.write(f"const int8_t k{idx} [{k_tiled.size}] = {{ {array_str} }};\n")

        x_tiled = tile_matrix(layer["x"], m, k)
        a_tiled = tile_matrix(layer["a"], m, n)
        np.savetxt(f"data/x{idx}.txt", np.tile(x_tiled, (iterations, 1)).reshape(-1, 16), fmt="%s", delimiter=" ")
        np.savetxt(f"data/a{idx}.txt", np.tile(a_tiled, (iterations, 1)).reshape(-1, 16), fmt="%s", delimiter=" ")
        
        # Calculate tiling parameters
        t_m = layer['x'].shape[0] // m
        t_k = layer['x'].shape[1] // k
        t_n = k_matrix.shape[1] // n
        is_relu_str = str(is_relu).lower()
        
        # Generate code for graph.cpp if requested
        if graph_cpp_content is not None:
            # Add class member declaration
            if class_members is not None:
                class_members.append(f"  // Dense graph for layer {idx}")
                class_members.append(f"  DenseGraph<{m}, {k}, {n}, {t_m}, {t_k}, {t_n}, {shift}, {is_relu_str}> dense_graph_{idx} {{ k{idx} }};")
            
            # Add layer pointer assignment
            graph_cpp_content.append(f"    // Assign layer {idx} pointer")
            graph_cpp_content.append(f"    layers[{idx}] = &dense_graph_{idx};") 
            
            # Add connection code
            if layer_connections is not None:
                num_bytes = layer['x'].size * layer['x'].itemsize
                if idx == 0:
                    layer_connections.append(f"    // Connect input to first layer")
                    layer_connections.append(f"    connect<window<{num_bytes:>5}>>(AIE_IN.out[0], dense_graph_{idx}.in);")
                else:
                    layer_connections.append(f"    // Connect layer {idx-1} to layer {idx}")
                    layer_connections.append(f"    connect<window<{num_bytes:>5}>>(layers[{idx-1}]->out[0], dense_graph_{idx}.in);")
                layer_connections.append("")
            
    elif layer_type == 'residual':
        # Extract input and residual tensors
        x = layer['x']
        residual = layer['residual']
        
        # Save input/output data for testing
        x_tiled = tile_matrix(x, m, n)  # Using m,n since both tensors should have same shape
        a_tiled = tile_matrix(layer["a"], m, n)
        np.savetxt(f"data/x{idx}.txt", np.tile(x_tiled, (iterations, 1)).reshape(-1, 16), fmt="%s", delimiter=" ")
        np.savetxt(f"data/a{idx}.txt", np.tile(a_tiled, (iterations, 1)).reshape(-1, 16), fmt="%s", delimiter=" ")
        
        # Find the previous layer that produced the residual input
        residual_idx = None
        for i in range(idx):
            if i < len(layers) and np.array_equal(layers[i]['a'], residual):
                residual_idx = i
                break
        
        if residual_idx is None:
            raise ValueError(f"Could not find source layer for residual connection in layer {idx}")
        
        # Calculate tiling parameters for residual
        t_m = x.shape[0] // m
        t_n = x.shape[1] // n
        
        # Generate code for graph.cpp if requested
        if graph_cpp_content is not None:
            # Add class member declaration
            if class_members is not None:
                class_members.append(f"  // Residual graph for layer {idx} with residual from layer {residual_idx}")
                class_members.append(f"  ResidualGraph<{m}, {n}, {t_m}, {t_n}> residual_graph_{idx};") 
            
            # Add layer pointer assignment
            graph_cpp_content.append(f"    // Assign layer {idx} pointer")
            graph_cpp_content.append(f"    layers[{idx}] = &residual_graph_{idx};") 
            
            # Add connection code
            if layer_connections is not None:
                # Connect main input
                num_bytes = x.size * x.itemsize
                if idx == 0:
                    layer_connections.append(f"    // Connect input to first layer")
                    layer_connections.append(f"    connect<window<{num_bytes:>5}>>(AIE_IN.out[0], residual_graph_{idx}.in1);") 
                else:
                    layer_connections.append(f"    // Connect layer {idx-1} to layer {idx}")
                    layer_connections.append(f"    connect<window<{num_bytes:>5}>>(layers[{idx-1}]->out[0], residual_graph_{idx}.in1);") 
                
                # Connect residual input
                residual_bytes = residual.size * residual.itemsize
                layer_connections.append(f"    // Connect residual from layer {residual_idx} to layer {idx}")
                layer_connections.append(f"    connect<window<{residual_bytes:>5}>>(layers[{residual_idx}]->out[0], residual_graph_{idx}.in2);") 
                layer_connections.append("")
    
    elif layer_type == 'mha':
        # Extract config parameters
        config = layer['config']
        num_heads = config['num_heads']
        d_model = config['d_model']
        
        # Extract weight matrices
        Wq = config['Wq']
        Wk = config['Wk']
        Wv = config['Wv']
        Wo = config['Wo']
        
        # Extract shift values
        shift_q = config['shift_q']
        shift_k = config['shift_k']
        shift_v = config['shift_v']
        shift_o = config['shift_o']
        shift_s = config['shift_s']
        shift_c = config['shift_c']
        
        # Extract tiling parameters if available
        m_local = config.get('m', m)
        k_local = config.get('k', k)
        n_local = config.get('n', n)
        
        # Generate weights files for all matrices
        for name, matrix in [
            (f"Wq{idx}", Wq),
            (f"Wk{idx}", Wk),
            (f"Wv{idx}", Wv),
            (f"Wo{idx}", Wo)
        ]:
            # Tile and save weights
            matrix_tiled = tile_matrix(matrix, k, n)
            np.savetxt(f"data/{name}.txt", matrix, fmt="%d")
            array_str = ', '.join(str(x) for x in matrix_tiled)
            with open("aie/weights.h", 'a') as f:
                f.write(f"const int8_t {name} [{matrix_tiled.size}] = {{ {array_str} }};\n")
        
        # Save input and output data
        x_tiled = tile_matrix(layer["x"], m, k)
        a_tiled = tile_matrix(layer["a"], m, n)
        np.savetxt(f"data/x{idx}.txt", np.tile(x_tiled, (iterations, 1)).reshape(-1, 16), fmt="%s", delimiter=" ")
        np.savetxt(f"data/a{idx}.txt", np.tile(a_tiled, (iterations, 1)).reshape(-1, 16), fmt="%s", delimiter=" ")
        
        # Calculate tensor dimensions based on input shape
        B, T, C = 1, layer['x'].shape[0], layer['x'].shape[1]  # Assuming 2D input (T,C)
        if layer['x'].ndim == 3:
            B, T, C = layer['x'].shape
        
        # Calculate tiling parameters
        t_m = layer['x'].shape[0] // m
        t_k = layer['x'].shape[1] // k
        t_n = d_model // n
        
        # Generate code for graph.cpp if requested
        if graph_cpp_content is not None:
            # Add class member declaration
            if class_members is not None:
                class_members.append(f"  // MHA graph for layer {idx}")
                class_members.append(f"  MHAGraph<{m}, {k}, {n}, {num_heads}, {d_model}, {shift_q}, {shift_k}, {shift_v}, {shift_o}, {shift_s}, {shift_c}> mha_graph_{idx} {{ Wq{idx}, Wk{idx}, Wv{idx}, Wo{idx} }};")
            
            # Add layer pointer assignment
            graph_cpp_content.append(f"    // Assign layer {idx} pointer")
            graph_cpp_content.append(f"    layers[{idx}] = &mha_graph_{idx};") 
            
            # Add connection code
            if layer_connections is not None:
                num_bytes = layer['x'].size * layer['x'].itemsize
                if idx == 0:
                    layer_connections.append(f"    // Connect input to first layer")
                    layer_connections.append(f"    connect<window<{num_bytes:>5}>>(AIE_IN.out[0], mha_graph_{idx}.in);") 
                else:
                    layer_connections.append(f"    // Connect layer {idx-1} to layer {idx}")
                    layer_connections.append(f"    connect<window<{num_bytes:>5}>>(layers[{idx-1}]->out[0], mha_graph_{idx}.in);") 
                layer_connections.append("")
    
    else:
        raise ValueError(f"Unknown layer type: {layer_type}")

def generate_graph_cpp(layers, m, k, n, iterations):
    """Write input/output data files. Generate graph.cpp code
    
    Args:
        layers: List of layer dictionaries
        m, k, n: Tiling parameters
        iterations: Number of iterations
        
    Returns:
        String containing the complete graph.cpp file content
    """
    # Start with includes and class definition
    graph_cpp_content = [
        "#include <adf.h>",
        "#include \"include.h\"",
        "#include <vector>",
        "#include \"model.h\"",
        "#include \"dense_graph.h\"",
        "#include \"mha_graph.h\"",
        "#include \"residual_graph.h\"",
        "#include \"weights.h\"",
        "",
        "using namespace adf;",
        "",
        "class mainGraph : public adf::graph {",
        "private:"
    ]
    
    # Class member declarations for graph objects
    class_members = []
    
    # Process each layer to generate class member declarations
    for i, layer in enumerate(layers):
        layer_type = layer['type']
        
        if layer_type == 'dense':
            # Extract config parameters
            config = layer['config']
            k_matrix = config['k']
            shift = config['shift']
            is_relu = config['is_relu']
            is_relu_str = str(is_relu).lower()
            
            # Calculate tiling parameters
            t_m = layer['x'].shape[0] // m
            t_k = layer['x'].shape[1] // k
            t_n = k_matrix.shape[1] // n
            
            # Add class member declaration
            class_members.append(f"  // Dense graph for layer {i}")
            class_members.append(f"  DenseGraph<{m}, {k}, {n}, {t_m}, {t_k}, {t_n}, {shift}, {is_relu_str}> dense_graph_{i} {{ k{i} }};")
            
        elif layer_type == 'residual':
            # Find residual source layer
            residual_idx = None
            for j in range(i):
                if np.array_equal(layers[j]['a'], layer['residual']):
                    residual_idx = j
                    break
            
            if residual_idx is None:
                raise ValueError(f"Could not find source layer for residual connection in layer {i}")
            
            # Calculate tiling parameters
            t_m = layer['x'].shape[0] // m
            t_n = layer['x'].shape[1] // n
            
            # Add class member declaration
            class_members.append(f"  // Residual graph for layer {i} with residual from layer {residual_idx}")
            class_members.append(f"  ResidualGraph<{m}, {n}, {t_m}, {t_n}> residual_graph_{i};") 
            
        elif layer_type == 'mha':
            # Extract config parameters
            config = layer['config']
            num_heads = config['num_heads']
            d_model = config['d_model']
            shift_q = config['shift_q']
            shift_k = config['shift_k']
            shift_v = config['shift_v']
            shift_o = config['shift_o']
            shift_s = config['shift_s']
            shift_c = config['shift_c']
            
            # Calculate tiling parameters
            t_m = layer['x'].shape[0] // m
            t_k = layer['x'].shape[1] // k
            t_n = d_model // n
            
            # Add class member declaration
            class_members.append(f"  // MHA graph for layer {i}")
            class_members.append(f"  MHAGraph<{m}, {k}, {n}, {num_heads}, {d_model}, {shift_q}, {shift_k}, {shift_v}, {shift_o}, {shift_s}, {shift_c}> mha_graph_{i} {{ Wq{i}, Wk{i}, Wv{i}, Wo{i} }};")
    
    # Add layers array after graph declarations
    graph_cpp_content.extend(class_members)
    graph_cpp_content.append("  graph* layers [N_LAYERS];") 
    graph_cpp_content.append("")
    graph_cpp_content.append("public:")
    graph_cpp_content.append("  input_plio  AIE_IN;")
    graph_cpp_content.append("  output_plio AIE_OUT;")
    graph_cpp_content.append("")
    graph_cpp_content.append("  mainGraph(){")
    graph_cpp_content.append("")
    graph_cpp_content.append("    AIE_IN = input_plio::create(plio_128_bits, \"data/x0.txt\");")
    graph_cpp_content.append("    AIE_OUT = output_plio::create(plio_128_bits, \"data/out_sim.txt\");")
    graph_cpp_content.append("")
    
    # Process each layer - this will write weights.h and data files and add layer assignments
    for i, layer in enumerate(layers):
        # Process layer to write weights and data files
        process_layer(i, layer, m, k, n, iterations)
        
        # Add layer pointer assignment
        layer_type = layer['type']
        if layer_type == 'dense':
            graph_cpp_content.append(f"    // Assign layer {i} pointer")
            graph_cpp_content.append(f"    layers[{i}] = &dense_graph_{i};") 
        elif layer_type == 'residual':
            graph_cpp_content.append(f"    // Assign layer {i} pointer")
            graph_cpp_content.append(f"    layers[{i}] = &residual_graph_{i};") 
        elif layer_type == 'mha':
            graph_cpp_content.append(f"    // Assign layer {i} pointer")
            graph_cpp_content.append(f"    layers[{i}] = &mha_graph_{i};") 
        
        # Add connection code
        num_bytes = layer['x'].size * layer['x'].itemsize
        if i == 0:
            graph_cpp_content.append(f"    // Connect input to first layer")
            if layer_type == 'dense':
                graph_cpp_content.append(f"    connect<window<{num_bytes:>5}>>(AIE_IN.out[0], dense_graph_{i}.in);") 
            elif layer_type == 'residual':
                graph_cpp_content.append(f"    connect<window<{num_bytes:>5}>>(AIE_IN.out[0], residual_graph_{i}.in1);") 
            elif layer_type == 'mha':
                graph_cpp_content.append(f"    connect<window<{num_bytes:>5}>>(AIE_IN.out[0], mha_graph_{i}.in);") 
        else:
            graph_cpp_content.append(f"    // Connect layer {i-1} to layer {i}")
            if layer_type == 'dense':
                graph_cpp_content.append(f"    connect<window<{num_bytes:>5}>>(layers[{i-1}]->out[0], dense_graph_{i}.in);") 
            elif layer_type == 'residual':
                graph_cpp_content.append(f"    connect<window<{num_bytes:>5}>>(layers[{i-1}]->out[0], residual_graph_{i}.in1);") 
                
                # Find residual source layer
                residual_idx = None
                for j in range(i):
                    if np.array_equal(layers[j]['a'], layer['residual']):
                        residual_idx = j
                        break
                
                # Connect residual input
                residual_bytes = layer['residual'].size * layer['residual'].itemsize
                graph_cpp_content.append(f"    // Connect residual from layer {residual_idx} to layer {i}")
                graph_cpp_content.append(f"    connect<window<{residual_bytes:>5}>>(layers[{residual_idx}]->out[0], residual_graph_{i}.in2);") 
            elif layer_type == 'mha':
                graph_cpp_content.append(f"    connect<window<{num_bytes:>5}>>(layers[{i-1}]->out[0], mha_graph_{i}.in);") 
        
        graph_cpp_content.append("")
    
    # Connect the last layer to output
    graph_cpp_content.append(f"    // Connect last layer to output")
    graph_cpp_content.append(f"    connect<window<{layers[-1]['a'].size * layers[-1]['a'].itemsize:>5}>>(layers[{len(layers)-1}]->out[0], AIE_OUT.in[0]);")
    
    # Close the class and add main function
    graph_cpp_content.extend([
        "  }",
        "};",
        "",
        "mainGraph mygraph;",
        "",
        "int main(void) {",
        "  mygraph.init();",
        "  mygraph.run(ITERATIONS);",
        "  mygraph.end();",
        "  return 0;",
        "}",
        ""
    ])
    
    return "\n".join(graph_cpp_content)


def print_layers_brief(layers):
    for i, L in enumerate(layers):
        layer_type = L['type']
        name = L.get('name', f'{layer_type}{i}')
        xsh = tuple(L['x'].shape)
        
        if layer_type == 'dense':
            config = L['config']
            ksh = tuple(config['k'].shape)
            sh = config['shift']
            act = 'ReLU' if config['is_relu'] else 'Linear'
            print(f"{i:02d} {name:12s}  x{ xsh }  @  k{ ksh }  -> shift={sh}  act={act}")
        elif layer_type == 'mha':
            config = L['config']
            print(f"{i:02d} {name:12s}  x{ xsh }  @ MHA layer")
        elif layer_type == 'residual':
            residual_shape = tuple(L['residual'].shape)
            print(f"{i:02d} {name:12s}  x{ xsh }  +  residual{ residual_shape }")
        else:
            print(f"{i:02d} {name:12s}  x{ xsh }  @ Unknown layer type: {layer_type}")


def golden_fc(x, k, is_relu, shift):
    """Perform dense (fully connected) operation with int8 quantization.
    """
    y = np.matmul(x.astype(np.int32), k.astype(np.int32))
    y = (y >> shift).astype(np.int8)
    a = np.maximum(0, y) if is_relu else y
    
    layer_fc = [{
        'type': 'dense',
        'x': x,
        'a': a,
        'config': {
            'k': k,
            'y': y,
            'shift': shift,
            'is_relu': is_relu
        }
    }]
    return a, layer_fc 
   

if __name__ == "__main__":
    ################################################## PYTHON REFERENCE ##################################################
    layers = []
    out_dim = 8

    # Keras-like dims
    in_particles, num_feature, ff_dim = 150, 3, 64
    num_feature_pad = 32   # pad to multiple of 8 for AIE tiling

    ###################################################################### below is layer 1
    # ---- Input + padding (3 -> 32) ----
    dummy_inp = np.random.randint(-128, 128, size=(in_particles, num_feature), dtype=np.int8)
    pad_inp   = np.zeros((in_particles, num_feature_pad), dtype=np.int8)
    pad_inp[:, :num_feature] = dummy_inp

    # ---- Dense to reach MHA width: (150,32) · (32,64) -> (150,64) ----
    W_fc1 = np.random.randint(-128, 128, size=(num_feature_pad, ff_dim), dtype=np.int8)
    a1, L1 = golden_fc(pad_inp, W_fc1, is_relu=True, shift=3)
    layers += L1

    # ---- MHA 1 + residual ----
    numheads = 4
    mha1 = NumpyMHALinear(d_model=ff_dim, num_heads=numheads, name_prefix="mha1", seed=0)
    att1 = mha1(a1, a1, a1, layers=layers)      # self-attention: Q=K=V=a1
    a2   = residual_add_int8(att1, a1, layers=layers)  # Add()([x, emb])

    ###################################################################### below is layer 2
    # ---- Two dense (FF) + residual (Block 1) ----
    W_ff1a = np.random.randint(-128, 128, size=(ff_dim, ff_dim), dtype=np.int8)
    h1, L2a = golden_fc(a2, W_ff1a, is_relu=True, shift=3); layers += L2a

    W_ff1b = np.random.randint(-128, 128, size=(ff_dim, ff_dim), dtype=np.int8)
    h2, L2b = golden_fc(h1, W_ff1b, is_relu=True, shift=3); layers += L2b
    # Apply residual connection separately
    lo1 = residual_add_int8(h2, a2, layers=layers)

    # ---- MHA 2 + residual ----
    mha2 = NumpyMHALinear(d_model=ff_dim, num_heads=numheads, name_prefix="mha2", seed=1)
    att2 = mha2(lo1, lo1, lo1, layers=layers)
    a3   = residual_add_int8(att2, lo1, layers=layers)
    ####################################################################### below is layer 3
    # ---- Two dense (FF) + residual (Block 2) ----
    W_ff2a = np.random.randint(-128, 128, size=(ff_dim, ff_dim), dtype=np.int8)
    h3, L3a = golden_fc(a3, W_ff2a, is_relu=True, shift=3); layers += L3a

    W_ff2b = np.random.randint(-128, 128, size=(ff_dim, ff_dim), dtype=np.int8)
    h4, L3b = golden_fc(h3, W_ff2b, is_relu=True, shift=3); layers += L3b
    # Apply residual connection separately
    lo2 = residual_add_int8(h4, a3, layers=layers)

    
    ####################################################################### below is layer 4 - out
    # ---- “Pooling” skipped. Two final dense to 5 logits ----
    # pooling:     x = keras.layers.GlobalAveragePooling1D(data_format='channels_last')(lo2)
    W_out1 = np.random.randint(-128, 128, size=(ff_dim, ff_dim), dtype=np.int8)
    z,  L4a = golden_fc(lo2, W_out1, is_relu=True,  shift=3); layers += L4a

    W_out2 = np.random.randint(-128, 128, size=(ff_dim, out_dim), dtype=np.int8)
    out, L4b = golden_fc(z, W_out2, is_relu=False, shift=3); layers += L4b

    ################################################## PYTHON REFERENCE END #################################################  




    m, k, n = 2,8,8 # k==n such that output matrix can be fed as input without re-tiling
    iterations = 5

    # 0. Do a cleanup

    for path in [
        "data", "aie/graph.cpp", "aie/include.h", "aie/weights.h", "aie/layer_graph.h", "aie/model.cc", "aie/model.h",
        "*.log", "aiesimulator_output", "Work", ".Xil", 
        ".AIE_SIM_CMD_LINE_OPTIONS", "ISS_RPC_SERVER_PORT",
        "libadf.a", "Map_Report.csv", "pl_sample_counts",
        "plio_throughput_info.json", "sol.db", "aiesim.vcd"
    ]:
        for p in glob.glob(path):
            if os.path.exists(p):
                if os.path.isdir(p):
                    shutil.rmtree(p, ignore_errors=True)
                else:
                    os.remove(p)
    
    # Create directories only if they don't exist
    os.makedirs("data", exist_ok=True)
    os.makedirs("aie", exist_ok=True)

    # 1. Initialize include.h with common parameters
    with open("aie/include.h", "w") as f:
        f.write(f'#define N_LAYERS {len(layers)}\n#define ITERATIONS {iterations}')
        
    # 2. Initialize weights.h file with header
    with open("aie/weights.h", "w") as f:
        f.write("// Auto-generated weights for AIE implementation\n")
        f.write("#pragma once\n\n")

    # 3. Generate graph.cpp file directly
    # This also processes each layer and writes weights/data files in a single pass
    graph_cpp_content = generate_graph_cpp(layers, m, k, n, iterations)
    with open("aie/graph.cpp", "w") as f:
        f.write(graph_cpp_content)
        
    # 4. Write reference output for verification
    np.savetxt("data/out_ref.txt", layers[-1]['a'], fmt="%d")
    """
    # 4. Run AIE

    subprocess.run(["./run.sh"], check=True)

    # 5. Verify output

    aie_out_path = "aiesimulator_output/data/out_sim.txt"
    assert os.path.exists(aie_out_path), f"Error: Output file {aie_out_path} does not exist."

    with open(aie_out_path, "r") as infile, open("data/out_sim.txt", "w") as outfile:
        for line in infile:
            if not line.startswith("T"):
                outfile.write(line)

    out_sim = np.loadtxt("data/out_sim.txt").astype(np.int32)
    out_ref = np.loadtxt("data/out_ref.txt").astype(np.int32)

    if out_sim.shape == out_ref.shape and np.array_equal(out_sim, out_ref):
        print(f"\n\n Success: Outputs match ({out_sim.shape})\n\n{out_sim}\n\n")
    else:
        print("\n\nError: Output does not match\n")
        print(f"Simulation Output ({out_sim.shape}):\n{out_sim}\n")
        print(f"Expected output ({out_ref.shape}):\n{out_ref}\n")
    """