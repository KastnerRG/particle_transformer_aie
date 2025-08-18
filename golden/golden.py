import numpy as np
import os, glob, shutil, subprocess
from utils.np_mha_linear import NumpyMHALinear, residual_add_int8


def tile_matrix(matrix, row_tiles, col_tiles): # (R,C) -> (R/r, C/c, r, c).flatten()
    rows, cols = matrix.shape
    assert rows % row_tiles == 0 and cols % col_tiles == 0, "Matrix must be divisible by block sizes"
    reshaped = matrix.reshape(rows // row_tiles, row_tiles, cols // col_tiles, col_tiles)
    transposed = reshaped.transpose(0, 2, 1, 3) # (R/r, C/c, r, c)
    return transposed.flatten()

def process_layer(idx, layer, m, k, n, iterations):
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

        # model.cc - each layer as function
        t_m = layer['x'].shape[0] // m
        t_k = layer['x'].shape[1] // k
        t_n = k_matrix.shape[1] // n
        is_relu_str = str(is_relu).lower()
        """
        with open("aie/model.cc", "a") as f:
            f.write(f"void f{idx}(input_window_int8* __restrict x, output_window_int8 * __restrict a) ")
            f.write(f"{{ dense<{m}, {k}, {n}, {t_m}, {t_k}, {t_n}, {shift}, {is_relu_str}> (x, a, k{idx}); }}\n")
        
        # model.h - Function prototypes
        with open("aie/model.h", "a") as f:
            f.write(f"void f{idx}( input_window_int8  * __restrict, output_window_int8 * __restrict);\n")
        """
        # layer_graph.h - create and connect layers
        num_bytes = layer['x'].size * layer['x'].itemsize
        in_port = "AIE_IN" if idx == 0 else f"layers[{idx-1}]"

        # Calculate tiling parameters
        t_m = layer['x'].shape[0] // m
        t_k = layer['x'].shape[1] // k
        t_n = k_matrix.shape[1] // n

        with open("aie/layer_graph.h", "a") as f:
            if layer_type == 'dense':
                f.write(f"// Create DenseGraph for layer {idx}\n")
                f.write(f"DenseGraph<{m}, {k}, {n}, {t_m}, {t_k}, {t_n}, {shift}, {is_relu_str}> ")
                f.write(f"dense_graph_{idx}(k{idx});\n")
                f.write(f"layers[{idx}] = &dense_graph_{idx};\n")
                f.write(f"connect<window<{num_bytes:>5}>>({in_port}.out[0], dense_graph_{idx}.in);\n\n")
            # Dense layer processing ends here
            
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
        
        # layer_graph.h - create and connect layers
        num_bytes = x.size * x.itemsize
        in_port = "AIE_IN" if idx == 0 else f"layers[{idx-1}]"
        residual_port = f"layers[{residual_idx}]"
        
        # Calculate tiling parameters for residual
        t_m = x.shape[0] // m
        t_n = x.shape[1] // n
        
        with open("aie/layer_graph.h", "a") as f:
            f.write(f"// Create ResidualGraph for layer {idx} with residual from layer {residual_idx}\n")
            f.write(f"ResidualGraph<{m}, {n}, {t_m}, {t_n}> ")
            f.write(f"residual_graph_{idx};\n")
            f.write(f"layers[{idx}] = &residual_graph_{idx};\n")
            f.write(f"connect<window<{num_bytes:>5}>>({in_port}.out[0], residual_graph_{idx}.in[0]);\n")
            
            # Connect residual input
            residual_bytes = residual.size * residual.itemsize
            f.write(f"connect<window<{residual_bytes:>5}>>({residual_port}.out[0], residual_graph_{idx}.in[1]);\n\n")
    
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
        m = config.get('m', m)
        k = config.get('k', k)
        n = config.get('n', n)
        
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
        
        # Declare MHAGraph in layer_graph.h
        with open("aie/layer_graph.h", "a") as f:
            num_bytes = layer['x'].size * layer['x'].itemsize
            in_port = "AIE_IN" if idx == 0 else f"layers[{idx-1}]"
            
            # Create MHAGraph instance
            f.write(f"// Create MHAGraph for layer {idx}\n")
            f.write(f"MHAGraph<{m}, {k}, {n}, {num_heads}, {d_model}, {shift_q}, {shift_k}, {shift_v}, {shift_o}, {shift_s}, {shift_c}> ")
            f.write(f"mha_graph_{idx}(Wq{idx}, Wk{idx}, Wv{idx}, Wo{idx});\n")
            
            # Store in layers array
            f.write(f"layers[{idx}] = &mha_graph_{idx};\n")
            
            # Connect input to MHAGraph input port
            f.write(f"connect<window<{num_bytes:>5}>>({in_port}.out[0], mha_graph_{idx}.in);\n\n")
        
    else:
        raise ValueError(f"Unknown layer type: {layer_type}")

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
    
    Args:
        x: Input tensor (int8)
        k: Weight matrix (int8)
        is_relu: Whether to apply ReLU activation
        shift: Right shift amount for quantization
        
    Returns:
        Tuple of (output tensor, layer info)
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
        "data", "aie/include.h", "aie/weights.h", "aie/layer_graph.h", "aie/model.cc", "aie/model.h",
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

    # 1. include.h - common parameters
    
    with open("aie/include.h", "w") as f:
        f.write(f'#define N_LAYERS {len(layers)}\n#define ITERATIONS {iterations}')

    # 3. Process each layer: write weights.h, x.txt, a.txt, model.cc, model.h, layer_graph.h

    for i, layer in enumerate(layers):
        process_layer(i, layer, m, k, n, iterations)
    
    tiled_mat = tile_matrix(layers[-1]['a'], m, n)
    np.savetxt("data/out_ref.txt", np.tile(tiled_mat, (iterations, 1)).reshape(-1, 16), fmt="%s", delimiter=" ")

    # 4. Postamble of layer_graph.h - connect last layer to AIE_OUT
    
    with open("aie/layer_graph.h", "a") as f:
        f.write(f"connect<window<{layers[-1]['a'].size * layers[-1]['a'].itemsize:>5}>>(layers[{len(layers)-1}].out[0], AIE_OUT.in[0]);\n")
    
"""
    # 5. Run AIE

    subprocess.run(["./run.sh"], check=True)

    # 6. Verify output

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