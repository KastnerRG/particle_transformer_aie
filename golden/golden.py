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

    # Generate weights & intermediate input/output matrices
    k_tiled = tile_matrix(layer["k"], k, n)
    np.savetxt(f"data/k{idx}.txt", layer["k"], fmt="%d")
    array_str = ', '.join(str(x) for x in k_tiled)
    # with open("aie/weights.h", 'a') as f:
    #     f.write(f"#include <cstdint>\n__attribute__((section(\".data\"))) const int8_t k{idx} [{k_tiled.size}] = {{ {array_str} }};\n")

    x_tiled = tile_matrix(layer["x"], m, k)
    a_tiled = tile_matrix(layer["a"], m, n)
    np.savetxt(f"data/x{idx}.txt", np.tile(x_tiled, (iterations, 1)).reshape(-1, 16), fmt="%s", delimiter=" ")
    np.savetxt(f"data/a{idx}.txt", np.tile(a_tiled, (iterations, 1)).reshape(-1, 16), fmt="%s", delimiter=" ")

    t_m = layer['x'].shape[0] // m
    t_k = layer['x'].shape[1] // k
    t_n = layer['k'].shape[1] // n
    shift = layer['shift']
    is_relu = str(layer['is_relu']).lower()

    #define DO_RELU {str(self.relu).lower()}
    # aie/layer_{idx}.cc
    with open(f"aie/layer_{idx}.cc", "a") as f:
        f.write(f'''
#include <cstdint>
__attribute__((section(".data"))) alignas(32) int8_t k_p [{k_tiled.size}] = {{ {", ".join(str(int(g)) for g in k_tiled)} }};

#include "kernels.h"

void f{idx}(input_window_int8 * __restrict x, output_window_int8 * __restrict a){{ dense<{m}, {k}, {n}, {t_m}, {t_k}, {t_n}, {shift}, {is_relu}> (x, a, k_p);}}
''')
    
    # with open(f"data/x{idx}.txt", "ab") as f:
    #     np.savetxt(f, np.zeros((4, 16), dtype=int), fmt="%s", delimiter=" ")

    # model.cc - each layer as function



    # with open("aie/model.cc", "a") as f:
    #     f.write(f"void f{idx}(input_window_int8* __restrict x, output_window_int8 * __restrict a) ")
    #     f.write(f"{{ dense<{m}, {k}, {n}, {t_m}, {t_k}, {t_n}, {shift}, {is_relu}> (x, a, k{idx}); }}\n")

    # model.h - Function prototypes

    # with open("aie/model.h", "a") as f:
    #     f.write(f"void f{idx}( input_window_int8  * __restrict, output_window_int8 * __restrict);\n")

    # layer_graph.h - create and connect layers

    num_bytes = layer['x'].size * layer['x'].itemsize
    in_port = "AIE_IN" if idx == 0 else f"layers[{idx-1}]"

    with open("aie/layer_graph.h", "a") as f:
        f.write(f"layers[{idx}] = kernel::create(f{idx});\n")
        f.write(f'source(layers[{idx}]) = "layer_{idx}.cc";\n')
        f.write(f'runtime<ratio>(layers[{idx}]) = 1.0;\n')
        f.write(f"connect<window<{num_bytes}>>({in_port}.out[0], layers[{idx}].in[0]);\n\n")
        if idx == 0 and num_bytes > 32768:
            f.write(f"single_buffer(layers[{idx}].in[0]);\n")

def print_layers_brief(layers):
    for i, L in enumerate(layers):
        name = L.get('name', f'dense{i}')
        xsh = tuple(L['x'].shape)
        ksh = tuple(L['k'].shape)
        sh  = L['shift']
        act = 'ReLU' if L['is_relu'] else 'Linear'
        print(f"{i:02d} {name:12s}  x{ xsh }  @  k{ ksh }  -> shift={sh}  act={act}")

def golden_fc(x, k, is_relu, shift):
    y = np.matmul(x.astype(np.int32), k.astype(np.int32))
    y = (y >> shift).astype(np.int8)
    a = np.maximum(0, y) if is_relu else y
    layer_fc =  [{'x': x, 'k': k, 'y': y, 'a': a, 'shift': shift, 'is_relu': is_relu}]
    return a, layer_fc 

if __name__ == "__main__":
    ################################################## PYTHON REFERENCE ##################################################
    layers = []
    # out_dim = 5

    # Keras-like dims
    in_particles, num_feature, ff_dim = 150, 3, 64
    num_feature_pad = 8   # pad to multiple of 8 for AIE tiling

    ###################################################################### below is layer 1
    # ---- Input + padding (3 -> 32) ----
    dummy_inp = np.random.randint(-128, 128, size=(in_particles, num_feature), dtype=np.int8)
    pad_inp   = np.zeros((in_particles, num_feature_pad), dtype=np.int8)
    pad_inp[:, :num_feature] = dummy_inp

    # ---- Dense to reach MHA width: (150,32) · (32,64) -> (150,64) ----
    W_fc1 = np.random.randint(-128, 128, size=(num_feature_pad, ff_dim), dtype=np.int8)
    a1, L1 = golden_fc(pad_inp, W_fc1, is_relu=True, shift=2)
    layers += L1

    # # ---- MHA 1 + residual ----
    # numheads = 4
    # mha1 = NumpyMHALinear(d_model=ff_dim, num_heads=numheads, name_prefix="mha1", seed=0)
    # att1 = mha1(a1, a1, a1, layers=layers)      # self-attention: Q=K=V=a1
    # a2   = residual_add_int8(att1, a1)          # Add()([x, emb])

    # ###################################################################### below is layer 2
    # # ---- Two dense (FF) + residual (Block 1) ----
    # W_ff1a = np.random.randint(-128, 128, size=(ff_dim, ff_dim), dtype=np.int8)
    # h1, L2a = golden_fc(a2, W_ff1a, is_relu=True, shift=3); layers += L2a

    # W_ff1b = np.random.randint(-128, 128, size=(ff_dim, ff_dim), dtype=np.int8)
    # h2, L2b = golden_fc(h1, W_ff1b, is_relu=True, shift=3); layers += L2b

    # lo1 = residual_add_int8(h2, a2)

    # # ---- MHA 2 + residual ----
    # mha2 = NumpyMHALinear(d_model=ff_dim, num_heads=numheads, name_prefix="mha2", seed=1)
    # att2 = mha2(lo1, lo1, lo1, layers=layers)
    # a3   = residual_add_int8(att2, lo1)
    # ####################################################################### below is layer 3
    # # ---- Two dense (FF) + residual (Block 2) ----
    # W_ff2a = np.random.randint(-128, 128, size=(ff_dim, ff_dim), dtype=np.int8)
    # h3, L3a = golden_fc(a3, W_ff2a, is_relu=True, shift=3); layers += L3a

    # W_ff2b = np.random.randint(-128, 128, size=(ff_dim, ff_dim), dtype=np.int8)
    # h4, L3b = golden_fc(h3, W_ff2b, is_relu=True, shift=3); layers += L3b

    # lo2 = residual_add_int8(h4, a3)

    
    # ####################################################################### below is layer 4 - out
    # # ---- “Pooling” skipped (as you noted). Two final dense to 5 logits ----
    # # pooling:     x = keras.layers.GlobalAveragePooling1D(data_format='channels_last')(lo2)
    # W_out1 = np.random.randint(-128, 128, size=(ff_dim, ff_dim), dtype=np.int8)
    # z,  L4a = golden_fc(lo2, W_out1, is_relu=True,  shift=3); layers += L4a

    # W_out2 = np.random.randint(-128, 128, size=(ff_dim, out_dim), dtype=np.int8)
    # out, L4b = golden_fc(z, W_out2, is_relu=False, shift=3); layers += L4b

    ################################################## PYTHON REFERENCE END #################################################  




    m, k, n = 2,8,8 # k==n such that output matrix can be fed as input without re-tiling
    iterations = 1

    # 0. Do a cleanup

    for path in [
        "data", "aie/layer_graph.h", "aie/include.h", "aie/model.cc", "aie/model.h",
        "aie/weights.h", "aie/layer_0.cc", "aie/layer_1.cc", 
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
    
    os.makedirs("data")

    # 1. include.h - common parameters
    
    with open("aie/include.h", "w") as f:
        f.write(f'#define N_LAYERS {len(layers)}\n#define ITERATIONS {iterations}\n')
        for idx in range(len(layers)):
                f.write(f'void f{idx}(input_window_int8 * __restrict, output_window_int8 * __restrict);\n')

    # 2. Preamble of model.cc - each layer as function

    # with open("aie/model.cc", "w") as f:
    #     f.write('#include "kernels.h"\n#include "weights.h"\n')


    # 3. Process each layer: write weights.h, x.txt, a.txt, model.cc, model.h, layer_graph.h

    for i, layer in enumerate(layers):
        process_layer(i, layer, m, k, n, iterations)
    
    tiled_mat = tile_matrix(layers[-1]['a'], m, n)
    np.savetxt("data/out_ref.txt", np.tile(tiled_mat, (iterations, 1)).reshape(-1, 16), fmt="%s", delimiter=" ")

    # 4. Postamble of layer_graph.h - connect last layer to AIE_OUT
    
    out_bytes = layers[-1]['a'].size * layers[-1]['a'].itemsize
    with open("aie/layer_graph.h", "a") as f:
        f.write(f"connect<window<{out_bytes:>5}>>(layers[{len(layers)-1}].out[0], AIE_OUT.in[0]);\n")    

    # 5. Run AIE (graph.cpp)

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
