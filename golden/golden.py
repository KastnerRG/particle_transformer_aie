import numpy as np
import os, glob, shutil, subprocess
from utils.np_mha_linear import NumpyMHALinear, residual_add_int8


def tile_matrix(matrix, row_tiles, col_tiles): # (R,C) -> (R/r, C/c, r, c).flatten()
    rows, cols = matrix.shape
    assert rows % row_tiles == 0 and cols % col_tiles == 0, "Matrix must be divisible by block sizes"
    reshaped = matrix.reshape(rows // row_tiles, row_tiles, cols // col_tiles, col_tiles)
    transposed = reshaped.transpose(0, 2, 1, 3) # (R/r, C/c, r, c)
    return transposed.flatten()

def pad_rows(matrix, target_rows=160):
    rows, cols = matrix.shape
    if rows < target_rows:
        pad = np.zeros((target_rows - rows, cols), dtype=matrix.dtype)
        return np.vstack([matrix, pad])
    return matrix

def process_dense_layer(idx, layer, m, k, n, iterations):

    # Generate weights & intermediate input/output matrices
    k_tiled = tile_matrix(layer["k"], k, n)
    np.savetxt(f"data/k{idx}.txt", layer["k"], fmt="%d")
    # array_str = ', '.join(str(x) for x in k_tiled)
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
__attribute__((section(".data"))) alignas(32) int8_t k_p [{k_tiled.size}] = {{ {", ".join(str(int(x)) for x in k_tiled)} }};

#include "kernels.h"

void f{idx}(input_stream_int8 * __restrict x, output_stream_int8 * __restrict a){{ dense<{m}, {k}, {n}, {t_m}, {t_k}, {t_n}, {shift}, {is_relu}> (x, a, k_p);}}
''')


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
        f.write(f"connect<stream>({in_port}.out[0], layers[{idx}].in[0]);\n\n")
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

def process_mha_layer(idx, m, k, n, layer_q, layer_k, layer_v, layer_o, iterations):
     # Generate weights & intermediate input/output matrices
    Wq_tiled   = tile_matrix(pad_rows(layer_q["k"]), k, n)
    Wk_tiled   = tile_matrix(pad_rows(layer_k["k"]), k, n)
    Wv_tiled   = tile_matrix(pad_rows(layer_v["k"]), k, n)
    Wo_tiled   = tile_matrix(pad_rows(layer_o["k"]), k, n)
    out_x_tiled = tile_matrix(pad_rows(layer_o["x"]), m, k)

    np.savetxt(f"data/mha_Wq{idx}.txt", layer_q["k"], fmt="%d")
    np.savetxt(f"data/mha_Wk{idx}.txt", layer_k["k"], fmt="%d")
    np.savetxt(f"data/mha_Wv{idx}.txt", layer_v["k"], fmt="%d")
    np.savetxt(f"data/mha_Wo{idx}.txt", layer_o["k"], fmt="%d")
    np.savetxt(f"data/out_x{idx}.txt", np.tile(out_x_tiled, (iterations, 1)).reshape(-1, 16), fmt="%s", delimiter=" ")

    # pad input
    layer_q["x"] = pad_rows(layer_q["x"], target_rows=160)
    x_tiled = tile_matrix(layer_q["x"], m, k)
    # a_tiled = tile_matrix(layer_o["a"], m, n)
    np.savetxt(f"data/x{idx}.txt", np.tile(x_tiled, (iterations, 1)).reshape(-1, 16), fmt="%s", delimiter=" ")
    # np.savetxt(f"data/a{idx}.txt", np.tile(a_tiled, (iterations, 1)).reshape(-1, 16), fmt="%s", delimiter=" ")

    m = 2
    k = 8
    n = 8
    num_heads = 1
    d_model = 64
    T = 160 # first dimension of input to mha
    SHIFT_Q = 10
    SHIFT_K = 11
    SHIFT_V = 11
    SHIFT_S = 8
    SHIFT_C = 10
    SHIFT_O = 10 # layer_o["shift"]

    head_dim = d_model // num_heads

    # Q (160,64)@(64,64) TODO: repeat for each head
    with open(f"aie/layer_{idx}_q.cc", "a") as f:
        f.write(f'''
#include <cstdint>
__attribute__((section(".data"))) alignas(32) int8_t k_p [{Wq_tiled.size}] = {{ {", ".join(str(int(x)) for x in Wq_tiled)} }};

#include "kernels.h"

void q{idx}(input_stream_int8 * __restrict x, output_stream_int8 * __restrict a){{ dense<{m}, {k}, {n}, {T//m}, {d_model//k}, {head_dim//n}, {SHIFT_Q}, false>(x, a, k_p);}}
''')
    # K (160,64)@(64,64) TODO: repeat for each head
    with open(f"aie/layer_{idx}_k.cc", "a") as f:
        f.write(f'''
#include <cstdint>
__attribute__((section(".data"))) alignas(32) int8_t k_p [{Wk_tiled.size}] = {{ {", ".join(str(int(x)) for x in Wk_tiled)} }};

#include "kernels.h"

void k{idx}(input_stream_int8 * __restrict x, output_stream_int8 * __restrict a){{ dense<{m}, {k}, {n}, {T//m}, {d_model//k}, {head_dim//n}, {SHIFT_K}, false>(x, a, k_p);}}
''')
    # V (160,64)@(64,64) TODO: repeat for each head
    with open(f"aie/layer_{idx}_v.cc", "a") as f:
        f.write(f'''
#include <cstdint>
__attribute__((section(".data"))) alignas(32) int8_t k_p [{Wv_tiled.size}] = {{ {", ".join(str(int(x)) for x in Wv_tiled)} }};

#include "kernels.h"

void v{idx}(input_stream_int8 * __restrict x, output_stream_int8 * __restrict a){{ dense<{m}, {k}, {n}, {T//m}, {d_model//k}, {head_dim//n}, {SHIFT_V}, false>(x, a, k_p);}}
''')
    # score = (Q @ K^T)  (160,64)@(64,160)  TODO: repeat for each head
    with open(f"aie/layer_{idx}_attn.cc", "a") as f:
        f.write(f'''
#include "kernels.h"

void attn{idx}(output_stream_int8 * __restrict o_head, input_stream_int8 * __restrict q_head, input_stream_int8 * __restrict k_head){{ attention<{m}, {k}, {n}, {T//m}, {d_model//k}, {T//n}, {d_model}, {T}, {SHIFT_S}>(q_head, k_head, o_head);}}
''')
    # head = (scores @ V) (160,160)@(160,64) TODO: repeat for each head
    with open(f"aie/layer_{idx}_head.cc", "a") as f:
        f.write(f'''
#include "kernels.h"

void head{idx}(input_stream_int8 * __restrict x, input_stream_int8 * __restrict v, output_stream_int8 * __restrict a){{ head<{m}, {k}, {n}, {T//m}, {T//k}, {head_dim//n}, {SHIFT_C}>(x, a, v);}}
''')
    head_idx = 0
    # (concatenated heads @ Wo) (160,64)@(64,64)
    with open(f"aie/layer_{idx}_out.cc", "a") as f:
        f.write(f'''
#include <cstdint>
__attribute__((section(".data"))) alignas(32) int8_t k_p [{Wo_tiled.size}] = {{ {", ".join(str(int(x)) for x in Wo_tiled)} }};

#include "kernels.h"

void out{idx}(input_stream_int8 * __restrict x, output_stream_int8 * __restrict a){{ output<{m}, {k}, {n}, {d_model}, {T}, {SHIFT_O}>(x, a, k_p);}}
''')
    
    # in_bytes = layer_q['x'].size * layer_q['x'].itemsize
    in_port = "AIE_IN" if idx == 0 else f"layers[{idx-1}]"
    # q_bytes = layer_q['a'].size * layer_q['a'].itemsize
    # k_bytes = layer_k['a'].size * layer_k['a'].itemsize
    # v_bytes = layer_v['a'].size * layer_v['a'].itemsize
    # attn_bytes = T*T # large window size causes memory error
    # head_bytes = layer_o['x'].size * layer_o['x'].itemsize

    with open("aie/layer_graph.h", "a") as f: # TODO: set mha index properly
        f.write(f"mha[{0}] = kernel::create(q{idx});\n")
        f.write(f'source(mha[{0}]) = "layer_{idx}_q.cc";\n')
        f.write(f'runtime<ratio>(mha[{0}]) = 1.0;\n')
        f.write(f"connect<stream>({in_port}.out[0], mha[{0}].in[0]);\n\n")

        f.write(f"mha[{1}] = kernel::create(k{idx});\n")
        f.write(f'source(mha[{1}]) = "layer_{idx}_k.cc";\n')
        f.write(f'runtime<ratio>(mha[{1}]) = 1.0;\n')
        f.write(f"connect<stream>({in_port}.out[0], mha[{1}].in[0]);\n\n")

        f.write(f"mha[{2}] = kernel::create(v{idx});\n")
        f.write(f'source(mha[{2}]) = "layer_{idx}_v.cc";\n')
        f.write(f'runtime<ratio>(mha[{2}]) = 1.0;\n')
        f.write(f"connect<stream>({in_port}.out[0], mha[{2}].in[0]);\n\n")

        f.write(f"mha[{3}] = kernel::create(attn{idx});\n")
        f.write(f'source(mha[{3}]) = "layer_{idx}_attn.cc";\n')
        f.write(f'runtime<ratio>(mha[{3}]) = 1.0;\n')
        f.write(f"connect<stream>(mha[{0}].out[0], mha[{3}].in[0]);\n\n")
        f.write(f"connect<stream>(mha[{1}].out[0], mha[{3}].in[1]);\n\n")

        f.write(f"mha[{4}] = kernel::create(head{idx});\n")
        f.write(f'source(mha[{4}]) = "layer_{idx}_head.cc";\n')
        f.write(f'runtime<ratio>(mha[{4}]) = 1.0;\n')
        f.write(f"connect<stream>(mha[{3}].out[0], mha[{4}].in[0]);\n\n")
        f.write(f"connect<stream>(mha[{2}].out[0], mha[{4}].in[1]);\n\n")

        f.write(f"mha[{5}] = kernel::create(out{idx});\n")
        f.write(f'source(mha[{5}]) = "layer_{idx}_out.cc";\n')
        f.write(f'runtime<ratio>(mha[{5}]) = 1.0;\n')
        f.write(f"connect<stream>(mha[{4}].out[0], mha[{5}].in[0]);\n\n")

def to_btc(t):
    if t.ndim == 2:  # (T,C)
        return t[None, ...]
    if t.ndim == 3:  # (B,T,C)
        return t
    raise ValueError("Expected (T,C) or (B,T,C)")


if __name__ == "__main__":
    ################################################## PYTHON REFERENCE ##################################################
    seed = 0
    rng = np.random.default_rng(seed)
    
    layers = []
    # out_dim = 5

    # Keras-like dims
    in_particles, num_feature, ff_dim = 150, 3, 64
    num_feature_pad = 8   # pad to multiple of 8 for AIE tiling

    ###################################################################### below is layer 1
    # ---- Input + padding (3 -> 8) ----
    dummy_inp = rng.integers(-128, 128, size=(in_particles, num_feature), dtype=np.int8)
    pad_inp   = np.zeros((in_particles, num_feature_pad), dtype=np.int8)
    pad_inp[:, :num_feature] = dummy_inp

    # ---- Dense to reach MHA width: (150,8) · (8,64) -> (150,64) ----
    W_fc1 = rng.integers(-128, 128, size=(num_feature_pad, ff_dim), dtype=np.int8)
    a1, L1 = golden_fc(pad_inp, W_fc1, is_relu=True, shift=2)
    layers += L1

    # ---- MHA 1 + residual ----
    numheads = 1 #4
    Wq = rng.integers(-128, 128, size=(ff_dim, ff_dim), dtype=np.int8)
    Wk = rng.integers(-128, 128, size=(ff_dim, ff_dim), dtype=np.int8)
    Wv = rng.integers(-128, 128, size=(ff_dim, ff_dim), dtype=np.int8)
    Wo = rng.integers(-128, 128, size=(ff_dim, ff_dim), dtype=np.int8)
    Wq_btc = to_btc(Wq)
    Wk_btc = to_btc(Wk)
    Wv_btc = to_btc(Wv)
    Wo_btc = to_btc(Wo)
    mha1 = NumpyMHALinear(d_model=ff_dim, num_heads=numheads, name_prefix="mha1", seed=seed, Wq=Wq, Wk=Wk, Wv=Wv, Wo=Wo)
    att1 = mha1(a1, a1, a1, layers=layers)      # self-attention: Q=K=V=a1
    # a2   = residual_add_int8(att1, a1)          # Add()([x, emb])

    # ###################################################################### below is layer 2
    # # ---- Two dense (FF) + residual (Block 1) ----
    # W_ff1a = np.random.randint(-128, 128, size=(ff_dim, ff_dim), dtype=np.int8)
    # h1, L2a = golden_fc(a2, W_ff1a, is_relu=True, shift=3); 
    # layers += L2a

    # W_ff1b = np.random.randint(-128, 128, size=(ff_dim, ff_dim), dtype=np.int8)
    # h2, L2b = golden_fc(h1, W_ff1b, is_relu=True, shift=3); 
    # layers += L2b

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
        "aie/weights.h", "aie/layer_0.cc",  "aie/layer_1_attn.cc", "aie/layer_1_head.cc", "aie/layer_1_k.cc", "aie/layer_1_out.cc", "aie/layer_1_q.cc", "aie/layer_1_v.cc", 
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

    print_layers_brief(layers)

    # 1. include.h - common parameters
    
    with open("aie/include.h", "w") as f:
        f.write(f'#define N_LAYERS {1}\n#define ITERATIONS {iterations}\n')
        f.write(f'void f{0}(input_stream_int8 * __restrict, output_stream_int8 * __restrict);\n')
        f.write(f'void q{1}(input_stream_int8 * __restrict, output_stream_int8 * __restrict);\n')
        f.write(f'void k{1}(input_stream_int8 * __restrict, output_stream_int8 * __restrict);\n')
        f.write(f'void v{1}(input_stream_int8 * __restrict, output_stream_int8 * __restrict);\n')
        f.write(f'void attn{1}(output_stream_int8 * __restrict, input_stream_int8 * __restrict, input_stream_int8 * __restrict);\n')
        f.write(f'void head{1}(input_stream_int8 * __restrict, input_stream_int8 * __restrict, output_stream_int8 * __restrict);\n')
        f.write(f'void out{1}(input_stream_int8 * __restrict, output_stream_int8 * __restrict);\n')

    # 2. Preamble of model.cc - each layer as function

    # with open("aie/model.cc", "w") as f:
    #     f.write('#include "kernels.h"\n#include "weights.h"\n')


    # 3. Process each layer: write weights.h, x.txt, a.txt, model.cc, model.h, layer_graph.h

    process_dense_layer(0, layers[0], m, k, n, iterations)
    process_mha_layer(1, m, k, n, layers[1], layers[2], layers[3], layers[4], iterations)
    
    tiled_mat = tile_matrix(layers[-1]['a'], m, n)
    np.savetxt("data/out_ref.txt", np.tile(tiled_mat, (iterations, 1)).reshape(-1, 16), fmt="%s", delimiter=" ")

    # 4. Postamble of layer_graph.h - connect last layer to AIE_OUT
    
    out_bytes = layers[-1]['a'].size * layers[-1]['a'].itemsize
    with open("aie/layer_graph.h", "a") as f:
        f.write(f"connect<stream>(mha[{5}].out[0], AIE_OUT.in[0]);\n")    

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
