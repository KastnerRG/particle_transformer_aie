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

def process_mha_layer(idx, m, k, n, layer_q, layer_k, layer_v, layer_o, iterations, num_heads):
    # Extract parameters
    m = 4
    k = 8
    n = 8
    d_model = 64
    T = 160 # first dimension of input to mha
    SHIFT_Q = 10
    SHIFT_K = 11
    SHIFT_V = 11
    # SHIFT_S = 8
    # SHIFT_C = 10
    SHIFT_S = [6, 6, 7, 7]  # per-head
    SHIFT_C = [10, 10, 10, 10]
    SHIFT_O = 9

    head_dim = d_model // num_heads

    # Split weight matrices by head: (64,64) -> num_heads x (64,16)
    # Each head gets columns [h*head_dim : (h+1)*head_dim]
    Wq_heads = []
    Wk_heads = []
    Wv_heads = []

    for h in range(num_heads):
        col_start = h * head_dim
        col_end = (h + 1) * head_dim

        # Extract per-head weight slices: (64, head_dim)
        Wq_h = layer_q["k"][:, col_start:col_end]  # (64, 16) for 4 heads
        Wk_h = layer_k["k"][:, col_start:col_end]
        Wv_h = layer_v["k"][:, col_start:col_end]

        # Tile each head's weight matrix
        Wq_h_tiled = tile_matrix(Wq_h, k, n)
        Wk_h_tiled = tile_matrix(Wk_h, k, n)
        Wv_h_tiled = tile_matrix(Wv_h, k, n)

        Wq_heads.append(Wq_h_tiled)
        Wk_heads.append(Wk_h_tiled)
        Wv_heads.append(Wv_h_tiled)

        # Save per-head weights to separate files
        np.savetxt(f"data/mha_Wq{idx}_head{h}.txt", Wq_h, fmt="%d")
        np.savetxt(f"data/mha_Wk{idx}_head{h}.txt", Wk_h, fmt="%d")
        np.savetxt(f"data/mha_Wv{idx}_head{h}.txt", Wv_h, fmt="%d")

    # Output projection is still full width: (64,64)
    Wo_tiled = tile_matrix(layer_o["k"], k, n)
    np.savetxt(f"data/mha_Wo{idx}.txt", layer_o["k"], fmt="%d")

    # Input and output data
    x_tiled = tile_matrix(layer_q["x"], m, k)
    out_x_tiled = tile_matrix(layer_o["x"], m, k)

    np.savetxt(f"data/x{idx}.txt", np.tile(x_tiled, (iterations, 1)).reshape(-1, 16), fmt="%s", delimiter=" ")
    np.savetxt(f"data/out_x{idx}.txt", np.tile(out_x_tiled, (iterations, 1)).reshape(-1, 16), fmt="%s", delimiter=" ")

    # Generate Q, K, V projection kernels for each head
    for h in range(num_heads):
        # Q projection: (160,64) @ (64, head_dim) -> (160, head_dim)
        with open(f"aie/layer_{idx}_q_head{h}.cc", "w") as f:
            f.write(f'''#include <cstdint>
__attribute__((section(".data"))) alignas(32) int8_t k_p [{Wq_heads[h].size}] = {{ {", ".join(str(int(x)) for x in Wq_heads[h])} }};

#include "kernels.h"

void q{idx}_head{h}(input_stream_int8 * __restrict x, output_stream_int8 * __restrict a){{ dense<{m}, {k}, {n}, {T//m}, {d_model//k}, {head_dim//n}, {SHIFT_Q}, false>(x, a, k_p);}}
''')

        # K projection: (160,64) @ (64, head_dim) -> (160, head_dim)
        with open(f"aie/layer_{idx}_k_head{h}.cc", "w") as f:
            f.write(f'''#include <cstdint>
__attribute__((section(".data"))) alignas(32) int8_t k_p [{Wk_heads[h].size}] = {{ {", ".join(str(int(x)) for x in Wk_heads[h])} }};

#include "kernels.h"

void k{idx}_head{h}(input_stream_int8 * __restrict x, output_stream_int8 * __restrict a){{ dense<{m}, {k}, {n}, {T//m}, {d_model//k}, {head_dim//n}, {SHIFT_K}, false>(x, a, k_p);}}
''')

        # V projection: (160,64) @ (64, head_dim) -> (160, head_dim)
        with open(f"aie/layer_{idx}_v_head{h}.cc", "w") as f:
            f.write(f'''#include <cstdint>
__attribute__((section(".data"))) alignas(32) int8_t k_p [{Wv_heads[h].size}] = {{ {", ".join(str(int(x)) for x in Wv_heads[h])} }};

#include "kernels.h"

void v{idx}_head{h}(input_stream_int8 * __restrict x, output_stream_int8 * __restrict a){{ dense<{m}, {k}, {n}, {T//m}, {d_model//k}, {head_dim//n}, {SHIFT_V}, false>(x, a, k_p);}}
''')

        # Scores: Q @ K^T for this head (160, head_dim) @ (head_dim, 160) -> (160, 160)
        with open(f"aie/layer_{idx}_scores_head{h}.cc", "w") as f:
            f.write(f'''#include "kernels.h"

void scores{idx}_head{h}(input_stream_int8 * __restrict q_head, input_stream_int8 * __restrict k_head, output_stream_int8 * __restrict o_head){{ scores<{m}, {k}, {n}, {T//m}, {head_dim//k}, {head_dim//n}, {head_dim}, {T}, {SHIFT_S[h]}>(q_head, k_head, o_head);}}
''')

        # Context: (scores @ V) for this head (160, 160) @ (160, head_dim) -> (160, head_dim)
        with open(f"aie/layer_{idx}_context_head{h}.cc", "w") as f:
            f.write(f'''#include "kernels.h"

void context{idx}_head{h}(input_stream_int8 * __restrict x, input_stream_int8 * __restrict v, output_stream_int8 * __restrict a){{ context<{m}, {k}, {n}, {T//m}, {T//k}, {head_dim//n}, {SHIFT_C[h]}>(x, v, a);}}
''')

    # (h0+h1)->concat_0, (h2+h3)->concat_1, then output kernel does final concat
    with open(f"aie/layer_{idx}_concat.cc", "w") as f:
        f.write(f'''#include "kernels.h"

// Concat head 0 and head 1: (160,16) + (160,16) -> (160,32)
void concat{idx}_0(input_stream_int8 * __restrict sA, input_stream_int8 * __restrict sB, output_stream_int8 * __restrict sC){{
  concat<{m}, {n}, {T//m}, {head_dim//n}>(sA, sB, sC);
}}

// Concat head 2 and head 3: (160,16) + (160,16) -> (160,32)
void concat{idx}_1(input_stream_int8 * __restrict sA, input_stream_int8 * __restrict sB, output_stream_int8 * __restrict sC){{
  concat<{m}, {n}, {T//m}, {head_dim//n}>(sA, sB, sC);
}}
''')

    # Output projection: uses output() kernel which concatenates 2 inputs (concat_0 + concat_1) + matmul
    # (160,32) + (160,32) -> (160,64) @ (64,64) -> (160,64)
    with open(f"aie/layer_{idx}_out.cc", "w") as f:
        f.write(f'''#include <cstdint>
__attribute__((section(".data"))) alignas(32) int8_t k_p [{Wo_tiled.size}] = {{ {", ".join(str(int(x)) for x in Wo_tiled)} }};

#include "kernels.h"

void out{idx}(input_stream_int8 * __restrict sA, input_stream_int8 * __restrict sB, output_stream_int8 * __restrict a){{output<{m}, {k}, {n}, {T//m}, {d_model//k}, {d_model//n}, {SHIFT_O}>(sA, sB, a, k_p);}}
''')



    in_port = "AIE_IN" if idx == 0 else f"layers[{idx-1}]"

    with open("aie/layer_graph.h", "a") as f:
        # Generate graph connections for each head
        for h in range(num_heads):
            base = h * 5  # Each head has 5 kernels: q, k, v, scores, context

            # Kernel indices within this head
            q_idx = base + 0
            k_idx = base + 1
            v_idx = base + 2
            scores_idx = base + 3
            context_idx = base + 4

            # Q projection kernel
            f.write(f"mha[{q_idx}] = kernel::create(q{idx}_head{h});\n")
            f.write(f'source(mha[{q_idx}]) = "layer_{idx}_q_head{h}.cc";\n')
            f.write(f'runtime<ratio>(mha[{q_idx}]) = 1.0;\n')
            f.write(f"connect<stream>({in_port}.out[0], mha[{q_idx}].in[0]);\n\n")

            # K projection kernel
            f.write(f"mha[{k_idx}] = kernel::create(k{idx}_head{h});\n")
            f.write(f'source(mha[{k_idx}]) = "layer_{idx}_k_head{h}.cc";\n')
            f.write(f'runtime<ratio>(mha[{k_idx}]) = 1.0;\n')
            f.write(f"connect<stream>({in_port}.out[0], mha[{k_idx}].in[0]);\n\n")

            # V projection kernel
            f.write(f"mha[{v_idx}] = kernel::create(v{idx}_head{h});\n")
            f.write(f'source(mha[{v_idx}]) = "layer_{idx}_v_head{h}.cc";\n')
            f.write(f'runtime<ratio>(mha[{v_idx}]) = 1.0;\n')
            f.write(f"connect<stream>({in_port}.out[0], mha[{v_idx}].in[0]);\n\n")

            # Scores kernel (Q @ K^T)
            f.write(f"mha[{scores_idx}] = kernel::create(scores{idx}_head{h});\n")
            f.write(f'source(mha[{scores_idx}]) = "layer_{idx}_scores_head{h}.cc";\n')
            f.write(f'runtime<ratio>(mha[{scores_idx}]) = 1.0;\n')
            f.write(f"connect<stream> s{h}_qk(mha[{q_idx}].out[0], mha[{scores_idx}].in[0]);\n")
            f.write(f"fifo_depth(s{h}_qk) = {int(T*head_dim/4)};\n")
            f.write(f"connect<stream>(mha[{k_idx}].out[0], mha[{scores_idx}].in[1]);\n\n")

            # Context kernel (scores @ V)
            f.write(f"mha[{context_idx}] = kernel::create(context{idx}_head{h});\n")
            f.write(f'source(mha[{context_idx}]) = "layer_{idx}_context_head{h}.cc";\n')
            f.write(f'runtime<ratio>(mha[{context_idx}]) = 1.0;\n')
            f.write(f"connect<stream> s{h}_sv(mha[{scores_idx}].out[0], mha[{context_idx}].in[0]);\n")
            f.write(f"fifo_depth(s{h}_sv) = {int(T*T/4)};\n")
            f.write(f"connect<stream>(mha[{v_idx}].out[0], mha[{context_idx}].in[1]);\n\n")

        # concat_0_idx: concatenates head 0 and head 1
        # concat_1_idx: concatenates head 2 and head 3
        concat_0_idx = num_heads * 5  # 20
        concat_1_idx = num_heads * 5 + 1  # 21
        out_idx = num_heads * 5 + 2  # 22

        # Create concat_0 kernel: head0 + head1 -> (160,32)
        f.write(f"mha[{concat_0_idx}] = kernel::create(concat{idx}_0);\n")
        f.write(f'source(mha[{concat_0_idx}]) = "layer_{idx}_concat.cc";\n')
        f.write(f'runtime<ratio>(mha[{concat_0_idx}]) = 1.0;\n')
        f.write(f"connect<stream>(mha[{4}].out[0], mha[{concat_0_idx}].in[0]);\n")  # head 0 context
        f.write(f"connect<stream>(mha[{9}].out[0], mha[{concat_0_idx}].in[1]);\n\n")  # head 1 context

        # Create concat_1 kernel: head2 + head3 -> (160,32)
        f.write(f"mha[{concat_1_idx}] = kernel::create(concat{idx}_1);\n")
        f.write(f'source(mha[{concat_1_idx}]) = "layer_{idx}_concat.cc";\n')
        f.write(f'runtime<ratio>(mha[{concat_1_idx}]) = 1.0;\n')
        f.write(f"connect<stream>(mha[{14}].out[0], mha[{concat_1_idx}].in[0]);\n")  # head 2 context
        f.write(f"connect<stream>(mha[{19}].out[0], mha[{concat_1_idx}].in[1]);\n\n")  # head 3 context

        # Create output projection kernel: concat_0 + concat_1 -> (160,64) via output() kernel
        f.write(f"mha[{out_idx}] = kernel::create(out{idx});\n")
        f.write(f'source(mha[{out_idx}]) = "layer_{idx}_out.cc";\n')
        f.write(f'runtime<ratio>(mha[{out_idx}]) = 1.0;\n')
        f.write(f"connect<stream>(mha[{concat_0_idx}].out[0], mha[{out_idx}].in[0]);\n")
        f.write(f"connect<stream>(mha[{concat_1_idx}].out[0], mha[{out_idx}].in[1]);\n\n")

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
    num_particles_pad = 160 # pad to multiple of 8 for AIE tiling in mha layer.
    ###################################################################### below is layer 1
    # ---- Input + padding (3 -> 8) ----
    dummy_inp = rng.integers(-128, 128, size=(in_particles, num_feature), dtype=np.int8)
    pad_inp   = np.zeros((num_particles_pad, num_feature_pad), dtype=np.int8)
    pad_inp[:in_particles, :num_feature] = dummy_inp

    # ---- Dense to reach MHA width: (160,8) · (8,64) -> (160,64) ----
    W_fc1 = rng.integers(-128, 128, size=(num_feature_pad, ff_dim), dtype=np.int8)
    a1, L1 = golden_fc(pad_inp, W_fc1, is_relu=True, shift=2)
    # print(f"Dense: x_in = {pad_inp.shape}, W = {W_fc1.shape}, x_out = {a1.shape}")
    layers += L1

    
    # ---- MHA 1 + residual ----
    numheads = 4
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




    m, k, n = 4,8,8 # k==n such that output matrix can be fed as input without re-tiling
    iterations = 1

    # 0. Do a cleanup

    for path in [
        "data", "aie/layer_graph.h", "aie/include.h", "aie/model.cc", "aie/model.h",
        "aie/weights.h", "aie/layer_0.cc", "aie/layer_1_*.cc",
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

    # 1. include.h - common parameters and function declarations

    with open("aie/include.h", "w") as f:
        f.write(f'#define N_LAYERS {1}\n#define ITERATIONS {iterations}\n\n')

        # Dense layer function
        f.write(f'void f{0}(input_stream_int8 * __restrict, output_stream_int8 * __restrict);\n\n')

        # MHA per-head functions
        for h in range(numheads):
            f.write(f'void q{1}_head{h}(input_stream_int8 * __restrict, output_stream_int8 * __restrict);\n')
            f.write(f'void k{1}_head{h}(input_stream_int8 * __restrict, output_stream_int8 * __restrict);\n')
            f.write(f'void v{1}_head{h}(input_stream_int8 * __restrict, output_stream_int8 * __restrict);\n')
            f.write(f'void scores{1}_head{h}(input_stream_int8 * __restrict, input_stream_int8 * __restrict, output_stream_int8 * __restrict);\n')
            f.write(f'void context{1}_head{h}(input_stream_int8 * __restrict, input_stream_int8 * __restrict, output_stream_int8 * __restrict);\n')
            f.write(f'\n')
        if numheads == 4:
            f.write(f'void concat{1}_0(input_stream_int8 * __restrict, input_stream_int8 * __restrict, output_stream_int8 * __restrict);\n')
            f.write(f'void concat{1}_1(input_stream_int8 * __restrict, input_stream_int8 * __restrict, output_stream_int8 * __restrict);\n\n')

        if numheads == 1:
            f.write(f'void out{1}(input_stream_int8 * __restrict, output_stream_int8 * __restrict);\n')
        else:
            f.write(f'void out{1}(input_stream_int8 * __restrict, input_stream_int8 * __restrict, output_stream_int8 * __restrict);\n')

    # 2. Process each layer: write weights.h, x.txt, a.txt, layer_graph.h

    process_dense_layer(0, layers[0], m, k, n, iterations)
    process_mha_layer(1, m, k, n, layers[1], layers[2], layers[3], layers[4], iterations, numheads)
    
    tiled_mat = tile_matrix(layers[-1]['a'], m, n)
    np.savetxt("data/out_ref.txt", np.tile(tiled_mat, (iterations, 1)).reshape(-1, 16), fmt="%s", delimiter=" ")

    # 3. Postamble of layer_graph.h - connect last layer to AIE_OUT

    out_bytes = layers[-1]['a'].size * layers[-1]['a'].itemsize
    # For num_heads=4: out_idx = 4*5 + 2 = 22 (5 kernels/head + 2 concat + 1 output)
    final_out_idx = numheads * 5 + 2
    with open("aie/layer_graph.h", "a") as f:
        f.write(f"connect<stream>(mha[{final_out_idx}].out[0], AIE_OUT.in[0]);\n")    

    # 4. Run AIE (graph.cpp)

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
