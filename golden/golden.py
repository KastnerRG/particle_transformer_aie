import numpy as np
import os, glob, shutil, subprocess

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
    with open("aie/weights.h", 'a') as f:
        f.write(f"const int8_t k{idx} [{k_tiled.size}] = {{ {array_str} }};\n")

    x_tiled = tile_matrix(layer["x"], m, k)
    a_tiled = tile_matrix(layer["a"], m, n)
    np.savetxt(f"data/x{idx}.txt", np.tile(x_tiled, (iterations, 1)).reshape(-1, 16), fmt="%s", delimiter=" ")
    np.savetxt(f"data/a{idx}.txt", np.tile(a_tiled, (iterations, 1)).reshape(-1, 16), fmt="%s", delimiter=" ")

    # model.cc - each layer as function

    t_m = layer['x'].shape[0] // m
    t_k = layer['x'].shape[1] // k
    t_n = layer['k'].shape[1] // n
    shift = layer['shift']
    is_relu = str(layer['is_relu']).lower()

    with open("aie/model.cc", "a") as f:
        f.write(f"void f{idx}(input_window_int8* __restrict x, output_window_int8 * __restrict a) ")
        f.write(f"{{ dense<{m}, {k}, {n}, {t_m}, {t_k}, {t_n}, {shift}, {is_relu}> (x, a, k{idx}); }}\n")

    # model.h - Function prototypes

    with open("aie/model.h", "a") as f:
        f.write(f"void f{idx}( input_window_int8  * __restrict, output_window_int8 * __restrict);\n")

    # layer_graph.h - create and connect layers

    num_bytes = layer['x'].size * layer['x'].itemsize
    in_port = "AIE_IN" if idx == 0 else f"layers[{idx-1}]"

    with open("aie/layer_graph.h", "a") as f:
        f.write(f"layers[{idx}] = kernel::create(f{idx});\n")
        f.write(f"connect<window<{num_bytes:>5}>>({in_port}.out[0], layers[{idx}].in[0]);\n\n")


if __name__ == "__main__":
    
    layers = []

    is_relu = True
    shift = 2
    x = np.random.randint(0, 128, size=(16, 32), dtype=np.int8)
    k = np.random.randint(0, 128, size=(32, 32), dtype=np.int8)
    y = np.matmul(x.astype(np.int32), k.astype(np.int32))
    y = (y >> shift).astype(np.int8)
    a = np.maximum(0, y) if is_relu else y
    layers += [{'x': x, 'k': k, 'y': y, 'a': a, 'shift': shift, 'is_relu': is_relu}]

    is_relu = False
    shift = 3
    x = a
    k = np.random.randint(0, 128, size=(32, 64), dtype=np.int8)
    y = np.matmul(x.astype(np.int32), k.astype(np.int32))
    y = (y >> shift).astype(np.int8)
    a = np.maximum(0, y) if is_relu else y
    layers += [{'x': x, 'k': k, 'y': y, 'a': a, 'shift': shift, 'is_relu': is_relu}]

    is_relu = True
    shift = 4
    x = a
    k = np.random.randint(0, 128, size=(64, 32), dtype=np.int8)
    y = np.matmul(x.astype(np.int32), k.astype(np.int32))
    y = (y >> shift).astype(np.int8)
    a = np.maximum(0, y) if is_relu else y
    layers += [{'x': x, 'k': k, 'y': y, 'a': a, 'shift': shift, 'is_relu': is_relu}]

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
    
    os.makedirs("data")

    # 1. include.h - common parameters
    
    with open("aie/include.h", "w") as f:
        f.write(f'#define N_LAYERS {len(layers)}\n#define ITERATIONS {iterations}')

    # 2. Preamble of model.cc - each layer as function

    with open("aie/model.cc", "w") as f:
        f.write('#include "kernels.h"\n#include "weights.h"\n')

    # 3. Process each layer: write weights.h, x.txt, a.txt, model.cc, model.h, layer_graph.h

    for i, layer in enumerate(layers):
        process_layer(i, layer, m, k, n, iterations)
    
    tiled_mat = tile_matrix(layers[-1]['a'], m, n)
    np.savetxt("data/out_ref.txt", np.tile(tiled_mat, (iterations, 1)).reshape(-1, 16), fmt="%s", delimiter=" ")

    # 4. Postamble of layer_graph.h - connect last layer to AIE_OUT
    
    out_bytes = layers[-1]['a'].size * layers[-1]['a'].itemsize
    with open("aie/layer_graph.h", "a") as f:
        f.write(f"connect<window<{out_bytes:>5}>>(layers[{len(layers)-1}].out[0], AIE_OUT.in[0]);\n")

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
