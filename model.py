"""
AIE Model class for building and executing neural network models on AIE.

The AIEModel manages the complete workflow:
1. Build DAG of layers
2. Compute golden reference (NumPy)
3. Generate AIE C++ code
4. Compile and simulate
5. Validate results
"""

import numpy as np
import os
import glob
import shutil
import subprocess
from typing import List, Optional, Tuple, Any, Dict
from utils.tiling import tile_matrix


class AIEModel:
    """
    Main model class for AIE neural network execution.
    """

    def __init__(self, m: int = 4, k: int = 8, n: int = 8, iterations: int = 1):
        self.m = m
        self.k = k
        self.n = n
        self.iterations = iterations

        self.layers: List[Any] = []  # Ordered list of layers
        self.input_map: Dict[Any, List[Tuple[Any, int]]] = {}  # layer -> [(src_layer, port), ...]

        self.input_data: Optional[np.ndarray] = None

    def add_layer(self, layer, inputs: List[Any]):
        """
        Add a layer to the model.

        Automatically assigns layer index and tiling parameters.

        Args:
            layer: Layer object (DenseLayer, MHALayer, etc.)
            inputs: List of source layers (required)
                   - [None]: connect to AIE_IN
                   - [layer1]: single input
                   - [layer1, layer2]: multiple inputs (e.g., ResAdd)
                   Port index is automatically set to the last kernel in each source layer.

        Returns:
            The layer object (for chaining)
        """
        # Assign index
        layer.idx = len(self.layers)

        # Set tiling parameters
        layer.m = self.m
        layer.k = self.k
        layer.n = self.n

        self.layers.append(layer)

        if inputs is None or not isinstance(inputs, list) or len(inputs) == 0:
            raise ValueError("inputs must be a non-empty list. Use [None] to connect to AIE_IN.")

        input_tuples = []
        for src in inputs:
            if src is None:
                input_tuples.append((None, 0))
            else:
                port_idx = src.num_kernels() - 1
                input_tuples.append((src, port_idx))
        self.input_map[layer] = input_tuples

        return layer

    def forward(self, input_data: np.ndarray) -> np.ndarray:
        """
        Execute the complete AIE workflow.

        Steps:
        1. Compute golden reference (NumPy)
        2. Save golden outputs to files
        3. Generate AIE C++ code
        4. Compile with Vitis
        5. Run simulation
        6. Validate against golden

        Args:
            input_data: Input array (will be saved and tiled)

        Returns:
            Final layer output from golden computation
        """
        self.input_data = input_data

        print("=" * 60)
        print("AIE Model Forward Pass")
        print("=" * 60)

        print("\n[1/6] Computing golden reference...")
        self._compute_golden()

        print("\n[2/6] Creating include.h header...")
        self._generate_include_header()

        print("\n[3/6] Generating AIE kernel code...")
        self._generate_kernels()

        print("\n[4/6] Generating graph connections...")
        self._generate_graph()

        print("\n[5/6] Compiling and simulating...")
        self._run_simulation()

        print("\n[6/6] Validating results...")
        match = self._validate_output()

        if match:
            print("\n" + "=" * 60)
            print("SUCCESS: AIE output matches golden reference!")
            print("=" * 60)
        else:
            print("\n" + "=" * 60)
            print("ERROR: Output mismatch!")
            print("=" * 60)

        return self.layers[-1].outputs['a']

    def _get_layer_inputs(self, layer) -> List[np.ndarray]:
        """
        Get input data for a layer from connected sources.

        Args:
            layer: Layer to get inputs for

        Returns:
            List of input arrays
        """
        inputs = self.input_map.get(layer, [])
        input_data = []

        for src_layer, port_idx in inputs:
            if src_layer is None:
                # Input from AIE_IN
                input_data.append(self.input_data)
            else:
                # Input from another layer's output
                input_data.append(src_layer.outputs['a'])

        return input_data

    def _compute_golden(self):
        """
        Compute golden reference for all layers using NumPy.

        Also saves tiled golden outputs to data/ directory for validation.
        """
        if os.path.exists('data'):
            shutil.rmtree('data')
        os.makedirs('data')

        for i, layer in enumerate(self.layers):
            inputs = self._get_layer_inputs(layer)

            print(f"  Layer {i} ({layer.name}): ", end='')
            output = layer._compute_golden(inputs)
            print(f"output shape = {output.shape}")

            output_tiled = tile_matrix(output, self.m, self.n)
            tiled_repeated = np.tile(output_tiled, (self.iterations, 1))
            np.savetxt(f"data/a{i}_golden.txt",
                      tiled_repeated.reshape(-1, 16),
                      fmt="%s", delimiter=" ")

        final_output = self.layers[-1].outputs['a']
        final_tiled = tile_matrix(final_output, self.m, self.n)
        final_repeated = np.tile(final_tiled, (self.iterations, 1))
        np.savetxt("data/out_ref.txt",
                  final_repeated.reshape(-1, 16),
                  fmt="%s", delimiter=" ")

        input_tiled = tile_matrix(self.input_data, self.m, self.k)
        input_repeated = np.tile(input_tiled, (self.iterations, 1))
        np.savetxt("data/input.txt",
                  input_repeated.reshape(-1, 16),
                  fmt="%s", delimiter=" ")

        print(f"  ✓ Golden computation complete. Saved to data/")

    def _generate_kernels(self):
        """Generate C++ kernel files (layer_X.cc) for all layers."""
        for path in glob.glob("aie/layer_*.cc"):
            os.remove(path)

        for layer in self.layers:
            with open(f"aie/layer_{layer.idx}.cc", "w") as f:
                layer.generate_kernel_code(f)

        print(f"  ✓ Generated {len(self.layers)} kernel files")

    def _generate_graph(self):
        """Generate graph.cpp and layer_graph.h (connections separated)."""
        with open("aie/layer_graph.h", "w") as lg:
            for layer in self.layers:
                input_ports = self._get_layer_input_ports(layer)
                layer.generate_graph_code(lg, input_ports)
            final_layer = self.layers[-1]
            lg.write(f'        connect<stream>({final_layer.get_output_port()}.out[0], AIE_OUT.in[0]);\n')

        with open("aie/graph.cpp", "w") as f:
            f.write('#include <adf.h>\n')
            f.write('#include "include.h"\n')
            f.write('#include <vector>\n\n')
            f.write('using namespace adf;\n\n')

            f.write('class simpleGraph : public adf::graph {\n')
            f.write('public:\n')
            f.write('  input_plio  AIE_IN;\n')
            f.write('  output_plio AIE_OUT;\n\n')

            for layer in self.layers:
                f.write(f'  kernel {layer.name} [{layer.num_kernels()}];\n')
            f.write('\n  simpleGraph(){\n\n')

            input_bytes = self.input_data.size * self.input_data.itemsize * self.iterations
            output_bytes = self.layers[-1].outputs['a'].size * self.layers[-1].outputs['a'].itemsize * self.iterations
            f.write(f'    AIE_IN = input_plio::create("DataIn", plio_128_bits, "data/input.txt", {input_bytes});\n')
            f.write(f'    AIE_OUT = output_plio::create("DataOut", plio_128_bits, "data/out_sim.txt", {output_bytes});\n\n')

            f.write('    #include "layer_graph.h"\n\n')

            f.write('  }\n');
            f.write('};\n\n')

            f.write('simpleGraph mygraph;\n\n')
            f.write('int main(void) {\n')
            f.write('  mygraph.init();\n')
            f.write('  adf::event::handle h_latency =\n')
            f.write('    adf::event::start_profiling(mygraph.AIE_IN, mygraph.AIE_OUT,\n')
            f.write('                                adf::event::io_stream_start_difference_cycles);\n')
            f.write('  mygraph.run(ITERATIONS);\n')
            f.write('  mygraph.end();\n')
            f.write('  long long latency_cycles = adf::event::read_profiling(h_latency);\n')
            f.write('  adf::event::stop_profiling(h_latency);\n')
            f.write('  const int AIE_clock_Hz = 1200000000;\n')
            f.write('  printf("\\n\\n\\n--------GRAPH LATENCY    (First in  -> First out) : %lld cycles, %.1f ns\\n\\n\\n",\n')
            f.write('         latency_cycles, (1e9 * (double)latency_cycles) / AIE_clock_Hz);\n')
            f.write('  return 0;\n')
            f.write('}\n')

    def _get_layer_input_ports(self, layer) -> List[str]:
        """
        Get input port names for a layer.
        
        Args:
            layer: Layer to get input ports for

        Returns:
            List of port names (e.g., ['AIE_IN', 'dense_0'])
        """
        inputs = self.input_map.get(layer, [])
        port_names = []

        for src_layer, port_idx in inputs:
            if src_layer is None:
                port_names.append("AIE_IN")
            else:
                port_names.append(src_layer.get_output_port(port_idx))

        return port_names

    def _generate_include_header(self):
        """Generate include.h header (layers append their own function declarations)."""
        with open("aie/include.h", "w") as f:
            f.write(f'#define N_LAYERS {len(self.layers)}\n')
            f.write(f'#define ITERATIONS {self.iterations}\n\n')

        print(f"  ✓ Created include.h header (function declarations added by layers)")

    def _run_simulation(self):
        """Run Vitis compilation and AIE simulation."""
        try:
            subprocess.run(["./run.sh"], check=True, cwd=".")
            print(f"  ✓ Compilation and simulation complete")
        except subprocess.CalledProcessError as e:
            print(f"  ✗ Error during compilation/simulation: {e}")
            raise

    def _validate_output(self) -> bool:
        """
        Validate AIE simulation output against golden reference.

        Returns:
            True if outputs match, False otherwise
        """
        aie_out_path = "aiesimulator_output/data/out_sim.txt"

        if not os.path.exists(aie_out_path):
            print(f"  ✗ AIE output file not found: {aie_out_path}")
            return False

        # Clean AIE output (remove timestamp lines)
        with open(aie_out_path, "r") as infile, open("data/out_sim.txt", "w") as outfile:
            for line in infile:
                if not line.startswith("T"):
                    outfile.write(line)

        # Load and compare
        out_sim = np.loadtxt("data/out_sim.txt").astype(np.int32)
        out_ref = np.loadtxt("data/out_ref.txt").astype(np.int32)

        if out_sim.shape != out_ref.shape:
            print(f"  ✗ Shape mismatch: sim={out_sim.shape}, ref={out_ref.shape}")
            return False

        if not np.array_equal(out_sim, out_ref):
            diff = np.sum(out_sim != out_ref)
            print(f"  ✗ {diff} elements differ")
            print(f"    Simulation output:\n{out_sim[:5]}")
            print(f"    Reference output:\n{out_ref[:5]}")
            return False

        print(f"  ✓ Outputs match! Shape: {out_sim.shape}")
        return True

    def __repr__(self) -> str:
        """String representation."""
        return f"AIEModel(m={self.m}, k={self.k}, n={self.n}, layers={len(self.layers)})"