import numpy as np
from typing import List
from .base import AIELayer


class ResAddLayer(AIELayer):

    def __init__(
        self,
        name: str,
        shift: int = 0
    ):
        """
        Initialize residual addition layer.

        Args:
            name: Layer name
            shift: Right shift for requantization (default: 0 for no shift)

        Note: Tiling parameters (m, k, n) are set by AIEModel when layer is added.
        """
        super().__init__(name, 'resadd', params={
            'shift': shift
        })
        # Note: SHIFT is currently not supported in the AIE resadd kernel; kept for future use
        self.shift = shift

        self.m = None
        self.k = None
        self.n = None

    def _compute_golden(self, inputs: List[np.ndarray]) -> np.ndarray:
        """
        Compute residual addition output using NumPy (internal method).

        Args:
            inputs: List containing two input arrays of same shape

        Returns:
            Output array of same shape as inputs
        """
        self.validate_inputs(inputs, expected_count=2)
        x1 = inputs[0]
        x2 = inputs[1]

        assert self.m is not None and self.k is not None and self.n is not None, \
            f"Tiling parameters not set. Layer must be added to AIEModel first."

        assert x1.shape == x2.shape, \
            f"Input shapes must match: x1={x1.shape}, x2={x2.shape}"

        assert x1.shape[0] % self.m == 0, \
            f"Batch dimension {x1.shape[0]} must be divisible by m={self.m}"
        assert x1.shape[1] % self.n == 0, \
            f"Feature dimension {x1.shape[1]} must be divisible by n={self.n}"

        y = x1.astype(np.int32) + x2.astype(np.int32)
        a = np.clip(y, -128, 127).astype(np.int8)

        self.outputs['x1'] = x1 
        self.outputs['x2'] = x2
        self.outputs['y'] = y
        self.outputs['a'] = a

        self._golden_computed = True
        return a

    def generate_kernel_code(self, f) -> None:
        batch, features = self.outputs['x1'].shape
        t_m = batch // self.m
        t_n = features // self.n

        f.write('#include <cstdint>\n')
        f.write('#include "kernels.h"\n\n')

        f.write(f'void f{self.idx}(input_stream_int8 * __restrict x1, input_stream_int8 * __restrict x2, output_stream_int8 * __restrict a){{ ')
        f.write(f'resadd<{self.m}, {self.n}, {t_m}, {t_n}>')
        f.write(' (x1, x2, a);}\n')

        self._generate_include_code()

    def _generate_include_code(self) -> None:
        with open("aie/include.h", "a") as f:
            f.write(f'void f{self.idx}(input_stream_int8 * __restrict, input_stream_int8 * __restrict, output_stream_int8 * __restrict);\n')

    def generate_graph_code(self, f, input_ports: List[str]) -> None:
        self.validate_inputs(input_ports, expected_count=2)
        in_port1 = input_ports[0]
        in_port2 = input_ports[1]

        f.write(f"        {self.name}[0] = kernel::create(::f{self.idx});\n")
        f.write(f'        source({self.name}[0]) = "layer_{self.idx}.cc";\n')
        f.write(f'        runtime<ratio>({self.name}[0]) = 1.0;\n')

        T = self.outputs['x1'].shape[0] if 'x1' in self.outputs else self.m
        d_model = self.outputs['x1'].shape[1] if 'x1' in self.outputs else self.n
        fifo_depth_val = int((T * d_model) / 4)

        f.write(f"        connect<stream> s{self.idx}_in0({in_port1}.out[0], {self.name}[0].in[0]);\n")
        f.write(f"        fifo_depth(s{self.idx}_in0) = {fifo_depth_val};\n")
        f.write(f"        connect<stream> s{self.idx}_in1({in_port2}.out[0], {self.name}[0].in[1]);\n")
        f.write(f"        fifo_depth(s{self.idx}_in1) = {fifo_depth_val};\n\n")

    def num_kernels(self) -> int:
        return 1

    def get_output_port(self, port_idx: int = 0) -> str:
        return f"{self.name}[0]"

    def __repr__(self) -> str:
        """String representation for debugging."""
        idx_str = f"idx={self.idx}" if self.idx is not None else "idx=unassigned"
        return (f"ResAddLayer({idx_str}, name='{self.name}', "
                f"shift={self.shift})")
