import numpy as np
from typing import List, Optional
from .base import AIELayer
from utils.tiling import tile_matrix


class DenseLayer(AIELayer):

    def __init__(
        self,
        name: str,
        weight: np.ndarray,
        shift: int,
        relu: bool = False
    ):
        """
        Initialize dense layer.

        Args:
            name: Layer name
            weight: Weight matrix (int8), shape (in_features, out_features)
            shift: Right shift for requantization
            relu: Apply ReLU activation

        Note: Tiling parameters (m, k, n) are set by AIEModel when layer is added.
        """
        super().__init__(name, 'dense', params={
            'weight': weight,
            'shift': shift,
            'relu': relu
        })

        self.weight = weight
        self.shift = shift
        self.relu = relu

        self.m = None
        self.k = None
        self.n = None

    def _compute_golden(self, inputs: List[np.ndarray]) -> np.ndarray:
        self.validate_inputs(inputs, expected_count=1)
        x = inputs[0]

        assert self.m is not None and self.k is not None and self.n is not None, \
            f"Tiling parameters not set. Layer must be added to AIEModel first."

        in_features, out_features = self.weight.shape
        assert in_features % self.k == 0, \
            f"Weight in_features {in_features} must be divisible by k={self.k}"
        assert out_features % self.n == 0, \
            f"Weight out_features {out_features} must be divisible by n={self.n}"
        assert x.shape[1] % self.k == 0, \
            f"Input dimension {x.shape[1]} must be divisible by k={self.k}"
        assert x.shape[0] % self.m == 0, \
            f"Batch dimension {x.shape[0]} must be divisible by m={self.m}"

        y = np.matmul(x.astype(np.int32), self.weight.astype(np.int32))

        y = (y >> self.shift).astype(np.int8)

        if self.relu:
            a = np.maximum(0, y)
        else:
            a = y

        self.outputs['x'] = x
        self.outputs['y'] = y
        self.outputs['a'] = a

        return a

    def generate_kernel_code(self, f) -> None:
        weight_tiled = tile_matrix(self.weight, self.k, self.n)

        in_features, out_features = self.weight.shape

        batch = self.outputs['x'].shape[0]
        t_m = batch // self.m

        t_k = in_features // self.k
        t_n = out_features // self.n

        f.write('#include <cstdint>\n')
        f.write(f'__attribute__((section(".data"))) alignas(32) int8_t k_p [{weight_tiled.size}] = {{ ')
        f.write(', '.join(str(int(x)) for x in weight_tiled))
        f.write(' };\n\n')

        f.write('#include "kernels.h"\n\n')

        relu_str = 'true' if self.relu else 'false'
        f.write(f'void f{self.idx}(input_stream_int8 * __restrict x, output_stream_int8 * __restrict a){{ ')
        f.write(f'dense<{self.m}, {self.k}, {self.n}, {t_m}, {t_k}, {t_n}, {self.shift}, {relu_str}>')
        f.write(' (x, a, k_p);}\n')

        self._generate_include_code()

    def _generate_include_code(self) -> None:
        with open("aie/include.h", "a") as f:
            f.write(f'void f{self.idx}(input_stream_int8 * __restrict, output_stream_int8 * __restrict);\n')

    def generate_graph_code(self, f, input_ports: List[str]) -> None:
        self.validate_inputs(input_ports, expected_count=1)
        in_port = input_ports[0]
        f.write(f"        {self.name}[0] = kernel::create(::f{self.idx});\n")
        f.write(f'        source({self.name}[0]) = "layer_{self.idx}.cc";\n')
        f.write(f'        runtime<ratio>({self.name}[0]) = 1.0;\n')

        f.write(f"        connect<stream>({in_port}.out[0], {self.name}[0].in[0]);\n\n")

    def num_kernels(self) -> int:
        return 1

    def get_output_port(self, port_idx: int = 0) -> str:
        return f"{self.name}[0]"

    def __repr__(self) -> str:
        """String representation for debugging."""
        in_f, out_f = self.weight.shape
        act = 'ReLU' if self.relu else 'Linear'
        idx_str = f"idx={self.idx}" if self.idx is not None else "idx=unassigned"
        return (f"DenseLayer({idx_str}, name='{self.name}', "
                f"weight={self.weight.shape}, shift={self.shift}, act={act})")