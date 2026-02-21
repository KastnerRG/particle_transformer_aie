import numpy as np
from typing import List

from .base import AIELayer
from utils.softmax_lut import build_exp_lut_256, softmax_lut_int8_rowwise, lut_to_c_initializer


class SoftmaxLayer(AIELayer):

    def __init__(self, name: str, out_scale: int = 127):
        super().__init__(name, 'softmax', params={'out_scale': out_scale})
        self.out_scale = out_scale
        self.m = None
        self.k = None
        self.n = None
        self.exp_lut = build_exp_lut_256()

    def _compute_golden(self, inputs: List[np.ndarray]) -> np.ndarray:
        self.validate_inputs(inputs, expected_count=1)
        x = inputs[0]

        assert self.m is not None and self.n is not None, \
            "Tiling parameters not set. Layer must be added to AIEModel first."
        assert x.shape[0] % self.m == 0, f"Row dimension {x.shape[0]} must be divisible by m={self.m}"
        assert x.shape[1] % self.n == 0, f"Col dimension {x.shape[1]} must be divisible by n={self.n}"

        y = softmax_lut_int8_rowwise(x.astype(np.int8), exp_lut=self.exp_lut, out_scale=self.out_scale)

        self.outputs['x'] = x.astype(np.int8)
        self.outputs['a'] = y
        self._golden_computed = True
        return y

    def generate_kernel_code(self, f) -> None:
        batch, features = self.outputs['x'].shape
        t_m = batch // self.m
        t_n = features // self.n

        f.write('#include <cstdint>\n')
        f.write(f'__attribute__((section(".data"))) alignas(32) int16_t lut_p [256] = {{ ')
        f.write(lut_to_c_initializer(self.exp_lut))
        f.write(' };\n\n')
        f.write('#include "kernels.h"\n\n')
        f.write(f'void f{self.idx}(input_stream_int8 * __restrict x, output_stream_int8 * __restrict a){{ ')
        f.write(f'softmax_rows<{self.m}, {self.n}, {t_m}, {t_n}, {self.out_scale}>(x, a, lut_p);')
        f.write(' }\n')

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
