import numpy as np
from typing import List, Optional
from .base import AIELayer
from utils.tiling import tile_matrix
from utils.np_mha_linear import NumpyMHALinear


class MHALayer(AIELayer):

    def __init__(
        self,
        name: str,
        Wq: np.ndarray,
        Wk: np.ndarray,
        Wv: np.ndarray,
        Wo: np.ndarray,
        num_heads: int,
        scale_q: Optional[int] = None,
        shift_q: Optional[int] = None,
        scale_k: Optional[int] = None,
        shift_k: Optional[int] = None,
        scale_v: Optional[int] = None,
        shift_v: Optional[int] = None,
        scale_s: Optional[List[int]] = None,
        shift_s: Optional[List[int]] = None,
        scale_c: Optional[List[int]] = None,
        shift_c: Optional[List[int]] = None,
        scale_o: Optional[int] = None,
        shift_o: Optional[int] = None,
        d_model: int = 64,
        T: int = 160,
        enable_softmax: bool = True,
    ):
        """
        Initialize MHA layer.

        Args:
            name: Layer name
            Wq: Query weight matrix (d_model, d_model), int8
            Wk: Key weight matrix (d_model, d_model), int8
            Wv: Value weight matrix (d_model, d_model), int8
            Wo: Output weight matrix (d_model, d_model), int8
            num_heads: Number of attention heads (1 or 4)
            scale_*/shift_*: Requantization values. If omitted, inferred dynamically for testing.
            d_model: Model dimension
            T: Sequence length (padded)
            enable_softmax: If False, skip score softmax and use raw quantized scores

        Note: Tiling parameters (m, k, n) are set by AIEModel when layer is added.
        """
        super().__init__(name, 'mha', params={
            'Wq': Wq,
            'Wk': Wk,
            'Wv': Wv,
            'Wo': Wo,
            'num_heads': num_heads,
            'd_model': d_model,
            'T': T,
            'enable_softmax': enable_softmax,
        })

        self.Wq = Wq
        self.Wk = Wk
        self.Wv = Wv
        self.Wo = Wo
        self.num_heads = num_heads
        self.d_model = d_model
        self.T = T
        self.enable_softmax = bool(enable_softmax)
        self.head_dim = d_model // num_heads

        self.m = None
        self.k = None
        self.n = None

        assert d_model % num_heads == 0, f"d_model {d_model} must be divisible by num_heads {num_heads}"
        assert num_heads in [1, 4], f"Only num_heads=1 or 4 supported, got {num_heads}"

        self.layer_q = None
        self.layer_k = None
        self.layer_v = None
        self.layer_o = None
        self.scale_q = scale_q
        self.shift_q = shift_q
        self.scale_k = scale_k
        self.shift_k = shift_k
        self.scale_v = scale_v
        self.shift_v = shift_v
        self.scale_s = scale_s if scale_s is not None else [1 for _ in range(self.num_heads)]
        self.shift_s = shift_s if shift_s is not None else [15 for _ in range(self.num_heads)]
        self.scale_c = scale_c if scale_c is not None else [1 for _ in range(self.num_heads)]
        self.shift_c = shift_c if shift_c is not None else [15 for _ in range(self.num_heads)]
        self.scale_o = scale_o
        self.shift_o = shift_o
        self.dynamic_quant = False

        self.Wq_heads = []
        self.Wk_heads = []
        self.Wv_heads = []
        self.Wo_tiled = None

    def _compute_golden(self, inputs: List[np.ndarray]) -> np.ndarray:
        self.validate_inputs(inputs, expected_count=1)
        x = inputs[0]

        assert self.m is not None and self.k is not None and self.n is not None, \
            f"Tiling parameters not set. Layer must be added to AIEModel first."

        assert x.shape == (self.T, self.d_model), \
            f"Expected input shape ({self.T}, {self.d_model}), got {x.shape}"

        layers_list = []
        mha = NumpyMHALinear(
            d_model=self.d_model,
            num_heads=self.num_heads,
            name_prefix=self.name,
            Wq=self.Wq,
            Wk=self.Wk,
            Wv=self.Wv,
            Wo=self.Wo,
            enable_softmax=self.enable_softmax,
            use_dynamic_quant=self.dynamic_quant,
            scale_q=self.scale_q,
            shift_q=self.shift_q,
            scale_k=self.scale_k,
            shift_k=self.shift_k,
            scale_v=self.scale_v,
            shift_v=self.shift_v,
            scale_s=self.scale_s,
            shift_s=self.shift_s,
            scale_c=self.scale_c,
            shift_c=self.shift_c,
            scale_o=self.scale_o,
            shift_o=self.shift_o,
        )

        output = mha(x, x, x, layers=layers_list)

        assert len(layers_list) == 4, f"Expected 4 layers from MHA, got {len(layers_list)}"
        self.layer_q = layers_list[0]
        self.layer_k = layers_list[1]
        self.layer_v = layers_list[2]
        self.layer_o = layers_list[3]
        self.scale_q = self.layer_q['scale']
        self.shift_q = self.layer_q['shift']
        self.scale_k = self.layer_k['scale']
        self.shift_k = self.layer_k['shift']
        self.scale_v = self.layer_v['scale']
        self.shift_v = self.layer_v['shift']
        self.scale_s = self.layer_q['scale_scores']
        self.shift_s = self.layer_q['shift_scores']
        self.scale_c = self.layer_q['scale_context']
        self.shift_c = self.layer_q['shift_context']
        self.scale_o = self.layer_o['scale']
        self.shift_o = self.layer_o['shift']

        # Split and tile weight matrices per head
        for h in range(self.num_heads):
            col_start = h * self.head_dim
            col_end = (h + 1) * self.head_dim

            Wq_h = self.Wq[:, col_start:col_end]
            Wk_h = self.Wk[:, col_start:col_end]
            Wv_h = self.Wv[:, col_start:col_end]

            Wq_h_tiled = tile_matrix(Wq_h, self.k, self.n)
            Wk_h_tiled = tile_matrix(Wk_h, self.k, self.n)
            Wv_h_tiled = tile_matrix(Wv_h, self.k, self.n)

            self.Wq_heads.append(Wq_h_tiled)
            self.Wk_heads.append(Wk_h_tiled)
            self.Wv_heads.append(Wv_h_tiled)

        self.Wo_tiled = tile_matrix(self.Wo, self.k, self.n)

        self.outputs['a'] = output
        self._golden_computed = True

        return output

    def generate_kernel_code(self, f) -> None:
        assert self._golden_computed, "Must call compute_golden() before generating code"

        try:
            f.write(f"// MHA layer {self.idx}: uses per-head/source compilation units (layer_{self.idx}_*.cc).\n")
        except Exception:
            pass

        for h in range(self.num_heads):
            with open(f"aie/layer_{self.idx}_q_head{h}.cc", "w") as fq:
                fq.write('#include <cstdint>\n')
                fq.write(f'__attribute__((section(".data"))) alignas(32) int8_t k_p [{self.Wq_heads[h].size}] = {{ ')
                fq.write(', '.join(str(int(x)) for x in self.Wq_heads[h]))
                fq.write(' };\n\n')
                fq.write('#include "kernels.h"\n\n')
                fq.write(f'void q{self.idx}_head{h}(input_stream_int8 * __restrict x, output_stream_int8 * __restrict a){{ ')
                fq.write(f'dense<{self.m}, {self.k}, {self.n}, {self.T//self.m}, {self.d_model//self.k}, {self.head_dim//self.n}, {self.shift_q}, {self.scale_q}, false>')
                fq.write('(x, a, k_p);}\n')

            with open(f"aie/layer_{self.idx}_k_head{h}.cc", "w") as fk:
                fk.write('#include <cstdint>\n')
                fk.write(f'__attribute__((section(".data"))) alignas(32) int8_t k_p [{self.Wk_heads[h].size}] = {{ ')
                fk.write(', '.join(str(int(x)) for x in self.Wk_heads[h]))
                fk.write(' };\n\n')
                fk.write('#include "kernels.h"\n\n')
                fk.write(f'void k{self.idx}_head{h}(input_stream_int8 * __restrict x, output_stream_int8 * __restrict a){{ ')
                fk.write(f'dense<{self.m}, {self.k}, {self.n}, {self.T//self.m}, {self.d_model//self.k}, {self.head_dim//self.n}, {self.shift_k}, {self.scale_k}, false>')
                fk.write('(x, a, k_p);}\n')

            with open(f"aie/layer_{self.idx}_v_head{h}.cc", "w") as fv:
                fv.write('#include <cstdint>\n')
                fv.write(f'__attribute__((section(".data"))) alignas(32) int8_t k_p [{self.Wv_heads[h].size}] = {{ ')
                fv.write(', '.join(str(int(x)) for x in self.Wv_heads[h]))
                fv.write(' };\n\n')
                fv.write('#include "kernels.h"\n\n')
                fv.write(f'void v{self.idx}_head{h}(input_stream_int8 * __restrict x, output_stream_int8 * __restrict a){{ ')
                fv.write(f'dense<{self.m}, {self.k}, {self.n}, {self.T//self.m}, {self.d_model//self.k}, {self.head_dim//self.n}, {self.shift_v}, {self.scale_v}, false>')
                fv.write('(x, a, k_p);}\n')

            with open(f"aie/layer_{self.idx}_scores_head{h}.cc", "w") as fs:
                fs.write('#include "kernels.h"\n\n')
                fs.write(f'void scores{self.idx}_head{h}(input_stream_int8 * __restrict q_head, input_stream_int8 * __restrict k_head, output_stream_int8 * __restrict o_head){{ ')
                if self.enable_softmax:
                    fs.write(f'scores<{self.m}, {self.k}, {self.n}, {self.T//self.m}, {self.head_dim//self.k}, {self.head_dim//self.n}, {self.head_dim}, {self.T}, {self.shift_q + self.shift_k}, {self.scale_q * self.scale_k}, 0, 1, true>')
                else:
                    fs.write(f'scores<{self.m}, {self.k}, {self.n}, {self.T//self.m}, {self.head_dim//self.k}, {self.head_dim//self.n}, {self.head_dim}, {self.T}, {self.shift_q + self.shift_k}, {self.scale_q * self.scale_k}, {self.shift_s[h]}, {self.scale_s[h]}, false>')
                fs.write('(q_head, k_head, o_head);}\n')

            with open(f"aie/layer_{self.idx}_context_head{h}.cc", "w") as fc:
                fc.write('#include "kernels.h"\n\n')
                fc.write(f'void context{self.idx}_head{h}(input_stream_int8 * __restrict x, input_stream_int8 * __restrict v, output_stream_int8 * __restrict a){{ ')
                fc.write(f'context<{self.m}, {self.k}, {self.n}, {self.T//self.m}, {self.T//self.k}, {self.head_dim//self.n}, {self.shift_c[h]}, {self.scale_c[h]}>')
                fc.write('(x, v, a);}\n')

        # Generate concat and output kernels
        if self.num_heads == 4:
            with open(f"aie/layer_{self.idx}_concat.cc", "w") as fc:
                fc.write('#include "kernels.h"\n\n')
                fc.write(f'// Concat head 0 and head 1: ({self.T},{self.head_dim}) + ({self.T},{self.head_dim}) -> ({self.T},{self.head_dim*2})\n')
                fc.write(f'void concat{self.idx}_0(input_stream_int8 * __restrict sA, input_stream_int8 * __restrict sB, output_stream_int8 * __restrict sC){{\n')
                fc.write(f'concat<{self.m}, {self.n}, {self.T//self.m}, {self.head_dim//self.n}>(sA, sB, sC);\n')
                fc.write('}\n\n')
                fc.write(f'// Concat head 2 and head 3: ({self.T},{self.head_dim}) + ({self.T},{self.head_dim}) -> ({self.T},{self.head_dim*2})\n')
                fc.write(f'void concat{self.idx}_1(input_stream_int8 * __restrict sA, input_stream_int8 * __restrict sB, output_stream_int8 * __restrict sC){{\n')
                fc.write(f'concat<{self.m}, {self.n}, {self.T//self.m}, {self.head_dim//self.n}>(sA, sB, sC);\n')
                fc.write('}\n')

            with open(f"aie/layer_{self.idx}_out.cc", "w") as fo:
                fo.write('#include <cstdint>\n')
                fo.write(f'__attribute__((section(".data"))) alignas(32) int8_t k_p [{self.Wo_tiled.size}] = {{ ')
                fo.write(', '.join(str(int(x)) for x in self.Wo_tiled))
                fo.write(' };\n\n')
                fo.write('#include "kernels.h"\n\n')
                fo.write(f'void out{self.idx}(input_stream_int8 * __restrict sA, input_stream_int8 * __restrict sB, output_stream_int8 * __restrict a){{')
                fo.write(f'output<{self.m}, {self.k}, {self.n}, {self.T//self.m}, {self.d_model//self.k}, {self.d_model//self.n}, {self.shift_o}, {self.scale_o}>')
                fo.write('(sA, sB, a, k_p);}\n')

        elif self.num_heads == 1:
            with open(f"aie/layer_{self.idx}_out.cc", "w") as fo:
                fo.write('#include <cstdint>\n')
                fo.write(f'__attribute__((section(".data"))) alignas(32) int8_t k_p [{self.Wo_tiled.size}] = {{ ')
                fo.write(', '.join(str(int(x)) for x in self.Wo_tiled))
                fo.write(' };\n\n')
                fo.write('#include "kernels.h"\n\n')
                fo.write(f'void out{self.idx}(input_stream_int8 * __restrict x, output_stream_int8 * __restrict a){{ ')
                fo.write(f'dense<{self.m}, {self.k}, {self.n}, {self.T//self.m}, {self.d_model//self.k}, {self.head_dim//self.n}, {self.shift_o}, {self.scale_o}, true>')
                fo.write('(x, a, k_p);}\n')

        self._generate_include_code()

    def _generate_include_code(self) -> None:
        """Append function declarations to include.h (private method)."""
        with open("aie/include.h", "a") as f:
            for h in range(self.num_heads):
                f.write(f'void q{self.idx}_head{h}(input_stream_int8 * __restrict, output_stream_int8 * __restrict);\n')
                f.write(f'void k{self.idx}_head{h}(input_stream_int8 * __restrict, output_stream_int8 * __restrict);\n')
                f.write(f'void v{self.idx}_head{h}(input_stream_int8 * __restrict, output_stream_int8 * __restrict);\n')
                f.write(f'void scores{self.idx}_head{h}(input_stream_int8 * __restrict, input_stream_int8 * __restrict, output_stream_int8 * __restrict);\n')
                f.write(f'void context{self.idx}_head{h}(input_stream_int8 * __restrict, input_stream_int8 * __restrict, output_stream_int8 * __restrict);\n')

            if self.num_heads == 4:
                f.write(f'void concat{self.idx}_0(input_stream_int8 * __restrict, input_stream_int8 * __restrict, output_stream_int8 * __restrict);\n')
                f.write(f'void concat{self.idx}_1(input_stream_int8 * __restrict, input_stream_int8 * __restrict, output_stream_int8 * __restrict);\n')
                f.write(f'void out{self.idx}(input_stream_int8 * __restrict, input_stream_int8 * __restrict, output_stream_int8 * __restrict);\n\n')
            else:
                f.write(f'void out{self.idx}(input_stream_int8 * __restrict, output_stream_int8 * __restrict);\n\n')

    def generate_graph_code(self, f, input_ports: List[str]) -> None:
        """
        Generate graph connectivity code for MHA.

        Args:
            f: File handle to write to
            input_ports: List with single input port name
        """
        self.validate_inputs(input_ports, expected_count=1)
        in_port = input_ports[0]

        for h in range(self.num_heads):
            base = h * 5  # Each head has 5 kernels: q, k, v, scores, context

            q_idx = base + 0
            k_idx = base + 1
            v_idx = base + 2
            scores_idx = base + 3
            context_idx = base + 4

            f.write(f"        {self.name}[{q_idx}] = kernel::create(::q{self.idx}_head{h});\n")
            f.write(f'        source({self.name}[{q_idx}]) = "layer_{self.idx}_q_head{h}.cc";\n')
            f.write(f'        runtime<ratio>({self.name}[{q_idx}]) = 1.0;\n')
            f.write(f"        connect<stream>({in_port}.out[0], {self.name}[{q_idx}].in[0]);\n\n")

            f.write(f"        {self.name}[{k_idx}] = kernel::create(::k{self.idx}_head{h});\n")
            f.write(f'        source({self.name}[{k_idx}]) = "layer_{self.idx}_k_head{h}.cc";\n')
            f.write(f'        runtime<ratio>({self.name}[{k_idx}]) = 1.0;\n')
            f.write(f"        connect<stream>({in_port}.out[0], {self.name}[{k_idx}].in[0]);\n\n")

            f.write(f"        {self.name}[{v_idx}] = kernel::create(::v{self.idx}_head{h});\n")
            f.write(f'        source({self.name}[{v_idx}]) = "layer_{self.idx}_v_head{h}.cc";\n')
            f.write(f'        runtime<ratio>({self.name}[{v_idx}]) = 1.0;\n')
            f.write(f"        connect<stream>({in_port}.out[0], {self.name}[{v_idx}].in[0]);\n\n")

            f.write(f"        {self.name}[{scores_idx}] = kernel::create(::scores{self.idx}_head{h});\n")
            f.write(f'        source({self.name}[{scores_idx}]) = "layer_{self.idx}_scores_head{h}.cc";\n')
            f.write(f'        runtime<ratio>({self.name}[{scores_idx}]) = 1.0;\n')
            f.write(f"        connect<stream> s{self.idx}_{h}_qk({self.name}[{q_idx}].out[0], {self.name}[{scores_idx}].in[0]);\n")
            f.write(f"        fifo_depth(s{self.idx}_{h}_qk) = {int(self.T*self.head_dim/4)};\n")
            f.write(f"        connect<stream>({self.name}[{k_idx}].out[0], {self.name}[{scores_idx}].in[1]);\n\n")

            f.write(f"        {self.name}[{context_idx}] = kernel::create(::context{self.idx}_head{h});\n")
            f.write(f'        source({self.name}[{context_idx}]) = "layer_{self.idx}_context_head{h}.cc";\n')
            f.write(f'        runtime<ratio>({self.name}[{context_idx}]) = 1.0;\n')
            f.write(f"        connect<stream> s{self.idx}_{h}_sv({self.name}[{scores_idx}].out[0], {self.name}[{context_idx}].in[0]);\n")
            f.write(f"        fifo_depth(s{self.idx}_{h}_sv) = {int(self.T*self.T/4)};\n")
            f.write(f"        connect<stream>({self.name}[{v_idx}].out[0], {self.name}[{context_idx}].in[1]);\n\n")

        if self.num_heads == 4:
            concat_0_idx = self.num_heads * 5      # 20
            concat_1_idx = self.num_heads * 5 + 1  # 21
            out_idx = self.num_heads * 5 + 2       # 22

            head0_ctx = 4
            head1_ctx = 9
            head2_ctx = 14
            head3_ctx = 19

            # Create concat_0 kernel: head0 + head1
            f.write(f"        {self.name}[{concat_0_idx}] = kernel::create(::concat{self.idx}_0);\n")
            f.write(f'        source({self.name}[{concat_0_idx}]) = "layer_{self.idx}_concat.cc";\n')
            f.write(f'        runtime<ratio>({self.name}[{concat_0_idx}]) = 1.0;\n')
            f.write(f"        connect<stream>({self.name}[{head0_ctx}].out[0], {self.name}[{concat_0_idx}].in[0]);\n")
            f.write(f"        connect<stream>({self.name}[{head1_ctx}].out[0], {self.name}[{concat_0_idx}].in[1]);\n\n")

            # Create concat_1 kernel: head2 + head3
            f.write(f"        {self.name}[{concat_1_idx}] = kernel::create(::concat{self.idx}_1);\n")
            f.write(f'        source({self.name}[{concat_1_idx}]) = "layer_{self.idx}_concat.cc";\n')
            f.write(f'        runtime<ratio>({self.name}[{concat_1_idx}]) = 1.0;\n')
            f.write(f"        connect<stream>({self.name}[{head2_ctx}].out[0], {self.name}[{concat_1_idx}].in[0]);\n")
            f.write(f"        connect<stream>({self.name}[{head3_ctx}].out[0], {self.name}[{concat_1_idx}].in[1]);\n\n")

            # Create output projection kernel
            f.write(f"        {self.name}[{out_idx}] = kernel::create(::out{self.idx});\n")
            f.write(f'        source({self.name}[{out_idx}]) = "layer_{self.idx}_out.cc";\n')
            f.write(f'        runtime<ratio>({self.name}[{out_idx}]) = 1.0;\n')
            f.write(f"        connect<stream>({self.name}[{concat_0_idx}].out[0], {self.name}[{out_idx}].in[0]);\n")
            f.write(f"        connect<stream>({self.name}[{concat_1_idx}].out[0], {self.name}[{out_idx}].in[1]);\n\n")

        elif self.num_heads == 1:
            out_idx = 5
            context_idx = 4

            f.write(f"        {self.name}[{out_idx}] = kernel::create(::out{self.idx});\n")
            f.write(f'        source({self.name}[{out_idx}]) = "layer_{self.idx}_out.cc";\n')
            f.write(f'        runtime<ratio>({self.name}[{out_idx}]) = 1.0;\n')
            f.write(f'        connect<stream>({self.name}[{context_idx}].out[0], {self.name}[{out_idx}].in[0]);\n\n')

    def num_kernels(self) -> int:
        """Return number of AIE kernels for this MHA layer."""
        if self.num_heads == 4:
            return 23  # 4*5 + 2 concat + 1 output
        elif self.num_heads == 1:
            return 6   # 1*5 + 1 output
        else:
            raise ValueError(f"Unsupported num_heads: {self.num_heads}")

    def get_output_port(self, port_idx: int = 0) -> str:
        """Get output port name (the output projection kernel)."""
        if self.num_heads == 4:
            return f"{self.name}[22]"
        elif self.num_heads == 1:
            return f"{self.name}[5]"
        else:
            raise ValueError(f"Unsupported num_heads: {self.num_heads}")

    def __repr__(self) -> str:
        """String representation for debugging."""
        idx_str = f"idx={self.idx}" if self.idx is not None else "idx=unassigned"
        return (f"MHALayer({idx_str}, name='{self.name}', "
                f"num_heads={self.num_heads}, d_model={self.d_model}, "
                f"num_kernels={self.num_kernels()})")
