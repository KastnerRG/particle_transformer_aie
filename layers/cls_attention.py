import numpy as np
from typing import List, Optional

from .base import AIELayer
from utils.tiling import tile_matrix
from utils.np_mha_linear import NumpyMHALinear
from utils.softmax_lut import build_exp_lut_256, lut_to_c_initializer


class CLSAttentionLayer(AIELayer):
    """
    CLS cross-attention:
      - Mode A: internal CLS query + particle K/V input.
      - Mode B: explicit CLS input + particle K/V input.
    """

    def __init__(
        self,
        name: str,
        Wq: np.ndarray,
        Wk: np.ndarray,
        Wv: np.ndarray,
        Wo: np.ndarray,
        num_heads: int,
        d_model: int,
        T_kv: int,
        T_q: int,
        internal_cls_query: bool,
        cls_token: Optional[np.ndarray] = None,
        bq: Optional[np.ndarray] = None,
        bk: Optional[np.ndarray] = None,
        bv: Optional[np.ndarray] = None,
        bo: Optional[np.ndarray] = None,
        use_softmax: bool = True,
        add_query_residual: bool = False,
        output_relu: bool = False,
    ):
        super().__init__(name, 'cls_attention', params={
            'Wq': Wq,
            'Wk': Wk,
            'Wv': Wv,
            'Wo': Wo,
            'bq': bq,
            'bk': bk,
            'bv': bv,
            'bo': bo,
            'num_heads': num_heads,
            'd_model': d_model,
            'T_kv': T_kv,
            'T_q': T_q,
            'internal_cls_query': internal_cls_query,
            'use_softmax': use_softmax,
            'add_query_residual': add_query_residual,
            'output_relu': output_relu,
        })

        self.Wq = Wq.astype(np.int8)
        self.Wk = Wk.astype(np.int8)
        self.Wv = Wv.astype(np.int8)
        self.Wo = Wo.astype(np.int8)
        self.bq = None if bq is None else np.clip(bq, -128, 127).astype(np.int8)
        self.bk = None if bk is None else np.clip(bk, -128, 127).astype(np.int8)
        self.bv = None if bv is None else np.clip(bv, -128, 127).astype(np.int8)
        self.bo = None if bo is None else np.clip(bo, -128, 127).astype(np.int8)

        self.num_heads = num_heads
        self.d_model = d_model
        self.T_kv = T_kv
        self.T_q = T_q
        self.head_dim = d_model // num_heads
        self.internal_cls_query = internal_cls_query
        self.add_query_residual = add_query_residual
        self.use_softmax = use_softmax
        self.output_relu = output_relu

        self.cls_token = None if cls_token is None else np.clip(cls_token, -128, 127).astype(np.int8)
        if self.internal_cls_query:
            assert self.cls_token is not None, "cls_token is required for internal CLS query mode."
            assert self.cls_token.shape == (self.d_model,), \
                f"cls_token shape must be ({self.d_model},), got {self.cls_token.shape}"

        assert d_model % num_heads == 0, f"d_model {d_model} must be divisible by num_heads {num_heads}"
        assert num_heads in [1, 4], f"Only num_heads=1 or 4 supported, got {num_heads}"

        self.m = None
        self.k = None
        self.n = None

        self.shift_q = None
        self.shift_k = None
        self.shift_v = None
        self.shift_s = None
        self.shift_c = None
        self.shift_o = None

        self.Wq_heads = []
        self.Wk_heads = []
        self.Wv_heads = []
        self.Wo_tiled = None

        self.bq_heads = []
        self.bk_heads = []
        self.bv_heads = []
        self.q_proj_heads_const_tiled = []

        self.softmax_lut = build_exp_lut_256()

    def _split_head_bias(self, b: Optional[np.ndarray]) -> List[Optional[np.ndarray]]:
        if b is None:
            return [None for _ in range(self.num_heads)]
        assert b.shape == (self.d_model,), f"Expected bias shape ({self.d_model},), got {b.shape}"
        out = []
        for h in range(self.num_heads):
            s = h * self.head_dim
            e = (h + 1) * self.head_dim
            out.append(b[s:e].astype(np.int8))
        return out

    def _compute_golden(self, inputs: List[np.ndarray]) -> np.ndarray:
        expected_count = 1 if self.internal_cls_query else 2
        self.validate_inputs(inputs, expected_count=expected_count)

        assert self.m is not None and self.k is not None and self.n is not None, \
            "Tiling parameters not set. Layer must be added to AIEModel first."
        assert self.T_q % self.m == 0, f"T_q={self.T_q} must be divisible by m={self.m}"
        assert self.T_kv % self.m == 0, f"T_kv={self.T_kv} must be divisible by m={self.m}"

        if self.internal_cls_query:
            kv = inputs[0]
            q_in = np.repeat(self.cls_token.reshape(1, -1), self.T_q, axis=0).astype(np.int8)
        else:
            q_in = inputs[0]
            kv = inputs[1]

        assert q_in.shape == (self.T_q, self.d_model), \
            f"Expected CLS input shape ({self.T_q}, {self.d_model}), got {q_in.shape}"
        assert kv.shape == (self.T_kv, self.d_model), \
            f"Expected KV input shape ({self.T_kv}, {self.d_model}), got {kv.shape}"

        layers_list = []
        mha = NumpyMHALinear(
            d_model=self.d_model,
            num_heads=self.num_heads,
            name_prefix=self.name,
            Wq=self.Wq,
            Wk=self.Wk,
            Wv=self.Wv,
            Wo=self.Wo,
            bq=self.bq,
            bk=self.bk,
            bv=self.bv,
            bo=self.bo,
            use_softmax=self.use_softmax,
            output_relu=self.output_relu,
            exp_lut=self.softmax_lut,
        )

        out = mha(q_in, kv, kv, layers=layers_list)

        layer_q = layers_list[0]
        layer_k = layers_list[1]
        layer_v = layers_list[2]
        layer_o = layers_list[3]

        self.shift_q = layer_q['shift']
        self.shift_k = layer_k['shift']
        self.shift_v = layer_v['shift']
        self.shift_s = layer_q['shift_scores']
        self.shift_c = layer_q['shift_context']
        self.shift_o = layer_o['shift']

        q_proj_full = layer_q['a'].reshape(self.T_q, self.d_model).astype(np.int8)

        self.Wq_heads = []
        self.Wk_heads = []
        self.Wv_heads = []
        self.q_proj_heads_const_tiled = []

        for h in range(self.num_heads):
            s = h * self.head_dim
            e = (h + 1) * self.head_dim
            self.Wq_heads.append(tile_matrix(self.Wq[:, s:e], self.k, self.n))
            self.Wk_heads.append(tile_matrix(self.Wk[:, s:e], self.k, self.n))
            self.Wv_heads.append(tile_matrix(self.Wv[:, s:e], self.k, self.n))
            self.q_proj_heads_const_tiled.append(tile_matrix(q_proj_full[:, s:e], self.m, self.n))

        self.Wo_tiled = tile_matrix(self.Wo, self.k, self.n)
        self.bq_heads = self._split_head_bias(self.bq)
        self.bk_heads = self._split_head_bias(self.bk)
        self.bv_heads = self._split_head_bias(self.bv)

        if self.add_query_residual:
            out = np.clip(out.astype(np.int32) + q_in.astype(np.int32), -128, 127).astype(np.int8)

        self.outputs['q_in'] = q_in
        self.outputs['kv_in'] = kv
        self.outputs['a'] = out
        self._golden_computed = True
        return out

    def _write_q_kernel(self, h: int):
        if self.internal_cls_query:
            with open(f"aie/layer_{self.idx}_q_head{h}.cc", "w") as fq:
                fq.write('#include <cstdint>\n')
                q_const = self.q_proj_heads_const_tiled[h]
                fq.write(f'__attribute__((section(".data"))) alignas(32) int8_t q_p [{q_const.size}] = {{ ')
                fq.write(', '.join(str(int(x)) for x in q_const))
                fq.write(' };\n\n')
                fq.write('#include "kernels.h"\n\n')
                fq.write(
                    f'void q{self.idx}_head{h}(output_stream_int8 * __restrict a){{ '
                    f'emit_const<{self.m}, {self.n}, {self.T_q//self.m}, {self.head_dim//self.n}>(a, q_p);'
                    f'}}\n'
                )
        else:
            with open(f"aie/layer_{self.idx}_q_head{h}.cc", "w") as fq:
                fq.write('#include <cstdint>\n')
                fq.write(f'__attribute__((section(".data"))) alignas(32) int8_t k_p [{self.Wq_heads[h].size}] = {{ ')
                fq.write(', '.join(str(int(x)) for x in self.Wq_heads[h]))
                fq.write(' };\n\n')
                if self.bq_heads[h] is not None:
                    fq.write(f'__attribute__((section(".data"))) alignas(32) int8_t b_p [{self.bq_heads[h].size}] = {{ ')
                    fq.write(', '.join(str(int(x)) for x in self.bq_heads[h]))
                    fq.write(' };\n\n')
                fq.write('#include "kernels.h"\n\n')
                fq.write(f'void q{self.idx}_head{h}(input_stream_int8 * __restrict x, output_stream_int8 * __restrict a){{ ')
                if self.bq_heads[h] is not None:
                    fq.write(
                        f'dense_bias<{self.m}, {self.k}, {self.n}, {self.T_q//self.m}, {self.d_model//self.k}, '
                        f'{self.head_dim//self.n}, {self.shift_q}, false>(x, a, k_p, b_p);'
                    )
                else:
                    fq.write(
                        f'dense<{self.m}, {self.k}, {self.n}, {self.T_q//self.m}, {self.d_model//self.k}, '
                        f'{self.head_dim//self.n}, {self.shift_q}, false>(x, a, k_p);'
                    )
                fq.write('}\n')

    def _write_kv_kernel(self, which: str, h: int, weight_tiled: np.ndarray, bias: Optional[np.ndarray], shift: int):
        with open(f"aie/layer_{self.idx}_{which}_head{h}.cc", "w") as f:
            f.write('#include <cstdint>\n')
            f.write(f'__attribute__((section(".data"))) alignas(32) int8_t k_p [{weight_tiled.size}] = {{ ')
            f.write(', '.join(str(int(x)) for x in weight_tiled))
            f.write(' };\n\n')
            if bias is not None:
                f.write(f'__attribute__((section(".data"))) alignas(32) int8_t b_p [{bias.size}] = {{ ')
                f.write(', '.join(str(int(x)) for x in bias.reshape(-1)))
                f.write(' };\n\n')
            f.write('#include "kernels.h"\n\n')
            f.write(f'void {which}{self.idx}_head{h}(input_stream_int8 * __restrict x, output_stream_int8 * __restrict a){{ ')
            if bias is not None:
                f.write(
                    f'dense_bias<{self.m}, {self.k}, {self.n}, {self.T_kv//self.m}, {self.d_model//self.k}, '
                    f'{self.head_dim//self.n}, {shift}, false>(x, a, k_p, b_p);'
                )
            else:
                f.write(
                    f'dense<{self.m}, {self.k}, {self.n}, {self.T_kv//self.m}, {self.d_model//self.k}, '
                    f'{self.head_dim//self.n}, {shift}, false>(x, a, k_p);'
                )
            f.write('}\n')

    def generate_kernel_code(self, f) -> None:
        assert self._golden_computed, "Must call compute_golden() before generating code"
        try:
            f.write(f"// CLS cross-attention layer {self.idx}\n")
        except Exception:
            pass

        for h in range(self.num_heads):
            self._write_q_kernel(h)
            self._write_kv_kernel('k', h, self.Wk_heads[h], self.bk_heads[h], self.shift_k)
            self._write_kv_kernel('v', h, self.Wv_heads[h], self.bv_heads[h], self.shift_v)

            with open(f"aie/layer_{self.idx}_scores_head{h}.cc", "w") as fs:
                fs.write('#include "kernels.h"\n\n')
                fs.write(
                    f'void scores{self.idx}_head{h}(input_stream_int8 * __restrict q_head, '
                    f'input_stream_int8 * __restrict k_head, output_stream_int8 * __restrict o_head){{ '
                )
                fs.write(
                    f'scores_cross<{self.m}, {self.k}, {self.n}, {self.T_q//self.m}, {self.T_kv//self.m}, '
                    f'{self.head_dim//self.n}, {self.head_dim}, {self.shift_s[h]}>(q_head, k_head, o_head);'
                )
                fs.write('}\n')

            if self.use_softmax:
                with open(f"aie/layer_{self.idx}_softmax_head{h}.cc", "w") as fsm:
                    fsm.write('#include <cstdint>\n')
                    fsm.write(f'__attribute__((section(".data"))) alignas(32) int16_t lut_p [256] = {{ ')
                    fsm.write(lut_to_c_initializer(self.softmax_lut))
                    fsm.write(' };\n\n')
                    fsm.write('#include "kernels.h"\n\n')
                    fsm.write(
                        f'void softmax{self.idx}_head{h}(input_stream_int8 * __restrict x, output_stream_int8 * __restrict y){{ '
                    )
                    fsm.write(
                        f'softmax_rows<{self.m}, {self.m}, {self.T_q//self.m}, {self.T_kv//self.m}>(x, y, lut_p);'
                    )
                    fsm.write('}\n')

            with open(f"aie/layer_{self.idx}_context_head{h}.cc", "w") as fc:
                fc.write('#include "kernels.h"\n\n')
                fc.write(
                    f'void context{self.idx}_head{h}(input_stream_int8 * __restrict x, input_stream_int8 * __restrict v, '
                    f'output_stream_int8 * __restrict a){{ '
                )
                fc.write(
                    f'context_cross<{self.m}, {self.n}, {self.T_q//self.m}, {self.T_kv//self.m}, '
                    f'{self.head_dim//self.n}, {self.shift_c[h]}>(x, v, a);'
                )
                fc.write('}\n')

        if self.num_heads == 4:
            with open(f"aie/layer_{self.idx}_concat.cc", "w") as fc:
                fc.write('#include "kernels.h"\n\n')
                fc.write(
                    f'void concat{self.idx}_0(input_stream_int8 * __restrict sA, input_stream_int8 * __restrict sB, '
                    f'output_stream_int8 * __restrict sC){{\n'
                )
                fc.write(f'concat<{self.m}, {self.n}, {self.T_q//self.m}, {self.head_dim//self.n}>(sA, sB, sC);\n')
                fc.write('}\n\n')
                fc.write(
                    f'void concat{self.idx}_1(input_stream_int8 * __restrict sA, input_stream_int8 * __restrict sB, '
                    f'output_stream_int8 * __restrict sC){{\n'
                )
                fc.write(f'concat<{self.m}, {self.n}, {self.T_q//self.m}, {self.head_dim//self.n}>(sA, sB, sC);\n')
                fc.write('}\n')

            with open(f"aie/layer_{self.idx}_out.cc", "w") as fo:
                fo.write('#include <cstdint>\n')
                fo.write(f'__attribute__((section(".data"))) alignas(32) int8_t k_p [{self.Wo_tiled.size}] = {{ ')
                fo.write(', '.join(str(int(x)) for x in self.Wo_tiled))
                fo.write(' };\n\n')
                if self.bo is not None:
                    fo.write(f'__attribute__((section(".data"))) alignas(32) int8_t b_p [{self.bo.size}] = {{ ')
                    fo.write(', '.join(str(int(x)) for x in self.bo.reshape(-1)))
                    fo.write(' };\n\n')
                if self.add_query_residual:
                    fo.write(f'__attribute__((section(".data"))) alignas(32) int8_t r_p [{self.cls_token.size}] = {{ ')
                    fo.write(', '.join(str(int(x)) for x in self.cls_token.reshape(-1)))
                    fo.write(' };\n\n')
                fo.write('#include "kernels.h"\n\n')
                fo.write(
                    f'void out{self.idx}(input_stream_int8 * __restrict sA, input_stream_int8 * __restrict sB, '
                    f'output_stream_int8 * __restrict a){{'
                )
                is_relu = 'true' if self.output_relu else 'false'
                use_bias = 'true' if self.bo is not None else 'false'
                use_residual = 'true' if self.add_query_residual else 'false'
                bias_ptr = 'b_p' if self.bo is not None else 'nullptr'
                res_ptr = 'r_p' if self.add_query_residual else 'nullptr'
                fo.write(
                    f'output<{self.m}, {self.k}, {self.n}, {self.T_q//self.m}, {self.d_model//self.k}, {self.d_model//self.n}, '
                    f'{self.shift_o}, {is_relu}, {use_bias}, {use_residual}>(sA, sB, a, k_p, {bias_ptr}, {res_ptr});'
                )
                fo.write('}\n')

        elif self.num_heads == 1:
            with open(f"aie/layer_{self.idx}_out.cc", "w") as fo:
                fo.write('#include <cstdint>\n')
                fo.write(f'__attribute__((section(".data"))) alignas(32) int8_t k_p [{self.Wo_tiled.size}] = {{ ')
                fo.write(', '.join(str(int(x)) for x in self.Wo_tiled))
                fo.write(' };\n\n')
                if self.bo is not None:
                    fo.write(f'__attribute__((section(".data"))) alignas(32) int8_t b_p [{self.bo.size}] = {{ ')
                    fo.write(', '.join(str(int(x)) for x in self.bo.reshape(-1)))
                    fo.write(' };\n\n')
                fo.write('#include "kernels.h"\n\n')
                fo.write(f'void out{self.idx}(input_stream_int8 * __restrict x, output_stream_int8 * __restrict a){{ ')
                if self.bo is not None:
                    fo.write(
                        f'dense_bias<{self.m}, {self.k}, {self.n}, {self.T_q//self.m}, {self.d_model//self.k}, '
                        f'{self.d_model//self.n}, {self.shift_o}, false>(x, a, k_p, b_p);'
                    )
                else:
                    fo.write(
                        f'dense<{self.m}, {self.k}, {self.n}, {self.T_q//self.m}, {self.d_model//self.k}, '
                        f'{self.d_model//self.n}, {self.shift_o}, false>(x, a, k_p);'
                    )
                fo.write('}\n')

        self._generate_include_code()

    def _generate_include_code(self) -> None:
        with open("aie/include.h", "a") as f:
            for h in range(self.num_heads):
                if self.internal_cls_query:
                    f.write(f'void q{self.idx}_head{h}(output_stream_int8 * __restrict);\n')
                else:
                    f.write(f'void q{self.idx}_head{h}(input_stream_int8 * __restrict, output_stream_int8 * __restrict);\n')
                f.write(f'void k{self.idx}_head{h}(input_stream_int8 * __restrict, output_stream_int8 * __restrict);\n')
                f.write(f'void v{self.idx}_head{h}(input_stream_int8 * __restrict, output_stream_int8 * __restrict);\n')
                f.write(
                    f'void scores{self.idx}_head{h}(input_stream_int8 * __restrict, input_stream_int8 * __restrict, '
                    f'output_stream_int8 * __restrict);\n'
                )
                if self.use_softmax:
                    f.write(f'void softmax{self.idx}_head{h}(input_stream_int8 * __restrict, output_stream_int8 * __restrict);\n')
                f.write(
                    f'void context{self.idx}_head{h}(input_stream_int8 * __restrict, input_stream_int8 * __restrict, '
                    f'output_stream_int8 * __restrict);\n'
                )

            if self.num_heads == 4:
                f.write(f'void concat{self.idx}_0(input_stream_int8 * __restrict, input_stream_int8 * __restrict, output_stream_int8 * __restrict);\n')
                f.write(f'void concat{self.idx}_1(input_stream_int8 * __restrict, input_stream_int8 * __restrict, output_stream_int8 * __restrict);\n')
                f.write(f'void out{self.idx}(input_stream_int8 * __restrict, input_stream_int8 * __restrict, output_stream_int8 * __restrict);\n\n')
            else:
                f.write(f'void out{self.idx}(input_stream_int8 * __restrict, output_stream_int8 * __restrict);\n\n')

    def _head_kernel_stride(self) -> int:
        return 6 if self.use_softmax else 5

    def generate_graph_code(self, f, input_ports: List[str]) -> None:
        expected = 1 if self.internal_cls_query else 2
        self.validate_inputs(input_ports, expected_count=expected)

        if self.internal_cls_query:
            token_port = input_ports[0]
            cls_port = None
        else:
            cls_port = input_ports[0]
            token_port = input_ports[1]

        stride = self._head_kernel_stride()
        for h in range(self.num_heads):
            base = h * stride
            q_idx = base + 0
            k_idx = base + 1
            v_idx = base + 2
            scores_idx = base + 3
            softmax_idx = base + 4 if self.use_softmax else None
            context_idx = base + 5 if self.use_softmax else base + 4

            f.write(f"        {self.name}[{q_idx}] = kernel::create(::q{self.idx}_head{h});\n")
            f.write(f'        source({self.name}[{q_idx}]) = "layer_{self.idx}_q_head{h}.cc";\n')
            f.write(f'        runtime<ratio>({self.name}[{q_idx}]) = 1.0;\n')
            if not self.internal_cls_query:
                f.write(f"        connect<stream>({cls_port}.out[0], {self.name}[{q_idx}].in[0]);\n")
            f.write('\n')

            f.write(f"        {self.name}[{k_idx}] = kernel::create(::k{self.idx}_head{h});\n")
            f.write(f'        source({self.name}[{k_idx}]) = "layer_{self.idx}_k_head{h}.cc";\n')
            f.write(f'        runtime<ratio>({self.name}[{k_idx}]) = 1.0;\n')
            f.write(f"        connect<stream>({token_port}.out[0], {self.name}[{k_idx}].in[0]);\n\n")

            f.write(f"        {self.name}[{v_idx}] = kernel::create(::v{self.idx}_head{h});\n")
            f.write(f'        source({self.name}[{v_idx}]) = "layer_{self.idx}_v_head{h}.cc";\n')
            f.write(f'        runtime<ratio>({self.name}[{v_idx}]) = 1.0;\n')
            f.write(f"        connect<stream>({token_port}.out[0], {self.name}[{v_idx}].in[0]);\n\n")

            f.write(f"        {self.name}[{scores_idx}] = kernel::create(::scores{self.idx}_head{h});\n")
            f.write(f'        source({self.name}[{scores_idx}]) = "layer_{self.idx}_scores_head{h}.cc";\n')
            f.write(f'        runtime<ratio>({self.name}[{scores_idx}]) = 1.0;\n')
            f.write(f"        connect<stream> s{self.idx}_{h}_qk({self.name}[{q_idx}].out[0], {self.name}[{scores_idx}].in[0]);\n")
            f.write(f"        fifo_depth(s{self.idx}_{h}_qk) = {int(self.T_q * self.head_dim / 4)};\n")
            f.write(f"        connect<stream>({self.name}[{k_idx}].out[0], {self.name}[{scores_idx}].in[1]);\n\n")

            if self.use_softmax:
                f.write(f"        {self.name}[{softmax_idx}] = kernel::create(::softmax{self.idx}_head{h});\n")
                f.write(f'        source({self.name}[{softmax_idx}]) = "layer_{self.idx}_softmax_head{h}.cc";\n')
                f.write(f'        runtime<ratio>({self.name}[{softmax_idx}]) = 1.0;\n')
                f.write(f"        connect<stream>({self.name}[{scores_idx}].out[0], {self.name}[{softmax_idx}].in[0]);\n\n")

            f.write(f"        {self.name}[{context_idx}] = kernel::create(::context{self.idx}_head{h});\n")
            f.write(f'        source({self.name}[{context_idx}]) = "layer_{self.idx}_context_head{h}.cc";\n')
            f.write(f'        runtime<ratio>({self.name}[{context_idx}]) = 1.0;\n')
            src_for_context = softmax_idx if self.use_softmax else scores_idx
            f.write(
                f"        connect<stream> s{self.idx}_{h}_sv({self.name}[{src_for_context}].out[0], "
                f"{self.name}[{context_idx}].in[0]);\n"
            )
            f.write(f"        fifo_depth(s{self.idx}_{h}_sv) = {int(self.T_q * self.T_kv / 4)};\n")
            f.write(f"        connect<stream>({self.name}[{v_idx}].out[0], {self.name}[{context_idx}].in[1]);\n\n")

        if self.num_heads == 4:
            concat_0_idx = self.num_heads * stride
            concat_1_idx = self.num_heads * stride + 1
            out_idx = self.num_heads * stride + 2

            head0_ctx = 0 * stride + (5 if self.use_softmax else 4)
            head1_ctx = 1 * stride + (5 if self.use_softmax else 4)
            head2_ctx = 2 * stride + (5 if self.use_softmax else 4)
            head3_ctx = 3 * stride + (5 if self.use_softmax else 4)

            f.write(f"        {self.name}[{concat_0_idx}] = kernel::create(::concat{self.idx}_0);\n")
            f.write(f'        source({self.name}[{concat_0_idx}]) = "layer_{self.idx}_concat.cc";\n')
            f.write(f'        runtime<ratio>({self.name}[{concat_0_idx}]) = 1.0;\n')
            f.write(f"        connect<stream>({self.name}[{head0_ctx}].out[0], {self.name}[{concat_0_idx}].in[0]);\n")
            f.write(f"        connect<stream>({self.name}[{head1_ctx}].out[0], {self.name}[{concat_0_idx}].in[1]);\n\n")

            f.write(f"        {self.name}[{concat_1_idx}] = kernel::create(::concat{self.idx}_1);\n")
            f.write(f'        source({self.name}[{concat_1_idx}]) = "layer_{self.idx}_concat.cc";\n')
            f.write(f'        runtime<ratio>({self.name}[{concat_1_idx}]) = 1.0;\n')
            f.write(f"        connect<stream>({self.name}[{head2_ctx}].out[0], {self.name}[{concat_1_idx}].in[0]);\n")
            f.write(f"        connect<stream>({self.name}[{head3_ctx}].out[0], {self.name}[{concat_1_idx}].in[1]);\n\n")

            f.write(f"        {self.name}[{out_idx}] = kernel::create(::out{self.idx});\n")
            f.write(f'        source({self.name}[{out_idx}]) = "layer_{self.idx}_out.cc";\n')
            f.write(f'        runtime<ratio>({self.name}[{out_idx}]) = 1.0;\n')
            f.write(f"        connect<stream>({self.name}[{concat_0_idx}].out[0], {self.name}[{out_idx}].in[0]);\n")
            f.write(f"        connect<stream>({self.name}[{concat_1_idx}].out[0], {self.name}[{out_idx}].in[1]);\n\n")

        elif self.num_heads == 1:
            out_idx = stride
            context_idx = stride - 1
            f.write(f"        {self.name}[{out_idx}] = kernel::create(::out{self.idx});\n")
            f.write(f'        source({self.name}[{out_idx}]) = "layer_{self.idx}_out.cc";\n')
            f.write(f'        runtime<ratio>({self.name}[{out_idx}]) = 1.0;\n')
            f.write(f"        connect<stream>({self.name}[{context_idx}].out[0], {self.name}[{out_idx}].in[0]);\n\n")

    def num_kernels(self) -> int:
        stride = self._head_kernel_stride()
        if self.num_heads == 4:
            return self.num_heads * stride + 3
        if self.num_heads == 1:
            return self.num_heads * stride + 1
        raise ValueError(f"Unsupported num_heads: {self.num_heads}")

    def get_output_port(self, port_idx: int = 0) -> str:
        return f"{self.name}[{self.num_kernels() - 1}]"
