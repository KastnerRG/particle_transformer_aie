import numpy as np

from .softmax_lut import build_exp_lut_256, softmax_lut_int8_rowwise


def _choose_shift(acc_int32: np.ndarray) -> int:
    max_abs = int(np.max(np.abs(acc_int32))) if acc_int32.size else 0
    if max_abs <= 127:
        return 0
    shift = 0
    limit = 127
    while max_abs > limit and shift < 31:
        shift += 1
        limit <<= 1
    return shift


def _quantize_gemm(
    x_int8_2d: np.ndarray,
    w_int8_2d: np.ndarray,
    bias_int8_1d: np.ndarray | None = None,
    relu: bool = False,
) -> tuple[np.ndarray, int]:
    """
    y = x @ w with int32 accum, then >> shift, optional int8 bias add, and saturation to int8.
    """
    acc = x_int8_2d.astype(np.int32) @ w_int8_2d.astype(np.int32)
    shift = _choose_shift(acc)
    y = (acc >> shift).astype(np.int32)
    if bias_int8_1d is not None:
        y = y + bias_int8_1d.astype(np.int32).reshape(1, -1)
    y = np.clip(y, -128, 127).astype(np.int8)
    if relu:
        y = np.maximum(y, 0)
    return y, shift


class NumpyMHALinear:
    """
    Int8 MHA with optional LUT softmax and support for cross-attention (Tq != Tk).
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        name_prefix: str,
        Wq: np.ndarray,
        Wk: np.ndarray,
        Wv: np.ndarray,
        Wo: np.ndarray,
        bq: np.ndarray | None = None,
        bk: np.ndarray | None = None,
        bv: np.ndarray | None = None,
        bo: np.ndarray | None = None,
        use_softmax: bool = True,
        output_relu: bool = False,
        exp_lut: np.ndarray | None = None,
    ) -> None:
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.C = d_model
        self.H = num_heads
        self.dh = d_model // num_heads
        self.name = str(name_prefix)

        self.Wq = Wq.astype(np.int8)
        self.Wk = Wk.astype(np.int8)
        self.Wv = Wv.astype(np.int8)
        self.Wo = Wo.astype(np.int8)
        self.bq = None if bq is None else np.clip(bq, -128, 127).astype(np.int8)
        self.bk = None if bk is None else np.clip(bk, -128, 127).astype(np.int8)
        self.bv = None if bv is None else np.clip(bv, -128, 127).astype(np.int8)
        self.bo = None if bo is None else np.clip(bo, -128, 127).astype(np.int8)

        self.use_softmax = use_softmax
        self.output_relu = output_relu
        self.exp_lut = build_exp_lut_256() if exp_lut is None else exp_lut

    def __call__(self, q, k=None, v=None, layers=None, training=False):
        def to_btc(t):
            if t.ndim == 2:
                return t[None, ...], True
            if t.ndim == 3:
                return t, False
            raise ValueError("Expected (T,C) or (B,T,C)")

        q_btc, squeezed = to_btc(q)
        k_btc, _ = to_btc(k if k is not None else q)
        v_btc, _ = to_btc(v if v is not None else q)

        Bq, Tq, Cq = q_btc.shape
        Bk, Tk, Ck = k_btc.shape
        Bv, Tv, Cv = v_btc.shape

        assert Bq == Bk == Bv, "Batch dimensions must match"
        assert Cq == Ck == Cv == self.C, f"Expected channel dim {self.C}"
        assert Tk == Tv, "Key/value sequence lengths must match"

        q8 = np.clip(q_btc, -128, 127).astype(np.int8)
        k8 = np.clip(k_btc, -128, 127).astype(np.int8)
        v8 = np.clip(v_btc, -128, 127).astype(np.int8)

        q2d = q8.reshape(Bq * Tq, self.C)
        k2d = k8.reshape(Bk * Tk, self.C)
        v2d = v8.reshape(Bv * Tv, self.C)

        q_proj, sh_q = _quantize_gemm(q2d, self.Wq, self.bq)
        k_proj, sh_k = _quantize_gemm(k2d, self.Wk, self.bk)
        v_proj, sh_v = _quantize_gemm(v2d, self.Wv, self.bv)

        if layers is not None:
            layers.append(
                {
                    "name": f"{self.name}_Wq",
                    "x": q2d,
                    "k": self.Wq,
                    "b": self.bq,
                    "y": q_proj,
                    "a": q_proj,
                    "shift": sh_q,
                    "is_relu": False,
                    "shift_scores": None,
                    "shift_context": None,
                    "use_softmax": self.use_softmax,
                }
            )
            layers.append(
                {
                    "name": f"{self.name}_Wk",
                    "x": k2d,
                    "k": self.Wk,
                    "b": self.bk,
                    "y": k_proj,
                    "a": k_proj,
                    "shift": sh_k,
                    "is_relu": False,
                }
            )
            layers.append(
                {
                    "name": f"{self.name}_Wv",
                    "x": v2d,
                    "k": self.Wv,
                    "b": self.bv,
                    "y": v_proj,
                    "a": v_proj,
                    "shift": sh_v,
                    "is_relu": False,
                }
            )

        qh = q_proj.reshape(Bq, Tq, self.H, self.dh).transpose(0, 2, 1, 3)  # (B,H,Tq,dh)
        kh = k_proj.reshape(Bk, Tk, self.H, self.dh).transpose(0, 2, 1, 3)  # (B,H,Tk,dh)
        vh = v_proj.reshape(Bv, Tv, self.H, self.dh).transpose(0, 2, 1, 3)  # (B,H,Tk,dh)

        ctx_h = np.zeros((Bq, self.H, Tq, self.dh), dtype=np.int8)
        sh_s_heads = np.empty(self.H, dtype=int)
        sh_c_heads = np.empty(self.H, dtype=int)

        for b in range(Bq):
            for h in range(self.H):
                Q = qh[b, h].astype(np.int32)     # (Tq, dh)
                Kt = kh[b, h].astype(np.int32).T  # (dh, Tk)
                scores_acc = Q @ Kt               # (Tq, Tk)

                sh_s = _choose_shift(scores_acc)
                sh_s_heads[h] = sh_s
                scores_q = np.clip(scores_acc >> sh_s, -128, 127).astype(np.int8)

                if self.use_softmax:
                    attn_q = softmax_lut_int8_rowwise(scores_q, exp_lut=self.exp_lut)
                else:
                    attn_q = scores_q

                V = vh[b, h].astype(np.int32)            # (Tk, dh)
                ctx_acc = attn_q.astype(np.int32) @ V    # (Tq, dh)

                sh_c = _choose_shift(ctx_acc)
                sh_c_heads[h] = sh_c
                ctx_q = np.clip(ctx_acc >> sh_c, -128, 127).astype(np.int8)
                ctx_h[b, h] = ctx_q

        ctx = ctx_h.transpose(0, 2, 1, 3).reshape(Bq, Tq, self.C)
        ctx2d = ctx.reshape(Bq * Tq, self.C)
        out_proj, sh_o = _quantize_gemm(ctx2d, self.Wo, self.bo, relu=self.output_relu)

        if layers is not None:
            layers.append(
                {
                    "name": f"{self.name}_Wo",
                    "x": ctx2d,
                    "k": self.Wo,
                    "b": self.bo,
                    "y": out_proj,
                    "a": out_proj,
                    "shift": sh_o,
                    "is_relu": self.output_relu,
                }
            )
            wq_idx = len(layers) - 4
            layers[wq_idx]["shift_scores"] = sh_s_heads.tolist()
            layers[wq_idx]["shift_context"] = sh_c_heads.tolist()

        out = out_proj.reshape(Bq, Tq, self.C)
        return out[0] if squeezed else out
