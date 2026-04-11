# np_mha_linear.py
import numpy as np


SOFTMAX_BIT = 7


def _int_softmax(scores_int, scaling_factor=1.0):
    """
    Integer softmax approximation adapted from IntSoftmax.

    Args:
        scores_int: int32 array with shape (..., T)
        scaling_factor: scalar used by integer exp approximation.

    Returns:
        probs_int: int32 array where each row approximately sums to 2**SOFTMAX_BIT.
    """
    x = scores_int.astype(np.int64, copy=False)

    # Normalize rows for numerical stability.
    x = x - np.max(x, axis=-1, keepdims=True)

    x0 = -0.6931
    n = 30
    coef0 = 0.35815147
    coef1 = 0.96963238 / coef0
    coef2 = 1.0 / coef0

    x0_int = int(np.floor(x0 / scaling_factor))
    x = np.maximum(x, n * x0_int)

    # q = floor(x / x0_int), r = x - x0_int * q
    q = np.floor_divide(x, x0_int)
    r = x - x0_int * q

    b_int = int(np.floor(coef1 / scaling_factor))
    c_int = int(np.floor(coef2 / (scaling_factor ** 2)))

    z = r * (r + b_int) + c_int
    # Mirror AIE kernel clamp to keep left shift bounded and deterministic.
    shift = np.clip((n - q).astype(np.int64), 0, 62)
    exp_int = z << shift
    exp_int = np.maximum(exp_int, 0)

    exp_sum = np.sum(exp_int, axis=-1, keepdims=True)
    exp_sum = np.maximum(exp_sum, 1)

    factor = np.floor_divide(1 << 32, exp_sum)
    probs_int = np.floor_divide(exp_int * factor, 1 << (32 - SOFTMAX_BIT))
    return probs_int.astype(np.int32)

def _choose_scale_and_shift(acc_int32, shift=15):
    max_abs = int(np.max(np.abs(acc_int32))) if acc_int32.size else 0
    if max_abs <= 127:
        return 1, 0
    float_scale = 127.0 / max_abs
    int_scale = int(np.round(float_scale * (1 << shift)))
    return int_scale, shift


def _resolve_scale_shift(acc_int32, scale=None, shift=None):
    if (scale is None) != (shift is None):
        raise ValueError(f"scale and shift must be both provided or both None, got scale={scale}, shift={shift}")
    if scale is None and shift is None:
        return _choose_scale_and_shift(acc_int32)
    return int(scale), int(shift)

def _quantize_gemm(x_int8_2d, W_int8_2d, relu=False, scale=None, shift=None):
    """
    y = x @ W with int32 accum, then (acc * scale) >> shift and saturate to int8.
    """
    acc = x_int8_2d.astype(np.int32) @ W_int8_2d.astype(np.int32)
    scale, shift = _resolve_scale_shift(acc, scale, shift)
    scaled_acc = acc.astype(np.int64) * scale
    y = np.around(scaled_acc / (1 << shift)).astype(np.int32)
    y = np.clip(y, -128, 127).astype(np.int8)
    if relu:
        y = np.maximum(y, 0)
    return y, scale, shift

class NumpyMHALinear:
    """
    Linear-only Multi-Head Attention (NumPy, int8 I/O, exportable Q/K/V/O GEMMs).

    - Call with (B,T,C) or (T,C). If k/v are None, uses q (self-attention).
    - Records four linear ops into `layers`: {name, x, k, y, a, shift, is_relu=False}
      where x/k/y/a are all int8. (No nonlinear ops are recorded.)
    - Set `name_prefix` to get entries like mha1_Wq, mha1_Wk, ...
    """
    def __init__(
        self,
        d_model,
        num_heads,
        name_prefix,
        Wq,
        Wk,
        Wv,
        Wo,
        softmax_scaling=None,
        enable_softmax=True,
        scale_q=None,
        shift_q=None,
        scale_k=None,
        shift_k=None,
        scale_v=None,
        shift_v=None,
        scale_s=None,
        shift_s=None,
        scale_c=None,
        shift_c=None,
        scale_o=None,
        shift_o=None,
        use_dynamic_quant=False,
    ):
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.C = d_model
        self.H = num_heads
        self.dh = d_model // num_heads
        self.name = str(name_prefix)
        
        # All weight matrices are now required parameters
        self.Wq = Wq
        self.Wk = Wk
        self.Wv = Wv
        self.Wo = Wo
        self.softmax_scaling = None if softmax_scaling is None else float(softmax_scaling)
        self.enable_softmax = bool(enable_softmax)

        self.scale_q = scale_q
        self.shift_q = shift_q
        self.scale_k = scale_k
        self.shift_k = shift_k
        self.scale_v = scale_v
        self.shift_v = shift_v
        self.scale_s = scale_s
        self.shift_s = shift_s
        self.scale_c = scale_c
        self.shift_c = shift_c
        self.scale_o = scale_o
        self.shift_o = shift_o
        self._use_dynamic_quant = bool(use_dynamic_quant)
        if not self._use_dynamic_quant:
            required = [
                self.scale_q, self.shift_q, self.scale_k, self.shift_k,
                self.scale_v, self.shift_v, self.scale_s, self.shift_s,
                self.scale_c, self.shift_c, self.scale_o, self.shift_o,
            ]
            if any(v is None for v in required):
                raise ValueError("Static quantization requires all quantization factors.")

    def __call__(self, q, k=None, v=None, layers=None, training=False):
        # Normalize to (B,T,C)
        def to_btc(t):
            if t.ndim == 2:  # (T,C)
                return t[None, ...], True
            if t.ndim == 3:  # (B,T,C)
                return t, False
            raise ValueError("Expected (T,C) or (B,T,C)")
        q_btc, squeezed = to_btc(q)
        k_btc, _ = to_btc(k if k is not None else q)
        v_btc, _ = to_btc(v if v is not None else q)

        B, T, C = q_btc.shape
        assert C == self.C, f"Expected last dim {self.C}, got {C}"

        # Ensure int8 inputs
        q8 = np.clip(q_btc, -128, 127).astype(np.int8)
        k8 = np.clip(k_btc, -128, 127).astype(np.int8)
        v8 = np.clip(v_btc, -128, 127).astype(np.int8)

        # ----- Linear Q/K/V (exportable) -----
        BT = B * T
        q2d = q8.reshape(BT, C)
        k2d = k8.reshape(BT, C)
        v2d = v8.reshape(BT, C)

        q_proj, sc_q, sh_q = _quantize_gemm(q2d, self.Wq, scale=None if self._use_dynamic_quant else self.scale_q, shift=None if self._use_dynamic_quant else self.shift_q)  # (BT,C) int8; (160,64)@(64,64)
        k_proj, sc_k, sh_k = _quantize_gemm(k2d, self.Wk, scale=None if self._use_dynamic_quant else self.scale_k, shift=None if self._use_dynamic_quant else self.shift_k)
        v_proj, sc_v, sh_v = _quantize_gemm(v2d, self.Wv, scale=None if self._use_dynamic_quant else self.scale_v, shift=None if self._use_dynamic_quant else self.shift_v)

        if self._use_dynamic_quant:
            print(f"NumpyMHALinear {self.name}: using dynamic quantization (choose_scale_and_shift)")
        else:
            print(f"NumpyMHALinear {self.name}: using static quantization")

        # Match fused AIE kernel scaling convention with scale+shift:
        # s = (scale_q * scale_k) / 2^(shift_q + shift_k)
        raw_scores_shift = int(sh_q) + int(sh_k)
        raw_scores_scale = int(sc_q) * int(sc_k)
        inferred_softmax_scaling = np.ldexp(float(raw_scores_scale), -raw_scores_shift)
        softmax_scaling = inferred_softmax_scaling if self.softmax_scaling is None else self.softmax_scaling

        if layers is not None:
            # Store Q layer with shift_scores and shift_context for later retrieval
            layers.append({'name': f'{self.name}_Wq', 'x': q2d, 'k': self.Wq,
                           'y': q_proj, 'a': q_proj, 'scale': sc_q, 'shift': sh_q, 'is_relu': False,
                           'scale_scores': None, 'shift_scores': None,
                           'scale_context': None, 'shift_context': None})
            layers.append({'name': f'{self.name}_Wk', 'x': k2d, 'k': self.Wk,
                           'y': k_proj, 'a': k_proj, 'scale': sc_k, 'shift': sh_k, 'is_relu': False})
            layers.append({'name': f'{self.name}_Wv', 'x': v2d, 'k': self.Wv,
                           'y': v_proj, 'a': v_proj, 'scale': sc_v, 'shift': sh_v, 'is_relu': False})

        qh = q_proj.reshape(B, T, self.H, self.dh).transpose(0, 2, 1, 3)  # (B,H,T,dh)
        kh = k_proj.reshape(B, T, self.H, self.dh).transpose(0, 2, 1, 3)
        vh = v_proj.reshape(B, T, self.H, self.dh).transpose(0, 2, 1, 3)

        # ----- Integer attention core with optional softmax -----
        ctx_h = np.empty_like(vh)  # (B,H,T,dh), int8
        sc_s_heads = np.empty(self.H, dtype=int)
        sh_s_heads = np.empty(self.H, dtype=int)
        sc_c_heads = np.empty(self.H, dtype=int)
        sh_c_heads = np.empty(self.H, dtype=int)

        for b in range(B):
            for h in range(self.H):
                Q = qh[b, h].astype(np.int32)              # (T,dh)
                Kt = kh[b, h].astype(np.int32).T           # (dh,T)
                scores_acc = Q @ Kt                         # (T,T) int32 //LAYER

                if self.enable_softmax:
                    # Integer softmax: takes int32 accumulation directly, outputs int8 scaled to 2^SOFTMAX_BIT
                    attn_int = _int_softmax(
                        scores_acc,
                        scaling_factor=softmax_scaling,
                    )
                    sc_s = raw_scores_scale
                    sh_s = raw_scores_shift
                else:
                    # Match AIE scores kernel no-softmax path: quantize raw score accumulations.
                    sc_s = None if self._use_dynamic_quant else self.scale_s[h]
                    sh_s = None if self._use_dynamic_quant else self.shift_s[h]
                    sc_s, sh_s = _resolve_scale_shift(scores_acc, sc_s, sh_s)
                    scores_scaled = scores_acc.astype(np.int64) * sc_s
                    scores_scaled_rounded = np.around(scores_scaled / (1 << sh_s)).astype(np.int32)
                    attn_int = np.clip(scores_scaled_rounded, -128, 127).astype(np.int32)

                sc_s_heads[h] = sc_s
                sh_s_heads[h] = sh_s

                V = vh[b, h].astype(np.int32)               # (T,dh), promote for accum
                ctx_acc = attn_int @ V                       # (T,dh) int32

                if self.enable_softmax:
                    # Undo softmax fixed-point scale (sum approximately equals 2**SOFTMAX_BIT).
                    ctx_acc = ctx_acc >> SOFTMAX_BIT

                sc_c_in = None if self._use_dynamic_quant else self.scale_c[h]
                sh_c_in = None if self._use_dynamic_quant else self.shift_c[h]
                sc_c, sh_c = _resolve_scale_shift(ctx_acc, sc_c_in, sh_c_in)
                sc_c_heads[h] = sc_c
                sh_c_heads[h] = sh_c
                ctx_scaled = ctx_acc.astype(np.int64) * sc_c
                ctx_scaled_rounded = np.around(ctx_scaled / (1 << sh_c)).astype(np.int32)
                ctx_q = np.clip(ctx_scaled_rounded, -128, 127).astype(np.int8)

                ctx_h[b, h] = ctx_q

        # Concat heads -> (B,T,C) int8
        ctx = ctx_h.transpose(0, 2, 1, 3).reshape(B, T, C)

        # ----- Output linear (exportable) -----
        ctx2d = ctx.reshape(BT, C)  # int8
        out_proj, sc_o, sh_o = _quantize_gemm(ctx2d, self.Wo, relu=True, scale=None if self._use_dynamic_quant else self.scale_o, shift=None if self._use_dynamic_quant else self.shift_o)

        # for debug
        if layers is not None:
            layers.append({'name': f'{self.name}_Wo', 'x': ctx2d, 'k': self.Wo,
                           'y': out_proj, 'a': out_proj, 'scale': sc_o, 'shift': sh_o, 'is_relu': True})

            # Update the Wq layer with the computed shift values for scores and context
            # Find the Wq layer (it's 4 layers back from the current position)
            wq_idx = len(layers) - 4
            layers[wq_idx]['scale_scores'] = sc_s_heads.tolist()
            layers[wq_idx]['shift_scores'] = sh_s_heads.tolist()
            layers[wq_idx]['scale_context'] = sc_c_heads.tolist()
            layers[wq_idx]['shift_context'] = sh_c_heads.tolist()

        print(f"SCALE_Q = {sc_q}, SHIFT_Q = {sh_q}")
        print(f"SCALE_K = {sc_k}, SHIFT_K = {sh_k}")
        print(f"SCALE_V = {sc_v}, SHIFT_V = {sh_v}")
        print(f"ENABLE_SOFTMAX = {self.enable_softmax}")
        print(f"SCALE_S (per head) = {sc_s_heads}")
        print(f"SHIFT_S (per head) = {sh_s_heads}")
        print(f"SCALE_C (per head) = {sc_c_heads}")
        print(f"SHIFT_C (per head) = {sh_c_heads}")
        print(f"SCALE_O = {sc_o}, SHIFT_O = {sh_o}")

        out = out_proj.reshape(B, T, C)  # int8
        return out[0] if squeezed else out
