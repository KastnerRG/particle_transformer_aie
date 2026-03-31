# np_mha_linear.py
import numpy as np


def _int_softmax(scores_int, output_bit=7, scaling_factor=1.0):
    """
    Integer softmax approximation adapted from IntSoftmax.

    Args:
        scores_int: int32 array with shape (..., T)
        output_bit: softmax output bitwidth (default 7)
        scaling_factor: scalar used by integer exp approximation.

    Returns:
        probs_int: int32 array where each row approximately sums to 2**output_bit.
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
    probs_int = np.floor_divide(exp_int * factor, 1 << (32 - output_bit))
    return probs_int.astype(np.int32)

def _choose_shift(acc_int32):
    max_abs = int(np.max(np.abs(acc_int32))) if acc_int32.size else 0
    if max_abs <= 127: 
        return 0
    # ceil(log2(max_abs/127))
    # (avoid log for determinism)
    shift = 0
    limit = 127
    while max_abs > limit and shift < 31:
        shift += 1
        limit <<= 1
    return shift

def _quantize_gemm(x_int8_2d, W_int8_2d, relu=False):
    """
    y = x @ W with int32 accum, then >> shift and saturate to int8.
    x_int8_2d: (N, C), W_int8_2d: (C, C) -> y_int8: (N, C)
    """
    acc = x_int8_2d.astype(np.int32) @ W_int8_2d.astype(np.int32)
    shift = _choose_shift(acc)
    y = (acc >> shift).astype(np.int32)
    y = np.clip(y, -128, 127).astype(np.int8)
    if relu:
        y = np.maximum(y, 0)
    return y, shift

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
        softmax_bit=7,
        softmax_scaling=None,
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
        self.softmax_bit = int(softmax_bit)
        self.softmax_scaling = None if softmax_scaling is None else float(softmax_scaling)

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

        q_proj, sh_q = _quantize_gemm(q2d, self.Wq)  # (BT,C) int8; (160,64)@(64,64)
        k_proj, sh_k = _quantize_gemm(k2d, self.Wk)
        v_proj, sh_v = _quantize_gemm(v2d, self.Wv)

        # Match fused AIE kernel scaling convention: s = 2^-(shift_q + shift_k).
        # Match AIE kernel clamp: B is limited to [0, 18].
        b_softmax = min(max(0, int(sh_q) + int(sh_k)), 18)
        inferred_softmax_scaling = np.ldexp(1.0, -b_softmax)
        softmax_scaling = inferred_softmax_scaling if self.softmax_scaling is None else self.softmax_scaling

        if layers is not None:
            # Store Q layer with shift_scores and shift_context for later retrieval
            layers.append({'name': f'{self.name}_Wq', 'x': q2d, 'k': self.Wq,
                           'y': q_proj, 'a': q_proj, 'shift': sh_q, 'is_relu': False,
                           'shift_scores': None, 'shift_context': None})  # Will be filled after attention
            layers.append({'name': f'{self.name}_Wk', 'x': k2d, 'k': self.Wk,
                           'y': k_proj, 'a': k_proj, 'shift': sh_k, 'is_relu': False})
            layers.append({'name': f'{self.name}_Wv', 'x': v2d, 'k': self.Wv,
                           'y': v_proj, 'a': v_proj, 'shift': sh_v, 'is_relu': False})

        qh = q_proj.reshape(B, T, self.H, self.dh).transpose(0, 2, 1, 3)  # (B,H,T,dh)
        kh = k_proj.reshape(B, T, self.H, self.dh).transpose(0, 2, 1, 3)
        vh = v_proj.reshape(B, T, self.H, self.dh).transpose(0, 2, 1, 3)

        # ----- Integer-only attention core with IntSoftmax approximation -----
        ctx_h = np.empty_like(vh)  # (B,H,T,dh), int8
        sh_s_heads = np.empty(self.H, dtype=int)
        sh_c_heads = np.empty(self.H, dtype=int)

        for b in range(B):
            for h in range(self.H):
                Q = qh[b, h].astype(np.int32)              # (T,dh)
                Kt = kh[b, h].astype(np.int32).T           # (dh,T)
                scores_acc = Q @ Kt                         # (T,T) int32 //LAYER

                # Integer softmax: takes int32 accumulation directly, outputs int8 scaled to 2^softmax_bit
                attn_int = _int_softmax(
                    scores_acc,
                    output_bit=self.softmax_bit,
                    scaling_factor=softmax_scaling,
                )
                # Track softmax shift for reference (not used in quantization)
                sh_s = 0  # Shift is handled internally by _int_softmax
                sh_s_heads[h] = sh_s

                V = vh[b, h].astype(np.int32)               # (T,dh), promote for accum
                ctx_acc = attn_int @ V                       # (T,dh) int32

                # Undo softmax fixed-point scale (sum approximately equals 2**softmax_bit).
                ctx_acc = ctx_acc >> self.softmax_bit

                sh_c = _choose_shift(ctx_acc)
                sh_c_heads[h] = sh_c + self.softmax_bit
                ctx_q = np.clip(ctx_acc >> sh_c, -128, 127).astype(np.int8)  # (T,dh)

                ctx_h[b, h] = ctx_q

        # Concat heads -> (B,T,C) int8
        ctx = ctx_h.transpose(0, 2, 1, 3).reshape(B, T, C)

        # ----- Output linear (exportable) -----
        ctx2d = ctx.reshape(BT, C)  # int8
        out_proj, sh_o = _quantize_gemm(ctx2d, self.Wo, relu=True)

        # for debug
        if layers is not None:
            layers.append({'name': f'{self.name}_Wo', 'x': ctx2d, 'k': self.Wo,
                           'y': out_proj, 'a': out_proj, 'shift': sh_o, 'is_relu': True})

            # Update the Wq layer with the computed shift values for scores and context
            # Find the Wq layer (it's 4 layers back from the current position)
            wq_idx = len(layers) - 4
            layers[wq_idx]['shift_scores'] = sh_s_heads.tolist()
            layers[wq_idx]['shift_context'] = sh_c_heads.tolist()

        print(f"SHIFT_Q = {sh_q}")
        print(f"SHIFT_K = {sh_k}")
        print(f"SHIFT_V = {sh_v}")
        print(f"SHIFT_S (per head) = {sh_s_heads}")
        print(f"SHIFT_C (per head) = {sh_c_heads}")
        print(f"SHIFT_O = {sh_o}")

        out = out_proj.reshape(B, T, C)  # int8
        return out[0] if squeezed else out
