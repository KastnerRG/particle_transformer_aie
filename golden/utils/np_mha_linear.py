# np_mha_linear.py
import numpy as np

def residual_add_int8(x_int8, skip_int8):
    """Keras Add()([x, skip]) with int8 saturation."""
    y = x_int8.astype(np.int16) + skip_int8.astype(np.int16)
    y = np.clip(y, -128, 127).astype(np.int8)
    return y

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

def _quantize_gemm(x_int8_2d, W_int8_2d):
    """
    y = x @ W with int32 accum, then >> shift and saturate to int8.
    x_int8_2d: (N, C), W_int8_2d: (C, C) -> y_int8: (N, C)
    """
    acc = x_int8_2d.astype(np.int32) @ W_int8_2d.astype(np.int32)
    shift = _choose_shift(acc)
    print(f"SHIFT = {shift}")
    y = (acc >> shift).astype(np.int32)
    y = np.clip(y, -128, 127).astype(np.int8)
    return y, shift

class NumpyMHALinear:
    """
    Linear-only Multi-Head Attention (NumPy, int8 I/O, exportable Q/K/V/O GEMMs).

    - Call with (B,T,C) or (T,C). If k/v are None, uses q (self-attention).
    - Records four linear ops into `layers`: {name, x, k, y, a, shift, is_relu=False}
      where x/k/y/a are all int8. (No nonlinear ops are recorded.)
    - Set `name_prefix` to get entries like mha1_Wq, mha1_Wk, ...
    """
    def __init__(self, d_model, num_heads=4, name_prefix="mha1", seed=0,
                 Wq=None, Wk=None, Wv=None, Wo=None):
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.C = d_model
        self.H = num_heads
        self.dh = d_model // num_heads
        self.name = str(name_prefix)
        rng = np.random.default_rng(seed)

        def initW():
            return rng.integers(-128, 128, size=(d_model, d_model), dtype=np.int8)
            print("making a weight")

        self.Wq = Wq if Wq is not None else initW()
        self.Wk = Wk if Wk is not None else initW()
        self.Wv = V if (V:=Wv) is not None else initW()  # small trick to keep name short
        self.Wo = Wo if Wo is not None else initW()

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

        q_proj, sh_q = _quantize_gemm(q2d, self.Wq)  # (BT,C) int8; (150,64)@(64,64)
        k_proj, sh_k = _quantize_gemm(k2d, self.Wk)
        v_proj, sh_v = _quantize_gemm(v2d, self.Wv)

        if layers is not None:
            layers.append({'name': f'{self.name}_Wq', 'x': q2d, 'k': self.Wq,
                           'y': q_proj, 'a': q_proj, 'shift': sh_q, 'is_relu': False})
            layers.append({'name': f'{self.name}_Wk', 'x': k2d, 'k': self.Wk,
                           'y': k_proj, 'a': k_proj, 'shift': sh_k, 'is_relu': False})
            layers.append({'name': f'{self.name}_Wv', 'x': v2d, 'k': self.Wv,
                           'y': v_proj, 'a': v_proj, 'shift': sh_v, 'is_relu': False})

        qh = q_proj.reshape(B, T, self.H, self.dh).transpose(0, 2, 1, 3)  # (B,H,T,dh)
        kh = k_proj.reshape(B, T, self.H, self.dh).transpose(0, 2, 1, 3)
        vh = v_proj.reshape(B, T, self.H, self.dh).transpose(0, 2, 1, 3)

        # ----- Linear attention core (NO softmax; nonlinear parts commented) -----
        ctx_h = np.empty_like(vh)  # (B,H,T,dh), int8

        for b in range(B):
            for h in range(self.H):
                Q = qh[b, h].astype(np.int32)              # (T,dh)
                Kt = kh[b, h].astype(np.int32).T           # (dh,T)
                scores_acc = Q @ Kt                         # (T,T) int32 //LAYER

                # # NONLINEAR (commented): scaling and softmax
                # # scale = 1.0 / math.sqrt(self.dh)     # float
                # # scores = scores_acc * scale
                # # attn = softmax(scores, axis=-1)      # non-linear
                # # if training and dropout>0: apply dropout mask
                # Quantize scores to int8
                sh_s = _choose_shift(scores_acc)
                print(f"SHIFT_S = {sh_s}")
                scores_q = np.clip(scores_acc >> sh_s, -128, 127).astype(np.int8)  # (T,T)

                V = vh[b, h].astype(np.int32)               # (T,dh), promote for accum
                ctx_acc = scores_q.astype(np.int32) @ V     # (T,dh) int32

                sh_c = _choose_shift(ctx_acc)
                print(f"SHIFT_C = {sh_c}")
                ctx_q = np.clip(ctx_acc >> sh_c, -128, 127).astype(np.int8)  # (T,dh)

                ctx_h[b, h] = ctx_q

        # Concat heads -> (B,T,C) int8
        ctx = ctx_h.transpose(0, 2, 1, 3).reshape(B, T, C)

        # ----- Output linear (exportable) -----
        ctx2d = ctx.reshape(BT, C)  # int8
        out_proj, sh_o = _quantize_gemm(ctx2d, self.Wo)

        # for debug
        if layers is not None:
            layers.append({'name': f'{self.name}_Wo', 'x': ctx2d, 'k': self.Wo,
                           'y': out_proj, 'a': out_proj, 'shift': sh_o, 'is_relu': False})

        out = out_proj.reshape(B, T, C)  # int8
        return out[0] if squeezed else out
