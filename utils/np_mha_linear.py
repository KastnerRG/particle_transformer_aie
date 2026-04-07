# np_mha_linear.py
import numpy as np

def _choose_scale_and_shift(acc_int32, shift=15):
    max_abs = int(np.max(np.abs(acc_int32))) if acc_int32.size else 0
    if max_abs <= 127: 
        return 1, 0
    float_scale = 127.0 / max_abs
    int_scale = int(np.round(float_scale * (1 << shift)))
    return int_scale, shift

def _quantize_gemm(x_int8_2d, W_int8_2d, relu=False):
    """
    y = x @ W with int32 accum, then (acc * scale) >> shift and saturate to int8.
    """
    acc = x_int8_2d.astype(np.int32) @ W_int8_2d.astype(np.int32)
    scale, shift = _choose_scale_and_shift(acc)
    scaled_acc = acc.astype(np.int64) * scale # Emulate the 64-bit accumulator during multiplication
    # scaled_acc_rounded = scaled_acc + (1 << (shift - 1)) # Add half of the divisor to emulate symmetric rounding
    # y = (scaled_acc_rounded >> shift).astype(np.int32)
    y = np.around(scaled_acc / (1 << shift)).astype(np.int32) # bankers rounding
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
    def __init__(self, d_model, num_heads, name_prefix, Wq, Wk, Wv, Wo):
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

        q_proj, sc_q, sh_q = _quantize_gemm(q2d, self.Wq)  # (BT,C) int8; (160,64)@(64,64)
        k_proj, sc_k, sh_k = _quantize_gemm(k2d, self.Wk)
        v_proj, sc_v, sh_v = _quantize_gemm(v2d, self.Wv)

        if layers is not None:
            # Store Q layer with shift_scores and shift_context for later retrieval
            layers.append({'name': f'{self.name}_Wq', 'x': q2d, 'k': self.Wq,
                           'y': q_proj, 'a': q_proj, 'scale': sc_q, 'shift': sh_q, 'is_relu': False,
                           'scale_scores': None, 'shift_scores': None, 
                           'scale_context': None, 'shift_context': None})  # Will be filled after attention
            layers.append({'name': f'{self.name}_Wk', 'x': k2d, 'k': self.Wk,
                           'y': k_proj, 'a': k_proj, 'scale': sc_k, 'shift': sh_k, 'is_relu': False})
            layers.append({'name': f'{self.name}_Wv', 'x': v2d, 'k': self.Wv,
                           'y': v_proj, 'a': v_proj, 'scale': sc_v, 'shift': sh_v, 'is_relu': False})

        qh = q_proj.reshape(B, T, self.H, self.dh).transpose(0, 2, 1, 3)  # (B,H,T,dh)
        kh = k_proj.reshape(B, T, self.H, self.dh).transpose(0, 2, 1, 3)
        vh = v_proj.reshape(B, T, self.H, self.dh).transpose(0, 2, 1, 3)

        # ----- Linear attention core (NO softmax; nonlinear parts commented) -----
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

                # # NONLINEAR (commented): scaling and softmax
                # # scale = 1.0 / math.sqrt(self.dh)     # float
                # # scores = scores_acc * scale
                # # attn = softmax(scores, axis=-1)      # non-linear
                # # if training and dropout>0: apply dropout mask

                # Quantize scores to int8 using multiply and shift
                sc_s, sh_s = _choose_scale_and_shift(scores_acc)
                sc_s_heads[h] = sc_s
                sh_s_heads[h] = sh_s
                
                scores_scaled = scores_acc.astype(np.int64) * sc_s
                # scores_scaled_rounded = scores_scaled + (1 << (sh_s - 1))
                scores_scaled_rounded = np.around(scores_scaled / (1 << sh_s)).astype(np.int32) # bankers rounding
                scores_q = np.clip(scores_scaled_rounded, -128, 127).astype(np.int8)

                V = vh[b, h].astype(np.int32)               # (T,dh), promote for accum
                ctx_acc = scores_q.astype(np.int32) @ V     # (T,dh) int32

                # Quantize context to int8 using multiply and shift
                sc_c, sh_c = _choose_scale_and_shift(ctx_acc)
                sc_c_heads[h] = sc_c
                sh_c_heads[h] = sh_c
                
                ctx_scaled = ctx_acc.astype(np.int64) * sc_c
                # ctx_scaled_rounded = ctx_scaled + (1 << (sh_c - 1))
                ctx_scaled_rounded = np.around(ctx_scaled / (1 << sh_c)).astype(np.int32) # bankers rounding
                ctx_q = np.clip(ctx_scaled_rounded, -128, 127).astype(np.int8)

                ctx_h[b, h] = ctx_q

        # Concat heads -> (B,T,C) int8
        ctx = ctx_h.transpose(0, 2, 1, 3).reshape(B, T, C)

        # ----- Output linear (exportable) -----
        ctx2d = ctx.reshape(BT, C)  # int8
        out_proj, sc_o, sh_o = _quantize_gemm(ctx2d, self.Wo, relu=True)

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
        print(f"SCALE_S (per head) = {sc_s_heads}")
        print(f"SHIFT_S (per head) = {sh_s_heads}")
        print(f"SCALE_C (per head) = {sc_c_heads}")
        print(f"SHIFT_C (per head) = {sh_c_heads}")
        print(f"SCALE_O = {sc_o}, SHIFT_O = {sh_o}")

        out = out_proj.reshape(B, T, C)  # int8
        return out[0] if squeezed else out
