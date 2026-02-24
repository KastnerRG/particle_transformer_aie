from __future__ import annotations

from dataclasses import dataclass
from math import lcm
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from model import AIEModel
from layers import DenseLayer, MHALayer, ResAddLayer, CLSAttentionLayer, SoftmaxLayer
from utils.weight_loader import WeightLoader


def _pad_to_multiple(x: int, mult: int) -> int:
    return ((x + mult - 1) // mult) * mult


def _expand_weight(weight: np.ndarray, in_padded: int, out_padded: int) -> np.ndarray:
    w = np.zeros((in_padded, out_padded), dtype=np.int8)
    in_copy = min(in_padded, weight.shape[0])
    out_copy = min(out_padded, weight.shape[1])
    w[:in_copy, :out_copy] = weight[:in_copy, :out_copy].astype(np.int8)
    return w


def _expand_bias(bias: Optional[np.ndarray], out_padded: int) -> Optional[np.ndarray]:
    if bias is None:
        return None
    b = np.zeros((out_padded,), dtype=np.int8)
    copy = min(out_padded, bias.shape[0])
    b[:copy] = bias[:copy].astype(np.int8)
    return b


def _pad_input_particles(x: np.ndarray, padded_particles: int, padded_features: int) -> np.ndarray:
    assert x.ndim == 2, f"Expected input shape (R,D), got {x.shape}"
    out = np.zeros((padded_particles, padded_features), dtype=np.int8)
    r = min(padded_particles, x.shape[0])
    d = min(padded_features, x.shape[1])
    out[:r, :d] = np.clip(x[:r, :d], -128, 127).astype(np.int8)
    return out


@dataclass
class ParticleTransformerAIEConfig:
    num_particles: int = 128
    input_dim: int = 17
    embed_dims: Tuple[int, ...] = (128, 256, 128)
    ffn_dim: int = 512
    num_heads: int = 4
    num_blocks: int = 4
    num_cls_blocks: int = 2
    num_classes: int = 10

    m: int = 4
    k: int = 8
    n: int = 8
    iterations: int = 1

    seed: int = 0
    manifest_path: Optional[str] = None
    use_manifest_if_available: bool = True
    default_shift: int = 3


def _resolve_dims(cfg: ParticleTransformerAIEConfig) -> Dict[str, int | List[int]]:
    kn_mult = lcm(cfg.k, cfg.n)
    dmodel_mult = cfg.num_heads * kn_mult

    embed_padded: List[int] = []
    for i, d in enumerate(cfg.embed_dims):
        mult = dmodel_mult if i == (len(cfg.embed_dims) - 1) else kn_mult
        embed_padded.append(_pad_to_multiple(int(d), mult))

    dims = {
        "num_particles_pad": _pad_to_multiple(cfg.num_particles, cfg.m),
        "input_dim_pad": _pad_to_multiple(cfg.input_dim, cfg.k),
        "embed_dims_pad": embed_padded,
        "ffn_dim_pad": _pad_to_multiple(cfg.ffn_dim, kn_mult),
        "num_classes_pad": _pad_to_multiple(cfg.num_classes, cfg.n),
        "cls_q_len_pad": _pad_to_multiple(1, cfg.m),
    }
    return dims


def _shift_or_default(matmul_shift: Optional[int], default_shift: int) -> int:
    if matmul_shift is None:
        return default_shift
    return int(max(0, matmul_shift))


def _linear_from_loader(
    loader: WeightLoader,
    op_name: str,
    logical_in: int,
    logical_out: int,
    padded_in: int,
    padded_out: int,
    default_shift: int,
) -> Tuple[np.ndarray, Optional[np.ndarray], int]:
    p = loader.linear(op_name, logical_in, logical_out, with_bias=True)
    w = _expand_weight(p.weight, in_padded=padded_in, out_padded=padded_out)
    b = _expand_bias(p.bias, out_padded=padded_out)
    shift = _shift_or_default(p.matmul_shift, default_shift)
    return w, b, shift


def build_model(
    cfg: ParticleTransformerAIEConfig,
) -> Tuple[AIEModel, Dict[str, object]]:
    if cfg.num_cls_blocks < 1:
        raise ValueError("num_cls_blocks must be >= 1 for CLS-attention pooling.")

    dims = _resolve_dims(cfg)
    num_particles_pad = int(dims["num_particles_pad"])
    input_dim_pad = int(dims["input_dim_pad"])
    embed_dims_pad = list(dims["embed_dims_pad"])
    ffn_dim_pad = int(dims["ffn_dim_pad"])
    num_classes_pad = int(dims["num_classes_pad"])
    cls_q_len_pad = int(dims["cls_q_len_pad"])
    d_model = embed_dims_pad[-1]

    loader = WeightLoader(
        manifest_path=cfg.manifest_path,
        seed=cfg.seed,
        use_manifest_if_available=cfg.use_manifest_if_available,
    )

    model = AIEModel(m=cfg.m, k=cfg.k, n=cfg.n, iterations=cfg.iterations)

    # Encoder: D -> embed_dims...
    prev_logical = cfg.input_dim
    prev_padded = input_dim_pad
    prev_layer = None

    for i, (logical_out, padded_out) in enumerate(zip(cfg.embed_dims, embed_dims_pad)):
        op_name = f"mod.embed.embed.{1 + 3 * i}"
        w, b, shift = _linear_from_loader(
            loader=loader,
            op_name=op_name,
            logical_in=prev_logical,
            logical_out=logical_out,
            padded_in=prev_padded,
            padded_out=padded_out,
            default_shift=cfg.default_shift,
        )
        layer = DenseLayer(name=f"embed_{i}", weight=w, shift=shift, bias=b, relu=True)
        model.add_layer(layer, inputs=[None] if prev_layer is None else [prev_layer])
        prev_layer = layer
        prev_logical = logical_out
        prev_padded = padded_out

    token_layer = prev_layer

    # Transformer blocks.
    for i in range(cfg.num_blocks):
        Wq, bq, _ = _linear_from_loader(loader, f"mod.blocks.{i}.attn.q_proj", d_model, d_model, d_model, d_model, cfg.default_shift)
        Wk, bk, _ = _linear_from_loader(loader, f"mod.blocks.{i}.attn.k_proj", d_model, d_model, d_model, d_model, cfg.default_shift)
        Wv, bv, _ = _linear_from_loader(loader, f"mod.blocks.{i}.attn.v_proj", d_model, d_model, d_model, d_model, cfg.default_shift)
        Wo, bo, _ = _linear_from_loader(loader, f"mod.blocks.{i}.attn.out_proj", d_model, d_model, d_model, d_model, cfg.default_shift)

        attn = MHALayer(
            name=f"blocks_{i}_attn",
            Wq=Wq,
            Wk=Wk,
            Wv=Wv,
            Wo=Wo,
            bq=bq,
            bk=bk,
            bv=bv,
            bo=bo,
            num_heads=cfg.num_heads,
            d_model=d_model,
            T=num_particles_pad,
            use_softmax=True,
            output_relu=False,
        )
        model.add_layer(attn, inputs=[token_layer])

        res_attn = ResAddLayer(name=f"blocks_{i}_res_attn")
        model.add_layer(res_attn, inputs=[attn, token_layer])

        W1, b1, s1 = _linear_from_loader(
            loader, f"mod.blocks.{i}.fc1", d_model, cfg.ffn_dim, d_model, ffn_dim_pad, cfg.default_shift
        )
        ff1 = DenseLayer(name=f"blocks_{i}_ff1", weight=W1, shift=s1, bias=b1, relu=True)
        model.add_layer(ff1, inputs=[res_attn])

        W2, b2, s2 = _linear_from_loader(
            loader, f"mod.blocks.{i}.fc2", cfg.ffn_dim, d_model, ffn_dim_pad, d_model, cfg.default_shift
        )
        ff2 = DenseLayer(name=f"blocks_{i}_ff2", weight=W2, shift=s2, bias=b2, relu=False)
        model.add_layer(ff2, inputs=[ff1])

        res_ffn = ResAddLayer(name=f"blocks_{i}_res_ffn")
        model.add_layer(res_ffn, inputs=[ff2, res_attn])
        token_layer = res_ffn

    # CLS blocks.
    cls_layer = None
    for i in range(cfg.num_cls_blocks):
        Wq, bq, _ = _linear_from_loader(loader, f"mod.cls_blocks.{i}.attn.q_proj", d_model, d_model, d_model, d_model, cfg.default_shift)
        Wk, bk, _ = _linear_from_loader(loader, f"mod.cls_blocks.{i}.attn.k_proj", d_model, d_model, d_model, d_model, cfg.default_shift)
        Wv, bv, _ = _linear_from_loader(loader, f"mod.cls_blocks.{i}.attn.v_proj", d_model, d_model, d_model, d_model, cfg.default_shift)
        Wo, bo, _ = _linear_from_loader(loader, f"mod.cls_blocks.{i}.attn.out_proj", d_model, d_model, d_model, d_model, cfg.default_shift)

        internal_mode = (i == 0)
        cls_token = loader.cls_token(d_model) if internal_mode else None

        cls_attn = CLSAttentionLayer(
            name=f"cls_blocks_{i}_attn",
            Wq=Wq,
            Wk=Wk,
            Wv=Wv,
            Wo=Wo,
            bq=bq,
            bk=bk,
            bv=bv,
            bo=bo,
            num_heads=cfg.num_heads,
            d_model=d_model,
            T_kv=num_particles_pad,
            T_q=cls_q_len_pad,
            internal_cls_query=internal_mode,
            cls_token=cls_token,
            use_softmax=True,
            add_query_residual=internal_mode,
            output_relu=False,
        )

        if internal_mode:
            model.add_layer(cls_attn, inputs=[token_layer])
            cls_res_attn = cls_attn
        else:
            model.add_layer(cls_attn, inputs=[cls_layer, token_layer])
            cls_res_attn = ResAddLayer(name=f"cls_blocks_{i}_res_attn")
            model.add_layer(cls_res_attn, inputs=[cls_attn, cls_layer])

        W1, b1, s1 = _linear_from_loader(
            loader, f"mod.cls_blocks.{i}.fc1", d_model, cfg.ffn_dim, d_model, ffn_dim_pad, cfg.default_shift
        )
        cls_ff1 = DenseLayer(name=f"cls_blocks_{i}_ff1", weight=W1, shift=s1, bias=b1, relu=True)
        model.add_layer(cls_ff1, inputs=[cls_res_attn])

        W2, b2, s2 = _linear_from_loader(
            loader, f"mod.cls_blocks.{i}.fc2", cfg.ffn_dim, d_model, ffn_dim_pad, d_model, cfg.default_shift
        )
        cls_ff2 = DenseLayer(name=f"cls_blocks_{i}_ff2", weight=W2, shift=s2, bias=b2, relu=False)
        model.add_layer(cls_ff2, inputs=[cls_ff1])

        cls_res_ffn = ResAddLayer(name=f"cls_blocks_{i}_res_ffn")
        model.add_layer(cls_res_ffn, inputs=[cls_ff2, cls_res_attn])
        cls_layer = cls_res_ffn

    # Final classifier + softmax.
    W_fc, b_fc, s_fc = _linear_from_loader(
        loader, "mod.fc.0", d_model, cfg.num_classes, d_model, num_classes_pad, cfg.default_shift
    )
    logits_layer = DenseLayer(name="fc_out", weight=W_fc, shift=s_fc, bias=b_fc, relu=False)
    model.add_layer(logits_layer, inputs=[cls_layer])

    probs_layer = SoftmaxLayer(name="softmax_out", out_scale=127)
    model.add_layer(probs_layer, inputs=[logits_layer])

    meta = {
        "dims": dims,
        "weight_mode": loader.mode_str(),
        "logits_layer": logits_layer,
        "probs_layer": probs_layer,
    }
    return model, meta


def build_and_run(
    cfg: Optional[ParticleTransformerAIEConfig] = None,
    input_particles: Optional[np.ndarray] = None,
) -> Dict[str, np.ndarray | int | str | Dict[str, object]]:
    cfg = cfg or ParticleTransformerAIEConfig()
    dims = _resolve_dims(cfg)
    rng = np.random.default_rng(cfg.seed)

    if input_particles is None:
        input_particles = rng.integers(
            -128,
            128,
            size=(cfg.num_particles, cfg.input_dim),
            dtype=np.int8,
        )

    x_pad = _pad_input_particles(
        input_particles,
        padded_particles=int(dims["num_particles_pad"]),
        padded_features=int(dims["input_dim_pad"]),
    )

    model, meta = build_model(cfg)
    probs_padded = model.forward(x_pad)
    logits_padded = meta["logits_layer"].outputs['a']

    probs_vec = probs_padded[0, :cfg.num_classes].astype(np.int8)
    pred_class = int(np.argmax(probs_vec.astype(np.int32)))

    print("\nParticleTransformer AIE inference complete")
    print(f"  weight mode      : {meta['weight_mode']}")
    print(f"  input padded     : {x_pad.shape}")
    print(f"  logits padded    : {logits_padded.shape}")
    print(f"  probs padded     : {probs_padded.shape}")
    print(f"  predicted class  : {pred_class}")

    return {
        "logits_padded": logits_padded,
        "probs_padded": probs_padded,
        "probs": probs_vec,
        "pred_class": pred_class,
        "meta": meta,
    }


if __name__ == "__main__":
    build_and_run()
