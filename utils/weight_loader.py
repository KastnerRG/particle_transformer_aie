"""
Weight loading helpers with manifest-backed lookup and random int8 fallback.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np


def _as_int8(arr: np.ndarray) -> np.ndarray:
    return np.clip(np.round(arr), -128, 127).astype(np.int8)


def _adapt_weight_shape(weight: np.ndarray, in_features: int, out_features: int) -> np.ndarray:
    if weight.ndim != 2:
        raise ValueError(f"Weight must be rank-2, got shape {weight.shape}")

    if weight.shape == (in_features, out_features):
        return weight.astype(np.int8)

    if weight.shape == (out_features, in_features):
        return weight.T.astype(np.int8)

    if weight.size == in_features * out_features:
        # Fallback reshape assumes source stored as (out, in) like PyTorch Linear.
        return weight.reshape(out_features, in_features).T.astype(np.int8)

    raise ValueError(
        f"Cannot adapt weight shape {weight.shape} to ({in_features}, {out_features})."
    )


def _adapt_bias_shape(
    bias: np.ndarray,
    out_features: int,
    bias_shift: Optional[int] = None,
    output_shift: Optional[int] = None,
) -> np.ndarray:
    b = np.asarray(bias).reshape(-1)
    if b.size != out_features:
        if b.size == 1:
            b = np.repeat(b, out_features)
        else:
            raise ValueError(f"Cannot adapt bias size {b.size} to {out_features}.")

    if np.issubdtype(b.dtype, np.floating):
        return _as_int8(b)

    # Quantized int32 bias path (best-effort downshift to output scale).
    if np.issubdtype(b.dtype, np.integer):
        shift_down = 0
        if bias_shift is not None and output_shift is not None:
            shift_down = max(0, int(bias_shift) - int(output_shift))
        v = b.astype(np.int64)
        if shift_down > 0:
            v = np.right_shift(v, shift_down)
        return np.clip(v, -128, 127).astype(np.int8)

    return _as_int8(b.astype(np.float32))


@dataclass
class LinearParams:
    weight: np.ndarray
    bias: Optional[np.ndarray]
    matmul_shift: Optional[int] = None


class WeightLoader:
    """
    Resolve INT8 weights/biases by op-name from an export manifest, with fallback.
    """

    def __init__(
        self,
        manifest_path: Optional[str] = None,
        seed: int = 0,
        use_manifest_if_available: bool = True,
    ) -> None:
        self.rng = np.random.default_rng(seed)
        self.manifest_path = manifest_path
        self.use_manifest_if_available = use_manifest_if_available
        self._entries: Dict[str, dict] = {}
        self._manifest_dir: Optional[str] = None

        if manifest_path and use_manifest_if_available and os.path.exists(manifest_path):
            with open(manifest_path, "r", encoding="utf-8") as f:
                manifest = json.load(f)
            self._entries = {op["name"]: op for op in manifest.get("ops", []) if "name" in op}
            self._manifest_dir = os.path.dirname(os.path.abspath(manifest_path))

    @property
    def has_manifest(self) -> bool:
        return bool(self._entries) and self._manifest_dir is not None

    def _random_weight(self, in_features: int, out_features: int) -> np.ndarray:
        return self.rng.integers(-128, 128, size=(in_features, out_features), dtype=np.int8)

    def _random_bias(self, out_features: int) -> np.ndarray:
        return self.rng.integers(-32, 32, size=(out_features,), dtype=np.int8)

    def _load_entry_linear(
        self,
        name: str,
        in_features: int,
        out_features: int,
        with_bias: bool,
    ) -> Optional[LinearParams]:
        if not self.has_manifest or name not in self._entries:
            return None

        entry = self._entries[name]
        weight_file = entry.get("weight_file")
        if not weight_file:
            return None

        weight_path = os.path.join(self._manifest_dir, weight_file)
        if not os.path.exists(weight_path):
            return None

        weight_raw = np.load(weight_path)
        weight = _adapt_weight_shape(weight_raw, in_features, out_features)

        bias = None
        if with_bias:
            bias_file = entry.get("bias_file")
            if bias_file:
                bias_path = os.path.join(self._manifest_dir, bias_file)
                if os.path.exists(bias_path):
                    bias_raw = np.load(bias_path)
                    bias = _adapt_bias_shape(
                        bias_raw,
                        out_features=out_features,
                        bias_shift=entry.get("bias_shift"),
                        output_shift=entry.get("output_shift"),
                    )

        matmul_shift = entry.get("matmul_shift")
        if matmul_shift is not None:
            matmul_shift = int(matmul_shift)
        return LinearParams(weight=weight, bias=bias, matmul_shift=matmul_shift)

    def linear(
        self,
        name: str,
        in_features: int,
        out_features: int,
        with_bias: bool = True,
    ) -> LinearParams:
        loaded = self._load_entry_linear(name, in_features, out_features, with_bias)
        if loaded is not None:
            if with_bias and loaded.bias is None:
                loaded.bias = self._random_bias(out_features)
            return loaded

        weight = self._random_weight(in_features, out_features)
        bias = self._random_bias(out_features) if with_bias else None
        return LinearParams(weight=weight, bias=bias, matmul_shift=None)

    def cls_token(self, d_model: int) -> np.ndarray:
        return self.rng.integers(-32, 32, size=(d_model,), dtype=np.int8)

    def mode_str(self) -> str:
        return "manifest+fallback" if self.has_manifest else "random"


def split_heads_projection(
    matrix: np.ndarray,
    num_heads: int,
) -> Tuple[np.ndarray, ...]:
    """
    Split (in_features, d_model) or (d_model,) by head on the last dimension.
    """
    if matrix.ndim == 2:
        _, d_model = matrix.shape
        assert d_model % num_heads == 0
        return tuple(
            matrix[:, h * (d_model // num_heads):(h + 1) * (d_model // num_heads)]
            for h in range(num_heads)
        )
    if matrix.ndim == 1:
        d_model = matrix.shape[0]
        assert d_model % num_heads == 0
        return tuple(
            matrix[h * (d_model // num_heads):(h + 1) * (d_model // num_heads)]
            for h in range(num_heads)
        )
    raise ValueError(f"Unsupported matrix rank for head split: {matrix.ndim}")
