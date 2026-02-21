"""
Lookup-table softmax helpers for int8 pipelines.
"""

from __future__ import annotations

import numpy as np


def build_exp_lut_256(temperature: float = 16.0, scale: int = 1024) -> np.ndarray:
    """
    Build a 256-entry LUT for exp(x / temperature), where x is signed int8.

    Index mapping:
      lut[i] corresponds to signed value (i - 128).
    """
    values = np.arange(-128, 128, dtype=np.int16).astype(np.float64)
    lut = np.exp(values / float(temperature)) * float(scale)
    lut = np.clip(np.round(lut), 1, np.iinfo(np.int16).max).astype(np.int16)
    return lut


def softmax_lut_int8_rowwise(
    x_int8: np.ndarray,
    exp_lut: np.ndarray | None = None,
    out_scale: int = 127,
) -> np.ndarray:
    """
    Row-wise int8 softmax using a 256-entry LUT.

    Args:
        x_int8: Array with class/scores dimension on the last axis.
        exp_lut: Optional LUT from build_exp_lut_256().
        out_scale: Output quantization scale. Typical choice is 127.

    Returns:
        int8 array in [0, out_scale], same shape as input.
    """
    if exp_lut is None:
        exp_lut = build_exp_lut_256()

    if x_int8.dtype != np.int8:
        x = np.clip(x_int8, -128, 127).astype(np.int8)
    else:
        x = x_int8

    x2 = x.reshape(-1, x.shape[-1]).astype(np.int16)
    y2 = np.zeros_like(x2, dtype=np.int8)

    lut_i32 = exp_lut.astype(np.int32)
    for r in range(x2.shape[0]):
        row = x2[r]
        row_max = int(row.max())
        delta = np.clip(row - row_max, -128, 127).astype(np.int16)
        idx = (delta + 128).astype(np.int16)
        numer = lut_i32[idx]
        denom = int(numer.sum())

        if denom <= 0:
            # Defensive fallback; this should not occur with a min-LUT value of 1.
            y_row = np.zeros_like(row, dtype=np.int16)
            y_row[0] = out_scale
        else:
            y_row = (numer * int(out_scale) + (denom // 2)) // denom

        y2[r] = np.clip(y_row, 0, out_scale).astype(np.int8)

    return y2.reshape(x.shape)


def lut_to_c_initializer(lut: np.ndarray) -> str:
    """
    Convert LUT to a C initializer list.
    """
    flat = lut.reshape(-1)
    return ", ".join(str(int(v)) for v in flat)

