"""
Matrix tiling utilities for AIE streaming.
"""

import numpy as np


def tile_matrix(matrix: np.ndarray, row_tiles: int, col_tiles: int) -> np.ndarray:
    """
    Tile matrix for AIE streaming.

    Reshapes (R,C) -> (R/r, C/c, r, c).flatten()

    Args:
        matrix: Input matrix of shape (R, C)
        row_tiles: Tile size for rows (r)
        col_tiles: Tile size for columns (c)

    Returns:
        Flattened tiled matrix

    Example:
        >>> matrix = np.arange(16).reshape(4, 4)
        >>> tiled = tile_matrix(matrix, 2, 2)
        >>> # Result: [0,1,4,5, 2,3,6,7, 8,9,12,13, 10,11,14,15]
    """
    rows, cols = matrix.shape
    assert rows % row_tiles == 0 and cols % col_tiles == 0, \
        f"Matrix shape {matrix.shape} must be divisible by tile sizes ({row_tiles}, {col_tiles})"

    reshaped = matrix.reshape(rows // row_tiles, row_tiles, cols // col_tiles, col_tiles)
    transposed = reshaped.transpose(0, 2, 1, 3)  # (R/r, C/c, r, c)
    return transposed.flatten()