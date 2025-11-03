"""
AIE Layer implementations.

Each layer type is self-contained with:
- Golden computation (NumPy reference)
- Code generation (C++ kernel instantiation)
- Graph generation (connectivity)
"""

from .base import AIELayer
from .dense import DenseLayer
from .mha import MHALayer
from .resadd import ResAddLayer

__all__ = ['AIELayer', 'DenseLayer', 'MHALayer', 'ResAddLayer']