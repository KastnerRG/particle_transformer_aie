"""
AIE Layer implementations.

Each layer type is self-contained with:
- Golden computation (NumPy reference)
- Code generation (C++ kernel instantiation)
- Graph generation (connectivity)
"""

from .base import AIELayer
from .dense import DenseLayer
from .dense_softmax import DenseSoftmaxLayer
from .mha import MHALayer
from .resadd import ResAddLayer

__all__ = ['AIELayer', 'DenseLayer', 'DenseSoftmaxLayer', 'MHALayer', 'ResAddLayer']