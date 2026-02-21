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
from .cls_attention import CLSAttentionLayer
from .softmax import SoftmaxLayer

__all__ = [
    'AIELayer',
    'DenseLayer',
    'MHALayer',
    'ResAddLayer',
    'CLSAttentionLayer',
    'SoftmaxLayer',
]
