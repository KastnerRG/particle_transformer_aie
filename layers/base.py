"""
Base class for all AIE layers.

All layer types (Dense, MHA, ResAdd, etc.) inherit from AIELayer.
Each layer is self-contained and handles:
1. Golden computation (NumPy reference)
2. Kernel code generation (layer_X.cc)
3. Graph code generation (connectivity in layer_graph.h)
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
import numpy as np


class AIELayer(ABC):
    """
    Abstract base class for AIE layers.

    Attributes:
        idx: Layer index in the model (assigned by AIEModel.add_layer())
        name: Human-readable layer name (e.g., 'dense_0', 'mha_1')
        layer_type: Type identifier (e.g., 'dense', 'mha', 'resadd')
        params: Dictionary of layer-specific parameters
        outputs: Dictionary storing computed outputs from golden computation
                 Key 'a' is the primary output used by downstream layers
    """

    def __init__(self, name: str, layer_type: str, params: Optional[Dict[str, Any]] = None):
        """
        Initialize base layer.

        Args:
            name: Human-readable name
            layer_type: Type identifier for the layer
            params: Optional dict of layer-specific parameters
        """
        self.idx = None  # Will be assigned by AIEModel.add_layer()
        self.name = name
        self.layer_type = layer_type
        self.params = params or {}

        # Outputs dictionary - populated by _compute_golden()
        # 'a' is the primary output, other keys can store intermediates
        self.outputs: Dict[str, np.ndarray] = {}

        # Flag to track if golden has been computed
        self._golden_computed = False

    @abstractmethod
    def _compute_golden(self, inputs: List[np.ndarray]) -> np.ndarray:
        """
        Compute golden reference using NumPy.

        This is an internal method called by AIEModel.forward().
        Users should not call this directly.

        Args:
            inputs: List of input arrays from connected layers
                   For single-input layers: [input_array]
                   For multi-input layers (like ResAdd): [input1, input2, ...]

        Returns:
            Primary output array (also stored in self.outputs['a'])

        Implementations should:
        1. Perform NumPy computation
        2. Store result in self.outputs['a']
        3. Store any intermediate results needed for validation
        4. Set self._golden_computed = True
        5. Return the primary output
        """
        pass

    @abstractmethod
    def generate_kernel_code(self, f) -> None:
        """
        Generate C++ kernel instantiation code (layer_X.cc file).

        Args:
            f: File handle to write to

        Implementations should write the complete .cc file including:
        - Includes
        - Weight arrays (quantized, as int8 arrays)
        - Kernel function call with weights and parameters

        Example for dense layer:
            f.write('#include "kernels.h"\\n\\n')
            f.write('alignas(32) int8 weights[...] = {...};\\n')
            f.write('void f0(input_stream<int16>* in, output_stream<int16>* out) {\\n')
            f.write('    dense<...>(in, out, weights, shift);\\n')
            f.write('}\\n')
        """
        pass

    @abstractmethod
    def generate_graph_code(self, f, input_ports: List[str]) -> None:
        """
        Generate graph connectivity code (for graph.cpp).

        Args:
            f: File handle to write to
            input_ports: List of input port names this layer connects to
                        Provided by the model based on the DAG topology
                        Examples: ['AIE_IN', 'dense_0', 'mha_1']

        Implementations should write the graph definition including:
        - Kernel creation
        - Source file specification
        - Runtime ratio
        - Input/output connections
        - FIFO depths if needed

        Example for single-input layer:
            f.write(f'{self.name}[0] = kernel::create(f{self.idx});\\n')
            f.write(f'source({self.name}[0]) = "layer_{self.idx}.cc";\\n')
            f.write(f'runtime<ratio>({self.name}[0]) = 1.0;\\n')
            f.write(f'connect<stream>({input_ports[0]}.out[0], {self.name}[0].in[0]);\\n\\n')
        """
        pass

    @abstractmethod
    def num_kernels(self) -> int:
        """
        Return the number of AIE kernels this layer uses.

        Returns:
            Number of kernels (e.g., 1 for Dense, 23 for 4-head MHA)
        """
        pass

    def get_output_port(self, port_idx: int = 0) -> str:
        """
        Get the output port name for this layer.

        Args:
            port_idx: Output port index (kernel index within this layer's array)

        Returns:
            Port name string (e.g., 'dense_0[0]', 'mha_1[22]')

        Note: Most layers override this. Default implementation assumes single kernel.
        """
        return f"{self.name}[{port_idx}]"

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize layer to dictionary format.

        Returns:
            Dict with layer metadata and parameters
        """
        return {
            'idx': self.idx,
            'name': self.name,
            'type': self.layer_type,
            'params': self.params,
            'output_shape': self.outputs.get('a', None).shape if 'a' in self.outputs else None
        }

    def validate_inputs(self, inputs: List[np.ndarray], expected_count: int) -> None:
        """
        Helper to validate input count.

        Args:
            inputs: List of input arrays
            expected_count: Expected number of inputs

        Raises:
            ValueError if input count doesn't match
        """
        if len(inputs) != expected_count:
            raise ValueError(
                f"Layer {self.name} (type={self.layer_type}) expects {expected_count} "
                f"input(s), got {len(inputs)}"
            )

    def __repr__(self) -> str:
        """String representation for debugging."""
        return f"{self.__class__.__name__}(idx={self.idx}, name='{self.name}')"