"""
Multi-Layer Perceptron (MLP) Model
Input: 160x8 -> Hidden: 160x64 -> Output: 160x8
"""

import numpy as np
from model import AIEModel
from layers import DenseLayer


def build_and_run(seed: int = 0):
    rng = np.random.default_rng(seed)

    # Input parameters
    num_particles_pad = 160
    input_dim = 8
    hidden_dim = 64
    output_dim = 8

    # Create dummy input: (160, 8)
    dummy_inp = rng.integers(-128, 128, size=(num_particles_pad, input_dim), dtype=np.int8)

    use_dynamic_quant = False

    # Create AIEModel
    m, k, n = 4, 8, 8
    model = AIEModel(m=m, k=k, n=n, iterations=1, dynamic_quant=use_dynamic_quant)

    # First dense layer: 8 -> 64
    W_fc1 = rng.integers(-128, 128, size=(input_dim, hidden_dim), dtype=np.int8)
    fc1 = DenseLayer(name='fc1', weight=W_fc1, shift=15, scale=73, relu=True)
    model.add_layer(fc1, inputs=[None])  # connect to AIE_IN

    # Second dense layer: 64 -> 8
    W_fc2 = rng.integers(-128, 128, size=(hidden_dim, output_dim), dtype=np.int8)
    fc2 = DenseLayer(name='fc2', weight=W_fc2, shift=15, scale=58, relu=False)
    model.add_layer(fc2, inputs=[fc1])

    # Forward pass
    y = model.forward(dummy_inp)
    print(f"\nMLP Model completed. Output shape: {y.shape}")
    print(f"Expected output shape: ({num_particles_pad}, {output_dim})")
    return y


if __name__ == "__main__":
    build_and_run()
