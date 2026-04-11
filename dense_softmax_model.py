import numpy as np

from model import AIEModel
from layers import DenseSoftmaxLayer


def build_and_run(seed: int = 0):
    rng = np.random.default_rng(seed)

    batch = 160
    features = 64

    x = rng.integers(-128, 128, size=(batch, features), dtype=np.int8)
    w = rng.integers(-128, 128, size=(features, features), dtype=np.int8)

    model = AIEModel(m=4, k=8, n=8, iterations=1, dynamic_quant=False)

    dense_softmax = DenseSoftmaxLayer(
        name='dense_softmax_0',
        weight=w,
        shift_in=3,
        scale_in=1,
    )
    model.add_layer(dense_softmax, inputs=[None])

    y = model.forward(x)
    print(f"\nModel completed. Output shape: {y.shape}")
    return y


if __name__ == "__main__":
    build_and_run()