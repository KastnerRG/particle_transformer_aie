import numpy as np
from model import AIEModel
from layers import DenseLayer, MHALayer, ResAddLayer


def build_and_run(seed: int = 0):
    rng = np.random.default_rng(seed)

    in_particles, num_feature, ff_dim = 150, 3, 64
    num_feature_pad = 8
    num_particles_pad = 160

    dummy_inp = rng.integers(-128, 128, size=(in_particles, num_feature), dtype=np.int8)
    pad_inp = np.zeros((num_particles_pad, num_feature_pad), dtype=np.int8)
    pad_inp[:in_particles, :num_feature] = dummy_inp

    m, k, n = 4, 8, 8
    model = AIEModel(m=m, k=k, n=n, iterations=1)

    W_fc0 = rng.integers(-128, 128, size=(num_feature_pad, ff_dim), dtype=np.int8)
    dense0 = DenseLayer(name='dense_0', weight=W_fc0, shift=2, relu=True)
    model.add_layer(dense0, inputs=[None])  # connect to AIE_IN under the hood

    Wq = rng.integers(-128, 128, size=(ff_dim, ff_dim), dtype=np.int8)
    Wk = rng.integers(-128, 128, size=(ff_dim, ff_dim), dtype=np.int8)
    Wv = rng.integers(-128, 128, size=(ff_dim, ff_dim), dtype=np.int8)
    Wo = rng.integers(-128, 128, size=(ff_dim, ff_dim), dtype=np.int8)

    mha1 = MHALayer(name='mha_1', Wq=Wq, Wk=Wk, Wv=Wv, Wo=Wo, num_heads=4, d_model=ff_dim, T=num_particles_pad)
    model.add_layer(mha1, inputs=[dense0])
    res1 = ResAddLayer(name='resadd_1')
    model.add_layer(res1, inputs=[mha1, dense0])

    W_ff1a = rng.integers(-128, 128, size=(ff_dim, ff_dim), dtype=np.int8)
    ff1a = DenseLayer(name='ff1a', weight=W_ff1a, shift=3, relu=True)
    model.add_layer(ff1a, inputs=[res1])

    W_ff1b = rng.integers(-128, 128, size=(ff_dim, ff_dim), dtype=np.int8)
    ff1b = DenseLayer(name='ff1b', weight=W_ff1b, shift=3, relu=True)
    model.add_layer(ff1b, inputs=[ff1a])

    res2 = ResAddLayer(name='resadd_2')
    model.add_layer(res2, inputs=[ff1b, res1])

    mha2 = MHALayer(name='mha_2', Wq=Wq, Wk=Wk, Wv=Wv, Wo=Wo, num_heads=4, d_model=ff_dim, T=num_particles_pad)
    model.add_layer(mha2, inputs=[res2])

    res3 = ResAddLayer(name='resadd_3')
    model.add_layer(res3, inputs=[mha2, res2])

    W_ff2a = rng.integers(-128, 128, size=(ff_dim, ff_dim), dtype=np.int8)
    ff2a = DenseLayer(name='ff2a', weight=W_ff2a, shift=3, relu=True)
    model.add_layer(ff2a, inputs=[res3])

    W_ff2b = rng.integers(-128, 128, size=(ff_dim, ff_dim), dtype=np.int8)
    ff2b = DenseLayer(name='ff2b', weight=W_ff2b, shift=3, relu=True)
    model.add_layer(ff2b, inputs=[ff2a])

    res4 = ResAddLayer(name='resadd_4')
    model.add_layer(res4, inputs=[ff2b, res3])

    W_out1 = rng.integers(-128, 128, size=(ff_dim, ff_dim), dtype=np.int8)
    out1 = DenseLayer(name='out1', weight=W_out1, shift=3, relu=True)
    model.add_layer(out1, inputs=[res4])

    out_dim = 8
    W_out2 = rng.integers(-128, 128, size=(ff_dim, out_dim), dtype=np.int8)
    out2 = DenseLayer(name='out2', weight=W_out2, shift=3, relu=False)
    model.add_layer(out2, inputs=[out1])

    y = model.forward(pad_inp)
    print(f"\nModel completed. Output shape: {y.shape}")
    return y


if __name__ == "__main__":
    build_and_run()
