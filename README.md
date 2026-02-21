# particle_transformer_aie

INT8-oriented Particle Transformer inference flow for AIE:
- Configurable encoder + transformer blocks + CLS cross-attention blocks
- LUT softmax for attention and final class probabilities
- AIE codegen + simulation via `AIEModel`

## Run

```bash
python particle_transformer.py
```

This will:
1. Build a default JetClass-like model:
   - `num_particles=128`
   - `input_dim=17`
   - `embed_dims=[128,256,128]`
   - `num_heads=4`
   - `num_blocks=4`
   - `num_cls_blocks=2`
   - `num_classes=10`
2. Pad dimensions for tile safety (`m=4`, `k=8`, `n=8`)
3. Generate AIE kernels/graph and run simulation through `run.sh`
4. Return:
   - `logits_padded`
   - `probs_padded`
   - `probs` (first row, sliced to `num_classes`)
   - `pred_class`

## Configure Dimensions

```python
from particle_transformer import ParticleTransformerAIEConfig, build_and_run

cfg = ParticleTransformerAIEConfig(
    num_particles=150,
    input_dim=20,
    embed_dims=(128, 256, 128),
    num_heads=4,
    num_blocks=4,
    num_cls_blocks=2,
    num_classes=12,
)

out = build_and_run(cfg)
```

## Weight Source Modes

`WeightLoader` supports two modes:
- `manifest+fallback`: if `manifest_path` is provided and found, loads matching ops from manifest/NPY files and falls back to random int8 for missing ops.
- `random`: full deterministic random int8 initialization.

Set via:
- `manifest_path`
- `use_manifest_if_available`

in `ParticleTransformerAIEConfig`.

## Notes

- Execution path is INT8-centric (not INT4).
- Non-tile-aligned dimensions are padded automatically.
- CLS query length is replicated to tile-aligned length (`m`).
- Final prediction uses row `0` and first `num_classes` entries after padded softmax.
