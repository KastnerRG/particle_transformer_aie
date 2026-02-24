# AIE Particle Transformer Change Report

## Scope
This report covers only the AIE-side implementation work in `particle_transformer_aie/` (layers, kernels, utilities, and model builder flow), not training/export code in the main PyTorch repo.

## Functionality Increases

### 1) Bias-aware dense path across the stack
- Prior: Dense and attention projections were effectively weight-only in the AIE generation flow.
- Needed: Quantized inference needed explicit int8 bias support for encoder/FFN/classifier and q/k/v/o projections.
- Problem: Bias had to be applied in a tile-safe way inside int8 kernels while preserving saturation behavior and codegen compatibility.
- Implemented: Added `dense_bias` in `aie/kernels.h`, added optional `bias` in `DenseLayer`, and propagated optional `bq/bk/bv/bo` through `MHALayer` and `CLSAttentionLayer`, with generated kernel calls switching between `dense` and `dense_bias` as needed.

### 2) LUT softmax for attention and classifier output
- Prior: No softmax implementation
- Needed: Deterministic int8-compatible softmax for both attention scores and final class logits.
- Problem: True exp/normalize is expensive in integer AIE kernels and needed to stay tile-stream compatible.
- Implemented: Added `utils/softmax_lut.py` (`build_exp_lut_256`, `softmax_lut_int8_rowwise`, C initializer helper), added `softmax_rows` kernel template in `aie/kernels.h`, integrated per-head softmax nodes in `MHALayer`/`CLSAttentionLayer`, and added a standalone `SoftmaxLayer` for final classifier probabilities.

### 3) True CLS cross-attention layer
- Prior: No implementation of CLS cross-attention blocks (although the ingredients existed)
- Needed: CLS pooling path where query length and token length are decoupled, including first-block internal CLS token mode.
- Problem: Self-attention kernels assumed equal query/key sequence structure and graph wiring needed two-mode behavior (internal vs external CLS query).
- Implemented: Added `layers/cls_attention.py` with:
  - Mode A: internal CLS query via `emit_const` kernel source.
  - Mode B: explicit CLS input + particle K/V stream.
  - Per-head q/k/v, scores, softmax, context, concat, out generation with cross-attention kernels.

### 4) Configurable end-to-end particle transformer builder
- Prior: `particle_transformer.py` was a fixed/random demo pipeline.
- Needed: A shape-configurable architecture matching ParT inference structure capable of running a real model.
- Problem: Dimensions must remain tile-safe (`m/k/n`) while preserving logical model dimensions and class slicing.
- Implemented: Rebuilt `particle_transformer.py` around `ParticleTransformerAIEConfig`, `build_model`, and `build_and_run`, with:
  - Encoder `D -> 128 -> 256 -> 128` (configurable),
  - `num_blocks` transformer blocks (default 4),
  - `num_cls_blocks` CLS cross-attention blocks (default 2),
  - Final FC + softmax output,
  - Structured outputs: `logits_padded`, `probs_padded`, `probs`, `pred_class`.

### 5) Dual weight-source behavior with random fallback
- Prior: Model assembly relied on ad hoc random tensors.
- Needed: Smooth path for manifest-backed int8 artifacts when available, with deterministic fallback when missing.
- Problem: Exported artifacts can differ in stored layout/orientation and may have partial coverage across ops.
- Implemented: Added `utils/weight_loader.py` with manifest-aware op lookup, shape adaptation, bias adaptation, optional int32-bias downshift using manifest shifts, and random fallback for missing entries.

## Code Fixes / Robustness Improvements

### 1) Explicit tile divisibility and shape checks
- Prior: Several paths assumed tile-compatible inputs without strict validation.
- Needed: Early failures for invalid dimensions to avoid silent mis-wiring or malformed generated kernels.
- Problem: Dynamic configs can produce non-divisible shapes that only fail late at codegen/sim time.
- Implemented: Added asserts in `DenseLayer`, `MHALayer`, `CLSAttentionLayer`, and `SoftmaxLayer` for divisibility and expected tensor shapes before kernel generation.

### 2) Consistent int8 saturation/clipping semantics
- Prior: Behavior around intermediate clipping/bias handling was not uniformly enforced across new paths.
- Needed: Predictable int8 math behavior between NumPy golden and generated kernels.
- Problem: Mixed integer types and optional bias/residual paths can diverge if clipping is inconsistent.
- Implemented: Standardized clipping/saturation in NumPy golden utilities and kernel-side paths (`dense_bias`, `output`, LUT softmax clamping to `[0, out_scale]`).

### 3) Cross-attention reference correctness in NumPy golden
- Prior: Golden MHA helper was self-attention-oriented.
- Needed: Golden reference had to support `Tq != Tk` for CLS cross-attention and LUT-softmax behavior.
- Problem: Without this, shift extraction and codegen params for cross-attention would be unreliable.
- Implemented: Extended `utils/np_mha_linear.py` to support cross-attention (`q`,`k`,`v` decoupling), optional LUT softmax, bias-aware projections, and per-head score/context shift capture.

## Performance-Oriented Improvements

### 1) Fused output-stage additions in AIE kernel
- Prior: Output projection and optional additions were separate conceptual operations.
- Needed: Reduce extra passes where possible in projection output stage.
- Problem: Additional kernels/streams for simple adds increase overhead and graph complexity.
- Implemented: Extended `output` template in `aie/kernels.h` with compile-time flags `use_bias` and `use_residual`, enabling fused projection + optional bias/residual add + activation in one kernel body.

### 2) Internal CLS query emission from constant tiled buffer
- Prior: CLS query would require separate runtime stream preparation.
- Needed: Efficient first CLS block query source without external producer complexity.
- Problem: Query length is tiny (`1`, padded to `m`) and awkward for normal streaming setup.
- Implemented: Added `emit_const` kernel and used it in `CLSAttentionLayer` internal mode to source tiled CLS query data directly.

## AIE Kernel Extensions

Added/extended templates in `aie/kernels.h`:
- `dense_bias<...>`: dense matmul + requant + optional relu + int8 bias add with saturation.
- `softmax_rows<...>`: row-wise LUT softmax for tiled int8 score matrices.
- `scores_cross<...>`: cross-attention score computation with decoupled query/key tile counts.
- `context_cross<...>`: cross-attention context projection with decoupled query/key tile counts.
- `output<..., use_bias, use_residual>`: extended output projection supporting optional bias/residual fusion.
- `emit_const<...>`: constant tiled stream emitter (used for internal CLS query path).

## Particle Transformer Implementation (`particle_transformer.py`)

### Usability and configurability
- `ParticleTransformerAIEConfig` centralizes model dimensions, block counts, tiling (`m/k/n`), seed, manifest mode, and default shift behavior.
- `build_model(cfg)` assembles the full graph with manifest-backed or random int8 params.
- `build_and_run(cfg, input_particles)` handles input padding, graph execution, and post-processing into user-facing outputs.

### Tile-safe padding behavior
- Particle count padded to multiple of `m`.
- Feature/input dims padded to multiple of `k`.
- Hidden/class dims padded to multiples compatible with `k/n` and head partitioning.
- CLS query length padded/replicated to tile-aligned length (`m`).

### Implemented ParT structure
- 3-layer embedding encoder (`D -> 128 -> 256 -> 128`, configurable).
- 4 transformer token blocks by default (MHA + residual + FFN + residual).
- 2 CLS cross-attention blocks by default (attention + residual policy + FFN + residual).
- Final FC layer to `num_classes` (padded internally), then LUT softmax.
- Final prediction extraction from row `0`, sliced to logical `num_classes`.
