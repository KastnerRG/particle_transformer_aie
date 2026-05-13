# Particle Transformer on Versal AI Engines

A reusable software framework for mapping quantized transformer models to AMD Versal VCK190 AI Engine (AIE). This project demonstrates an integer-only transformer implementation for jet tagging at particle accelerators, featuring a code-generation framework that automatically converts high-level Python model descriptions into optimized Vitis AIE graphs.

**Paper:** "Reconfigurable Computing Challenge: Transformer for Jet Tagging on Versal AI Engines" - FCCM Journal
**Authors:** Gram Koski, Sean Lipps, Zhenghua Ma, G Abarajithan, Ryan Kastner (UC San Diego)

## Project Overview

This framework addresses the challenge of deploying transformer-based neural networks in real-time, resource-constrained edge inference systems. It is motivated by low-level triggers for jet tagging at particle accelerators, where the ideal constraints are:

- **Input Rate:** 40 MHz collision rate
- **Latency Budget:** A few microseconds for Level-1 Trigger decisions
- **Throughput Requirement:** O(10⁵) events per second
- **Constraints:** Tight on-detector power and resource budgets

Traditional CPU/GPU platforms cannot meet these requirements. **This is a preliminary, unoptimized implementation that demonstrates the framework's feasibility and establishes a foundation for future optimization.** Currently, performance is limited by bottlenecks such as integer-only softmax (55+ µs latency) that prevent real-time operation at production scale. This project leverages the AMD Versal VCK190 SoC's AI Engine array for low-latency, high-throughput ML inference using quantized integer-only arithmetic, with potential for improvement in future work.

### Key Contributions

1. **Modular Code-Generation Framework:** Automatically generates Vitis AIE graphs from high-level Python model descriptions with composable building blocks (Dense, MHA, ResAdd, DenseSoftmax layers)

2. **Integer-Only Transformer:** Fully quantized implementation using int8 weights/activations and int32 accumulators with fixed-point rescaling, including a novel integer-only softmax

3. **Head-Parallel MHA Mapping:** Assigns attention heads to parallel AIE tiles, achieving ~4× throughput improvement (1 head: 734.6 µs vs. 4 heads: 187.0 µs)

4. **Validation Framework:** Every model automatically runs both AIE emulation and a NumPy golden reference for numerical correctness verification

## Installation and Setup

### Requirements

- **Vitis 2024.1** or later (with appropriate licenses)
- **Python 3.8+**
- **NumPy** and dependencies listed in `environment.yml`

### Setup

```bash
# Create conda environment
conda env create -f environment.yml
conda activate particle-transformer

# Verify installation
python -c "import numpy; print('NumPy OK')"
```

## The AIEModel Framework

The framework exposes a clean Python API for building transformer models without manually writing C++ kernels or Vitis graph code:

```python
from model import AIEModel
from layers import DenseLayer, MHALayer, ResAddLayer

# Create model with AIE grid parameters (m, k, n)
model = AIEModel(m=4, k=8, n=8, iterations=1, dynamic_quant=True)

# Add layers
dense = DenseLayer(name='dense_0', weight=W, bias=b, relu=True)
model.add_layer(dense, inputs=[None])

mha = MHALayer(name='mha_1', Wq=Wq, Wk=Wk, Wv=Wv, Wo=Wo, num_heads=4, ...)
model.add_layer(mha, inputs=[dense])

# Forward pass: generates code, compiles, emulates, validates
output = model.forward(input_data)
```

### Framework Workflow

1. **Model Definition** → High-level Python layer specification
2. **Golden Reference** → NumPy implementation for validation
3. **Code Generation** → C++ kernels and Vitis graph instantiation
4. **Compilation** → Vitis 2024.1 compilation
5. **AIE Emulation** → Cycle-accurate simulation
6. **Numerical Validation** → AIE output vs. NumPy reference comparison

## Examples

Each example demonstrates different aspects of the framework. All use quantized int8 arithmetic on a 160×8 input tensor (padded from 150×3 jets with 3 features).

### 1. `skeleton.py` - Transformer Backbone (No Softmax)

**Purpose:** Validate core transformer computation without softmax overhead
**Use Case:** Demonstrates the skeleton model evaluated in the paper's benchmarks

**Architecture:**
- Dense layer (3→64) with ReLU
- Multi-head attention (4 heads, no softmax, no bias)
- Residual connection (skip)
- 2 stacked transformer blocks, each with:
  - MHA layer (4 heads, no softmax)
  - Feed-forward: Dense(64→64)→Dense(64→64) with ReLU
  - Residual connections
- Output projection: Dense(64→8)

**Key Features:**
- Omits softmax and bias for latency isolation
- Uses dynamic quantization (calibrated from input)
- Directly maps to Table I results in the paper
- **Performance:** 4 heads: 187.0 ms latency, 10,041.8 samples/s; 1 head: 734.6 ms latency, 2,775.3 samples/s

**Run:** `python examples/skeleton.py`

### 2. `mlp.py` - Simple Multi-Layer Perceptron

**Purpose:** Demonstrate basic dense layer operations and quantization schemes
**Use Case:** Validates foundational building blocks; lightweight test for framework debugging

**Architecture:**
- FC1: 8→64 with bias and ReLU (shift=15, scale=73)
- FC2: 64→8 with bias, no ReLU (shift=15, scale=58)

**Key Features:**
- Single MLP with explicit static quantization parameters
- No dynamic quantization; fixed quantization scales and shifts
- Small model suitable for quick iteration and testing
- Good starting point for understanding DenseLayer quantization

**Run:** `python examples/mlp.py`

### 3. `dense_softmax_model.py` - Dense Layer with Integer Softmax

**Purpose:** Benchmark the integer-only softmax implementation
**Use Case:** Demonstrates softmax layer as a separate building block; crucial for identifying performance bottlenecks

**Architecture:**
- DenseSoftmax: 64→64 layer combining dense projection and softmax
- Input: 160×64 quantized tensor
- Output: 160×64 softmax probabilities

**Key Features:**
- Isolated softmax performance analysis
- Fixed quantization (no dynamic scaling)
- **Performance Impact (from paper Table II):**
  - Dense only: 218.3 ns latency, 697.7 MB/s throughput
  - Dense + bias: 1,695.8 ns latency, 22.4 MB/s throughput  
  - Dense + softmax: 55,199.2 ns latency, 4.6 MB/s throughput
  - **Softmax introduces ~250× latency increase, identifying it as the dominant bottleneck**

**Run:** `python examples/dense_softmax_model.py`

### 4. `particle_transformer_no_softmax.py` - Full Transformer (No Softmax, Dynamic Quant)

**Purpose:** Complete transformer pipeline with automatic quantization calibration
**Use Case:** Production-oriented model using dynamic quantization; equivalent to skeleton but with calibration

**Architecture:** Same as `skeleton.py` but with:
- All layers include bias terms
- Dynamic quantization enabled (auto-calibrated)
- Quantization factors derived from reference forward pass

**Key Differences from skeleton:**
- ✓ Bias terms on dense and attention layers
- ✓ Dynamic quantization mode (automatic parameter derivation)
- ✗ Still no softmax (avoiding the identified bottleneck)
- ✗ No softmax comparison needed

**Run:** `python examples/particle_transformer_no_softmax.py`

### 5. `particle_transformer.py` - Full Transformer (With Softmax, Static Quant)

**Purpose:** Complete production transformer matching the paper's full implementation
**Use Case:** Reference implementation with optional softmax enabling; demonstrates static quantization

**Architecture:** Full transformer with:
- Dense layer with bias and explicit quantization (shift=15, scale=115)
- MHA with per-head static quantization scales:
  - `scale_q, shift_q`: Query projection
  - `scale_k, shift_k`: Key projection
  - `scale_v, shift_v`: Value projection
  - `scale_s, shift_s` (4 values): Per-head softmax scaling
  - `scale_c, shift_c` (4 values): Per-head context scaling
  - `scale_o, shift_o`: Output projection
- Feed-forward layers with explicit quantization parameters

**Key Differences from particle_transformer_no_softmax:**
- ✓ Static quantization with explicit per-layer scales/shifts
- ✓ Optional softmax enabling (parameter: `enable_softmax=True/False`)
- ✓ Per-head quantization for improved precision
- ✗ More memory bandwidth due to softmax data movement

**Configuration Options:**
```python
# Disable softmax (faster)
model.forward(input_data, enable_softmax=False)  

# Enable softmax (production, higher latency)
model.forward(input_data, enable_softmax=True)
```

**Run:** `python examples/particle_transformer.py`

---

## Example Comparison Table

| Example | Purpose | Softmax | Quantization | Bias | Complexity | Use Case |
|---------|---------|---------|--------------|------|------------|----------|
| `skeleton.py` | Core validation | ✗ | Dynamic | ✗ | Medium | Baseline performance, paper Table I |
| `mlp.py` | Foundational blocks | ✗ | Static | ✓ | Low | Framework testing, quick iteration |
| `dense_softmax_model.py` | Softmax analysis | ✓ | Static | ✓ | Very Low | Bottleneck identification, paper Table II |
| `particle_transformer_no_softmax.py` | Transformer backbone | ✗ | Dynamic | ✓ | High | Production without softmax |
| `particle_transformer.py` | Full transformer | ✓ | Static | ✓ | High | Complete reference implementation |

## Performance Results

All experiments conducted on the AMD Versal VCK190 SoC using AIE hardware emulation with randomly initialized weights (as noted in the paper).

### Table I: Skeleton Model Latency & Throughput (No Bias, No Softmax)
| Configuration | Latency (ns) | Throughput (MB/s) | Throughput (samples/s) |
|---------------|--------------|-------------------|------------------------|
| 4 heads | 187,014.2 | 12.85 | 10,041.8 |
| 1 head | 734,559.2 | 3.55 | 2,775.3 |
| **Speedup** | **3.93×** | **3.62×** | **3.62×** |

**Key Insight:** Head-parallel execution achieves ~4× throughput improvement by distributing attention heads across AIE tiles.

### Table II: Impact of Bias and Softmax on Single Dense Layer
| Configuration | Latency (ns) | Throughput (MB/s) |
|---------------|--------------|-------------------|
| Dense only | 218.3 | 697.66 |
| Dense + bias | 1,695.8 | 22.41 |
| Dense + softmax (no bias) | 55,199.2 | 4.60 |
| Dense + bias + softmax | 55,392.5 | 4.59 |
| **Softmax Overhead** | **~250×** | **~150×** |

**Key Insight:** Integer-only softmax and associated data movement form the dominant bottleneck, introducing orders-of-magnitude latency increase. This is why `skeleton.py` and `particle_transformer_no_softmax.py` omit softmax for real-time inference.

## Quantization Strategy

All models use symmetric per-tensor quantization with int8 weights and activations:

- **Weight Format:** int8 (range: -128 to 127)
- **Activation Format:** int8 (range: -128 to 127)
- **Accumulator Format:** int32 (for multiplication products)
- **Rescaling:** Fixed-point with per-layer scale and shift parameters

### Dynamic vs. Static Quantization

**Dynamic Quantization** (`dynamic_quant=True`):
- Automatically calibrates quantization scales from the reference forward pass
- Useful for randomly initialized weights or unknown data distributions
- Used by `skeleton.py` and `particle_transformer_no_softmax.py`

**Static Quantization** (`dynamic_quant=False`):
- User-specified scale and shift parameters
- Better for production with known input distributions
- Enables per-head quantization for improved precision (MHA layers)
- Used by `mlp.py`, `dense_softmax_model.py`, and `particle_transformer.py`

## Framework Architecture

### Layer Types

1. **DenseLayer** - Quantized matrix multiplication with optional bias and ReLU
2. **MHALayer** - Multi-head attention with per-head quantization and optional softmax
3. **ResAddLayer** - Residual connection (element-wise addition)
4. **DenseSoftmaxLayer** - Combined dense + integer softmax operation

### Code Generation Pipeline

```
Python Model Definition
    ↓
AIEModel.add_layer() [builds DAG]
    ↓
AIEModel.forward() calls:
    ├→ _compute_golden() [NumPy reference]
    ├→ _generate_code() [C++ kernels & graph]
    ├→ _compile_and_simulate() [Vitis emulation]
    └→ _validate() [compare outputs]
    ↓
Output validation or error report
```

### Directory Structure

```
aie/
├─ kernels.h          # AIE kernel library (C++)
examples/
├─ skeleton.py                      # Core transformer validation
├─ mlp.py                          # Simple MLP test
├─ dense_softmax_model.py          # Softmax benchmarking
├─ particle_transformer.py          # Full transformer (with softmax)
└─ particle_transformer_no_softmax.py  # Full transformer (no softmax)
layers/
├─ __init__.py
├─ base.py            # AIELayer abstract base class
├─ dense.py           # DenseLayer implementation
├─ dense_softmax.py   # DenseSoftmaxLayer implementation
├─ mha.py             # MHALayer (multi-head attention)
└─ resadd.py          # ResAddLayer (residual addition)
utils/
├─ integer_modules.py # Integer arithmetic utilities
├─ np_mha_linear.py   # NumPy attention reference
└─ tiling.py          # Tensor tiling utilities
model.py              # AIEModel framework
```

## Future Work

1. **Optimized Softmax Implementation:** Reduce integer softmax latency from 55+ µs to acceptable levels for real-time inference

2. **Extended Layer Support:** Add integer-only LayerNorm, pooling, and other common transformer components

3. **Advanced Quantization:** Move beyond symmetric per-tensor scaling to per-channel or mixed-precision schemes

4. **Trained Model Integration:** Evaluate with actual trained jet tagging models and realistic calibration data (currently using random weights for validation)

5. **Enhanced Tiling Strategies:** Explore alternative tile mappings and load-balancing schemes for non-uniform workloads

## Building and Running

```bash
# Run the skeleton model (baseline)
python examples/skeleton.py

# Run the simple MLP test
python examples/mlp.py

# Run softmax benchmarking
python examples/dense_softmax_model.py

# Run full transformer (no softmax)
python examples/particle_transformer_no_softmax.py

# Run full transformer (with softmax)
python examples/particle_transformer.py
```

## Citation

If you use this framework in your research, please cite:

```bibtex
@article{koski2024transformer,
  title={Reconfigurable Computing Challenge: Transformer for Jet Tagging on Versal AI Engines},
  author={Koski, Gram and Lipps, Sean and Ma, Zhenghua and Abarajithan, G and Kastner, Ryan},
  journal={FCCM},
  year={2024}
}
```

## License

Open-source software released for research and development.

## Contact

For questions or contributions, contact the authors at UC San Diego Department of Computer Science and Engineering.

