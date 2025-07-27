# Porting nanogpt Linear Layers to AMD AI Engine (AIE)

## Notes
- The goal is to port all linear (matmul) layers from nanogpt to the AMD AI Engine (AIE) using int8 precision.
- Output values will be gibberish, but the focus is on ensuring computational accuracy alignment between PyTorch and AIE implementations.
- There are two possible approaches: (1) quantize the nanogpt model first and then create the skeleton, or (2) create the skeleton with int8 support and implement scaling/clamping as needed.
- Exact scaling/clamping must be matched between PyTorch and AIE for valid accuracy comparisons.
- nanogpt code is present in the external directory of the repo.
- linear_skeleton directory is currently empty and ready for new code.

## Implementation Strategy
1. Create a simplified skeleton of nanogpt that includes only the essential linear layers.
2. Implement int8 quantization for these layers with explicit scaling/clamping.
3. Ensure the PyTorch implementation matches exactly what will be done on the AIE.
4. Add instrumentation to compare outputs between the two platforms.

## Task List
- [x] Decide on approach: quantize first or skeleton with int8 support (we will go with the direct approach without quantizing first)
- [ ] Create a PyTorch linear skeleton of nanogpt supporting int8 operations
- [ ] Implement correct scaling/clamping in PyTorch model
- [ ] Prepare equivalent linear layers for AIE
- [ ] Verify computational alignment between PyTorch and AIE outputs

## Current Goal
Create PyTorch int8 linear skeleton with scaling/clamping