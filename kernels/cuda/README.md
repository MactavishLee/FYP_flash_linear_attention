# `kernels/cuda/` — CUDA C++ only

One `.cu` file per kernel you write by hand for the PMPP ladder:

| File (you add it) | Ladder rung |
|---|---|
| `vec_add.cu` | vector add (starter is already here) |
| `gemm_naive.cu` | naive matrix multiplication |
| `gemm_tiled.cu` | shared-memory tiled GEMM |
| `reduce.cu` | sum / max reduction |
| `softmax.cu` | row softmax built from reductions |
| `scan.cu` | prefix sum |

Rules:

- Device code stays in this folder. Do not put PyTorch model code here.
- Each file exposes a function Python can call (`vec_add`, `gemm_naive`, …) through `python/cuda_ops.py`.
- Compile only on the remote NVIDIA machine or Colab. This Mac has no `nvcc`.
- These kernels are **not** imported by the FlashAttention kernel. Tiling is reused as a technique inside a new fused kernel, not by launching `gemm_tiled` three times from Python.
