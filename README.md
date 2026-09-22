# gpu-attn-kernels

Learning repo for writing a FlashAttention-style forward and a linear-attention forward in half a year. The 26-week plan is `~/Desktop/gpu-attention-kernel-6month-plan.md`.

Python is the backbone. CUDA C++ is how the PMPP exercises are written. Triton is how the first attention kernels are written. This Mac only edits and runs the reference tests. Kernels compile on a remote NVIDIA GPU (lab A800/5090, Colab, or Lightning).

## Where each language goes

```text
gpu-attn-kernels/
  ref/                      PyTorch oracles. No kernels. Safe on the Mac.
  kernels/cuda/             CUDA C++ (.cu). One file per PMPP kernel.
  kernels/triton/           Triton kernels (.py syntax, GPU device code).
  python/                   Host Python: wrappers, naive attention, the model.
  tests/                    pytest. CPU tests always; GPU tests skip without CUDA.
  notes/                    weekly notes, shapes, atol.
  scripts/check_gpu.py      run on the remote machine.
```

```text
tests
  │
  ├─ compare against ref/ops.py
  │
  └─ call python/                 ← this is the only API the model uses
        ├─ cuda_ops.py    ──load──►  kernels/cuda/*.cu
        ├─ triton_ops.py  ──import─► kernels/triton/*.py
        ├─ naive_attention.py      Python calls matmul, then softmax, then matmul
        └─ backbone.py             Linears / residual / MLP stay here
```

## What you are allowed to compose in Python

| You want | Where it lives | Python may call the CUDA matmul? |
|---|---|---|
| Vector add, tiled GEMM, reduction, scan | `kernels/cuda/` | n/a — each is its own kernel |
| Naive attention | `python/naive_attention.py` | **Yes.** Three launches. The N×N matrix is stored. |
| Non-causal linear attention | two GEMMs | **Yes.** `phi(Q) @ (phi(K).T @ V)` really is two matmuls. |
| FlashAttention-style forward | `kernels/triton/flash_attn.py` (you write it) | **No.** One fused launch. Tiling stays inside the kernel. |
| Causal / chunkwise linear attention | `kernels/triton/linear_attn.py` (you write it) | **No.** The running state `S_t` lives inside the kernel. |

## Commands

On the Mac (references only):

```bash
python -m pip install -r requirements.txt
python -m pytest tests/test_refs.py
```

On the NVIDIA machine, from this repo:

```bash
python -m pip install -r requirements.txt triton
python scripts/check_gpu.py
python -m pytest tests/test_cuda_vec_add.py tests/test_triton_vec_add.py
```

`kernels/cuda/vec_add.cu` and `kernels/triton/vec_add.py` are starters so the folders have a real call path. Rewrite the CUDA one yourself in week 1.
