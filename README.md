# FYP-Flash-Linear-Attention

This is my FYP project: implement Flash and Linear Attention by Triton & CUDA, and integrated it into a open-sourced linear LLM. 

Reference book: Programming Massively Parallel Processors (PMPP)

Python is the backbone. 
CUDA C++ is how some kernels are written. 
Triton is how the attention kernels are written. 

## Where each language goes

```text
gpu-attn-kernels/
  ref/                      PyTorch oracles for references. inplemented by python/pytorch. No kernels. Safe on the Mac.
  kernels/cuda/             CUDA C++ (.cu). One file per PMPP kernel.
  kernels/triton/           Triton kernels (.py syntax, GPU device code).
  tests/                    testing performance of kernels and attention module.
  notes/                    weekly notes and other technical documents.
  
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

