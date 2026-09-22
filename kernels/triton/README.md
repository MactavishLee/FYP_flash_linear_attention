# `kernels/triton/` — Triton kernels

Triton source is `.py`, but it is **device code**, not the model backbone. Put it here, not in `python/`.

| File (you add it) | When |
|---|---|
| `vec_add.py` | first Triton program (starter is already here) |
| `gemm.py` | Triton matmul, after the CUDA tiled GEMM is correct |
| `flash_attn.py` | fused FlashAttention-style **forward**. One launch. |
| `linear_attn.py` | recurrent `S_t`, then chunkwise |

`python/triton_ops.py` imports these and is what tests call.
Do not start `flash_attn.py` until CUDA tiled GEMM, reduction/softmax, and scan exist.
