"""Starter Triton vector add. Runs only where Triton has a CUDA (or HIP) device."""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _vec_add_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    tl.store(out_ptr + offsets, x + y, mask=mask)


def vec_add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    if x.shape != y.shape:
        raise ValueError(f"shape mismatch: {tuple(x.shape)} vs {tuple(y.shape)}")
    out = torch.empty_like(x)
    n = x.numel()
    block = 128
    grid = (triton.cdiv(n, block),)
    _vec_add_kernel[grid](x, y, out, n, BLOCK=block)
    return out
