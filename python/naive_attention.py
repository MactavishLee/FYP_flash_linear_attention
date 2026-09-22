"""Naive attention as Python orchestration of separate kernels.

This is the step *before* FlashAttention. It is correct, and it materializes
the N x N score matrix because each kernel returns to Python.

FlashAttention must not call this. It is a new fused kernel under kernels/triton/.
"""

from __future__ import annotations

from collections.abc import Callable

import torch

Matmul = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
Softmax = Callable[[torch.Tensor], torch.Tensor]


def naive_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    matmul: Matmul,
    softmax: Softmax,
) -> torch.Tensor:
    """softmax(Q K^T / sqrt(d)) V, using whatever matmul/softmax you pass in.

    Pass ref.ops.matmul / ref.ops.row_softmax on a laptop.
    Pass CUDA kernels once gemm_naive.cu and softmax.cu exist.
    """
    scale = q.shape[-1] ** -0.5
    scores = matmul(q, k.transpose(-2, -1)) * scale  # N x N lives in GPU memory
    probs = softmax(scores)  # another N x N
    return matmul(probs, v)
