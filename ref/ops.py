"""Ground-truth ops. Run anywhere PyTorch runs, including this Mac."""

from __future__ import annotations

import torch


def vec_add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return x + y


def matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return a @ b


def row_softmax(x: torch.Tensor) -> torch.Tensor:
    # Stable row-wise softmax. The same max-then-sum pattern becomes
    # online softmax (m, ell) inside the FlashAttention-style kernel.
    m = x.amax(dim=-1, keepdim=True)
    e = torch.exp(x - m)
    return e / e.sum(dim=-1, keepdim=True)


def attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Materialized softmax attention. Writes the N x N score matrix."""
    scale = q.shape[-1] ** -0.5
    scores = matmul(q, k.transpose(-2, -1)) * scale
    return matmul(row_softmax(scores), v)


def elu_plus_one(x: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.elu(x) + 1


def linear_attention_noncausal(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    phi=elu_plus_one,
) -> torch.Tensor:
    """Non-causal linear attention via associativity: phi(Q) (phi(K)^T V).

    This one *is* two matmuls. Causal linear attention is not: it needs the
    running state S_t inside one kernel.
    """
    qh = phi(q)
    kh = phi(k)
    state = matmul(kh.transpose(-2, -1), v)
    return matmul(qh, state)
