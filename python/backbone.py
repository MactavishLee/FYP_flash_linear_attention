"""Placeholder for the Python model around the kernel.

Linears, residuals, MLP, and the training loop stay here.
The attention call is one function, swapped as you climb the ladder:

    from ref.ops import attention as attn                 # week 0, any machine
    from python.naive_attention import naive_attention    # separate CUDA kernels
    from python.triton_ops import flash_attn              # add this in month 5

Do not rewrite this file in CUDA.
"""

from __future__ import annotations

import torch
from torch import nn


class AttentionBlock(nn.Module):
    def __init__(self, d_model: int, attn_fn):
        super().__init__()
        self.q = nn.Linear(d_model, d_model, bias=False)
        self.k = nn.Linear(d_model, d_model, bias=False)
        self.v = nn.Linear(d_model, d_model, bias=False)
        self.out = nn.Linear(d_model, d_model, bias=False)
        self.attn_fn = attn_fn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq, d_model). Single head until you add a head split.
        q, k, v = self.q(x), self.k(x), self.v(x)
        return self.out(self.attn_fn(q, k, v))
