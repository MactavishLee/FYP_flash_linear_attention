"""PyTorch reference implementations. These are the correctness oracle.

No CUDA and no Triton live here. Tests compare every kernel against this package.
"""

from ref.ops import (
    attention,
    linear_attention_noncausal,
    matmul,
    row_softmax,
    vec_add,
)

__all__ = [
    "attention",
    "linear_attention_noncausal",
    "matmul",
    "row_softmax",
    "vec_add",
]
