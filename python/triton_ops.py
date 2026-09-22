"""Python entry points for kernels/triton/.

Tests and the backbone import this module, not the Triton files directly,
so the launch API stays in one place.
"""

from __future__ import annotations

from kernels.triton.vec_add import vec_add

__all__ = ["vec_add"]
