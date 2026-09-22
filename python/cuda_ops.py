"""Load CUDA kernels from kernels/cuda/*.cu and expose them as Python functions.

Compilation happens on the machine that imports this module. On macOS that import
fails (no nvcc). Tests skip when torch.cuda.is_available() is false.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.cpp_extension import load

_CUDA_DIR = Path(__file__).resolve().parents[1] / "kernels" / "cuda"
_MODULES: dict[str, object] = {}


def _load(name: str, source_name: str):
    if name not in _MODULES:
        _MODULES[name] = load(
            name=name,
            sources=[str(_CUDA_DIR / source_name)],
            extra_cuda_cflags=["-O2"],
            verbose=False,
        )
    return _MODULES[name]


def vec_add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return _load("attnlab_vec_add", "vec_add.cu").vec_add(x, y)


# Later, same pattern:
# def gemm_naive(a, b): return _load("attnlab_gemm_naive", "gemm_naive.cu").gemm_naive(a, b)
# def gemm_tiled(a, b): return _load("attnlab_gemm_tiled", "gemm_tiled.cu").gemm_tiled(a, b)
