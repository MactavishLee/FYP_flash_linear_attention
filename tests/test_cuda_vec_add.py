"""Runs only on a machine with an NVIDIA GPU and nvcc."""

import pytest
import torch

pytest.importorskip("torch")

if not torch.cuda.is_available():
    pytest.skip("CUDA kernel tests need a remote NVIDIA GPU", allow_module_level=True)

from python.cuda_ops import vec_add as cuda_vec_add
from ref.ops import vec_add as ref_vec_add


def test_cuda_vec_add_matches_reference():
    x = torch.randn(10_000, device="cuda")
    y = torch.randn(10_000, device="cuda")
    torch.testing.assert_close(cuda_vec_add(x, y), ref_vec_add(x, y))
