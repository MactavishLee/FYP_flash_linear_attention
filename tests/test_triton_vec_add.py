"""Runs only on a machine with Triton and an NVIDIA GPU."""

import pytest
import torch

pytest.importorskip("triton")

if not torch.cuda.is_available():
    pytest.skip("Triton kernel tests need a remote NVIDIA GPU", allow_module_level=True)

from python.triton_ops import vec_add as triton_vec_add
from ref.ops import vec_add as ref_vec_add


def test_triton_vec_add_matches_reference():
    x = torch.randn(10_000, device="cuda")
    y = torch.randn(10_000, device="cuda")
    torch.testing.assert_close(triton_vec_add(x, y), ref_vec_add(x, y))
