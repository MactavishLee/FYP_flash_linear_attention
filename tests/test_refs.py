"""CPU-safe tests. These run on the Mac; they never launch a kernel."""

import torch

from python.backbone import AttentionBlock
from python.naive_attention import naive_attention
from ref.ops import attention, linear_attention_noncausal, matmul, row_softmax, vec_add


def test_vec_add_and_matmul():
    x = torch.randn(128)
    y = torch.randn(128)
    torch.testing.assert_close(vec_add(x, y), x + y)
    a = torch.randn(32, 16)
    b = torch.randn(16, 8)
    torch.testing.assert_close(matmul(a, b), a @ b)


def test_naive_attention_matches_reference():
    torch.manual_seed(0)
    q = torch.randn(2, 16, 8)
    k = torch.randn(2, 16, 8)
    v = torch.randn(2, 16, 8)
    got = naive_attention(q, k, v, matmul=matmul, softmax=row_softmax)
    torch.testing.assert_close(got, attention(q, k, v))


def test_linear_attention_matches_explicit_formula():
    torch.manual_seed(1)
    q = torch.randn(2, 20, 8)
    k = torch.randn(2, 20, 8)
    v = torch.randn(2, 20, 4)
    phi = torch.nn.functional.elu
    # identity-like check against the associated product with the same phi
    got = linear_attention_noncausal(q, k, v, phi=lambda t: phi(t) + 1)
    qh, kh = phi(q) + 1, phi(k) + 1
    expect = qh @ (kh.transpose(-2, -1) @ v)
    torch.testing.assert_close(got, expect)


def test_backbone_uses_python_attention():
    torch.manual_seed(2)
    block = AttentionBlock(d_model=8, attn_fn=attention)
    x = torch.randn(2, 6, 8)
    y = block(x)
    assert y.shape == x.shape
