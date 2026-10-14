"""Paper 04: Layer Normalization & RMSNorm
Unit Tests Suite.
"""

import math
import torch
import torch.nn as nn


def test_layernorm_feature_dim_stats(ln_cls):
    """Tests that LayerNorm normalizes each sample independently across the feature dimension."""
    all_ok = True
    torch.manual_seed(42)

    # 1. Test 2D input (B=4, D=16)
    x2d = torch.randn(4, 16) * 5.0 + 3.0
    ln2d = ln_cls(normalized_shape=16, elementwise_affine=False)
    out2d = ln2d(x2d)

    mean2d = out2d.mean(dim=-1)
    std2d = out2d.std(dim=-1, unbiased=False)

    if not torch.allclose(mean2d, torch.zeros_like(mean2d), atol=1e-3):
        print(f"  [FAIL] 2D LayerNorm mean not 0.0 (got {mean2d.abs().max().item():.4f})")
        all_ok = False
    if not torch.allclose(std2d, torch.ones_like(std2d), atol=1e-3):
        print(f"  [FAIL] 2D LayerNorm std not 1.0 (got {(std2d - 1.0).abs().max().item():.4f})")
        all_ok = False

    # 2. Test 3D sequence input (B=2, T=8, D=32)
    x3d = torch.randn(2, 8, 32) * 4.0 - 2.0
    ln3d = ln_cls(normalized_shape=32, elementwise_affine=False)
    out3d = ln3d(x3d)

    mean3d = out3d.mean(dim=-1)
    std3d = out3d.std(dim=-1, unbiased=False)

    if not torch.allclose(mean3d, torch.zeros_like(mean3d), atol=1e-3):
        print("  [FAIL] 3D LayerNorm token means not 0.0")
        all_ok = False
    if not torch.allclose(std3d, torch.ones_like(std3d), atol=1e-3):
        print("  [FAIL] 3D LayerNorm token stds not 1.0")
        all_ok = False

    if all_ok:
        print("  [PASS] LayerNorm correctly standardizes each sample/token independently (mean 0.0, std 1.0)!")
    return all_ok


def test_layernorm_affine(ln_cls):
    """Tests learnable affine transformation (gamma, beta)."""
    all_ok = True
    torch.manual_seed(42)

    x = torch.randn(4, 16)
    ln = ln_cls(normalized_shape=16, elementwise_affine=True)
    with torch.no_grad():
        ln.weight.fill_(3.0)
        ln.bias.fill_(-2.0)

    out = ln(x)
    mean = out.mean(dim=-1)
    std = out.std(dim=-1, unbiased=False)

    expected_mean = torch.full_like(mean, -2.0)
    expected_std = torch.full_like(std, 3.0)

    if not torch.allclose(mean, expected_mean, atol=1e-3):
        print("  [FAIL] LayerNorm affine bias shift failed!")
        all_ok = False
    if not torch.allclose(std, expected_std, atol=1e-3):
        print("  [FAIL] LayerNorm affine weight scale failed!")
        all_ok = False

    if all_ok:
        print("  [PASS] LayerNorm affine parameters (gamma=3.0, beta=-2.0) scale and shift correctly!")
    return all_ok


def test_layernorm_equivalence_to_pytorch(ln_cls):
    """Validates exact numerical equivalence against PyTorch native nn.LayerNorm."""
    all_ok = True
    torch.manual_seed(42)

    x = torch.randn(8, 12, 64)
    custom_ln = ln_cls(normalized_shape=64)
    torch_ln = nn.LayerNorm(normalized_shape=64)

    with torch.no_grad():
        torch_ln.weight.copy_(custom_ln.weight)
        torch_ln.bias.copy_(custom_ln.bias)

    out_c = custom_ln(x)
    out_t = torch_ln(x)
    diff = (out_c - out_t).abs().max().item()

    if diff > 1e-5:
        print(f"  [FAIL] LayerNorm divergence from PyTorch: max diff = {diff:.2e}")
        all_ok = False
    else:
        print(f"  [PASS] Custom LayerNorm is mathematically identical to PyTorch LayerNorm (diff: {diff:.2e})!")

    return all_ok


def test_rmsnorm_scaling(rmsnorm_cls):
    """Tests RMSNorm root mean square scaling and unit RMS output."""
    all_ok = True
    torch.manual_seed(42)

    dim = 32
    x = torch.randn(4, 8, dim) * 3.0
    rms_norm = rmsnorm_cls(dim=dim)

    out = rms_norm(x)

    # Compute root mean square of the output: sqrt(mean(out^2))
    rms_val = torch.sqrt(torch.mean(out ** 2, dim=-1))

    if not torch.allclose(rms_val, torch.ones_like(rms_val), atol=1e-3):
        max_err = (rms_val - 1.0).abs().max().item()
        print(f"  [FAIL] RMSNorm output does not have unit RMS (max err: {max_err:.4f})")
        all_ok = False
    else:
        print("  [PASS] RMSNorm scales vectors to exact unit root-mean-square!")

    return all_ok


def run_all_tests():
    """Runs all test gates from the terminal."""
    print("=" * 60)
    print("Running Paper 04 (LayerNorm & RMSNorm) Test Suite")
    print("=" * 60)

    try:
        from model import LayerNorm, RMSNorm
    except ImportError:
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from model import LayerNorm, RMSNorm

    try:
        ok1 = test_layernorm_feature_dim_stats(LayerNorm)
        ok2 = test_layernorm_affine(LayerNorm)
        ok3 = test_layernorm_equivalence_to_pytorch(LayerNorm)
        ok4 = test_rmsnorm_scaling(RMSNorm)

        if all([ok1, ok2, ok3, ok4]):
            print("\n" + "=" * 60)
            print("ALL PAPER 04 TESTS PASSED! YOU INVENTED LAYERNORM & RMSNORM!")
            print("=" * 60)
    except NotImplementedError:
        print("\n[PENDING] Implement LayerNorm and RMSNorm in 'papers/04_layer_norm/model.py' to pass tests.")


if __name__ == "__main__":
    run_all_tests()
