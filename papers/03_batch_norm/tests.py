"""Paper 03: Batch Normalization (Ioffe & Szegedy, 2015)
Unit Tests Suite.
"""

import math
import torch
import torch.nn as nn


def test_batchnorm1d_training_stats(bn1d_cls):
    """Tests that BatchNorm1d normalizes training batch to zero mean and unit variance."""
    all_ok = True
    torch.manual_seed(42)

    # Input with mean=5.0 and std=3.0
    x = torch.randn(64, 16) * 3.0 + 5.0
    bn = bn1d_cls(num_features=16, affine=False)
    bn.train()
    out = bn(x)

    mean = out.mean(dim=0)
    std = out.std(dim=0, unbiased=False)

    if not torch.allclose(mean, torch.zeros_like(mean), atol=1e-3):
        print(f"  [FAIL] Training output mean not 0.0 (max mean: {mean.abs().max().item():.4f})")
        all_ok = False
    if not torch.allclose(std, torch.ones_like(std), atol=1e-3):
        print(f"  [FAIL] Training output std not 1.0 (max std diff: {(std - 1.0).abs().max().item():.4f})")
        all_ok = False

    if all_ok:
        print("  [PASS] BatchNorm1d correctly standardizes batch to mean 0.0 and std 1.0!")
    return all_ok


def test_batchnorm1d_affine(bn1d_cls):
    """Tests learnable affine transformation (gamma, beta)."""
    all_ok = True
    torch.manual_seed(42)

    x = torch.randn(64, 8)
    bn = bn1d_cls(num_features=8, affine=True)
    with torch.no_grad():
        bn.weight.fill_(2.5)
        bn.bias.fill_(-1.5)

    bn.train()
    out = bn(x)

    mean = out.mean(dim=0)
    std = out.std(dim=0, unbiased=False)

    expected_mean = torch.full_like(mean, -1.5)
    expected_std = torch.full_like(std, 2.5)

    if not torch.allclose(mean, expected_mean, atol=1e-3):
        print("  [FAIL] Affine bias shift failed!")
        all_ok = False
    if not torch.allclose(std, expected_std, atol=1e-3):
        print("  [FAIL] Affine weight scale failed!")
        all_ok = False

    if all_ok:
        print("  [PASS] Learnable affine parameters (gamma=2.5, beta=-1.5) scale and shift correctly!")
    return all_ok


def test_batchnorm1d_running_stats_and_eval(bn1d_cls):
    """Tests EMA running buffers update and deterministic behavior during eval() mode."""
    all_ok = True
    torch.manual_seed(42)

    bn = bn1d_cls(num_features=4, momentum=0.1)
    bn.train()

    # Pass multiple batches
    for _ in range(20):
        x = torch.randn(32, 4) * 2.0 + 3.0
        bn(x)

    # Running statistics should approximate mean=3.0, var=4.0
    r_mean = bn.running_mean.mean().item()
    r_var = bn.running_var.mean().item()

    if not (2.5 < r_mean < 3.5):
        print(f"  [FAIL] Running mean did not converge to ~3.0 (got {r_mean:.3f})")
        all_ok = False
    if not (3.0 < r_var < 5.0):
        print(f"  [FAIL] Running variance did not converge to ~4.0 (got {r_var:.3f})")
        all_ok = False

    # Switch to eval mode and test with single sample (B=1)
    bn.eval()
    single_x = torch.randn(1, 4) * 2.0 + 3.0
    try:
        out_eval = bn(single_x)
        if out_eval.shape != (1, 4):
            print("  [FAIL] Eval output shape mismatch on B=1")
            all_ok = False
    except Exception as e:
        print(f"  [FAIL] Failed to evaluate on B=1: {e}")
        all_ok = False

    if all_ok:
        print(f"  [PASS] Running stats updated (mean: {r_mean:.2f}, var: {r_var:.2f}) and eval(B=1) works flawlessly!")
    return all_ok


def test_batchnorm2d_spatial(bn2d_cls):
    """Tests 4D spatial BatchNorm2d across (B, H, W) for each channel C."""
    all_ok = True
    torch.manual_seed(42)

    # Input: (B=8, C=4, H=16, W=16)
    x = torch.randn(8, 4, 16, 16) * 4.0 + 2.0
    bn = bn2d_cls(num_features=4, affine=False)
    bn.train()
    out = bn(x)

    # Reduction over dims (0, 2, 3) should yield mean 0, std 1 for each of 4 channels
    mean_per_c = out.mean(dim=(0, 2, 3))
    std_per_c = out.std(dim=(0, 2, 3), unbiased=False)

    if not torch.allclose(mean_per_c, torch.zeros_like(mean_per_c), atol=1e-3):
        print("  [FAIL] BatchNorm2d channel means are not 0.0")
        all_ok = False
    if not torch.allclose(std_per_c, torch.ones_like(std_per_c), atol=1e-3):
        print("  [FAIL] BatchNorm2d channel stds are not 1.0")
        all_ok = False

    if all_ok:
        print("  [PASS] BatchNorm2d correctly normalizes across all spatial pixels per channel!")
    return all_ok


def test_equivalence_to_pytorch(bn1d_cls, bn2d_cls):
    """Verifies numerical equivalence against PyTorch's native nn.BatchNorm1d and nn.BatchNorm2d."""
    all_ok = True
    torch.manual_seed(42)

    # 1. Test 1D Equivalence
    x1d = torch.randn(16, 8)
    custom1d = bn1d_cls(num_features=8, momentum=0.1)
    torch1d = nn.BatchNorm1d(num_features=8, momentum=0.1)

    with torch.no_grad():
        torch1d.weight.copy_(custom1d.weight)
        torch1d.bias.copy_(custom1d.bias)

    out_c1 = custom1d(x1d)
    out_t1 = torch1d(x1d)
    diff1d = (out_c1 - out_t1).abs().max().item()

    if diff1d > 1e-4:
        print(f"  [FAIL] BatchNorm1d divergence from PyTorch: max diff = {diff1d:.2e}")
        all_ok = False

    # 2. Test 2D Equivalence
    x2d = torch.randn(8, 4, 12, 12)
    custom2d = bn2d_cls(num_features=4, momentum=0.1)
    torch2d = nn.BatchNorm2d(num_features=4, momentum=0.1)

    with torch.no_grad():
        torch2d.weight.copy_(custom2d.weight)
        torch2d.bias.copy_(custom2d.bias)

    out_c2 = custom2d(x2d)
    out_t2 = torch2d(x2d)
    diff2d = (out_c2 - out_t2).abs().max().item()

    if diff2d > 1e-4:
        print(f"  [FAIL] BatchNorm2d divergence from PyTorch: max diff = {diff2d:.2e}")
        all_ok = False

    if all_ok:
        print(f"  [PASS] Custom BatchNorm is mathematically identical to PyTorch native (diff1d: {diff1d:.2e}, diff2d: {diff2d:.2e})!")
    return all_ok


def run_all_tests():
    """Runs all test gates from the terminal."""
    print("=" * 60)
    print("Running Paper 03 (Batch Normalization) Test Suite")
    print("=" * 60)

    try:
        from model import BatchNorm1d, BatchNorm2d
    except ImportError:
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from model import BatchNorm1d, BatchNorm2d

    try:
        ok1 = test_batchnorm1d_training_stats(BatchNorm1d)
        ok2 = test_batchnorm1d_affine(BatchNorm1d)
        ok3 = test_batchnorm1d_running_stats_and_eval(BatchNorm1d)
        ok4 = test_batchnorm2d_spatial(BatchNorm2d)
        ok5 = test_equivalence_to_pytorch(BatchNorm1d, BatchNorm2d)

        if all([ok1, ok2, ok3, ok4, ok5]):
            print("\n" + "=" * 60)
            print("ALL PAPER 03 TESTS PASSED! YOU INVENTED BATCH NORMALIZATION!")
            print("=" * 60)
    except NotImplementedError:
        print("\n[PENDING] Implement BatchNorm1d and BatchNorm2d in 'papers/03_batch_norm/model.py' to pass tests.")


if __name__ == "__main__":
    run_all_tests()
