"""Paper 01: Delving Deep into Rectifiers (Kaiming Initialization)
Unit Tests Suite (matching Deep Learning course exercise format).
"""

import math
import torch
import torch.nn as nn


def test_calculate_gain(gain_fn):
    """Tests calculate_gain across standard activations."""
    all_ok = True

    cases = [
        ("linear", None, 1.0),
        ("conv2d", None, 1.0),
        ("sigmoid", None, 1.0),
        ("tanh", None, 5.0 / 3.0),
        ("relu", None, math.sqrt(2.0)),
        ("leaky_relu", 0.0, math.sqrt(2.0)),
        ("leaky_relu", 0.2, math.sqrt(2.0 / (1.0 + 0.2**2))),
        ("leaky_relu", 0.5, math.sqrt(2.0 / (1.0 + 0.5**2))),
    ]

    for nonlin, param, expected in cases:
        try:
            val = gain_fn(nonlin, param)
            if not math.isclose(val, expected, rel_tol=1e-4):
                print(f"  [FAIL] gain('{nonlin}', {param}) = {val:.5f}, expected {expected:.5f}")
                all_ok = False
        except Exception as e:
            print(f"  [FAIL] gain('{nonlin}', {param}) raised exception: {e}")
            all_ok = False

    if all_ok:
        print("  [PASS] calculate_gain is correct across all activations!")
    return all_ok


def test_calculate_fan(fan_fn):
    """Tests _calculate_fan for Linear (2D) and Conv2d (4D) weights."""
    all_ok = True

    # 1. Linear: weight shape (out_features=128, in_features=64)
    linear_w = torch.empty(128, 64)
    fan_in = fan_fn(linear_w, mode="fan_in")
    fan_out = fan_fn(linear_w, mode="fan_out")
    if fan_in != 64 or fan_out != 128:
        print(f"  [FAIL] Linear(64, 128): fan_in={fan_in} (expected 64), fan_out={fan_out} (expected 128)")
        all_ok = False

    # 2. Conv2d: weight shape (C_out=32, C_in=16, k_h=3, k_w=3)
    conv_w = torch.empty(32, 16, 3, 3)
    c_fan_in = fan_fn(conv_w, mode="fan_in")
    c_fan_out = fan_fn(conv_w, mode="fan_out")
    # expected: fan_in = 16 * 3 * 3 = 144, fan_out = 32 * 3 * 3 = 288
    if c_fan_in != 144 or c_fan_out != 288:
        print(f"  [FAIL] Conv2d(16->32, 3x3): fan_in={c_fan_in} (expected 144), fan_out={c_fan_out} (expected 288)")
        all_ok = False

    if all_ok:
        print("  [PASS] _calculate_fan correctly handles Linear (2D) and Conv2d (4D) tensors!")
    return all_ok


def test_kaiming_uniform(uniform_fn):
    """Tests kaiming_uniform_ for variance and distribution bounds."""
    all_ok = True
    torch.manual_seed(42)

    # Large weight matrix for statistical significance
    w = torch.empty(1000, 1000)
    uniform_fn(w, a=0.0, mode="fan_in", nonlinearity="relu")

    # For fan_in=1000, gain=sqrt(2): std = sqrt(2/1000) = 0.04472
    # bound = sqrt(3) * std = 0.07746
    expected_std = math.sqrt(2.0 / 1000.0)
    expected_bound = math.sqrt(3.0) * expected_std

    actual_std = w.std().item()
    actual_max = w.abs().max().item()

    if not math.isclose(actual_std, expected_std, rel_tol=0.05):
        print(f"  [FAIL] kaiming_uniform_ std = {actual_std:.5f}, expected ~{expected_std:.5f}")
        all_ok = False

    if actual_max > expected_bound * 1.01:
        print(f"  [FAIL] kaiming_uniform_ max value {actual_max:.5f} exceeds bound {expected_bound:.5f}")
        all_ok = False

    if all_ok:
        print(f"  [PASS] kaiming_uniform_ variance (~{actual_std:.4f}) and bounds match theory!")
    return all_ok


def test_kaiming_normal(normal_fn):
    """Tests kaiming_normal_ for zero mean and target variance."""
    all_ok = True
    torch.manual_seed(42)

    w = torch.empty(1000, 1000)
    normal_fn(w, a=0.0, mode="fan_in", nonlinearity="relu")

    expected_std = math.sqrt(2.0 / 1000.0)
    actual_mean = w.mean().item()
    actual_std = w.std().item()

    if abs(actual_mean) > 0.005:
        print(f"  [FAIL] kaiming_normal_ mean {actual_mean:.5f} is not close to 0.0")
        all_ok = False

    if not math.isclose(actual_std, expected_std, rel_tol=0.05):
        print(f"  [FAIL] kaiming_normal_ std = {actual_std:.5f}, expected ~{expected_std:.5f}")
        all_ok = False

    if all_ok:
        print(f"  [PASS] kaiming_normal_ mean ({actual_mean:.4f}) and std ({actual_std:.4f}) match theory!")
    return all_ok


def test_init_weights(init_fn):
    """Tests init_weights helper on a multi-layer model with Linear, Conv2d, and biases."""
    all_ok = True
    torch.manual_seed(42)

    class DummyNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(3, 16, 3, bias=True)
            self.fc = nn.Linear(64, 10, bias=True)

    net = DummyNet()
    # Fill biases with non-zero to verify they get zeroed
    nn.init.constant_(net.conv.bias, 5.0)
    nn.init.constant_(net.fc.bias, -3.0)

    init_fn(net, nonlinearity="relu", distribution="normal")

    if not torch.allclose(net.conv.bias, torch.zeros_like(net.conv.bias)):
        print("  [FAIL] init_weights did not zero out Conv2d bias!")
        all_ok = False
    if not torch.allclose(net.fc.bias, torch.zeros_like(net.fc.bias)):
        print("  [FAIL] init_weights did not zero out Linear bias!")
        all_ok = False

    if all_ok:
        print("  [PASS] init_weights correctly applies initialization and zeroes biases!")
    return all_ok


def test_30_layer_variance(init_fn, mlp_cls):
    """Stress-tests that Kaiming init prevents activation collapse in a 30-layer deep network."""
    all_ok = True
    torch.manual_seed(42)
    x = torch.randn(64, 64)

    # 1. Default (uninitialized / Xavier-like default)
    net_default = mlp_cls(in_dim=64, hidden_dim=64, num_layers=30)
    _, stats_default = net_default.forward_with_layer_stats(x)
    final_std_default = stats_default[-1][1]

    # 2. Your Kaiming init
    net_kaiming = mlp_cls(in_dim=64, hidden_dim=64, num_layers=30)
    init_fn(net_kaiming, nonlinearity="relu", distribution="normal")
    _, stats_kaiming = net_kaiming.forward_with_layer_stats(x)
    final_std_kaiming = stats_kaiming[-1][1]

    print("\n  Activation Standard Deviation Across Depth:")
    print(f"    Layer  1 std -> Default: {stats_default[0][1]:.4f}  | Your Kaiming: {stats_kaiming[0][1]:.4f}")
    print(f"    Layer 15 std -> Default: {stats_default[14][1]:.6f}  | Your Kaiming: {stats_kaiming[14][1]:.4f}")
    print(f"    Layer 30 std -> Default: {stats_default[29][1]:.8f}  | Your Kaiming: {stats_kaiming[29][1]:.4f}")

    if final_std_kaiming < 0.05:
        print(f"  [FAIL] Signal collapsed at layer 30 (std = {final_std_kaiming:.6f})!")
        all_ok = False
    else:
        print(f"\n  [PASS] Signal preserved! Layer 30 std = {final_std_kaiming:.4f} vs Default: {final_std_default:.8f}")

    return all_ok, stats_default, stats_kaiming


def run_all_tests():
    """Runs all test gates from the terminal."""
    print("=" * 60)
    print("Running Paper 01 (Kaiming Init) Test Suite")
    print("=" * 60)

    try:
        from model import calculate_gain, _calculate_fan, kaiming_uniform_, kaiming_normal_, init_weights, DeepMLP
    except ImportError:
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from model import calculate_gain, _calculate_fan, kaiming_uniform_, kaiming_normal_, init_weights, DeepMLP

    try:
        ok1 = test_calculate_gain(calculate_gain)
        ok2 = test_calculate_fan(_calculate_fan)
        ok3 = test_kaiming_uniform(kaiming_uniform_)
        ok4 = test_kaiming_normal(kaiming_normal_)
        ok5 = test_init_weights(init_weights)
        ok6, _, _ = test_30_layer_variance(init_weights, DeepMLP)

        if all([ok1, ok2, ok3, ok4, ok5, ok6]):
            print("\n" + "=" * 60)
            print("ALL PAPER 01 TESTS PASSED! YOU INVENTED KAIMING INIT!")
            print("=" * 60)
    except NotImplementedError as e:
        print(f"\n[PENDING] {e}")
        print("Implement the missing functions in 'papers/01_kaiming_init/model.py' to pass tests.")


if __name__ == "__main__":
    run_all_tests()
