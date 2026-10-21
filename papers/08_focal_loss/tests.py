"""Paper 08: Focal Loss for Dense Object Detection
Unit Tests Suite.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def test_gamma_zero_equals_ce(bfl_cls, fl_cls):
    """When gamma=0 (and unweighted), Focal Loss must match standard Cross Entropy."""
    all_ok = True
    torch.manual_seed(42)

    # 1. Binary Focal Loss with gamma=0, alpha=None
    inputs_b = torch.randn(8, 16)
    targets_b = torch.randint(0, 2, (8, 16)).float()

    bfl = bfl_cls(alpha=None, gamma=0.0, reduction="none")
    bce_expected = F.binary_cross_entropy_with_logits(inputs_b, targets_b, reduction="none")
    bfl_out = bfl(inputs_b, targets_b)

    diff_b = (bfl_out - bce_expected).abs().max().item()
    if diff_b > 1e-5:
        print(f"  [FAIL] BinaryFocalLoss(gamma=0) diverges from BCE: max diff = {diff_b:.2e}")
        all_ok = False
    else:
        print(f"  [PASS] BinaryFocalLoss(gamma=0) matches standard BCE exactly (diff: {diff_b:.2e})!")

    # 2. Multi-class Focal Loss with gamma=0, alpha=None
    inputs_m = torch.randn(8, 10)
    targets_m = torch.randint(0, 10, (8,))

    fl = fl_cls(alpha=None, gamma=0.0, reduction="none")
    ce_expected = F.cross_entropy(inputs_m, targets_m, reduction="none")
    fl_out = fl(inputs_m, targets_m)

    diff_m = (fl_out - ce_expected).abs().max().item()
    if diff_m > 1e-5:
        print(f"  [FAIL] FocalLoss(gamma=0) diverges from CE: max diff = {diff_m:.2e}")
        all_ok = False
    else:
        print(f"  [PASS] Multi-class FocalLoss(gamma=0) matches standard CE exactly (diff: {diff_m:.2e})!")

    return all_ok


def test_easy_example_downweighting(bfl_cls):
    """Validates that easy examples (p_t -> 1) are heavily suppressed by (1 - p_t)^gamma."""
    torch.manual_seed(42)

    # Logit = 5.0 -> sigmoid(5.0) = 0.9933
    # Easy sample: target = 1 (p_t = 0.9933)
    # Hard sample: target = 0 (p_t = 0.0067)
    inputs = torch.tensor([5.0, 5.0])
    targets = torch.tensor([1.0, 0.0])

    ce_loss = bfl_cls(alpha=None, gamma=0.0, reduction="none")(inputs, targets)
    fl_loss = bfl_cls(alpha=None, gamma=2.0, reduction="none")(inputs, targets)

    # Ratio of easy loss to hard loss under CE vs Focal Loss
    ratio_ce = ce_loss[0] / ce_loss[1]
    ratio_fl = fl_loss[0] / fl_loss[1]

    # With gamma=2, (1 - 0.9933)^2 / (1 - 0.0067)^2 = 4.5e-5 / 0.9866 ~ 4.5e-5
    suppression_factor = ratio_ce / ratio_fl

    if suppression_factor < 500:
        print(f"  [FAIL] Easy examples were not sufficiently suppressed! Factor: {suppression_factor:.1f}")
        return False

    print(f"  [PASS] Easy example suppressed by {suppression_factor:.1f}x relative to hard example (gamma=2.0)!")
    return True


def test_multiclass_and_spatial_shapes(fl_cls):
    """Tests multi-class Focal Loss across 1D vectors and 2D spatial segmentation maps."""
    all_ok = True
    torch.manual_seed(42)

    # 1. Classification (B=4, C=5)
    logits_1d = torch.randn(4, 5)
    targets_1d = torch.randint(0, 5, (4,))
    fl = fl_cls(gamma=2.0, reduction="mean")
    loss_mean = fl(logits_1d, targets_1d)

    if loss_mean.dim() != 0:
        print(f"  [FAIL] Multi-class mean reduction did not produce scalar, got dim {loss_mean.dim()}")
        all_ok = False

    # 2. Dense spatial segmentation (B=2, C=4, H=16, W=16)
    logits_2d = torch.randn(2, 4, 16, 16)
    targets_2d = torch.randint(0, 4, (2, 16, 16))
    loss_2d = fl(logits_2d, targets_2d)

    if loss_2d.dim() != 0 or torch.isnan(loss_2d):
        print("  [FAIL] Spatial dense multi-class Focal Loss failed or produced NaN!")
        all_ok = False
    else:
        print("  [PASS] Multi-class Focal Loss supports both 1D vectors and 2D dense spatial maps!")

    return all_ok


def test_numerical_stability(bfl_cls, fl_cls):
    """Validates that extreme logits (+/- 1000) do NOT cause NaNs in loss or gradients."""
    all_ok = True

    extreme_inputs = torch.tensor([-500.0, -100.0, 0.0, 100.0, 500.0], requires_grad=True)
    extreme_targets = torch.tensor([0.0, 1.0, 1.0, 0.0, 1.0])

    bfl = bfl_cls(alpha=0.25, gamma=2.0, reduction="mean")
    loss = bfl(extreme_inputs, extreme_targets)

    if torch.isnan(loss) or torch.isinf(loss):
        print(f"  [FAIL] Extreme logits produced NaN or Inf in BinaryFocalLoss: {loss.item()}")
        all_ok = False

    loss.backward()
    if torch.isnan(extreme_inputs.grad).any() or torch.isinf(extreme_inputs.grad).any():
        print("  [FAIL] BinaryFocalLoss backward gradients contain NaNs or Infs under extreme logits!")
        all_ok = False
    else:
        print("  [PASS] Numerical stability verified: zero NaNs under extreme logit ranges (-500 to +500)!")

    return all_ok


def test_extreme_imbalance_overfit(bfl_cls):
    """Sanity check: confirms model trains cleanly under 99:1 extreme class imbalance."""
    torch.manual_seed(42)

    # 100 anchors: 99 background (0), 1 foreground object (1)
    x = torch.randn(100, 16)
    y = torch.zeros(100, 1)
    y[42] = 1.0  # single positive object

    linear = nn.Linear(16, 1)
    from core.optimizers import AdamW

    optimizer = AdamW(linear.parameters(), lr=0.1)
    focal_criterion = bfl_cls(alpha=0.75, gamma=2.0, reduction="mean")

    initial_loss = None
    for step in range(60):
        optimizer.zero_grad()
        logits = linear(x)
        loss = focal_criterion(logits, y)
        loss.backward()
        optimizer.step()
        if step == 0:
            initial_loss = loss.item()

    final_loss = loss.item()
    pred_prob = torch.sigmoid(linear(x[42:43])).item()

    if pred_prob < 0.7:
        print(f"  [FAIL] Model failed to detect rare foreground object under 99:1 imbalance (pred prob: {pred_prob:.4f})")
        return False

    print(f"  [PASS] 99:1 Extreme Imbalance gate passed: rare target probability reached {pred_prob * 100:.1f}%!")
    return True


def run_all_tests():
    """Runs all Paper 08 verification gates."""
    print("=" * 60)
    print("Running Paper 08 (Focal Loss) Test Suite")
    print("=" * 60)

    try:
        from model import BinaryFocalLoss, FocalLoss
    except ImportError:
        import sys
        from pathlib import Path

        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from model import BinaryFocalLoss, FocalLoss

    try:
        ok1 = test_gamma_zero_equals_ce(BinaryFocalLoss, FocalLoss)
        ok2 = test_easy_example_downweighting(BinaryFocalLoss)
        ok3 = test_multiclass_and_spatial_shapes(FocalLoss)
        ok4 = test_numerical_stability(BinaryFocalLoss, FocalLoss)
        ok5 = test_extreme_imbalance_overfit(BinaryFocalLoss)

        if all([ok1, ok2, ok3, ok4, ok5]):
            print("\n" + "=" * 60)
            print("ALL PAPER 08 TESTS PASSED! YOU INVENTED FOCAL LOSS!")
            print("=" * 60)
    except NotImplementedError:
        print("\n[PENDING] Implement BinaryFocalLoss and FocalLoss in 'papers/08_focal_loss/model.py' to pass tests.")


if __name__ == "__main__":
    run_all_tests()
