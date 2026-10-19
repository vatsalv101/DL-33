"""Paper 07: MobileNetV2: Inverted Residuals and Linear Bottlenecks
Unit Tests Suite.
"""

import torch
import torch.nn as nn


def test_inverted_residual_stride1(inverted_residual_cls):
    """Tests InvertedResidual with stride=1 (residual shortcut active)."""
    all_ok = True
    torch.manual_seed(42)

    blk = inverted_residual_cls(in_channels=24, out_channels=24, stride=1, expand_ratio=6)
    x = torch.randn(2, 24, 28, 28)
    out = blk(x)

    if out.shape != (2, 24, 28, 28):
        print(f"  [FAIL] InvertedResidual stride=1 shape mismatch: expected (2, 24, 28, 28), got {out.shape}")
        all_ok = False
    elif not getattr(blk, "use_res_connect", False):
        print("  [FAIL] InvertedResidual did not activate use_res_connect when in_channels == out_channels and stride == 1!")
        all_ok = False
    else:
        print("  [PASS] InvertedResidual (stride=1, expand=6) preserves shape and activates shortcut connection!")

    return all_ok


def test_inverted_residual_stride2(inverted_residual_cls):
    """Tests InvertedResidual with stride=2 (spatial downsampling, no shortcut)."""
    all_ok = True
    torch.manual_seed(42)

    blk = inverted_residual_cls(in_channels=24, out_channels=32, stride=2, expand_ratio=6)
    x = torch.randn(2, 24, 28, 28)
    out = blk(x)

    if out.shape != (2, 32, 14, 14):
        print(f"  [FAIL] InvertedResidual stride=2 shape mismatch: expected (2, 32, 14, 14), got {out.shape}")
        all_ok = False
    elif getattr(blk, "use_res_connect", False):
        print("  [FAIL] InvertedResidual should NOT activate shortcut connection when stride == 2 or channels change!")
        all_ok = False
    else:
        print("  [PASS] InvertedResidual (stride=2, expand=6) correctly downsamples (28x28 -> 14x14) without shortcut!")

    return all_ok


def test_linear_bottleneck_no_relu(inverted_residual_cls):
    """Verifies that the final projection layer is LINEAR (no non-linear activation)."""
    blk = inverted_residual_cls(in_channels=16, out_channels=24, stride=1, expand_ratio=6)
    last_submodule = list(blk.conv.children())[-1]

    # The last layer must be BatchNorm2d (or Conv2d), NEVER ReLU/ReLU6
    if isinstance(last_submodule, (nn.ReLU, nn.ReLU6)):
        print(f"  [FAIL] Linear Bottleneck violated! Found {type(last_submodule).__name__} as final activation.")
        return False

    print("  [PASS] Linear Bottleneck verified: final projection has no ReLU (preserves low-dim manifold)!")
    return True


def test_mobilenet_v2_shapes(mobilenet_v2_fn):
    """Validates full MobileNetV2 architecture forward pass."""
    all_ok = True
    torch.manual_seed(42)

    # 1. ImageNet standard: 1000 classes
    m1000 = mobilenet_v2_fn(num_classes=1000)
    m1000.eval()
    x = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        out1000 = m1000(x)

    if out1000.shape != (2, 1000):
        print(f"  [FAIL] MobileNetV2 output shape expected (2, 1000), got {out1000.shape}")
        all_ok = False
    else:
        print("  [PASS] MobileNetV2 forward pass output matches (B, 1000)!")

    # 2. Custom robot policy classification / discrete actions: 10 classes
    m10 = mobilenet_v2_fn(num_classes=10)
    m10.eval()
    with torch.no_grad():
        out10 = m10(x)

    if out10.shape != (2, 10):
        print(f"  [FAIL] MobileNetV2 output shape expected (2, 10), got {out10.shape}")
        all_ok = False
    else:
        print("  [PASS] MobileNetV2 forward pass with custom classes matches (B, 10)!")

    return all_ok


def test_single_batch_overfit(mobilenet_v2_fn):
    """Sanity check: confirms MobileNetV2 can rapidly overfit a toy classification batch."""
    torch.manual_seed(42)
    model = mobilenet_v2_fn(num_classes=4)
    model.train()

    from core.optimizers import AdamW

    optimizer = AdamW(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()

    x = torch.randn(4, 3, 32, 32)
    y = torch.tensor([0, 1, 2, 3])

    initial_loss = None
    for step in range(30):
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        if step == 0:
            initial_loss = loss.item()

    final_loss = loss.item()
    if final_loss > 0.1:
        print(f"  [FAIL] Overfit test did not reach loss < 0.1 (init: {initial_loss:.4f}, final: {final_loss:.4f})")
        return False

    print(f"  [PASS] Single-batch overfit gate passed (loss: {initial_loss:.4f} -> {final_loss:.6f})!")
    return True


def run_all_tests():
    """Runs all Paper 07 verification gates."""
    print("=" * 60)
    print("Running Paper 07 (MobileNetV2) Test Suite")
    print("=" * 60)

    try:
        from model import InvertedResidual, MobileNetV2, mobilenet_v2
    except ImportError:
        import sys
        from pathlib import Path

        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from model import InvertedResidual, MobileNetV2, mobilenet_v2

    try:
        ok1 = test_inverted_residual_stride1(InvertedResidual)
        ok2 = test_inverted_residual_stride2(InvertedResidual)
        ok3 = test_linear_bottleneck_no_relu(InvertedResidual)
        ok4 = test_mobilenet_v2_shapes(mobilenet_v2)
        ok5 = test_single_batch_overfit(mobilenet_v2)

        if all([ok1, ok2, ok3, ok4, ok5]):
            print("\n" + "=" * 60)
            print("ALL PAPER 07 TESTS PASSED! YOU INVENTED MOBILENETV2!")
            print("=" * 60)
    except NotImplementedError:
        print("\n[PENDING] Implement InvertedResidual and MobileNetV2 in 'papers/07_mobilenet_v2/model.py' to pass tests.")


if __name__ == "__main__":
    run_all_tests()
