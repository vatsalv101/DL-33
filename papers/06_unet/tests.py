"""Paper 06: U-Net: Convolutional Networks for Biomedical Image Segmentation
Unit Tests Suite.
"""

import torch
import torch.nn as nn


def test_double_conv_shape(double_conv_cls):
    """Tests that DoubleConv preserves spatial dimensions."""
    all_ok = True
    torch.manual_seed(42)

    dc = double_conv_cls(in_channels=3, out_channels=16)
    x = torch.randn(2, 3, 64, 64)
    out = dc(x)

    if out.shape != (2, 16, 64, 64):
        print(f"  [FAIL] DoubleConv shape mismatch: expected (2, 16, 64, 64), got {out.shape}")
        all_ok = False
    else:
        print("  [PASS] DoubleConv correctly preserves spatial resolution (64x64)!")

    return all_ok


def test_down_shape(down_cls):
    """Tests that Down block halves spatial resolution via MaxPool."""
    all_ok = True
    torch.manual_seed(42)

    down = down_cls(in_channels=16, out_channels=32)
    x = torch.randn(2, 16, 64, 64)
    out = down(x)

    if out.shape != (2, 32, 32, 32):
        print(f"  [FAIL] Down block shape mismatch: expected (2, 32, 32, 32), got {out.shape}")
        all_ok = False
    else:
        print("  [PASS] Down block halves spatial dimensions (64x64 -> 32x32)!")

    return all_ok


def test_up_concatenation_shape(up_cls):
    """Tests Up block upsampling, skip connection concatenation, and odd padding."""
    all_ok = True
    torch.manual_seed(42)

    # 1. Even spatial dimensions
    up_even = up_cls(in_channels=64, out_channels=32, bilinear=True)
    x1_even = torch.randn(2, 32, 16, 16)
    x2_even = torch.randn(2, 32, 32, 32)
    out_even = up_even(x1_even, x2_even)

    if out_even.shape != (2, 32, 32, 32):
        print(f"  [FAIL] Up block shape mismatch (even): expected (2, 32, 32, 32), got {out_even.shape}")
        all_ok = False

    # 2. Odd spatial dimensions (tests padding logic)
    up_odd = up_cls(in_channels=64, out_channels=32, bilinear=True)
    x1_odd = torch.randn(2, 32, 15, 15)
    x2_odd = torch.randn(2, 32, 31, 31)
    out_odd = up_odd(x1_odd, x2_odd)

    if out_odd.shape != (2, 32, 31, 31):
        print(f"  [FAIL] Up block shape mismatch (odd padding): expected (2, 32, 31, 31), got {out_odd.shape}")
        all_ok = False

    if all_ok:
        print("  [PASS] Up block correctly upsamples, concatenates skip connections, and handles padding!")

    return all_ok


def test_unet_full_resolution_preservation(unet_cls):
    """Verifies that full U-Net preserves input (H, W) identically at output."""
    all_ok = True
    torch.manual_seed(42)

    model = unet_cls(in_channels=3, num_classes=1, bilinear=True)
    model.eval()

    x = torch.randn(2, 3, 128, 128)
    with torch.no_grad():
        out = model(x)

    if out.shape != (2, 1, 128, 128):
        print(f"  [FAIL] U-Net output shape mismatch: expected (2, 1, 128, 128), got {out.shape}")
        all_ok = False
    else:
        print("  [PASS] U-Net preserves exact spatial resolution: (B, 3, 128, 128) -> (B, 1, 128, 128)!")

    # Multi-class output check (e.g. 5 semantic classes)
    model5 = unet_cls(in_channels=3, num_classes=5, bilinear=True)
    model5.eval()
    with torch.no_grad():
        out5 = model5(x)

    if out5.shape != (2, 5, 128, 128):
        print(f"  [FAIL] Multi-class U-Net shape mismatch: expected (2, 5, 128, 128), got {out5.shape}")
        all_ok = False
    else:
        print("  [PASS] Multi-class U-Net output matches (B, 5, 128, 128)!")

    return all_ok


def test_unet_overfit_toy_mask(unet_cls):
    """Sanity check: confirms U-Net can rapidly memorize a dense binary segmentation mask."""
    torch.manual_seed(42)
    model = unet_cls(in_channels=1, num_classes=1, bilinear=True)
    model.train()

    from core.optimizers import AdamW

    optimizer = AdamW(model.parameters(), lr=0.01)
    criterion = nn.BCEWithLogitsLoss()

    # Toy input image and target circular mask
    x = torch.randn(2, 1, 32, 32)
    target = torch.zeros(2, 1, 32, 32)
    target[:, :, 8:24, 8:24] = 1.0  # square target mask

    initial_loss = None
    for step in range(30):
        optimizer.zero_grad()
        pred = model(x)
        loss = criterion(pred, target)
        loss.backward()
        optimizer.step()
        if step == 0:
            initial_loss = loss.item()

    final_loss = loss.item()
    if final_loss > 0.05:
        print(f"  [FAIL] Overfit test did not reach loss < 0.05 (init: {initial_loss:.4f}, final: {final_loss:.4f})")
        return False

    print(f"  [PASS] Single-batch segmentation overfit passed (loss: {initial_loss:.4f} -> {final_loss:.6f})!")
    return True


def run_all_tests():
    """Runs all Paper 06 verification gates."""
    print("=" * 60)
    print("Running Paper 06 (U-Net) Test Suite")
    print("=" * 60)

    try:
        from model import DoubleConv, Down, Up, UNet
    except ImportError:
        import sys
        from pathlib import Path

        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from model import DoubleConv, Down, Up, UNet

    try:
        ok1 = test_double_conv_shape(DoubleConv)
        ok2 = test_down_shape(Down)
        ok3 = test_up_concatenation_shape(Up)
        ok4 = test_unet_full_resolution_preservation(UNet)
        ok5 = test_unet_overfit_toy_mask(UNet)

        if all([ok1, ok2, ok3, ok4, ok5]):
            print("\n" + "=" * 60)
            print("ALL PAPER 06 TESTS PASSED! YOU INVENTED U-NET!")
            print("=" * 60)
    except NotImplementedError:
        print("\n[PENDING] Implement Down, Up, and UNet in 'papers/06_unet/model.py' to pass tests.")


if __name__ == "__main__":
    run_all_tests()
