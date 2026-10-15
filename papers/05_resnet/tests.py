"""Paper 05: Deep Residual Learning for Image Recognition (ResNet)
Unit Tests Suite.
"""

import torch
import torch.nn as nn


def test_basic_block_shapes(basic_block_cls):
    """Tests BasicBlock shapes with stride=1 (identity) and stride=2 (downsample)."""
    all_ok = True
    torch.manual_seed(42)

    # 1. Stride 1: in_planes == planes (identity shortcut)
    blk_s1 = basic_block_cls(in_planes=64, planes=64, stride=1)
    x1 = torch.randn(2, 64, 28, 28)
    out1 = blk_s1(x1)
    if out1.shape != (2, 64, 28, 28):
        print(f"  [FAIL] BasicBlock stride=1 shape mismatch: expected (2, 64, 28, 28), got {out1.shape}")
        all_ok = False

    # 2. Stride 2: in_planes != planes (downsample projection)
    downsample = nn.Sequential(
        nn.Conv2d(64, 128, kernel_size=1, stride=2, bias=False),
        nn.BatchNorm2d(128),
    )
    blk_s2 = basic_block_cls(in_planes=64, planes=128, stride=2, downsample=downsample)
    out2 = blk_s2(x1)
    if out2.shape != (2, 128, 14, 14):
        print(f"  [FAIL] BasicBlock stride=2 shape mismatch: expected (2, 128, 14, 14), got {out2.shape}")
        all_ok = False

    if all_ok:
        print("  [PASS] BasicBlock preserves tensor shapes on stride=1 and downsamples on stride=2!")
    return all_ok


def test_bottleneck_shapes(bottleneck_cls):
    """Tests Bottleneck shapes with 4x expansion."""
    all_ok = True
    torch.manual_seed(42)

    # 1. Stride 1: in_planes=64 -> planes=64 -> out=256
    downsample = nn.Sequential(
        nn.Conv2d(64, 256, kernel_size=1, stride=1, bias=False),
        nn.BatchNorm2d(256),
    )
    btn_s1 = bottleneck_cls(in_planes=64, planes=64, stride=1, downsample=downsample)
    x = torch.randn(2, 64, 56, 56)
    out1 = btn_s1(x)
    if out1.shape != (2, 256, 56, 56):
        print(f"  [FAIL] Bottleneck stride=1 shape mismatch: expected (2, 256, 56, 56), got {out1.shape}")
        all_ok = False

    # 2. Stride 2: in_planes=256 -> planes=128 -> out=512, downsampled
    downsample2 = nn.Sequential(
        nn.Conv2d(256, 512, kernel_size=1, stride=2, bias=False),
        nn.BatchNorm2d(512),
    )
    btn_s2 = bottleneck_cls(in_planes=256, planes=128, stride=2, downsample=downsample2)
    x2 = torch.randn(2, 256, 56, 56)
    out2 = btn_s2(x2)
    if out2.shape != (2, 512, 28, 28):
        print(f"  [FAIL] Bottleneck stride=2 shape mismatch: expected (2, 512, 28, 28), got {out2.shape}")
        all_ok = False

    if all_ok:
        print("  [PASS] Bottleneck correctly applies 1x1->3x3->1x1 compression and 4x channel expansion!")
    return all_ok


def test_resnet_architectures(resnet18_fn, resnet50_fn):
    """Validates full ResNet-18 and ResNet-50 pipeline forward pass."""
    all_ok = True
    torch.manual_seed(42)

    # ResNet-18
    m18 = resnet18_fn(num_classes=1000)
    m18.eval()
    x18 = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        out18 = m18(x18)
    if out18.shape != (2, 1000):
        print(f"  [FAIL] ResNet-18 output shape expected (2, 1000), got {out18.shape}")
        all_ok = False
    else:
        print("  [PASS] ResNet-18 forward pass output shape matches (B, 1000)!")

    # ResNet-50
    m50 = resnet50_fn(num_classes=10)
    m50.eval()
    x50 = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        out50 = m50(x50)
    if out50.shape != (2, 10):
        print(f"  [FAIL] ResNet-50 output shape expected (2, 10), got {out50.shape}")
        all_ok = False
    else:
        print("  [PASS] ResNet-50 forward pass output shape matches (B, 10)!")

    return all_ok


def test_gradient_highway(resnet18_fn):
    """Verifies that gradients flow cleanly to early stem layers without vanishing."""
    torch.manual_seed(42)
    model = resnet18_fn(num_classes=10)
    model.train()

    x = torch.randn(4, 3, 64, 64)
    out = model(x)
    loss = out.sum()
    loss.backward()

    stem_grad = model.conv1.weight.grad
    if stem_grad is None:
        print("  [FAIL] Stem conv1.weight has no gradients!")
        return False

    if torch.isnan(stem_grad).any():
        print("  [FAIL] Stem conv1.weight gradients contain NaNs!")
        return False

    grad_norm = stem_grad.norm().item()
    if grad_norm < 1e-6:
        print(f"  [FAIL] Gradient vanishing detected! Stem grad norm: {grad_norm:.2e}")
        return False

    print(f"  [PASS] Residual Gradient Highway active: stem conv1 received healthy grad norm {grad_norm:.4f}!")
    return True


def test_single_batch_overfit(resnet18_fn):
    """Sanity check: confirms ResNet-18 rapidly overfits a single batch."""
    torch.manual_seed(42)
    model = resnet18_fn(num_classes=4)
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

    print(f"  [PASS] Single-batch overfit gate passed (loss dropped: {initial_loss:.4f} -> {final_loss:.6f})!")
    return True


def run_all_tests():
    """Runs all Paper 05 verification gates."""
    print("=" * 60)
    print("Running Paper 05 (ResNet) Test Suite")
    print("=" * 60)

    try:
        from model import BasicBlock, Bottleneck, resnet18, resnet50
    except ImportError:
        import sys
        from pathlib import Path

        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from model import BasicBlock, Bottleneck, resnet18, resnet50

    try:
        ok1 = test_basic_block_shapes(BasicBlock)
        ok2 = test_bottleneck_shapes(Bottleneck)
        ok3 = test_resnet_architectures(resnet18, resnet50)
        ok4 = test_gradient_highway(resnet18)
        ok5 = test_single_batch_overfit(resnet18)

        if all([ok1, ok2, ok3, ok4, ok5]):
            print("\n" + "=" * 60)
            print("ALL PAPER 05 TESTS PASSED! YOU INVENTED RESNET!")
            print("=" * 60)
    except NotImplementedError:
        print("\n[PENDING] Implement BasicBlock, Bottleneck, and ResNet in 'papers/05_resnet/model.py' to pass tests.")


if __name__ == "__main__":
    run_all_tests()
