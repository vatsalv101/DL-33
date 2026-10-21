"""Paper 09: PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation
Unit Tests Suite.
"""

import torch
import torch.nn as nn


def test_tnet_shape_and_initialization(tnet_cls):
    """Tests that T-Net outputs (B, K, K) affine matrix and initializes near Identity."""
    all_ok = True
    torch.manual_seed(42)

    tnet = tnet_cls(k=3)
    tnet.eval()
    x = torch.randn(2, 3, 128)
    with torch.no_grad():
        trans = tnet(x)

    if trans.shape != (2, 3, 3):
        print(f"  [FAIL] T-Net output shape mismatch: expected (2, 3, 3), got {trans.shape}")
        all_ok = False

    # Check that initial predicted transformation is close to identity matrix I_3
    eye3 = torch.eye(3).unsqueeze(0).repeat(2, 1, 1)
    diff = (trans - eye3).abs().max().item()
    if diff > 0.1:
        print(f"  [FAIL] T-Net did not initialize near Identity matrix (max diff: {diff:.4f})")
        all_ok = False
    else:
        print(f"  [PASS] T-Net outputs (B, 3, 3) matrix initialized near Identity (diff: {diff:.4f})!")

    return all_ok


def test_permutation_invariance(pointnet_cls):
    """The Fundamental Theorem of PointNet: classification output must be strictly invariant to point ordering."""
    torch.manual_seed(42)

    model = pointnet_cls(num_classes=10, feature_transform=False)
    model.eval()

    # Original point cloud: 1 object with 512 points
    x = torch.randn(1, 3, 512)

    # Random permutation of points along dimension 2
    perm = torch.randperm(512)
    x_perm = x[:, :, perm]

    with torch.no_grad():
        out_orig, _, _ = model(x)
        out_perm, _, _ = model(x_perm)

    diff = (out_orig - out_perm).abs().max().item()
    if diff > 1e-4:
        print(f"  [FAIL] PointNet is NOT permutation invariant! Max logit difference under point shuffling: {diff:.2e}")
        return False

    print(f"  [PASS] Permutation Invariance Proven: shuffling 512 points produced identical logits (diff: {diff:.2e})!")
    return True


def test_classification_and_segmentation_shapes(pointnet_cls, pointnet_seg):
    """Validates tensor contracts for both classification and per-point segmentation."""
    all_ok = True
    torch.manual_seed(42)

    # 1. 3D Classification (ModelNet40: 40 classes)
    m_cls = pointnet_cls(num_classes=40, feature_transform=True)
    m_cls.eval()
    x = torch.randn(2, 3, 256)
    with torch.no_grad():
        logits_cls, trans, trans_feat = m_cls(x)

    if logits_cls.shape != (2, 40):
        print(f"  [FAIL] PointNetCls shape mismatch: expected (2, 40), got {logits_cls.shape}")
        all_ok = False
    else:
        print("  [PASS] PointNetCls forward pass matches (B, num_classes) = (2, 40)!")

    # 2. 3D Part Segmentation (ShapeNet: 16 parts, per point)
    m_seg = pointnet_seg(num_classes=16, feature_transform=True)
    m_seg.eval()
    with torch.no_grad():
        logits_seg, _, _ = m_seg(x)

    if logits_seg.shape != (2, 16, 256):
        print(f"  [FAIL] PointNetSeg shape mismatch: expected (2, 16, 256), got {logits_seg.shape}")
        all_ok = False
    else:
        print("  [PASS] PointNetSeg per-point segmentation matches (B, num_classes, N) = (2, 16, 256)!")

    return all_ok


def test_single_batch_overfit(pointnet_cls):
    """Sanity check: confirms PointNet can rapidly memorize a toy 3D point cloud classification batch."""
    torch.manual_seed(42)
    model = pointnet_cls(num_classes=4, feature_transform=False)
    model.train()

    from core.optimizers import AdamW

    optimizer = AdamW(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()

    x = torch.randn(4, 3, 128)
    y = torch.tensor([0, 1, 2, 3])

    initial_loss = None
    for step in range(35):
        optimizer.zero_grad()
        logits, _, _ = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        if step == 0:
            initial_loss = loss.item()

    final_loss = loss.item()
    if final_loss > 0.05:
        print(f"  [FAIL] Overfit test did not reach loss < 0.05 (init: {initial_loss:.4f}, final: {final_loss:.4f})")
        return False

    print(f"  [PASS] Single-batch 3D point cloud overfit passed (loss: {initial_loss:.4f} -> {final_loss:.6f})!")
    return True


def run_all_tests():
    """Runs all Paper 09 verification gates."""
    print("=" * 60)
    print("Running Paper 09 (PointNet) Test Suite")
    print("=" * 60)

    try:
        from model import TNet, PointNetEncoder, PointNetCls, PointNetSeg
    except ImportError:
        import sys
        from pathlib import Path

        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from model import TNet, PointNetEncoder, PointNetCls, PointNetSeg

    try:
        ok1 = test_tnet_shape_and_initialization(TNet)
        ok2 = test_permutation_invariance(PointNetCls)
        ok3 = test_classification_and_segmentation_shapes(PointNetCls, PointNetSeg)
        ok4 = test_single_batch_overfit(PointNetCls)

        if all([ok1, ok2, ok3, ok4]):
            print("\n" + "=" * 60)
            print("ALL PAPER 09 TESTS PASSED! YOU INVENTED POINTNET!")
            print("=" * 60)
    except NotImplementedError:
        print("\n[PENDING] Implement TNet, PointNetEncoder, PointNetCls, and PointNetSeg in 'papers/09_pointnet/model.py' to pass tests.")


if __name__ == "__main__":
    run_all_tests()
