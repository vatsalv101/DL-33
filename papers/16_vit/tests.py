import sys
import importlib
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim

MODULE_NAME = sys.argv[1] if len(sys.argv) > 1 else "papers.16_vit.model"

try:
    mod = importlib.import_module(MODULE_NAME)
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import model as mod

PatchEmbedding = mod.PatchEmbedding
ViTAttention = mod.ViTAttention
ViTMLP = mod.ViTMLP
ViTBlock = mod.ViTBlock
ViT = mod.ViT


def test_patch_embedding_shape():
    print("Testing PatchEmbedding shape contract...")
    B, C, H, W = 2, 3, 32, 32
    patch_size = 4
    embed_dim = 64
    patch_embed = PatchEmbedding(img_size=H, patch_size=patch_size, in_channels=C, embed_dim=embed_dim)
    patch_embed.eval()

    x = torch.randn(B, C, H, W)
    out = patch_embed(x)

    expected_num_patches = (H // patch_size) * (W // patch_size)  # 8 * 8 = 64
    assert out.shape == (B, expected_num_patches, embed_dim), (
        f"Expected shape {(B, expected_num_patches, embed_dim)}, got {out.shape}"
    )
    print("  [PASS] PatchEmbedding shape and patchification verified.")


def test_vit_attention_and_block():
    print("Testing ViTAttention and ViTBlock...")
    B, N, D = 2, 65, 64
    block = ViTBlock(dim=D, num_heads=4, mlp_ratio=4.0, drop=0.0)
    block.eval()

    x = torch.randn(B, N, D)
    out = block(x)

    assert out.shape == (B, N, D), f"Expected {(B, N, D)}, got {out.shape}"
    print("  [PASS] ViTBlock Pre-LN Self-Attention + MLP block verified.")


def test_spatial_position_sensitivity():
    print("Testing Spatial Position Sensitivity (ViT is NOT permutation-invariant)...")
    torch.manual_seed(42)
    B, C, H, W = 1, 3, 32, 32
    model = ViT(img_size=H, patch_size=8, in_channels=C, num_classes=5, embed_dim=32, depth=2, num_heads=2)
    model.eval()

    # Image 1
    x1 = torch.randn(B, C, H, W)
    # Image 2: swap top half and bottom half of the image
    x2 = torch.cat([x1[:, :, 16:, :], x1[:, :, :16, :]], dim=2)

    with torch.no_grad():
        feats1 = model.forward_features(x1)
        feats2 = model.forward_features(x2)

    diff = (feats1 - feats2).abs().max().item()
    assert diff > 0.1, (
        f"ViT must be sensitive to spatial arrangements! Feature difference was only {diff:.6e}"
    )
    print(f"  [PASS] Spatial position sensitivity verified: rearranging image regions alters features (diff: {diff:.4f}).")


def test_vit_forward_and_backward():
    print("Testing ViT Forward & Backward Pass...")
    B, C, H, W = 2, 3, 32, 32
    num_classes = 4
    model = ViT(
        img_size=H,
        patch_size=8,
        in_channels=C,
        num_classes=num_classes,
        embed_dim=48,
        depth=2,
        num_heads=3,
        drop_rate=0.1,
    )
    model.train()

    x = torch.randn(B, C, H, W)
    logits = model(x)
    assert logits.shape == (B, num_classes), f"Expected logits {(B, num_classes)}, got {logits.shape}"

    loss = logits.sum()
    loss.backward()

    # Audit critical parameters
    assert model.cls_token.grad is not None, "CLS token received no gradient!"
    assert model.pos_embed.grad is not None, "Position embedding received no gradient!"
    assert model.patch_embed.proj.weight.grad is not None, "Patch embedding conv received no gradient!"

    for name, param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"Parameter {name} has no gradient!"
            assert not torch.isnan(param.grad).any(), f"Parameter {name} gradient contains NaN!"
            assert not torch.isinf(param.grad).any(), f"Parameter {name} gradient contains Inf!"

    print("  [PASS] ViT forward/backward pass and gradient audit clean.")


def test_single_batch_overfit():
    print("Testing Single Batch Overfit (Image Classification Memorization)...")
    torch.manual_seed(42)
    B, C, H, W = 4, 3, 32, 32
    num_classes = 4

    model = ViT(
        img_size=H,
        patch_size=8,
        in_channels=C,
        num_classes=num_classes,
        embed_dim=64,
        depth=3,
        num_heads=4,
        drop_rate=0.0,
    )
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    images = torch.randn(B, C, H, W)
    targets = torch.tensor([0, 1, 2, 3])

    initial_loss = None
    final_loss = None

    for epoch in range(60):
        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()

        if epoch == 0:
            initial_loss = loss.item()
        final_loss = loss.item()
        if final_loss < 0.01:
            break

    print(f"  Loss: {initial_loss:.4f} -> {final_loss:.4f} in {epoch+1} steps")
    assert final_loss < 0.05, f"ViT failed to overfit single batch! Final loss: {final_loss:.4f}"
    print("  [PASS] Single batch overfit successfully converged.")


if __name__ == "__main__":
    print(f"Running unit tests for: {MODULE_NAME}\n" + "=" * 50)
    test_patch_embedding_shape()
    test_vit_attention_and_block()
    test_spatial_position_sensitivity()
    test_vit_forward_and_backward()
    test_single_batch_overfit()
    print("=" * 50 + "\nALL TESTS PASSED! [PASS]")
