import sys
import importlib
import torch
import torch.nn as nn
import torch.optim as optim


def run_tests():
    module_name = sys.argv[1] if len(sys.argv) > 1 else "papers.23_mae.model"
    try:
        mod = importlib.import_module(module_name)
    except Exception as e:
        print(f"[FAIL] Could not import {module_name}: {e}")
        sys.exit(1)

    patchify = getattr(mod, "patchify")
    unpatchify = getattr(mod, "unpatchify")
    random_masking = getattr(mod, "random_masking")
    MaskedAutoencoderViT = getattr(mod, "MaskedAutoencoderViT")

    torch.manual_seed(42)

    # ---------------------------------------------------------
    # Test 1: Patchify and Unpatchify Roundtrip
    # ---------------------------------------------------------
    print("Running Test 1: Patchify & Unpatchify Roundtrip...", end=" ")
    try:
        B, C, H, W = 2, 3, 16, 16
        patch_size = 4
        imgs = torch.randn(B, C, H, W)

        patches = patchify(imgs, patch_size)
        num_patches = (H // patch_size) * (W // patch_size)
        patch_dim = patch_size * patch_size * C
        assert patches.shape == (B, num_patches, patch_dim), f"Expected {(B, num_patches, patch_dim)}, got {patches.shape}"

        reconstructed = unpatchify(patches, patch_size, in_chans=C)
        assert reconstructed.shape == imgs.shape, f"Expected {imgs.shape}, got {reconstructed.shape}"
        assert torch.allclose(imgs, reconstructed, atol=1e-5), "Patchify/Unpatchify roundtrip mismatch!"
        print("[PASS]")
    except Exception as e:
        print(f"[FAIL]: {e}")
        sys.exit(1)

    # ---------------------------------------------------------
    # Test 2: Random Masking Ratio & Unshuffle Restoration
    # ---------------------------------------------------------
    print("Running Test 2: Random Masking & Restoration...", end=" ")
    try:
        B, N, D = 4, 16, 32
        mask_ratio = 0.75
        x = torch.randn(B, N, D)

        x_masked, mask, ids_restore = random_masking(x, mask_ratio)
        expected_kept = int(N * (1 - mask_ratio))  # 4
        assert x_masked.shape == (B, expected_kept, D), f"Expected {(B, expected_kept, D)}, got {x_masked.shape}"
        assert mask.shape == (B, N), f"Expected mask shape {(B, N)}, got {mask.shape}"
        assert abs(mask.mean().item() - 0.75) < 1e-5, f"Expected 75% masked, got {mask.mean().item()}"

        # Test unshuffle restoration of a dummy full sequence
        dummy_full = torch.cat([x_masked, torch.zeros(B, N - expected_kept, D)], dim=1)
        restored = torch.gather(dummy_full, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, D))
        assert restored.shape == (B, N, D), f"Restored shape should be {(B, N, D)}, got {restored.shape}"
        print("[PASS]")
    except Exception as e:
        print(f"[FAIL]: {e}")
        sys.exit(1)

    # ---------------------------------------------------------
    # Test 3: MAE Forward Pass & Gradient Flow
    # ---------------------------------------------------------
    print("Running Test 3: MAE Forward Pass & Gradients...", end=" ")
    try:
        mae = MaskedAutoencoderViT(
            img_size=16,
            patch_size=4,
            in_chans=3,
            embed_dim=64,
            depth=2,
            num_heads=2,
            decoder_embed_dim=32,
            decoder_depth=1,
            decoder_num_heads=2,
        )
        imgs = torch.randn(2, 3, 16, 16)
        loss, pred, mask = mae(imgs, mask_ratio=0.75)

        assert loss.ndim == 0 and loss.item() > 0.0, "Loss must be positive scalar"
        assert pred.shape == (2, 16, 48), f"Expected pred shape (2, 16, 48), got {pred.shape}"
        assert mask.shape == (2, 16), f"Expected mask shape (2, 16), got {mask.shape}"

        loss.backward()
        # Verify gradients in encoder and decoder
        assert mae.patch_embed.proj.weight.grad is not None, "Encoder patch_embed grad missing"
        assert mae.decoder_pred.weight.grad is not None, "Decoder pred head grad missing"
        assert mae.decoder_embed.weight.grad is not None, "Decoder embed grad missing"
        print("[PASS]")
    except Exception as e:
        print(f"[FAIL]: {e}")
        sys.exit(1)

    # ---------------------------------------------------------
    # Test 4: Single-Batch Overfit Gate (Pixel Reconstruction)
    # ---------------------------------------------------------
    print("Running Test 4: Single-Batch Overfitting...", end=" ")
    try:
        torch.manual_seed(42)
        mae = MaskedAutoencoderViT(
            img_size=16,
            patch_size=4,
            in_chans=3,
            embed_dim=64,
            depth=2,
            num_heads=2,
            decoder_embed_dim=32,
            decoder_depth=1,
            decoder_num_heads=2,
            norm_pix_loss=True,
        )
        optimizer = optim.AdamW(mae.parameters(), lr=2e-3, weight_decay=1e-4)
        toy_img = torch.randn(1, 3, 16, 16)

        initial_loss = None
        for step in range(80):
            optimizer.zero_grad()
            # Train on fixed image with 75% random masking
            loss, pred, mask = mae(toy_img, mask_ratio=0.75)
            if initial_loss is None:
                initial_loss = loss.item()
            loss.backward()
            optimizer.step()

        final_loss = loss.item()
        assert final_loss < initial_loss * 0.35 or final_loss < 0.25, (
            f"Overfit failed: initial {initial_loss:.4f} -> final {final_loss:.4f}"
        )
        print(f"[PASS] (Loss {initial_loss:.3f} -> {final_loss:.3f})")
    except Exception as e:
        print(f"[FAIL]: {e}")
        sys.exit(1)

    print("\nAll MAE unit tests passed successfully!")


if __name__ == "__main__":
    run_tests()
