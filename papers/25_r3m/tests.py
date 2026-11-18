import sys
import importlib
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


def run_tests():
    module_name = sys.argv[1] if len(sys.argv) > 1 else "papers.25_r3m.model"
    try:
        mod = importlib.import_module(module_name)
    except Exception as e:
        print(f"[FAIL] Could not import {module_name}: {e}")
        sys.exit(1)

    R3MTextEncoder = getattr(mod, "R3MTextEncoder")
    R3M = getattr(mod, "R3M")

    torch.manual_seed(42)

    # ---------------------------------------------------------
    # Test 1: Embedding Shapes & L2 Normalization
    # ---------------------------------------------------------
    print("Running Test 1: R3M Embedding Shapes & Normalization...", end=" ")
    try:
        B, embed_dim = 4, 32
        model = R3M(embed_dim=embed_dim, feat_dim=64, vocab_size=100)

        imgs = torch.randn(B, 3, 16, 16)
        text = torch.randint(0, 100, (B, 10))

        z_img = model.encode_image(imgs)
        z_txt = model.encode_text(text)

        assert z_img.shape == (B, embed_dim), f"Expected {(B, embed_dim)}, got {z_img.shape}"
        assert z_txt.shape == (B, embed_dim), f"Expected {(B, embed_dim)}, got {z_txt.shape}"

        assert torch.allclose(z_img.norm(dim=-1), torch.ones(B), atol=1e-5), "Image embeddings must have unit L2 norm"
        assert torch.allclose(z_txt.norm(dim=-1), torch.ones(B), atol=1e-5), "Text embeddings must have unit L2 norm"
        print("[PASS]")
    except Exception as e:
        print(f"[FAIL]: {e}")
        sys.exit(1)

    # ---------------------------------------------------------
    # Test 2: Sub-Loss Contracts (TCN, Lang, L1 Reg)
    # ---------------------------------------------------------
    print("Running Test 2: Loss Mathematical Contracts...", end=" ")
    try:
        model = R3M(embed_dim=32, feat_dim=64, vocab_size=100, temperature=0.1)
        z = F.normalize(torch.randn(4, 32), dim=-1)

        loss_tcn = model.compute_tcn_loss(z, z)
        assert loss_tcn.ndim == 0 and loss_tcn.item() < 0.2, "Identical frames should yield very low TCN loss"

        loss_lang = model.compute_lang_loss(z, z)
        assert loss_lang.ndim == 0 and loss_lang.item() < 0.2, "Identical features should yield very low Lang loss"

        loss_reg = model.compute_l1_reg(z)
        assert loss_reg.ndim == 0 and loss_reg.item() > 0.0, "L1 norm must be positive scalar"
        print("[PASS]")
    except Exception as e:
        print(f"[FAIL]: {e}")
        sys.exit(1)

    # ---------------------------------------------------------
    # Test 3: Multi-Task Forward Pass & Gradient Audit
    # ---------------------------------------------------------
    print("Running Test 3: Forward Pass & Gradient Audit...", end=" ")
    try:
        B = 4
        model = R3M(embed_dim=32, feat_dim=64, vocab_size=100)

        im_anchor = torch.randn(B, 3, 16, 16)
        im_pos = torch.randn(B, 3, 16, 16)
        text = torch.randint(0, 100, (B, 8))

        total_loss, l_tcn, l_lang, l_reg = model(im_anchor, im_pos, text)
        assert total_loss.ndim == 0 and total_loss.item() > 0.0
        assert l_tcn.ndim == 0 and l_lang.ndim == 0 and l_reg.ndim == 0

        total_loss.backward()
        assert model.visual_proj.weight.grad is not None, "Visual projection grad missing"
        assert model.text.proj.weight.grad is not None, "Text projection grad missing"
        print("[PASS]")
    except Exception as e:
        print(f"[FAIL]: {e}")
        sys.exit(1)

    # ---------------------------------------------------------
    # Test 4: Robotic Representation Multi-Task Overfitting Gate
    # ---------------------------------------------------------
    print("Running Test 4: Multi-Task Overfitting Gate...", end=" ")
    try:
        torch.manual_seed(42)
        B = 4
        model = R3M(embed_dim=32, feat_dim=64, vocab_size=100, temperature=0.1)
        optimizer = optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-4)

        # 4 synthetic video sub-actions:
        # anchor frames and adjacent positive frames with small temporal delta
        base_frames = torch.randn(B, 3, 16, 16)
        im_anchor = base_frames + 0.02 * torch.randn_like(base_frames)
        im_pos = base_frames + 0.02 * torch.randn_like(base_frames)

        # 4 distinct task narrations
        task_narrations = torch.tensor([
            [1, 2, 3, 4],
            [10, 20, 30, 40],
            [50, 60, 70, 80],
            [85, 90, 95, 99],
        ], dtype=torch.long)

        initial_loss = None
        for step in range(60):
            optimizer.zero_grad()
            total_loss, l_tcn, l_lang, l_reg = model(im_anchor, im_pos, task_narrations)
            if initial_loss is None:
                initial_loss = total_loss.item()
            total_loss.backward()
            optimizer.step()

        final_loss = total_loss.item()
        assert final_loss < 0.25 or final_loss < initial_loss * 0.2, (
            f"Overfit failed: initial {initial_loss:.4f} -> final {final_loss:.4f}"
        )

        # Verify retrieval accuracy
        with torch.no_grad():
            z_a = model.encode_image(im_anchor)
            z_p = model.encode_image(im_pos)
            z_t = model.encode_text(task_narrations)

            tcn_acc = (z_a @ z_p.T).argmax(dim=-1).eq(torch.arange(B)).float().mean().item()
            lang_acc = (z_a @ z_t.T).argmax(dim=-1).eq(torch.arange(B)).float().mean().item()

        assert tcn_acc == 1.0 and lang_acc == 1.0, f"Expected 100% retrieval, got TCN={tcn_acc}, Lang={lang_acc}"
        print(f"[PASS] (Loss {initial_loss:.3f} -> {final_loss:.3f}, TCN=100%, Lang=100%)")
    except Exception as e:
        print(f"[FAIL]: {e}")
        sys.exit(1)

    print("\nAll R3M unit tests passed successfully!")


if __name__ == "__main__":
    run_tests()
