import sys
import importlib
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


def run_tests():
    module_name = sys.argv[1] if len(sys.argv) > 1 else "papers.24_clip.model"
    try:
        mod = importlib.import_module(module_name)
    except Exception as e:
        print(f"[FAIL] Could not import {module_name}: {e}")
        sys.exit(1)

    VisionBackbone = getattr(mod, "VisionBackbone")
    TextTransformerEncoder = getattr(mod, "TextTransformerEncoder")
    CLIP = getattr(mod, "CLIP")

    torch.manual_seed(42)

    # ---------------------------------------------------------
    # Test 1: Multimodal Embedding Shapes & L2 Normalization
    # ---------------------------------------------------------
    print("Running Test 1: Multimodal Embedding Shapes & Normalization...", end=" ")
    try:
        B, embed_dim = 4, 32
        clip = CLIP(embed_dim=embed_dim, image_dim=32, text_dim=32, vocab_size=100)

        imgs = torch.randn(B, 3, 16, 16)
        text = torch.randint(0, 100, (B, 10))

        img_embeds = clip.encode_image(imgs)
        txt_embeds = clip.encode_text(text)

        assert img_embeds.shape == (B, embed_dim), f"Expected {(B, embed_dim)}, got {img_embeds.shape}"
        assert txt_embeds.shape == (B, embed_dim), f"Expected {(B, embed_dim)}, got {txt_embeds.shape}"

        # Verify unit L2 norms
        img_norms = img_embeds.norm(dim=-1)
        txt_norms = txt_embeds.norm(dim=-1)
        assert torch.allclose(img_norms, torch.ones_like(img_norms), atol=1e-5), "Image embeddings must have unit L2 norm"
        assert torch.allclose(txt_norms, torch.ones_like(txt_norms), atol=1e-5), "Text embeddings must have unit L2 norm"
        print("[PASS]")
    except Exception as e:
        print(f"[FAIL]: {e}")
        sys.exit(1)

    # ---------------------------------------------------------
    # Test 2: Symmetric Forward Pass & Gradient Audit
    # ---------------------------------------------------------
    print("Running Test 2: Symmetric Forward & Gradient Flow...", end=" ")
    try:
        B = 4
        clip = CLIP(embed_dim=32, image_dim=32, text_dim=32, vocab_size=100)
        imgs = torch.randn(B, 3, 16, 16)
        text = torch.randint(0, 100, (B, 8))

        loss, logits_per_image, logits_per_text = clip(imgs, text)

        assert logits_per_image.shape == (B, B), f"Expected {(B, B)}, got {logits_per_image.shape}"
        assert logits_per_text.shape == (B, B), f"Expected {(B, B)}, got {logits_per_text.shape}"
        assert torch.allclose(logits_per_image, logits_per_text.T), "Logits must be transposed symmetric"

        loss.backward()
        assert clip.visual_projection.weight.grad is not None, "Visual projection grad missing"
        assert clip.text_projection.weight.grad is not None, "Text projection grad missing"
        assert clip.logit_scale.grad is not None, "Logit scale temperature grad missing"
        print("[PASS]")
    except Exception as e:
        print(f"[FAIL]: {e}")
        sys.exit(1)

    # ---------------------------------------------------------
    # Test 3: Zero-Shot Classification Mechanics
    # ---------------------------------------------------------
    print("Running Test 3: Zero-Shot Classification...", end=" ")
    try:
        clip = CLIP(embed_dim=32, image_dim=32, text_dim=32, vocab_size=100)
        clip.eval()

        # Synthetic prompt tokens for 2 classes
        prompt_class0 = torch.tensor([[10, 20, 30]])  # "photo of red"
        prompt_class1 = torch.tensor([[40, 50, 60]])  # "photo of blue"
        prompts = torch.cat([prompt_class0, prompt_class1], dim=0)

        with torch.no_grad():
            text_weights = clip.encode_text(prompts)  # (2, embed_dim)
            img = torch.randn(1, 3, 16, 16)
            img_embed = clip.encode_image(img)  # (1, embed_dim)
            logits = img_embed @ text_weights.T  # (1, 2)

        assert logits.shape == (1, 2), f"Expected logits shape (1, 2), got {logits.shape}"
        print("[PASS]")
    except Exception as e:
        print(f"[FAIL]: {e}")
        sys.exit(1)

    # ---------------------------------------------------------
    # Test 4: Multimodal Cross-Modal Overfit Gate
    # ---------------------------------------------------------
    print("Running Test 4: Multimodal Overfitting Gate...", end=" ")
    try:
        torch.manual_seed(42)
        B = 4
        clip = CLIP(embed_dim=32, image_dim=32, text_dim=32, vocab_size=100)
        optimizer = optim.AdamW(clip.parameters(), lr=3e-3, weight_decay=1e-4)

        # 4 distinct synthetic image-text pairs
        toy_imgs = torch.randn(B, 3, 16, 16)
        toy_texts = torch.tensor([
            [1, 2, 3, 4],
            [10, 20, 30, 40],
            [50, 60, 70, 80],
            [85, 90, 95, 99],
        ], dtype=torch.long)

        initial_loss = None
        for step in range(60):
            optimizer.zero_grad()
            loss, l_i, l_t = clip(toy_imgs, toy_texts)
            if initial_loss is None:
                initial_loss = loss.item()
            loss.backward()
            optimizer.step()

        final_loss = loss.item()
        # Verify cross-modal alignment accuracy
        targets = torch.arange(B)
        i2t_acc = (l_i.argmax(dim=-1) == targets).float().mean().item()
        t2i_acc = (l_t.argmax(dim=-1) == targets).float().mean().item()

        assert final_loss < 0.20 or final_loss < initial_loss * 0.2, (
            f"Overfit failed: initial {initial_loss:.4f} -> final {final_loss:.4f}"
        )
        assert i2t_acc == 1.0 and t2i_acc == 1.0, f"Expected 100% retrieval accuracy, got {i2t_acc}, {t2i_acc}"
        print(f"[PASS] (Loss {initial_loss:.3f} -> {final_loss:.3f}, Acc 100%)")
    except Exception as e:
        print(f"[FAIL]: {e}")
        sys.exit(1)

    print("\nAll CLIP unit tests passed successfully!")


if __name__ == "__main__":
    run_tests()
