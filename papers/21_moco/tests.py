import sys
import importlib
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


def run_tests():
    module_name = sys.argv[1] if len(sys.argv) > 1 else "papers.21_moco.model"
    try:
        mod = importlib.import_module(module_name)
    except Exception as e:
        print(f"[FAIL] Could not import {module_name}: {e}")
        sys.exit(1)

    MoCoBackbone = getattr(mod, "MoCoBackbone")
    MoCo = getattr(mod, "MoCo")

    torch.manual_seed(42)

    # ---------------------------------------------------------
    # Test 1: Momentum Update Mechanics
    # ---------------------------------------------------------
    print("Running Test 1: Momentum Update Mechanics...", end=" ")
    try:
        m = 0.9
        model = MoCo(dim=32, K=64, m=m)
        # Verify initial weight equality
        q_weight = next(model.encoder_q.parameters())
        k_weight = next(model.encoder_k.parameters())
        assert torch.allclose(q_weight, k_weight), "Initial key encoder must match query encoder"

        # Clone initial key weight before update
        initial_k = k_weight.clone()

        # Perturb query encoder
        with torch.no_grad():
            q_weight.add_(10.0)

        # Apply momentum update
        model._momentum_update_key_encoder()
        expected_k = initial_k * m + q_weight * (1.0 - m)
        assert torch.allclose(k_weight, expected_k, atol=1e-5), (
            f"Momentum update failed! Expected {expected_k[0, 0]}, got {k_weight[0, 0]}"
        )
        print("[PASS]")
    except Exception as e:
        print(f"[FAIL]: {e}")
        sys.exit(1)

    # ---------------------------------------------------------
    # Test 2: Dynamic Queue FIFO & Wrap-Around
    # ---------------------------------------------------------
    print("Running Test 2: Dynamic Queue FIFO & Wrap-Around...", end=" ")
    try:
        dim, K = 8, 16
        model = MoCo(dim=dim, K=K)
        assert model.queue.shape == (dim, K), f"Expected queue shape {(dim, K)}, got {model.queue.shape}"

        # Insert 8 keys
        k1 = torch.ones(8, dim)
        model._dequeue_and_enqueue(k1)
        assert model.queue_ptr[0].item() == 8, f"Expected ptr=8, got {model.queue_ptr[0].item()}"
        assert torch.allclose(model.queue[:, :8], k1.T), "First 8 slots should match k1"

        # Insert 8 more keys (queue is now full)
        k2 = torch.full((8, dim), 2.0)
        model._dequeue_and_enqueue(k2)
        assert model.queue_ptr[0].item() == 0, f"Expected ptr wrap-around to 0, got {model.queue_ptr[0].item()}"
        assert torch.allclose(model.queue[:, 8:], k2.T), "Last 8 slots should match k2"

        # Insert 4 more keys (overwriting oldest entries 0..3)
        k3 = torch.full((4, dim), 3.0)
        model._dequeue_and_enqueue(k3)
        assert model.queue_ptr[0].item() == 4, f"Expected ptr=4, got {model.queue_ptr[0].item()}"
        assert torch.allclose(model.queue[:, :4], k3.T), "Oldest entries 0..3 should be overwritten with k3"
        print("[PASS]")
    except Exception as e:
        print(f"[FAIL]: {e}")
        sys.exit(1)

    # ---------------------------------------------------------
    # Test 3: MoCo Forward Pass & Detached Target Audit
    # ---------------------------------------------------------
    print("Running Test 3: Forward Pass & Detached Target Audit...", end=" ")
    try:
        dim, K = 32, 64
        model = MoCo(dim=dim, K=K, T=0.1)
        im_q = torch.randn(4, 3, 16, 16)
        im_k = torch.randn(4, 3, 16, 16)

        logits, labels = model(im_q, im_k)
        assert logits.shape == (4, 1 + K), f"Expected logits shape {(4, 1 + K)}, got {logits.shape}"
        assert labels.shape == (4,), f"Expected labels shape (4,), got {labels.shape}"
        assert (labels == 0).all(), "All positive matches must be at index 0"

        # Backpropagation audit
        loss = F.cross_entropy(logits, labels)
        loss.backward()

        # Query encoder must receive gradients
        q_has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.encoder_q.parameters())
        assert q_has_grad, "Query encoder must receive gradients"

        # Key encoder MUST NOT receive gradients
        for p in model.encoder_k.parameters():
            assert p.grad is None, "Detached target violation! Key encoder must not receive backprop gradients"
        print("[PASS]")
    except Exception as e:
        print(f"[FAIL]: {e}")
        sys.exit(1)

    # ---------------------------------------------------------
    # Test 4: Contrastive Convergence (Overfit Gate)
    # ---------------------------------------------------------
    print("Running Test 4: Contrastive Convergence...", end=" ")
    try:
        torch.manual_seed(42)
        N = 8
        dim = 32
        K = 32

        base_images = torch.randn(N, 3, 16, 16)
        model = MoCo(dim=dim, K=K, m=0.99, T=0.1)
        optimizer = optim.AdamW(model.encoder_q.parameters(), lr=3e-3)

        # Warm up queue with image representations
        with torch.no_grad():
            for _ in range(K // N + 1):
                warm_q = base_images + 0.05 * torch.randn_like(base_images)
                warm_k = base_images + 0.05 * torch.randn_like(base_images)
                model(warm_q, warm_k)

        initial_loss = None
        for step in range(80):
            optimizer.zero_grad()
            # Stochastic views
            im_q = base_images + 0.05 * torch.randn_like(base_images)
            im_k = base_images + 0.05 * torch.randn_like(base_images)

            logits, labels = model(im_q, im_k)
            loss = F.cross_entropy(logits, labels)
            if initial_loss is None:
                initial_loss = loss.item()
            loss.backward()
            optimizer.step()

        final_loss = loss.item()
        acc = (logits.argmax(dim=-1) == labels).float().mean().item()

        assert final_loss < 0.6 or final_loss < initial_loss * 0.5, (
            f"Overfit failed: initial {initial_loss:.4f} -> final {final_loss:.4f}"
        )
        assert acc >= 0.85, f"Expected top-1 accuracy >= 85%, got {acc * 100:.1f}%"
        print(f"[PASS] (Loss {initial_loss:.3f} -> {final_loss:.3f}, Acc {acc * 100:.1f}%)")
    except Exception as e:
        print(f"[FAIL]: {e}")
        sys.exit(1)

    print("\nAll MoCo unit tests passed successfully!")


if __name__ == "__main__":
    run_tests()
