import sys
import importlib
import torch
import torch.nn as nn
import torch.optim as optim


def run_tests():
    module_name = sys.argv[1] if len(sys.argv) > 1 else "papers.20_simclr.model"
    try:
        mod = importlib.import_module(module_name)
    except Exception as e:
        print(f"[FAIL] Could not import {module_name}: {e}")
        sys.exit(1)

    nt_xent_loss = getattr(mod, "nt_xent_loss")
    SimCLRProjectionHead = getattr(mod, "SimCLRProjectionHead")
    SmallResNetBackbone = getattr(mod, "SmallResNetBackbone")
    SimCLR = getattr(mod, "SimCLR")
    LinearProbe = getattr(mod, "LinearProbe")

    torch.manual_seed(42)

    # ---------------------------------------------------------
    # Test 1: NT-Xent Loss Mathematical Contracts
    # ---------------------------------------------------------
    print("Running Test 1: NT-Xent Loss Mathematical Contracts...", end=" ")
    try:
        N, D = 16, 32
        # Case A: Identical positive pairs with sharp temperature
        z1 = torch.randn(N, D)
        z2 = z1.clone()
        loss_perfect, acc_perfect = nt_xent_loss(z1, z2, temperature=0.05)
        assert abs(acc_perfect.item() - 1.0) < 1e-5, f"Expected perfect acc 1.0, got {acc_perfect.item()}"
        assert loss_perfect.item() < 0.01, f"Loss should be near zero for identical pairs with low tau, got {loss_perfect.item()}"

        loss_soft, acc_soft = nt_xent_loss(z1, z2, temperature=0.5)
        assert abs(acc_soft.item() - 1.0) < 1e-5, f"Expected perfect acc 1.0, got {acc_soft.item()}"
        assert loss_soft.item() < 2.0, f"Loss should be bounded, got {loss_soft.item()}"

        # Case B: Symmetry test: nt_xent_loss(z1, z2) == nt_xent_loss(z2, z1)
        z_a = torch.randn(N, D)
        z_b = torch.randn(N, D)
        loss_ab, acc_ab = nt_xent_loss(z_a, z_b, temperature=0.5)
        loss_ba, acc_ba = nt_xent_loss(z_b, z_a, temperature=0.5)
        assert abs(loss_ab.item() - loss_ba.item()) < 1e-5, f"Loss should be symmetric: {loss_ab} vs {loss_ba}"
        print("[PASS]")
    except Exception as e:
        print(f"[FAIL]: {e}")
        sys.exit(1)

    # ---------------------------------------------------------
    # Test 2: SimCLR Projection Head
    # ---------------------------------------------------------
    print("Running Test 2: SimCLR Projection Head...", end=" ")
    try:
        head = SimCLRProjectionHead(in_dim=64, hidden_dim=64, out_dim=32)
        h = torch.randn(8, 64, requires_grad=True)
        z = head(h)
        assert z.shape == (8, 32), f"Expected shape (8, 32), got {z.shape}"

        loss = z.sum()
        loss.backward()
        assert head.net[0].weight.grad is not None, "Linear1 grad should not be None"
        assert head.net[3].weight.grad is not None, "Linear2 grad should not be None"
        print("[PASS]")
    except Exception as e:
        print(f"[FAIL]: {e}")
        sys.exit(1)

    # ---------------------------------------------------------
    # Test 3: SimCLR Forward Pass & Gradient Audit
    # ---------------------------------------------------------
    print("Running Test 3: SimCLR Forward Pass & Gradients...", end=" ")
    try:
        model = SimCLR(feat_dim=64, proj_dim=32, temperature=0.5)
        x1 = torch.randn(4, 3, 32, 32)
        x2 = torch.randn(4, 3, 32, 32)

        loss, acc, z1, z2, h1, h2 = model(x1, x2)
        assert h1.shape == (4, 64) and h2.shape == (4, 64)
        assert z1.shape == (4, 32) and z2.shape == (4, 32)
        assert loss.ndim == 0 and loss.item() > 0.0

        loss.backward()
        assert model.projector.net[0].weight.grad is not None, "Projector grad should not be None"
        assert model.backbone.conv1.weight.grad is not None, "Backbone conv1 grad should not be None"
        print("[PASS]")
    except Exception as e:
        print(f"[FAIL]: {e}")
        sys.exit(1)

    # ---------------------------------------------------------
    # Test 4: Contrastive Learning Overfit Gate
    # ---------------------------------------------------------
    print("Running Test 4: Contrastive Convergence (Overfit Gate)...", end=" ")
    try:
        torch.manual_seed(42)
        N = 8
        feat_dim = 64
        proj_dim = 32

        # Create N distinct synthetic image clusters (base image + noise)
        base_images = torch.randn(N, 3, 16, 16)
        # 2 views of each image with random perturbation
        x1 = base_images + 0.1 * torch.randn_like(base_images)
        x2 = base_images + 0.1 * torch.randn_like(base_images)

        model = SimCLR(feat_dim=feat_dim, proj_dim=proj_dim, temperature=0.2)
        optimizer = optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-4)

        initial_loss = None
        for step in range(60):
            optimizer.zero_grad()
            # Generate slight fresh view jitter each step to simulate live data augmentations
            v1 = x1 + 0.02 * torch.randn_like(x1)
            v2 = x2 + 0.02 * torch.randn_like(x2)
            loss, acc, _, _, _, _ = model(v1, v2)
            if initial_loss is None:
                initial_loss = loss.item()
            loss.backward()
            optimizer.step()

        final_loss = loss.item()
        final_acc = acc.item()
        assert final_loss < 0.35 or final_loss < initial_loss * 0.25, (
            f"Contrastive overfit failed: initial {initial_loss:.4f} -> final {final_loss:.4f}"
        )
        assert final_acc >= 0.85, f"Contrastive accuracy should exceed 85%, got {final_acc * 100:.1f}%"
        print(f"[PASS] (Loss {initial_loss:.3f} -> {final_loss:.3f}, Acc {final_acc * 100:.1f}%)")
    except Exception as e:
        print(f"[FAIL]: {e}")
        sys.exit(1)

    # ---------------------------------------------------------
    # Test 5: Downstream Linear Probe Evaluation
    # ---------------------------------------------------------
    print("Running Test 5: Downstream Linear Probe on Frozen Encoder...", end=" ")
    try:
        model.eval()
        probe = LinearProbe(feat_dim=feat_dim, num_classes=2)
        probe_opt = optim.AdamW(probe.parameters(), lr=1e-2)

        # Labels: first N/2 images are class 0, second N/2 are class 1
        labels = torch.tensor([0] * (N // 2) + [1] * (N // 2), dtype=torch.long)

        with torch.no_grad():
            h_frozen = model.encode(base_images)

        for _ in range(40):
            probe_opt.zero_grad()
            logits = probe(h_frozen)
            loss_probe = nn.functional.cross_entropy(logits, labels)
            loss_probe.backward()
            probe_opt.step()

        probe_acc = (logits.argmax(dim=-1) == labels).float().mean().item()
        assert probe_acc == 1.0, f"Linear probe on frozen features should achieve 100% on toy clusters, got {probe_acc}"
        print("[PASS]")
    except Exception as e:
        print(f"[FAIL]: {e}")
        sys.exit(1)

    print("\nAll SimCLR unit tests passed successfully!")


if __name__ == "__main__":
    run_tests()
