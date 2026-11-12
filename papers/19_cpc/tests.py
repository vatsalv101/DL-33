import sys
import math
import importlib
import torch
import torch.nn as nn
import torch.optim as optim


def run_tests():
    module_name = sys.argv[1] if len(sys.argv) > 1 else "papers.19_cpc.model"
    try:
        mod = importlib.import_module(module_name)
    except Exception as e:
        print(f"[FAIL] Could not import {module_name}: {e}")
        sys.exit(1)

    info_nce_loss = getattr(mod, "info_nce_loss")
    CPCEncoder = getattr(mod, "CPCEncoder")
    CPCModel = getattr(mod, "CPCModel")

    torch.manual_seed(42)

    # ---------------------------------------------------------
    # Test 1: InfoNCE Loss & MI Lower Bound Baseline
    # ---------------------------------------------------------
    print("Running Test 1: InfoNCE Loss & MI Lower Bound...", end=" ")
    try:
        N, D = 32, 64
        # Case A: Perfect positive alignment
        keys = torch.randn(N, D)
        queries = keys.clone()
        loss_perfect, acc_perfect = info_nce_loss(queries, keys, temperature=0.1, normalize=True)
        assert abs(acc_perfect.item() - 1.0) < 1e-5, f"Expected acc 1.0, got {acc_perfect.item()}"
        assert loss_perfect.item() < 0.05, f"Expected very small loss for exact match, got {loss_perfect.item()}"

        # Case B: Uncorrelated random features -> loss should approximate log(N)
        rand_q = torch.randn(N, D)
        rand_k = torch.randn(N, D)
        loss_rand, _ = info_nce_loss(rand_q, rand_k, temperature=0.5, normalize=True)
        expected_bound = math.log(N)
        assert abs(loss_rand.item() - expected_bound) < 0.6, (
            f"Random guessing loss {loss_rand.item():.3f} should be close to log(N)={expected_bound:.3f}"
        )
        print("[PASS]")
    except Exception as e:
        print(f"[FAIL]: {e}")
        sys.exit(1)

    # ---------------------------------------------------------
    # Test 2: CPCEncoder & Autoregressive Causality
    # ---------------------------------------------------------
    print("Running Test 2: Encoder Shapes & Context Causality...", end=" ")
    try:
        model = CPCModel(input_dim=16, latent_dim=32, context_dim=32, prediction_steps=4)
        model.eval()
        x = torch.randn(2, 12, 16)
        z, c = model.encode(x)
        assert z.shape == (2, 12, 32), f"Expected z shape (2, 12, 32), got {z.shape}"
        assert c.shape == (2, 12, 32), f"Expected c shape (2, 12, 32), got {c.shape}"

        # Causality check: modifying x[:, 6:, :] must not alter c[:, :6, :]
        x_mod = x.clone()
        x_mod[:, 6:, :] += 20.0
        _, c_mod = model.encode(x_mod)
        assert torch.allclose(c[:, :6, :], c_mod[:, :6, :], atol=1e-5), (
            "Causality violation! Context c_t should not depend on future tokens."
        )
        print("[PASS]")
    except Exception as e:
        print(f"[FAIL]: {e}")
        sys.exit(1)

    # ---------------------------------------------------------
    # Test 3: Forward Pass & Gradient Audit
    # ---------------------------------------------------------
    print("Running Test 3: CPC Forward Pass & Gradients...", end=" ")
    try:
        model = CPCModel(input_dim=16, latent_dim=32, context_dim=32, prediction_steps=3)
        x = torch.randn(4, 10, 16)
        loss, acc, z, c = model(x)
        assert loss.ndim == 0 and loss.item() > 0.0
        assert 0.0 <= acc.item() <= 1.0

        loss.backward()
        assert model.encoder.net[0].weight.grad is not None, "Encoder grad should not be None"
        assert model.ar_gru.weight_ih_l0.grad is not None, "GRU grad should not be None"
        assert model.predictors[0].weight.grad is not None, "Predictor W_k grad should not be None"
        print("[PASS]")
    except Exception as e:
        print(f"[FAIL]: {e}")
        sys.exit(1)

    # ---------------------------------------------------------
    # Test 4: Temporal Representation Learning (Overfit Gate)
    # ---------------------------------------------------------
    print("Running Test 4: Temporal Predictive Coding Convergence...", end=" ")
    try:
        torch.manual_seed(42)
        B, L, D_in = 8, 16, 8
        # Construct structured predictable temporal waves with unique phases per sample
        t = torch.linspace(0, 4 * math.pi, L).unsqueeze(0).unsqueeze(-1)  # (1, L, 1)
        phases = torch.linspace(0, 2 * math.pi, B).unsqueeze(1).unsqueeze(-1)  # (B, 1, 1)
        freqs = torch.arange(1, D_in + 1, dtype=torch.float32).view(1, 1, D_in) * 0.5
        x = torch.sin(t * freqs + phases)

        model = CPCModel(input_dim=D_in, latent_dim=32, context_dim=32, prediction_steps=3, temperature=0.1)
        optimizer = optim.AdamW(model.parameters(), lr=3e-3)

        initial_loss = None
        for step in range(120):
            optimizer.zero_grad()
            loss, acc, _, _ = model(x)
            if initial_loss is None:
                initial_loss = loss.item()
            loss.backward()
            optimizer.step()

        final_loss = loss.item()
        final_acc = acc.item()
        assert final_acc > 0.80, f"Expected contrastive accuracy > 80%, got {final_acc * 100:.1f}%"
        assert final_loss < initial_loss * 0.35, f"Expected significant loss drop: {initial_loss:.3f} -> {final_loss:.3f}"
        print(f"[PASS] (Loss {initial_loss:.3f} -> {final_loss:.3f}, Acc {final_acc * 100:.1f}%)")
    except Exception as e:
        print(f"[FAIL]: {e}")
        sys.exit(1)

    print("\nAll CPC / InfoNCE unit tests passed successfully!")


if __name__ == "__main__":
    run_tests()
