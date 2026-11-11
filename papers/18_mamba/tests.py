import sys
import importlib
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

def run_tests():
    module_name = sys.argv[1] if len(sys.argv) > 1 else "papers.18_mamba.model"
    try:
        mod = importlib.import_module(module_name)
    except Exception as e:
        print(f"[FAIL] Could not import {module_name}: {e}")
        sys.exit(1)

    MambaConfig = getattr(mod, "MambaConfig")
    selective_scan = getattr(mod, "selective_scan")
    SelectiveSSM = getattr(mod, "SelectiveSSM")
    MambaBlock = getattr(mod, "MambaBlock")
    MambaLM = getattr(mod, "MambaLM")

    torch.manual_seed(42)

    # ---------------------------------------------------------
    # Test 1: Selective Scan Shapes, Values & Causality
    # ---------------------------------------------------------
    print("Running Test 1: Selective Scan (Shapes & Causality)...", end=" ")
    try:
        B, L, D, N = 2, 8, 16, 4
        u = torch.randn(B, L, D)
        delta = F.softplus(torch.randn(B, L, D))
        A = -torch.exp(torch.randn(D, N))
        B_mat = torch.randn(B, L, N)
        C_mat = torch.randn(B, L, N)
        D_skip = torch.randn(D)

        y = selective_scan(u, delta, A, B_mat, C_mat, D_skip)
        assert y.shape == (B, L, D), f"Expected shape {(B, L, D)}, got {y.shape}"

        # Strict Causality check: altering future inputs must not change past outputs
        u_mod = u.clone()
        u_mod[:, 5:, :] += 10.0  # Modify tokens from index 5 onwards
        y_mod = selective_scan(u_mod, delta, A, B_mat, C_mat, D_skip)

        assert torch.allclose(y[:, :5, :], y_mod[:, :5, :], atol=1e-5), (
            "Causality violation in selective_scan! Modifying future steps altered earlier steps."
        )
        print("[PASS]")
    except Exception as e:
        print(f"[FAIL]: {e}")
        sys.exit(1)

    # ---------------------------------------------------------
    # Test 2: SelectiveSSM & Parameterization
    # ---------------------------------------------------------
    print("Running Test 2: SelectiveSSM (S6)...", end=" ")
    try:
        ssm = SelectiveSSM(d_inner=32, d_state=8, dt_rank=8)
        x = torch.randn(2, 6, 32, requires_grad=True)
        out = ssm(x)
        assert out.shape == (2, 6, 32), f"Expected shape (2, 6, 32), got {out.shape}"

        # Audit gradients
        loss = out.sum()
        loss.backward()

        assert ssm.A_log.grad is not None, "A_log gradient should not be None"
        assert ssm.D.grad is not None, "D gradient should not be None"
        assert ssm.x_proj.weight.grad is not None, "x_proj gradient should not be None"
        assert ssm.dt_proj.weight.grad is not None, "dt_proj gradient should not be None"
        print("[PASS]")
    except Exception as e:
        print(f"[FAIL]: {e}")
        sys.exit(1)

    # ---------------------------------------------------------
    # Test 3: MambaBlock Forward
    # ---------------------------------------------------------
    print("Running Test 3: MambaBlock...", end=" ")
    try:
        config = MambaConfig(d_model=32, d_state=8, d_conv=4, expand=2)
        block = MambaBlock(config)
        x = torch.randn(2, 10, 32)
        out = block(x)
        assert out.shape == (2, 10, 32), f"Expected shape (2, 10, 32), got {out.shape}"
        print("[PASS]")
    except Exception as e:
        print(f"[FAIL]: {e}")
        sys.exit(1)

    # ---------------------------------------------------------
    # Test 4: MambaLM Causality Audit
    # ---------------------------------------------------------
    print("Running Test 4: MambaLM Causality...", end=" ")
    try:
        config = MambaConfig(d_model=32, d_state=8, d_conv=4, expand=2, vocab_size=50, n_layers=2)
        model = MambaLM(config)
        model.eval()

        input_ids = torch.randint(0, 50, (1, 10))
        with torch.no_grad():
            logits1 = model(input_ids)

        # Perturb token at index 7
        input_ids_mod = input_ids.clone()
        input_ids_mod[0, 7] = (input_ids_mod[0, 7] + 5) % 50

        with torch.no_grad():
            logits2 = model(input_ids_mod)

        # Logits at positions 0..6 must remain identical
        assert torch.allclose(logits1[:, :7, :], logits2[:, :7, :], atol=1e-5), (
            "Causality leak! MambaLM predicted earlier positions using future tokens."
        )
        print("[PASS]")
    except Exception as e:
        print(f"[FAIL]: {e}")
        sys.exit(1)

    # ---------------------------------------------------------
    # Test 5: Single-Batch Overfit & Autoregressive Generation
    # ---------------------------------------------------------
    print("Running Test 5: Single-Batch Overfit & Generation...", end=" ")
    try:
        torch.manual_seed(42)
        config = MambaConfig(d_model=32, d_state=8, d_conv=4, expand=2, vocab_size=30, n_layers=2)
        model = MambaLM(config)
        optimizer = optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-4)

        # Toy target sequence
        target_seq = torch.tensor([[3, 7, 12, 15, 22, 9, 14, 18, 5, 27, 2, 11]], dtype=torch.long)
        inputs = target_seq[:, :-1]
        targets = target_seq[:, 1:]

        initial_loss = None
        for step in range(60):
            optimizer.zero_grad()
            logits = model(inputs)
            loss = F.cross_entropy(logits.reshape(-1, config.vocab_size), targets.reshape(-1))
            if initial_loss is None:
                initial_loss = loss.item()
            loss.backward()
            optimizer.step()

        final_loss = loss.item()
        assert final_loss < 0.20 or final_loss < initial_loss * 0.1, (
            f"Overfit failed: initial {initial_loss:.4f} -> final {final_loss:.4f}"
        )

        # Test generation with greedy decoding (temperature -> 0)
        prompt = target_seq[:, :4]
        generated = model.generate(prompt, max_new_tokens=len(target_seq[0]) - 4, temperature=1e-5)
        matches = (generated[0] == target_seq[0]).sum().item()
        assert matches >= 10, f"Model should memorize sequence: got {generated.tolist()} vs {target_seq.tolist()}"

        print(f"[PASS] (Loss {initial_loss:.3f} -> {final_loss:.3f})")
    except Exception as e:
        print(f"[FAIL]: {e}")
        sys.exit(1)

    print("\nAll Mamba unit tests passed successfully!")


if __name__ == "__main__":
    run_tests()
