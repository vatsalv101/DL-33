"""Paper 02: Adam & AdamW (Decoupled Weight Decay Optimizer)
Unit Tests Suite.
"""

import math
import torch
import torch.nn as nn


def test_state_initialization(optimizer_cls):
    """Tests that optimizer state buffers ('step', 'exp_avg', 'exp_avg_sq') are created correctly."""
    all_ok = True
    w = torch.tensor([1.0, -2.0, 3.0], requires_grad=True)
    opt = optimizer_cls([w], lr=0.01)

    loss = (w ** 2).sum()
    loss.backward()
    opt.step()

    state = opt.state[w]
    if "step" not in state or state["step"] != 1:
        print("  [FAIL] State 'step' counter not properly initialized to 1")
        all_ok = False
    if "exp_avg" not in state or state["exp_avg"].shape != w.shape:
        print("  [FAIL] Buffer 'exp_avg' missing or shape mismatch")
        all_ok = False
    if "exp_avg_sq" not in state or state["exp_avg_sq"].shape != w.shape:
        print("  [FAIL] Buffer 'exp_avg_sq' missing or shape mismatch")
        all_ok = False

    if all_ok:
        print("  [PASS] State initialization ('step', 'exp_avg', 'exp_avg_sq') is correct!")
    return all_ok


def test_decoupled_weight_decay(optimizer_cls):
    """Tests that decoupled weight decay applies directly to parameters even when gradients are zero."""
    all_ok = True
    w = torch.tensor([10.0], requires_grad=True)
    lr = 0.1
    wd = 0.05
    opt = optimizer_cls([w], lr=lr, weight_decay=wd)

    # Fake zero gradient: loss = 0 * w
    loss = 0.0 * w.sum()
    loss.backward()
    opt.step()

    # Expected: w_new = w * (1 - lr * wd) = 10.0 * (1 - 0.005) = 9.95
    expected = 10.0 * (1.0 - lr * wd)
    actual = w.item()

    if not math.isclose(actual, expected, rel_tol=1e-5):
        print(f"  [FAIL] Decoupled weight decay: w={actual:.5f}, expected {expected:.5f}")
        all_ok = False
    else:
        print(f"  [PASS] Decoupled weight decay applied independently of gradients (w: 10.0 -> {actual:.4f})!")

    return all_ok


def test_bias_correction_step_one(optimizer_cls):
    """Tests analytical step 1 update under constant gradient."""
    all_ok = True
    w = torch.tensor([2.0], requires_grad=True)
    lr = 0.01
    beta1, beta2 = 0.9, 0.999
    eps = 1e-8
    opt = optimizer_cls([w], lr=lr, betas=(beta1, beta2), eps=eps, weight_decay=0.0)

    # Constant gradient g = 1.0
    w.grad = torch.tensor([1.0])
    opt.step()

    # Step 1 analytical values:
    # m_1 = (1 - beta1) * 1.0 = 0.1
    # v_1 = (1 - beta2) * 1.0^2 = 0.001
    # bias1 = 1 - beta1 = 0.1  => m_hat = 0.1 / 0.1 = 1.0
    # bias2 = 1 - beta2 = 0.001 => v_hat = 0.001 / 0.001 = 1.0
    # step_size = 0.01
    # delta = -0.01 * (1.0 / (sqrt(1.0) + eps)) = -0.01
    expected_w = 2.0 - lr

    if not math.isclose(w.item(), expected_w, rel_tol=1e-5):
        print(f"  [FAIL] Step 1 update value {w.item():.6f}, expected {expected_w:.6f}")
        all_ok = False
    else:
        print("  [PASS] Step 1 bias correction matches exact analytical formula!")

    return all_ok


def test_equivalence_to_pytorch(optimizer_cls):
    """Validates mathematical equivalence against PyTorch's native AdamW on quadratic loss."""
    all_ok = True
    torch.manual_seed(42)

    w_custom = torch.tensor([[1.5, -2.0], [0.5, 3.0]], requires_grad=True)
    w_torch = w_custom.clone().detach().requires_grad_(True)

    opt_custom = optimizer_cls([w_custom], lr=0.01, weight_decay=0.05)
    opt_torch = torch.optim.AdamW([w_torch], lr=0.01, weight_decay=0.05)

    max_diff = 0.0
    for step in range(1, 16):
        loss_custom = ((w_custom - 2.0) ** 2).sum()
        loss_torch = ((w_torch - 2.0) ** 2).sum()

        loss_custom.backward()
        loss_torch.backward()

        opt_custom.step()
        opt_torch.step()

        opt_custom.zero_grad()
        opt_torch.zero_grad()

        diff = (w_custom - w_torch).abs().max().item()
        max_diff = max(max_diff, diff)

    if max_diff >= 1e-6:
        print(f"  [FAIL] Diverged from PyTorch reference: max diff = {max_diff:.8e} (expected < 1e-6)")
        all_ok = False
    else:
        print(f"  [PASS] Custom AdamW is mathematically identical to PyTorch AdamW (max diff: {max_diff:.2e})!")

    return all_ok


def test_rosenbrock_convergence(optimizer_cls):
    """Stress-tests optimizer on the non-convex Rosenbrock valley."""
    all_ok = True
    # f(x, y) = (1 - x)^2 + 100 * (y - x^2)^2, global minimum at (1, 1)
    xy = torch.tensor([-1.5, 2.0], requires_grad=True)
    opt = optimizer_cls([xy], lr=0.08, weight_decay=0.0)

    for _ in range(800):
        opt.zero_grad()
        x, y = xy[0], xy[1]
        loss = (1.0 - x) ** 2 + 100.0 * (y - x ** 2) ** 2
        loss.backward()
        opt.step()

    final_loss = loss.item()
    dist_to_opt = ((xy - torch.tensor([1.0, 1.0])) ** 2).sum().sqrt().item()

    if final_loss > 0.05 or dist_to_opt > 0.2:
        print(f"  [FAIL] Rosenbrock optimization did not converge (loss={final_loss:.4f}, dist={dist_to_opt:.4f})")
        all_ok = False
    else:
        print(f"  [PASS] Solved Rosenbrock valley! Converged to ({xy[0].item():.3f}, {xy[1].item():.3f}) [Loss: {final_loss:.6f}]")

    return all_ok


def run_all_tests():
    """Runs all test gates from the terminal."""
    print("=" * 60)
    print("Running Paper 02 (AdamW) Test Suite")
    print("=" * 60)

    try:
        from model import AdamW
    except ImportError:
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from model import AdamW

    try:
        ok1 = test_state_initialization(AdamW)
        ok2 = test_decoupled_weight_decay(AdamW)
        ok3 = test_bias_correction_step_one(AdamW)
        ok4 = test_equivalence_to_pytorch(AdamW)
        ok5 = test_rosenbrock_convergence(AdamW)

        if all([ok1, ok2, ok3, ok4, ok5]):
            print("\n" + "=" * 60)
            print("ALL PAPER 02 TESTS PASSED! YOU INVENTED ADAMW!")
            print("=" * 60)
    except NotImplementedError:
        print("\n[PENDING] Implement AdamW.step() in 'papers/02_adamw/model.py' to pass tests.")


if __name__ == "__main__":
    run_all_tests()
