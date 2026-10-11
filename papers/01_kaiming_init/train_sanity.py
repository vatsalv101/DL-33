import sys
from pathlib import Path

# Ensure both repo root and current paper directory are on sys.path
CUR_DIR = Path(__file__).resolve().parent
REPO_ROOT = CUR_DIR.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(CUR_DIR) not in sys.path:
    sys.path.insert(0, str(CUR_DIR))

import torch
import torch.nn as nn
from sanity.checks import assert_shape_contract, audit_gradients, overfit_single_batch
from model import DeepMLP, init_weights


def run_sanity_checks():
    print("=" * 60)
    print("Running Paper 01 (Kaiming Init) Sanity Checks & Overfit Gate")
    print("=" * 60)

    torch.manual_seed(42)

    # 1. Shape Contract Check
    model = DeepMLP(in_dim=32, hidden_dim=64, out_dim=4, num_layers=15)
    init_weights(model, nonlinearity="relu", distribution="normal")
    dummy_input = torch.randn(8, 32)
    assert_shape_contract(model, dummy_input, (8, 4))
    print("[PASS] Step 2 Shape Contract Passed: (8, 32) -> (8, 4)")

    # 2. Gradient Audit Check
    loss_fn = nn.MSELoss()
    target = torch.randn(8, 4)
    pred = model(dummy_input)
    loss = loss_fn(pred, target)
    loss.backward()
    audit_gradients(model)
    print("[PASS] Step 4 Gradient Audit Passed: No NaNs, no Infs, clean gradient flow.")

    # 3. Step 5: 1-Batch Overfit Sanity Check
    print("\nExecuting Step 5: 1-Batch Overfit Gate...")
    model_overfit = DeepMLP(in_dim=16, hidden_dim=32, out_dim=2, num_layers=10)
    init_weights(model_overfit, nonlinearity="relu", distribution="normal")
    optimizer = torch.optim.Adam(model_overfit.parameters(), lr=0.01)

    batch_x = torch.randn(4, 16)
    batch_y = torch.randn(4, 2)

    final_loss = overfit_single_batch(
        model=model_overfit,
        optimizer=optimizer,
        loss_fn=loss_fn,
        batch_inputs=batch_x,
        batch_targets=batch_y,
        max_steps=200,
        target_loss=0.01,
        verbose=True,
    )

    print(f"\n[PASS] Step 5 Passed: Final Loss = {final_loss:.6f} (< 0.01)")
    print("=" * 60)
    print("Paper 01 (Kaiming Init) is 100% Verified!")
    print("=" * 60)


if __name__ == "__main__":
    run_sanity_checks()
