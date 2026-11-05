import sys
import importlib
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim

MODULE_NAME = sys.argv[1] if len(sys.argv) > 1 else "papers.15_lora.model"

try:
    mod = importlib.import_module(MODULE_NAME)
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import model as mod

LoRALinear = mod.LoRALinear
mark_only_lora_as_trainable = mod.mark_only_lora_as_trainable
apply_lora_to_model = mod.apply_lora_to_model


def test_step_zero_exact_identity():
    print("Testing Step-Zero Exact Identity Invariant (B = 0)...")
    torch.manual_seed(42)
    in_features, out_features, r = 32, 64, 4
    base_linear = nn.Linear(in_features, out_features)
    lora_linear = LoRALinear(base_layer=base_linear, r=r, lora_alpha=8.0)
    lora_linear.eval()

    x = torch.randn(2, 5, in_features)

    with torch.no_grad():
        out_base = base_linear(x)
        out_lora = lora_linear(x)

    # At step 0, Delta W = B * A = 0 * A = 0, so outputs must be bit-for-bit identical!
    diff = (out_base - out_lora).abs().max().item()
    assert diff == 0.0, f"Step 0 identity violated! Difference between base and LoRA output was {diff:.6e}"
    print(f"  [PASS] Step 0 identity verified: Delta W is exactly zero (diff: {diff:.2e}).")


def test_forward_and_low_rank_delta():
    print("Testing LoRA Forward Computation & Low-Rank Update...")
    torch.manual_seed(42)
    in_features, out_features, r = 16, 32, 4
    lora_alpha = 8.0
    layer = LoRALinear(in_features=in_features, out_features=out_features, r=r, lora_alpha=lora_alpha)
    layer.eval()

    # Perturb matrix B so Delta W != 0
    nn.init.normal_(layer.lora_B.weight, mean=0.0, std=0.1)

    x = torch.randn(2, 3, in_features)
    out = layer(x)
    assert out.shape == (2, 3, out_features), f"Expected shape {(2, 3, out_features)}, got {out.shape}"

    # Verify manual computation: base + (x @ (alpha/r * B * A).T)
    scaling = lora_alpha / r
    delta_W = (layer.lora_B.weight @ layer.lora_A.weight) * scaling
    expected_out = layer.linear(x) + torch.matmul(x, delta_W.t())

    diff = (out - expected_out).abs().max().item()
    assert diff < 1e-5, f"Manual LoRA computation mismatch! diff: {diff:.6e}"
    print(f"  [PASS] Low-rank forward pass matches mathematical contract (diff: {diff:.2e}).")


def test_merge_and_unmerge():
    print("Testing LoRA Weight Merge & Unmerge (Zero-Latency Inference)...")
    torch.manual_seed(42)
    in_features, out_features, r = 32, 48, 8
    layer = LoRALinear(in_features=in_features, out_features=out_features, r=r, lora_alpha=16.0)
    layer.eval()

    # Give non-zero weights to B
    nn.init.normal_(layer.lora_B.weight, std=0.05)
    orig_weight = layer.linear.weight.clone()

    x = torch.randn(4, in_features)

    # 1. Unmerged forward
    with torch.no_grad():
        out_unmerged = layer(x)

    # 2. Merge weights
    layer.merge()
    assert layer.merged is True, "layer.merged must be True after merge()"
    assert not torch.allclose(layer.linear.weight, orig_weight), "Base weight must be modified after merge()"

    with torch.no_grad():
        out_merged = layer(x)

    diff_merge = (out_unmerged - out_merged).abs().max().item()
    assert diff_merge < 1e-5, f"Merged output does not match unmerged output! diff: {diff_merge:.6e}"

    # 3. Unmerge weights
    layer.unmerge()
    assert layer.merged is False, "layer.merged must be False after unmerge()"
    assert torch.allclose(layer.linear.weight, orig_weight, atol=1e-6), "Base weight was not restored after unmerge!"

    with torch.no_grad():
        out_restored = layer(x)

    diff_restored = (out_unmerged - out_restored).abs().max().item()
    assert diff_restored < 1e-6, f"Restored output does not match! diff: {diff_restored:.6e}"

    print(f"  [PASS] Merge/unmerge cycles verified with exact output preservation (diff: {diff_merge:.2e}).")


def test_gradient_isolation_and_freezing():
    print("Testing Gradient Isolation via mark_only_lora_as_trainable...")
    in_features, out_features, r = 16, 24, 4

    model = nn.Sequential(
        LoRALinear(in_features=in_features, out_features=out_features, r=r),
        nn.ReLU(),
        LoRALinear(in_features=out_features, out_features=8, r=r),
    )

    # Freeze base model, train only LoRA
    mark_only_lora_as_trainable(model, bias="none")

    # Verify requires_grad status
    for name, param in model.named_parameters():
        if "lora_" in name:
            assert param.requires_grad, f"LoRA parameter {name} should have requires_grad=True!"
        else:
            assert not param.requires_grad, f"Base parameter {name} should have requires_grad=False!"

    # Forward & backward pass
    x = torch.randn(2, in_features)
    out = model(x)
    loss = out.sum()
    loss.backward()

    # Audit gradients
    for name, param in model.named_parameters():
        if "lora_" in name:
            assert param.grad is not None, f"LoRA parameter {name} must have gradients!"
            assert not torch.isnan(param.grad).any(), f"LoRA parameter {name} gradient has NaNs!"
        else:
            assert param.grad is None, f"Frozen base parameter {name} must NOT receive gradients!"

    print("  [PASS] Gradient isolation clean: base model parameters completely frozen.")


def test_single_batch_overfit_with_frozen_base():
    print("Testing Single-Batch Adaptation with 100% Frozen Base Network...")
    torch.manual_seed(42)
    B, in_dim, hidden_dim, out_dim = 4, 16, 32, 4
    r = 4

    # Base model with randomly initialized frozen weights
    model = nn.Sequential(
        LoRALinear(in_features=in_dim, out_features=hidden_dim, r=r, lora_alpha=8.0),
        nn.GELU(),
        LoRALinear(in_features=hidden_dim, out_features=out_dim, r=r, lora_alpha=8.0),
    )
    mark_only_lora_as_trainable(model, bias="none")

    # Only pass parameters with requires_grad=True to optimizer
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    assert len(trainable_params) == 4, f"Expected 4 LoRA parameter tensors (2x A, 2x B), got {len(trainable_params)}"

    optimizer = optim.Adam(trainable_params, lr=1e-2)
    criterion = nn.MSELoss()

    x = torch.randn(B, in_dim)
    target = torch.randn(B, out_dim)

    initial_loss = None
    final_loss = None

    for epoch in range(80):
        optimizer.zero_grad()
        pred = model(x)
        loss = criterion(pred, target)
        loss.backward()
        optimizer.step()

        if epoch == 0:
            initial_loss = loss.item()
        final_loss = loss.item()
        if final_loss < 0.01:
            break

    print(f"  Loss: {initial_loss:.4f} -> {final_loss:.4f} in {epoch+1} steps")
    assert final_loss < 0.05, f"LoRA adapter failed to adapt frozen base model! Final loss: {final_loss:.4f}"
    print("  [PASS] Single batch adaptation overfit successfully converged.")


def test_apply_lora_to_model():
    print("Testing apply_lora_to_model Automatic Model Surgery...")
    class ToyTransformerBlock(nn.Module):
        def __init__(self):
            super().__init__()
            self.q_proj = nn.Linear(32, 32)
            self.v_proj = nn.Linear(32, 32)
            self.out_proj = nn.Linear(32, 32)
            self.ln = nn.LayerNorm(32)

    block = ToyTransformerBlock()
    # Apply LoRA only to attention query and value projections
    apply_lora_to_model(block, target_module_names=["q_proj", "v_proj"], r=4, lora_alpha=8.0)

    assert isinstance(block.q_proj, LoRALinear), "q_proj should be wrapped in LoRALinear!"
    assert isinstance(block.v_proj, LoRALinear), "v_proj should be wrapped in LoRALinear!"
    assert isinstance(block.out_proj, nn.Linear) and not isinstance(block.out_proj, LoRALinear), (
        "out_proj should remain standard nn.Linear!"
    )
    assert isinstance(block.ln, nn.LayerNorm), "LayerNorm should remain untouched!"

    print("  [PASS] apply_lora_to_model targeted injection verified.")


if __name__ == "__main__":
    print(f"Running unit tests for: {MODULE_NAME}\n" + "=" * 50)
    test_step_zero_exact_identity()
    test_forward_and_low_rank_delta()
    test_merge_and_unmerge()
    test_gradient_isolation_and_freezing()
    test_single_batch_overfit_with_frozen_base()
    test_apply_lora_to_model()
    print("=" * 50 + "\nALL TESTS PASSED! [PASS]")
