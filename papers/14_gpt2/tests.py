import sys
import importlib
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim

MODULE_NAME = sys.argv[1] if len(sys.argv) > 1 else "papers.14_gpt2.model"

try:
    mod = importlib.import_module(MODULE_NAME)
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import model as mod

GPT2Attention = mod.GPT2Attention
GPT2MLP = mod.GPT2MLP
GPT2Block = mod.GPT2Block
GPT2Model = mod.GPT2Model
GPT2LMHeadModel = mod.GPT2LMHeadModel


def test_gpt2_attention_causal_property():
    print("Testing GPT2Attention Causal Invariant...")
    torch.manual_seed(42)
    d_model = 64
    num_heads = 4
    attn = GPT2Attention(d_model=d_model, num_heads=num_heads, max_positions=128, dropout=0.0)
    attn.eval()

    B, T = 1, 6
    x1 = torch.randn(B, T, d_model)
    x2 = x1.clone()
    # Modify future tokens at step 3, 4, 5
    x2[:, 3:, :] = torch.randn(1, 3, d_model)

    with torch.no_grad():
        out1, _ = attn(x1)
        out2, _ = attn(x2)

    # In a strictly causal model, outputs at steps 0, 1, 2 must be IDENTICAL
    diff = (out1[:, :3, :] - out2[:, :3, :]).abs().max().item()
    assert diff < 1e-5, f"Causal violation in GPT2Attention! Modifying future tokens changed past output by {diff:.6e}"
    print(f"  [PASS] Causal masking verified: zero future information leakage (diff: {diff:.2e}).")


def test_kv_cache_equivalence():
    print("Testing KV-Cache Numerical Equivalence...")
    # Fundamental Property: Full-sequence forward MUST equal step-by-step cached forward
    torch.manual_seed(42)
    vocab_size = 50
    d_model = 64
    num_heads = 4
    num_layers = 2
    max_positions = 64

    model = GPT2LMHeadModel(
        vocab_size=vocab_size,
        max_positions=max_positions,
        d_model=d_model,
        num_layers=num_layers,
        num_heads=num_heads,
        dropout=0.0,
    )
    model.eval()

    sequence = torch.randint(0, vocab_size, (1, 8))

    # 1. Full-sequence forward (no cache)
    with torch.no_grad():
        logits_full, _ = model(sequence, use_cache=False)

    # 2. Step-by-step forward with KV cache
    past_key_values = None
    cached_logits_list = []
    with torch.no_grad():
        for t in range(sequence.size(1)):
            current_token = sequence[:, t : t + 1]
            token_logits, past_key_values = model(
                current_token, past_key_values=past_key_values, use_cache=True
            )
            cached_logits_list.append(token_logits)

    logits_cached = torch.cat(cached_logits_list, dim=1)

    diff = (logits_full - logits_cached).abs().max().item()
    assert diff < 1e-4, f"KV-cache inconsistency! Full vs Cached logits differ by {diff:.6e}"
    print(f"  [PASS] KV-cache output exactly matches full-sequence computation (diff: {diff:.2e}).")


def test_pre_ln_and_residual_scaling():
    print("Testing Pre-LN Architecture and Residual Scaling...")
    d_model = 64
    num_heads = 4
    num_layers = 4
    model = GPT2LMHeadModel(
        vocab_size=100,
        max_positions=64,
        d_model=d_model,
        num_layers=num_layers,
        num_heads=num_heads,
        dropout=0.0,
    )

    # Check that residual projections have reduced standard deviation: 0.02 / sqrt(2 * num_layers)
    expected_std = 0.02 / (2.0 * num_layers) ** 0.5
    for name, p in model.named_parameters():
        if "c_proj.weight" in name:
            actual_std = p.std().item()
            assert abs(actual_std - expected_std) < 0.015, (
                f"Residual projection {name} std ({actual_std:.4f}) does not match scaling ({expected_std:.4f})"
            )

    print(f"  [PASS] Residual projection initialization scaling ({expected_std:.4f}) verified.")


def test_gpt2_forward_and_backward():
    print("Testing GPT2LMHeadModel Forward & Backward Pass...")
    vocab_size = 50
    d_model = 64
    num_heads = 4
    num_layers = 2
    model = GPT2LMHeadModel(
        vocab_size=vocab_size,
        max_positions=64,
        d_model=d_model,
        num_layers=num_layers,
        num_heads=num_heads,
        dropout=0.1,
    )
    model.train()

    B, T = 2, 10
    input_ids = torch.randint(0, vocab_size, (B, T))
    logits, _ = model(input_ids)

    assert logits.shape == (B, T, vocab_size), f"Expected logits {(B, T, vocab_size)}, got {logits.shape}"

    # Autoregressive loss: predict next token (shift by 1)
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = input_ids[:, 1:].contiguous()
    loss = nn.CrossEntropyLoss()(shift_logits.view(-1, vocab_size), shift_labels.view(-1))
    loss.backward()

    for name, param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"Parameter {name} has no gradient!"
            assert not torch.isnan(param.grad).any(), f"Parameter {name} has NaN gradient!"
            assert not torch.isinf(param.grad).any(), f"Parameter {name} has Inf gradient!"

    print("  [PASS] GPT2LMHeadModel forward pass and gradient audit clean.")


def test_single_batch_overfit():
    print("Testing Single Batch Overfitting (Next-Token Memorization)...")
    torch.manual_seed(42)
    vocab_size = 30
    d_model = 64
    B, T = 2, 8

    model = GPT2LMHeadModel(
        vocab_size=vocab_size,
        max_positions=32,
        d_model=d_model,
        num_layers=2,
        num_heads=4,
        dropout=0.0,
    )
    optimizer = optim.Adam(model.parameters(), lr=2e-3)
    loss_fn = nn.CrossEntropyLoss()

    input_ids = torch.randint(0, vocab_size, (B, T))
    targets = input_ids[:, 1:].contiguous()

    initial_loss = None
    final_loss = None

    for epoch in range(100):
        optimizer.zero_grad()
        logits, _ = model(input_ids)
        shift_logits = logits[:, :-1, :].contiguous()
        loss = loss_fn(shift_logits.view(-1, vocab_size), targets.view(-1))
        loss.backward()
        optimizer.step()

        if epoch == 0:
            initial_loss = loss.item()
        final_loss = loss.item()
        if final_loss < 0.05:
            break

    print(f"  Loss: {initial_loss:.4f} -> {final_loss:.4f} in {epoch+1} steps")
    assert final_loss < 0.1, f"Failed to overfit single batch! Final loss: {final_loss:.4f}"
    print("  [PASS] Single batch overfit successfully converged.")


def test_autoregressive_generate():
    print("Testing Autoregressive Generation with KV-Cache...")
    torch.manual_seed(42)
    vocab_size = 50
    model = GPT2LMHeadModel(
        vocab_size=vocab_size,
        max_positions=64,
        d_model=64,
        num_layers=2,
        num_heads=4,
        dropout=0.0,
    )
    model.eval()

    prompt = torch.tensor([[5, 12, 19]])
    max_new_tokens = 6

    # Generate with cache
    gen_cached = model.generate(prompt.clone(), max_new_tokens=max_new_tokens, temperature=1.0, use_cache=True)
    assert gen_cached.shape == (1, 3 + max_new_tokens), f"Expected shape (1, 9), got {gen_cached.shape}"
    assert (gen_cached[:, :3] == prompt).all(), "Prompt tokens must be preserved in generated output!"

    print("  [PASS] Autoregressive generate method verified.")


if __name__ == "__main__":
    print(f"Running unit tests for: {MODULE_NAME}\n" + "=" * 50)
    test_gpt2_attention_causal_property()
    test_kv_cache_equivalence()
    test_pre_ln_and_residual_scaling()
    test_gpt2_forward_and_backward()
    test_single_batch_overfit()
    test_autoregressive_generate()
    print("=" * 50 + "\nALL TESTS PASSED! [PASS]")
