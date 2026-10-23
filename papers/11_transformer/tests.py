import sys
import importlib
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim

# Dynamic import to support testing student or solution
MODULE_NAME = sys.argv[1] if len(sys.argv) > 1 else "papers.11_transformer.model"

try:
    mod = importlib.import_module(MODULE_NAME)
except ModuleNotFoundError:
    # Fallback to local import if run from within papers/11_transformer/
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import model as mod

ScaledDotProductAttention = mod.ScaledDotProductAttention
MultiHeadAttention = mod.MultiHeadAttention
PositionwiseFeedForward = mod.PositionwiseFeedForward
PositionalEncoding = mod.PositionalEncoding
TransformerEncoderLayer = mod.TransformerEncoderLayer
TransformerEncoder = mod.TransformerEncoder
TransformerDecoderLayer = mod.TransformerDecoderLayer
TransformerDecoder = mod.TransformerDecoder
Transformer = mod.Transformer


def test_scaled_dot_product_attention():
    print("Testing ScaledDotProductAttention...")
    attn = ScaledDotProductAttention(dropout=0.0)
    attn.eval()

    B, H, Tq, Tk, d_k = 2, 4, 8, 8, 32
    q = torch.randn(B, H, Tq, d_k)
    k = torch.randn(B, H, Tk, d_k)
    v = torch.randn(B, H, Tk, d_k)

    # 1. Unmasked forward
    out, weights = attn(q, k, v)
    assert out.shape == (B, H, Tq, d_k), f"Expected out shape {(B, H, Tq, d_k)}, got {out.shape}"
    assert weights.shape == (B, H, Tq, Tk), f"Expected weights shape {(B, H, Tq, Tk)}, got {weights.shape}"
    assert torch.allclose(weights.sum(dim=-1), torch.ones_like(weights.sum(dim=-1)), atol=1e-5), \
        "Attention weights must sum to 1.0 along key dimension"

    # 2. Masked forward (mask out last 2 keys)
    mask = torch.ones(B, 1, 1, Tk, dtype=torch.bool)
    mask[:, :, :, -2:] = False
    out_m, weights_m = attn(q, k, v, mask=mask)
    assert (weights_m[:, :, :, -2:] < 1e-4).all(), \
        f"Masked positions should have attention weights ~0, max was {weights_m[:, :, :, -2:].max().item()}"

    print("  [PASS] ScaledDotProductAttention shape, softmax sum, and masking verified.")


def test_multi_head_attention():
    print("Testing MultiHeadAttention...")
    d_model = 64
    num_heads = 4
    mha = MultiHeadAttention(d_model=d_model, num_heads=num_heads, dropout=0.0)
    mha.eval()

    B, Tq, Tk = 2, 10, 14
    q = torch.randn(B, Tq, d_model)
    k = torch.randn(B, Tk, d_model)
    v = torch.randn(B, Tk, d_model)

    # Cross-attention shape test
    out, weights = mha(q, k, v)
    assert out.shape == (B, Tq, d_model), f"Expected out {(B, Tq, d_model)}, got {out.shape}"
    assert weights.shape == (B, num_heads, Tq, Tk), f"Expected weights {(B, num_heads, Tq, Tk)}, got {weights.shape}"

    # Self-attention shape test
    out_self, _ = mha(q, q, q)
    assert out_self.shape == (B, Tq, d_model), f"Expected self out {(B, Tq, d_model)}, got {out_self.shape}"

    print("  [PASS] MultiHeadAttention projections and head splitting verified.")


def test_positional_encoding():
    print("Testing PositionalEncoding...")
    d_model = 64
    max_len = 100
    pe_module = PositionalEncoding(d_model=d_model, max_len=max_len, dropout=0.0)
    pe_module.eval()

    B, T = 2, 25
    x = torch.zeros(B, T, d_model)
    out = pe_module(x)

    assert out.shape == (B, T, d_model), f"Expected shape {(B, T, d_model)}, got {out.shape}"
    # Positional encodings must be non-zero and differ across timesteps
    assert not torch.allclose(out[:, 0, :], out[:, 1, :]), "Different positions must have distinct encodings!"
    assert not torch.allclose(out[:, 0, :], torch.zeros_like(out[:, 0, :])), "Positional encoding must not be all zeros!"

    print("  [PASS] PositionalEncoding dimension and position distinction verified.")


def test_causal_masking_autoregressive_property():
    print("Testing Causal Masking (Autoregressive Invariant)...")
    # Autoregressive requirement: output at time step t must NOT depend on token at t+1, t+2, ...
    torch.manual_seed(42)
    src_vocab = 30
    trg_vocab = 30
    d_model = 32
    num_heads = 2
    num_layers = 2
    d_ff = 64

    model = Transformer(
        src_vocab_size=src_vocab,
        trg_vocab_size=trg_vocab,
        d_model=d_model,
        num_heads=num_heads,
        num_encoder_layers=num_layers,
        num_decoder_layers=num_layers,
        d_ff=d_ff,
        dropout=0.0,
    )
    model.eval()

    src = torch.randint(1, src_vocab, (1, 6))
    trg1 = torch.tensor([[1, 5, 9, 12, 18, 22]])
    # trg2 shares first 3 tokens with trg1, but alters the 4th, 5th, 6th tokens
    trg2 = torch.tensor([[1, 5, 9, 25, 27, 28]])

    sz = trg1.size(1)
    causal_mask = Transformer.generate_square_subsequent_mask(sz)

    with torch.no_grad():
        out1 = model(src, trg1, trg_mask=causal_mask)
        out2 = model(src, trg2, trg_mask=causal_mask)

    # Positions 0, 1, 2 must produce IDENTICAL output vectors!
    diff = (out1[:, :3, :] - out2[:, :3, :]).abs().max().item()
    assert diff < 1e-4, f"Causal violation! Modifying future tokens changed past logits by {diff:.6e}"

    print(f"  [PASS] Causal masking verified: zero future information leakage (diff: {diff:.2e}).")


def test_transformer_forward_and_backward():
    print("Testing Transformer Forward and Backward Pass...")
    B, T_src, T_trg = 2, 8, 7
    src_vocab = 50
    trg_vocab = 50
    d_model = 64
    num_heads = 4

    model = Transformer(
        src_vocab_size=src_vocab,
        trg_vocab_size=trg_vocab,
        d_model=d_model,
        num_heads=num_heads,
        num_encoder_layers=2,
        num_decoder_layers=2,
        d_ff=128,
        dropout=0.1,
    )
    model.train()

    src = torch.randint(1, src_vocab, (B, T_src))
    trg = torch.randint(1, trg_vocab, (B, T_trg))
    trg_mask = Transformer.generate_square_subsequent_mask(T_trg)

    logits = model(src, trg, trg_mask=trg_mask)
    assert logits.shape == (B, T_trg, trg_vocab), \
        f"Expected logits shape {(B, T_trg, trg_vocab)}, got {logits.shape}"

    targets = torch.randint(0, trg_vocab, (B, T_trg))
    loss = nn.CrossEntropyLoss()(logits.reshape(-1, trg_vocab), targets.reshape(-1))
    loss.backward()

    for name, param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"Parameter {name} has no gradient!"
            assert not torch.isnan(param.grad).any(), f"Parameter {name} gradient contains NaN!"
            assert not torch.isinf(param.grad).any(), f"Parameter {name} gradient contains Inf!"

    print("  [PASS] Transformer forward/backward pass & gradient audit clean.")


def test_single_batch_overfit():
    print("Testing Single Batch Overfitting (Memorization Sanity)...")
    torch.manual_seed(42)
    B = 2
    T_src = 6
    T_trg = 6
    vocab_size = 25
    d_model = 64

    model = Transformer(
        src_vocab_size=vocab_size,
        trg_vocab_size=vocab_size,
        d_model=d_model,
        num_heads=4,
        num_encoder_layers=2,
        num_decoder_layers=2,
        d_ff=128,
        dropout=0.0,
    )
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    # Toy task: reverse sequence
    src = torch.randint(1, vocab_size, (B, T_src))
    trg_in = torch.cat([torch.zeros(B, 1, dtype=torch.long), torch.flip(src, dims=[1])[:, :-1]], dim=1)
    trg_target = torch.flip(src, dims=[1])
    trg_mask = Transformer.generate_square_subsequent_mask(T_trg)

    initial_loss = None
    final_loss = None

    for epoch in range(120):
        optimizer.zero_grad()
        logits = model(src, trg_in, trg_mask=trg_mask)
        loss = criterion(logits.reshape(-1, vocab_size), trg_target.reshape(-1))
        loss.backward()
        optimizer.step()

        if epoch == 0:
            initial_loss = loss.item()
        final_loss = loss.item()
        if final_loss < 0.05:
            break

    print(f"  Single batch loss: {initial_loss:.4f} -> {final_loss:.4f} in {epoch+1} steps")
    assert final_loss < 0.1, f"Failed to overfit single batch! Final loss was {final_loss:.4f}"
    print("  [PASS] Single batch overfit successfully converged.")


if __name__ == "__main__":
    print(f"Running unit tests for: {MODULE_NAME}\n" + "=" * 50)
    test_scaled_dot_product_attention()
    test_multi_head_attention()
    test_positional_encoding()
    test_causal_masking_autoregressive_property()
    test_transformer_forward_and_backward()
    test_single_batch_overfit()
    print("=" * 50 + "\nALL TESTS PASSED! [PASS]")
