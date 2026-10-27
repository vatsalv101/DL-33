import sys
import importlib
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim

MODULE_NAME = sys.argv[1] if len(sys.argv) > 1 else "papers.12_rope.model"

try:
    mod = importlib.import_module(MODULE_NAME)
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import model as mod

rotate_half = mod.rotate_half
apply_rotary_pos_emb = mod.apply_rotary_pos_emb
RotaryPositionalEmbedding = mod.RotaryPositionalEmbedding
RoPEMultiHeadAttention = mod.RoPEMultiHeadAttention
PositionwiseFeedForward = mod.PositionwiseFeedForward
RoFormerEncoderLayer = mod.RoFormerEncoderLayer
RoFormerEncoder = mod.RoFormerEncoder


def test_rotate_half():
    print("Testing rotate_half...")
    x = torch.tensor([[1.0, 2.0, 3.0, 4.0]])  # d = 4, x1=[1,2], x2=[3,4]
    rot = rotate_half(x)
    expected = torch.tensor([[-3.0, -4.0, 1.0, 2.0]])
    assert torch.allclose(rot, expected), f"Expected {expected}, got {rot}"

    # 4 consecutive 90-degree rotations must equal identity: R^4 = I
    rot4 = rotate_half(rotate_half(rotate_half(rotate_half(x))))
    assert torch.allclose(rot4, x), "Four rotations of 90 degrees must equal identity!"
    print("  [PASS] rotate_half and 4-fold cyclic rotation verified.")


def test_rotary_embedding_shapes():
    print("Testing RotaryPositionalEmbedding shapes & dynamic caching...")
    d_k = 32
    rope = RotaryPositionalEmbedding(dim=d_k, max_seq_len=64)
    cos, sin = rope(seq_len=50)

    assert cos.shape == (1, 1, 50, d_k), f"Expected cos shape (1, 1, 50, {d_k}), got {cos.shape}"
    assert sin.shape == (1, 1, 50, d_k), f"Expected sin shape (1, 1, 50, {d_k}), got {sin.shape}"

    # Test dynamic cache extension beyond initial max_seq_len (64 -> 128)
    cos_ext, sin_ext = rope(seq_len=128)
    assert cos_ext.shape == (1, 1, 128, d_k), f"Dynamic cache failed, got {cos_ext.shape}"
    print("  [PASS] RotaryPositionalEmbedding shape contracts and dynamic caching verified.")


def test_relative_position_invariance():
    print("Testing Relative Position Invariance Property of RoPE...")
    # Fundamental Property: <RoPE(q, m), RoPE(k, n)> == <RoPE(q, m + s), RoPE(k, n + s)>
    torch.manual_seed(42)
    d_k = 64
    rope = RotaryPositionalEmbedding(dim=d_k, max_seq_len=512)

    # Let m = 10, n = 15 (relative offset = -5)
    # Let shift s = 25 -> m+s = 35, n+s = 40
    m, n, s = 10, 15, 25
    max_idx = max(m + s, n + s) + 1
    cos, sin = rope(seq_len=max_idx)

    q = torch.randn(1, 1, 1, d_k)
    k = torch.randn(1, 1, 1, d_k)

    q_m = apply_rotary_pos_emb(q, cos[:, :, m:m+1, :], sin[:, :, m:m+1, :])
    k_n = apply_rotary_pos_emb(k, cos[:, :, n:n+1, :], sin[:, :, n:n+1, :])
    dot_mn = (q_m * k_n).sum().item()

    q_ms = apply_rotary_pos_emb(q, cos[:, :, m+s:m+s+1, :], sin[:, :, m+s:m+s+1, :])
    k_ns = apply_rotary_pos_emb(k, cos[:, :, n+s:n+s+1, :], sin[:, :, n+s:n+s+1, :])
    dot_shifted = (q_ms * k_ns).sum().item()

    diff = abs(dot_mn - dot_shifted)
    assert diff < 1e-5, f"Relative position invariance violated! diff={diff:.6e}"
    print(f"  [PASS] Relative shift invariance holds: <q_m, k_n> == <q_(m+s), k_(n+s)> (diff: {diff:.2e}).")


def test_long_term_decay():
    print("Testing Natural Long-Term Attention Decay...")
    # Section 3.4.3 of Su et al. (2021): For identical/correlated representations,
    # <RoPE(q, m), RoPE(q, n)> = sum_i cos((m - n) * theta_i), which naturally decays as |m - n| increases.
    d_k = 64
    seq_len = 128
    rope = RotaryPositionalEmbedding(dim=d_k, max_seq_len=seq_len)
    cos, sin = rope(seq_len=seq_len)

    # Identical base query/key representation across sequence
    q = torch.ones(1, 1, seq_len, d_k)
    q_rot = apply_rotary_pos_emb(q, cos, sin)

    scores = torch.matmul(q_rot, q_rot.transpose(-2, -1)).squeeze().detach()

    # Compare correlation at short distances (offset 1 to 5) vs long distances (offset 50 to 80)
    diag_means = []
    for offset in range(1, 80):
        diag_means.append(torch.diagonal(scores, offset=offset).mean().item())

    short_dist_corr = sum(diag_means[:5]) / 5
    long_dist_corr = sum(diag_means[50:80]) / 30

    print(f"  Short-range correlation (1-5): {short_dist_corr:.4f} | Long-range correlation (50-80): {long_dist_corr:.4f}")
    assert short_dist_corr > long_dist_corr, f"Expected short-range correlation > long-range, got {short_dist_corr} vs {long_dist_corr}"
    print("  [PASS] Natural long-range attention decay confirmed.")


def test_rope_mha_shapes_and_masking():
    print("Testing RoPEMultiHeadAttention shapes and masking...")
    d_model = 64
    num_heads = 4
    mha = RoPEMultiHeadAttention(d_model=d_model, num_heads=num_heads, dropout=0.0)
    mha.eval()

    B, T = 2, 16
    x = torch.randn(B, T, d_model)

    # 1. Unmasked
    out, weights = mha(x)
    assert out.shape == (B, T, d_model), f"Expected out {(B, T, d_model)}, got {out.shape}"
    assert weights.shape == (B, num_heads, T, T), f"Expected weights {(B, num_heads, T, T)}, got {weights.shape}"

    # 2. Causal masking
    causal_mask = torch.tril(torch.ones(T, T, dtype=torch.bool))
    out_c, weights_c = mha(x, mask=causal_mask)
    upper_tri_max = torch.triu(weights_c, diagonal=1).max().item()
    assert upper_tri_max < 1e-4, f"Causal mask leak in RoPE MHA! Max upper tri value: {upper_tri_max}"

    print("  [PASS] RoPEMultiHeadAttention forward pass and causal mask verified.")


def test_roformer_encoder_gradient_flow():
    print("Testing RoFormerEncoder Forward & Backward Gradient Flow...")
    vocab_size = 50
    d_model = 64
    num_heads = 4
    num_layers = 2
    d_ff = 128

    model = RoFormerEncoder(
        vocab_size=vocab_size,
        d_model=d_model,
        num_layers=num_layers,
        num_heads=num_heads,
        d_ff=d_ff,
        dropout=0.1,
    )
    model.train()

    B, T = 2, 12
    input_ids = torch.randint(0, vocab_size, (B, T))
    out = model(input_ids)

    assert out.shape == (B, T, d_model), f"Expected output {(B, T, d_model)}, got {out.shape}"

    loss = out.sum()
    loss.backward()

    for name, param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"Parameter {name} has no gradient!"
            assert not torch.isnan(param.grad).any(), f"Parameter {name} has NaN gradients!"
            assert not torch.isinf(param.grad).any(), f"Parameter {name} has Inf gradients!"

    print("  [PASS] RoFormerEncoder gradient audit clean (no NaN/Inf).")


def test_single_batch_overfit():
    print("Testing Single Batch Overfit (Sequence Reconstruction)...")
    torch.manual_seed(42)
    B, T, vocab_size, d_model = 2, 8, 20, 64

    model = RoFormerEncoder(
        vocab_size=vocab_size,
        d_model=d_model,
        num_layers=2,
        num_heads=4,
        d_ff=128,
        dropout=0.0,
    )
    head = nn.Linear(d_model, vocab_size)
    optimizer = optim.Adam(list(model.parameters()) + list(head.parameters()), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    input_ids = torch.randint(1, vocab_size, (B, T))
    targets = input_ids.clone()

    initial_loss = None
    final_loss = None

    for epoch in range(60):
        optimizer.zero_grad()
        hidden = model(input_ids)
        logits = head(hidden)
        loss = criterion(logits.view(-1, vocab_size), targets.view(-1))
        loss.backward()
        optimizer.step()

        if epoch == 0:
            initial_loss = loss.item()
        final_loss = loss.item()
        if final_loss < 0.05:
            break

    print(f"  Loss: {initial_loss:.4f} -> {final_loss:.4f} in {epoch+1} steps")
    assert final_loss < 0.1, f"Failed to overfit single batch! Final loss was {final_loss:.4f}"
    print("  [PASS] Single batch overfit successfully converged.")


if __name__ == "__main__":
    print(f"Running unit tests for: {MODULE_NAME}\n" + "=" * 50)
    test_rotate_half()
    test_rotary_embedding_shapes()
    test_relative_position_invariance()
    test_long_term_decay()
    test_rope_mha_shapes_and_masking()
    test_roformer_encoder_gradient_flow()
    test_single_batch_overfit()
    print("=" * 50 + "\nALL TESTS PASSED! [PASS]")
