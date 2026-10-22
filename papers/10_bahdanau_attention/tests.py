"""Paper 10: Neural Machine Translation by Jointly Learning to Align and Translate (Bahdanau Attention)
Unit Tests Suite.
"""

import torch
import torch.nn as nn


def test_bahdanau_attention_shapes_and_properties(attn_cls):
    """Tests Bahdanau Attention shapes and confirms attention weights sum to 1.0."""
    all_ok = True
    torch.manual_seed(42)

    B, src_len = 4, 12
    enc_hidden_dim = 64
    dec_hidden_dim = 64
    attn_dim = 32

    attn = attn_cls(enc_hidden_dim, dec_hidden_dim, attn_dim)
    attn.eval()

    hidden = torch.randn(B, dec_hidden_dim)
    encoder_outputs = torch.randn(B, src_len, enc_hidden_dim * 2)

    with torch.no_grad():
        context, attn_weights = attn(hidden, encoder_outputs)

    # 1. Shape Checks
    if context.shape != (B, enc_hidden_dim * 2):
        print(f"  [FAIL] Context shape mismatch: expected ({B}, {enc_hidden_dim * 2}), got {context.shape}")
        all_ok = False

    if attn_weights.shape != (B, src_len):
        print(f"  [FAIL] Attention weights shape mismatch: expected ({B}, {src_len}), got {attn_weights.shape}")
        all_ok = False

    # 2. Probability distribution check (sum to 1.0 across src_len)
    sums = attn_weights.sum(dim=1)
    ones = torch.ones_like(sums)
    if not torch.allclose(sums, ones, atol=1e-4):
        print("  [FAIL] Attention weights do NOT sum to 1.0 across source positions!")
        all_ok = False
    else:
        print("  [PASS] Bahdanau Attention produces valid probability distribution (sum = 1.0) and correct context shape!")

    return all_ok


def test_encoder_shapes(encoder_cls):
    """Tests bidirectional encoder output and initial decoder hidden state projection."""
    all_ok = True
    torch.manual_seed(42)

    B, src_len = 4, 15
    input_dim = 100
    emb_dim = 32
    enc_hidden_dim = 64
    dec_hidden_dim = 64

    encoder = encoder_cls(input_dim, emb_dim, enc_hidden_dim, dec_hidden_dim)
    encoder.eval()

    src = torch.randint(0, input_dim, (B, src_len))
    with torch.no_grad():
        outputs, s_0 = encoder(src)

    if outputs.shape != (B, src_len, enc_hidden_dim * 2):
        print(f"  [FAIL] Encoder outputs shape mismatch: expected ({B}, {src_len}, {enc_hidden_dim * 2}), got {outputs.shape}")
        all_ok = False

    if s_0.shape != (B, dec_hidden_dim):
        print(f"  [FAIL] Initial decoder state shape mismatch: expected ({B}, {dec_hidden_dim}), got {s_0.shape}")
        all_ok = False

    if all_ok:
        print("  [PASS] Bidirectional Encoder outputs correctly shaped annotations and initial decoder hidden state!")

    return all_ok


def test_decoder_step(decoder_cls, attn_cls):
    """Tests single decoding step with attention context integration."""
    all_ok = True
    torch.manual_seed(42)

    B, src_len = 4, 10
    output_dim = 80
    emb_dim = 32
    enc_hidden_dim = 64
    dec_hidden_dim = 64
    attn_dim = 32

    attn = attn_cls(enc_hidden_dim, dec_hidden_dim, attn_dim)
    decoder = decoder_cls(output_dim, emb_dim, enc_hidden_dim, dec_hidden_dim, dropout=0.1, attention=attn)
    decoder.eval()

    input_token = torch.randint(0, output_dim, (B,))
    hidden = torch.randn(B, dec_hidden_dim)
    encoder_outputs = torch.randn(B, src_len, enc_hidden_dim * 2)

    with torch.no_grad():
        pred, new_hidden, attn_weights = decoder(input_token, hidden, encoder_outputs)

    if pred.shape != (B, output_dim):
        print(f"  [FAIL] Decoder prediction shape mismatch: expected ({B}, {output_dim}), got {pred.shape}")
        all_ok = False

    if new_hidden.shape != (B, dec_hidden_dim):
        print(f"  [FAIL] Decoder new_hidden shape mismatch: expected ({B}, {dec_hidden_dim}), got {new_hidden.shape}")
        all_ok = False

    if attn_weights.shape != (B, src_len):
        print(f"  [FAIL] Decoder attn_weights shape mismatch: expected ({B}, {src_len}), got {attn_weights.shape}")
        all_ok = False

    if all_ok:
        print("  [PASS] Decoder step integrates attention context and produces vocabulary prediction logits!")

    return all_ok


def test_seq2seq_forward_pass(seq2seq_cls, encoder_cls, decoder_cls, attn_cls):
    """Validates full end-to-end Seq2Seq sequence translation forward pass."""
    all_ok = True
    torch.manual_seed(42)

    B, src_len, trg_len = 2, 8, 10
    src_vocab, trg_vocab = 50, 60
    emb_dim = 32
    enc_hidden_dim = 64
    dec_hidden_dim = 64
    attn_dim = 32

    attn = attn_cls(enc_hidden_dim, dec_hidden_dim, attn_dim)
    enc = encoder_cls(src_vocab, emb_dim, enc_hidden_dim, dec_hidden_dim)
    dec = decoder_cls(trg_vocab, emb_dim, enc_hidden_dim, dec_hidden_dim, dropout=0.1, attention=attn)
    model = seq2seq_cls(enc, dec)
    model.eval()

    src = torch.randint(0, src_vocab, (B, src_len))
    trg = torch.randint(0, trg_vocab, (B, trg_len))

    with torch.no_grad():
        outputs, attentions = model(src, trg, teacher_forcing_ratio=0.0)

    if outputs.shape != (B, trg_len, trg_vocab):
        print(f"  [FAIL] Seq2Seq outputs shape mismatch: expected ({B}, {trg_len}, {trg_vocab}), got {outputs.shape}")
        all_ok = False

    if attentions.shape != (B, trg_len, src_len):
        print(f"  [FAIL] Seq2Seq attentions shape mismatch: expected ({B}, {trg_len}, {src_len}), got {attentions.shape}")
        all_ok = False

    if all_ok:
        print("  [PASS] Full Seq2Seq model correctly tracks sequence decoding steps and attention heatmaps!")

    return all_ok


def test_seq2seq_overfit_toy_translation(seq2seq_cls, encoder_cls, decoder_cls, attn_cls):
    """Sanity check: confirms Seq2Seq with Bahdanau attention memorizes an inverse copy sequence."""
    torch.manual_seed(42)

    src_vocab = 20
    trg_vocab = 20
    emb_dim = 32
    enc_hidden_dim = 64
    dec_hidden_dim = 64
    attn_dim = 32

    attn = attn_cls(enc_hidden_dim, dec_hidden_dim, attn_dim)
    enc = encoder_cls(src_vocab, emb_dim, enc_hidden_dim, dec_hidden_dim, dropout=0.0)
    dec = decoder_cls(trg_vocab, emb_dim, enc_hidden_dim, dec_hidden_dim, dropout=0.0, attention=attn)
    model = seq2seq_cls(enc, dec)
    model.train()

    from core.optimizers import AdamW

    optimizer = AdamW(model.parameters(), lr=0.02)
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    # Toy sentence: reverse sequence mapping
    # src: [1, 5, 8, 12, 17, 2]
    # trg: [1, 17, 12, 8, 5, 2]
    src = torch.tensor([[1, 5, 8, 12, 17, 2], [1, 3, 6, 9, 14, 2]])
    trg = torch.tensor([[1, 17, 12, 8, 5, 2], [1, 14, 9, 6, 3, 2]])

    initial_loss = None
    for step in range(40):
        optimizer.zero_grad()
        output, _ = model(src, trg, teacher_forcing_ratio=0.5)

        # Output shape: (B, trg_len, trg_vocab)
        # Exclude the <sos> token at index 0 for loss calculation
        output_dim = output.shape[-1]
        loss = criterion(output[:, 1:].reshape(-1, output_dim), trg[:, 1:].reshape(-1))
        loss.backward()
        optimizer.step()

        if step == 0:
            initial_loss = loss.item()

    final_loss = loss.item()
    if final_loss > 0.05:
        print(f"  [FAIL] Seq2Seq overfit did not reach loss < 0.05 (init: {initial_loss:.4f}, final: {final_loss:.4f})")
        return False

    print(f"  [PASS] Single-batch Seq2Seq translation overfit passed (loss: {initial_loss:.4f} -> {final_loss:.6f})!")
    return True


def run_all_tests():
    """Runs all Paper 10 verification gates."""
    print("=" * 60)
    print("Running Paper 10 (Bahdanau Attention) Test Suite")
    print("=" * 60)

    try:
        from model import BahdanauAttention, Encoder, Decoder, Seq2Seq
    except ImportError:
        import sys
        from pathlib import Path

        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from model import BahdanauAttention, Encoder, Decoder, Seq2Seq

    try:
        ok1 = test_bahdanau_attention_shapes_and_properties(BahdanauAttention)
        ok2 = test_encoder_shapes(Encoder)
        ok3 = test_decoder_step(Decoder, BahdanauAttention)
        ok4 = test_seq2seq_forward_pass(Seq2Seq, Encoder, Decoder, BahdanauAttention)
        ok5 = test_seq2seq_overfit_toy_translation(Seq2Seq, Encoder, Decoder, BahdanauAttention)

        if all([ok1, ok2, ok3, ok4, ok5]):
            print("\n" + "=" * 60)
            print("ALL PAPER 10 TESTS PASSED! YOU INVENTED ATTENTION!")
            print("=" * 60)
    except NotImplementedError:
        print("\n[PENDING] Implement BahdanauAttention, Encoder, Decoder, and Seq2Seq in 'papers/10_bahdanau_attention/model.py' to pass tests.")


if __name__ == "__main__":
    run_all_tests()
