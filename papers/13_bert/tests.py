import sys
import importlib
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim

MODULE_NAME = sys.argv[1] if len(sys.argv) > 1 else "papers.13_bert.model"

try:
    mod = importlib.import_module(MODULE_NAME)
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import model as mod

BERTEmbedding = mod.BERTEmbedding
BERTSelfAttention = mod.BERTSelfAttention
BERTSelfOutput = mod.BERTSelfOutput
BERTAttention = mod.BERTAttention
BERTIntermediate = mod.BERTIntermediate
BERTOutput = mod.BERTOutput
BERTLayer = mod.BERTLayer
BERTEncoder = mod.BERTEncoder
BERTPooler = mod.BERTPooler
BERTModel = mod.BERTModel
BERTLMPredictionHead = mod.BERTLMPredictionHead
BERTForPreTraining = mod.BERTForPreTraining


def test_bert_embedding():
    print("Testing BERTEmbedding...")
    vocab_size = 100
    hidden_size = 64
    max_len = 128
    emb = BERTEmbedding(vocab_size=vocab_size, hidden_size=hidden_size, max_position_embeddings=max_len, hidden_dropout_prob=0.0)
    emb.eval()

    B, T = 2, 10
    input_ids = torch.randint(0, vocab_size, (B, T))
    token_type_ids = torch.zeros(B, T, dtype=torch.long)
    token_type_ids[:, 5:] = 1  # Sentence B

    out1 = emb(input_ids, token_type_ids=token_type_ids)
    assert out1.shape == (B, T, hidden_size), f"Expected shape {(B, T, hidden_size)}, got {out1.shape}"

    # Verify that changing token_type_ids changes representation
    token_type_ids_alt = torch.zeros_like(token_type_ids)
    out2 = emb(input_ids, token_type_ids=token_type_ids_alt)
    assert not torch.allclose(out1[:, 5:, :], out2[:, 5:, :]), "Segment embeddings must alter token representations!"

    print("  [PASS] BERTEmbedding combined token + position + segment sum verified.")


def test_bert_attention_bidirectional():
    print("Testing BERT Attention Bidirectional Context...")
    hidden_size = 64
    num_heads = 4
    attn = BERTSelfAttention(hidden_size=hidden_size, num_attention_heads=num_heads, attention_probs_dropout_prob=0.0)
    attn.eval()

    B, T = 1, 6
    x1 = torch.randn(B, T, hidden_size)
    x2 = x1.clone()
    # Modify future token at step 5
    x2[:, 5, :] = torch.randn(hidden_size)

    out1, probs1 = attn(x1)
    out2, probs2 = attn(x2)

    # In BERT, unlike causal GPT, altering step 5 MUST change step 0 representation (Bidirectionality)
    diff_at_step_0 = (out1[:, 0, :] - out2[:, 0, :]).abs().max().item()
    assert diff_at_step_0 > 1e-4, "BERT attention must be bidirectional! Future tokens must affect earlier tokens."

    # Test padding mask: position 4 and 5 masked out
    additive_mask = torch.zeros(1, 1, 1, T)
    additive_mask[:, :, :, -2:] = -10000.0
    _, probs_masked = attn(x1, attention_mask=additive_mask)
    assert (probs_masked[:, :, :, -2:] < 1e-4).all(), "Masked tokens should have ~0 attention probability!"

    print(f"  [PASS] Bidirectional attention verified (diff at step 0: {diff_at_step_0:.4f}) and padding mask clean.")


def test_bert_pooler():
    print("Testing BERTPooler...")
    hidden_size = 64
    pooler = BERTPooler(hidden_size=hidden_size)
    pooler.eval()

    B, T = 2, 8
    hidden_states = torch.randn(B, T, hidden_size)
    pooled = pooler(hidden_states)

    assert pooled.shape == (B, hidden_size), f"Expected {(B, hidden_size)}, got {pooled.shape}"
    assert (pooled >= -1.0).all() and (pooled <= 1.0).all(), "Pooled output must be bounded in [-1, 1] by Tanh!"
    print("  [PASS] BERTPooler [CLS] extraction and Tanh projection verified.")


def test_bert_model_forward():
    print("Testing BERTModel forward pass...")
    vocab_size = 100
    hidden_size = 64
    model = BERTModel(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=128,
        hidden_dropout_prob=0.0,
    )
    model.eval()

    B, T = 2, 12
    input_ids = torch.randint(0, vocab_size, (B, T))
    attention_mask = torch.ones(B, T)
    attention_mask[:, -3:] = 0  # 3 padding tokens

    seq_out, pooled_out = model(input_ids, attention_mask=attention_mask)
    assert seq_out.shape == (B, T, hidden_size), f"Expected seq_out {(B, T, hidden_size)}, got {seq_out.shape}"
    assert pooled_out.shape == (B, hidden_size), f"Expected pooled_out {(B, hidden_size)}, got {pooled_out.shape}"
    print("  [PASS] BERTModel sequence_output and pooled_output shapes verified.")


def test_bert_for_pretraining_heads():
    print("Testing BERTForPreTraining heads and gradient flow...")
    vocab_size = 100
    hidden_size = 64
    model = BERTForPreTraining(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=128,
        hidden_dropout_prob=0.1,
    )
    model.train()

    B, T = 2, 8
    input_ids = torch.randint(0, vocab_size, (B, T))
    token_type_ids = torch.zeros(B, T, dtype=torch.long)
    token_type_ids[:, 4:] = 1

    mlm_logits, nsp_logits = model(input_ids, token_type_ids=token_type_ids)
    assert mlm_logits.shape == (B, T, vocab_size), f"Expected mlm_logits {(B, T, vocab_size)}, got {mlm_logits.shape}"
    assert nsp_logits.shape == (B, 2), f"Expected nsp_logits {(B, 2)}, got {nsp_logits.shape}"

    loss = mlm_logits.sum() + nsp_logits.sum()
    loss.backward()

    for name, param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"Parameter {name} has no gradient!"
            assert not torch.isnan(param.grad).any(), f"Parameter {name} gradient contains NaN!"
            assert not torch.isinf(param.grad).any(), f"Parameter {name} gradient contains Inf!"

    print("  [PASS] Pre-training MLM and NSP output shapes & gradient flow verified.")


def test_single_batch_overfit_mlm_nsp():
    print("Testing Single-Batch Overfit (Joint MLM + NSP)...")
    torch.manual_seed(42)
    vocab_size = 30
    hidden_size = 64
    B, T = 2, 8

    model = BERTForPreTraining(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=128,
        hidden_dropout_prob=0.0,
    )
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    loss_fn_mlm = nn.CrossEntropyLoss()
    loss_fn_nsp = nn.CrossEntropyLoss()

    input_ids = torch.randint(2, vocab_size, (B, T))
    # Insert special tokens: [CLS]=1 at idx 0, [SEP]=2 at idx 4 and idx 7
    input_ids[:, 0] = 1
    input_ids[:, 4] = 2
    input_ids[:, 7] = 2

    token_type_ids = torch.zeros(B, T, dtype=torch.long)
    token_type_ids[:, 5:] = 1

    # Mask token at index 2 (replace with mask_id = 0)
    mlm_target = input_ids.clone()
    input_ids[:, 2] = 0

    nsp_target = torch.tensor([1, 0])  # IsNext, NotNext

    initial_loss = None
    final_loss = None

    for epoch in range(60):
        optimizer.zero_grad()
        mlm_logits, nsp_logits = model(input_ids, token_type_ids=token_type_ids)
        loss_mlm = loss_fn_mlm(mlm_logits.view(-1, vocab_size), mlm_target.view(-1))
        loss_nsp = loss_fn_nsp(nsp_logits, nsp_target)
        loss = loss_mlm + loss_nsp
        loss.backward()
        optimizer.step()

        if epoch == 0:
            initial_loss = loss.item()
        final_loss = loss.item()
        if final_loss < 0.05:
            break

    print(f"  Loss: {initial_loss:.4f} -> {final_loss:.4f} in {epoch+1} steps")
    assert final_loss < 0.1, f"Failed to overfit joint MLM+NSP! Final loss: {final_loss:.4f}"
    print("  [PASS] Single batch overfit successfully converged.")


if __name__ == "__main__":
    print(f"Running unit tests for: {MODULE_NAME}\n" + "=" * 50)
    test_bert_embedding()
    test_bert_attention_bidirectional()
    test_bert_pooler()
    test_bert_model_forward()
    test_bert_for_pretraining_heads()
    test_single_batch_overfit_mlm_nsp()
    print("=" * 50 + "\nALL TESTS PASSED! [PASS]")
