# Paper #13: BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding

> **Devlin, J., Chang, M.-W., Lee, K., & Toutanova, K. (2018 / NAACL 2019)**  
> *Google AI Language*  
> [arXiv:1810.04805](https://arxiv.org/abs/1810.04805)

---

## 1. Historical Breaking Point: The Unidirectional Dilemma

In 2018, pre-trained language models were divided into two flawed paradigms:
1. **Strictly Left-to-Right (Autoregressive) Transformers (OpenAI GPT-1)**:  
   Trained with causal masking where token $i$ can only attend to previous tokens $1 \dots i-1$. While optimal for text generation, this is fundamentally sub-optimal for sentence-level understanding, question answering, and token classification (e.g. Named Entity Recognition), where meaning depends heavily on **both** left and right context.
2. **Shallow Bidirectional Concatenation (ELMo)**:  
   Trained two separate independent LSTMs (one left-to-right and one right-to-left) and concatenated their final representations: $[h_{\text{forward}}; h_{\text{backward}}]$. However, shallow concatenation is fundamentally weaker than a **deep bidirectional** Transformer where every self-attention layer jointly cross-references tokens from both directions simultaneously.

### The Fundamental Obstacle
Why couldn't standard Transformers be trained bidirectionally before BERT?  
If a multi-layer Transformer allows bidirectional self-attention, tokens in higher layers would trivially "see themselves" in the lower layers. In standard autoregressive pre-training, predicting $x_t$ given all tokens $(x_1, \dots, x_T)$ results in degenerate identity shortcut learning ($\hat{x}_t = x_t$)!

---

## 2. The Thought Experiment: The Cloze Task & Masked Language Modeling

Devlin et al. resolved the identity leak by adapting the **Cloze procedure** (Taylor, 1953):

Instead of predicting the next token, **corrupt a fraction (15%) of the input tokens at random**, and train the deep bidirectional Transformer to reconstruct only the corrupted tokens:

### The 80-10-10 Masking Recipe
For the $15\%$ of tokens selected for prediction:
- **80% of the time**: Replace with the special token `[MASK]`.
- **10% of the time**: Replace with a random vocabulary token (forces the encoder to maintain contextual representations for every token, preventing it from relying solely on `[MASK]`).
- **10% of the time**: Leave the token unchanged (biases representation toward the actual observed token).

### Next Sentence Prediction (NSP)
Many downstream NLP tasks (QA, Natural Language Inference) depend on understanding the relationship between two sentences. BERT introduces paired sentence pre-training:
- Input format: `[CLS] Sentence A [SEP] Sentence B [SEP]`
- 50% of the time: $B$ is the actual next sentence (`IsNext = 1`).
- 50% of the time: $B$ is a random sentence from the corpus (`NotNext = 0`).
- The `[CLS]` token representation is pooled via $\tanh(W_{\text{pool}} h_{\text{CLS}} + b)$ to classify sentence adjacency.

---

## 3. Mathematical Formulation & Architecture

### 3.1 Tripartite Embeddings
Input embeddings are the sum of three distinct learned representations:
$$\mathbf{E}_i = \mathbf{E}_{\text{token}}(w_i) + \mathbf{E}_{\text{pos}}(i) + \mathbf{E}_{\text{segment}}(s_i)$$
followed by Layer Normalization and Dropout:
$$\mathbf{h}_i^{(0)} = \text{LayerNorm}(\mathbf{E}_i)$$

- $\mathbf{E}_{\text{token}} \in \mathbb{R}^{V \times H}$: Standard WordPiece token embeddings.
- $\mathbf{E}_{\text{pos}} \in \mathbb{R}^{T_{\max} \times H}$: Learned absolute positional embeddings.
- $\mathbf{E}_{\text{segment}} \in \mathbb{R}^{2 \times H}$: Segment indicator (0 for sentence A, 1 for sentence B).

### 3.2 Bidirectional Transformer Layer
Each layer $l \in [1, L]$ executes:
$$\hat{\mathbf{h}}^{(l)} = \text{LayerNorm}\left(\mathbf{h}^{(l-1)} + \text{Dropout}(\text{MultiHeadAttention}(\mathbf{h}^{(l-1)}))\right)$$
$$\mathbf{h}^{(l)} = \text{LayerNorm}\left(\hat{\mathbf{h}}^{(l)} + \text{Dropout}(\text{FFN}(\hat{\mathbf{h}}^{(l)}))\right)$$
where $\text{FFN}(x) = \text{GELU}(x W_1 + b_1) W_2 + b_2$, with $W_1 \in \mathbb{R}^{H \times 4H}$ and $W_2 \in \mathbb{R}^{4H \times H}$.

### 3.3 The Gaussian Error Linear Unit (GELU)
BERT replaces standard ReLU with the smooth probabilistic activation GELU (Hendrycks & Gimpel, 2016):
$$\text{GELU}(x) = x \Phi(x) = x P(X \le x) \approx 0.5 x \left(1 + \tanh\left(\sqrt{\frac{2}{\pi}} (x + 0.044715 x^3)\right)\right)$$

### 3.4 Pre-training Heads
1. **MLM Head**:  
   Projects final hidden state of masked tokens through a transformation with tied input embeddings:
   $$\mathbf{y}_{\text{MLM}} = \text{softmax}\left(\text{LayerNorm}(\text{GELU}(\mathbf{h}_i^{(L)} W_{\text{transform}} + b)) W_{\text{token}}^T + b_{\text{vocab}}\right)$$
2. **NSP Head**:  
   Pooled representation of `[CLS]`:
   $$\mathbf{h}_{\text{pooled}} = \tanh(\mathbf{h}_0^{(L)} W_{\text{pool}} + b_{\text{pool}})$$
   $$\mathbf{y}_{\text{NSP}} = \text{softmax}(\mathbf{h}_{\text{pooled}} W_{\text{NSP}} + b_{\text{NSP}}) \in \mathbb{R}^2$$

---

## 4. Tensor Contracts & Architectural Blueprint

```
Tokens:     [CLS]  Token_A1   Token_A2   [SEP]  Token_B1  [SEP]
Segments:     0       0          0         0       1        1
Positions:    0       1          2         3       4        5
              │       │          │         │       │        │
              ▼       ▼          ▼         ▼       ▼        ▼
       [ Token Emb + Position Emb + Segment Emb ] -> LayerNorm
                                │
                 [ Bidirectional Transformer Stack ]
                                │
              ┌─────────────────┴─────────────────┐
              ▼                                   ▼
        [CLS] Pooler (Tanh)                All Token Representations
              │                                   │
      NSP Binary Head                     MLM Vocab Projection Head
```

| Component | Input Shape | Output Shape | Notes |
| :--- | :--- | :--- | :--- |
| `BERTEmbedding` | `input_ids`: `(B, T)`, `token_type_ids`: `(B, T)` | `(B, T, H)` | Token + Pos + Segment + LN + Dropout |
| `BERTAttention` | `(B, T, H)`, `mask`: `(B, 1, 1, T)` | `(B, T, H)` | Bidirectional Self-Attention + Residual + LN |
| `BERTIntermediate` & `BERTOutput` | `(B, T, H)` | `(B, T, H)` | GELU MLP ($H \to 4H \to H$) + Residual + LN |
| `BERTLayer` | `(B, T, H)` | `(B, T, H)` | Full Transformer encoder block |
| `BERTPooler` | `sequence_output`: `(B, T, H)` | `(B, H)` | Extracts `[CLS]` (token 0) $\to$ Dense + Tanh |
| `BERTModel` | `input_ids`, `token_type_ids`, `attention_mask` | `(sequence_output, pooled_output)` | Core trunk |
| `BERTForPreTraining` | `input_ids`, `token_type_ids`, `attention_mask` | `(mlm_logits: (B, T, V), nsp_logits: (B, 2))` | Pre-training heads |

---

## 5. Downstream Unlocks

- **Fine-Tuning Paradigm**: Pioneered the modern pre-train $\to$ fine-tune NLP paradigm across classification, QA, and NER.
- **RoBERTa / DeBERTa**: Refinements in dynamic masking, removing NSP, and disentangled relative attention.
- **Vision Transformers (MAE - Paper #23)**: Masked Autoencoders directly transplant BERT's masked token reconstruction onto visual image patches.
- **Cross-Modal Robotics (RT-1 / OpenVLA)**: Bidirectional language conditioning encoders for robotic task instructions.

## Further Reading

For more, [visit this chat](https://chatgpt.com/c/6ae9a7f5-3c28-83ed-9fad-d503059a6c4e).

