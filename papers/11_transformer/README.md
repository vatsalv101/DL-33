# Paper #11: Attention Is All You Need (The Transformer)

> **Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł., & Polosukhin, I. (2017).**  
> *Advances in Neural Information Processing Systems (NeurIPS 2017)*  
> [arXiv:1706.03762](https://arxiv.org/abs/1706.03762)

---

## 1. Historical Breaking Point: The Recurrence Bottleneck

In 2017, the state of the art in sequence transduction (translation, summarization, speech) was dominated by recurrent neural networks (RNNs, LSTMs, GRUs) equipped with encoder-decoder attention (Bahdanau et al., 2014; Luong et al., 2015).

However, recurrence suffers from three fundamental structural pathologies:
1. **The Sequential Execution Bottleneck ($O(T)$ Serial Computation)**:  
   Because hidden state $h_t$ is an explicit function of previous state $h_{t-1}$ ($h_t = f(h_{t-1}, x_t)$), computation along the sequence length $T$ cannot be parallelized across hardware cores (GPUs/TPUs). Batching is possible across sentences, but within a sentence, time steps must execute serially.
2. **Information Attenuation Over Path Lengths**:  
   In an RNN, the maximum path length between token $x_1$ and token $x_T$ is $O(T)$. Even with gating (LSTM forget gates) and skip connections, gradient signals degrade exponentially over long sequences, creating difficult optimization landscapes.
3. **Capacity Bottleneck in Token Mixing**:  
   In RNNs, hidden state representations must simultaneously encode syntax, memory, semantics, and temporal order within a single recurrent state vector.

---

## 2. The Thought Experiment: Pure Content-Addressed Routing

What if we **completely discard recurrence and convolutions**?

Imagine a sequence not as a chronological timeline, but as an **unordered set of tokens situated in a metric space**.
- Each token generates a **Query** ($Q$): *"What kind of information am I looking for?"*
- Each token advertises a **Key** ($K$): *"What information do I contain?"*
- Each token holds a **Value** ($V$): *"What representation should be transmitted if I am selected?"*

Every token computes its affinity with every other token in **a single matrix multiplication** ($O(1)$ sequential operations). Order is restored by injecting explicit coordinate vectors—**Positional Encodings**—directly into the representations.

---

## 3. First-Principles Mathematical Derivation

### 3.1 Scaled Dot-Product Attention
Given matrices $Q \in \mathbb{R}^{T_q \times d_k}$, $K \in \mathbb{R}^{T_k \times d_k}$, and $V \in \mathbb{R}^{T_k \times d_v}$:
$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{Q K^T}{\sqrt{d_k}} + M\right) V$$

#### Why Scale by $\frac{1}{\sqrt{d_k}}$?
Assume query vector components $q_i$ and key vector components $k_i$ are independent identically distributed (i.i.d.) random variables with mean $0$ and variance $1$:
$$\mathbb{E}[q_i] = 0, \quad \text{Var}(q_i) = 1, \qquad \mathbb{E}[k_i] = 0, \quad \text{Var}(k_i) = 1$$
Their dot product is the sum of $d_k$ independent random variables:
$$z = q \cdot k = \sum_{i=1}^{d_k} q_i k_i$$
$$\mathbb{E}[z] = \sum_{i=1}^{d_k} \mathbb{E}[q_i]\mathbb{E}[k_i] = 0$$
$$\text{Var}(z) = \sum_{i=1}^{d_k} \text{Var}(q_i k_i) = \sum_{i=1}^{d_k} \left(\text{Var}(q_i)\text{Var}(k_i) + \text{Var}(q_i)\mathbb{E}[k_i]^2 + \text{Var}(k_i)\mathbb{E}[q_i]^2\right) = d_k \times 1 = d_k$$

As dimension $d_k$ grows large, the variance of the dot product grows linearly with $d_k$, meaning magnitude $|z|$ scales with $\sqrt{d_k}$. When fed into the softmax function:
$$\text{softmax}(z)_i = \frac{e^{z_i}}{\sum_j e^{z_j}}$$
Extreme values cause softmax to saturate (producing outputs near $0$ or $1$) where derivatives vanish:
$$\frac{\partial \text{softmax}(z)_i}{\partial z_j} = \text{softmax}(z)_i(\delta_{ij} - \text{softmax}(z)_j) \approx 0$$
Dividing by $\sqrt{d_k}$ normalizes the variance back to $1$, keeping gradients active throughout training.

---

### 3.2 Multi-Head Attention (MHA)
Instead of performing a single attention function with $d_{model}$-dimensional queries, keys, and values, Multi-Head Attention linearly projects $Q, K, V$ with $h$ different, learned linear projections to $d_k, d_k, d_v$ dimensions respectively:
$$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \dots, \text{head}_h) W^O$$
$$\text{where } \text{head}_i = \text{Attention}(Q W_i^Q, K W_i^K, V W_i^V)$$
where $W_i^Q \in \mathbb{R}^{d_{model} \times d_k}$, $W_i^K \in \mathbb{R}^{d_{model} \times d_k}$, $W_i^V \in \mathbb{R}^{d_{model} \times d_v}$, and $W^O \in \mathbb{R}^{h d_v \times d_{model}}$.

Typically, $d_k = d_v = d_{model} / h$. This allows the model to jointly attend to information from different representation subspaces at different positions without increasing total computational cost.

---

### 3.3 Positional Encoding
Because self-attention contains no recurrence and no convolution, it is **permutation-equivariant**:
$$\text{Attention}(P Q, P K, P V) = P \text{Attention}(Q, K, V)$$
for any permutation matrix $P$.

To give the model sequence awareness, fixed sinusoidal position encodings of dimension $d_{model}$ are added to input embeddings:
$$PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i / d_{model}}}\right)$$
$$PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i / d_{model}}}\right)$$

**Linear Relative Shift Property**:  
For any fixed offset $k$, $PE_{pos + k}$ can be represented as a linear function of $PE_{pos}$. By standard angle addition formulas:
$$\sin(\omega (pos + k)) = \sin(\omega pos)\cos(\omega k) + \cos(\omega pos)\sin(\omega k)$$
$$\cos(\omega (pos + k)) = \cos(\omega pos)\cos(\omega k) - \sin(\omega pos)\sin(\omega k)$$
This allows the attention mechanism to learn relative positional relationships directly through linear projection matrices.

---

### 3.4 Position-Wise Feed-Forward Networks
Each layer in encoder and decoder contains a fully connected feed-forward network applied identically and separately to each position:
$$\text{FFN}(x) = \max(0, x W_1 + b_1) W_2 + b_2$$
where $W_1 \in \mathbb{R}^{d_{model} \times d_{ff}}$ and $W_2 \in \mathbb{R}^{d_{ff} \times d_{model}}$ (typically $d_{ff} = 4 \times d_{model}$).

---

### 3.5 Causal (Autoregressive) Masking
In the decoder, self-attention must prevent tokens from attending to subsequent (future) positions during training:
$$M_{i, j} = \begin{cases} 0 & \text{if } i \ge j \\ -\infty & \text{if } i < j \end{cases}$$
When added to $\frac{QK^T}{\sqrt{d_k}}$, all elements where $j > i$ become $-\infty$, which evaluates to $e^{-\infty} = 0$ after softmax.

---

## 4. Tensor Contracts & Architectural Blueprint

```
Encoder Layer:
  x -> [Self-Attention] -> + -> [LayerNorm] -> [FeedForward] -> + -> [LayerNorm] -> out
  |                        ^                   |                ^
  +------------------------+                   +----------------+

Decoder Layer:
  x -> [Masked Self-Attention] -> + -> [LayerNorm] -> [Cross-Attention] -> + -> [LayerNorm] -> [FeedForward] -> + -> [LayerNorm] -> out
  |                               ^                   |   (from Enc)       ^                    |                ^
  +-------------------------------+                   +--------------------+                    +----------------+
```

| Component | Input Shape | Output Shape | Notes |
| :--- | :--- | :--- | :--- |
| `ScaledDotProductAttention` | $Q, K, V$: `(B, H, T, d_k)` | Output: `(B, H, T, d_k)`, Weights: `(B, H, T, T)` | Mask filled with $-1e9$ |
| `MultiHeadAttention` | $Q, K, V$: `(B, T, d_model)` | Output: `(B, T, d_model)`, Weights: `(B, H, T_q, T_k)` | Projects, splits heads, merges |
| `PositionalEncoding` | `(B, T, d_model)` | `(B, T, d_model)` | Added to embeddings |
| `TransformerEncoderLayer` | `(B, T_src, d_model)` | `(B, T_src, d_model)` | Self-attention + FFN |
| `TransformerDecoderLayer` | `(B, T_trg, d_model)`, `memory` | `(B, T_trg, d_model)` | Causal Self-Attn + Cross-Attn + FFN |
| `Transformer` | `src`: `(B, T_src)`, `trg`: `(B, T_trg)` | `(B, T_trg, trg_vocab_size)` | Full Seq2Seq pipeline |

---

## 5. Downstream Unlocks

Mastering the canonical Transformer unlocks:
- **Paper #12**: RoFormer (RoPE) — replacing absolute sinusoids with rotary complex inner products.
- **Paper #13 & #14**: BERT & GPT-2 — bidirectional encoder and causal autoregressive decoder pre-training.
- **Paper #16**: Vision Transformer (ViT) — treating image patches as token sequences.
- **Paper #68 & #74**: Decision Transformer & Trajectory Transformer — casting reinforcement learning as sequence modeling.
- **Paper #79 & #80**: Octo & OpenVLA — robot manipulation policy transformers conditioned on language and multi-camera visual tokens.
