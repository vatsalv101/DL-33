# Paper 10: Neural Machine Translation by Jointly Learning to Align and Translate (Bahdanau Attention)
**Authors**: Dzmitry Bahdanau, Kyunghyun Cho, Yoshua Bengio (2014)  
**Paper Link**: [arXiv:1409.0473](https://arxiv.org/abs/1409.0473)

---

## 1. The Historical Breaking Point (The Fixed-Vector Bottleneck)

In 2014, the standard architecture for sequence-to-sequence translation (Sutskever et al., Cho et al.) was the **Encoder-Decoder RNN**:
1. An **Encoder RNN** reads the entire source sentence token-by-token.
2. The final hidden state $h_{T_x} \in \mathbb{R}^H$ is declared the **single summary vector** $c$ of the entire sentence.
3. A **Decoder RNN** generates the target translated sentence word-by-word, conditioned exclusively on this single vector $c$.

```
               Classical Seq2Seq (The Information Bottleneck)

 Source: "The" ──> "quick" ──> "brown" ──> "fox" ──> [ Bottleneck Vector c ]
                                                              │
 Target: "Le" <──── "renard" <──── "brun" <──── "rapide" <────┘
```

### Why Classical Seq2Seq Hit a Wall:
* **The Information Bottleneck**: Compressing all semantic meaning, word orders, clauses, and subtle grammar of a 40–50 word sentence into a single 512-dimensional vector caused catastrophic information loss.
* **Length Degradation**: For sentences longer than 15–20 words, translation performance (BLEU score) plunged off a cliff!
* **Decayed Memory**: The decoder had no direct connection to the source words. Early source tokens (like the subject of a sentence) had their signals washed away across 30 recurrent time-steps.

---

## 2. The Thought Experiment: "If I Were Inventing It"

Sitting at the blackboard, Bahdanau, Cho, and Bengio asked:
> *"Why are we throwing away all the intermediate encoder hidden states? When a human translator translates word $t$, they don't rely purely on a vague memory of the whole sentence; their eyes look back at the specific source words relevant to word $t$!"*

### The Solution: Keep Every State and Learn an Alignment (Attention!)
1. **Retain All Annotations**: Instead of keeping only the last hidden state, keep **every** hidden state produced by the encoder:
   $$H = [h_1, h_2, \dots, h_{T_x}], \quad h_i \in \mathbb{R}^{2 \cdot H_{\text{enc}}}$$
2. **Dynamic Alignment (Query vs Keys)**:
   At each decoding step $t$, compare the decoder's current state $s_{t-1}$ (the **Query**) with every source hidden state $h_i$ (the **Keys**) to compute an alignment score $e_{t, i}$.
3. **Soft Attention Distribution**:
   Normalize scores via softmax to get attention weights $\alpha_{t, i} \in [0, 1]$:
   $$\alpha_{t, i} = \frac{\exp(e_{t, i})}{\sum_{j=1}^{T_x} \exp(e_{t, j})}$$
4. **Dynamic Context Vector**:
   Compute a tailored context vector $c_t$ as the weighted average of source representations:
   $$c_t = \sum_{i=1}^{T_x} \alpha_{t, i} h_i$$

This was the **genesis of Attention in Artificial Intelligence**, unlocking direct connections between any two tokens in a sequence regardless of distance!

---

## 3. Mathematical Formulation (Additive / Bahdanau Attention)

Given:
* Decoder hidden state at previous step: $s_{t-1} \in \mathbb{R}^{H_{\text{dec}}}$
* Encoder bidirectional annotations: $h_i = [\vec{h}_i; \overleftarrow{h}_i] \in \mathbb{R}^{2 \cdot H_{\text{enc}}}$ for $i \in \{1, \dots, T_x\}$

### 1. The Additive Score Function
Bahdanau designed a small single-layer feedforward network with a non-linear $\tanh$ activation:
$$e_{t, i} = v_a^T \tanh(W_a s_{t-1} + U_a h_i)$$
Where:
* $W_a \in \mathbb{R}^{A \times H_{\text{dec}}}$ projects the decoder query.
* $U_a \in \mathbb{R}^{A \times (2 H_{\text{enc}})}$ projects the encoder keys.
* $v_a \in \mathbb{R}^A$ is a learnable alignment vector mapping to a scalar score.

### 2. Softmax Normalization
$$\alpha_t = \text{softmax}(e_t) \in \mathbb{R}^{T_x}, \quad \sum_{i=1}^{T_x} \alpha_{t, i} = 1$$

### 3. Context Aggregation
$$c_t = \sum_{i=1}^{T_x} \alpha_{t, i} h_i \in \mathbb{R}^{2 H_{\text{enc}}}$$

### 4. Decoder Recurrence & Output Projection
The decoder GRU updates using the previous output embedding $y_{t-1}$ concatenated with context $c_t$:
$$s_t = \text{GRUCell}([y_{t-1}; c_t], s_{t-1})$$
Target vocabulary logits:
$$\hat{y}_t = W_{\text{out}} [s_t; c_t; y_{t-1}] + b_{\text{out}}$$

---

## 4. Tensor Shape Contract

| Tensor | Description | Shape |
| :--- | :--- | :--- |
| `src` | Source input tokens | $(B, T_x)$ |
| `trg` | Target input tokens | $(B, T_y)$ |
| `encoder_outputs` | All bidirectional encoder annotations | $(B, T_x, 2 H_{\text{enc}})$ |
| `hidden` ($s_t$) | Decoder GRU hidden state | $(B, H_{\text{dec}})$ |
| `attn_weights` ($\alpha_t$) | Softmax attention distribution | $(B, T_x)$ |
| `context` ($c_t$) | Dynamically weighted context vector | $(B, 2 H_{\text{enc}})$ |
| `output_logits` | Vocabulary classification logits | $(B, \text{vocab\_size}_{\text{trg}})$ |

---

## 5. Downstream Unlocks in our 80-Paper Roadmap

Bahdanau Attention is the direct evolutionary ancestor of the modern Transformer:
* **#11 Attention Is All You Need (Transformer)**: Replaces additive attention with Scaled Dot-Product Multi-Head Self-Attention ($QK^T / \sqrt{d_k}$) and removes recurrence entirely.
* **#12 RoPE & #14 GPT-2**: Autoregressive causal attention with KV-caching.
* **#17 DETR & #24 CLIP**: Cross-attention aligning vision features with text queries.
* **#52 Decision Transformer**: Conditioning action generation on return-to-go sequences in offline RL.
