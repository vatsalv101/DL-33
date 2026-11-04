# Paper #14: GPT-2: Language Models are Unsupervised Multitask Learners

> **Radford, A., Wu, J., Child, R., Luan, D., Amodei, D., & Sutskever, I. (2019)**  
> *OpenAI Technical Report*  
> [OpenAI Paper Link](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)

---

## 1. Historical Breaking Point: The Fine-Tuning Bottleneck & Inference Complexity

Following GPT-1 (Radford et al., 2018) and BERT (Devlin et al., 2018), machine learning was locked into the **pre-train then fine-tune** paradigm:
- A large model was pre-trained on generic text, but required task-specific labeled datasets, custom task heads, and supervised gradient steps for every distinct objective (translation, summarization, QA).
- This raised a fundamental question: *Could a general sequence model perform multiple tasks in a zero-shot regime without any parameter updates or task-specific fine-tuning?*

Simultaneously, two severe architectural engineering bottlenecks existed in generative Transformers:
1. **Post-LN Vanishing/Exploding Gradients in Deep Stacks**:  
   In the original Post-LayerNorm Transformer (Vaswani et al., 2017), activations at layer $l$ passed through:
   $$x_{l+1} = \text{LayerNorm}(x_l + \text{SubLayer}(x_l))$$
   In deep models (24–48 layers), the variance at the residual connection compounded before normalization, making early gradient warmups fragile and causing optimization instability.
2. **The Quadratic Inference Tax ($O(T^2)$ Token Generation)**:  
   During autoregressive generation, generating token $t+1$ requires computing attention over tokens $1 \dots t$. Naively passing the entire sequence into the model at every generation step recomputes key and value projections for all previous tokens, scaling inference time as $O(T^2)$.

---

## 2. The Thought Experiment: Task Conditioning & Cached Memory

### 2.1 Language Modeling as Unsupervised Multitask Learning
Radford et al. formulated all language tasks under conditional probability estimation:
$$P(\text{output} \mid \text{input}, \text{task})$$
Instead of altering architecture or adding heads, task specification is delivered via **natural language prompting**:
- Translation: `translate to french, [english text], [target prompt]`
- Reading Comprehension: `[context], answer question: [question], [answer prompt]`

With sufficient capacity and broad diverse web-text pre-training (WebText), the autoregressive objective:
$$\mathcal{L}(\theta) = \sum_{i=1}^T \log P(x_i \mid x_1, \dots, x_{i-1}; \theta)$$
forces the network to model reasoning, syntax, and task-switching zero-shot!

### 2.2 KV-Cache: The $O(1)$ Token Step
Notice that during causal generation, when generating token $x_{t+1}$:
- The past keys $K_{1:t}$ and values $V_{1:t}$ depend **only** on tokens $x_{1:t}$.
- Because of causal masking, past keys and values **never change** when a new token $x_{t+1}$ is appended!
- Therefore, we can cache past key and value projections:
  $$K_{\text{all}} = [K_{\text{past}} \,;\, K_{\text{new}}], \qquad V_{\text{all}} = [V_{\text{past}} \,;\, V_{\text{new}}]$$
- We only need to project a single Query vector $q_{t+1} \in \mathbb{R}^{1 \times d}$, compute attention over $K_{\text{all}}$, and append $(K_{\text{new}}, V_{\text{new}})$ to the cache. This reduces computation per generated token from $O(t)$ to $O(1)$ projections!

---

## 3. Mathematical Architecture & Innovations

### 3.1 Pre-Layer Normalization (Pre-LN)
GPT-2 relocated Layer Normalization to the **input** of each sub-block:
$$x^{(l)}_{\text{mid}} = x^{(l-1)} + \text{Dropout}(\text{CausalSelfAttn}(\text{LayerNorm}(x^{(l-1)})))$$
$$x^{(l)} = x^{(l)}_{\text{mid}} + \text{Dropout}(\text{MLP}(\text{LayerNorm}(x^{(l)}_{\text{mid}})))$$
An additional Layer Normalization was added after the final Transformer block prior to the output logit projection:
$$x_{\text{final}} = \text{LayerNorm}(x^{(L)})$$

**Gradient Highway Benefit**:  
Notice that in Pre-LN, the identity path extends directly from the input embedding to the final layer:
$$x^{(L)} = x^{(0)} + \sum_{l=1}^L \Delta x^{(l)}$$
Gradients flow directly back through the residual highway without passing through repeated LayerNorm divisions, preventing vanishing gradients in deep networks.

### 3.2 Residual Weight Scaling at Initialization
To account for variance accumulation along the residual path as depth increases, GPT-2 scales the weights of residual projection layers (the output projection of MHA and the second linear layer of the MLP) at initialization by:
$$\frac{1}{\sqrt{2 N}}$$
where $N$ is the number of Transformer layers (since each layer contains 2 residual additions).

### 3.3 Causal Self-Attention with KV-Cache
Given query token at position $t$:
$$q_t = x_t W_q, \quad k_t = x_t W_k, \quad v_t = x_t W_v$$
Cache update:
$$K \leftarrow [K_{\text{cached}}, k_t], \quad V \leftarrow [V_{\text{cached}}, v_t]$$
$$\text{Attention}(q_t, K, V) = \text{softmax}\left(\frac{q_t K^T}{\sqrt{d_k}}\right) V$$

---

## 4. Tensor Contracts & Architectural Blueprint

```
Token IDs: (B, T) ───────> wte (Token Embedding) ────────┐
Position IDs: (B, T) ────> wpe (Learned Pos Embedding) ───┴──> (+) ──> Dropout
                                                                  │
┌─────────────────────────── GPT-2 Block Stack ───────────────────┘
│
│  x -> [LayerNorm] -> [CausalSelfAttention (with KV-Cache)] -> [Dropout] -> (+) -> x_mid
│  |                                                                          ^
│  └──────────────────────────────────────────────────────────────────────────┘
│  x_mid -> [LayerNorm] -> [MLP (GELU, 4*d_model)] ----------> [Dropout] -> (+) -> x_out
│  |                                                                          ^
│  └──────────────────────────────────────────────────────────────────────────┘
│
└─────────────────────────────────────────────────────────────────┐
                                                                  ▼
                                                             [LayerNorm] (Final LN)
                                                                  │
                                                        [Linear / Tied LM Head]
                                                                  │
                                                                  ▼
                                                      Logits: (B, T, vocab_size)
```

| Component | Input Shape | Output Shape | Notes |
| :--- | :--- | :--- | :--- |
| `GPT2Attention` | `x`: `(B, T, d_model)`, `layer_past`: `(K, V)` | `(B, T, d_model)`, `present_kv`: `(K, V)` | Supports training & cached inference |
| `GPT2MLP` | `x`: `(B, T, d_model)` | `(B, T, d_model)` | Linear $d \to 4d$, GELU, Linear $4d \to d$ |
| `GPT2Block` | `x`: `(B, T, d_model)`, `layer_past` | `(B, T, d_model)`, `present_kv` | Pre-LN Causal Attn + Pre-LN MLP |
| `GPT2Model` | `input_ids`: `(B, T)`, `past_key_values` | `(B, T, d_model)`, `pasts` | Backbone with final LayerNorm |
| `GPT2LMHeadModel` | `input_ids`: `(B, T)` | `logits`: `(B, T, vocab_size)` | Full language model with generation |
| `generate` | `prompt_ids`, `max_new_tokens` | `(B, prompt_len + max_new_tokens)` | Top-k, temperature, with KV-caching |

---

## 5. Downstream Unlocks

- **Paper #15 (LoRA)**: Parameter-efficient fine-tuning applied directly to GPT-2 attention projection weights.
- **Paper #52 (Decision Transformer)**: Replaces MDP value estimation by framing Offline RL as autoregressive GPT-2 modeling over (return-to-go, state, action) trajectories.
- **Paper #71 (Behavior Transformer - BeT)**: Generates robot actions via tokenized GPT-2 with k-means cluster heads.
- **Paper #78 & #79 (RT-1, OpenVLA)**: Autoregressive action token generation for robot manipulation.

## Further Reading

For more, [visit this chat](https://chatgpt.com/c/6aeaec84-9714-83ed-8e17-b1bcb234a372).

Second part, [visit this chat](https://chatgpt.com/c/6aeafa41-fed0-83eb-afa0-211b4fa164de).

