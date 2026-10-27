# Paper #12: RoFormer: Enhanced Transformer with Rotary Position Embedding (RoPE)

> **Su, J., Lu, Y., Pan, S., Ahmed, B., Liu, Y., & Wang, Y. (2021).**  
> *Neurocomputing / arXiv:2104.09864*  
> [arXiv:2104.09864](https://arxiv.org/abs/2104.09864)

---

## 1. Historical Breaking Point: Absolute vs. Relative Positions

In the original Transformer (Vaswani et al., 2017), positions were injected **additively** into input token embeddings:
$$\tilde{\mathbf{x}}_i = \mathbf{x}_i + \mathbf{p}_i$$

When queries and keys are formed via linear projections $W_q, W_k$, their inner product decomposes into:
$$\mathbf{q}_m^T \mathbf{k}_n = (\mathbf{x}_m + \mathbf{p}_m)^T W_q^T W_k (\mathbf{x}_n + \mathbf{p}_n) = \underbrace{\mathbf{x}_m^T W_q^T W_k \mathbf{x}_n}_{\text{content-content}} + \underbrace{\mathbf{x}_m^T W_q^T W_k \mathbf{p}_n}_{\text{content-position}} + \underbrace{\mathbf{p}_m^T W_q^T W_k \mathbf{x}_n}_{\text{position-content}} + \underbrace{\mathbf{p}_m^T W_q^T W_k \mathbf{p}_n}_{\text{position-position}}$$

This additive formulation introduces critical flaws:
1. **Entangled Representation Space**: Content and positional information interfere within the same embedding vector space.
2. **Context Length Extrapolation Failure**: Absolute encodings $\mathbf{p}_m$ do not naturally extrapolate to sequence lengths beyond the pre-training context window.
3. **Flawed Relative Distance Modeling**: Models like Transformer-XL (Dai et al., 2019) and T5 (Raffel et al., 2020) introduced relative position biases $b_{m - n}$ added directly to the attention logits $A_{m, n}$. However, these methods modify the attention matrix directly, breaking compatibility with modern fused kernels (e.g. FlashAttention) and linear attention formulations.

---

## 2. The Thought Experiment: Inner Product Invariance Under Rotation

Jianlin Su et al. formulated the fundamental question:
> *Can we define a coordinate transformation $\mathbf{f}_q(\mathbf{x}_m, m)$ and $\mathbf{f}_k(\mathbf{x}_n, n)$ applied locally to query and key vectors such that their inner product depends purely on their relative offset $(m - n)$ and their contents?*

$$\langle \mathbf{f}_q(\mathbf{x}_m, m), \mathbf{f}_k(\mathbf{x}_n, n) \rangle = g(\mathbf{x}_m, \mathbf{x}_n, m - n)$$

### The 2D Complex Plane Intuition
Consider a 2D vector represented as a complex number $z = x^{(1)} + i x^{(2)} \in \mathbb{C}$.  
Multiplying $z$ by $e^{i m \theta}$ rotates it by angle $m \theta$ in the complex plane.

Now evaluate the Hermitian inner product between query at position $m$ and key at position $n$:
$$\langle \mathbf{q}_m e^{i m \theta}, \mathbf{k}_n e^{i n \theta} \rangle = \text{Re}\left[ (\mathbf{q}_m e^{i m \theta}) (\mathbf{k}_n e^{i n \theta})^* \right] = \text{Re}\left[ \mathbf{q}_m \mathbf{k}_n^* e^{i (m - n) \theta} \right]$$

Notice the magic: **The absolute positions $m$ and $n$ vanish entirely**, leaving only the relative distance $(m - n)$!

---

## 3. Mathematical Derivation: d-Dimensional RoPE

To extend this rotation to $d$-dimensional representations (where $d$ is the head dimension, assumed even), we divide the $d$-dimensional space into $d/2$ independent orthogonal 2D planes:

$$R_{\Theta, m}^d = \text{diag}\left( R_{\theta_1, m}, R_{\theta_2, m}, \dots, R_{\theta_{d/2}, m} \right)$$

where each $2 \times 2$ rotation sub-matrix is:
$$R_{\theta_i, m} = \begin{pmatrix} \cos(m \theta_i) & -\sin(m \theta_i) \\ \sin(m \theta_i) & \cos(m \theta_i) \end{pmatrix}, \qquad \theta_i = 10000^{-2(i-1)/d}, \quad i \in \{1, \dots, d/2\}$$

### Fast Vectorized Implementation: `rotate_half`
Multiplying explicitly by a $d \times d$ block matrix is computationally wasteful. Instead, note that for any 2D vector $\begin{pmatrix} x^{(1)} \\ x^{(2)} \end{pmatrix}$:
$$\begin{pmatrix} \cos\theta & -\sin\theta \\ \sin\theta & \cos\theta \end{pmatrix} \begin{pmatrix} x^{(1)} \\ x^{(2)} \end{pmatrix} = \begin{pmatrix} x^{(1)} \\ x^{(2)} \end{pmatrix} \cos\theta + \begin{pmatrix} -x^{(2)} \\ x^{(1)} \end{pmatrix} \sin\theta$$

In modern implementations (LLaMA, Mistral, HuggingFace), this is generalized across the head dimension by splitting the vector into two halves:
$$\mathbf{x} = [\mathbf{x}_{1}, \mathbf{x}_{2}], \quad \text{where } \mathbf{x}_1, \mathbf{x}_2 \in \mathbb{R}^{d/2}$$
$$\text{rotate\_half}(\mathbf{x}) = [-\mathbf{x}_{2}, \mathbf{x}_{1}]$$
$$\text{RoPE}(\mathbf{x}, m) = (\mathbf{x} \odot \cos(m \Theta)) + (\text{rotate\_half}(\mathbf{x}) \odot \sin(m \Theta))$$
where $\cos(m \Theta)$ and $\sin(m \Theta)$ are replicated across both halves:
$$\cos(m \Theta) = [\cos(m \theta_1), \dots, \cos(m \theta_{d/2}), \cos(m \theta_1), \dots, \cos(m \theta_{d/2})]$$

This achieves exact rotation via **two element-wise tensor multiplications and one tensor concatenation**, with zero matrix overhead!

---

## 4. Key Mathematical Properties

### 1. Relative Shift Invariance
For any arbitrary shift $k \in \mathbb{Z}$:
$$\langle \text{RoPE}(\mathbf{q}, m + k), \text{RoPE}(\mathbf{k}, n + k) \rangle = \langle \text{RoPE}(\mathbf{q}, m), \text{RoPE}(\mathbf{k}, n) \rangle$$

### 2. Natural Long-Range Decay
Because $\theta_i = 10000^{-2(i-1)/d}$ spans multiple frequency scales, high-frequency dimensions rotate rapidly while low-frequency dimensions rotate slowly. Summing across all $d/2$ sub-spaces causes destructive interference for large offsets $|m - n|$:
$$\sum_{i=1}^{d/2} \cos((m - n)\theta_i) \to 0 \quad \text{as } |m - n| \to \infty$$
The attention mechanism is naturally biased toward nearby tokens without requiring hand-crafted distance penalties!

### 3. Linear & FlashAttention Compatibility
Because RoPE operates directly on $\mathbf{q}_m$ and $\mathbf{k}_n$ *before* the dot product:
$$\text{Attention}(\tilde{Q}, \tilde{K}, V) = \text{softmax}\left(\frac{\tilde{Q} \tilde{K}^T}{\sqrt{d_k}}\right) V$$
It leaves the attention operation unchanged, making it 100% compatible with kernel fusion (FlashAttention) and KV-cache autoregressive inference.

---

## 5. Architectural Blueprint

```
Token x_m --------> Q Projection --------> RoPE(Q, m) -----\
                                                             ---> Scaled Dot-Product Attn ---> Output
Token x_n --------> K Projection --------> RoPE(K, n) -----/
Token x_n --------> V Projection -------------------------/
```

| Component | Input Shape | Output Shape | Notes |
| :--- | :--- | :--- | :--- |
| `RotaryPositionalEmbedding` | `seq_len` | `cos`, `sin`: `(1, 1, seq_len, d)` | Precomputes and caches frequencies |
| `apply_rotary_pos_emb` | `x: (B, H, T, d)`, `cos`, `sin` | `(B, H, T, d)` | Rotates vector using `rotate_half` |
| `RoPEMultiHeadAttention` | `x: (B, T, d_model)` | `(B, T, d_model)` | Projects $Q, K, V$, rotates $Q, K$, attends |
| `RoFormerEncoderLayer` | `x: (B, T, d_model)` | `(B, T, d_model)` | RoPE Self-Attn + Pre-LN/Post-LN + FFN |
| `RoFormerEncoder` | `x: (B, T, d_model)` | `(B, T, d_model)` | Full sequence encoder with RoPE |

---

## 6. Downstream Unlocks

RoPE is the de-facto standard positional embedding in state-of-the-art architectures:
- **LLaMA 1 / 2 / 3 & Mistral & Gemma**: Standard backbone positional encoding.
- **Paper #14 (GPT-2) & Modern Decoders**: RoPE powers efficient KV-cache extensions (YaRN, RoPE scaling).
- **Paper #79 & #80 (OpenVLA, Octo)**: Vision-Language-Action robotics transformers use RoPE to encode long trajectories of visual and proprioceptive state tokens.
