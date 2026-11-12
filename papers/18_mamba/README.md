# Paper #18: Mamba: Linear-Time Sequence Modeling with Selective State Spaces

> **Gu, A., & Dao, T. (2023 / arXiv:2312.00752)**  
> *Carnegie Mellon University & Princeton University*  
> [arXiv:2312.00752](https://arxiv.org/abs/2312.00752)

---

## 1. Historical Breaking Point: The Quadratic Bottleneck vs. The Expressive Limits of LTI SSMs

The Transformer architecture (Vaswani et al., 2017) dominated sequence modeling due to its expressive multi-head self-attention. However, self-attention suffers from two fundamental scalability bottlenecks:
1. **$O(L^2)$ Computational & Memory Complexity**: Every token attends to every previous token, creating prohibitive quadratic compute and activation memory for long sequences (e.g. $L \ge 8\text{k}, 32\text{k}, 1\text{M}$).
2. **$O(L)$ KV-Cache Inference Footprint**: Autoregressive decoding requires caching all previous keys and values, requiring vast GPU VRAM and memory bandwidth per active user.

To achieve $O(L)$ linear-time scaling, researchers introduced **Linear Time-Invariant (LTI) State Space Models (SSMs)** like **S4** (Gu et al., 2021). S4 maps continuous 1D inputs $x(t) \in \mathbb{R}$ through hidden latent states $h(t) \in \mathbb{R}^N$:
$$\dot{h}(t) = A h(t) + B x(t), \quad y(t) = C h(t)$$

Because parameters $(A, B, C)$ in S4 were **static constants across time** (Time-Invariant):
- **Training Mode**: S4 could be unrolled into a global 1D convolution of length $L$ via FFT in $O(L \log L)$ time.
- **Inference Mode**: S4 operated as a constant-memory $O(1)$ recurrent step.

### The Fatal Flaw of LTI Models: Content-Blind Memorization
Because $(A, B, C)$ did not depend on the input tokens $x_t$, LTI models could **not select, filter, or discard context dynamically based on content**. An LTI model applies the exact same decay and projection to a crucial keyword as it does to a punctuation mark. As a consequence, S4 fundamentally failed at:
- **Selective Copying**: Memorizing specific marked tokens across arbitrary noise sequences.
- **Induction Heads & Associative Recall**: Looking up values associated with previously seen keys (the bedrock of In-Context Learning).

---

## 2. The Thought Experiment: Selective State Spaces (S6)

Imagine modifying the continuous system so that the matrices $B$, $C$, and the discretization step size $\Delta$ are **functions of the current input token $x_t$**:

$$B_t = s_B(x_t), \quad C_t = s_C(x_t), \quad \Delta_t = \text{softplus}(\text{Linear}(x_t) + \Delta_{\text{bias}})$$

Now consider the physical intuition of the step size $\Delta_t$:
- If $\Delta_t \to \infty$ (large step): $\bar{A}_t = \exp(\Delta_t A) \to 0$ and $\bar{B}_t \approx \Delta_t B_t$ is large. The model **resets its previous state $h_{t-1}$** and heavily focuses on the current input $x_t$ (like encountering an important new topic sentence).
- If $\Delta_t \to 0$ (tiny step): $\bar{A}_t \to I$ and $\bar{B}_t \to 0$. The model **preserves its past hidden state $h_{t-1}$ untouched** and completely ignores the current input $x_t$ (like skipping unimportant filler words or noise).

By making $(\Delta, B, C)$ input-dependent, the model transitions from a **Linear Time-Invariant (LTI)** system to a **Linear Time-Varying (LTV)** system.

| Property | Self-Attention (Transformer) | LTI SSM (S4) | Selective SSM / S6 (Mamba) |
| :--- | :--- | :--- | :--- |
| **Training Complexity** | $O(L^2 \cdot d)$ | $O(L \log L \cdot d)$ (FFT Conv) | $O(L \cdot d \cdot N)$ (Scan) |
| **Inference Step** | $O(L \cdot d)$ (KV-Cache) | $O(1 \cdot d \cdot N)$ (State) | $O(1 \cdot d \cdot N)$ (State) |
| **Inference Memory** | $O(L \cdot d)$ | $O(d \cdot N)$ | $O(d \cdot N)$ |
| **Context Selection** | Content-aware dynamic routing | Content-blind static weights | **Content-aware dynamic selection** |

---

## 3. Mathematical Architecture & Step-by-Step Derivation

```
                        Input x: (B, L, d_model)
                                   │
                                   ▼
                   ┌───────────────────────────────┐
                   │ Linear Projection: 2*d_inner   │
                   └───────┬───────────────┬───────┘
                           │ (Branch 1)    │ (Branch 2: Gate)
                           ▼               │
               ┌───────────────────────┐   │
               │ 1D Causal Conv1d       │   │
               │ + SiLU Activation     │   │
               └───────────┬───────────┘   │
                           │               │
                           ▼               │
               ┌───────────────────────┐   │
               │ Selective SSM (S6)    │   │
               │  - Project Δ, B, C    │   │
               │  - Discretize A, B    │   │
               │  - Selective Scan     │   │
               │  - Add Residual D*x   │   │
               └───────────┬───────────┘   │
                           │               │
                           ▼               ▼
                           [  * Elementwise SiLU Gate  ]
                                   │
                                   ▼
                   ┌───────────────────────────────┐
                   │ Output Linear: d_inner -> d   │
                   └───────────────┬───────────────┘
                                   │
                                   ▼
                        Output y: (B, L, d_model)
```

### 3.1 Continuous to Discrete State Space Formulation
The continuous state space equation:
$$h'(t) = A h(t) + B x(t)$$
$$y(t) = C h(t) + D x(t)$$

Using **Zero-Order Hold (ZOH)** discretization with step size $\Delta$:
$$\bar{A} = \exp(\Delta A)$$
$$\bar{B} = (\Delta A)^{-1}(\exp(\Delta A) - I) \cdot \Delta B \approx \Delta B$$

In Mamba:
- $A \in \mathbb{R}^{D \times N}$ is structured as a diagonal matrix. To guarantee stability and prevent explosive states, $A$ is parameterized as $A = -\exp(\text{A\_log})$ where $\text{A\_log}$ is initialized using the HiPPO heuristic:
  $$A_{d, n} = -(n + 1), \quad n \in \{0, \dots, N-1\}$$
- $B \in \mathbb{R}^{B \times L \times N}$ is projected from input $x$: $B = \text{Linear}_{d \to N}(x)$
- $C \in \mathbb{R}^{B \times L \times N}$ is projected from input $x$: $C = \text{Linear}_{d \to N}(x)$
- $\Delta \in \mathbb{R}^{B \times L \times D}$ is projected from input $x$ via a low-rank bottleneck ($\text{rank} = \lceil d / 16 \rceil$):
  $$\Delta = \text{softplus}(\text{Linear}_{\text{rank} \to D}(\text{Linear}_{D \to \text{rank}}(x)) + \Delta_{\text{bias}})$$
- $D \in \mathbb{R}^D$ is a learnable skip-connection parameter.

### 3.2 The Selective Recurrence (Sequential Scan)
At each sequence step $t \in \{0, \dots, L-1\}$:
1. Discretize transition and input matrices:
   $$\bar{A}_t = \exp(\Delta_t \odot A) \in \mathbb{R}^{B \times D \times N}$$
   $$\bar{B}_t = \Delta_t \odot B_t \in \mathbb{R}^{B \times D \times N}$$
2. Update hidden state:
   $$h_t = \bar{A}_t \odot h_{t-1} + \bar{B}_t \odot x_t \in \mathbb{R}^{B \times D \times N}$$
3. Compute output:
   $$y_t = \sum_{n=1}^N C_{t, n} \cdot h_{t, :, n} + D \odot x_t \in \mathbb{R}^{B \times D}$$

### 3.3 The Mamba Block Architecture
The Mamba Block fuses the S6 selective SSM with the gated MLP structure (similar to SwiGLU):
1. **Input Projection**: $u, z = \text{Linear}(x) \in \mathbb{R}^{B \times L \times 2 D_{\text{inner}}}$.
2. **Depthwise Convolution**: $u_{\text{conv}} = \text{SiLU}(\text{Conv1D}(u)) \in \mathbb{R}^{B \times L \times D_{\text{inner}}}$.
3. **Selective SSM**: $y_{\text{ssm}} = \text{S6}(u_{\text{conv}}) \in \mathbb{R}^{B \times L \times D_{\text{inner}}}$.
4. **Multiplicative Gating**: $y_{\text{gated}} = y_{\text{ssm}} \odot \text{SiLU}(z)$.
5. **Output Projection**: $\text{out} = \text{Linear}(y_{\text{gated}}) \in \mathbb{R}^{B \times L \times D_{\text{model}}}$.

---

## 4. Tensor Contracts & Architectural Blueprint

| Component | Input Shape | Output Shape | Notes |
| :--- | :--- | :--- | :--- |
| `selective_scan` | `u: (B, L, D)`, `delta: (B, L, D)`, `A: (D, N)`, `B: (B, L, N)`, `C: (B, L, N)`, `D: (D)` | `y: (B, L, D)` | Content-dependent recurrent scan |
| `SelectiveSSM` | `x: (B, L, D)` | `y: (B, L, D)` | Computes $\Delta, B, C$, discretizes, and scans |
| `MambaBlock` | `x: (B, L, d_model)` | `out: (B, L, d_model)` | Causal Conv1d + S6 SSM + SiLU gate |
| `MambaLM` | `token_ids: (B, L)` | `logits: (B, L, vocab_size)` | Full causal language model stack |

---

## 5. Downstream Unlocks

- **Mamba-2 & State Space Duality (SSD)**: Establishing theoretical equivalence between structured state space models and structured masked attention.
- **Long-Context Reasoning**: Processing $100\text{k}+$ token prompts without KV-cache blowout.
- **Linear-Time Multimodal & World Models**: Replacing quadratic self-attention in Video diffusion and continuous robot trajectory generation (e.g. Mamba-ND, Decision Mamba).

---

## Further Reading

For more, [visit this chat](https://chatgpt.com/c/6af1961e-d914-83eb-8094-59af29350018).
For next part, [visit this chat](https://chatgpt.com/c/6af41867-3be8-83ed-b08e-7c6b0960e30c).
For code part, [visit this chat](https://chatgpt.com/c/6af424e1-3990-83ed-8a82-3e9e67245e32).
