# Paper #16: An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale (ViT)

> **Dosovitskiy, A., Beyer, L., Kolesnikov, A., Weissenborn, D., Zhai, X., Unterthiner, T., Dehghani, M., Minderer, M., Heigold, G., Gelly, S., Uszkoreit, J., & Houlsby, N. (2020 / ICLR 2021)**  
> *Google Research, Brain Team*  
> [arXiv:2010.11929](https://arxiv.org/abs/2010.11929)

---

## 1. Historical Breaking Point: The 2D Inductive Bias Monopoly

From AlexNet (2012) to ResNet (2015), computer vision was dominated by Convolutional Neural Networks (CNNs). Convolutions hardcode two foundational inductive biases directly into the network architecture:
1. **Translation Equivariance**: Shifting an image shifts feature activations identically ($f(g(x)) = g(f(x))$).
2. **Locality (Local Receptive Fields)**: Early convolutional kernels operate strictly over tiny $3 \times 3$ or $5 \times 5$ pixel neighborhoods.

### The Scaling Ceiling
While these inductive biases make CNNs sample-efficient on small datasets (e.g. ImageNet-1K), they act as an **inflexible inductive straitjacket** at planetary scale (JFT-300M, LAION-5B):
- Convolutions cannot dynamically route information globally across distant regions of an image in a single layer.
- Long-range relationships require deep hierarchies of repeated pooling and stride operations.
- NLP had already discarded recurrent and convolutional inductive biases in favor of the pure, general-purpose Transformer (Vaswani et al., 2017). Could the exact same architecture replace convolutions entirely in Computer Vision?

---

## 2. The Thought Experiment: "An Image is Worth 16x16 Words"

How do we feed a 2D image $x \in \mathbb{R}^{H \times W \times C}$ into a standard 1D sequence Transformer?

If we treat every individual pixel as a token, an image of size $224 \times 224$ produces $T = 50,176$ tokens. Because self-attention scales quadratically with sequence length ($O(T^2)$), pixel-level attention requires over $2.5 \times 10^9$ attention operations per layer—computationally prohibitive!

Dosovitskiy et al. proposed a simple, brilliant bridge:
> *Slice the 2D image into a grid of non-overlapping $P \times P$ patches (e.g., $16 \times 16$), flatten each patch into a vector, and treat each patch exactly like a word token in an NLP sentence!*

For $H = W = 224$ and $P = 16$:
$$N = \frac{H \cdot W}{P^2} = \frac{224 \cdot 224}{16 \cdot 16} = 14 \times 14 = 196 \text{ tokens}$$
$T = 196$ is roughly the length of a short paragraph—completely tractable for standard Transformer encoders!

---

## 3. Mathematical Architecture & Step-by-Step Derivation

### 3.1 Linear Patch Projection (Patch Embedding)
Given an image $\mathbf{x} \in \mathbb{R}^{B \times C \times H \times W}$, we partition it into $N$ patches $\mathbf{x}_p \in \mathbb{R}^{B \times N \times (P^2 \cdot C)}$.
Each flattened patch is projected to the Transformer hidden dimension $D$ via a learned linear projection $\mathbf{E} \in \mathbb{R}^{(P^2 \cdot C) \times D}$:
$$\mathbf{z}_0 = [\mathbf{x}_{\text{class}} \,;\, \mathbf{x}_p^1 \mathbf{E} \,;\, \mathbf{x}_p^2 \mathbf{E} \,;\, \dots \,;\, \mathbf{x}_p^N \mathbf{E}] + \mathbf{E}_{\text{pos}}$$

#### The 2D Convolution Equivalence
Flattening patches and multiplying by matrix $\mathbf{E}$ is mathematically identical to a single 2D Convolution:
$$\text{Conv2d}(\text{in\_channels}=C, \text{out\_channels}=D, \text{kernel\_size}=P, \text{stride}=P)$$
Setting stride equal to kernel size $P$ guarantees non-overlapping patch sampling!

### 3.2 Prepend Learnable Class Token (`[CLS]`)
Analogous to BERT's `[CLS]` token, a learnable embedding $\mathbf{x}_{\text{class}} \in \mathbb{R}^{1 \times 1 \times D}$ is prepended to the patch sequence at position 0:
$$\mathbf{z}_0 \in \mathbb{R}^{B \times (N + 1) \times D}$$
Because self-attention is isotropic, the `[CLS]` token aggregates information from all $N$ visual patches across all layers, serving as a global visual representation unaffected by arbitrary spatial biases.

### 3.3 Learnable 1D Position Embeddings
Because pure self-attention contains zero spatial coordinate awareness, learnable 1D positional embeddings $\mathbf{E}_{\text{pos}} \in \mathbb{R}^{1 \times (N + 1) \times D}$ are added element-wise:
$$\mathbf{z}_0 \leftarrow \mathbf{z}_0 + \mathbf{E}_{\text{pos}}$$
During training, the model autonomously discovers 2D spatial grid topology (patches closer in 2D space learn similar position vectors).

### 3.4 Pre-LN Transformer Encoder Stack
The sequence of $N + 1$ tokens passes through $L$ identical Transformer encoder blocks:
$$\mathbf{z}'_l = \text{MSA}(\text{LayerNorm}(\mathbf{z}_{l-1})) + \mathbf{z}_{l-1}, \quad l = 1 \dots L$$
$$\mathbf{z}_l = \text{MLP}(\text{LayerNorm}(\mathbf{z}'_l)) + \mathbf{z}'_l, \quad l = 1 \dots L$$

where $\text{MLP}(x) = \text{Linear}_{D \to 4D} \to \text{GELU} \to \text{Dropout} \to \text{Linear}_{4D \to D} \to \text{Dropout}$.

### 3.5 Classification Head (MLP Head)
The representation of the `[CLS]` token at the final layer $L$ is extracted and normalized:
$$\mathbf{y} = \text{Linear}(\text{LayerNorm}(\mathbf{z}_L^0))$$

---

## 4. Tensor Contracts & Architectural Blueprint

```
Input Image: (B, C, H, W)
      │
      ▼
[Conv2d(kernel=P, stride=P)] ────────> Shape: (B, D, H/P, W/P)
      │
      ▼
Flatten & Transpose ──────────────────> Shape: (B, N, D)
      │
      ▼
Prepend [CLS] Token (B, 1, D) ────────> Shape: (B, N + 1, D)
      │
      ▼
Add Positional Embeddings (1, N+1, D) ─> Shape: (B, N + 1, D)
      │
      ▼
┌────────────────── ViT Encoder Stack (x L) ──────────────────┐
│                                                             │
│  z -> [LayerNorm] -> [MultiHeadSelfAttention] -> (+) -> z'  │
│  |                                                ^         │
│  └────────────────────────────────────────────────┘         │
│  z' -> [LayerNorm] -> [MLP (GELU, 4*D)] --------> (+) -> z  │
│  |                                                ^         │
│  └────────────────────────────────────────────────┘         │
└─────────────────────────────────────────────────────────────┘
      │
      ▼
[LayerNorm(D)]
      │
      ▼
Extract [CLS] Token (index 0) ────────> Shape: (B, D)
      │
      ▼
[Linear(D, num_classes)] ─────────────> Logits: (B, num_classes)
```

| Component | Input Shape | Output Shape | Notes |
| :--- | :--- | :--- | :--- |
| `PatchEmbedding` | `(B, C, H, W)` | `(B, N, D)` | $N = (H/P) \times (W/P)$ patches |
| `ViTBlock` | `(B, N + 1, D)` | `(B, N + 1, D)` | Pre-LN MHA + Pre-LN GELU MLP |
| `ViT` | `(B, C, H, W)` | `(B, num_classes)` | Full classifier with CLS token |

---

## 5. Downstream Unlocks

- **Paper #22 & #23 (DINO & MAE)**: Self-supervised pre-training on images using ViT backbones.
- **Paper #24 (CLIP)**: Aligning ViT image embeddings with Transformer text representations.
- **Paper #72 & #79 (ACT & OpenVLA)**: ViT backbones generating visual observation tokens for end-to-end robot manipulation.

---

## Further Reading

For more, [visit this chat](https://chatgpt.com/c/6aeeee21-8108-83eb-beb0-cdc349efda21).
