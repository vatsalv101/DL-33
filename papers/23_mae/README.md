# Paper #23: Masked Autoencoders Are Scalable Vision Learners (MAE)

> **He, K., Chen, X., Xie, S., Li, Y., Bojanowski, P., & Girshick, R. (2022 / CVPR 2022)**  
> *Facebook AI Research (FAIR)*  
> [arXiv:2111.06377](https://arxiv.org/abs/2111.06377)

---

## 1. Historical Breaking Point: Why BERT Failed on Images

In Natural Language Processing, Masked Language Modeling (BERT, Paper #13) became the undisputed foundational pre-training paradigm: mask 15% of tokens and predict them using a bidirectional Transformer encoder.

For years, researchers attempted to transplant masked autoencoding to computer vision, but results consistently lagged behind contrastive learning (SimCLR, MoCo). Kaiming He et al. diagnosed the root causes:
1. **Information Density Mismatch**:
   Language is human-generated, highly semantic, and information-dense: predicting a missing word requires syntactic parsing and semantic comprehension. Images, by contrast, possess **heavy spatial redundancy**: a missing pixel or small patch can easily be interpolated from adjacent pixels without understanding scene semantics. Masking only 15% of an image produces a trivial interpolation problem!
2. **Grid Rigidity of Convolutions**:
   Standard CNNs operate on fixed 2D spatial grids. They cannot process an arbitrary, sparse subset of patches without inserting artificial zero-padding or destroying convolutional strides.
3. **Compute Inefficiency**:
   Autoencoders that processed both visible and masked tokens simultaneously wasted massive GPU compute and memory on reconstructing empty mask tokens through deep encoders.

---

## 2. The Thought Experiment: Asymmetric Masked Autoencoding at 75% Masking

Kaiming He et al. introduced three conceptual breakthroughs:

```
      Raw Image x: (B, 3, H, W)
                 │
                 ▼
     [ Patchify: P x P Patches ]  ──> N = (H/P) * (W/P) Patches
                 │
                 ▼
    ┌─────────────────────────┐
    │ Random Masking: 75%     │
    └────┬───────────────┬────┘
         │ (25% Visible) │ (75% Dropped)
         ▼               ▼
  [ Deep Encoder ]     (Skipped entirely! 4x to 16x compute speedup!)
         │
         ▼
    Latent Tokens
         │
         ▼
  [ Insert Shared Learnable [MASK] Tokens + Full Positional Embeddings ]
         │
         ▼
  [ Lightweight Decoder ]  (Processes full N tokens)
         │
         ▼
  [ Predict Normalized Pixels per Patch: P x P x 3 ]
         │
         ▼
  [ MSE Loss computed ONLY on the 75% Masked Patches! ]
```

1. **High Masking Ratio (75%–80%)**:
   By masking out 75% of the image, neighboring pixel interpolation is rendered impossible. The model cannot cheat; it must understand holistic 3D geometry and semantics (e.g. observing a small portion of a dog's snout and deducing where ears, eyes, and fur belong).
2. **Asymmetric Architecture**:
   - **Heavy Encoder**: Operates **strictly on the 25% visible patches**! Masked patches are completely discarded before entering the encoder. Because self-attention scales as $O(L^2)$, processing only $25\%$ of patches yields a **$16\times$ reduction in compute and memory**!
   - **Lightweight Decoder**: A shallow Transformer (e.g. 8 layers with half the embedding dimension) reconstructs pixels from the combined sequence of encoded visible tokens and learnable `[MASK]` tokens.
3. **Direct Pixel Reconstruction (MSE on Masked Patches)**:
   The model directly predicts normalized raw pixel values for each masked patch, supervised with Mean Squared Error (MSE).

---

## 3. Mathematical Architecture & Step-by-Step Derivation

### 3.1 Patchify & Unpatchify Operations
Given an image $x \in \mathbb{R}^{B \times 3 \times H \times W}$ and patch size $P$:
- Number of patches along height and width: $h = H / P$, $w = W / P$, total patches $N = h \cdot w$.
- **Patchify**: Reshape to $(B, N, P^2 \cdot 3)$.
- **Unpatchify**: Invert back to $(B, 3, H, W)$.

### 3.2 Random Masking without Replacement
For a sequence of $N$ patches:
1. Generate random noise $r \sim \mathcal{U}(0, 1)$ of shape $(B, N)$.
2. Sort noise: `ids_shuffle = torch.argsort(r, dim=1)`.
3. Compute inverse permutation: `ids_restore = torch.argsort(ids_shuffle, dim=1)`.
4. Keep the first $N_{\text{vis}} = \text{int}(N \cdot (1 - \text{mask\_ratio}))$ indices:
   $$\text{visible\_tokens} = x[\text{batch}, \text{ids\_shuffle}[:, :N_{\text{vis}}]]$$
5. Binary mask indicator: $1$ for masked patches, $0$ for visible patches:
   $$\text{mask} \in \{0, 1\}^{B \times N}$$

### 3.3 Reconstruction Loss with Per-Patch Normalization
Normalize each ground-truth patch to zero mean and unit variance:
$$\mu = \frac{1}{P^2 \cdot 3} \sum p_i, \quad \sigma^2 = \frac{1}{P^2 \cdot 3} \sum (p_i - \mu)^2$$
$$\tilde{p} = \frac{p - \mu}{\sqrt{\sigma^2 + \epsilon}}$$

The loss is computed as Mean Squared Error **exclusively on the masked patches**:
$$\mathcal{L}_{\text{MAE}} = \frac{1}{\sum \text{mask}} \sum_{i=1}^B \sum_{j=1}^N \text{mask}_{i, j} \cdot \| \hat{p}_{i, j} - \tilde{p}_{i, j} \|_2^2$$

---

## 4. Tensor Contracts & Architectural Blueprint

| Component | Input Shape | Output Shape | Notes |
| :--- | :--- | :--- | :--- |
| `patchify` | `(B, 3, H, W)` | `(B, N, P*P*3)` | Flattens spatial patches into tokens |
| `unpatchify` | `(B, N, P*P*3)` | `(B, 3, H, W)` | Reconstructs image from patch tokens |
| `random_masking` | `(B, N, D)` | `(x_vis: (B, N_vis, D), mask: (B, N), ids_restore: (B, N))` | Shuffles and slices sequence |
| `MaskedAutoencoderViT` | `x: (B, 3, H, W)` | `(loss: scalar, pred: (B, N, P*P*3), mask: (B, N))` | Full MAE self-supervised forward |

---

## 5. Downstream Unlocks

- **MAE for Video & 3D (VideoMAE, PointMAE)**: Extending 75%–90% spatiotemporal masking to raw video frames and 3D point clouds.
- **Vision Foundation Encoders**: Fine-tuning pre-trained MAE encoders set new SOTA records on ImageNet-1K, COCO detection, and ADE20K segmentation.
- **Robotic Pre-Training**: Visual encoders pre-trained with MAE learn spatial 3D scene geometry from passive unlabelled video streams, transferring directly to visuomotor control policies!

---

## Further Reading

For more, [visit this chat](https://chatgpt.com/c/6afae1e5-cd68-83eb-ba07-be7f53197bfe).
For next, [visit this chat](https://chatgpt.com/c/6afaee72-c4c4-83eb-9142-01ba71b0c0ac).
