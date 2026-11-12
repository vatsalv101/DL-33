# Paper #19: Representation Learning with Contrastive Predictive Coding (CPC / InfoNCE)

> **Oord, A. v. d., Li, Y., & Vinyals, O. (2018 / arXiv:1807.03748)**  
> *DeepMind*  
> [arXiv:1807.03748](https://arxiv.org/abs/1807.03748)

---

## 1. Historical Breaking Point: The Failure of Pixel-Level Generative Reconstruction

Before Contrastive Predictive Coding (CPC), unsupervised representation learning relied on two dominant strategies:
1. **Pixel-Space Reconstruction (Autoencoders / Autoregressive Predictors)**: Training a model to predict exact raw observations $p(x_{t+k} \mid c_t)$ via MSE or pixel-wise cross-entropy.
   - **The Flaw**: High-dimensional sensory data (audio waveforms, video frames, sensor streams) is dominated by unpredictable local stochastic noise (e.g. exact background texture or static). Predicting raw pixels wastes massive model capacity on irrelevant high-frequency details rather than learning abstract semantic concepts (like phonemes, objects, or robot intent).
2. **Direct Mutual Information Maximization**: Directly computing Mutual Information $I(x_{t+k}; c_t) = \iint p(x, c) \log \frac{p(x|c)}{p(x)} \, dx \, dc$ is intractable in high-dimensional continuous spaces because the conditional generative density $p(x|c)$ is unknown.

---

## 2. The Thought Experiment: Density Ratio Estimation via Contrastive Guessing

Oord et al. asked:
> *Instead of generating the future observation $x_{t+k}$ from scratch, can we simply ask the model to **identify** the true future representation among a set of fake distractors?*

Suppose we encode observations into compact latents $z_t = g_{\text{enc}}(x_t)$ and summarize past history up to step $t$ with an autoregressive model $c_t = g_{\text{ar}}(z_{\le t})$.
To predict $k$ steps into the future without modeling $p(x_{t+k} \mid c_t)$, we instead model the **density ratio**:
$$f_k(x_{t+k}, c_t) \propto \frac{p(x_{t+k} \mid c_t)}{p(x_{t+k})}$$

Notice that the ratio $\frac{p(x_{t+k} \mid c_t)}{p(x_{t+k})}$ preserves all shared mutual information between the future and the past, while completely discarding factors that are independent of context! We approximate this log-density ratio with a simple **log-bilinear score**:
$$f_k(x_{t+k}, c_t) = \exp\left( z_{t+k}^T W_k c_t \right)$$

---

## 3. Mathematical Architecture & Step-by-Step Derivation

```
  Input Sequence:  x_1      x_2      ...      x_t     ...    x_{t+k}
                    │        │                 │                │
                    ▼        ▼                 ▼                ▼
  Encoder g_enc:   z_1      z_2      ...      z_t     ...    z_{t+k}  (True Future)
                    │        │                 │                ▲
                    └───────>┴─────> [g_ar] ──> c_t ──> [W_k] ──┘ (Contrastive Score)
                                                         │
                                              Negatives: z_j (j != t+k)
```

### 3.1 The InfoNCE Loss Function
Given a set $X = \{x_1, \dots, x_N\}$ containing **one positive sample** $x_{\text{pos}} \sim p(x_{t+k} \mid c_t)$ and **$N-1$ negative distractor samples** drawn from the marginal proposal distribution $p(x)$, the **InfoNCE** loss is the cross-entropy of classifying the positive sample:

$$\mathcal{L}_N = -\mathbb{E}_X \left[ \log \frac{f_k(x_{\text{pos}}, c_t)}{\sum_{x_j \in X} f_k(x_j, c_t)} \right]$$

### 3.2 Connection to Mutual Information Lower Bound
Let's prove why minimizing $\mathcal{L}_N$ maximizes Mutual Information $I(x_{t+k}; c_t)$!
The optimal probability for $x_i$ being the positive sample (via Bayes' theorem) is:
$$p(d = i \mid X, c_t) = \frac{p(x_i \mid c_t) \prod_{l \neq i} p(x_l)}{\sum_{j=1}^N p(x_j \mid c_t) \prod_{l \neq j} p(x_l)} = \frac{\frac{p(x_i \mid c_t)}{p(x_i)}}{\sum_{j=1}^N \frac{p(x_j \mid c_t)}{p(x_j)}}$$

Thus, the optimal critic $f_k(x, c_t)$ is directly proportional to $\frac{p(x \mid c_t)}{p(x)}$. Substituting this optimal ratio back into the InfoNCE loss:

$$\mathcal{L}_N^{\text{opt}} = -\mathbb{E}_X \log \left[ \frac{\frac{p(x_{t+k} \mid c_t)}{p(x_{t+k})}}{\frac{p(x_{t+k} \mid c_t)}{p(x_{t+k})} + \sum_{x_j \in X_{\text{neg}}} \frac{p(x_j \mid c_t)}{p(x_j)}} \right]$$

$$\approx \mathbb{E}_X \log \left[ 1 + \frac{p(x_{t+k})}{p(x_{t+k} \mid c_t)} (N - 1) \, \mathbb{E}_{x_j} \frac{p(x_j \mid c_t)}{p(x_j)} \right] = \mathbb{E}_X \log \left[ 1 + (N - 1) \frac{p(x_{t+k})}{p(x_{t+k} \mid c_t)} \right]$$

$$\ge \mathbb{E}_X \log \left[ N \frac{p(x_{t+k})}{p(x_{t+k} \mid c_t)} \right] = \log(N) - I(x_{t+k}; c_t)$$

Rearranging gives the celebrated **InfoNCE Mutual Information Bound**:
$$I(x_{t+k}; c_t) \ge \log(N) - \mathcal{L}_N$$

- **Tightness**: Increasing the number of negative samples $N$ raises $\log(N)$ and forces the encoder to learn sharper representations!
- **Random Guessing Baseline**: If the representation carries zero mutual information ($I = 0$), the loss equals $\log(N)$ (uniform guessing over $N$ choices).

---

## 4. Tensor Contracts & Architectural Blueprint

| Component | Input Shape | Output Shape | Notes |
| :--- | :--- | :--- | :--- |
| `info_nce_loss` | `queries: (N, D)`, `keys: (N, D)` | `loss: scalar`, `acc: scalar` | In-batch negatives; similarity $S = \frac{QK^T}{\tau}$ |
| `CPCEncoder` | `x: (B, L, input_dim)` | `z: (B, L, latent_dim)` | Non-linear feature extractor $g_{\text{enc}}$ |
| `CPCModel` | `x: (B, L, input_dim)` | `(loss, accuracy, z, c)` | Multi-horizon $k=1 \dots K$ predictive coding |

---

## 5. Downstream Unlocks

- **Paper #20 (SimCLR) & Paper #21 (MoCo)**: Extending InfoNCE from temporal sequence prediction to image-level instance discrimination using data augmentations.
- **Paper #24 (CLIP)**: Applying symmetric cross-modal InfoNCE loss across paired image and text embeddings.
- **Paper #25 (R3M)**: Combining Time-Contrastive Learning (TCN + InfoNCE) over robot manipulation videos to learn generalizable robotic visual representations!

---

## Further Reading

For more, [visit this chat](https://chatgpt.com/c/6af57196-8110-83ed-8bfd-0f3c924c86b6).
