# Paper #24: Learning Transferable Visual Models From Natural Language Supervision (CLIP)

> **Radford, A., Kim, J. W., Hallacy, C., Ramesh, A., Goh, G., Agarwal, S., Sastry, G., Askell, A., Mishkin, P., Clark, J., Krueger, G., & Sutskever, I. (2021 / ICML 2021)**  
> *OpenAI*  
> [arXiv:2103.00020](https://arxiv.org/abs/2103.00020)

---

## 1. Historical Breaking Point: The Fixed-Class Label Straitjacket

For over a decade, visual representation learning was held back by the canonical **ImageNet classification paradigm**:
1. **Discrete Categorical Blindness**:
   Standard vision models were trained with Cross-Entropy to predict a 1-of-1,000 discrete integer index. A model had no semantic understanding of relationships between classes (e.g. knowing a "husky" is a type of "dog" or resembles a "wolf").
2. **Brittle Downstream Transfer (Zero-Shot Impossibility)**:
   Whenever an engineer wanted to deploy a vision model to a new task (e.g. medical radiology, autonomous vehicles, or robotics), they had to discard the final layer and annotate thousands of specialized images to train a new classifier head.
3. **Information Bottleneck of Manual Annotations**:
   Compressing an entire visual scene containing dozens of interacting objects and attributes into a single scalar category label throws away 99% of the scene's semantic information.

---

## 2. The Thought Experiment: Natural Language as the Supervision Medium

Radford et al. asked:
> *Can we train visual models by teaching them to understand images paired with **unconstrained natural language text** scraped from the internet (400M pairs)?*

```
          Images I_1 ... I_N                  Texts T_1 ... T_N
                  │                                   │
                  ▼                                   ▼
          [ Image Encoder ]                   [ Text Encoder ]
                  │                                   │
                  ▼                                   ▼
             Features h_i                        Features h_t
                  │                                   │
                  ▼                                   ▼
          [ Proj: W_i -> D ]                  [ Proj: W_t -> D ]
                  │                                   │
                  ▼                                   ▼
           Norm Embeds I_e                     Norm Embeds T_e
                  │                                   │
                  └─────────────────┬─────────────────┘
                                    ▼
                     [ Cross-Modal Similarity Matrix ]
                               S = (I_e · T_e^T) * exp(τ)
                                    │
                  ┌─────────────────┴─────────────────┐
                  ▼                                   ▼
        Image-to-Text Loss:                 Text-to-Image Loss:
       CrossEntropy(S, labels)             CrossEntropy(S^T, labels)
                  └─────────────────┬─────────────────┘
                                    ▼
                      Total Loss = 0.5 * (L_i + L_t)
```

Instead of predicting exact text strings word-for-word (which is computationally expensive), CLIP uses a **symmetric cross-modal contrastive objective**:
1. Within an $N \times N$ batch of paired images and texts:
   - The $N$ diagonal entries $(I_i, T_i)$ represent the true positive multimodal matches.
   - The $N^2 - N$ off-diagonal entries represent negative distractor pairs.
2. Maximize cosine similarity between aligned image-text pairs while pushing unaligned pairs apart!
3. **Zero-Shot Classification via Text Prompting**:
   To classify an image into $K$ target categories without training:
   - Format category names into prompt sentences: `"a photo of a {category}"`.
   - Embed prompt sentences using the frozen text encoder.
   - The predicted class is simply the prompt embedding with the highest cosine similarity to the image embedding!

---

## 3. Mathematical Architecture & Step-by-Step Derivation

### 3.1 Multimodal Feature Extraction & Projection
Given a batch of $N$ image-text pairs $(I, T)$:
1. **Image Embedding**:
   $$h_i = f_{\text{img}}(I) \in \mathbb{R}^{N \times d_i}$$
   $$I_e = \text{L2\_Normalize}(h_i W_i) \in \mathbb{R}^{N \times D}$$
2. **Text Embedding**:
   $$h_t = f_{\text{txt}}(T) \in \mathbb{R}^{N \times d_t}$$
   $$T_e = \text{L2\_Normalize}(h_t W_t) \in \mathbb{R}^{N \times D}$$
   where $W_i \in \mathbb{R}^{d_i \times D}$ and $W_t \in \mathbb{R}^{d_t \times D}$ map both modalities into a shared joint embedding space of dimension $D$.

### 3.2 Learnable Log-Temperature Scaling
The cross-modal cosine similarity matrix is:
$$S_{j, k} = (I_{e, j} \cdot T_{e, k}) \cdot e^{\tau}$$
where $\tau$ is a learnable scalar parameter initialized to $\ln(1 / 0.07) \approx 2.659$, clamped at $\ln(100) \approx 4.605$ to prevent numerical instability in exponential logits.

### 3.3 Symmetric Cross-Entropy Loss
Because matching is bidirectional:
- **Image-to-Text Loss**:
  $$\mathcal{L}_i = -\frac{1}{N} \sum_{j=1}^N \log \frac{\exp(S_{j, j})}{\sum_{k=1}^N \exp(S_{j, k})}$$
- **Text-to-Image Loss**:
  $$\mathcal{L}_t = -\frac{1}{N} \sum_{k=1}^N \log \frac{\exp(S_{k, k})}{\sum_{j=1}^N \exp(S_{j, k})}$$
- **Total Symmetric CLIP Loss**:
  $$\mathcal{L} = \frac{\mathcal{L}_i + \mathcal{L}_t}{2}$$

---

## 4. Tensor Contracts & Architectural Blueprint

| Component | Input Shape | Output Shape | Notes |
| :--- | :--- | :--- | :--- |
| `CLIP.encode_image` | `images: (N, 3, H, W)` | `(N, embed_dim)` | L2-normalized image embeddings |
| `CLIP.encode_text` | `text_tokens: (N, L)` | `(N, embed_dim)` | L2-normalized text embeddings |
| `CLIP.forward` | `images: (N, 3, H, W)`, `text: (N, L)` | `(loss, logits_per_image, logits_per_text)` | Full symmetric multimodal loss |
| `zero_shot_classifier` | `classnames: List[str]`, `templates: List[str]` | `(num_classes, embed_dim)` | Normalized zero-shot category weights |

---

## 5. Downstream Unlocks

- **Paper #31 (Classifier-Free Diffusion Guidance / Stable Diffusion)**: CLIP text encoders provide the universal semantic conditioning vectors driving modern text-to-image diffusion models!
- **Vision-Language-Action Models (OpenVLA, PaLM-E, RT-2)**: Robotic foundation models adapt pre-trained CLIP backbones to ground visual robot camera feeds with natural language task instructions!
- **Zero-Shot Object Detection & Segmentation (OWL-ViT, CLIPSeg)**: Grounding open-vocabulary queries directly into spatial bounding boxes.

---

## Further Reading

For more, [visit this chat](https://chatgpt.com/c/6afc2a90-0760-83eb-95b4-a04eab597d52).
