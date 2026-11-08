# Paper #17: End-to-End Object Detection with Transformers (DETR)

> **Carion, N., Massa, F., Synnaeve, G., Uszkoreit, J., Kirillov, A., & Zagoruyko, S. (2020 / ECCV 2020)**  
> *Facebook AI Research (FAIR)*  
> [arXiv:2005.12872](https://arxiv.org/abs/2005.12872)

---

## 1. Historical Breaking Point: Handcrafted Heuristics in 2D Detection

Before DETR, modern object detection (Faster R-CNN, SSD, YOLO) was hindered by complex hand-engineered surrogate tasks and post-processing heuristics:
1. **Prior Anchor Boxes**:  
   Detectors required dense grids of thousands of candidate bounding boxes spanning arbitrary scales, aspect ratios, and spatial positions. Tuning anchor geometries was brittle and dataset-dependent.
2. **Heuristic Label Assignment**:  
   Ground-truth objects were assigned to anchors via arbitrary IoU threshold rules (e.g. $\text{IoU} > 0.7$ positive, $< 0.3$ negative). This introduced severe class imbalance and conflicting assignment ambiguities for small or overlapping objects.
3. **Non-Maximum Suppression (NMS)**:  
   Because thousands of redundant anchors fire for a single object, inference depended on greedy Non-Maximum Suppression to discard duplicates. NMS is non-differentiable, parameter-sensitive, and breaks end-to-end gradient optimization.

---

## 2. The Thought Experiment: Direct Set Prediction & Hungarian Bipartite Matching

Carion et al. asked a fundamental question:
> *Can we cast object detection as a direct **set prediction** problem without anchors or NMS?*

Imagine a model that always outputs a fixed-size set of $N$ predictions:
$$\hat{y} = \{(\hat{c}_i, \hat{b}_i)\}_{i=1}^N$$
where $N$ is set larger than the typical maximum number of objects in an image (e.g., $N = 100$).
- $\hat{c}_i \in [0, 1]^{K+1}$: Softmax class probability distribution over $K$ object classes plus a special "no object" background class $\varnothing$.
- $\hat{b}_i \in [0, 1]^4$: Normalized bounding box coordinates $(x_{\text{center}}, y_{\text{center}}, w, h)$ predicted via a sigmoid activation.

### The Bipartite Matching Objective
To train with direct sets where predictions are unordered, ground truth $y$ is padded with $\varnothing$ up to size $N$.  
We then find the optimal one-to-one bipartite matching permutation $\hat{\sigma} \in \mathfrak{S}_N$ that minimizes the matching cost:

$$\hat{\sigma} = \arg\min_{\sigma \in \mathfrak{S}_N} \sum_{i=1}^N \mathcal{L}_{\text{match}}(y_i, \hat{y}_{\sigma(i)})$$

where for each ground truth object $y_i = (c_i, b_i)$:
$$\mathcal{L}_{\text{match}}(y_i, \hat{y}_{\sigma(i)}) = - \hat{p}_{\sigma(i)}(c_i) + \mathbb{I}_{\{c_i \ne \varnothing\}} \mathcal{L}_{\text{box}}(b_i, \hat{b}_{\sigma(i)})$$

This assignment can be solved in $O(N^3)$ polynomial time using the **Hungarian Algorithm** (`scipy.optimize.linear_sum_assignment`). Once matched, the loss function updates parameters using the optimal assignment $\hat{\sigma}$.

---

## 3. Mathematical Architecture & Step-by-Step Derivation

### 3.1 Bounding Box Losses: $L_1$ + Generalized IoU (GIoU)
Predicting bounding box coordinates using only $L_1$ loss scales with box size: small boxes incur small absolute loss errors even when misaligned, while large boxes incur huge errors.

DETR combines scale-invariant **Generalized IoU (GIoU)** loss with $L_1$:
$$\mathcal{L}_{\text{box}}(b, \hat{b}) = \lambda_{\text{L1}} \|b - \hat{b}\|_1 + \lambda_{\text{giou}} \mathcal{L}_{\text{giou}}(b, \hat{b})$$

#### Generalized IoU Formulation
For two bounding boxes $A$ and $B$, let $C$ be the smallest convex box containing both $A$ and $B$:
$$\text{IoU} = \frac{|A \cap B|}{|A \cup B|}, \qquad \text{GIoU} = \text{IoU} - \frac{|C \setminus (A \cup B)|}{|C|}$$
$$\mathcal{L}_{\text{giou}}(A, B) = 1 - \text{GIoU}(A, B)$$
- When boxes overlap perfectly: $\text{GIoU} = 1 \implies \mathcal{L}_{\text{giou}} = 0$.
- When boxes do not overlap: $\text{IoU} = 0$, but GIoU provides meaningful gradient signals directing the predicted box toward the target box!

### 3.2 DETR Architectural Pipeline

```
Image: (B, 3, H, W)
      │
      ▼
[CNN Backbone] ───────────────────────────> Feature Map: (B, C, H', W')
      │
      ▼
[1x1 Conv Projection] ────────────────────> (B, d_model, H', W')
      │
      ▼
Flatten Spatial Dims + Add 2D Positional Encodings
      │
      ▼
[Transformer Encoder (x L_enc)] ─────────> Memory: (B, H'*W', d_model)
      │
      ▼
[Transformer Decoder (x L_dec)]
(Cross-Attention: Queries = Object Queries (N, d_model), Keys/Values = Memory)
      │
      ▼
Output Embeddings: (B, N, d_model)
      │
      ├───> [Linear(d_model, num_classes + 1)] ────> Class Logits: (B, N, num_classes + 1)
      │
      └───> [3-Layer MLP + Sigmoid] ───────────────> Bounding Boxes: (B, N, 4)
```

### 3.3 Object Queries
The Transformer decoder is prompted with $N$ **learnable Object Queries** $\mathbf{q} \in \mathbb{R}^{N \times d_{\text{model}}}$.
- Through cross-attention, each query independently inspects the encoder memory representations.
- Through decoder self-attention, the object queries interact with one another, preventing duplicate detections and learning unique role specializations (e.g. "query 12 looks for large objects on the bottom left").

---

## 4. Tensor Contracts & Architectural Blueprint

| Component | Input Shape | Output Shape | Notes |
| :--- | :--- | :--- | :--- |
| `box_cxcywh_to_xyxy` | `(..., 4)`: $[x_c, y_c, w, h]$ | `(..., 4)`: $[x_1, y_1, x_2, y_2]$ | Normalized in $[0, 1]$ |
| `generalized_box_iou` | `boxes1: (N, 4)`, `boxes2: (M, 4)` | `(N, M)` | GIoU values in $[-1, 1]$ |
| `HungarianMatcher` | `outputs`, `targets` | `List[Tuple[Tensor, Tensor]]` | Optimal indices per batch |
| `SetCriterion` | `outputs`, `targets` | Scalar loss dictionary | Classification + L1 + GIoU |
| `DETR` | `images: (B, 3, H, W)` | `{"pred_logits": (B, N, K+1), "pred_boxes": (B, N, 4)}` | Full end-to-end detector |

---

## 5. Downstream Unlocks

- **Deformable DETR & RT-DETR**: Accelerating convergence and real-time detection via multi-scale sparse sampling.
- **Mask2Former**: Universal image segmentation framing semantic, instance, and panoptic segmentation as set queries.
- **Paper #72 (Action Chunking with Transformers / ACT)**: The foundational imitation learning policy architecture for bimanual robot manipulation (ALOHA) adapts DETR's object query decoder to predict sequences of future robot action chunks!

---

## Further Reading

For more, [visit this chat](https://chatgpt.com/c/6af067a9-a134-83eb-ab26-f13e8d5b64c6).
Next part, [visit this chat](https://chatgpt.com/c/6af0862a-3abc-83ed-b607-99d54efde82a).
Code part, [visit this chat](https://chatgpt.com/c/6af09c3d-69f4-83eb-947d-394916b2c014).
