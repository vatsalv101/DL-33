# Paper 05: Deep Residual Learning for Image Recognition (ResNet)
**Authors**: Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun (2015)  
**Paper Link**: [arXiv:1512.03385](https://arxiv.org/abs/1512.03385)

---

## 1. The Historical Breaking Point (The Degradation Problem)

After Kaiming Initialization (Paper #1) and Batch Normalization (Paper #3), deep networks (10 to 30 layers) could finally be trained with stable activation variances and non-zero gradients.

Naturally, researchers asked:
> *"If 20 layers are good, will 50 or 100 layers be twice as good?"*

They stacked more convolutional layers (creating "Plain-20" and "Plain-56" networks) and observed an unexpected catastrophe:

```
Training Error: Plain-56 >> Plain-20  (Much worse!)
Testing Error:  Plain-56 >> Plain-20  (Much worse!)
```

### Why this shocked the entire AI community:
1. **It was NOT Overfitting**: If a 56-layer network were overfitting, its *training* error would be near zero while test error exploded. Here, the *training error itself was far higher*.
2. **It was NOT Vanishing Gradients**: With Batch Normalization and Kaiming Init, backward gradients were demonstrably non-zero at all layers.

This anomaly became known as the **Degradation Problem**: deep plain networks are fundamentally difficult to optimize. In theory, a deeper model should have a solution space at least as good as a shallower model, because extra layers could simply learn the **identity mapping** ($H(x) = x$). Yet in practice, gradient descent through chained non-linearities struggles to find even the identity mapping!

---

## 2. The Thought Experiment: "If I Were Inventing It"

Sitting at the drawing board, Kaiming He and his team realized:
> *"Multiple stacked non-linear layers struggle to approximate identity mappings ($H(x) = x$). What if we don't force them to learn $H(x)$ directly? What if we let them learn the residual difference instead?"*

Instead of asking layers to output the full desired mapping:
$$H(x)$$

Define the residual function:
$$F(x) := H(x) - x$$

Rearrange to get:
$$H(x) = F(x) + x$$

### The Intuition:
* If the identity mapping is optimal, the optimizer simply needs to push the weights of $F(x)$ toward **zero** ($F(x) \to 0$).
* Pushing weights to zero with weight decay / L2 regularization is natural for stochastic gradient descent.
* In a plain network, learning $W \cdot x = x$ requires setting weight matrices to exact identity matrices $I$, which SGD almost never discovers across cascaded ReLU activations.

### The Gradient Highway
When computing backpropagation through a residual block $y = F(x) + x$:
$$\frac{\partial \mathcal{L}}{\partial x} = \frac{\partial \mathcal{L}}{\partial y} \cdot \frac{\partial y}{\partial x} = \frac{\partial \mathcal{L}}{\partial y} \left( \frac{\partial F(x)}{\partial x} + I \right) = \frac{\partial \mathcal{L}}{\partial y} \frac{\partial F(x)}{\partial x} + \frac{\partial \mathcal{L}}{\partial y}$$

Notice the second term: $+\frac{\partial \mathcal{L}}{\partial y}$!
Even if the gradient of the residual path $\frac{\partial F(x)}{\partial x}$ approaches zero, the gradient flows directly back to earlier layers unimpeded through the identity shortcut $I$. This created an unobstructed **gradient highway** that enabled training networks with **152 to 1,000+ layers** for the first time in human history.

---

## 3. Mathematical Derivation & Architectures

ResNet uses two primary residual block designs:

### 1. BasicBlock (for ResNet-18 and ResNet-34)
Designed for shallower networks with moderate channel counts:
* **Residual path $F(x)$**:
  $$\text{Conv3x3}(C_{\text{in}} \to C_{\text{out}}, \text{stride}=s) \to \text{BatchNorm} \to \text{ReLU} \to \text{Conv3x3}(C_{\text{out}} \to C_{\text{out}}, \text{stride}=1) \to \text{BatchNorm}$$
* **Shortcut path**:
  * If $C_{\text{in}} == C_{\text{out}}$ and $\text{stride} == 1$: identity shortcut $x$.
  * If dimensions change ($C_{\text{in}} \ne C_{\text{out}}$ or $\text{stride} > 1$): projection shortcut $\text{Conv1x1}(C_{\text{in}} \to C_{\text{out}}, \text{stride}=s) \to \text{BatchNorm}$.
* **Addition & Activation**:
  $$y = \text{ReLU}(F(x) + \text{shortcut}(x))$$

### 2. Bottleneck Block (for ResNet-50, ResNet-101, ResNet-152)
To train 50+ layer networks efficiently without ballooning computational cost (FLOPs), the authors introduced a 3-layer **bottleneck**:
* **$1 \times 1$ Conv**: Reduces channel dimension by $4\times$ (compression).
* **$3 \times 3$ Conv**: Performs spatial filtering in the low-dimensional bottleneck.
* **$1 \times 1$ Conv**: Restores and expands channels by $4\times$ ($C_{\text{out}} = 4 \times \text{planes}$).
* **Expansion factor**: `expansion = 4`.

```
        BasicBlock (18/34)               Bottleneck (50/101/152)
           Input: [C]                          Input: [4*C]
            │       ╲                            │        ╲
       Conv 3x3      │                      Conv 1x1 (C)   │
          BN         │ (Shortcut)                BN        │ (Shortcut)
         ReLU        │                          ReLU       │
          │          │                           │         │
       Conv 3x3      │                      Conv 3x3 (C)   │
          BN         │                           BN        │
          │         ╱                           ReLU       │
          ▼        ╱                             │         │
         [+] ◄────┘                         Conv 1x1 (4*C) │
          │                                      BN        │
         ReLU                                    │        ╱
          │                                      ▼       ╱
       Output: [C]                              [+] ◄───┘
                                                 │
                                                ReLU
                                                 │
                                            Output: [4*C]
```

---

## 4. Tensor Shape Contract (Full ResNet Pipeline)

For an input image tensor $x \in \mathbb{R}^{B \times 3 \times 224 \times 224}$:

| Stage | Operations | Output Tensor Shape |
| :--- | :--- | :--- |
| **Stem** | Conv $7\times 7$, stride 2, padding 3 $\to$ BN $\to$ ReLU $\to$ MaxPool $3\times 3$, stride 2, pad 1 | $(B, 64, 56, 56)$ |
| **Stage 1 (`layer1`)** | $N_1$ blocks, channels $= 64$, stride $= 1$ | $(B, 64 \times E, 56, 56)$ |
| **Stage 2 (`layer2`)** | $N_2$ blocks, channels $= 128$, first block stride $= 2$ | $(B, 128 \times E, 28, 28)$ |
| **Stage 3 (`layer3`)** | $N_3$ blocks, channels $= 256$, first block stride $= 2$ | $(B, 256 \times E, 14, 14)$ |
| **Stage 4 (`layer4`)** | $N_4$ blocks, channels $= 512$, first block stride $= 2$ | $(B, 512 \times E, 7, 7)$ |
| **Head** | AdaptiveAvgPool2d((1, 1)) $\to$ Flatten $\to$ Linear($512 \times E$, num_classes) | $(B, \text{num\_classes})$ |

*(Where $E = 1$ for `BasicBlock` and $E = 4$ for `Bottleneck`)*.

---

## 5. Downstream Unlocks in our 80-Paper Curriculum

Implementing ResNet unlocks:
* **#6 U-Net**: ResNet encoder backbones for diffusion and segmentation.
* **#7 MobileNetV2**: Inverted residuals with linear bottlenecks.
* **#17 DETR**: End-to-end Transformer object detection with ResNet-50 visual feature backbone.
* **#20 SimCLR / #21 MoCo**: Contrastive self-supervised visual representation learning.
* **#24 CLIP**: OpenAI CLIP image encoder backbone (alongside ViT).
* **#25 R3M**: Universal visual representations for robot manipulation.


## Further Reading

For more, [visit this chat](https://chatgpt.com/c/6ad09c30-82b8-83eb-90f0-aa89205a5749).