# Paper 07: MobileNetV2: Inverted Residuals and Linear Bottlenecks
**Authors**: Mark Sandler, Andrew Howard, Menglong Zhu, Andrey Zhmoginov, Liang-Chieh Chen (2018)  
**Paper Link**: [arXiv:1801.04381](https://arxiv.org/abs/1801.04381)

---

## 1. The Historical Breaking Point (The Edge Robotics & Mobile Crisis)

By 2017, ResNet-50 and VGG-16 were the dominant backbones in computer vision. However, they were computationally prohibitive for real-time edge robotics, embedded microcontrollers, drones, and mobile devices:
* **ResNet-50**: $\approx 25.6\text{M}$ parameters, $3.8\text{B}$ FLOPs ($\sim 40\text{ms}$ on mobile CPU).
* **VGG-16**: $\approx 138\text{M}$ parameters, $15.5\text{B}$ FLOPs.

Standard 2D convolutions jointly compute **spatial correlations** (across $K \times K$ kernel pixels) and **channel correlations** (across $C_{\text{in}} \times C_{\text{out}}$ channels) in a single dense step:
$$\text{Cost}_{\text{standard}} = H \cdot W \cdot C_{\text{in}} \cdot C_{\text{out}} \cdot K^2$$

### MobileNetV1 & The Collapse of Low-Dimensional Manifolds
MobileNetV1 factorized standard convolution into **Depthwise Separable Convolutions**:
1. **Depthwise Conv ($K \times K$, groups=$C_{\text{in}}$)**: Filters each input channel independently ($H \cdot W \cdot C_{\text{in}} \cdot K^2$).
2. **Pointwise Conv ($1 \times 1$)**: Linearly combines channels ($H \cdot W \cdot C_{\text{in}} \cdot C_{\text{out}}$).

$$\text{Computation Ratio} = \frac{H \cdot W \cdot C_{\text{in}} \cdot K^2 + H \cdot W \cdot C_{\text{in}} \cdot C_{\text{out}}}{H \cdot W \cdot C_{\text{in}} \cdot C_{\text{out}} \cdot K^2} = \frac{1}{C_{\text{out}}} + \frac{1}{K^2} \approx \frac{1}{9}$$
A staggering **$8\times$ to $9\times$ reduction in FLOPs!**

**BUT MobileNetV1 had a fatal flaw**:
In small, efficient networks, intermediate channels are narrow (e.g. $C=16$ or $C=24$). Passing low-dimensional representations through **ReLU** $(\max(0, x))$ zeroes out all negative activations, **destroying the manifold structure**. The authors proved mathematically:
> *"An operation like ReLU will inevitably lose information if the input manifold is mapped into a low-dimensional subspace."*

---

## 2. The Thought Experiment: "If I Were Inventing It"

Sitting at the blackboard, Sandler et al. devised two breakthroughs:

### 1. Inverted Residuals: Expand First, Then Squeeze!
In classical ResNet bottlenecks:
$$\text{Wide input } (256) \xrightarrow{1\times 1} \text{Compressed bottleneck } (64) \xrightarrow{3\times 3} \text{Wide output } (256)$$
Shortcuts connect the *wide* representations.

Sandler flipped this inside out:
$$\text{Compact input } (32) \xrightarrow{1\times 1 \text{ expand } (t=6)} \text{High-dimensional space } (192) \xrightarrow{\text{Depthwise } 3\times 3} \xrightarrow{1\times 1 \text{ project}} \text{Compact output } (32)$$
Shortcuts connect the **compact bottlenecks**!

* Depthwise convolution performs rich non-linear filtering in the **high-dimensional expanded space** ($6\times$ channels).
* Memory footprint is drastically reduced because residual buffers store only compact low-dimensional representations between blocks!

### 2. The Linear Bottleneck (Never ReLU the Bottleneck!)
Because the output of the block is in a compact low-dimensional space, applying ReLU would collapse the manifold.
**Solution**: Remove ReLU from the final $1\times 1$ projection layer!
The projection is purely **linear**:
$$y = \text{BN}(\text{Conv1x1}(x)) \quad \text{(No ReLU!)}$$

---

## 3. Mathematical Derivation: The Inverted Residual Block

For input $x \in \mathbb{R}^{B \times C_{\text{in}} \times H \times W}$, expansion factor $t$ (default 6), and stride $s \in \{1, 2\}$:

1. **Expansion ($1 \times 1$ Conv)** (if $t > 1$):
   $$\tilde{x} = \text{ReLU6}(\text{BN}(\text{Conv1x1}(x))), \quad \tilde{x} \in \mathbb{R}^{B \times (t \cdot C_{\text{in}}) \times H \times W}$$
   *(Note: ReLU6$(\cdot) = \min(\max(0, \cdot), 6)$ prevents activation saturation in low-precision integer / FP16 quantization).*

2. **Spatial Filtering (Depthwise $3 \times 3$ Conv with `groups` = channels)**:
   $$z = \text{ReLU6}(\text{BN}(\text{Conv3x3}(\tilde{x}, \text{stride}=s, \text{groups}=t \cdot C_{\text{in}})))$$

3. **Linear Bottleneck Projection ($1 \times 1$ Conv)**:
   $$\hat{y} = \text{BN}(\text{Conv1x1}(z, \text{out\_channels}=C_{\text{out}})) \quad \text{\textbf{[LINEAR: NO RELU]}}$$

4. **Residual Shortcut**:
   $$y = \begin{cases} x + \hat{y} & \text{if } s = 1 \text{ and } C_{\text{in}} == C_{\text{out}} \\ \hat{y} & \text{otherwise} \end{cases}$$

---

## 4. Tensor Shape Contract (Full MobileNetV2 Pipeline)

Input image: $x \in \mathbb{R}^{B \times 3 \times 224 \times 224}$

| Stage | Operator | $t$ (Expansion) | $c$ (Channels Out) | $n$ (Blocks) | $s$ (Stride) | Output Tensor Shape |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **Stem** | Conv3x3, BN, ReLU6 | - | 32 | 1 | 2 | $(B, 32, 112, 112)$ |
| **Stage 1** | InvertedResidual | 1 | 16 | 1 | 1 | $(B, 16, 112, 112)$ |
| **Stage 2** | InvertedResidual | 6 | 24 | 2 | 2 | $(B, 24, 56, 56)$ |
| **Stage 3** | InvertedResidual | 6 | 32 | 3 | 2 | $(B, 32, 28, 28)$ |
| **Stage 4** | InvertedResidual | 6 | 64 | 4 | 2 | $(B, 64, 14, 14)$ |
| **Stage 5** | InvertedResidual | 6 | 96 | 3 | 1 | $(B, 96, 14, 14)$ |
| **Stage 6** | InvertedResidual | 6 | 160 | 3 | 2 | $(B, 160, 7, 7)$ |
| **Stage 7** | InvertedResidual | 6 | 320 | 1 | 1 | $(B, 320, 7, 7)$ |
| **Conv1x1** | Conv1x1, BN, ReLU6 | - | 1280 | 1 | 1 | $(B, 1280, 7, 7)$ |
| **Head** | AvgPool2d(7) $\to$ Dropout(0.2) $\to$ Linear | - | num_classes | - | - | $(B, \text{num\_classes})$ |

---

## 5. Downstream Unlocks in our 80-Paper Roadmap

MobileNetV2 unlocks real-time, low-latency vision for robot control:
* **Edge Object Detection (SSDLite / YOLO-Nano)**: Real-time 60+ FPS bounding box detection on embedded Jetson or Raspberry Pi robot controllers.
* **#25 R3M & Robot Representation Learning**: Lightweight sensory encoders for robotic manipulation.
* **On-Robot Policy Inference (ACT / Diffusion Policy)**: Running real-time closed-loop visuomotor policies on mobile manipulators where GPU memory is severely constrained.

## Further Reading

For more, [visit this chat](https://chatgpt.com/c/6ad5cf5c-d9d4-83ed-a759-6f0386c736a0).
