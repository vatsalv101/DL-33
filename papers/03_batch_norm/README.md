# Paper 03: Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift
**Authors**: Sergey Ioffe & Christian Szegedy (2015)  
**Paper Link**: [arXiv:1502.03167](https://arxiv.org/abs/1502.03167)

---

## 1. The Historical Breaking Point (What Broke Before This?)

In 2012–2015, training deep neural networks was notoriously slow, brittle, and unstable.

The culprit was **Internal Covariate Shift**:
* As early layers update their weights, the distribution of their outputs changes.
* Layer 15 constantly has the rug pulled out from under it—every parameter update in layers 1–14 changes the mean and variance of layer 15's inputs.
* To prevent activations from exploding or saturating non-linearities (like sigmoid/tanh), practitioners were forced to:
  1. Use microscopic learning rates ($\eta \le 10^{-4}$), making training painfully slow.
  2. Rely on ultra-fragile weight initialization.
  3. Avoid saturating non-linearities entirely.

---

## 2. The Thought Experiment: "If I Were Inventing It"

Sitting at your desk, you ask:
> *"Why don't we force every layer's inputs to have zero mean and unit variance across the mini-batch?"*

$$\hat{x} = \frac{x - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}}$$

### The Catch (Why Naive Normalization Fails)
If you simply normalize $x \to \hat{x}$, **you restrict the layer's capacity to represent functions!**
* For Sigmoid, constraining inputs to $\mathcal{N}(0, 1)$ forces all values into the linear middle zone $[-1, 1]$. The network can never use the non-linear curvature of the activation function!

### The Solution: Learnable Affine Transform ($\gamma$ and $\beta$)
Introduce two learnable parameters per channel: **$\gamma$ (scale)** and **$\beta$ (shift)**:
$$y = \gamma \hat{x} + \beta$$

* The network standardizes the inputs first to stabilize dynamics.
* Then, if optimal representation requires mean $2.0$ and variance $4.0$, gradient descent simply learns $\beta = 2.0$ and $\gamma = 2.0$.
* The network can even learn $\gamma = \sqrt{\sigma_B^2}$ and $\beta = \mu_B$ to completely undo the normalization if that is what the task requires!

### The Inference Dilemma
During deployment or evaluation (`eval()`), you might process a single image ($B=1$). You cannot compute batch variance from 1 sample ($\sigma^2 = 0$).
* **Fix**: During training, track a running mean and running variance using Exponential Moving Average (EMA).
* During `eval()`, freeze all statistics and normalize using the running buffers!

---

## 3. Mathematical Derivation (Step-by-Step)

### Step 1: Mini-Batch Statistics (Training Mode)
For a batch $B$ of feature vectors:
$$\mu_B = \frac{1}{m} \sum_{i=1}^m x_i$$
$$\sigma_B^2 = \frac{1}{m} \sum_{i=1}^m (x_i - \mu_B)^2$$
*(For Conv2d with shape $(B, C, H, W)$, pooling occurs over $m = B \times H \times W$ per channel $C$)*.

### Step 2: Normalization
$$\hat{x}_i = \frac{x_i - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}}$$
where $\epsilon = 10^{-5}$ prevents division by zero.

### Step 3: Scale and Shift
$$y_i = \gamma \hat{x}_i + \beta$$
where $\gamma$ is initialized to $\mathbf{1}$ and $\beta$ is initialized to $\mathbf{0}$.

### Step 4: Running Buffer Updates (EMA)
During training, update running buffers without computing gradients:
$$\mu_{\text{running}} \leftarrow (1 - \text{momentum}) \cdot \mu_{\text{running}} + \text{momentum} \cdot \mu_B$$
$$\sigma^2_{\text{running}} \leftarrow (1 - \text{momentum}) \cdot \sigma^2_{\text{running}} + \text{momentum} \cdot \sigma_B^2$$
*(PyTorch convention: default `momentum = 0.1`)*.

During `eval()` mode:
$$\hat{x} = \frac{x - \mu_{\text{running}}}{\sqrt{\sigma^2_{\text{running}} + \epsilon}}, \quad y = \gamma \hat{x} + \beta$$

---

## 4. Tensor Shape Contract

### BatchNorm1d
* **Input**: $(B, C)$ or $(B, C, L)$
* **Normalized over**: Batch dimension $B$ (and sequence length $L$)
* **Parameters**: $\gamma \in \mathbb{R}^C$, $\beta \in \mathbb{R}^C$
* **Buffers**: `running_mean` $\in \mathbb{R}^C$, `running_var` $\in \mathbb{R}^C$
* **Output**: Same shape as input

### BatchNorm2d
* **Input**: $(B, C, H, W)$
* **Normalized over**: Spatial & batch dimensions $(B, H, W)$ per channel $C$
* **Parameters**: $\gamma \in \mathbb{R}^C$, $\beta \in \mathbb{R}^C$
* **Buffers**: `running_mean` $\in \mathbb{R}^C$, `running_var` $\in \mathbb{R}^C$
* **Output**: $(B, C, H, W)$

---

## 5. Downstream Unlocks (The 80-Paper Roadmap)

* **Immediate Dependencies**: Built on PyTorch tensor primitives.
* **Direct Unlocks**:
  * **#5 ResNet**: Every residual block requires BatchNorm2d after each $3\times 3$ convolution.
  * **#6 U-Net**: Feature map normalization in contracting and expansive paths.
  * **#7 MobileNetV2**: Inverted bottleneck blocks with BatchNorm2d.
  * **#9 PointNet**: 1D shared MLPs processing 3D point clouds rely on BatchNorm1d.
