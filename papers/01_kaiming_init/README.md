# Paper 01: Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification
**Authors**: Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun (2015)  
**Paper Link**: [arXiv:1502.01852](https://arxiv.org/abs/1502.01852)

---

## 1. The Historical Breaking Point (What Broke Before This?)

In 2010–2014, deep neural networks were initialized using **Xavier (Glorot) Initialization**:
$$\text{Var}[W] = \frac{2}{n_{\text{in}} + n_{\text{out}}}$$

Xavier initialization was derived under the assumption that activation functions are **linear or symmetric around zero** (like $\tanh(x)$ or the identity).

When the deep learning revolution switched to the **Rectified Linear Unit (ReLU)**:
$$f(x) = \max(0, x)$$
training deep networks suddenly broke down. If you initialized a 20-to-30 layer ReLU network with Xavier initialization:
1. Every ReLU layer sets approximately half of its inputs to zero ($x < 0 \to 0$).
2. The output variance drops by a factor of $2$ at every single layer:
   $$\text{Var}[y_l] \approx \left(\frac{1}{2}\right) \text{Var}[y_{l-1}]$$
3. By layer 30, the signal variance is scaled by:
   $$\left(\frac{1}{2}\right)^{30} \approx 9.3 \times 10^{-10}$$
4. The signal completely vanishes. Activations collapse to zero, backward gradients underflow, and deep networks cannot learn at all without artificial tricks like multi-stage greedy pretraining.

---

## 2. The Thought Experiment: "If I Were Inventing It"

Imagine you are sitting at your desk looking at a 30-layer network whose forward signal vanishes. You trace the signal layer by layer:

1. **The Culprit**: At each layer, ReLU zeroes out all negative activations. If inputs are roughly symmetric around zero, half the distribution is discarded.
2. **The Question**: If the activation function slices the variance in half at every layer, how do we prevent the signal from dying?
3. **The Solution**: We must double the variance of the weights!
   $$\text{Var}[W] = \frac{2}{n_{\text{in}}}$$
   By injecting twice the variance into the weight initialization, each linear layer amplifies the variance by $2\times$, which perfectly cancels out the $1/2\times$ reduction caused by the ReLU gate.

Now, whether the network has 5 layers or 50 layers, the signal variance remains constant ($\approx 1.0$) from the input layer all the way to the final classification logits!

---

## 3. Mathematical Derivation (Step-by-Step)

### Forward Pass Variance Preservation

Consider a linear or convolutional layer:
$$y_l = W_l x_l + b_l$$
where $W_l \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}$, $x_l \in \mathbb{R}^{d_{\text{in}}}$, and $b_l = 0$. Let $n_l = d_{\text{in}}$ (for Conv2d, $n_l = C_{\text{in}} \cdot k_h \cdot k_w$).

Assuming $W_l$ and $x_l$ are independent, and elements of $W_l$ have zero mean ($\mathbb{E}[w_l] = 0$):
$$\text{Var}[y_l] = n_l \text{Var}[w_l] \mathbb{E}[x_l^2]$$

Now, $x_l$ is the output of the previous activation: $x_l = f(y_{l-1}) = \max(0, y_{l-1})$.  
Assuming $y_{l-1}$ has a symmetric distribution around zero with zero mean:
$$\mathbb{E}[x_l^2] = \int_{-\infty}^{\infty} (\max(0, y))^2 p(y) \, dy = \int_{0}^{\infty} y^2 p(y) \, dy = \frac{1}{2} \mathbb{E}[y_{l-1}^2] = \frac{1}{2} \text{Var}[y_{l-1}]$$

Substituting back into our variance equation:
$$\text{Var}[y_l] = n_l \text{Var}[w_l] \cdot \left(\frac{1}{2} \text{Var}[y_{l-1}]\right) = \left(\frac{1}{2} n_l \text{Var}[w_l]\right) \text{Var}[y_{l-1}]$$

To ensure the signal variance does not explode or vanish across $L$ layers:
$$\frac{1}{2} n_l \text{Var}[w_l] = 1 \implies \text{Var}[w_l] = \frac{2}{n_l}$$

For a Gaussian distribution $\mathcal{N}(0, \sigma^2)$:
$$\sigma = \sqrt{\frac{2}{n_{\text{in}}}}$$

For a Uniform distribution $\mathcal{U}(-a, a)$ with variance $\sigma^2 = \frac{a^2}{3}$:
$$\text{bound} = \sqrt{\frac{6}{n_{\text{in}}}}$$

### Generalization to LeakyReLU (PReLU)
If $f(x) = \max(0, x) + a \min(0, x)$, then $\mathbb{E}[x^2] = \frac{1 + a^2}{2} \text{Var}[y]$, yielding:
$$\text{Var}[w_l] = \frac{2}{(1 + a^2) n_l}$$

---

## 4. Tensor Shape Contract

| Parameter / Layer | Shape | $\text{fan}_{\text{in}}$ | $\text{fan}_{\text{out}}$ |
| :--- | :--- | :--- | :--- |
| **Linear** | $(d_{\text{out}}, d_{\text{in}})$ | $d_{\text{in}}$ | $d_{\text{out}}$ |
| **Conv2d** | $(C_{\text{out}}, C_{\text{in}}, k_h, k_w)$ | $C_{\text{in}} \cdot k_h \cdot k_w$ | $C_{\text{out}} \cdot k_h \cdot k_w$ |
| **Conv1d** | $(C_{\text{out}}, C_{\text{in}}, k)$ | $C_{\text{in}} \cdot k$ | $C_{\text{out}} \cdot k$ |

---

## 5. Downstream Unlocks (The 80-Paper Roadmap)

* **Immediate Dependencies**: None (Rank #1, the root of all training).
* **Direct Unlocks**:
  * **#5 ResNet**: Deep residual networks rely on Kaiming init to train 50–152 layer models.
  * **#6 U-Net**: Encoder-decoder convolutional blocks initialized with variance preservation.
  * **#9 PointNet**: Deep shared MLPs on unordered 3D point clouds.
  * **#34–44 RL Critics & Actors**: Stable Q-networks and policy networks in DQN, TD3, and SAC.
