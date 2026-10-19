# Paper 02: Adam & AdamW: Stochastic Optimization with Decoupled Weight Decay
**Authors**: Diederik P. Kingma & Jimmy Ba (2014) / Ilya Loshchilov & Frank Hutter (2017)  
**Paper Links**: [arXiv:1412.6980 (Adam)](https://arxiv.org/abs/1412.6980) | [arXiv:1711.05101 (AdamW)](https://arxiv.org/abs/1711.05101)

---

## 1. The Historical Breaking Point (What Broke Before This?)

In 2012–2014, deep learning models relied on Stochastic Gradient Descent (SGD) with momentum:
$$\theta_t = \theta_{t-1} - \eta g_t + \beta (\theta_{t-1} - \theta_{t-2})$$

SGD has a fundamental limitation: **it applies a single global scalar learning rate $\eta$ across all parameters**.
* Parameters with steep gradients oscillate wildly across ravines.
* Parameters with sparse or tiny gradients barely move.

1. **AdaGrad** tried to adapt learning rates coordinate-by-coordinate by dividing by the cumulative sum of historical squared gradients $\sum_{\tau=1}^t g_\tau^2$. But because the sum grows monotonically, the effective learning rate decays to zero, prematurely freezing training.
2. **RMSProp** solved AdaGrad's premature freezing by replacing the sum with an Exponential Moving Average (EMA) of squared gradients, but lacked momentum.
3. **The Adam Breakthrough & The Hidden Defect**: Kingma & Ba combined momentum (1st moment $m_t$) and coordinate scaling (2nd moment $v_t$). It quickly became the default deep learning optimizer—**until researchers added L2 regularization**.
   * In standard deep learning frameworks, L2 regularization ($\frac{1}{2}\lambda \|\theta\|^2$) was implemented by adding $\lambda \theta$ directly to the gradient: $\tilde{g}_t = g_t + \lambda \theta$.
   * Look at what happens when $\tilde{g}_t$ enters Adam's adaptive update:
     $$\theta_t = \theta_{t-1} - \frac{\eta}{\sqrt{v_t} + \epsilon} (m_t + \lambda \theta_{t-1})$$
   * The weight decay penalty was divided by $\sqrt{v_t}$!
   * Weights with large or frequent gradients had their weight decay heavily suppressed, while weights with tiny gradients were aggressively over-decayed. On deep Transformers and ConvNets, this crippled generalization.

---

## 2. The Thought Experiment: "If I Were Inventing It"

Sitting at the blackboard, you want an optimizer that:
1. **Tracks velocity (1st moment $m_t$)**: Keeps rolling through flat valleys via exponential smoothing on gradients (momentum).
2. **Normalizes gradient variance (2nd moment $v_t$)**: Scales each parameter's step inversely by the magnitude of its historical gradient fluctuations.
3. **Corrects for cold starts**: At step $t=1$, buffers initialized at $m_0 = 0$ and $v_0 = 0$ heavily underestimate the true moments. We must rescale them by dividing by $(1 - \beta^t)$ so the optimizer acts with full confidence from step 1.
4. **Decouples weight decay (AdamW)**: True weight decay is an exponential shrinkage on the weights themselves, independent of gradient scale. We must subtract $\eta \lambda \theta_{t-1}$ directly outside the adaptive division!

---

## 3. Mathematical Derivation (Step-by-Step)

### Step 1: Exponential Moving Averages
Given gradient $g_t = \nabla_\theta \mathcal{L}(\theta_{t-1})$:
$$m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t \quad (\text{1st Moment / Momentum})$$
$$v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2 \quad (\text{2nd Uncentered Moment / Variance})$$
Standard hyperparameters: $\beta_1 = 0.9$, $\beta_2 = 0.999$, $\epsilon = 10^{-8}$.

### Step 2: Exact Bias Correction Derivation
Unrolling $m_t$ starting from $m_0 = 0$:
$$m_t = (1 - \beta_1) \sum_{i=1}^t \beta_1^{t-i} g_i$$
Taking expectations under the assumption that true gradient $\mathbb{E}[g_i] \approx \mathbb{E}[g_t]$:
$$\mathbb{E}[m_t] = \mathbb{E}[g_t] (1 - \beta_1) \sum_{i=1}^t \beta_1^{t-i} = \mathbb{E}[g_t] (1 - \beta_1) \frac{1 - \beta_1^t}{1 - \beta_1} = \mathbb{E}[g_t] (1 - \beta_1^t)$$

Dividing by $(1 - \beta_1^t)$ recovers the unbiased estimator:
$$\hat{m}_t = \frac{m_t}{1 - \beta_1^t}$$
Similarly for the second moment:
$$\hat{v}_t = \frac{v_t}{1 - \beta_2^t}$$

### Step 3: Decoupled Weight Decay Update (AdamW)
Apply weight decay directly to the weights, completely decoupled from the gradient moments:
$$\theta_t = \theta_{t-1} - \eta \lambda \theta_{t-1} - \frac{\eta}{\sqrt{\hat{v}_t} + \epsilon} \hat{m}_t$$

---

## 4. Tensor Shape Contract

For each parameter tensor $\theta \in \mathbb{R}^{d_1 \times \dots \times d_k}$:

| State Variable | Shape | Initial Value | Memory Footprint |
| :--- | :--- | :--- | :--- |
| **`step`** | Scalar $(1,)$ | $0$ | Int64 |
| **`exp_avg` ($m_t$)** | $(d_1, \dots, d_k)$ | $\mathbf{0}$ | Float32 (same as $\theta$) |
| **`exp_avg_sq` ($v_t$)** | $(d_1, \dots, d_k)$ | $\mathbf{0}$ | Float32 (same as $\theta$) |

Total optimizer memory overhead: **$2\times$ model parameter size**.

---

## 5. Downstream Unlocks (The 80-Paper Roadmap)

* **Immediate Dependencies**: Built directly on PyTorch tensor operations.
* **Direct Unlocks**:
  * **All 78 Downstream Papers (#3–#80)**: AdamW is the universal optimization engine for Transformers (#11–18), Self-Supervised Encoders (#19–25), Diffusion Models (#29–33), Deep RL Critics & Actors (#34–44), and Vision-Language-Action Models (#70–80).

## Further Reading

For more, [visit this chat](https://chatgpt.com/c/6acca609-c3e4-83eb-b4b2-aed54a1b844e).