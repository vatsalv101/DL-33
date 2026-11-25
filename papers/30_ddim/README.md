# Paper 30: Denoising Diffusion Implicit Models (DDIM)

> **"Denoising diffusion probabilistic models (DDPMs) have achieved high quality image generation... However, sampling from DDPMs requires simulating a Markov chain for many steps (e.g. 1000 steps), which is computationally expensive. We propose Denoising Diffusion Implicit Models (DDIMs), a class of non-Markovian diffusion processes that have the same training objective as DDPMs but can produce high quality samples 10x to 50x faster."**
> — Jiaming Song, Chenlin Meng, Stefano Ermon (Stanford University, ICLR 2021)

---

## 1. The Breaking Point: The 1000-Step Bottleneck & Non-Invertibility

In 2020, DDPM (Ho et al.) demonstrated that diffusion models can match or outperform GANs in visual fidelity and mode coverage. However, DDPM suffered from two severe practical limitations:

```
DDPM Markov Chain (1000 Sequential Steps):
------------------------------------------
x_T ~ N(0, I) ---> x_999 ---> x_998 ---> ... ---> x_1 ---> x_0
      |              |              |              |
   U-Net          U-Net          U-Net          U-Net
  Forward        Forward        Forward        Forward
   Pass 1         Pass 2         Pass 3        Pass 1000

Total time per image: ~15 to 45 seconds!
In addition, random noise z ~ N(0, I) injected at EVERY step makes DDPM NON-INVERTIBLE!
Given an image x_0, you cannot recover its latent code x_T.
```

1. **Computational Inefficiency**: Sampling a single batch of images requires $T=1000$ sequential evaluations of a deep, multi-million parameter U-Net. Skipping steps in a Markov chain breaks the mathematical assumptions, degrading sample quality drastically.
2. **Absence of a Latent Representation**: Because random Gaussian noise $z_t \sim \mathcal{N}(0, I)$ is added at every reverse step, DDPM is stochastic. Two generations from the same initial $x_T$ take completely divergent paths. There is no bijective mapping between latent space and data space, making latent interpolation, real-image inversion, and controlled editing impossible.

---

## 2. The Thought Experiment: The Non-Markovian Family

Jiaming Song et al. asked a fundamental mathematical question:
*Does the forward diffusion process have to be Markovian?*

Recall that to train the U-Net denoiser $\epsilon_\theta(x_t, t)$ with the simplified objective:
$$L_{\text{simple}}(\theta) = \mathbb{E}_{t, x_0, \epsilon}\left[\|\epsilon - \epsilon_\theta(x_t, t)\|^2\right]$$
the **ONLY** requirement is the marginal distribution at step $t$:
$$q(x_t | x_0) = \mathcal{N}\left(x_t; \sqrt{\bar{\alpha}_t} x_0, (1 - \bar{\alpha}_t) I\right)$$

```
Key Insight:
Many DIFFERENT forward joint distributions q(x_{1:T} | x_0) share the EXACT SAME marginals q(x_t | x_0)!
Therefore, ANY model trained on DDPM can be reused for DDIM with ZERO retraining!
```

By designing a non-Markovian forward distribution where $x_{t-1}$ depends on **both** $x_t$ and $x_0$, we can choose how much stochastic noise is added at each reverse step. Setting the noise to zero transforms the generative process into a **deterministic Ordinary Differential Equation (Probability Flow ODE)**!

---

## 3. Mathematical Derivation: The Parameterized Inference Family

### 3.1 The Non-Markovian Forward Process

Song et al. define a family of inference distributions $q_\sigma(x_{1:T} | x_0)$ satisfying $q_\sigma(x_t | x_0) = \mathcal{N}(\sqrt{\bar{\alpha}_t}x_0, (1 - \bar{\alpha}_t)I)$:
$$q_\sigma(x_{t-1} | x_t, x_0) = \mathcal{N}\left( \sqrt{\bar{\alpha}_{t-1}} x_0 + \sqrt{1 - \bar{\alpha}_{t-1} - \sigma_t^2} \frac{x_t - \sqrt{\bar{\alpha}_t} x_0}{\sqrt{1 - \bar{\alpha}_t}}, \sigma_t^2 I \right)$$

where $\sigma_t$ is a free parameter controlling the stochasticity of the reverse transition:
$$\sigma_t = \eta \cdot \sqrt{\frac{1 - \bar{\alpha}_{t-1}}{1 - \bar{\alpha}_t}} \sqrt{1 - \frac{\bar{\alpha}_t}{\bar{\alpha}_{t-1}}}, \quad \eta \in [0, 1]$$

### 3.2 The Two Extremes:
1. **$\eta = 1$ (DDPM)**:
   $$\sigma_t = \tilde{\beta}_t^{1/2}$$
   The process becomes strictly Markovian, recovering the exact stochastic DDPM reverse step.
2. **$\eta = 0$ (DDIM)**:
   $$\sigma_t = 0$$
   The forward and reverse processes become **completely deterministic**!

---

### 3.3 The Deterministic Reverse Step ($\eta = 0$)

When $\sigma_t = 0$, the reverse transition is a deterministic function of $x_t$ and the predicted noise $\epsilon_\theta(x_t, t)$:
1. First, compute the **predicted clean image** $x_0$:
   $$\hat{x}_0 = \frac{x_t - \sqrt{1 - \bar{\alpha}_t} \epsilon_\theta(x_t, t)}{\sqrt{\bar{\alpha}_t}}$$
2. Next, calculate the **direction pointing to $x_t$**:
   $$d_t = \sqrt{1 - \bar{\alpha}_{t-1} - \sigma_t^2} \cdot \epsilon_\theta(x_t, t)$$
3. Compute $x_{t-1}$:
   $$x_{t-1} = \sqrt{\bar{\alpha}_{t-1}} \hat{x}_0 + d_t + \sigma_t \epsilon_t$$

When $\sigma_t = 0$:
$$x_{t-1} = \sqrt{\bar{\alpha}_{t-1}} \left( \frac{x_t - \sqrt{1 - \bar{\alpha}_t} \epsilon_\theta(x_t, t)}{\sqrt{\bar{\alpha}_t}} \right) + \sqrt{1 - \bar{\alpha}_{t-1}} \cdot \epsilon_\theta(x_t, t)$$

Notice: **No random noise is sampled!** The mapping from initial noise $x_T$ to data $x_0$ is a deterministic bijection.

---

### 3.4 Sub-Sequence Acceleration ($S \ll T$)

Because the process does not require simulating every consecutive integer step, we can select a subset of $S$ sampling timesteps:
$$\tau = [\tau_1, \tau_2, \dots, \tau_S] \subset [1, 2, \dots, T]$$
where $\tau_1 < \tau_2 < \dots < \tau_S = T$ (for instance, $S=20$ or $S=50$ linearly or quadratically spaced steps out of $1000$).

Sampling along sub-sequence $\tau$:
$$x_{\tau_{i-1}} = \sqrt{\bar{\alpha}_{\tau_{i-1}}} \left( \frac{x_{\tau_i} - \sqrt{1 - \bar{\alpha}_{\tau_i}} \epsilon_\theta(x_{\tau_i}, \tau_i)}{\sqrt{\bar{\alpha}_{\tau_i}}} \right) + \sqrt{1 - \bar{\alpha}_{\tau_{i-1}} - \sigma_{\tau_i}^2} \epsilon_\theta(x_{\tau_i}, \tau_i) + \sigma_{\tau_i} \epsilon$$

**Speedup**: From $1000$ steps down to $20$ or $50$ steps—a **$20\times$ to $50\times$ acceleration** with virtually zero perceptual degradation!

---

### 3.5 Exact Latent Inversion (DDIM Inversion)

Because the probability flow ODE is time-reversible, we can invert a real image $x_0$ into its unique latent representation $x_T$ by running the step forward in time:
$$x_{\tau_{i+1}} = \sqrt{\bar{\alpha}_{\tau_{i+1}}} \left( \frac{x_{\tau_i} - \sqrt{1 - \bar{\alpha}_{\tau_i}} \epsilon_\theta(x_{\tau_i}, \tau_i)}{\sqrt{\bar{\alpha}_{\tau_i}}} \right) + \sqrt{1 - \bar{\alpha}_{\tau_{i+1}}} \epsilon_\theta(x_{\tau_i}, \tau_i)$$

This enables:
- **Consistent Image Reconstruction**: Inverting $x_0 \to x_T$ and sampling $x_T \to x_0$ reproduces the exact original image!
- **Latent Space Semantic Interpolation**: Linear interpolation between two real image latents $x_T^{(1)}$ and $x_T^{(2)}$ produces smooth visual morphing.

---

## 4. Tensor Contracts & Dimension Transitions

```
Sub-sequence of S steps: tau = [tau_1, ..., tau_S] (e.g., length 20)
Initial Noise: x_tau_S ~ N(0, I), shape (B, C, H, W)
      |
      v
For step i = S down to 1:
      |
      +---> t_curr = tau_i,  t_prev = tau_{i-1} (or 0)
      |
      v
  U-Net Forward Pass:
    eps_pred = model(x_t, t_curr)  # Shape: (B, C, H, W)
      |
      v
  Compute predicted x_0:
    pred_x_0 = (x_t - sqrt(1 - alpha_bar_curr) * eps_pred) / sqrt(alpha_bar_curr)
    pred_x_0 = clamp(pred_x_0, -1, 1)
      |
      v
  Direction pointing to x_t:
    dir_xt = sqrt(1 - alpha_bar_prev - sigma^2) * eps_pred
      |
      v
  Next state:
    x_prev = sqrt(alpha_bar_prev) * pred_x_0 + dir_xt + sigma * eps
      |
      v
Final Generated Image x_0: (B, C, H, W) in [-1, 1]
```

---

## 5. Downstream Unlocks

1. **Classifier-Free Guidance (#31)**: Fast guided text-to-image synthesis in 20–50 steps.
2. **Latent Diffusion & Stable Diffusion (#32)**: The default standard sampler for Stable Diffusion v1/v2/XL.
3. **Diffusion-Based Image Editing & Inpainting (Prompt-to-Prompt, Null-Text Inversion)**: Modifying prompts while keeping DDIM inverted latents fixed to alter image semantics while preserving identity.
4. **Diffusion Policy Execution (#74, #75)**: High-frequency robot control requiring action generation in $< 50\text{ms}$.

---

## Further Reading
- Song, J., Meng, C., & Ermon, S. (2020). *Denoising Diffusion Implicit Models*. ICLR 2021.
- Dhariwal, P., & Nichol, A. (2021). *Diffusion Models Beat GANs on Image Synthesis*. NeurIPS 2021.
- [visit this chat](https://chatgpt.com/c/6b06b872-3d20-83eb-9443-0503a15a65cf)
