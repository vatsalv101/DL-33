# Paper 29: Denoising Diffusion Probabilistic Models (DDPM)

> **"We show that progress in diffusion probabilistic models can be achieved through a connection between diffusion models and denoising score matching with Langevin dynamics... We present high quality image synthesis results that are competitive with GANs while providing mode coverage and a simple training objective."**
> — Jonathan Ho, Ajay Jain, Pieter Abbeel (UC Berkeley, NeurIPS 2020)

---

## 1. The Breaking Point: The Generative Learning Trilemma

Prior to 2020, deep generative modeling was constrained by a fundamental structural trade-off known as the **Generative Learning Trilemma**:

```
                       The Generative Trilemma
                                Fast Sampling
                                     /\
                                    /  \
                                   /    \
                                  / GANs \
                                 /________\
             Sample Quality    /            \    Mode Diversity
             (Fidelity)       /______________\   (Log-Likelihood)
                             VAEs / Flows      Autoregressive
```

1. **GANs**: Fast $O(1)$ sampling and sharp outputs, but notoriously unstable adversarial minimax dynamics, prone to mode collapse, and incapable of full density coverage.
2. **VAEs & Normalizing Flows**: Stable likelihood-based training with density coverage, but constrained by blurry MSE reconstructions (VAEs) or restrictive invertible Jacobian transformations (Flows).
3. **Autoregressive Models (PixelCNN / Transformers)**: High perceptual quality and stable likelihoods, but excruciatingly slow sequential sampling ($O(N)$ for $N$ pixels/tokens).

None of these paradigms offered **stable regression-based training, complete mode coverage, and state-of-the-art high-frequency perceptual fidelity**.

---

## 2. The Thought Experiment: Thermodynamic Time-Reversal

Imagine dropping ink into a glass of water.
- **Forward Process (Thermodynamics)**: As time proceeds, the ordered droplet of ink diffuses uniformly throughout the water via Brownian motion until it reaches thermodynamic maximum entropy (pure white noise). This forward process destroys information and requires **no learning**!
- **Reverse Process (Time-Reversal)**: Can we reverse time? If you could record the microscopic movement of every particle and know the exact displacement at each infinitesimal time step $t \to t - 1$, you could systematically push each particle backward, reassembling the uniform ink water back into the pristine droplet!

```
Data x_0 -------------------> x_1 ---------> ... ---------> x_T ~ N(0, I)
  (Pristine Image)      Forward Process q(x_t | x_{t-1})      (Pure Gaussian Noise)
                          (Fixed, No Learning)
                                  |
                                  v
Data x_0 <------------------- x_1 <--------- ... <--------- x_T ~ N(0, I)
  (Generated Image)     Reverse Process p_theta(x_{t-1} | x_t)  (Sample from Prior)
                          (Learned Neural Network)
```

By decomposing the generative task into $T=1000$ tiny denoising steps, the reverse transition $p_\theta(x_{t-1} | x_t)$ becomes approximately Gaussian, turning an impossible one-step generation problem into a sequence of tractable linear regressions!

---

## 3. Mathematical Derivation: Forward Schedule & Noise Prediction

### 3.1 The Forward Markovian Noising Process

Given a data distribution $x_0 \sim q(x)$, the forward process adds Gaussian noise according to a variance schedule $\beta_1, \beta_2, \dots, \beta_T$:
$$q(x_t | x_{t-1}) = \mathcal{N}\left(x_t; \sqrt{1 - \beta_t} x_{t-1}, \beta_t I\right)$$

Let $\alpha_t = 1 - \beta_t$ and $\bar{\alpha}_t = \prod_{s=1}^t \alpha_s$.
Using the additive property of independent Gaussians, we derive the **closed-form marginal distribution at arbitrary step $t$**:
$$x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1 - \bar{\alpha}_t} \epsilon, \quad \text{where } \epsilon \sim \mathcal{N}(0, I)$$

**Critical Property**: We do NOT need to simulate $t$ intermediate steps! We can sample $x_t$ at any random step $t$ in $O(1)$ time.

---

### 3.2 The Tractable Posterior $q(x_{t-1} | x_t, x_0)$

Although the unconditioned reverse step $q(x_{t-1} | x_t)$ is intractable, conditioning on $x_0$ makes it analytically tractable via Bayes' rule:
$$q(x_{t-1} | x_t, x_0) = \mathcal{N}\left(x_{t-1}; \tilde{\mu}_t(x_t, x_0), \tilde{\beta}_t I\right)$$

Expanding Gaussian densities yields:
$$\tilde{\mu}_t(x_t, x_0) = \frac{\sqrt{\bar{\alpha}_{t-1}} \beta_t}{1 - \bar{\alpha}_t} x_0 + \frac{\sqrt{\alpha_t}(1 - \bar{\alpha}_{t-1})}{1 - \bar{\alpha}_t} x_t$$
$$\tilde{\beta}_t = \frac{1 - \bar{\alpha}_{t-1}}{1 - \bar{\alpha}_t} \beta_t$$

---

### 3.3 Reparameterization as Noise Prediction $\epsilon_\theta(x_t, t)$

From our forward closed-form equation, we can express $x_0$ in terms of $x_t$ and the noise $\epsilon$:
$$x_0 = \frac{x_t - \sqrt{1 - \bar{\alpha}_t} \epsilon}{\sqrt{\bar{\alpha}_t}}$$

Substituting this into the posterior mean $\tilde{\mu}_t$:
$$\mu_\theta(x_t, t) = \frac{1}{\sqrt{\alpha_t}} \left( x_t - \frac{\beta_t}{\sqrt{1 - \bar{\alpha}_t}} \epsilon_\theta(x_t, t) \right)$$

Instead of predicting the clean image $x_0$ or the mean $\mu$, the neural network $\epsilon_\theta(x_t, t)$ simply **predicts the noise vector $\epsilon$ that was added to $x_0$**!

---

### 3.4 The Simplified Loss Function

Ho et al. showed that dropping the complex variational weighting coefficients in the Evidence Lower Bound (ELBO) yields superior visual quality:
$$L_{\text{simple}}(\theta) = \mathbb{E}_{t \sim [1, T], x_0, \epsilon \sim \mathcal{N}(0, I)} \left[ \left\| \epsilon - \epsilon_\theta\left( \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1 - \bar{\alpha}_t} \epsilon, t \right) \right\|_2^2 \right]$$

Training algorithm:
1. Sample $x_0 \sim q(x)$ and $t \sim \text{Uniform}(\{1, \dots, T\})$.
2. Sample $\epsilon \sim \mathcal{N}(0, I)$.
3. Take gradient step on $\nabla_\theta \|\epsilon - \epsilon_\theta(\sqrt{\bar{\alpha}_t} x_0 + \sqrt{1 - \bar{\alpha}_t}\epsilon, t)\|^2$.

---

### 3.5 The Reverse Sampling Algorithm

To generate an image from scratch:
1. Sample $x_T \sim \mathcal{N}(0, I)$.
2. For $t = T, T-1, \dots, 1$:
   - Sample $z \sim \mathcal{N}(0, I)$ if $t > 1$, else $z = 0$.
   - Compute:
     $$x_{t-1} = \frac{1}{\sqrt{\alpha_t}} \left( x_t - \frac{\beta_t}{\sqrt{1 - \bar{\alpha}_t}} \epsilon_\theta(x_t, t) \right) + \sigma_t z$$
     where $\sigma_t = \sqrt{\tilde{\beta}_t}$.
3. Return $x_0$.

---

## 4. Tensor Contracts & Architectural Topology

```
Input Image x_0: (B, C, H, W) in [-1, 1]
Timestep t: (B,) integers in [0, T-1]
      |
      +------------------------+
      |                        |
      v                        v
Sinusoidal Embedding:   Sample eps ~ N(0, I)
  t_emb: (B, D_emb)            |
      |                        v
      |                 q_sample(x_0, t, eps):
      |                   x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * eps
      |                   Shape: (B, C, H, W)
      |                        |
      +----------+-------------+
                 |
                 v
        [ U-Net Denoiser ]
         - Down Blocks with Timestep Injections
         - Mid Bottleneck Residual Block
         - Up Blocks with Skip Connections
                 |
                 v
        Predicted Noise eps_theta: (B, C, H, W)
                 |
                 v
        Loss = MSE(eps, eps_theta)
```

---

## 5. Downstream Unlocks

1. **Fast Sampling via DDIM (#30)**: Non-Markovian deterministic sampling skipping steps from 1000 down to 20–50 steps.
2. **Classifier-Free Guidance (CFG #31)**: Conditioning diffusion models on text prompts without external classifiers.
3. **Latent Diffusion Models (LDM / Stable Diffusion #32)**: Running DDPM in the latent space of a pre-trained VQ-VAE/VAE.
4. **Diffusion Policy for Robotics (#74, #75)**: Formulating multi-modal robot action trajectories as a denoising diffusion process over action horizons.

---

## Further Reading
- Ho, J., Jain, A., & Abbeel, P. (2020). *Denoising Diffusion Probabilistic Models*. NeurIPS 2020.
- Sohl-Dickstein, J., et al. (2015). *Deep Unsupervised Learning using Nonequilibrium Thermodynamics*. ICML 2015.
- [visit this chat](https://chatgpt.com/c/6b05a740-f7b0-83eb-8920-1ccfa2f1d5af)
- [For next chat](https://chatgpt.com/c/6b05bb0c-9534-83ed-b541-96f31d489066)
