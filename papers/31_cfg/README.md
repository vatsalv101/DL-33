# Paper 31: Classifier-Free Diffusion Guidance (CFG)

> **"Classifier guidance is a recently developed technique for diffusion models... While classifier guidance is effective, it complicates the training pipeline because it requires training a separate classifier model... We show that classifier guidance can be performed without a classifier by jointly training a conditional and unconditional diffusion model with a single neural network."**
> — Jonathan Ho & Tim Salimans (NeurIPS 2022 Workshop / arXiv 2022)

---

## 1. The Breaking Point: The Fragility of Classifier Guidance

In 2021, Dhariwal & Nichol demonstrated that diffusion models could surpass GANs in image quality by steering the diffusion process using the gradients of a pre-trained image classifier:
$$\nabla_{x_t} \log p(x_t | y) = \nabla_{x_t} \log p(x_t) + s \cdot \nabla_{x_t} \log p(y | x_t)$$
where $s > 1$ is the guidance scale factor.

```
Classifier Guidance Pipeline (Fragile & Complex):
-------------------------------------------------
Noisy Image x_t ---------> [ Pre-trained Noisy Classifier ] ---------> Grad_xt log p(y | x_t)
                                                                           |
                                                                           v
Noisy Image x_t ---------> [ Diffusion U-Net Denoiser ] -------------> Score eps_theta
                                                                           |
                                                                           v
                                                          Combined Guided Score
```

While effective in principle, **Classifier Guidance suffered from three fatal engineering and theoretical flaws**:
1. **The Noisy Classifier Burden**: Standard ImageNet classifiers operate on clean images $x_0$. To guide diffusion at step $t \in [1, 1000]$, practitioners had to train a completely separate, specialized classifier on noisy images $x_t$ across all noise levels $\bar{\alpha}_t$.
2. **Adversarial Exploitation**: Classifiers are vulnerable to high-frequency adversarial perturbations. The gradient $\nabla_{x_t} \log p(y | x_t)$ often points toward unnatural adversarial artifacts that fool the classifier rather than generating realistic semantics.
3. **Inability to Guide Complex Modalities**: Training a noisy classifier is manageable for discrete classes (e.g. 1000 ImageNet categories), but becomes virtually impossible for free-form natural language prompts, bounding boxes, or robot task trajectories.

---

## 2. The Thought Experiment: Deriving the Classifier from Within

Jonathan Ho and Tim Salimans realized that the diffusion model was already learning everything needed to compute the classifier's gradients.

Recall Bayes' Rule:
$$p(y | x_t) = \frac{p(x_t | y) p(y)}{p(x_t)}$$

Taking the gradient of the log-posterior with respect to $x_t$:
$$\nabla_{x_t} \log p(y | x_t) = \nabla_{x_t} \log p(x_t | y) - \nabla_{x_t} \log p(x_t) + \underbrace{\nabla_{x_t} \log p(y)}_{= 0}$$

```
Profound Insight:
The classifier score gradient is simply the DIFFERENCE between:
1. The conditional score:   Grad_xt log p(x_t | y)
2. The unconditional score: Grad_xt log p(x_t)

You do NOT need a classifier! You only need to know how the score changes
when conditioning information is present versus when it is absent!
```

---

## 3. Mathematical Derivation: Classifier-Free Guidance

### 3.1 Score Formulation
Substituting the Bayesian gradient into Dhariwal & Nichol's guided score:
$$\begin{aligned}
\tilde{\nabla}_{x_t} \log p(x_t | y) &= \nabla_{x_t} \log p(x_t) + s \cdot \nabla_{x_t} \log p(y | x_t) \\
&= \nabla_{x_t} \log p(x_t) + s \left[ \nabla_{x_t} \log p(x_t | y) - \nabla_{x_t} \log p(x_t) \right] \\
&= (1 - s) \nabla_{x_t} \log p(x_t) + s \nabla_{x_t} \log p(x_t | y)
\end{aligned}$$

Using the identity relating score to predicted noise $\nabla_{x_t} \log p(x_t) = -\frac{1}{\sqrt{1 - \bar{\alpha}_t}} \epsilon_\theta(x_t, t)$:
$$\tilde{\epsilon}_\theta(x_t, c) = (1 - s) \epsilon_\theta(x_t, \emptyset) + s \epsilon_\theta(x_t, c)$$

Rearranging into the standard extrapolation form:
$$\tilde{\epsilon}_\theta(x_t, c) = \epsilon_\theta(x_t, \emptyset) + s \cdot \left( \epsilon_\theta(x_t, c) - \epsilon_\theta(x_t, \emptyset) \right)$$

where:
- $\epsilon_\theta(x_t, c)$ is the **conditional noise prediction** (e.g. conditioned on class $c$ or text prompt).
- $\epsilon_\theta(x_t, \emptyset)$ is the **unconditional noise prediction** (conditioned on a null/dummy token $\emptyset$).
- $s$ is the **guidance scale**:
  - $s = 0$: Purely unconditional generation (ignores prompt).
  - $s = 1$: Standard conditional generation without guidance.
  - $s > 1$ (typically $3.0$ to $7.5$): Extrapolates along the conditioning direction away from the prior, aggressively amplifying prompt adherence and sharp contrast.

---

### 3.2 Training via Conditioning Dropout

Instead of training two separate networks (one conditional and one unconditional), we train **a single unified neural network** $\epsilon_\theta(x_t, t, c)$ with **Conditioning Dropout**:

```
During Training Step:
Sample batch of (image x_0, condition c).
For each element in batch, with probability p_uncond (e.g. 10% - 20%):
    Replace c with null token: c <- \emptyset

Train network with standard simplified MSE loss:
L_simple = E_{t, x_0, c, eps} [ || eps - eps_theta(x_t, t, c) ||^2 ]
```

When $c = \emptyset$, the network learns the unconditional density $p(x)$; when $c \ne \emptyset$, it learns $p(x|c)$. Both modes share the same feature representations and U-Net weights!

---

### 3.3 Sampling with Batch Doubling

At each inference timestep $\tau_i$, we pass a doubled batch through the U-Net in parallel:
```
Batch 0..B-1:      Noisy images x_t conditioned on c
Batch B..2B-1:     Identical noisy images x_t conditioned on \emptyset

Single U-Net Forward Pass:
[ eps_cond, eps_uncond ] = model([ x_t, x_t ], [ t, t ], [ c, \emptyset ])

Extrapolate Guided Noise:
eps_guided = eps_uncond + s * (eps_cond - eps_uncond)
```

Then take standard DDPM or DDIM step using $\tilde{\epsilon}_{\text{guided}}$!

---

### 3.4 Negative Prompting

CFG also unlocks **Negative Conditioning**:
Instead of subtracting the unconditional baseline $\emptyset$, we subtract the prediction of an unwanted condition $c_{\text{neg}}$ (e.g. *"blurry, low quality, distorted"*):
$$\tilde{\epsilon} = \epsilon_\theta(x_t, c_{\text{neg}}) + s \cdot \left( \epsilon_\theta(x_t, c_{\text{pos}}) - \epsilon_\theta(x_t, c_{\text{neg}}) \right)$$
The generative trajectory actively steers *away* from negative features!

---

## 4. Tensor Contracts & Architectural Topology

```
Input:
  Image x_t: (B, C, H, W)
  Timestep t: (B,)
  Class/Prompt condition c: (B,) integers or (B, L, D_text)
  Guidance scale s: float (e.g. 3.0)

Condition Embedding Layer:
  Embedding table of size (num_classes + 1, D_cond)
  Index 0..num_classes-1 : Class tokens
  Index num_classes      : Null/Unconditional token \emptyset

Doubled Batch Forward Pass:
  Inputs: cat([x_t, x_t], dim=0) -> (2B, C, H, W)
  Conditions: cat([c, null_c], dim=0) -> (2B,)

U-Net Denoiser Output:
  out: (2B, C, H, W)
  eps_cond, eps_uncond = chunk(out, 2, dim=0)

Extrapolated Prediction:
  eps_guided = eps_uncond + s * (eps_cond - eps_uncond) -> (B, C, H, W)
```

---

## 5. Downstream Unlocks

1. **Text-to-Image Foundation (Stable Diffusion, Imagen, DALL-E 2)**: CFG is the single most important technique enabling diffusion models to accurately follow complex text prompts.
2. **Robotics Diffusion Policy (#74, #75)**: High guidance scales ($s=2.0$ to $5.0$) force the robot end-effector to adhere strictly to multi-modal language goals without drifting into unconditioned trajectory averages.
3. **Latent Diffusion Models (#32)**: CFG operates identically inside compressed latent spaces.

---

## Further Reading
- Ho, J., & Salimans, T. (2022). *Classifier-Free Diffusion Guidance*. NeurIPS 2022 Workshop.
- Dhariwal, P., & Nichol, A. (2021). *Diffusion Models Beat GANs on Image Synthesis*. NeurIPS 2021.
- [visit this chat]()
