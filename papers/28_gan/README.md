# Paper 28: Generative Adversarial Networks & Wasserstein GAN with Gradient Penalty (GAN & WGAN-GP)

> **"In this paper, we propose a new framework for estimating generative models via an adversarial process... We can train both models using only the highly successful backpropagation and dropout algorithms."**
> — Ian J. Goodfellow et al. (NeurIPS 2014)
>
> **"We show that WGANs with gradient penalty (WGAN-GP) do not suffer from the capacity underuse and pathological behaviors of weight clipping, enabling stable training of a wide variety of GAN architectures."**
> — Ishaan Gulrajani, Faruk Ahmed, Martin Arjovsky, Vincent Dumoulin, Aaron Courville (NeurIPS 2017)

---

## 1. The Breaking Point: The Geometry of Probability Distances

In 2014, Goodfellow et al. introduced Generative Adversarial Networks (GANs) as a two-player zero-sum minimax game between a Generator $G$ and a Discriminator $D$:
$$\min_G \max_D V(D, G) = \mathbb{E}_{x \sim p_{\text{data}}}[\log D(x)] + \mathbb{E}_{z \sim p_z}[\log(1 - D(G(z)))]$$

When the discriminator is optimal ($D^*(x) = \frac{p_{\text{data}}(x)}{p_{\text{data}}(x) + p_g(x)}$), the minimax game reduces to minimizing the **Jensen-Shannon (JS) Divergence**:
$$V(D^*, G) = -\log 4 + 2 \cdot D_{\text{JS}}(p_{\text{data}} \parallel p_g)$$

```
Classic GAN Manifold Problem:
-----------------------------
High-dimensional image space R^D (e.g., 28x28 = 784 dimensions):
- Real data p_data lies on a low-dimensional sub-manifold M_r
- Generated data p_g lies on a low-dimensional sub-manifold M_g
- Probability of two low-dimensional manifolds intersecting in R^784 is ZERO!

Support(p_data) \cap Support(p_g) = \emptyset
  ===>  D_JS(p_data || p_g) = log(2) = CONSTANT!
  ===>  Grad_theta_G D_JS = 0  (Vanishing Gradients!)
```

### The Three Pathologies of Classic GANs:
1. **Vanishing Gradients**: Because real and generated data manifolds have disjoint supports, an optimal discriminator easily achieves 100% classification accuracy, resulting in zero gradients flowing back to the generator.
2. **Non-Saturating Heuristic & Mode Collapse**: To avoid vanishing gradients, Goodfellow proposed maximizing $\log D(G(z))$ instead of minimizing $\log(1 - D(G(z)))$. Mathematically, this minimizes:
   $$D_{\text{KL}}(p_g \parallel p_{\text{data}}) - 2 D_{\text{JS}}(p_{\text{data}} \parallel p_g)$$
   The asymmetric KL divergence heavily penalizes generating implausible samples ($p_g(x) > 0$ when $p_{\text{data}}(x) = 0$), but exerts zero penalty when the generator ignores entire modes ($p_{\text{data}}(x) > 0$ when $p_g(x) = 0$). The generator collapses to outputting only a single safe image: **Mode Collapse**!
3. **Instability & Divergence**: Minimax optimization without smooth continuity produces oscillatory limit cycles rather than converging to Nash equilibria.

---

## 2. The Thought Experiment: The Earth Mover's Distance

Imagine two piles of earth on a table: pile $P_1$ and pile $P_2$.
- **JS / KL Divergence View**: If pile $P_1$ is at $x=0$ and pile $P_2$ is moved from $x=10$ to $x=100$, JS divergence says the difference is identical ($\log 2$). It has no concept of spatial distance!
- **Wasserstein-1 (Earth Mover's) View**: The work required to move pile $P_1$ to $P_2$ is mass $\times$ distance. If $P_2$ moves from $x=10$ to $x=2$, the Wasserstein distance decreases continuously from $10$ to $2$.

```
Distribution P_0 at x = 0; Distribution P_theta at x = theta:
-------------------------------------------------------------
Distance / Divergence       Value as theta -> 0       Differentiable?
Total Variation (TV)        1 (if theta != 0)         NO (step function)
KL Divergence               +inf (if theta != 0)      NO
JS Divergence               log(2) (if theta != 0)    NO (flat plateau)
Wasserstein-1 Distance      |theta|                   YES! (Smooth linear slope)
```

The Wasserstein distance provides a smooth, meaningful gradient **even when the distributions have strictly disjoint supports**!

---

## 3. Mathematical Derivation: Kantorovich-Rubinstein Duality & Gradient Penalty

### 3.1 The Primal Form
The Wasserstein-1 distance between continuous probability distributions $p_r$ (real) and $p_g$ (generated) is:
$$W_1(p_r, p_g) = \inf_{\gamma \in \Pi(p_r, p_g)} \mathbb{E}_{(x, y) \sim \gamma}[\|x - y\|_2]$$
where $\Pi(p_r, p_g)$ denotes the set of all joint distributions $\gamma(x, y)$ whose marginals are $p_r$ and $p_g$.

---

### 3.2 The Kantorovich-Rubinstein Dual Form
Solving the infimum over all joint couplings is intractable. By the Kantorovich-Rubinstein theorem:
$$W_1(p_r, p_g) = \sup_{\|D\|_L \le 1} \mathbb{E}_{x \sim p_r}[D(x)] - \mathbb{E}_{\tilde{x} \sim p_g}[D(\tilde{x})]$$
where the supremum is over the set of all **1-Lipschitz continuous functions** $D: \mathcal{X} \to \mathbb{R}$.

In WGAN, $D$ is no longer a classifier outputting probabilities in $[0, 1]$; it is an unbounded **Critic** scoring realness:
$$\mathcal{L}_{\text{critic}} = \mathbb{E}_{\tilde{x} \sim p_g}[D(\tilde{x})] - \mathbb{E}_{x \sim p_r}[D(x)]$$
$$\mathcal{L}_{\text{generator}} = -\mathbb{E}_{z \sim p_z}[D(G(z))]$$

---

### 3.3 The 1-Lipschitz Constraint & Gradient Penalty (WGAN-GP)

A differentiable function $D$ is 1-Lipschitz if and only if its gradients have norm at most 1 everywhere:
$$\|\nabla_x D(x)\|_2 \le 1 \quad \forall x$$

- **Arjovsky et al. (WGAN)** clipped critic weights to $[-c, c]$. This clipped capacity and caused pathological exploding/vanishing gradients.
- **Gulrajani et al. (WGAN-GP)** proved that the optimal critic $D^*$ has unit gradient norm almost everywhere along straight lines connecting real points $x \sim p_r$ and generated points $\tilde{x} \sim p_g$:
  $$\|\nabla_{\hat{x}} D^*(\hat{x})\|_2 = 1$$
  where $\hat{x} = \epsilon x + (1 - \epsilon)\tilde{x}$ for $\epsilon \sim U(0, 1)$.

The **Gradient Penalty (GP)** penalizes the critic for deviating from unit gradient norm:
$$\mathcal{L}_{\text{GP}} = \mathbb{E}_{\hat{x} \sim p_{\hat{x}}} \left[ \left( \|\nabla_{\hat{x}} D(\hat{x})\|_2 - 1 \right)^2 \right]$$

The complete WGAN-GP critic loss is:
$$\mathcal{L}_{\text{critic}} = \underbrace{\mathbb{E}[D(\tilde{x})] - \mathbb{E}[D(x)]}_{\text{Wasserstein Loss}} + \underbrace{\lambda \cdot \mathbb{E}_{\hat{x}}\left[\left(\|\nabla_{\hat{x}} D(\hat{x})\|_2 - 1\right)^2\right]}_{\text{Gradient Penalty} \ (\lambda = 10)}$$

> **Critical Architectural Rule**: The Critic must **not use Batch Normalization**, because Batch Normalization creates inter-sample dependencies across the batch, violating the validity of the gradient penalty on individual samples! Use **LayerNorm**, **SpectralNorm**, or no normalization in the Critic.

---

## 4. Tensor Contracts & Dimension Transitions

```
Generator:
  Input Latent Vector: z ~ N(0, I), shape (B, D_latent)
      |
      v
  [ Linear Projection ]: (B, C_g * 7 * 7)
      |
      v
  Reshape to Spatial Feature Map: (B, C_g, 7, 7)
      |
      v
  ConvTranspose2d / Upsample (stride 2): (B, C_g // 2, 14, 14)
      |
      v
  ConvTranspose2d / Upsample (stride 2): (B, C_out, 28, 28)
      |
      v
  Tanh Activation: Output Image x_fake in [-1, 1], shape (B, C_out, 28, 28)

Critic:
  Input Image: x in [-1, 1], shape (B, C_in, 28, 28)
      |
      v
  Conv2d (stride 2) + LeakyReLU(0.2): (B, C_d, 14, 14)
      |
      v
  Conv2d (stride 2) + LayerNorm + LeakyReLU(0.2): (B, C_d * 2, 7, 7)
      |
      v
  Flatten: (B, C_d * 2 * 7 * 7)
      |
      v
  Linear (unbounded scalar output): D(x), shape (B, 1)  [NO SIGMOID!]
```

---

## 5. Downstream Unlocks

1. **High-Resolution Synthesis (StyleGAN 1/2/3)**: WGAN-GP gradient regularization established the stability foundation allowing StyleGAN to synthesize photorealistic $1024 \times 1024$ human faces without collapse.
2. **Image Translation & Restoration (CycleGAN, Pix2Pix, ESRGAN)**: Combining adversarial WGAN-GP losses with $L_1$ and perceptual VGG features enables unpaired domain-to-domain translation and super-resolution.
3. **Adversarial Imitation Learning (GAIL & AIRL - Stage 8)**: The discriminator scores trajectories against expert demonstrations, computing an implicit reward function without reward engineering.

---

## Further Reading
- Goodfellow, I., et al. (2014). *Generative Adversarial Nets*. NeurIPS 2014.
- Arjovsky, M., Chintala, S., & Bottou, L. (2017). *Wasserstein Generative Adversarial Networks*. ICML 2017.
- Gulrajani, I., Ahmed, F., Arjovsky, M., Dumoulin, V., & Courville, A. (2017). *Improved Training of Wasserstein GANs*. NeurIPS 2017.
- [visit this chat]()
