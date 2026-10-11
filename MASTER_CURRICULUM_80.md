# Master Curriculum: 1-to-80 Deep Learning, Reinforcement Learning & Robotics (`dl-rl-robotics-80`)

This monorepo implements 80 foundational papers from scratch in PyTorch, sorted by strict code dependency (topological sort) with zero forward references.

---

## The 6-Step Paper-to-PyTorch Pipeline

Every paper from #1 to #80 follows this strict execution workflow:

1. **Surgical Extraction (README.md)**:
   - Written from a **First-Principles Inventor's Perspective**:
     1. The Historical Breaking Point (What broke before this paper?)
     2. The Thought Experiment: "If I Were Inventing It"
     3. Mathematical Derivation (Step-by-step)
     4. Tensor Shape Contract: `(B, C, H, W) -> ...`
     5. Downstream Unlocks in the 80-Paper Roadmap
2. **The Tensor Shape Contract (`model.py`)**:
   - Explicit comment block at the top tracing every dimension transformation.
3. **Bottom-Up Submodule Coding with `torch.randn()`**:
   - Code smallest leaf blocks first using `torch.einsum`.
   - Standalone `if __name__ == "__main__":` test passing dummy tensors.
4. **The Gradient & Detached-Target Audit**:
   - Verify target networks are wrapped in `torch.no_grad()`.
   - Polyak/EMA parameters update cleanly.
   - Actors do not leak gradients into critic parameters during policy updates.
   - Assert zero NaNs in gradients.
5. **The 1-Batch Overfit Sanity Check (`train_sanity.py`)**:
   - Train on a single batch of 4-8 samples for 200 iterations (or 2-step toy env for RL).
   - Loss must drop to $\approx 0.000$ before declaring the model verified.
6. **Downscaled Benchmark Validation & Core Export**:
   - Validate on fast proxy benchmarks (MNIST, CIFAR-10, Pendulum, CartPole, Push-T).
   - Export reusable blocks to `core/` for downstream papers.

---

## The 80-Paper Topological Progression

### Stage 1: Optimization, Normalization & 2D/3D Backbones (Papers 1–9)
- **#1 Delving Deep into Rectifiers (Kaiming Init)** (He et al., 2015) — Variance-preserving init for deep ReLU networks.
- **#2 Adam: A Method for Stochastic Optimization (+ AdamW)** (Kingma & Ba, 2014 / Loshchilov & Hutter, 2017) — Decoupled weight decay optimizer.
- **#3 Batch Normalization** (Ioffe & Szegedy, 2015) — BatchNorm1d/2d with running buffers.
- **#4 Layer Normalization (+ RMSNorm)** (Ba et al., 2016 / Zhang & Sennrich, 2019) — Channel/token normalization.
- **#5 Deep Residual Learning (ResNet)** (He et al., 2015) — Residual shortcuts and bottleneck blocks.
- **#6 U-Net: Convolutional Networks for Image Segmentation** (Ronneberger et al., 2015) — Multi-scale skip connections (backbone for DDPM & Diffusion Policy).
- **#7 MobileNetV2: Inverted Residuals & Linear Bottlenecks** (Sandler et al., 2018) — Depthwise separable convolutions for robot edge inference.
- **#8 Focal Loss for Dense Object Detection** (Lin et al., 2017) — Class-imbalanced loss and stable `logsumexp`.
- **#9 PointNet / PointNet++: Deep Learning on Point Sets** (Qi et al., 2017) — Unordered 3D point cloud processing via shared MLPs and symmetric max-pooling.

### Stage 2: Sequence Modeling, Attention & Transformers (Papers 10–18)
- **#10 Neural Machine Translation (Additive Attention)** (Bahdanau et al., 2014) — GRU seq2seq with additive alignment.
- **#11 Attention Is All You Need (Transformer)** (Vaswani et al., 2017) — Scaled dot-product Multi-Head Attention.
- **#12 RoFormer: Rotary Position Embedding (RoPE)** (Su et al., 2021) — Complex plane rotary embeddings.
- **#13 BERT: Bidirectional Transformers** (Devlin et al., 2018) — Bidirectional masked token encoder.
- **#14 GPT-2: Unsupervised Multitask Learners** (Radford et al., 2019) — Causal autoregressive decoder with KV-cache.
- **#15 LoRA: Low-Rank Adaptation** (Hu et al., 2021) — Parameter-efficient adapter wrapper over `nn.Linear`.
- **#16 An Image is Worth 16x16 Words (ViT)** (Dosovitskiy et al., 2020) — Image patchification into Transformer encoder.
- **#17 End-to-End Object Detection with Transformers (DETR)** (Carion et al., 2020) — Bipartite matching + Transformer decoder.
- **#18 Mamba: Linear-Time Sequence Modeling (SSM)** (Gu & Dao, 2023) — Selective state-space scan.

### Stage 3: Self-Supervised & Multimodal Visual Pre-Training (Papers 19–25)
- **#19 Contrastive Predictive Coding (InfoNCE)** (Oord et al., 2018) — Mutual information lower bound loss.
- **#20 SimCLR: Contrastive Learning of Visual Representations** (Chen et al., 2020) — ResNet + projection head + InfoNCE.
- **#21 MoCo: Momentum Contrast** (He et al., 2020) — EMA target encoder and dictionary queue.
- **#22 DINO: Self-Supervised Vision Transformers** (Caron et al., 2021) — Self-distillation with centering and sharpening.
- **#23 Masked Autoencoders (MAE)** (He et al., 2021) — Patch masking with pixel reconstruction.
- **#24 CLIP: Learning Transferable Visual Models** (Radford et al., 2021) — Vision-language contrastive alignment.
- **#25 R3M: Universal Visual Representation for Robot Manipulation** (Nair et al., 2022) — Time-contrastive ego-video pretraining.

### Stage 4: Generative Modeling — VAEs, GANs, Diffusion & Flow Matching (Papers 26–33)
- **#26 Auto-Encoding Variational Bayes (VAE)** (Kingma & Welling, 2013) — Gaussian reparameterization and ELBO.
- **#27 VQ-VAE: Neural Discrete Representation Learning** (Oord et al., 2017) — Discrete vector codebook and straight-through estimator.
- **#28 GAN / WGAN-GP** (Goodfellow et al., 2014 / Gulrajani et al., 2017) — Adversarial minimax and gradient penalty.
- **#29 DDPM: Denoising Diffusion Probabilistic Models** (Ho et al., 2020) — U-Net with sinusoidal time embeddings for denoising.
- **#30 DDIM: Denoising Diffusion Implicit Models** (Song et al., 2020) — Deterministic fast 15-step sampling.
- **#31 Classifier-Free Diffusion Guidance (CFG)** (Ho & Salimans, 2022) — Conditional diffusion with conditioning dropout.
- **#32 Latent Diffusion Models (LDM)** (Rombach et al., 2022) — Cross-attention diffusion in VAE latent space.
- **#33 Flow Matching for Generative Modeling** (Lipman et al., 2022) — Straight Optimal Transport ODE vector fields.

### Stage 5: Deep RL I — Value-Based & Policy Gradient Foundations (Papers 34–43)
- **#34 Playing Atari with Deep RL (DQN)** (Mnih et al., 2013) — Experience Replay and Bellman TD target network.
- **#35 Double DQN** (van Hasselt et al., 2015) — Target network action selection decoupling.
- **#36 Prioritized Experience Replay (PER)** (Schaul et al., 2015) — Binary SumTree TD-error sampling.
- **#37 Dueling Network Architectures** (Wang et al., 2015) — Decomposed $V(s)$ and $A(s,a)$ streams.
- **#38 Distributional RL (C51)** (Bellemare et al., 2017) — 51-atom categorical return distribution projection.
- **#39 Trust Region Policy Optimization (TRPO)** (Schulman et al., 2015) — Natural policy gradients via conjugate gradient.
- **#40 Generalized Advantage Estimation (GAE)** (Schulman et al., 2015) — Multi-step exponentially weighted advantage.
- **#41 Proximal Policy Optimization (PPO)** (Schulman et al., 2017) — Clipped surrogate objective with GAE.
- **#42 Continuous Control with Deep RL (DDPG)** (Lillicrap et al., 2015) — Deterministic policy gradient actor-critic.
- **#43 TD3: Twin Delayed DDPG** (Fujimoto et al., 2018) — Clipped double Q-learning, delayed actor updates, target smoothing.

### Stage 6: Deep RL II — Max-Entropy, Exploration & Offline/Preference RL (Papers 44–54)
- **#44 Soft Actor-Critic (SAC)** (Haarnoja et al., 2018) — Maximum entropy RL with auto-tuned temperature $\alpha$.
- **#45 Hindsight Experience Replay (HER)** (Andrychowicz et al., 2017) — Goal relabeling for sparse reward robotics.
- **#46 Random Network Distillation (RND)** (Burda et al., 2018) — Intrinsic prediction curiosity bonus.
- **#47 Successor Features & GPI** (Barreto et al., 2017) — Decoupled dynamics and reward weights for zero-shot transfer.
- **#48 BCQ: Batch-Constrained Deep RL** (Fujimoto et al., 2019) — CVAE action generator + TD3 critic.
- **#49 Conservative Q-Learning (CQL)** (Kumar et al., 2020) — Log-sum-exp OOD action penalty.
- **#50 Advantage-Weighted Regression (AWR)** (Peng et al., 2019) — Supervised policy regression with GAE weights.
- **#51 Implicit Q-Learning (IQL)** (Kostrikov et al., 2021) — Expectile regression without OOD action queries.
- **#52 Decision Transformer** (Chen et al., 2021) — Offline RL as GPT-2 sequence modeling over return-to-go.
- **#53 RL from Human Preferences (RLHF)** (Christiano et al., 2017) — Bradley-Terry reward model + PPO.
- **#54 Direct Preference Optimization (DPO)** (Rafailov et al., 2023) — Closed-form preference policy optimization.

### Stage 7: Model-Based RL, World Models & Trajectory Planning (Papers 55–61)
- **#55 PETS: Probabilistic Ensembles with Trajectory Sampling** (Chua et al., 2018) — Gaussian ensemble + Cross-Entropy Method.
- **#56 MBPO: Model-Based Policy Optimization** (Janner et al., 2019) — Branched model rollouts into SAC buffer.
- **#57 MPPI: Path Integral Control** (Williams et al., 2017) — GPU-vectorized temperature-weighted trajectory optimization.
- **#58 World Models** (Ha & Schmidhuber, 2018) — VAE compression + MDN-RNN latent physics.
- **#59 Dreamer (V1–V3)** (Hafner et al., 2019) — RSSM world model with actor-critic latent imagination.
- **#60 TD-MPC2** (Hansen et al., 2023) — Latent dynamics + SAC value bootstrapping + MPPI planning.
- **#61 MuZero** (Schrittwieser et al., 2019) — MCTS planning over learned latent dynamics.

### Stage 8: Imitation Learning, Sim-to-Real & Agile Locomotion (Papers 62–69)
- **#62 DAgger: Dataset Aggregation** (Ross et al., 2011) — Online expert query to overcome covariate shift.
- **#63 GAIL: Generative Adversarial Imitation Learning** (Ho & Ermon, 2016) — GAN discriminator matching expert occupancy.
- **#64 AIRL: Adversarial Inverse RL** (Fu et al., 2017) — Disentangled, morphology-portable reward recovery.
- **#65 Domain Randomization for Sim-to-Real** (Tobin et al., 2017) — Mass, friction, latency, and visual randomization.
- **#66 RMA: Rapid Motor Adaptation** (Kumar et al., 2021) — Privileged teacher + 1D-CNN online adaptation student.
- **#67 Parallel GPU Deep RL Locomotion** (Rudin et al., 2021) — Massively parallel 4,096-env PPO terrain locomotion.
- **#68 AMP: Adversarial Motion Priors** (Peng et al., 2021) — Mocap-constrained adversarial locomotion.
- **#69 Eureka: Human-Level Reward Design via LLMs** (Ma et al., 2023) — Evolutionary reward synthesis for GPU RL.

### Stage 9: Modern Robot Manipulation & Vision-Language-Action Models (Papers 70–80)
- **#70 Implicit Behavioral Cloning (IBC)** (Florence et al., 2021) — Energy-based policy for discontinuous actions.
- **#71 Behavior Transformers (BeT)** (Shafiullah et al., 2022) — GPT-2 + $k$-means action clustering + offset regression.
- **#72 ACT: Action Chunking with Transformers** (Zhao et al., 2023) — ResNet + DETR + CVAE trajectory chunking.
- **#73 Diffuser: Planning with Diffusion** (Janner et al., 2022) — 1D U-Net trajectory generation and inpainting.
- **#74 Diffusion Policy: Visuomotor Policy Learning** (Chi et al., 2023) — DDIM + CFG + chunking for manipulation.
- **#75 3D Diffusion Policy (DP3)** (Ze et al., 2024) — PointNet encoder + Diffusion Policy for 3D manipulation.
- **#76 Transporter Networks** (Zeng et al., 2020) — SE(2)/SE(3) equivariant spatial visual manipulation.
- **#77 Perceiver-Actor (PerAct)** (Shridhar et al., 2022) — 3D voxel cross-attention with language conditioning.
- **#78 RT-1: Robotics Transformer** (Brohan et al., 2022) — MobileNet + FiLM tokenized manipulation.
- **#79 OpenVLA / RT-2** (Kim et al., 2024) — Vision-Language-Action tokenization via ViT + LLM + LoRA.
- **#80 $\pi_0$ (Pi-Zero): Vision-Language-Action Flow Model** (Black et al., 2024) — VLM backbone + Action Chunking + Flow Matching.
