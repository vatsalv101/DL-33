# DL-33

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **Foundational Papers from Scratch in PyTorch** (Stages 1–4: Papers #1–#33) across Deep Learning, Computer Vision, Transformers, Self-Supervised Learning & Generative Modeling — organized with strict topological code dependencies and zero forward references.

---

## 🧭 Philosophy & Vision

Most paper implementations exist as isolated repositories with disparate interfaces, implicit assumptions, and uncontrolled dependencies. **DL-RL-Robotics-80** builds an interconnected, cumulative library of milestone papers from first principles:

* **Strict Topological Sort:** Modules build upon previous implementations. Paper #11 (Transformer) builds upon Paper #4 (LayerNorm) and Paper #10 (Bahdanau Attention); Paper #32 (Latent Diffusion) composes Paper #6 (U-Net) and Paper #26 (VAE).
* **Zero Magic:** No high-level wrappers hiding the math. Every layer, attention head, loss function, and update step is exposed and audited.
* **Rigorous Verification Gate:** Every implementation must pass a 6-step verification pipeline before its building blocks are exported to the shared `core/` library.

---

## 🔬 The 6-Step Paper-to-PyTorch Pipeline

Each paper follows a deterministic validation workflow:

```
[1. Surgical Extraction] ──> [2. Tensor Shape Contract] ──> [3. Bottom-Up Submodule Coding]
                                                                        │
[6. Benchmark & Core Export] <── [5. Single-Batch Overfit] <── [4. Gradient & Target Audit]
```

1. **Surgical Extraction (`README.md` per paper):**
   * The Historical Breaking Point (What broke before this paper?)
   * The Thought Experiment: *"If I Were Inventing It"*
   * Mathematical Derivation & Tensor Dimension Contracts
2. **The Tensor Shape Contract (`model.py`):**
   * Explicit dimension tracing throughout every transformation (`assert_shape_contract`).
3. **Bottom-Up Submodule Coding:**
   * Leaf blocks implemented with explicit tensor operations (`torch.einsum`).
4. **Gradient & Detached-Target Audit:**
   * Automated verification (`audit_gradients`) asserting target networks remain untainted, no gradient leaks across actor/critic, and zero NaNs/Infs.
5. **Single-Batch Overfit Gate (`train_sanity.py`):**
   * Verification on a 1-batch toy problem driving training loss $\to 0.00$ before scaling.
6. **Benchmark & Core Export:**
   * Verification on fast proxy benchmarks (MNIST, CIFAR, Pendulum, Push-T) and promotion of reusable modules to `core/`.

---

## 🗺️ Roadmap (Stages 1–4: Papers #1–#33)

<details open>
<summary><b>Stage 1: Optimization, Normalization & 2D/3D Backbones (#1–#9)</b></summary>

- **#1 Delving Deep into Rectifiers (Kaiming Init)** — He et al., 2015
- **#2 Adam & AdamW Optimizers** — Kingma & Ba, 2014 / Loshchilov & Hutter, 2017
- **#3 Batch Normalization** — Ioffe & Szegedy, 2015
- **#4 Layer Normalization & RMSNorm** — Ba et al., 2016 / Zhang & Sennrich, 2019
- **#5 Deep Residual Learning (ResNet)** — He et al., 2015
- **#6 U-Net: Convolutional Networks for Segmentation** — Ronneberger et al., 2015
- **#7 MobileNetV2: Inverted Residuals & Bottlenecks** — Sandler et al., 2018
- **#8 Focal Loss for Dense Object Detection** — Lin et al., 2017
- **#9 PointNet / PointNet++: Deep Learning on Point Sets** — Qi et al., 2017
</details>

<details>
<summary><b>Stage 2: Sequence Modeling, Attention & Transformers (#10–#18)</b></summary>

- **#10 Neural Machine Translation (Additive Attention)** — Bahdanau et al., 2014
- **#11 Attention Is All You Need (Transformer)** — Vaswani et al., 2017
- **#12 RoFormer: Rotary Position Embedding (RoPE)** — Su et al., 2021
- **#13 BERT: Bidirectional Transformers** — Devlin et al., 2018
- **#14 GPT-2: Autoregressive Decoder & KV-Cache** — Radford et al., 2019
- **#15 LoRA: Low-Rank Adaptation** — Hu et al., 2021
- **#16 Vision Transformer (ViT)** — Dosovitskiy et al., 2020
- **#17 End-to-End Object Detection (DETR)** — Carion et al., 2020
- **#18 Mamba: Linear-Time Sequence Modeling (SSM)** — Gu & Dao, 2023
</details>

<details>
<summary><b>Stage 3: Self-Supervised & Multimodal Pre-Training (#19–#25)</b></summary>

- **#19 Contrastive Predictive Coding (InfoNCE)** — Oord et al., 2018
- **#20 SimCLR: Visual Contrastive Learning** — Chen et al., 2020
- **#21 MoCo: Momentum Contrast** — He et al., 2020
- **#22 DINO: Self-Supervised Vision Transformers** — Caron et al., 2021
- **#23 Masked Autoencoders (MAE)** — He et al., 2021
- **#24 CLIP: Contrastive Language-Image Pre-Training** — Radford et al., 2021
- **#25 R3M: Visual Representation for Robot Manipulation** — Nair et al., 2022
</details>

<details>
<summary><b>Stage 4: Generative Modeling — VAEs, GANs, Diffusion & Flow (#26–#33)</b></summary>

- **#26 Auto-Encoding Variational Bayes (VAE)** — Kingma & Welling, 2013
- **#27 VQ-VAE: Discrete Representation Learning** — Oord et al., 2017
- **#28 GAN & WGAN-GP** — Goodfellow et al., 2014 / Gulrajani et al., 2017
- **#29 DDPM: Denoising Diffusion Probabilistic Models** — Ho et al., 2020
- **#30 DDIM: Denoising Diffusion Implicit Models** — Song et al., 2020
- **#31 Classifier-Free Diffusion Guidance (CFG)** — Ho & Salimans, 2022
- **#32 Latent Diffusion Models (LDM)** — Rombach et al., 2022
- **#33 Flow Matching for Generative Modeling** — Lipman et al., 2022
</details>

<!-- Stages 5–9 commented out during plan reconsideration:
<details>
<summary><b>Stage 5: Deep RL I — Value-Based & Policy Gradients (#34–#43)</b></summary>

- **#34 Playing Atari with Deep RL (DQN)** — Mnih et al., 2013
- **#35 Double DQN** — van Hasselt et al., 2015
- **#36 Prioritized Experience Replay (PER)** — Schaul et al., 2015
- **#37 Dueling Network Architectures** — Wang et al., 2015
- **#38 Distributional RL (C51)** — Bellemare et al., 2017
- **#39 Trust Region Policy Optimization (TRPO)** — Schulman et al., 2015
- **#40 Generalized Advantage Estimation (GAE)** — Schulman et al., 2015
- **#41 Proximal Policy Optimization (PPO)** — Schulman et al., 2017
- **#42 Continuous Control with Deep RL (DDPG)** — Lillicrap et al., 2015
- **#43 Twin Delayed DDPG (TD3)** — Fujimoto et al., 2018
</details>

<details>
<summary><b>Stage 6: Deep RL II — Max-Entropy, Exploration & Offline RL (#44–#54)</b></summary>

- **#44 Soft Actor-Critic (SAC)** — Haarnoja et al., 2018
- **#45 Hindsight Experience Replay (HER)** — Andrychowicz et al., 2017
- **#46 Random Network Distillation (RND)** — Burda et al., 2018
- **#47 Successor Features & Generalized Policy Improvement** — Barreto et al., 2017
- **#48 Batch-Constrained Deep RL (BCQ)** — Fujimoto et al., 2019
- **#49 Conservative Q-Learning (CQL)** — Kumar et al., 2020
- **#50 Advantage-Weighted Regression (AWR)** — Peng et al., 2019
- **#51 Implicit Q-Learning (IQL)** — Kostrikov et al., 2021
- **#52 Decision Transformer** — Chen et al., 2021
- **#53 RL from Human Preferences (RLHF)** — Christiano et al., 2017
- **#54 Direct Preference Optimization (DPO)** — Rafailov et al., 2023
</details>

<details>
<summary><b>Stage 7: Model-Based RL & World Models (#55–#61)</b></summary>

- **#55 PETS: Probabilistic Ensembles with Trajectory Sampling** — Chua et al., 2018
- **#56 Model-Based Policy Optimization (MBPO)** — Janner et al., 2019
- **#57 Model Predictive Path Integral Control (MPPI)** — Williams et al., 2017
- **#58 World Models** — Ha & Schmidhuber, 2018
- **#59 Dreamer (V1–V3)** — Hafner et al., 2019
- **#60 TD-MPC2** — Hansen et al., 2023
- **#61 MuZero** — Schrittwieser et al., 2019
</details>

<details>
<summary><b>Stage 8: Imitation Learning & Agile Locomotion (#62–#69)</b></summary>

- **#62 DAgger: Dataset Aggregation** — Ross et al., 2011
- **#63 GAIL: Generative Adversarial Imitation Learning** — Ho & Ermon, 2016
- **#64 AIRL: Adversarial Inverse RL** — Fu et al., 2017
- **#65 Domain Randomization for Sim-to-Real** — Tobin et al., 2017
- **#66 RMA: Rapid Motor Adaptation** — Kumar et al., 2021
- **#67 Parallel GPU Deep RL Locomotion** — Rudin et al., 2021
- **#68 AMP: Adversarial Motion Priors** — Peng et al., 2021
- **#69 Eureka: Reward Design via LLMs** — Ma et al., 2023
</details>

<details>
<summary><b>Stage 9: Robot Manipulation & Vision-Language-Action Models (#70–#80)</b></summary>

- **#70 Implicit Behavioral Cloning (IBC)** — Florence et al., 2021
- **#71 Behavior Transformers (BeT)** — Shafiullah et al., 2022
- **#72 Action Chunking with Transformers (ACT)** — Zhao et al., 2023
- **#73 Diffuser: Planning with Diffusion** — Janner et al., 2022
- **#74 Diffusion Policy: Visuomotor Policy Learning** — Chi et al., 2023
- **#75 3D Diffusion Policy (DP3)** — Ze et al., 2024
- **#76 Transporter Networks** — Zeng et al., 2020
- **#77 Perceiver-Actor (PerAct)** — Shridhar et al., 2022
- **#78 RT-1: Robotics Transformer** — Brohan et al., 2022
- **#79 OpenVLA / RT-2** — Kim et al., 2024
- **#80 $\pi_0$ (Pi-Zero): VLA Flow Matching Model** — Black et al., 2024
</details>
-->

---

## 🚀 Getting Started

### Installation

Clone the repository and install dependencies with [`uv`](https://github.com/astral-sh/uv):

```bash
git clone https://github.com/vatsalv101/DL-RL-Robotics-80.git
cd DL-RL-Robotics-80

uv sync
```

Or install with `pip`:

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -e .
```

### Running Sanity Checks

Run verification tests across the codebase:

```bash
pytest tests/
```

---

## 📂 Repository Structure

```text
DL-RL-Robotics-80/
├── MASTER_CURRICULUM_80.md     # Complete topological roadmap & paper specifications
├── core/                       # Verified shared building blocks exported from papers
│   └── __init__.py
├── sanity/                     # Automated shape, gradient, and overfit verification harness
│   ├── checks.py
│   └── __init__.py
├── tests/                      # Unit tests & verification suites
│   └── test_sanity_harness.py
├── pyproject.toml              # Environment dependencies and project metadata
└── README.md
```

---

## 📜 License

This project is licensed under the MIT License.
