from typing import Optional, Tuple, List
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def extract(a: torch.Tensor, t: torch.Tensor, x_shape: Tuple[int, ...]) -> torch.Tensor:
    """
    Extracts appropriate coefficients at specified timesteps t and reshapes to broadcast with x_shape.
    """
    b = t.shape[0]
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


class SinusoidalPosEmb(nn.Module):
    """
    Sinusoidal positional embedding for discrete diffusion timesteps t.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        device = t.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class DiffusionResidualBlock(nn.Module):
    """
    Residual block with timestep embedding injection for diffusion models.
    """

    def __init__(self, in_channels: int, out_channels: int, time_emb_dim: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.gn1 = nn.GroupNorm(8, out_channels)
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, out_channels),
        )
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.gn2 = nn.GroupNorm(8, out_channels)

        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = F.silu(self.gn1(self.conv1(x)))
        time_proj = self.time_mlp(t_emb)[:, :, None, None]
        h = h + time_proj
        h = F.silu(self.gn2(self.conv2(h)))
        return h + self.shortcut(x)


class UNetDenoiser(nn.Module):
    """
    Compact U-Net denoiser predicting noise eps_theta(x_t, t).
    Supports 28x28 (MNIST) and 32x32 (CIFAR) image inputs.
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        hidden_dims: Tuple[int, ...] = (32, 64, 128),
        time_emb_dim: int = 64,
    ):
        super().__init__()
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim * 2),
            nn.SiLU(),
            nn.Linear(time_emb_dim * 2, time_emb_dim),
        )

        c1, c2, c3 = hidden_dims
        # Downsampling path
        self.init_conv = nn.Conv2d(in_channels, c1, kernel_size=3, padding=1)
        self.down1 = DiffusionResidualBlock(c1, c1, time_emb_dim)
        self.downsample1 = nn.Conv2d(c1, c2, kernel_size=4, stride=2, padding=1)

        self.down2 = DiffusionResidualBlock(c2, c2, time_emb_dim)
        self.downsample2 = nn.Conv2d(c2, c3, kernel_size=4, stride=2, padding=1)

        # Bottleneck
        self.mid1 = DiffusionResidualBlock(c3, c3, time_emb_dim)
        self.mid2 = DiffusionResidualBlock(c3, c3, time_emb_dim)

        # Upsampling path
        self.upsample1 = nn.ConvTranspose2d(c3, c2, kernel_size=4, stride=2, padding=1)
        self.up1 = DiffusionResidualBlock(c2 * 2, c2, time_emb_dim)

        self.upsample2 = nn.ConvTranspose2d(c2, c1, kernel_size=4, stride=2, padding=1)
        self.up2 = DiffusionResidualBlock(c1 * 2, c1, time_emb_dim)

        self.out_conv = nn.Conv2d(c1, out_channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t_emb = self.time_mlp(t)

        x1 = self.init_conv(x)
        x1 = self.down1(x1, t_emb)
        x2 = self.downsample1(x1)

        x2 = self.down2(x2, t_emb)
        x3 = self.downsample2(x2)

        x_mid = self.mid1(x3, t_emb)
        x_mid = self.mid2(x_mid, t_emb)

        u1 = self.upsample1(x_mid)
        u1 = torch.cat([u1, x2], dim=1)
        u1 = self.up1(u1, t_emb)

        u2 = self.upsample2(u1)
        u2 = torch.cat([u2, x1], dim=1)
        u2 = self.up2(u2, t_emb)

        return self.out_conv(u2)


class GaussianDiffusion(nn.Module):
    """
    Denoising Diffusion Probabilistic Model (DDPM) (Ho et al., 2020).
    Implements closed-form forward diffusion, simple noise-prediction loss,
    and ancestral reverse sampling.
    """

    def __init__(
        self,
        model: nn.Module,
        timesteps: int = 1000,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
    ):
        super().__init__()
        self.model = model
        self.timesteps = timesteps

        # Linear variance schedule beta_t
        betas = torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float32)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

        # Register forward diffusion constants as buffers
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)

        # Calculations for diffusion q(x_t | x_0)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer(
            "sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod)
        )

        # Calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        self.register_buffer("posterior_variance", posterior_variance)
        self.register_buffer(
            "posterior_log_variance_clipped",
            torch.log(torch.clamp(posterior_variance, min=1e-20)),
        )
        self.register_buffer(
            "posterior_mean_coef1",
            betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod),
        )
        self.register_buffer(
            "posterior_mean_coef2",
            (1.0 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - alphas_cumprod),
        )

    def q_sample(
        self,
        x_start: torch.Tensor,
        t: torch.Tensor,
        noise: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward diffusion process: closed-form sampling of x_t at timestep t:
        x_t = sqrt(alpha_bar_t) * x_start + sqrt(1 - alpha_bar_t) * noise
        """
        if noise is None:
            noise = torch.randn_like(x_start)

        sqrt_alpha_bar = extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alpha_bar = extract(
            self.sqrt_one_minus_alphas_cumprod, t, x_start.shape
        )
        return sqrt_alpha_bar * x_start + sqrt_one_minus_alpha_bar * noise

    def compute_loss(
        self,
        x_start: torch.Tensor,
        t: Optional[torch.Tensor] = None,
        noise: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Computes L_simple: MSE between injected noise and network predicted noise.
        """
        b = x_start.shape[0]
        device = x_start.device

        if t is None:
            t = torch.randint(0, self.timesteps, (b,), device=device, dtype=torch.long)
        if noise is None:
            noise = torch.randn_like(x_start)

        x_t = self.q_sample(x_start, t, noise)
        noise_pred = self.model(x_t, t)
        return F.mse_loss(noise_pred, noise)

    def p_mean_variance(
        self, x: torch.Tensor, t: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Predicts mean and variance of reverse step p_theta(x_{t-1} | x_t).
        """
        noise_pred = self.model(x, t)

        # Predict x_0 from x_t and noise_pred
        sqrt_recip_alphas_cumprod = extract(
            1.0 / self.sqrt_alphas_cumprod, t, x.shape
        )
        sqrt_recipm1_alphas_cumprod = extract(
            self.sqrt_one_minus_alphas_cumprod / self.sqrt_alphas_cumprod, t, x.shape
        )
        x_recon = sqrt_recip_alphas_cumprod * x - sqrt_recipm1_alphas_cumprod * noise_pred

        # Clamp x_0 to [-1, 1] for stable sampling
        x_recon = torch.clamp(x_recon, -1.0, 1.0)

        # Posterior mean
        c1 = extract(self.posterior_mean_coef1, t, x.shape)
        c2 = extract(self.posterior_mean_coef2, t, x.shape)
        model_mean = c1 * x_recon + c2 * x

        model_variance = extract(self.posterior_variance, t, x.shape)
        model_log_variance = extract(self.posterior_log_variance_clipped, t, x.shape)
        return model_mean, model_variance, model_log_variance

    @torch.no_grad()
    def p_sample(self, x: torch.Tensor, t: int) -> torch.Tensor:
        """
        Single ancestral reverse step: samples x_{t-1} given x_t.
        """
        b = x.shape[0]
        device = x.device
        t_batch = torch.full((b,), t, device=device, dtype=torch.long)

        model_mean, _, model_log_variance = self.p_mean_variance(x, t_batch)
        noise = torch.randn_like(x) if t > 0 else 0.0
        return model_mean + torch.exp(0.5 * model_log_variance) * noise

    @torch.no_grad()
    def sample(
        self, shape: Tuple[int, ...], device: Optional[torch.device] = None
    ) -> torch.Tensor:
        """
        Full reverse diffusion generation starting from x_T ~ N(0, I).
        """
        if device is None:
            device = next(self.model.parameters()).device

        img = torch.randn(shape, device=device)
        for t in reversed(range(self.timesteps)):
            img = self.p_sample(img, t)
        return img
