from typing import Optional, Tuple, List, Union
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from core.ddpm import SinusoidalPosEmb, extract
from core.ddim import DDIMScheduler


class ConditionedResidualBlock(nn.Module):
    """
    Residual block with combined timestep and class conditioning injection.
    """

    def __init__(self, in_channels: int, out_channels: int, cond_dim: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.gn1 = nn.GroupNorm(8, out_channels)
        self.cond_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_dim, out_channels),
        )
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.gn2 = nn.GroupNorm(8, out_channels)

        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor, cond_emb: torch.Tensor) -> torch.Tensor:
        # TODO: 
        h = F.silu(self.gn1(self.conv1(x)))
        proj = self.cond_mlp(cond_emb)[:, :, None, None]
        h = h + proj
        h = F.silu(self.gn2(self.conv2(h)))
        return h + self.shortcut(x)


class ClassConditionedUNet(nn.Module):
    """
    U-Net Denoiser supporting Classifier-Free Guidance via label embedding
    and a reserved unconditional null token.
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        num_classes: int = 10,
        hidden_dims: Tuple[int, ...] = (32, 64, 128),
        cond_dim: int = 64,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.null_token_id = num_classes
        self.cond_dim = cond_dim

        # Timestep embedding MLP
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(cond_dim),
            nn.Linear(cond_dim, cond_dim * 2),
            nn.SiLU(),
            nn.Linear(cond_dim * 2, cond_dim),
        )

        # Class embedding table: index num_classes is the unconditional null token \emptyset
        self.class_embedding = nn.Embedding(num_classes + 1, cond_dim)
        self.class_mlp = nn.Sequential(
            nn.Linear(cond_dim, cond_dim),
            nn.SiLU(),
            nn.Linear(cond_dim, cond_dim),
        )

        c1, c2, c3 = hidden_dims
        # Downsampling path
        self.init_conv = nn.Conv2d(in_channels, c1, kernel_size=3, padding=1)
        self.down1 = ConditionedResidualBlock(c1, c1, cond_dim)
        self.downsample1 = nn.Conv2d(c1, c2, kernel_size=4, stride=2, padding=1)

        self.down2 = ConditionedResidualBlock(c2, c2, cond_dim)
        self.downsample2 = nn.Conv2d(c2, c3, kernel_size=4, stride=2, padding=1)

        # Bottleneck
        self.mid1 = ConditionedResidualBlock(c3, c3, cond_dim)
        self.mid2 = ConditionedResidualBlock(c3, c3, cond_dim)

        # Upsampling path
        self.upsample1 = nn.ConvTranspose2d(c3, c2, kernel_size=4, stride=2, padding=1)
        self.up1 = ConditionedResidualBlock(c2 * 2, c2, cond_dim)

        self.upsample2 = nn.ConvTranspose2d(c2, c1, kernel_size=4, stride=2, padding=1)
        self.up2 = ConditionedResidualBlock(c1 * 2, c1, cond_dim)

        self.out_conv = nn.Conv2d(c1, out_channels, kernel_size=3, padding=1)

    def get_conditioning(self, t: torch.Tensor, y: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Computes combined conditioning vector: cond = time_emb + class_emb."""
        # TODO: 
        t_emb = self.time_mlp(t)
        if y is None:
            y = torch.full((t.shape[0],), self.null_token_id, device=t.device, dtype=torch.long)
        c_emb = self.class_mlp(self.class_embedding(y))
        return t_emb + c_emb

    def forward(
        self, x: torch.Tensor, t: torch.Tensor, y: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # TODO: 
        cond_emb = self.get_conditioning(t, y)

        x1 = self.init_conv(x)
        x1 = self.down1(x1, cond_emb)
        x2 = self.downsample1(x1)

        x2 = self.down2(x2, cond_emb)
        x3 = self.downsample2(x2)

        x_mid = self.mid1(x3, cond_emb)
        x_mid = self.mid2(x_mid, cond_emb)

        u1 = self.upsample1(x_mid)
        u1 = torch.cat([u1, x2], dim=1)
        u1 = self.up1(u1, cond_emb)

        u2 = self.upsample2(u1)
        u2 = torch.cat([u2, x1], dim=1)
        u2 = self.up2(u2, cond_emb)

        return self.out_conv(u2)


class ClassifierFreeDiffusion(nn.Module):
    """
    Classifier-Free Diffusion Guidance (CFG) wrapper (Ho & Salimans, 2022).
    Supports joint conditional/unconditional training and extrapolated guided sampling.
    """

    def __init__(
        self,
        model: ClassConditionedUNet,
        timesteps: int = 1000,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
    ):
        super().__init__()
        self.model = model
        self.timesteps = timesteps

        betas = torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float32)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))

        self.scheduler = DDIMScheduler(
            num_train_timesteps=timesteps,
            beta_start=beta_start,
            beta_end=beta_end,
            clip_sample=True,
        )

    def q_sample(
        self, x_start: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if noise is None:
            noise = torch.randn_like(x_start)
        sqrt_alpha = extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus = extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
        return sqrt_alpha * x_start + sqrt_one_minus * noise

    def compute_loss(
        self,
        x_start: torch.Tensor,
        y: torch.Tensor,
        p_uncond: float = 0.1,
        t: Optional[torch.Tensor] = None,
        noise: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Computes L_simple with conditioning dropout.
        With probability p_uncond, class label y is replaced by the null token.
        """
        # TODO: 
        b = x_start.shape[0]
        device = x_start.device

        if t is None:
            t = torch.randint(0, self.timesteps, (b,), device=device, dtype=torch.long)
        if noise is None:
            noise = torch.randn_like(x_start)

        if p_uncond > 0.0:
            mask = torch.rand(b, device=device) < p_uncond
            y_train = torch.where(
                mask,
                torch.full_like(y, self.model.null_token_id),
                y,
            )
        else:
            y_train = y

        x_t = self.q_sample(x_start, t, noise)
        noise_pred = self.model(x_t, t, y_train)
        return F.mse_loss(noise_pred, noise)

    def predict_guided_noise(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        y: torch.Tensor,
        guidance_scale: float = 3.0,
        y_neg: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Extrapolates guided noise via doubled parallel forward pass:
        eps_guided = eps_uncond + guidance_scale * (eps_cond - eps_uncond)
        """
        # TODO: 
        b = x_t.shape[0]
        device = x_t.device

        if guidance_scale == 1.0 and y_neg is None:
            return self.model(x_t, t, y)

        if y_neg is None:
            y_uncond = torch.full((b,), self.model.null_token_id, device=device, dtype=torch.long)
        else:
            y_uncond = y_neg

        x_in = torch.cat([x_t, x_t], dim=0)
        t_in = torch.cat([t, t], dim=0)
        y_in = torch.cat([y, y_uncond], dim=0)

        noise_pred = self.model(x_in, t_in, y_in)
        eps_cond, eps_uncond = noise_pred.chunk(2, dim=0)

        eps_guided = eps_uncond + guidance_scale * (eps_cond - eps_uncond)
        return eps_guided
    
    @torch.no_grad()
    def sample(
        self,
        y: torch.Tensor,
        shape: Optional[Tuple[int, ...]] = None,
        num_inference_steps: int = 20,
        guidance_scale: float = 3.0,
        eta: float = 0.0,
        y_neg: Optional[torch.Tensor] = None,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        """
        Accelerated DDIM sampling with Classifier-Free Guidance.
        """
        # TODO: 
        if device is None:
            device = next(self.model.parameters()).device

        b = y.shape[0]
        if shape is None:
            shape = (b, self.model.init_conv.in_channels, 28, 28)

        y = y.to(device)
        if y_neg is not None:
            y_neg = y_neg.to(device)

        timesteps = self.scheduler.set_timesteps(num_inference_steps, device=device)
        img = torch.randn(shape, device=device)

        for i, t in enumerate(timesteps):
            prev_t = timesteps[i + 1] if i + 1 < len(timesteps) else -1
            t_batch = torch.full((b,), t, device=device, dtype=torch.long)

            guided_noise = self.predict_guided_noise(
                img, t_batch, y, guidance_scale=guidance_scale, y_neg=y_neg
            )

            img, _ = self.scheduler.step(
                model_output=guided_noise,
                timestep=t.item(),
                prev_timestep=prev_t.item() if isinstance(prev_t, torch.Tensor) else prev_t,
                sample=img,
                eta=eta,
            )

        return img

