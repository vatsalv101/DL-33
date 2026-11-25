from typing import Optional, Tuple, List
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from core.ddpm import UNetDenoiser, SinusoidalPosEmb, DiffusionResidualBlock


class DDIMScheduler(nn.Module):
    """
    Denoising Diffusion Implicit Models (DDIM) Scheduler (Song et al., ICLR 2021).
    Provides accelerated non-Markovian sampling and exact deterministic latent inversion.
    """

    def __init__(
        self,
        num_train_timesteps: int = 1000,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
        clip_sample: bool = True,
    ):
        super().__init__()
        self.num_train_timesteps = num_train_timesteps
        self.clip_sample = clip_sample

        betas = torch.linspace(beta_start, beta_end, num_train_timesteps, dtype=torch.float32)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)

        self.timesteps: Optional[torch.Tensor] = None

    def set_timesteps(self, num_inference_steps: int, device: Optional[torch.device] = None) -> torch.Tensor:
        """
        Creates evenly spaced sub-sequence of timesteps from 0 to num_train_timesteps - 1.
        """
        step_ratio = self.num_train_timesteps // num_inference_steps
        timesteps = (
            torch.arange(0, num_inference_steps, device=device) * step_ratio
        ).round().long()
        self.timesteps = timesteps.flip(0)  # [tau_S, tau_{S-1}, ..., tau_1]
        return self.timesteps

    def step(
        self,
        model_output: torch.Tensor,
        timestep: int,
        prev_timestep: int,
        sample: torch.Tensor,
        eta: float = 0.0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Performs one reverse DDIM step from timestep to prev_timestep.

        Args:
            model_output: Predicted noise eps_theta, shape (B, C, H, W)
            timestep: Current timestep tau_i
            prev_timestep: Target previous timestep tau_{i-1} (or -1 for final clean image)
            sample: Current noisy image x_{tau_i}
            eta: Stochasticity weight (0.0 = deterministic DDIM, 1.0 = DDPM)

        Returns:
            Tuple of (prev_sample x_{tau_{i-1}}, predicted clean image pred_x_0)
        """
        # TODO: 
        alpha_bar_t = self.alphas_cumprod[timestep]
        alpha_bar_t_prev = (
            self.alphas_cumprod[prev_timestep]
            if prev_timestep >= 0
            else torch.tensor(1.0, device=sample.device)
        )

        pred_x_0 = (sample - torch.sqrt(1.0 - alpha_bar_t) * model_output) / torch.sqrt(alpha_bar_t)
        if self.clip_sample:
            pred_x_0 = torch.clamp(pred_x_0, -1.0, 1.0)
        variance = (
            (1.0 - alpha_bar_t_prev) / (1.0 - alpha_bar_t)
            * (1.0 - alpha_bar_t / alpha_bar_t_prev)
        )
        sigma = eta * torch.sqrt(torch.clamp(variance, min=0.0))

        dir_xt = torch.sqrt(torch.clamp(1.0 - alpha_bar_t_prev - sigma ** 2, min=0.0)) * model_output

        noise = torch.randn_like(sample) if eta > 0.0 else 0.0
        prev_sample = torch.sqrt(alpha_bar_t_prev) * pred_x_0 + dir_xt + sigma * noise

        return prev_sample, pred_x_0
    
    def reverse_step(
        self,
        model_output: torch.Tensor,
        timestep: int,
        next_timestep: int,
        sample: torch.Tensor,
    ) -> torch.Tensor:
        """
        Performs one forward ODE step from timestep to next_timestep (DDIM Inversion).
        """
        # TODO: 
        alpha_bar_t = self.alphas_cumprod[timestep]
        alpha_bar_t_next = self.alphas_cumprod[next_timestep]

        pred_x_0 = (sample - torch.sqrt(1.0 - alpha_bar_t) * model_output) / torch.sqrt(alpha_bar_t)
        if self.clip_sample:
            pred_x_0 = torch.clamp(pred_x_0, -1.0, 1.0)

        next_sample = (
            torch.sqrt(alpha_bar_t_next) * pred_x_0
            + torch.sqrt(1.0 - alpha_bar_t_next) * model_output
        )
        return next_sample


class DDIM(nn.Module):
    """
    Denoising Diffusion Implicit Model wrapper uniting pre-trained U-Net with DDIMScheduler.
    """

    def __init__(
        self,
        model: nn.Module,
        num_train_timesteps: int = 1000,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
        clip_sample: bool = True,
    ):
        super().__init__()
        self.model = model
        self.scheduler = DDIMScheduler(
            num_train_timesteps=num_train_timesteps,
            beta_start=beta_start,
            beta_end=beta_end,
            clip_sample=clip_sample,
        )

    @torch.no_grad()
    def sample(
        self,
        shape: Tuple[int, ...],
        num_inference_steps: int = 20,
        eta: float = 0.0,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        """
        Generates images in num_inference_steps using DDIM sampling.
        """
        # TODO: 
        if device is None:
            device = next(self.model.parameters()).device

        timesteps = self.scheduler.set_timesteps(num_inference_steps, device=device)
        img = torch.randn(shape, device=device)

        for i, t in enumerate(timesteps):
            prev_t = timesteps[i + 1] if i + 1 < len(timesteps) else -1
            t_batch = torch.full((shape[0],), t, device=device, dtype=torch.long)
            noise_pred = self.model(img, t_batch)
            img, _ = self.scheduler.step(
                model_output=noise_pred,
                timestep=t.item(),
                prev_timestep=prev_t.item() if isinstance(prev_t, torch.Tensor) else prev_t,
                sample=img,
                eta=eta,
            )
        return img


    @torch.no_grad()
    def invert(
        self,
        x_0: torch.Tensor,
        num_inference_steps: int = 20,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        """
        Inverts real image x_0 to its deterministic latent representation x_T via DDIM Inversion.
        """
        # TODO: 
        if device is None:
            device = x_0.device

        timesteps = self.scheduler.set_timesteps(num_inference_steps, device=device).flip(0)
        img = x_0.clone()

        for i, t in enumerate(timesteps[:-1]):
            next_t = timesteps[i + 1]
            t_batch = torch.full((x_0.shape[0],), t, device=device, dtype=torch.long)
            noise_pred = self.model(img, t_batch)
            img = self.scheduler.reverse_step(
                model_output=noise_pred,
                timestep=t.item(),
                next_timestep=next_t.item(),
                sample=img,
            )

        return img

    @torch.no_grad()
    def interpolate(
        self,
        x_1: torch.Tensor,
        x_2: torch.Tensor,
        steps: int = 10,
        num_inference_steps: int = 20,
        eta: float = 0.0,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        """
        Interpolates smoothly between two real images via latent inversion and reconstruction.
        """
        # TODO: 
        if device is None:
            device = x_1.device

        z_1 = self.invert(x_1, num_inference_steps=num_inference_steps, device=device)
        z_2 = self.invert(x_2, num_inference_steps=num_inference_steps, device=device)

        alphas = torch.linspace(0.0, 1.0, steps, device=device)[:, None, None, None]
        z_interp = (1.0 - alphas) * z_1 + alphas * z_2

        timesteps = self.scheduler.set_timesteps(num_inference_steps, device=device)
        img = z_interp

        for i, t in enumerate(timesteps):
            prev_t = timesteps[i + 1] if i + 1 < len(timesteps) else -1
            t_batch = torch.full((steps,), t, device=device, dtype=torch.long)
            noise_pred = self.model(img, t_batch)
            img, _ = self.scheduler.step(
                model_output=noise_pred,
                timestep=t.item(),
                prev_timestep=prev_t.item() if isinstance(prev_t, torch.Tensor) else prev_t,
                sample=img,
                eta=eta,
            )
        return img
