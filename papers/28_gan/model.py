from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_gradient_penalty(
    critic: nn.Module,
    real_images: torch.Tensor,
    fake_images: torch.Tensor,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """
    Computes 1-Lipschitz Gradient Penalty for WGAN-GP:
    GP = E_x_hat [(||grad_x_hat D(x_hat)||_2 - 1)^2]
    where x_hat = epsilon * real + (1 - epsilon) * fake for epsilon ~ U(0, 1).

    Args:
        critic: Critic network D(x)
        real_images: Batch of real images, shape (B, C, H, W)
        fake_images: Batch of generated images, shape (B, C, H, W)
        device: Torch device

    Returns:
        Scalar gradient penalty tensor
    """
    # TODO: Return mean((gradient_norm - 1.0) ** 2)
    if device is None:
        device = real_images.device

    batch_size = real_images.shape[0]
    epsilon = torch.rand(batch_size, 1, 1, 1, device=device)
    interpolates = (epsilon * real_images + (1.0 - epsilon) * fake_images).requires_grad_(True)

    d_interpolates = critic(interpolates)

    grad_outputs = torch.ones_like(d_interpolates)
    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=grad_outputs,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]

    gradients = gradients.view(batch_size, -1)
    gradient_norm = gradients.norm(2, dim=1)

    gradient_penalty = torch.mean((gradient_norm - 1.0) ** 2)
    return gradient_penalty

def critic_loss(
    critic: nn.Module,
    real_images: torch.Tensor,
    fake_images: torch.Tensor,
    lambda_gp: float = 10.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Computes WGAN-GP Critic loss:
    L_critic = E[D(fake)] - E[D(real)] + lambda_gp * GP

    Returns:
        Tuple: (total_critic_loss, wasserstein_loss, gradient_penalty)
    """
    # TODO: Return (total_loss, w_loss, gp)
    d_real = critic(real_images)
    d_fake = critic(fake_images)

    w_loss = torch.mean(d_fake) - torch.mean(d_real)
    gp = compute_gradient_penalty(critic, real_images, fake_images, device=real_images.device)

    total_loss = w_loss + lambda_gp * gp
    return total_loss, w_loss, gp



def generator_loss(critic: nn.Module, fake_images: torch.Tensor) -> torch.Tensor:
    """
    Computes WGAN Generator loss:
    L_gen = -E[D(fake)]
    """
    # TODO: Evaluate critic on fake_images: d_fake
    d_fake = critic(fake_images)
    return -torch.mean(d_fake)


class Generator(nn.Module):
    """
    Convolutional generator mapping noise z ~ N(0, I) to image in [-1, 1].
    """

    def __init__(
        self,
        latent_dim: int = 64,
        img_channels: int = 1,
        hidden_dim: int = 64,
        img_size: int = 28,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.img_size = img_size
        self.init_size = img_size // 4
        self.hidden_dim = hidden_dim

        self.fc = nn.Linear(latent_dim, hidden_dim * 4 * self.init_size * self.init_size)

        self.deconv1 = nn.ConvTranspose2d(hidden_dim * 4, hidden_dim * 2, kernel_size=4, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(hidden_dim * 2)

        self.deconv2 = nn.ConvTranspose2d(hidden_dim * 2, hidden_dim, kernel_size=4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(hidden_dim)

        self.out_conv = nn.Conv2d(hidden_dim, img_channels, kernel_size=3, padding=1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x = self.fc(z)
        x = x.view(-1, self.hidden_dim * 4, self.init_size, self.init_size)
        x = F.relu(self.bn1(self.deconv1(x)))
        x = F.relu(self.bn2(self.deconv2(x)))
        return torch.tanh(self.out_conv(x))


class Critic(nn.Module):
    """
    Convolutional critic mapping image in [-1, 1] to unbounded scalar score D(x).
    Uses Layer/Group Normalization (never Batch Normalization to preserve GP independence).
    """

    def __init__(
        self,
        img_channels: int = 1,
        hidden_dim: int = 64,
        img_size: int = 28,
    ):
        super().__init__()
        self.conv1 = nn.Conv2d(img_channels, hidden_dim, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(hidden_dim, hidden_dim * 2, kernel_size=4, stride=2, padding=1)
        self.gn2 = nn.GroupNorm(1, hidden_dim * 2)

        self.conv3 = nn.Conv2d(hidden_dim * 2, hidden_dim * 4, kernel_size=3, stride=1, padding=1)
        self.gn3 = nn.GroupNorm(1, hidden_dim * 4)

        final_spatial = (img_size // 4) ** 2
        self.fc = nn.Linear(hidden_dim * 4 * final_spatial, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.leaky_relu(self.conv1(x), 0.2)
        x = F.leaky_relu(self.gn2(self.conv2(x)), 0.2)
        x = F.leaky_relu(self.gn3(self.conv3(x)), 0.2)
        x = x.flatten(start_dim=1)
        return self.fc(x)



class WGANGP(nn.Module):
    """
    Wasserstein GAN with Gradient Penalty (Gulrajani et al., NeurIPS 2017).
    """

    def __init__(
        self,
        latent_dim: int = 64,
        img_channels: int = 1,
        hidden_dim: int = 64,
        img_size: int = 28,
        lambda_gp: float = 10.0,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.lambda_gp = lambda_gp

        self.generator = Generator(
            latent_dim=latent_dim,
            img_channels=img_channels,
            hidden_dim=hidden_dim,
            img_size=img_size,
        )
        self.critic = Critic(
            img_channels=img_channels,
            hidden_dim=hidden_dim,
            img_size=img_size,
        )

    def generate(self, z: torch.Tensor) -> torch.Tensor:
        # TODO: Delegate to self.generator
        return self.generator(z)

    def score(self, x: torch.Tensor) -> torch.Tensor:
        # TODO: Delegate to self.critic
        return self.critic(x)

    @torch.no_grad()
    def sample(
        self, num_samples: int = 16, device: Optional[torch.device] = None
    ) -> torch.Tensor:
        """Samples generated images from latent prior N(0, I)."""
        # TODO: Sample random z from standard normal distribution
        # TODO: Pass through generator and return sampled images
        if device is None:
            device = next(self.generator.parameters()).device
        z = torch.randn(num_samples, self.latent_dim, device=device)
        return self.generator(z)
