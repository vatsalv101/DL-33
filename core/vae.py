from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """
    Reparameterization Trick: z = mu + sigma * eps, where eps ~ N(0, I).
    Enables backpropagation through stochastic latent bottleneck.

    Args:
        mu: Mean tensor of shape (B, D)
        logvar: Log-variance tensor log(sigma^2) of shape (B, D)

    Returns:
        Sampled latent vector z of shape (B, D)
    """
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + eps * std


def vae_loss(
    recon_x: torch.Tensor,
    x: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    beta: float = 1.0,
    recon_type: str = "mse",
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Computes negative Evidence Lower Bound (ELBO):
    Loss = Reconstruction_Loss + beta * KL_Divergence

    KL Divergence between N(mu, sigma^2) and N(0, I):
    D_KL = -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)

    Args:
        recon_x: Reconstructed inputs, shape (B, C, H, W)
        x: Ground truth inputs, shape (B, C, H, W)
        mu: Latent mean, shape (B, D)
        logvar: Latent log-variance, shape (B, D)
        beta: Weighting factor for KL divergence (beta-VAE)
        recon_type: "mse" (Gaussian likelihood) or "bce" (Bernoulli likelihood)

    Returns:
        Tuple: (total_loss, recon_loss, kl_loss) all averaged per sample (batch mean).
    """
    batch_size = x.shape[0]

    if recon_type == "bce":
        recon_loss = F.binary_cross_entropy(recon_x, x, reduction="sum") / batch_size
    elif recon_type == "mse":
        recon_loss = F.mse_loss(recon_x, x, reduction="sum") / batch_size
    else:
        raise ValueError(f"Unknown recon_type: {recon_type}")

    kl_per_sample = -0.5 * torch.sum(1.0 + logvar - mu.pow(2) - logvar.exp(), dim=-1)
    kl_loss = torch.mean(kl_per_sample)

    total_loss = recon_loss + beta * kl_loss
    return total_loss, recon_loss, kl_loss


class ConvEncoder(nn.Module):
    """
    Convolutional encoder mapping image (B, C, H, W) to (mu, logvar).
    """

    def __init__(
        self,
        in_channels: int = 1,
        latent_dim: int = 20,
        hidden_dims: Tuple[int, ...] = (32, 64),
        img_size: int = 28,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.latent_dim = latent_dim
        self.img_size = img_size

        modules = []
        c_in = in_channels
        for c_out in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(c_in, c_out, kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2d(c_out),
                    nn.LeakyReLU(0.2, inplace=True),
                )
            )
            c_in = c_out
        self.conv = nn.Sequential(*modules)

        # Compute flat feature shape dynamically
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, img_size, img_size)
            dummy_out = self.conv(dummy)
            self.spatial_shape = dummy_out.shape[1:]  # (C, H, W)
            self.flat_dim = dummy_out.numel()

        self.fc_mu = nn.Linear(self.flat_dim, latent_dim)
        self.fc_logvar = nn.Linear(self.flat_dim, latent_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        feat = self.conv(x)
        feat = feat.flatten(start_dim=1)
        mu = self.fc_mu(feat)
        logvar = self.fc_logvar(feat)
        return mu, logvar


class ConvDecoder(nn.Module):
    """
    Convolutional decoder mapping latent code z (B, D) to reconstructed image (B, C, H, W).
    """

    def __init__(
        self,
        latent_dim: int = 20,
        out_channels: int = 1,
        hidden_dims: Tuple[int, ...] = (64, 32),
        spatial_shape: Tuple[int, int, int] = (64, 7, 7),
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.spatial_shape = spatial_shape
        self.flat_dim = spatial_shape[0] * spatial_shape[1] * spatial_shape[2]

        self.fc = nn.Linear(latent_dim, self.flat_dim)

        modules = []
        c_in = spatial_shape[0]
        # Transpose conv layers
        for i, c_out in enumerate(hidden_dims[1:]):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(
                        c_in, c_out, kernel_size=4, stride=2, padding=1
                    ),
                    nn.BatchNorm2d(c_out),
                    nn.LeakyReLU(0.2, inplace=True),
                )
            )
            c_in = c_out

        # Final projection to out_channels
        modules.append(
            nn.Sequential(
                nn.ConvTranspose2d(
                    c_in, out_channels, kernel_size=4, stride=2, padding=1
                ),
                nn.Sigmoid(),
            )
        )
        self.deconv = nn.Sequential(*modules)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        feat = self.fc(z)
        feat = feat.view(-1, *self.spatial_shape)
        return self.deconv(feat)


class VAE(nn.Module):
    """
    Variational Autoencoder (VAE) (Kingma & Welling, 2013).
    """

    def __init__(
        self,
        in_channels: int = 1,
        latent_dim: int = 20,
        hidden_dims: Tuple[int, ...] = (32, 64),
        img_size: int = 28,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.latent_dim = latent_dim
        self.img_size = img_size

        self.encoder = ConvEncoder(
            in_channels=in_channels,
            latent_dim=latent_dim,
            hidden_dims=hidden_dims,
            img_size=img_size,
        )

        decoder_hidden = tuple(reversed(hidden_dims))
        self.decoder = ConvDecoder(
            latent_dim=latent_dim,
            out_channels=in_channels,
            hidden_dims=decoder_hidden,
            spatial_shape=self.encoder.spatial_shape,
        )

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.encoder(x)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        return reparameterize(mu, logvar)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Full forward pass: encode -> reparameterize -> decode.
        Returns:
            recon_x: (B, C, H, W)
            mu: (B, D)
            logvar: (B, D)
        """
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z)
        return recon_x, mu, logvar

    @torch.no_grad()
    def sample(
        self, num_samples: int = 16, device: Optional[torch.device] = None
    ) -> torch.Tensor:
        """
        Sample new images from the prior N(0, I).
        """
        if device is None:
            device = next(self.parameters()).device
        z = torch.randn(num_samples, self.latent_dim, device=device)
        return self.decode(z)

    @torch.no_grad()
    def interpolate(
        self, z1: torch.Tensor, z2: torch.Tensor, steps: int = 10
    ) -> torch.Tensor:
        """
        Linear interpolation between two latent points z1 and z2.
        """
        alphas = torch.linspace(0.0, 1.0, steps, device=z1.device).view(-1, 1)
        z_interp = (1.0 - alphas) * z1.unsqueeze(0) + alphas * z2.unsqueeze(0)
        return self.decode(z_interp)
