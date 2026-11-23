from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class VQVAEResidualBlock(nn.Module):
    """
    Residual block with 3x3 convolutions and ReLU activations for VQ-VAE.
    """

    def __init__(self, channels: int, hidden_dim: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(channels, hidden_dim, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class VectorQuantizer(nn.Module):
    """
    Vector Quantization (VQ) bottleneck layer (van den Oord et al., NeurIPS 2017).

    Maps continuous encoder representations z_e to discrete codebook vectors z_q
    via minimum Euclidean distance lookup, using the Straight-Through Estimator (STE).
    """

    def __init__(
        self,
        num_embeddings: int = 128,
        embedding_dim: int = 32,
        commitment_cost: float = 0.25,
    ):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost

        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight.data.uniform_(
            -1.0 / num_embeddings, 1.0 / num_embeddings
        )

    def forward(
        self, z: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass with straight-through gradient estimation.

        Args:
            z: Continuous latents from encoder, shape (B, D, H, W)

        Returns:
            quantized: Quantized latents with straight-through gradients, shape (B, D, H, W)
            loss: Combined VQ loss + commitment loss, scalar
            perplexity: Codebook perplexity measuring dictionary utilization, scalar
            encoding_indices: Codebook indices chosen, shape (B, H, W)
        """
        # Permute (B, D, H, W) -> (B, H, W, D)
        z_permuted = z.permute(0, 2, 3, 1).contiguous()
        flat_z = z_permuted.view(-1, self.embedding_dim)

        # Pairwise Euclidean distances: ||z - e_j||^2 = ||z||^2 - 2 z @ e_j^T + ||e_j||^2
        distances = (
            torch.sum(flat_z ** 2, dim=1, keepdim=True)
            + torch.sum(self.embedding.weight ** 2, dim=1)
            - 2.0 * torch.matmul(flat_z, self.embedding.weight.t())
        )

        # Discrete codebook indices
        encoding_indices = torch.argmin(distances, dim=1)

        # Look up quantized representations
        quantized = self.embedding(encoding_indices).view_as(z_permuted)

        # VQ loss (trains codebook) + Commitment loss (trains encoder)
        vq_loss = F.mse_loss(quantized, z_permuted.detach())
        commitment_loss = F.mse_loss(quantized.detach(), z_permuted)
        loss = vq_loss + self.commitment_cost * commitment_loss

        # Straight-Through Estimator: forward uses quantized, backward flows to z_permuted
        quantized_st = z_permuted + (quantized - z_permuted).detach()

        # Reshape back to channel-first: (B, D, H, W)
        quantized_out = quantized_st.permute(0, 3, 1, 2).contiguous()

        # Compute codebook perplexity: P = exp(-sum(p_k * log(p_k)))
        encodings = torch.zeros(
            encoding_indices.shape[0], self.num_embeddings, device=z.device
        )
        encodings.scatter_(1, encoding_indices.unsqueeze(1), 1.0)
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        B, _, H, W = z.shape
        return quantized_out, loss, perplexity, encoding_indices.view(B, H, W)


class VQVAEEncoder(nn.Module):
    """
    Convolutional encoder mapping image (B, C_in, H, W) to (B, D, H/4, W/4).
    """

    def __init__(
        self,
        in_channels: int = 1,
        hidden_dim: int = 64,
        embedding_dim: int = 32,
    ):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, hidden_dim // 2, kernel_size=4, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(hidden_dim // 2)
        self.conv2 = nn.Conv2d(hidden_dim // 2, hidden_dim, kernel_size=4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(hidden_dim)

        self.res1 = VQVAEResidualBlock(hidden_dim, hidden_dim)
        self.res2 = VQVAEResidualBlock(hidden_dim, hidden_dim)

        self.proj = nn.Conv2d(hidden_dim, embedding_dim, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.res1(x)
        x = self.res2(x)
        return self.proj(x)


class VQVAEDecoder(nn.Module):
    """
    Convolutional decoder mapping quantized latents (B, D, H/4, W/4) to reconstructed image (B, C_in, H, W).
    """

    def __init__(
        self,
        embedding_dim: int = 32,
        hidden_dim: int = 64,
        out_channels: int = 1,
    ):
        super().__init__()
        self.proj = nn.Conv2d(embedding_dim, hidden_dim, kernel_size=3, padding=1)
        self.res1 = VQVAEResidualBlock(hidden_dim, hidden_dim)
        self.res2 = VQVAEResidualBlock(hidden_dim, hidden_dim)

        self.deconv1 = nn.ConvTranspose2d(hidden_dim, hidden_dim // 2, kernel_size=4, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(hidden_dim // 2)
        self.deconv2 = nn.ConvTranspose2d(hidden_dim // 2, out_channels, kernel_size=4, stride=2, padding=1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x = self.proj(z)
        x = self.res1(x)
        x = self.res2(x)
        x = F.relu(self.bn1(self.deconv1(x)))
        x = torch.sigmoid(self.deconv2(x))
        return x


class VQVAE(nn.Module):
    """
    Vector Quantized Variational AutoEncoder (VQ-VAE) (van den Oord et al., 2017).
    """

    def __init__(
        self,
        in_channels: int = 1,
        hidden_dim: int = 64,
        embedding_dim: int = 32,
        num_embeddings: int = 128,
        commitment_cost: float = 0.25,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings

        self.encoder = VQVAEEncoder(
            in_channels=in_channels,
            hidden_dim=hidden_dim,
            embedding_dim=embedding_dim,
        )
        self.quantizer = VectorQuantizer(
            num_embeddings=num_embeddings,
            embedding_dim=embedding_dim,
            commitment_cost=commitment_cost,
        )
        self.decoder = VQVAEDecoder(
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            out_channels=in_channels,
        )

    def encode(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encodes image to continuous and quantized representations."""
        z_e = self.encoder(x)
        z_q, vq_loss, perplexity, indices = self.quantizer(z_e)
        return z_q, vq_loss, perplexity, indices

    def decode(self, z_q: torch.Tensor) -> torch.Tensor:
        """Decodes quantized representations to reconstructed image."""
        return self.decoder(z_q)

    def decode_code(self, indices: torch.Tensor) -> torch.Tensor:
        """Decodes discrete codebook indices directly to reconstructed image."""
        B, H, W = indices.shape
        quantized = self.quantizer.embedding(indices)  # (B, H, W, D)
        quantized = quantized.permute(0, 3, 1, 2).contiguous()  # (B, D, H, W)
        return self.decoder(quantized)

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Full forward pass:
        Returns:
            x_recon: Reconstructed image (B, C, H, W)
            vq_loss: Vector quantization and commitment loss
            perplexity: Codebook perplexity
            indices: Chosen discrete codebook indices (B, H', W')
        """
        z_e = self.encoder(x)
        z_q, vq_loss, perplexity, indices = self.quantizer(z_e)
        x_recon = self.decoder(z_q)
        return x_recon, vq_loss, perplexity, indices
