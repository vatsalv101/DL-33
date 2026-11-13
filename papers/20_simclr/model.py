from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from core.models import ResNet, BasicBlock


def nt_xent_loss(
    z1: torch.Tensor,
    z2: torch.Tensor,
    temperature: float = 0.5,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Computes Normalized Temperature-scaled Cross Entropy (NT-Xent) loss between two augmented views.

    Args:
        z1: Projected representations of view 1 of shape (N, D)
        z2: Projected representations of view 2 of shape (N, D)
        temperature: Softmax temperature parameter tau
    Returns:
        Tuple[Tensor, Tensor]: (scalar NT-Xent loss, top-1 contrastive accuracy)
    """
    # TODO:
    N, D = z1.shape

    z1_norm = F.normalize(z1, dim=-1)
    z2_norm = F.normalize(z2, dim=-1)

    z = torch.cat([z1_norm, z2_norm], dim=0)

    sim_matrix = torch.matmul(z, z.T) / temperature

    diag_mask = torch.eye(2 * N, dtype=torch.bool, device=z.device)
    sim_matrix.masked_fill_(diag_mask, -1e9)

    targets = torch.cat([
        torch.arange(N, 2 * N, device=z.device),
        torch.arange(0, N, device=z.device),
    ], dim=0)

    loss = F.cross_entropy(sim_matrix, targets)
    acc = (sim_matrix.argmax(dim=-1) == targets).float().mean()
    return loss, acc


class SimCLRProjectionHead(nn.Module):
    """
    2-Layer Non-Linear MLP projection head g(h) = W2 * ReLU(W1 * h).
    Discards augmentation nuisance variables while preserving downstream semantic features in h.
    """

    def __init__(self, in_dim: int = 512, hidden_dim: int = 512, out_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim, bias=True),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        # TODO: Forward h through self.net
        return self.net(h)

class SmallResNetBackbone(nn.Module):
    """
    Compact ResNet feature extractor for rapid training and testing.
    """

    def __init__(self, in_channels: int = 3, feat_dim: int = 128):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.layer1 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, feat_dim, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(feat_dim),
            nn.ReLU(inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO: Forward x through conv1, layer1, layer2, and pool, then flatten
        x = F.relu(self.bn1(self.conv1(x)), inplace=True)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.pool(x)
        return torch.flatten(x, 1)


class SimCLR(nn.Module):
    """
    SimCLR framework:
    - Base encoder f(·) producing representation h
    - Projection head g(·) producing latent vector z
    - NT-Xent loss optimization
    """

    def __init__(
        self,
        backbone: Optional[nn.Module] = None,
        feat_dim: int = 128,
        proj_dim: int = 64,
        temperature: float = 0.5,
    ):
        super().__init__()
        self.backbone = backbone if backbone is not None else SmallResNetBackbone(feat_dim=feat_dim)
        self.projector = SimCLRProjectionHead(in_dim=feat_dim, hidden_dim=feat_dim, out_dim=proj_dim)
        self.temperature = temperature

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Extracts representations h used for downstream transfer tasks."""
        # TODO: Forward through backbone
        return self.backbone(x)

    def forward(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor,
        temperature: Optional[float] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass for a pair of augmented views (x1, x2).
        Returns:
            Tuple: (loss, acc, z1, z2, h1, h2)
        """
        # TODO:
        tau = temperature if temperature is not None else self.temperature

        h1 = self.backbone(x1)
        h2 = self.backbone(x2)

        z1 = self.projector(h1)
        z2 = self.projector(h2)

        loss, acc = nt_xent_loss(z1, z2, temperature=tau)
        return loss, acc, z1, z2, h1, h2


class LinearProbe(nn.Module):
    """
    Linear classifier trained on frozen backbone representations h to evaluate transfer quality.
    """

    def __init__(self, feat_dim: int, num_classes: int):
        super().__init__()
        self.classifier = nn.Linear(feat_dim, num_classes)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        # TODO: Forward h through self.classifier
        return self.classifier(h)
