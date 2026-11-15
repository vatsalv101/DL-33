from typing import Optional, Tuple, List
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

from core.vit import ViT


class DINOHead(nn.Module):
    """
    DINO projection head:
    3-layer MLP with L2-normalized bottleneck followed by a weight-normalized prototype layer.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int = 256,
        hidden_dim: int = 128,
        bottleneck_dim: int = 64,
        norm_last_layer: bool = True,
    ):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, bottleneck_dim),
        )
        self.last_layer = nn.Linear(bottleneck_dim, out_dim, bias=False)
        self.norm_last_layer = norm_last_layer

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO:
        x = self.mlp(x)
        x = F.normalize(x, dim=-1, p=2)
        if self.norm_last_layer:
            weight = F.normalize(self.last_layer.weight, dim=-1, p=2)
            logits = F.linear(x, weight)
        else:
            logits = self.last_layer(x)
        return logits


class DINOLoss(nn.Module):
    """
    Self-distillation cross-entropy loss with Centering and Sharpening.
    - Centering (subtraction of running mean) prevents dominant-dimension collapse.
    - Sharpening (lower teacher temperature) prevents uniform-distribution collapse.
    """

    def __init__(
        self,
        out_dim: int,
        student_temp: float = 0.1,
        center_momentum: float = 0.9,
    ):
        super().__init__()
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.register_buffer("center", torch.zeros(1, out_dim))

    @torch.no_grad()
    def update_center(self, teacher_output: torch.Tensor):
        """
        Momentum update of the center buffer:
        c = m * c + (1 - m) * mean(teacher_output)
        """
        # TODO:
        batch_center = torch.mean(teacher_output, dim=0, keepdim=True)
        self.center = self.center * self.center_momentum + batch_center * (1.0 - self.center_momentum)

    def forward(
        self,
        student_output: torch.Tensor,
        teacher_output: torch.Tensor,
        teacher_temp: float = 0.04,
    ) -> torch.Tensor:
        """
        Computes symmetric cross-entropy loss across 2 global views.

        Args:
            student_output: Concatenated student predictions (2*B, out_dim)
            teacher_output: Concatenated teacher predictions (2*B, out_dim)
            teacher_temp:   Sharpening temperature for teacher predictions
        Returns:
            Scalar cross-entropy loss
        """
        # TODO:
        s_out = student_output / self.student_temp
        s_chunk = s_out.chunk(2, dim=0)

        t_out = F.softmax((teacher_output - self.center) / teacher_temp, dim=-1).detach()
        t_chunk = t_out.chunk(2, dim=0)

        loss1 = torch.sum(-t_chunk[1] * F.log_softmax(s_chunk[0], dim=-1), dim=-1).mean()
        loss2 = torch.sum(-t_chunk[0] * F.log_softmax(s_chunk[1], dim=-1), dim=-1).mean()
        total_loss = (loss1 + loss2) / 2.0

        self.update_center(teacher_output)

        return total_loss


class CompactViT(nn.Module):
    """
    Lightweight Vision Transformer backbone for rapid local execution and testing.
    """

    def __init__(
        self,
        img_size: int = 16,
        patch_size: int = 4,
        in_chans: int = 3,
        embed_dim: int = 64,
        depth: int = 2,
        num_heads: int = 2,
    ):
        super().__init__()
        self.vit = ViT(
            img_size=img_size,
            patch_size=patch_size,
            in_channels=in_chans,
            num_classes=embed_dim,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=2.0,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Extract [CLS] token representation from forward_features
        return self.vit.forward_features(x)[:, 0]


class DINO(nn.Module):
    """
    DINO: Self-distillation with no labels.
    Comprises student and teacher networks with EMA updates and DINOLoss.
    """

    def __init__(
        self,
        backbone: Optional[nn.Module] = None,
        embed_dim: int = 64,
        out_dim: int = 128,
        student_temp: float = 0.1,
        teacher_temp: float = 0.04,
        momentum_teacher: float = 0.999,
    ):
        super().__init__()
        self.student_temp = student_temp
        self.teacher_temp = teacher_temp
        self.momentum_teacher = momentum_teacher

        # Base backbone
        base = backbone if backbone is not None else CompactViT(embed_dim=embed_dim)
        head = DINOHead(in_dim=embed_dim, out_dim=out_dim)

        # Student network: backbone + head
        self.student = nn.Sequential(base, head)

        # Teacher network: identical architecture, no gradients
        self.teacher = copy.deepcopy(self.student)
        for param in self.teacher.parameters():
            param.requires_grad = False

        self.loss_fn = DINOLoss(out_dim=out_dim, student_temp=student_temp)

    @torch.no_grad()
    def _update_teacher(self):
        """EMA update for teacher network: theta_t = m * theta_t + (1 - m) * theta_s"""
        # TODO: Update param_t.data = param_t.data * self.momentum_teacher + param_s.data * (1.0 - self.momentum_teacher)
        for param_s, param_t in zip(self.student.parameters(), self.teacher.parameters()):
            param_t.data = param_t.data * self.momentum_teacher + param_s.data * (1.0 - self.momentum_teacher)

    def forward(
        self, x1: torch.Tensor, x2: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass for two views x1, x2 of shape (B, 3, H, W).
        Returns:
            Tuple: (loss, student_output, teacher_output)
        """
        # TODO:
        s_out1 = self.student(x1)
        s_out2 = self.student(x2)
        s_out = torch.cat([s_out1, s_out2], dim=0)

        # Teacher forward pass (no gradients, with EMA update)
        with torch.no_grad():
            self._update_teacher()
            t_out1 = self.teacher(x1)
            t_out2 = self.teacher(x2)
            t_out = torch.cat([t_out1, t_out2], dim=0)

        loss = self.loss_fn(s_out, t_out, teacher_temp=self.teacher_temp)
        return loss, s_out, t_out
