"""Core verified loss functions for dl-rl-robotics-80.

Paper 08: Focal Loss for Dense Object Detection (BinaryFocalLoss & FocalLoss)
Implemented and verified by the student.
"""

from typing import Optional, Union
import torch
import torch.nn as nn
import torch.nn.functional as F


class BinaryFocalLoss(nn.Module):
    """
    Numerically stable Binary Focal Loss for dense binary classification and detection.
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    Args:
        alpha     (float): Weighting factor in [0, 1] for class 1 (1 - alpha for class 0). Default: 0.25.
        gamma     (float): Focusing parameter for modulating factor (1 - p_t)^gamma. Default: 2.0.
        reduction   (str): Specifies reduction: 'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(
        self,
        alpha: Optional[float] = 0.25,
        gamma: float = 2.0,
        reduction: str = "mean",
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        targets = targets.type_as(inputs)
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
        p = torch.sigmoid(inputs)
        p_t = targets * p + (1.0 - targets) * (1.0 - p)
        modulating_factor = (1.0 - p_t) ** self.gamma

        if self.alpha is not None:
            alpha_t = targets * self.alpha + (1.0 - targets) * (1.0 - self.alpha)
            loss = alpha_t * modulating_factor * bce_loss
        else:
            loss = modulating_factor * bce_loss

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss


class FocalLoss(nn.Module):
    """
    Multi-class Focal Loss for dense prediction across C classes.
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    Args:
        alpha (float or Tensor, optional): Class weighting factor(s).
        gamma                     (float): Focusing parameter for modulating factor. Default: 2.0.
        reduction                   (str): 'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(
        self,
        alpha: Optional[Union[float, torch.Tensor]] = None,
        gamma: float = 2.0,
        reduction: str = "mean",
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(inputs, targets, reduction="none")
        p = F.softmax(inputs, dim=1)

        if inputs.dim() > 2:
            targets_expanded = targets.unsqueeze(1)
            p_t = p.gather(1, targets_expanded).squeeze(1)
        else:
            p_t = p.gather(1, targets.unsqueeze(1)).squeeze(1)

        modulating_factor = (1.0 - p_t) ** self.gamma
        loss = modulating_factor * ce_loss

        if self.alpha is not None:
            if isinstance(self.alpha, (float, int)):
                loss = self.alpha * loss
            elif isinstance(self.alpha, torch.Tensor):
                alpha = self.alpha.to(inputs.device)
                alpha_t = alpha[targets]
                loss = alpha_t * loss

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss
