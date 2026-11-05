import math
from typing import List, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRALinear(nn.Module):
    """
    Low-Rank Adaptation (LoRA) wrapper around a Linear layer.
    h = W_0 x + (alpha / r) * B * A * x

    Args:
        base_layer   (nn.Linear, optional): Existing linear layer to wrap.
        in_features  (int, optional): Input features (if base_layer not provided).
        out_features (int, optional): Output features (if base_layer not provided).
        r            (int): Low-rank adapter dimension. Default: 4.
        lora_alpha   (float): Scaling constant. Default: 1.0.
        lora_dropout (float): Dropout probability applied to adapter input. Default: 0.0.
        bias         (bool): Whether base linear layer includes bias. Default: True.
    """

    def __init__(
        self,
        base_layer: Optional[nn.Linear] = None,
        in_features: Optional[int] = None,
        out_features: Optional[int] = None,
        r: int = 4,
        lora_alpha: float = 1.0,
        lora_dropout: float = 0.0,
        bias: bool = True,
    ):
        super().__init__()
        assert (base_layer is not None) or (in_features is not None and out_features is not None), (
            "Must provide either base_layer or both in_features and out_features."
        )

        if base_layer is not None:
            self.in_features = base_layer.in_features
            self.out_features = base_layer.out_features
            self.linear = base_layer
        else:
            self.in_features = in_features
            self.out_features = out_features
            self.linear = nn.Linear(in_features, out_features, bias=bias)

        self.r = r
        self.lora_alpha = lora_alpha
        self.scaling = lora_alpha / r if r > 0 else 1.0

        if r > 0:
            self.lora_A = nn.Linear(self.in_features, r, bias=False)
            self.lora_B = nn.Linear(r, self.out_features, bias=False)
            self.lora_dropout = nn.Dropout(p=lora_dropout) if lora_dropout > 0.0 else nn.Identity()
            self.reset_lora_parameters()
        else:
            self.lora_A = None
            self.lora_B = None
            self.lora_dropout = nn.Identity()

        self.merged = False

    def reset_lora_parameters(self):
        """
        Initializes matrix A with Kaiming uniform and matrix B with exact zeros.
        Guarantees Delta W = 0 at training step 0.
        """
        if self.r > 0:
            nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B.weight)

    def get_delta_weight(self) -> torch.Tensor:
        """
        Computes Delta W = (alpha / r) * B * A of shape (out_features, in_features).

        Returns:
            Tensor: Delta weight matrix of shape (out_features, in_features).
        """
        ################################################################
        # TODO: Implement get_delta_weight
        # 1. If self.r == 0, return zeros of shape (out_features, in_features)
        if self.r == 0:
            return torch.zeros((self.out_features, self.in_features), device=self.linear.weight.device)
        
        # 2. Return (self.lora_B.weight @ self.lora_A.weight) * self.scaling
        return (self.lora_B.weight @ self.lora_A.weight) * self.scaling
        ################################################################

    def merge(self):
        """
        Merges adapter weights into the base layer weight for zero-latency inference:
        W_0 <- W_0 + Delta W
        """
        ################################################################
        # TODO: Implement merge
        if not self.merged and self.r > 0:
            self.linear.weight.data += self.get_delta_weight()
            self.merged = True
        ################################################################

    def unmerge(self):
        """
        Subtracts adapter weights from base layer weight to restore original weights:
        W_0 <- W_0 - Delta W
        """
        ################################################################
        # TODO: Implement unmerge
        if self.merged and self.r > 0:
            self.linear.weight.data -= self.get_delta_weight()
            self.merged = False
        ################################################################

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for LoRALinear layer:
        h = W_0 x + (alpha / r) * B * A * x

        Args:
            x (Tensor): Input tensor of shape (..., in_features).

        Returns:
            Tensor: Output tensor of shape (..., out_features).
        """
        ################################################################
        # TODO: Implement LoRALinear forward pass
        if self.merged or self.r == 0:
            return self.linear(x)
        base_out = self.linear(x)
        lora_out = self.lora_B(self.lora_A(self.lora_dropout(x))) * self.scaling
        return base_out + lora_out
        ################################################################


def mark_only_lora_as_trainable(model: nn.Module, bias: str = "none"):
    """
    Freezes all base parameters in the model except LoRA adapter weights.

    Args:
        model (nn.Module): Model containing LoRALinear layers.
        bias  (str): 'none' (freeze all biases), 'all' (train all biases),
                     or 'lora_only' (train only biases in LoRALinear layers).
    """
    ################################################################
    # TODO: Implement mark_only_lora_as_trainable
    for name, param in model.named_parameters():
        if "lora_" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False

    if bias == "none":
        return
    elif bias == "all":
        for name, param in model.named_parameters():
            if "bias" in name:
                param.requires_grad = True
    elif bias == "lora_only":
        for m in model.modules():
            if isinstance(m, LoRALinear) and hasattr(m.linear, "bias") and m.linear.bias is not None:
                m.linear.bias.requires_grad = True
    else:
        raise ValueError(f"Unsupported bias option: {bias}")
    ################################################################


def apply_lora_to_model(
    model: nn.Module,
    target_module_names: List[str],
    r: int = 4,
    lora_alpha: float = 1.0,
    lora_dropout: float = 0.0,
) -> nn.Module:
    """
    Recursively replaces target nn.Linear layers matching target_module_names with LoRALinear.

    Args:
        model               (nn.Module): Target model (e.g. Transformer, GPT-2).
        target_module_names (List[str]): Names of linear submodules to wrap (e.g. ['c_attn', 'c_proj']).
        r                   (int): LoRA rank.
        lora_alpha          (float): LoRA scaling constant.
        lora_dropout        (float): Dropout probability.

    Returns:
        nn.Module: Model with specified linear layers wrapped in LoRALinear.
    """
    ################################################################
    # TODO: Implement apply_lora_to_model
    for name, child in model.named_children():
        if isinstance(child, nn.Linear) and any(target in name for target in target_module_names):
            setattr(
                model,
                name,
                LoRALinear(
                    base_layer=child,
                    r=r,
                    lora_alpha=lora_alpha,
                    lora_dropout=lora_dropout,
                ),
            )
        else:
            apply_lora_to_model(child, target_module_names, r, lora_alpha, lora_dropout)
    return model
    ################################################################
