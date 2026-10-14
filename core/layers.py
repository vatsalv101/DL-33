"""Core verified layers for dl-rl-robotics-80.

Paper 01: Delving Deep into Rectifiers (Kaiming Initialization)
Paper 03: Batch Normalization (BatchNorm1d & BatchNorm2d)
Implemented and verified by the student.
"""

import math
from typing import Optional, Union
import torch
import torch.nn as nn


def calculate_gain(nonlinearity: str, param: Optional[float] = None) -> float:
    if nonlinearity in ("sigmoid", "linear", "conv2d"):
        return 1.0
    elif nonlinearity == "tanh":
        return 5.0 / 3.0
    elif nonlinearity == "relu":
        return math.sqrt(2.0)
    elif nonlinearity == "leaky_relu":
        a = 0.01 if param is None else float(param)
        return math.sqrt(2.0 / (1.0 + a**2))
    else:
        raise ValueError(f"Unsupported nonlinearity: {nonlinearity}")


def _calculate_fan(tensor: torch.Tensor, mode: str = "fan_in") -> int:
    d = tensor.dim()
    if d < 2:
        raise ValueError(f"Fan calculation requires at least 2 dimensions, got {d}")

    ncin = tensor.size(1)
    ncout = tensor.size(0)
    rfs = 1

    if d > 2:
        for s in tensor.shape[2:]:
            rfs *= s

    if mode == "fan_in":
        return ncin * rfs
    elif mode == "fan_out":
        return ncout * rfs
    else:
        raise ValueError(f"Invalid mode '{mode}'. Choose 'fan_in' or 'fan_out'.")


def kaiming_uniform_(
    tensor: torch.Tensor,
    a: float = 0.0,
    mode: str = "fan_in",
    nonlinearity: str = "leaky_relu",
) -> torch.Tensor:
    gain = calculate_gain(nonlinearity, a)
    fin = _calculate_fan(tensor, mode)
    std = gain / math.sqrt(fin)
    bound = math.sqrt(3) * std
    with torch.no_grad():
        return tensor.uniform_(-bound, bound)


def kaiming_normal_(
    tensor: torch.Tensor,
    a: float = 0.0,
    mode: str = "fan_in",
    nonlinearity: str = "leaky_relu",
) -> torch.Tensor:
    gain = calculate_gain(nonlinearity, a)
    fin = _calculate_fan(tensor, mode)
    std = gain / math.sqrt(fin)
    with torch.no_grad():
        return tensor.normal_(0, std)


def init_weights(
    module: nn.Module,
    a: float = 0.0,
    mode: str = "fan_in",
    nonlinearity: str = "relu",
    distribution: str = "normal",
) -> None:
    for m in module.modules():
        if isinstance(m, (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d)):
            if distribution == "normal":
                kaiming_normal_(m.weight, a=a, mode=mode, nonlinearity=nonlinearity)
            else:
                kaiming_uniform_(m.weight, a=a, mode=mode, nonlinearity=nonlinearity)
            if m.bias is not None:
                nn.init.zeros_(m.bias)


class BatchNorm1d(nn.Module):
    """Applies Batch Normalization over a 2D or 3D input."""

    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        momentum: float = 0.1,
        affine: bool = True,
    ):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine

        if self.affine:
            self.weight = nn.Parameter(torch.ones(num_features))
            self.bias = nn.Parameter(torch.zeros(num_features))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

        self.register_buffer("running_mean", torch.zeros(num_features))
        self.register_buffer("running_var", torch.ones(num_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dim = (0,) if x.dim() == 2 else (0, 2)

        if self.training:
            mean = x.mean(dim)
            var = x.var(dim, unbiased=False)
            n = x.numel() / self.num_features
            unbiased_var = var * (n / (n - 1)) if n > 1 else var

            with torch.no_grad():
                self.running_mean.lerp_(mean, self.momentum)
                self.running_var.lerp_(unbiased_var, self.momentum)
        else:
            mean = self.running_mean
            var = self.running_var

        shape = [1, self.num_features] + [1] * (x.dim() - 2)
        mean = mean.view(shape)
        var = var.view(shape)

        x_hat = (x - mean) / torch.sqrt(var + self.eps)

        if self.affine:
            w = self.weight.view(shape)
            b = self.bias.view(shape)
            return x_hat * w + b
        return x_hat


class BatchNorm2d(nn.Module):
    """Applies Batch Normalization over a 4D input (spatial Conv2d feature maps)."""

    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        momentum: float = 0.1,
        affine: bool = True,
    ):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine

        if self.affine:
            self.weight = nn.Parameter(torch.ones(num_features))
            self.bias = nn.Parameter(torch.zeros(num_features))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

        self.register_buffer("running_mean", torch.zeros(num_features))
        self.register_buffer("running_var", torch.ones(num_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dims = (0, 2, 3)

        if self.training:
            mean = x.mean(dims)
            unbiased_var = x.var(dims, unbiased=True)

            with torch.no_grad():
                self.running_mean.lerp_(mean, self.momentum)
                self.running_var.lerp_(unbiased_var, self.momentum)
        else:
            mean = self.running_mean
            unbiased_var = self.running_var

        shape = (1, self.num_features, 1, 1)
        mean = mean.view(shape)
        var = unbiased_var.view(shape)

        x_hat = (x - mean) / torch.sqrt(var + self.eps)

        if self.affine:
            w = self.weight.view(shape)
            b = self.bias.view(shape)
            return x_hat * w + b
        return x_hat
