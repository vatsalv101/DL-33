"""Core verified layers for dl-rl-robotics-80.

Paper 01: Delving Deep into Rectifiers (Kaiming Initialization)
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
