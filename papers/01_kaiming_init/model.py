import math
from typing import Optional, Tuple, Union
import torch
import torch.nn as nn


def calculate_gain(nonlinearity: str, param: Optional[float] = None) -> float:
    """
    Computes the recommended gain value for a given nonlinearity.

    Args:
        nonlinearity (str): Name of activation function ('linear', 'conv2d', 'sigmoid', 'tanh', 'relu', 'leaky_relu').
        param      (float): Optional parameter, e.g. negative slope for leaky_relu.

    Returns:
        float: Scaling gain value.
    """
    ################################################################
    # TODO
    if nonlinearity in ('sigmoid', 'linear', 'conv2d'):
        return 1.0
    elif nonlinearity == 'tanh':
        return 5.0/3.0
    elif nonlinearity == 'relu':
        return math.sqrt(2.0)
    elif nonlinearity == 'leaky_relu':
        a = 0.01 if param is None else float(param)
        return math.sqrt(2.0 / (1.0 + a**2))
    else:
        raise ValueError(f"Unsupported nonlinearity: {nonlinearity}")

    ################################################################


def _calculate_fan(tensor: torch.Tensor, mode: str = "fan_in") -> int:
    """
    Computes the fan-in or fan-out for a weight tensor.

    Args:
        tensor (Tensor): Weight tensor of shape (out_features, in_features) or (C_out, C_in, k_h, k_w).
        mode     (str): 'fan_in' or 'fan_out'.

    Returns:
        int: Number of input or output connections.
    """
    ################################################################
    # TODO
    d = tensor.dim()
    if d < 2:
        raise ValueError(f"Fan calculation requires at least 2 dimensions, got {d}")

    ncin = tensor.size(1)
    ncout = tensor.size(0)
    rfs = 1

    if d > 2:
        for s in tensor.shape[2:]:
            rfs *= s

    if mode == 'fan_in':
        return ncin * rfs
    elif mode == 'fan_out':
        return ncout * rfs
    else:
        raise ValueError(f"Invalid mode '{mode}'. Choose 'fan_in' or 'fan_out'.")

    ################################################################


def kaiming_uniform_(
    tensor: torch.Tensor,
    a: float = 0.0,
    mode: str = "fan_in",
    nonlinearity: str = "leaky_relu",
) -> torch.Tensor:
    """
    Fills the input tensor with values according to Kaiming uniform initialization.

    Args:
        tensor       (Tensor): An n-dimensional torch.Tensor.
        a             (float): Negative slope of the rectifier (0 for ReLU).
        mode            (str): 'fan_in' or 'fan_out'.
        nonlinearity    (str): Non-linear function name.

    Returns:
        Tensor: The initialized tensor.
    """
    ################################################################
    # TODO
    gain = calculate_gain(nonlinearity, a)
    fin = _calculate_fan(tensor, mode)
    std = gain / math.sqrt(fin)
    bound = math.sqrt(3) * std
    with torch.no_grad():
        return tensor.uniform_(-bound, bound) 
    ################################################################


def kaiming_normal_(
    tensor: torch.Tensor,
    a: float = 0.0,
    mode: str = "fan_in",
    nonlinearity: str = "leaky_relu",
) -> torch.Tensor:
    """
    Fills the input tensor with values according to Kaiming normal initialization.

    Args:
        tensor       (Tensor): An n-dimensional torch.Tensor.
        a             (float): Negative slope of the rectifier (0 for ReLU).
        mode            (str): 'fan_in' or 'fan_out'.
        nonlinearity    (str): Non-linear function name.

    Returns:
        Tensor: The initialized tensor.
    """
    ################################################################
    # TODO
    gain = calculate_gain(nonlinearity, a)
    fin = _calculate_fan(tensor, mode)
    std = gain / math.sqrt(fin)
    with torch.no_grad():
        return tensor.normal_(0, std) 
    ################################################################


def init_weights(
    module: nn.Module,
    a: float = 0.0,
    mode: str = "fan_in",
    nonlinearity: str = "relu",
    distribution: str = "normal",
) -> None:
    """
    Recursively initializes all Linear and Conv layers in a module using Kaiming init,
    and sets biases to zero if present.

    Args:
        module            (nn.Module): Network or module to initialize.
        a                     (float): Negative slope for leaky_relu.
        mode                    (str): 'fan_in' or 'fan_out'.
        nonlinearity            (str): Activation function name.
        distribution            (str): 'normal' or 'uniform'.
    """
    ################################################################
    # TODO
    for m in module.modules():
        if isinstance(m, (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d)):
            if distribution == "normal":
                kaiming_normal_(m.weight, a=a, mode=mode, nonlinearity=nonlinearity)
            else:
                kaiming_uniform_(m.weight, a=a, mode=mode, nonlinearity=nonlinearity)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        
    ################################################################


class DeepMLP(nn.Module):
    """30-layer MLP for stress-testing activation variance."""

    def __init__(
        self,
        in_dim: int = 64,
        hidden_dim: int = 64,
        out_dim: int = 1,
        num_layers: int = 30,
        activation: str = "relu",
    ):
        super().__init__()
        self.num_layers = num_layers
        layers = []
        for i in range(num_layers):
            d_in = in_dim if i == 0 else hidden_dim
            d_out = out_dim if i == num_layers - 1 else hidden_dim
            layers.append(nn.Linear(d_in, d_out))
            if i < num_layers - 1:
                layers.append(nn.ReLU() if activation == "relu" else nn.LeakyReLU(0.1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def forward_with_layer_stats(self, x: torch.Tensor) -> Tuple[torch.Tensor, list]:
        stats = []
        h = x
        for layer in self.net:
            h = layer(h)
            if isinstance(layer, nn.Linear):
                stats.append((f"Layer_{len(stats)+1}", h.std().item()))
        return h, stats
