"""Paper 01: Delving Deep into Rectifiers (Kaiming Initialization)
Authors: Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun (2015)

=============================================================================
TODAY'S MISSION: Implement Variance-Preserving Initialization from Scratch!
=============================================================================
Follow the docstrings, mathematical formulas, and TODO blocks below.
When you finish writing your code, run:
    uv run python papers/01_kaiming_init/model.py
and:
    uv run python papers/01_kaiming_init/train_sanity.py
"""

import math
from typing import Optional, Tuple, Union
import torch
import torch.nn as nn


def calculate_gain(nonlinearity: str, param: Optional[float] = None) -> float:
    """Calculates the scaling gain factor to preserve variance through an activation.

    Mathematical intuition:
      - Linear / Identity / Sigmoid: Signal variance is uncompressed -> gain = 1.0
      - Tanh: Compresses variance slightly -> gain = 5/3 (~1.667)
      - ReLU: Sets all negative values (half the signal) to 0, halving variance:
              E[x^2] = 1/2 Var[y]  =>  gain = sqrt(2.0) (~1.414)
      - LeakyReLU(a): Negative values scaled by slope 'a':
              E[x^2] = (1 + a^2) / 2 Var[y]  =>  gain = sqrt(2.0 / (1 + a^2))

    Inputs:
      - nonlinearity (str): 'linear', 'conv2d', 'sigmoid', 'tanh', 'relu', 'leaky_relu'
      - param (float, optional): Negative slope 'a' for 'leaky_relu' (defaults to 0.01 if None)

    Output:
      - float: Exact scaling gain value.
    """
    # TODO: Implement gain calculation based on the formulas above
    raise NotImplementedError("TODO: Implement calculate_gain(nonlinearity, param)")


def _calculate_fan(tensor: torch.Tensor, mode: str = "fan_in") -> int:
    """Computes fan_in or fan_out for 2D (Linear) or 3D/4D/5D (Convolutional) tensors.

    Mathematical intuition:
      - fan_in:  Number of input connections contributing to each output neuron.
                 Preserves forward pass activation variance.
      - fan_out: Number of output connections receiving each input neuron.
                 Preserves backward pass gradient variance.

    Tensor shapes:
      - Linear weight: (out_features, in_features)
          fan_in  = in_features
          fan_out = out_features
      - Conv2d weight: (C_out, C_in, k_h, k_w)
          receptive_field = k_h * k_w
          fan_in  = C_in * receptive_field
          fan_out = C_out * receptive_field

    Inputs:
      - tensor (torch.Tensor): Weight tensor (must have at least 2 dimensions).
      - mode (str): 'fan_in' or 'fan_out'.

    Output:
      - int: Computed fan count.
    """
    # TODO: Implement _calculate_fan based on tensor dimensions and mode
    raise NotImplementedError("TODO: Implement _calculate_fan(tensor, mode)")


def kaiming_uniform_(
    tensor: torch.Tensor,
    a: float = 0.0,
    mode: str = "fan_in",
    nonlinearity: str = "leaky_relu",
) -> torch.Tensor:
    """Fills the input Tensor with values drawn from Uniform(-bound, bound).

    Mathematical derivation:
      For Uniform(-bound, bound), variance is: Var = (2 * bound)^2 / 12 = bound^2 / 3.
      To achieve target variance Var = (gain / sqrt(fan))^2:
          bound^2 / 3 = std^2  =>  bound = sqrt(3) * std
      where std = gain / sqrt(fan).

    Inputs:
      - tensor (torch.Tensor): The weight tensor to initialize in-place.
      - a (float): Negative slope for leaky_relu (0 for standard ReLU).
      - mode (str): 'fan_in' (default) or 'fan_out'.
      - nonlinearity (str): Activation name (e.g. 'relu', 'leaky_relu').

    Output:
      - torch.Tensor: The initialized tensor (in-place modification).
    """
    # TODO:
    # 1. Compute fan using _calculate_fan(tensor, mode=mode)
    # 2. Compute gain using calculate_gain(nonlinearity, param=a)
    # 3. Compute std = gain / sqrt(fan)
    # 4. Compute bound = sqrt(3.0) * std
    # 5. Inside 'with torch.no_grad():', fill tensor using tensor.uniform_(-bound, bound)
    raise NotImplementedError("TODO: Implement kaiming_uniform_")


def kaiming_normal_(
    tensor: torch.Tensor,
    a: float = 0.0,
    mode: str = "fan_in",
    nonlinearity: str = "leaky_relu",
) -> torch.Tensor:
    """Fills the input Tensor with values drawn from Normal(0, std^2).

    Mathematical derivation:
      std = gain / sqrt(fan)
      Sample W ~ N(0, std^2).

    Inputs:
      - tensor (torch.Tensor): The weight tensor to initialize in-place.
      - a (float): Negative slope for leaky_relu (0 for standard ReLU).
      - mode (str): 'fan_in' (default) or 'fan_out'.
      - nonlinearity (str): Activation name (e.g. 'relu', 'leaky_relu').

    Output:
      - torch.Tensor: The initialized tensor (in-place modification).
    """
    # TODO:
    # 1. Compute fan using _calculate_fan(tensor, mode=mode)
    # 2. Compute gain using calculate_gain(nonlinearity, param=a)
    # 3. Compute std = gain / sqrt(fan)
    # 4. Inside 'with torch.no_grad():', fill tensor using tensor.normal_(0.0, std)
    raise NotImplementedError("TODO: Implement kaiming_normal_")


def init_weights(
    module: nn.Module,
    a: float = 0.0,
    mode: str = "fan_in",
    nonlinearity: str = "relu",
    distribution: str = "normal",
) -> None:
    """Recursively initializes all Linear and Conv weights in a module.

    Inputs:
      - module (nn.Module): PyTorch module / network to initialize.
      - a (float): LeakyReLU slope (0 for standard ReLU).
      - mode (str): 'fan_in' or 'fan_out'.
      - nonlinearity (str): 'relu' or other activation.
      - distribution (str): 'normal' for kaiming_normal_, 'uniform' for kaiming_uniform_.

    Side-effect:
      - For each child layer that is nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d:
          - Initializes m.weight using kaiming_normal_ or kaiming_uniform_.
          - If m.bias is not None, zeros out m.bias using nn.init.zeros_(m.bias).
    """
    # TODO:
    # Iterate through module.modules() and initialize weights and biases
    raise NotImplementedError("TODO: Implement init_weights(module, ...)")


# =============================================================================
# Verification Architecture: 30-Layer Deep MLP
# =============================================================================
class DeepMLP(nn.Module):
    """A 30-layer deep network designed to stress-test activation variance."""

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
        """Tracks activation standard deviation layer-by-layer."""
        stats = []
        h = x
        for layer in self.net:
            h = layer(h)
            if isinstance(layer, nn.Linear):
                stats.append((f"Layer_{len(stats)+1}", h.std().item()))
        return h, stats


if __name__ == "__main__":
    print("=" * 60)
    print("Testing Your Kaiming Init Implementation on a 30-Layer Network")
    print("=" * 60)
    torch.manual_seed(42)
    x = torch.randn(64, 64)

    # 1. Uninitialized network (standard PyTorch default init)
    model_default = DeepMLP(in_dim=64, hidden_dim=64, num_layers=30)
    _, stats_default = model_default.forward_with_layer_stats(x)

    # 2. Your Kaiming-initialized network
    model_kaiming = DeepMLP(in_dim=64, hidden_dim=64, num_layers=30)
    try:
        init_weights(model_kaiming, nonlinearity="relu", distribution="normal")
        _, stats_kaiming = model_kaiming.forward_with_layer_stats(x)

        print("\nActivation Std Dev Comparison:")
        print(f"  Layer  1 std -> Default: {stats_default[0][1]:.4f}  | Your Kaiming: {stats_kaiming[0][1]:.4f}")
        print(f"  Layer 15 std -> Default: {stats_default[14][1]:.6f}  | Your Kaiming: {stats_kaiming[14][1]:.4f}")
        print(f"  Layer 30 std -> Default: {stats_default[29][1]:.8f}  | Your Kaiming: {stats_kaiming[29][1]:.4f}")

        assert stats_kaiming[-1][1] > 0.05, "Signal variance collapsed to 0! Check your math."
        print("\n[SUCCESS] Your Kaiming Init implementation successfully kept the signal alive across all 30 layers!")
    except NotImplementedError as e:
        print(f"\n[PENDING] {e}")
        print("Open 'papers/01_kaiming_init/model.py' and implement the functions!")
