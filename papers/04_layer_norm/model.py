from typing import Tuple, Union
import torch
import torch.nn as nn


class LayerNorm(nn.Module):
    """
    Applies Layer Normalization over a mini-batch of inputs.

    Args:
        normalized_shape (int or tuple): Input shape from an expected input of size (*, normalized_shape).
        eps                     (float): A value added to the denominator for numerical stability. Default: 1e-5.
        elementwise_affine       (bool): A boolean value that when set to True, has learnable per-element affine parameters. Default: True.
    """

    def __init__(
        self,
        normalized_shape: Union[int, Tuple[int, ...]],
        eps: float = 1e-5,
        elementwise_affine: bool = True,
    ):
        super().__init__()
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = tuple(normalized_shape)
        self.eps = eps
        self.elementwise_affine = elementwise_affine

        if self.elementwise_affine:
            self.weight = nn.Parameter(torch.ones(self.normalized_shape))
            self.bias = nn.Parameter(torch.zeros(self.normalized_shape))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for LayerNorm.

        Args:
            x (Tensor): Input tensor of shape (*, normalized_shape).

        Returns:
            Tensor: Normalized tensor of the same shape as input.
        """
        ################################################################
        # TODO: Implement LayerNorm forward pass
        # 1. Identify reduction dimensions from normalized_shape
        dims = -1
        # 2. Compute mean and unbiased=False variance along those dimensions
        mean = x.mean(dim = dims, keepdim=True)
        var = x.var(dim = dims, unbiased = False, keepdim=True)

        # 3. Standardize x to zero mean and unit variance
        x_hat = (x - mean) / torch.sqrt(var + self.eps)
        # 4. Apply learnable affine parameters if elementwise_affine is True

        if self.elementwise_affine:
            return x_hat * self.weight + self.bias

        return x_hat
        ################################################################


class RMSNorm(nn.Module):
    """
    Applies Root Mean Square Layer Normalization (RMSNorm).

    Args:
        dim   (int): Dimension of the features to normalize.
        eps (float): Small value added to denominator for numerical stability. Default: 1e-6.
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for RMSNorm.

        Args:
            x (Tensor): Input tensor of shape (*, dim).

        Returns:
            Tensor: Scaled tensor of the same shape as input.
        """
        ################################################################
        # TODO: Implement RMSNorm forward pass
        # 1. Compute root mean square (RMS) along the feature dimension
        rms = torch.sqrt(1/self.dim*(torch.sum((x**2), dim=-1, keepdim=True)+self.eps))
        # 2. Normalize x by RMS and apply learnable weight
        return self.weight * x / rms
        ################################################################
