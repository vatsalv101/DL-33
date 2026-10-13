import torch
import torch.nn as nn


class BatchNorm1d(nn.Module):
    """
    Applies Batch Normalization over a 2D or 3D input.

    Args:
        num_features     (int): Number of features/channels C.
        eps            (float): Small value added to denominator for numerical stability. Default: 1e-5.
        momentum       (float): Value used for the running_mean and running_var computation. Default: 0.1.
        affine          (bool): When True, has learnable affine parameters (gamma, beta). Default: True.
    """

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
        """
        Forward pass for BatchNorm1d.

        Args:
            x (Tensor): Input of shape (B, C) or (B, C, L).

        Returns:
            Tensor: Normalized tensor of the same shape as input.
        """
        ################################################################
        # TODO: Implement BatchNorm1d forward pass
        # 1. Determine dimensions over which to compute mean and variance:
        #    - If 2D (B, C): reduce over dim=0
        #    - If 3D (B, C, L): reduce over dim=(0, 2)
        dim = (0,) if x.dim() == 2 else (0, 2)

        # 2. In training mode (self.training):
        #    - Compute batch mean and unbiased/biased variance
        #    - Update self.running_mean and self.running_var via EMA
        #    - Normalize x using batch statistics
        if self.training:
            mean = x.mean(dim)
            var = x.var(dim, unbiased = False)
            n = x.numel() / self.num_features
            unbiased_var = var * (n / (n - 1)) if n > 1 else var

            with torch.no_grad():
                self.running_mean.lerp_(mean, self.momentum)
                self.running_var.lerp_(unbiased_var, self.momentum)

        # 3. In evaluation mode (not self.training):
        #    - Normalize x using self.running_mean and self.running_var
        else:
            mean = self.running_mean
            var = self.running_var

        shape = [1, self.num_features] + [1] * (x.dim() - 2)
        mean = mean.view(shape)
        var = var.view(shape)

        x_hat = (x-mean) / torch.sqrt(var+self.eps)
        # 4. If self.affine is True, scale by self.weight and shift by self.bias
        if self.affine:
            w = self.weight.view(shape)
            b = self.bias.view(shape)
            return x_hat * w + b
        return x_hat
        ################################################################


class BatchNorm2d(nn.Module):
    """
    Applies Batch Normalization over a 4D input (spatial Conv2d feature maps).

    Args:
        num_features     (int): Number of channels C.
        eps            (float): Small value added to denominator for numerical stability. Default: 1e-5.
        momentum       (float): Value used for the running_mean and running_var computation. Default: 0.1.
        affine          (bool): When True, has learnable affine parameters (gamma, beta). Default: True.
    """

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
        """
        Forward pass for BatchNorm2d.

        Args:
            x (Tensor): Input of shape (B, C, H, W).

        Returns:
            Tensor: Normalized tensor of shape (B, C, H, W).
        """
        ################################################################
        # TODO: Implement BatchNorm2d forward pass
        # 1. Reduce over spatial and batch dimensions: dim=(0, 2, 3)
        dims = (0,2,3)

        if self.training:
            mean = x.mean(dims)
            var = x.var(dims, unbiased=False)
            unbiased_var = x.var(dims, unbiased = True)

            with torch.no_grad():
                self.running_mean.lerp_(mean, self.momentum)
                self.running_var.lerp_(unbiased_var, self.momentum)

        else:
            mean = self.running_mean 
            var = self.running_var

        shape = (1, self.num_features, 1, 1)
        mean = mean.view(shape)
        var = var.view(shape)

        x_hat = (x - mean) / torch.sqrt(var + self.eps)

        if self.affine:
            w = self.weight.view(shape)
            b = self.bias.view(shape)
            return x_hat * w + b
        return x_hat
        ################################################################
