from typing import List, Optional
import torch
import torch.nn as nn


class ConvBNReLU(nn.Sequential):
    """
    Standard Convolution + BatchNorm + ReLU6 sequential block.

    Args:
        in_planes   (int): Number of input channels.
        out_planes  (int): Number of output channels.
        kernel_size (int): Size of the convolving kernel. Default: 3.
        stride      (int): Stride of the convolution. Default: 1.
        groups      (int): Number of blocked connections from input to output channels. Default: 1.
    """

    def __init__(
        self,
        in_planes: int,
        out_planes: int,
        kernel_size: int = 3,
        stride: int = 1,
        groups: int = 1,
    ):
        padding = (kernel_size - 1) // 2
        super().__init__(
            nn.Conv2d(
                in_planes,
                out_planes,
                kernel_size,
                stride,
                padding,
                groups=groups,
                bias=False,
            ),
            nn.BatchNorm2d(out_planes),
            nn.ReLU6(inplace=True),
        )


class InvertedResidual(nn.Module):
    """
    Inverted Residual Block with Linear Bottleneck.
    (1x1 Expansion -> 3x3 Depthwise Conv -> 1x1 Linear Pointwise Projection).

    Args:
        in_channels  (int): Number of input channels.
        out_channels (int): Number of output channels.
        stride       (int): Stride for the depthwise convolution (1 or 2).
        expand_ratio (int): Channel expansion factor for the internal bottleneck.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int,
        expand_ratio: int,
    ):
        super().__init__()
        self.stride = stride
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.expand_ratio = expand_ratio

        ################################################################
        # TODO: Implement InvertedResidual __init__

        self.use_res_connect = self.stride == 1 and self.in_channels == self.out_channels

        hidden_dim = int(round(self.in_channels * self.expand_ratio))
        layers = []

        if expand_ratio != 1:
            layers.append(ConvBNReLU(self.in_channels, hidden_dim, 1))

        layers.extend([
            ConvBNReLU(hidden_dim, hidden_dim, 3, stride, hidden_dim),
            nn.Conv2d(hidden_dim, self.out_channels, 1, 1)
        ])

        self.conv = nn.Sequential(*layers)
        ################################################################

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for InvertedResidual block.

        Args:
            x (Tensor): Input tensor of shape (B, in_channels, H, W).

        Returns:
            Tensor: Output tensor of shape (B, out_channels, H', W').
        """
        ################################################################
        # TODO: Implement InvertedResidual forward pass
        if self.use_res_connect:
            return x + self.conv(x)
        return self.conv(x)
        ################################################################


class MobileNetV2(nn.Module):
    """
    MobileNetV2 neural network architecture.

    Args:
        num_classes (int): Number of classification target classes. Default: 1000.
        width_mult (float): Width multiplier to scale channels globally. Default: 1.0.
    """

    def __init__(self, num_classes: int = 1000, width_mult: float = 1.0):
        super().__init__()

        # Inverted residual setting table: [t, c, n, s]
        # t: expansion factor
        # c: output channels
        # n: number of repeated blocks
        # s: stride of first block
        self.cfgs = [
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]

        ################################################################
        # TODO: Implement MobileNetV2 __init__
        input_channel = int(32 * width_mult)
        last_channel = int(1280 * max(1.0, width_mult))
        layers = [ConvBNReLU(3, input_channel, stride = 2)]
        for t, c, n, s in self.cfgs:
            output_channel = int(c * width_mult)
            for i in range(n):
                stride = s if i == 0 else 1
                layers.append(
                    InvertedResidual(
                        input_channel,
                        output_channel,
                        stride,
                        t,
                    )
                )
                input_channel = output_channel
        layers.append(ConvBNReLU(input_channel, last_channel, 1))
        self.feature = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(last_channel, num_classes)
        )
        ################################################################

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for MobileNetV2.

        Args:
            x (Tensor): Input image tensor of shape (B, 3, H, W).

        Returns:
            Tensor: Logits of shape (B, num_classes).
        """
        ################################################################
        # TODO: Implement MobileNetV2 forward pass
        x = self.feature(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)
        ################################################################


def mobilenet_v2(num_classes: int = 1000, width_mult: float = 1.0) -> MobileNetV2:
    """Constructs a MobileNetV2 architecture."""
    return MobileNetV2(num_classes=num_classes, width_mult=width_mult)
