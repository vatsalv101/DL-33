from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """
    Two consecutive 3x3 convolutions with BatchNorm and ReLU.
    (Conv3x3 -> BN -> ReLU -> Conv3x3 -> BN -> ReLU)

    Args:
        in_channels  (int): Number of input channels.
        out_channels (int): Number of output channels.
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for DoubleConv block."""
        return self.double_conv(x)


class Down(nn.Module):
    """
    Downscaling with 2x2 MaxPool followed by DoubleConv.

    Args:
        in_channels  (int): Number of input channels.
        out_channels (int): Number of output channels.
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for Down block."""
        ################################################################
        # TODO: Implement Down forward pass
        # Pass x through maxpool and double_conv
        return self.maxpool_conv(x)
        ################################################################


class Up(nn.Module):
    """
    Upscaling followed by DoubleConv, incorporating skip connection concatenation.

    Args:
        in_channels  (int): Total channels entering the DoubleConv (after concatenation).
        out_channels (int): Channels exiting the DoubleConv.
        bilinear    (bool): If True, use bilinear interpolation for upsampling. Default: True.
    """

    def __init__(self, in_channels: int, out_channels: int, bilinear: bool = True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels)
        else:
            self.up = nn.ConvTranspose2d(
                in_channels ,
                in_channels // 2,
                kernel_size=2,
                stride=2,
            )
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for Up block.

        Args:
            x1 (Tensor): Input tensor to be upsampled from previous decoder stage.
            x2 (Tensor): Skip connection tensor from corresponding encoder stage.

        Returns:
            Tensor: Refined tensor after upsampling, concatenation, and DoubleConv.
        """
        ################################################################
        # TODO: 
        x1 = self.up(x1)
        diffy = x2.shape[2] - x1.shape[2]
        diffx = x2.shape[3] - x1.shape[3]

        if diffy != 0 or diffx != 0:
            x1 = F.pad(x1, [diffx//2, diffx - diffx // 2, diffy // 2, diffy - diffy // 2])

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)
        ################################################################


class OutConv(nn.Module):
    """
    Final 1x1 convolution mapping feature maps to class logits.

    Args:
        in_channels  (int): Number of input channels.
        out_channels (int): Number of target classes / output channels.
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for OutConv."""
        return self.conv(x)


class UNet(nn.Module):
    """
    Full U-Net Architecture.

    Args:
        in_channels  (int): Number of input image channels (e.g. 3 for RGB). Default: 3.
        num_classes  (int): Number of segmentation target classes. Default: 1.
        bilinear    (bool): Whether to use bilinear interpolation for upsampling. Default: True.
    """

    def __init__(self, in_channels: int = 3, num_classes: int = 1, bilinear: bool = True):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.bilinear = bilinear

        factor = 2 if bilinear else 1

        # Encoder (Contracting Path)
        self.inc = DoubleConv(in_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024 // factor)

        # Decoder (Expanding Path)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)

        # Output Classifier
        self.outc = OutConv(64, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for U-Net.

        Args:
            x (Tensor): Input image tensor of shape (B, in_channels, H, W).

        Returns:
            Tensor: Output mask/logits of shape (B, num_classes, H, W).
        """
        ################################################################
        # TODO: Implement UNet forward pass
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        logits = self.outc(x)
        return logits
        ################################################################
