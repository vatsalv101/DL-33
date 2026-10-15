from typing import List, Optional, Type, Union
import torch
import torch.nn as nn


def conv3x3(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding."""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False,
    )


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution."""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=1,
        stride=stride,
        bias=False,
    )


class BasicBlock(nn.Module):
    """
    Basic Residual Block for ResNet-18 and ResNet-34.

    Args:
        in_planes  (int): Number of input channels.
        planes     (int): Number of intermediate/output channels.
        stride     (int): Stride for the first convolutional layer. Default: 1.
        downsample (nn.Module, optional): Downsample projection module for the shortcut.
    """

    expansion: int = 1

    def __init__(
        self,
        in_planes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
    ):
        super().__init__()
        self.conv1 = conv3x3(in_planes, planes, stride=stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, stride=1)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for BasicBlock: H(x) = ReLU(F(x) + shortcut(x)).

        Args:
            x (Tensor): Input tensor of shape (B, in_planes, H, W).

        Returns:
            Tensor: Output tensor of shape (B, planes, H', W').
        """
        ################################################################
        # TODO: Implement BasicBlock forward pass
        # 1. Save identity shortcut (apply downsample if provided)
        identity = x
        # 2. Pass x through conv1 -> bn1 -> relu -> conv2 -> bn2
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        # 3. Add shortcut to residual output
        if self.downsample is not None:
            identity = self.downsample(x)

        out = out + identity
        # 4. Apply final ReLU activation
        return self.relu(out)

        ################################################################


class Bottleneck(nn.Module):
    """
    Bottleneck Block for ResNet-50, ResNet-101, and ResNet-152.

    Args:
        in_planes  (int): Number of input channels.
        planes     (int): Base channel dimension (output is planes * expansion).
        stride     (int): Stride for the 3x3 conv layer. Default: 1.
        downsample (nn.Module, optional): Downsample projection module for the shortcut.
    """

    expansion: int = 4

    def __init__(
        self,
        in_planes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
    ):
        super().__init__()
        self.conv1 = conv1x1(in_planes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes, stride=stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for Bottleneck block: 1x1 -> 3x3 -> 1x1 with residual add.

        Args:
            x (Tensor): Input tensor of shape (B, in_planes, H, W).

        Returns:
            Tensor: Output tensor of shape (B, planes * 4, H', W').
        """
        ################################################################
        # TODO: Implement Bottleneck forward pass
        # 1. Save identity shortcut (apply downsample if provided)
        identity = x

        # 2. Pass x through conv1 -> bn1 -> relu -> conv2 -> bn2 -> relu -> conv3 -> bn3
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)

        # 3. Add shortcut to residual output
        if self.downsample is not None:
            identity = self.downsample(x)
        # 4. Apply final ReLU activation
        return self.relu(identity + out)
        ################################################################


class ResNet(nn.Module):
    """
    Deep Residual Network architecture.

    Args:
        block       (Type[BasicBlock or Bottleneck]): Residual block type.
        layers      (List[int]): Number of blocks in each of the 4 stages.
        num_classes (int): Number of classification target classes. Default: 1000.
    """

    def __init__(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        layers: List[int],
        num_classes: int = 1000,
    ):
        super().__init__()
        self.in_planes = 64

        # Stem: 7x7 conv, BN, ReLU, MaxPool
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # 4 Residual Stages
        self.layer1 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        # Global Average Pooling and Classifier
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        planes: int,
        blocks: int,
        stride: int = 1,
    ) -> nn.Sequential:
        """
        Constructs one residual stage containing multiple residual blocks.

        Args:
            block  (Type): Block class (BasicBlock or Bottleneck).
            planes (int): Channel capacity for this stage.
            blocks (int): Number of residual blocks in this stage.
            stride (int): Stride for the first block. Default: 1.

        Returns:
            nn.Sequential: Sequential container with all blocks for this stage.
        """
        ################################################################
        # TODO: Implement _make_layer
        # 1. Determine if a downsample shortcut is required for the first block:
        #    downsample needed if stride != 1 or self.in_planes != planes * block.expansion
        downsample = None
        if stride != 1 or self.in_planes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.in_planes, planes * block.expansion, stride = stride),
                nn.BatchNorm2d(planes * block.expansion)
            )
        # 2. Append the first block with the specified stride and downsample
        layers = []
        layers.append(block(self.in_planes, planes, stride = stride, downsample = downsample))

        # 3. Update self.in_planes = planes * block.expansion
        self.in_planes = planes * block.expansion
    
        # 4. Append remaining (blocks - 1) blocks with stride=1 and downsample=None
        for _ in range(1, blocks):
            layers.append(block(self.in_planes, planes))
        # 5. Return nn.Sequential(*layers)
        return nn.Sequential(*layers)
        raise NotImplementedError
        ################################################################

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for full ResNet network.

        Args:
            x (Tensor): Input tensor of shape (B, 3, H, W).

        Returns:
            Tensor: Logits of shape (B, num_classes).
        """
        ################################################################
        # TODO: Implement ResNet forward pass
        # 1. Pass input through stem: conv1 -> bn1 -> relu -> maxpool
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        # 2. Pass through the 4 stages: layer1 -> layer2 -> layer3 -> layer4
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

        raise NotImplementedError
        ################################################################


def resnet18(num_classes: int = 1000) -> ResNet:
    """Constructs a ResNet-18 model."""
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes)


def resnet34(num_classes: int = 1000) -> ResNet:
    """Constructs a ResNet-34 model."""
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes=num_classes)


def resnet50(num_classes: int = 1000) -> ResNet:
    """Constructs a ResNet-50 model."""
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes=num_classes)
