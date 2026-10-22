"""Core verified model architectures for dl-rl-robotics-80.

Paper 05: Deep Residual Learning for Image Recognition (ResNet)
Paper 06: U-Net: Convolutional Networks for Biomedical Image Segmentation (U-Net)
Paper 07: MobileNetV2: Inverted Residuals and Linear Bottlenecks (MobileNetV2)
Paper 09: PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation (PointNet)
Implemented and verified by the student.
"""

from typing import List, Optional, Tuple, Type, Union
import torch
import torch.nn as nn
import torch.nn.functional as F


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
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = out + identity
        return self.relu(out)


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
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        return self.relu(identity + out)


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
        downsample = None
        if stride != 1 or self.in_planes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.in_planes, planes * block.expansion, stride=stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(
                self.in_planes,
                planes,
                stride=stride,
                downsample=downsample,
            )
        )
        self.in_planes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_planes, planes))

        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


def resnet18(num_classes: int = 1000) -> ResNet:
    """Constructs a ResNet-18 model."""
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes)


def resnet34(num_classes: int = 1000) -> ResNet:
    """Constructs a ResNet-34 model."""
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes=num_classes)


def resnet50(num_classes: int = 1000) -> ResNet:
    """Constructs a ResNet-50 model."""
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes=num_classes)


class DoubleConv(nn.Module):
    """
    Two consecutive 3x3 convolutions with BatchNorm and ReLU.
    (Conv3x3 -> BN -> ReLU -> Conv3x3 -> BN -> ReLU)
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
        return self.double_conv(x)


class Down(nn.Module):
    """
    Downscaling with 2x2 MaxPool followed by DoubleConv.
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.maxpool_conv(x)


class Up(nn.Module):
    """
    Upscaling followed by DoubleConv, incorporating skip connection concatenation.
    """

    def __init__(self, in_channels: int, out_channels: int, bilinear: bool = True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels)
        else:
            self.up = nn.ConvTranspose2d(
                in_channels // 2,
                in_channels // 2,
                kernel_size=2,
                stride=2,
            )
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        if diffY != 0 or diffX != 0:
            x1 = F.pad(
                x1,
                [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2],
            )

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    """
    Final 1x1 convolution mapping feature maps to class logits.
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class UNet(nn.Module):
    """
    Full U-Net Architecture.
    """

    def __init__(self, in_channels: int = 3, num_classes: int = 1, bilinear: bool = True):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.bilinear = bilinear

        factor = 2 if bilinear else 1

        self.inc = DoubleConv(in_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024 // factor)

        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)

        self.outc = OutConv(64, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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


class ConvBNReLU(nn.Sequential):
    """
    Standard Convolution + BatchNorm + ReLU6 sequential block.
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

        self.use_res_connect = self.stride == 1 and in_channels == out_channels

        hidden_dim = int(round(in_channels * expand_ratio))
        layers = []

        if expand_ratio != 1:
            layers.append(ConvBNReLU(in_channels, hidden_dim, kernel_size=1))

        layers.extend([
            ConvBNReLU(hidden_dim, hidden_dim, stride=stride, groups=hidden_dim),
            nn.Conv2d(hidden_dim, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
        ])

        self.conv = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_res_connect:
            return x + self.conv(x)
        return self.conv(x)


class MobileNetV2(nn.Module):
    """
    MobileNetV2 neural network architecture.
    """

    def __init__(self, num_classes: int = 1000, width_mult: float = 1.0):
        super().__init__()

        input_channel = int(32 * width_mult)
        last_channel = int(1280 * max(1.0, width_mult))

        self.cfgs = [
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]

        features = [ConvBNReLU(3, input_channel, stride=2)]

        for t, c, n, s in self.cfgs:
            output_channel = int(c * width_mult)
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(
                    InvertedResidual(
                        input_channel,
                        output_channel,
                        stride=stride,
                        expand_ratio=t,
                    )
                )
                input_channel = output_channel

        features.append(ConvBNReLU(input_channel, last_channel, kernel_size=1))
        self.features = nn.Sequential(*features)

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(last_channel, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def mobilenet_v2(num_classes: int = 1000, width_mult: float = 1.0) -> MobileNetV2:
    """Constructs a MobileNetV2 architecture."""
    return MobileNetV2(num_classes=num_classes, width_mult=width_mult)


class TNet(nn.Module):
    """
    Mini Spatial Transformer Network (T-Net) that predicts an affine transformation matrix.

    Args:
        k (int): Dimension of the points / features to transform (e.g. 3 for coordinates, 64 for features).
    """

    def __init__(self, k: int = 3):
        super().__init__()
        self.k = k
        self.conv1 = nn.Conv1d(k, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k * k)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)
        self.relu = nn.ReLU(inplace=True)

        nn.init.constant_(self.fc3.weight, 0)
        nn.init.constant_(self.fc3.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batchsize = x.size(0)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = self.relu(self.bn4(self.fc1(x)))
        x = self.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = torch.eye(self.k, device=x.device).view(1, self.k * self.k).repeat(batchsize, 1)
        x = x + iden
        x = x.view(-1, self.k, self.k)
        return x


class PointNetEncoder(nn.Module):
    """
    PointNet backbone encoder extracting global and per-point local features.

    Args:
        global_feat       (bool): If True, returns global feature vector. If False, returns concatenated local+global features.
        feature_transform (bool): If True, uses an intermediate 64x64 feature T-Net. Default: False.
    """

    def __init__(self, global_feat: bool = True, feature_transform: bool = False):
        super().__init__()
        self.stn = TNet(k=3)
        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.global_feat = global_feat
        self.feature_transform = feature_transform
        if self.feature_transform:
            self.fstn = TNet(k=64)
        self.relu = nn.ReLU(inplace=True)

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        n_pts = x.size(2)
        trans = self.stn(x)
        x = torch.bmm(trans, x)
        x = self.relu(self.bn1(self.conv1(x)))

        if self.feature_transform:
            trans_feat = self.fstn(x)
            x = torch.bmm(trans_feat, x)
        else:
            trans_feat = None

        pointfeat = x
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        if self.global_feat:
            return x, trans, trans_feat
        else:
            x = x.view(-1, 1024, 1).repeat(1, 1, n_pts)
            return torch.cat([x, pointfeat], 1), trans, trans_feat


class PointNetCls(nn.Module):
    """
    PointNet for 3D Shape Classification.

    Args:
        num_classes       (int): Number of target 3D object categories (e.g. 40 for ModelNet40).
        feature_transform (bool): Whether to use intermediate feature transform.
    """

    def __init__(self, num_classes: int = 40, feature_transform: bool = False):
        super().__init__()
        self.feat = PointNetEncoder(global_feat=True, feature_transform=feature_transform)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)
        self.dropout = nn.Dropout(p=0.3)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.relu = nn.ReLU(inplace=True)

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        x, trans, trans_feat = self.feat(x)
        x = self.relu(self.bn1(self.fc1(x)))
        x = self.relu(self.bn2(self.dropout(self.fc2(x))))
        x = self.fc3(x)
        return x, trans, trans_feat


class PointNetSeg(nn.Module):
    """
    PointNet for per-point 3D Part Segmentation.

    Args:
        num_classes       (int): Number of part segmentation classes (e.g. 16 for ShapeNet).
        feature_transform (bool): Whether to use intermediate feature transform.
    """

    def __init__(self, num_classes: int = 16, feature_transform: bool = False):
        super().__init__()
        self.num_classes = num_classes
        self.feat = PointNetEncoder(global_feat=False, feature_transform=feature_transform)
        self.conv1 = nn.Conv1d(1088, 512, 1)
        self.conv2 = nn.Conv1d(512, 256, 1)
        self.conv3 = nn.Conv1d(256, 128, 1)
        self.conv4 = nn.Conv1d(128, num_classes, 1)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(128)
        self.relu = nn.ReLU(inplace=True)

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        x, trans, trans_feat = self.feat(x)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.conv4(x)
        return x, trans, trans_feat


def feature_transform_regularizer(trans: torch.Tensor) -> torch.Tensor:
    """
    Orthogonality regularization loss: L_reg = ||I - A * A^T||_F^2 / (2 * B).

    Args:
        trans (Tensor): Transformation matrix of shape (B, K, K).

    Returns:
        Tensor: Scalar regularization loss.
    """
    d = trans.size()[1]
    batch_size = trans.size()[0]
    i = torch.eye(d, device=trans.device).unsqueeze(0).repeat(batch_size, 1, 1)
    loss = torch.mean(
        torch.norm(torch.bmm(trans, trans.transpose(2, 1)) - i, dim=(1, 2)) ** 2
    )
    return loss / (2 * d)



