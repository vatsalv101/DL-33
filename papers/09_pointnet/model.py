from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class TNet(nn.Module):
    """
    Mini Spatial Transformer Network (T-Net) that predicts an affine transformation matrix.

    Args:
        k (int): Dimension of the points / features to transform (e.g. 3 for coordinates, 64 for features).
    """

    def __init__(self, k: int = 3):
        super().__init__()
        self.k = k

        ################################################################
        # TODO: Implement TNet __init__
        self.conv = nn.Sequential(
            nn.Conv1d(k, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 1024, 1),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
        )

        self.fc = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
        )
        self.last = nn.Linear(256, k*k)

        nn.init.constant_(self.last.weight, 0)
        nn.init.constant_(self.last.bias, 0)
        ################################################################

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for T-Net.

        Args:
            x (Tensor): Input tensor of shape (B, k, N).

        Returns:
            Tensor: Predicted transformation matrix of shape (B, k, k).
        """
        ################################################################
        # TODO: Implement TNet forward pass
        # 1. Pass x through conv layers
        batchsize = x.shape[0]
        x = self.conv(x)
        x = torch.amax(x, -1)
        x = self.fc(x)
        x = self.last(x)
        iden = torch.eye(self.k, device = x.device).view(1, self.k * self.k).repeat(batchsize,1)
        x = x + iden
        x = x.view(-1, self.k, self.k)

        return x
        ################################################################


class PointNetEncoder(nn.Module):
    """
    PointNet backbone encoder extracting global and per-point local features.

    Args:
        global_feat       (bool): If True, returns global feature vector. If False, returns concatenated local+global features.
        feature_transform (bool): If True, uses an intermediate 64x64 feature T-Net. Default: False.
    """

    def __init__(self, global_feat: bool = True, feature_transform: bool = False):
        super().__init__()
        self.global_feat = global_feat
        self.feature_transform = feature_transform

        ################################################################
        # TODO: Implement PointNetEncoder __init__
        self.stn = TNet(3)
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
        ################################################################

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Forward pass for PointNetEncoder.

        Args:
            x (Tensor): Input point coordinates of shape (B, 3, N).

        Returns:
            Tuple[Tensor, Optional[Tensor], Optional[Tensor]]:
                - features: Global feature (B, 1024) or concatenated local+global (B, 1088, N).
                - trans: Input coordinate transformation matrix (B, 3, 3).
                - trans_feat: Optional feature transformation matrix (B, 64, 64) or None.
        """
        ################################################################
        # TODO: Implement PointNetEncoder forward pass
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
        x = torch.amax(x, 2)

        if self.global_feat:
            return x, trans, trans_feat
        else:
            x = x.view(-1, 1024, 1).repeat(1, 1, n_pts)
            return torch.cat([x, pointfeat], 1), trans, trans_feat
        ################################################################


class PointNetCls(nn.Module):
    """
    PointNet for 3D Shape Classification.

    Args:
        num_classes       (int): Number of target 3D object categories (e.g. 40 for ModelNet40).
        feature_transform (bool): Whether to use intermediate feature transform.
    """

    def __init__(self, num_classes: int = 40, feature_transform: bool = False):
        super().__init__()
        self.feature_transform = feature_transform

        ################################################################
        # TODO: Implement PointNetCls __init__
        # 1. Encoder: PointNetEncoder(global_feat=True, feature_transform=feature_transform)
        self.encoder = PointNetEncoder(global_feat = True, feature_transform=feature_transform)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)
        self.dropout = nn.Dropout(.30)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.relu = nn.ReLU(inplace = True)
        ################################################################

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Forward pass for 3D classification.

        Args:
            x (Tensor): Input point cloud of shape (B, 3, N).

        Returns:
            Tuple[Tensor, Optional[Tensor], Optional[Tensor]]:
                - Logits of shape (B, num_classes)
                - Input transformation matrix (B, 3, 3)
                - Feature transformation matrix (B, 64, 64) or None
        """
        ################################################################
        # TODO: Implement PointNetCls forward pass
        # 1. Pass x through encoder
        x, trans, trans_feat = self.encoder(x)
        x = self.relu(self.bn1(self.fc1(x)))
        x = self.relu(self.bn2(self.dropout(self.fc2(x))))
        x = self.fc3(x)
        return x, trans, trans_feat
        ################################################################


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
        self.feature_transform = feature_transform

        ################################################################
        # TODO: Implement PointNetSeg __init__
        self.num_classes = num_classes
        self.feat = PointNetEncoder(global_feat=False, feature_transform = feature_transform)
        self.conv1 = nn.Conv1d(1088, 512, 1)
        self.conv2 = nn.Conv1d(512, 256, 1)
        self.conv3 = nn.Conv1d(256, 128, 1)
        self.conv4 = nn.Conv1d(128, num_classes, 1)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(128)
        self.relu = nn.ReLU(inplace=True)
        ################################################################

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Forward pass for 3D part segmentation.

        Args:
            x (Tensor): Input point cloud of shape (B, 3, N).

        Returns:
            Tuple[Tensor, Optional[Tensor], Optional[Tensor]]:
                - Per-point class logits of shape (B, num_classes, N)
                - Input transformation matrix (B, 3, 3)
                - Feature transformation matrix (B, 64, 64) or None
        """
        ################################################################
        # TODO: Implement PointNetSeg forward pass
        x, trans, trans_feat = self.feat(x)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.conv4(x)
        return x, trans, trans_feat
        ################################################################


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
