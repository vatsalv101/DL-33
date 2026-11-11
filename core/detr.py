import math
from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment


def box_cxcywh_to_xyxy(x: torch.Tensor) -> torch.Tensor:
    """Converts bounding boxes from (center_x, center_y, width, height) to (x1, y1, x2, y2)."""
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h), (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=-1)


def box_xyxy_to_cxcywh(x: torch.Tensor) -> torch.Tensor:
    """Converts bounding boxes from (x1, y1, x2, y2) to (center_x, center_y, width, height)."""
    x0, y0, x1, y1 = x.unbind(-1)
    b = [(x0 + x1) / 2, (y0 + y1) / 2, (x1 - x0), (y1 - y0)]
    return torch.stack(b, dim=-1)


def box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Computes pairwise IoU between two sets of boxes in (x1, y1, x2, y2) format.
    Args:
        boxes1: Tensor of shape (N, 4)
        boxes2: Tensor of shape (M, 4)
    Returns:
        Tuple[Tensor, Tensor]: iou of shape (N, M) and union of shape (N, M)
    """
    area1 = (boxes1[:, 2] - boxes1[:, 0]).clamp(min=0) * (boxes1[:, 3] - boxes1[:, 1]).clamp(min=0)
    area2 = (boxes2[:, 2] - boxes2[:, 0]).clamp(min=0) * (boxes2[:, 3] - boxes2[:, 1]).clamp(min=0)

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N, M, 2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N, M, 2]

    wh = (rb - lt).clamp(min=0)  # [N, M, 2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N, M]

    union = area1[:, None] + area2 - inter
    iou = inter / (union + 1e-6)
    return iou, union


def generalized_box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """
    Computes Generalized IoU (GIoU) between two sets of boxes in (x1, y1, x2, y2) format.
    GIoU = IoU - (area(C) - union) / area(C)
    where C is the smallest convex enclosing box.
    """
    iou, union = box_iou(boxes1, boxes2)

    lt = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])

    wh = (rb - lt).clamp(min=0)
    area_c = wh[:, :, 0] * wh[:, :, 1]

    giou = iou - (area_c - union) / (area_c + 1e-6)
    return giou


class PositionEmbeddingSine(nn.Module):
    """
    Standard 2D Sinusoidal Positional Encoding for image feature maps.
    """

    def __init__(self, num_pos_feats: int = 64, temperature: int = 10000, normalize: bool = True):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        self.scale = 2 * math.pi

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (B, C, H, W)
        Returns:
            Tensor: Positional embeddings of shape (B, num_pos_feats * 2, H, W)
        """
        B, C, H, W = x.shape
        not_mask = torch.ones((B, H, W), device=x.device)
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)

        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t

        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)

        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)  # (B, 2*num_pos_feats, H, W)
        return pos


class MLP(nn.Module):
    """Simple Multi-layer Perceptron (used for bounding box regression)."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class HungarianMatcher(nn.Module):
    """
    Computes optimal one-to-one assignment between predictions and ground truth targets
    using the Hungarian algorithm (linear_sum_assignment).

    Args:
        cost_class: Relative weight of classification cost
        cost_bbox:  Relative weight of L1 bounding box cost
        cost_giou:  Relative weight of GIoU cost
    """

    def __init__(self, cost_class: float = 1.0, cost_bbox: float = 5.0, cost_giou: float = 2.0):
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou

    @torch.no_grad()
    def forward(
        self, outputs: Dict[str, torch.Tensor], targets: List[Dict[str, torch.Tensor]]
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Args:
            outputs: Dict containing:
                "pred_logits": (B, num_queries, num_classes)
                "pred_boxes":  (B, num_queries, 4) in (cx, cy, w, h)
            targets: List of dicts of length B, each containing:
                "labels": (N_gt,) class indices
                "boxes":  (N_gt, 4) in (cx, cy, w, h)
        Returns:
            List of size B, where each element is a tuple of (pred_indices, target_indices).
        """
        bs, num_queries = outputs["pred_logits"].shape[:2]

        out_prob = outputs["pred_logits"].softmax(-1)  # [B, num_queries, num_classes]
        out_bbox = outputs["pred_boxes"]  # [B, num_queries, 4]

        indices = []
        for i in range(bs):
            tgt_ids = targets[i]["labels"]
            tgt_bbox = targets[i]["boxes"]

            if len(tgt_ids) == 0:
                indices.append((torch.empty(0, dtype=torch.int64), torch.empty(0, dtype=torch.int64)))
                continue

            # Class cost: -prob for ground-truth class
            cost_class = -out_prob[i, :, tgt_ids]  # [num_queries, N_gt]

            # L1 box cost
            cost_bbox = torch.cdist(out_bbox[i], tgt_bbox, p=1)  # [num_queries, N_gt]

            # GIoU cost
            cost_giou = -generalized_box_iou(
                box_cxcywh_to_xyxy(out_bbox[i]), box_cxcywh_to_xyxy(tgt_bbox)
            )

            # Total cost matrix
            C = self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou
            C = C.cpu()

            src_idx, tgt_idx = linear_sum_assignment(C)
            indices.append((
                torch.as_tensor(src_idx, dtype=torch.int64),
                torch.as_tensor(tgt_idx, dtype=torch.int64),
            ))

        return indices


class SetCriterion(nn.Module):
    """
    Computes bipartite matching loss for DETR:
    1. CrossEntropy on labels (with background class weight eos_coef)
    2. L1 Loss on matched bounding boxes
    3. GIoU Loss on matched bounding boxes
    """

    def __init__(
        self,
        num_classes: int,
        matcher: HungarianMatcher,
        weight_dict: Dict[str, float],
        eos_coef: float = 0.1,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef

        # Class weights: background class (index num_classes) receives eos_coef
        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[-1] = self.eos_coef
        self.register_buffer("empty_weight", empty_weight)

    def forward(
        self, outputs: Dict[str, torch.Tensor], targets: List[Dict[str, torch.Tensor]]
    ) -> Dict[str, torch.Tensor]:
        indices = self.matcher(outputs, targets)

        # 1. Classification Loss
        src_logits = outputs["pred_logits"]
        idx = self._get_src_permutation_idx(indices)

        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(
            src_logits.shape[:2],
            self.num_classes,
            dtype=torch.int64,
            device=src_logits.device,
        )
        target_classes[idx] = target_classes_o

        loss_ce = F.cross_entropy(
            src_logits.transpose(1, 2), target_classes, self.empty_weight
        )

        # 2. Box Losses (L1 + GIoU)
        src_boxes = outputs["pred_boxes"][idx]
        target_boxes = torch.cat([t["boxes"][i] for t, (_, i) in zip(targets, indices)], dim=0)

        num_boxes = max(len(target_boxes), 1)
        if len(target_boxes) > 0:
            loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction="sum") / num_boxes
            loss_giou = (
                1.0
                - torch.diag(
                    generalized_box_iou(
                        box_cxcywh_to_xyxy(src_boxes), box_cxcywh_to_xyxy(target_boxes)
                    )
                )
            ).sum() / num_boxes
        else:
            loss_bbox = src_boxes.sum() * 0.0
            loss_giou = src_boxes.sum() * 0.0

        losses = {
            "loss_ce": loss_ce,
            "loss_bbox": loss_bbox,
            "loss_giou": loss_giou,
        }
        return losses

    def _get_src_permutation_idx(self, indices):
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx


class SimpleBackbone(nn.Module):
    """Compact convolutional backbone for feature extraction."""

    def __init__(self, in_channels: int = 3, out_channels: int = 128):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, out_channels, kernel_size=3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        return x


class DETR(nn.Module):
    """
    End-to-End Object Detection with Transformers (Carion et al., 2020).

    Args:
        num_classes: Number of foreground classes.
        num_queries: Number of learned object queries. Default: 10.
        hidden_dim:  Transformer hidden dimension. Default: 128.
        nheads:      Number of attention heads. Default: 4.
        num_encoder_layers: Number of Transformer encoder layers. Default: 2.
        num_decoder_layers: Number of Transformer decoder layers. Default: 2.
    """

    def __init__(
        self,
        num_classes: int,
        num_queries: int = 10,
        hidden_dim: int = 128,
        nheads: int = 4,
        num_encoder_layers: int = 2,
        num_decoder_layers: int = 2,
        backbone: Optional[nn.Module] = None,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.num_queries = num_queries
        self.hidden_dim = hidden_dim

        self.backbone = backbone if backbone is not None else SimpleBackbone(out_channels=hidden_dim)
        self.input_proj = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1)

        self.pos_encoder = PositionEmbeddingSine(num_pos_feats=hidden_dim // 2, normalize=True)

        self.transformer = nn.Transformer(
            d_model=hidden_dim,
            nhead=nheads,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,
            batch_first=True,
        )

        self.query_embed = nn.Embedding(num_queries, hidden_dim)

        # Output projection heads
        self.class_embed = nn.Linear(hidden_dim, num_classes + 1)
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, num_layers=3)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: Input images of shape (B, 3, H, W).
        Returns:
            Dict containing:
                "pred_logits": (B, num_queries, num_classes + 1)
                "pred_boxes":  (B, num_queries, 4) in (cx, cy, w, h) normalized in [0, 1].
        """
        B = x.shape[0]

        # 1. Feature extraction through CNN backbone
        features = self.backbone(x)  # (B, hidden_dim, H', W')
        src = self.input_proj(features)  # (B, hidden_dim, H', W')
        pos = self.pos_encoder(src)  # (B, hidden_dim, H', W')

        # Flatten spatial dimensions for Transformer: (B, H'*W', hidden_dim)
        src = src.flatten(2).transpose(1, 2)
        pos = pos.flatten(2).transpose(1, 2)

        # 2. Prepare Object Queries: (B, num_queries, hidden_dim)
        query_embed = self.query_embed.weight.unsqueeze(0).expand(B, -1, -1)
        tgt = torch.zeros_like(query_embed)

        # 3. Transformer Encoder-Decoder forward
        hs = self.transformer(
            src=src + pos,
            tgt=tgt + query_embed,
        )  # (B, num_queries, hidden_dim)

        # 4. Heads
        outputs_class = self.class_embed(hs)
        outputs_coord = self.bbox_embed(hs).sigmoid()  # normalized coordinates in [0, 1]

        return {"pred_logits": outputs_class, "pred_boxes": outputs_coord}
