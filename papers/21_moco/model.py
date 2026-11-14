from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class MoCoBackbone(nn.Module):
    """
    Feature extractor with an optional 2-layer MLP projection head (MoCo v2).
    """

    def __init__(self, in_channels: int = 3, feat_dim: int = 128, proj_dim: int = 64, mlp: bool = True):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.layer1 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, feat_dim, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(feat_dim),
            nn.ReLU(inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        if mlp:
            self.head = nn.Sequential(
                nn.Linear(feat_dim, feat_dim),
                nn.ReLU(inplace=True),
                nn.Linear(feat_dim, proj_dim),
            )
        else:
            self.head = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO: Forward x through conv1, layer1, layer2, pool, and head
        x = F.relu(self.bn1(self.conv1(x)), inplace=True)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.pool(x)
        feat = torch.flatten(x, 1)
        out = self.head(feat)
        return out

class MoCo(nn.Module):
    """
    Momentum Contrast (MoCo / MoCo v2) for Unsupervised Visual Representation Learning.
    Decouples dictionary size from minibatch size using a dynamic queue and momentum encoder.
    """

    def __init__(
        self,
        base_encoder: Optional[nn.Module] = None,
        dim: int = 64,
        K: int = 256,
        m: float = 0.999,
        T: float = 0.07,
    ):
        super().__init__()
        self.K = K
        self.m = m
        self.T = T
        self.dim = dim

        # Create query encoder and key encoder
        self.encoder_q = base_encoder if base_encoder is not None else MoCoBackbone(proj_dim=dim)
        # Instantiate identical architecture for key encoder
        self.encoder_k = (
            type(self.encoder_q)(proj_dim=dim)
            if base_encoder is None
            else type(base_encoder)()
        )

        # Initialize key encoder with query encoder parameters and freeze gradients
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False

        # Create the negative queue buffer: shape (dim, K)
        self.register_buffer("queue", torch.randn(dim, K))
        self.queue = F.normalize(self.queue, dim=0)

        # Create queue pointer buffer
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of key encoder parameters:
        theta_k = m * theta_k + (1 - m) * theta_q
        """
        # TODO: Update param_k.data = param_k.data * self.m + param_q.data * (1.0 - self.m)
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1.0 - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys: torch.Tensor):
        """
        Enqueues current batch of keys and dequeues the oldest keys (FIFO).
        """
        # TODO:
        batch_size = keys.shape[0]
        ptr = int(self.queue_ptr[0])

        assert self.K % batch_size == 0 or self.K >= batch_size, (
            f"Queue size {self.K} should be >= batch size {batch_size}"
        )

        if ptr + batch_size <= self.K:
            self.queue[:, ptr : ptr + batch_size] = keys.T
        else:
            overflow = (ptr + batch_size) - self.K
            self.queue[:, ptr : self.K] = keys.T[:, : self.K - ptr]
            self.queue[:, :overflow] = keys.T[:, self.K - ptr :]

        # Move pointer forward
        ptr = (ptr + batch_size) % self.K
        self.queue_ptr[0] = ptr

    def forward(
        self, im_q: torch.Tensor, im_k: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Input:
            im_q: a batch of query images (N, 3, H, W)
            im_k: a batch of key images (N, 3, H, W)
        Output:
            logits: (N, 1 + K) cosine similarities with temperature scaling
            labels: (N,) ground-truth targets (all zeros)
        """
        # TODO:
        q = self.encoder_q(im_q)
        q = F.normalize(q, dim=1)

        with torch.no_grad():
            self._momentum_update_key_encoder()
            k = self.encoder_k(im_k)
            k = F.normalize(k, dim=1)

        l_pos = (q * k).sum(dim=-1, keepdim=True)
        l_neg = torch.matmul(q, self.queue.clone().detach())
        logits = torch.cat([l_pos, l_neg], dim=1) / self.T
        labels = torch.zeros(logits.shape[0], dtype=torch.long, device=logits.device)
        self._dequeue_and_enqueue(k)

        return logits, labels

