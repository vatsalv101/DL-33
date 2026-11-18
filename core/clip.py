from typing import Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from core.vit import ViT


class VisionBackbone(nn.Module):
    """
    Compact Vision Transformer backbone for image feature extraction.
    """

    def __init__(
        self,
        img_size: int = 16,
        patch_size: int = 4,
        in_chans: int = 3,
        embed_dim: int = 64,
        depth: int = 2,
        num_heads: int = 2,
    ):
        super().__init__()
        self.vit = ViT(
            img_size=img_size,
            patch_size=patch_size,
            in_channels=in_chans,
            num_classes=embed_dim,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=2.0,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.vit.forward_features(x)[:, 0]


class TextTransformerEncoder(nn.Module):
    """
    Causal / Bidirectional Transformer for text token sequence encoding.
    """

    def __init__(
        self,
        vocab_size: int = 1000,
        max_seq_len: int = 32,
        text_dim: int = 64,
        depth: int = 2,
        num_heads: int = 2,
    ):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, text_dim)
        self.position_embedding = nn.Parameter(torch.zeros(1, max_seq_len, text_dim))
        nn.init.normal_(self.position_embedding, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=text_dim,
            nhead=num_heads,
            dim_feedforward=text_dim * 2,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.ln_final = nn.LayerNorm(text_dim)

    def forward(self, text: torch.Tensor) -> torch.Tensor:
        B, L = text.shape
        x = self.token_embedding(text) + self.position_embedding[:, :L, :]
        x = self.transformer(x)
        x = self.ln_final(x)
        return x.mean(dim=1)


class CLIP(nn.Module):
    """
    Contrastive Language-Image Pre-Training (Radford et al., 2021).
    Jointly trains an image encoder and text encoder using symmetric cross-modal InfoNCE loss.
    """

    def __init__(
        self,
        embed_dim: int = 64,
        image_dim: int = 64,
        text_dim: int = 64,
        vocab_size: int = 1000,
        max_seq_len: int = 32,
        image_encoder: Optional[nn.Module] = None,
        text_encoder: Optional[nn.Module] = None,
        init_temperature: float = 0.07,
    ):
        super().__init__()
        self.embed_dim = embed_dim

        self.visual = image_encoder if image_encoder is not None else VisionBackbone(embed_dim=image_dim)
        self.text = (
            text_encoder
            if text_encoder is not None
            else TextTransformerEncoder(
                vocab_size=vocab_size, max_seq_len=max_seq_len, text_dim=text_dim
            )
        )

        self.visual_projection = nn.Linear(image_dim, embed_dim, bias=False)
        self.text_projection = nn.Linear(text_dim, embed_dim, bias=False)

        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1.0 / init_temperature))

    def encode_image(self, image: torch.Tensor) -> torch.Tensor:
        h = self.visual(image)
        e = self.visual_projection(h)
        return F.normalize(e, dim=-1)

    def encode_text(self, text: torch.Tensor) -> torch.Tensor:
        h = self.text(text)
        e = self.text_projection(h)
        return F.normalize(e, dim=-1)

    def forward(
        self, image: torch.Tensor, text: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        image_features = self.encode_image(image)
        text_features = self.encode_text(text)

        logit_scale = self.logit_scale.exp().clamp(max=100.0)
        logits_per_image = logit_scale * torch.matmul(image_features, text_features.T)
        logits_per_text = logits_per_image.T

        labels = torch.arange(image.shape[0], device=image.device)

        loss_i = F.cross_entropy(logits_per_image, labels)
        loss_t = F.cross_entropy(logits_per_text, labels)
        total_loss = (loss_i + loss_t) / 2.0

        return total_loss, logits_per_image, logits_per_text
