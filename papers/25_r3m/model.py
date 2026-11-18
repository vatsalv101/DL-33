from typing import Optional, Tuple, Dict
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from core.simclr import SmallResNetBackbone


class R3MTextEncoder(nn.Module):
    """
    Language encoder mapping text narrations to representation vectors.
    """

    def __init__(
        self,
        vocab_size: int = 1000,
        max_seq_len: int = 32,
        text_dim: int = 64,
        embed_dim: int = 64,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, text_dim)
        self.pos_embedding = nn.Parameter(torch.zeros(1, max_seq_len, text_dim))
        nn.init.normal_(self.pos_embedding, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=text_dim,
            nhead=2,
            dim_feedforward=text_dim * 2,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.proj = nn.Linear(text_dim, embed_dim, bias=False)

    def forward(self, text: torch.Tensor) -> torch.Tensor:
        # TODO:
        B, L = text.shape
        x = self.embedding(text) + self.pos_embedding[:, :L, :]
        x = self.transformer(x)
        feat = x.mean(dim=1)
        return self.proj(feat)

class R3M(nn.Module):
    """
    R3M: A Universal Visual Representation for Robot Manipulation (Nair et al., 2022).
    Combines:
    1. Time-Contrastive Learning (TCN)
    2. Visual-Language Alignment (CLIP)
    3. L1 Sparsity Regularization
    """

    def __init__(
        self,
        embed_dim: int = 64,
        feat_dim: int = 128,
        vocab_size: int = 1000,
        max_seq_len: int = 32,
        image_encoder: Optional[nn.Module] = None,
        text_encoder: Optional[nn.Module] = None,
        temperature: float = 0.1,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.temperature = temperature

        self.visual = (
            image_encoder
            if image_encoder is not None
            else SmallResNetBackbone(feat_dim=feat_dim)
        )
        self.visual_proj = nn.Linear(feat_dim, embed_dim, bias=False)

        self.text = (
            text_encoder
            if text_encoder is not None
            else R3MTextEncoder(
                vocab_size=vocab_size,
                max_seq_len=max_seq_len,
                text_dim=64,
                embed_dim=embed_dim,
            )
        )

    def encode_image(self, image: torch.Tensor) -> torch.Tensor:
        """Extracts L2-normalized visual representations."""
        # TODO: Forward through self.visual, self.visual_proj, and F.normalize along dim=-1
        feat = self.visual(image)
        emb = self.visual_proj(feat)
        return F.normalize(emb, dim=-1)

    def encode_text(self, text: torch.Tensor) -> torch.Tensor:
        """Extracts L2-normalized text representations."""
        # TODO: Forward through self.text and F.normalize along dim=-1
        emb = self.text(text)
        return F.normalize(emb, dim=-1)
    
    def compute_tcn_loss(
        self, z_anchor: torch.Tensor, z_pos: torch.Tensor
    ) -> torch.Tensor:
        """
        Time-contrastive InfoNCE loss across video frames.
        z_anchor and z_pos are temporally adjacent frames.
        """
        # TODO:
        sim = torch.matmul(z_anchor, z_pos.T) / self.temperature
        labels = torch.arange(z_anchor.shape[0], device=z_anchor.device)
        return F.cross_entropy(sim, labels)

    def compute_lang_loss(
        self, z_visual: torch.Tensor, z_text: torch.Tensor
    ) -> torch.Tensor:
        """
        Symmetric cross-modal visual-language InfoNCE loss.
        """
        # TODO:
        sim_i2t = torch.matmul(z_visual, z_text.T) / self.temperature
        sim_t2i = sim_i2t.T
        labels = torch.arange(z_visual.shape[0], device=z_visual.device)
        loss_i2t = F.cross_entropy(sim_i2t, labels)
        loss_t2i = F.cross_entropy(sim_t2i, labels)
        return 0.5 * (loss_i2t + loss_t2i)

    def compute_l1_reg(self, z: torch.Tensor) -> torch.Tensor:
        """L1 penalty on representation coordinates."""
        # TODO: Return z.abs().mean()
        return z.abs().mean()

    def forward(
        self,
        im_anchor: torch.Tensor,
        im_pos: torch.Tensor,
        text: torch.Tensor,
        lambda_tcn: float = 1.0,
        lambda_lang: float = 1.0,
        lambda_reg: float = 1e-4,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Multi-task forward pass.
        Returns:
            Tuple: (total_loss, loss_tcn, loss_lang, loss_reg)
        """
        # TODO:
        z_anchor = self.encode_image(im_anchor)
        z_pos = self.encode_image(im_pos)
        z_text = self.encode_text(text)

        loss_tcn = self.compute_tcn_loss(z_anchor, z_pos)
        loss_lang = self.compute_lang_loss(z_anchor, z_text)
        loss_reg = self.compute_l1_reg(z_anchor)

        total_loss = lambda_tcn * loss_tcn + lambda_lang * loss_lang + lambda_reg * loss_reg
        return total_loss, loss_tcn, loss_lang, loss_reg

