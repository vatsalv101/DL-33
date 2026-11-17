from typing import Tuple, Optional
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from core.vit import PatchEmbedding, ViTBlock


def patchify(imgs: torch.Tensor, patch_size: int) -> torch.Tensor:
    """
    Slices images into patches and flattens each patch.

    Args:
        imgs: Tensor of shape (B, 3, H, W)
        patch_size: Spatial dimensions P of each patch
    Returns:
        Tensor of shape (B, num_patches, P * P * 3)
    """
    p = patch_size
    assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

    h = w = imgs.shape[2] // p
    x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
    x = torch.einsum("nchpwq->nhwpqc", x)
    x = x.reshape(shape=(imgs.shape[0], h * w, p * p * 3))
    return x


def unpatchify(x: torch.Tensor, patch_size: int, in_chans: int = 3) -> torch.Tensor:
    """
    Reconstructs 2D images from flattened patch tokens.

    Args:
        x: Tensor of shape (B, num_patches, P * P * in_chans)
        patch_size: Spatial dimension P
        in_chans: Number of image channels (default: 3)
    Returns:
        Tensor of shape (B, in_chans, H, W)
    """
    p = patch_size
    h = w = int(x.shape[1] ** 0.5)
    assert h * w == x.shape[1]

    x = x.reshape(shape=(x.shape[0], h, w, p, p, in_chans))
    x = torch.einsum("nhwpqc->nchpwq", x)
    imgs = x.reshape(shape=(x.shape[0], in_chans, h * p, w * p))
    return imgs


def random_masking(
    x: torch.Tensor, mask_ratio: float
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Performs per-sample random masking without replacement by shuffling in the sequence dimension.

    Args:
        x: Patch sequence of shape (B, N, D)
        mask_ratio: Fraction of patches to remove (e.g. 0.75)
    Returns:
        Tuple:
            - x_masked: Kept visible tokens of shape (B, N_vis, D)
            - mask: Binary mask of shape (B, N) where 0 is kept and 1 is removed
            - ids_restore: Index tensor to restore original sequence order of shape (B, N)
    """
    N, D = x.shape[1], x.shape[2]
    len_keep = int(N * (1.0 - mask_ratio))

    noise = torch.rand(x.shape[0], N, device=x.device)

    ids_shuffle = torch.argsort(noise, dim=1)
    ids_restore = torch.argsort(ids_shuffle, dim=1)

    ids_keep = ids_shuffle[:, :len_keep]
    x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

    mask = torch.ones([x.shape[0], N], device=x.device)
    mask[:, :len_keep] = 0
    mask = torch.gather(mask, dim=1, index=ids_restore)

    return x_masked, mask, ids_restore


class MaskedAutoencoderViT(nn.Module):
    """
    Masked Autoencoder with Vision Transformer backbone (He et al., 2022).
    Asymmetric architecture:
    - Deep encoder operates strictly on visible patches (25%)
    - Shallow decoder operates on full sequence with learnable [MASK] tokens
    """

    def __init__(
        self,
        img_size: int = 16,
        patch_size: int = 4,
        in_chans: int = 3,
        embed_dim: int = 64,
        depth: int = 2,
        num_heads: int = 2,
        decoder_embed_dim: int = 32,
        decoder_depth: int = 1,
        decoder_num_heads: int = 2,
        mlp_ratio: float = 2.0,
        norm_pix_loss: bool = True,
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.norm_pix_loss = norm_pix_loss

        # 1. MAE Encoder (ViT on visible tokens)
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))

        self.blocks = nn.ModuleList([
            ViTBlock(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=True,
            )
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)

        # 2. MAE Decoder (Lightweight ViT on full sequence)
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, decoder_embed_dim))

        self.decoder_blocks = nn.ModuleList([
            ViTBlock(
                dim=decoder_embed_dim,
                num_heads=decoder_num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=True,
            )
            for _ in range(decoder_depth)
        ])
        self.decoder_norm = nn.LayerNorm(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size * patch_size * in_chans, bias=True)

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.pos_embed, std=0.02)
        nn.init.normal_(self.decoder_pos_embed, std=0.02)
        nn.init.normal_(self.cls_token, std=0.02)
        nn.init.normal_(self.mask_token, std=0.02)

    def forward_encoder(
        self, x: torch.Tensor, mask_ratio: float
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.patch_embed(x)
        x = x + self.pos_embed[:, 1:, :]
        x, mask, ids_restore = random_masking(x, mask_ratio)

        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        for block in self.blocks:
            x = block(x)
        x = self.norm(x)

        return x, mask, ids_restore

    def forward_decoder(self, x: torch.Tensor, ids_restore: torch.Tensor) -> torch.Tensor:
        x = self.decoder_embed(x)

        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))
        x = torch.cat([x[:, :1, :], x_], dim=1)

        x = x + self.decoder_pos_embed

        for block in self.decoder_blocks:
            x = block(x)
        x = self.decoder_norm(x)

        x = self.decoder_pred(x)
        x = x[:, 1:, :]
        return x

    def forward_loss(
        self, imgs: torch.Tensor, pred: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        target = patchify(imgs, self.patch_size)

        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1e-6) ** 0.5

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)
        loss = (loss * mask).sum() / mask.sum()
        return loss

    def forward(
        self, imgs: torch.Tensor, mask_ratio: float = 0.75
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        latent, mask, ids_restore = self.forward_encoder(imgs, mask_ratio)
        pred = self.forward_decoder(latent, ids_restore)
        loss = self.forward_loss(imgs, pred, mask)
        return loss, pred, mask
