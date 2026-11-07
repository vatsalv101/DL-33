import math
from typing import Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F


class PatchEmbedding(nn.Module):
    """
    Image to Patch Embedding: Slices a 2D image into non-overlapping PxP patches
    and linearly projects each flattened patch to dimension D.

    Args:
        img_size   (int): Input image height/width (assumed square). Default: 224.
        patch_size (int): Patch size P (assumed square). Default: 16.
        in_channels (int): Number of input image channels. Default: 3.
        embed_dim  (int): Patch embedding dimension D. Default: 768.
    """

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        embed_dim: int = 768,
    ):
        super().__init__()
        assert img_size % patch_size == 0, f"img_size ({img_size}) must be divisible by patch_size ({patch_size})"
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = img_size // patch_size
        self.num_patches = self.grid_size * self.grid_size

        # 2D Conv with kernel_size = stride = patch_size is mathematically equivalent to patch projection
        self.proj = nn.Conv2d(
            in_channels=in_channels,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (Tensor): Input images of shape (B, in_channels, H, W).

        Returns:
            Tensor: Patch sequence of shape (B, num_patches, embed_dim).
        """
        ################################################################
        # TODO: Implement PatchEmbedding forward pass
        B, C, H, W = x.shape
        assert H == self.img_size and W == self.img_size, (
            f"Input image size ({H}x{W}) doesn't match model ({self.img_size}x{self.img_size})"
        )
        x = self.proj(x)  
        x = x.flatten(2).transpose(1, 2)  
        return x
        ################################################################


class ViTAttention(nn.Module):
    """
    Multi-Head Self-Attention for Vision Transformer.

    Args:
        dim         (int): Hidden embedding dimension.
        num_heads   (int): Number of attention heads.
        qkv_bias    (bool): Whether to include bias in QKV projection. Default: True.
        attn_drop (float): Attention dropout. Default: 0.0.
        proj_drop (float): Output projection dropout. Default: 0.0.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ):
        super().__init__()
        assert dim % num_heads == 0, f"dim ({dim}) must be divisible by num_heads ({num_heads})"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (Tensor): Input token sequence of shape (B, N, dim).

        Returns:
            Tensor: Attended token sequence of shape (B, N, dim).
        """
        ################################################################
        # TODO: Implement ViTAttention forward pass
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
        ################################################################


class ViTMLP(nn.Module):
    """
    Feed-Forward Network with GELU activation:
    Linear(dim -> hidden_dim) -> GELU -> Dropout -> Linear(hidden_dim -> dim) -> Dropout

    Args:
        in_features  (int): Input dimension.
        hidden_features (int): Intermediate hidden dimension (typically 4 * in_features).
        drop        (float): Dropout probability. Default: 0.0.
    """

    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        drop: float = 0.0,
    ):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, in_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (Tensor): Shape (B, N, in_features).

        Returns:
            Tensor: Shape (B, N, in_features).
        """
        ################################################################
        # TODO: Implement ViTMLP forward pass
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x
        ################################################################


class ViTBlock(nn.Module):
    """
    Vision Transformer Encoder Layer with Pre-Layer Normalization:
    x -> x + Attention(LayerNorm(x))
    x -> x + MLP(LayerNorm(x))

    Args:
        dim        (int): Embedding dimension.
        num_heads  (int): Number of attention heads.
        mlp_ratio (float): Ratio of MLP hidden dimension to embedding dimension. Default: 4.0.
        qkv_bias   (bool): Whether to include bias in QKV projection. Default: True.
        drop      (float): Dropout probability. Default: 0.0.
        attn_drop (float): Attention dropout. Default: 0.0.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = ViTAttention(
            dim=dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = ViTMLP(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            drop=drop,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (Tensor): Input token representations of shape (B, N, dim).

        Returns:
            Tensor: Output token representations of shape (B, N, dim).
        """
        ################################################################
        # TODO: Implement ViTBlock forward pass
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x
        ################################################################


class ViT(nn.Module):
    """
    Vision Transformer (Dosovitskiy et al., 2020) for Image Classification:
    Image -> PatchEmbedding -> Prepend [CLS] -> Add Position Embeddings -> L x ViTBlocks -> LayerNorm -> Linear Head

    Args:
        img_size    (int): Input image size. Default: 224.
        patch_size  (int): Patch size. Default: 16.
        in_channels (int): Input image channels. Default: 3.
        num_classes (int): Number of target classes. Default: 1000.
        embed_dim   (int): Hidden embedding dimension. Default: 768.
        depth       (int): Number of ViTBlock layers. Default: 12.
        num_heads   (int): Number of attention heads. Default: 12.
        mlp_ratio (float): Ratio of MLP hidden dim to embed_dim. Default: 4.0.
        qkv_bias    (bool): Whether to include bias in QKV. Default: True.
        drop_rate  (float): Dropout rate. Default: 0.0.
        attn_drop_rate (float): Attention dropout rate. Default: 0.0.
    """

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        num_classes: int = 1000,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.embed_dim = embed_dim

        self.patch_embed = PatchEmbedding(
            img_size=img_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim,
        )
        num_patches = self.patch_embed.num_patches

        # Learnable class token [CLS] prepended to the patch sequence
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        # Learnable 1D position embeddings
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        # Transformer Encoder Blocks
        self.blocks = nn.ModuleList(
            [
                ViTBlock(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                )
                for _ in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)

        # Classifier head
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        self._init_weights()

    def _init_weights(self):
        """Initialize weights: normal_(0, 0.02) for embeddings and linear weights, zeros for bias."""
        nn.init.normal_(self.pos_embed, std=0.02)
        nn.init.normal_(self.cls_token, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extracts feature representations from input images.

        Args:
            x (Tensor): Input image tensor of shape (B, C, H, W).

        Returns:
            Tensor: Output token representations of shape (B, num_patches + 1, embed_dim).
        """
        ################################################################
        # TODO: Implement forward_features
        B = x.shape[0]
        x = self.patch_embed(x)  

        cls_tokens = self.cls_token.expand(B, -1, -1)  
        x = torch.cat((cls_tokens, x), dim=1)  
        x = self.pos_drop(x + self.pos_embed)

        for block in self.blocks:
            x = block(x)

        x = self.norm(x)
        return x
        ################################################################

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Full classification forward pass.

        Args:
            x (Tensor): Input image tensor of shape (B, in_channels, H, W).

        Returns:
            Tensor: Class logits of shape (B, num_classes).
        """
        ################################################################
        # TODO: Implement ViT forward pass
        x = self.forward_features(x)
        cls_token_final = x[:, 0]  
        logits = self.head(cls_token_final)
        return logits
        ################################################################
