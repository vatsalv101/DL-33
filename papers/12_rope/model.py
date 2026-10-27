import math
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """
    Rotates half the hidden dimensions of the input tensor:
    [-x_{d/2:}, x_{:d/2}]

    Args:
        x (Tensor): Input tensor of shape (..., d).

    Returns:
        Tensor: Rotated tensor of same shape (..., d).
    """
    ################################################################
    # TODO: Implement rotate_half
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)
    ################################################################


def apply_rotary_pos_emb(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> torch.Tensor:
    """
    Applies Rotary Position Embedding to input tensor x:
    RoPE(x, m) = (x * cos) + (rotate_half(x) * sin)

    Args:
        x   (Tensor): Query or Key tensor of shape (B, H, T, d) or broadcastable.
        cos (Tensor): Cosine frequencies of shape (1, 1, T, d) or broadcastable.
        sin (Tensor): Sine frequencies of shape (1, 1, T, d) or broadcastable.

    Returns:
        Tensor: Position-embedded tensor of same shape as x.
    """
    ################################################################
    # TODO: Implement apply_rotary_pos_emb
    return (x * cos) + (rotate_half(x) * sin)
    ################################################################


class RotaryPositionalEmbedding(nn.Module):
    """
    Rotary Position Embedding (RoPE) cache module.
    Computes and caches frequency tensors:
    theta_i = 1 / (base^(2i / dim)), for i in [0, dim/2)

    Args:
        dim         (int): Head embedding dimension (must be even).
        max_seq_len (int): Maximum sequence length to precompute. Default: 4096.
        base      (float): Exponential frequency base. Default: 10000.0.
    """

    def __init__(
        self, dim: int, max_seq_len: int = 4096, base: float = 10000.0
    ):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base

        ################################################################
        # TODO: Implement RotaryPositionalEmbedding __init__
        # 1. Compute inv_freq = 1 / (base^(2i / dim))
        inv_freq = 1 / (base ** (torch.arange(0, dim, 2)/dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len: int):
        self.max_seq_len = max(self.max_seq_len, seq_len)
        t = torch.arange(self.max_seq_len, dtype=torch.float32, device=self.inv_freq.device)
        freqs = torch.outer(t, self.inv_freq)  
        emb = torch.cat((freqs, freqs), dim=-1)  
        cos = emb.cos().unsqueeze(0).unsqueeze(0)  
        sin = emb.sin().unsqueeze(0).unsqueeze(0)  
        self.register_buffer("cos_cached", cos, persistent=False)
        self.register_buffer("sin_cached", sin, persistent=False)
        ################################################################

    def forward(
        self,
        seq_len: int,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Retrieves cached cosine and sine frequency tensors for given sequence length.

        Args:
            seq_len (int): Length of the sequence.
            device  (torch.device, optional): Target device.
            dtype   (torch.dtype, optional): Target dtype.

        Returns:
            Tuple[Tensor, Tensor]: (cos, sin) tensors of shape (1, 1, seq_len, dim).
        """
        ################################################################
        # TODO: Implement RotaryPositionalEmbedding forward pass
        if seq_len > self.max_seq_len:
            self._build_cache(seq_len)
        cos = self.cos_cached[:, :, :seq_len, :].to(device = device, dtype=dtype)
        sin = self.sin_cached[:, :, :seq_len, :].to(device = device, dtype=dtype)
        return cos, sin
        ################################################################


class RoPEMultiHeadAttention(nn.Module):
    """
    Multi-Head Attention with Rotary Position Embedding applied to Q and K.

    Args:
        d_model     (int): Total embedding dimension.
        num_heads   (int): Number of parallel attention heads. Must divide d_model.
        dropout   (float): Dropout probability. Default: 0.1.
        max_seq_len (int): Max sequence length for RoPE cache. Default: 4096.
        base      (float): Base for RoPE frequency calculation. Default: 10000.0.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout: float = 0.1,
        max_seq_len: int = 4096,
        base: float = 10000.0,
    ):
        super().__init__()
        assert d_model % num_heads == 0, f"d_model ({d_model}) must be divisible by num_heads ({num_heads})"
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        ################################################################
        # TODO: Implement RoPEMultiHeadAttention __init__
        self.wq = nn.Linear(d_model, d_model, bias = False)
        self.wk = nn.Linear(d_model, d_model, bias = False)
        self.wv = nn.Linear(d_model, d_model, bias = False)
        self.wo = nn.Linear(d_model, d_model, bias = False)

        self.dropout = nn.Dropout(dropout)
        self.rope = RotaryPositionalEmbedding(self.d_k, max_seq_len, base)

        pass
        ################################################################

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        cos: Optional[torch.Tensor] = None,
        sin: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for RoPE Multi-Head Attention.

        Args:
            x    (Tensor): Input token representations of shape (B, T, d_model).
            mask (Tensor, optional): Attention mask broadcastable to (B, num_heads, T, T).
            cos  (Tensor, optional): Precomputed cos tensor of shape (1, 1, T, d_k).
            sin  (Tensor, optional): Precomputed sin tensor of shape (1, 1, T, d_k).

        Returns:
            Tuple[Tensor, Tensor]:
                - output: Projected attention output of shape (B, T, d_model).
                - attn_weights: Attention distribution of shape (B, num_heads, T, T).
        """
        ################################################################
        # TODO: Implement RoPEMultiHeadAttention forward pass
        bs, seq_len, _ = x.shape

        q = self.wq(x).view(bs, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        k = self.wk(x).view(bs, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        v = self.wv(x).view(bs, seq_len, self.num_heads, self.d_k).transpose(1, 2)

        if cos is None or sin is None:
            cos, sin = self.rope(seq_len, device = x.device, dtype=q.dtype)

        q = apply_rotary_pos_emb(q, cos, sin)
        k = apply_rotary_pos_emb(k, cos, sin)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)

        if mask is not None:
            if mask.dim() == 2:
                mask = mask.unsqueeze(0).unsqueeze(0)
            elif mask.dim() == 3:
                mask = mask.unsqueeze(1)

            if mask.dtype == torch.bool:
                scores = scores.masked_fill(~mask, -1e9)
            elif mask.dtype.is_floating_point and (mask < -1).any():
                scores = scores + mask
            else:
                scores = scores.masked_fill(mask == 0, -1e9)

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        out = torch.matmul(attn_weights, v)

        out = out.transpose(1, 2).contiguous().view(bs, seq_len, self.d_model)
        out = self.wo(out)
        return out, attn_weights
        ################################################################


class PositionwiseFeedForward(nn.Module):
    """
    Position-wise Feed-Forward Network: FFN(x) = max(0, x W_1 + b_1) W_2 + b_2

    Args:
        d_model (int): Model embedding dimension.
        d_ff    (int): Hidden dimension.
        dropout (float): Dropout probability. Default: 0.1.
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff

        ################################################################
        # TODO: Implement PositionwiseFeedForward __init__
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        pass
        ################################################################

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (Tensor): Input tensor of shape (B, T, d_model).

        Returns:
            Tensor: Output tensor of shape (B, T, d_model).
        """
        ################################################################
        # TODO: Implement PositionwiseFeedForward forward pass
        return self.fc2(self.dropout(self.relu(self.fc1(x))))
        ################################################################


class RoFormerEncoderLayer(nn.Module):
    """
    Single RoFormer Encoder Layer (Pre-LN architecture):
    x -> x + Dropout(RoPEMHA(LayerNorm(x))) -> x + Dropout(FFN(LayerNorm(x)))

    Args:
        d_model     (int): Embedding dimension.
        num_heads   (int): Number of attention heads.
        d_ff        (int): Feed-forward dimension.
        dropout   (float): Dropout probability. Default: 0.1.
        max_seq_len (int): Maximum sequence length. Default: 4096.
        base      (float): RoPE frequency base. Default: 10000.0.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        dropout: float = 0.1,
        max_seq_len: int = 4096,
        base: float = 10000.0,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff

        ################################################################
        # TODO: Implement RoFormerEncoderLayer __init__
        self.self_attn = RoPEMultiHeadAttention(
            d_model, num_heads, dropout=dropout, max_seq_len=max_seq_len, base=base
        )
        self.ffn = PositionwiseFeedForward(d_model, d_ff, dropout=dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        ################################################################

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        cos: Optional[torch.Tensor] = None,
        sin: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass for RoFormer Encoder Layer.

        Args:
            x    (Tensor): Input representations of shape (B, T, d_model).
            mask (Tensor, optional): Mask tensor.
            cos  (Tensor, optional): Precomputed RoPE cos tensor.
            sin  (Tensor, optional): Precomputed RoPE sin tensor.

        Returns:
            Tensor: Encoded representations of shape (B, T, d_model).
        """
        ################################################################
        # TODO: Implement RoFormerEncoderLayer forward pass
        normed = self.norm1(x)
        attn_out, _ = self.self_attn(normed, mask=mask, cos=cos, sin=sin)
        x = x + self.dropout1(attn_out)

        ffn_out = self.ffn(self.norm2(x))
        x = x + self.dropout2(ffn_out)
        return x
        ################################################################


class RoFormerEncoder(nn.Module):
    """
    RoFormer Sequence Encoder consisting of token embeddings and a stack of RoFormerEncoderLayers.
    Note: NO additive positional embedding is added to token embeddings!

    Args:
        vocab_size  (int): Vocabulary size.
        d_model     (int): Embedding dimension.
        num_layers  (int): Number of encoder layers.
        num_heads   (int): Number of attention heads.
        d_ff        (int): Feed-forward hidden dimension.
        max_seq_len (int): Maximum sequence length. Default: 4096.
        dropout   (float): Dropout probability. Default: 0.1.
        base      (float): RoPE frequency base. Default: 10000.0.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        max_seq_len: int = 4096,
        dropout: float = 0.1,
        base: float = 10000.0,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_layers = num_layers

        ################################################################
        # TODO: Implement RoFormerEncoder __init__
        self.d_model = d_model
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.rope = RotaryPositionalEmbedding(d_model // num_heads, max_seq_len=max_seq_len, base=base)

        self.layers = nn.ModuleList(
            [
                RoFormerEncoderLayer(
                    d_model, num_heads, d_ff, dropout=dropout, max_seq_len=max_seq_len, base=base
                )
                for _ in range(num_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        ################################################################

    def forward(
        self,
        input_ids: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            input_ids (Tensor): Token indices of shape (B, T).
            mask      (Tensor, optional): Mask tensor broadcastable to attention map.

        Returns:
            Tensor: Encoded representations of shape (B, T, d_model).
        """
        ################################################################
        # TODO: Implement RoFormerEncoder forward pass
        seq_len = input_ids.shape[1]
        x = self.dropout(self.token_embedding(input_ids))

        cos, sin = self.rope(seq_len, device=input_ids.device, dtype=x.dtype)

        for layer in self.layers:
            x = layer(x, mask=mask, cos=cos, sin=sin)

        return self.final_norm(x)
        ################################################################
