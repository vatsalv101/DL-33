import math
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class ScaledDotProductAttention(nn.Module):
    """
    Scaled Dot-Product Attention mechanism.
    Attention(Q, K, V) = softmax(Q K^T / sqrt(d_k) + mask) V

    Args:
        dropout (float): Dropout probability applied to attention weights. Default: 0.0.
    """

    def __init__(self, dropout: float = 0.0):
        super().__init__()
        self.dropout_p = dropout

        ################################################################
        # TODO: Implement ScaledDotProductAttention __init__
        self.dropout = nn.Dropout(dropout)
        pass
        ################################################################

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for Scaled Dot-Product Attention.

        Args:
            q    (Tensor): Query tensor of shape (B, ..., T_q, d_k).
            k    (Tensor): Key tensor of shape (B, ..., T_k, d_k).
            v    (Tensor): Value tensor of shape (B, ..., T_k, d_v).
            mask (Tensor, optional): Mask tensor broadcastable to attention scores.
                                    0 or False denotes positions to mask out (filled with -1e9).

        Returns:
            Tuple[Tensor, Tensor]:
                - output: Attended values of shape (B, ..., T_q, d_v).
                - attn_weights: Attention probabilities of shape (B, ..., T_q, T_k).
        """
        ################################################################
        # TODO: Implement ScaledDotProductAttention forward pass
        d_k = q.shape[-1]
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)

        if mask is not None:
            if mask.dtype == torch.bool:
                scores = scores.masked_fill(~mask, -1e9)
            elif mask.dtype.is_floating_point and (mask < -1).any():
                scores = scores + mask
            else:
                scores = scores.masked_fill(mask == 0, -1e9)

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        output = torch.matmul(attn_weights, v)
        return output, attn_weights
        ################################################################


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention module.
    Projects Q, K, V into h representation subspaces, computes parallel scaled dot-product
    attentions, concatenates outputs, and projects through an output linear layer.

    Args:
        d_model   (int): Total embedding dimension.
        num_heads (int): Number of parallel attention heads. Must divide d_model.
        dropout (float): Dropout probability. Default: 0.1.
    """

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0, f"d_model ({d_model}) must be divisible by num_heads ({num_heads})"
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        ################################################################
        # TODO: Implement MultiHeadAttention __init__
        self.wq = nn.Linear(d_model, d_model)
        self.wk = nn.Linear(d_model, d_model)
        self.wv = nn.Linear(d_model, d_model)
        self.wo = nn.Linear(d_model, d_model)

        self.attention = ScaledDotProductAttention(dropout=dropout)
        pass
        ################################################################

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for Multi-Head Attention.

        Args:
            q    (Tensor): Query tensor of shape (B, T_q, d_model).
            k    (Tensor): Key tensor of shape (B, T_k, d_model).
            v    (Tensor): Value tensor of shape (B, T_k, d_model).
            mask (Tensor, optional): Mask tensor broadcastable to (B, num_heads, T_q, T_k).

        Returns:
            Tuple[Tensor, Tensor]:
                - output: Projected attention output of shape (B, T_q, d_model).
                - attn_weights: Attention weights of shape (B, num_heads, T_q, T_k).
        """
        ################################################################
        # TODO: Implement MultiHeadAttention forward pass
        bs = q.shape[0]

        q = self.wq(q).view(bs, -1, self.num_heads, self.d_k).transpose(1, 2)
        k = self.wk(k).view(bs, -1, self.num_heads, self.d_k).transpose(1, 2)
        v = self.wv(v).view(bs, -1, self.num_heads, self.d_k).transpose(1, 2)

        if mask is not None:
            if mask.dim() == 2:
                mask = mask.unsqueeze(0).unsqueeze(0)
            elif mask.dim() == 3:
                mask = mask.unsqueeze(1)

        out, attn_weights = self.attention(q, k, v, mask = mask)
        out = out.transpose(1, 2).contiguous().view(bs, -1, self.d_model)
        out = self.wo(out)
        return out, attn_weights
        ################################################################


class PositionwiseFeedForward(nn.Module):
    """
    Position-wise Feed-Forward Network: FFN(x) = max(0, x W_1 + b_1) W_2 + b_2

    Args:
        d_model (int): Model embedding dimension.
        d_ff    (int): Inner hidden dimension (typically 4 * d_model).
        dropout (float): Dropout probability. Default: 0.1.
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff

        ################################################################
        # TODO: Implement PositionwiseFeedForward __init__
        self.fc1 = nn.Linear(d_model, 4 * d_model)
        self.fc2 = nn.Linear(4 * d_model, d_model)
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


class PositionalEncoding(nn.Module):
    """
    Fixed Sinusoidal Positional Encoding.
    PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

    Args:
        d_model (int): Model embedding dimension.
        max_len (int): Maximum sequence length to precompute. Default: 5000.
        dropout (float): Dropout probability. Default: 0.1.
    """

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len

        ################################################################
        # TODO: Implement PositionalEncoding __init__
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)                             
        pass
        ################################################################

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (Tensor): Input sequence embeddings of shape (B, T, d_model).

        Returns:
            Tensor: Embeddings with positional encodings added, shape (B, T, d_model).
        """
        ################################################################
        # TODO: Implement PositionalEncoding forward pass
        x = x + self.pe[:, : x.shape[1], :]
        return self.dropout(x)
        ################################################################


class TransformerEncoderLayer(nn.Module):
    """
    Single Transformer Encoder Layer:
    x -> LayerNorm(x + Dropout(SelfAttention(x))) -> LayerNorm(x + Dropout(FFN(x)))

    Args:
        d_model   (int): Model embedding dimension.
        num_heads (int): Number of attention heads.
        d_ff      (int): Feed-forward hidden dimension.
        dropout (float): Dropout probability. Default: 0.1.
    """

    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff

        ################################################################
        # TODO: Implement TransformerEncoderLayer __init__
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ffn = PositionwiseFeedForward(d_model, d_ff, dropout = dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        pass
        ################################################################

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x    (Tensor): Input representations of shape (B, T, d_model).
            mask (Tensor, optional): Source mask broadcastable to attention map.

        Returns:
            Tensor: Encoded representations of shape (B, T, d_model).
        """
        ################################################################
        # TODO: Implement TransformerEncoderLayer forward pass
        attn_out, _ = self.self_attn(x, x, x, mask = mask)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_out))
        return x
        ################################################################


class TransformerEncoder(nn.Module):
    """
    Transformer Encoder consisting of a stack of N TransformerEncoderLayers.

    Args:
        num_layers (int): Number of stacked encoder layers.
        d_model    (int): Model embedding dimension.
        num_heads  (int): Number of attention heads.
        d_ff       (int): Feed-forward hidden dimension.
        dropout  (float): Dropout probability. Default: 0.1.
    """

    def __init__(
        self,
        num_layers: int,
        d_model: int,
        num_heads: int,
        d_ff: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_layers = num_layers

        ################################################################
        # TODO: Implement TransformerEncoder __init__
        self.layers = nn.ModuleList(
            [
                TransformerEncoderLayer(d_model, num_heads, d_ff, dropout=dropout)
                for _ in range(num_layers)
            ]
        )
        ################################################################

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x    (Tensor): Input token embeddings of shape (B, T, d_model).
            mask (Tensor, optional): Source mask broadcastable to attention map.

        Returns:
            Tensor: Encoded memory representations of shape (B, T, d_model).
        """
        ################################################################
        # TODO: Implement TransformerEncoder forward pass
        for layer in self.layers:
            x = layer(x, mask = mask)
        return x
        ################################################################


class TransformerDecoderLayer(nn.Module):
    """
    Single Transformer Decoder Layer:
    1. Masked Self-Attention + Residual + LayerNorm
    2. Cross-Attention (Query = decoder state, Key & Value = encoder memory) + Residual + LayerNorm
    3. Position-wise Feed-Forward + Residual + LayerNorm

    Args:
        d_model   (int): Model embedding dimension.
        num_heads (int): Number of attention heads.
        d_ff      (int): Feed-forward hidden dimension.
        dropout (float): Dropout probability. Default: 0.1.
    """

    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff

        ################################################################
        # TODO: Implement TransformerDecoderLayer __init__
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout=dropout)
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout=dropout)
        self.ffn = PositionwiseFeedForward(d_model, d_ff, dropout=dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        pass
        ################################################################

    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        trg_mask: Optional[torch.Tensor] = None,
        memory_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x           (Tensor): Target sequence representation of shape (B, T_trg, d_model).
            memory      (Tensor): Encoder output memory of shape (B, T_src, d_model).
            trg_mask    (Tensor, optional): Causal/target mask for self-attention.
            memory_mask (Tensor, optional): Source mask for cross-attention.

        Returns:
            Tensor: Updated target representations of shape (B, T_trg, d_model).
        """
        ################################################################
        # TODO: Implement TransformerDecoderLayer forward pass
        attn1, _ = self.self_attn(x, x, x, mask=trg_mask)
        x = self.norm1(x + self.dropout1(attn1))

        attn2, _ = self.cross_attn(x, memory, memory, mask = memory_mask)
        x = self.norm2(x + self.dropout2(attn2))

        ffn_out = self.ffn(x)
        x = self.norm3(x + self.dropout3(ffn_out))
        return x

        ################################################################


class TransformerDecoder(nn.Module):
    """
    Transformer Decoder consisting of a stack of N TransformerDecoderLayers.

    Args:
        num_layers (int): Number of stacked decoder layers.
        d_model    (int): Model embedding dimension.
        num_heads  (int): Number of attention heads.
        d_ff       (int): Feed-forward hidden dimension.
        dropout  (float): Dropout probability. Default: 0.1.
    """

    def __init__(
        self,
        num_layers: int,
        d_model: int,
        num_heads: int,
        d_ff: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_layers = num_layers

        ################################################################
        # TODO: Implement TransformerDecoder __init__
        self.layers = nn.ModuleList(
            [
                TransformerDecoderLayer(d_model, num_heads, d_ff, dropout = dropout)
                for _ in range(num_layers)
            ]
        )
        pass
        ################################################################

    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        trg_mask: Optional[torch.Tensor] = None,
        memory_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x           (Tensor): Target sequence representations of shape (B, T_trg, d_model).
            memory      (Tensor): Encoder memory of shape (B, T_src, d_model).
            trg_mask    (Tensor, optional): Target causal mask.
            memory_mask (Tensor, optional): Source memory mask.

        Returns:
            Tensor: Decoded representations of shape (B, T_trg, d_model).
        """
        ################################################################
        # TODO: Implement TransformerDecoder forward pass
        for layer in self.layers:
            x = layer(x, memory, trg_mask = trg_mask, memory_mask = memory_mask)
        return x
        ################################################################


class Transformer(nn.Module):
    """
    Full Encoder-Decoder Transformer Architecture (Vaswani et al., 2017).

    Args:
        src_vocab_size     (int): Size of source vocabulary.
        trg_vocab_size     (int): Size of target vocabulary.
        d_model            (int): Embedding dimension. Default: 512.
        num_heads          (int): Number of attention heads. Default: 8.
        num_encoder_layers (int): Number of encoder layers. Default: 6.
        num_decoder_layers (int): Number of decoder layers. Default: 6.
        d_ff               (int): Feed-forward hidden dimension. Default: 2048.
        max_len            (int): Maximum sequence length. Default: 5000.
        dropout          (float): Dropout probability. Default: 0.1.
    """

    def __init__(
        self,
        src_vocab_size: int,
        trg_vocab_size: int,
        d_model: int = 512,
        num_heads: int = 8,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 6,
        d_ff: int = 2048,
        max_len: int = 5000,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.scale = math.sqrt(d_model)

        ################################################################
        # TODO: Implement Transformer __init__
        self.src_embedding = nn.Embedding(src_vocab_size, d_model)
        self.trg_embedding = nn.Embedding(trg_vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len=max_len, dropout=dropout)

        self.encoder = TransformerEncoder(num_encoder_layers, d_model, num_heads, d_ff, dropout=dropout)
        self.decoder = TransformerDecoder(num_decoder_layers, d_model, num_heads, d_ff, dropout = dropout)
        self.fc_out = nn.Linear(d_model, trg_vocab_size)
        pass
        ################################################################

    @staticmethod
    def generate_square_subsequent_mask(
        sz: int, device: Optional[torch.device] = None
    ) -> torch.Tensor:
        """
        Generate a lower-triangular causal mask where 1 (True) means valid and 0 (False) means masked.

        Args:
            sz     (int): Sequence length.
            device (torch.device, optional): Target device.

        Returns:
            Tensor: Boolean causal mask of shape (sz, sz).
        """
        mask = torch.tril(torch.ones(sz, sz, device=device, dtype=torch.bool))
        return mask

    def encode(
        self, src: torch.Tensor, src_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Encode source sequence tokens into memory representations.

        Args:
            src      (Tensor): Source token indices of shape (B, T_src).
            src_mask (Tensor, optional): Source mask.

        Returns:
            Tensor: Encoded representations of shape (B, T_src, d_model).
        """
        ################################################################
        # TODO: Implement Transformer encode
        src_emb = self.src_embedding(src) * self.scale
        src_emb = self.pos_encoder(src_emb)
        return self.encoder(src_emb, mask=src_mask)
        ################################################################

    def decode(
        self,
        trg: torch.Tensor,
        memory: torch.Tensor,
        trg_mask: Optional[torch.Tensor] = None,
        memory_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Decode target tokens conditioned on encoder memory.

        Args:
            trg         (Tensor): Target token indices of shape (B, T_trg).
            memory      (Tensor): Encoder memory of shape (B, T_src, d_model).
            trg_mask    (Tensor, optional): Causal mask for target self-attention.
            memory_mask (Tensor, optional): Source mask for cross-attention.

        Returns:
            Tensor: Decoded representations of shape (B, T_trg, d_model).
        """
        ################################################################
        # TODO: Implement Transformer decode
        trg_emb = self.trg_embedding(trg) * self.scale
        trg_emb = self.pos_encoder(trg_emb)
        return self.decoder(trg_emb, memory, trg_mask, memory_mask)
        ################################################################

    def forward(
        self,
        src: torch.Tensor,
        trg: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        trg_mask: Optional[torch.Tensor] = None,
        memory_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Full forward pass for Transformer Seq2Seq.

        Args:
            src         (Tensor): Source sequence tokens of shape (B, T_src).
            trg         (Tensor): Target sequence tokens of shape (B, T_trg).
            src_mask    (Tensor, optional): Source attention mask.
            trg_mask    (Tensor, optional): Target attention mask (e.g. causal mask).
            memory_mask (Tensor, optional): Memory cross-attention mask.

        Returns:
            Tensor: Target token logits of shape (B, T_trg, trg_vocab_size).
        """
        ################################################################
        # TODO: Implement Transformer forward pass
        memory = self.encode(src, src_mask=src_mask)
        dec_out = self.decode(
            trg, memory, trg_mask = trg_mask, memory_mask = memory_mask
        )
        logits = self.fc_out(dec_out)
        return logits
        
        ################################################################
