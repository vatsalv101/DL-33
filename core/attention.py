from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class BahdanauAttention(nn.Module):
    """
    Bahdanau (Additive) Attention Mechanism.
    Score: e_{t,i} = v_a^T * tanh(W_a * s_{t-1} + U_a * h_i)

    Args:
        enc_hidden_dim (int): Hidden dimension of the encoder per direction.
        dec_hidden_dim (int): Hidden dimension of the decoder.
        attn_dim       (int): Internal attention bottleneck dimension.
    """

    def __init__(self, enc_hidden_dim: int, dec_hidden_dim: int, attn_dim: int):
        super().__init__()
        self.enc_hidden_dim = enc_hidden_dim
        self.dec_hidden_dim = dec_hidden_dim
        self.attn_dim = attn_dim

        self.W_a = nn.Linear(dec_hidden_dim, attn_dim, bias=False)
        self.U_a = nn.Linear(enc_hidden_dim * 2, attn_dim, bias=False)
        self.v_a = nn.Linear(attn_dim, 1, bias=False)

    def forward(
        self, hidden: torch.Tensor, encoder_outputs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for Bahdanau Attention.

        Args:
            hidden          (Tensor): Decoder hidden state s_{t-1} of shape (B, dec_hidden_dim).
            encoder_outputs (Tensor): Bidirectional encoder outputs of shape (B, src_len, enc_hidden_dim * 2).

        Returns:
            Tuple[Tensor, Tensor]:
                - context: Context vector c_t of shape (B, enc_hidden_dim * 2).
                - attn_weights: Normalized attention weights of shape (B, src_len).
        """
        query = self.W_a(hidden).unsqueeze(1)
        keys = self.U_a(encoder_outputs)
        energy = self.v_a(torch.tanh(query + keys)).squeeze(2)
        attn_weights = F.softmax(energy, dim=1)
        context = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs).squeeze(1)
        return context, attn_weights


class Encoder(nn.Module):
    """
    Bidirectional GRU Encoder for Seq2Seq.

    Args:
        input_dim      (int): Source vocabulary size.
        emb_dim        (int): Token embedding dimension.
        enc_hidden_dim (int): Hidden dimension of GRU per direction.
        dec_hidden_dim (int): Decoder hidden dimension (for initial state projection).
        dropout      (float): Dropout probability. Default: 0.5.
    """

    def __init__(
        self,
        input_dim: int,
        emb_dim: int,
        enc_hidden_dim: int,
        dec_hidden_dim: int,
        dropout: float = 0.5,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.emb_dim = emb_dim
        self.enc_hidden_dim = enc_hidden_dim
        self.dec_hidden_dim = dec_hidden_dim

        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.rnn = nn.GRU(emb_dim, enc_hidden_dim, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(enc_hidden_dim * 2, dec_hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for Encoder.

        Args:
            src (Tensor): Source sequence tokens of shape (B, src_len).

        Returns:
            Tuple[Tensor, Tensor]:
                - encoder_outputs of shape (B, src_len, enc_hidden_dim * 2).
                - initial decoder hidden state of shape (B, dec_hidden_dim).
        """
        embedded = self.dropout(self.embedding(src))
        output, hidden = self.rnn(embedded)
        cat_hidden = torch.cat([hidden[-2], hidden[-1]], dim=1)
        s_0 = torch.tanh(self.fc(cat_hidden))
        return output, s_0


class Decoder(nn.Module):
    """
    Attention-guided GRU Decoder.

    Args:
        output_dim     (int): Target vocabulary size.
        emb_dim        (int): Target token embedding dimension.
        enc_hidden_dim (int): Encoder hidden dimension per direction.
        dec_hidden_dim (int): Decoder hidden dimension.
        dropout      (float): Dropout probability.
        attention (Attention): BahdanauAttention module.
    """

    def __init__(
        self,
        output_dim: int,
        emb_dim: int,
        enc_hidden_dim: int,
        dec_hidden_dim: int,
        dropout: float,
        attention: nn.Module,
    ):
        super().__init__()
        self.output_dim = output_dim
        self.attention = attention

        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.rnn = nn.GRU(emb_dim + enc_hidden_dim * 2, dec_hidden_dim, batch_first=True)
        self.fc_out = nn.Linear(dec_hidden_dim + enc_hidden_dim * 2 + emb_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        input: torch.Tensor,
        hidden: torch.Tensor,
        encoder_outputs: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward step for Decoder for a single time step.

        Args:
            input           (Tensor): Target token at current step of shape (B,).
            hidden          (Tensor): Decoder state s_{t-1} of shape (B, dec_hidden_dim).
            encoder_outputs (Tensor): Encoder annotations of shape (B, src_len, enc_hidden_dim * 2).

        Returns:
            Tuple[Tensor, Tensor, Tensor]:
                - prediction: Vocabulary logits of shape (B, output_dim).
                - hidden: Updated decoder state s_t of shape (B, dec_hidden_dim).
                - attn_weights: Attention distribution of shape (B, src_len).
        """
        embedded = self.dropout(self.embedding(input)).unsqueeze(1)
        context, attn_weights = self.attention(hidden, encoder_outputs)
        rnn_input = torch.cat([embedded, context.unsqueeze(1)], dim=2)
        output, new_hidden = self.rnn(rnn_input, hidden.unsqueeze(0))
        output = output.squeeze(1)
        embedded = embedded.squeeze(1)
        prediction = self.fc_out(torch.cat([output, context, embedded], dim=1))
        return prediction, new_hidden.squeeze(0), attn_weights


class Seq2Seq(nn.Module):
    """
    Full Encoder-Decoder Seq2Seq architecture with Bahdanau Attention.

    Args:
        encoder (Encoder): The bidirectional encoder.
        decoder (Decoder): The attention-guided decoder.
        pad_idx     (int): Padding token index. Default: 0.
    """

    def __init__(self, encoder: Encoder, decoder: Decoder, pad_idx: int = 0):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.pad_idx = pad_idx

    def forward(
        self,
        src: torch.Tensor,
        trg: torch.Tensor,
        teacher_forcing_ratio: float = 0.5,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for Seq2Seq model across target sequence.

        Args:
            src                   (Tensor): Source sequence tokens of shape (B, src_len).
            trg                   (Tensor): Target sequence tokens of shape (B, trg_len).
            teacher_forcing_ratio  (float): Probability of using ground truth token as next input.

        Returns:
            Tuple[Tensor, Tensor]:
                - outputs: Logits across target sequence of shape (B, trg_len, output_dim).
                - attentions: Attention distributions across target steps of shape (B, trg_len, src_len).
        """
        bs = src.shape[0]
        tl = trg.shape[1]
        tvs = self.decoder.output_dim
        sl = src.shape[1]

        output = torch.zeros(bs, tl, tvs, device=src.device)
        attentions = torch.zeros(bs, tl, sl, device=src.device)

        encoder_outputs, hidden = self.encoder(src)
        input_token = trg[:, 0]

        for t in range(1, tl):
            logits, hidden, attn = self.decoder(input_token, hidden, encoder_outputs)
            output[:, t] = logits
            attentions[:, t] = attn
            teacher_force = torch.rand(1).item() < teacher_forcing_ratio
            top1 = logits.argmax(1)
            input_token = trg[:, t] if teacher_force else top1
        return output, attentions


import math


class ScaledDotProductAttention(nn.Module):
    """
    Scaled Dot-Product Attention mechanism.
    Attention(Q, K, V) = softmax(Q K^T / sqrt(d_k) + mask) V

    Args:
        dropout (float): Dropout probability applied to attention weights. Default: 0.0.
    """

    def __init__(self, dropout: float = 0.0):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
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

        self.wq = nn.Linear(d_model, d_model)
        self.wk = nn.Linear(d_model, d_model)
        self.wv = nn.Linear(d_model, d_model)
        self.wo = nn.Linear(d_model, d_model)

        self.attention = ScaledDotProductAttention(dropout=dropout)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        bs = q.shape[0]

        q = self.wq(q).view(bs, -1, self.num_heads, self.d_k).transpose(1, 2)
        k = self.wk(k).view(bs, -1, self.num_heads, self.d_k).transpose(1, 2)
        v = self.wv(v).view(bs, -1, self.num_heads, self.d_k).transpose(1, 2)

        if mask is not None:
            if mask.dim() == 2:
                mask = mask.unsqueeze(0).unsqueeze(0)
            elif mask.dim() == 3:
                mask = mask.unsqueeze(1)

        out, attn_weights = self.attention(q, k, v, mask=mask)
        out = out.transpose(1, 2).contiguous().view(bs, -1, self.d_model)
        out = self.wo(out)
        return out, attn_weights


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

        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.dropout(self.relu(self.fc1(x))))


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

        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, : x.shape[1], :]
        return self.dropout(x)


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

        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ffn = PositionwiseFeedForward(d_model, d_ff, dropout=dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        attn_out, _ = self.self_attn(x, x, x, mask=mask)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout2(ffn_out))
        return x


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

        self.layers = nn.ModuleList(
            [
                TransformerEncoderLayer(d_model, num_heads, d_ff, dropout=dropout)
                for _ in range(num_layers)
            ]
        )

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, mask=mask)
        return x


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

        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout=dropout)
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout=dropout)
        self.ffn = PositionwiseFeedForward(d_model, d_ff, dropout=dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        trg_mask: Optional[torch.Tensor] = None,
        memory_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        attn1, _ = self.self_attn(x, x, x, mask=trg_mask)
        x = self.norm1(x + self.dropout1(attn1))

        attn2, _ = self.cross_attn(x, memory, memory, mask=memory_mask)
        x = self.norm2(x + self.dropout2(attn2))

        ffn_out = self.ffn(x)
        x = self.norm3(x + self.dropout3(ffn_out))
        return x


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

        self.layers = nn.ModuleList(
            [
                TransformerDecoderLayer(d_model, num_heads, d_ff, dropout=dropout)
                for _ in range(num_layers)
            ]
        )

    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        trg_mask: Optional[torch.Tensor] = None,
        memory_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, memory, trg_mask=trg_mask, memory_mask=memory_mask)
        return x


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

        self.src_embedding = nn.Embedding(src_vocab_size, d_model)
        self.trg_embedding = nn.Embedding(trg_vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len=max_len, dropout=dropout)

        self.encoder = TransformerEncoder(num_encoder_layers, d_model, num_heads, d_ff, dropout=dropout)
        self.decoder = TransformerDecoder(num_decoder_layers, d_model, num_heads, d_ff, dropout=dropout)
        self.fc_out = nn.Linear(d_model, trg_vocab_size)

    @staticmethod
    def generate_square_subsequent_mask(
        sz: int, device: Optional[torch.device] = None
    ) -> torch.Tensor:
        """
        Generate a lower-triangular causal mask where 1 (True) means valid and 0 (False) means masked.
        """
        mask = torch.tril(torch.ones(sz, sz, device=device, dtype=torch.bool))
        return mask

    def encode(
        self, src: torch.Tensor, src_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        src_emb = self.src_embedding(src) * self.scale
        src_emb = self.pos_encoder(src_emb)
        return self.encoder(src_emb, mask=src_mask)

    def decode(
        self,
        trg: torch.Tensor,
        memory: torch.Tensor,
        trg_mask: Optional[torch.Tensor] = None,
        memory_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        trg_emb = self.trg_embedding(trg) * self.scale
        trg_emb = self.pos_encoder(trg_emb)
        return self.decoder(trg_emb, memory, trg_mask=trg_mask, memory_mask=memory_mask)

    def forward(
        self,
        src: torch.Tensor,
        trg: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        trg_mask: Optional[torch.Tensor] = None,
        memory_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        memory = self.encode(src, src_mask=src_mask)
        dec_out = self.decode(
            trg, memory, trg_mask=trg_mask, memory_mask=memory_mask
        )
        logits = self.fc_out(dec_out)
        return logits
