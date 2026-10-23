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
