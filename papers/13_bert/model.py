import math
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class BERTEmbedding(nn.Module):
    """
    BERT Embedding layer combining Token, Position, and Segment embeddings:
    Embedding = LayerNorm(TokenEmb + PosEmb + SegmentEmb) + Dropout

    Args:
        vocab_size               (int): Vocabulary size.
        hidden_size              (int): Hidden embedding dimension.
        max_position_embeddings  (int): Maximum sequence length for learned positions. Default: 512.
        type_vocab_size          (int): Number of segment types (usually 2: sentence A & B). Default: 2.
        hidden_dropout_prob    (float): Dropout probability. Default: 0.1.
        layer_norm_eps         (float): Epsilon for LayerNorm. Default: 1e-12.
    """

    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        max_position_embeddings: int = 512,
        type_vocab_size: int = 2,
        hidden_dropout_prob: float = 0.1,
        layer_norm_eps: float = 1e-12,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size

        self.word_embeddings = nn.Embedding(vocab_size, hidden_size)
        self.position_embeddings = nn.Embedding(max_position_embeddings, hidden_size)
        self.token_type_embeddings = nn.Embedding(type_vocab_size, hidden_size)

        self.LayerNorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.dropout = nn.Dropout(hidden_dropout_prob)

    def forward(
        self,
        input_ids: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass for BERT embeddings.

        Args:
            input_ids      (Tensor): Token indices of shape (B, T).
            token_type_ids (Tensor, optional): Segment indices of shape (B, T).
            position_ids   (Tensor, optional): Position indices of shape (B, T).

        Returns:
            Tensor: Normalized token embeddings of shape (B, T, hidden_size).
        """
        ################################################################
        # TODO: Implement BERTEmbedding forward pass
        seq_length = input_ids.size(1)

        if position_ids is None:
            position_ids = torch.arange(seq_length, dtype = torch.long, device = input_ids.device).unsqueeze(0)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        words_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = words_embeddings + position_embeddings + token_type_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings
        ################################################################


class BERTSelfAttention(nn.Module):
    """
    BERT Multi-Head Self-Attention.
    Unlike standard autoregressive decoders, this attention is fully bidirectional.

    Args:
        hidden_size                 (int): Model dimension.
        num_attention_heads         (int): Number of attention heads.
        attention_probs_dropout_prob (float): Attention dropout. Default: 0.1.
    """

    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        attention_probs_dropout_prob: float = 0.1,
    ):
        super().__init__()
        assert hidden_size % num_attention_heads == 0, (
            f"hidden_size ({hidden_size}) must be divisible by num_attention_heads ({num_attention_heads})"
        )
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = hidden_size // num_attention_heads
        self.all_head_size = hidden_size

        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(attention_probs_dropout_prob)

    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        # Reshapes (B, T, H) -> (B, num_heads, T, head_dim)
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            hidden_states  (Tensor): Shape (B, T, hidden_size).
            attention_mask (Tensor, optional): Additive mask where masked tokens are -10000.0.

        Returns:
            Tuple[Tensor, Tensor]: (context_layer of shape (B, T, hidden_size), attention_probs)
        """
        ################################################################
        # TODO: Implement BERTSelfAttention forward pass
        query_layer = self.transpose_for_scores(self.query(hidden_states))
        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2)) / math.sqrt(self.attention_head_size)

        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask

        attention_probs = F.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        return context_layer, attention_probs
        ################################################################


class BERTSelfOutput(nn.Module):
    """
    Residual projection and LayerNorm after self-attention:
    x = LayerNorm(x + Dropout(Dense(hidden_states)))
    """

    def __init__(
        self,
        hidden_size: int,
        hidden_dropout_prob: float = 0.1,
        layer_norm_eps: float = 1e-12,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.dropout = nn.Dropout(hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states (Tensor): Attention output of shape (B, T, hidden_size).
            input_tensor  (Tensor): Residual input of shape (B, T, hidden_size).

        Returns:
            Tensor: Output representations of shape (B, T, hidden_size).
        """
        ################################################################
        # TODO: Implement BERTSelfOutput forward pass
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states
        ################################################################


class BERTAttention(nn.Module):
    """
    BERT Attention block combining Self-Attention and Self-Output projection.
    """

    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        attention_probs_dropout_prob: float = 0.1,
        hidden_dropout_prob: float = 0.1,
        layer_norm_eps: float = 1e-12,
    ):
        super().__init__()
        self.self = BERTSelfAttention(
            hidden_size, num_attention_heads, attention_probs_dropout_prob
        )
        self.output = BERTSelfOutput(
            hidden_size, hidden_dropout_prob, layer_norm_eps
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            hidden_states  (Tensor): Input representations of shape (B, T, hidden_size).
            attention_mask (Tensor, optional): Additive attention mask.

        Returns:
            Tuple[Tensor, Tensor]: (attention_output, attention_probs)
        """
        ################################################################
        # TODO: Implement BERTAttention forward pass
        self_output, attention_probs = self.self(hidden_states, attention_mask)
        attention_output = self.output(self_output, hidden_states)
        return attention_output, attention_probs
        ################################################################


class BERTIntermediate(nn.Module):
    """
    BERT Intermediate feed-forward projection with GELU activation:
    hidden_size -> intermediate_size
    """

    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size

        self.dense = nn.Linear(hidden_size, intermediate_size)
        self.intermediate_act_fn = nn.GELU()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states (Tensor): Shape (B, T, hidden_size).

        Returns:
            Tensor: Shape (B, T, intermediate_size).
        """
        ################################################################
        # TODO: Implement BERTIntermediate forward pass
        return self.intermediate_act_fn(self.dense(hidden_states))
        ################################################################


class BERTOutput(nn.Module):
    """
    BERT Output feed-forward projection back to hidden_size with residual + LayerNorm:
    LayerNorm(input_tensor + Dropout(Dense(hidden_states)))
    """

    def __init__(
        self,
        intermediate_size: int,
        hidden_size: int,
        hidden_dropout_prob: float = 0.1,
        layer_norm_eps: float = 1e-12,
    ):
        super().__init__()
        self.intermediate_size = intermediate_size
        self.hidden_size = hidden_size

        self.dense = nn.Linear(intermediate_size, hidden_size)
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.dropout = nn.Dropout(hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states (Tensor): Intermediate output of shape (B, T, intermediate_size).
            input_tensor  (Tensor): Attention output of shape (B, T, hidden_size).

        Returns:
            Tensor: Shape (B, T, hidden_size).
        """
        ################################################################
        # TODO: Implement BERTOutput forward pass
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states
        ################################################################


class BERTLayer(nn.Module):
    """
    Full BERT Transformer Layer:
    Attention Block -> Feed-Forward Block
    """

    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        intermediate_size: int,
        hidden_dropout_prob: float = 0.1,
        attention_probs_dropout_prob: float = 0.1,
        layer_norm_eps: float = 1e-12,
    ):
        super().__init__()
        self.attention = BERTAttention(
            hidden_size,
            num_attention_heads,
            attention_probs_dropout_prob,
            hidden_dropout_prob,
            layer_norm_eps,
        )
        self.intermediate = BERTIntermediate(hidden_size, intermediate_size)
        self.output = BERTOutput(
            intermediate_size, hidden_size, hidden_dropout_prob, layer_norm_eps
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            hidden_states  (Tensor): Input representations of shape (B, T, hidden_size).
            attention_mask (Tensor, optional): Additive mask.

        Returns:
            Tensor: Output representations of shape (B, T, hidden_size).
        """
        ################################################################
        # TODO: Implement BERTLayer forward pass
        attention_output, _ = self.attention(hidden_states, attention_mask)
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output
        ################################################################


class BERTEncoder(nn.Module):
    """
    Stack of N BERT Layers.
    """

    def __init__(
        self,
        num_hidden_layers: int,
        hidden_size: int,
        num_attention_heads: int,
        intermediate_size: int,
        hidden_dropout_prob: float = 0.1,
        attention_probs_dropout_prob: float = 0.1,
        layer_norm_eps: float = 1e-12,
    ):
        super().__init__()
        self.num_hidden_layers = num_hidden_layers
        self.layer = nn.ModuleList(
            [
                BERTLayer(
                    hidden_size,
                    num_attention_heads,
                    intermediate_size,
                    hidden_dropout_prob,
                    attention_probs_dropout_prob,
                    layer_norm_eps,
                )
                for _ in range(num_hidden_layers)
            ]
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            hidden_states  (Tensor): Token embeddings of shape (B, T, hidden_size).
            attention_mask (Tensor, optional): Additive mask.

        Returns:
            Tensor: Final hidden states of shape (B, T, hidden_size).
        """
        ################################################################
        # TODO: Implement BERTEncoder forward pass
        for layer_module in self.layer:
            hidden_states = layer_module(hidden_states, attention_mask)
        return hidden_states
        ################################################################


class BERTPooler(nn.Module):
    """
    Pools the [CLS] representation (first token) through a Linear projection and Tanh activation.
    """

    def __init__(self, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states (Tensor): Sequence hidden states of shape (B, T, hidden_size).

        Returns:
            Tensor: Pooled [CLS] output of shape (B, hidden_size).
        """
        ################################################################
        # TODO: Implement BERTPooler forward pass
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output
        ################################################################


class BERTModel(nn.Module):
    """
    Core BERT Transformer Model:
    Embeddings -> Encoder Stack -> [CLS] Pooler

    Args:
        vocab_size                  (int): Vocabulary size.
        hidden_size                 (int): Hidden embedding dimension. Default: 256.
        num_hidden_layers           (int): Number of encoder layers. Default: 4.
        num_attention_heads         (int): Number of attention heads. Default: 4.
        intermediate_size           (int): Intermediate FFN dimension. Default: 1024.
        max_position_embeddings     (int): Maximum sequence length. Default: 512.
        type_vocab_size             (int): Segment vocabulary size. Default: 2.
        hidden_dropout_prob       (float): Dropout probability. Default: 0.1.
        attention_probs_dropout_prob (float): Attention dropout. Default: 0.1.
        layer_norm_eps            (float): LayerNorm epsilon. Default: 1e-12.
    """

    def __init__(
        self,
        vocab_size: int,
        hidden_size: int = 256,
        num_hidden_layers: int = 4,
        num_attention_heads: int = 4,
        intermediate_size: int = 1024,
        max_position_embeddings: int = 512,
        type_vocab_size: int = 2,
        hidden_dropout_prob: float = 0.1,
        attention_probs_dropout_prob: float = 0.1,
        layer_norm_eps: float = 1e-12,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size

        self.embeddings = BERTEmbedding(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            max_position_embeddings=max_position_embeddings,
            type_vocab_size=type_vocab_size,
            hidden_dropout_prob=hidden_dropout_prob,
            layer_norm_eps=layer_norm_eps,
        )
        self.encoder = BERTEncoder(
            num_hidden_layers=num_hidden_layers,
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            intermediate_size=intermediate_size,
            hidden_dropout_prob=hidden_dropout_prob,
            attention_probs_dropout_prob=attention_probs_dropout_prob,
            layer_norm_eps=layer_norm_eps,
        )
        self.pooler = BERTPooler(hidden_size=hidden_size)

    def get_extended_attention_mask(
        self, attention_mask: torch.Tensor, input_shape: Tuple[int, ...]
    ) -> torch.Tensor:
        """
        Converts 2D (B, T) mask (where 1 is valid, 0 is masked) into 4D (B, 1, 1, T) additive mask
        with -10000.0 for masked positions.
        """
        if attention_mask.dim() == 2:
            extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        elif attention_mask.dim() == 3:
            extended_attention_mask = attention_mask.unsqueeze(1)
        else:
            extended_attention_mask = attention_mask

        extended_attention_mask = extended_attention_mask.to(dtype=torch.float32)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        return extended_attention_mask

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            input_ids      (Tensor): Token indices of shape (B, T).
            attention_mask (Tensor, optional): Binary mask of shape (B, T) where 1 is keep, 0 is mask.
            token_type_ids (Tensor, optional): Segment indices of shape (B, T).
            position_ids   (Tensor, optional): Position indices of shape (B, T).

        Returns:
            Tuple[Tensor, Tensor]:
                - sequence_output: Contextual token representations of shape (B, T, hidden_size).
                - pooled_output: [CLS] representation of shape (B, hidden_size).
        """
        ################################################################
        # TODO: Implement BERTModel forward pass
        embedding_output = self.embeddings(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
        )

        extended_attention_mask = None
        if attention_mask is not None:
            extended_attention_mask = self.get_extended_attention_mask(
                attention_mask, input_ids.shape
            )

        sequence_output = self.encoder(
            embedding_output, attention_mask=extended_attention_mask
        )
        pooled_output = self.pooler(sequence_output)
        return sequence_output, pooled_output
        ################################################################


class BERTLMPredictionHead(nn.Module):
    """
    Masked Language Model prediction head:
    hidden_states -> Linear -> GELU -> LayerNorm -> Linear(vocab_size)
    """

    def __init__(self, hidden_size: int, vocab_size: int, layer_norm_eps: float = 1e-12):
        super().__init__()
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size

        self.dense = nn.Linear(hidden_size, hidden_size)
        self.transform_act_fn = nn.GELU()
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.decoder = nn.Linear(hidden_size, vocab_size, bias=True)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states (Tensor): Sequence representations of shape (B, T, hidden_size).

        Returns:
            Tensor: Vocabulary logits of shape (B, T, vocab_size).
        """
        ################################################################
        # TODO: Implement BERTLMPredictionHead forward pass
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        hidden_states = self.decoder(hidden_states)
        return hidden_states
        ################################################################


class BERTForPreTraining(nn.Module):
    """
    BERT Model with both pre-training heads:
    1. Masked Language Modeling (MLM): Predicts token identity for all positions.
    2. Next Sentence Prediction (NSP): Predicts binary sentence adjacency from [CLS].

    Args:
        vocab_size          (int): Vocabulary size.
        hidden_size         (int): Hidden embedding dimension. Default: 256.
        num_hidden_layers   (int): Number of encoder layers. Default: 4.
        num_attention_heads (int): Number of attention heads. Default: 4.
        intermediate_size   (int): Intermediate FFN dimension. Default: 1024.
    """

    def __init__(
        self,
        vocab_size: int,
        hidden_size: int = 256,
        num_hidden_layers: int = 4,
        num_attention_heads: int = 4,
        intermediate_size: int = 1024,
        **kwargs,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size

        self.bert = BERTModel(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            intermediate_size=intermediate_size,
            **kwargs,
        )
        self.cls_mlm = BERTLMPredictionHead(hidden_size, vocab_size)
        self.cls_nsp = nn.Linear(hidden_size, 2)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for pre-training.

        Args:
            input_ids      (Tensor): Token indices of shape (B, T).
            attention_mask (Tensor, optional): Binary mask of shape (B, T).
            token_type_ids (Tensor, optional): Segment indices of shape (B, T).

        Returns:
            Tuple[Tensor, Tensor]:
                - prediction_scores: Vocabulary logits of shape (B, T, vocab_size).
                - seq_relationship_score: Binary NSP logits of shape (B, 2).
        """
        ################################################################
        # TODO: Implement BERTForPreTraining forward pass
        sequence_output, pooled_output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        prediction_scores = self.cls_mlm(sequence_output)
        seq_relationship_score = self.cls_nsp(pooled_output)
        return prediction_scores, seq_relationship_score
        ################################################################
