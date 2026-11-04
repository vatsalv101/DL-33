import math
from typing import List, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F


class GPT2Attention(nn.Module):
    """
    Causal Multi-Head Self-Attention with Key-Value (KV) Caching.

    Args:
        d_model       (int): Hidden embedding dimension.
        num_heads     (int): Number of attention heads. Must divide d_model.
        max_positions (int): Maximum sequence length for the causal mask buffer. Default: 1024.
        dropout     (float): Dropout probability. Default: 0.1.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        max_positions: int = 1024,
        dropout: float = 0.1,
    ):
        super().__init__()
        assert d_model % num_heads == 0, f"d_model ({d_model}) must be divisible by num_heads ({num_heads})"
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        # Combined QKV projection
        self.c_attn = nn.Linear(d_model, 3 * d_model)
        # Output projection
        self.c_proj = nn.Linear(d_model, d_model)

        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

        # Causal mask buffer: lower triangular
        bias = torch.tril(torch.ones(max_positions, max_positions, dtype=torch.bool))
        self.register_buffer("bias", bias.view(1, 1, max_positions, max_positions), persistent=False)

    def forward(
        self,
        x: torch.Tensor,
        layer_past: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Forward pass with optional KV caching.

        Args:
            x          (Tensor): Input representations of shape (B, T, d_model).
            layer_past (Tuple[Tensor, Tensor], optional): Cached (key, value) tensors from previous steps,
                                                         each of shape (B, num_heads, past_len, head_dim).
            use_cache  (bool): Whether to return the updated (key, value) cache.

        Returns:
            Tuple[Tensor, Optional[Tuple[Tensor, Tensor]]]:
                - output: Projected attention output of shape (B, T, d_model).
                - present: Updated (key, value) cache if use_cache is True, else None.
        """
        ################################################################
        # TODO: Implement GPT2Attention forward pass
        B, T, C = x.size()

        # 1. Project QKV using self.c_attn and chunk into (q, k, v)
        qkv = self.c_attn(x)
        q, k, v = qkv.chunk(3, dim=-1)

        # 2. Reshape into (B, num_heads, T, head_dim)
        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        # 3. If layer_past is provided, concatenate with past_k and past_v along seq dimension
        past_len = 0
        if layer_past is not None:
            past_k, past_v = layer_past
            past_len = past_k.size(-2)
            k = torch.cat((past_k, k), dim=-2)
            v = torch.cat((past_v, v), dim=-2)

        # 4. Form present = (k, v) if use_cache else None
        present = (k, v) if use_cache else None
        total_len = k.size(-2)

        # 5. Compute scaled dot-product attention with causal mask slice self.bias[:, :, past_len:past_len+T, :total_len]
        att = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.head_dim)
        causal_mask = self.bias[:, :, past_len : past_len + T, :total_len]
        att = att.masked_fill(~causal_mask, -1e4)

        # 6. Apply dropout, multiply with v, reshape back, and project with self.c_proj
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)

        y = torch.matmul(att, v)  
        y = y.transpose(1, 2).contiguous().view(B, T, C) 

        y = self.resid_dropout(self.c_proj(y))
        return y, present
        ################################################################


class GPT2MLP(nn.Module):
    """
    Position-wise Feed-Forward Network with GELU activation:
    Linear(d_model -> 4 * d_model) -> GELU -> Linear(4 * d_model -> d_model) -> Dropout

    Args:
        d_model (int): Hidden embedding dimension.
        dropout (float): Dropout probability. Default: 0.1.
    """

    def __init__(self, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.c_fc = nn.Linear(d_model, 4 * d_model)
        self.c_proj = nn.Linear(4 * d_model, d_model)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (Tensor): Input representations of shape (B, T, d_model).

        Returns:
            Tensor: Output representations of shape (B, T, d_model).
        """
        ################################################################
        # TODO: Implement GPT2MLP forward pass
        h = self.act(self.c_fc(x))
        h = self.c_proj(h)
        h = self.dropout(h)
        return h
        ################################################################


class GPT2Block(nn.Module):
    """
    GPT-2 Transformer Decoder Layer with Pre-Layer Normalization:
    x -> x + Attention(LayerNorm(x))
    x -> x + MLP(LayerNorm(x))

    Args:
        d_model        (int): Hidden embedding dimension.
        num_heads      (int): Number of attention heads.
        max_positions  (int): Maximum sequence length. Default: 1024.
        dropout      (float): Dropout probability. Default: 0.1.
        layer_norm_eps (float): Epsilon for LayerNorm. Default: 1e-5.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        max_positions: int = 1024,
        dropout: float = 0.1,
        layer_norm_eps: float = 1e-5,
    ):
        super().__init__()
        self.ln_1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.attn = GPT2Attention(
            d_model=d_model,
            num_heads=num_heads,
            max_positions=max_positions,
            dropout=dropout,
        )
        self.ln_2 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.mlp = GPT2MLP(d_model=d_model, dropout=dropout)

    def forward(
        self,
        x: torch.Tensor,
        layer_past: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Args:
            x          (Tensor): Input representations of shape (B, T, d_model).
            layer_past (Tuple[Tensor, Tensor], optional): Cached KV tensors.
            use_cache  (bool): Whether to return updated cache.

        Returns:
            Tuple[Tensor, Optional[Tuple[Tensor, Tensor]]]: (output, present_kv)
        """
        ################################################################
        # TODO: Implement GPT2Block forward pass
        # 1. Pre-LN Self-Attention with residual connection
        attn_out, present = self.attn(self.ln_1(x), layer_past=layer_past, use_cache=use_cache)
        x = x + attn_out

        # 2. Pre-LN MLP with residual connection
        x = x + self.mlp(self.ln_2(x))
        return x, present
        ################################################################


class GPT2Model(nn.Module):
    """
    Core GPT-2 Backbone Transformer:
    Token + Position Embeddings -> Stack of Pre-LN GPT2Blocks -> Final LayerNorm

    Args:
        vocab_size     (int): Size of vocabulary. Default: 50257.
        max_positions  (int): Maximum sequence length. Default: 1024.
        d_model        (int): Embedding dimension. Default: 768.
        num_layers     (int): Number of Transformer decoder layers. Default: 12.
        num_heads      (int): Number of attention heads. Default: 12.
        dropout      (float): Dropout probability. Default: 0.1.
        layer_norm_eps (float): LayerNorm epsilon. Default: 1e-5.
    """

    def __init__(
        self,
        vocab_size: int = 50257,
        max_positions: int = 1024,
        d_model: int = 768,
        num_layers: int = 12,
        num_heads: int = 12,
        dropout: float = 0.1,
        layer_norm_eps: float = 1e-5,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_positions = max_positions
        self.d_model = d_model
        self.num_layers = num_layers

        self.wte = nn.Embedding(vocab_size, d_model)
        self.wpe = nn.Embedding(max_positions, d_model)
        self.drop = nn.Dropout(dropout)

        self.h = nn.ModuleList(
            [
                GPT2Block(
                    d_model=d_model,
                    num_heads=num_heads,
                    max_positions=max_positions,
                    dropout=dropout,
                    layer_norm_eps=layer_norm_eps,
                )
                for _ in range(num_layers)
            ]
        )
        self.ln_f = nn.LayerNorm(d_model, eps=layer_norm_eps)

    def forward(
        self,
        input_ids: torch.Tensor,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        position_ids: Optional[torch.Tensor] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[List[Tuple[torch.Tensor, torch.Tensor]]]]:
        """
        Args:
            input_ids       (Tensor): Token indices of shape (B, T).
            past_key_values (List[Tuple[Tensor, Tensor]], optional): Cached KV tuples per layer.
            position_ids    (Tensor, optional): Custom position indices.
            use_cache       (bool): Whether to return updated KV cache.

        Returns:
            Tuple[Tensor, Optional[List[Tuple[Tensor, Tensor]]]]:
                - hidden_states: Output representations of shape (B, T, d_model).
                - presents: Updated KV caches for all layers.
        """
        ################################################################
        # TODO: Implement GPT2Model forward pass
        B, T = input_ids.size()
        # 1. Determine past_len from past_key_values if present
        past_len = 0
        if past_key_values is not None:
            past_len = past_key_values[0][0].size(-2)

        # 2. Build position_ids if None (shape (1, T) starting at past_len)
        if position_ids is None:
            position_ids = torch.arange(
                past_len, past_len + T, dtype=torch.long, device=input_ids.device
            ).unsqueeze(0)

        # 3. Sum token and position embeddings, apply self.drop
        token_embeds = self.wte(input_ids)
        pos_embeds = self.wpe(position_ids)
        hidden_states = self.drop(token_embeds + pos_embeds)

        # 4. Iterate over self.h blocks passing layer_past and collecting presents if use_cache
        presents = [] if use_cache else None

        for i, block in enumerate(self.h):
            layer_past = past_key_values[i] if past_key_values is not None else None
            hidden_states, present = block(
                hidden_states, layer_past=layer_past, use_cache=use_cache
            )
            if use_cache:
                presents.append(present)
        # 5. Apply self.ln_f final LayerNorm
        hidden_states = self.ln_f(hidden_states)
        return hidden_states, presents
        ################################################################


class GPT2LMHeadModel(nn.Module):
    """
    GPT-2 Language Model with language modeling head (tied weights with token embeddings).

    Args:
        vocab_size     (int): Size of vocabulary. Default: 50257.
        max_positions  (int): Maximum sequence length. Default: 1024.
        d_model        (int): Embedding dimension. Default: 768.
        num_layers     (int): Number of Transformer decoder layers. Default: 12.
        num_heads      (int): Number of attention heads. Default: 12.
        dropout      (float): Dropout probability. Default: 0.1.
        layer_norm_eps (float): LayerNorm epsilon. Default: 1e-5.
    """

    def __init__(
        self,
        vocab_size: int = 50257,
        max_positions: int = 1024,
        d_model: int = 768,
        num_layers: int = 12,
        num_heads: int = 12,
        dropout: float = 0.1,
        layer_norm_eps: float = 1e-5,
    ):
        super().__init__()
        self.transformer = GPT2Model(
            vocab_size=vocab_size,
            max_positions=max_positions,
            d_model=d_model,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout,
            layer_norm_eps=layer_norm_eps,
        )
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        # Tie weights: lm_head.weight = transformer.wte.weight
        self.lm_head.weight = self.transformer.wte.weight

        self._init_weights()

    def _init_weights(self):
        """
        Initialize weights with standard normal (std=0.02) and scale residual projections
        by 1 / sqrt(2 * num_layers) as specified in Section 2.3 of Radford et al.
        """
        for name, p in self.named_parameters():
            if "weight" in name and not "ln" in name:
                nn.init.normal_(p, mean=0.0, std=0.02)
                if "c_proj.weight" in name:
                    p.data.mul_(1.0 / math.sqrt(2.0 * self.transformer.num_layers))
            elif "bias" in name:
                nn.init.zeros_(p)

    def forward(
        self,
        input_ids: torch.Tensor,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        position_ids: Optional[torch.Tensor] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[List[Tuple[torch.Tensor, torch.Tensor]]]]:
        """
        Args:
            input_ids       (Tensor): Token indices of shape (B, T).
            past_key_values (List[Tuple[Tensor, Tensor]], optional): Cached KV tensors.
            position_ids    (Tensor, optional): Custom positions.
            use_cache       (bool): Whether to return updated KV cache.

        Returns:
            Tuple[Tensor, Optional[List[Tuple[Tensor, Tensor]]]]:
                - logits: Unnormalized vocabulary logits of shape (B, T, vocab_size).
                - presents: Updated KV caches if use_cache is True.
        """
        ################################################################
        # TODO: Implement GPT2LMHeadModel forward pass
        # 1. Forward through self.transformer
        hidden_states, presents = self.transformer(
            input_ids=input_ids,
            past_key_values=past_key_values,
            position_ids=position_ids,
            use_cache=use_cache,
        )
        # 2. Project hidden states through self.lm_head to vocabulary logits
        logits = self.lm_head(hidden_states)
        return logits, presents
        ################################################################

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        eos_token_id: Optional[int] = None,
        use_cache: bool = True,
    ) -> torch.Tensor:
        """
        Autoregressive generation with optional KV-cache.

        Args:
            input_ids      (Tensor): Prompt token IDs of shape (B, T).
            max_new_tokens    (int): Number of new tokens to generate.
            temperature     (float): Sampling temperature (> 0).
            top_k           (int, optional): Top-k sampling threshold.
            eos_token_id    (int, optional): End-of-sequence token ID.
            use_cache        (bool): Whether to use KV caching for fast generation.

        Returns:
            Tensor: Generated token IDs of shape (B, T + generated_tokens).
        """
        ################################################################
        # TODO: Implement generate
        self.eval()
        past_key_values = None

        # 1. Loop max_new_tokens times
        for _ in range(max_new_tokens):
            
            # 2. If use_cache and past_key_values exists, feed only input_ids[:, -1:]
            if use_cache and past_key_values is not None:
                current_input = input_ids[:, -1:]
            else:
                current_input = input_ids

            # 3. Call self.forward to get next token logits and updated past_key_values
            logits, past_key_values = self.forward(
                current_input,
                past_key_values=past_key_values if use_cache else None,
                use_cache=use_cache,
            )
            # 4. Scale logits by temperature and apply optional top_k filtering
            next_token_logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(next_token_logits, min(top_k, next_token_logits.size(-1)))
                next_token_logits[next_token_logits < v[:, [-1]]] = -float("Inf")

            # 5. Sample next token with torch.multinomial and append to input_ids
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            input_ids = torch.cat((input_ids, next_token), dim=1)

            if eos_token_id is not None and (next_token == eos_token_id).all():
                break

        return input_ids
        ################################################################
