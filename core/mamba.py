import math
from dataclasses import dataclass
from typing import Optional, Union, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from core.layers import RMSNorm


@dataclass
class MambaConfig:
    d_model: int = 64
    d_state: int = 16          # SSM state expansion factor (N)
    d_conv: int = 4            # Local causal convolution width
    expand: int = 2            # Block expansion factor (E)
    dt_rank: Union[int, str] = "auto"
    dt_min: float = 0.001
    dt_max: float = 0.1
    dt_init: str = "random"
    dt_scale: float = 1.0
    dt_init_floor: float = 1e-4
    vocab_size: int = 1000
    n_layers: int = 2
    bias: bool = False
    conv_bias: bool = True

    def __post_init__(self):
        self.d_inner = int(self.expand * self.d_model)
        if self.dt_rank == "auto":
            self.dt_rank = math.ceil(self.d_model / 16)


def selective_scan(
    u: torch.Tensor,
    delta: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    D: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Pure PyTorch implementation of the Selective Scan algorithm (S6 recurrence).
    """
    bs, seq_len, d_inner = u.shape
    d_state = A.shape[1]

    h = torch.zeros(bs, d_inner, d_state, device=u.device, dtype=u.dtype)
    ys = []

    for t in range(seq_len):
        delta_t = delta[:, t, :, None]   # (B, D, 1)
        u_t = u[:, t, :, None]           # (B, D, 1)
        B_t = B[:, t, None, :]           # (B, 1, N)
        C_t = C[:, t, None, :]           # (B, 1, N)

        A_bar = torch.exp(delta_t * A)   # (B, D, N)
        B_bar = delta_t * B_t            # (B, D, N)

        h = A_bar * h + B_bar * u_t      # (B, D, N)
        y_t = torch.sum(h * C_t, dim=-1) # (B, D)
        ys.append(y_t)

    y = torch.stack(ys, dim=1)           # (B, L, D)

    if D is not None:
        y = y + u * D

    return y


class SelectiveSSM(nn.Module):
    """Selective State Space Module (S6) with input-dependent (Delta, B, C) projections."""

    def __init__(
        self,
        d_inner: int,
        d_state: int = 16,
        dt_rank: int = 16,
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        dt_init: str = "random",
        dt_scale: float = 1.0,
        dt_init_floor: float = 1e-4,
    ):
        super().__init__()
        self.d_inner = d_inner
        self.d_state = d_state
        self.dt_rank = dt_rank

        A = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(d_inner))

        self.x_proj = nn.Linear(d_inner, dt_rank + 2 * d_state, bias=False)
        self.dt_proj = nn.Linear(dt_rank, d_inner, bias=True)

        dt = torch.exp(
            torch.rand(d_inner) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_dbl = self.x_proj(x)
        delta_raw, B, C = torch.split(
            x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1
        )
        delta = F.softplus(self.dt_proj(delta_raw))
        A = -torch.exp(self.A_log.float())
        return selective_scan(x, delta, A, B, C, self.D)


class MambaBlock(nn.Module):
    """Standard Mamba Block with 1D Causal Conv, Selective SSM, and SiLU Gating."""

    def __init__(self, config: MambaConfig):
        super().__init__()
        self.config = config
        d_model = config.d_model
        d_inner = config.d_inner
        d_conv = config.d_conv

        self.in_proj = nn.Linear(d_model, 2 * d_inner, bias=config.bias)
        self.conv1d = nn.Conv1d(
            in_channels=d_inner,
            out_channels=d_inner,
            kernel_size=d_conv,
            groups=d_inner,
            padding=d_conv - 1,
            bias=config.conv_bias,
        )
        self.ssm = SelectiveSSM(
            d_inner=d_inner,
            d_state=config.d_state,
            dt_rank=config.dt_rank,
            dt_min=config.dt_min,
            dt_max=config.dt_max,
            dt_init=config.dt_init,
            dt_scale=config.dt_scale,
            dt_init_floor=config.dt_init_floor,
        )
        self.out_proj = nn.Linear(d_inner, d_model, bias=config.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, _ = x.shape
        xz = self.in_proj(x)
        x_branch, z_branch = xz.chunk(2, dim=-1)

        x_conv = self.conv1d(x_branch.transpose(1, 2))[:, :, :L].transpose(1, 2)
        x_conv = F.silu(x_conv)

        y_ssm = self.ssm(x_conv)
        y = y_ssm * F.silu(z_branch)
        return self.out_proj(y)


class ResidualBlock(nn.Module):
    """Mamba block wrapped with Pre-RMSNorm and residual connection."""

    def __init__(self, config: MambaConfig):
        super().__init__()
        self.norm = RMSNorm(config.d_model)
        self.block = MambaBlock(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(self.norm(x))


class MambaLM(nn.Module):
    """End-to-End Mamba Causal Language Model."""

    def __init__(self, config: MambaConfig):
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.layers = nn.ModuleList([ResidualBlock(config) for _ in range(config.n_layers)])
        self.norm_f = RMSNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.lm_head.weight = self.embedding.weight

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.embedding(input_ids)
        for layer in self.layers:
            x = layer(x)
        x = self.norm_f(x)
        return self.lm_head(x)

    @torch.no_grad()
    def generate(
        self,
        prompt_ids: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
    ) -> torch.Tensor:
        for _ in range(max_new_tokens):
            logits = self.forward(prompt_ids)
            next_token_logits = logits[:, -1, :] / max(temperature, 1e-5)
            if top_k is not None:
                v, _ = torch.topk(next_token_logits, min(top_k, next_token_logits.size(-1)))
                next_token_logits[next_token_logits < v[:, [-1]]] = -float("Inf")
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            prompt_ids = torch.cat([prompt_ids, next_token], dim=1)
        return prompt_ids
