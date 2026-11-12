from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


def info_nce_loss(
    queries: torch.Tensor,
    keys: torch.Tensor,
    temperature: float = 0.1,
    normalize: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Computes the InfoNCE contrastive loss over aligned positive pairs with in-batch negatives.

    Args:
        queries: Predicted representations of shape (N, D)
        keys:    Target representations of shape (N, D), where keys[i] is the positive match for queries[i]
        temperature: Softmax temperature scaling factor
        normalize: If True, L2-normalize queries and keys before computing dot products
    Returns:
        Tuple[Tensor, Tensor]: (scalar InfoNCE loss, classification accuracy)
    """
    # TODO:
    if normalize:
        queries = F.normalize(queries, dim=-1)
        keys = F.normalize(keys, dim=-1)

    logits = torch.matmul(queries, keys.T) / temperature

    labels = torch.arange(queries.shape[0], device=queries.device)
    loss = F.cross_entropy(logits, labels)
    acc = (logits.argmax(dim=-1) == labels).float().mean()
    return loss, acc


class CPCEncoder(nn.Module):
    """
    Non-linear encoder g_enc mapping raw observations x_t to compact latents z_t.
    """

    def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class CPCModel(nn.Module):
    """
    Contrastive Predictive Coding (Oord et al., 2018).
    Predicts future latent representations k steps ahead using an autoregressive context model
    and contrasts them against in-batch and in-sequence negative distractors via InfoNCE.
    """

    def __init__(
        self,
        input_dim: int,
        latent_dim: int = 64,
        context_dim: int = 64,
        prediction_steps: int = 4,
        temperature: float = 0.1,
        normalize: bool = True,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.context_dim = context_dim
        self.prediction_steps = prediction_steps
        self.temperature = temperature
        self.normalize = normalize

        self.encoder = CPCEncoder(input_dim, hidden_dim=latent_dim * 2, latent_dim=latent_dim)
        self.ar_gru = nn.GRU(latent_dim, context_dim, batch_first=True)

        # Linear prediction heads W_k for each future horizon k = 1..prediction_steps
        self.predictors = nn.ModuleList(
            [nn.Linear(context_dim, latent_dim, bias=False) for _ in range(prediction_steps)]
        )

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extracts latent sequence z and causal context sequence c.
        Args:
            x: Input sequence of shape (B, L, input_dim)
        Returns:
            Tuple[Tensor, Tensor]: z of shape (B, L, latent_dim), c of shape (B, L, context_dim)
        """
        # TODO:
        z = self.encoder(x)
        c, _ = self.ar_gru(z)
        return z, c

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Computes multi-horizon CPC InfoNCE loss and accuracy.
        Args:
            x: Input sequence of shape (B, L, input_dim)
        Returns:
            Tuple[Tensor, Tensor, Tensor, Tensor]: (loss, accuracy, z, c)
        """
        # TODO:
        B, L, _ = x.shape
        z, c = self.encode(x)

        total_loss = 0.0
        total_acc = 0.0
        steps_computed = 0

        for k in range(1, self.prediction_steps + 1):
            if k >= L:
                break

            c_t = c[:, :-k, :].reshape(-1, self.context_dim)
            z_future = z[:, k:, :].reshape(-1, self.latent_dim)

            z_pred = self.predictors[k - 1](c_t)

            loss_k, acc_k = info_nce_loss(
                z_pred,
                z_future,
                temperature=self.temperature,
                normalize=self.normalize,
            )
            total_loss = total_loss + loss_k
            total_acc = total_acc + acc_k
            steps_computed += 1

        avg_loss = total_loss / max(steps_computed, 1)
        avg_acc = total_acc / max(steps_computed, 1)
        return avg_loss, avg_acc, z, c
