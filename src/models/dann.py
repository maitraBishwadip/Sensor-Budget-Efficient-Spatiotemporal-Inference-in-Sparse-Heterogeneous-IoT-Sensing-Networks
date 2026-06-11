"""Graph-DANN wrapper — adversarial domain adaptation around an ST-GNN.

Composes the ST-GNN encoder of choice (STGNN_GAT or STGNN_SAGE) with a
Gradient-Reversal Layer followed by a 3-class city discriminator over the
size-invariant graph-pooled embedding produced by the encoder.

Loss during training:
    L = L_forecast(y, ŷ) − λ · L_city(d_city, d̂)
The negative sign is implemented by the GRL on the forward path of the
discriminator branch (Ganin & Lempitsky, ICML 2015).
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
from torch.autograd import Function


class _GradReverseFn(Function):
    """The classic gradient-reversal autograd function."""

    @staticmethod
    def forward(ctx, x: torch.Tensor, lambda_: float) -> torch.Tensor:
        ctx.lambda_ = float(lambda_)
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[torch.Tensor, None]:
        return -ctx.lambda_ * grad_output, None


def grad_reverse(x: torch.Tensor, lambda_: float) -> torch.Tensor:
    return _GradReverseFn.apply(x, lambda_)


class CityDiscriminator(nn.Module):
    """3-class MLP over the pooled graph embedding."""

    def __init__(self, in_dim: int, hidden_dim: int = 64, n_cities: int = 3, dropout: float = 0.2) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_cities),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class GraphDANN(nn.Module):
    """Wraps an ST-GNN encoder with the adversarial city head.

    The encoder must accept (x, edge_index, edge_weight, return_embedding=True)
    and return a (forecast, embedding) tuple — both STGNN_GAT and STGNN_SAGE
    satisfy this contract.

    A LayerNorm is applied to the graph-pooled embedding before the GRL.
    The raw mean+max pool scale depends on |V| (Delhi's 40-node max-pool sits
    on a different magnitude than Guwahati's 4-node max-pool), which trivializes
    the city discriminator and produces destructively large reversed gradients
    back into the encoder. Normalizing per-feature decouples the discriminator's
    job from |V| and tracks the modern reimplementation practice (e.g.
    GraphNorm, Cai et al. ICML-21).
    """

    def __init__(
        self,
        encoder: nn.Module,
        n_cities: int = 3,
        discriminator_hidden: int = 64,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        if not hasattr(encoder, "embedding_dim"):
            raise ValueError("encoder must expose an 'embedding_dim' attribute")
        self.embed_norm = nn.LayerNorm(encoder.embedding_dim)
        self.discriminator = CityDiscriminator(
            in_dim=encoder.embedding_dim,
            hidden_dim=discriminator_hidden,
            n_cities=n_cities,
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor,
        lambda_: float = 0.0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        y, z = self.encoder(x, edge_index, edge_weight, return_embedding=True)
        z = self.embed_norm(z)
        z_rev = grad_reverse(z, lambda_)
        city_logits = self.discriminator(z_rev)
        return y, city_logits


def lambda_schedule(progress: float, gamma: float = 10.0) -> float:
    """Ganin's classic λ schedule: λ(p) = 2/(1+exp(-γp)) − 1, p ∈ [0, 1]."""
    p = float(progress)
    return float(2.0 / (1.0 + torch.exp(torch.tensor(-gamma * p))).item() - 1.0)
