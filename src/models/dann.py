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


class GraphDualDANN(nn.Module):
    """Dual (marginal + conditional) adversarial domain adaptation.

    Extends GraphDANN with a *conditional* discriminator in the CDAN style
    (Long et al., NeurIPS-18): besides aligning the marginal embedding
    distribution P(z) (the GraphDANN branch), it aligns the forecast-
    conditional distribution P(z | y_hat) by feeding a second discriminator
    the multilinear conditioned feature T = z (x) g(y_hat), where g is a small
    learned summary of the graph-level forecast. This targets the documented
    failure mode of marginal-only DANN in *regression* (de Mathelin et al.,
    2020): aligning P(z) alone leaves the conditional structure unaligned, so
    accuracy does not move. Dual alignment couples both, in the spirit of the
    marginal/conditional duality of Dual Transfer Learning (Long et al.,
    SDM-12).

    forward() returns (forecast, marginal_logits, conditional_logits).
    """

    def __init__(
        self,
        encoder: nn.Module,
        n_cities: int = 3,
        discriminator_hidden: int = 64,
        cond_dim: int = 8,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        if not hasattr(encoder, "embedding_dim"):
            raise ValueError("encoder must expose an 'embedding_dim' attribute")
        d = encoder.embedding_dim
        self.cond_dim = cond_dim
        # marginal branch (identical to GraphDANN)
        self.embed_norm = nn.LayerNorm(d)
        self.marginal_disc = CityDiscriminator(d, discriminator_hidden, n_cities)
        # conditional branch: summarize the forecast, then a multilinear map
        self.forecast_proj = nn.Sequential(nn.Linear(2, cond_dim), nn.ReLU())
        self.cond_norm = nn.LayerNorm(d * cond_dim)
        self.conditional_disc = CityDiscriminator(d * cond_dim, discriminator_hidden, n_cities)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor,
        lambda_: float = 0.0,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        y, z = self.encoder(x, edge_index, edge_weight, return_embedding=True)
        z = self.embed_norm(z)

        # --- marginal alignment: P(z) ---
        m_logits = self.marginal_disc(grad_reverse(z, lambda_))

        # --- conditional alignment: P(z | y_hat) via multilinear map ---
        # Graph-level forecast summary g = MLP([mean_n y_hat, std_n y_hat]).
        g_in = torch.stack([y.mean(dim=1), y.std(dim=1)], dim=-1)   # [B, 2]
        g = self.forecast_proj(g_in)                               # [B, cond_dim]
        # T = z (x) g  (outer product, flattened) — the CDAN conditioned feature.
        T = torch.bmm(z.unsqueeze(2), g.unsqueeze(1)).flatten(1)    # [B, d*cond_dim]
        T = self.cond_norm(T)
        c_logits = self.conditional_disc(grad_reverse(T, lambda_))
        return y, m_logits, c_logits


def lambda_schedule(progress: float, gamma: float = 10.0) -> float:
    """Ganin's classic λ schedule: λ(p) = 2/(1+exp(-γp)) − 1, p ∈ [0, 1]."""
    p = float(progress)
    return float(2.0 / (1.0 + torch.exp(torch.tensor(-gamma * p))).item() - 1.0)
