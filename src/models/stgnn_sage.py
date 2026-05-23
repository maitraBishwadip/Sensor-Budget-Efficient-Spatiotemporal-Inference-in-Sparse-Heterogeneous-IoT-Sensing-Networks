"""Inductive Spatio-Temporal GNN with GraphSAGE-style mean aggregation.

Same input/output contract as STGNN_GAT — useful as an ablation backbone
that swaps attention for plain neighbourhood mean aggregation.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .stgnn_gat import TemporalConv


class SAGELayer(nn.Module):
    """Edge-weighted mean-aggregation layer (GraphSAGE variant).

    h_v <- ReLU( W_self · h_v  +  W_neigh · mean_{u ∈ N(v)} ( w_uv · h_u ) )
    """

    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.lin_self = nn.Linear(in_dim, out_dim)
        self.lin_neigh = nn.Linear(in_dim, out_dim)
        nn.init.xavier_uniform_(self.lin_self.weight)
        nn.init.xavier_uniform_(self.lin_neigh.weight)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,                # [B, N, in_dim]
        edge_index: torch.Tensor,       # [2, E]
        edge_weight: torch.Tensor,      # [E, 1]
    ) -> torch.Tensor:
        B, N, _ = x.shape
        src, dst = edge_index[0], edge_index[1]
        w = edge_weight.squeeze(-1) if edge_weight.dim() == 2 else edge_weight  # [E]

        # Weighted neighbour features.
        msg = x[:, src, :] * w.view(1, -1, 1)                    # [B, E, in_dim]

        # Sum to destination then divide by sum of weights per destination.
        in_dim = x.shape[-1]
        agg = torch.zeros(B, N, in_dim, device=x.device).scatter_add_(
            1, dst.view(1, -1, 1).expand(B, -1, in_dim), msg
        )
        deg = torch.zeros(B, N, device=x.device).scatter_add_(
            1, dst.unsqueeze(0).expand(B, -1), w.unsqueeze(0).expand(B, -1)
        )                                                        # [B, N]
        agg = agg / (deg.unsqueeze(-1) + 1e-8)                    # mean

        out = F.relu(self.lin_self(x) + self.lin_neigh(agg))
        return self.dropout(out)


class STGNN_SAGE(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_dim: int = 32,
        sage_dim: int = 32,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.t1 = TemporalConv(in_features, hidden_dim, kernel_size=3, dilation=1)
        self.sage1 = SAGELayer(hidden_dim, sage_dim, dropout=dropout)
        self.sage2 = SAGELayer(sage_dim, sage_dim, dropout=dropout)
        self.t2 = TemporalConv(sage_dim, hidden_dim, kernel_size=3, dilation=2)
        self.norm = nn.LayerNorm(hidden_dim)
        self.head = nn.Linear(hidden_dim, 1)
        self._embedding_dim = hidden_dim * 2

    @property
    def embedding_dim(self) -> int:
        return self._embedding_dim

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor,
        return_embedding: bool = False,
    ):
        z = self.t1(x)
        B, H, N, D = z.shape

        # Vectorized: collapse (batch, time) so each layer runs once on B*H graphs.
        z_flat = z.reshape(B * H, N, D)
        z_flat = self.sage1(z_flat, edge_index, edge_weight)
        z_flat = self.sage2(z_flat, edge_index, edge_weight)
        z = z_flat.reshape(B, H, N, -1)

        z = self.t2(z)
        z = self.norm(z)
        last = z[:, -1, :, :]
        y = self.head(last).squeeze(-1)

        if return_embedding:
            mean_pool = last.mean(dim=1)
            max_pool, _ = last.max(dim=1)
            graph_emb = torch.cat([mean_pool, max_pool], dim=-1)
            return y, graph_emb
        return y
