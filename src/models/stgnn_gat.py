"""Inductive Spatio-Temporal GNN with GAT spatial layers.

Architecture:
    Input [B, H, N, F] ──┐
                         ├─ Temporal 1D-Conv per node ──┐
                         │                              │
                         │                              ▼
                         │            GAT layer x L  on each time step
                         │                              │
                         │                              ▼
                         │            Temporal 1D-Conv per node
                         │                              │
                         │                              ▼
                         └────────  Last-step Linear ── Output [B, N]

Parameter count is independent of |V|, so the same model can be applied to a
4-node Guwahati graph or a 39-node Delhi graph without any shape change.
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class GATLayer(nn.Module):
    """Single-head, edge-weighted Graph Attention layer.

    Implemented from scratch (no torch-geometric dependency) so the project
    can run on a CPU-only laptop without pulling in PyG compiled extensions.
    Supports an external edge weight that multiplicatively scales attention.
    """

    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.W = nn.Linear(in_dim, out_dim, bias=False)
        self.a_src = nn.Parameter(torch.empty(out_dim))
        self.a_dst = nn.Parameter(torch.empty(out_dim))
        nn.init.xavier_uniform_(self.W.weight)
        nn.init.normal_(self.a_src, std=0.1)
        nn.init.normal_(self.a_dst, std=0.1)
        self.dropout = nn.Dropout(dropout)
        self.leaky = nn.LeakyReLU(0.2)

    def forward(
        self,
        x: torch.Tensor,                # [B, N, in_dim]
        edge_index: torch.Tensor,       # [2, E]
        edge_weight: torch.Tensor,      # [E, 1] or [E]
    ) -> torch.Tensor:
        B, N, _ = x.shape
        Wx = self.W(x)                          # [B, N, out_dim]
        src, dst = edge_index[0], edge_index[1] # [E]

        Wx_src = Wx[:, src, :]                  # [B, E, out_dim]
        Wx_dst = Wx[:, dst, :]                  # [B, E, out_dim]

        e = self.leaky(
            (Wx_src * self.a_src).sum(-1) + (Wx_dst * self.a_dst).sum(-1)
        )                                       # [B, E]

        if edge_weight is not None:
            w = edge_weight.squeeze(-1) if edge_weight.dim() == 2 else edge_weight
            e = e + torch.log(w.clamp(min=1e-8)).unsqueeze(0)  # log-additive

        # Per-destination softmax.
        e = e - e.max(dim=1, keepdim=True).values
        e_exp = torch.exp(e)                                # [B, E]
        denom = torch.zeros(B, N, device=x.device).scatter_add_(
            1, dst.unsqueeze(0).expand(B, -1), e_exp
        )                                                   # [B, N]
        alpha = e_exp / (denom[:, dst] + 1e-8)               # [B, E]
        alpha = self.dropout(alpha)

        # Aggregate neighbour features.
        msg = Wx_src * alpha.unsqueeze(-1)                   # [B, E, out_dim]
        out = torch.zeros(B, N, self.out_dim, device=x.device).scatter_add_(
            1, dst.view(1, -1, 1).expand(B, -1, self.out_dim), msg
        )                                                   # [B, N, out_dim]
        return out


class TemporalConv(nn.Module):
    """Dilated 1-D causal convolution applied independently per node."""

    def __init__(self, in_dim: int, out_dim: int, kernel_size: int = 3, dilation: int = 1) -> None:
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels=in_dim,
            out_channels=out_dim,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=(kernel_size - 1) * dilation,
        )
        self.activation = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, H, N, in_dim] -> [B, H, N, out_dim]"""
        B, H, N, F = x.shape
        # Apply 1-D conv along the time axis, treating each (B, N) pair as a sample.
        z = x.permute(0, 2, 3, 1).reshape(B * N, F, H)       # [B*N, F, H]
        z = self.conv(z)[..., :H]                            # causal trim
        z = self.activation(z)
        out_dim = z.shape[1]
        z = z.reshape(B, N, out_dim, H).permute(0, 3, 1, 2)  # [B, H, N, out_dim]
        return z


class STGNN_GAT(nn.Module):
    """Inductive ST-GNN with two GAT layers sandwiched between two TCN blocks.

    Forward signature:
        x: [B, H, N, F]
        edge_index: [2, E] (long)
        edge_weight: [E, 1] (float)
        return_embedding: if True, also return the graph-pooled embedding
                          used as the input to the DANN discriminator.
    Output: [B, N]   (PM2.5 at the next horizon step, per station)
    """

    def __init__(
        self,
        in_features: int,
        hidden_dim: int = 32,
        gat_dim: int = 32,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.t1 = TemporalConv(in_features, hidden_dim, kernel_size=3, dilation=1)
        self.gat1 = GATLayer(hidden_dim, gat_dim, dropout=dropout)
        self.gat2 = GATLayer(gat_dim, gat_dim, dropout=dropout)
        self.t2 = TemporalConv(gat_dim, hidden_dim, kernel_size=3, dilation=2)
        self.norm = nn.LayerNorm(hidden_dim)
        self.head = nn.Linear(hidden_dim, 1)
        self._embedding_dim = hidden_dim * 2  # mean + max pool

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
        # First temporal block.
        z = self.t1(x)                          # [B, H, N, hidden_dim]
        B, H, N, D = z.shape

        # Vectorized: collapse (batch, time) to a single batch dim so the
        # GAT layer runs once over B*H graph instances. This removes the
        # per-time-step Python loop and gives ~Hx CPU speedup.
        z_flat = z.reshape(B * H, N, D)
        z_flat = self.gat1(z_flat, edge_index, edge_weight)
        z_flat = F.elu(z_flat)
        z_flat = self.gat2(z_flat, edge_index, edge_weight)
        z_flat = F.elu(z_flat)
        z = z_flat.reshape(B, H, N, -1)         # [B, H, N, gat_dim]

        # Second temporal block.
        z = self.t2(z)                          # [B, H, N, hidden_dim]
        z = self.norm(z)

        # Take the last time step for the per-node forecast.
        last = z[:, -1, :, :]                   # [B, N, hidden_dim]
        y = self.head(last).squeeze(-1)         # [B, N]

        if return_embedding:
            # Graph-level pooled embedding (size-invariant, used by DANN).
            mean_pool = last.mean(dim=1)         # [B, hidden_dim]
            max_pool, _ = last.max(dim=1)        # [B, hidden_dim]
            graph_emb = torch.cat([mean_pool, max_pool], dim=-1)  # [B, 2*hidden_dim]
            return y, graph_emb
        return y
