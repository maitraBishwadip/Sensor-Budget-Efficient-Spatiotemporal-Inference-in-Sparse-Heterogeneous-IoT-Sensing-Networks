"""Minimal from-scratch STGCN (Yu, Yin & Zhu, IJCAI 2018) forecasting baseline.

Reviewer-requested standard ST-GNN comparator (issue C4/R1-2). Follows the original
design — two ST-Conv blocks, each a temporal gated convolution (GLU) sandwich around a
first-order graph convolution, followed by a temporal output layer — implemented with
plain PyTorch only (no PyG), matching this repository's constraints.

Adaptation to this testbed: history H=8 steps at 3-h cadence, so the temporal kernel is
K_t=2 (the original used K_t=3 on H=12); two blocks shrink T 8->6->4 and the output layer
collapses the remaining 4 steps. The graph convolution uses the symmetric-normalized
adjacency with self-loops built from the same k-NN Gaussian edge weights as the GAT model
(Eq. (1) of the paper), so both models see identical graph information.

Interface matches STGNN_GAT: forward(x[B,H,N,F], edge_index, edge_weight) -> [B,N].
"""
from __future__ import annotations

import torch
import torch.nn as nn


def _normalized_adj(edge_index: torch.Tensor, edge_weight: torch.Tensor, n: int) -> torch.Tensor:
    """Dense sym-normalized adjacency with self-loops: D^{-1/2}(A_sym + I)D^{-1/2}."""
    A = torch.zeros(n, n, dtype=edge_weight.dtype, device=edge_weight.device)
    A[edge_index[0], edge_index[1]] = edge_weight.reshape(-1)
    A = torch.maximum(A, A.t())                       # symmetrize the directed k-NN graph
    A = A + torch.eye(n, dtype=A.dtype, device=A.device)
    d = A.sum(dim=1)
    d_inv_sqrt = torch.pow(d.clamp(min=1e-12), -0.5)
    return d_inv_sqrt.unsqueeze(1) * A * d_inv_sqrt.unsqueeze(0)


class TemporalGatedConv(nn.Module):
    """Gated 1-D convolution over time (GLU), applied per node. [B,C,N,T] -> [B,C',N,T-Kt+1]."""

    def __init__(self, c_in: int, c_out: int, kt: int):
        super().__init__()
        self.conv = nn.Conv2d(c_in, 2 * c_out, kernel_size=(1, kt))
        self.res = nn.Conv2d(c_in, c_out, kernel_size=(1, 1))
        self.kt = kt

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        p, q = self.conv(x).chunk(2, dim=1)
        res = self.res(x)[:, :, :, self.kt - 1:]
        return (p + res) * torch.sigmoid(q)


class STConvBlock(nn.Module):
    """TemporalGatedConv -> first-order graph conv (ReLU) -> TemporalGatedConv -> LayerNorm."""

    def __init__(self, c_in: int, c_spat: int, c_out: int, kt: int, n_hint: int, dropout: float):
        super().__init__()
        self.t1 = TemporalGatedConv(c_in, c_out, kt)
        self.theta = nn.Linear(c_out, c_spat)
        self.t2 = TemporalGatedConv(c_spat, c_out, kt)
        self.norm = nn.LayerNorm(c_out)   # over channels; shape-independent of N
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, A_hat: torch.Tensor) -> torch.Tensor:
        x = self.t1(x)                                     # [B,C,N,T]
        x = x.permute(0, 3, 2, 1)                          # [B,T,N,C]
        x = torch.relu(torch.einsum("mn,btnc->btmc", A_hat, self.theta(x)))
        x = x.permute(0, 3, 2, 1)                          # [B,C,N,T]
        x = self.t2(x)
        x = self.norm(x.permute(0, 3, 2, 1)).permute(0, 3, 2, 1)
        return self.dropout(x)


class STGCN(nn.Module):
    def __init__(self, in_features: int, hidden_dim: int = 64, spat_dim: int = 16,
                 kt: int = 2, history: int = 8, dropout: float = 0.15):
        super().__init__()
        self.block1 = STConvBlock(in_features, spat_dim, hidden_dim, kt, 0, dropout)
        self.block2 = STConvBlock(hidden_dim, spat_dim, hidden_dim, kt, 0, dropout)
        t_rem = history - 4 * (kt - 1)                     # time steps left after two blocks
        if t_rem < 1:
            raise ValueError(f"history {history} too short for kt={kt}")
        self.out_temporal = TemporalGatedConv(hidden_dim, hidden_dim, t_rem)
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                edge_weight: torch.Tensor) -> torch.Tensor:
        # x: [B, H, N, F] -> [B, F, N, H]
        n = x.shape[2]
        A_hat = _normalized_adj(edge_index, edge_weight.to(x.dtype), n)
        h = x.permute(0, 3, 2, 1)
        h = self.block1(h, A_hat)
        h = self.block2(h, A_hat)
        h = self.out_temporal(h)                           # [B,C,N,1]
        return self.head(h[:, :, :, 0].permute(0, 2, 1)).squeeze(-1)   # [B,N]
