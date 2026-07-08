"""Minimal from-scratch Graph WaveNet (Wu, Pan, Long, Jiang & Zhang, IJCAI 2019).

Reviewer-requested standard ST-GNN comparator (issue C4/R1-2), completing the
STGCN/Graph-WaveNet pair named by Reviewer 1. Follows the original design ---
stacked gated dilated causal temporal convolutions (WaveNet), each followed by a
diffusion graph convolution over the forward/backward random-walk matrices plus a
learned *adaptive* adjacency softmax(ReLU(E1 E2^T)), with residual and skip
connections --- implemented with plain PyTorch only (no PyG).

Adaptation to this testbed: history H=8 at 3-h cadence, so two blocks of two
layers with dilations (1, 2, 1, 3) give receptive field 8 = H (the original used
H=12 with dilations 1,2,1,2,...). Note the adaptive-adjacency node embeddings are
per-node parameters, so unlike the paper's inductive encoder this baseline is
|V|-specific --- appropriate for the source-only forecasting comparison only.

Interface matches STGNN_GAT: forward(x[B,H,N,F], edge_index, edge_weight) -> [B,N].
"""
from __future__ import annotations

import torch
import torch.nn as nn


def _random_walk_pair(edge_index: torch.Tensor, edge_weight: torch.Tensor,
                      n: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Dense forward/backward random-walk transition matrices of the k-NN graph."""
    A = torch.zeros(n, n, dtype=edge_weight.dtype, device=edge_weight.device)
    A[edge_index[0], edge_index[1]] = edge_weight.reshape(-1)
    A = torch.maximum(A, A.t())                       # symmetrize the directed k-NN graph
    A_f = A / A.sum(dim=1, keepdim=True).clamp(min=1e-12)
    A_b = A.t() / A.t().sum(dim=1, keepdim=True).clamp(min=1e-12)
    return A_f, A_b


class _DiffusionConv2d(nn.Module):
    """K-order diffusion convolution over a list of supports. [B,C,N,T] -> [B,C',N,T]."""

    def __init__(self, c_in: int, c_out: int, n_supports: int, orders: int = 2,
                 dropout: float = 0.15):
        super().__init__()
        self.orders = orders
        n_mat = n_supports * orders + 1
        self.proj = nn.Conv2d(n_mat * c_in, c_out, kernel_size=(1, 1))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, supports: list[torch.Tensor]) -> torch.Tensor:
        out = [x]
        for A in supports:
            xk = torch.einsum("mn,bcnt->bcmt", A, x)
            out.append(xk)
            for _ in range(2, self.orders + 1):
                xk = torch.einsum("mn,bcnt->bcmt", A, xk)
                out.append(xk)
        return self.dropout(self.proj(torch.cat(out, dim=1)))


class GraphWaveNet(nn.Module):
    def __init__(self, in_features: int, n_nodes: int, residual_channels: int = 32,
                 dilation_channels: int = 32, skip_channels: int = 64,
                 end_channels: int = 128, kernel_size: int = 2, orders: int = 2,
                 adapt_dim: int = 10, dropout: float = 0.15,
                 dilations: tuple[int, ...] = (1, 2, 1, 3)):
        super().__init__()
        self.start_conv = nn.Conv2d(in_features, residual_channels, kernel_size=(1, 1))
        self.e1 = nn.Parameter(torch.randn(n_nodes, adapt_dim) * 0.1)
        self.e2 = nn.Parameter(torch.randn(n_nodes, adapt_dim) * 0.1)

        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        self.gconvs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for d in dilations:
            self.filter_convs.append(nn.Conv2d(residual_channels, dilation_channels,
                                               kernel_size=(1, kernel_size), dilation=(1, d)))
            self.gate_convs.append(nn.Conv2d(residual_channels, dilation_channels,
                                             kernel_size=(1, kernel_size), dilation=(1, d)))
            self.gconvs.append(_DiffusionConv2d(dilation_channels, residual_channels,
                                                n_supports=3, orders=orders, dropout=dropout))
            self.skip_convs.append(nn.Conv2d(dilation_channels, skip_channels,
                                             kernel_size=(1, 1)))
            self.norms.append(nn.BatchNorm2d(residual_channels))

        self.end1 = nn.Conv2d(skip_channels, end_channels, kernel_size=(1, 1))
        self.end2 = nn.Conv2d(end_channels, 1, kernel_size=(1, 1))

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                edge_weight: torch.Tensor) -> torch.Tensor:
        # x: [B, H, N, F] -> [B, F, N, T]
        n = x.shape[2]
        A_f, A_b = _random_walk_pair(edge_index, edge_weight.to(x.dtype), n)
        A_adp = torch.softmax(torch.relu(self.e1 @ self.e2.t()), dim=1)
        supports = [A_f, A_b, A_adp]

        h = self.start_conv(x.permute(0, 3, 2, 1))
        skip = 0
        for f, g, gc, sc, bn in zip(self.filter_convs, self.gate_convs,
                                    self.gconvs, self.skip_convs, self.norms):
            residual = h
            h = torch.tanh(f(residual)) * torch.sigmoid(g(residual))
            s = sc(h)
            skip = s if isinstance(skip, int) else skip[..., -s.size(3):] + s
            h = gc(h, supports) + residual[..., -h.size(3):]
            h = bn(h)

        out = torch.relu(skip)
        out = torch.relu(self.end1(out))
        out = self.end2(out)                              # [B, 1, N, T_f]
        return out[:, 0, :, -1]                           # [B, N]
