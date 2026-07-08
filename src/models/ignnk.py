"""Minimal from-scratch IGNNK (Wu, Zhuang, Labbe & Sun, AAAI 2021) kriging baseline.

Reviewer-requested learned-kriging comparator for the virtual-sensing ladder (the
"KCN/IGNNK not compared" gap, issue C4/R2-M4). IGNNK reconstructs the signal at
masked (unsensored) nodes from the observed nodes' readings over a short window,
using diffusion graph convolutions (Graph WaveNet's D-GCN) in an
encoder-residual-decoder stack:

    X [B, N, h] -> D_GCN(h -> z) -> ReLU -> D_GCN(z -> z) + skip -> ReLU
                -> D_GCN(z -> h) -> X_hat [B, N, h]

The diffusion convolution runs a K-order Chebyshev-style recursion on the forward
and backward random-walk matrices of the (dense) Gaussian-kernel adjacency — the
same sigma = 5 km kernel as Eq. (1) of the paper — implemented with plain PyTorch
only (no PyG), matching this repository's constraints.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


def random_walk_supports(W: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
    """Forward/backward random-walk transition matrices of a dense adjacency."""
    A = torch.from_numpy(W.astype(np.float32))
    A_f = A / A.sum(dim=1, keepdim=True).clamp(min=1e-12)
    A_b = A.t() / A.t().sum(dim=1, keepdim=True).clamp(min=1e-12)
    return A_f, A_b


class DiffusionGraphConv(nn.Module):
    """K-order diffusion convolution over both walk directions. [B,N,C] -> [B,N,C']."""

    def __init__(self, c_in: int, c_out: int, orders: int = 2):
        super().__init__()
        self.orders = orders
        n_mat = 2 * orders + 1                       # identity + K hops per direction
        self.theta = nn.Parameter(torch.empty(n_mat * c_in, c_out))
        self.bias = nn.Parameter(torch.zeros(c_out))
        nn.init.xavier_normal_(self.theta)

    def forward(self, x: torch.Tensor, A_f: torch.Tensor, A_b: torch.Tensor) -> torch.Tensor:
        supports = [x]
        for A in (A_f, A_b):
            x1 = torch.einsum("mn,bnc->bmc", A, x)
            supports.append(x1)
            xk_prev, xk = x, x1
            for _ in range(2, self.orders + 1):
                xk_next = 2 * torch.einsum("mn,bnc->bmc", A, xk) - xk_prev
                supports.append(xk_next)
                xk_prev, xk = xk, xk_next
        h = torch.cat(supports, dim=-1)              # [B, N, n_mat*C]
        return h @ self.theta + self.bias


class IGNNK(nn.Module):
    def __init__(self, history: int = 8, hidden_dim: int = 100, orders: int = 2):
        super().__init__()
        self.gc1 = DiffusionGraphConv(history, hidden_dim, orders)
        self.gc2 = DiffusionGraphConv(hidden_dim, hidden_dim, orders)
        self.gc3 = DiffusionGraphConv(hidden_dim, history, orders)

    def forward(self, x: torch.Tensor, A_f: torch.Tensor, A_b: torch.Tensor) -> torch.Tensor:
        # x: [B, N, h] window of the (masked) signal; returns the reconstructed window.
        h1 = torch.relu(self.gc1(x, A_f, A_b))
        h2 = torch.relu(self.gc2(h1, A_f, A_b) + h1)
        return self.gc3(h2, A_f, A_b)
