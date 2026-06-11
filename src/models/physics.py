"""Graph advection–diffusion–reaction (ADR) physics residual for Variant E (GNN-PINN-TL).

Implements the formulation fixed in reports/PINN_PHYSICS.md:

  - State: climatology-residual concentration in physical units, C' = C - clim [µg/m³]
           (so the emission source term S' ≈ 0).
  - Full unsteady forward-Euler ADR residual at node i, with Δt = 3 h:

        R_i = (Ĉ'_i(t+Δt) - C'^obs_i(t)) / Δt
              + α · (𝒜(t) C'^obs)_i        # advection, measured-wind upwind operator
              + K · (L C'^obs)_i            # diffusion, mass-conserving graph Laplacian
              + λ · C'^obs_i                # deposition / first-order loss

        L_phys = mean_{b,i} R_i²

  - Spatial operators act on the *observed* current field C'^obs(t) (known at all nodes);
    only Ĉ'(t+Δt) is the prediction. α, K, λ ≥ 0 are learnable (softplus) effective
    physical coefficients identified from data.

Operators live on the sensor graph (edges), never a collocation grid — the regime where
classic PINNs fail on sparse/irregular data (FieldFormer, arXiv:2510.03589).
"""
from __future__ import annotations

from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.graph_construction import bearing_deg, haversine_km, knn_graph


def build_geometry(coords: List[Tuple[float, float]]) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return (bearing_deg [N,N], dist_km [N,N]).

    bearing[i,j] = compass bearing from i to j; dist[i,j] = haversine km (diagonal set to a
    large value so 1/dist on the diagonal vanishes; the diagonal is also masked out downstream).
    """
    n = len(coords)
    bearing = np.zeros((n, n), dtype=np.float32)
    dist = np.full((n, n), 1e6, dtype=np.float32)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            bearing[i, j] = bearing_deg(*coords[i], *coords[j])
            dist[i, j] = max(haversine_km(*coords[i], *coords[j]), 1e-3)
    return torch.from_numpy(bearing), torch.from_numpy(dist)


def build_diffusion_weights(coords: List[Tuple[float, float]], k: int = 3,
                            sigma_km: float = 5.0) -> torch.Tensor:
    """Symmetric Gaussian-distance adjacency W [N,N] of the k-NN graph (for the diffusion
    Laplacian L = diag(W·1) - W, applied as (LC)_i = Σ_j W_ij (C_i - C_j))."""
    n = len(coords)
    ei, ew = knn_graph(coords, k=min(k, max(1, n - 1)), sigma_km=sigma_km)
    W = torch.zeros(n, n, dtype=torch.float32)
    src, dst = ei[0].long(), ei[1].long()
    W[src, dst] = ew.squeeze(-1).float()
    W = 0.5 * (W + W.t())                      # symmetrize → mass-conserving diffusion
    W.fill_diagonal_(0.0)
    return W


def apply_diffusion(C: torch.Tensor, W: torch.Tensor) -> torch.Tensor:
    """(L C)_i = Σ_j W_ij (C_i - C_j) = deg_i·C_i - (W C)_i.  C:[B,N], W:[N,N] → [B,N]."""
    deg = W.sum(dim=1)                          # [N]
    return C * deg.unsqueeze(0) - C @ W         # W symmetric ⇒ (W C)[b,i] = (C @ W)[b,i]


def apply_advection(C: torch.Tensor, wind_speed: torch.Tensor, wind_dir_deg: torch.Tensor,
                    bearing: torch.Tensor, dist: torch.Tensor) -> torch.Tensor:
    """First-order upwind advection on the measured-wind graph.

    (𝒜 C)_i = Σ_j u_ij (C_i - C_j),  u_ij = [ v_i · cos(φ_i^flow - θ_ij) ]_+ / d_ij,
    where φ^flow = WD + 180° is the direction the wind blows toward.

    C, wind_speed, wind_dir_deg: [B, N];  bearing, dist: [N, N]  →  [B, N].
    """
    B, N = C.shape
    phi_flow = (wind_dir_deg + 180.0) % 360.0                       # [B,N] direction blown toward
    ang = torch.deg2rad(phi_flow.unsqueeze(2) - bearing.unsqueeze(0))  # [B,N,N]
    u = torch.relu(wind_speed.unsqueeze(2) * torch.cos(ang)) / dist.unsqueeze(0)  # [B,N,N]
    eye = torch.eye(N, device=C.device, dtype=C.dtype).unsqueeze(0)
    u = u * (1.0 - eye)                                             # no self-advection
    return C * u.sum(dim=2) - torch.einsum("bij,bj->bi", u, C)


class PhysicsADR(nn.Module):
    """Forward-Euler advection–diffusion–reaction residual penalty with learnable
    positive effective coefficients (α advection, K diffusion, λ deposition)."""

    def __init__(self, dt_hours: float = 3.0) -> None:
        super().__init__()
        self.dt = float(dt_hours)
        self._log_alpha = nn.Parameter(torch.zeros(()))   # softplus(0) ≈ 0.69
        self._log_K = nn.Parameter(torch.zeros(()))
        self._log_lambda = nn.Parameter(torch.zeros(()))

    @property
    def alpha(self) -> torch.Tensor: return F.softplus(self._log_alpha)
    @property
    def K(self) -> torch.Tensor: return F.softplus(self._log_K)
    @property
    def lam(self) -> torch.Tensor: return F.softplus(self._log_lambda)

    def coeffs(self) -> dict:
        return {"alpha": float(self.alpha.detach()), "K": float(self.K.detach()),
                "lambda": float(self.lam.detach())}

    def forward(self, C_pred: torch.Tensor, C_obs: torch.Tensor,
                wind_speed: torch.Tensor, wind_dir_deg: torch.Tensor,
                bearing: torch.Tensor, dist: torch.Tensor, W_diff: torch.Tensor,
                scale: float = 1.0) -> torch.Tensor:
        # C_pred, C_obs: [B,N] climatology-residual concentration in µg/m³.
        # The residual is non-dimensionalized by `scale` (the characteristic concentration
        # std σ_y) so L_phys is comparable to the z-scored data MSE and λ_phys is interpretable
        # and city-invariant; the learned α, K, λ remain physical. See PINN_PHYSICS.md §4.
        if not (torch.isfinite(C_pred).all() and torch.isfinite(C_obs).all()):
            return torch.zeros((), dtype=C_pred.dtype, device=C_pred.device)
        dCdt = (C_pred - C_obs) / self.dt
        adv = apply_advection(C_obs, wind_speed, wind_dir_deg, bearing, dist)
        dif = apply_diffusion(C_obs, W_diff)
        R = (dCdt + self.alpha * adv + self.K * dif + self.lam * C_obs) / max(scale, 1e-6)
        return (R ** 2).mean()


class SpatialPhysics(nn.Module):
    """Steady-state spatial advection–diffusion residual on the PREDICTED field, for
    *virtual sensing* (estimating PM2.5 at unsensored nodes).

    L_phys = mean‖ α·(𝒜 Ĉ) + K·(L Ĉ) ‖² / scale²,  α,K = softplus(·) > 0.

    No temporal term: unsensored nodes have no observed current concentration, so the prior
    acts purely spatially — penalizing transport-imbalance of the predicted field propagates
    concentration from sensored anchors to unsensored nodes along wind + diffusion paths.
    Wind is measured (available at all nodes from reanalysis). See reports/PINN_PHYSICS.md."""

    def __init__(self, use_adv: bool = True, use_diff: bool = True) -> None:
        super().__init__()
        # Mode flags enable the wind-vs-smoothness ablation: both (transport physics),
        # advection-only (pure wind), or diffusion-only (generic graph smoothness ≈ Kriging).
        self.use_adv, self.use_diff = use_adv, use_diff
        self._log_alpha = nn.Parameter(torch.zeros(()))
        self._log_K = nn.Parameter(torch.zeros(()))

    @property
    def alpha(self) -> torch.Tensor: return F.softplus(self._log_alpha)
    @property
    def K(self) -> torch.Tensor: return F.softplus(self._log_K)

    def coeffs(self) -> dict:
        return {"alpha": float(self.alpha.detach()), "K": float(self.K.detach()),
                "use_adv": self.use_adv, "use_diff": self.use_diff}

    def forward(self, C_pred: torch.Tensor, wind_speed: torch.Tensor, wind_dir_deg: torch.Tensor,
                bearing: torch.Tensor, dist: torch.Tensor, W_diff: torch.Tensor,
                scale: float = 1.0) -> torch.Tensor:
        if not torch.isfinite(C_pred).all():
            return torch.zeros((), dtype=C_pred.dtype, device=C_pred.device)
        R = torch.zeros_like(C_pred)
        if self.use_adv:
            R = R + self.alpha * apply_advection(C_pred, wind_speed, wind_dir_deg, bearing, dist)
        if self.use_diff:
            R = R + self.K * apply_diffusion(C_pred, W_diff)
        R = R / max(scale, 1e-6)
        return (R ** 2).mean()
