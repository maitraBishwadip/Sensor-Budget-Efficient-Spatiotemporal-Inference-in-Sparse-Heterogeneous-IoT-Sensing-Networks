"""Graph construction utilities.

Given a city's station coordinates, builds three families of graphs:
  1. k-NN distance graph with Gaussian-decay edge weights.
  2. Wind-aware directed graph weighted by cosine(wind, edge direction).
  3. Hybrid: wind graph + a feature-induced learnable residual is applied
     inside the model at training time; here we just emit the wind graph
     plus the node feature tensor needed by the learnable head.

All builders return PyG-style (edge_index, edge_attr) tensors with float32
weights and int64 indices.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
import torch


EARTH_RADIUS_KM = 6371.0


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance between two (lat, lon) points in kilometres."""
    lat1r, lat2r = np.radians(lat1), np.radians(lat2)
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1r) * np.cos(lat2r) * np.sin(dlon / 2) ** 2
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(a))


def pairwise_distance_matrix(coords: List[Tuple[float, float]]) -> np.ndarray:
    n = len(coords)
    D = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        for j in range(n):
            if i != j:
                D[i, j] = haversine_km(*coords[i], *coords[j])
    return D


def bearing_deg(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Compass bearing from point 1 to point 2, in degrees [0, 360)."""
    lat1r, lat2r = np.radians(lat1), np.radians(lat2)
    dlon = np.radians(lon2 - lon1)
    x = np.sin(dlon) * np.cos(lat2r)
    y = np.cos(lat1r) * np.sin(lat2r) - np.sin(lat1r) * np.cos(lat2r) * np.cos(dlon)
    return float((np.degrees(np.arctan2(x, y)) + 360.0) % 360.0)


def knn_graph(
    coords: List[Tuple[float, float]],
    k: int,
    sigma_km: float = 5.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """k-NN distance graph with Gaussian-decay edge weights.

    Returns:
        edge_index: LongTensor of shape [2, E]
        edge_attr:  FloatTensor of shape [E, 1]  (weights in (0, 1])
    """
    n = len(coords)
    if k >= n:
        k = max(1, n - 1)
    D = pairwise_distance_matrix(coords)
    sources, dests, weights = [], [], []
    for i in range(n):
        order = np.argsort(D[i])
        order = [j for j in order if j != i][:k]
        for j in order:
            w = float(np.exp(-(D[i, j] ** 2) / (2.0 * sigma_km ** 2)))
            sources.append(i)
            dests.append(j)
            weights.append(w)
    edge_index = torch.tensor([sources, dests], dtype=torch.long)
    edge_attr = torch.tensor(weights, dtype=torch.float32).unsqueeze(-1)
    return edge_index, edge_attr


def wind_aware_graph(
    coords: List[Tuple[float, float]],
    prevailing_wind_deg: float,
    decay_km: float = 10.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Directed graph whose edges follow the prevailing wind direction.

    An edge i -> j receives weight = max(0, cos(theta_wind - theta_ij)) *
    exp(-d_ij / decay_km), where theta_ij is the bearing from i to j.
    Self-loops are skipped. Zero-weight edges are dropped.
    """
    n = len(coords)
    sources, dests, weights = [], [], []
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            d_ij = haversine_km(*coords[i], *coords[j])
            theta_ij = bearing_deg(*coords[i], *coords[j])
            cosw = np.cos(np.radians(prevailing_wind_deg - theta_ij))
            w = max(0.0, float(cosw)) * float(np.exp(-d_ij / decay_km))
            if w > 1e-4:
                sources.append(i)
                dests.append(j)
                weights.append(w)
    if not sources:
        # Fall back to k-NN if the wind graph is empty for any reason.
        return knn_graph(coords, k=min(3, n - 1))
    edge_index = torch.tensor([sources, dests], dtype=torch.long)
    edge_attr = torch.tensor(weights, dtype=torch.float32).unsqueeze(-1)
    return edge_index, edge_attr


def build_city_graph(
    coords: List[Tuple[float, float]],
    strategy: str = "knn",
    k: int = 3,
    sigma_km: float = 5.0,
    prevailing_wind_deg: float = 270.0,  # ~westerly is a reasonable default for IGP
    decay_km: float = 10.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Front-door factory. `strategy` ∈ {"knn", "wind", "hybrid"}.

    For "hybrid" we return the wind graph here; the learnable residual lives
    inside the model (see models/stgnn_gat.py — InductiveAdjacencyResidual).
    """
    strategy = strategy.lower()
    if strategy == "knn":
        return knn_graph(coords, k=k, sigma_km=sigma_km)
    if strategy == "wind":
        return wind_aware_graph(coords, prevailing_wind_deg, decay_km=decay_km)
    if strategy == "hybrid":
        return wind_aware_graph(coords, prevailing_wind_deg, decay_km=decay_km)
    raise ValueError(f"unknown graph strategy: {strategy}")
