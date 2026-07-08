"""Sensing-graph geometry statistics (reviewer request: contextualize sigma = 5 km).

For each deployment, reports the haversine-distance statistics of the k = 3
nearest-neighbour edges actually used by Eq. (1): per-edge mean/median/max, the
mean distance to the single nearest neighbour, and the Gaussian edge-weight
range exp(-d^2 / 2 sigma^2) that the distances induce. This is what lets a
reader judge how the fixed sigma = 5 km decay behaves on graphs whose spacing
differs by an order of magnitude (Delhi 40 nodes vs Guwahati 4).

Writes results/graph_geometry.json.

Run:  python -u -m analysis.graph_geometry
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.graph_construction import pairwise_distance_matrix  # noqa: E402
from src.train_gnn import KNN_K, load_all_cities             # noqa: E402

OUT = Path("results/graph_geometry.json")
SIGMA_KM = 5.0


def main():
    cities = load_all_cities(fixed=True)
    results = {}
    for cname, c in cities.items():
        N = len(c.coords)
        D = pairwise_distance_matrix(c.coords).astype(np.float64)
        k = min(KNN_K, N - 1)
        knn_d = []       # all k-NN edge distances (the |E| = kN edges of Eq. (1))
        nn1_d = []       # distance to the single nearest neighbour, per node
        for i in range(N):
            order = np.argsort(D[i])
            order = [j for j in order if j != i][:k]
            knn_d.extend(D[i, j] for j in order)
            nn1_d.append(D[i, order[0]])
        knn_d = np.asarray(knn_d)
        nn1_d = np.asarray(nn1_d)
        w = np.exp(-(knn_d ** 2) / (2 * SIGMA_KM ** 2))
        results[cname] = {
            "N": int(N), "k": int(k), "n_edges": int(len(knn_d)),
            "knn_dist_mean_km": float(knn_d.mean()),
            "knn_dist_median_km": float(np.median(knn_d)),
            "knn_dist_max_km": float(knn_d.max()),
            "knn_dist_min_km": float(knn_d.min()),
            "nn1_dist_mean_km": float(nn1_d.mean()),
            "edge_weight_mean": float(w.mean()),
            "edge_weight_min": float(w.min()),
            "edge_weight_max": float(w.max()),
        }
        print(f"{cname:8s} N={N:2d} |E|={len(knn_d):3d}  kNN dist (km): "
              f"mean={knn_d.mean():5.2f} median={np.median(knn_d):5.2f} "
              f"max={knn_d.max():5.2f} min={knn_d.min():4.2f}  "
              f"1-NN mean={nn1_d.mean():5.2f}  w(sigma=5km): "
              f"[{w.min():.3f}, {w.max():.3f}] mean={w.mean():.3f}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
