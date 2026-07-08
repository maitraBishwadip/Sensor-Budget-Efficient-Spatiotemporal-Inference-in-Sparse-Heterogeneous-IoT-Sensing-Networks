"""Adaptive edge-weight scale on the sparsest deployment (reviewer request).

The attention-entropy audit (analysis/attention_entropy.py) shows that under
the shared sigma = 5 km the distance prior's within-neighbourhood spread on
Guwahati (std ~3.2 logits) swamps the feature logits, so the GAT layers there
follow the distance prior almost exactly. The proposed remedy is a per-node
scale sigma_j = mean(d_.j) over each destination's incoming k-NN edges, which
restores within-neighbourhood contrast to the same order on every deployment.
This script evaluates that remedy where it could matter, on Guwahati:

  1. source-only training with the adaptive-sigma graph (fixed recipe), against
     the fixed-sigma reference in results/gnn/gnn_gat_source_only_fixed.json;
  2. the Delhi->Guwahati d=30% cold-start transfer cell with the adaptive-sigma
     target graph (three-way verified), against the fixed-sigma cell in
     results/gnn_tl/variantA_gat_fixed.json;
  3. the induced within-neighbourhood spread of log w under both schemes
     (geometry only, no training).

Checkpoints land in models/adaptive_sigma/ so the paper's artifacts are
untouched. Writes results/adaptive_sigma.json.

Run:  python -u -m analysis.adaptive_sigma
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import src.train_gnn as tg                                # noqa: E402
import src.train_gnn_tl as tl                             # noqa: E402
from src.graph_construction import pairwise_distance_matrix  # noqa: E402

OUT = Path("results/adaptive_sigma.json")
CKPT_DIR = Path("models/adaptive_sigma")


def adaptive_knn_graph(coords: List[Tuple[float, float]], k: int = 3,
                       **_ignored) -> Tuple[torch.Tensor, torch.Tensor]:
    """Same k-NN edges as knn_graph, but w_ij = exp(-d_ij^2 / (2 sigma_j^2))
    with sigma_j the mean distance of destination j's incoming edges."""
    n = len(coords)
    k = min(k, n - 1)
    D = pairwise_distance_matrix(coords)
    edges = []
    for i in range(n):
        order = [j for j in np.argsort(D[i]) if j != i][:k]
        edges.extend((i, j, float(D[i, j])) for j in order)
    dsts = np.array([e[1] for e in edges])
    dists = np.array([e[2] for e in edges])
    sigma_j = np.array([dists[dsts == j].mean() if (dsts == j).any() else 1.0
                        for j in range(n)])
    weights = np.exp(-(dists ** 2) / (2.0 * sigma_j[dsts] ** 2))
    edge_index = torch.tensor([[e[0] for e in edges], [e[1] for e in edges]], dtype=torch.long)
    edge_attr = torch.tensor(weights, dtype=torch.float32).unsqueeze(-1)
    return edge_index, edge_attr


def logw_spread(coords, k=3, builder=None) -> float:
    """Mean within-neighbourhood std of log w over destination nodes."""
    if builder is None:
        from src.graph_construction import knn_graph
        builder = knn_graph
    edge_index, edge_attr = builder(coords, k=k)
    logw = torch.log(edge_attr.squeeze(-1).clamp(min=1e-8)).numpy()
    dst = edge_index[1].numpy()
    # ddof=1 matches the sample std reported by analysis/attention_entropy.py.
    stds = [logw[dst == j].std(ddof=1) for j in np.unique(dst) if (dst == j).sum() >= 2]
    return float(np.mean(stds))


def main():
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    cities = tg.load_all_cities(fixed=True)
    guw = cities["Guwahati"].coords

    results = {
        "logw_spread_guwahati": {
            "fixed_sigma5": logw_spread(guw),
            "adaptive": logw_spread(guw, builder=adaptive_knn_graph),
        }
    }
    print(f"Guwahati within-neighbourhood std(log w): "
          f"fixed={results['logw_spread_guwahati']['fixed_sigma5']:.2f}  "
          f"adaptive={results['logw_spread_guwahati']['adaptive']:.2f}")

    # Reroute graph construction and checkpoint output; sources stay fixed-sigma.
    tg.build_city_graph = lambda coords, strategy="knn", k=3, **kw: adaptive_knn_graph(coords, k=k)
    tl.build_city_graph = tg.build_city_graph
    tg.MODELS_DIR = CKPT_DIR
    tl.MODELS_DIR = CKPT_DIR

    print("\n=== Guwahati source-only, adaptive sigma ===")
    metrics, _ = tg.train_base_gnn(cities["Guwahati"], backbone="gat", fixed=True,
                                   ckpt_extra="_adaptive")
    results["guwahati_source_only_adaptive"] = metrics

    print("\n=== Delhi->Guwahati d=30% transfer, adaptive-sigma target graph ===")
    cell = tl.transfer_variant_a(cities, "Delhi", "Guwahati", 0.30,
                                 backbone="gat", fixed=True)
    results["delhi_to_guwahati_d30_adaptive"] = cell

    with open("results/gnn/gnn_gat_source_only_fixed.json") as fh:
        results["guwahati_source_only_fixed_reference"] = json.load(fh)["Guwahati"]
    with open("results/gnn_tl/variantA_gat_fixed.json") as fh:
        ref = json.load(fh)
    key = [k for k in ref if "Delhi" in k and "Guwahati" in k]
    results["delhi_to_guwahati_d30_fixed_reference"] = {k: ref[k] for k in key}

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
