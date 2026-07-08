"""Attention-entropy audit of the fixed-sigma distance prior (reviewer request).

The GAT logit e_ij = LeakyReLU(a_s^T W x_i + a_d^T W x_j) + log w_ij carries a
distance prior whose magnitude grows quadratically with edge length: on the
sparsest deployment (Guwahati, mean k-NN edge 12.0 km, max 21.3 km at sigma =
5 km) log w reaches -9.1. The question raised in review is whether that term
swamps the feature-driven part and flattens the layer into a static
distance-weighted average. Because the per-neighbourhood softmax is
shift-invariant, the deciding quantity is not the absolute size of log w but
its spread *within* each destination's neighbourhood, compared with the spread
of the feature logits over the same edges. This script measures both, plus the
attention distributions they produce, on the trained checkpoints:

  - within-neighbourhood std of each logit component (feature vs distance);
  - normalized entropy H/ln(deg) of the full attention, of the distance-only
    softmax (feature term removed), and of the feature-only softmax
    (distance term removed);
  - mean total-variation distance between full and distance-only attention,
    and the fraction of neighbourhoods whose top-weighted neighbour changes
    when the feature term is included.

Conditions: the three source-only checkpoints, plus the Delhi->Guwahati and
Delhi->Kolkata d=30% transferred checkpoints (a dense-trained model running
on a sparser target, the setting the concern is about).

Writes results/attention_entropy.json.

Run:  python -u -m analysis.attention_entropy
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.graph_construction import build_city_graph                  # noqa: E402
from src.train_gnn import (                                          # noqa: E402
    GRAPH_STRATEGY, HISTORY, HORIZON, KNN_K, build_backbone, load_all_cities,
)
from src.utils import load_checkpoint, make_windows_masked, set_seed  # noqa: E402

OUT = Path("results/attention_entropy.json")
MAX_WINDOWS = 64  # per condition; each window contributes H=8 graph instances
EPS = 1e-12


def neighbourhoods(edge_index: torch.Tensor, n_nodes: int):
    """List of incoming-edge index arrays, one per destination node."""
    dst = edge_index[1].numpy()
    return [np.where(dst == j)[0] for j in range(n_nodes)]


def entropy_norm(alpha: torch.Tensor) -> float:
    """Mean normalized entropy of attention rows [B?, deg] (1 = uniform)."""
    deg = alpha.shape[-1]
    if deg < 2:
        return float("nan")
    h = -(alpha * torch.log(alpha + EPS)).sum(dim=-1) / np.log(deg)
    return float(h.mean())


def layer_stats(layer, x_flat: torch.Tensor, edge_index: torch.Tensor,
                logw: torch.Tensor, nbhd) -> tuple[dict, torch.Tensor]:
    """Attention statistics for one GAT layer, plus the layer output.

    Reproduces GATLayer.forward in eval mode (dropout inactive) so the three
    logit variants share the exact same Wx features.
    """
    src, dst = edge_index[0], edge_index[1]
    Wx = layer.W(x_flat)                                   # [B, N, D]
    e_feat = layer.leaky(
        (Wx[:, src, :] * layer.a_src).sum(-1) + (Wx[:, dst, :] * layer.a_dst).sum(-1)
    )                                                      # [B, E]
    e_full = e_feat + logw.unsqueeze(0)                    # [B, E]

    B = x_flat.shape[0]
    per = {k: [] for k in ("ent_full", "ent_dist", "ent_feat",
                           "std_feat", "tv", "argmax_shift")}
    std_dist = []
    out = torch.zeros(B, x_flat.shape[1], layer.out_dim)
    for j, ids in enumerate(nbhd):
        if len(ids) < 2:
            continue
        lw = logw[ids]                                     # [deg]
        ef = e_feat[:, ids]                                # [B, deg]
        a_full = torch.softmax(e_full[:, ids], dim=-1)     # [B, deg]
        a_dist = torch.softmax(lw, dim=-1)                 # [deg]
        a_feat = torch.softmax(ef, dim=-1)                 # [B, deg]

        per["ent_full"].append(entropy_norm(a_full))
        per["ent_dist"].append(entropy_norm(a_dist))
        per["ent_feat"].append(entropy_norm(a_feat))
        per["std_feat"].append(float(ef.std(dim=-1).mean()))
        std_dist.append(float(lw.std()))
        per["tv"].append(float(0.5 * (a_full - a_dist.unsqueeze(0)).abs().sum(-1).mean()))
        per["argmax_shift"].append(
            float((a_full.argmax(-1) != int(a_dist.argmax())).float().mean())
        )
        # Aggregate with the full attention, as the layer itself does.
        out[:, j, :] = (a_full.unsqueeze(-1) * Wx[:, src[ids], :]).sum(dim=1)

    stats = {k: float(np.mean(v)) for k, v in per.items()}
    stats["std_dist"] = float(np.mean(std_dist))
    stats["std_ratio_feat_over_dist"] = stats["std_feat"] / max(stats["std_dist"], EPS)
    return stats, out


def run_condition(name: str, city, ckpt: Path) -> dict:
    set_seed(42)
    F = city.feature_tensor.shape[-1]
    model = build_backbone("gat", F, fixed=True)
    load_checkpoint(model, ckpt)
    model.eval()

    edge_index, edge_attr = build_city_graph(city.coords, strategy=GRAPH_STRATEGY, k=KNN_K)
    w = edge_attr.squeeze(-1)
    logw = torch.log(w.clamp(min=1e-8))
    nbhd = neighbourhoods(edge_index, len(city.coords))

    X, _, _ = make_windows_masked(
        city.feature_tensor, city.target_tensor, city.test_mask,
        history=HISTORY, horizon=HORIZON, climatology_tensor=city.climatology_tensor,
    )
    stride = max(1, X.shape[0] // MAX_WINDOWS)
    X = X[::stride][:MAX_WINDOWS]
    x = torch.from_numpy(X).float()

    with torch.no_grad():
        z = model.t1(x)                                    # [B, H, N, D]
        B, H, N, D = z.shape
        z1 = z.reshape(B * H, N, D)
        s1, out1 = layer_stats(model.gat1, z1, edge_index, logw, nbhd)
        z2 = torch.nn.functional.elu(out1)
        s2, _ = layer_stats(model.gat2, z2, edge_index, logw, nbhd)

    res = {
        "checkpoint": str(ckpt), "N": int(N), "graph_instances": int(B * H),
        "logw_min": float(logw.min()), "logw_max": float(logw.max()),
        "gat1": s1, "gat2": s2,
    }
    print(f"\n{name}  (N={N}, log-w range [{res['logw_min']:.2f}, {res['logw_max']:.2f}])")
    for lname, s in (("gat1", s1), ("gat2", s2)):
        print(f"  {lname}: std(feat)={s['std_feat']:.2f} vs std(dist)={s['std_dist']:.2f} "
              f"(ratio {s['std_ratio_feat_over_dist']:.2f})  "
              f"entropy full/dist/feat = {s['ent_full']:.3f}/{s['ent_dist']:.3f}/{s['ent_feat']:.3f}  "
              f"TV(full,dist)={s['tv']:.3f}  argmax-shift={s['argmax_shift']:.2f}")
    return res


def main():
    cities = load_all_cities(fixed=True)
    conditions = [
        ("Delhi (source)",    cities["Delhi"],    Path("models/gnn/gat_source_Delhi_fixed.pt")),
        ("Kolkata (source)",  cities["Kolkata"],  Path("models/gnn/gat_source_Kolkata_fixed.pt")),
        ("Guwahati (source)", cities["Guwahati"], Path("models/gnn/gat_source_Guwahati_fixed.pt")),
        ("Delhi->Guwahati TL d=30", cities["Guwahati"],
         Path("models/gnn_tl/gat_tl_Delhi_to_Guwahati_d30_fixed.pt")),
        ("Delhi->Kolkata TL d=30", cities["Kolkata"],
         Path("models/gnn_tl/gat_tl_Delhi_to_Kolkata_d30_fixed.pt")),
    ]
    results = {name: run_condition(name, city, ckpt) for name, city, ckpt in conditions}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
