"""Export REAL size-invariant graph embeddings for the city-invariance figure.

Compares the pooled graph embedding z = [mean||max] for the three cities under:
  (A) the source-pretrained encoder  (no adversarial alignment)   <- "before"
  (B) the Graph-DANN encoder         (adversarial alignment)      <- "after"

Also reports a linear-probe city-classification accuracy on the raw 128-d
embeddings (5-fold logistic regression): high for (A), near chance 1/3 for (B)
is the quantitative signature of city-invariance.

Run:  python paper_figs/export_embeddings.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.graph_construction import build_city_graph                       # noqa: E402
from src.train_gnn import (                                               # noqa: E402
    GRAPH_STRATEGY, KNN_K, build_backbone, load_all_cities, make_city_loader_masked,
)
from src.utils import load_checkpoint                                     # noqa: E402
from sklearn.linear_model import LogisticRegression                       # noqa: E402
from sklearn.model_selection import cross_val_score                       # noqa: E402
from sklearn.preprocessing import StandardScaler                          # noqa: E402

OUT = ROOT / "paper_figs" / "_assets"
CITIES = ["Delhi", "Kolkata", "Guwahati"]
CAP = 300  # windows per city for a balanced, fast t-SNE


def embeddings(encoder, city, norm=None, cap=CAP):
    ei, ew = build_city_graph(city.coords, strategy=GRAPH_STRATEGY, k=KNN_K)
    _, X, _, _ = make_city_loader_masked(city, "test", 64, shuffle=False)
    if X.shape[0] > cap:
        idx = np.linspace(0, X.shape[0] - 1, cap).astype(int)
        X = X[idx]
    out = []
    encoder.eval()
    with torch.no_grad():
        for i in range(0, X.shape[0], 64):
            xb = torch.from_numpy(X[i:i + 64]).float()
            _, z = encoder(xb, ei, ew, return_embedding=True)
            if norm is not None:
                z = norm(z)
            out.append(z.numpy())
    return np.concatenate(out, axis=0)


def probe(Z, y):
    Zs = StandardScaler().fit_transform(Z)
    acc = cross_val_score(LogisticRegression(max_iter=2000, C=1.0),
                          Zs, y, cv=5).mean()
    return float(acc)


def main():
    cities = load_all_cities(fixed=True)

    # (A) source encoder (Delhi-pretrained, no alignment)
    enc_src = build_backbone("gat", 14, fixed=True)
    load_checkpoint(enc_src, ROOT / "models/gnn/gat_source_Delhi_fixed.pt")

    # (B) Graph-DANN encoder (Delhi->Kolkata joint), encoder + embed_norm only
    blob = torch.load(ROOT / "models/gnn_dann/gat_dann_joint_Delhi_to_Kolkata_fixed.pt",
                      map_location="cpu", weights_only=False)
    sd = blob["state_dict"]
    enc_dann = build_backbone("gat", 14, fixed=True)
    enc_dann.load_state_dict({k[len("encoder."):]: v for k, v in sd.items()
                              if k.startswith("encoder.")})
    embed_norm = nn.LayerNorm(enc_dann.embedding_dim)
    embed_norm.load_state_dict({k[len("embed_norm."):]: v for k, v in sd.items()
                                if k.startswith("embed_norm.")})

    Zs_src, Zs_dann, labels = [], [], []
    for ci, name in enumerate(CITIES):
        za = embeddings(enc_src, cities[name])
        zb = embeddings(enc_dann, cities[name], norm=embed_norm)
        n = min(len(za), len(zb))
        Zs_src.append(za[:n]); Zs_dann.append(zb[:n])
        labels.append(np.full(n, ci))
        print(f"  {name:9s}: {n} windows")
    Z_src = np.concatenate(Zs_src); Z_dann = np.concatenate(Zs_dann)
    y = np.concatenate(labels)

    acc_src = probe(Z_src, y)
    acc_dann = probe(Z_dann, y)
    print(f"\nlinear-probe city accuracy  (chance = {1/3:.2f}):")
    print(f"  source encoder (no align) : {acc_src:.3f}")
    print(f"  Graph-DANN encoder        : {acc_dann:.3f}")

    np.savez(OUT / "embeddings_cityinvariance.npz",
             Z_src=Z_src.astype(np.float32), Z_dann=Z_dann.astype(np.float32),
             y=y.astype(np.int64), cities=np.array(CITIES),
             acc_src=acc_src, acc_dann=acc_dann)
    print(f"saved -> {OUT / 'embeddings_cityinvariance.npz'}")


if __name__ == "__main__":
    main()
