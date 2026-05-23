"""Variant B (headline): Graph-DANN.

Trains the inductive ST-GNN jointly on source + target with the gradient-
reversal adversarial city classifier. Then performs the same d% target
fine-tune used by Variant A. Evaluated on the target's test partition.

Loss per mini-batch:
    L = MSE(ŷ, y_pm25) − λ(p) · CE(ĉ, c_city)

where the negative is realized by the GRL on the forward path of the city
classifier branch.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.graph_construction import build_city_graph
from src.models.dann import GraphDANN
from src.train_gnn import (
    BATCH_SIZE, DEVICE, DROPOUT, GAT_DIM, GRAPH_STRATEGY, HIDDEN,
    HISTORY, HORIZON, KNN_K, build_backbone, evaluate_gnn,
    load_all_cities, make_city_loader,
)
from src.utils import (
    CityTensors,
    inverse_transform_target,
    load_checkpoint,
    make_windows,
    save_checkpoint,
    set_seed,
    write_results,
)


MODELS_DIR = Path("models/gnn_dann")
RESULTS_DIR = Path("results/gnn_dann")

EPOCHS_JOINT = 8        # reduced for CPU compute economy
EPOCHS_FT = 6
LR = 1e-3
FT_LR = 1e-4
WEIGHT_DECAY = 1e-5
DANN_GAMMA = 10.0


# --------------------------------------------------------------------------- #
# Pair-wise windowing across all cities — needed so the city classifier sees
# examples from every city in every mini-batch.
# --------------------------------------------------------------------------- #

CITY_TO_LABEL = {"Delhi": 0, "Kolkata": 1, "Guwahati": 2}


DANN_SUBSAMPLE_MAX = 2000   # cap per-city training windows for DANN


def build_per_city_train_arrays(
    cities: Dict[str, CityTensors],
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    out = {}
    for name, c in cities.items():
        X, Y = make_windows(c.feature_tensor, c.target_tensor, HISTORY, HORIZON, 0, c.train_end_idx)
        if X.shape[0] > DANN_SUBSAMPLE_MAX:
            stride = max(1, X.shape[0] // DANN_SUBSAMPLE_MAX)
            X = X[::stride][:DANN_SUBSAMPLE_MAX]
            Y = Y[::stride][:DANN_SUBSAMPLE_MAX]
        out[name] = (X, Y)
        print(f"  [{name}] train windows (after subsample): {X.shape}")
    return out


def build_per_city_edges(
    cities: Dict[str, CityTensors],
) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
    return {
        name: build_city_graph(c.coords, strategy=GRAPH_STRATEGY, k=KNN_K)
        for name, c in cities.items()
    }


def sample_minibatch(
    per_city_windows: Dict[str, Tuple[np.ndarray, np.ndarray]],
    per_city_indices: Dict[str, np.ndarray],
    per_city_cursors: Dict[str, int],
    batch_per_city: int,
    rng: np.random.Generator,
) -> Tuple[Dict[str, np.ndarray], Dict[str, int]]:
    """Round-robin one mini-batch per city. Returns dict of city -> (xb, yb)
    and updated cursors. Reshuffles when a city's cursor wraps around."""
    out_x: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    new_cursors = dict(per_city_cursors)
    for name, (X, Y) in per_city_windows.items():
        if X.shape[0] == 0:
            continue
        idx = per_city_indices[name]
        cur = new_cursors[name]
        if cur + batch_per_city > len(idx):
            rng.shuffle(idx)
            cur = 0
        sel = idx[cur : cur + batch_per_city]
        new_cursors[name] = cur + batch_per_city
        out_x[name] = (X[sel], Y[sel])
    return out_x, new_cursors


def lambda_progress(global_step: int, total_steps: int, gamma: float = DANN_GAMMA) -> float:
    p = max(0.0, min(1.0, global_step / max(1, total_steps)))
    val = 2.0 / (1.0 + float(np.exp(-gamma * p))) - 1.0
    return float(val)


# --------------------------------------------------------------------------- #
# Train
# --------------------------------------------------------------------------- #

def train_graph_dann(
    cities: Dict[str, CityTensors],
    source: str,
    target: str,
    d_pct: float,
    backbone: str = "gat",
) -> Dict[str, float]:
    set_seed(42)
    src = cities[source]
    tgt = cities[target]
    F = src.feature_tensor.shape[-1]
    encoder = build_backbone(backbone, F).to(DEVICE)
    model = GraphDANN(encoder, n_cities=3).to(DEVICE)

    edges = build_per_city_edges(cities)

    # Phase 1: joint training on source + (unlabeled-PM2.5 but labeled-city) target.
    per_city = build_per_city_train_arrays(cities)
    indices_by_city = {
        name: np.arange(per_city[name][0].shape[0])
        for name in per_city
        if per_city[name][0].shape[0] > 0
    }
    rng = np.random.default_rng(42)
    for arr in indices_by_city.values():
        rng.shuffle(arr)
    cursors = {name: 0 for name in indices_by_city}
    batch_per_city = max(8, BATCH_SIZE // 3)

    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    mse = nn.MSELoss()
    ce = nn.CrossEntropyLoss()

    # Steps per joint epoch is determined by the *source* city's data.
    steps_per_epoch = max(1, per_city[source][0].shape[0] // batch_per_city)
    total_steps = EPOCHS_JOINT * steps_per_epoch
    global_step = 0

    print(f"[Graph-DANN {source}->{target}] joint training: {EPOCHS_JOINT} ep × {steps_per_epoch} steps")
    for ep in range(EPOCHS_JOINT):
        model.train()
        sum_loss_y, sum_loss_d, n = 0.0, 0.0, 0
        for step in range(steps_per_epoch):
            lam = lambda_progress(global_step, total_steps)
            mb, cursors = sample_minibatch(per_city, indices_by_city, cursors, batch_per_city, rng)

            # Source supervised step.
            if source in mb:
                xb, yb = mb[source]
                xb_t = torch.from_numpy(xb).float()
                yb_t = torch.from_numpy(yb).float()
                ei, ew = edges[source]
                y_hat, city_logits = model(xb_t, ei, ew, lambda_=lam)
                loss_y = mse(y_hat, yb_t)
                lab = torch.full((xb_t.size(0),), CITY_TO_LABEL[source], dtype=torch.long)
                loss_d_src = ce(city_logits, lab)
            else:
                loss_y = torch.tensor(0.0)
                loss_d_src = torch.tensor(0.0)

            # Target adversarial step (unlabeled for PM2.5; labeled for city).
            if target in mb:
                xb, _ = mb[target]
                xb_t = torch.from_numpy(xb).float()
                ei, ew = edges[target]
                _, city_logits = model(xb_t, ei, ew, lambda_=lam)
                lab = torch.full((xb_t.size(0),), CITY_TO_LABEL[target], dtype=torch.long)
                loss_d_tgt = ce(city_logits, lab)
            else:
                loss_d_tgt = torch.tensor(0.0)

            # Third-city (replay) step also labels the discriminator — helps stability.
            third = [c for c in cities if c not in (source, target)]
            if third and third[0] in mb:
                xb, _ = mb[third[0]]
                xb_t = torch.from_numpy(xb).float()
                ei, ew = edges[third[0]]
                _, city_logits = model(xb_t, ei, ew, lambda_=lam)
                lab = torch.full((xb_t.size(0),), CITY_TO_LABEL[third[0]], dtype=torch.long)
                loss_d_third = ce(city_logits, lab)
            else:
                loss_d_third = torch.tensor(0.0)

            loss_d = loss_d_src + loss_d_tgt + loss_d_third
            loss = loss_y + loss_d            # GRL handles the sign on the encoder
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            sum_loss_y += float(loss_y.item())
            sum_loss_d += float(loss_d.item())
            n += 1
            global_step += 1

        print(f"  joint ep {ep+1:02d}/{EPOCHS_JOINT} L_y={sum_loss_y/max(n,1):.4f} L_d={sum_loss_d/max(n,1):.4f} lam={lam:.3f}")

    # Phase 2: target fine-tune on d% of target training data (Variant-A style).
    X_tgt, Y_tgt = per_city[target]
    keep = rng.choice(X_tgt.shape[0], size=max(1, int(d_pct * X_tgt.shape[0])), replace=False)
    keep.sort()
    X_ft, Y_ft = X_tgt[keep], Y_tgt[keep]
    ei, ew = edges[target]
    ds = torch.utils.data.TensorDataset(torch.from_numpy(X_ft).float(), torch.from_numpy(Y_ft).float())
    train_loader = torch.utils.data.DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    ft_opt = torch.optim.Adam(model.parameters(), lr=FT_LR, weight_decay=WEIGHT_DECAY)

    _, X_va, Y_va = make_city_loader(tgt, tgt.train_end_idx, tgt.val_end_idx, BATCH_SIZE, False)
    _, X_te, Y_te = make_city_loader(tgt, tgt.val_end_idx, tgt.test_end_idx, BATCH_SIZE, False)

    best_val = -1e9
    ckpt = MODELS_DIR / f"{backbone}_dann_{source}_to_{target}_d{int(d_pct*100)}.pt"
    for ep in range(EPOCHS_FT):
        model.train()
        for xb, yb in train_loader:
            ft_opt.zero_grad()
            y_hat, _ = model(xb, ei, ew, lambda_=0.0)   # turn DANN off during fine-tune
            loss = mse(y_hat, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            ft_opt.step()
        val_metrics = evaluate_gnn(model.encoder, X_va, Y_va, ei, ew, tgt.target_scaler)
        print(f"  ft ep {ep+1:02d}/{EPOCHS_FT}  val_R2={val_metrics['R2']:.4f}")
        if val_metrics["R2"] > best_val:
            best_val = val_metrics["R2"]
            save_checkpoint(model, ckpt, extra={"val_r2": best_val})

    load_checkpoint(model, ckpt)
    test_metrics = evaluate_gnn(model.encoder, X_te, Y_te, ei, ew, tgt.target_scaler)
    print(f"[Graph-DANN {source}->{target} d={d_pct:.0%}] test = {test_metrics}")
    return test_metrics


def main(args):
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    cities = load_all_cities()

    pairs = [
        ("Delhi", "Kolkata"),
        ("Delhi", "Guwahati"),
        ("Kolkata", "Guwahati"),
    ]
    d_values = [0.30]   # one d% per pair for compute economy
    results = {}
    for src, tgt in pairs:
        for d in d_values:
            key = f"{src}->{tgt}@{int(d*100)}"
            results[key] = train_graph_dann(cities, src, tgt, d, backbone=args.backbone)

    write_results(RESULTS_DIR / f"variantB_dann_{args.backbone}.json", results)
    print(f"\nwrote {RESULTS_DIR / f'variantB_dann_{args.backbone}.json'}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--backbone", choices=["gat", "sage"], default="gat")
    main(p.parse_args())
