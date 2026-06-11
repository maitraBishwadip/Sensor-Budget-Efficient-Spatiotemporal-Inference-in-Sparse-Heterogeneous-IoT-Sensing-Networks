"""Train base ST-GNN models per city (no transfer).

For each city, builds a per-city graph from station coordinates, trains the
chosen backbone (GAT or GraphSAGE), evaluates on train/val/test, and saves
the checkpoint to models/gnn/. Results land in results/gnn/.

Two modes (selected via `--fixed`):

  - Default (legacy):    EPOCHS=10, HIDDEN=24, chronological split, no
                         climatology, uniform LR. Matches the original numbers
                         reported in results/gnn/gnn_gat_source_only.json.

  - `--fixed` (default-on for new runs): per-city configs that mirror the
    LSTM recipe (epochs 25/40/60 with early stopping, hidden=64, bigger
    batches, per-city LR), interleaved train/val/test split, and per-(month,
    hour) climatology residual normalization. Output lands in
    results/gnn/gnn_<backbone>_source_only_fixed.json so the old numbers
    remain on disk for direct comparison.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn

from src.graph_construction import build_city_graph
from src.models.stgnn_gat import STGNN_GAT, STGNN_GAT_GRU
from src.models.stgnn_sage import STGNN_SAGE
from src.utils import (
    CityTensors,
    all_metrics,
    inverse_transform_target,
    load_city_tensor,
    make_windows,
    make_windows_masked,
    save_checkpoint,
    load_checkpoint,
    set_seed,
    write_results,
)


PROCESSED_DIR = Path("dataset/processed")
MODELS_DIR = Path("models/gnn")
RESULTS_DIR = Path("results/gnn")

HISTORY = 8
HORIZON = 1
HIDDEN = 24
GAT_DIM = 24
DROPOUT = 0.1
BATCH_SIZE = 32
EPOCHS = 10            # reduced for CPU compute economy
LR = 1e-3
WEIGHT_DECAY = 1e-5
DEVICE = "cpu"
GRAPH_STRATEGY = "knn"
KNN_K = 3
TRAIN_SUBSAMPLE_MAX = 4000   # cap train windows for fast iteration on Delhi


# --------------------------------------------------------------------------- #
# `--fixed` training recipe — mirrors the per-city LSTM configuration so the
# GNN gets a fair shot at converging. Lifts hidden dim, gives early stopping,
# uses per-city LR and epochs.
# --------------------------------------------------------------------------- #

FIXED_CITY_CFG = {
    "Delhi":    {"epochs": 25, "batch_size": 128, "lr": 1e-3, "patience": 6,  "train_subsample_max": 20000},
    "Kolkata":  {"epochs": 40, "batch_size": 64,  "lr": 1e-3, "patience": 8,  "train_subsample_max": None},
    "Guwahati": {"epochs": 60, "batch_size": 32,  "lr": 5e-4, "patience": 10, "train_subsample_max": None},
}

FIXED_HIDDEN = 64
FIXED_GAT_DIM = 64
FIXED_DROPOUT = 0.15


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def build_backbone(name: str, in_features: int, *, fixed: bool = False) -> nn.Module:
    name = name.lower()
    hd = FIXED_HIDDEN if fixed else HIDDEN
    gd = FIXED_GAT_DIM if fixed else GAT_DIM
    dp = FIXED_DROPOUT if fixed else DROPOUT
    if name == "gat":
        return STGNN_GAT(in_features, hidden_dim=hd, gat_dim=gd, dropout=dp)
    if name == "gatgru":
        return STGNN_GAT_GRU(in_features, hidden_dim=hd, gat_dim=gd, dropout=dp)
    if name == "sage":
        return STGNN_SAGE(in_features, hidden_dim=hd, sage_dim=gd, dropout=dp)
    raise ValueError(f"unknown backbone: {name}")


def make_city_loader(
    city: CityTensors,
    start_idx: int,
    end_idx: int,
    batch_size: int,
    shuffle: bool,
    subsample_max: int | None = None,
):
    X, Y = make_windows(city.feature_tensor, city.target_tensor, HISTORY, HORIZON, start_idx, end_idx)
    if X.shape[0] == 0:
        return None, X, Y
    if subsample_max is not None and X.shape[0] > subsample_max:
        # Deterministic uniform subsample by stride to keep coverage.
        stride = max(1, X.shape[0] // subsample_max)
        X = X[::stride][:subsample_max]
        Y = Y[::stride][:subsample_max]
    X_t = torch.from_numpy(X).float()
    Y_t = torch.from_numpy(Y).float()
    ds = torch.utils.data.TensorDataset(X_t, Y_t)
    loader = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=0)
    return loader, X, Y


def make_city_loader_masked(
    city: CityTensors,
    split: str,                   # "train" | "val" | "test"
    batch_size: int,
    shuffle: bool,
    subsample_max: int | None = None,
):
    """Mask-based windowing. Returns (loader, X, Y, C) where C is the
    climatology at the prediction step (None if city has no climatology).
    """
    mask = getattr(city, f"{split}_mask")
    X, Y, C = make_windows_masked(
        city.feature_tensor, city.target_tensor, mask,
        history=HISTORY, horizon=HORIZON,
        climatology_tensor=city.climatology_tensor,
    )
    if X.shape[0] == 0:
        return None, X, Y, C
    if subsample_max is not None and X.shape[0] > subsample_max:
        stride = max(1, X.shape[0] // subsample_max)
        X = X[::stride][:subsample_max]
        Y = Y[::stride][:subsample_max]
        if C is not None:
            C = C[::stride][:subsample_max]
    X_t = torch.from_numpy(X).float()
    Y_t = torch.from_numpy(Y).float()
    ds = torch.utils.data.TensorDataset(X_t, Y_t)
    loader = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=0)
    return loader, X, Y, C


def train_one_epoch_gnn(model, loader, opt, loss_fn, edge_index, edge_weight) -> float:
    model.train()
    total = 0.0
    n = 0
    for xb, yb in loader:
        opt.zero_grad()
        pred = model(xb, edge_index, edge_weight)
        loss = loss_fn(pred, yb)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        total += float(loss.item()) * xb.size(0)
        n += xb.size(0)
    return total / max(n, 1)


def evaluate_gnn(
    model,
    X: np.ndarray,
    Y: np.ndarray,
    edge_index,
    edge_weight,
    target_scaler,
    batch_size: int = 64,
    climatology: np.ndarray | None = None,
) -> Dict[str, float]:
    """Evaluate the model in *raw PM2.5* space.

    If `climatology` is provided, treat Y and model outputs as z-scored
    residuals; add climatology back after inverse z-score.
    """
    if X.shape[0] == 0:
        return {"R2": float("nan"), "MAE": float("nan"), "RMSE": float("nan"), "MAPE": float("nan")}
    model.eval()
    preds_all, trues_all = [], []
    with torch.no_grad():
        for i in range(0, X.shape[0], batch_size):
            xb = torch.from_numpy(X[i : i + batch_size]).float()
            p = model(xb, edge_index, edge_weight).cpu().numpy()
            preds_all.append(p)
            trues_all.append(Y[i : i + batch_size])
    y_pred_norm = np.concatenate(preds_all, axis=0)
    y_true_norm = np.concatenate(trues_all, axis=0)
    clim = climatology if climatology is not None else None
    y_true = inverse_transform_target(target_scaler, y_true_norm.reshape(-1), clim.reshape(-1) if clim is not None else None)
    y_pred = inverse_transform_target(target_scaler, y_pred_norm.reshape(-1), clim.reshape(-1) if clim is not None else None)
    return all_metrics(y_true, y_pred)


def train_base_gnn(
    city: CityTensors,
    backbone: str = "gat",
    *,
    fixed: bool = False,
) -> Tuple[Dict[str, float], Path]:
    """Train one source-only ST-GNN. With `fixed=True` switches to per-city LSTM-
    grade hyperparameters, mask-based windowing, and climatology-aware eval.
    """
    set_seed(42)
    F = city.feature_tensor.shape[-1]
    model = build_backbone(backbone, F, fixed=fixed).to(DEVICE)
    edge_index, edge_weight = build_city_graph(city.coords, strategy=GRAPH_STRATEGY, k=KNN_K)
    print(f"[{city.city}/{backbone}{' /FIX' if fixed else ''}] graph: |V|={len(city.coords)}  |E|={edge_index.shape[1]}  "
          f"params={sum(p.numel() for p in model.parameters())}")

    if fixed:
        cfg = FIXED_CITY_CFG[city.city]
        train_loader, _, _, _ = make_city_loader_masked(
            city, "train", cfg["batch_size"], shuffle=True, subsample_max=cfg["train_subsample_max"],
        )
        _, X_va, Y_va, C_va = make_city_loader_masked(city, "val", cfg["batch_size"], shuffle=False)
        _, X_te, Y_te, C_te = make_city_loader_masked(city, "test", cfg["batch_size"], shuffle=False)
        epochs = cfg["epochs"]
        lr = cfg["lr"]
        patience = cfg["patience"]
    else:
        train_loader, _, _ = make_city_loader(
            city, 0, city.train_end_idx, BATCH_SIZE, True, subsample_max=TRAIN_SUBSAMPLE_MAX,
        )
        _, X_va, Y_va = make_city_loader(city, city.train_end_idx, city.val_end_idx, BATCH_SIZE, False)
        _, X_te, Y_te = make_city_loader(city, city.val_end_idx, city.test_end_idx, BATCH_SIZE, False)
        C_va = C_te = None
        epochs, lr, patience = EPOCHS, LR, None

    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=3) if fixed else None
    loss_fn = nn.MSELoss()
    best_val = -1e9
    patience_left = patience if patience is not None else None
    ckpt_suffix = "_fixed" if fixed else ""
    ckpt = MODELS_DIR / f"{backbone}_source_{city.city}{ckpt_suffix}.pt"

    for ep in range(epochs):
        t_ep = time.time()
        tr = train_one_epoch_gnn(model, train_loader, opt, loss_fn, edge_index, edge_weight)
        val_metrics = evaluate_gnn(model, X_va, Y_va, edge_index, edge_weight, city.target_scaler, climatology=C_va)
        if scheduler is not None:
            scheduler.step(val_metrics["R2"])
        improved = val_metrics["R2"] > best_val + 1e-4
        if improved:
            best_val = val_metrics["R2"]
            save_checkpoint(model, ckpt, extra={"val_r2": best_val, "feature_dim": F, "backbone": backbone})
            if patience is not None:
                patience_left = patience
        elif patience is not None:
            patience_left -= 1
        flag = "*" if improved else " "
        lr_now = opt.param_groups[0]["lr"]
        print(f"  ep {ep+1:02d}/{epochs} {flag} loss={tr:.4f}  val_R2={val_metrics['R2']:+.4f}  "
              f"val_MAE={val_metrics['MAE']:.3f}  lr={lr_now:.2e}  "
              f"patience={patience_left}  ({time.time()-t_ep:.1f}s)")
        if patience is not None and patience_left is not None and patience_left <= 0:
            print(f"  early-stopped at epoch {ep+1} (best val R2={best_val:.4f})")
            break

    load_checkpoint(model, ckpt)
    test_metrics = evaluate_gnn(model, X_te, Y_te, edge_index, edge_weight, city.target_scaler, climatology=C_te)
    print(f"[{city.city}/{backbone}] test = {test_metrics}")
    return test_metrics, ckpt


def load_all_cities(*, fixed: bool = False) -> Dict[str, CityTensors]:
    with open(PROCESSED_DIR / "metadata.json") as fh:
        metas = json.load(fh)
    cities: Dict[str, CityTensors] = {}
    for name in ["Delhi", "Kolkata", "Guwahati"]:
        cities[name] = load_city_tensor(
            PROCESSED_DIR / f"{name.lower()}_processed.csv",
            metas[name],
            metas[name]["feature_cols"],
            metas[name]["target_col"],
            interleaved_split=fixed,
            climatology_residual=fixed,
        )
    return cities


def main(args):
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    cities = load_all_cities(fixed=args.fixed)
    results = {}
    for name in ["Delhi", "Kolkata", "Guwahati"]:
        metrics, _ = train_base_gnn(cities[name], backbone=args.backbone, fixed=args.fixed)
        results[name] = metrics
    suffix = "_fixed" if args.fixed else ""
    out = RESULTS_DIR / f"gnn_{args.backbone}_source_only{suffix}.json"
    write_results(out, results)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--backbone", choices=["gat", "gatgru", "sage"], default="gat")
    p.add_argument("--fixed", action="store_true",
                   help="Enable interleaved split + climatology residual + per-city LSTM-grade recipe.")
    main(p.parse_args())
