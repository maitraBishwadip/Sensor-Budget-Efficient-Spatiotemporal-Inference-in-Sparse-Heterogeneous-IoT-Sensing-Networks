"""Robustness battery.

Loads every saved checkpoint and runs four kinds of evaluation per city:

  1. Standard test-partition metrics (R2, MAE, RMSE, MAPE)
  2. Seasonality split: separate metrics on Winter / Spring / Summer / Monsoon
     test-set windows. A robust model should not collapse in any single
     season — we flag negative-R2 buckets.
  3. Cross-city zero-shot: take a model trained on city X and evaluate it
     on city Y's test partition with no fine-tune. Catches over-fitting to
     one city's topology.
  4. Per-station error breakdown for the target city: surfaces whether the
     gain comes from a few well-monitored stations or is uniform.

Writes a single combined JSON to results/robustness.json plus a pretty
console table.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import torch

from src.graph_construction import build_city_graph
from src.models.lstm_baseline import LSTMForecaster
from src.train_gnn import (
    BATCH_SIZE, DROPOUT, GAT_DIM, GRAPH_STRATEGY, HIDDEN, HISTORY, HORIZON,
    KNN_K, build_backbone, evaluate_gnn, load_all_cities, make_city_loader,
)
from src.train_lstm import (
    HIDDEN as LSTM_HIDDEN, LAYERS, DROPOUT as LSTM_DROPOUT,
    build_station_dataset, evaluate as evaluate_lstm,
)
from src.utils import (
    CityTensors, all_metrics, inverse_transform_target, load_checkpoint,
    make_windows, write_results,
)


PROCESSED_DIR = Path("dataset/processed")
MODELS_DIR = Path("models")
RESULTS_DIR = Path("results")


# --------------------------------------------------------------------------- #
# Seasonal mask
# --------------------------------------------------------------------------- #

def season_of(month: int) -> str:
    if month in (12, 1, 2, 10, 11):
        return "Winter"
    if month in (3, 4):
        return "Spring"
    if month in (5, 6):
        return "Summer"
    return "Monsoon"


def seasonal_split_indices(
    city: CityTensors,
    start_idx: int,
    end_idx: int,
) -> Dict[str, np.ndarray]:
    """Returns dict {season -> window-index-array} aligned to make_windows()
    output for the same (start_idx, end_idx) range."""
    ts = city.timestamps
    # window i corresponds to timestamps[start + i + HISTORY + HORIZON - 1]
    target_positions = np.arange(start_idx, end_idx - HISTORY - HORIZON + 1) + HISTORY + HORIZON - 1
    months = pd.DatetimeIndex(ts).month[target_positions].to_numpy()
    seasons = np.array([season_of(int(m)) for m in months])
    return {s: np.where(seasons == s)[0] for s in ["Winter", "Spring", "Summer", "Monsoon"]}


# --------------------------------------------------------------------------- #
# Per-station error breakdown
# --------------------------------------------------------------------------- #

def per_station_breakdown(
    pred: np.ndarray,    # [W, N]
    true: np.ndarray,    # [W, N]
    stations: List[str],
    target_scaler,
) -> Dict[str, Dict[str, float]]:
    pred_u = inverse_transform_target(target_scaler, pred)
    true_u = inverse_transform_target(target_scaler, true)
    out = {}
    for j, s in enumerate(stations):
        out[s] = all_metrics(true_u[:, j], pred_u[:, j])
    return out


# --------------------------------------------------------------------------- #
# GNN evaluation utility variants
# --------------------------------------------------------------------------- #

def gnn_predict_all(
    model: torch.nn.Module,
    X: np.ndarray,
    edge_index: torch.Tensor,
    edge_weight: torch.Tensor,
    batch_size: int = 64,
) -> np.ndarray:
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, X.shape[0], batch_size):
            xb = torch.from_numpy(X[i : i + batch_size]).float()
            p = model(xb, edge_index, edge_weight).cpu().numpy()
            preds.append(p)
    if not preds:
        return np.zeros((0, X.shape[2] if X.ndim == 4 else 0), dtype=np.float32)
    return np.concatenate(preds, axis=0)


# --------------------------------------------------------------------------- #
# Audit functions
# --------------------------------------------------------------------------- #

def audit_gnn_source(city: CityTensors, ckpt_path: Path, backbone: str) -> Dict[str, Any]:
    F = city.feature_tensor.shape[-1]
    model = build_backbone(backbone, F)
    load_checkpoint(model, ckpt_path)
    edge_index, edge_weight = build_city_graph(city.coords, strategy=GRAPH_STRATEGY, k=KNN_K)

    _, X_te, Y_te = make_city_loader(city, city.val_end_idx, city.test_end_idx, BATCH_SIZE, False)
    standard = evaluate_gnn(model, X_te, Y_te, edge_index, edge_weight, city.target_scaler)

    # Seasonal split.
    seasonal_idx = seasonal_split_indices(city, city.val_end_idx, city.test_end_idx)
    seasonal_metrics = {}
    if X_te.shape[0] > 0:
        pred_te = gnn_predict_all(model, X_te, edge_index, edge_weight)
        for season, sel in seasonal_idx.items():
            if sel.size == 0:
                seasonal_metrics[season] = None
                continue
            seasonal_metrics[season] = all_metrics(
                inverse_transform_target(city.target_scaler, Y_te[sel].reshape(-1)),
                inverse_transform_target(city.target_scaler, pred_te[sel].reshape(-1)),
            )

        per_station = per_station_breakdown(pred_te, Y_te, city.stations, city.target_scaler)
    else:
        per_station = {}

    return {
        "standard_test": standard,
        "seasonal": seasonal_metrics,
        "per_station": per_station,
    }


def audit_gnn_zero_shot_cross_city(
    source_city: CityTensors,
    target_city: CityTensors,
    source_ckpt: Path,
    backbone: str,
) -> Dict[str, float]:
    """Apply a city-X-trained model to city-Y's test set with no fine-tune."""
    F = source_city.feature_tensor.shape[-1]
    model = build_backbone(backbone, F)
    load_checkpoint(model, source_ckpt)
    edge_index, edge_weight = build_city_graph(target_city.coords, strategy=GRAPH_STRATEGY, k=KNN_K)
    _, X_te, Y_te = make_city_loader(
        target_city, target_city.val_end_idx, target_city.test_end_idx, BATCH_SIZE, False
    )
    return evaluate_gnn(model, X_te, Y_te, edge_index, edge_weight, target_city.target_scaler)


def audit_lstm_source(city: CityTensors, ckpt_path: Path) -> Dict[str, Any]:
    F = city.feature_tensor.shape[-1]
    model = LSTMForecaster(F, LSTM_HIDDEN, LAYERS, LSTM_DROPOUT)
    load_checkpoint(model, ckpt_path)
    X_te, Y_te = build_station_dataset(city, city.val_end_idx, city.test_end_idx)
    standard = evaluate_lstm(model, X_te, Y_te, city.target_scaler)
    return {"standard_test": standard}


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #

def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    cities = load_all_cities()
    audit: Dict[str, Any] = {"lstm_source": {}, "gnn_source": {}, "gnn_zero_shot_cross_city": {}}

    # LSTM source-only audits
    for name in ["Delhi", "Kolkata", "Guwahati"]:
        ckpt = MODELS_DIR / "lstm" / f"lstm_source_{name}.pt"
        if ckpt.exists():
            audit["lstm_source"][name] = audit_lstm_source(cities[name], ckpt)

    # GNN source-only audits
    backbone = "gat"
    for name in ["Delhi", "Kolkata", "Guwahati"]:
        ckpt = MODELS_DIR / "gnn" / f"{backbone}_source_{name}.pt"
        if ckpt.exists():
            audit["gnn_source"][name] = audit_gnn_source(cities[name], ckpt, backbone)

    # GNN zero-shot cross-city
    for src in ["Delhi", "Kolkata", "Guwahati"]:
        ckpt = MODELS_DIR / "gnn" / f"{backbone}_source_{src}.pt"
        if not ckpt.exists():
            continue
        audit["gnn_zero_shot_cross_city"][src] = {}
        for tgt in ["Delhi", "Kolkata", "Guwahati"]:
            if tgt == src:
                continue
            audit["gnn_zero_shot_cross_city"][src][tgt] = audit_gnn_zero_shot_cross_city(
                cities[src], cities[tgt], ckpt, backbone
            )

    out_path = RESULTS_DIR / "robustness.json"
    write_results(out_path, audit)
    print(f"wrote {out_path}\n")

    # Console summary.
    print("==== LSTM source-only test metrics ====")
    for c, m in audit["lstm_source"].items():
        print(f"  {c:8s}  R2={m['standard_test']['R2']:.4f}  MAE={m['standard_test']['MAE']:.4f}")
    print("\n==== GNN source-only test metrics ====")
    for c, m in audit["gnn_source"].items():
        print(f"  {c:8s}  R2={m['standard_test']['R2']:.4f}  MAE={m['standard_test']['MAE']:.4f}")
        for season, sm in m["seasonal"].items():
            if sm is None:
                continue
            print(f"     {season:7s}  R2={sm['R2']:+.4f}  MAE={sm['MAE']:.4f}")
    print("\n==== GNN zero-shot cross-city ====")
    for src, dct in audit["gnn_zero_shot_cross_city"].items():
        for tgt, m in dct.items():
            print(f"  {src} -> {tgt:8s}  R2={m['R2']:+.4f}  MAE={m['MAE']:.4f}")


if __name__ == "__main__":
    main()
