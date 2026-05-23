"""Train the LSTM baseline (replicates thesis Tables 5.1 and 5.2).

Two stages:
  1) Source-only LSTM per city (Delhi, Kolkata, Guwahati) -- thesis Table 5.1.
  2) LSTM transfer learning: load source checkpoint, fine-tune on d% of the
     target city -- thesis Table 5.2 (Variant A applied to LSTM).

Per-city hyperparameters are tuned to reflect the dataset size:
  - Delhi (large): bigger batch, capped train sequences, fewer epochs.
  - Kolkata (medium): full data, moderate epochs.
  - Guwahati (small): full data, more epochs and lower LR for stable convergence.

Run:
    python -u -m src.train_lstm --mode source       # thesis Table 5.1
    python -u -m src.train_lstm --mode tl           # thesis Table 5.2
    python -u -m src.train_lstm --mode all
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.models.lstm_baseline import LSTMForecaster
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


# --------------------------------------------------------------------------- #
# Paths & global constants
# --------------------------------------------------------------------------- #

PROCESSED_DIR = Path("dataset/processed")
MODELS_DIR = Path("models/lstm")
RESULTS_DIR = Path("results/lstm")

HISTORY = 8         # 8 x 3h = 24h history (matches thesis)
HORIZON = 1         # 1 x 3h = 3h forecast (matches thesis)
HIDDEN = 64
LAYERS = 2
DROPOUT = 0.2
WEIGHT_DECAY = 1e-5
DEVICE = "cpu"


# Per-city training recipes. Smaller cities get more epochs + lower LR so
# the model has time to converge on the smaller dataset.
CITY_TRAIN_CFG = {
    "Delhi":    {"epochs": 25, "batch_size": 512, "lr": 1e-3, "train_subsample_max": 50000, "patience": 6},
    "Kolkata":  {"epochs": 40, "batch_size": 256, "lr": 1e-3, "train_subsample_max": None,  "patience": 8},
    "Guwahati": {"epochs": 60, "batch_size": 128, "lr": 5e-4, "train_subsample_max": None,  "patience": 10},
}

# Transfer-learning fine-tune hyperparameters per target.
CITY_FT_CFG = {
    "Delhi":    {"epochs": 25, "batch_size": 256, "lr": 2e-4, "patience": 6},
    "Kolkata":  {"epochs": 30, "batch_size": 256, "lr": 2e-4, "patience": 8},
    "Guwahati": {"epochs": 40, "batch_size": 128, "lr": 1e-4, "patience": 10},
}


# --------------------------------------------------------------------------- #
# Dataset construction (per-station sequences flattened into one tensor)
# --------------------------------------------------------------------------- #

def build_station_dataset(
    city: CityTensors,
    start_idx: int,
    end_idx: int,
    subsample_max: int | None = None,
    tag: str = "",
) -> Tuple[np.ndarray, np.ndarray]:
    """Legacy chronological window builder. Returns [W*N, H, F] and [W*N]."""
    X, Y = make_windows(
        city.feature_tensor, city.target_tensor,
        history=HISTORY, horizon=HORIZON,
        start_idx=start_idx, end_idx=end_idx,
    )
    if X.shape[0] == 0:
        return X.reshape(0, HISTORY, -1), Y.reshape(0)
    W, H, N, F = X.shape
    X_flat = X.transpose(0, 2, 1, 3).reshape(W * N, H, F)
    Y_flat = Y.reshape(W * N)
    if subsample_max is not None and X_flat.shape[0] > subsample_max:
        rng = np.random.default_rng(42)
        sel = rng.choice(X_flat.shape[0], size=subsample_max, replace=False)
        sel.sort()
        X_flat = X_flat[sel]
        Y_flat = Y_flat[sel]
        print(f"   [{tag}] subsampled {W*N} -> {X_flat.shape[0]} sequences")
    return X_flat, Y_flat


def build_station_dataset_masked(
    city: CityTensors,
    split: str,                  # "train" | "val" | "test"
    subsample_max: int | None = None,
    tag: str = "",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Mask-based window builder with optional climatology vector.

    Returns (X_flat[W*N,H,F], Y_flat[W*N], C_flat[W*N] or None).
    """
    mask = getattr(city, f"{split}_mask")
    X, Y, C = make_windows_masked(
        city.feature_tensor, city.target_tensor, mask,
        history=HISTORY, horizon=HORIZON,
        climatology_tensor=city.climatology_tensor,
    )
    if X.shape[0] == 0:
        return X.reshape(0, HISTORY, -1), Y.reshape(0), (None if C is None else C.reshape(0))
    W, H, N, F = X.shape
    X_flat = X.transpose(0, 2, 1, 3).reshape(W * N, H, F)
    Y_flat = Y.reshape(W * N)
    C_flat = None if C is None else C.reshape(W * N)
    if subsample_max is not None and X_flat.shape[0] > subsample_max:
        rng = np.random.default_rng(42)
        sel = rng.choice(X_flat.shape[0], size=subsample_max, replace=False)
        sel.sort()
        X_flat = X_flat[sel]
        Y_flat = Y_flat[sel]
        if C_flat is not None:
            C_flat = C_flat[sel]
        print(f"   [{tag}] subsampled {W*N} -> {X_flat.shape[0]} sequences")
    return X_flat, Y_flat, C_flat


def to_loader(X: np.ndarray, Y: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    ds = TensorDataset(torch.from_numpy(X), torch.from_numpy(Y).unsqueeze(-1))
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=0)


# --------------------------------------------------------------------------- #
# Train / eval
# --------------------------------------------------------------------------- #

def train_one_epoch(model, loader, opt, loss_fn) -> float:
    model.train()
    total, n = 0.0, 0
    for xb, yb in loader:
        xb = xb.to(DEVICE)
        yb = yb.to(DEVICE)
        opt.zero_grad()
        pred = model(xb)
        loss = loss_fn(pred, yb)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        total += float(loss.item()) * xb.size(0)
        n += xb.size(0)
    return total / max(n, 1)


def evaluate(
    model: nn.Module,
    X: np.ndarray,
    Y: np.ndarray,
    target_scaler,
    batch_size: int = 512,
    climatology: np.ndarray | None = None,
) -> Dict[str, float]:
    if X.shape[0] == 0:
        return {"R2": float("nan"), "MAE": float("nan"), "RMSE": float("nan"), "MAPE": float("nan")}
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, X.shape[0], batch_size):
            xb = torch.from_numpy(X[i : i + batch_size]).to(DEVICE)
            p = model(xb).cpu().numpy().reshape(-1)
            preds.append(p)
    y_pred_norm = np.concatenate(preds, axis=0)
    y_pred = inverse_transform_target(target_scaler, y_pred_norm, climatology)
    y_true = inverse_transform_target(target_scaler, Y, climatology)
    return all_metrics(y_true, y_pred)


def train_source_only(cities: Dict[str, CityTensors], target_city: str, *, fixed: bool = False) -> Dict[str, float]:
    """Train an LSTM from scratch on `target_city` (thesis Table 5.1 cell).
    With `fixed=True`, use interleaved split + climatology residual."""
    cfg = CITY_TRAIN_CFG[target_city]
    print(f"\n{'='*68}\n[LSTM SOURCE-ONLY {target_city}{' /FIX' if fixed else ''}]  cfg={cfg}\n{'='*68}")
    set_seed(42)

    city = cities[target_city]
    F = city.feature_tensor.shape[-1]
    print(f"   feature_dim={F}  stations={len(city.coords)}  T={city.feature_tensor.shape[0]}")

    print(f"   building train/val/test datasets ...")
    t0 = time.time()
    if fixed:
        X_tr, Y_tr, C_tr = build_station_dataset_masked(
            city, "train", subsample_max=cfg["train_subsample_max"], tag=f"{target_city}/train",
        )
        X_va, Y_va, C_va = build_station_dataset_masked(city, "val", tag=f"{target_city}/val")
        X_te, Y_te, C_te = build_station_dataset_masked(city, "test", tag=f"{target_city}/test")
    else:
        X_tr, Y_tr = build_station_dataset(
            city, 0, city.train_end_idx,
            subsample_max=cfg["train_subsample_max"], tag=f"{target_city}/train",
        )
        X_va, Y_va = build_station_dataset(city, city.train_end_idx, city.val_end_idx, tag=f"{target_city}/val")
        X_te, Y_te = build_station_dataset(city, city.val_end_idx, city.test_end_idx, tag=f"{target_city}/test")
        C_va = C_te = None
    print(f"   train={X_tr.shape}  val={X_va.shape}  test={X_te.shape}  (built in {time.time()-t0:.1f}s)")

    model = LSTMForecaster(F, HIDDEN, LAYERS, DROPOUT).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"   model: LSTMForecaster(hidden={HIDDEN}, layers={LAYERS}, dropout={DROPOUT}) -- {n_params} params")

    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=3)
    loss_fn = nn.MSELoss()
    train_loader = to_loader(X_tr, Y_tr, cfg["batch_size"], shuffle=True)

    best_val_r2 = -1e9
    patience_left = cfg["patience"]
    suffix = "_fixed" if fixed else ""
    ckpt = MODELS_DIR / f"lstm_source_{target_city}{suffix}.pt"
    epochs = cfg["epochs"]
    print(f"   training for up to {epochs} epochs with patience={cfg['patience']}")

    for ep in range(epochs):
        t_ep = time.time()
        tr_loss = train_one_epoch(model, train_loader, opt, loss_fn)
        val_metrics = evaluate(model, X_va, Y_va, city.target_scaler, climatology=C_va)
        scheduler.step(val_metrics["R2"])
        lr_now = opt.param_groups[0]["lr"]
        improved = val_metrics["R2"] > best_val_r2 + 1e-4
        if improved:
            best_val_r2 = val_metrics["R2"]
            save_checkpoint(model, ckpt, extra={"val_r2": best_val_r2, "feature_dim": F})
            patience_left = cfg["patience"]
        else:
            patience_left -= 1
        flag = "*" if improved else " "
        print(
            f"   ep {ep+1:02d}/{epochs} {flag} loss={tr_loss:.4f}  "
            f"val_R2={val_metrics['R2']:+.4f}  val_MAE={val_metrics['MAE']:.3f}  "
            f"lr={lr_now:.2e}  patience={patience_left}  ({time.time()-t_ep:.1f}s)"
        )
        if patience_left <= 0:
            print(f"   early-stopped at epoch {ep+1} (best val R2={best_val_r2:.4f})")
            break

    load_checkpoint(model, ckpt)
    test_metrics = evaluate(model, X_te, Y_te, city.target_scaler, climatology=C_te)
    print(f"   TEST {target_city}: {test_metrics}")
    return test_metrics


def train_transfer(
    cities: Dict[str, CityTensors],
    source: str,
    target: str,
    d_pct: float,
    *,
    fixed: bool = False,
) -> Dict[str, float]:
    """LSTM Variant-A TL: load source checkpoint, fine-tune on d% of target."""
    ft_cfg = CITY_FT_CFG[target]
    print(f"\n{'='*68}\n[LSTM TL {source} -> {target} d={d_pct:.0%}{' /FIX' if fixed else ''}]  ft_cfg={ft_cfg}\n{'='*68}")
    set_seed(42)

    src = cities[source]
    tgt = cities[target]
    F = src.feature_tensor.shape[-1]
    assert F == tgt.feature_tensor.shape[-1]

    model = LSTMForecaster(F, HIDDEN, LAYERS, DROPOUT).to(DEVICE)
    suffix = "_fixed" if fixed else ""
    src_ckpt = MODELS_DIR / f"lstm_source_{source}{suffix}.pt"
    if not src_ckpt.exists():
        raise RuntimeError(f"missing source checkpoint: {src_ckpt}")
    load_checkpoint(model, src_ckpt)
    print(f"   loaded source checkpoint: {src_ckpt}")

    print(f"   building target {target} d={d_pct:.0%} fine-tune set ...")
    if fixed:
        X_tgt_tr, Y_tgt_tr, C_tgt_tr = build_station_dataset_masked(tgt, "train", tag=f"{target}/train")
        X_va, Y_va, C_va = build_station_dataset_masked(tgt, "val", tag=f"{target}/val")
        X_te, Y_te, C_te = build_station_dataset_masked(tgt, "test", tag=f"{target}/test")
    else:
        X_tgt_tr, Y_tgt_tr = build_station_dataset(tgt, 0, tgt.train_end_idx, tag=f"{target}/train")
        C_tgt_tr = None
        X_va, Y_va = build_station_dataset(tgt, tgt.train_end_idx, tgt.val_end_idx)
        X_te, Y_te = build_station_dataset(tgt, tgt.val_end_idx, tgt.test_end_idx)
        C_va = C_te = None

    rng = np.random.default_rng(42)
    n_keep = max(1, int(d_pct * X_tgt_tr.shape[0]))
    keep = rng.choice(X_tgt_tr.shape[0], size=n_keep, replace=False)
    keep.sort()
    X_ft, Y_ft = X_tgt_tr[keep], Y_tgt_tr[keep]
    print(f"   fine-tune={X_ft.shape}  val={X_va.shape}  test={X_te.shape}")

    opt = torch.optim.Adam(model.parameters(), lr=ft_cfg["lr"], weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=3)
    loss_fn = nn.MSELoss()
    train_loader = to_loader(X_ft, Y_ft, ft_cfg["batch_size"], shuffle=True)

    # First report zero-shot test metrics.
    zs_metrics = evaluate(model, X_te, Y_te, tgt.target_scaler, climatology=C_te)
    print(f"   zero-shot (d=0%) on {target}: {zs_metrics}")

    best_val_r2 = -1e9
    patience_left = ft_cfg["patience"]
    ckpt = MODELS_DIR / f"lstm_tl_{source}_to_{target}_d{int(d_pct*100)}{suffix}.pt"
    epochs = ft_cfg["epochs"]
    print(f"   fine-tuning for up to {epochs} epochs with patience={ft_cfg['patience']}")
    for ep in range(epochs):
        t_ep = time.time()
        tr_loss = train_one_epoch(model, train_loader, opt, loss_fn)
        val_metrics = evaluate(model, X_va, Y_va, tgt.target_scaler, climatology=C_va)
        scheduler.step(val_metrics["R2"])
        lr_now = opt.param_groups[0]["lr"]
        improved = val_metrics["R2"] > best_val_r2 + 1e-4
        if improved:
            best_val_r2 = val_metrics["R2"]
            save_checkpoint(model, ckpt, extra={"val_r2": best_val_r2})
            patience_left = ft_cfg["patience"]
        else:
            patience_left -= 1
        flag = "*" if improved else " "
        print(
            f"   ep {ep+1:02d}/{epochs} {flag} loss={tr_loss:.4f}  "
            f"val_R2={val_metrics['R2']:+.4f}  val_MAE={val_metrics['MAE']:.3f}  "
            f"lr={lr_now:.2e}  patience={patience_left}  ({time.time()-t_ep:.1f}s)"
        )
        if patience_left <= 0:
            print(f"   early-stopped at epoch {ep+1} (best val R2={best_val_r2:.4f})")
            break

    load_checkpoint(model, ckpt)
    test_metrics = evaluate(model, X_te, Y_te, tgt.target_scaler, climatology=C_te)
    print(f"   TEST {source}->{target} d={d_pct:.0%}: {test_metrics}")
    return test_metrics


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #

def load_all_cities(*, fixed: bool = False) -> Dict[str, CityTensors]:
    print("="*68)
    print(f"LOADING ALL CITIES{' (FIXED PROTOCOL)' if fixed else ''}")
    print("="*68)
    with open(PROCESSED_DIR / "metadata.json") as fh:
        metas = json.load(fh)
    cities: Dict[str, CityTensors] = {}
    for name in ["Delhi", "Kolkata", "Guwahati"]:
        t0 = time.time()
        cities[name] = load_city_tensor(
            PROCESSED_DIR / f"{name.lower()}_processed.csv",
            metas[name],
            metas[name]["feature_cols"],
            metas[name]["target_col"],
            interleaved_split=fixed,
            climatology_residual=fixed,
        )
        print(
            f"   loaded {name:9s}  features={cities[name].feature_tensor.shape}  "
            f"target={cities[name].target_tensor.shape}  ({time.time()-t0:.1f}s)"
        )
    return cities


def merge_into_existing_results(path: Path, key: str, value: dict) -> None:
    """Merge a new entry into the results JSON without clobbering other keys."""
    existing = {}
    if path.exists():
        try:
            with open(path) as fh:
                existing = json.load(fh)
        except Exception:
            existing = {}
    existing.setdefault("source_only", {})
    existing.setdefault("transfer", {})
    if key.startswith("source:"):
        existing["source_only"][key.split(":", 1)[1]] = value
    elif key.startswith("transfer:"):
        existing["transfer"][key.split(":", 1)[1]] = value
    write_results(path, existing)


def main(args):
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    cities = load_all_cities(fixed=args.fixed)
    suffix = "_fixed" if args.fixed else ""
    results_path = RESULTS_DIR / f"lstm_results{suffix}.json"

    if args.mode in ("source", "all"):
        print(f"\n{'#'*68}\n# STAGE 1: SOURCE-ONLY LSTM (per-city)\n{'#'*68}")
        for c in ["Delhi", "Kolkata", "Guwahati"]:
            metrics = train_source_only(cities, c, fixed=args.fixed)
            merge_into_existing_results(results_path, f"source:{c}", metrics)
            print(f"\n   ==> merged source_only[{c}] into {results_path}")

    if args.mode in ("tl", "all"):
        print(f"\n{'#'*68}\n# STAGE 2: TRANSFER-LEARNING LSTM\n{'#'*68}")
        pairs = [
            ("Delhi", "Kolkata"),
            ("Delhi", "Guwahati"),
            ("Kolkata", "Guwahati"),
            ("Guwahati", "Kolkata"),
            ("Kolkata", "Delhi"),
            ("Guwahati", "Delhi"),
        ]
        d_values = [0.15, 0.30, 0.45, 0.60]
        for src, tgt in pairs:
            for d in d_values:
                metrics = train_transfer(cities, src, tgt, d, fixed=args.fixed)
                key = f"{src}->{tgt}@{int(d*100)}"
                merge_into_existing_results(results_path, f"transfer:{key}", metrics)
                print(f"   ==> merged transfer[{key}] into {results_path}")

    print(f"\nALL DONE. Wrote {results_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["source", "tl", "all"], default="all")
    p.add_argument("--fixed", action="store_true",
                   help="Enable interleaved split + climatology residual.")
    main(p.parse_args())
