"""Common utilities: sequence windowing, metrics, seeding, checkpoint I/O.

`load_city_tensor` accepts two opt-in flags (default off so existing thesis-
comparable results stay reproducible):

  - `interleaved_split`: instead of a chronological 70/15/15, take every Nth
    sample for val and test so each split spans the full year. Fixes the
    Kolkata/Guwahati issue where the test set is winter-only because they
    only have one year of data.

  - `climatology_residual`: subtract per-(month, hour) climatology (fit on
    train slice) from PM2.5 before z-scoring. Model then predicts the
    *deviation from seasonal climatology*. At inference, the climatology
    is added back. Reduces the cross-city/cross-year mean shift that
    drives the Delhi-2021 vs Guwahati-2023 transfer gap.
"""

from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler


# --------------------------------------------------------------------------- #
# Reproducibility
# --------------------------------------------------------------------------- #

def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(False)


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #

def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    return 1.0 - ss_res / max(ss_tot, 1e-12)


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))


def mape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1.0) -> float:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    denom = np.maximum(np.abs(y_true), eps)
    return float(np.mean(np.abs(y_true - y_pred) / denom) * 100.0)


def all_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "R2": r2_score(y_true, y_pred),
        "MAE": mae(y_true, y_pred),
        "RMSE": rmse(y_true, y_pred),
        "MAPE": mape(y_true, y_pred),
    }


# --------------------------------------------------------------------------- #
# Per-city tensor packing (the central data structure for GNN training)
# --------------------------------------------------------------------------- #

@dataclass
class CityTensors:
    """Time-aligned per-city tensors, shape [T, N, F] for features.

    T = number of timestamps; N = number of stations; F = feature dim.
    target shape: [T, N].

    When `climatology_residual=True` was passed to `load_city_tensor`,
    `target_tensor` holds z-scored *residuals* (target − climatology), and
    `climatology_tensor` holds the per-(t, station) climatology mean used for
    add-back at inference. Otherwise climatology_tensor is None.

    When `interleaved_split=True`, `train_mask` / `val_mask` / `test_mask` hold
    boolean masks of shape [T] selecting which timesteps belong to each split.
    The legacy `train_end_idx` / `val_end_idx` / `test_end_idx` are still set
    (to the largest masked index, for backward compat) but partition lookups
    should prefer the masks.
    """
    city: str
    feature_tensor: np.ndarray   # [T, N, F]
    target_tensor: np.ndarray    # [T, N] -- z-scored residual if climatology_residual=True
    timestamps: pd.DatetimeIndex
    stations: List[str]
    coords: List[Tuple[float, float]]
    scaler: StandardScaler       # fit on train features (flattened)
    target_scaler: StandardScaler  # fit on train target (or residual)
    train_end_idx: int
    val_end_idx: int
    test_end_idx: int
    climatology_tensor: Optional[np.ndarray] = None  # [T, N], raw-PM2.5 climatology
    train_mask: Optional[np.ndarray] = None  # [T] bool
    val_mask: Optional[np.ndarray] = None    # [T] bool
    test_mask: Optional[np.ndarray] = None   # [T] bool


def _compute_climatology(target_tensor: np.ndarray, timestamps: pd.DatetimeIndex,
                          train_mask: np.ndarray) -> np.ndarray:
    """Per-(month, hour) climatology of PM2.5, fit on the train slice only.

    Returns a [T, N] tensor: at each (t, n), the mean PM2.5 over the train
    timesteps that share `timestamp.month` and `timestamp.hour` with t. This
    is the seasonal-diurnal expectation that the model will be asked to
    predict *deviations from*.
    """
    T, N = target_tensor.shape
    months = timestamps.month.to_numpy()
    hours = timestamps.hour.to_numpy()
    clim = np.zeros((T, N), dtype=np.float32)
    for n in range(N):
        for m in range(1, 13):
            for h in range(0, 24, 3):  # 3-hour cadence
                mask = train_mask & (months == m) & (hours == h)
                if mask.any():
                    mean_val = float(np.nanmean(target_tensor[mask, n]))
                else:
                    # Fall back to per-station train mean if this (m, h) bucket is empty.
                    mean_val = float(np.nanmean(target_tensor[train_mask, n]))
                # Fill all timesteps in this (m, h) bucket, regardless of train/val/test.
                bucket = (months == m) & (hours == h)
                clim[bucket, n] = mean_val
    return clim


def _interleaved_split(T: int, train_frac: float = 0.70, val_frac: float = 0.15,
                        seed: int = 42) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Deterministic interleaved split.

    Take every (1/val_frac)th sample for val (offset 0), every (1/test_frac)th
    sample for test (offset 1), everything else for train. By using strides
    rather than random sampling we keep the split deterministic *and* spread
    val/test uniformly across the full year so each split spans every season.
    """
    test_frac = 1.0 - train_frac - val_frac
    val_stride = max(2, int(round(1.0 / val_frac)))
    test_stride = max(2, int(round(1.0 / test_frac)))
    idx = np.arange(T)
    val_mask = (idx % val_stride == 0)
    test_mask = (idx % test_stride == 1) & ~val_mask  # offset 1 keeps val/test disjoint
    train_mask = ~(val_mask | test_mask)
    return train_mask, val_mask, test_mask


def load_city_tensor(
    processed_csv: Path,
    metadata: dict,
    feature_cols: List[str],
    target_col: str,
    *,
    interleaved_split: bool = False,
    climatology_residual: bool = False,
) -> CityTensors:
    df = pd.read_csv(processed_csv, parse_dates=["From Date"])
    city = str(df["city"].iloc[0])

    # Sort by timestamp then station; pivot to [T, N, F]
    df = df.sort_values(["From Date", "Station Name"]).reset_index(drop=True)
    stations = sorted(df["Station Name"].unique().tolist())
    coords = [(metadata["coords"][s][0], metadata["coords"][s][1]) for s in stations]

    # Build the common timestamp index (intersection across stations)
    ts_sets = [set(g["From Date"].unique()) for _, g in df.groupby("Station Name")]
    common_ts = sorted(set.intersection(*ts_sets))
    if not common_ts:
        # Fall back to all timestamps with NaN for missing station-timestamps.
        common_ts = sorted(df["From Date"].unique())
    timestamps = pd.DatetimeIndex(common_ts)

    T = len(timestamps)
    N = len(stations)
    F = len(feature_cols)

    feature_tensor = np.full((T, N, F), np.nan, dtype=np.float32)
    target_tensor = np.full((T, N), np.nan, dtype=np.float32)

    # Vectorized fill via integer indexing — much faster than iterrows().
    ts_to_i = pd.Series(np.arange(T), index=timestamps)
    st_to_i = pd.Series(np.arange(N), index=stations)
    ti_arr = ts_to_i.reindex(df["From Date"]).to_numpy()
    si_arr = st_to_i.reindex(df["Station Name"]).to_numpy()
    valid = (~pd.isna(ti_arr)) & (~pd.isna(si_arr))
    ti_arr = ti_arr[valid].astype(np.int64)
    si_arr = si_arr[valid].astype(np.int64)
    feat_vals = df.loc[valid, feature_cols].to_numpy(dtype=np.float32)
    tgt_vals = df.loc[valid, target_col].to_numpy(dtype=np.float32)
    feature_tensor[ti_arr, si_arr, :] = feat_vals
    target_tensor[ti_arr, si_arr] = tgt_vals

    # Time-domain forward then backward fill per station-feature.
    for n in range(N):
        df_n = pd.DataFrame(feature_tensor[:, n, :]).ffill().bfill().fillna(0.0)
        feature_tensor[:, n, :] = df_n.to_numpy(dtype=np.float32)
        tgt = pd.Series(target_tensor[:, n]).ffill().bfill().fillna(0.0)
        target_tensor[:, n] = tgt.to_numpy(dtype=np.float32)

    train_end = pd.Timestamp(metadata["train_end"])
    val_end = pd.Timestamp(metadata["val_end"])
    test_end = pd.Timestamp(metadata["test_end"])

    if interleaved_split:
        train_mask, val_mask, test_mask = _interleaved_split(T)
        # Keep the *_end_idx fields well-defined for callers that still use them:
        # set each to the largest True index of the corresponding mask, +1.
        train_end_idx = int(np.where(train_mask)[0].max()) + 1 if train_mask.any() else T
        val_end_idx = int(np.where(val_mask)[0].max()) + 1 if val_mask.any() else T
        test_end_idx = T
    else:
        train_end_idx = int(np.searchsorted(timestamps.values, np.datetime64(train_end), side="right"))
        val_end_idx = int(np.searchsorted(timestamps.values, np.datetime64(val_end), side="right"))
        test_end_idx = T
        train_mask = np.zeros(T, dtype=bool)
        train_mask[:train_end_idx] = True
        val_mask = np.zeros(T, dtype=bool)
        val_mask[train_end_idx:val_end_idx] = True
        test_mask = np.zeros(T, dtype=bool)
        test_mask[val_end_idx:] = True

    # Optional: compute per-(month, hour) climatology from the train slice and
    # store the *residual* (target − climatology) in target_tensor. Z-scoring
    # then operates on the residual.
    raw_target = target_tensor.copy()
    climatology_tensor: Optional[np.ndarray] = None
    if climatology_residual:
        climatology_tensor = _compute_climatology(raw_target, timestamps, train_mask)
        target_tensor = raw_target - climatology_tensor  # residual

    # Fit scalers on the train slice only.
    flat_train_feats = feature_tensor[train_mask].reshape(-1, F)
    scaler = StandardScaler().fit(flat_train_feats)
    feature_tensor = scaler.transform(feature_tensor.reshape(-1, F)).reshape(T, N, F).astype(np.float32)

    target_scaler = StandardScaler().fit(target_tensor[train_mask].reshape(-1, 1))
    target_tensor = target_scaler.transform(target_tensor.reshape(-1, 1)).reshape(T, N).astype(np.float32)

    return CityTensors(
        city=city,
        feature_tensor=feature_tensor,
        target_tensor=target_tensor,
        timestamps=timestamps,
        stations=stations,
        coords=coords,
        scaler=scaler,
        target_scaler=target_scaler,
        train_end_idx=train_end_idx,
        val_end_idx=val_end_idx,
        test_end_idx=test_end_idx,
        climatology_tensor=climatology_tensor,
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask,
    )


# --------------------------------------------------------------------------- #
# Sequence windowing
# --------------------------------------------------------------------------- #

def make_windows(
    feature_tensor: np.ndarray,  # [T, N, F]
    target_tensor: np.ndarray,   # [T, N]
    history: int = 8,            # 8 * 3h = 24 h history
    horizon: int = 1,            # 1 * 3h = 3 h forecast horizon
    start_idx: int = 0,
    end_idx: int | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Returns:
        X: [num_windows, history, N, F]
        Y: [num_windows, N]   (PM2.5 at t + horizon)
    """
    if end_idx is None:
        end_idx = feature_tensor.shape[0]
    T = end_idx - start_idx
    if T <= history + horizon:
        return np.zeros((0, history, feature_tensor.shape[1], feature_tensor.shape[2]), dtype=np.float32), \
               np.zeros((0, feature_tensor.shape[1]), dtype=np.float32)

    xs, ys = [], []
    for t in range(start_idx, end_idx - history - horizon + 1):
        xs.append(feature_tensor[t : t + history])
        ys.append(target_tensor[t + history + horizon - 1])
    X = np.stack(xs, axis=0).astype(np.float32)
    Y = np.stack(ys, axis=0).astype(np.float32)
    return X, Y


def make_windows_masked(
    feature_tensor: np.ndarray,  # [T, N, F]
    target_tensor: np.ndarray,   # [T, N]
    split_mask: np.ndarray,      # [T] bool — which target timesteps belong to this split
    history: int = 8,
    horizon: int = 1,
    climatology_tensor: Optional[np.ndarray] = None,  # [T, N], raw-PM2.5 climatology
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """Same as `make_windows`, but selects windows by `split_mask` on the
    prediction timestep `t + history + horizon - 1`. Returns the per-window
    climatology at the prediction step if climatology_tensor is provided.

    This is what lets interleaved splits work end-to-end: a window's history
    can overlap with windows from other splits (cheap and necessary at
    3-hour cadence), but the supervision target is uniquely owned by one split.
    """
    T = feature_tensor.shape[0]
    if T <= history + horizon:
        empty_X = np.zeros((0, history, feature_tensor.shape[1], feature_tensor.shape[2]), dtype=np.float32)
        empty_Y = np.zeros((0, feature_tensor.shape[1]), dtype=np.float32)
        empty_C = None if climatology_tensor is None else np.zeros((0, feature_tensor.shape[1]), dtype=np.float32)
        return empty_X, empty_Y, empty_C

    xs, ys, cs = [], [], []
    last_t = T - history - horizon + 1
    for t in range(last_t):
        pred_t = t + history + horizon - 1
        if not split_mask[pred_t]:
            continue
        xs.append(feature_tensor[t : t + history])
        ys.append(target_tensor[pred_t])
        if climatology_tensor is not None:
            cs.append(climatology_tensor[pred_t])
    if not xs:
        empty_X = np.zeros((0, history, feature_tensor.shape[1], feature_tensor.shape[2]), dtype=np.float32)
        empty_Y = np.zeros((0, feature_tensor.shape[1]), dtype=np.float32)
        empty_C = None if climatology_tensor is None else np.zeros((0, feature_tensor.shape[1]), dtype=np.float32)
        return empty_X, empty_Y, empty_C
    X = np.stack(xs, axis=0).astype(np.float32)
    Y = np.stack(ys, axis=0).astype(np.float32)
    C = np.stack(cs, axis=0).astype(np.float32) if cs else None
    return X, Y, C


# --------------------------------------------------------------------------- #
# Checkpointing
# --------------------------------------------------------------------------- #

def save_checkpoint(model: torch.nn.Module, path: Path, extra: dict | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    blob = {"state_dict": model.state_dict()}
    if extra is not None:
        blob.update(extra)
    torch.save(blob, path)


def load_checkpoint(model: torch.nn.Module, path: Path) -> dict:
    blob = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(blob["state_dict"])
    return blob


def write_results(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        json.dump(payload, fh, indent=2, default=str)


def inverse_transform_target(scaler: StandardScaler, y: np.ndarray,
                              climatology: Optional[np.ndarray] = None) -> np.ndarray:
    """Inverse z-score back to raw PM2.5. If `climatology` is given (shape
    must broadcast to y's shape), it is treated as a *residual climatology*
    add-back: the inverse z-score is the residual, and the result is
    residual + climatology = absolute PM2.5.
    """
    y_arr = np.asarray(y)
    flat = scaler.inverse_transform(y_arr.reshape(-1, 1)).reshape(y_arr.shape)
    if climatology is not None:
        flat = flat + np.asarray(climatology).reshape(flat.shape)
    return flat
