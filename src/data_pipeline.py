"""Data cleaning and feature engineering for the PM2.5 GNN-TL project.

Loads the three raw CPCB-derived CSVs (Delhi, Kolkata, Guwahati), harmonizes
the feature set to a common schema, adds cyclic temporal encodings, imputes
remaining missing values, and writes a processed file per city plus a metadata
JSON describing the schema and the per-station train/val/test splits.

Run:
    python -m src.data_pipeline
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

DATA_DIR = Path("dataset")
OUT_DIR = DATA_DIR / "processed"

# Columns we will keep across every city. The Delhi file has fewer pollutants
# than Kolkata/Guwahati, so the intersection is the working schema. Anything
# city-specific (NH3, NOx, O3, PM10) is dropped — keeping the schema
# consistent is required for transfer learning.
COMMON_FEATURES = ["PM2.5", "AT", "RH", "WS", "Sin_WD", "Cos_WD"]

# Cyclic time features we derive from the timestamp.
CYCLIC_FEATURES = ["sin_hour", "cos_hour", "sin_month", "cos_month"]

# One-hot season columns we standardize to (4 columns).
SEASON_COLS = ["Season_Winter", "Season_Spring", "Season_Summer", "Season_Monsoon"]

ALL_FEATURE_COLS = COMMON_FEATURES + CYCLIC_FEATURES + SEASON_COLS
TARGET_COL = "PM2.5"


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

@dataclass
class CitySplitMeta:
    """Per-station train/val/test split boundaries (timestamp indices)."""

    city: str
    stations: List[str]
    coords: Dict[str, Tuple[float, float]]  # station -> (lat, lon)
    n_records_per_station: Dict[str, int]
    train_end: str   # ISO timestamp marking the end of the train period
    val_end: str     # end of validation period
    test_end: str    # end of test period
    feature_cols: List[str]
    target_col: str


def month_to_season(month: int) -> str:
    """India-centric season mapping consistent with the Kolkata/Guwahati CSVs.

    Winter: Dec, Jan, Feb. Spring: Mar, Apr. Summer: May, Jun.
    Monsoon: Jul, Aug, Sep. Autumn (post-monsoon) folded into Winter for
    consistency with the existing 4-class one-hot in Kolkata/Guwahati CSVs.
    """
    if month in (12, 1, 2, 10, 11):
        return "Season_Winter"
    if month in (3, 4):
        return "Season_Spring"
    if month in (5, 6):
        return "Season_Summer"
    return "Season_Monsoon"  # 7, 8, 9


def add_cyclic_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Adds sin/cos encodings for hour-of-day and month-of-year."""
    df = df.copy()
    ts = pd.to_datetime(df["From Date"])
    df["hour"] = ts.dt.hour
    df["month"] = ts.dt.month
    df["sin_hour"] = np.sin(2 * np.pi * df["hour"] / 24.0)
    df["cos_hour"] = np.cos(2 * np.pi * df["hour"] / 24.0)
    df["sin_month"] = np.sin(2 * np.pi * (df["month"] - 1) / 12.0)
    df["cos_month"] = np.cos(2 * np.pi * (df["month"] - 1) / 12.0)
    return df


def ensure_season_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Ensures all four Season_* columns exist; derives them from the month
    if they are missing (Delhi case)."""
    df = df.copy()
    has_season = any(c in df.columns for c in SEASON_COLS)
    if not has_season:
        season = df["month"].apply(lambda m: month_to_season(int(m)))
        for col in SEASON_COLS:
            df[col] = (season == col).astype(float)
    else:
        for col in SEASON_COLS:
            if col not in df.columns:
                df[col] = 0.0
            df[col] = df[col].astype(float)
    return df


def impute_per_station(df: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
    """Time-ordered forward-then-backward fill within each station, followed
    by per-station mean fill, followed by global mean fill. Drops rows that
    are still NaN after all three passes (should be rare)."""
    df = df.copy().sort_values(["Station Name", "From Date"]).reset_index(drop=True)
    grouped = df.groupby("Station Name", group_keys=False)
    df[feature_cols] = grouped[feature_cols].apply(
        lambda g: g.ffill().bfill()
    )
    # Per-station mean fill for stations that are still NaN in a column.
    df[feature_cols] = grouped[feature_cols].apply(
        lambda g: g.fillna(g.mean(numeric_only=True))
    )
    # Global mean fill for completely-missing columns (shouldn't happen with
    # the common schema, but defensive).
    df[feature_cols] = df[feature_cols].fillna(df[feature_cols].mean(numeric_only=True))
    return df.dropna(subset=feature_cols).reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Per-city loaders
# --------------------------------------------------------------------------- #

def load_city(name: str, path: Path) -> pd.DataFrame:
    """Loads one city CSV and reduces it to the common schema."""
    df = pd.read_csv(path)
    df["From Date"] = pd.to_datetime(df["From Date"])
    df["To Date"] = pd.to_datetime(df["To Date"])

    # Coerce numeric features (some files have stray strings).
    for col in COMMON_FEATURES:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = add_cyclic_time_features(df)
    df = ensure_season_columns(df)

    keep = [
        "From Date", "Station Name", "Latitude", "Longitude",
        *COMMON_FEATURES, *CYCLIC_FEATURES, *SEASON_COLS,
    ]
    df = df[keep].copy()
    df["city"] = name
    return df


# --------------------------------------------------------------------------- #
# Splits
# --------------------------------------------------------------------------- #

def chronological_split(
    df: pd.DataFrame,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
) -> Tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp]:
    """Returns the (train_end, val_end, test_end) timestamps for the city.

    The same boundaries are applied across all stations of the city so that
    the train/val/test windows are time-aligned.
    """
    timestamps = df["From Date"].sort_values().unique()
    n = len(timestamps)
    train_end = pd.Timestamp(timestamps[int(n * train_frac)])
    val_end = pd.Timestamp(timestamps[int(n * (train_frac + val_frac))])
    test_end = pd.Timestamp(timestamps[-1])
    return train_end, val_end, test_end


# --------------------------------------------------------------------------- #
# Main pipeline
# --------------------------------------------------------------------------- #

def process_city(name: str, path: Path) -> Tuple[pd.DataFrame, CitySplitMeta]:
    print(f"[{name}] loading {path.name} ...")
    df = load_city(name, path)
    n_before = len(df)
    df = impute_per_station(df, ALL_FEATURE_COLS)
    n_after = len(df)
    print(f"[{name}] {n_before} -> {n_after} rows after imputation/dropna")

    train_end, val_end, test_end = chronological_split(df)

    stations = sorted(df["Station Name"].unique().tolist())
    coords = {
        s: (
            float(df.loc[df["Station Name"] == s, "Latitude"].iloc[0]),
            float(df.loc[df["Station Name"] == s, "Longitude"].iloc[0]),
        )
        for s in stations
    }
    n_per_station = df.groupby("Station Name").size().to_dict()

    meta = CitySplitMeta(
        city=name,
        stations=stations,
        coords={s: list(c) for s, c in coords.items()},  # JSON-safe
        n_records_per_station={s: int(n_per_station[s]) for s in stations},
        train_end=train_end.isoformat(),
        val_end=val_end.isoformat(),
        test_end=test_end.isoformat(),
        feature_cols=ALL_FEATURE_COLS,
        target_col=TARGET_COL,
    )
    print(
        f"[{name}] stations={len(stations)}  "
        f"train_end={meta.train_end}  val_end={meta.val_end}  test_end={meta.test_end}"
    )
    return df, meta


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    cities = {
        "Delhi": DATA_DIR / "delhi_processed - delhi_processed.csv",
        "Kolkata": DATA_DIR / "Kolkata-Data-3HR.csv",
        "Guwahati": DATA_DIR / "Guwahati-Data-3HR.csv",
    }

    metas: Dict[str, dict] = {}
    for name, path in cities.items():
        df, meta = process_city(name, path)
        out_csv = OUT_DIR / f"{name.lower()}_processed.csv"
        df.to_csv(out_csv, index=False)
        metas[name] = asdict(meta)
        print(f"[{name}] wrote {out_csv}\n")

    with open(OUT_DIR / "metadata.json", "w") as fh:
        json.dump(metas, fh, indent=2, default=str)
    print(f"wrote metadata -> {OUT_DIR / 'metadata.json'}")


if __name__ == "__main__":
    main()
