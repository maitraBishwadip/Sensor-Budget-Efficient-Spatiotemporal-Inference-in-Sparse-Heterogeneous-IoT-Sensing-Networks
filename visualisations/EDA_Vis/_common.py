"""Shared utilities for the Q1-grade EDA visualization scripts.

Provides:
  * paper-style matplotlib defaults (Type-42 fonts, 9-pt body, embedded)
  * data loaders for the three city CSVs with column normalization
  * map decorations: scale bar, north arrow, India inset
  * a small categorical palette and named city colors (colorblind-safe)
  * panel-label helper for (a)/(b)/(c) tags
"""
from __future__ import annotations

import os
from pathlib import Path

# Override any system PROJ paths (e.g. PostgreSQL/PostGIS) with pyproj's
# bundled database. Must happen before pyproj/geopandas/contextily import.
for _k in ("PROJ_LIB", "PROJ_DATA", "GDAL_DATA"):
    os.environ.pop(_k, None)
try:
    import pyproj
    _pyproj_db = str(Path(pyproj.datadir.get_data_dir()))
    os.environ["PROJ_DATA"] = _pyproj_db
    os.environ["PROJ_LIB"] = _pyproj_db
except Exception:
    pass

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle
from matplotlib.lines import Line2D
from matplotlib.transforms import blended_transform_factory

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "dataset"
PROCESSED_DIR = DATA_DIR / "processed"
FIG_DIR = Path(__file__).resolve().parent / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

CITY_FILES = {
    "Delhi":    DATA_DIR / "delhi_processed - delhi_processed.csv",
    "Kolkata":  DATA_DIR / "Kolkata-Data-3HR.csv",
    "Guwahati": DATA_DIR / "Guwahati-Data-3HR.csv",
}

# Okabe-Ito colorblind-safe categorical palette
CITY_COLORS = {
    "Delhi":    "#D55E00",   # vermillion
    "Kolkata":  "#0072B2",   # blue
    "Guwahati": "#009E73",   # bluish green
}

CITY_CENTROIDS = {
    "Delhi":    (28.6139, 77.2090),
    "Kolkata":  (22.5726, 88.3639),
    "Guwahati": (26.1445, 91.7362),
}

# India bounding box (lat, lon) — for the inset
INDIA_BBOX = dict(lon=(67.0, 98.5), lat=(6.0, 36.5))

# Journal column widths in inches (≈ mm/25.4)
COL_SINGLE = 3.31     # 84 mm
COL_DOUBLE = 6.85     # 174 mm


def set_paper_style():
    mpl.rcParams.update({
        "figure.dpi": 110,
        "savefig.dpi": 400,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.03,
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.titleweight": "bold",
        "axes.labelsize": 9,
        "axes.labelweight": "regular",
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.5,
        "ytick.major.width": 0.5,
        "axes.grid": True,
        "grid.alpha": 0.20,
        "grid.linestyle": "--",
        "grid.linewidth": 0.4,
        "legend.frameon": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def load_city(city: str) -> pd.DataFrame:
    df = pd.read_csv(CITY_FILES[city], low_memory=False)
    df["From Date"] = pd.to_datetime(df["From Date"], errors="coerce")
    df = df.dropna(subset=["From Date", "PM2.5", "Latitude", "Longitude"])
    df["PM2.5"] = pd.to_numeric(df["PM2.5"], errors="coerce")
    df = df.dropna(subset=["PM2.5"])
    df["city"] = city
    df["hour"] = df["From Date"].dt.hour
    df["month"] = df["From Date"].dt.month
    df["date"] = df["From Date"].dt.date
    df["season"] = df["month"].map(_season)
    return df


def _season(m: int) -> str:
    if m in (12, 1, 2):
        return "Winter"
    if m in (3, 4, 5):
        return "Pre-monsoon"
    if m in (6, 7, 8, 9):
        return "Monsoon"
    return "Post-monsoon"


def load_all() -> dict[str, pd.DataFrame]:
    return {c: load_city(c) for c in CITY_FILES}


def station_summary(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby("Station Name")
          .agg(lat=("Latitude", "mean"),
               lon=("Longitude", "mean"),
               pm25_mean=("PM2.5", "mean"),
               pm25_std=("PM2.5", "std"),
               n=("PM2.5", "size"))
          .reset_index()
    )


def panel_label(ax, label: str, *, x=0.02, y=0.98, **kw):
    """Bold (a)/(b)/(c)... panel label in the top-left of axes coords."""
    ax.text(x, y, f"({label})", transform=ax.transAxes,
            fontsize=10, fontweight="bold",
            ha="left", va="top",
            bbox=dict(facecolor="white", edgecolor="none",
                      boxstyle="round,pad=0.18", alpha=0.85),
            **kw)


def add_scale_bar(ax, length_m, *, location=(0.62, 0.06),
                  color="black", linewidth=2.2, label=None):
    """Scale bar in EPSG:3857 metres. `length_m` is the bar length in metres."""
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    x0 = xmin + (xmax - xmin) * location[0]
    y0 = ymin + (ymax - ymin) * location[1]
    x1 = x0 + length_m
    ax.plot([x0, x1], [y0, y0], color=color, lw=linewidth,
            solid_capstyle="butt", zorder=10)
    ax.plot([x0, x0], [y0, y0 + (ymax - ymin) * 0.012],
            color=color, lw=linewidth, zorder=10)
    ax.plot([x1, x1], [y0, y0 + (ymax - ymin) * 0.012],
            color=color, lw=linewidth, zorder=10)
    if label is None:
        label = f"{length_m / 1000:g} km"
    ax.text((x0 + x1) / 2, y0 + (ymax - ymin) * 0.022, label,
            ha="center", va="bottom", fontsize=7, color=color, zorder=10,
            bbox=dict(facecolor="white", edgecolor="none",
                      boxstyle="round,pad=0.18", alpha=0.85))


def add_north_arrow(ax, *, location=(0.93, 0.85), size=0.05, color="black"):
    """A small north arrow in axes-fraction coords. Stays fully inside the axes."""
    x, y = location
    arrow = FancyArrowPatch(
        (x, y - size), (x, y + size),
        transform=ax.transAxes,
        arrowstyle="-|>", mutation_scale=10,
        linewidth=1.2, color=color, zorder=10,
    )
    ax.add_patch(arrow)
    ax.text(x, y + size + 0.015, "N", transform=ax.transAxes,
            ha="center", va="bottom", fontsize=7.5, fontweight="bold",
            color=color, zorder=10,
            bbox=dict(facecolor="white", edgecolor="none",
                      boxstyle="round,pad=0.10", alpha=0.85))


def add_india_inset(parent_ax, city: str, *, loc="lower right",
                    width="28%", height="28%", border_pad=0.4):
    """Inset minimap of India with the city highlighted."""
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    iax = inset_axes(parent_ax, width=width, height=height,
                     loc=loc, borderpad=border_pad)
    lat = INDIA_BBOX["lat"]
    lon = INDIA_BBOX["lon"]
    iax.add_patch(Rectangle((lon[0], lat[0]), lon[1] - lon[0], lat[1] - lat[0],
                            facecolor="#f0f0f0", edgecolor="#666", lw=0.5))
    # Coarse outline (no shapefile dependency): a few state-level reference dots
    for c, (la, lo) in CITY_CENTROIDS.items():
        iax.scatter(lo, la, s=14, facecolor="white",
                    edgecolor="#888", lw=0.5, zorder=2)
    la, lo = CITY_CENTROIDS[city]
    iax.scatter(lo, la, s=70, facecolor=CITY_COLORS[city],
                edgecolor="black", lw=0.7, zorder=3)
    iax.set_xlim(*lon); iax.set_ylim(*lat)
    iax.set_xticks([]); iax.set_yticks([])
    iax.grid(False)
    for s in iax.spines.values():
        s.set_edgecolor("#666"); s.set_linewidth(0.5)
    iax.text(0.04, 0.95, "India", transform=iax.transAxes,
             fontsize=6.5, va="top", ha="left", color="#444",
             fontweight="bold")
    return iax


def pick_scale_bar_length(ax_xrange_m: float) -> float:
    """Choose a 'nice' scale-bar length given the axes width in metres."""
    target = ax_xrange_m * 0.18
    candidates = [1_000, 2_000, 5_000, 10_000, 20_000, 25_000, 50_000,
                  100_000, 200_000]
    return min(candidates, key=lambda c: abs(c - target))


def save(fig, name: str):
    out_pdf = FIG_DIR / f"{name}.pdf"
    out_png = FIG_DIR / f"{name}.png"
    fig.savefig(out_pdf)
    fig.savefig(out_png)
    print(f"  saved -> {out_pdf.name} / {out_png.name}")
