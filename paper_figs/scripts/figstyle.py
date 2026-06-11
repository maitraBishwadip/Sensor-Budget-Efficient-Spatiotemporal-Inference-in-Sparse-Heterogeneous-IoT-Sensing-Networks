"""Shared Matplotlib styling for the IEEE IoT-J paper figures.

All figure scripts import from here so fonts, sizes, palette, and output paths are
consistent. Figures are written as vector PDF (for LaTeX) plus a PNG preview to
``paper_figs/``. Paths are resolved relative to this file, so scripts run from any CWD.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# --- paths ---------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]          # repo root (d:/GNN_TL)
RESULTS = ROOT / "results"
OUT = ROOT / "paper_figs"

# --- IEEE column widths (inches) -----------------------------------------
COL_W = 3.45            # single column
DCOL_W = 7.16           # full text width (double column)

# --- colour palette (colour-blind safe, Wong) ----------------------------
PALETTE = {
    "gnn": "#0072B2",        # blue   -> our model
    "lstm": "#9A9A9A",       # grey   -> station-independent baseline
    "transfer": "#0072B2",   # blue   -> transfer
    "scratch": "#D55E00",    # orange -> from-scratch
    "zeroshot": "#999999",   # grey   -> zero-shot
    "accent": "#CC3311",     # red    -> highlight / delta
    "ok": "#009E73",         # green
    "neutral": "#444444",
}


def _init_rc() -> None:
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif", "Nimbus Roman"],
        "mathtext.fontset": "stix",
        "font.size": 8,
        "axes.titlesize": 8,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "axes.linewidth": 0.6,
        "grid.linewidth": 0.4,
        "lines.linewidth": 1.3,
        "lines.markersize": 4,
        "axes.grid": True,
        "grid.alpha": 0.35,
        "axes.axisbelow": True,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "pdf.fonttype": 42,      # editable/embeddable fonts
        "ps.fonttype": 42,
    })


_init_rc()


def save(fig, name: str) -> None:
    """Write ``<name>.pdf`` and ``<name>.png`` into paper_figs/ and report the path."""
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"{name}.{ext}")
    plt.close(fig)
    print(f"  wrote {OUT / (name + '.pdf')}")
