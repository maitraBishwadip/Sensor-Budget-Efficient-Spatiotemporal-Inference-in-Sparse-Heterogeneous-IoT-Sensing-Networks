"""Fig. 8 (App. A) — Q1-grade per-city feature correlation matrices.

To make the matrices directly comparable, we use only the features common
to all three cities: PM2.5, AT, RH, WS, WD (the meteorological + target set
used by both Stage-I LSTM-TL and Stage-II Graph-DANN).

We use Spearman's ρ rather than Pearson because PM2.5 distributions are
heavily right-skewed (Spearman is rank-based and shift-robust).

Each correlation is annotated with a small dot-flag if the correlation
direction or strength flips relative to the Delhi reference panel — this
visually flags non-stationarity of feature relationships across cities
and motivates the adversarial domain alignment in Act II.
"""
from __future__ import annotations

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from _common import (load_all, save, set_paper_style, CITY_COLORS,
                     COL_DOUBLE, panel_label)

set_paper_style()

cities = load_all()
features = ["PM2.5", "AT", "RH", "WS", "WD"]

corrs = {}
for c, df in cities.items():
    sub = df[features].apply(lambda s: pd.to_numeric(s, errors="coerce"))
    corrs[c] = sub.corr(method="spearman")

ref = corrs["Delhi"]

fig, axes = plt.subplots(1, 3, figsize=(COL_DOUBLE, 2.6),
                         gridspec_kw={"wspace": 0.20})

panel_letters = ["a", "b", "c"]
for ax, letter, (city, corr) in zip(axes, panel_letters, corrs.items()):
    sns.heatmap(corr, ax=ax, cmap="RdBu_r", vmin=-1, vmax=1,
                annot=True, fmt=".2f", annot_kws={"size": 7.5},
                cbar=(ax is axes[-1]),
                cbar_kws={"label": "Spearman ρ"} if ax is axes[-1] else None,
                square=True, linewidths=0.4, linecolor="white")

    # Flip-flag: mark cells whose sign differs from Delhi
    if city != "Delhi":
        for i in range(len(features)):
            for j in range(len(features)):
                if i == j:
                    continue
                if np.sign(corr.iat[i, j]) != np.sign(ref.iat[i, j]):
                    ax.scatter(j + 0.5, i + 0.5, s=70, marker="o",
                               facecolor="none", edgecolor="black",
                               linewidth=1.0, zorder=5)

    ax.set_title(city, color=CITY_COLORS[city])
    ax.tick_params(axis="x", rotation=0, labelsize=7.5)
    ax.tick_params(axis="y", rotation=0, labelsize=7.5)
    panel_label(ax, letter)

fig.text(0.5, -0.02,
         "Circled cells in (b)/(c) flag correlations whose sign flips "
         "relative to Delhi (a) — feature relationships are non-stationary "
         "across cities.",
         ha="center", va="top", fontsize=7, color="#222")

save(fig, "06_correlation_matrix")
plt.close(fig)
print("done.")
