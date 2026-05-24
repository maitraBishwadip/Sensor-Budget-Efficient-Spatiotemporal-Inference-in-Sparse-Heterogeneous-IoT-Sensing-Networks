"""Fig. 6 — Q1-grade month × hour PM2.5 heatmaps per city.

Per panel:
  - month × hour mean PM2.5, in 3-letter month labels
  - per-panel min/max stated in title (the comparison story is *patterns*,
    not absolute values, so per-panel normalization is correct here)
  - marginal bar plots: monthly mean (left) and diurnal mean (top) with
    standard-error whiskers across stations
  - colorbar per panel (since they're normalized per-city)
"""
from __future__ import annotations

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns

from _common import (load_all, save, set_paper_style, CITY_COLORS,
                     COL_DOUBLE, panel_label)

set_paper_style()

cities = load_all()
month_labels = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
hour_labels = [f"{h:02d}" for h in range(0, 24, 3)]

fig = plt.figure(figsize=(COL_DOUBLE, 4.8))
outer = GridSpec(1, 3, wspace=0.4, figure=fig)

panel_letters = ["a", "b", "c"]
for col, letter, (city, df) in zip(range(3), panel_letters, cities.items()):
    inner = outer[0, col].subgridspec(
        2, 2,
        width_ratios=[1, 4], height_ratios=[1, 4],
        wspace=0.05, hspace=0.05,
    )
    ax_main = fig.add_subplot(inner[1, 1])
    ax_top  = fig.add_subplot(inner[0, 1], sharex=ax_main)
    ax_left = fig.add_subplot(inner[1, 0], sharey=ax_main)

    pivot = (df.groupby(["month", "hour"])["PM2.5"]
               .mean().unstack("hour")
               .reindex(index=range(1, 13),
                        columns=[0, 3, 6, 9, 12, 15, 18, 21]))

    vmax = float(np.nanpercentile(pivot.values, 99))
    vmin = float(np.nanpercentile(pivot.values, 1))

    sns.heatmap(pivot, ax=ax_main, cmap="rocket_r",
                vmin=vmin, vmax=vmax,
                cbar=True,
                cbar_kws={"label": "PM$_{2.5}$ ($\\mu$g m$^{-3}$)",
                          "shrink": 0.85, "pad": 0.06},
                linewidths=0.25, linecolor="white",
                xticklabels=hour_labels, yticklabels=month_labels)
    ax_main.set_xlabel("Hour of day (UTC+5:30)")
    ax_main.set_ylabel("")
    ax_main.tick_params(axis="y", labelleft=False)

    # Top marginal: diurnal mean (errorbars across stations)
    diurnal = (df.groupby(["Station Name", "hour"])["PM2.5"]
                 .mean().reset_index())
    means = diurnal.groupby("hour")["PM2.5"].mean()
    sems = diurnal.groupby("hour")["PM2.5"].sem()
    ax_top.bar(np.arange(8) + 0.5, means.reindex([0, 3, 6, 9, 12, 15, 18, 21]).values,
               yerr=sems.reindex([0, 3, 6, 9, 12, 15, 18, 21]).values,
               width=0.7, color=CITY_COLORS[city], alpha=0.65,
               edgecolor="black", linewidth=0.4, error_kw=dict(lw=0.6))
    ax_top.set_xticks([])
    ax_top.set_ylabel("μ", fontsize=7)
    ax_top.tick_params(axis="y", labelsize=7)
    ax_top.grid(False)

    # Left marginal: monthly mean
    monthly = (df.groupby(["Station Name", "month"])["PM2.5"]
                 .mean().reset_index())
    mmeans = monthly.groupby("month")["PM2.5"].mean().reindex(range(1, 13))
    msems  = monthly.groupby("month")["PM2.5"].sem().reindex(range(1, 13))
    ax_left.barh(np.arange(12) + 0.5, mmeans.values,
                 xerr=msems.values,
                 height=0.7, color=CITY_COLORS[city], alpha=0.65,
                 edgecolor="black", linewidth=0.4, error_kw=dict(lw=0.6))
    ax_left.invert_xaxis()
    ax_left.set_yticks(np.arange(12) + 0.5)
    ax_left.set_yticklabels(month_labels, fontsize=8)
    ax_left.set_xlabel("μ", fontsize=7)
    ax_left.tick_params(axis="x", labelsize=7)
    ax_left.grid(False)

    ax_top.set_title(f"{city}  ·  {vmin:.0f}–{vmax:.0f} $\\mu$g m$^{{-3}}$",
                     color=CITY_COLORS[city])
    panel_label(ax_top, letter)

save(fig, "04_seasonal_diurnal_heatmaps")
plt.close(fig)
print("done.")
