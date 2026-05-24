"""Fig. 7 — Q1-grade wind roses coloured by PM2.5 concentration.

Three polar panels, one per city, sharing:
  - common PM2.5 bin set (so legend colour ↔ value is identical)
  - common nsector = 16 (same angular resolution)
  - radial axis label hidden (only relative frequency is meaningful)
  - horizontal common legend below the figure
  - small N annotation per panel
"""
from __future__ import annotations

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import matplotlib.pyplot as plt
from windrose import WindroseAxes

from _common import (load_all, save, set_paper_style, CITY_COLORS,
                     COL_DOUBLE)

set_paper_style()

cities = load_all()

# Bin edges chosen to span the joint PM2.5 distribution.
bins = [0, 30, 60, 90, 120, 180, 250, 500]
nsector = 16

fig = plt.figure(figsize=(COL_DOUBLE, 3.6))
axes_list = []

for i, (city, df) in enumerate(cities.items(), start=1):
    sub = df.dropna(subset=["WS", "WD", "PM2.5"]).copy()
    sub = sub[(sub["WS"] >= 0) & (sub["WS"] < 30)]
    sub = sub[(sub["WD"] >= 0) & (sub["WD"] <= 360)]
    ax = fig.add_subplot(1, 3, i, projection="windrose")
    if len(sub) == 0:
        ax.text(0.5, 0.5, "no wind data", ha="center", va="center",
                transform=ax.transAxes)
        continue
    ax.bar(sub["WD"].values, sub["PM2.5"].values,
           bins=bins, nsector=nsector, normed=True,
           opening=0.85, edgecolor="white", linewidth=0.3,
           cmap=plt.cm.magma_r)
    ax.set_yticklabels([])
    ax.set_title(f"{city}  ·  n = {len(sub):,}",
                 color=CITY_COLORS[city], pad=12)
    ax.tick_params(axis="x", labelsize=7)
    axes_list.append(ax)

# Standardize radial limits across panels (use the max of all).
max_rmax = max(ax.get_rmax() for ax in axes_list)
for ax in axes_list:
    ax.set_rmax(max_rmax)

# Common horizontal legend below the figure
from matplotlib.patches import Patch
cmap = plt.cm.magma_r
norm_vals = np.linspace(0.05, 0.95, len(bins) - 1)
handles = []
for i in range(len(bins) - 1):
    label = f"{bins[i]}–{bins[i+1]}" if i < len(bins) - 2 else f"≥{bins[i]}"
    handles.append(Patch(facecolor=cmap(norm_vals[i]), edgecolor="white",
                         label=label))
fig.legend(handles=handles, loc="lower center",
           bbox_to_anchor=(0.5, -0.04),
           ncol=len(bins) - 1, fontsize=7,
           title="PM$_{2.5}$ bin ($\\mu$g m$^{-3}$)", title_fontsize=8,
           frameon=False)

save(fig, "05_wind_rose")
plt.close(fig)
print("done.")
