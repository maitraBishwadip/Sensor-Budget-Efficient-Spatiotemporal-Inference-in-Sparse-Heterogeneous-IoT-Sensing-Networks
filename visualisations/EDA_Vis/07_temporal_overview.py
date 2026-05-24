"""Fig. 9 — Q1-grade temporal overview of city-mean PM2.5.

Stack of three panels (one per city) sharing a common y-axis cap so visual
comparison is valid, with:
  - daily city-mean (line) ± 1 SD across stations (band)
  - season-band shading (Winter / Pre-monsoon / Monsoon / Post-monsoon)
    in muted Okabe-Ito tints
  - WHO 24-h (15 µg/m³, green dotted) and NAAQS 24-h (60 µg/m³, black dashed)
    threshold lines, labelled once
  - annotation of the top-3 daily-mean peaks per city (likely Diwali,
    crop-residue burning episodes)
  - data span and N stations annotated per panel
"""
from __future__ import annotations

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from _common import (load_all, save, set_paper_style, CITY_COLORS,
                     COL_DOUBLE, panel_label)

set_paper_style()

cities = load_all()

SEASON_COLORS = {
    "Winter":       "#56B4E9",   # Okabe-Ito sky
    "Pre-monsoon":  "#F0E442",   # yellow
    "Monsoon":      "#009E73",   # green
    "Post-monsoon": "#E69F00",   # orange
}
SEASON_OF_MONTH = {12: "Winter", 1: "Winter", 2: "Winter",
                   3: "Pre-monsoon", 4: "Pre-monsoon", 5: "Pre-monsoon",
                   6: "Monsoon", 7: "Monsoon", 8: "Monsoon", 9: "Monsoon",
                   10: "Post-monsoon", 11: "Post-monsoon"}

# Determine a shared y-axis cap (clip extreme spikes for visibility)
caps = []
for df in cities.values():
    daily = df.groupby("date")["PM2.5"].mean()
    caps.append(daily.quantile(0.99))
y_cap = max(caps) * 1.05

fig, axes = plt.subplots(3, 1, figsize=(COL_DOUBLE, 6.0),
                         sharex=False)

panel_letters = ["a", "b", "c"]
for ax, letter, (city, df) in zip(axes, panel_letters, cities.items()):
    daily = (df.groupby("date")["PM2.5"]
               .agg(mean="mean", std="std", n="size")
               .reset_index())
    daily["date"] = pd.to_datetime(daily["date"])
    daily = daily.sort_values("date").reset_index(drop=True)

    # Season bands: one rectangle per month
    months = pd.date_range(daily["date"].min(),
                           daily["date"].max() + pd.Timedelta(days=1),
                           freq="MS")
    for d in months:
        s = SEASON_OF_MONTH[d.month]
        end = d + pd.offsets.MonthEnd(0)
        ax.axvspan(d, end, color=SEASON_COLORS[s], alpha=0.10, lw=0)

    ax.fill_between(daily["date"],
                    (daily["mean"] - daily["std"]).clip(lower=0),
                    daily["mean"] + daily["std"],
                    color=CITY_COLORS[city], alpha=0.18, lw=0)
    ax.plot(daily["date"], daily["mean"], color=CITY_COLORS[city],
            lw=0.9)

    ax.axhline(60, color="black", ls="--", lw=0.6, alpha=0.6)
    ax.axhline(15, color="#2ca02c", ls=":", lw=0.6, alpha=0.7)
    if ax is axes[0]:
        ax.text(daily["date"].iloc[-1], 60, " NAAQS 24-h",
                ha="left", va="center", fontsize=7, alpha=0.7)
        ax.text(daily["date"].iloc[-1], 15, " WHO 24-h",
                ha="left", va="center", fontsize=7,
                color="#2ca02c", alpha=0.85)

    # Annotate top-3 peaks
    top = daily.nlargest(3, "mean")
    for _, r in top.iterrows():
        ax.annotate(r["date"].strftime("%d %b"),
                    xy=(r["date"], r["mean"]),
                    xytext=(0, 8), textcoords="offset points",
                    ha="center", va="bottom", fontsize=6.5,
                    color="#333",
                    arrowprops=dict(arrowstyle="-", color="#333", lw=0.4))

    ax.set_ylabel("PM$_{2.5}$  ($\\mu$g m$^{-3}$)")
    ax.set_ylim(0, y_cap)
    ax.set_title(f"{city}  ·  {df['Station Name'].nunique()} stations  ·  "
                 f"{daily['date'].min().date()} → {daily['date'].max().date()}",
                 color=CITY_COLORS[city], loc="left")
    panel_label(ax, letter)

axes[-1].set_xlabel("Date")

legend_handles = [Patch(facecolor=c, alpha=0.5, label=name)
                  for name, c in SEASON_COLORS.items()]
fig.legend(handles=legend_handles, loc="upper center",
           bbox_to_anchor=(0.5, 1.005), ncol=4, frameon=False,
           fontsize=8)

save(fig, "07_temporal_overview")
plt.close(fig)
print("done.")
