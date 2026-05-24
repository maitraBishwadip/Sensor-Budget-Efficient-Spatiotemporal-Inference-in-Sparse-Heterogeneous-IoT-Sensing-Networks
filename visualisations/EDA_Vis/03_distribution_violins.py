"""Fig. 4 — Q1-grade cross-city PM2.5 distributions.

Three panels:
  (a) ECDF (linear scale) of PM2.5 per city, with WHO 24-h and NAAQS 24-h
      guideline lines annotated. Reveals where each city's distribution
      crosses each threshold.
  (b) Half-violin + box of log10 PM2.5 per city, with KS pairwise p-values
      and N stated under each violin.
  (c) Per-station mean ± std strip plot; horizontal bar = city mean.

The panel of pairwise KS p-values formalizes the "domain shift" claim that
motivates DANN — reviewers will not accept this claim without a test.
"""
from __future__ import annotations

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ks_2samp

from _common import (load_all, save, set_paper_style, CITY_COLORS,
                     COL_DOUBLE, panel_label)

set_paper_style()

cities = load_all()
combined = pd.concat(
    [df.assign(city=c) for c, df in cities.items()], ignore_index=True
)
combined["log_pm25"] = np.log10(combined["PM2.5"].clip(lower=1))

order = ["Delhi", "Kolkata", "Guwahati"]
palette = [CITY_COLORS[c] for c in order]

fig, axes = plt.subplots(1, 3, figsize=(COL_DOUBLE, 3.2),
                         gridspec_kw={"wspace": 0.35})

# ---- (a) ECDF ------------------------------------------------------------
ax = axes[0]
cap = np.quantile(combined["PM2.5"], 0.995)
for c in order:
    vals = np.sort(combined.loc[combined.city == c, "PM2.5"].clip(upper=cap))
    y = np.arange(1, len(vals) + 1) / len(vals)
    ax.plot(vals, y, color=CITY_COLORS[c], lw=1.6, label=c)
for x, name, col in [(15, "WHO 24-h (15)", "#2ca02c"),
                     (60, "NAAQS 24-h (60)", "black")]:
    ax.axvline(x, color=col, ls="--", lw=0.7, alpha=0.7)
    ax.text(x, 1.02, name, rotation=0, fontsize=7,
            ha="center", va="bottom", color=col)
ax.set_xlim(0, cap)
ax.set_ylim(0, 1.0)
ax.set_xlabel("PM$_{2.5}$  ($\\mu$g m$^{-3}$)")
ax.set_ylabel("Empirical CDF")
ax.legend(loc="lower right")
ax.set_title("Cumulative distribution")
panel_label(ax, "a")

# ---- (b) Log-violin + KS p-values -----------------------------------------
ax = axes[1]
sns.violinplot(data=combined, x="city", y="log_pm25", order=order,
               hue="city", palette=palette, legend=False,
               inner=None, cut=0, linewidth=0.6, ax=ax)
sns.boxplot(data=combined, x="city", y="log_pm25", order=order,
            width=0.10, showcaps=False,
            boxprops={"zorder": 3, "facecolor": "white", "edgecolor": "black",
                      "linewidth": 0.7},
            whiskerprops={"linewidth": 0.6}, fliersize=0, ax=ax)
ax.set_ylabel("$\\log_{10}$ PM$_{2.5}$")
ax.set_xlabel("")
ax.set_title("Per-city distributions")
# Annotate N below each violin
for i, c in enumerate(order):
    n = (combined.city == c).sum()
    ax.text(i, -0.05, f"n = {n:,}", transform=ax.get_xaxis_transform(),
            ha="center", va="top", fontsize=7, color="#444")
ax.margins(y=0.08)
panel_label(ax, "b")

# KS pairwise table in the panel-c title area
ks_lines = []
for i, ci in enumerate(order):
    for cj in order[i + 1:]:
        a = combined.loc[combined.city == ci, "PM2.5"].values
        b = combined.loc[combined.city == cj, "PM2.5"].values
        stat, p = ks_2samp(a, b)
        ks_lines.append(f"{ci[:3]}–{cj[:3]}: D={stat:.2f}, p={p:.1e}")

# ---- (c) Per-station strip + city mean -----------------------------------
ax = axes[2]
station_means = (combined.groupby(["city", "Station Name"])["PM2.5"]
                 .agg(["mean", "std"]).reset_index())
sns.stripplot(data=station_means, x="city", y="mean", order=order,
              hue="city", palette=palette, legend=False,
              size=7, edgecolor="black", linewidth=0.5,
              jitter=0.18, ax=ax)
for i, c in enumerate(order):
    mu = station_means.loc[station_means.city == c, "mean"].mean()
    ax.hlines(mu, i - 0.28, i + 0.28, color=CITY_COLORS[c],
              lw=2.2, zorder=2)
ax.set_ylabel("Station-mean PM$_{2.5}$  ($\\mu$g m$^{-3}$)")
ax.set_xlabel("")
ax.set_title("Within-city heterogeneity")
panel_label(ax, "c")

# KS table as a text box anchored to figure
ks_text = "Kolmogorov–Smirnov pairwise:\n" + "\n".join(ks_lines)
fig.text(0.5, -0.02, ks_text, ha="center", va="top", fontsize=7,
         family="monospace", color="#222")

save(fig, "03_distribution_violins")
plt.close(fig)
print("done.")
