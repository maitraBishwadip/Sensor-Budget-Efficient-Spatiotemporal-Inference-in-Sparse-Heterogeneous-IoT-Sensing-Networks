"""F1: System architecture, in a clean colour-coded block style (editor's IoT test).

Sparse, heterogeneous sensor deployments -> edge gateway -> graph construction ->
inductive |V|-independent ST-GNN encoder -> two edge inference services (virtual sensing
+ forecasting). A dense source deployment feeds a cold-start transfer arrow into the encoder.

Writes: paper_figs/fig_architecture.{pdf,png}
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from figstyle import DCOL_W, save

C = dict(grey="#ECEFF1", purple="#D1C4E9", tan="#FFE0B2", yellow="#FFF59D",
         blue="#BBDEFB", green="#C8E6C9", orange="#FFCC80", srcfill="#FCE9D6")
EC = "#5f6368"


def box(ax, cx, cy, w, h, text, fc, ec=EC, fs=7.0, bold=False, tc="black"):
    ax.add_patch(FancyBboxPatch((cx - w / 2, cy - h / 2), w, h,
                 boxstyle="round,pad=0.02,rounding_size=0.06",
                 linewidth=0.9, edgecolor=ec, facecolor=fc, zorder=3))
    ax.text(cx, cy, text, ha="center", va="center", fontsize=fs, zorder=4,
            fontweight="bold" if bold else "normal", color=tc)


def arrow(ax, p0, p1, color=EC, lw=1.2, style="-|>", rad=0.0, ls="-"):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle=style, mutation_scale=10,
                 lw=lw, color=color, linestyle=ls,
                 connectionstyle=f"arc3,rad={rad}", zorder=2))


fig, ax = plt.subplots(figsize=(DCOL_W, 2.75))
ax.set_xlim(0, 12.7)
ax.set_ylim(0, 5.0)
ax.axis("off")

ymid = 3.45
# --- main pipeline -------------------------------------------------------
box(ax, 1.20, ymid, 2.05, 1.5, "Sparse,\nheterogeneous\nsensor nodes\n(40 / 10 / 4)", C["grey"])
box(ax, 3.55, ymid, 1.8, 1.15, "Edge gateway\nharmonize +\nclimatology residual", C["tan"])
box(ax, 5.80, ymid, 1.8, 1.0, "Graph build\n$k$-NN + wind-aware", C["purple"])
box(ax, 8.15, ymid, 2.0, 1.3, "Inductive\nST-GNN encoder\n($|V|$-independent)", C["blue"], bold=True)
for a, b in [(2.225, 2.65), (4.45, 4.90), (6.70, 7.15)]:
    arrow(ax, (a, ymid), (b, ymid))
ax.text(8.15, 2.55, "23.7k params  $\\cdot$  93 KB  $\\cdot$  $<$3 ms/inf. (CPU)",
        ha="center", va="center", fontsize=5.9, color="#1f5fa8")

# --- inference-services group -------------------------------------------
gx0, gx1, gy0, gy1 = 9.55, 12.6, 2.18, 4.7
ax.add_patch(FancyBboxPatch((gx0, gy0), gx1 - gx0, gy1 - gy0,
             boxstyle="round,pad=0.02,rounding_size=0.06", linewidth=0.8,
             edgecolor="#9e9e9e", facecolor="#FafafA", linestyle=(0, (4, 2)), zorder=1))
ax.text((gx0 + gx1) / 2, gy1 - 0.17, "Edge inference services", ha="center",
        fontsize=6.4, color="#555", style="italic")
box(ax, 11.07, 3.95, 2.45, 0.8, "Virtual sensing\n(unsensored nodes)", C["green"])
box(ax, 11.07, 2.75, 2.45, 0.8, "Forecasting  ($+3$ h)", C["orange"])
arrow(ax, (9.15, 3.6), (9.85, 3.95), rad=0.12)
arrow(ax, (9.15, 3.2), (9.85, 2.78), rad=-0.12)

# --- cold-start transfer feed -------------------------------------------
box(ax, 1.20, 1.05, 2.05, 0.85, "Dense source\ndeployment", C["srcfill"], ec=C["orange"])
arrow(ax, (2.25, 1.05), (7.5, 2.80), color="#E8830C", lw=1.3, rad=-0.10)
ax.text(4.9, 1.30, "cold-start transfer (pre-train $\\rightarrow$ fine-tune)",
        fontsize=6.3, color="#E8830C", style="italic", ha="center")

fig.tight_layout(pad=0.2)
save(fig, "fig_architecture")
