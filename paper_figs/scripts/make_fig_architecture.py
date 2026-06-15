"""F1: System architecture, dense but compact (landscape) process diagram.

(Top) a left-to-right process spine: process inputs -> edge gateway -> graph build ->
inductive |V|-independent ST-GNN encoder -> edge control -> process outputs, with the
data baseline labelled on each transition. (Bottom) a grouped "Sensor-Budget Levers"
band (virtual sensing / cold-start transfer / graceful degradation) the encoder enables,
with a cold-start-transfer feedback arc. Kept short (~3.2 in) for the 8-page budget.

Writes: paper_figs/fig_architecture.{pdf,png}
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch, FancyArrowPatch
from figstyle import DCOL_W, save

INK = "#111111"
SUB = "#333333"
GRP = "#7A7F85"
XFER = "#B5500F"


def box(ax, cx, cy, w, h, title, subs=None, *, tfs=7.0, sfs=5.9, lw=1.2, ec=INK, fc="white"):
    ax.add_patch(Rectangle((cx - w / 2, cy - h / 2), w, h, facecolor=fc,
                 edgecolor=ec, linewidth=lw, zorder=3))
    if subs:
        ax.text(cx, cy + h / 2 - 0.20, title, ha="center", va="top",
                fontsize=tfs, fontweight="bold", color=INK, zorder=4)
        ax.text(cx, cy + h / 2 - 0.52, "\n".join(subs), ha="center", va="top",
                fontsize=sfs, color=SUB, zorder=4, linespacing=1.4)
    else:
        ax.text(cx, cy, title, ha="center", va="center", fontsize=tfs,
                fontweight="bold", color=INK, zorder=4)


def arrow(ax, p0, p1, *, lw=1.2, head=9, color=INK, ls="-"):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=head,
                 lw=lw, color=color, shrinkA=0, shrinkB=0, linestyle=ls, zorder=2))


def route(ax, pts, *, lw=1.3, head=9, color=INK, ls="-"):
    xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
    ax.plot(xs, ys, color=color, lw=lw, ls=ls, zorder=2,
            solid_capstyle="round", solid_joinstyle="round")
    ax.add_patch(FancyArrowPatch(pts[-2], pts[-1], arrowstyle="-|>", mutation_scale=head,
                 lw=lw, color=color, shrinkA=0, shrinkB=0, linestyle=ls, zorder=2))


def lab(ax, x, y, s, *, fs=6.0, ha="center", color=INK, style="normal", rot=0, weight="normal"):
    ax.text(x, y, s, ha=ha, va="center", fontsize=fs, color=color,
            style=style, rotation=rot, zorder=5, fontweight=weight)


fig, ax = plt.subplots(figsize=(DCOL_W, 3.2))
ax.set_xlim(0, 16); ax.set_ylim(0, 7.15); ax.axis("off")

# ---- top: process spine ------------------------------------------------
SY, BH = 5.5, 1.45
G = (2.45, 2.8); K = (5.75, 2.8); E = (9.15, 3.05); C = (12.55, 2.8)
box(ax, G[0], SY, G[1], BH, "Edge Gateway",
    ["harmonize → 3-h cadence", "causal impute · climatology"])
box(ax, K[0], SY, K[1], BH, "Graph Construction",
    ["$k$-NN ($k{=}3$) + wind", "Gaussian $w_{ij}$,  $\\sigma{=}5$ km"])
box(ax, E[0], SY, E[1], BH, "Inductive ST-GNN Encoder",
    ["$|V|$-independent · $\\approx$24k", "93 KB · $<$3 ms / CPU"], lw=1.8)
box(ax, C[0], SY, C[1], BH, "Edge Control",
    ["inverse-standardize +", "climatology · alerts"])

# process inputs
lab(ax, G[0], 6.92, "PROCESS INPUTS", fs=7.4, weight="bold")
lab(ax, G[0], 6.60, "raw streams  ·  40 / 10 / 4 nodes", fs=5.7, color=SUB)
arrow(ax, (G[0], 6.42), (G[0], SY + BH / 2))

# spine transitions with baselines
for (c0, w0), (c1, w1), s in [(G, K, "$\\tilde{y}$"), (K, E, "$\\mathcal{G}=(V,E,W)$"), (E, C, "$\\hat{y}$")]:
    arrow(ax, (c0 + w0 / 2, SY), (c1 - w1 / 2, SY))
    lab(ax, (c0 + w0 / 2 + c1 - w1 / 2) / 2, SY + BH / 2 + 0.22, s, fs=5.8, color=SUB)

# process outputs
arrow(ax, (C[0] + C[1] / 2, SY), (C[0] + C[1] / 2 + 0.85, SY))
lab(ax, 15.2, SY + 0.16, "PROCESS", fs=7.2, weight="bold")
lab(ax, 15.2, SY - 0.18, "OUTPUTS", fs=7.2, weight="bold")

# ---- bottom: sensor-budget levers band ---------------------------------
ex0, ex1, ey0, ey1 = 0.35, 15.65, 0.45, 3.6
ax.add_patch(FancyBboxPatch((ex0, ey0), ex1 - ex0, ey1 - ey0,
             boxstyle="round,pad=0.02,rounding_size=0.06", linewidth=1.0,
             edgecolor=GRP, facecolor="#FCFCFD", linestyle=(0, (5, 2.5)), zorder=1))
lab(ax, 8.0, ey1 - 0.24, "Sensor-Budget Levers  —  where the graph earns its keep",
    fs=6.4, color=GRP, style="italic")

LY = 1.78
box(ax, 3.0, LY, 4.55, 1.55, "Virtual Sensing",
    ["estimate unsensored $U$ from sensored $S$", "$R^2$ 0.73–0.77   (+0.08–0.16)"])
box(ax, 8.0, LY, 4.65, 1.55, "Cold-start Transfer",
    ["pre-train → fine-tune, $\\leq$15% labels", "beats zero-shot & scratch 24 / 24"])
box(ax, 13.0, LY, 4.55, 1.55, "Graceful Degradation",
    ["unsensored $R^2 \\geq 0.80$", "down to 4 sensors"])

# encoder enables the levers; cold-start transfer feeds back
arrow(ax, (E[0] + 0.5, SY - BH / 2), (E[0] + 0.5, ey1), head=10)
lab(ax, E[0] + 0.5 + 0.12, (SY - BH / 2 + ey1) / 2, "enables", fs=6.0, ha="left", style="italic")
route(ax, [(8.0, LY + 1.55 / 2), (8.0, ey1 + 0.18), (E[0] - 0.6, ey1 + 0.18), (E[0] - 0.6, SY - BH / 2)],
      lw=1.4, head=9, color=XFER)
lab(ax, E[0] - 0.6 - 0.15, (ey1 + SY - BH / 2) / 2 + 0.1, "inherit  $\\theta_s$",
    fs=6.0, ha="right", color=XFER, style="italic")

fig.tight_layout(pad=0.15)
save(fig, "fig_architecture")
