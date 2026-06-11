"""F3: Inductive ST-GNN encoder, redrawn in a clean colour-coded block style.

Top: the spatio-temporal pipeline (TCN -> GAT x2 -> TCN -> LayerNorm -> head).
Bottom: a zoom-in on one edge-weighted graph-attention layer (the GAT internals),
echoing the "build up / zoom" style of standard architecture diagrams.

Writes: paper_figs/fig_encoder.{pdf,png}
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from figstyle import DCOL_W, save

C = dict(grey="#ECEFF1", purple="#D1C4E9", tan="#FFE0B2", yellow="#FFF59D",
         blue="#BBDEFB", green="#C8E6C9")
EC = "#5f6368"


def box(ax, cx, cy, w, h, text, fc, fs=6.8, bold=False):
    ax.add_patch(FancyBboxPatch((cx - w / 2, cy - h / 2), w, h,
                 boxstyle="round,pad=0.02,rounding_size=0.05",
                 linewidth=0.8, edgecolor=EC, facecolor=fc, zorder=3))
    ax.text(cx, cy, text, ha="center", va="center", fontsize=fs, zorder=4,
            fontweight="bold" if bold else "normal")


def arrow(ax, p0, p1, color=EC, lw=1.0, style="-|>", ls="-"):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle=style, mutation_scale=8,
                 lw=lw, color=color, linestyle=ls, zorder=2))


fig, ax = plt.subplots(figsize=(DCOL_W, 2.8))
ax.set_xlim(0, 12)
ax.set_ylim(0, 4.4)
ax.axis("off")

# --- top: encoder pipeline ----------------------------------------------
yt, bw, bh = 3.5, 1.18, 0.98
xs = [0.85 + i * 1.46 for i in range(8)]
stack = [("$X$\n$[B,H,N,F]$", C["grey"]), ("TConv\n(d=1)", C["tan"]),
         ("GAT", C["blue"]), ("GAT", C["blue"]), ("TConv\n(d=2)", C["tan"]),
         ("Layer\nNorm", C["yellow"]), ("Linear\nhead", C["green"]),
         (r"$\hat{y}$" + "\n$[B,N]$", C["grey"])]
for x, (t, c) in zip(xs, stack):
    box(ax, x, yt, bw, bh, t, c, bold=(c == C["blue"]))
for i in range(7):
    arrow(ax, (xs[i] + bw / 2, yt), (xs[i + 1] - bw / 2, yt))
ax.text(6.0, 4.28, r"Inductive ST-GNN encoder  ($|V|$-independent)",
        ha="center", fontsize=8, fontweight="bold")

# --- bottom: zoom into one GAT layer ------------------------------------
cx0, cx1, cy0, cy1 = 1.6, 10.4, 0.42, 2.05
ax.add_patch(FancyBboxPatch((cx0, cy0), cx1 - cx0, cy1 - cy0,
             boxstyle="round,pad=0.02,rounding_size=0.06",
             linewidth=0.8, edgecolor=C["blue"], facecolor="#F5F9FE", zorder=1))
ax.text((cx0 + cx1) / 2, cy1 - 0.16, "Edge-weighted graph attention (one GAT layer)",
        ha="center", fontsize=6.8, color="#1f5fa8", style="italic")
# dashed zoom connectors from the first GAT box
arrow(ax, (xs[2] - bw / 2, yt - bh / 2), (cx0 + 0.25, cy1), color=C["blue"], lw=0.7, style="-", ls=(0, (4, 3)))
arrow(ax, (xs[2] + bw / 2, yt - bh / 2), (cx1 - 0.25, cy1), color=C["blue"], lw=0.7, style="-", ls=(0, (4, 3)))

yb, iw, ih = 1.0, 1.25, 0.72
ix = [2.5 + i * 1.45 for i in range(6)]
inner = [("$x_i,\\,x_j$", C["grey"]), ("Linear $W$", C["purple"]),
         ("score $e_{ij}$\n$+\\log w_{ij}$", C["tan"]), ("softmax\n$\\alpha_{ij}$", C["yellow"]),
         ("aggregate", C["blue"]), ("$h_j$", C["green"])]
for x, (t, c) in zip(ix, inner):
    box(ax, x, yb, iw, ih, t, c, fs=6.3)
for i in range(5):
    arrow(ax, (ix[i] + iw / 2, yb), (ix[i + 1] - iw / 2, yb))

fig.tight_layout(pad=0.2)
save(fig, "fig_encoder")
