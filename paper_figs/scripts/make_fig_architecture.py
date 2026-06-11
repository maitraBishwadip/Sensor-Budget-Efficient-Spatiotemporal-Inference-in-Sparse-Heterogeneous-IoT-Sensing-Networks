"""F1: System architecture, publication-formal colour-coded block diagram.

Pipeline: sparse heterogeneous sensor nodes -> edge gateway -> graph construction ->
inductive |V|-independent ST-GNN encoder -> two edge inference services (virtual sensing
+ forecasting). A dense source deployment feeds a cold-start transfer arc into the encoder.

Writes: paper_figs/fig_architecture.{pdf,png}
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from figstyle import DCOL_W, save

C = dict(grey="#EAEDEF", purple="#D6CCEC", tan="#FBE0B6", yellow="#FBF1A6",
         blue="#BBD9F4", green="#C6E6CC", orange="#FAC887", srcfill="#FBEAD6")
BORD = "#4A4F54"          # uniform box border
LINE = "#3F4A54"          # connector colour
TRANSFER = "#C0641A"      # cold-start accent


def box(ax, cx, cy, w, h, text, fc, ec=BORD, fs=7.0, bold=False):
    ax.add_patch(FancyBboxPatch((cx - w / 2, cy - h / 2), w, h,
                 boxstyle="round,pad=0.02,rounding_size=0.05",
                 linewidth=1.0, edgecolor=ec, facecolor=fc, zorder=3))
    ax.text(cx, cy, text, ha="center", va="center", fontsize=fs, zorder=4,
            fontweight="bold" if bold else "normal", color="#1c1c1c")


def connect(ax, p0, p1, color=LINE, lw=1.3, rad=0.0):
    """A single clean directed connector with a consistent arrowhead."""
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=11,
                 lw=lw, color=color, shrinkA=0, shrinkB=0,
                 connectionstyle=f"arc3,rad={rad}", joinstyle="round", zorder=2))


fig, ax = plt.subplots(figsize=(DCOL_W, 2.85))
ax.set_xlim(0, 12.8)
ax.set_ylim(0, 5.0)
ax.axis("off")

y = 3.55
# --- spine boxes (centre, width) ----------------------------------------
A = (1.25, 2.05); Bx = (3.70, 1.90); Cx = (5.95, 1.85); D = (8.35, 2.05)
box(ax, A[0],  y, A[1],  1.5,  "Sparse,\nheterogeneous\nsensor nodes\n(40 / 10 / 4)", C["grey"])
box(ax, Bx[0], y, Bx[1], 1.18, "Edge gateway\nharmonize +\nclimatology residual", C["tan"])
box(ax, Cx[0], y, Cx[1], 1.02, "Graph build\n$k$-NN + wind-aware", C["purple"])
box(ax, D[0],  y, D[1],  1.32, "Inductive\nST-GNN encoder\n($|V|$-independent)", C["blue"], bold=True)

# straight edge-to-edge connectors along the spine
for (c0, w0), (c1, w1) in [(A, Bx), (Bx, Cx), (Cx, D)]:
    connect(ax, (c0 + w0 / 2, y), (c1 - w1 / 2, y))

# --- edge inference services group --------------------------------------
gx0, gx1, gy0, gy1 = 9.75, 12.75, 2.12, 4.78
ax.add_patch(FancyBboxPatch((gx0, gy0), gx1 - gx0, gy1 - gy0,
             boxstyle="round,pad=0.02,rounding_size=0.05", linewidth=0.9,
             edgecolor="#9aa0a6", facecolor="#FBFBFC", linestyle=(0, (5, 2.5)), zorder=1))
ax.text((gx0 + gx1) / 2, gy1 - 0.18, "Edge inference services", ha="center",
        fontsize=6.4, color="#5f6368", style="italic")
E1 = (11.25, 3.98); E2 = (11.25, 2.74); ew, eh = 2.5, 0.82
box(ax, E1[0], E1[1], ew, eh, "Virtual sensing\n(unsensored nodes)", C["green"])
box(ax, E2[0], E2[1], ew, eh, "Forecasting  ($+3$ h)", C["orange"])

# symmetric fan from the encoder's right edge to each service
d_right = D[0] + D[1] / 2
connect(ax, (d_right, y + 0.22), (E1[0] - ew / 2, E1[1]), rad=0.16)
connect(ax, (d_right, y - 0.22), (E2[0] - ew / 2, E2[1]), rad=-0.16)

# --- cold-start transfer feed -------------------------------------------
S = (1.25, 1.02, 2.05, 0.85)            # cx, cy, w, h
box(ax, S[0], S[1], S[2], S[3], "Dense source\ndeployment", C["srcfill"], ec=TRANSFER)
connect(ax, (S[0] + S[2] / 2, S[1] + 0.20), (D[0], y - D[1] / 2),
        color=TRANSFER, lw=1.5, rad=-0.16)
ax.text(5.15, 1.5, "cold-start transfer  (pre-train $\\rightarrow$ fine-tune)",
        fontsize=6.4, color=TRANSFER, style="italic", ha="center")

fig.tight_layout(pad=0.25)
save(fig, "fig_architecture")
