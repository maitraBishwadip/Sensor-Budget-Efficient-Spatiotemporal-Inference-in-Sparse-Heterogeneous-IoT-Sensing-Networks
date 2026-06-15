"""F3: Inductive ST-GNN encoder, dense but compact (landscape) block diagram.

(Top) a left-to-right layer spine (Input -> TConv -> GAT x2 -> TCN -> LayerNorm ->
head -> PM2.5) with the running tensor shape labelled on every transition.
(Bottom) an operator-detail row carrying the temporal-conv, graph-attention, and
read-out equations. A size-invariance note asserts one weight set runs on N=40,10,4.
Kept short (~3.1 in tall) so the full-width float fits the 8-page budget.

Writes: paper_figs/fig_encoder.{pdf,png}
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
SHAPE = "#1f5fa8"


def box(ax, cx, cy, w, h, title, subs=None, *, tfs=6.8, sfs=6.1, lw=1.2, ec=INK, fc="white"):
    ax.add_patch(Rectangle((cx - w / 2, cy - h / 2), w, h, facecolor=fc,
                 edgecolor=ec, linewidth=lw, zorder=3))
    if subs:
        ax.text(cx, cy + h / 2 - 0.22, title, ha="center", va="top",
                fontsize=tfs, fontweight="bold", color=INK, zorder=4)
        ax.text(cx, cy + h / 2 - 0.56, "\n".join(subs), ha="center", va="top",
                fontsize=sfs, color=SUB, zorder=4, linespacing=1.45)
    else:
        ax.text(cx, cy, title, ha="center", va="center", fontsize=tfs,
                fontweight="bold", color=INK, zorder=4, linespacing=1.25)


def arrow(ax, p0, p1, *, lw=1.2, head=9, color=INK, ls="-"):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=head,
                 lw=lw, color=color, shrinkA=0, shrinkB=0, linestyle=ls, zorder=2))


def lab(ax, x, y, s, *, fs=6.0, ha="center", color=INK, style="normal", rot=0, weight="normal"):
    ax.text(x, y, s, ha=ha, va="center", fontsize=fs, color=color,
            style=style, rotation=rot, zorder=5, fontweight=weight)


fig, ax = plt.subplots(figsize=(DCOL_W, 3.12))
ax.set_xlim(0, 16); ax.set_ylim(0, 6.97); ax.axis("off")

lab(ax, 7.0, 6.72, "Inductive ST-GNN encoder  —  no parameter shape depends on $|V|$",
    fs=8.0, weight="bold")

# ---- top: left-to-right layer spine ------------------------------------
SY, BH, BW = 5.05, 0.96, 1.80
xs = [1.05 + i * 2.0 for i in range(7)]
names = ["Input\n$X$", "TConv$_1$\n$K{=}3,\\,\\delta{=}1$", "GAT$_1$\n$+$ ELU",
         "GAT$_2$\n$+$ ELU", "TConv$_2$\n$K{=}3,\\,\\delta{=}2$",
         "LayerNorm\nlast step", "Linear\nhead"]
shapes = ["$[B,H,N,F]$", "$[B,H,N,64]$", "$[B{\\cdot}H,N,64]$",
          "$[B{\\cdot}H,N,64]$", "$[B,H,N,64]$", "$[B,N,64]$"]
for x, t in zip(xs, names):
    box(ax, x, SY, BW, BH, t)
for i in range(6):
    x0, x1 = xs[i] + BW / 2, xs[i + 1] - BW / 2
    arrow(ax, (x0, SY), (x1, SY))
    lab(ax, (x0 + x1) / 2, SY + BH / 2 + 0.26, shapes[i], fs=5.2, color=SHAPE)
# output
arrow(ax, (xs[6] + BW / 2, SY), (xs[6] + BW / 2 + 0.95, SY))
lab(ax, xs[6] + BW / 2 + 0.55, SY + BH / 2 + 0.26, "$[B,N]$", fs=5.2, color=SHAPE)
lab(ax, xs[6] + BW / 2 + 1.7, SY, "$\\hat{y}$ :  PM$_{2.5}$\n(+3 h)", fs=7.0, weight="bold")

# ---- dashed detail connectors to the operator row ----------------------
OY, OH = 2.02, 2.04
ot, og, oh = 2.85, 8.05, 13.15
det = dict(ls=(0, (4, 3)), color=GRP, lw=0.8, head=7)
arrow(ax, (xs[1], SY - BH / 2), (ot, OY + OH / 2), **det)
arrow(ax, (xs[2], SY - BH / 2), (og - 1.0, OY + OH / 2), **det)
arrow(ax, (xs[3], SY - BH / 2), (og + 1.0, OY + OH / 2), **det)
arrow(ax, (xs[6], SY - BH / 2), (oh, OY + OH / 2), **det)

# ---- bottom: operator detail -------------------------------------------
box(ax, ot, OY, 4.9, OH, "Temporal block  (dilated causal TCN)",
    ["1-D conv per node  ·  $K{=}3$  ·  $\\delta\\in\\{1,2\\}$",
     "$(\\mathrm{TConv}\\,x)_{n,t}=\\sum_{k}\\Theta_k\\,x_{n,\\,t-\\delta k}+b$"], lw=1.0)
box(ax, og, OY, 5.3, OH, "Edge-weighted graph attention  (GAT)",
    ["$e_{ij}=\\mathrm{LeakyReLU}(a_s^{\\top}Wx_i+a_d^{\\top}Wx_j)+\\log w_{ij}$",
     "$\\alpha_{ij}=\\mathrm{softmax}_j(e_{ij})$ ,   "
     "$h_j=\\sum_{i\\in\\mathcal{N}(j)}\\alpha_{ij}\\,Wx_i$"], lw=1.0, sfs=6.0)
box(ax, oh, OY, 4.7, OH, "Read-out",
    ["per-node linear head  →  $\\hat{y}$",
     "$|\\theta|$ independent of node count $N$"], lw=1.0)

lab(ax, 8.0, 0.42, "size-invariant:  the same $\\theta$ ($\\approx$24k params) runs verbatim on  $N = 40,\\ 10,\\ 4$",
    fs=6.2, color=GRP, style="italic")

fig.tight_layout(pad=0.15)
save(fig, "fig_encoder")
