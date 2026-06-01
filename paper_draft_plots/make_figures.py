"""
Generate the six figures referenced in reports/PAPER_DRAFT.md.

Outputs (300 dpi) are written next to this script as both PDF (vector, for
LaTeX) and PNG (raster, for the Markdown draft preview):

    fig1_motivation.{pdf,png}
    fig2_city_graphs.{pdf,png}
    fig3_encoder_blocks.{pdf,png}
    fig4_graph_dann.{pdf,png}
    fig5_stage1_heatmaps.{pdf,png}
    fig6_headline_bars.{pdf,png}

Run:  python paper_draft_plots/make_figures.py
"""
from __future__ import annotations
import json
import math
import os
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
from matplotlib.lines import Line2D

# ----------------------------------------------------------------------------
# Journal-style defaults. Single-column ~3.5 in, full-width (2-col span) 7.16 in.
# ----------------------------------------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))
META = os.path.normpath(os.path.join(HERE, "..", "dataset", "processed", "metadata.json"))
COL1, COL2 = 3.50, 7.16  # inches

plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 8,
    "axes.titlesize": 9,
    "axes.labelsize": 8,
    "axes.linewidth": 0.6,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
})

# Palette
C_DELHI, C_KOL, C_GUW = "#C0392B", "#2E86C1", "#27AE60"
C_LSTM, C_S1, C_S2 = "#7F8C8D", "#2E86C1", "#C0392B"
CITY_COLORS = {"Delhi": C_DELHI, "Kolkata": C_KOL, "Guwahati": C_GUW}


def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(HERE, f"{name}.{ext}"))
    plt.close(fig)
    print("wrote", name)


# ----------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------
PAIRS = ["Delhi→Kolkata", "Delhi→Guwahati", "Kolkata→Delhi",
         "Kolkata→Guwahati", "Guwahati→Delhi", "Guwahati→Kolkata"]
DCOLS = ["15%", "30%", "45%", "60%"]

# Stage 1 (PT-FT) transfer R^2 — full grid
S1 = np.array([
    [0.8113, 0.8158, 0.8182, 0.8166],
    [0.8233, 0.8165, 0.8273, 0.8220],
    [0.7953, 0.8085, 0.8198, 0.8225],
    [0.7955, 0.8044, 0.8038, 0.8157],
    [0.7887, 0.8030, 0.8123, 0.8173],
    [0.7951, 0.8085, 0.8103, 0.8177],
])
# Scratch R^2 (random init + target FT only); depends on (target, d)
SCRATCH = np.array([
    [0.7576, 0.7803, 0.7925, 0.8031],
    [0.7756, 0.7826, 0.7887, 0.8083],
    [0.7635, 0.7905, 0.7989, 0.8105],
    [0.7807, 0.7924, 0.7970, 0.7981],
    [0.7635, 0.7905, 0.7989, 0.8105],
    [0.7576, 0.7803, 0.7925, 0.8031],
])
# Zero-shot R^2 (source-only, no FT); constant across d
ZS = np.array([0.7282, 0.6841, 0.6032, 0.6698, 0.6317, 0.6980])
ZERO = np.repeat(ZS[:, None], 4, axis=1)

# Headline @ d=30 %
LSTM30 = np.array([0.8521, 0.8224, 0.8510, 0.8129, 0.8491, 0.8420])
S1_30 = np.array([0.8158, 0.8165, 0.8085, 0.8044, 0.8030, 0.8085])
S2_30 = np.array([0.8171, 0.8131, 0.8087, 0.7916, 0.8029, 0.8039])

STATION_COUNTS = {"Delhi": 40, "Kolkata": 10, "Guwahati": 4}


# ----------------------------------------------------------------------------
# Graph helpers (k-NN, k=3, haversine) from real station coordinates
# ----------------------------------------------------------------------------
def haversine(a, b):
    R = 6371.0
    la1, lo1, la2, lo2 = map(math.radians, [a[0], a[1], b[0], b[1]])
    dlat, dlon = la2 - la1, lo2 - lo1
    h = math.sin(dlat / 2) ** 2 + math.cos(la1) * math.cos(la2) * math.sin(dlon / 2) ** 2
    return 2 * R * math.asin(math.sqrt(h))


def knn_edges(coords, k=3):
    names = list(coords)
    pts = np.array([coords[n] for n in names])
    edges = []
    for i in range(len(names)):
        d = np.array([haversine(pts[i], pts[j]) if i != j else 1e9 for j in range(len(names))])
        for j in np.argsort(d)[:k]:
            edges.append((i, int(j)))
    return names, pts, edges


# ----------------------------------------------------------------------------
# Figure 1 — Motivation (3 panels)
# ----------------------------------------------------------------------------
def fig1():
    fig = plt.figure(figsize=(COL2, 2.55))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.05, 1.0, 1.05], wspace=0.32)

    # Panel A: station-count disparity
    axA = fig.add_subplot(gs[0])
    cities = ["Delhi", "Kolkata", "Guwahati"]
    vals = [STATION_COUNTS[c] for c in cities]
    bars = axA.bar(cities, vals, color=[CITY_COLORS[c] for c in cities],
                   width=0.62, edgecolor="black", linewidth=0.5)
    for b, v in zip(bars, vals):
        axA.text(b.get_x() + b.get_width() / 2, v + 1, str(v),
                 ha="center", va="bottom", fontsize=8, fontweight="bold")
    axA.set_ylabel("CPCB stations  |V|")
    axA.set_ylim(0, 46)
    axA.set_title("(A)  Monitoring-density gap", fontsize=8.5, loc="left")
    axA.annotate("10×", xy=(0, 40), xytext=(2, 40), ha="center", va="center",
                 fontsize=8, fontweight="bold", color="black",
                 arrowprops=dict(arrowstyle="<->", lw=0.8))
    axA.spines[["top", "right"]].set_visible(False)
    axA.tick_params(axis="x", labelsize=7, rotation=12)

    # Panel B: station-independent LSTM-TL (isolated nodes, no edges)
    axB = fig.add_subplot(gs[1])
    axB.set_title("(B)  Station-independent\nLSTM-TL", fontsize=8.5, loc="left")
    rng = np.random.default_rng(3)
    nb = 7
    px, py = rng.uniform(0.1, 0.9, nb), rng.uniform(0.15, 0.85, nb)
    for x, y in zip(px, py):
        axB.add_patch(Circle((x, y), 0.045, color=C_KOL, ec="black", lw=0.4, zorder=3))
        tx = np.linspace(0, 0.12, 12)
        axB.plot(x - 0.06 + tx, y + 0.075 + 0.02 * np.sin(tx * 90), lw=0.5, color="0.45")
    axB.text(0.5, 0.015, "each station modelled alone — no spatial links",
             ha="center", va="bottom", fontsize=6.6, style="italic")
    axB.set_xlim(0, 1); axB.set_ylim(0, 1); axB.axis("off")

    # Panel C: inductive GNN-TL (graph + cross-city transfer)
    axC = fig.add_subplot(gs[2])
    axC.set_title("(C)  Inductive\ngraph-TL", fontsize=8.5, loc="left")
    # source (dense) graph
    s_xy = np.array([[0.18, 0.78], [0.30, 0.86], [0.10, 0.62], [0.27, 0.64],
                     [0.40, 0.74], [0.20, 0.50], [0.36, 0.52]])
    for i in range(len(s_xy)):
        for j in range(i + 1, len(s_xy)):
            if np.linalg.norm(s_xy[i] - s_xy[j]) < 0.20:
                axC.plot(*zip(s_xy[i], s_xy[j]), lw=0.5, color=C_DELHI, alpha=0.6, zorder=1)
    axC.scatter(s_xy[:, 0], s_xy[:, 1], s=24, color=C_DELHI, ec="black", lw=0.4, zorder=3)
    axC.text(0.25, 0.93, "Delhi (40)", ha="center", fontsize=6.6, color=C_DELHI)
    # target (sparse) graph
    t_xy = np.array([[0.74, 0.34], [0.86, 0.40], [0.78, 0.20], [0.90, 0.24]])
    for i in range(len(t_xy)):
        for j in range(i + 1, len(t_xy)):
            axC.plot(*zip(t_xy[i], t_xy[j]), lw=0.5, color=C_GUW, alpha=0.6, zorder=1)
    axC.scatter(t_xy[:, 0], t_xy[:, 1], s=24, color=C_GUW, ec="black", lw=0.4, zorder=3)
    axC.text(0.82, 0.47, "Guwahati (4)", ha="center", fontsize=6.6, color=C_GUW)
    axC.add_patch(FancyArrowPatch((0.42, 0.58), (0.72, 0.34),
                  arrowstyle="-|>", mutation_scale=11, lw=1.3, color="black", zorder=4))
    axC.text(0.55, 0.50, "same θ\ntransferred", ha="center", fontsize=6.4,
             fontweight="bold", rotation=-23)
    axC.set_xlim(0, 1); axC.set_ylim(0, 1); axC.axis("off")

    fig.suptitle("Sparse-network cities need transfer learning; a graph model adds the spatial coupling the LSTM ignores",
                 fontsize=8.2, y=1.04)
    save(fig, "fig1_motivation")


# ----------------------------------------------------------------------------
# Figure 2 — the three city graphs (real coords, k-NN k=3)
# ----------------------------------------------------------------------------
def fig2():
    meta = json.load(open(META))
    fig, axes = plt.subplots(1, 3, figsize=(COL2, 2.7))
    for ax, city in zip(axes, ["Delhi", "Kolkata", "Guwahati"]):
        names, pts, edges = knn_edges(meta[city]["coords"], k=3)
        lon, lat = pts[:, 1], pts[:, 0]
        col = CITY_COLORS[city]
        for i, j in edges:
            ax.plot([lon[i], lon[j]], [lat[i], lat[j]], lw=0.45, color=col, alpha=0.45, zorder=1)
        ax.scatter(lon, lat, s=22 if city != "Delhi" else 14, color=col,
                   ec="black", lw=0.35, zorder=3)
        ax.set_title(f"{city}\n|V|={len(names)}, |E|={len(edges)}, deg=3.0",
                     fontsize=8, color=col)
        ax.set_xlabel("lon"); ax.set_aspect("equal", adjustable="datalim")
        ax.tick_params(labelsize=6)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("lat")
    fig.suptitle("k-NN station graphs (k = 3, Gaussian-decay edge weights, σ = 5 km) — one inductive encoder runs on all three",
                 fontsize=8.0, y=1.02)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    save(fig, "fig2_city_graphs")


# ----------------------------------------------------------------------------
# Figure 3 — encoder block diagram
# ----------------------------------------------------------------------------
def _box(ax, x, y, w, h, text, fc, ec="black"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.02",
                 linewidth=0.7, edgecolor=ec, facecolor=fc, zorder=2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=7.2, zorder=3)


def _arrow(ax, x0, y0, x1, y1):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>",
                 mutation_scale=9, lw=0.9, color="black", zorder=1))


def fig3():
    fig, ax = plt.subplots(figsize=(COL2, 2.35))
    ax.set_xlim(0, 16); ax.set_ylim(0, 5); ax.axis("off")
    blocks = [
        ("Input\n[B,H,N,F]", "#ECF0F1"),
        ("TemporalConv₁\n(per-node, d=1)", "#FCF3CF"),
        ("GAT₁ → ELU\n(spatial)", "#D6EAF8"),
        ("GAT₂ → ELU\n(spatial)", "#D6EAF8"),
        ("TemporalConv₂\n(per-node, d=2)", "#FCF3CF"),
        ("LayerNorm +\nLinear head", "#ECF0F1"),
        ("Output\n[B,N]", "#E8F8F5"),
    ]
    w, h, gap = 1.95, 1.5, 0.27
    x = 0.15
    centers = []
    for txt, fc in blocks:
        _box(ax, x, 2.0, w, h, txt, fc)
        centers.append(x + w)
        x += w + gap
    for c in centers[:-1]:
        _arrow(ax, c, 2.75, c + gap, 2.75)
    # shape annotations under arrows
    shapes = ["[B,H,N,64]", "→[B·H,N,64]", "[B·H,N,64]", "→[B,H,N,64]", "[B,N,64]"]
    xs = [centers[i] + gap / 2 for i in range(len(shapes))]
    for sx, s in zip(xs, shapes):
        ax.text(sx, 1.78, s, ha="center", va="top", fontsize=5.6, color="0.35")
    # banner
    ax.add_patch(FancyBboxPatch((0.15, 4.05), 15.55, 0.62,
                 boxstyle="round,pad=0.02,rounding_size=0.05",
                 fc="#FDEDEC", ec=C_DELHI, lw=0.8))
    ax.text(7.9, 4.36, "No block has a parameter whose shape depends on |V|  →  same ~25k-param θ runs on N = 40, 10, or 4",
            ha="center", va="center", fontsize=7.2, color=C_DELHI, fontweight="bold")
    ax.text(7.9, 0.7, "GAT layers are vectorized over the B·H per-timestep graphs; TemporalConv mixes only along time, per node.",
            ha="center", va="center", fontsize=6.4, style="italic", color="0.3")
    save(fig, "fig3_encoder_blocks")


# ----------------------------------------------------------------------------
# Figure 4 — Graph-DANN architecture + lambda(p) inset
# ----------------------------------------------------------------------------
def fig4():
    fig, ax = plt.subplots(figsize=(COL2, 3.05))
    ax.set_xlim(0, 16); ax.set_ylim(0, 9); ax.axis("off")

    # round-robin minibatch (3 cities)
    cnames = [("Delhi (source)", C_DELHI), ("Guwahati (target,\nlabels withheld)", C_GUW),
              ("Kolkata (3rd-city\nreplay)", C_KOL)]
    for i, (nm, c) in enumerate(cnames):
        y = 6.6 - i * 2.2
        _box(ax, 0.2, y, 2.5, 1.5, nm, "#FBFCFC", ec=c)
        _arrow(ax, 2.7, y + 0.75, 3.7, 4.5 + (1 - i) * 0.0)
    ax.text(1.45, 8.5, "round-robin\nminibatch", ha="center", fontsize=6.6, fontweight="bold")

    _box(ax, 3.75, 3.7, 2.7, 1.6, "Inductive\nST-GNN encoder\n(shared θ)", "#D6EAF8")
    _arrow(ax, 6.45, 4.5, 7.35, 4.5)
    _box(ax, 7.4, 3.7, 2.5, 1.6, "mean + max\ngraph readout\nz = [mean‖max]", "#E8F8F5")

    # forecast head (top branch)
    _arrow(ax, 9.9, 5.0, 10.9, 6.2)
    _box(ax, 10.95, 5.7, 2.4, 1.2, "Forecast head\nMSE(ŷ, y)", "#FCF3CF")

    # GRL + discriminator (bottom branch)
    _arrow(ax, 9.9, 4.0, 10.9, 2.8)
    _box(ax, 10.95, 2.2, 1.7, 1.2, "GRL\n(×−λ)", "#FADBD8")
    _arrow(ax, 12.65, 2.8, 13.4, 2.8)
    _box(ax, 13.45, 2.1, 2.3, 1.4, "3-way city\ndiscriminator\nCE(city)", "#F9EBEA")
    ax.text(13.5, 0.95, "encoder is trained to FOOL the discriminator\n→ city-invariant embedding",
            ha="center", fontsize=6.2, style="italic", color="0.3")

    # lambda(p) inset — placed in the free top-centre region (above the encoder)
    iax = fig.add_axes([0.47, 0.74, 0.18, 0.17])
    p = np.linspace(0, 1, 200)
    lam = 2 / (1 + np.exp(-10 * p)) - 1
    iax.plot(p, lam, color=C_DELHI, lw=1.2)
    iax.set_title("λ(p) warm-up", fontsize=6.2)
    iax.set_xlabel("training progress p", fontsize=5.6)
    iax.set_ylabel("λ", fontsize=5.6)
    iax.tick_params(labelsize=5)
    iax.set_xlim(0, 1); iax.set_ylim(0, 1.05)

    fig.suptitle("Graph-DANN: adversarial alignment of the size-invariant graph embedding across cities",
                 fontsize=8.2, y=0.99)
    save(fig, "fig4_graph_dann")


# ----------------------------------------------------------------------------
# Figure 5 — Stage 1 verification small multiples (zero-shot / scratch / transfer)
# ----------------------------------------------------------------------------
def fig5():
    fig, axes = plt.subplots(1, 3, figsize=(COL2, 3.0))
    mats = [("Zero-shot", ZERO), ("Scratch", SCRATCH), ("Transfer (PT-FT)", S1)]
    vmin, vmax = 0.60, 0.84
    im = None
    for ax, (title, M) in zip(axes, mats):
        im = ax.imshow(M, cmap="viridis", vmin=vmin, vmax=vmax, aspect="auto")
        ax.set_title(title, fontsize=8.2)
        ax.set_xticks(range(4)); ax.set_xticklabels(DCOLS, fontsize=6.5)
        ax.set_xlabel("target data fraction d")
        for i in range(6):
            for j in range(4):
                ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center",
                        fontsize=5.7, color="white" if M[i, j] < 0.77 else "black")
    axes[0].set_yticks(range(6)); axes[0].set_yticklabels(PAIRS, fontsize=6.2)
    axes[1].set_yticks([]); axes[2].set_yticks([])
    cbar = fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02)
    cbar.set_label("test R²", fontsize=7)
    cbar.ax.tick_params(labelsize=6)
    fig.suptitle("Stage 1 (PT-FT): transfer beats both zero-shot and scratch in all 24 cells",
                 fontsize=8.2, x=0.45, y=1.0)
    save(fig, "fig5_stage1_heatmaps")


# ----------------------------------------------------------------------------
# Figure 6 — headline bar chart @ d=30 %
# ----------------------------------------------------------------------------
def fig6():
    fig, ax = plt.subplots(figsize=(COL2, 3.0))
    x = np.arange(len(PAIRS))
    w = 0.27
    ax.bar(x - w, LSTM30, w, label="LSTM-TL (fixed)", color=C_LSTM, ec="black", lw=0.4)
    ax.bar(x, S1_30, w, label="GNN Stage 1 (PT-FT)", color=C_S1, ec="black", lw=0.4)
    ax.bar(x + w, S2_30, w, label="GNN Stage 2 (Graph-DANN)", color=C_S2, ec="black", lw=0.4)
    for xi, vals in zip(x, zip(LSTM30, S1_30, S2_30)):
        for off, v in zip((-w, 0, w), vals):
            ax.text(xi + off, v + 0.004, f"{v:.2f}", ha="center", va="bottom", fontsize=5.2, rotation=90)
    ax.set_ylim(0.74, 0.875)
    ax.set_ylabel("transfer test R²  (d = 30 %)")
    ax.set_xticks(x); ax.set_xticklabels(PAIRS, rotation=18, ha="right", fontsize=6.6)
    ax.legend(loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.13))
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", lw=0.4, alpha=0.4)
    save(fig, "fig6_headline_bars")


if __name__ == "__main__":
    fig1(); fig2(); fig3(); fig4(); fig5(); fig6()
    print("All figures written to", HERE)
