"""
Generate the six figures referenced in reports/PAPER_DRAFT.md.

Outputs (300 dpi) are written next to this script as both PDF (vector, for
LaTeX) and PNG (raster, for the Markdown draft preview):

    fig1_motivation.{pdf,png}      (draft Fig 1)
    fig_pipeline.{pdf,png}         (draft Fig 2 — end-to-end methodology)
    fig_lstm_arch.{pdf,png}        (draft Fig 3 — LSTM-TL architecture)
    fig3_encoder_blocks.{pdf,png}  (draft Fig 4 — ST-GNN encoder blocks)
    fig_gnn_blocks.{pdf,png}       (draft Fig 5 — TCN + GAT building blocks)
    fig2_city_graphs.{pdf,png}     (draft Fig 6 — k-NN city graphs)
    fig4_graph_dann.{pdf,png}      (draft Fig 7 — Graph-DANN)
    fig5_stage1_heatmaps.{pdf,png} (draft Fig 8 — Stage-1 verification)
    fig6_headline_bars.{pdf,png}   (draft Fig 9 — cross-method headline)

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


# ----------------------------------------------------------------------------
# Figure — end-to-end methodology pipeline (data → models → results)
# ----------------------------------------------------------------------------
def fig_pipeline():
    fig, ax = plt.subplots(figsize=(COL2, 5.2))
    ax.set_xlim(0, 16); ax.set_ylim(0, 26); ax.axis("off")

    def box(x, y, w, h, t, fc, ec="black", fs=7.0):
        ax.add_patch(FancyBboxPatch((x, y), w, h,
                     boxstyle="round,pad=0.02,rounding_size=0.12",
                     lw=0.8, edgecolor=ec, facecolor=fc, zorder=2))
        ax.text(x + w / 2, y + h / 2, t, ha="center", va="center", fontsize=fs, zorder=3)

    def arr(x0, y0, x1, y1, c="black", lw=1.1):
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>",
                     mutation_scale=11, lw=lw, color=c, zorder=1))

    cx = 8.0
    # 1 — data collection
    box(cx - 5.4, 23.4, 10.8, 1.9,
        "1)  DATA COLLECTION  —  CPCB CAAQM, 3-hour cadence\nDelhi (40 stations, 2021–22) · Kolkata (10, 2023) · Guwahati (4, 2023)",
        "#ECF0F1", fs=7.2)
    arr(cx, 23.4, cx, 22.7)
    # 2 — preprocessing & protocol
    box(cx - 5.4, 20.6, 10.8, 2.1,
        "2)  PRE-PROCESSING & PROTOCOL\nschema harmonization · cyclic time + season encodings (F = 14) · causal imputation\ninterleaved 70/15/15 split · per-(month, hour) climatology-residual + z-score (train-only)",
        "#FCF3CF", fs=6.8)
    arr(cx, 20.6, cx, 19.9)
    # 3 — graph construction (feeds GNN branch only)
    box(cx - 5.4, 18.1, 10.8, 1.7,
        "3)  GRAPH CONSTRUCTION (GNN branch)  —  per-city k-NN (k = 3), Gaussian edge weights w_ij = exp(−d²/2σ²), σ = 5 km",
        "#D6EAF8", fs=6.8)
    arr(cx - 3.0, 18.1, cx - 3.0, 17.0)   # to LSTM branch
    arr(cx + 3.0, 18.1, cx + 3.0, 17.0)   # to GNN branch

    # ---- two parallel branches ----
    lx, gx = cx - 3.0, cx + 3.0
    # branch headers
    box(lx - 2.5, 15.3, 5.0, 1.5, "LSTM-TL branch\n(station-independent)", "#EAECEE", ec=C_LSTM, fs=7.0)
    box(gx - 2.5, 15.3, 5.0, 1.5, "GNN-TL branch\n(inductive ST-GNN)", "#D6EAF8", ec=C_S1, fs=7.0)
    arr(lx, 15.3, lx, 14.6); arr(gx, 15.3, gx, 14.6)
    # Stage 0
    box(lx - 2.5, 13.1, 5.0, 1.5, "Stage 0 — pre-train\non source city", "#FBFCFC", ec=C_LSTM, fs=6.8)
    box(gx - 2.5, 13.1, 5.0, 1.5, "Stage 0 — pre-train\non source city", "#FBFCFC", ec=C_S1, fs=6.8)
    arr(lx, 13.1, lx, 12.4); arr(gx, 13.1, gx, 12.4)
    # Stage 1 / Stage 2
    box(lx - 2.5, 10.6, 5.0, 1.8, "Stage 1 — fine-tune\non target d %\n(no graph, no adversary)", "#FBFCFC", ec=C_LSTM, fs=6.6)
    box(gx - 2.5, 10.6, 5.0, 1.8,
        "Stage 1 — PT-FT  on target d %\n— or —\nStage 2 — Graph-DANN\n(adversarial alignment) then FT", "#EBF5FB", ec=C_S1, fs=6.4)
    # converge
    arr(lx, 10.6, cx - 0.4, 9.2); arr(gx, 10.6, cx + 0.4, 9.2)

    box(cx - 5.4, 7.4, 10.8, 1.8,
        "4)  THREE-WAY VERIFICATION (per source→target × d cell)\nzero-shot  vs  scratch (random init)  vs  transfer  →  real iff transfer > both",
        "#D5F5E3", ec=C_GUW, fs=6.9)
    arr(cx, 7.4, cx, 6.7)
    box(cx - 5.4, 5.0, 10.8, 1.6,
        "5)  EVALUATION on held-out target test\nR² · MAE · RMSE · MAPE  in raw µg/m³ (inverse z-score + climatology add-back)",
        "#FEF9E7", fs=6.9)
    arr(cx, 5.0, cx, 4.3)
    box(cx - 5.4, 2.2, 10.8, 2.0,
        "6)  RESULTS  —  24-cell grid per method (6 pairs × 4 d)\nPT-FT 24/24 · Graph-DANN 23/24 real transfer · GNN within ≈0.04 R² of LSTM\nand uniquely transfers across |V| in {4, 10, 40} with no shape change",
        "#FDEDEC", ec=C_DELHI, fs=6.8)

    fig.suptitle("End-to-end experimental methodology", fontsize=9.5, y=0.99, fontweight="bold")
    save(fig, "fig_pipeline")


# ----------------------------------------------------------------------------
# Figure — LSTM architecture
# ----------------------------------------------------------------------------
def fig_lstm_arch():
    fig, ax = plt.subplots(figsize=(COL2, 3.0))
    ax.set_xlim(0, 16); ax.set_ylim(0, 9); ax.axis("off")

    # input window grid [H x F] for one station
    ax.text(1.5, 8.3, "one station's\n24 h window", ha="center", fontsize=6.8, fontweight="bold")
    gx0, gy0, cw, ch = 0.5, 4.2, 0.26, 0.26
    H, Fsmall = 8, 6
    for r in range(Fsmall):
        for c in range(H):
            ax.add_patch(FancyBboxPatch((gx0 + c * cw, gy0 + r * ch), cw * 0.9, ch * 0.9,
                         boxstyle="square,pad=0", lw=0.3, ec="0.5", fc="#D6EAF8"))
    ax.text(gx0 + H * cw / 2, gy0 - 0.35, "x₁..x₈  (H=8)", ha="center", fontsize=6.2)
    ax.text(gx0 - 0.35, gy0 + Fsmall * ch / 2, "F=14", rotation=90, va="center", fontsize=6.2)
    arr = lambda a, b, cc=("0",): None

    def A(x0, y0, x1, y1, c="black", lw=1.0):
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>",
                     mutation_scale=9, lw=lw, color=c, zorder=1))

    A(3.1, 5.3, 3.9, 5.3)

    # unrolled 2-layer LSTM (4 visible steps + ellipsis)
    steps_x = [4.2, 5.5, 6.8, 8.6]
    labels = ["t=1", "t=2", "t=3", "t=8"]
    for layer, yy in [(1, 4.4), (2, 6.0)]:
        for k, x in enumerate(steps_x):
            ax.add_patch(FancyBboxPatch((x, yy), 1.0, 1.1,
                         boxstyle="round,pad=0.02,rounding_size=0.06", lw=0.7,
                         ec="black", fc="#AED6F1" if layer == 1 else "#85C1E9"))
            ax.text(x + 0.5, yy + 0.55, f"LSTM\nL{layer}", ha="center", va="center", fontsize=6.0)
            if k < len(steps_x) - 1:
                A(x + 1.0, yy + 0.55, steps_x[k + 1], yy + 0.55, lw=0.8)
        # vertical (layer1 -> layer2)
    for x in steps_x:
        A(x + 0.5, 5.5, x + 0.5, 6.0, lw=0.8)
    ax.text(7.85, 4.95, "···", fontsize=11, ha="center")
    ax.text(6.4, 7.4, "2-layer LSTM (hidden = 64, dropout 0.2), shared across all stations",
            ha="center", fontsize=6.6, style="italic")

    # last hidden -> FC head
    A(9.6, 6.55, 10.4, 6.55)
    ax.add_patch(FancyBboxPatch((10.4, 6.0), 1.7, 1.1, boxstyle="round,pad=0.02,rounding_size=0.06",
                 lw=0.7, ec="black", fc="#FCF3CF"))
    ax.text(11.25, 6.55, "h₈ (64)", ha="center", va="center", fontsize=6.6)
    A(12.1, 6.55, 12.8, 6.55)
    ax.add_patch(FancyBboxPatch((12.8, 6.0), 1.9, 1.1, boxstyle="round,pad=0.02,rounding_size=0.06",
                 lw=0.7, ec="black", fc="#FCF3CF"))
    ax.text(13.75, 6.55, "FC 64→64\n→ReLU→ 64→1", ha="center", va="center", fontsize=6.0)
    A(14.7, 6.55, 15.4, 6.55)
    ax.text(15.5, 6.55, "ŷ\n(+3 h\nPM2.5)", ha="left", va="center", fontsize=6.4, fontweight="bold")

    # gate equations box
    ax.add_patch(FancyBboxPatch((10.2, 2.3, ), 5.5, 2.2, boxstyle="round,pad=0.04,rounding_size=0.05",
                 lw=0.6, ec="0.5", fc="#FBFCFC"))
    ax.text(12.95, 4.2, "LSTM cell (per step)", ha="center", fontsize=6.4, fontweight="bold")
    ax.text(10.4, 3.95, "i,f,o = σ(W·[hₜ₋₁,xₜ]+b)\ng = tanh(W_g·[hₜ₋₁,xₜ]+b_g)\ncₜ = f·cₜ₋₁ + i·g\nhₜ = o·tanh(cₜ)",
            ha="left", va="top", fontsize=6.0)

    # TL note
    ax.add_patch(FancyBboxPatch((0.5, 2.3), 8.8, 1.0, boxstyle="round,pad=0.03,rounding_size=0.05",
                 lw=0.7, ec=C_LSTM, fc="#EAECEE"))
    ax.text(4.9, 2.8, "Transfer: pre-train on source → fine-tune on target d %.\n~50k params, independent of N (applied per station).",
            ha="center", va="center", fontsize=6.3)

    fig.suptitle("Station-independent LSTM-TL architecture", fontsize=9.0, y=1.0, fontweight="bold")
    save(fig, "fig_lstm_arch")


# ----------------------------------------------------------------------------
# Figure — GNN building blocks: dilated causal TCN + edge-weighted GAT
# ----------------------------------------------------------------------------
def fig_gnn_blocks():
    fig, axes = plt.subplots(1, 2, figsize=(COL2, 3.1), gridspec_kw={"width_ratios": [1.0, 1.05]})

    # (a) dilated causal TemporalConv
    axA = axes[0]; axA.set_xlim(0, 10); axA.set_ylim(0, 10); axA.axis("off")
    axA.set_title("(a)  TemporalConv — dilated causal 1-D conv (per node)", fontsize=7.6, loc="left")
    xs = np.linspace(0.8, 9.2, 8)
    yb, yt = 2.2, 6.6
    for i, x in enumerate(xs):
        axA.add_patch(Circle((x, yb), 0.28, fc="#D6EAF8", ec="black", lw=0.5, zorder=3))
        axA.text(x, yb - 0.7, f"t{i+1}", ha="center", fontsize=5.8)
    axA.add_patch(Circle((xs[-1], yt), 0.30, fc="#FCF3CF", ec="black", lw=0.6, zorder=3))
    axA.text(xs[-1], yt + 0.55, "out (t8)", ha="center", fontsize=5.8)
    # dilation-1 kernel (t6,t7,t8) and dilation-2 (t4,t6,t8) feeding the output — show receptive field
    for idx, c in [(7, C_DELHI), (5, C_S1), (3, "#27AE60")]:
        axA.add_patch(FancyArrowPatch((xs[idx], yb + 0.28), (xs[-1], yt - 0.30),
                      arrowstyle="-|>", mutation_scale=7, lw=0.8, color=c, alpha=0.8, zorder=2))
    axA.text(5.0, 8.7, "kernel size 3, dilation 1 then 2;\nleft-padded + causal trim (no future leak)",
             ha="center", fontsize=6.2, style="italic")
    axA.text(5.0, 0.7, "parameters: (in·out·3) — independent of |V|", ha="center",
             fontsize=6.2, color=C_DELHI, fontweight="bold")

    # (b) edge-weighted GAT aggregation
    axB = axes[1]; axB.set_xlim(0, 10); axB.set_ylim(0, 10); axB.axis("off")
    axB.set_title("(b)  GAT layer — edge-weighted attention aggregation", fontsize=7.6, loc="left")
    j = (5.0, 4.6)
    nbrs = [(1.6, 7.4), (1.4, 3.0), (4.0, 8.4)]
    alphas = ["α₁", "α₂", "α₃"]
    axB.add_patch(Circle(j, 0.5, fc="#FADBD8", ec=C_DELHI, lw=1.0, zorder=4))
    axB.text(j[0], j[1], "node j", ha="center", va="center", fontsize=6.4, zorder=5)
    for (nx, ny), a in zip(nbrs, alphas):
        axB.add_patch(Circle((nx, ny), 0.42, fc="#D6EAF8", ec="black", lw=0.6, zorder=4))
        axB.text(nx, ny, "i", ha="center", va="center", fontsize=6.2, zorder=5)
        axB.add_patch(FancyArrowPatch((nx, ny), j, arrowstyle="-|>", mutation_scale=8,
                      lw=1.0, color="0.3", zorder=2,
                      connectionstyle="arc3,rad=0.08"))
        mx, my = (nx + j[0]) / 2, (ny + j[1]) / 2
        axB.text(mx, my + 0.25, a, ha="center", fontsize=6.2, color=C_DELHI, fontweight="bold")
    # equation box
    axB.add_patch(FancyBboxPatch((0.4, 0.3), 9.2, 1.95, boxstyle="round,pad=0.04,rounding_size=0.05",
                  lw=0.6, ec="0.5", fc="#FBFCFC"))
    axB.text(5.0, 1.95, "eᵢⱼ = LeakyReLU(a_src·Wxᵢ + a_dst·Wxⱼ) + log wᵢⱼ", ha="center", fontsize=6.2)
    axB.text(5.0, 1.35, "αᵢⱼ = softmaxⱼ(eᵢⱼ)        hⱼ = Σᵢ αᵢⱼ · Wxᵢ", ha="center", fontsize=6.2)
    axB.text(5.0, 0.6, "parameters: out·(in+2) — no per-node term, independent of |V|",
             ha="center", fontsize=6.0, color=C_DELHI, fontweight="bold")

    fig.suptitle("Inductive ST-GNN building blocks", fontsize=9.0, y=1.0, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    save(fig, "fig_gnn_blocks")


if __name__ == "__main__":
    fig1(); fig2(); fig3(); fig4(); fig5(); fig6()
    fig_pipeline(); fig_lstm_arch(); fig_gnn_blocks()
    print("All figures written to", HERE)
