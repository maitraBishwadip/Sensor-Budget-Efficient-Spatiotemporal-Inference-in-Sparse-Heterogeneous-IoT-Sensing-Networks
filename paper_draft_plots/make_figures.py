"""
Generate the figures referenced in reports/PAPER_DRAFT.md.

Outputs (300 dpi) are written next to this script as both PDF (vector, for
LaTeX) and PNG (raster, for the Markdown draft preview):

    fig1_motivation.{pdf,png}      (draft Fig 1)
    fig_pipeline.{pdf,png}         (draft Fig 2 — end-to-end methodology)
    fig_lstm_arch.{pdf,png}        (draft Fig 3 — LSTM-TL architecture)
    fig3_encoder_blocks.{pdf,png}  (draft Fig 4 — ST-GNN encoder layer stack)
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
import matplotlib.patheffects as pe
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Polygon, Ellipse, Rectangle

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
    "savefig.pad_inches": 0.04,
})

# ----------------------------------------------------------------------------
# Palette  (fill, edge) pairs — soft academic tones
# ----------------------------------------------------------------------------
INK = "#23313A"
ARROW = "#1F5C8B"
GROUP_EC = "#9FB0B8"
DATA = ("#EEF2F4", "#8FA0A8")
AMBER = ("#FBEDCB", "#D7A93F")
BLUE = ("#D9E8F6", "#4E8FC6")
BLUES = ("#C7DBF0", "#356BA6")
TEAL = ("#D6EEE8", "#46A091")
RED = ("#F8DDD8", "#C0564B")
GREEN = ("#DCEFD9", "#5AA85F")
PURP = ("#E7E0F1", "#8A6BB1")
C_DELHI, C_KOL, C_GUW = "#C0392B", "#2E86C1", "#27AE60"
C_LSTM, C_S1, C_S2 = "#7F8C8D", "#2E86C1", "#C0392B"
CITY_COLORS = {"Delhi": C_DELHI, "Kolkata": C_KOL, "Guwahati": C_GUW}


def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(HERE, f"{name}.{ext}"))
    plt.close(fig)
    print("wrote", name)


# ----------------------------------------------------------------------------
# Shared academic drawing toolkit
# ----------------------------------------------------------------------------
def _shadow(alpha=0.30):
    return [pe.withSimplePatchShadow(offset=(1.6, -1.6), shadow_rgbFace="#9AA7AD", alpha=alpha)]


def node(ax, x, y, w, h, text, style, fs=8.0, lw=1.2, rounding=0.05,
         shadow=True, weight="normal", tcolor=None):
    fc, ec = style
    p = FancyBboxPatch((x, y), w, h,
                       boxstyle=f"round,pad=0.015,rounding_size={rounding}",
                       fc=fc, ec=ec, lw=lw, zorder=3)
    if shadow:
        p.set_path_effects(_shadow())
    ax.add_patch(p)
    if text:
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                fontsize=fs, color=tcolor or INK, zorder=4, fontweight=weight)
    return (x, y, w, h)


def port(box, side):
    x, y, w, h = box
    return {"l": (x, y + h / 2), "r": (x + w, y + h / 2),
            "t": (x + w / 2, y + h), "b": (x + w / 2, y),
            "c": (x + w / 2, y + h / 2)}[side]


def arrow(ax, p0, p1, color=ARROW, lw=2.0, ms=15, rad=0.0, style="-|>", alpha=1.0):
    a = FancyArrowPatch(p0, p1, arrowstyle=style, mutation_scale=ms, lw=lw,
                        color=color, alpha=alpha, zorder=2,
                        connectionstyle=f"arc3,rad={rad}",
                        shrinkA=1.5, shrinkB=1.5, capstyle="round")
    ax.add_patch(a)


def group(ax, x, y, w, h, title, ec=GROUP_EC, fc="#FBFCFD"):
    p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.06",
                       fc=fc, ec=ec, lw=1.3, ls=(0, (6, 3)), zorder=0)
    ax.add_patch(p)
    if title:
        ax.text(x + 0.20, y + h - 0.12, title, ha="left", va="top",
                fontsize=8.4, color="#5B6B73", fontweight="bold", style="italic")


def slab(ax, x, y, w, h, style, label, shape=None, depth=0.45, fs=7.4):
    """Pseudo-3D tensor slab for layer-stack diagrams."""
    fc, ec = style
    # top + right faces (darker)
    def shade(c, f):
        c = np.array(matplotlib.colors.to_rgb(c)); return tuple(c * f)
    top = Polygon([[x, y + h], [x + w, y + h], [x + w + depth, y + h + depth], [x + depth, y + h + depth]],
                  closed=True, fc=shade(fc, 0.88), ec=ec, lw=0.9, zorder=3)
    side = Polygon([[x + w, y], [x + w + depth, y + depth], [x + w + depth, y + h + depth], [x + w, y + h]],
                   closed=True, fc=shade(fc, 0.74), ec=ec, lw=0.9, zorder=3)
    ax.add_patch(top); ax.add_patch(side)
    front = Rectangle((x, y), w, h, fc=fc, ec=ec, lw=1.0, zorder=4)
    front.set_path_effects(_shadow(0.22))
    ax.add_patch(front)
    ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=fs,
            color=INK, zorder=5)
    if shape:
        ax.text(x + w / 2 + depth / 2, y - 0.28, shape, ha="center", va="top",
                fontsize=6.3, color="#5B6B73", zorder=5)
    return (x, y, w, h)


# --- small icons -----------------------------------------------------------
def icon_stations(ax, cx, cy, s=1.0, seed=7):
    ax.add_patch(Ellipse((cx, cy), 1.5 * s, 1.05 * s, fc="#F3CFA6", ec="#C98A4E",
                 lw=0.9, zorder=4, path_effects=_shadow(0.18)))
    rng = np.random.default_rng(seed)
    for _ in range(15):
        a = rng.uniform(0, 2 * math.pi); r = rng.uniform(0, 1)
        dx, dy = 0.62 * s * math.sqrt(r) * math.cos(a), 0.42 * s * math.sqrt(r) * math.sin(a)
        ax.add_patch(Circle((cx + dx, cy + dy), 0.045 * s, fc="#7B3F12", ec="white", lw=0.2, zorder=5))


def icon_sheets(ax, x, y, w=0.7, h=0.92, s=1.0, fc="#EAF1F8", ec="#5B89B5"):
    for k in range(2, -1, -1):
        ax.add_patch(FancyBboxPatch((x + k * 0.08 * s, y - k * 0.08 * s), w * s, h * s,
                     boxstyle="round,pad=0.01,rounding_size=0.03", fc=fc, ec=ec, lw=0.8,
                     zorder=4 + (2 - k)))


def icon_layers(ax, cx, cy, s=1.0):
    for k in range(4):
        yy = cy + k * 0.24 * s
        poly = [[cx - 0.62 * s, yy], [cx + 0.40 * s, yy],
                [cx + 0.62 * s, yy + 0.2 * s], [cx - 0.40 * s, yy + 0.2 * s]]
        ax.add_patch(Polygon(poly, closed=True, fc="#F2C8A2", ec="#B5743A", lw=0.7,
                     alpha=0.96, zorder=4 + k, path_effects=_shadow(0.12) if k == 0 else None))


def icon_heat(fig, ax_box):
    """Small spatial-estimate heatmap as an inset (ax_box in figure fraction)."""
    iax = fig.add_axes(ax_box)
    rng = np.random.default_rng(4)
    z = rng.standard_normal((9, 9))
    for _ in range(4):  # smooth
        z = (z + np.roll(z, 1, 0) + np.roll(z, -1, 0) + np.roll(z, 1, 1) + np.roll(z, -1, 1)) / 5
    iax.imshow(z, cmap="jet", interpolation="bicubic", aspect="auto")
    iax.set_xticks([]); iax.set_yticks([])
    for sp in iax.spines.values():
        sp.set_edgecolor("#5B6B73"); sp.set_linewidth(0.8)


# ----------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------
PAIRS = ["Delhi→Kolkata", "Delhi→Guwahati", "Kolkata→Delhi",
         "Kolkata→Guwahati", "Guwahati→Delhi", "Guwahati→Kolkata"]
DCOLS = ["15%", "30%", "45%", "60%"]

S1 = np.array([
    [0.8113, 0.8158, 0.8182, 0.8166],
    [0.8233, 0.8165, 0.8273, 0.8220],
    [0.7953, 0.8085, 0.8198, 0.8225],
    [0.7955, 0.8044, 0.8038, 0.8157],
    [0.7887, 0.8030, 0.8123, 0.8173],
    [0.7951, 0.8085, 0.8103, 0.8177],
])
SCRATCH = np.array([
    [0.7576, 0.7803, 0.7925, 0.8031],
    [0.7756, 0.7826, 0.7887, 0.8083],
    [0.7635, 0.7905, 0.7989, 0.8105],
    [0.7807, 0.7924, 0.7970, 0.7981],
    [0.7635, 0.7905, 0.7989, 0.8105],
    [0.7576, 0.7803, 0.7925, 0.8031],
])
ZS = np.array([0.7282, 0.6841, 0.6032, 0.6698, 0.6317, 0.6980])
ZERO = np.repeat(ZS[:, None], 4, axis=1)
LSTM30 = np.array([0.8521, 0.8224, 0.8510, 0.8129, 0.8491, 0.8420])
S1_30 = np.array([0.8158, 0.8165, 0.8085, 0.8044, 0.8030, 0.8085])
S2_30 = np.array([0.8171, 0.8131, 0.8087, 0.7916, 0.8029, 0.8039])
STATION_COUNTS = {"Delhi": 40, "Kolkata": 10, "Guwahati": 4}


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


# ============================================================================
# Figure 1 — Motivation
# ============================================================================
def fig1():
    fig = plt.figure(figsize=(COL2, 2.6))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.05, 1.0, 1.05], wspace=0.30)

    axA = fig.add_subplot(gs[0])
    cities = ["Delhi", "Kolkata", "Guwahati"]
    vals = [STATION_COUNTS[c] for c in cities]
    bars = axA.bar(cities, vals, color=[CITY_COLORS[c] for c in cities],
                   width=0.62, edgecolor="black", linewidth=0.5)
    for b, v in zip(bars, vals):
        axA.text(b.get_x() + b.get_width() / 2, v + 1, str(v), ha="center", va="bottom",
                 fontsize=8, fontweight="bold")
    axA.set_ylabel("CPCB stations  |V|"); axA.set_ylim(0, 46)
    axA.set_title("(A)  Monitoring-density gap", fontsize=8.5, loc="left")
    axA.annotate("", xy=(0, 41), xytext=(2, 41), arrowprops=dict(arrowstyle="<->", lw=0.8))
    axA.text(1, 43, "10×", ha="center", fontsize=8, fontweight="bold")
    axA.spines[["top", "right"]].set_visible(False)
    axA.tick_params(axis="x", labelsize=7, rotation=12)

    axB = fig.add_subplot(gs[1]); axB.set_title("(B)  Station-independent\nLSTM-TL", fontsize=8.5, loc="left")
    rng = np.random.default_rng(3)
    px, py = rng.uniform(0.12, 0.88, 7), rng.uniform(0.18, 0.82, 7)
    for x, y in zip(px, py):
        axB.add_patch(Circle((x, y), 0.045, color=C_KOL, ec="black", lw=0.4, zorder=3))
        tx = np.linspace(0, 0.12, 12)
        axB.plot(x - 0.06 + tx, y + 0.08 + 0.02 * np.sin(tx * 90), lw=0.5, color="0.45")
    axB.text(0.5, 0.02, "each station modelled alone — no spatial links", ha="center",
             va="bottom", fontsize=6.6, style="italic")
    axB.set_xlim(0, 1); axB.set_ylim(0, 1); axB.axis("off")

    axC = fig.add_subplot(gs[2]); axC.set_title("(C)  Inductive\ngraph-TL", fontsize=8.5, loc="left")
    s_xy = np.array([[0.18, 0.78], [0.30, 0.86], [0.10, 0.62], [0.27, 0.64],
                     [0.40, 0.74], [0.20, 0.50], [0.36, 0.52]])
    for i in range(len(s_xy)):
        for j in range(i + 1, len(s_xy)):
            if np.linalg.norm(s_xy[i] - s_xy[j]) < 0.20:
                axC.plot(*zip(s_xy[i], s_xy[j]), lw=0.5, color=C_DELHI, alpha=0.6, zorder=1)
    axC.scatter(s_xy[:, 0], s_xy[:, 1], s=24, color=C_DELHI, ec="black", lw=0.4, zorder=3)
    axC.text(0.25, 0.93, "Delhi (40)", ha="center", fontsize=6.6, color=C_DELHI)
    t_xy = np.array([[0.74, 0.34], [0.86, 0.40], [0.78, 0.20], [0.90, 0.24]])
    for i in range(len(t_xy)):
        for j in range(i + 1, len(t_xy)):
            axC.plot(*zip(t_xy[i], t_xy[j]), lw=0.5, color=C_GUW, alpha=0.6, zorder=1)
    axC.scatter(t_xy[:, 0], t_xy[:, 1], s=24, color=C_GUW, ec="black", lw=0.4, zorder=3)
    axC.text(0.82, 0.47, "Guwahati (4)", ha="center", fontsize=6.6, color=C_GUW)
    axC.add_patch(FancyArrowPatch((0.42, 0.58), (0.72, 0.34), arrowstyle="-|>",
                  mutation_scale=11, lw=1.3, color="black", zorder=4))
    axC.text(0.55, 0.50, "same θ\ntransferred", ha="center", fontsize=6.4, fontweight="bold", rotation=-23)
    axC.set_xlim(0, 1); axC.set_ylim(0, 1); axC.axis("off")
    fig.suptitle("Sparse-network cities need transfer learning; a graph model adds the spatial coupling the LSTM ignores",
                 fontsize=8.1, y=1.04)
    save(fig, "fig1_motivation")


# ============================================================================
# Figure 2 — end-to-end methodology (grouped, iconified flowchart)
# ============================================================================
def fig_pipeline():
    fig, ax = plt.subplots(figsize=(COL2, 4.4))
    ax.set_xlim(0, 34); ax.set_ylim(0, 21); ax.axis("off")

    # ---- LEFT group: data preparation ----
    group(ax, 0.4, 1.0, 13.0, 19.2, "Data preparation")
    icon_stations(ax, 2.4, 17.6, s=1.25)
    ax.text(2.4, 16.0, "CPCB CAAQM\nnetwork", ha="center", fontsize=6.6, color=INK)
    d1 = node(ax, 4.6, 16.4, 8.4, 2.5,
              "Ground PM2.5  +  meteorology\nDelhi 40 · Kolkata 10 · Guwahati 4\n(3-hourly)", DATA, fs=6.8)
    arrow(ax, port(d1, "b"), (8.8, 14.5))
    d2 = node(ax, 2.0, 10.9, 10.8, 3.4,
              "Pre-processing & imputation\n• feature engineering  (F = 14: PM2.5, AT, RH, WS,\n  WD sin/cos, cyclic time, season)\n• causal gap-filling  • interleaved 70/15/15 split\n• per-(month,hour) climatology-residual target",
              AMBER, fs=6.4)
    arrow(ax, port(d2, "b"), (7.4, 9.2))
    d3 = node(ax, 2.0, 7.2, 5.1, 1.8, "Cleaned per-city\nPM2.5 tensors", GREEN, fs=6.6)
    d4 = node(ax, 7.6, 7.2, 5.1, 1.8, "Cleaned\nmeteorology", GREEN, fs=6.6)
    arrow(ax, port(d3, "b"), (8.0, 4.7), rad=0.05)
    arrow(ax, port(d4, "b"), (8.4, 4.7), rad=-0.05)

    # ---- central hub ----
    hub = node(ax, 5.6, 3.0, 5.6, 1.7, "Input data", BLUES, fs=8.4, weight="bold",
               rounding=0.45, tcolor="white")
    arrow(ax, port(hub, "r"), (15.2, 3.85), lw=2.4)

    # ---- RIGHT group: modelling & cross-city transfer ----
    group(ax, 15.4, 1.0, 18.2, 19.2, "Modelling & cross-city transfer")

    g = node(ax, 16.0, 2.9, 7.2, 1.9, "k-NN graph construction\n(k = 3, Gaussian edge weights)", BLUE, fs=6.6)
    arrow(ax, (15.2, 3.85), port(g, "l"))
    # models
    m1 = node(ax, 16.0, 11.6, 7.6, 2.6,
              "LSTM-TL\n(station-independent)\npre-train → fine-tune", TEAL, fs=6.8, weight="bold")
    m2 = node(ax, 16.0, 7.2, 7.6, 3.4,
              "Inductive ST-GNN\nStage 0 pre-train →\nStage 1 PT-FT  /  Stage 2 Graph-DANN", BLUES, fs=6.8, weight="bold")
    arrow(ax, port(g, "t"), port(m2, "b"))
    arrow(ax, port(m2, "t"), port(m1, "b"), rad=0.0)
    # verification + evaluation
    v = node(ax, 24.4, 9.4, 8.8, 2.6,
             "Three-way verification\nzero-shot · scratch · transfer\n(real iff transfer > both)", GREEN, fs=6.6)
    arrow(ax, port(m1, "r"), port(v, "l"), rad=0.10)
    arrow(ax, port(m2, "r"), port(v, "l"), rad=-0.10)
    ev = node(ax, 24.4, 6.3, 8.8, 2.2,
              "Evaluation on target test\nR² · MAE · RMSE · MAPE  (raw µg/m³)", PURP, fs=6.6)
    arrow(ax, port(v, "b"), port(ev, "t"))

    # forecast-horizon icon (above the models) + spatial-estimate heatmap (output)
    icon_layers(ax, 20.0, 15.3, s=1.4)
    ax.text(20.0, 18.3, "multi-step\nforecast", ha="center", fontsize=6.6, color=INK)
    arrow(ax, port(m1, "t"), (19.8, 15.4), rad=0.0)

    # spatial-estimate heatmap drawn directly in data coordinates (aligns cleanly)
    rng = np.random.default_rng(4); z = rng.standard_normal((9, 9))
    for _ in range(4):
        z = (z + np.roll(z, 1, 0) + np.roll(z, -1, 0) + np.roll(z, 1, 1) + np.roll(z, -1, 1)) / 5
    hx0, hx1, hy0, hy1 = 27.6, 32.6, 1.7, 4.9
    ax.imshow(z, cmap="jet", interpolation="bicubic", extent=[hx0, hx1, hy0, hy1],
              aspect="auto", zorder=5)
    ax.add_patch(Rectangle((hx0, hy0), hx1 - hx0, hy1 - hy0, fill=False, ec="#5B6B73", lw=0.9, zorder=6))
    ax.text((hx0 + hx1) / 2, hy0 - 0.4, "spatial estimate", ha="center", fontsize=6.4, color=INK)
    arrow(ax, port(ev, "b"), ((hx0 + hx1) / 2, hy1), rad=0.10)

    fig.suptitle("End-to-end experimental methodology", fontsize=10.5, y=0.99, fontweight="bold")
    save(fig, "fig_pipeline")


# ============================================================================
# Figure 3 — LSTM-TL architecture (layered NN diagram)
# ============================================================================
def fig_lstm_arch():
    fig, ax = plt.subplots(figsize=(COL2, 3.4))
    ax.set_xlim(0, 34); ax.set_ylim(0, 18); ax.axis("off")

    # input sequence: H columns, each a stack of feature cells
    ax.text(3.2, 16.6, "Input window  (one station)", ha="center", fontsize=7.2, fontweight="bold")
    H, Fc = 8, 7
    x0, y0, cw, ch = 0.8, 6.0, 0.52, 0.62
    for c in range(H):
        for r in range(Fc):
            ax.add_patch(Rectangle((x0 + c * (cw + 0.05), y0 + r * ch), cw, ch * 0.92,
                         fc="#D9E8F6", ec="#7FA8CC", lw=0.4, zorder=3))
    ax.annotate("", xy=(x0 + H * (cw + 0.05), 5.4), xytext=(x0, 5.4),
                arrowprops=dict(arrowstyle="-", lw=0.8, color="0.4"))
    ax.text(x0 + H * (cw + 0.05) / 2, 4.8, "t = 1 … 8   (H = 8)", ha="center", fontsize=6.4)
    ax.text(x0 - 0.45, y0 + Fc * ch / 2, "F = 14", rotation=90, va="center", fontsize=6.4)

    arrow(ax, (5.7, 8.4), (7.3, 8.4), lw=2.0)

    # unrolled 2-layer LSTM
    sx = [7.6, 10.0, 12.4, 15.6]
    labels = ["h₁", "h₂", "h₃", "h₈"]
    rows = {1: 6.4, 2: 9.8}
    for layer, yy in rows.items():
        st = TEAL if layer == 1 else (("#BfE3DA", "#3C8C7E"))
        for k, x in enumerate(sx):
            node(ax, x, yy, 1.7, 1.7, f"LSTM\ncell", st, fs=6.2, rounding=0.12)
            if k < len(sx) - 1:
                arrow(ax, (x + 1.7, yy + 0.85), (sx[k + 1], yy + 0.85), lw=1.3, ms=10)
    for x in sx:
        arrow(ax, (x + 0.85, rows[1] + 1.7), (x + 0.85, rows[2]), lw=1.3, ms=10)
    ax.text(13.6, 4.6, "···", fontsize=13, ha="center")
    ax.text(11.6, 12.2, "2-layer LSTM  (hidden = 64, dropout 0.2) — shared across all stations",
            ha="center", fontsize=6.8, style="italic")

    # dense head as neuron columns
    arrow(ax, (17.3, 10.65), (18.7, 10.65), lw=2.0)
    def neurons(cx, n, yy0, col):
        ys = np.linspace(yy0, yy0 + 4.0, n)
        for yv in ys:
            ax.add_patch(Circle((cx, yv), 0.32, fc=col, ec="#2E5E8C", lw=0.7, zorder=4,
                         path_effects=_shadow(0.15)))
        return cx, ys
    cxh, ysh = neurons(19.6, 5, 8.6, "#FCEFC7")
    cx1, ys1 = neurons(22.6, 5, 8.6, "#FCEFC7")
    cxo, yso = neurons(25.6, 1, 10.5, "#F8DDD8")
    for a in ysh:
        for b in ys1:
            ax.plot([cxh, cx1], [a, b], lw=0.3, color="0.7", zorder=2)
    for b in ys1:
        ax.plot([cx1, cxo], [b, yso[0]], lw=0.3, color="0.7", zorder=2)
    ax.text(19.6, 13.1, "h₈ (64)", ha="center", fontsize=6.6)
    ax.text(22.6, 13.1, "FC 64→64\n+ ReLU", ha="center", fontsize=6.4)
    ax.text(25.6, 13.1, "FC →1", ha="center", fontsize=6.4)
    arrow(ax, (26.0, 10.65), (27.4, 10.65), lw=2.0)
    ax.text(28.9, 10.65, "ŷ\n(+3 h\nPM2.5)", ha="center", va="center", fontsize=7.0, fontweight="bold")

    # gate-equation card
    node(ax, 18.6, 1.2, 14.6, 4.6, "", DATA, shadow=True, rounding=0.05)
    ax.text(25.9, 5.2, "LSTM cell (per step)", ha="center", fontsize=7.0, fontweight="bold")
    ax.text(19.2, 4.4,
            "i, f, o = σ(W·[hₜ₋₁, xₜ] + b)       g = tanh(W_g·[hₜ₋₁, xₜ] + b_g)\n"
            "cₜ = f · cₜ₋₁ + i · g                      hₜ = o · tanh(cₜ)",
            ha="left", va="top", fontsize=6.6)
    ax.text(25.9, 1.7, "~50 k parameters, independent of N (applied per station)",
            ha="center", fontsize=6.3, color="#5B6B73", style="italic")

    # transfer ribbon
    node(ax, 0.8, 1.2, 16.0, 1.7,
         "Transfer:  pre-train on source city  →  fine-tune on target d %  →  evaluate on target test",
         GREEN, fs=6.6)

    fig.suptitle("Station-independent LSTM-TL architecture", fontsize=10.0, y=1.0, fontweight="bold")
    save(fig, "fig_lstm_arch")


# ============================================================================
# Figure 4 — ST-GNN encoder as a pseudo-3D layer stack
# ============================================================================
def fig3():
    fig, ax = plt.subplots(figsize=(COL2, 3.0))
    ax.set_xlim(0, 34); ax.set_ylim(0, 15); ax.axis("off")

    y = 5.0
    layers = [
        ("Input\n[B,H,N,F]", DATA, "F = 14", 3.0, 4.0),
        ("Temporal\nConv₁  (d=1)", AMBER, "[B,H,N,64]", 3.2, 4.4),
        ("GAT₁\n+ ELU", BLUE, "[B·H,N,64]", 3.0, 4.8),
        ("GAT₂\n+ ELU", BLUE, "[B·H,N,64]", 3.0, 4.8),
        ("Temporal\nConv₂  (d=2)", AMBER, "[B,H,N,64]", 3.2, 4.4),
        ("Layer\nNorm", TEAL, "[B,N,64]", 2.2, 4.4),
        ("Linear\nhead", PURP, "[B,N]", 2.2, 4.0),
    ]
    x = 0.8
    prev_r = None
    for i, (lab, st, shp, w, h) in enumerate(layers):
        yy = y + (15 - 5 - h) / 2 - 1.4
        b = slab(ax, x, yy, w, h, st, lab, shape=shp, depth=0.5, fs=6.8)
        r = (x + w + 0.5, yy + h / 2)
        if prev_r is not None:
            arrow(ax, prev_r, (x, yy + h / 2), lw=1.9, ms=12)
        prev_r = r
        x += w + 1.7
    arrow(ax, prev_r, (x, y + (15 - 5 - 4.0) / 2 - 1.4 + 2.0), lw=1.9, ms=12)
    ax.text(x + 0.2, y + (15 - 5 - 4.0) / 2 - 1.4 + 2.0, "PM2.5\n(+3 h)", ha="left", va="center",
            fontsize=6.8, fontweight="bold")

    # banner
    node(ax, 0.8, 11.7, 32.4, 2.0, "", RED, rounding=0.06, shadow=True)
    ax.text(17.0, 12.7,
            "No block carries a parameter whose shape depends on |V|  →  the same ~25 k-parameter θ runs on N = 40, 10, or 4",
            ha="center", va="center", fontsize=7.4, fontweight="bold", color=RED[1])
    ax.text(17.0, 1.1,
            "GAT layers are vectorized over the B·H per-timestep graphs;  TemporalConv mixes only along time, per node.",
            ha="center", fontsize=6.6, style="italic", color="#5B6B73")

    fig.suptitle("Inductive ST-GNN encoder — layer stack", fontsize=10.0, y=1.0, fontweight="bold")
    save(fig, "fig3_encoder_blocks")


# ============================================================================
# Figure 5 — GNN building blocks: dilated causal TCN + edge-weighted GAT
# ============================================================================
def fig_gnn_blocks():
    fig, axes = plt.subplots(1, 2, figsize=(COL2, 3.2), gridspec_kw={"width_ratios": [1.0, 1.05]})

    # (a) dilated causal TemporalConv
    axA = axes[0]; axA.set_xlim(0, 10); axA.set_ylim(0, 10); axA.axis("off")
    axA.set_title("(a)  TemporalConv — dilated causal conv (per node)", fontsize=7.6, loc="left")
    xs = np.linspace(1.0, 9.0, 8)
    yb, ym, yt = 1.8, 4.7, 7.8
    # input row
    for i, x in enumerate(xs):
        axA.add_patch(Circle((x, yb), 0.30, fc="#D9E8F6", ec="#4E8FC6", lw=0.7, zorder=3,
                      path_effects=_shadow(0.12)))
        axA.text(x, yb - 0.62, f"t{i+1}", ha="center", fontsize=5.6, color="#5B6B73")
    # hidden (dilation 1) row — a few
    hid = xs[2:]
    for x in hid:
        axA.add_patch(Circle((x, ym), 0.27, fc="#FBEDCB", ec="#D7A93F", lw=0.7, zorder=3))
    # output (dilation 2)
    axA.add_patch(Circle((xs[-1], yt), 0.33, fc="#F8DDD8", ec="#C0564B", lw=0.9, zorder=4,
                  path_effects=_shadow(0.15)))
    axA.text(xs[-1], yt + 0.6, "out (t8)", ha="center", fontsize=5.8)
    # dilation-1 edges: each hidden from 3 inputs (t-2,t-1,t)
    for hx, ix in zip(hid, range(2, 8)):
        for s in (ix - 2, ix - 1, ix):
            axA.plot([xs[s], hx], [yb + 0.30, ym - 0.27], lw=0.5, color="#C9A24A", alpha=0.6, zorder=2)
    # dilation-2 edges into output: t4,t6,t8
    for s, c in [(3, "#27AE60"), (5, "#2E86C1"), (7, "#C0392B")]:
        axA.add_patch(FancyArrowPatch((hid[s - 2] if s - 2 < len(hid) else xs[s], ym + 0.27),
                      (xs[-1], yt - 0.33), arrowstyle="-|>", mutation_scale=8, lw=1.0,
                      color=c, alpha=0.85, zorder=3, connectionstyle="arc3,rad=0.05"))
    axA.text(5.0, 9.3, "kernel 3, dilation 1 then 2; left-padded + causal trim (no future leak)",
             ha="center", fontsize=6.0, style="italic")
    axA.text(5.0, 0.4, "parameters: (in · out · 3) — independent of |V|", ha="center",
             fontsize=6.2, color=RED[1], fontweight="bold")

    # (b) edge-weighted GAT aggregation
    axB = axes[1]; axB.set_xlim(0, 10); axB.set_ylim(0, 10); axB.axis("off")
    axB.set_title("(b)  GAT layer — edge-weighted attention", fontsize=7.6, loc="left")
    j = (6.4, 5.0)
    nbrs = [(2.0, 7.8), (1.7, 4.0), (3.8, 8.6)]
    alphas = ["α₁", "α₂", "α₃"]
    for (nx, ny), a in zip(nbrs, alphas):
        axB.add_patch(FancyArrowPatch((nx, ny), j, arrowstyle="-|>", mutation_scale=10,
                      lw=1.4, color="#7FA8CC", zorder=2, connectionstyle="arc3,rad=0.10"))
        mx, my = (nx + j[0]) / 2, (ny + j[1]) / 2
        axB.text(mx - 0.1, my + 0.35, a, ha="center", fontsize=6.6, color=RED[1], fontweight="bold")
    for (nx, ny) in nbrs:
        axB.add_patch(Circle((nx, ny), 0.46, fc="#D9E8F6", ec="#4E8FC6", lw=0.8, zorder=4,
                      path_effects=_shadow(0.15)))
        axB.text(nx, ny, "i", ha="center", va="center", fontsize=6.6, zorder=5)
    axB.add_patch(Circle(j, 0.56, fc="#F8DDD8", ec="#C0564B", lw=1.1, zorder=5,
                  path_effects=_shadow(0.2)))
    axB.text(j[0], j[1], "j", ha="center", va="center", fontsize=7.4, fontweight="bold", zorder=6)
    node(axB, 0.4, 0.3, 9.2, 2.45, "", DATA, rounding=0.05, shadow=False)
    axB.text(5.0, 2.35, "eᵢⱼ = LeakyReLU(a_src·Wxᵢ + a_dst·Wxⱼ) + log wᵢⱼ", ha="center", fontsize=6.3)
    axB.text(5.0, 1.65, "αᵢⱼ = softmaxⱼ(eᵢⱼ)         hⱼ = Σᵢ αᵢⱼ · Wxᵢ", ha="center", fontsize=6.3)
    axB.text(5.0, 0.7, "parameters: out · (in+2) — no per-node term, independent of |V|",
             ha="center", fontsize=6.0, color=RED[1], fontweight="bold")

    fig.suptitle("Inductive ST-GNN building blocks", fontsize=10.0, y=1.0, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    save(fig, "fig_gnn_blocks")


# ============================================================================
# Figure 6 — three city graphs (real coords)
# ============================================================================
def fig2():
    meta = json.load(open(META))
    fig, axes = plt.subplots(1, 3, figsize=(COL2, 2.7))
    for ax, city in zip(axes, ["Delhi", "Kolkata", "Guwahati"]):
        names, pts, edges = knn_edges(meta[city]["coords"], k=3)
        lon, lat = pts[:, 1], pts[:, 0]
        col = CITY_COLORS[city]
        for i, j in edges:
            ax.plot([lon[i], lon[j]], [lat[i], lat[j]], lw=0.5, color=col, alpha=0.45, zorder=1)
        ax.scatter(lon, lat, s=22 if city != "Delhi" else 14, color=col, ec="black", lw=0.35, zorder=3)
        ax.set_title(f"{city}\n|V|={len(names)}, |E|={len(edges)}, deg=3.0", fontsize=8, color=col)
        ax.set_xlabel("lon"); ax.set_aspect("equal", adjustable="datalim")
        ax.tick_params(labelsize=6); ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("lat")
    fig.suptitle("k-NN station graphs (k = 3, Gaussian-decay edge weights, σ = 5 km) — one inductive encoder runs on all three",
                 fontsize=8.0, y=1.02)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    save(fig, "fig2_city_graphs")


# ============================================================================
# Figure 7 — Graph-DANN
# ============================================================================
def fig4():
    fig, ax = plt.subplots(figsize=(COL2, 3.1))
    ax.set_xlim(0, 16); ax.set_ylim(0, 9); ax.axis("off")

    cnames = [("Delhi (source)", C_DELHI), ("Guwahati (target,\nlabels withheld)", C_GUW),
              ("Kolkata (3rd-city\nreplay)", C_KOL)]
    boxes = []
    for i, (nm, c) in enumerate(cnames):
        yy = 6.6 - i * 2.15
        b = node(ax, 0.3, yy, 2.6, 1.6, nm, ("#FBFCFD", c), fs=6.2)
        boxes.append(b)
    ax.text(1.6, 8.5, "round-robin\nminibatch", ha="center", fontsize=6.6, fontweight="bold")
    enc = node(ax, 3.9, 3.7, 2.9, 1.7, "Inductive\nST-GNN encoder\n(shared θ)", BLUES, fs=6.6, weight="bold")
    for b in boxes:
        arrow(ax, port(b, "r"), port(enc, "l"), rad=0.0, lw=1.6, ms=12)
    rd = node(ax, 7.4, 3.7, 2.6, 1.7, "mean + max\ngraph readout\nz = [mean‖max]", TEAL, fs=6.3)
    arrow(ax, port(enc, "r"), port(rd, "l"))
    fh = node(ax, 11.0, 5.7, 2.5, 1.3, "Forecast head\nMSE(ŷ, y)", AMBER, fs=6.4)
    grl = node(ax, 11.0, 2.0, 1.9, 1.3, "GRL\n(×−λ)", RED, fs=6.6, weight="bold")
    disc = node(ax, 13.4, 1.85, 2.3, 1.6, "3-way city\ndiscriminator\nCE(city)", ("#F9EAE8", C_DELHI), fs=6.2)
    arrow(ax, port(rd, "r"), port(fh, "l"), rad=0.12)
    arrow(ax, port(rd, "r"), port(grl, "l"), rad=-0.12)
    arrow(ax, port(grl, "r"), port(disc, "l"))
    ax.text(13.6, 0.85, "encoder trained to FOOL the discriminator\n→ city-invariant embedding",
            ha="center", fontsize=6.0, style="italic", color="#5B6B73")

    iax = fig.add_axes([0.46, 0.74, 0.18, 0.17])
    p = np.linspace(0, 1, 200); lam = 2 / (1 + np.exp(-10 * p)) - 1
    iax.plot(p, lam, color=C_DELHI, lw=1.4)
    iax.set_title("λ(p) warm-up", fontsize=6.2); iax.set_xlabel("progress p", fontsize=5.6)
    iax.set_ylabel("λ", fontsize=5.6); iax.tick_params(labelsize=5)
    iax.set_xlim(0, 1); iax.set_ylim(0, 1.05)
    for sp in iax.spines.values():
        sp.set_linewidth(0.6)

    fig.suptitle("Graph-DANN — adversarial alignment of the size-invariant graph embedding",
                 fontsize=9.2, y=0.99, fontweight="bold")
    save(fig, "fig4_graph_dann")


# ============================================================================
# Figure 8 — Stage-1 verification small multiples
# ============================================================================
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
                ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=5.7,
                        color="white" if M[i, j] < 0.77 else "black")
    axes[0].set_yticks(range(6)); axes[0].set_yticklabels(PAIRS, fontsize=6.2)
    axes[1].set_yticks([]); axes[2].set_yticks([])
    cbar = fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02)
    cbar.set_label("test R²", fontsize=7); cbar.ax.tick_params(labelsize=6)
    fig.suptitle("Stage 1 (PT-FT): transfer beats both zero-shot and scratch in all 24 cells",
                 fontsize=8.2, x=0.45, y=1.0)
    save(fig, "fig5_stage1_heatmaps")


# ============================================================================
# Figure 9 — headline bars
# ============================================================================
def fig6():
    fig, ax = plt.subplots(figsize=(COL2, 3.0))
    x = np.arange(len(PAIRS)); w = 0.27
    ax.bar(x - w, LSTM30, w, label="LSTM-TL (fixed)", color=C_LSTM, ec="black", lw=0.4)
    ax.bar(x, S1_30, w, label="GNN Stage 1 (PT-FT)", color=C_S1, ec="black", lw=0.4)
    ax.bar(x + w, S2_30, w, label="GNN Stage 2 (Graph-DANN)", color=C_S2, ec="black", lw=0.4)
    for xi, vals in zip(x, zip(LSTM30, S1_30, S2_30)):
        for off, v in zip((-w, 0, w), vals):
            ax.text(xi + off, v + 0.004, f"{v:.2f}", ha="center", va="bottom", fontsize=5.2, rotation=90)
    ax.set_ylim(0.74, 0.875); ax.set_ylabel("transfer test R²  (d = 30 %)")
    ax.set_xticks(x); ax.set_xticklabels(PAIRS, rotation=18, ha="right", fontsize=6.6)
    ax.legend(loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.13))
    ax.spines[["top", "right"]].set_visible(False); ax.grid(axis="y", lw=0.4, alpha=0.4)
    save(fig, "fig6_headline_bars")


if __name__ == "__main__":
    fig1(); fig2(); fig3(); fig4(); fig5(); fig6()
    fig_pipeline(); fig_lstm_arch(); fig_gnn_blocks()
    print("All figures written to", HERE)
