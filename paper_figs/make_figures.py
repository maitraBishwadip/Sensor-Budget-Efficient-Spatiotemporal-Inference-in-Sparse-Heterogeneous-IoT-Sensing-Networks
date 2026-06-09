"""
High-quality figure set for the IEEE TGRS submission (6 figures) -- v2.

Revisions addressing review feedback:
  - removed µg/m³ clutter from the study-area map; fixed overlapping city labels
  - de-congested the framework diagram (shorter text, larger boxes, nothing clipped)
  - redesigned the encoder figure to a cleaner, publishable layout
  - refined, thinner, consistent arrows throughout
  - diversified the palette; perceptual multi-hue colormaps for heatmap/scatter
  - forecast trace uses a recognizable Kolkata station (Victoria)

Run:  python paper_figs/export_predictions.py   (once, for Fig 6 data)
      python paper_figs/make_figures.py
"""
from __future__ import annotations

import json
import math
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patheffects as pe
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle

warnings.filterwarnings("ignore")

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
ASSETS = HERE / "_assets"
META = ROOT / "dataset" / "processed" / "metadata.json"

COL1, COL2 = 3.5, 7.16

plt.rcParams.update({
    "figure.dpi": 160, "savefig.dpi": 400,
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Nimbus Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 8.0, "axes.titlesize": 9.0, "axes.labelsize": 8.0,
    "axes.linewidth": 0.7, "axes.edgecolor": "#3a4650",
    "xtick.labelsize": 7.0, "ytick.labelsize": 7.0,
    "xtick.major.width": 0.7, "ytick.major.width": 0.7,
    "legend.fontsize": 7.0, "legend.frameon": False,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.04,
})

# --- refined, multi-hue palette ------------------------------------------- #
INK = "#1f2a30"
MUT = "#5d6b73"
ARROW = "#46555e"
HALO = [pe.withStroke(linewidth=2.2, foreground="white")]

C_DELHI, C_KOL, C_GUW = "#c0392b", "#1f6cb0", "#1e8449"
CITY_C = {"Delhi": C_DELHI, "Kolkata": C_KOL, "Guwahati": C_GUW}
C_LSTM, C_S1, C_S2 = "#6b7177", "#1f6cb0", "#c0392b"

# block styles (fill, edge) -- distinct hues per functional role
B_DATA = ("#e9eef1", "#7f8d96")
B_PREP = ("#fbeaccc"[:7], "#c79338")
B_PREP = ("#fbecca", "#c79338")
B_GRAPH = ("#d6ebe4", "#2f9c84")
B_ENC = ("#dee2f4", "#4a5cae")
B_S1 = ("#d3e6f5", "#1f6cb0")
B_S2 = ("#f7ddd5", "#c0392b")
B_VERIF = ("#ecdff1", "#8a5bb0")
B_EVAL = ("#dcefd6", "#4f9e57")
B_HILITE = ("#fdf1e2", "#d79b3a")

CMAP = "viridis"        # perceptual, multi-hue (not all-blue)

PAIRS = ["Delhi→Kolkata", "Delhi→Guwahati", "Kolkata→Delhi",
         "Kolkata→Guwahati", "Guwahati→Delhi", "Guwahati→Kolkata"]
DCOLS = ["15", "30", "45", "60"]

S1 = np.array([
    [0.8113, 0.8158, 0.8182, 0.8166], [0.8233, 0.8165, 0.8273, 0.8220],
    [0.7953, 0.8085, 0.8198, 0.8225], [0.7955, 0.8044, 0.8038, 0.8157],
    [0.7887, 0.8030, 0.8123, 0.8173], [0.7951, 0.8085, 0.8103, 0.8177]])
SCRATCH = np.array([
    [0.7576, 0.7803, 0.7925, 0.8031], [0.7756, 0.7826, 0.7887, 0.8083],
    [0.7635, 0.7905, 0.7989, 0.8105], [0.7807, 0.7924, 0.7970, 0.7981],
    [0.7635, 0.7905, 0.7989, 0.8105], [0.7576, 0.7803, 0.7925, 0.8031]])
ZS = np.array([0.7282, 0.6841, 0.6032, 0.6698, 0.6317, 0.6980])
ZERO = np.repeat(ZS[:, None], 4, axis=1)


def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(HERE / f"{name}.{ext}")
    plt.close(fig)
    print("wrote", name)


def haversine(a, b):
    R = 6371.0
    la1, lo1, la2, lo2 = map(math.radians, [a[0], a[1], b[0], b[1]])
    dlat, dlon = la2 - la1, lo2 - lo1
    h = math.sin(dlat / 2) ** 2 + math.cos(la1) * math.cos(la2) * math.sin(dlon / 2) ** 2
    return 2 * R * math.asin(math.sqrt(h))


def knn_edges(coords, k=3):
    names = list(coords)
    pts = np.array([coords[n] for n in names], dtype=float)
    edges = []
    for i in range(len(names)):
        d = np.array([haversine(pts[i], pts[j]) if i != j else 1e9 for j in range(len(names))])
        for j in np.argsort(d)[:k]:
            edges.append((i, int(j)))
    return names, pts, edges


def load_meta():
    return json.load(open(META))


# --- clean diagram toolkit ------------------------------------------------- #
def _sh(a=0.16):
    return [pe.withSimplePatchShadow(offset=(1.0, -1.0), shadow_rgbFace="#9aa7ad", alpha=a)]


def box(ax, x, y, w, h, text, style, fs=7.4, lw=1.0, r=0.035, weight="normal",
        tcolor=None, shadow=True, head=None, head_c=None):
    fc, ec = style
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0.010,rounding_size={r}",
                 fc=fc, ec=ec, lw=lw, zorder=3, path_effects=_sh() if shadow else None))
    ty = y + h / 2
    if head:
        ax.text(x + w / 2, y + h - 0.30, head, ha="center", va="top", fontsize=fs + 0.8,
                fontweight="bold", color=head_c or ec, zorder=4)
        ty = y + (h - 0.55) / 2 - 0.05
    if text:
        ax.text(x + w / 2, ty, text, ha="center", va="center", fontsize=fs,
                color=tcolor or INK, zorder=4, fontweight=weight, linespacing=1.32)
    return (x, y, w, h)


def port(b, s):
    x, y, w, h = b
    return {"l": (x, y + h / 2), "r": (x + w, y + h / 2),
            "t": (x + w / 2, y + h), "b": (x + w / 2, y)}[s]


def arr(ax, p0, p1, color=ARROW, lw=1.05, ms=8, rad=0.0):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=ms, lw=lw,
                 color=color, zorder=2, connectionstyle=f"arc3,rad={rad}",
                 shrinkA=3.5, shrinkB=3.5, joinstyle="round", capstyle="round"))


def container(ax, x, y, w, h, title, ec="#b3bdc4"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.04",
                 fc="#fbfcfd", ec=ec, lw=1.0, ls=(0, (4, 3)), zorder=0))
    if title:
        ax.text(x + 0.25, y + h - 0.18, title, ha="left", va="top", fontsize=7.6,
                color=MUT, fontweight="bold", style="italic", zorder=1)


def nodecluster(ax, cx, cy, n, color, rad=0.55, seed=0):
    """tiny graph glyph: n nodes on a ring with a few edges."""
    rng = np.random.default_rng(seed)
    ang = np.linspace(0, 2 * math.pi, n, endpoint=False) + rng.uniform(0, 1)
    px = cx + rad * np.cos(ang); py = cy + rad * np.sin(ang)
    for i in range(n):
        for j in range(i + 1, n):
            if rng.random() < 0.5:
                ax.plot([px[i], px[j]], [py[i], py[j]], lw=0.5, color=color, alpha=0.5, zorder=2)
    ax.scatter(px, py, s=12, color=color, ec="white", lw=0.4, zorder=3)


# ============================================================================
# FIGURE 1 — Study area & monitoring-network heterogeneity
# ============================================================================
def fig1_study_area():
    import geopandas as gpd
    meta = load_meta()
    counts = {c: len(meta[c]["coords"]) for c in ["Delhi", "Kolkata", "Guwahati"]}

    fig = plt.figure(figsize=(COL2, 3.0))
    gs = fig.add_gridspec(3, 2, width_ratios=[1.34, 1.0], wspace=0.10, hspace=0.55)
    axM = fig.add_subplot(gs[:, 0])

    try:
        india = gpd.read_file(ASSETS / "india_states.geojson")
        india.plot(ax=axM, facecolor="#eef1f3", edgecolor="#c2ccd2", linewidth=0.35, zorder=1)
        india.boundary.plot(ax=axM, color="#aab5bb", linewidth=0.3, zorder=1)
    except Exception as e:
        print("  [fig1] basemap skipped:", e)

    cents = {}
    for c in ["Delhi", "Kolkata", "Guwahati"]:
        pts = np.array(list(meta[c]["coords"].values()), dtype=float)
        cents[c] = (pts[:, 1].mean(), pts[:, 0].mean())
    # label placement chosen to avoid collisions (Delhi NW, Kolkata S, Guwahati NE)
    lblpos = {"Delhi": (-10, 7, "right", "bottom"),
              "Kolkata": (-2, -12, "center", "top"),
              "Guwahati": (10, 9, "left", "bottom")}
    for c, (lo, la) in cents.items():
        axM.scatter([lo], [la], s=105, marker="*", color=CITY_C[c], ec="white",
                    lw=0.9, zorder=6, path_effects=_sh(0.30))
        dx, dy, ha, va = lblpos[c]
        axM.annotate(f"{c}\n|V| = {counts[c]}", (lo, la), xytext=(dx, dy),
                     textcoords="offset points", fontsize=6.8, color=CITY_C[c],
                     fontweight="bold", ha=ha, va=va, zorder=7, path_effects=HALO,
                     arrowprops=dict(arrowstyle="-", lw=0.5, color=CITY_C[c], alpha=0.6))
    axM.set_xlim(67, 98); axM.set_ylim(7.5, 37)
    axM.set_title("(a)  CPCB monitoring cities", fontsize=8.2, loc="left")
    axM.set_xlabel("Longitude (°E)", fontsize=7); axM.set_ylabel("Latitude (°N)", fontsize=7)
    axM.tick_params(labelsize=6)
    axM.set_aspect(1.0 / math.cos(math.radians(23)))

    tags = ["(b)", "(c)", "(d)"]
    for row, city in enumerate(["Delhi", "Kolkata", "Guwahati"]):
        ax = fig.add_subplot(gs[row, 1])
        names, pts, edges = knn_edges(meta[city]["coords"], k=3)
        lon, lat = pts[:, 1], pts[:, 0]
        col = CITY_C[city]
        for i, j in edges:
            ax.plot([lon[i], lon[j]], [lat[i], lat[j]], lw=0.5, color=col, alpha=0.4, zorder=1)
        ax.scatter(lon, lat, s=(11 if city == "Delhi" else 30), color=col,
                   ec="white", lw=0.45, zorder=3)
        ax.set_title(f"{tags[row]} {city}:  |V|={len(names)},  |E|={len(edges)}",
                     fontsize=7.0, loc="left", color=col, pad=2)
        ax.set_aspect("equal", adjustable="datalim")
        ax.tick_params(labelsize=5.2, length=2); ax.margins(0.20)
        for sp in ax.spines.values():
            sp.set_edgecolor("#aab5bb"); sp.set_linewidth(0.5)

    fig.suptitle("Order-of-magnitude heterogeneity in monitoring density (|V| = 40 / 10 / 4)",
                 fontsize=8.2, y=1.01, fontweight="bold")
    save(fig, "fig1_study_area")


# ============================================================================
# FIGURE 2 — Proposed framework (de-congested)
# ============================================================================
def fig2_framework():
    fig, ax = plt.subplots(figsize=(COL2, 3.5))
    ax.set_xlim(0, 48); ax.set_ylim(0, 24); ax.axis("off")

    # Column 1 — data + preprocessing
    container(ax, 0.4, 0.8, 12.2, 22.4, "Data & preprocessing")
    d = box(ax, 1.1, 16.6, 10.8, 5.4,
            "CPCB CAAQM network\nPM2.5 (BAM) + AT, RH,\nWS, WD   ·   3-hourly\nDelhi 40 / Kol 10 / Guw 4",
            B_DATA, fs=6.7, head="Instruments")
    p = box(ax, 1.1, 8.0, 10.8, 7.2,
            "• IDW + Kalman gap-fill\n• 14 engineered features\n"
            "• climatology-residual\n   target\n• 70/15/15 interleaved\n   split",
            B_PREP, fs=6.6, head="Preprocessing")
    g = box(ax, 1.1, 3.4, 10.8, 3.0, "k-NN graph,  k = 3\nGaussian weights, σ = 5 km", B_GRAPH, fs=6.6)
    arr(ax, port(d, "b"), port(p, "t")); arr(ax, port(p, "b"), port(g, "t"))

    # Column 2 — encoder (shared)
    s0 = box(ax, 14.3, 18.6, 11.0, 2.6, "Stage 0\nsource-only pre-train", B_DATA, fs=6.6)
    enc = box(ax, 14.3, 9.0, 11.0, 7.4,
              "TCN → GAT×2 → TCN\n\n~25k parameters,\nno |V| dependence (θ)",
              B_ENC, fs=6.8, head="Shared inductive\nST-GNN encoder", weight="normal")
    arr(ax, port(g, "r"), port(enc, "l"), rad=-0.05)
    arr(ax, port(s0, "b"), port(enc, "t"))

    # Column 3 — two transfer branches
    container(ax, 27.0, 0.8, 20.6, 22.4, "Cross-city transfer")
    s1 = box(ax, 27.7, 13.4, 11.2, 4.4,
             "load θ → fine-tune\non target d%  (MSE)", B_S1, fs=6.7, head="Stage 1 — PT-FT")
    s2 = box(ax, 27.7, 6.4, 11.2, 5.6,
             "adversarial align of\nmean+max readout\n(GRL + 3-city disc.) → FT",
             B_S2, fs=6.6, head="Stage 2 — Graph-DANN")
    arr(ax, port(enc, "r"), port(s1, "l"), rad=0.10)
    arr(ax, port(enc, "r"), port(s2, "l"), rad=-0.10)

    v = box(ax, 40.3, 13.6, 7.0, 4.0, "zero-shot ·\nscratch · transfer",
            B_VERIF, fs=6.4, head="3-way verify")
    ev = box(ax, 40.3, 7.4, 7.0, 4.0, "R² · MAE ·\nRMSE · MAPE", B_EVAL, fs=6.4, head="Evaluate")
    arr(ax, port(s1, "r"), port(v, "l"), rad=0.05)
    arr(ax, port(s2, "r"), port(ev, "l"), rad=-0.05)
    arr(ax, port(v, "b"), port(ev, "t"))

    fig.suptitle("Proposed cross-city PM$_{2.5}$ transfer-learning framework",
                 fontsize=10.0, y=1.0, fontweight="bold")
    save(fig, "fig2_framework")


# ============================================================================
# FIGURE 3 — Inductive ST-GNN encoder (redesigned)
# ============================================================================
def fig3_encoder():
    fig, ax = plt.subplots(figsize=(COL2, 2.7))
    ax.set_xlim(0, 48); ax.set_ylim(0, 16); ax.axis("off")

    layers = [
        ("Input", "[B,H,N,F]", B_DATA),
        ("TConv₁\nd=1", "[B,H,N,64]", B_PREP),
        ("GAT₁\n+ELU", "[B·H,N,64]", B_ENC),
        ("GAT₂\n+ELU", "[B·H,N,64]", B_ENC),
        ("TConv₂\nd=2", "[B,H,N,64]", B_PREP),
        ("Layer\nNorm", "[B,N,64]", B_GRAPH),
        ("Linear\nhead", "[B,N]", B_VERIF),
    ]
    x, w, gap, y, h = 0.8, 5.3, 1.2, 7.3, 3.6
    prev = None
    for lab, shp, st in layers:
        b = box(ax, x, y, w, h, lab, st, fs=7.0, shadow=True)
        ax.text(x + w / 2, y - 0.5, shp, ha="center", va="top", fontsize=5.8, color=MUT)
        if prev:
            arr(ax, prev, (x, y + h / 2), lw=1.1, ms=8)
        prev = (x + w, y + h / 2)
        x += w + gap
    arr(ax, prev, (x + 0.2, y + h / 2), lw=1.1, ms=8)
    ax.text(x + 0.5, y + h / 2, "PM$_{2.5}$\n(+3 h)", ha="left", va="center",
            fontsize=6.8, fontweight="bold", color=C_DELHI)

    # subtle inductivity callout with three small graph glyphs (replaces big red block)
    cy = 13.5
    ax.text(2.0, cy, "Size-invariant:", fontsize=7.6, fontweight="bold", color=INK, va="center")
    ax.text(9.4, cy, "the same  θ  runs verbatim on", fontsize=7.2, color=INK, va="center")
    gx = [27.5, 33.5, 39.5]
    for cx, (n, col, lab) in zip(gx, [(7, C_DELHI, "N=40"), (5, C_KOL, "N=10"), (4, C_GUW, "N=4")]):
        nodecluster(ax, cx, cy, n, col, rad=0.85, seed=n)
        ax.text(cx, cy - 1.7, lab, ha="center", fontsize=6.2, color=col, fontweight="bold")
    ax.annotate("", xy=(26.0, cy), xytext=(24.0, cy),
                arrowprops=dict(arrowstyle="-|>", lw=1.0, color=ARROW))
    ax.plot([0.8, 44.5], [11.6, 11.6], lw=0.6, color="#c2ccd2", zorder=0)

    # equations strip
    ax.text(24.0, 1.6,
            r"$e_{ij}=\mathrm{LeakyReLU}(a_s^\top Wx_i+a_d^\top Wx_j)+\log w_{ij}$,   "
            r"$\alpha_{ij}=\mathrm{softmax}_j(e_{ij})$,   $h_j=\sum_i\alpha_{ij}Wx_i$,   "
            r"$z=[\mathrm{mean}_n h\,\Vert\,\mathrm{max}_n h]$",
            ha="center", va="center", fontsize=6.2, color=INK)

    fig.suptitle("Inductive ST-GNN encoder — no parameter shape depends on $|V|$",
                 fontsize=9.4, y=1.0, fontweight="bold")
    save(fig, "fig3_encoder")


# ============================================================================
# FIGURE 4 — Graph-DANN (refined arrows)
# ============================================================================
def fig4_graph_dann():
    fig, ax = plt.subplots(figsize=(COL2, 3.0))
    ax.set_xlim(0, 17); ax.set_ylim(0, 9.4); ax.axis("off")

    cities = [("Delhi  (source)", C_DELHI), ("Guwahati\n(target, no labels)", C_GUW),
              ("Kolkata\n(3rd-city replay)", C_KOL)]
    bxs = []
    for i, (nm, c) in enumerate(cities):
        b = box(ax, 0.3, 6.7 - i * 2.3, 3.0, 1.8, nm, ("#fbfcfd", c), fs=6.2)
        bxs.append(b)
    ax.text(1.8, 8.95, "round-robin minibatch", ha="center", fontsize=6.3, fontweight="bold", color=MUT)

    enc = box(ax, 4.4, 3.9, 3.1, 1.9, "Inductive\nST-GNN encoder\n(shared θ)", B_ENC, fs=6.3, weight="bold")
    for b in bxs:
        arr(ax, port(b, "r"), port(enc, "l"), lw=1.0, ms=7.5)
    rd = box(ax, 8.2, 3.9, 2.9, 1.9, "mean + max\nreadout\n$z=[\\mathrm{mean}\\Vert\\mathrm{max}]$",
             B_GRAPH, fs=6.1)
    arr(ax, port(enc, "r"), port(rd, "l"))

    fh = box(ax, 12.0, 5.7, 2.7, 1.5, "Forecast head\nMSE(ŷ, y)", B_PREP, fs=6.3)
    grl = box(ax, 12.0, 1.9, 2.1, 1.5, "GRL\n×(−λ)", B_S2, fs=6.4, weight="bold")
    disc = box(ax, 14.5, 1.85, 2.4, 1.6, "3-way city\ndiscriminator\nCE(city)", ("#f7ddd5", C_DELHI), fs=6.0)
    arr(ax, port(rd, "r"), port(fh, "l"), rad=0.10)
    arr(ax, port(rd, "r"), port(grl, "l"), rad=-0.10)
    arr(ax, port(grl, "r"), port(disc, "l"))
    ax.text(13.6, 0.75, "encoder trained to fool the discriminator → city-invariant embedding",
            ha="center", fontsize=5.8, style="italic", color=MUT)

    iax = fig.add_axes([0.49, 0.71, 0.155, 0.19])
    pp = np.linspace(0, 1, 200); lam = 2 / (1 + np.exp(-10 * pp)) - 1
    iax.plot(pp, lam, color=C_DELHI, lw=1.5)
    iax.set_title(r"$\lambda(p)$ warm-up", fontsize=6.0, pad=1)
    iax.set_xlabel("progress p", fontsize=5.4, labelpad=1)
    iax.tick_params(labelsize=4.8, length=2); iax.set_xlim(0, 1); iax.set_ylim(0, 1.05)
    for sp in iax.spines.values():
        sp.set_linewidth(0.5)

    fig.suptitle("Graph-DANN — adversarial alignment of the size-invariant graph embedding",
                 fontsize=9.0, y=1.0, fontweight="bold")
    save(fig, "fig4_graph_dann")


# ============================================================================
# FIGURE 5 — verification heatmaps + data-efficiency (viridis)
# ============================================================================
def fig5_verification():
    fig = plt.figure(figsize=(COL2, 3.0))
    gs = fig.add_gridspec(1, 5, width_ratios=[1, 1, 1, 0.07, 1.18], wspace=0.16)
    mats = [("Zero-shot", ZERO), ("Scratch", SCRATCH), ("Transfer (PT-FT)", S1)]
    vmin, vmax = 0.60, 0.84
    im = None
    for k, (title, M) in enumerate(mats):
        ax = fig.add_subplot(gs[k])
        im = ax.imshow(M, cmap=CMAP, vmin=vmin, vmax=vmax, aspect="auto")
        ax.set_title(title, fontsize=7.8, pad=3)
        ax.set_xticks(range(4)); ax.set_xticklabels(DCOLS, fontsize=6.2)
        ax.set_xlabel("target fraction d (%)", fontsize=6.6)
        for i in range(6):
            for j in range(4):
                v = M[i, j]
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=5.5,
                        color="white" if v < 0.74 else "#222222")
        if k == 0:
            ax.set_yticks(range(6)); ax.set_yticklabels(PAIRS, fontsize=5.9)
        else:
            ax.set_yticks([])
        ax.tick_params(length=2)
    cax = fig.add_subplot(gs[3])
    cb = fig.colorbar(im, cax=cax)
    cb.set_label("test R²", fontsize=6.8); cb.ax.tick_params(labelsize=5.8)

    axc = fig.add_subplot(gs[4])
    dvals = np.array([15, 30, 45, 60])
    for r, pair, c in [(0, "Delhi→Kolkata", C_DELHI), (2, "Kolkata→Delhi", C_KOL)]:
        axc.plot(dvals, S1[r], "-o", color=c, lw=1.5, ms=3.2, label=f"{pair}")
        axc.plot(dvals, SCRATCH[r], "--s", color=c, lw=1.0, ms=2.6, alpha=0.75)
        axc.axhline(ZS[r], color=c, lw=0.8, ls=":", alpha=0.7)
    axc.set_title("(d)  Data-efficiency", fontsize=7.8, loc="left", pad=3)
    axc.set_xlabel("target fraction d (%)", fontsize=6.6); axc.set_ylabel("test R²", fontsize=6.6)
    axc.set_xticks(dvals); axc.tick_params(labelsize=6.0, length=2)
    axc.grid(True, lw=0.4, alpha=0.35); axc.spines[["top", "right"]].set_visible(False)
    axc.legend(fontsize=5.4, loc="lower right", handlelength=1.6, title="solid=transfer",
               title_fontsize=5.2)
    axc.text(0.03, 0.07, "dashed = scratch\ndotted = zero-shot", transform=axc.transAxes,
             fontsize=5.0, color=MUT, style="italic", va="bottom")

    fig.suptitle("Transfer beats both zero-shot and scratch in all 24 PT-FT cells; margin is largest at low d",
                 fontsize=7.4, x=0.5, y=1.02)
    save(fig, "fig5_verification")


# ============================================================================
# FIGURE 6 — forecast fidelity (viridis; recognizable station)
# ============================================================================
def fig6_forecast():
    npz = ASSETS / "hero_delhi_to_kolkata_d30.npz"
    if not npz.exists():
        print("  [fig6] missing predictions; run export_predictions.py first"); return
    d = np.load(npz, allow_pickle=True)
    pred, true = d["pred"], d["true"]
    stations = list(d["stations"]); ts = pd.to_datetime(d["timestamps"])
    r2, mae, rmse = float(d["r2"]), float(d["mae"]), float(d["rmse"])

    fig = plt.figure(figsize=(COL2, 3.0))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.5], wspace=0.32)

    axA = fig.add_subplot(gs[0])
    yo, yp = true.reshape(-1), pred.reshape(-1)
    hi = float(np.nanpercentile(np.concatenate([yo, yp]), 99.0))
    hb = axA.hexbin(yo, yp, gridsize=32, cmap=CMAP, bins="log", mincnt=1, linewidths=0,
                    extent=(0, hi, 0, hi))
    axA.plot([0, hi], [0, hi], color="#d62728", lw=1.2, ls="--", zorder=4)
    axA.set_xlim(0, hi); axA.set_ylim(0, hi); axA.set_aspect("equal")
    axA.set_xlabel(r"Observed PM$_{2.5}$ ($\mu$g m$^{-3}$)", fontsize=6.8)
    axA.set_ylabel(r"Predicted PM$_{2.5}$ ($\mu$g m$^{-3}$)", fontsize=6.8, labelpad=1)
    axA.set_title("(a)  Predicted vs observed", fontsize=7.8, loc="left")
    axA.tick_params(labelsize=6.0, length=2)
    axA.text(0.05, 0.95, f"R² = {r2:.3f}\nMAE = {mae:.1f}\nRMSE = {rmse:.1f}",
             transform=axA.transAxes, va="top", ha="left", fontsize=6.5,
             bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#aab5bb", lw=0.6))
    cb = fig.colorbar(hb, ax=axA, fraction=0.046, pad=0.02)
    cb.set_label("count (log)", fontsize=6.0); cb.ax.tick_params(labelsize=5.6)
    axA.spines[["top", "right"]].set_visible(False)

    axB = fig.add_subplot(gs[1])
    order = np.argsort(ts.values); ts_s = ts.values[order]
    target = "Victoria" if "Victoria" in stations else stations[int(np.argmax(true.std(0)))]
    sidx = stations.index(target)
    to = true[order, sidx]; tp = pred[order, sidx]
    n = min(160, len(ts_s))
    axB.plot(ts_s[:n], to[:n], color="#2a3439", lw=1.3, label="Observed", zorder=3)
    axB.plot(ts_s[:n], tp[:n], color="#1f6cb0", lw=1.2, label="Predicted (PT-FT)", zorder=4)
    axB.fill_between(ts_s[:n], to[:n], tp[:n], color="#f0a020", alpha=0.22, lw=0, zorder=2)
    axB.set_title(f"(b)  Test-window trace — {target} station, Kolkata", fontsize=7.6, loc="left")
    axB.set_ylabel(r"PM$_{2.5}$ ($\mu$g m$^{-3}$)", fontsize=6.8, labelpad=2)
    axB.tick_params(labelsize=6.0, length=2)
    axB.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    axB.xaxis.set_major_locator(mdates.AutoDateLocator(maxticks=7))
    axB.legend(fontsize=6.2, loc="upper right", ncol=2)
    axB.grid(True, lw=0.4, alpha=0.3); axB.spines[["top", "right"]].set_visible(False)
    axB.margins(x=0.01)

    fig.suptitle("Forecast fidelity on the hero cell (Delhi→Kolkata, PT-FT, d = 30%) — real model output",
                 fontsize=7.6, y=1.02)
    save(fig, "fig6_forecast")


# ============================================================================
# FIGURE 7 — per-station spatial skill of the transferred model
# ============================================================================
def fig7_per_station():
    npz = ASSETS / "hero_delhi_to_kolkata_d30.npz"
    if not npz.exists():
        print("  [fig7] missing predictions; run export_predictions.py first"); return
    d = np.load(npz, allow_pickle=True)
    pred, true = d["pred"], d["true"]
    stations = list(d["stations"])
    meta = load_meta()
    coords = meta["Kolkata"]["coords"]

    r2s, maes = [], []
    for j in range(pred.shape[1]):
        yt, yp = true[:, j], pred[:, j]
        ss_res = float(((yt - yp) ** 2).sum())
        ss_tot = float(((yt - yt.mean()) ** 2).sum())
        r2s.append(1 - ss_res / max(ss_tot, 1e-9))
        maes.append(float(np.abs(yt - yp).mean()))
    r2s = np.array(r2s); maes = np.array(maes)

    names, pts, edges = knn_edges({s: coords[s] for s in stations}, k=3)
    lon, lat = pts[:, 1], pts[:, 0]

    def short(s):
        p = s.split()
        return p[0] if len(p) == 1 else f"{p[0]} {p[1][0]}."

    fig = plt.figure(figsize=(COL2, 2.7))
    # 4 columns: map | colorbar | spacer | bars  -> keeps the colorbar label
    # off the bar-chart station names.
    gs = fig.add_gridspec(1, 4, width_ratios=[1.0, 0.045, 0.20, 0.92], wspace=0.14)

    # (a) spatial skill map
    axA = fig.add_subplot(gs[0])
    for i, j in edges:
        axA.plot([lon[i], lon[j]], [lat[i], lat[j]], lw=0.5, color="#b9c3c9", alpha=0.8, zorder=1)
    sc = axA.scatter(lon, lat, c=r2s, s=165, cmap=CMAP, vmin=0.64, vmax=0.90,
                     ec="white", lw=0.8, zorder=3)
    nums = {s: k + 1 for k, s in enumerate(stations)}
    for k, s in enumerate(stations):
        tc = "white" if r2s[k] < 0.80 else "#1f2a30"
        axA.text(lon[k], lat[k], str(nums[s]), ha="center", va="center",
                 fontsize=5.2, fontweight="bold", color=tc, zorder=5)
    axA.set_title("(a)  Per-station skill — Kolkata", fontsize=7.8, loc="left")
    axA.set_xlabel("Longitude (°E)", fontsize=6.8); axA.set_ylabel("Latitude (°N)", fontsize=6.8)
    axA.tick_params(labelsize=5.8, length=2); axA.set_aspect("equal", adjustable="datalim")
    axA.margins(0.15)
    cax = fig.add_subplot(gs[1])
    cb = fig.colorbar(sc, cax=cax)
    cb.ax.set_title("R²", fontsize=6.4, pad=3); cb.ax.tick_params(labelsize=5.6)

    # (b) per-station MAE bars (sorted)
    axB = fig.add_subplot(gs[3])
    order = np.argsort(maes)
    yps = np.arange(len(order))
    axB.barh(yps, maes[order], color=C_KOL, ec="white", lw=0.4, height=0.72)
    axB.set_yticks(yps)
    axB.set_yticklabels([f"{i + 1}. {short(stations[i])}" for i in order], fontsize=5.6)
    axB.invert_yaxis()
    for k, i in enumerate(order):
        axB.text(maes[i] + 0.15, k, f"{maes[i]:.1f}", va="center", fontsize=5.4, color="#2a3439")
    axB.set_xlabel("station MAE (µg m$^{-3}$)", fontsize=6.8)
    axB.set_title("(b)  Per-station error", fontsize=7.8, loc="left")
    axB.tick_params(labelsize=5.8, length=2)
    axB.set_xlim(0, maes.max() * 1.18)
    axB.spines[["top", "right"]].set_visible(False)
    axB.grid(axis="x", lw=0.4, alpha=0.3)

    fig.suptitle("Transferred skill is spatially distributed across all 10 Kolkata monitors "
                 f"(R² = {r2s.min():.2f}–{r2s.max():.2f})",
                 fontsize=7.5, y=1.02)
    save(fig, "fig7_per_station")


if __name__ == "__main__":
    fig1_study_area()
    fig2_framework()
    fig3_encoder()
    fig4_graph_dann()
    fig5_verification()
    fig6_forecast()
    fig7_per_station()
    print("All 7 figures written to", HERE)
