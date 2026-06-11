"""F4 (HERO): Virtual sensing -- estimating the signal at UNSENSORED nodes.

Compares the inductive ST-GNN against a station-independent meteorology-only LSTM at
held-out unsensored nodes. The GNN can borrow from sensored neighbours over the graph;
the station-independent LSTM cannot -- this is the GNN's non-removable value.

Reads:  results/gnn_vsense/vsense_gat.json, results/lstm/lstm_vsense_peru.json
Writes: paper_figs/fig_vsense_hero.{pdf,png}
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import matplotlib.pyplot as plt
from figstyle import RESULTS, DCOL_W, PALETTE, save

vs = json.loads((RESULTS / "gnn_vsense" / "vsense_gat.json").read_text())
base = json.loads((RESULTS / "lstm" / "lstm_vsense_peru.json").read_text())

# (city, k) in presentation order (densest sensed first)
CONFIGS = [("Delhi", 16, 40), ("Delhi", 8, 40), ("Kolkata", 4, 10)]

labels, lstm_r2, gnn_r2, lstm_mae, gnn_mae, n_uns = [], [], [], [], [], []
for city, k, N in CONFIGS:
    b = base[f"{city}|k={k}"]
    v = vs[f"{city}|k={k}|lam=0.0"]
    labels.append(f"{city}\n{k}/{N} sensed")
    lstm_r2.append(b["lstm_metonly_U"]["R2"])
    gnn_r2.append(b["gnn_U"])
    lstm_mae.append(b["lstm_metonly_U"]["MAE"])
    gnn_mae.append(v["MAE"])
    n_uns.append(b["n_unsensored"])

x = np.arange(len(CONFIGS))
w = 0.38
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(DCOL_W, 2.7))

# --- panel (a): R^2 at unsensored nodes ---------------------------------
ax1.bar(x - w / 2, lstm_r2, w, label="Met-only LSTM", color=PALETTE["lstm"], edgecolor="k", linewidth=0.4)
ax1.bar(x + w / 2, gnn_r2, w, label="ST-GNN (ours)", color=PALETTE["gnn"], edgecolor="k", linewidth=0.4)
for i in range(len(x)):
    d = gnn_r2[i] - lstm_r2[i]
    ax1.annotate(f"+{d:.2f}", (x[i] + w / 2, gnn_r2[i]), textcoords="offset points",
                 xytext=(0, 2), ha="center", fontsize=6.5, color=PALETTE["accent"], fontweight="bold")
ax1.set_ylabel(r"$R^2$ at unsensored nodes")
ax1.set_ylim(0, 0.9)
ax1.set_xticks(x)
ax1.set_xticklabels(labels)
ax1.set_title("(a) Spatial inference skill")

# --- panel (b): MAE at unsensored nodes ---------------------------------
ax2.bar(x - w / 2, lstm_mae, w, label="Met-only LSTM", color=PALETTE["lstm"], edgecolor="k", linewidth=0.4)
ax2.bar(x + w / 2, gnn_mae, w, label="ST-GNN (ours)", color=PALETTE["gnn"], edgecolor="k", linewidth=0.4)
ax2.set_ylabel(r"MAE ($\mu$g m$^{-3}$)")
ax2.set_xticks(x)
ax2.set_xticklabels(labels)
ax2.set_title("(b) Spatial inference error")
ax2.legend(loc="upper right", framealpha=0.9)

fig.tight_layout()
save(fig, "fig_vsense_hero")
