"""F5: Sensor-budget / graceful degradation under cold-start transfer.

How accuracy at unsensored nodes varies with the number of physical sensor nodes k
retained in the target deployment, for a model cold-started from a dense source.
Data-only setting (lambda=0). Zero-shot (source-only, no fine-tune) shown as reference.

Reads:  results/gnn_pinn/sensor_budget.json
Writes: paper_figs/fig_sensor_budget.{pdf,png}
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import matplotlib.pyplot as plt
from figstyle import RESULTS, DCOL_W, PALETTE, save

sb = json.loads((RESULTS / "gnn_pinn" / "sensor_budget.json").read_text())

# collect lambda=0 rows per transfer direction
dirs = defaultdict(list)
for v in sb.values():
    if abs(v["lambda"]) > 1e-9:
        continue
    dirs[(v["source"], v["target"], v["N"])].append((v["k"], v["R2"], v["zero_shot_R2"]))

ORDER = [("Delhi", "Kolkata", 10), ("Kolkata", "Delhi", 40)]
fig, axes = plt.subplots(1, 2, figsize=(DCOL_W, 2.6))

for ax, key in zip(axes, ORDER):
    rows = sorted(dirs[key])
    ks = [r[0] for r in rows]
    r2 = [r[1] for r in rows]
    zs = [r[2] for r in rows]
    src, tgt, N = key
    ax.plot(ks, r2, "-o", color=PALETTE["gnn"], label="ST-GNN (cold-start)")
    ax.plot(ks, zs, "--s", color=PALETTE["zeroshot"], markersize=3, label="Zero-shot (no fine-tune)")
    ax.axhline(0.80, color=PALETTE["accent"], lw=0.8, ls=":", alpha=0.8)
    ax.set_xlabel(f"# sensor nodes retained ($k$ of {N})")
    ax.set_title(f"({'a' if key==ORDER[0] else 'b'}) {src}$\\rightarrow${tgt}")
    ax.set_xticks(ks)
    ax.set_ylim(0.55, 0.9)

axes[0].set_ylabel(r"$R^2$ at unsensored nodes")
axes[0].legend(loc="lower left", framealpha=0.9)
fig.tight_layout()
save(fig, "fig_sensor_budget")
