"""F5: Sensor-budget / graceful degradation under cold-start transfer.

Two views of a shrinking sensor budget k, per transfer direction:

  1. RETAINED view (results/gnn_pinn/sensor_budget.json, lambda=0): the dropped
     N-k nodes are removed from the network and forecasting is evaluated at the
     retained k sensors. This is the ">=0.80 R2" graceful-degradation claim.
  2. COVERAGE view (results/gnn_vsense/sensor_budget_vsense.json): all N nodes
     stay in the graph, PM2.5 is masked at the N-k dropped nodes, and inference
     is evaluated *at those unsensored nodes* (the Lever-1 task under budget
     shrinkage). Necessarily lower; the honest bound on coverage.

Each view carries its own zero-shot (no fine-tune) reference.

Reads:  results/gnn_pinn/sensor_budget.json
        results/gnn_vsense/sensor_budget_vsense.json
Writes: paper_figs/fig_sensor_budget.{pdf,png}
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import matplotlib.pyplot as plt
from figstyle import RESULTS, DCOL_W, PALETTE, save

retained = json.loads((RESULTS / "gnn_pinn" / "sensor_budget.json").read_text())
coverage = json.loads((RESULTS / "gnn_vsense" / "sensor_budget_vsense.json").read_text())


def _series(blob, source, target, lam_only=False):
    rows = []
    for v in blob.values():
        if v["source"] != source or v["target"] != target:
            continue
        if lam_only and abs(v.get("lambda", 0.0)) > 1e-9:
            continue
        rows.append((v["k"], v["R2"], v["zero_shot_R2"]))
    rows.sort()
    return [r[0] for r in rows], [r[1] for r in rows], [r[2] for r in rows]


ORDER = [("Delhi", "Kolkata", 10), ("Kolkata", "Delhi", 40)]
YLIMS = [(0.45, 0.92), (0.0, 0.92)]
fig, axes = plt.subplots(1, 2, figsize=(DCOL_W, 2.7))

for ax, (src, tgt, N), ylim in zip(axes, ORDER, YLIMS):
    kr, r2r, zsr = _series(retained, src, tgt, lam_only=True)
    kc, r2c, zsc = _series(coverage, src, tgt)

    ax.plot(kr, r2r, "-o", color=PALETTE["gnn"],
            label="Forecasting at retained $k$ sensors")
    ax.plot(kr, zsr, "--o", color=PALETTE["gnn"], alpha=0.45, markersize=3,
            markerfacecolor="none", label="  ... zero-shot")
    ax.plot(kc, r2c, "-^", color=PALETTE["ok"],
            label="Virtual sensing at dropped $N{-}k$ nodes")
    ax.plot(kc, zsc, "--^", color=PALETTE["ok"], alpha=0.45, markersize=3,
            markerfacecolor="none", label="  ... zero-shot")
    ax.axhline(0.80, color=PALETTE["accent"], lw=0.8, ls=":", alpha=0.8)

    ax.set_xlabel(f"# sensor nodes retained ($k$ of {N})")
    ax.set_title(f"({'a' if tgt == 'Kolkata' else 'b'}) {src}$\\rightarrow${tgt}")
    ax.set_xticks(sorted(set(kr) | set(kc)))
    ax.set_ylim(*ylim)

axes[0].set_ylabel(r"$R^2$")
axes[1].legend(loc="lower right", framealpha=0.9)
fig.tight_layout()
save(fig, "fig_sensor_budget")
