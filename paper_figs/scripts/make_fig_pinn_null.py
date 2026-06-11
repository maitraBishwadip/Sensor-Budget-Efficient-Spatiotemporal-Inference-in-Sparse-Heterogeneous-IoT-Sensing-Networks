"""F7 (honest negative): the advection-diffusion physics prior gives no significant gain.

Multi-seed ablation of the spatial physics residual on the virtual-sensing task.
Bars = mean Delta-R^2 (physics minus data-only) with +/-1 s.d. error bars, for the
three physics modes (both / advection-only / diffusion-only). The error bars straddle
zero -- any marginal effect is generic graph smoothing, not transport physics.

Reads:  results/gnn_vsense/vsense_ablation.json
Writes: paper_figs/fig_pinn_null.{pdf,png}
"""
from __future__ import annotations

import json
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import matplotlib.pyplot as plt
from figstyle import RESULTS, COL_W, PALETTE, save

ab = json.loads((RESULTS / "gnn_vsense" / "vsense_ablation.json").read_text())

pat = re.compile(r"(\w+)\|k=(\d+)\|seed=(\d+)")
groups = defaultdict(lambda: defaultdict(list))  # config -> mode -> [delta]
for key, v in ab.items():
    m = pat.match(key)
    cfg = f"{m.group(1)} (k={m.group(2)})"
    for mode in ("both", "adv", "diff"):
        groups[cfg][mode].append(v[f"d_{mode}"])

CFGS = list(groups.keys())
MODES = [("both", "Adv+Diff"), ("adv", "Adv only"), ("diff", "Diff only")]
colors = [PALETTE["gnn"], PALETTE["scratch"], PALETTE["ok"]]

x = np.arange(len(CFGS))
w = 0.25
fig, ax = plt.subplots(figsize=(COL_W, 2.5))
for j, (mode, mlabel) in enumerate(MODES):
    means = [np.mean(groups[c][mode]) for c in CFGS]
    sds = [np.std(groups[c][mode], ddof=1) for c in CFGS]
    ax.bar(x + (j - 1) * w, means, w, yerr=sds, capsize=2.5, label=mlabel,
           color=colors[j], edgecolor="k", linewidth=0.4, error_kw={"lw": 0.8})

ax.axhline(0, color="k", lw=0.7)
ax.set_xticks(x)
ax.set_xticklabels(CFGS, fontsize=6.5)
ax.set_ylabel(r"$\Delta R^2$ (physics $-$ data-only)")
ax.set_title("Physics prior: no significant gain (n.s.)")
ax.legend(loc="upper right", fontsize=6, framealpha=0.9)
fig.tight_layout()
save(fig, "fig_pinn_null")
