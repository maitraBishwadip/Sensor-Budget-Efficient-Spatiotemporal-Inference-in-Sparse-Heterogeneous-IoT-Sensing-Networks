"""F6: Cold-start data-efficiency -- transfer vs scratch vs zero-shot across target budget.

For each of the six ordered deployment pairs, target-test R^2 as a function of the
fraction d% of target labels used, comparing pre-train+fine-tune (transfer) to a
from-scratch model and the zero-shot source. Transfer dominates, with the largest
margin where target data is scarcest.

Reads:  results/gnn_tl/variantA_gat_fixed.json
Writes: paper_figs/fig_dataeff.{pdf,png}
"""
from __future__ import annotations

import json
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import matplotlib.pyplot as plt
from figstyle import RESULTS, DCOL_W, PALETTE, save

tl = json.loads((RESULTS / "gnn_tl" / "variantA_gat_fixed.json").read_text())

pat = re.compile(r"(\w+)->(\w+)@(\d+)")
cells = defaultdict(dict)  # (src,tgt) -> d -> dict
for key, v in tl.items():
    m = pat.match(key)
    src, tgt, d = m.group(1), m.group(2), int(m.group(3))
    cells[(src, tgt)][d] = v

PAIRS = [("Delhi", "Kolkata"), ("Delhi", "Guwahati"), ("Kolkata", "Delhi"),
         ("Kolkata", "Guwahati"), ("Guwahati", "Delhi"), ("Guwahati", "Kolkata")]
ds = [15, 30, 45, 60]

fig, axes = plt.subplots(2, 3, figsize=(DCOL_W, 3.6), sharex=True, sharey=True)
for ax, pair in zip(axes.ravel(), PAIRS):
    c = cells[pair]
    transfer = [c[d]["transfer"]["R2"] for d in ds]
    scratch = [c[d]["scratch"]["R2"] for d in ds]
    zshot = [c[d]["zero_shot"]["R2"] for d in ds]
    ax.plot(ds, transfer, "-o", color=PALETTE["transfer"], label="Transfer")
    ax.plot(ds, scratch, "--s", color=PALETTE["scratch"], markersize=3, label="From scratch")
    ax.plot(ds, zshot, ":^", color=PALETTE["zeroshot"], markersize=3, label="Zero-shot")
    ax.set_title(f"{pair[0]}$\\rightarrow${pair[1]}", fontsize=7)
    ax.set_xticks(ds)

for ax in axes[-1]:
    ax.set_xlabel(r"target labels $d$ (\%)" if plt.rcParams["text.usetex"] else "target labels d (%)")
for ax in axes[:, 0]:
    ax.set_ylabel(r"target $R^2$")
axes[0, 0].legend(loc="lower right", fontsize=6, framealpha=0.9)
fig.tight_layout()
save(fig, "fig_dataeff")
