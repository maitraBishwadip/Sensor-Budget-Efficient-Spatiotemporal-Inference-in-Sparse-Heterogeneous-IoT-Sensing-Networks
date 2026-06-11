"""Virtual-sensing validation: multi-seed robustness + wind-vs-smoothness ablation.

Decides whether the small positive physics gain in virtual sensing (results/gnn_vsense/) is
(a) real across node-split seeds, and (b) genuine TRANSPORT physics (the wind/advection term)
vs generic graph smoothness (diffusion only ~ Kriging/Tikhonov).

For each (city, k) and each node-split seed we train 4 models at the SAME split:
  base = data-only (lambda=0);  both = adv+diff;  adv = advection-only;  diff = diffusion-only.
Paired per-seed deltas  Δ_mode = R2(mode) - R2(base)  are aggregated; we test
  - Δ_both, Δ_adv, Δ_diff vs 0      (does physics help?)
  - Δ_adv vs Δ_diff (paired)        (does the WIND add value over smoothing? = the physics claim)

Run:  python -u -m analysis.vsense_ablation
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.train_gnn import load_all_cities                              # noqa: E402
from src.train_gnn_pinn import WS_IDX, SIN_WD_IDX, COS_WD_IDX          # noqa: E402
from src.train_gnn_vsense import virtual_sense                         # noqa: E402
from src.utils import write_results                                    # noqa: E402

try:
    from scipy import stats
    HAVE_SCIPY = True
except Exception:
    HAVE_SCIPY = False

OUT = Path("results/gnn_vsense/vsense_ablation.json")
LAMBDA = 0.2
CONFIGS = [("Kolkata", 4, [0, 1, 2, 3, 4]), ("Delhi", 16, [0, 1, 2])]   # (city, k, seeds)


def _wind_consts(c):
    fs, fm = c.scaler.scale_, c.scaler.mean_
    return (float(fs[WS_IDX]), float(fm[WS_IDX]), float(fs[SIN_WD_IDX]), float(fm[SIN_WD_IDX]),
            float(fs[COS_WD_IDX]), float(fm[COS_WD_IDX]))


def _ttest_1samp(x):
    x = np.asarray(x, float)
    if HAVE_SCIPY and len(x) > 1:
        t, p = stats.ttest_1samp(x, 0.0)
        return float(t), float(p)
    return float("nan"), float("nan")


def _ttest_rel(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if HAVE_SCIPY and len(a) > 1:
        t, p = stats.ttest_rel(a, b)
        return float(t), float(p)
    return float("nan"), float("nan")


def main():
    cities = load_all_cities(fixed=True)
    results = {}
    for cname, k, seeds in CONFIGS:
        c = cities[cname]
        wc = _wind_consts(c)
        sy, my = float(c.target_scaler.scale_[0]), float(c.target_scaler.mean_[0])
        for seed in seeds:
            common = dict(backbone="gat", wind_consts=wc, sy=sy, my=my, seed_split=seed)
            base = virtual_sense(c, k, 0.0, **common)["R2"]
            both = virtual_sense(c, k, LAMBDA, phys_use_adv=True, phys_use_diff=True, **common)["R2"]
            adv = virtual_sense(c, k, LAMBDA, phys_use_adv=True, phys_use_diff=False, **common)["R2"]
            diff = virtual_sense(c, k, LAMBDA, phys_use_adv=False, phys_use_diff=True, **common)["R2"]
            results[f"{cname}|k={k}|seed={seed}"] = {
                "base": base, "both": both, "adv": adv, "diff": diff,
                "d_both": both - base, "d_adv": adv - base, "d_diff": diff - base}
            write_results(OUT, results)
            print(f"  [{cname} k={k} seed={seed}] base={base:.4f} "
                  f"d_both={both-base:+.4f} d_adv={adv-base:+.4f} d_diff={diff-base:+.4f}")

    print("\n" + "=" * 78)
    print("VSENSE ABLATION SUMMARY (lambda=0.2)  -- dR2 = mode - data-only, paired per seed")
    print("=" * 78)
    for cname, k, seeds in CONFIGS:
        rows = [results[f"{cname}|k={k}|seed={s}"] for s in seeds]
        d_both = [r["d_both"] for r in rows]; d_adv = [r["d_adv"] for r in rows]; d_diff = [r["d_diff"] for r in rows]
        print(f"\n{cname} k={k} (n={len(seeds)} seeds):")
        for name, d in [("both", d_both), ("adv-only", d_adv), ("diff-only", d_diff)]:
            t, p = _ttest_1samp(d)
            print(f"  dR2[{name:9s}] mean={np.mean(d):+.4f} +/- {np.std(d, ddof=1):.4f}   "
                  f"(t={t:+.2f}, p={p:.3f} vs 0)")
        t, p = _ttest_rel(d_adv, d_diff)
        print(f"  WIND TEST  adv-only vs diff-only:  mean diff={np.mean(np.array(d_adv)-np.array(d_diff)):+.4f}  "
              f"(paired t={t:+.2f}, p={p:.3f})  <- >0 & sig => advection adds over smoothing")
    print(f"\nwrote {OUT}  (scipy={'yes' if HAVE_SCIPY else 'NO - p-values nan'})")


if __name__ == "__main__":
    main()
