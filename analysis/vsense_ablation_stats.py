"""Statistical hardening of the physics-prior null (reviewer issue C5/R2-M5).

Reads the per-seed paired deltas in results/gnn_vsense/vsense_ablation.json and reports,
per (city, k) and physics mode:

  - mean +/- sd and a 95% t-interval on dR2 = R2(mode) - R2(data-only);
  - the one-sample t-test against 0 (as before);
  - a TOST equivalence test against the pre-registered practical-significance margin
    |dR2| < DELTA (= 0.01 R2): p_TOST = max of the two one-sided p-values. p_TOST < 0.05
    means the effect is statistically confined WITHIN the margin — i.e. practically null —
    which is a positive statement of absence, not a mere failure to reject.

This addresses the criticism that "no significant gain" from an n=3/n=5 t-test is weak
evidence of absence, and resolves the diffusion-only p=0.05 Kolkata cell: its effect is
real but bounded well inside the practical margin.

Run:  python -u -m analysis.vsense_ablation_stats
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

IN = Path("results/gnn_vsense/vsense_ablation.json")
OUT = Path("results/gnn_vsense/vsense_ablation_stats.json")
DELTA = 0.01          # pre-registered practical-significance margin on dR2
CONFIGS = [("Kolkata", 4), ("Delhi", 16)]


def summarize(d: np.ndarray) -> dict:
    n = len(d)
    mean = float(np.mean(d))
    sd = float(np.std(d, ddof=1))
    se = sd / np.sqrt(n)
    tcrit = stats.t.ppf(0.975, n - 1)
    ci = (mean - tcrit * se, mean + tcrit * se)
    t0, p0 = stats.ttest_1samp(d, 0.0)
    # TOST: H0a: mu <= -DELTA (p1);  H0b: mu >= +DELTA (p2);  equivalence if max(p1,p2) < .05
    t1 = (mean + DELTA) / se
    p1 = float(1 - stats.t.cdf(t1, n - 1))
    t2 = (mean - DELTA) / se
    p2 = float(stats.t.cdf(t2, n - 1))
    return {
        "n": n, "mean": mean, "sd": sd,
        "ci95": [float(ci[0]), float(ci[1])],
        "t_vs_0": float(t0), "p_vs_0": float(p0),
        "tost_margin": DELTA, "p_tost": max(p1, p2),
        "equivalent_at_05": bool(max(p1, p2) < 0.05),
    }


def main():
    data = json.loads(IN.read_text())
    out = {}
    for cname, k in CONFIGS:
        rows = {key: v for key, v in data.items() if key.startswith(f"{cname}|k={k}|")}
        seeds = sorted(rows)
        for mode in ("d_both", "d_adv", "d_diff"):
            d = np.array([rows[s][mode] for s in seeds], float)
            s = summarize(d)
            out[f"{cname}|k={k}|{mode}"] = s
            print(f"{cname} k={k} {mode:7s}: mean={s['mean']:+.4f}+/-{s['sd']:.4f} "
                  f"CI95=[{s['ci95'][0]:+.4f},{s['ci95'][1]:+.4f}] p_vs_0={s['p_vs_0']:.3f} "
                  f"p_TOST={s['p_tost']:.4f} equiv(|d|<{DELTA})={s['equivalent_at_05']}")
        # paired advection-vs-diffusion contrast (the transport-physics claim)
        da = np.array([rows[s]["d_adv"] for s in seeds], float)
        dd = np.array([rows[s]["d_diff"] for s in seeds], float)
        s = summarize(da - dd)
        out[f"{cname}|k={k}|adv_minus_diff"] = s
        print(f"{cname} k={k} adv-diff: mean={s['mean']:+.4f} "
              f"CI95=[{s['ci95'][0]:+.4f},{s['ci95'][1]:+.4f}] p_vs_0={s['p_vs_0']:.3f} "
              f"p_TOST={s['p_tost']:.4f} equiv={s['equivalent_at_05']}")

    OUT.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
