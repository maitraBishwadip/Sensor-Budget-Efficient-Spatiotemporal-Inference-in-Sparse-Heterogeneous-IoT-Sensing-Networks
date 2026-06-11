"""Sensor-budget / node-dropping experiment — the Variant E (GNN-PINN-TL) hero test.

Strips a *target* deployment down to k active sensor nodes (source stays full) and asks
whether the ADR physics prior recovers accuracy the data-only model loses as nodes vanish.
This is the decisive "more value from fewer sensors" test: we expect the physics gain

    Δ(k) = R²(λ_phys > 0) − R²(λ_phys = 0)

to GROW (turn positive) as k shrinks, because few nodes under-determine the spatial field
and the transport prior fills the gap — whereas at full nodes the data already determines it
(see the pilot null in reports/PINN_PHYSICS.md / project memory).

λ=0 and λ>0 run through the *same* trainer (transfer_variant_e), so the only difference is the
physics term. Nodes are dropped in nested subsets (fixed permutation, first k) so the curve is
not confounded by which nodes are kept.

Run:  python -u -m analysis.sensor_budget        (or: python -u analysis/sensor_budget.py)
"""
from __future__ import annotations

import dataclasses
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.train_gnn import load_all_cities                      # noqa: E402
from src.train_gnn_pinn import transfer_variant_e              # noqa: E402
from src.utils import CityTensors, write_results               # noqa: E402

OUT = Path("results/gnn_pinn/sensor_budget.json")
D_PCT = 0.30
LAMBDAS = [0.0, 0.2, 0.5]
CONFIGS = [("Delhi", "Kolkata"), ("Kolkata", "Delhi")]          # (source full, target stripped)


def subset_city_nodes(city: CityTensors, idx) -> CityTensors:
    """Return a CityTensors keeping only node indices `idx` (axis 1). Global scalers and
    per-timestep masks are unchanged; climatology and station/coord lists are subset."""
    idx = list(idx)
    clim = None if city.climatology_tensor is None else city.climatology_tensor[:, idx]
    return dataclasses.replace(
        city,
        feature_tensor=city.feature_tensor[:, idx, :],
        target_tensor=city.target_tensor[:, idx],
        stations=[city.stations[i] for i in idx],
        coords=[city.coords[i] for i in idx],
        climatology_tensor=clim,
    )


def main():
    cities = load_all_cities(fixed=True)
    results = {}
    for s, t in CONFIGS:
        N = len(cities[t].coords)
        ks = [k for k in ([4, 6, 8, N] if N <= 10 else [4, 8, 16, N]) if k <= N]
        perm = np.random.default_rng(0).permutation(N)           # fixed → nested subsets
        for k in sorted(set(ks)):
            idx = sorted(perm[:k].tolist())
            cmod = dict(cities)
            cmod[t] = subset_city_nodes(cities[t], idx)
            for lam in LAMBDAS:
                print(f"\n##### {s}->{t}  k={k}/{N}  lambda_phys={lam} #####")
                r = transfer_variant_e(cmod, s, t, D_PCT, fixed=True, lambda_phys=lam)
                results[f"{s}->{t}|k={k}|lam={lam}"] = {
                    "R2": r["transfer"]["R2"], "MAE": r["transfer"]["MAE"],
                    "zero_shot_R2": r["zero_shot"]["R2"], "k": k, "N": N,
                    "lambda": lam, "source": s, "target": t, "node_idx": idx,
                    "coeffs": r.get("coeffs"), "L_phys": r.get("L_phys_last"),
                }
                write_results(OUT, results)

    print("\n" + "=" * 72)
    print("SENSOR-BUDGET SUMMARY - dR2 (physics minus data-only) vs active nodes k")
    print("=" * 72)
    for s, t in CONFIGS:
        ks = sorted({v["k"] for v in results.values() if v["source"] == s and v["target"] == t})
        print(f"\n{s} -> {t}   (target stripped to k of {cities[t].feature_tensor.shape[1]} nodes, d={int(D_PCT*100)}%)")
        header = "  k    R2(lam=0)  MAE(lam=0) | " + "  ".join(f"dR2(lam={l})" for l in LAMBDAS if l > 0)
        print(header)
        for k in ks:
            r0 = results[f"{s}->{t}|k={k}|lam=0.0"]
            deltas = []
            for l in LAMBDAS:
                if l == 0:
                    continue
                rl = results[f"{s}->{t}|k={k}|lam={l}"]["R2"]
                deltas.append(f"{rl - r0['R2']:+.4f}")
            print(f"  {k:<3}  {r0['R2']:.4f}   {r0['MAE']:6.3f}  | " + "    ".join(deltas))
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
