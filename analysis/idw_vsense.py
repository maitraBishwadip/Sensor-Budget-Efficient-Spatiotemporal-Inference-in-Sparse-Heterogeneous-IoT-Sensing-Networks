"""Training-free interpolation baselines for virtual sensing (reviewer issue C4/R2-M4).

For each unsensored node u, estimate PM2.5 at time t as a weighted mean of the CONCURRENT
PM2.5 measured at the sensored nodes S:

  - IDW:  w_us = 1 / d_us^p            (inverse-distance weighting, p = 2)
  - GK :  w_us = exp(-d_us^2 / 2s^2)   (Gaussian kernel, sigma = 5 km, matching Eq. (1))

Node splits are IDENTICAL to src/train_gnn_vsense.py (np.random.default_rng(0).permutation(N),
sensored = first k), and evaluation runs on exactly the prediction timesteps of the GNN's test
windows (test_mask timesteps reachable by a history-H window), in raw ug/m3 space. Note the
baselines see the sensored PM2.5 AT the prediction timestep — information the ST-GNN does not
have (its inputs end one horizon step earlier) — so this is a strong, slightly favoured
comparator, not a strawman.

Writes results/gnn_vsense/idw_vsense.json (interleaved split) or
results/gnn_vsense/idw_vsense_chrono.json (--chrono; chronological block split).
Being training-free, the kernels have no train/test regime-mismatch problem, so
the chrono variant doubles as the protocol-sensitivity check for this baseline.

Run:  python -u -m analysis.idw_vsense [--chrono]
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.graph_construction import pairwise_distance_matrix     # noqa: E402
from src.train_gnn import HISTORY, HORIZON, load_all_cities      # noqa: E402
from src.utils import all_metrics                                # noqa: E402

GRID = {"Delhi": [8, 16], "Kolkata": [4]}
IDW_P = 2.0
SIGMA_KM = 5.0


def main(chrono: bool = False):
    import json

    out = Path("results/gnn_vsense/idw_vsense" + ("_chrono" if chrono else "") + ".json")
    cities = load_all_cities(fixed=True, chrono=chrono)
    results = {}
    for cname, ks in GRID.items():
        c = cities[cname]
        T, N = c.target_tensor.shape
        # Reconstruct raw PM2.5 [T, N]: inverse z-score of the residual + climatology add-back.
        raw = c.target_scaler.inverse_transform(
            c.target_tensor.reshape(-1, 1)).reshape(T, N).astype(np.float64)
        if c.climatology_tensor is not None:
            raw = raw + c.climatology_tensor
        D = pairwise_distance_matrix(c.coords).astype(np.float64)   # km

        # Same evaluation timesteps as the GNN test windows: pred_t in test_mask,
        # reachable by a full history window.
        first_pred = HISTORY + HORIZON - 1
        eval_t = np.where(c.test_mask)[0]
        eval_t = eval_t[eval_t >= first_pred]

        for k in ks:
            perm = np.random.default_rng(0).permutation(N)
            S = np.sort(perm[:k])
            U = np.sort(perm[k:])
            y_true = raw[np.ix_(eval_t, U)]                        # [T_eval, |U|]
            # Concurrent sensored readings (nowcast interpolation) and lagged ones
            # (t - HORIZON steps: exactly the information the ST-GNN's window ends with).
            y_S_now = raw[np.ix_(eval_t, S)]                       # [T_eval, |S|]
            y_S_lag = raw[np.ix_(eval_t - HORIZON, S)]

            for method in ("idw", "gk", "idw_lag", "gk_lag"):
                d = D[np.ix_(U, S)]                                # [|U|, |S|]
                if method.startswith("idw"):
                    w = 1.0 / np.maximum(d, 1e-6) ** IDW_P
                else:
                    w = np.exp(-(d ** 2) / (2 * SIGMA_KM ** 2))
                w = w / np.maximum(w.sum(axis=1, keepdims=True), 1e-12)
                y_S = y_S_lag if method.endswith("_lag") else y_S_now
                y_hat = y_S @ w.T                                  # [T_eval, |U|]
                m = all_metrics(y_true, y_hat)
                results[f"{cname}|k={k}|{method}"] = {
                    **{kk: float(vv) for kk, vv in m.items()},
                    "k": int(k), "N": int(N), "n_unsensored": int(len(U)),
                    "method": method, "n_eval_timesteps": int(len(eval_t)),
                }
                print(f"{cname:8s} k={k:<2d} {method.upper():7s}: R2={m['R2']:.4f} "
                      f"MAE={m['MAE']:.3f} RMSE={m['RMSE']:.3f}")

    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--chrono", action="store_true",
                   help="Chronological block split (protocol-sensitivity study).")
    main(chrono=p.parse_args().chrono)
