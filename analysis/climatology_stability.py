"""Climatology-slot stability analysis (reviewer request: short-record climatology).

The climatology-residual target of Eq. (2) subtracts a per-(month, hour) mean
c_n(m, h) fit on the training slice. On a one-year deployment a (month, hour)
slot recurs about once per day, so the training slice supplies roughly
30 x 0.714 ~ 21 samples per slot. This script quantifies how stable those slot
means are:

  1. Sample counts per (month, hour) slot on the train mask (min / mean / max
     over the 12 x 8 = 96 slots), per deployment.
  2. Split-half reliability: the train occurrences of each slot are divided
     into two disjoint interleaved halves; the climatology is computed on each
     half and compared. Reported are the mean absolute half-vs-half
     disagreement |c1 - c2| (in ug/m3 and as a fraction of the residual
     standard deviation the model actually regresses) and the Pearson
     correlation between the two half-climatologies over all (slot, station)
     pairs. Because each half uses ~10 samples per slot, this bounds the
     sampling noise of the full ~21-sample climatology from above.

Writes results/climatology_stability.json.

Run:  python -u -m analysis.climatology_stability
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.train_gnn import load_all_cities  # noqa: E402

OUT = Path("results/climatology_stability.json")


def main():
    cities = load_all_cities(fixed=True)
    results = {}
    for cname, c in cities.items():
        T, N = c.target_tensor.shape
        months = c.timestamps.month.to_numpy()
        hours = c.timestamps.hour.to_numpy()
        train = c.train_mask

        # Raw PM2.5 [T, N]: undo the z-score of the residual, add climatology back.
        raw = c.target_scaler.inverse_transform(
            c.target_tensor.reshape(-1, 1)).reshape(T, N).astype(np.float64)
        if c.climatology_tensor is not None:
            raw = raw + c.climatology_tensor

        counts, c1_list, c2_list, cf_list = [], [], [], []
        for m in range(1, 13):
            for h in range(0, 24, 3):
                idx = np.where(train & (months == m) & (hours == h))[0]
                counts.append(len(idx))
                if len(idx) < 4:
                    continue
                half1, half2 = idx[0::2], idx[1::2]       # disjoint interleaved halves
                c1_list.append(raw[half1].mean(axis=0))   # [N]
                c2_list.append(raw[half2].mean(axis=0))
                cf_list.append(raw[idx].mean(axis=0))
        counts = np.asarray(counts)
        c1 = np.stack(c1_list)   # [slots, N]
        c2 = np.stack(c2_list)

        # Residual scale the model regresses (std of PM2.5 minus full climatology, train slice).
        resid_std = float(c.target_scaler.scale_[0])
        dis = np.abs(c1 - c2).ravel()
        corr = float(np.corrcoef(c1.ravel(), c2.ravel())[0, 1])

        results[cname] = {
            "N": int(N), "T": int(T), "n_slots": int(len(counts)),
            "train_count_min": int(counts.min()),
            "train_count_mean": float(counts.mean()),
            "train_count_max": int(counts.max()),
            "n_empty_slots": int((counts == 0).sum()),
            "residual_std_ugm3": resid_std,
            "splithalf_mad_ugm3": float(dis.mean()),
            "splithalf_mad_over_resid_std": float(dis.mean() / resid_std),
            "splithalf_p95_ugm3": float(np.percentile(dis, 95)),
            "splithalf_corr": corr,
        }
        print(f"{cname:8s} slots={len(counts)} train count min/mean/max = "
              f"{counts.min()}/{counts.mean():.1f}/{counts.max()}  empty={int((counts == 0).sum())}")
        print(f"          split-half |c1-c2|: mean={dis.mean():.2f} ug/m3 "
              f"(= {dis.mean()/resid_std:.3f} x residual std {resid_std:.1f}), "
              f"p95={np.percentile(dis, 95):.2f}, corr={corr:.4f}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
