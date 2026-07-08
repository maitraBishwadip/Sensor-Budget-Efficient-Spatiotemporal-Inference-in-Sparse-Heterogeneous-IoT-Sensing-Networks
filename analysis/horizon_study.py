"""Multi-horizon sensitivity study: t+4 (12 h) versus the paper's t+1 (3 h).

Reviewer request: the paper's single 3-h horizon is reactive for municipal
alerting; show how the levers behave at a 12-h lookahead. Nothing in the
inductive encoder ties it to one step — the horizon only shifts the
supervision target — so this study retrains the identical architectures with
HORIZON = 4 (direct multi-step) and reports:

  1. Source-only forecasting on Delhi and Kolkata (LSTM and GAT-GNN), i.e.
     whether Negative 1 (the graph does not pay for plain forecasting) is
     horizon-specific.
  2. Cold-start transfer Delhi -> Kolkata at d = 30% with the full three-way
     verification (zero-shot / scratch / transfer), for both families.
  3. Virtual sensing at Delhi k = 16 and Kolkata k = 4 (lambda = 0), i.e.
     coverage of unsensored locations at the longer horizon.

All checkpoints/results are tagged _h4; the h = 1 artifacts are untouched.

Writes results/horizon/horizon_h4.json.

Run:  python -u -m analysis.horizon_study
"""
from __future__ import annotations

import os
import sys

os.environ["PM25_HORIZON"] = os.environ.get("PM25_HORIZON", "4")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json                                                        # noqa: E402
from pathlib import Path                                           # noqa: E402

from src import train_lstm                                         # noqa: E402
from src.train_gnn import HORIZON, load_all_cities, train_base_gnn # noqa: E402
from src.train_gnn_pinn import WS_IDX, SIN_WD_IDX, COS_WD_IDX      # noqa: E402
from src.train_gnn_tl import transfer_variant_a                    # noqa: E402
from src.train_gnn_vsense import virtual_sense                     # noqa: E402
from src.utils import write_results                                # noqa: E402

TAG = f"_h{HORIZON}"
OUT = Path(f"results/horizon/horizon{TAG}.json")


def main():
    print(f"### Multi-horizon study: HORIZON={HORIZON} steps ({HORIZON * 3} h) ###")
    assert train_lstm.HORIZON == HORIZON, "train_lstm did not pick up PM25_HORIZON"
    cities = load_all_cities(fixed=True)
    results = {"horizon_steps": HORIZON, "horizon_hours": HORIZON * 3}

    # ---- 1. Source-only forecasting (Negative-1 check at 12 h) ----
    for cname in ["Delhi", "Kolkata"]:
        m, _ = train_base_gnn(cities[cname], backbone="gat", fixed=True, ckpt_extra=TAG)
        results[f"gnn_source|{cname}"] = m
        write_results(OUT, results)
        results[f"lstm_source|{cname}"] = train_lstm.train_source_only(
            cities, cname, fixed=True, ckpt_extra=TAG)
        write_results(OUT, results)

    # ---- 2. Cold-start transfer with three-way verification ----
    results["gnn_tl|Delhi->Kolkata@30"] = transfer_variant_a(
        cities, "Delhi", "Kolkata", 0.30, backbone="gat", fixed=True,
        src_ckpt_extra=TAG)
    write_results(OUT, results)
    results["lstm_tl|Delhi->Kolkata@30"] = train_lstm.train_transfer(
        cities, "Delhi", "Kolkata", 0.30, fixed=True, ckpt_extra=TAG)
    write_results(OUT, results)

    # ---- 3. Virtual sensing at the longer horizon ----
    for cname, k in [("Delhi", 16), ("Kolkata", 4)]:
        c = cities[cname]
        fs, fm = c.scaler.scale_, c.scaler.mean_
        wind_consts = (float(fs[WS_IDX]), float(fm[WS_IDX]),
                       float(fs[SIN_WD_IDX]), float(fm[SIN_WD_IDX]),
                       float(fs[COS_WD_IDX]), float(fm[COS_WD_IDX]))
        sy, my = float(c.target_scaler.scale_[0]), float(c.target_scaler.mean_[0])
        results[f"gnn_vsense|{cname}|k={k}"] = virtual_sense(
            c, k, 0.0, backbone="gat", wind_consts=wind_consts, sy=sy, my=my,
            ckpt_extra=TAG)
        write_results(OUT, results)

    print("\n" + "=" * 64)
    print(f"HORIZON {HORIZON * 3}h SUMMARY (test R2)")
    print("=" * 64)
    for key, v in results.items():
        if isinstance(v, dict) and "R2" in v:
            print(f"  {key:32s} R2={v['R2']:+.4f} MAE={v['MAE']:.2f}")
        elif isinstance(v, dict) and "transfer" in v:
            print(f"  {key:32s} ZS={v['zero_shot']['R2']:+.3f} "
                  f"SC={v['scratch']['R2']:+.3f} TL={v['transfer']['R2']:+.3f} "
                  f"real={v['real_transfer']}")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
