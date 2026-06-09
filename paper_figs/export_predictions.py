"""Export REAL predicted-vs-observed PM2.5 for the hero transfer cell so the
forecast-fidelity figure (Fig. 6) uses genuine model output, not a mock-up.

Hero cell: Delhi -> Kolkata, Stage-1 PT-FT, d = 30 % (fixed protocol).
Verifies that the reproduced aggregate R2 matches the published value
(~0.8158, RESULTS.md S3.1.2) before saving arrays.

Run:  python paper_figs/export_predictions.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.graph_construction import build_city_graph                       # noqa: E402
from src.train_gnn import (                                               # noqa: E402
    GRAPH_STRATEGY, KNN_K, HISTORY, HORIZON,
    build_backbone, load_all_cities, make_city_loader_masked,
)
from src.utils import all_metrics, inverse_transform_target, load_checkpoint  # noqa: E402

OUT = ROOT / "paper_figs" / "_assets"
OUT.mkdir(parents=True, exist_ok=True)


def export_cell(target: str, ckpt: str, tag: str, expect_r2: float | None = None):
    cities = load_all_cities(fixed=True)
    city = cities[target]
    F = city.feature_tensor.shape[-1]

    model = build_backbone("gat", F, fixed=True)
    load_checkpoint(model, ROOT / ckpt)
    model.eval()

    edge_index, edge_weight = build_city_graph(city.coords, strategy=GRAPH_STRATEGY, k=KNN_K)
    _, X_te, Y_te, C_te = make_city_loader_masked(city, "test", 64, shuffle=False)

    preds = []
    with torch.no_grad():
        for i in range(0, X_te.shape[0], 64):
            xb = torch.from_numpy(X_te[i:i + 64]).float()
            preds.append(model(xb, edge_index, edge_weight).cpu().numpy())
    y_pred_norm = np.concatenate(preds, axis=0)               # [W, N] z-scored residual

    pred = inverse_transform_target(city.target_scaler, y_pred_norm, C_te)   # [W, N] raw ug/m3
    true = inverse_transform_target(city.target_scaler, Y_te,        C_te)   # [W, N] raw ug/m3

    m = all_metrics(true.reshape(-1), pred.reshape(-1))
    print(f"[{tag}] reproduced  R2={m['R2']:.4f}  MAE={m['MAE']:.3f}  "
          f"RMSE={m['RMSE']:.3f}  MAPE={m['MAPE']:.2f}  (W={pred.shape[0]}, N={pred.shape[1]})")
    if expect_r2 is not None:
        ok = abs(m["R2"] - expect_r2) < 0.01
        print(f"     match published R2={expect_r2}: {'OK' if ok else 'MISMATCH'}")

    # Prediction-step timestamps, in the same order make_windows_masked emits.
    T = city.feature_tensor.shape[0]
    pred_idx = [t + HISTORY + HORIZON - 1
                for t in range(T - HISTORY - HORIZON + 1)
                if city.test_mask[t + HISTORY + HORIZON - 1]]
    ts = city.timestamps[pred_idx].astype("datetime64[ns]").astype(str)
    assert len(ts) == pred.shape[0], (len(ts), pred.shape[0])

    np.savez(
        OUT / f"hero_{tag}.npz",
        pred=pred.astype(np.float32), true=true.astype(np.float32),
        stations=np.array(city.stations), timestamps=np.array(ts),
        r2=m["R2"], mae=m["MAE"], rmse=m["RMSE"], mape=m["MAPE"],
    )
    print(f"     saved -> {OUT / f'hero_{tag}.npz'}")


if __name__ == "__main__":
    export_cell(
        target="Kolkata",
        ckpt="models/gnn_tl/gat_tl_Delhi_to_Kolkata_d30_fixed.pt",
        tag="delhi_to_kolkata_d30",
        expect_r2=0.8158,
    )
