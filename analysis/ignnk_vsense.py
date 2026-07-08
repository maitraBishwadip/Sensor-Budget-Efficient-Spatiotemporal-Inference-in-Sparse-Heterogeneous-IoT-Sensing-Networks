"""Learned-kriging baseline (IGNNK) for virtual sensing (reviewer issue C4/R2-M4).

Places IGNNK (Wu et al., AAAI 2021) on the same virtual-sensing cells as Table III:
node splits IDENTICAL to src/train_gnn_vsense.py (np.random.default_rng(0).permutation(N),
sensored = first k), the same masked-reconstruction protocol as the ST-GNN (PM2.5 input
zeroed at the fixed unsensored set U plus a random half of S each step, loss at the
sensored nodes only), checkpoint selection on U-val R2, and evaluation on exactly the
prediction timesteps of the GNN's test windows, in raw ug/m3 space.

Two information regimes are reported, matching the kernel-interpolation ladder:
  - concurrent (t):   input window ends at the evaluation step (the nowcast regime of
                      kernel interp.(t)); the reconstruction at the window's last step
                      is the estimate.
  - lagged (t-1):     input window ends HORIZON steps earlier (exactly the information
                      the ST-GNN's window ends with); the reconstruction at that step
                      is persisted to the evaluation step, as kernel interp.(t-1) does.

Leakage control: training windows are anchored on train-owned timesteps and the
reconstruction loss is restricted to window steps owned by the training split, so no
validation/test-owned value is ever supervised.

Writes results/gnn_vsense/ignnk_vsense.json.

Run:  python -u -m analysis.ignnk_vsense
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.graph_construction import pairwise_distance_matrix        # noqa: E402
from src.models.ignnk import IGNNK, random_walk_supports           # noqa: E402
from src.train_gnn import HISTORY, HORIZON, load_all_cities        # noqa: E402
from src.utils import all_metrics, load_checkpoint, save_checkpoint, set_seed  # noqa: E402

OUT = Path("results/gnn_vsense/ignnk_vsense.json")
MODELS_DIR = Path("models/ignnk")
GRID = {"Delhi": [8, 16], "Kolkata": [4]}
SIGMA_KM = 5.0
HIDDEN = 100
ORDERS = 2
P_MASK = 0.5                     # fraction of S also masked each step (as the ST-GNN)
EPOCHS = 150
PATIENCE = 15
LR = 1e-3
BATCH = 32
DEVICE = "cpu"


def _windows(signal: np.ndarray, anchors: np.ndarray) -> np.ndarray:
    """Gather [len(anchors), N, h] windows of `signal` [T, N] ending at each anchor."""
    idx = anchors[:, None] - (HISTORY - 1) + np.arange(HISTORY)[None, :]   # [B, h]
    return signal[idx].transpose(0, 2, 1)                                  # [B, N, h]


def _reconstruct_at(model, signal, anchors, U, A_f, A_b) -> np.ndarray:
    """Masked forward pass; return the window-end reconstruction at U: [len(anchors), |U|]."""
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(anchors), 256):
            xb = _windows(signal, anchors[i:i + 256]).copy()
            xb[:, U, :] = 0.0
            pred = model(torch.from_numpy(xb).float(), A_f, A_b)
            out.append(pred[:, U, -1].numpy())
    return np.concatenate(out, axis=0)


def run_cell(city, k: int) -> dict:
    set_seed(42)
    T, N = city.target_tensor.shape
    signal = city.target_tensor.astype(np.float32)                 # z-scored residual [T, N]
    perm = np.random.default_rng(0).permutation(N)
    S = np.sort(perm[:k]); U = np.sort(perm[k:])

    D = pairwise_distance_matrix(city.coords).astype(np.float64)
    W = np.exp(-(D ** 2) / (2 * SIGMA_KM ** 2))
    np.fill_diagonal(W, 0.0)
    A_f, A_b = random_walk_supports(W)

    first = HISTORY - 1
    tr_anchors = np.where(city.train_mask)[0]; tr_anchors = tr_anchors[tr_anchors >= first]
    va_anchors = np.where(city.val_mask)[0];   va_anchors = va_anchors[va_anchors >= first]

    # Raw PM2.5 for scoring (inverse z-score + climatology add-back), as idw_vsense.py.
    raw = city.target_scaler.inverse_transform(
        signal.reshape(-1, 1)).reshape(T, N).astype(np.float64)
    if city.climatology_tensor is not None:
        raw = raw + city.climatology_tensor
    train_owned = city.train_mask.astype(bool)

    model = IGNNK(history=HISTORY, hidden_dim=HIDDEN, orders=ORDERS).to(DEVICE)
    n_par = sum(p.numel() for p in model.parameters())
    print(f"\n[IGNNK {city.city} k={k}]  |S|={k} |U|={len(U)}  params={n_par}  "
          f"train_anchors={len(tr_anchors)}  val_anchors={len(va_anchors)}")
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    mse = torch.nn.MSELoss(reduction="none")
    rng = np.random.default_rng(42)
    nR = max(1, int(P_MASK * len(S)))
    ckpt = MODELS_DIR / f"ignnk_{city.city}_k{k}_s0.pt"

    def val_r2() -> float:
        pred = _reconstruct_at(model, signal, va_anchors, U, A_f, A_b)
        true = signal[np.ix_(va_anchors, U)]
        ss_res = float(((true - pred) ** 2).sum())
        ss_tot = float(((true - true.mean()) ** 2).sum())
        return 1.0 - ss_res / max(ss_tot, 1e-12)

    best_val, patience_left = -1e9, PATIENCE
    for ep in range(EPOCHS):
        model.train()
        order = rng.permutation(len(tr_anchors))
        for i in range(0, len(order), BATCH):
            anchors = tr_anchors[order[i:i + BATCH]]
            xb = _windows(signal, anchors)                          # [B, N, h]
            yb = torch.from_numpy(xb.copy()).float()
            Ridx = S[rng.permutation(len(S))[:nR]]
            xbm = xb.copy(); xbm[:, np.concatenate([U, Ridx]), :] = 0.0
            pred = model(torch.from_numpy(xbm).float(), A_f, A_b)
            # loss at sensored nodes, restricted to train-owned window steps
            step_t = anchors[:, None] - (HISTORY - 1) + np.arange(HISTORY)[None, :]
            owned = torch.from_numpy(train_owned[step_t]).float()   # [B, h]
            err = mse(pred[:, S, :], yb[:, S, :])                   # [B, |S|, h]
            w = owned[:, None, :].expand_as(err)
            loss = (err * w).sum() / w.sum().clamp(min=1.0)
            opt.zero_grad(); loss.backward(); opt.step()
        vr = val_r2()
        star = " "
        if vr > best_val + 1e-4:
            best_val = vr; save_checkpoint(model, ckpt, extra={"val_r2": best_val})
            patience_left = PATIENCE; star = "*"
        else:
            patience_left -= 1
        print(f"  ep {ep+1:03d}/{EPOCHS} {star} U-val R2(z)={vr:+.4f} best={best_val:+.4f}")
        if patience_left <= 0:
            print(f"  early-stopped at epoch {ep+1} (best U-val R2={best_val:.4f})")
            break

    load_checkpoint(model, ckpt)
    first_pred = HISTORY + HORIZON - 1
    eval_t = np.where(city.test_mask)[0]
    eval_t = eval_t[eval_t >= first_pred]
    y_true = raw[np.ix_(eval_t, U)]

    res = {"k": int(k), "N": int(N), "n_unsensored": int(len(U)),
           "params": int(n_par), "val_R2_z": float(best_val),
           "n_eval_timesteps": int(len(eval_t))}
    for regime, anchors in (("t", eval_t), ("t_lag", eval_t - HORIZON)):
        pred_z = _reconstruct_at(model, signal, anchors, U, A_f, A_b)
        pred_raw = city.target_scaler.inverse_transform(
            pred_z.reshape(-1, 1)).reshape(pred_z.shape).astype(np.float64)
        if city.climatology_tensor is not None:
            # climatology of the *evaluation* step: lagged reconstructions are persisted
            # forward as estimates for eval_t, exactly as kernel interp.(t-1) persists.
            pred_raw = pred_raw + city.climatology_tensor[np.ix_(eval_t, U)]
        m = all_metrics(y_true, pred_raw)
        res[regime] = {kk: float(vv) for kk, vv in m.items()}
        print(f"[IGNNK {city.city} k={k}] {regime:5s}: R2={m['R2']:.4f} "
              f"MAE={m['MAE']:.3f} RMSE={m['RMSE']:.3f}")
    return res


def main():
    cities = load_all_cities(fixed=True)
    results = {}
    if OUT.exists():
        results = json.loads(OUT.read_text())
    for cname, ks in GRID.items():
        for k in ks:
            results[f"{cname}|k={k}"] = run_cell(cities[cname], k)
            OUT.parent.mkdir(parents=True, exist_ok=True)
            OUT.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
