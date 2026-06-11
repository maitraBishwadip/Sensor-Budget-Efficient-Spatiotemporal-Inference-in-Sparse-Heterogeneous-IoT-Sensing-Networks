"""Virtual sensing: estimate PM2.5 at UNSENSORED locations from a sparse sensor set.

The genuine "more value from fewer sensors" test (the forecast-at-retained-sensors regime is
data-determined and a null — see analysis/sensor_budget.py + project memory). Here the task is
spatial estimation at locations with NO PM2.5 sensor:

  - Split a city's N nodes into k SENSORED (S) and N-k UNSENSORED (U), fixed split.
  - Meteorology (AT/RH/WS/WD) is available at ALL nodes (realistic: reanalysis); only PM2.5 is
    sensor-specific, so the PM2.5 input channel is MASKED at U.
  - Masked-reconstruction training: each step also mask PM2.5 at a random subset R⊂S and
    supervise the model to reconstruct it — this teaches spatial inference from masked context.
  - Physics = SpatialPhysics (steady-state advection+diffusion on the predicted field), which
    propagates concentration sensored→unsensored along wind+diffusion paths.
  - EVALUATE at U only (held-out unsensored truth). Compare λ_phys=0 (data-only) vs λ_phys>0.

Hypothesis: physics helps here (unlike the retained-sensor regime) because U is under-determined.

Run:  python -u -m src.train_gnn_vsense --fixed
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn

from src.graph_construction import build_city_graph
from src.models.physics import SpatialPhysics, build_diffusion_weights, build_geometry
from src.train_gnn import (
    DEVICE, GRAPH_STRATEGY, HISTORY, HORIZON, KNN_K,
    build_backbone, load_all_cities, make_city_loader_masked,
)
from src.train_gnn_pinn import WS_IDX, SIN_WD_IDX, COS_WD_IDX, _wind_from_batch
from src.train_gnn_tl import WEIGHT_DECAY
from src.utils import (
    CityTensors, all_metrics, inverse_transform_target, make_windows_masked,
    save_checkpoint, load_checkpoint, set_seed, write_results,
)

MODELS_DIR = Path("models/gnn_vsense")
RESULTS_DIR = Path("results/gnn_vsense")
EPOCHS, LR, BATCH, PATIENCE = 40, 1e-3, 128, 8
P_MASK = 0.5                      # fraction of S also masked each step (reconstruction signal)


def _eval_at_nodes(model, X, Y, C, ei, ew, scaler, eval_nodes, mask_nodes, device) -> Dict[str, float]:
    """Mask PM2.5 input at `mask_nodes`, predict, score (raw µg/m³) at `eval_nodes`."""
    Xt = torch.from_numpy(X).float().to(device).clone()
    Xt[:, :, mask_nodes, 0] = 0.0
    model.eval()
    with torch.no_grad():
        pred = model(Xt, ei, ew).cpu().numpy()
    C_e = None if C is None else C[:, eval_nodes]
    pred_raw = inverse_transform_target(scaler, pred[:, eval_nodes], climatology=C_e)
    y_raw = inverse_transform_target(scaler, Y[:, eval_nodes], climatology=C_e)
    return all_metrics(y_raw, pred_raw)


def virtual_sense(city: CityTensors, k: int, lambda_phys: float, *, backbone="gat",
                  wind_consts=None, sy=1.0, my=0.0, seed_split=0,
                  phys_use_adv=True, phys_use_diff=True) -> Dict:
    set_seed(42)
    N = city.feature_tensor.shape[1]
    perm = np.random.default_rng(seed_split).permutation(N)
    S = np.sort(perm[:k]); U = np.sort(perm[k:])
    mode = "both" if (phys_use_adv and phys_use_diff) else ("adv" if phys_use_adv else "diff")

    ei, ew = build_city_graph(city.coords, strategy=GRAPH_STRATEGY, k=KNN_K)
    bearing, dist = build_geometry(city.coords)
    W_diff = build_diffusion_weights(city.coords, k=KNN_K)
    ei, ew, bearing, dist, W_diff = (t.to(DEVICE) for t in (ei, ew, bearing, dist, W_diff))

    X_tr, Y_tr, _ = make_windows_masked(city.feature_tensor, city.target_tensor, city.train_mask,
                                        HISTORY, HORIZON, climatology_tensor=city.climatology_tensor)
    _, X_va, Y_va, C_va = make_city_loader_masked(city, "val", 64, False)
    _, X_te, Y_te, C_te = make_city_loader_masked(city, "test", 64, False)

    ds = torch.utils.data.TensorDataset(torch.from_numpy(X_tr).float(), torch.from_numpy(Y_tr).float())
    loader = torch.utils.data.DataLoader(ds, batch_size=BATCH, shuffle=True)

    model = build_backbone(backbone, city.feature_tensor.shape[-1], fixed=True).to(DEVICE)
    phys = SpatialPhysics(use_adv=phys_use_adv, use_diff=phys_use_diff).to(DEVICE)
    params = list(model.parameters()) + list(phys.parameters())
    opt = torch.optim.Adam(params, lr=LR, weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=3)
    mse = nn.MSELoss()
    S_t = torch.as_tensor(S, device=DEVICE)
    nR = max(1, int(P_MASK * len(S)))

    ckpt = MODELS_DIR / f"{backbone}_vsense_{city.city}_k{k}_s{seed_split}_{mode}_lam{lambda_phys}.pt"
    best_val, patience_left = -1e9, PATIENCE
    rng = np.random.default_rng(123)
    for ep in range(EPOCHS):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            Ridx = S[rng.permutation(len(S))[:nR]]
            mask_cols = np.concatenate([U, Ridx])
            xbm = xb.clone(); xbm[:, :, mask_cols, 0] = 0.0
            opt.zero_grad()
            pred = model(xbm, ei, ew)                       # [B,N]
            loss_data = mse(pred[:, S_t], yb[:, S_t])
            ws, wd = _wind_from_batch(xb, *wind_consts)     # met available at all nodes
            loss_phys = phys(sy * pred + my, ws, wd, bearing, dist, W_diff, scale=sy)
            loss = loss_data + lambda_phys * loss_phys
            if not torch.isfinite(loss):
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
        vm = _eval_at_nodes(model, X_va, Y_va, C_va, ei, ew, city.target_scaler, U, U, DEVICE)
        sched.step(vm["R2"])
        if vm["R2"] > best_val + 1e-4:
            best_val = vm["R2"]; save_checkpoint(model, ckpt, extra={"val_r2": best_val}); patience_left = PATIENCE
        else:
            patience_left -= 1
        print(f"    ep {ep+1:02d}/{EPOCHS} U-val R2={vm['R2']:+.4f} best={best_val:+.4f}")
        if patience_left <= 0:
            break

    load_checkpoint(model, ckpt)
    te = _eval_at_nodes(model, X_te, Y_te, C_te, ei, ew, city.target_scaler, U, U, DEVICE)
    print(f"  [{city.city} k={k}/{N} lam={lambda_phys}] U-node test R2={te['R2']:.4f} "
          f"MAE={te['MAE']:.3f}  (val R2={best_val:.4f}, coeffs={phys.coeffs()})")
    return {"R2": te["R2"], "MAE": te["MAE"], "val_R2": best_val, "k": k, "N": int(N),
            "lambda": lambda_phys, "n_unsensored": int(len(U)), "coeffs": phys.coeffs()}


def main(args):
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    cities = load_all_cities(fixed=args.fixed)
    grid = {"Delhi": [8, 16], "Kolkata": [4]}
    lambdas = [0.0, 0.2, 0.5]
    results = {}
    for cname, ks in grid.items():
        c = cities[cname]
        fs, fm = c.scaler.scale_, c.scaler.mean_
        wind_consts = (float(fs[WS_IDX]), float(fm[WS_IDX]), float(fs[SIN_WD_IDX]), float(fm[SIN_WD_IDX]),
                       float(fs[COS_WD_IDX]), float(fm[COS_WD_IDX]))
        sy, my = float(c.target_scaler.scale_[0]), float(c.target_scaler.mean_[0])
        for k in ks:
            for lam in lambdas:
                print(f"\n##### Virtual sensing {cname} k={k} lambda={lam} #####")
                results[f"{cname}|k={k}|lam={lam}"] = virtual_sense(
                    c, k, lam, backbone=args.backbone, wind_consts=wind_consts, sy=sy, my=my)
                write_results(RESULTS_DIR / f"vsense_{args.backbone}.json", results)

    print("\n" + "=" * 64)
    print("VIRTUAL-SENSING SUMMARY - R2 at UNSENSORED nodes; dR2 = physics minus data-only")
    print("=" * 64)
    for cname, ks in grid.items():
        for k in ks:
            r0 = results[f"{cname}|k={k}|lam=0.0"]["R2"]
            line = f"  {cname:8s} k={k:<2d} (predict {results[f'{cname}|k={k}|lam=0.0']['n_unsensored']} unsensored): R2(lam=0)={r0:.4f}"
            for lam in lambdas:
                if lam == 0:
                    continue
                rl = results[f"{cname}|k={k}|lam={lam}"]["R2"]
                line += f"   dR2(lam={lam})={rl - r0:+.4f}"
            print(line)
    print(f"\nwrote {RESULTS_DIR / f'vsense_{args.backbone}.json'}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--backbone", choices=["gat", "sage"], default="gat")
    p.add_argument("--fixed", action="store_true")
    main(p.parse_args())
