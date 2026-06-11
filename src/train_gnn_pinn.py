"""Variant E: GNN-PINN-TL — pre-train + fine-tune with a full unsteady advection–
diffusion–reaction (ADR) physics prior on the predicted PM2.5 field.

Physics is a *property of the base model*; transfer runs on top. This pilot composes the
ADR prior (reports/PINN_PHYSICS.md) with the cheapest, cleanest transfer recipe (Variant A,
PT-FT), so the marginal effect of the physics term is isolated against
`results/gnn_tl/variantA_gat_fixed.json`. Fine-tune loss per batch:

    L = MSE(ŷ, y)  +  λ_phys · L_phys(Ĉ'(t+Δt), C'^obs(t), measured-wind, graph)

The physics term is evaluated on the climatology-residual concentration in physical units
(µg/m³), uses the measured time-varying wind to build the advection operator, and a
mass-conserving graph Laplacian for diffusion. It shares information across nodes via the
transport operator — the "more value from fewer sensors" lever, expected to help most on
the sparsest target (Guwahati, |V|=4) and lowest label budget.

Run (pilot):  python -m src.train_gnn_pinn --fixed --pilot
Run (full):   python -m src.train_gnn_pinn --fixed --full
Sweep:        python -m src.train_gnn_pinn --fixed --pilot --lambda-phys 0.1
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from src.graph_construction import build_city_graph
from src.models.physics import PhysicsADR, build_diffusion_weights, build_geometry
from src.train_gnn import (
    DEVICE, GRAPH_STRATEGY, HISTORY, HORIZON, KNN_K,
    build_backbone, evaluate_gnn, load_all_cities, make_city_loader_masked,
)
from src.train_gnn_tl import FIXED_FT_CFG, FREEZE_FRAC, SRC_MODELS_DIR, WEIGHT_DECAY
from src.utils import (
    CityTensors, load_checkpoint, save_checkpoint, set_seed, write_results,
)

MODELS_DIR = Path("models/gnn_pinn")
RESULTS_DIR = Path("results/gnn_pinn")

LAMBDA_PHYS = 0.05            # gentle by default; swept on the pilot.
DT_HOURS = HORIZON * 3.0     # forecast step in hours (3-hour cadence).
# Feature channel order (src/data_pipeline.COMMON_FEATURES): PM2.5, AT, RH, WS, Sin_WD, Cos_WD, ...
WS_IDX, SIN_WD_IDX, COS_WD_IDX = 3, 4, 5


def make_windows_phys(
    feature_tensor: np.ndarray, target_tensor: np.ndarray, split_mask: np.ndarray,
    history: int, horizon: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Like make_windows_masked, but also returns Y_prev = target at the last history step
    (t+history-1) — the observed current residual C'^obs(t) for the forward-Euler ∂C/∂t."""
    T = feature_tensor.shape[0]
    xs, ys, yps = [], [], []
    for t in range(T - history - horizon + 1):
        pred_t = t + history + horizon - 1
        if not split_mask[pred_t]:
            continue
        xs.append(feature_tensor[t:t + history])
        ys.append(target_tensor[pred_t])
        yps.append(target_tensor[pred_t - horizon])     # = target at t+history-1
    if not xs:
        N, Fd = feature_tensor.shape[1], feature_tensor.shape[2]
        return (np.zeros((0, history, N, Fd), np.float32),
                np.zeros((0, N), np.float32), np.zeros((0, N), np.float32))
    return (np.stack(xs).astype(np.float32),
            np.stack(ys).astype(np.float32), np.stack(yps).astype(np.float32))


def _wind_from_batch(xb: torch.Tensor, ws_s, ws_m, sin_s, sin_m, cos_s, cos_m
                     ) -> Tuple[torch.Tensor, torch.Tensor]:
    """Reconstruct measured (wind_speed [B,N], wind_dir_deg [B,N]) at the last history step
    by inverse-z-scoring the WS / Sin_WD / Cos_WD feature channels."""
    last = xb[:, -1, :, :]                                  # [B, N, F]
    ws = last[:, :, WS_IDX] * ws_s + ws_m
    sin_wd = last[:, :, SIN_WD_IDX] * sin_s + sin_m
    cos_wd = last[:, :, COS_WD_IDX] * cos_s + cos_m
    wd_deg = torch.rad2deg(torch.atan2(sin_wd, cos_wd))     # WD (direction wind comes from)
    return ws, wd_deg


def transfer_variant_e(
    cities: Dict[str, CityTensors], source: str, target: str, d_pct: float,
    *, backbone: str = "gat", fixed: bool = True, lambda_phys: float = LAMBDA_PHYS,
) -> Dict[str, Dict[str, float]]:
    set_seed(42)
    src, tgt = cities[source], cities[target]
    F_dim = src.feature_tensor.shape[-1]

    edge_index, edge_weight = build_city_graph(tgt.coords, strategy=GRAPH_STRATEGY, k=KNN_K)
    bearing, dist = build_geometry(tgt.coords)
    W_diff = build_diffusion_weights(tgt.coords, k=KNN_K)
    bearing, dist, W_diff = bearing.to(DEVICE), dist.to(DEVICE), W_diff.to(DEVICE)

    # Scaler constants for reconstructing physical fields inside the loss.
    sy = float(tgt.target_scaler.scale_[0]); my = float(tgt.target_scaler.mean_[0])
    fs, fm = tgt.scaler.scale_, tgt.scaler.mean_
    wind_consts = (float(fs[WS_IDX]), float(fm[WS_IDX]), float(fs[SIN_WD_IDX]), float(fm[SIN_WD_IDX]),
                   float(fs[COS_WD_IDX]), float(fm[COS_WD_IDX]))

    print(f"\n[PINN-TL {source}->{target} d={d_pct:.0%}] target |V|={len(tgt.coords)} "
          f"|E|={edge_index.shape[1]}  lambda_phys={lambda_phys}")

    src_ckpt = SRC_MODELS_DIR / f"{backbone}_source_{source}{'_fixed' if fixed else ''}.pt"
    if not src_ckpt.exists():
        raise RuntimeError(f"missing source checkpoint: {src_ckpt}")

    X_full, Y_full, Yp_full = make_windows_phys(
        tgt.feature_tensor, tgt.target_tensor, tgt.train_mask, HISTORY, HORIZON)
    _, X_va, Y_va, C_va = make_city_loader_masked(tgt, "val", 64, False)
    _, X_te, Y_te, C_te = make_city_loader_masked(tgt, "test", 64, False)

    cfg = FIXED_FT_CFG[target]
    rng = np.random.default_rng(42)
    keep = rng.choice(X_full.shape[0], size=max(1, int(d_pct * X_full.shape[0])), replace=False)
    keep.sort()
    ds = torch.utils.data.TensorDataset(
        torch.from_numpy(X_full[keep]).float(), torch.from_numpy(Y_full[keep]).float(),
        torch.from_numpy(Yp_full[keep]).float())
    loader = torch.utils.data.DataLoader(ds, batch_size=cfg["batch_size"], shuffle=True)

    # 0. Zero-shot (source weights, no fine-tune) — identical to Variant A's zero-shot.
    zs_model = build_backbone(backbone, F_dim, fixed=fixed).to(DEVICE)
    load_checkpoint(zs_model, src_ckpt)
    zs = evaluate_gnn(zs_model, X_te, Y_te, edge_index, edge_weight, tgt.target_scaler, climatology=C_te)
    print(f"  zero-shot test = R2={zs['R2']:.4f} MAE={zs['MAE']:.3f}")

    # 1. Transfer with physics: source warm-start -> fine-tune with λ_phys·L_phys.
    model = build_backbone(backbone, F_dim, fixed=fixed).to(DEVICE)
    load_checkpoint(model, src_ckpt)
    phys = PhysicsADR(dt_hours=DT_HOURS).to(DEVICE)
    params = list(model.parameters()) + list(phys.parameters())
    opt = torch.optim.Adam(params, lr=cfg["lr"], weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=3)
    mse = nn.MSELoss()
    freeze_for = int(cfg["epochs"] * FREEZE_FRAC)
    if freeze_for > 0:
        for p in model.t1.parameters():
            p.requires_grad = False

    tl_ckpt = MODELS_DIR / f"{backbone}_pinn_{source}_to_{target}_d{int(d_pct*100)}{'_fixed' if fixed else ''}.pt"
    set_seed(42)
    best_val, patience_left, last_phys = -1e9, cfg["patience"], 0.0
    for ep in range(cfg["epochs"]):
        if freeze_for > 0 and ep == freeze_for:
            for p in model.t1.parameters():
                p.requires_grad = True
        model.train()
        ep_phys, nb = 0.0, 0
        for xb, yb, ypb in loader:
            opt.zero_grad()
            pred = model(xb, edge_index, edge_weight)               # ŷ [B,N] (z-scored residual)
            C_pred = sy * pred + my                                 # µg/m³ residual
            C_obs = sy * ypb + my
            ws, wd_deg = _wind_from_batch(xb, *wind_consts)
            loss_phys = phys(C_pred, C_obs, ws, wd_deg, bearing, dist, W_diff, scale=sy)
            loss = mse(pred, yb) + lambda_phys * loss_phys
            if not torch.isfinite(loss):
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            ep_phys += float(loss_phys.detach()); nb += 1
        last_phys = ep_phys / max(nb, 1)
        vm = evaluate_gnn(model, X_va, Y_va, edge_index, edge_weight, tgt.target_scaler, climatology=C_va)
        sched.step(vm["R2"])
        if vm["R2"] > best_val + 1e-4:
            best_val = vm["R2"]; save_checkpoint(model, tl_ckpt, extra={"val_r2": best_val})
            patience_left = cfg["patience"]
        else:
            patience_left -= 1
        print(f"    ft ep {ep+1:02d}/{cfg['epochs']} val_R2={vm['R2']:+.4f} best={best_val:+.4f} L_phys={last_phys:.3f}")
        if patience_left is not None and patience_left <= 0:
            break

    load_checkpoint(model, tl_ckpt)
    tl = evaluate_gnn(model, X_te, Y_te, edge_index, edge_weight, tgt.target_scaler, climatology=C_te)
    coeffs = phys.coeffs()
    print(f"  ==> PINN transfer test = R2={tl['R2']:.4f} MAE={tl['MAE']:.3f} "
          f"(zero_shot={zs['R2']:.4f}) | L_phys={last_phys:.3f} "
          f"alpha={coeffs['alpha']:.3f} K={coeffs['K']:.3f} lambda={coeffs['lambda']:.3f}")

    return {"zero_shot": zs, "transfer": tl, "real_transfer": tl["R2"] > zs["R2"],
            "lambda_phys": lambda_phys, "L_phys_last": last_phys, "coeffs": coeffs}


def main(args):
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    cities = load_all_cities(fixed=args.fixed)
    if args.full:
        pairs = [("Delhi", "Kolkata"), ("Delhi", "Guwahati"), ("Kolkata", "Delhi"),
                 ("Kolkata", "Guwahati"), ("Guwahati", "Delhi"), ("Guwahati", "Kolkata")]
        d_values = [0.15, 0.30, 0.45, 0.60]
        out_name = f"variantE_pinn_{args.backbone}{'_fixed' if args.fixed else ''}.json"
    else:
        pairs = [("Delhi", "Kolkata"), ("Delhi", "Guwahati"), ("Kolkata", "Guwahati")]
        d_values = [0.30]
        out_name = f"variantE_pinn_{args.backbone}_pilot{'_fixed' if args.fixed else ''}.json"

    print("=" * 68)
    print(f"GNN-PINN-TL (Variant E, ADR)  backbone={args.backbone}  fixed={args.fixed}  "
          f"lambda_phys={args.lambda_phys}  dt={DT_HOURS}h  cells={len(pairs)*len(d_values)}")
    print("=" * 68)

    results: Dict[str, Dict] = {}
    for s, t in pairs:
        for d in d_values:
            results[f"{s}->{t}@{int(d*100)}"] = transfer_variant_e(
                cities, s, t, d, backbone=args.backbone, fixed=args.fixed, lambda_phys=args.lambda_phys)
            write_results(RESULTS_DIR / out_name, results)
    print(f"\nwrote {RESULTS_DIR / out_name}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--backbone", choices=["gat", "sage"], default="gat")
    p.add_argument("--fixed", action="store_true")
    p.add_argument("--pilot", action="store_true", help="3 pairs x d=0.30 (default if --full absent).")
    p.add_argument("--full", action="store_true", help="6 pairs x 4 d-fractions (24 cells).")
    p.add_argument("--lambda-phys", type=float, default=LAMBDA_PHYS)
    main(p.parse_args())
