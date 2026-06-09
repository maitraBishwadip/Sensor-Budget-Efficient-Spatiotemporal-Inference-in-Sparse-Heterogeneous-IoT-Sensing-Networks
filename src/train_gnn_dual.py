"""Variant C (dual TL): Dual marginal + conditional adversarial domain adaptation.

Extends Graph-DANN (Variant B) with a CDAN-style conditional discriminator
(Long et al., NeurIPS-18). The joint phase aligns BOTH:
  - the marginal embedding distribution P(z)              (as in Graph-DANN), and
  - the forecast-conditional distribution P(z | y_hat)    (the new dual term),
through a single gradient-reversal layer. This targets the regression-DA
limitation (de Mathelin et al., 2020) that marginal-only alignment does not
move accuracy, in the spirit of the marginal/conditional duality of Dual
Transfer Learning (Long et al., SDM-12).

Joint-phase loss per mini-batch:
    L = MSE(y_hat, y)  +  alpha_m * sum_c CE(D_marg)  +  alpha_c * sum_c CE(D_cond)

Reuses Variant B's protocol verbatim (warm-start, interleaved split,
climatology residual, per-city fine-tune, 3-way verification) so the dual
result is one-to-one comparable with PT-FT and Graph-DANN.

Run (pilot):  python -m src.train_gnn_dual --fixed --pilot
Run (full):   python -m src.train_gnn_dual --fixed
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torch.nn as nn

from src.models.dann import GraphDualDANN
from src.train_gnn import (
    DEVICE, build_backbone, evaluate_gnn, load_all_cities,
    make_city_loader, make_city_loader_masked,
)
from src.train_gnn_dann import (
    CITY_TO_LABEL, DANN_GAMMA, EPOCHS_JOINT, FIXED_FT_CFG, FREEZE_FRAC,
    JOINT_BATCH_PER_CITY, LR, SRC_MODELS_DIR, WEIGHT_DECAY,
    build_per_city_edges, build_per_city_train_arrays, lambda_progress,
    sample_minibatch,
)
from src.utils import (
    CityTensors, load_checkpoint, make_windows, make_windows_masked,
    save_checkpoint, set_seed, write_results,
)

MODELS_DIR = Path("models/gnn_dual")
RESULTS_DIR = Path("results/gnn_dual")

ALPHA_M = 0.10   # weight on the marginal discriminator (as Graph-DANN)
ALPHA_C = 0.05   # weight on the conditional discriminator (gentler auxiliary)
HISTORY, HORIZON = 8, 1


def train_graph_dual(
    cities: Dict[str, CityTensors],
    source: str,
    target: str,
    d_pct: float,
    *,
    backbone: str = "gat",
    fixed: bool = True,
    alpha_m: float = ALPHA_M,
    alpha_c: float = ALPHA_C,
) -> Dict[str, Dict[str, float]]:
    set_seed(42)
    src, tgt = cities[source], cities[target]
    F_dim = src.feature_tensor.shape[-1]
    encoder = build_backbone(backbone, F_dim, fixed=fixed).to(DEVICE)

    src_suffix = "_fixed" if fixed else ""
    src_ckpt = SRC_MODELS_DIR / f"{backbone}_source_{source}{src_suffix}.pt"
    if not src_ckpt.exists():
        raise RuntimeError(f"missing source checkpoint {src_ckpt}")
    load_checkpoint(encoder, src_ckpt)

    model = GraphDualDANN(encoder, n_cities=3).to(DEVICE)
    edges = build_per_city_edges(cities)
    per_city = build_per_city_train_arrays(cities, fixed=fixed)
    indices = {n: np.arange(per_city[n][0].shape[0]) for n in per_city if per_city[n][0].shape[0] > 0}
    rng = np.random.default_rng(42)
    for a in indices.values():
        rng.shuffle(a)
    cursors = {n: 0 for n in indices}

    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    mse, ce = nn.MSELoss(), nn.CrossEntropyLoss()
    steps_per_epoch = max(1, per_city[source][0].shape[0] // JOINT_BATCH_PER_CITY)
    total_steps = EPOCHS_JOINT * steps_per_epoch
    gstep = 0

    if fixed:
        _, Xv_s, Yv_s, Cv_s = make_city_loader_masked(src, "val", 64, False)
    else:
        _, Xv_s, Yv_s = make_city_loader(src, src.train_end_idx, src.val_end_idx, 64, False); Cv_s = None
    best_val = -1e9
    joint_ckpt = MODELS_DIR / f"{backbone}_dual_joint_{source}_to_{target}{'_fixed' if fixed else ''}.pt"

    print(f"\n[Dual-DANN {source}->{target}@{int(d_pct*100)}] joint {EPOCHS_JOINT}ep x {steps_per_epoch} "
          f"(alpha_m={alpha_m}, alpha_c={alpha_c})")
    for ep in range(EPOCHS_JOINT):
        model.train()
        sY = sM = sC = 0.0
        for _ in range(steps_per_epoch):
            lam = lambda_progress(gstep, total_steps)
            mb, cursors = sample_minibatch(per_city, indices, cursors, JOINT_BATCH_PER_CITY, rng)
            loss_y = torch.tensor(0.0)
            loss_m = torch.tensor(0.0)
            loss_c = torch.tensor(0.0)
            for name in mb:
                xb, yb = mb[name]
                xb_t = torch.from_numpy(xb).float()
                ei, ew = edges[name]
                y_hat, m_logits, c_logits = model(xb_t, ei, ew, lambda_=lam)
                lab = torch.full((xb_t.size(0),), CITY_TO_LABEL[name], dtype=torch.long)
                loss_m = loss_m + ce(m_logits, lab)
                loss_c = loss_c + ce(c_logits, lab)
                if name == source:  # PM2.5 labels only on source
                    loss_y = loss_y + mse(y_hat, torch.from_numpy(yb).float())
            loss = loss_y + alpha_m * loss_m + alpha_c * loss_c
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            sY += float(loss_y.detach()); sM += float(loss_m.detach()); sC += float(loss_c.detach()); gstep += 1

        vm = evaluate_gnn(model.encoder, Xv_s, Yv_s, edges[source][0], edges[source][1],
                          src.target_scaler, climatology=Cv_s)
        if vm["R2"] > best_val + 1e-4:
            best_val = vm["R2"]; save_checkpoint(model, joint_ckpt, extra={"src_val_r2": best_val})
            flag = "*"
        else:
            flag = " "
        print(f"  ep {ep+1:02d}/{EPOCHS_JOINT} {flag} Ly={sY/steps_per_epoch:.4f} "
              f"Lm={sM/steps_per_epoch:.3f} Lc={sC/steps_per_epoch:.3f} lam={lam:.2f} src_val_R2={vm['R2']:+.4f}")

    load_checkpoint(model, joint_ckpt)

    ei_t, ew_t = edges[target]
    if fixed:
        _, Xv, Yv, Cv = make_city_loader_masked(tgt, "val", 64, False)
        _, Xte, Yte, Cte = make_city_loader_masked(tgt, "test", 64, False)
    else:
        _, Xv, Yv = make_city_loader(tgt, tgt.train_end_idx, tgt.val_end_idx, 64, False)
        _, Xte, Yte = make_city_loader(tgt, tgt.val_end_idx, tgt.test_end_idx, 64, False); Cv = Cte = None

    zs = evaluate_gnn(model.encoder, Xte, Yte, ei_t, ew_t, tgt.target_scaler, climatology=Cte)
    print(f"  zero-shot (dual encoder, no FT) test = R2={zs['R2']:.4f} MAE={zs['MAE']:.3f}")

    # ── target fine-tune (adversary off) ──
    cfg = FIXED_FT_CFG[target]
    freeze_for = int(cfg["epochs"] * FREEZE_FRAC)
    if fixed:
        Xf, Yf, _ = make_windows_masked(tgt.feature_tensor, tgt.target_tensor, tgt.train_mask,
                                        history=HISTORY, horizon=HORIZON, climatology_tensor=tgt.climatology_tensor)
    else:
        Xf, Yf = make_windows(tgt.feature_tensor, tgt.target_tensor, HISTORY, HORIZON, 0, tgt.train_end_idx)
    rng_ft = np.random.default_rng(42)
    keep = rng_ft.choice(Xf.shape[0], size=max(1, int(d_pct * Xf.shape[0])), replace=False); keep.sort()
    Xft, Yft = Xf[keep], Yf[keep]
    ds = torch.utils.data.TensorDataset(torch.from_numpy(Xft).float(), torch.from_numpy(Yft).float())
    loader = torch.utils.data.DataLoader(ds, batch_size=cfg["batch_size"], shuffle=True)
    ft_opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(ft_opt, mode="max", factor=0.5, patience=3)
    if freeze_for > 0:
        for p in model.encoder.t1.parameters():
            p.requires_grad = False
    best_val = -1e9; patience = cfg["patience"]
    tl_ckpt = MODELS_DIR / f"{backbone}_dual_{source}_to_{target}_d{int(d_pct*100)}{'_fixed' if fixed else ''}.pt"
    for ep in range(cfg["epochs"]):
        if freeze_for > 0 and ep == freeze_for:
            for p in model.encoder.t1.parameters():
                p.requires_grad = True
        model.train()
        for xb, yb in loader:
            ft_opt.zero_grad()
            y_hat, _, _ = model(xb, ei_t, ew_t, lambda_=0.0)
            loss = mse(y_hat, yb); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); ft_opt.step()
        vm = evaluate_gnn(model.encoder, Xv, Yv, ei_t, ew_t, tgt.target_scaler, climatology=Cv)
        sched.step(vm["R2"])
        if vm["R2"] > best_val + 1e-4:
            best_val = vm["R2"]; save_checkpoint(model, tl_ckpt, extra={"val_r2": best_val}); patience = cfg["patience"]
        else:
            patience -= 1
        if patience is not None and patience <= 0:
            break
    load_checkpoint(model, tl_ckpt)
    tl = evaluate_gnn(model.encoder, Xte, Yte, ei_t, ew_t, tgt.target_scaler, climatology=Cte)
    print(f"  ==> DUAL transfer test = R2={tl['R2']:.4f} MAE={tl['MAE']:.3f} (zero_shot={zs['R2']:.4f})")
    return {"zero_shot": zs, "transfer": tl, "real_transfer": tl["R2"] > zs["R2"],
            "alpha_m": alpha_m, "alpha_c": alpha_c}


def main(args):
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    cities = load_all_cities(fixed=args.fixed)

    if args.pilot:
        pairs = [("Delhi", "Kolkata"), ("Delhi", "Guwahati"), ("Kolkata", "Guwahati")]
        d_values = [0.30]
        out_name = f"variantC_dual_{args.backbone}_pilot{'_fixed' if args.fixed else ''}.json"
    else:
        pairs = [("Delhi", "Kolkata"), ("Delhi", "Guwahati"), ("Kolkata", "Delhi"),
                 ("Kolkata", "Guwahati"), ("Guwahati", "Delhi"), ("Guwahati", "Kolkata")]
        d_values = [0.15, 0.30, 0.45, 0.60]
        out_name = f"variantC_dual_{args.backbone}{'_fixed' if args.fixed else ''}.json"

    results: Dict[str, Dict] = {}
    for s, t in pairs:
        for d in d_values:
            results[f"{s}->{t}@{int(d*100)}"] = train_graph_dual(
                cities, s, t, d, backbone=args.backbone, fixed=args.fixed,
                alpha_m=args.alpha_m, alpha_c=args.alpha_c)
            write_results(RESULTS_DIR / out_name, results)
    print(f"\nwrote {RESULTS_DIR / out_name}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--backbone", choices=["gat", "sage"], default="gat")
    p.add_argument("--fixed", action="store_true")
    p.add_argument("--pilot", action="store_true", help="3 priority cells @ d=30%% only")
    p.add_argument("--alpha-m", type=float, default=ALPHA_M)
    p.add_argument("--alpha-c", type=float, default=ALPHA_C)
    main(p.parse_args())
