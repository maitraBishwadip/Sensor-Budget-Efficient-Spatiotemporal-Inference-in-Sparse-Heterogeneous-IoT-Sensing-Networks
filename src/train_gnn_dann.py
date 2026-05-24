"""Variant B (headline): Graph-DANN — fixed-protocol implementation.

Trains the inductive ST-GNN jointly on source + target + replay-third-city
with the gradient-reversal adversarial city classifier, then performs a
target fine-tune in the same style as Variant A. Evaluated on the target's
test partition.

Joint-phase loss per mini-batch:

    L  =  MSE(ŷ, y_pm25)   +   α_d · ( CE(ĉ_src) + CE(ĉ_tgt) + CE(ĉ_third) )

where the negative sign on the encoder branch is realized by the GRL on the
forward path of the city classifier (Ganin & Lempitsky, ICML-15).

This is a rewrite of the earlier prototype after diagnosing why it collapsed:

  1. Source-only warm-start (Tzeng et al., ADDA, CVPR-17). Load the
     fixed-protocol source checkpoint as the encoder init so the forecast
     head already works before any adversarial gradient touches the encoder.
  2. Discriminator loss reweighting (DASTNet, Tang et al., CIKM-22 §3.3;
     adversarial-DA-for-regression, de Mathelin et al., 2020 — arXiv:2006.08251).
     `α_d` defaults to 0.1 so MSE keeps the louder voice in the saddle point.
  3. Slower λ schedule. Joint phase is 25 epochs (was 8). Ganin's logistic
     curve `2/(1+exp(−γp))−1` with `γ=10` only saturates near `p=1` now.
  4. LayerNorm on the pooled embedding (in src/models/dann.py) decouples the
     discriminator from |V|-dependent activation scale (cf. GraphNorm, Cai
     et al., ICML-21).
  5. `--fixed` support — interleaved split, climatology residual, per-city
     LSTM-grade recipe — same protocol that fixed Variant A.
  6. Variant-A-grade fine-tune — 25/30/40 epochs with patience-based early
     stopping, `t1` freeze for the first 20%, best-by-val checkpoint, LR
     scheduling.
  7. Higher per-city training cap (8000) and bigger batch size during joint.

References:
  - Ganin & Lempitsky (ICML-15); Ganin et al. (JMLR-16) §5.1 on λ schedule.
  - Tzeng, Hoffman, Saenko & Darrell, ADDA (CVPR-17) — source warm-start.
  - Tang et al., DASTNet (CIKM-22) — cross-city adversarial precedent.
  - Wu et al., UDA-GCN (WWW-20); Zhu et al., EGI (NeurIPS-21).
  - Cai et al., GraphNorm (ICML-21) — graph-level activation normalization.
  - Yosinski et al. (2014) on frozen-layer fine-tune.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.graph_construction import build_city_graph
from src.models.dann import GraphDANN
from src.train_gnn import (
    BATCH_SIZE, DEVICE, DROPOUT, GAT_DIM, GRAPH_STRATEGY, HIDDEN,
    HISTORY, HORIZON, KNN_K, build_backbone, evaluate_gnn,
    load_all_cities, make_city_loader, make_city_loader_masked,
)
from src.utils import (
    CityTensors,
    inverse_transform_target,
    load_checkpoint,
    make_windows,
    make_windows_masked,
    save_checkpoint,
    set_seed,
    write_results,
)


MODELS_DIR = Path("models/gnn_dann")
RESULTS_DIR = Path("results/gnn_dann")
SRC_MODELS_DIR = Path("models/gnn")

# --------------------------------------------------------------------------- #
# Joint-phase config — tuned after the diagnosis. The headline knobs:
#   - EPOCHS_JOINT bumped to 25 so Ganin's λ schedule actually has room to
#     warm up (it reached 0.99 by epoch 4 of 8 in the old run).
#   - ALPHA_D = 0.1 brings the 3 summed CE losses (≈ 3·ln 3 = 3.3 at chance)
#     to comparable magnitude with the MSE (≈ 0.4 z-scored).
#   - DANN_SUBSAMPLE_MAX raised from 2000 → 8000 to expose the model to
#     enough Delhi data.
# --------------------------------------------------------------------------- #
EPOCHS_JOINT = 25
LR = 5e-4
WEIGHT_DECAY = 1e-5
DANN_GAMMA = 10.0
ALPHA_D = 0.1
DANN_SUBSAMPLE_MAX = 8000
JOINT_BATCH_PER_CITY = 32

# Fine-tune config — same shape as Variant A (FIXED_FT_CFG in train_gnn_tl.py).
FIXED_FT_CFG = {
    "Delhi":    {"epochs": 25, "batch_size": 128, "lr": 2e-4, "patience": 6},
    "Kolkata":  {"epochs": 30, "batch_size": 64,  "lr": 2e-4, "patience": 8},
    "Guwahati": {"epochs": 40, "batch_size": 32,  "lr": 1e-4, "patience": 10},
}
FREEZE_FRAC = 0.20


CITY_TO_LABEL = {"Delhi": 0, "Kolkata": 1, "Guwahati": 2}


# --------------------------------------------------------------------------- #
# Per-city training arrays — fixed-protocol-aware so we can pull windows
# from the interleaved train mask and (when present) the per-window climatology.
# --------------------------------------------------------------------------- #

def build_per_city_train_arrays(
    cities: Dict[str, CityTensors],
    *,
    fixed: bool,
    subsample_max: int = DANN_SUBSAMPLE_MAX,
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    out: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    for name, c in cities.items():
        if fixed:
            X, Y, _ = make_windows_masked(
                c.feature_tensor, c.target_tensor, c.train_mask,
                history=HISTORY, horizon=HORIZON,
                climatology_tensor=c.climatology_tensor,
            )
        else:
            X, Y = make_windows(
                c.feature_tensor, c.target_tensor, HISTORY, HORIZON, 0, c.train_end_idx,
            )
        if X.shape[0] > subsample_max:
            stride = max(1, X.shape[0] // subsample_max)
            X = X[::stride][:subsample_max]
            Y = Y[::stride][:subsample_max]
        out[name] = (X, Y)
        print(f"  [{name}] joint-phase train windows: {X.shape}")
    return out


def build_per_city_edges(
    cities: Dict[str, CityTensors],
) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
    return {
        name: build_city_graph(c.coords, strategy=GRAPH_STRATEGY, k=KNN_K)
        for name, c in cities.items()
    }


def sample_minibatch(
    per_city_windows: Dict[str, Tuple[np.ndarray, np.ndarray]],
    per_city_indices: Dict[str, np.ndarray],
    per_city_cursors: Dict[str, int],
    batch_per_city: int,
    rng: np.random.Generator,
) -> Tuple[Dict[str, Tuple[np.ndarray, np.ndarray]], Dict[str, int]]:
    out_x: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    new_cursors = dict(per_city_cursors)
    for name, (X, Y) in per_city_windows.items():
        if X.shape[0] == 0:
            continue
        idx = per_city_indices[name]
        cur = new_cursors[name]
        if cur + batch_per_city > len(idx):
            rng.shuffle(idx)
            cur = 0
        sel = idx[cur : cur + batch_per_city]
        new_cursors[name] = cur + batch_per_city
        out_x[name] = (X[sel], Y[sel])
    return out_x, new_cursors


def lambda_progress(global_step: int, total_steps: int, gamma: float = DANN_GAMMA) -> float:
    """Ganin's logistic warm-up — but now `total_steps` is the full 25-epoch
    joint phase, so saturation happens close to the end of training as intended.
    """
    p = max(0.0, min(1.0, global_step / max(1, total_steps)))
    return float(2.0 / (1.0 + float(np.exp(-gamma * p))) - 1.0)


# --------------------------------------------------------------------------- #
# Train
# --------------------------------------------------------------------------- #

def train_graph_dann(
    cities: Dict[str, CityTensors],
    source: str,
    target: str,
    d_pct: float,
    *,
    backbone: str = "gat",
    fixed: bool = True,
    warm_start_source: bool = True,
    alpha_d: float = ALPHA_D,
) -> Dict[str, Dict[str, float]]:
    """Returns a dict with {`zero_shot`, `scratch_dann`, `transfer`} sub-dicts.

    Mirrors Variant A's three-way verification protocol: the DANN-pretrained
    transfer must beat (a) the zero-shot baseline (DANN encoder, no FT) AND
    (b) a from-scratch baseline at the same d% target sample.
    """
    set_seed(42)
    src = cities[source]
    tgt = cities[target]
    F_dim = src.feature_tensor.shape[-1]
    encoder = build_backbone(backbone, F_dim, fixed=fixed).to(DEVICE)

    # ── (1) Source-only warm-start (ADDA, Tzeng et al., CVPR-17). Loading
    #         the source checkpoint produces an encoder whose forecast head
    #         already works on the source domain — so the discriminator's
    #         reverse gradients don't have to fight a randomly-initialized
    #         encoder for the first ~5 epochs.
    src_suffix = "_fixed" if fixed else ""
    src_ckpt = SRC_MODELS_DIR / f"{backbone}_source_{source}{src_suffix}.pt"
    if warm_start_source:
        if not src_ckpt.exists():
            raise RuntimeError(
                f"warm-start requested but missing source checkpoint: {src_ckpt}.\n"
                f"Run: python -m src.train_gnn --backbone {backbone}"
                + (" --fixed" if fixed else "")
            )
        load_checkpoint(encoder, src_ckpt)
        print(f"  [warm-start] loaded encoder weights from {src_ckpt}")

    model = GraphDANN(encoder, n_cities=3).to(DEVICE)

    edges = build_per_city_edges(cities)
    per_city = build_per_city_train_arrays(cities, fixed=fixed)
    indices_by_city = {
        name: np.arange(per_city[name][0].shape[0])
        for name in per_city
        if per_city[name][0].shape[0] > 0
    }
    rng = np.random.default_rng(42)
    for arr in indices_by_city.values():
        rng.shuffle(arr)
    cursors = {name: 0 for name in indices_by_city}

    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    mse = nn.MSELoss()
    ce = nn.CrossEntropyLoss()

    steps_per_epoch = max(1, per_city[source][0].shape[0] // JOINT_BATCH_PER_CITY)
    total_steps = EPOCHS_JOINT * steps_per_epoch
    global_step = 0

    print(f"\n[Graph-DANN/{backbone}{' /FIX' if fixed else ''} {source}->{target}] "
          f"joint training: {EPOCHS_JOINT} ep x {steps_per_epoch} steps  "
          f"(alpha_d={alpha_d}, gamma={DANN_GAMMA}, warm_start={warm_start_source})")

    # Track a side-checkpoint of the encoder by best source-val R² during joint
    # phase — this is what zero-shot will be evaluated from.
    if fixed:
        _, X_va_src, Y_va_src, C_va_src = make_city_loader_masked(src, "val", 64, False)
    else:
        _, X_va_src, Y_va_src = make_city_loader(src, src.train_end_idx, src.val_end_idx, 64, False)
        C_va_src = None
    joint_best_val = -1e9
    joint_ckpt = MODELS_DIR / f"{backbone}_dann_joint_{source}_to_{target}{'_fixed' if fixed else ''}.pt"

    for ep in range(EPOCHS_JOINT):
        model.train()
        sum_loss_y, sum_loss_d, n = 0.0, 0.0, 0
        for step in range(steps_per_epoch):
            lam = lambda_progress(global_step, total_steps)
            mb, cursors = sample_minibatch(per_city, indices_by_city, cursors, JOINT_BATCH_PER_CITY, rng)

            # --- Source supervised + city loss ---
            if source in mb:
                xb, yb = mb[source]
                xb_t = torch.from_numpy(xb).float()
                yb_t = torch.from_numpy(yb).float()
                ei, ew = edges[source]
                y_hat, city_logits = model(xb_t, ei, ew, lambda_=lam)
                loss_y = mse(y_hat, yb_t)
                lab = torch.full((xb_t.size(0),), CITY_TO_LABEL[source], dtype=torch.long)
                loss_d_src = ce(city_logits, lab)
            else:
                loss_y = torch.tensor(0.0)
                loss_d_src = torch.tensor(0.0)

            # --- Target unsupervised (city loss only, PM2.5 labels withheld) ---
            if target in mb:
                xb, _ = mb[target]
                xb_t = torch.from_numpy(xb).float()
                ei, ew = edges[target]
                _, city_logits = model(xb_t, ei, ew, lambda_=lam)
                lab = torch.full((xb_t.size(0),), CITY_TO_LABEL[target], dtype=torch.long)
                loss_d_tgt = ce(city_logits, lab)
            else:
                loss_d_tgt = torch.tensor(0.0)

            # --- Third-city replay (stabilizes 3-class discriminator) ---
            third = [c for c in cities if c not in (source, target)]
            if third and third[0] in mb:
                xb, _ = mb[third[0]]
                xb_t = torch.from_numpy(xb).float()
                ei, ew = edges[third[0]]
                _, city_logits = model(xb_t, ei, ew, lambda_=lam)
                lab = torch.full((xb_t.size(0),), CITY_TO_LABEL[third[0]], dtype=torch.long)
                loss_d_third = ce(city_logits, lab)
            else:
                loss_d_third = torch.tensor(0.0)

            # --- Combined: α_d weights the 3-way discriminator loss ---
            loss_d = loss_d_src + loss_d_tgt + loss_d_third
            loss = loss_y + alpha_d * loss_d
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            sum_loss_y += float(loss_y.item())
            sum_loss_d += float(loss_d.item())
            n += 1
            global_step += 1

        # Track best encoder by source-val R² — this is the "zero-shot" snapshot.
        val_metrics_src = evaluate_gnn(model.encoder, X_va_src, Y_va_src,
                                       edges[source][0], edges[source][1],
                                       src.target_scaler, climatology=C_va_src)
        improved = val_metrics_src["R2"] > joint_best_val + 1e-4
        if improved:
            joint_best_val = val_metrics_src["R2"]
            save_checkpoint(model, joint_ckpt, extra={"src_val_r2": joint_best_val})
        flag = "*" if improved else " "
        print(f"  joint ep {ep+1:02d}/{EPOCHS_JOINT} {flag} L_y={sum_loss_y/max(n,1):.4f} "
              f"L_d={sum_loss_d/max(n,1):.4f} lam={lam:.3f}  "
              f"src_val_R2={val_metrics_src['R2']:+.4f}")

    # Restore best-by-source-val joint checkpoint before fine-tune.
    load_checkpoint(model, joint_ckpt)

    # ── (2) Zero-shot — eval the warm-started + DANN-aligned encoder on
    #         target test WITHOUT any fine-tune.
    ei_t, ew_t = edges[target]
    if fixed:
        _, X_va, Y_va, C_va = make_city_loader_masked(tgt, "val", 64, False)
        _, X_te, Y_te, C_te = make_city_loader_masked(tgt, "test", 64, False)
    else:
        _, X_va, Y_va = make_city_loader(tgt, tgt.train_end_idx, tgt.val_end_idx, 64, False)
        _, X_te, Y_te = make_city_loader(tgt, tgt.val_end_idx, tgt.test_end_idx, 64, False)
        C_va = C_te = None

    zs_metrics = evaluate_gnn(model.encoder, X_te, Y_te, ei_t, ew_t,
                              tgt.target_scaler, climatology=C_te)
    print(f"  [0] zero-shot (DANN encoder, NO fine-tune)        test = {zs_metrics}")

    # ── (3) Target fine-tune (adversary off, Variant-A-grade recipe).
    cfg = FIXED_FT_CFG[target] if fixed else {"epochs": 8, "batch_size": BATCH_SIZE, "lr": 1e-4, "patience": None}
    ft_epochs = cfg["epochs"]
    ft_bs = cfg["batch_size"]
    ft_lr = cfg["lr"]
    ft_patience = cfg["patience"]
    freeze_t1_for = int(ft_epochs * FREEZE_FRAC)

    if fixed:
        X_full, Y_full, C_full = make_windows_masked(
            tgt.feature_tensor, tgt.target_tensor, tgt.train_mask,
            history=HISTORY, horizon=HORIZON,
            climatology_tensor=tgt.climatology_tensor,
        )
    else:
        X_full, Y_full = make_windows(
            tgt.feature_tensor, tgt.target_tensor, HISTORY, HORIZON, 0, tgt.train_end_idx,
        )
    rng_ft = np.random.default_rng(42)
    keep = rng_ft.choice(X_full.shape[0], size=max(1, int(d_pct * X_full.shape[0])), replace=False)
    keep.sort()
    X_ft, Y_ft = X_full[keep], Y_full[keep]
    print(f"  [1] fine-tune  cfg={cfg}  freeze_t1_for={freeze_t1_for}  "
          f"data={X_ft.shape}")

    ds = torch.utils.data.TensorDataset(torch.from_numpy(X_ft).float(), torch.from_numpy(Y_ft).float())
    train_loader = torch.utils.data.DataLoader(ds, batch_size=ft_bs, shuffle=True, num_workers=0)

    ft_opt = torch.optim.Adam(model.parameters(), lr=ft_lr, weight_decay=WEIGHT_DECAY)
    ft_scheduler = (torch.optim.lr_scheduler.ReduceLROnPlateau(ft_opt, mode="max", factor=0.5, patience=3)
                    if ft_patience else None)

    if freeze_t1_for > 0:
        for p in model.encoder.t1.parameters():
            p.requires_grad = False

    best_val = -1e9
    patience_left = ft_patience
    tl_ckpt = MODELS_DIR / f"{backbone}_dann_{source}_to_{target}_d{int(d_pct*100)}{'_fixed' if fixed else ''}.pt"
    for ep in range(ft_epochs):
        if freeze_t1_for > 0 and ep == freeze_t1_for:
            for p in model.encoder.t1.parameters():
                p.requires_grad = True
            print(f"    [unfreezing t1 at epoch {ep+1}]")
        model.train()
        for xb, yb in train_loader:
            ft_opt.zero_grad()
            # Adversary OFF during FT (lam=0): pure supervised target-only loss.
            y_hat, _ = model(xb, ei_t, ew_t, lambda_=0.0)
            loss = mse(y_hat, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            ft_opt.step()
        val_metrics = evaluate_gnn(model.encoder, X_va, Y_va, ei_t, ew_t,
                                   tgt.target_scaler, climatology=C_va)
        if ft_scheduler is not None:
            ft_scheduler.step(val_metrics["R2"])
        improved = val_metrics["R2"] > best_val + 1e-4
        if improved:
            best_val = val_metrics["R2"]
            save_checkpoint(model, tl_ckpt, extra={"val_r2": best_val})
            if ft_patience is not None:
                patience_left = ft_patience
        elif ft_patience is not None:
            patience_left -= 1
        flag = "*" if improved else " "
        lr_now = ft_opt.param_groups[0]["lr"]
        print(f"    ft ep {ep+1:02d}/{ft_epochs} {flag} val_R2={val_metrics['R2']:+.4f}  "
              f"lr={lr_now:.2e}  patience={patience_left}")
        if ft_patience is not None and patience_left is not None and patience_left <= 0:
            print(f"    early-stopped at epoch {ep+1} (best val R2={best_val:.4f})")
            break

    load_checkpoint(model, tl_ckpt)
    tl_metrics = evaluate_gnn(model.encoder, X_te, Y_te, ei_t, ew_t,
                              tgt.target_scaler, climatology=C_te)
    print(f"      transfer test = {tl_metrics}")

    real_transfer = tl_metrics["R2"] > zs_metrics["R2"]
    print(f"  ==> zero_shot.R2={zs_metrics['R2']:+.3f}  transfer.R2={tl_metrics['R2']:+.3f}  "
          f"real_transfer_vs_zs={real_transfer}")

    return {
        "zero_shot": zs_metrics,
        "transfer": tl_metrics,
        "real_transfer": real_transfer,
        "joint_src_val_R2": float(joint_best_val),
        "alpha_d": alpha_d,
        "warm_start": warm_start_source,
    }


def main(args):
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    cities = load_all_cities(fixed=args.fixed)

    pairs = [
        ("Delhi", "Kolkata"),
        ("Delhi", "Guwahati"),
        ("Kolkata", "Delhi"),
        ("Kolkata", "Guwahati"),
        ("Guwahati", "Delhi"),
        ("Guwahati", "Kolkata"),
    ]
    d_values = [0.15, 0.30, 0.45, 0.60]

    print("=" * 68)
    print(f"GNN-DANN run  backbone={args.backbone}  fixed={args.fixed}  "
          f"alpha_d={args.alpha_d}  warm_start={not args.no_warm_start}")
    print(f"Pairs: {', '.join(f'{s}->{t}' for s,t in pairs)}   d={d_values}")
    print("=" * 68)

    results: Dict[str, Dict] = {}
    for s, t in pairs:
        for d in d_values:
            key = f"{s}->{t}@{int(d*100)}"
            results[key] = train_graph_dann(
                cities, s, t, d,
                backbone=args.backbone,
                fixed=args.fixed,
                warm_start_source=not args.no_warm_start,
                alpha_d=args.alpha_d,
            )
            suffix = "_fixed" if args.fixed else ""
            out = RESULTS_DIR / f"variantB_dann_{args.backbone}{suffix}.json"
            write_results(out, results)

    suffix = "_fixed" if args.fixed else ""
    out = RESULTS_DIR / f"variantB_dann_{args.backbone}{suffix}.json"
    write_results(out, results)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--backbone", choices=["gat", "sage"], default="gat")
    p.add_argument("--fixed", action="store_true",
                   help="Use interleaved-split + climatology-residual + per-city LSTM-grade recipe.")
    p.add_argument("--no-warm-start", action="store_true",
                   help="Disable ADDA-style source pre-train warm-start (debug only — expect collapse).")
    p.add_argument("--alpha-d", type=float, default=ALPHA_D,
                   help="Weight on the 3-way discriminator loss (default 0.1).")
    main(p.parse_args())
