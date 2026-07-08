"""Variant A: pre-train on source city + fine-tune on d% of target.

Loads the source-only ST-GNN checkpoint produced by train_gnn.py, then
fine-tunes it on the target city's graph using a small d% subsample of the
target's training data. Evaluates on the target's held-out test partition.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torch.nn as nn

from src.graph_construction import build_city_graph
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


MODELS_DIR = Path("models/gnn_tl")
RESULTS_DIR = Path("results/gnn_tl")
SRC_MODELS_DIR = Path("models/gnn")

EPOCHS_FT = 8
FT_LR = 1e-4
WEIGHT_DECAY = 1e-5
FREEZE_FRAC = 0.20   # freeze early layers for the first 20% of fine-tune

# `--fixed` fine-tune recipe (per target city, mirroring CITY_FT_CFG in LSTM).
FIXED_FT_CFG = {
    "Delhi":    {"epochs": 25, "batch_size": 128, "lr": 2e-4, "patience": 6},
    "Kolkata":  {"epochs": 30, "batch_size": 64,  "lr": 2e-4, "patience": 8},
    "Guwahati": {"epochs": 40, "batch_size": 32,  "lr": 1e-4, "patience": 10},
}


def _fit_one_run(
    model: nn.Module,
    train_loader,
    edge_index, edge_weight,
    X_va, Y_va, C_va,
    target_scaler,
    *,
    epochs: int, lr: float, patience: int | None,
    ckpt_path: Path,
    freeze_t1_for: int = 0,
) -> float:
    """Train `model` and return best val R²; write best-by-val checkpoint to `ckpt_path`."""
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=3) if patience else None
    loss_fn = nn.MSELoss()

    if freeze_t1_for > 0:
        for p in model.t1.parameters():
            p.requires_grad = False

    best_val = -1e9
    patience_left = patience
    for ep in range(epochs):
        if freeze_t1_for > 0 and ep == freeze_t1_for:
            for p in model.t1.parameters():
                p.requires_grad = True
            print(f"    [unfreezing t1 at epoch {ep+1}]")

        model.train()
        total, n = 0.0, 0
        for xb, yb in train_loader:
            opt.zero_grad()
            pred = model(xb, edge_index, edge_weight)
            loss = loss_fn(pred, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += float(loss.item()) * xb.size(0)
            n += xb.size(0)
        val_metrics = evaluate_gnn(model, X_va, Y_va, edge_index, edge_weight, target_scaler, climatology=C_va)
        if scheduler is not None:
            scheduler.step(val_metrics["R2"])
        improved = val_metrics["R2"] > best_val + 1e-4
        if improved:
            best_val = val_metrics["R2"]
            save_checkpoint(model, ckpt_path, extra={"val_r2": best_val})
            if patience is not None:
                patience_left = patience
        elif patience is not None:
            patience_left -= 1
        print(f"      ft ep {ep+1:02d}/{epochs} val_R2={val_metrics['R2']:+.4f} best={best_val:+.4f}")
        if patience is not None and patience_left is not None and patience_left <= 0:
            break
    return best_val


def transfer_variant_a(
    cities: Dict[str, CityTensors],
    source: str,
    target: str,
    d_pct: float,
    backbone: str = "gat",
    *,
    fixed: bool = False,
    run_baselines: bool = True,
    src_ckpt_extra: str = "",
    subsample: str = "random",
    seed: int = 42,
    freeze_frac: float = FREEZE_FRAC,
) -> Dict[str, Dict[str, float]]:
    """Returns a dict with three sub-dicts: `zero_shot`, `scratch`, `transfer`.

    Knowledge transfer is considered "real" iff
        transfer.R2 > zero_shot.R2  AND  transfer.R2 > scratch.R2.

    `src_ckpt_extra` selects an alternative source checkpoint (e.g. "_chrono").
    `subsample` chooses how the d% fine-tune keep-set is drawn: "random" (seeded
    uniform draw over the whole training year — the label-efficiency condition)
    or "prefix" (the first d% of windows in time order — the true cold-start
    condition, where a new deployment has only a contiguous history prefix).
    `seed` varies training stochasticity (init, shuffling) for seed sweeps; the
    d% keep-set draw stays fixed so every seed fine-tunes on identical data.
    `freeze_frac` sets the fraction of fine-tune epochs for which the input
    temporal block t1 stays frozen (sensitivity study; default FREEZE_FRAC).
    """
    set_seed(seed)
    src = cities[source]
    tgt = cities[target]
    F = src.feature_tensor.shape[-1]
    assert F == tgt.feature_tensor.shape[-1]

    # --- Source and target graph details (sanity log for the user) ---
    src_ei, src_ew = build_city_graph(src.coords, strategy=GRAPH_STRATEGY, k=KNN_K)
    edge_index, edge_weight = build_city_graph(tgt.coords, strategy=GRAPH_STRATEGY, k=KNN_K)
    print(f"\n[{backbone}-TL {source}->{target} d={d_pct:.0%}{' /FIX' if fixed else ''}]")
    print(f"  graphs: source({source}) |V|={len(src.coords)} |E|={src_ei.shape[1]} "
          f"avg_deg={src_ei.shape[1]/len(src.coords):.2f}  "
          f"target({target}) |V|={len(tgt.coords)} |E|={edge_index.shape[1]} "
          f"avg_deg={edge_index.shape[1]/len(tgt.coords):.2f}  strategy={GRAPH_STRATEGY}")

    # --- Build target full training set (with climatology when fixed) ---
    src_suffix = ("_fixed" if fixed else "") + src_ckpt_extra
    src_ckpt = SRC_MODELS_DIR / f"{backbone}_source_{source}{src_suffix}.pt"
    if not src_ckpt.exists():
        raise RuntimeError(f"missing source checkpoint: {src_ckpt}")

    # Build target full training set (with climatology when fixed).
    if fixed:
        X_full, Y_full, C_full = make_windows_masked(
            tgt.feature_tensor, tgt.target_tensor, tgt.train_mask,
            history=HISTORY, horizon=HORIZON,
            climatology_tensor=tgt.climatology_tensor,
        )
        _, X_va, Y_va, C_va = make_city_loader_masked(tgt, "val", BATCH_SIZE, False)
        _, X_te, Y_te, C_te = make_city_loader_masked(tgt, "test", BATCH_SIZE, False)
        cfg = FIXED_FT_CFG[target]
        epochs = cfg["epochs"]
        lr = cfg["lr"]
        ft_bs = cfg["batch_size"]
        patience = cfg["patience"]
    else:
        X_full, Y_full = make_windows(
            tgt.feature_tensor, tgt.target_tensor, HISTORY, HORIZON,
            start_idx=0, end_idx=tgt.train_end_idx,
        )
        C_full = None
        _, X_va, Y_va = make_city_loader(tgt, tgt.train_end_idx, tgt.val_end_idx, BATCH_SIZE, False)
        _, X_te, Y_te = make_city_loader(tgt, tgt.val_end_idx, tgt.test_end_idx, BATCH_SIZE, False)
        C_va = C_te = None
        epochs = EPOCHS_FT
        lr = FT_LR
        ft_bs = BATCH_SIZE
        patience = None

    if X_full.shape[0] == 0:
        raise RuntimeError(f"no target training windows for {target}")
    n_keep = max(1, int(d_pct * X_full.shape[0]))
    if subsample == "prefix":
        # True cold start: only the first d% of the target's history, in time order.
        keep = np.arange(n_keep)
    else:
        rng = np.random.default_rng(42)
        keep = rng.choice(X_full.shape[0], size=n_keep, replace=False)
        keep.sort()
    X_ft, Y_ft = X_full[keep], Y_full[keep]
    print(f"  fine-tune set: {X_ft.shape} [{subsample}]  (val={X_va.shape}  test={X_te.shape})")

    ds = torch.utils.data.TensorDataset(torch.from_numpy(X_ft).float(), torch.from_numpy(Y_ft).float())
    train_loader = torch.utils.data.DataLoader(ds, batch_size=ft_bs, shuffle=True, num_workers=0)

    ckpt_suffix = (("_fixed" if fixed else "") + src_ckpt_extra
                   + ("_prefix" if subsample == "prefix" else "")
                   + (f"_seed{seed}" if seed != 42 else "")
                   + (f"_fz{int(round(freeze_frac * 100))}" if freeze_frac != FREEZE_FRAC else ""))
    freeze_t1_for = int(epochs * freeze_frac)

    # =========================================================
    # 0. Zero-shot: load source weights, evaluate target test
    #    *without any fine-tune* — answers "did the source model
    #    learn anything that generalizes to the target?"
    # =========================================================
    zs_model = build_backbone(backbone, F, fixed=fixed).to(DEVICE)
    load_checkpoint(zs_model, src_ckpt)
    zs_metrics = evaluate_gnn(zs_model, X_te, Y_te, edge_index, edge_weight, tgt.target_scaler, climatology=C_te)
    print(f"  [0] zero-shot  (source weights, NO fine-tune)  test = {zs_metrics}")

    # =========================================================
    # 1. Transfer: load source weights, fine-tune on d% target
    # =========================================================
    transfer_model = build_backbone(backbone, F, fixed=fixed).to(DEVICE)
    load_checkpoint(transfer_model, src_ckpt)
    tl_ckpt = MODELS_DIR / f"{backbone}_tl_{source}_to_{target}_d{int(d_pct*100)}{ckpt_suffix}.pt"
    set_seed(seed)
    print(f"  [1] transfer  (source -> fine-tune {epochs} ep, freeze t1 for {freeze_t1_for})")
    _fit_one_run(
        transfer_model, train_loader, edge_index, edge_weight,
        X_va, Y_va, C_va, tgt.target_scaler,
        epochs=epochs, lr=lr, patience=patience,
        ckpt_path=tl_ckpt, freeze_t1_for=freeze_t1_for,
    )
    load_checkpoint(transfer_model, tl_ckpt)
    tl_metrics = evaluate_gnn(transfer_model, X_te, Y_te, edge_index, edge_weight, tgt.target_scaler, climatology=C_te)
    print(f"      transfer test = {tl_metrics}")

    # =========================================================
    # 2. Scratch baseline: random init, fine-tune on same d%.
    #    If transfer < scratch, source pre-training is *hurting*.
    # =========================================================
    scratch_metrics: Dict[str, float] = {"R2": float("nan"), "MAE": float("nan"), "RMSE": float("nan"), "MAPE": float("nan")}
    if run_baselines:
        scratch_model = build_backbone(backbone, F, fixed=fixed).to(DEVICE)  # fresh weights
        scratch_ckpt = MODELS_DIR / f"{backbone}_scratch_{target}_d{int(d_pct*100)}{ckpt_suffix}.pt"
        set_seed(seed)
        print(f"  [2] scratch   (random init, fine-tune {epochs} ep, no t1 freeze)")
        _fit_one_run(
            scratch_model, train_loader, edge_index, edge_weight,
            X_va, Y_va, C_va, tgt.target_scaler,
            epochs=epochs, lr=lr, patience=patience,
            ckpt_path=scratch_ckpt, freeze_t1_for=0,
        )
        load_checkpoint(scratch_model, scratch_ckpt)
        scratch_metrics = evaluate_gnn(scratch_model, X_te, Y_te, edge_index, edge_weight, tgt.target_scaler, climatology=C_te)
        print(f"      scratch  test = {scratch_metrics}")

    # ---- Transfer is "real" iff transfer beats both baselines ----
    real_transfer = (tl_metrics["R2"] > zs_metrics["R2"]) and (tl_metrics["R2"] > scratch_metrics["R2"])
    print(f"  ==> transfer.R2={tl_metrics['R2']:+.3f}  zero_shot.R2={zs_metrics['R2']:+.3f}  "
          f"scratch.R2={scratch_metrics['R2']:+.3f}  "
          f"real_transfer={real_transfer}")

    return {
        "zero_shot": zs_metrics,
        "scratch": scratch_metrics,
        "transfer": tl_metrics,
        "real_transfer": real_transfer,
        "source_graph": {"V": len(src.coords), "E": int(src_ei.shape[1]), "avg_deg": float(src_ei.shape[1]/len(src.coords))},
        "target_graph": {"V": len(tgt.coords), "E": int(edge_index.shape[1]), "avg_deg": float(edge_index.shape[1]/len(tgt.coords))},
    }


def main(args):
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    cities = load_all_cities(fixed=args.fixed, chrono=args.chrono)
    src_ckpt_extra = "_chrono" if args.chrono else ""

    if args.pairs:
        pairs = [tuple(p.split(">")) for p in args.pairs.split(",")]
    elif args.full:
        # All six ordered (source, target) pairs — exactly the layout of thesis Table 5.2.
        pairs = [
            ("Delhi", "Kolkata"),
            ("Kolkata", "Delhi"),
            ("Kolkata", "Guwahati"),
            ("Guwahati", "Kolkata"),
            ("Guwahati", "Delhi"),
            ("Delhi", "Guwahati"),
        ]
    else:
        pairs = [
            ("Delhi", "Kolkata"),
            ("Delhi", "Guwahati"),
            ("Kolkata", "Guwahati"),
        ]
    if args.d:
        d_values = [int(x) / 100.0 for x in args.d.split(",")]
    elif args.full or args.pairs:
        d_values = [0.15, 0.30, 0.45, 0.60]
    else:
        d_values = [0.30, 0.60]

    tag = (src_ckpt_extra + ("_prefix" if args.subsample == "prefix" else "")
           + (f"_seed{args.seed}" if args.seed != 42 else "")
           + (f"_fz{int(round(args.freeze_frac * 100))}" if args.freeze_frac != FREEZE_FRAC else ""))
    suffix = ("_fixed" if args.fixed else "") + tag
    out = RESULTS_DIR / f"variantA_{args.backbone}{suffix}.json"

    print("=" * 68)
    print(f"GNN TL run -- backbone={args.backbone}  fixed={args.fixed}  chrono={args.chrono}  "
          f"subsample={args.subsample}")
    print(f"Pairs ({len(pairs)}): {', '.join(f'{s}->{t}' for s, t in pairs)}")
    print(f"d-values: {d_values}")
    print(f"Total cells: {len(pairs)*len(d_values)}  ->  {out}")
    print("=" * 68)

    results = {}
    if out.exists():
        results = json.loads(out.read_text())   # resume/merge across targeted runs
    for src, tgt in pairs:
        for d in d_values:
            key = f"{src}->{tgt}@{int(d*100)}"
            results[key] = transfer_variant_a(
                cities, src, tgt, d, backbone=args.backbone, fixed=args.fixed,
                src_ckpt_extra=src_ckpt_extra, subsample=args.subsample,
                seed=args.seed, freeze_frac=args.freeze_frac)
            # Write incrementally so a mid-run interruption doesn't lose everything.
            write_results(out, results)

    write_results(out, results)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--backbone", choices=["gat", "gatgru", "sage"], default="gat")
    p.add_argument("--fixed", action="store_true",
                   help="Use interleaved-split + climatology-residual source checkpoint and recipe.")
    p.add_argument("--full", action="store_true",
                   help="Run all 6 (source,target) pairs x 4 d-fractions (24 cells). Default is 3 pairs x 2.")
    p.add_argument("--chrono", action="store_true",
                   help="Chronological block split + _chrono source checkpoints (protocol study).")
    p.add_argument("--subsample", choices=["random", "prefix"], default="random",
                   help="How the d%% fine-tune keep-set is drawn (prefix = true cold start).")
    p.add_argument("--pairs", default="",
                   help='Comma list of ordered pairs "Src>Tgt" to run (overrides --full).')
    p.add_argument("--d", default="",
                   help="Comma list of integer d percentages, e.g. 15,30.")
    p.add_argument("--seed", type=int, default=42,
                   help="Training seed (init/shuffle). The d%% keep-set draw stays "
                        "fixed, so seed sweeps compare on identical fine-tune data. "
                        "Non-default seeds tag checkpoints/results with _seed{N}.")
    p.add_argument("--freeze-frac", type=float, default=FREEZE_FRAC,
                   help="Fraction of fine-tune epochs with the input temporal block "
                        "frozen (sensitivity study). Non-default values tag "
                        "checkpoints/results with _fz{pct}.")
    main(p.parse_args())
