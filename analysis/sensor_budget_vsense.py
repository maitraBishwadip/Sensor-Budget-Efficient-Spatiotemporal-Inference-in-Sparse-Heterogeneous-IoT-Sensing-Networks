"""Graceful degradation at UNSENSORED nodes — the experiment the manuscript text describes.

The original analysis/sensor_budget.py deletes the dropped N-k nodes from the tensors and
graph entirely and reports next-step forecasting at the k RETAINED nodes (reviewer issue
R2-M1/C1: that is a different, easier task than "inference at unsensored nodes"). This
script runs the claimed experiment:

  - ALL N target nodes stay in the graph.
  - k nodes are SENSORED (S); the other N-k are UNSENSORED (U). The PM2.5 input channel is
    masked to zero at U (meteorology stays available everywhere), reusing the virtual-sensing
    convention of src/train_gnn_vsense.py.
  - The model warm-starts from the source checkpoint exactly as transfer_variant_e does
    (src/train_gnn_pinn.py): same checkpoint, same FIXED_FT_CFG recipe, same t1 freeze, and
    the same seeded d=30% subsample of the target's train windows.
  - Fine-tune loss is masked reconstruction on S ONLY (a k-sensor deployment has no labels
    at U): each step additionally masks a random R subset of S (P_MASK) and supervises S.
  - Evaluation: R2/MAE/RMSE at U on the held-out test partition (input masked at U).

Node subsets use the SAME fixed permutation as analysis/sensor_budget.py
(np.random.default_rng(0).permutation(N), first k, nested), so each k's sensored set is
identical to the retained set of the old experiment — the two curves are comparable.

Writes results/gnn_vsense/sensor_budget_vsense.json (incrementally, cell by cell).

Run:  python -u -m analysis.sensor_budget_vsense
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.graph_construction import build_city_graph                    # noqa: E402
from src.train_gnn import (                                            # noqa: E402
    DEVICE, GRAPH_STRATEGY, HISTORY, HORIZON, KNN_K,
    build_backbone, load_all_cities, make_city_loader_masked,
)
from src.train_gnn_tl import FIXED_FT_CFG, FREEZE_FRAC, SRC_MODELS_DIR, WEIGHT_DECAY  # noqa: E402
from src.train_gnn_vsense import P_MASK, _eval_at_nodes                # noqa: E402
from src.utils import (                                                # noqa: E402
    load_checkpoint, make_windows_masked, save_checkpoint, set_seed, write_results,
)

OUT = Path("results/gnn_vsense/sensor_budget_vsense.json")
MODELS_DIR = Path("models/gnn_vsense_budget")
D_PCT = 0.30
CONFIGS = [("Delhi", "Kolkata"), ("Kolkata", "Delhi")]   # (source, target) — Fig. 6 panels


def budget_vsense_cell(cities, source: str, target: str, k: int, *, backbone="gat") -> dict:
    set_seed(42)
    tgt = cities[target]
    F_dim = tgt.feature_tensor.shape[-1]
    N = tgt.feature_tensor.shape[1]

    # Same nested subsets as analysis/sensor_budget.py: S = retained set of the old runs.
    perm = np.random.default_rng(0).permutation(N)
    S = np.sort(perm[:k])
    U = np.sort(perm[k:])

    ei, ew = build_city_graph(tgt.coords, strategy=GRAPH_STRATEGY, k=KNN_K)
    ei, ew = ei.to(DEVICE), ew.to(DEVICE)

    X_tr, Y_tr, _ = make_windows_masked(tgt.feature_tensor, tgt.target_tensor, tgt.train_mask,
                                        HISTORY, HORIZON, climatology_tensor=tgt.climatology_tensor)
    _, X_va, Y_va, C_va = make_city_loader_masked(tgt, "val", 64, False)
    _, X_te, Y_te, C_te = make_city_loader_masked(tgt, "test", 64, False)

    # Same seeded d% subsample draw as transfer_variant_e / transfer_variant_a.
    cfg = FIXED_FT_CFG[target]
    rng = np.random.default_rng(42)
    keep = rng.choice(X_tr.shape[0], size=max(1, int(D_PCT * X_tr.shape[0])), replace=False)
    keep.sort()
    ds = torch.utils.data.TensorDataset(
        torch.from_numpy(X_tr[keep]).float(), torch.from_numpy(Y_tr[keep]).float())
    loader = torch.utils.data.DataLoader(ds, batch_size=cfg["batch_size"], shuffle=True)

    src_ckpt = SRC_MODELS_DIR / f"{backbone}_source_{source}_fixed.pt"
    if not src_ckpt.exists():
        raise RuntimeError(f"missing source checkpoint: {src_ckpt}")

    print(f"\n[budget-vsense {source}->{target} k={k}/{N} d={D_PCT:.0%}] "
          f"|S|={len(S)} |U|={len(U)}  ft={X_tr[keep].shape}")

    # Zero-shot reference: source weights, no fine-tune, PM2.5 masked at U, eval at U.
    zs_model = build_backbone(backbone, F_dim, fixed=True).to(DEVICE)
    load_checkpoint(zs_model, src_ckpt)
    zs = _eval_at_nodes(zs_model, X_te, Y_te, C_te, ei, ew, tgt.target_scaler, U, U, DEVICE)
    print(f"  zero-shot U-test: R2={zs['R2']:.4f} MAE={zs['MAE']:.3f}")

    # Transfer: warm-start from source, fine-tune with masked reconstruction on S.
    model = build_backbone(backbone, F_dim, fixed=True).to(DEVICE)
    load_checkpoint(model, src_ckpt)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=3)
    mse = torch.nn.MSELoss()
    freeze_for = int(cfg["epochs"] * FREEZE_FRAC)
    if freeze_for > 0:
        for p in model.t1.parameters():
            p.requires_grad = False

    S_t = torch.as_tensor(S, device=DEVICE)
    nR = max(1, int(P_MASK * len(S)))
    ckpt = MODELS_DIR / f"{backbone}_budget_{source}_to_{target}_k{k}_d{int(D_PCT*100)}.pt"
    set_seed(42)
    best_val, patience_left = -1e9, cfg["patience"]
    rng_mask = np.random.default_rng(123)
    for ep in range(cfg["epochs"]):
        if freeze_for > 0 and ep == freeze_for:
            for p in model.t1.parameters():
                p.requires_grad = True
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            Ridx = S[rng_mask.permutation(len(S))[:nR]]
            mask_cols = np.concatenate([U, Ridx])
            xbm = xb.clone()
            xbm[:, :, mask_cols, 0] = 0.0
            opt.zero_grad()
            pred = model(xbm, ei, ew)
            loss = mse(pred[:, S_t], yb[:, S_t])
            if not torch.isfinite(loss):
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        vm = _eval_at_nodes(model, X_va, Y_va, C_va, ei, ew, tgt.target_scaler, U, U, DEVICE)
        sched.step(vm["R2"])
        if vm["R2"] > best_val + 1e-4:
            best_val = vm["R2"]
            save_checkpoint(model, ckpt, extra={"val_r2": best_val})
            patience_left = cfg["patience"]
        else:
            patience_left -= 1
        print(f"    ft ep {ep+1:02d}/{cfg['epochs']} U-val R2={vm['R2']:+.4f} best={best_val:+.4f}")
        if patience_left <= 0:
            break

    load_checkpoint(model, ckpt)
    te = _eval_at_nodes(model, X_te, Y_te, C_te, ei, ew, tgt.target_scaler, U, U, DEVICE)
    # Companion number: same fine-tuned model scored at the sensored set (input unmasked
    # at S) — the retained-node view, for the corrected two-panel figure.
    te_S = _eval_at_nodes(model, X_te, Y_te, C_te, ei, ew, tgt.target_scaler, S, U, DEVICE)
    print(f"  ==> U-test R2={te['R2']:.4f} MAE={te['MAE']:.3f} RMSE={te['RMSE']:.3f} "
          f"(zero-shot {zs['R2']:.4f}; S-test R2={te_S['R2']:.4f})")
    return {
        "R2": te["R2"], "MAE": te["MAE"], "RMSE": te["RMSE"],
        "zero_shot_R2": zs["R2"], "zero_shot_MAE": zs["MAE"], "zero_shot_RMSE": zs["RMSE"],
        "sensored_R2": te_S["R2"], "val_R2": best_val,
        "k": int(k), "N": int(N), "n_unsensored": int(len(U)),
        "source": source, "target": target, "d_pct": D_PCT,
        "S_idx": [int(i) for i in S], "U_idx": [int(i) for i in U],
    }


def main():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    cities = load_all_cities(fixed=True)
    results = {}
    for s, t in CONFIGS:
        N = len(cities[t].coords)
        # k = N would leave U empty; the sweep stops one rung short of full coverage.
        ks = [k for k in ([4, 6, 8] if N <= 10 else [4, 8, 16]) if k < N]
        for k in ks:
            results[f"{s}->{t}|k={k}"] = budget_vsense_cell(cities, s, t, k)
            write_results(OUT, results)

    print("\n" + "=" * 72)
    print("GRACEFUL DEGRADATION AT UNSENSORED NODES — R2(U) vs sensored count k")
    print("=" * 72)
    for s, t in CONFIGS:
        rows = [v for v in results.values() if v["source"] == s and v["target"] == t]
        print(f"\n{s} -> {t}  (d={int(D_PCT*100)}% of target labels at S)")
        print("  k    R2(U)    MAE(U)   RMSE(U)  zero-shot  R2(S)")
        for v in sorted(rows, key=lambda r: r["k"]):
            print(f"  {v['k']:<3}  {v['R2']:.4f}  {v['MAE']:7.3f}  {v['RMSE']:7.3f}  "
                  f"{v['zero_shot_R2']:8.4f}  {v['sensored_R2']:.4f}")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
