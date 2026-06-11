"""Variant D (dual TL, regression-correct): marginal adversary + RSD subspace alignment.

Motivated by the literature review (reports/DUAL_TL_LITERATURE.md): for REGRESSION,
aligning representation *distributions* alters feature scale and impedes adaptation
(Chen, Wang, Wang & Long, "Representation Subspace Distance for DA Regression",
ICML 2021). RSD instead aligns the *orthogonal bases* of the source/target
representation subspaces (scale-free, Grassmann geometry). This "dual" combines:
  - marginal adversarial alignment of P(z)        (as Graph-DANN), and
  - a Grassmann principal-angle subspace-alignment penalty ||sin Theta||_1 between
    source/target embeddings, adapted from RSD (bases-mismatch penalty omitted; scale-free,
    regression-correct).

Joint-phase loss per step:
    L = MSE(y_s) + alpha_m * sum_c CE(D_marg) + beta_rsd * RSD(z_s, z_t)

Reuses Variant B's protocol (warm-start, interleaved split, climatology residual,
per-city fine-tune, 3-way verification) for a one-to-one comparison.

Run (pilot):  python -m src.train_gnn_rsd --fixed --pilot
Run (full):   python -m src.train_gnn_rsd --fixed
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torch.nn as nn

from src.models.dann import CityDiscriminator, grad_reverse
from src.train_gnn import (
    DEVICE, build_backbone, evaluate_gnn, load_all_cities,
    make_city_loader, make_city_loader_masked,
)
from src.train_gnn_dann import (
    CITY_TO_LABEL, EPOCHS_JOINT, FIXED_FT_CFG, FREEZE_FRAC, JOINT_BATCH_PER_CITY,
    LR, SRC_MODELS_DIR, WEIGHT_DECAY, build_per_city_edges,
    build_per_city_train_arrays, lambda_progress, sample_minibatch,
)
from src.utils import (
    CityTensors, load_checkpoint, make_windows, make_windows_masked,
    save_checkpoint, set_seed, write_results,
)

MODELS_DIR = Path("models/gnn_rsd")
RESULTS_DIR = Path("results/gnn_rsd")

ALPHA_M = 0.10    # marginal adversary weight
BETA_RSD = 0.05   # RSD subspace-alignment weight (gentle; RSD is unstable if too strong)
HISTORY, HORIZON = 8, 1


def rsd_loss(f_s: torch.Tensor, f_t: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    """Grassmann principal-angle subspace distance, adapted from RSD (Chen et al., ICML 2021).

    Returns ||sin Theta||_1 — the L1 norm of the sines of the principal angles between the
    source/target feature subspaces (scale-free). NOTE: this is the subspace-alignment term
    ONLY; the bases-mismatch penalty (BMP) of the RSD paper is intentionally omitted, and the
    norm is L1 (the paper uses Frobenius). f_s, f_t: [B, d] batches of embeddings.

    Numerically hardened: (i) skip if inputs are non-finite, (ii) clamp the
    principal-angle cosines to [-1+eps, 1-eps] so the sqrt(1-cos^2) gradient
    stays finite as the subspaces align (cos -> 1), which otherwise explodes
    and drives the encoder embeddings to NaN.
    """
    m = min(f_s.shape[0], f_t.shape[0])                          # equal-size subspaces
    f_s, f_t = f_s[:m], f_t[:m]
    zero = torch.zeros((), dtype=f_s.dtype, device=f_s.device)
    if not (torch.isfinite(f_s).all() and torch.isfinite(f_t).all()):
        return zero
    try:
        # Orthonormal bases via QR (stable backward, unlike SVD on a
        # rank-deficient d x m matrix). Columns of Q span the feature subspace.
        Q_s = torch.linalg.qr(f_s.t(), mode="reduced").Q         # [d, m]
        Q_t = torch.linalg.qr(f_t.t(), mode="reduced").Q         # [d, m]
        # Principal-angle cosines = singular values of Q_s^T Q_t.
        # svdvals backward is stable (no 1/(s_i^2 - s_j^2) terms, unlike svd U/V).
        cospa = torch.linalg.svdvals(Q_s.t() @ Q_t)
    except torch._C._LinAlgError:
        return zero
    cospa = torch.clamp(cospa, -1.0 + eps, 1.0 - eps)            # keep sqrt gradient finite
    sinpa = torch.sqrt(1.0 - cospa ** 2)
    return torch.norm(sinpa, p=1)                                # Grassmann projection distance


class _MarginalHead(nn.Module):
    def __init__(self, d: int, n_cities: int = 3):
        super().__init__()
        self.norm = nn.LayerNorm(d)
        self.disc = CityDiscriminator(d, 64, n_cities)

    def forward(self, z: torch.Tensor, lam: float) -> torch.Tensor:
        return self.disc(grad_reverse(self.norm(z), lam))


def train_graph_rsd(cities: Dict[str, CityTensors], source: str, target: str, d_pct: float, *,
                    backbone: str = "gat", fixed: bool = True,
                    alpha_m: float = ALPHA_M, beta_rsd: float = BETA_RSD) -> Dict[str, Dict[str, float]]:
    set_seed(42)
    src, tgt = cities[source], cities[target]
    F_dim = src.feature_tensor.shape[-1]
    encoder = build_backbone(backbone, F_dim, fixed=fixed).to(DEVICE)
    src_ckpt = SRC_MODELS_DIR / f"{backbone}_source_{source}{'_fixed' if fixed else ''}.pt"
    if not src_ckpt.exists():
        raise RuntimeError(f"missing source checkpoint {src_ckpt}")
    load_checkpoint(encoder, src_ckpt)
    marg = _MarginalHead(encoder.embedding_dim, 3).to(DEVICE)

    edges = build_per_city_edges(cities)
    per_city = build_per_city_train_arrays(cities, fixed=fixed)
    indices = {n: np.arange(per_city[n][0].shape[0]) for n in per_city if per_city[n][0].shape[0] > 0}
    rng = np.random.default_rng(42)
    for a in indices.values():
        rng.shuffle(a)
    cursors = {n: 0 for n in indices}

    params = list(encoder.parameters()) + list(marg.parameters())
    opt = torch.optim.Adam(params, lr=LR, weight_decay=WEIGHT_DECAY)
    mse, ce = nn.MSELoss(), nn.CrossEntropyLoss()
    steps_per_epoch = max(1, per_city[source][0].shape[0] // JOINT_BATCH_PER_CITY)
    total_steps = EPOCHS_JOINT * steps_per_epoch
    gstep = 0

    if fixed:
        _, Xv_s, Yv_s, Cv_s = make_city_loader_masked(src, "val", 64, False)
    else:
        _, Xv_s, Yv_s = make_city_loader(src, src.train_end_idx, src.val_end_idx, 64, False); Cv_s = None
    best_val = -1e9
    joint_ckpt = MODELS_DIR / f"{backbone}_rsd_joint_{source}_to_{target}{'_fixed' if fixed else ''}.pt"

    print(f"\n[Dual-RSD {source}->{target}@{int(d_pct*100)}] joint {EPOCHS_JOINT}ep x {steps_per_epoch} "
          f"(alpha_m={alpha_m}, beta_rsd={beta_rsd})")
    for ep in range(EPOCHS_JOINT):
        encoder.train(); marg.train()
        sY = sM = sR = 0.0
        for _ in range(steps_per_epoch):
            lam = lambda_progress(gstep, total_steps)
            mb, cursors = sample_minibatch(per_city, indices, cursors, JOINT_BATCH_PER_CITY, rng)
            loss_y = torch.tensor(0.0); loss_m = torch.tensor(0.0); loss_r = torch.tensor(0.0)
            z_cache = {}
            for name in mb:
                xb, yb = mb[name]
                xb_t = torch.from_numpy(xb).float()
                ei, ew = edges[name]
                y_hat, z = encoder(xb_t, ei, ew, return_embedding=True)
                z_cache[name] = z
                lab = torch.full((xb_t.size(0),), CITY_TO_LABEL[name], dtype=torch.long)
                loss_m = loss_m + ce(marg(z, lam), lab)
                if name == source:
                    loss_y = loss_y + mse(y_hat, torch.from_numpy(yb).float())
            # RSD between source and target embedding subspaces.
            if source in z_cache and target in z_cache:
                loss_r = rsd_loss(z_cache[source], z_cache[target])
            loss = loss_y + alpha_m * loss_m + beta_rsd * loss_r
            if not torch.isfinite(loss):
                continue  # skip pathological step rather than poison the weights
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step()
            sY += float(loss_y.detach()); sM += float(loss_m.detach()); sR += float(loss_r.detach()); gstep += 1

        vm = evaluate_gnn(encoder, Xv_s, Yv_s, edges[source][0], edges[source][1],
                          src.target_scaler, climatology=Cv_s)
        if vm["R2"] > best_val + 1e-4:
            best_val = vm["R2"]; save_checkpoint(encoder, joint_ckpt, extra={"src_val_r2": best_val}); flag = "*"
        else:
            flag = " "
        print(f"  ep {ep+1:02d}/{EPOCHS_JOINT} {flag} Ly={sY/steps_per_epoch:.4f} "
              f"Lm={sM/steps_per_epoch:.3f} Lrsd={sR/steps_per_epoch:.3f} lam={lam:.2f} src_val_R2={vm['R2']:+.4f}")

    load_checkpoint(encoder, joint_ckpt)

    ei_t, ew_t = edges[target]
    if fixed:
        _, Xv, Yv, Cv = make_city_loader_masked(tgt, "val", 64, False)
        _, Xte, Yte, Cte = make_city_loader_masked(tgt, "test", 64, False)
    else:
        _, Xv, Yv = make_city_loader(tgt, tgt.train_end_idx, tgt.val_end_idx, 64, False)
        _, Xte, Yte = make_city_loader(tgt, tgt.val_end_idx, tgt.test_end_idx, 64, False); Cv = Cte = None

    zs = evaluate_gnn(encoder, Xte, Yte, ei_t, ew_t, tgt.target_scaler, climatology=Cte)
    print(f"  zero-shot (RSD encoder, no FT) test = R2={zs['R2']:.4f} MAE={zs['MAE']:.3f}")

    cfg = FIXED_FT_CFG[target]
    freeze_for = int(cfg["epochs"] * FREEZE_FRAC)
    if fixed:
        Xf, Yf, _ = make_windows_masked(tgt.feature_tensor, tgt.target_tensor, tgt.train_mask,
                                        history=HISTORY, horizon=HORIZON, climatology_tensor=tgt.climatology_tensor)
    else:
        Xf, Yf = make_windows(tgt.feature_tensor, tgt.target_tensor, HISTORY, HORIZON, 0, tgt.train_end_idx)
    rng_ft = np.random.default_rng(42)
    keep = rng_ft.choice(Xf.shape[0], size=max(1, int(d_pct * Xf.shape[0])), replace=False); keep.sort()
    ds = torch.utils.data.TensorDataset(torch.from_numpy(Xf[keep]).float(), torch.from_numpy(Yf[keep]).float())
    loader = torch.utils.data.DataLoader(ds, batch_size=cfg["batch_size"], shuffle=True)
    ft_opt = torch.optim.Adam(encoder.parameters(), lr=cfg["lr"], weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(ft_opt, mode="max", factor=0.5, patience=3)
    if freeze_for > 0:
        for p in encoder.t1.parameters():
            p.requires_grad = False
    best_val = -1e9; patience = cfg["patience"]
    tl_ckpt = MODELS_DIR / f"{backbone}_rsd_{source}_to_{target}_d{int(d_pct*100)}{'_fixed' if fixed else ''}.pt"
    for ep in range(cfg["epochs"]):
        if freeze_for > 0 and ep == freeze_for:
            for p in encoder.t1.parameters():
                p.requires_grad = True
        encoder.train()
        for xb, yb in loader:
            ft_opt.zero_grad()
            y_hat = encoder(xb, ei_t, ew_t)
            loss = mse(y_hat, yb); loss.backward()
            torch.nn.utils.clip_grad_norm_(encoder.parameters(), 1.0); ft_opt.step()
        vm = evaluate_gnn(encoder, Xv, Yv, ei_t, ew_t, tgt.target_scaler, climatology=Cv)
        sched.step(vm["R2"])
        if vm["R2"] > best_val + 1e-4:
            best_val = vm["R2"]; save_checkpoint(encoder, tl_ckpt, extra={"val_r2": best_val}); patience = cfg["patience"]
        else:
            patience -= 1
        if patience is not None and patience <= 0:
            break
    load_checkpoint(encoder, tl_ckpt)
    tl = evaluate_gnn(encoder, Xte, Yte, ei_t, ew_t, tgt.target_scaler, climatology=Cte)
    print(f"  ==> RSD transfer test = R2={tl['R2']:.4f} MAE={tl['MAE']:.3f} (zero_shot={zs['R2']:.4f})")
    return {"zero_shot": zs, "transfer": tl, "real_transfer": tl["R2"] > zs["R2"],
            "alpha_m": alpha_m, "beta_rsd": beta_rsd}


def main(args):
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    cities = load_all_cities(fixed=args.fixed)
    if args.pilot:
        pairs = [("Delhi", "Kolkata"), ("Delhi", "Guwahati"), ("Kolkata", "Guwahati")]
        d_values = [0.30]
        out_name = f"variantD_rsd_{args.backbone}_pilot{'_fixed' if args.fixed else ''}.json"
    else:
        pairs = [("Delhi", "Kolkata"), ("Delhi", "Guwahati"), ("Kolkata", "Delhi"),
                 ("Kolkata", "Guwahati"), ("Guwahati", "Delhi"), ("Guwahati", "Kolkata")]
        d_values = [0.15, 0.30, 0.45, 0.60]
        out_name = f"variantD_rsd_{args.backbone}{'_fixed' if args.fixed else ''}.json"
    results: Dict[str, Dict] = {}
    for s, t in pairs:
        for d in d_values:
            results[f"{s}->{t}@{int(d*100)}"] = train_graph_rsd(
                cities, s, t, d, backbone=args.backbone, fixed=args.fixed,
                alpha_m=args.alpha_m, beta_rsd=args.beta_rsd)
            write_results(RESULTS_DIR / out_name, results)
    print(f"\nwrote {RESULTS_DIR / out_name}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--backbone", choices=["gat", "sage"], default="gat")
    p.add_argument("--fixed", action="store_true")
    p.add_argument("--pilot", action="store_true")
    p.add_argument("--alpha-m", type=float, default=ALPHA_M)
    p.add_argument("--beta-rsd", type=float, default=BETA_RSD)
    main(p.parse_args())
