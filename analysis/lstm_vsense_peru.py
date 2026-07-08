"""Rigorous per-U LSTM virtual-sensing baseline.

Trains a met-only LSTM on the k SENSORED stations and evaluates at the EXACT N-k UNsensored
stations used by the GNN virtual-sensing run (same node split, seed=0; configs match
results/gnn_vsense/vsense_gat.json: Delhi k=8, Delhi k=16, Kolkata k=4). This makes the
GNN-vs-LSTM comparison at unsensored nodes apples-to-apples: both train on the sensored set and
are scored on the identical unsensored set; the GNN borrows PM2.5 from neighbours via the graph,
the LSTM (station-independent) can use only local meteorology.

With --chrono, uses the chronological block split (Delhi only; the one-year
deployments are untrainable under it — paper §VI-F) and compares against
results/gnn_vsense/vsense_gat_chrono.json.

Run: python -u -m analysis.lstm_vsense_peru [--chrono]
"""
from __future__ import annotations

import argparse
import copy
import dataclasses
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.lstm_baseline import LSTMForecaster                                  # noqa: E402
from src.train_lstm import (                                                          # noqa: E402
    CITY_TRAIN_CFG, DROPOUT, HIDDEN, LAYERS, WEIGHT_DECAY,
    build_station_dataset_masked, evaluate, load_all_cities, to_loader, train_one_epoch,
)
from src.utils import CityTensors, write_results                                      # noqa: E402

PM25 = 0
CONFIGS = [("Delhi", 8), ("Delhi", 16), ("Kolkata", 4)]   # match results/gnn_vsense/vsense_gat.json


def subset(city: CityTensors, idx) -> CityTensors:
    idx = list(idx)
    clim = None if city.climatology_tensor is None else city.climatology_tensor[:, idx]
    return dataclasses.replace(
        city, feature_tensor=city.feature_tensor[:, idx, :], target_tensor=city.target_tensor[:, idx],
        stations=[city.stations[i] for i in idx], coords=[city.coords[i] for i in idx],
        climatology_tensor=clim)


def maskpm(X: np.ndarray) -> np.ndarray:
    X = X.copy(); X[:, :, PM25] = 0.0; return X       # met-only (no sensor at this location)


def main(chrono: bool = False):
    tag = "_chrono" if chrono else ""
    configs = [("Delhi", 8), ("Delhi", 16)] if chrono else CONFIGS
    cities = load_all_cities(fixed=True, chrono=chrono)
    gnn_path = Path(f"results/gnn_vsense/vsense_gat{tag}.json")
    gnn = json.load(open(gnn_path)) if gnn_path.exists() else {}
    out_path = Path(f"results/lstm/lstm_vsense_peru{tag}.json")
    out = {}
    for cname, k in configs:
        city, cfg = cities[cname], CITY_TRAIN_CFG[cname]
        N = city.feature_tensor.shape[1]
        perm = np.random.default_rng(0).permutation(N)        # SAME split as GNN virtual sensing
        S, U = np.sort(perm[:k]), np.sort(perm[k:])
        cS, cU = subset(city, S), subset(city, U)
        Xtr, Ytr, _ = build_station_dataset_masked(cS, "train", subsample_max=cfg["train_subsample_max"])
        Xva, Yva, Cva = build_station_dataset_masked(cU, "val")
        Xte, Yte, Cte = build_station_dataset_masked(cU, "test")
        Xtr, Xva, Xte = maskpm(Xtr), maskpm(Xva), maskpm(Xte)

        F = city.feature_tensor.shape[-1]
        model = LSTMForecaster(F, HIDDEN, LAYERS, DROPOUT)
        opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=WEIGHT_DECAY)
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=3)
        loss_fn = nn.MSELoss()
        loader = to_loader(Xtr, Ytr, cfg["batch_size"], True)
        best, best_state, pat = -1e9, None, cfg["patience"]
        for ep in range(cfg["epochs"]):
            train_one_epoch(model, loader, opt, loss_fn)
            vm = evaluate(model, Xva, Yva, city.target_scaler, climatology=Cva)
            sched.step(vm["R2"])
            if vm["R2"] > best + 1e-4:
                best, best_state, pat = vm["R2"], copy.deepcopy(model.state_dict()), cfg["patience"]
            else:
                pat -= 1
            print(f"  [{cname} k={k} LSTM@U] ep {ep+1:02d}/{cfg['epochs']} U-val R2={vm['R2']:+.4f} best={best:+.4f}")
            if pat <= 0:
                break
        if best_state is not None:
            model.load_state_dict(best_state)
        te = evaluate(model, Xte, Yte, city.target_scaler, climatology=Cte)
        g = gnn.get(f"{cname}|k={k}|lam=0.0", {}).get("R2")
        out[f"{cname}|k={k}"] = {"lstm_metonly_U": te, "gnn_U": g,
                                 "n_sensored": int(k), "n_unsensored": int(len(U))}
        write_results(out_path, out)
        print(f"  ==> {cname} k={k}: LSTM met-only @U test R2={te['R2']:.4f} MAE={te['MAE']:.3f} "
              f"(predict {len(U)} unsensored)")

    print("\n" + "=" * 64)
    print("PER-U VIRTUAL SENSING — GNN (neighbours) vs LSTM (met-only), same unsensored nodes")
    print("=" * 64)
    print("  config            LSTM@U   GNN@U    GNN-LSTM")
    for cname, k in configs:
        r = out[f"{cname}|k={k}"]
        l, g = r["lstm_metonly_U"]["R2"], r["gnn_U"]
        diff = "n/a" if g is None else f"{g - l:+.4f}"
        gs = "n/a" if g is None else f"{g:.4f}"
        print(f"  {cname+' k='+str(k):16s}  {l:.4f}   {gs}   {diff}")
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--chrono", action="store_true",
                   help="Chronological block split (Delhi configs only).")
    main(chrono=p.parse_args().chrono)
