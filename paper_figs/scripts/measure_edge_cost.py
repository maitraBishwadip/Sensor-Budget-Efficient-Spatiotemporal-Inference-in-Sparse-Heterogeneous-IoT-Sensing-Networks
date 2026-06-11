"""Measure the inductive ST-GNN encoder's edge-deployment cost.

Reports parameter count, on-disk footprint, and single-inference CPU latency for the
three deployment sizes (N = 4, 10, 40 sensor nodes), substantiating the claim that one
|V|-independent model runs on commodity/edge CPUs. Writes results/edge_cost.json.

Run:  python paper_figs/scripts/measure_edge_cost.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from src.graph_construction import build_city_graph
from src.train_gnn import build_backbone, HISTORY, GRAPH_STRATEGY, KNN_K

F = 14                      # feature dimension
SIZES = [4, 10, 40]         # Guwahati, Kolkata, Delhi
REPEATS, WARMUP = 200, 20
torch.manual_seed(0)
np.random.seed(0)


def synth_coords(n: int) -> list:
    """A plausible compact lat/lon cluster so the k-NN graph is well-formed."""
    rng = np.random.default_rng(0)
    lat = 28.5 + 0.3 * rng.random(n)
    lon = 77.0 + 0.3 * rng.random(n)
    return list(zip(lat.tolist(), lon.tolist()))


def main() -> None:
    model = build_backbone("gat", F, fixed=True).eval()
    n_params = sum(p.numel() for p in model.parameters())
    size_kb = n_params * 4 / 1024.0      # float32

    out = {"n_params": int(n_params), "size_kb": round(size_kb, 1), "history": HISTORY,
           "features": F, "device": "cpu", "torch": torch.__version__, "per_size": {}}

    print(f"ST-GNN encoder: {n_params:,} params  ({size_kb:.1f} KB float32)")
    print(f"{'N nodes':>8} | {'latency (ms)':>14} | {'throughput (graphs/s)':>22}")
    print("-" * 50)
    for n in SIZES:
        ei, ew = build_city_graph(synth_coords(n), strategy=GRAPH_STRATEGY, k=KNN_K)
        x = torch.randn(1, HISTORY, n, F)
        with torch.no_grad():
            for _ in range(WARMUP):
                model(x, ei, ew)
            t0 = time.perf_counter()
            for _ in range(REPEATS):
                model(x, ei, ew)
            dt = (time.perf_counter() - t0) / REPEATS
        out["per_size"][n] = {"latency_ms": round(dt * 1e3, 3), "throughput_hz": round(1.0 / dt, 1)}
        print(f"{n:>8} | {dt*1e3:>14.3f} | {1.0/dt:>22.1f}")

    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "edge_cost.json").write_text(json.dumps(out, indent=2))
    print(f"\nwrote {ROOT / 'results' / 'edge_cost.json'}")


if __name__ == "__main__":
    main()
