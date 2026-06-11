# GNN-TL — Transfer Learning over Sparse, Heterogeneous Sensor Networks for PM2.5 Forecasting

Short-horizon PM2.5 forecasting across three real, order-of-magnitude-different sensor deployments (Delhi 40 / Kolkata 10 / Guwahati 4 nodes), framed as a **sparse-IoT-sensor-network** problem. We compare a station-independent **LSTM** against an inductive, `|V|`-independent **Spatio-Temporal GNN** under transfer learning, and ask honestly *what helps*: **transfer learning (yes, decisively)**, **spatial coupling (not for forecasting, but clearly for virtual sensing at unsensored nodes)**, and **transport physics (no — a rigorous negative)**.

> The CPCB CAAQM network is treated as a real heterogeneous IoT sensor deployment; PM2.5 is the signal, **sensor sparsity is the problem**.

## The story (one coherent spine)

1. **Baselines** — station-independent **LSTM** vs spatial **GNN**, source-only per city. *Finding: LSTM ≥ GNN.*
2. **Transfer learning** — **LSTM-TL** and **GNN-TL** (pre-train + fine-tune): a sparse target deployment inherits a forecaster from a dense source. *Finding: TL is the decisive lever; a well-tuned LSTM-TL is as good as or better than GNN-TL.*
3. **Virtual sensing + physics probe** — the GNN's genuine edge: estimating PM2.5 at **unsensored** locations by borrowing from neighbours, beating a met-only LSTM by **+0.08 to +0.16 R²** at the unsensored nodes (the LSTM has no spatial mechanism). A physics-informed advection–diffusion prior (GNN-PINN) is tested as the candidate mechanism and found **not** to help — a well-controlled negative.

Reports: **consolidated technical report** [reports/MAIN_REPORT.md](reports/MAIN_REPORT.md); mathematical audit [reports/MATH_AUDIT.md](reports/MATH_AUDIT.md); data-leakage audit [reports/AUDIT.md](reports/AUDIT.md); sensor-sparsity & physics positioning [reports/SENSOR_EFFICIENCY_PINN_LITERATURE.md](reports/SENSOR_EFFICIENCY_PINN_LITERATURE.md); ADR physics derivation [reports/PINN_PHYSICS.md](reports/PINN_PHYSICS.md); physics ablation write-up [reports/PHYSICS_ABLATION_SECTION.md](reports/PHYSICS_ABLATION_SECTION.md); bibliography [reports/REFERENCES.md](reports/REFERENCES.md).

---

## Tech stack

- Python 3.10+, PyTorch ≥ 2.1 (CPU-only — no CUDA / PyG; GAT and SAGE layers implemented from scratch in [src/models/](src/models/))
- NumPy, Pandas, scikit-learn (preprocessing/metrics), SciPy (significance tests)
- Pinned in [requirements.txt](requirements.txt)

## Repository layout

```
GNN_TL/
├── src/
│   ├── data_pipeline.py          CSV → per-city tensors + metadata
│   ├── graph_construction.py     k-NN / wind / hybrid adjacency builders
│   ├── utils.py                  splits, windowing, climatology, scalers, metrics
│   ├── models/
│   │   ├── lstm_baseline.py      station-independent LSTM
│   │   ├── stgnn_gat.py          inductive GAT + TCN encoder
│   │   ├── stgnn_sage.py         inductive GraphSAGE + TCN (backbone ablation)
│   │   └── physics.py            graph advection–diffusion (ADR) operators + residual
│   ├── train_lstm.py             LSTM source-only + LSTM-TL
│   ├── train_gnn.py              GNN source-only
│   ├── train_gnn_tl.py           GNN-TL (pre-train + fine-tune)
│   ├── train_gnn_pinn.py         GNN-PINN-TL (physics-informed; rigorous-negative ablation)
│   ├── train_gnn_vsense.py       virtual sensing (estimate PM2.5 at unsensored nodes)
│   └── evaluate.py               robustness battery (seasonal / cross-city / per-station)
├── analysis/
│   ├── sensor_budget.py          node-dropping sensor-budget experiment
│   └── vsense_ablation.py        virtual-sensing multi-seed + wind-vs-smoothness ablation
├── dataset/                      CPCB raw + processed CSVs (raw not redistributed)
├── models/                       checkpoints: lstm/ gnn/ gnn_tl/ gnn_pinn/ gnn_vsense/
├── results/                      result JSONs: lstm/ gnn/ gnn_tl/ gnn_pinn/ gnn_vsense/
└── reports/                      written reports (see links above)
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Reproducing the experiments

```bash
# 1) Preprocess raw CPCB CSVs → dataset/processed/*.csv + metadata.json
python -m src.data_pipeline

# 2) Baselines + transfer (fixed protocol: interleaved split + climatology residual)
python -u -m src.train_lstm   --mode all --fixed          # LSTM source-only + LSTM-TL
python -u -m src.train_gnn    --backbone gat --fixed       # GNN source-only
python -u -m src.train_gnn_tl --backbone gat --fixed --full  # GNN-TL (24-cell grid + 3-way verification)

# 3) Physics-informed probe + sensor-sparsity (the "more value from fewer sensors" study)
python -u -m src.train_gnn_pinn   --fixed --full           # GNN-PINN-TL ablation
python -u -m analysis.sensor_budget                        # node-dropping sensor-budget
python -u -m src.train_gnn_vsense --fixed                  # virtual sensing at unsensored nodes
python -u -m analysis.vsense_ablation                      # multi-seed + wind-vs-smoothness ablation

# 4) Robustness battery + tabular report
python -u -m src.evaluate
python reports/generate_results_table.py
```

## Data access

Raw CPCB station data is **not redistributed** here — it is governed by CPCB's own terms. Place the raw CSVs into `dataset/` matching the names in [src/data_pipeline.py](src/data_pipeline.py) and run preprocessing.

## License

MIT for code. Raw CPCB data falls under CPCB's terms.
