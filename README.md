# GNN-TL — Graph Transfer Learning for PM2.5 Forecasting across Indian Cities

Cross-city PM2.5 forecasting under heterogeneous station topologies (Delhi 40 / Kolkata 10 / Guwahati 4) via an inductive Spatio-Temporal GNN (GAT + dilated TCN) with adversarial graph-level domain adaptation (Graph-DANN). Extends the LSTM-TL B.Tech thesis (Sanjeev, Prakash & Maitra, 2025) to the graph setting.

> Detailed numerical results live in [reports/RESULTS.md](reports/RESULTS.md); the full technical narrative is in [reports/MAIN_REPORT.md](reports/MAIN_REPORT.md); the consolidated research proposal is in [reports/ResearchProposal.md](reports/ResearchProposal.md); the bibliography is in [reports/REFERENCES.md](reports/REFERENCES.md); the architectural deep-dive of the GNN / GNN-TL implementation is in [reports/ARCHITECTURE.md](reports/ARCHITECTURE.md).

---

## Tech stack

- Python 3.10+, PyTorch ≥ 2.1 (CPU-only — no CUDA / PyG required; GAT and SAGE layers implemented from scratch in [src/models/](src/models/))
- NumPy, Pandas, scikit-learn for preprocessing and metrics
- Pinned in [requirements.txt](requirements.txt)

## Repository layout

```
GNN_TL/
├── src/                          source code
│   ├── data_pipeline.py          CSV → per-city tensors + metadata
│   ├── graph_construction.py     k-NN / wind / hybrid adjacency builders
│   ├── utils.py                  splits, windowing, climatology, scalers, metrics
│   ├── models/
│   │   ├── lstm_baseline.py      Stage I LSTM
│   │   ├── stgnn_gat.py          Inductive GAT + TCN
│   │   ├── stgnn_sage.py         Inductive GraphSAGE + TCN
│   │   └── dann.py               Gradient Reversal Layer + city discriminator
│   ├── train_lstm.py             LSTM source-only + TL trainer
│   ├── train_gnn.py              GNN source-only trainer
│   ├── train_gnn_tl.py           GNN-TL Variant A (with zero-shot + scratch verification)
│   ├── train_gnn_dann.py         GNN-TL Variant B (Graph-DANN)
│   └── evaluate.py
├── dataset/                      CPCB raw + processed CSVs (raw not redistributed)
├── models/                       saved checkpoints (.pt)
├── results/                      result JSONs (lstm/, gnn/, gnn_tl/)
├── analysis/                     diagnostic + comparison scripts
├── reports/                      written reports (MAIN_REPORT, RESULTS, REFERENCES, ResearchProposal, ARCHITECTURE)
└── requirements.txt
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

# 2) Legacy protocol (chronological split — the broken baseline)
python -u -m src.train_lstm --mode all
python -u -m src.train_gnn  --backbone gat

# 3) Fixed protocol (interleaved split + matched recipe + climatology residual)
python -u -m src.train_lstm --mode all --fixed
python -u -m src.train_gnn  --backbone gat --fixed
python -u -m src.train_gnn_tl --backbone gat --fixed --full   # 24-cell verification

# 4) Regenerate tabular report
python reports/generate_results_table.py
```

Total wall-clock on 8 vCPU / 16 GB / no GPU: ≈ 2.5 hours for the full fixed-protocol campaign.

## Data access

Raw CPCB station data is **not redistributed** here — it is governed by CPCB's own terms. Place the raw CSVs into `dataset/` matching the names in [src/data_pipeline.py](src/data_pipeline.py) and run preprocessing.



## License

MIT for code. Raw CPCB data falls under CPCB's terms.

