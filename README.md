# GNN-TL — Graph Neural Network Transfer Learning for PM2.5 Forecasting across Indian Cities

Cross-city PM2.5 forecasting under heterogeneous station topologies (Delhi 40 / Kolkata 10 / Guwahati 4) via an inductive Spatio-Temporal GNN (GAT + dilated TCN) with a two-phase transfer pipeline: supervised pre-train + fine-tune (PT-FT) and adversarial graph-level domain adaptation (Graph-DANN). The same `|V|`-independent encoder operates across all three city graphs without any parameter-shape change.

> The full technical report is in [reports/MAIN_REPORT.md](reports/MAIN_REPORT.md); detailed numerical results in [reports/RESULTS.md](reports/RESULTS.md); the journal-style manuscript in [reports/PAPER_DRAFT.md](reports/PAPER_DRAFT.md); the implementation/mechanism deep-dive in [reports/ARCHITECTURE.md](reports/ARCHITECTURE.md); the data-leakage & overfitting audit in [reports/AUDIT.md](reports/AUDIT.md); the bibliography in [reports/REFERENCES.md](reports/REFERENCES.md).

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
│   │   ├── lstm_baseline.py      Station-independent LSTM baseline
│   │   ├── stgnn_gat.py          Inductive GAT + TCN
│   │   ├── stgnn_sage.py         Inductive GraphSAGE + TCN
│   │   └── dann.py               Gradient Reversal Layer + city discriminator
│   ├── train_lstm.py             LSTM source-only + TL trainer
│   ├── train_gnn.py              GNN source-only trainer
│   ├── train_gnn_tl.py           GNN-TL Phase 1 PT-FT (with zero-shot + scratch verification)
│   ├── train_gnn_dann.py         GNN-TL Phase 2 Graph-DANN
│   └── evaluate.py
├── dataset/                      CPCB raw + processed CSVs (raw not redistributed)
├── models/                       saved checkpoints (.pt)
├── results/                      result JSONs (lstm/, gnn/, gnn_tl/)
├── analysis/                     diagnostic + comparison scripts
├── reports/                      written reports (MAIN_REPORT, RESULTS, PAPER_DRAFT, ARCHITECTURE, AUDIT, REFERENCES)
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

# 2) Final protocol (interleaved split + per-city recipe + climatology residual)
python -u -m src.train_lstm    --mode all --fixed              # LSTM baseline (source-only + TL)
python -u -m src.train_gnn     --backbone gat --fixed          # GNN source-only
python -u -m src.train_gnn_tl  --backbone gat --fixed --full   # Phase 1: 24-cell PT-FT + verification
python -u -m src.train_gnn_dann --backbone gat --fixed         # Phase 2: 24-cell Graph-DANN

# 3) Regenerate tabular report
python reports/generate_results_table.py
```

Total wall-clock on 8 vCPU / 16 GB / no GPU: ≈ 8 hours end-to-end for the full fixed-protocol campaign (both transfer phases).

## Data access

Raw CPCB station data is **not redistributed** here — it is governed by CPCB's own terms. Place the raw CSVs into `dataset/` matching the names in [src/data_pipeline.py](src/data_pipeline.py) and run preprocessing.



## License

MIT for code. Raw CPCB data falls under CPCB's terms.

