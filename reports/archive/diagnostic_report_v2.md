# Diagnostic Report v2 — After Fixes A + B + C

Fixes applied (see analysis/diagnostic_report.md for the why):

- **A: Interleaved 70/15/15 split.** Every Nth window goes to val/test, so each split spans the full year. Fixes the Kolkata/Guwahati 1-year-of-data trap where the winter peak was entirely in the test set.
- **B: GNN recipe matches LSTM.** Hidden=64 (was 24), epochs 25/40/60 with early stopping (was 10 fixed), per-city LR + batch size, no aggressive subsample cap.
- **C: Climatology residual.** Per-(month, hour) climatology computed from train slice; model predicts deviations from seasonal climatology; climatology added back at inference. AdaRNN-style temporal DA.

Run with `--fixed` flag on `src.train_lstm` and `src.train_gnn`. Legacy numbers stay in the un-suffixed result files; fixed numbers land in `*_fixed.json`.

## 1. Source-only (Table 5.1 analog)

| City | R² LSTM legacy | R² LSTM fixed | Δ R² | MAE LSTM legacy | MAE LSTM fixed | R² GNN legacy | R² GNN fixed | Δ R² GNN |
|------|--:|--:|--:|--:|--:|--:|--:|--:|
| Delhi | +0.845 | +0.850 | ++0.005 | 24.10 | 23.73 | +0.719 | +0.834 | ++0.115 |
| Kolkata | +0.786 | +0.863 | ++0.076 | 12.14 | 7.78 | +0.334 | +0.824 | ++0.491 |
| Guwahati | +0.414 | +0.832 | ++0.418 | 21.90 | 12.19 | -0.220 | +0.831 | ++1.051 |

## 2. Transfer learning (Table 5.2 analog)

### LSTM-TL

| Pair@d% | R² legacy | R² fixed | Δ R² | MAE legacy | MAE fixed |
|---|--:|--:|--:|--:|--:|
| Delhi->Guwahati@15 | +0.397 | +0.815 | +0.418 | 22.11 | 13.65 |
| Delhi->Guwahati@30 | +0.415 | +0.822 | +0.407 | 21.53 | 12.87 |
| Delhi->Guwahati@45 | +0.403 | +0.824 | +0.421 | 22.16 | 12.75 |
| Delhi->Guwahati@60 | +0.438 | +0.822 | +0.384 | 20.91 | 12.91 |
| Delhi->Kolkata@15 | +0.726 | +0.844 | +0.118 | 14.13 | 9.17 |
| Delhi->Kolkata@30 | +0.742 | +0.852 | +0.110 | 13.63 | 8.82 |
| Delhi->Kolkata@45 | +0.766 | +0.855 | +0.089 | 12.81 | 8.49 |
| Delhi->Kolkata@60 | +0.771 | +0.857 | +0.086 | 12.73 | 8.37 |
| Guwahati->Delhi@15 | +0.843 | +0.845 | +0.003 | 24.16 | 24.33 |
| Guwahati->Delhi@30 | +0.847 | +0.849 | +0.002 | 24.39 | 23.79 |
| Guwahati->Delhi@45 | +0.846 | +0.852 | +0.006 | 24.10 | 23.57 |
| Guwahati->Delhi@60 | +0.843 | +0.852 | +0.009 | 24.27 | 23.51 |
| Guwahati->Kolkata@15 | +0.754 | +0.826 | +0.073 | 13.85 | 9.49 |
| Guwahati->Kolkata@30 | +0.769 | +0.842 | +0.073 | 13.21 | 9.01 |
| Guwahati->Kolkata@45 | +0.767 | +0.849 | +0.082 | 13.42 | 8.78 |
| Guwahati->Kolkata@60 | +0.772 | +0.850 | +0.078 | 13.06 | 8.78 |
| Kolkata->Delhi@15 | +0.845 | +0.845 | -0.000 | 24.31 | 24.22 |
| Kolkata->Delhi@30 | +0.848 | +0.851 | +0.003 | 23.91 | 23.66 |
| Kolkata->Delhi@45 | +0.848 | +0.852 | +0.004 | 23.77 | 23.51 |
| Kolkata->Delhi@60 | +0.846 | +0.853 | +0.007 | 24.13 | 23.41 |
| Kolkata->Guwahati@15 | +0.397 | +0.809 | +0.413 | 21.83 | 13.69 |
| Kolkata->Guwahati@30 | +0.449 | +0.813 | +0.364 | 21.48 | 13.12 |
| Kolkata->Guwahati@45 | +0.431 | +0.820 | +0.389 | 21.21 | 12.84 |
| Kolkata->Guwahati@60 | +0.463 | +0.821 | +0.358 | 20.92 | 12.53 |

### GNN-TL Variant A (transfer with verification)

Each fixed-protocol cell ran three trainings on the *same* d% target sample:
(0) **zero-shot** — load source weights, no fine-tune; (1) **transfer** — load source weights, fine-tune; (2) **scratch** — random init, fine-tune. Knowledge transfer is "real" iff transfer > zero_shot AND transfer > scratch.

| Pair@d% | R2 legacy | R2 transfer (fixed) | R2 zero-shot | R2 scratch | gain vs scratch | real_transfer |
|---|--:|--:|--:|--:|--:|:-:|
| Delhi->Guwahati@15 | - | +0.823 | +0.684 | +0.776 | +0.048 | YES |
| Delhi->Guwahati@30 | - | +0.816 | +0.684 | +0.783 | +0.034 | YES |
| Delhi->Guwahati@45 | - | +0.827 | +0.684 | +0.789 | +0.039 | YES |
| Delhi->Guwahati@60 | - | +0.822 | +0.684 | +0.808 | +0.014 | YES |
| Delhi->Kolkata@15 | - | +0.811 | +0.728 | +0.758 | +0.054 | YES |
| Delhi->Kolkata@30 | - | +0.816 | +0.728 | +0.780 | +0.035 | YES |
| Delhi->Kolkata@45 | - | +0.818 | +0.728 | +0.793 | +0.026 | YES |
| Delhi->Kolkata@60 | - | +0.817 | +0.728 | +0.803 | +0.013 | YES |
| Guwahati->Delhi@15 | - | +0.789 | +0.632 | +0.764 | +0.025 | YES |
| Guwahati->Delhi@30 | - | +0.803 | +0.632 | +0.790 | +0.013 | YES |
| Guwahati->Delhi@45 | - | +0.812 | +0.632 | +0.799 | +0.013 | YES |
| Guwahati->Delhi@60 | - | +0.817 | +0.632 | +0.810 | +0.007 | YES |
| Guwahati->Kolkata@15 | - | +0.795 | +0.698 | +0.758 | +0.038 | YES |
| Guwahati->Kolkata@30 | - | +0.809 | +0.698 | +0.780 | +0.028 | YES |
| Guwahati->Kolkata@45 | - | +0.810 | +0.698 | +0.793 | +0.018 | YES |
| Guwahati->Kolkata@60 | - | +0.818 | +0.698 | +0.803 | +0.015 | YES |
| Kolkata->Delhi@15 | - | +0.795 | +0.603 | +0.764 | +0.032 | YES |
| Kolkata->Delhi@30 | - | +0.808 | +0.603 | +0.790 | +0.018 | YES |
| Kolkata->Delhi@45 | - | +0.820 | +0.603 | +0.799 | +0.021 | YES |
| Kolkata->Delhi@60 | - | +0.823 | +0.603 | +0.810 | +0.012 | YES |
| Kolkata->Guwahati@15 | - | +0.795 | +0.670 | +0.781 | +0.015 | YES |
| Kolkata->Guwahati@30 | - | +0.804 | +0.670 | +0.792 | +0.012 | YES |
| Kolkata->Guwahati@45 | - | +0.804 | +0.670 | +0.797 | +0.007 | YES |
| Kolkata->Guwahati@60 | - | +0.816 | +0.670 | +0.798 | +0.018 | YES |

**Verification result: 24/24 cells show real knowledge transfer.**

## 3. Conclusions

Below conclusions are auto-derived from the tables above.

- LSTM Guwahati source-only R² changed by **+0.418** after the fixes.
- GNN  Guwahati source-only R² changed by **+1.051** after the fixes.

### Interpretation guide

- If the GNN now beats LSTM on Guwahati under the fixed protocol, the architecture hypothesis (ST-GNN > LSTM) is supported.
- If the LSTM and GNN are close, the split + climatology + recipe was indeed the limiting factor, and the architectural gain alone is smaller than expected.
- If the GNN still underperforms, the next experiments to run are E (GraphNorm) and F (subgraph sampling on Delhi pre-training).
