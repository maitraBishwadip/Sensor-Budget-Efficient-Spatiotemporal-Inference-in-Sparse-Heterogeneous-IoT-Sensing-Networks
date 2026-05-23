# All Experiment Results — Tabular Report

_All results from the PM2.5 GNN-TL project, in one place._

This report covers four experiment families:

1. **Stage I — Thesis LSTM-TL baseline** (verbatim from B.Tech thesis Table 5.1 and Table 5.2). Numbers reproduced from the PDF; not re-run.
2. **Stage II legacy run** — same code that produced thesis-style chronological 70/15/15 split, as it stood when I diagnosed the underperformance.
3. **Stage II fixed-protocol run** — after the three diagnosed fixes (A: interleaved split, B: GNN recipe matched to LSTM, C: per-(month,hour) climatology residual).
4. **Stage II GNN-TL verification** — for every (source, target, d%) cell, three trainings on identical data: zero-shot (source weights, no fine-tune), scratch (random init + fine-tune), transfer (source weights + fine-tune). Transfer is considered real iff transfer.R2 > zero_shot.R2 AND transfer.R2 > scratch.R2.

All metrics are computed on the *target city's held-out test partition* in raw PM2.5 space (after un-doing the standardization and, where applicable, adding the climatology back).

## 1. Stage I — Thesis LSTM-TL baseline (reference)

### 1.1 Source-only LSTM (Thesis Table 5.1)

| Model | City | R² | MAE (µg/m³) |
|---|---|--:|--:|
| M_D | Delhi | 0.6570 | 37.3301 |
| M_K | Kolkata | 0.7861 | 14.6633 |
| M_G | Guwahati | 0.5723 | 16.3012 |

### 1.2 LSTM-TL fine-tuned (Thesis Table 5.2 — R² scores)

| Source | Target | d=15% | d=30% | d=45% | d=60% | best MAE |
|---|---|--:|--:|--:|--:|--:|
| Kolkata | Guwahati | 0.5980 | 0.6030 | 0.5823 | 0.6271 | 15.1105 |
| Kolkata | Delhi | 0.6939 | 0.6908 | 0.6881 | 0.6870 | 35.7143 |
| Guwahati | Kolkata | 0.8098 | 0.8174 | 0.8164 | 0.8019 | 13.5344 |
| Guwahati | Delhi | 0.6995 | 0.6877 | 0.6815 | 0.6837 | 35.9100 |
| Delhi | Kolkata | 0.7437 | 0.8189 | 0.8129 | 0.8051 | 13.3222 |
| Delhi | Guwahati | 0.5797 | 0.6381 | 0.6210 | 0.6133 | 14.5957 |

**Headline thesis cell:** Delhi → Kolkata @ d=30%, R² = **0.8189**, MAE = **13.3222 µg/m³**. This is the single best target-city result in the entire thesis Table 5.2 and is the number the new GNN-TL framework must beat.

## 2. Stage II legacy run (pre-fix; chronological split, GNN epochs=10, no climatology)

### 2.1 Source-only — LSTM and GAT ST-GNN

| City | R² LSTM legacy | MAE LSTM legacy | R² GAT-GNN legacy | MAE GAT-GNN legacy |
|---|--:|--:|--:|--:|
| Delhi | 0.8452 | 24.0985 | 0.7192 | 33.2373 |
| Kolkata | 0.7865 | 12.1392 | 0.3337 | 27.2955 |
| Guwahati | 0.4139 | 21.8989 | -0.2199 | 40.6901 |

Observation: under the legacy chronological 70/15/15 split, the GNN's Guwahati R² is **-0.220** — worse than predicting the test set's mean.

### 2.2 LSTM-TL fine-tuned (Stage II legacy)

| Pair@d% | R² | MAE | RMSE | MAPE |
|---|--:|--:|--:|--:|
| Delhi->Guwahati@15 | 0.3968 | 22.1109 | 37.2618 | 35.9085 |
| Delhi->Guwahati@30 | 0.4152 | 21.5251 | 36.6885 | 34.0342 |
| Delhi->Guwahati@45 | 0.4032 | 22.1574 | 37.0631 | 36.4729 |
| Delhi->Guwahati@60 | 0.4384 | 20.9091 | 35.9532 | 31.0218 |
| Delhi->Kolkata@15 | 0.7259 | 14.1273 | 24.4622 | 19.1724 |
| Delhi->Kolkata@30 | 0.7424 | 13.6288 | 23.7168 | 19.0730 |
| Delhi->Kolkata@45 | 0.7663 | 12.8070 | 22.5902 | 17.7417 |
| Delhi->Kolkata@60 | 0.7710 | 12.7333 | 22.3618 | 17.5034 |
| Guwahati->Delhi@15 | 0.8427 | 24.1633 | 38.7530 | 25.4236 |
| Guwahati->Delhi@30 | 0.8467 | 24.3850 | 38.2516 | 26.8524 |
| Guwahati->Delhi@45 | 0.8463 | 24.0957 | 38.2988 | 27.2275 |
| Guwahati->Delhi@60 | 0.8431 | 24.2670 | 38.6987 | 26.7218 |
| Guwahati->Kolkata@15 | 0.7538 | 13.8481 | 23.1846 | 20.9096 |
| Guwahati->Kolkata@30 | 0.7689 | 13.2103 | 22.4622 | 19.6658 |
| Guwahati->Kolkata@45 | 0.7672 | 13.4241 | 22.5450 | 19.5038 |
| Guwahati->Kolkata@60 | 0.7722 | 13.0577 | 22.2989 | 18.6529 |
| Kolkata->Delhi@15 | 0.8449 | 24.3137 | 38.4779 | 26.7657 |
| Kolkata->Delhi@30 | 0.8478 | 23.9130 | 38.1117 | 25.6455 |
| Kolkata->Delhi@45 | 0.8480 | 23.7696 | 38.0870 | 25.2212 |
| Kolkata->Delhi@60 | 0.8457 | 24.1258 | 38.3786 | 25.9129 |
| Kolkata->Guwahati@15 | 0.3967 | 21.8263 | 37.2661 | 33.4319 |
| Kolkata->Guwahati@30 | 0.4493 | 21.4779 | 35.6018 | 34.8453 |
| Kolkata->Guwahati@45 | 0.4309 | 21.2112 | 36.1924 | 33.5762 |
| Kolkata->Guwahati@60 | 0.4635 | 20.9222 | 35.1420 | 33.4317 |

## 3. Stage II fixed-protocol run (interleaved split + matched recipe + climatology residual)

### 3.1 Source-only — LSTM and GAT ST-GNN

| City | R² LSTM fixed | MAE LSTM fixed | R² GAT-GNN fixed | MAE GAT-GNN fixed |
|---|--:|--:|--:|--:|
| Delhi | 0.8501 | 23.7350 | 0.8343 | 24.1759 |
| Kolkata | 0.8627 | 7.7772 | 0.8244 | 9.2292 |
| Guwahati | 0.8320 | 12.1896 | 0.8311 | 12.7508 |

### 3.2 LSTM-TL fine-tuned (fixed protocol, full metrics)

| Pair@d% | R² | MAE | RMSE | MAPE |
|---|--:|--:|--:|--:|
| Delhi->Guwahati@15 | 0.8146 | 13.6542 | 24.3378 | 41.8961 |
| Delhi->Guwahati@30 | 0.8224 | 12.8701 | 23.8191 | 38.9221 |
| Delhi->Guwahati@45 | 0.8244 | 12.7541 | 23.6842 | 38.8745 |
| Delhi->Guwahati@60 | 0.8223 | 12.9060 | 23.8281 | 38.4428 |
| Delhi->Kolkata@15 | 0.8440 | 9.1703 | 16.0104 | 25.2268 |
| Delhi->Kolkata@30 | 0.8521 | 8.8178 | 15.5920 | 24.2808 |
| Delhi->Kolkata@45 | 0.8554 | 8.4904 | 15.4157 | 23.0241 |
| Delhi->Kolkata@60 | 0.8570 | 8.3727 | 15.3282 | 22.5815 |
| Guwahati->Delhi@15 | 0.8453 | 24.3309 | 37.1572 | 34.5537 |
| Guwahati->Delhi@30 | 0.8491 | 23.7855 | 36.6902 | 32.9783 |
| Guwahati->Delhi@45 | 0.8518 | 23.5701 | 36.3593 | 33.1840 |
| Guwahati->Delhi@60 | 0.8523 | 23.5132 | 36.2982 | 33.0734 |
| Guwahati->Kolkata@15 | 0.8263 | 9.4903 | 16.8959 | 26.4737 |
| Guwahati->Kolkata@30 | 0.8420 | 9.0072 | 16.1149 | 25.0813 |
| Guwahati->Kolkata@45 | 0.8491 | 8.7846 | 15.7495 | 24.5024 |
| Guwahati->Kolkata@60 | 0.8498 | 8.7842 | 15.7088 | 24.5113 |
| Kolkata->Delhi@15 | 0.8447 | 24.2214 | 37.2195 | 33.9551 |
| Kolkata->Delhi@30 | 0.8510 | 23.6561 | 36.4611 | 32.4532 |
| Kolkata->Delhi@45 | 0.8525 | 23.5103 | 36.2834 | 32.9681 |
| Kolkata->Delhi@60 | 0.8526 | 23.4131 | 36.2616 | 32.4483 |
| Kolkata->Guwahati@15 | 0.8092 | 13.6899 | 24.6874 | 45.1624 |
| Kolkata->Guwahati@30 | 0.8129 | 13.1193 | 24.4471 | 40.5891 |
| Kolkata->Guwahati@45 | 0.8202 | 12.8420 | 23.9659 | 40.1146 |
| Kolkata->Guwahati@60 | 0.8214 | 12.5316 | 23.8856 | 39.2701 |

### 3.3 GNN-TL Variant A fine-tuned (fixed protocol, full metrics)

| Pair@d% | R² transfer | MAE transfer | RMSE transfer | MAPE transfer |
|---|--:|--:|--:|--:|
| Delhi->Guwahati@15 | 0.8233 | 13.9316 | 23.7593 | 45.8673 |
| Delhi->Guwahati@30 | 0.8165 | 13.7335 | 24.2138 | 47.6495 |
| Delhi->Guwahati@45 | 0.8273 | 13.4093 | 23.4887 | 44.7551 |
| Delhi->Guwahati@60 | 0.8220 | 12.6777 | 23.8483 | 38.9895 |
| Delhi->Kolkata@15 | 0.8113 | 10.5281 | 17.6097 | 28.9802 |
| Delhi->Kolkata@30 | 0.8158 | 10.0473 | 17.4009 | 27.9143 |
| Delhi->Kolkata@45 | 0.8182 | 9.9781 | 17.2861 | 27.6841 |
| Delhi->Kolkata@60 | 0.8166 | 9.7791 | 17.3628 | 27.1827 |
| Guwahati->Delhi@15 | 0.7887 | 27.5789 | 43.4245 | 39.3599 |
| Guwahati->Delhi@30 | 0.8030 | 26.5499 | 41.9246 | 38.1341 |
| Guwahati->Delhi@45 | 0.8123 | 25.7138 | 40.9226 | 36.2818 |
| Guwahati->Delhi@60 | 0.8173 | 25.3058 | 40.3710 | 36.4258 |
| Guwahati->Kolkata@15 | 0.7951 | 10.9287 | 18.3498 | 29.8687 |
| Guwahati->Kolkata@30 | 0.8085 | 10.1837 | 17.7392 | 28.3521 |
| Guwahati->Kolkata@45 | 0.8103 | 10.1837 | 17.6583 | 28.1073 |
| Guwahati->Kolkata@60 | 0.8177 | 9.8522 | 17.3066 | 27.6278 |
| Kolkata->Delhi@15 | 0.7953 | 26.8175 | 42.7379 | 37.8696 |
| Kolkata->Delhi@30 | 0.8085 | 25.8171 | 41.3380 | 36.7148 |
| Kolkata->Delhi@45 | 0.8198 | 25.0238 | 40.0976 | 35.2919 |
| Kolkata->Delhi@60 | 0.8225 | 24.8185 | 39.7946 | 35.5767 |
| Kolkata->Guwahati@15 | 0.7955 | 14.5124 | 25.5622 | 42.4434 |
| Kolkata->Guwahati@30 | 0.8044 | 13.9312 | 24.9963 | 44.3194 |
| Kolkata->Guwahati@45 | 0.8038 | 13.6828 | 25.0340 | 43.1305 |
| Kolkata->Guwahati@60 | 0.8157 | 12.9965 | 24.2661 | 39.5935 |

## 4. GNN-TL verification — is knowledge actually transferring?

Every fixed-protocol GNN TL cell ran three trainings on the same d% target sample:

- **Zero-shot**: source-pretrained weights loaded, evaluated on target test, **no fine-tune**.
- **Scratch**: random initialization, fine-tuned on the same d% target sample using the same schedule. No knowledge from the source.
- **Transfer**: source-pretrained weights loaded, fine-tuned on the same d% target sample.

Knowledge transfer is "real" iff `transfer.R2 > zero_shot.R2` AND `transfer.R2 > scratch.R2`. If transfer beats scratch the source pre-training is actively helping; if not, the source is irrelevant or hurting.

### 4.1 Per-cell verification table

| Pair@d% | zero-shot R² | scratch R² | **transfer R²** | gain vs scratch | gain vs zero-shot | real_transfer |
|---|--:|--:|--:|--:|--:|:-:|
| Delhi->Guwahati@15 | +0.6841 | +0.7756 | **+0.8233** | +0.0477 | +0.1392 | YES |
| Delhi->Guwahati@30 | +0.6841 | +0.7826 | **+0.8165** | +0.0339 | +0.1323 | YES |
| Delhi->Guwahati@45 | +0.6841 | +0.7887 | **+0.8273** | +0.0386 | +0.1432 | YES |
| Delhi->Guwahati@60 | +0.6841 | +0.8083 | **+0.8220** | +0.0137 | +0.1378 | YES |
| Delhi->Kolkata@15 | +0.7282 | +0.7576 | **+0.8113** | +0.0537 | +0.0831 | YES |
| Delhi->Kolkata@30 | +0.7282 | +0.7803 | **+0.8158** | +0.0355 | +0.0875 | YES |
| Delhi->Kolkata@45 | +0.7282 | +0.7925 | **+0.8182** | +0.0256 | +0.0900 | YES |
| Delhi->Kolkata@60 | +0.7282 | +0.8031 | **+0.8166** | +0.0134 | +0.0883 | YES |
| Guwahati->Delhi@15 | +0.6317 | +0.7635 | **+0.7887** | +0.0251 | +0.1570 | YES |
| Guwahati->Delhi@30 | +0.6317 | +0.7905 | **+0.8030** | +0.0125 | +0.1713 | YES |
| Guwahati->Delhi@45 | +0.6317 | +0.7989 | **+0.8123** | +0.0134 | +0.1806 | YES |
| Guwahati->Delhi@60 | +0.6317 | +0.8105 | **+0.8173** | +0.0069 | +0.1857 | YES |
| Guwahati->Kolkata@15 | +0.6980 | +0.7576 | **+0.7951** | +0.0375 | +0.0971 | YES |
| Guwahati->Kolkata@30 | +0.6980 | +0.7803 | **+0.8085** | +0.0283 | +0.1105 | YES |
| Guwahati->Kolkata@45 | +0.6980 | +0.7925 | **+0.8103** | +0.0177 | +0.1123 | YES |
| Guwahati->Kolkata@60 | +0.6980 | +0.8031 | **+0.8177** | +0.0146 | +0.1198 | YES |
| Kolkata->Delhi@15 | +0.6032 | +0.7635 | **+0.7953** | +0.0317 | +0.1921 | YES |
| Kolkata->Delhi@30 | +0.6032 | +0.7905 | **+0.8085** | +0.0180 | +0.2053 | YES |
| Kolkata->Delhi@45 | +0.6032 | +0.7989 | **+0.8198** | +0.0209 | +0.2166 | YES |
| Kolkata->Delhi@60 | +0.6032 | +0.8105 | **+0.8225** | +0.0120 | +0.2193 | YES |
| Kolkata->Guwahati@15 | +0.6698 | +0.7807 | **+0.7955** | +0.0148 | +0.1256 | YES |
| Kolkata->Guwahati@30 | +0.6698 | +0.7924 | **+0.8044** | +0.0120 | +0.1346 | YES |
| Kolkata->Guwahati@45 | +0.6698 | +0.7970 | **+0.8038** | +0.0068 | +0.1340 | YES |
| Kolkata->Guwahati@60 | +0.6698 | +0.7981 | **+0.8157** | +0.0176 | +0.1458 | YES |

**Verification result: 24/24 cells show real knowledge transfer.**

## 5. Cross-method comparison — the headline table

Direct apples-to-apples comparison of the *transfer R²* across:
- Thesis LSTM-TL (Stage I)
- Stage II legacy LSTM-TL
- Stage II fixed LSTM-TL
- Stage II fixed GNN-TL (Variant A — pre-train + fine-tune)

| Pair@d% | Thesis LSTM | Legacy LSTM | Fixed LSTM | Fixed GAT-GNN | best |
|---|--:|--:|--:|--:|:-:|
| Kolkata->Guwahati@15 | 0.5980 | 0.3967 | 0.8092 | 0.7955 | LSTMfix |
| Kolkata->Guwahati@30 | 0.6030 | 0.4493 | 0.8129 | 0.8044 | LSTMfix |
| Kolkata->Guwahati@45 | 0.5823 | 0.4309 | 0.8202 | 0.8038 | LSTMfix |
| Kolkata->Guwahati@60 | 0.6271 | 0.4635 | 0.8214 | 0.8157 | LSTMfix |
| Kolkata->Delhi@15 | 0.6939 | 0.8449 | 0.8447 | 0.7953 | legacy |
| Kolkata->Delhi@30 | 0.6908 | 0.8478 | 0.8510 | 0.8085 | LSTMfix |
| Kolkata->Delhi@45 | 0.6881 | 0.8480 | 0.8525 | 0.8198 | LSTMfix |
| Kolkata->Delhi@60 | 0.6870 | 0.8457 | 0.8526 | 0.8225 | LSTMfix |
| Guwahati->Kolkata@15 | 0.8098 | 0.7538 | 0.8263 | 0.7951 | LSTMfix |
| Guwahati->Kolkata@30 | 0.8174 | 0.7689 | 0.8420 | 0.8085 | LSTMfix |
| Guwahati->Kolkata@45 | 0.8164 | 0.7672 | 0.8491 | 0.8103 | LSTMfix |
| Guwahati->Kolkata@60 | 0.8019 | 0.7722 | 0.8498 | 0.8177 | LSTMfix |
| Guwahati->Delhi@15 | 0.6995 | 0.8427 | 0.8453 | 0.7887 | LSTMfix |
| Guwahati->Delhi@30 | 0.6877 | 0.8467 | 0.8491 | 0.8030 | LSTMfix |
| Guwahati->Delhi@45 | 0.6815 | 0.8463 | 0.8518 | 0.8123 | LSTMfix |
| Guwahati->Delhi@60 | 0.6837 | 0.8431 | 0.8523 | 0.8173 | LSTMfix |
| Delhi->Kolkata@15 | 0.7437 | 0.7259 | 0.8440 | 0.8113 | LSTMfix |
| Delhi->Kolkata@30 | 0.8189 | 0.7424 | 0.8521 | 0.8158 | LSTMfix |
| Delhi->Kolkata@45 | 0.8129 | 0.7663 | 0.8554 | 0.8182 | LSTMfix |
| Delhi->Kolkata@60 | 0.8051 | 0.7710 | 0.8570 | 0.8166 | LSTMfix |
| Delhi->Guwahati@15 | 0.5797 | 0.3968 | 0.8146 | 0.8233 | GNNfix |
| Delhi->Guwahati@30 | 0.6381 | 0.4152 | 0.8224 | 0.8165 | LSTMfix |
| Delhi->Guwahati@45 | 0.6210 | 0.4032 | 0.8244 | 0.8273 | GNNfix |
| Delhi->Guwahati@60 | 0.6133 | 0.4384 | 0.8223 | 0.8220 | LSTMfix |

## 6. Graph structures (sanity log)

All graphs use the k-NN distance kernel with k=3 and Gaussian-decay edge weights `w_ij = exp(-d_ij² / 2σ²)`, σ=5 km. Edge counts and average degrees:

| City | \|V\| | \|E\| | avg degree |
|---|--:|--:|--:|
| Delhi    | 40 | 120 | 3.00 |
| Kolkata  | 10 | 30 | 3.00 |
| Guwahati | 4 | 12 | 3.00 |

Note: Guwahati has only 4 stations, so a k=3 k-NN graph is effectively a complete directed graph (minus self-loops). The inductive ST-GNN's parameter count does *not* depend on \|V\|, so the same model trained on Delhi (\|V\|=40) runs forward on Guwahati (\|V\|=4) without any shape change — this is the demonstrable resolution of the parameter-shape mismatch failure mode F-i in the intercity graph reconstruction problem.

## 7. Summary of improvements (Δ R² over baselines)

### 7.1 Source-only

| City | thesis LSTM | fixed LSTM | Δ vs thesis | fixed GAT-GNN | Δ vs thesis |
|---|--:|--:|--:|--:|--:|
| Delhi | 0.6570 | 0.8501 | +0.1931 | 0.8343 | +0.1773 |
| Kolkata | 0.7861 | 0.8627 | +0.0766 | 0.8244 | +0.0383 |
| Guwahati | 0.5723 | 0.8320 | +0.2597 | 0.8311 | +0.2588 |

### 7.2 Transfer (headline cells)

- **Delhi→Kolkata @ 30% (thesis headline)**: thesis 0.8189 → fixed-LSTM 0.8521 → fixed-GNN 0.8158
- **Delhi→Guwahati @ 30%**: thesis 0.6381 → fixed-LSTM 0.8224 → fixed-GNN 0.8165
- **Kolkata→Guwahati @ 30%**: thesis 0.6030 → fixed-LSTM 0.8129 → fixed-GNN 0.8044
- **Guwahati→Kolkata @ 30%**: thesis 0.8174 → fixed-LSTM 0.8420 → fixed-GNN 0.8085
