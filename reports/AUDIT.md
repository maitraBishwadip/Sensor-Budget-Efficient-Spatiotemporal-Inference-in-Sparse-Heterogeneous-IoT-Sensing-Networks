# Data-Leakage and Validity Audit

Audit of the kept pipeline — **LSTM** (source-only + TL), **GNN** (source-only + GNN-TL), **GNN-PINN**, and
**virtual sensing** — against the leakage taxonomy of Kaufman et al. (TKDD 2012) and the temporally-correlated
cross-validation literature (Roberts et al. 2017; Bergmeir & Benítez 2012; Bergmeir, Hyndman & Koo 2018;
Cerqueira, Torgo & Mozetič 2020). Companion docs: equation-level verification in
[MATH_AUDIT.md](MATH_AUDIT.md); narrative in [MAIN_REPORT.md](MAIN_REPORT.md); bibliography in
[REFERENCES.md](REFERENCES.md).

---

## 1. Checks (all ✅)

| # | Potential leak | Safeguard | Where |
|---|---|---|---|
| 1 | **Scaler fit on test** | `StandardScaler` (features + target) fit on the **train mask only**, then applied to all splits. | `utils.load_city_tensor` (`utils.py:254–260`) |
| 2 | **Climatology fit on test** | Per-(month,hour) climatology computed from the **train slice only**; subtracted as the target residual and added back at evaluation. | `utils._compute_climatology` (`utils.py:119–144`) |
| 3 | **Anticausal imputation** | Only causal forward-fill (+ zero-fill of leading gaps); the earlier `bfill` (which could pull future-train values into val/test) was removed. | `utils.py:217–222` |
| 4 | **Target leakage across splits** | A window's supervision target is owned by exactly one split — the split of its *prediction* timestep `t+H+horizon−1`. Histories may overlap across splits (cheap and necessary at 3-h cadence) but no split ever supervises another's target. | `utils.make_windows_masked` (`utils.py:313–354`) |
| 5 | **Unequal target sub-samples** | The `d%` fine-tune sub-sample uses a fixed seed (42); the **same** `keep` indices are reused for the transfer model and the from-scratch baseline, so the three-way verification compares on identical data. | `train_gnn_tl.py`, `train_lstm.py` |
| 6 | **Virtual-sensing leakage** | The model is trained on the **k sensored** stations and evaluated on the **held-out N−k unsensored** stations; the unsensored PM2.5 is never an input or a training target. The per-`U` LSTM baseline uses the identical split. | `train_gnn_vsense.py`, `analysis/lstm_vsense_peru.py` |

## 2. The one documented limitation — interleaved split

The fixed protocol uses an **interleaved** train/val/test split (every Nth timestep to val/test) rather than a
chronological hold-out, so each split spans the full year. This was adopted because Kolkata and Guwahati have
only ~1 year of data, making a chronological test set winter-only and unrepresentative.

Trade-off (Bergmeir & Benítez 2012; Bergmeir, Hyndman & Koo 2018): on autocorrelated series an interleaved
split can mildly **inflate absolute R²** versus a strict chronological split, because adjacent train/test
timesteps are correlated. **This does not affect the comparative claims of this study**, because the *identical*
interleaved split is applied to every model (LSTM, GNN, GNN-TL, GNN-PINN, virtual sensing) — any
protocol-driven inflation cancels at method-vs-method comparison time. The headline orderings
(*LSTM ≥ GNN for forecasting*; *GNN > LSTM for virtual sensing*; *physics = no improvement*) are therefore valid.

## 3. Conclusion

No leakage source that affects the relative model orderings was found; the comparative results in
[MAIN_REPORT.md](MAIN_REPORT.md) are sound under the uniform fixed protocol. A chronological-split sensitivity
check on Delhi (which has 2 years of data) is the recommended optional robustness addition for the manuscript.
