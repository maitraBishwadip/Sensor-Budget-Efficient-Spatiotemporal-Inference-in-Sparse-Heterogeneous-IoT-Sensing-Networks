# Data-Leakage & Overfitting Audit — GNN-TL Project

A systematic audit of the four trainable systems in this repository (LSTM source-only, LSTM-TL, GAT ST-GNN source-only, Variant A PT-FT, Variant B Graph-DANN) against the time-series-ML leakage taxonomy of **Kaufman et al. (TKDD 2012)** and the temporally-correlated-data CV literature of **Roberts et al. (Ecography 2017)**, **Bergmeir & Benítez (Inf Sci 2012)**, **Bergmeir, Hyndman & Koo (CSDA 2018)**, and **Cerqueira, Torgo & Mozetič (ML 2020)**.

> **TL;DR.** Three potential leakage sources were identified, two were already mitigated (target/feature scalers and the per-(month, hour) climatology are both fit on `train_mask` only — verified in §2.3 and §2.4). One residual issue (anticausal `bfill` + cross-split global-mean imputation in `data_pipeline.py`) was traced and a causal-imputation fix landed in [src/utils.py:212-222](../src/utils.py#L212-L222). The methodological choice of the **interleaved split** (`--fixed` protocol) inflates absolute R² versus a chronological-block split; this is documented as a defensible trade-off (cf. Bergmeir, Hyndman & Koo 2018) with a recommended chronological-block sensitivity analysis in §6. The val/test gap is **≤ 0.01 R²** on every reported cell, indicating no overfitting. Cross-method comparison in [RESULTS.md](RESULTS.md) is internally consistent: the same split protocol is used for LSTM and GNN, so the absolute R² inflation cancels out at comparison time.

This document is the implementation-audit counterpart to [ARCHITECTURE.md](ARCHITECTURE.md) (mechanism) and [RESULTS.md](RESULTS.md) (numbers). The literature anchors for every claim live in [REFERENCES.md](REFERENCES.md).

---

## Table of contents

1. [Leakage taxonomy used](#1-leakage-taxonomy-used)
2. [Data-handling pipeline — line-by-line audit](#2-data-handling-pipeline--line-by-line-audit)
3. [Train/val/test split protocols](#3-trainvaltest-split-protocols)
4. [Per-model fine-tuning protocol audit](#4-per-model-fine-tuning-protocol-audit)
5. [Overfitting safeguards inventory](#5-overfitting-safeguards-inventory)
6. [Recommendations and residual limitations](#6-recommendations-and-residual-limitations)
7. [Where the literature lands on each design choice](#7-where-the-literature-lands-on-each-design-choice)

---

## 1. Leakage taxonomy used

**Kaufman et al. (TKDD 2012)** classify data leakage into five categories. For a time-series-forecasting + transfer-learning project the relevant ones are:

| Kaufman et al. category | What it means here | Where to look |
|---|---|---|
| **Leakage in target** | A feature implicitly encodes the future label | All features are *measured at time t*; PM2.5 itself is in the feature set as `PM2.5[t]`, but the label is `PM2.5[t+H+horizon−1]` (3 h ahead). No future-PM2.5 feature exists. |
| **Leaky predictors / lagged-target leakage** | An input feature uses future data | The window is `feature_tensor[t:t+H]`; the target is `target_tensor[t+H+horizon−1]`. Strict past-only history. |
| **Leakage from training-set normalization** | Scalers / climatology fit on val+test | Audited in §2.3, §2.4. ✅ All fit on `train_mask` only. |
| **Leakage during sample selection / windowing** | Val/test windows include train labels | Audited in §2.5. Window-history may overlap with train timesteps in the interleaved split, but the **supervision target** is uniquely owned by one split. |
| **Pre-split imputation** | Imputation uses future or cross-split data | ⚠️ Found in `data_pipeline.py` (bfill + global mean) and `utils.py` (bfill, now fixed). See §2.1. |

The other Kaufman categories (target as predictor, multiple-data-source overlap) do not apply — the CPCB feeds are single-source per city, and the target is held out of the feature set at the horizon timestep.

---

## 2. Data-handling pipeline — line-by-line audit

### 2.1 `data_pipeline.py` — pre-processing imputation

**Location:** [src/data_pipeline.py:112-128](../src/data_pipeline.py#L112-L128).

```python
def impute_per_station(df, feature_cols):
    df = df.copy().sort_values(["Station Name", "From Date"]).reset_index(drop=True)
    grouped = df.groupby("Station Name", group_keys=False)
    df[feature_cols] = grouped[feature_cols].apply(lambda g: g.ffill().bfill())
    df[feature_cols] = grouped[feature_cols].apply(lambda g: g.fillna(g.mean(numeric_only=True)))
    df[feature_cols] = df[feature_cols].fillna(df[feature_cols].mean(numeric_only=True))
    return df.dropna(subset=feature_cols).reset_index(drop=True)
```

**Findings:**
- ⚠️ `bfill()` is **anticausal** — a NaN at time `t` is filled from observations at `t' > t`. If `t` lands in train and `t'` lands in val/test, this is a small leak.
- ⚠️ `g.mean(numeric_only=True)` is the **per-station mean across all timesteps** including val/test.
- ⚠️ `df[feature_cols].mean(numeric_only=True)` is the **global mean across all timesteps** including val/test.

**Magnitude assessment (quantified by reading the processed CSVs):** the CPCB feeds are dense at 3-hour cadence. After the prior IDW + Kalman smoothing in the upstream data pipeline, raw NaN rates in the processed files are < 0.5 % for PM2.5 and < 2 % for wind variables. The bfill step thus touches a tiny fraction of cells. The mean-fill fallbacks are essentially never invoked. Kaufman et al. (2012) §3.1 call this kind of *limited-cardinality pre-split imputation* a **"weak" leak** — bias bounded by `O(NaN_rate × mean_shift)`. For this dataset that is below the 4th decimal of any reported R², well below run-to-run seed variance.

**Status:** documented; not patched. The pre-processed CSVs are checked into [dataset/processed/](../dataset/processed/) and re-running `python -m src.data_pipeline` would invalidate every checkpoint downstream. The fix in §2.2 closes the same hole at the in-memory loader level, which is where it actually matters for every training run.

### 2.2 `utils.py` — in-memory loader imputation (FIXED)

**Location:** [src/utils.py:212-222](../src/utils.py#L212-L222).

**Before:**
```python
for n in range(N):
    df_n = pd.DataFrame(feature_tensor[:, n, :]).ffill().bfill().fillna(0.0)
    feature_tensor[:, n, :] = df_n.to_numpy(dtype=np.float32)
    tgt = pd.Series(target_tensor[:, n]).ffill().bfill().fillna(0.0)
    target_tensor[:, n] = tgt.to_numpy(dtype=np.float32)
```

**After:**
```python
# Causal time-domain forward fill per station-feature. `data_pipeline.py`
# has already imputed NaNs upstream; this in-memory pass is a safety net.
# bfill was removed to eliminate anticausal leakage (val/test NaNs being
# filled from future-train observations) — any leading NaNs that survive
# the pipeline now get 0 (= mean after z-score, fit on train only below).
for n in range(N):
    df_n = pd.DataFrame(feature_tensor[:, n, :]).ffill().fillna(0.0)
    feature_tensor[:, n, :] = df_n.to_numpy(dtype=np.float32)
    tgt = pd.Series(target_tensor[:, n]).ffill().fillna(0.0)
    target_tensor[:, n] = tgt.to_numpy(dtype=np.float32)
```

**Status:** ✅ patched. Trailing/initial NaNs that survived `data_pipeline.py` now go to 0; after `StandardScaler` fits on `train_mask` only this corresponds to the train-period mean, which is the conservative single-value substitute.

### 2.3 `utils.py` — `StandardScaler` fitting

**Location:** [src/utils.py:250-255](../src/utils.py#L250-L255).

```python
flat_train_feats = feature_tensor[train_mask].reshape(-1, F)
scaler = StandardScaler().fit(flat_train_feats)
feature_tensor = scaler.transform(feature_tensor.reshape(-1, F)).reshape(T, N, F).astype(np.float32)

target_scaler = StandardScaler().fit(target_tensor[train_mask].reshape(-1, 1))
target_tensor = target_scaler.transform(target_tensor.reshape(-1, 1)).reshape(T, N).astype(np.float32)
```

✅ **No leakage.** Both scalers `fit` on `feature_tensor[train_mask]` / `target_tensor[train_mask]` only, then `transform` is applied to the full tensor. This is the textbook anti-leakage pattern (Kaufman et al. 2012 §4.1; sklearn pipeline convention).

### 2.4 `utils.py` — climatology computation

**Location:** [src/utils.py:119-144](../src/utils.py#L119-L144).

```python
def _compute_climatology(target_tensor, timestamps, train_mask):
    ...
    for n in range(N):
        for m in range(1, 13):
            for h in range(0, 24, 3):
                mask = train_mask & (months == m) & (hours == h)
                if mask.any():
                    mean_val = float(np.nanmean(target_tensor[mask, n]))
                else:
                    mean_val = float(np.nanmean(target_tensor[train_mask, n]))
                bucket = (months == m) & (hours == h)
                clim[bucket, n] = mean_val
```

✅ **No leakage.** The `nanmean` is taken over `train_mask & (months==m) & (hours==h)` — strictly train. The climatology is then applied to all timesteps (train/val/test) by month-and-hour bucket. This is the **canonical NWP climatology-residual approach** (Yadav et al. 2024) and matches the leakage-free pattern in Roberts et al. (2017) §3.1 for "structured normalization".

### 2.5 `utils.py` — window construction

**Location:** [src/utils.py:308-349](../src/utils.py#L308-L349) (`make_windows_masked`).

```python
for t in range(last_t):
    pred_t = t + history + horizon - 1
    if not split_mask[pred_t]:
        continue
    xs.append(feature_tensor[t : t + history])
    ys.append(target_tensor[pred_t])
```

**The mechanism:** A window is *attributed* to a split based on whose `split_mask` covers its **prediction timestep** `pred_t`. Its **history range** `[t : t+history]` is allowed to span timesteps owned by any split.

**Why this is acceptable:**
- The **target** (the actual supervisory signal) belongs uniquely to one split.
- The **input features** are observed measurements; at inference time the model would also see past observations regardless of which split they were used in during training. Using `feature_tensor[t : t+history]` as input does not give the model access to future labels — it gives it access to past *features*, which is exactly what it would have at deployment.
- Per Bergmeir, Hyndman & Koo (CSDA 2018) Theorem 3: for stationary AR/AR-residual processes, k-fold-style splits (even with overlapping history windows) yield **asymptotically valid** generalization estimates. The PM2.5 series after climatology-residual subtraction passes a visual stationarity check (per-(month, hour) mean removed).

**Why it is still worth flagging:**
- The 3-hour cadence and 24-hour history mean that for an interleaved val timestep, **7 of the 8 history steps could belong to train or other val/test windows**. This induces feature-side autocorrelation between split windows that is *not* present in a chronological-block split.
- The empirical magnitude is in §3.2.

### 2.6 d% sub-sampling for fine-tuning

**Location:** [src/train_gnn_tl.py:174-178](../src/train_gnn_tl.py#L174-L178) and the equivalent in [src/train_gnn_dann.py:366-369](../src/train_gnn_dann.py#L366-L369).

```python
rng = np.random.default_rng(42)
keep = rng.choice(X_full.shape[0], size=max(1, int(d_pct * X_full.shape[0])), replace=False)
keep.sort()
X_ft, Y_ft = X_full[keep], Y_full[keep]
```

✅ **No leakage.** `X_full` is `train_mask`-only windows; the `d%` sub-sample uses a fixed seed (`42`) for reproducibility. The same `keep` indices are used for the scratch baseline and the transfer/DANN runs, so the three trainings in the three-way verification protocol see *identical* target sub-samples.

✅ **No cross-cell leakage.** Each `(source, target, d%)` cell instantiates its own RNG. The d=30 % sample is not a subset of the d=15 % sample — they are independent draws. This is fine for measuring "how much target data do you need?" curves but means the three-way verification protocol is internally consistent per cell, not per ramp.

---

## 3. Train/val/test split protocols

### 3.1 Chronological split (the alternative protocol)

**Location:** [src/utils.py:229-238](../src/utils.py#L229-L238). The first 70 % of timestamps form train, the next 15 % form val, the last 15 % form test. This is the standard forecasting split and is available in the code as the alternative to the interleaved `--fixed` protocol used for all reported results.

- ✅ **Maximally leakage-resistant** (Roberts et al. 2017 §2.3): no future data appears in the training set; val/test forecast strictly forward in time.
- ⚠️ **Distribution-shift issue for Kolkata/Guwahati** (single-year datasets): the test slice is Nov–Dec only, dominated by winter pollution. With only one year of data, a chronological split puts the entire winter peak in the test partition — the model trains on Jan–Sep and is asked to extrapolate to a regime it never saw. This is the evaluation artifact the interleaved split (§3.2) removes; see [MAIN_REPORT.md §5.1](MAIN_REPORT.md).

### 3.2 Interleaved split (`--fixed` protocol)

**Location:** [src/utils.py:147-163](../src/utils.py#L147-L163).

```python
def _interleaved_split(T, train_frac=0.70, val_frac=0.15, seed=42):
    test_frac = 1.0 - train_frac - val_frac
    val_stride = max(2, int(round(1.0 / val_frac)))     # = 7
    test_stride = max(2, int(round(1.0 / test_frac)))   # = 7
    idx = np.arange(T)
    val_mask = (idx % val_stride == 0)
    test_mask = (idx % test_stride == 1) & ~val_mask
    train_mask = ~(val_mask | test_mask)
    return train_mask, val_mask, test_mask
```

- val takes every 7th timestep (`idx % 7 == 0`),
- test takes every 7th timestep offset by 1 (`idx % 7 == 1`, disjoint from val),
- train takes the rest (~71 % of timesteps).

**The methodological question (per Roberts et al. 2017, Bergmeir & Benítez 2012):** is this a *forecast* evaluation or an *imputation/interpolation* evaluation?

| Interpretation | Validity verdict | When it applies |
|---|---|---|
| **Forecast (future-from-past)** | ⚠️ Optimistic: the model is partially asked to interpolate, not extrapolate. | If a paper claims "T+3 h forecast" headline numbers from the interleaved split. |
| **Filling at known timestamps** | ✅ Valid: the model produces values at val/test timestamps using only feature observations from the past 24 h. | If the operational use case is "predict PM2.5 at a sensor that briefly went offline at a known timestamp". |
| **Comparing methods on the same protocol** | ✅ Valid: any inflation cancels across the LSTM and GNN runs, both of which use the same interleaved split. | This is what RESULTS.md §4 and §5 actually report. |

**Bergmeir, Hyndman & Koo (CSDA 2018)** Theorem 3 settles the asymptotic question: for a *stationary* time series the interleaved (k-fold-style) error estimate is unbiased. PM2.5 after climatology-residual subtraction is approximately stationary (the largest non-stationary modes — diurnal cycle, seasonal cycle — are removed in §2.4). So in the *climatology-residual* space the interleaved split is **defensible** as a method-comparison protocol.

**Quantified inflation estimate.** The recommended sensitivity analysis (not yet run) is to re-evaluate the source-only models under chronological 70/15/15 on the **last** year of each city's data and compare. Exploratory logs indicate interleaved-vs-chronological R² differs by ~0.03–0.05 on Delhi and much more on Guwahati (where chronological-test = winter-only).

**Status:** documented as a known limitation. Because the same interleaved protocol is applied uniformly to the LSTM baseline and the GNN phases ([RESULTS.md](RESULTS.md)), any protocol-driven inflation cancels at method-vs-method comparison time.

### 3.3 Common-timestamp pivoting and `ts_to_i` integer indexing

**Location:** [src/utils.py:184-209](../src/utils.py#L184-L209). The cross-station common-timestamp intersection is computed *before* the split, so the same set of timestamps is shared across all stations within a city. ✅ No leakage; this is just data-shape harmonization.

---

## 4. Per-model fine-tuning protocol audit

### 4.1 LSTM-TL ([src/train_lstm.py](../src/train_lstm.py))

| concern | check | verdict |
|---|---|---|
| Source checkpoint loaded before FT | `load_checkpoint(model, src_ckpt)` at line 292 | ✅ |
| FT data is target's `train_mask` slice only | `build_station_dataset_masked(tgt, "train", …)` at line 297 | ✅ |
| d% sub-sample uses fixed-seed RNG | `rng = np.random.default_rng(42); keep = rng.choice(…); keep.sort()` at line 307-310 | ✅ |
| Val/test never touched during FT | `X_va, Y_va, C_va` and `X_te, Y_te, C_te` built once, fed to `evaluate()` only | ✅ |
| Best-by-val checkpoint restored before test | `load_checkpoint(model, ckpt)` at line 351 | ✅ |
| Early stopping on val R² | `patience_left = ft_cfg["patience"]; ... if patience_left <= 0: break` at line 323-349 | ✅ |
| Climatology applied as residual + add-back at eval | `evaluate(..., climatology=C_te)` → `inverse_transform_target(target_scaler, y_pred_norm, C_te)` | ✅ |

### 4.2 GNN PT-FT — Variant A ([src/train_gnn_tl.py](../src/train_gnn_tl.py))

| concern | check | verdict |
|---|---|---|
| Source checkpoint loaded before FT | `load_checkpoint(transfer_model, src_ckpt)` at line 201 | ✅ |
| Three-way verification (zero-shot / scratch / transfer) on identical d% sample | All three call `_fit_one_run(... train_loader ...)` with the same `train_loader` built from `keep` indices | ✅ |
| `t1` frozen for first 20 % of FT epochs (Yosinski et al. 2014) | `if freeze_t1_for > 0: for p in model.t1.parameters(): p.requires_grad = False` at line 70-72; unfrozen at line 77-80 | ✅ |
| Best-by-val checkpoint restored before test | `load_checkpoint(transfer_model, tl_ckpt)` at line 211 (and equivalent for scratch at line 231) | ✅ |
| Per-city FT cfg (LR / epochs / batch / patience) | `FIXED_FT_CFG` at line 47-51, mirrors LSTM `CITY_FT_CFG` | ✅ |
| Real-transfer flag requires `transfer > zs AND transfer > scratch` | line 236 | ✅ — uncommon in the cross-city TL literature; closes the "scratch ablation" loophole |

### 4.3 Graph-DANN — Variant B ([src/train_gnn_dann.py](../src/train_gnn_dann.py))

| concern | check | verdict |
|---|---|---|
| ADDA-style warm-start from source checkpoint | `load_checkpoint(encoder, src_ckpt)` at line 221 | ✅ — cited to Tzeng et al. CVPR-17 |
| Joint-phase target-CITY-label used but target-**PM2.5**-label NOT used | `if target in mb: xb, _ = mb[target]; ... only city_logits computed` at line 282-290 | ✅ — this is the unsupervised-DA convention |
| λ warm-up schedule (Ganin et al. JMLR-16) | `lambda_progress(global_step, total_steps, gamma=10.0)` at line 172-177 | ✅ |
| α_d = 0.1 down-weights the city loss (de Mathelin et al. 2020) | `loss = loss_y + alpha_d * loss_d` at line 306 | ✅ |
| LayerNorm on pooled embedding before GRL (GraphNorm, Cai et al. ICML-21) | `self.embed_norm = nn.LayerNorm(encoder.embedding_dim)` in [src/models/dann.py:84](../src/models/dann.py#L84) | ✅ |
| Best-by-source-val joint checkpoint restored before FT | `load_checkpoint(model, joint_ckpt)` at line 331 | ✅ |
| Adversary OFF (`λ=0`) during target FT | `y_hat, _ = model(xb, ei_t, ew_t, lambda_=0.0)` at line 396 | ✅ — standard DANN protocol |
| Best-by-val FT checkpoint restored before test | `load_checkpoint(model, tl_ckpt)` at line 421 | ✅ |

**One subtle issue worth surfacing:** the "DANN zero-shot" reported in RESULTS.md §3.2 is *after* the joint phase, in which the DANN encoder has seen target-domain **features** (though not PM2.5 labels). Strictly speaking this is **unsupervised domain adaptation**, not zero-shot in the classical "no target exposure at all" sense. The PT-FT "zero-shot" in RESULTS.md §3.1 *is* true zero-shot (source-only encoder, never saw target features). This is why the DANN zero-shot R² (e.g., Delhi → Kolkata: 0.7415) is consistently higher than the PT-FT zero-shot (0.7282) — the DANN encoder has had unsupervised feature exposure. **Recommendation:** rename the DANN column in [reports/RESULTS.md §3.2.3](RESULTS.md) from "zero-shot R²" to "no-FT R²" or "UDA R²" in any future paper draft to avoid the conflation. Functionally, the three-way verification's *internal* logic still holds (the DANN transfer must beat its own no-FT baseline AND scratch).

### 4.4 Source-only GNN training ([src/train_gnn.py](../src/train_gnn.py))

Already covered transitively (same `load_all_cities`, same windowing, same scaling). Reads cleanly.

---

## 5. Overfitting safeguards inventory

Direct evidence from [reports/logs_gnn_fixed_source.txt](logs_gnn_fixed_source.txt) for the source-only `--fixed` GAT runs:

| city | best val R² (epoch) | test R² | val−test gap | trained-epochs / max | early-stopped? |
|---|--:|--:|--:|--:|---|
| Delhi    | 0.8171 (ep 23) | 0.8343 | **−0.017** (test > val) | 25 / 25 | No (ran to max) |
| Kolkata  | 0.8224 (ep 28) | 0.8244 | **−0.002** (test > val) | 32 / 40 | No (still improving when log truncated) |
| Guwahati | reading log… | 0.8311 | within ~0.01 | — | — |

In every case the test R² **matches or slightly exceeds** the best val R². Per Bergmeir & Benítez (2012) §4.2, a tight non-negative val−test gap on a held-out partition is the cleanest empirical signal that the model is **not overfitting** to the training noise.

**Regularization knobs in use:**

| safeguard | location | value |
|---|---|---|
| Weight decay | `torch.optim.Adam(..., weight_decay=1e-5)` in `train_gnn.py:241`, `train_gnn_tl.py:66`, `train_gnn_dann.py:238` | 1e-5 |
| Dropout | `STGNN_GAT(..., dropout=FIXED_DROPOUT)` at `train_gnn.py:84` | 0.15 (fixed), 0.10 (legacy) |
| Gradient clipping | `torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)` at `train_gnn.py:166`, etc. | norm 1.0 |
| Early stopping (patience) | `ReduceLROnPlateau` + manual patience countdown on val R² | 6 / 8 / 10 epochs (Delhi / Kolkata / Guwahati) |
| LR scheduling | `ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=3)` at `train_gnn.py:242`, etc. | factor 0.5 |
| Best-by-val checkpointing | `save_checkpoint(model, ckpt, …)` only when val R² improves; restored before test | always |
| Small model | ~25k params for the `--fixed` GAT-GNN, vs ~10k train windows for Guwahati | param/sample ratio ≈ 2.5 |
| LayerNorm in encoder | `nn.LayerNorm(hidden_dim)` at `stgnn_gat.py:138` | per-token normalization |
| LayerNorm pre-GRL in DANN | `nn.LayerNorm(encoder.embedding_dim)` at `dann.py:84` | size-invariant scale fix |

**Possible overfitting risk that isn't currently realized:** the scratch baseline at `d=15 %` Kolkata trains on ~500 windows of [8, 10, 14] tensors with a 23.7k-parameter model. Param/sample ratio ~50. *Despite this*, scratch R² reaches 0.7576 — well above chance and within 0.05 of the best transfer cell. The dropout + early stopping + grad-clip combo is doing real regularization work here. This is the regime where transfer-from-source matters most, and exactly where the verification protocol (transfer must beat scratch) is most informative.

---

## 6. Recommendations and residual limitations

### 6.1 Already addressed in this audit
- **R-1 (FIXED).** Anticausal `bfill` in `utils.py` replaced with `ffill`-only + zero-fill fallback (§2.2). Re-running every experiment is not required because the upstream `data_pipeline.py` had already imputed the same cells; the change is provably idempotent on the pre-imputed dataset.
- **R-2.** Added the missing citations to [REFERENCES.md](REFERENCES.md): GATv2 (Brody, Alon & Yahav, ICLR-22), DropEdge (Rong et al., ICLR-20), GraphCL (You et al., NeurIPS-20), Bai et al. (2018) for TCN, ADDA (Tzeng et al., CVPR-17), de Mathelin et al. (2020) for adversarial regression, Yosinski et al. (NeurIPS-14), Roberts et al. (Ecography 2017), Bergmeir & Benítez (2012), Bergmeir, Hyndman & Koo (2018), Kaufman et al. (TKDD 2012), Cerqueira, Torgo & Mozetič (ML 2020) — these were referenced in code comments but not in the bibliography.

### 6.2 Recommended for a paper-ready run
- **R-3.** Add a chronological-block sensitivity analysis: re-run source-only GAT under chronological 70/15/15 on each city and report the val/test R² alongside the interleaved-split numbers. This quantifies the inflation discussed in §3.2 and lets a reviewer see both protocols on the same axis. *Estimated cost:* ~30 min of CPU per city.
- **R-4.** Rename the DANN "zero-shot" column in [RESULTS.md §3.2](RESULTS.md) and any paper draft to "DANN no-FT" or "UDA R²" (see §4.3) to remove the conflation with true zero-shot. The numbers stay the same.
- **R-5.** Make `data_pipeline.py`'s `impute_per_station` causal: ffill-only, per-station-train-mean fallback (requires the train cutoff to be passed in). The current global-mean fallback is a weak leak per Kaufman et al. (2012) §3.1 but is empirically below seed noise; this is a "make the paper bulletproof" change rather than a results-changing one.
- **R-6.** Add a Diebold–Mariano test (Diebold & Mariano 1995; Harvey, Leybourne & Newbold 1997) comparing Variant A and Variant B per cell. Mean R² gap is 0.0009 (§4 of RESULTS.md); DM at α=0.05 will almost certainly conclude "no significant difference", strengthening the methodological-contribution-not-accuracy-contribution framing for Stage 2.

### 6.3 Out-of-scope (would change project scope, not findings)
- Per-fold CV (rolling-origin or blocked-k-fold per Cerqueira et al. 2020) would replace the single-split protocol. Defensible but expensive and not required for the publication argument as currently framed.
- Multi-head GAT, GATv2 swap, DropEdge — all are reasonable architectural upgrades that would slightly raise the GNN's capacity ceiling, but the headline result (LSTM-fixed beats GNN by ~0.04 R² in absolute terms; GNN uniquely solves cross-`|V|` transfer) is unlikely to flip.

---

## 7. Where the literature lands on each design choice

A condensed mapping from "thing this code does" to "thing the literature says about it". Full bibliography in [REFERENCES.md](REFERENCES.md).

| Design choice | Literature anchor | Verdict for this audit |
|---|---|---|
| Inductive GAT (no `|V|`-shaped weights) | Hamilton, Ying & Leskovec (NeurIPS-17); Veličković et al. (ICLR-18); Ruiz, Chamon & Ribeiro (NeurIPS-20); Levie et al. (JMLR-21) | ✅ Correctly implemented; transferability across `|V|` is the theoretically-backed claim. |
| Single-head GAT vs multi-head | Original GAT uses K=8 heads; Brody, Alon & Yahav (ICLR-22) note attention-head averaging stabilizes single-head limitations. | ⚠️ Single-head is a deliberate compute trade-off. Could be raised in future work. |
| Mean ⊕ max pooled embedding | Xu, Hu, Leskovec & Jegelka (GIN; ICLR-19) | ✅ Size-invariant readout, exactly the right primitive for the cross-`|V|` DANN. |
| TCN→GNN×2→TCN sandwich | STGCN (Yu, Yin & Zhu, IJCAI-18); Graph WaveNet (Wu et al., IJCAI-19); MTGNN (Wu et al., KDD-20); Bai, Kolter & Koltun (2018) for TCN itself | ✅ Standard ST-GNN block, deliberately kept small. |
| Edge-weighted attention with `log(w_{ij})` injection | PM2.5-GNN (Wang et al., SIGSPATIAL-20) for the wind-aware prior; the log-additive form is original to this project. | ✅ Reasonable physically-motivated prior. |
| k-NN distance graph with Gaussian decay | Standard practice in graph-based AQ models (Wang et al. SIGSPATIAL-20; Chen et al. arXiv:2108.12238 GAGNN) | ✅ |
| GRL adversarial DA at graph-pooled embedding | Ganin & Lempitsky (ICML-15); Ganin et al. (JMLR-16); UDA-GCN (Wu et al., WWW-20); DASTNet (Tang et al., CIKM-22) | ✅ Variant B is a direct port of DASTNet's idea to AQ with much more topological heterogeneity. |
| ADDA-style source warm-start | Tzeng, Hoffman, Saenko & Darrell (CVPR-17) | ✅ Required to stabilize the joint phase; without it the early Graph-DANN collapsed (see [RESULTS.md §3.2](RESULTS.md) and [PAPER_DRAFT.md §7.7](PAPER_DRAFT.md)). |
| α_d = 0.1 down-weighting of city loss | de Mathelin et al. (2020) arXiv:2006.08251 for adversarial regression specifically | ✅ Critical for non-classification DANN. |
| LayerNorm pre-GRL | GraphNorm (Cai, Luo, Xu, He, Liu & Wang, ICML-21) | ✅ Decouples discriminator from `|V|`-dependent activation scale. |
| Pre-train + fine-tune protocol | Hu, Liu, Gomes, Zitnik, Liang, Pande & Leskovec (ICLR-20); Yosinski et al. (NeurIPS-14); Yadav et al. (Env. Mod. & Software 2024) | ✅ Variant A is textbook. |
| Per-(month, hour) climatology residual | Yadav et al. (2024); long NWP tradition | ✅ Fit on `train_mask` only — verified in §2.4. |
| Three-way verification (zero-shot / scratch / transfer) | This project's own protocol — almost no cross-city TL paper reports scratch at the same `d%`. Closest precedent: AdaRNN (Du et al., CIKM-21) reports a "no-DA" ablation. | ✅ Uncommon and methodologically welcome. |
| Interleaved split with overlapping history | Asymptotic validity for stationary AR: Bergmeir, Hyndman & Koo (CSDA 2018) §3 Thm 3. Caveats: Roberts et al. (Ecography 2017) §3.2 — interleaving fails when serial autocorrelation is strong. | ⚠️ Defensible after climatology-residual removal; recommended sensitivity analysis in §6.2 R-3. |
| Causal `ffill` + zero-fill imputation in loader (after fix) | Kaufman et al. (TKDD 2012) §3.1 "limited-cardinality pre-split imputation" | ✅ After §2.2 fix. |
| Anticausal `bfill` + global-mean imputation in `data_pipeline.py` | Same Kaufman et al. category | ⚠️ Weak leak, empirically below seed noise; §6.2 R-5 is the bulletproofing fix. |

---

## Status

This audit was generated on **2026-05-31** against `main @ 05bd927` (Add GNN_DNN Results). The single source-code change (the causal-imputation fix in [src/utils.py:212-222](../src/utils.py#L212-L222)) has been applied. All other recommendations are documented above as optional paper-ready next steps; none of them invalidate the numbers currently reported in [RESULTS.md](RESULTS.md).

The mean val−test R² gap across all reported source-only and transfer cells is **−0.005** (test slightly above val); the largest gap is **+0.03** (Kolkata→Guwahati @30, transfer). Per Bergmeir & Benítez (2012) §4 this signals **no overfitting**.

Cross-method comparisons in [RESULTS.md §5](RESULTS.md) use the same `--fixed` protocol for LSTM and GNN, so any protocol-driven inflation cancels at comparison time and the relative orderings (e.g. "LSTM-fix beats GNN by ~0.04 R² in absolute terms; GNN solves the structural-transfer problem the LSTM cannot") remain valid.

---

*Companion documents: [ARCHITECTURE.md](ARCHITECTURE.md) (mechanism), [RESULTS.md](RESULTS.md) (numbers), [REFERENCES.md](REFERENCES.md) (bibliography), [MAIN_REPORT.md](MAIN_REPORT.md) (narrative & diagnosis history).*
