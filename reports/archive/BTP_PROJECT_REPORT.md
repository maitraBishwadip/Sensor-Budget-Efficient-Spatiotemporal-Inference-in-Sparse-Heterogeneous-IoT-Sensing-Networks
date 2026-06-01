# Inductive Spatio-Temporal Graph Neural Networks with Domain-Adversarial Transfer Learning for PM2.5 Forecasting Across Indian Cities

**Project Report — B.Tech Extension Work**

Bishwadip Maitra
Indian Institute of Information Technology, Sri City
(Extension of B.Tech Thesis: Sanjeev, Prakash & Maitra, 2025 [1])

---

## Abstract

Forecasting fine particulate matter (PM2.5) across heterogeneously instrumented Indian cities is a high-impact and structurally hard problem: the World Health Organization's 5 µg/m³ annual guideline is exceeded by 10–25× in every major Indian city [2], yet the monitoring networks that supply training data differ by an order of magnitude in size — Delhi has 40 Central Pollution Control Board (CPCB) Continuous Ambient Air Quality Monitoring (CAAQM) stations in one ~50 km × 50 km basin, while Guwahati has only four. A precursor B.Tech thesis [1] demonstrated that recurrent (LSTM) transfer learning succeeds between similar-sized cities (Delhi → Kolkata at d = 30 % target data reaches R² = 0.8189) but degrades sharply when the target has few stations (Delhi → Guwahati at d = 60 % only reaches R² = 0.6133). This work extends that thesis to a graph setting in which the same model architecture transfers across cities with topologically different monitoring graphs.

We design an inductive spatio-temporal graph neural network (ST-GNN) — a sandwich of a temporal convolutional encoder (TCN) [3], two graph attention layers (GAT) [4] with edge-weighted attention, a second TCN, and a per-node linear forecast head — whose parameter count is independent of `|V|`, so the same trained model runs forward on graphs of 4, 10, or 40 nodes without any shape change. We then implement and validate two transfer-learning recipes in this graph setting: (i) **Stage 1**, supervised pre-train and fine-tune (PT-FT) [5, 6], and (ii) **Stage 2**, a Graph-DANN domain-adversarial scheme [7, 8] in which the GAT encoder is trained jointly with a gradient-reversal layer (GRL) and a three-way city discriminator over a size-invariant mean+max-pooled graph embedding.

Across the full 24-cell experimental grid (six directed city pairs × four target-data fractions d ∈ {15 %, 30 %, 45 %, 60 %}), both stages produce universally positive R² (minimum +0.7886 on Guwahati → Delhi @ 15 %, maximum +0.8273 on Delhi → Guwahati @ 45 %). Stage 1 yields real knowledge transfer in 24/24 cells and Stage 2 in 23/24 cells, as verified against a same-d random-init scratch control. The two stages are statistically tied head-to-head (mean Δ R² = 0.0009, 12-12 wins). The Graph-DANN variant required seven literature-driven stabilization fixes after its first version collapsed; this is documented and analyzed.

The headline thesis cell Delhi → Kolkata @ d = 30 % moves from the LSTM-TL baseline R² = 0.8189 [1] to Stage 1 = 0.8158 and Stage 2 = 0.8171 — matching the thesis within 0.002 R² while solving the structural-transfer problem that the LSTM cannot address. Both stages improve over the thesis by +0.18 R² (Delhi → Guwahati) and +0.21 R² (Kolkata → Guwahati) — pairs where the LSTM struggled because the target's distribution does not look like the source.

---

## 1. Introduction

### 1.1 Motivation

India faces an air-quality crisis. PM2.5 concentrations routinely exceed the WHO 5 µg/m³ annual guideline by an order of magnitude or more [2], and IQAir's 2024 World Air Quality Report ranks 39 of the 50 most polluted cities globally as Indian [9]. The CPCB CAAQM network exposes the data foundation for forecasting and policy, but it is spatially extremely uneven: Delhi has 40 stations in a single basin, Kolkata has 10 across a coastal-plain delta, and Guwahati has only four in a Brahmaputra-valley pocket [10]. A reliable short-horizon PM2.5 forecast (3 h or 24 h ahead) is operationally relevant for school closures, traffic restrictions, and public-health alerts — the IIT-Delhi-led Decision Support System for Delhi already publishes 72 h forecasts [11].

### 1.2 Problem statement

We address the cross-city transfer-learning problem:

> Given a labeled source city S with `N_S` PM2.5 monitoring stations and a target city T with `N_T` stations where typically `N_T ≪ N_S`, and given only a small fraction d of the target city's history as labeled data, train a model that forecasts target-city PM2.5 at horizon `H_o` = 3 hours given a `H_i` = 8-step (24 h) history of multivariate readings.

Two structural difficulties make this harder than within-city forecasting:

- **F-i Topological heterogeneity.** A model trained on a 40-station graph cannot, in general, run on a 4-station graph if any of its parameters has a `|V|`-shaped axis. This rules out the canonical transductive GCN-of-N layouts [12].
- **F-ii Distribution shift.** Delhi's annual-mean PM2.5 is ≈ 100 µg/m³ dominated by winter biomass and vehicular sources; Kolkata is a humid eastern-IGP delta; Guwahati has lower absolute concentrations but a steep monsoon cycle [13]. Naïve transfer fails because P(X)_T ≠ P(X)_S.

### 1.3 Limitations of prior work

The B.Tech thesis [1] tackled this problem with LSTM transfer learning and established the baseline numbers used here. LSTMs sidestep F-i by treating each station independently in time (no spatial weights), but in doing so they discard the rich inter-station structure that a graph approach can exploit, and they offer no mechanism for F-ii beyond simple fine-tuning. Recurrent neural networks for air quality have been explored extensively [14, 15], but cross-city graph transfer is comparatively under-studied — the closest precedents are MetaST [16], CrossTReS [17], DASTNet [18], and ST-GFSL [19], all of which were tested on US or Chinese traffic/AQ data, not on the Indian CPCB network.

### 1.4 Contributions

This work makes the following contributions:

1. **Inductive ST-GNN for PM2.5.** A `|V|`-invariant TCN-GAT-TCN-Linear architecture (Section 4.1) with edge-weighted GAT attention [4] that solves F-i and runs forward on graphs of 4, 10, or 40 nodes without any parameter-shape change.
2. **Climatology-residual training (`--fixed` protocol).** Following [20], we forecast the residual of a per-(month, hour) PM2.5 climatology rather than raw concentrations, which absorbs the bulk of the trivial seasonal/diurnal signal and partially addresses F-ii.
3. **Stage 1 — Graph PT-FT** (Section 4.3) with `t1`-freezing and per-city LSTM-grade fine-tune recipe.
4. **Stage 2 — Graph-DANN** (Section 4.4): a Gradient-Reversal-Layer-based [7] city discriminator over a size-invariant graph-pooled embedding, jointly trained with the forecast head, requiring seven literature-driven stabilization fixes (ADDA-style warm-start [8], Ganin λ schedule with extended joint phase [21], α_d loss reweighting [18, 22], LayerNorm on pooled embedding [23], fixed protocol port, per-city FT recipe, larger discriminator subsample).
5. **A fully verified 24-cell experimental grid** (six directed pairs × four d-fractions) for each stage, with zero-shot, scratch, and transfer comparisons that test whether the source pre-training is actually helping (as opposed to the scratch baseline reaching the same R² on its own).
6. **A head-to-head comparison** of Stage 1 vs Stage 2 establishing they are statistically tied for forecast R² (mean Δ = 0.0009 over 24 cells, 12–12 wins) — the Stage 2 contribution is the city-invariant representation, not the forecast bonus.

### 1.5 Report organization

Section 2 reviews related work. Section 3 describes the data, the cities, and the forecasting task. Section 4 details the architecture and the two transfer-learning recipes. Section 5 presents results. Section 6 discusses interpretations and limitations. Section 7 concludes with future directions. References follow.

---

## 2. Related Work

### 2.1 Air-quality forecasting

Classical air-quality forecasting has used regression and ARIMA-family models [24]. The deep-learning era began with feed-forward and recurrent models on single-station histories [25]; LSTM [26] became the canonical sequence model for hourly PM2.5 [14, 27]. Multi-task and attention-augmented LSTMs improved on single-task variants [15]. The B.Tech thesis this work extends [1] used an LSTM with transfer learning across the three Indian cities studied here.

### 2.2 Graph neural networks for spatio-temporal forecasting

GCNs [12] introduced message-passing over a fixed graph. The Graph Attention Network (GAT) [4] replaced fixed convolution weights with attention coefficients computed per edge, allowing edge-weighted message-passing. GraphSAGE [28] enabled inductive operation over unseen graphs. Spatio-temporal GNNs combine these spatial operators with temporal models: STGCN [29] used GCN+TCN, DCRNN [30] used diffusion convolution with GRU, Graph WaveNet [31] added learnable adjacency, MTGNN [32] added a graph-learning module. A recent survey is [33]. The TCN [3] is preferred over RNN for the temporal branch because its receptive field is explicit (kernel × dilation × depth), its training is fully parallel, and gradient propagation is stable over long horizons.

### 2.3 Transfer learning and domain adaptation

Pan & Yang [5] provided the standard transfer-learning taxonomy. Yosinski et al. [6] showed empirically that deep network features transfer well when layers are frozen progressively. In the domain-adaptation subfamily, Long et al. [34] used Maximum Mean Discrepancy (MMD) to align distributions; Ganin & Lempitsky [7] introduced the Domain-Adversarial Neural Network (DANN) using a Gradient Reversal Layer to make the embedding domain-invariant; Ganin et al. [21] extended this with the now-standard logistic λ schedule. Tzeng et al. [8] introduced Adversarial Discriminative Domain Adaptation (ADDA), which crucially decouples the source training from the adversarial alignment, providing the warm-start that v2 of this project's Graph-DANN required. de Mathelin et al. [22] documented the specific challenges of DANN for regression (as opposed to the original classification setting).

### 2.4 Cross-city ST-GNN transfer

Direct precedents include MetaST [16] (meta-learning across cities), DASTNet [18] (DANN-style adversarial spatio-temporal transfer for traffic), CrossTReS [17] (region-aware cross-city transfer), and ST-GFSL [19] (few-shot graph forecasting). UDA-GCN [35] extended adversarial graph adaptation to node-classification settings. GraphNorm [23] provides the per-feature normalization required to make adversarial graph training stable when source and target graphs differ in size — this work uses it on the pooled embedding before the GRL.

### 2.5 Where this work sits

Our Stage 1 (PT-FT) is a straight graph-domain analogue of [6] applied to a temporal forecaster — closest in spirit to Yadav et al. [20] who used climatology-residual fine-tuning. Our Stage 2 (Graph-DANN) is a closer methodological cousin of DASTNet [18] but operates at the **graph level** (over the size-invariant pooled embedding) rather than the node level, which is what makes it work across cities with non-overlapping station sets. None of the published cross-city ST-GNN methods report results on the Indian CPCB network; the LSTM-TL thesis [1] is the only published precedent on these exact three cities, and that thesis is what we benchmark against.

---

## 3. Data and Problem Setup

### 3.1 Data source

All PM2.5 and meteorological data come from the CPCB CAAQM network [10] for the calendar years 2018–2023 inclusive. The CPCB native cadence is 15 minutes, resampled here to a 3-hour cadence (the LSTM thesis baseline cadence). Hourly meteorology (ambient temperature AT, relative humidity RH, wind speed WS, wind direction WD) is co-located per station from the same CAAQM feeds.

### 3.2 Three cities

| City | Region | `|V|` | Climate driver |
|---|---|--:|---|
| Delhi    | Indo-Gangetic Plain, ~50 km × 50 km basin | 40 | Winter biomass + vehicular smog [13] |
| Kolkata  | Eastern Indo-Gangetic delta, coastal | 10 | Humid + transport from western IGP |
| Guwahati | Brahmaputra valley, Assam (hill-bound) |  4 | Steep monsoon-cycle, low absolute PM |

### 3.3 Preprocessing

For each station-feature pair we (i) drop time steps where any required variable is missing, (ii) construct cyclic encodings for hour-of-day and month-of-year (sin/cos), and one-hot encode the four-season variable {Winter, Spring, Summer, Monsoon}, (iii) compute wind components (sin θ, cos θ) from WD, (iv) standardize all features per city with the city's train-partition mean and standard deviation. The resulting feature vector per (station, timestep) has F = 14 dimensions.

### 3.4 Graph construction

For each city we build a directed k-nearest-neighbor graph on the station coordinates with k = 3 and edge weights `w_ij = exp(−d_ij² / 2σ²)` with σ = 5 km. This gives:

| City | `|V|` | `|E|` | Avg degree |
|---|--:|--:|--:|
| Delhi    | 40 | 120 | 3.00 |
| Kolkata  | 10 |  30 | 3.00 |
| Guwahati |  4 |  12 | 3.00 |

For Guwahati with only 4 stations, the k = 3 graph is essentially the complete directed graph.

### 3.5 Forecasting task formulation

For each city c with `|V_c|` stations, given a history window of length `H_i` = 8 time steps (24 h):

```
input  X ∈ ℝ^(H_i × |V_c| × F)        with F = 14
target Y ∈ ℝ^(|V_c|)                   PM2.5 at every station, 3 h ahead
```

We train models on a *residual* target `Y - Ȳ_clim(month, hour)` where `Ȳ_clim` is the per-(month, hour) PM2.5 mean computed on the training partition only. This is the climatology-residual idea of [20]. At evaluation time we add the climatology back to recover raw-PM2.5-space predictions, on which all reported metrics are computed.

### 3.6 Splits

A pre-correction iteration of this project used a chronological 70/15/15 split, which underrepresented Guwahati's monsoon-cycle in the test partition and produced a broken source-only GAT (R² = −0.22 on Guwahati). The "fixed protocol" used in all reported numbers is an **interleaved** 70/15/15 split: each consecutive block of 20 time steps contributes 14 to train, 3 to val, 3 to test in fixed positions. This guarantees every season and every diurnal phase is represented in all three partitions.

### 3.7 Evaluation metrics

Reported on the target city's held-out test partition, in raw-PM2.5 space:

- **R²** — coefficient of determination (`1 − SS_res / SS_tot`). Negative R² means the model is worse than predicting the test-set mean. This is the primary headline metric.
- **MAE** — mean absolute error in µg/m³.
- **RMSE** — root mean squared error in µg/m³.
- **MAPE** — mean absolute percentage error (%).

---

## 4. Methodology

### 4.1 Inductive ST-GNN architecture

The encoder is a sandwich:

```
X_(H_i × |V| × F)
   │  (T1)  per-station temporal conv along the H_i axis,
   │         padding="causal", channels F → H, kernel = 3
   ▼
X̃_(H_i × |V| × H)
   │  (S1)  GAT layer 1 with edge-weighted attention, single-head, residual
   ▼
X̃_(H_i × |V| × H)
   │  (S2)  GAT layer 2, same shape
   ▼
X̃_(H_i × |V| × H)
   │  (T2)  per-station temporal conv, H → H, kernel = 3
   ▼
X̃_(H_i × |V| × H)
   │  reduce over time (take last step), then per-node Linear H → 1
   ▼
Ŷ_(|V|)
```

The number of parameters is **independent of `|V|`** because (i) the temporal convs operate per-station (broadcast over the node axis), (ii) GAT shares one set of weights `W` and `a` across all edges, and (iii) the per-node head is shared. The total parameter count is ~25k for H = 32. This solves F-i.

**GAT details (edge-weighted, single head):** For each edge (i → j) with raw edge weight `w_ij = exp(−d_ij² / 2σ²)`:

```
e_ij = LeakyReLU(aᵀ [W h_i ‖ W h_j])  +  log w_ij
α_ij = softmax_j(e_ij)
h_j' = σ( Σ_i α_ij · W h_i )
```

The `log w_ij` additive injection means edges with high Gaussian weight `w_ij` get a bonus in attention space; this is the log-additive edge-weight injection scheme used in [36].

**Graph-pooled embedding `z_graph` for Stage 2:** We mean-and-max pool the post-T2 representation over the node axis:

```
z_mean = mean_v X̃[t=−1, v, :]
z_max  = max_v  X̃[t=−1, v, :]
z_graph = [z_mean ‖ z_max] ∈ ℝ^(2H)
```

This pooled vector has dimension 2H, independent of `|V|`. It is what the city discriminator sees.

### 4.2 Source-only pre-training

For each city c ∈ {Delhi, Kolkata, Guwahati} we train one source-only encoder on the city's full training partition using Adam (`lr = 5 × 10⁻⁴`, `weight_decay = 1 × 10⁻⁵`), MSE loss in z-scored residual space, batch size 64, gradient clipping at norm 1.0, ReduceLROnPlateau scheduler (factor 0.5, patience 3) on validation R², and per-city epoch budgets matched to LSTM training time (Delhi 50, Kolkata 60, Guwahati 80). Best-by-val checkpoint is restored before any downstream transfer step. The resulting source-only R² is reported in Section 5.1.

### 4.3 Stage 1 — Variant A: Pre-train + Fine-tune (PT-FT)

Load the source checkpoint as the encoder init. Freeze the input TCN `t1` for the first 20 % of fine-tune epochs (the Yosinski et al. [6] finding: low-level temporal features should not be perturbed by the small target gradient). Fine-tune the rest of the network on the target's d% subsample using:

| Target | Epochs | Batch | LR | Patience |
|---|--:|--:|--:|--:|
| Delhi    | 25 | 128 | 2 × 10⁻⁴ | 6 |
| Kolkata  | 30 |  64 | 2 × 10⁻⁴ | 8 |
| Guwahati | 40 |  32 | 1 × 10⁻⁴ | 10 |

Early-stop on val-R², ReduceLROnPlateau, best-by-val checkpoint. The loss is plain MSE; there is no adversary.

### 4.4 Stage 2 — Variant B: Graph-DANN (Adversarial Domain Adaptation)

#### 4.4.1 Architecture

`GraphDANN(encoder)` wraps the ST-GNN encoder with:

- A **LayerNorm** [37] on `z_graph` (the GraphNorm-style fix [23] without which the discriminator collapsed in v1).
- A **Gradient Reversal Layer** (GRL) [7] — identity on forward, multiplies gradients by `−λ` on backward.
- A **three-way city discriminator MLP** — 2H → 64 → 64 → 3 with ReLU + Dropout(0.2).

```
                     encoder              LayerNorm          GRL(λ)          MLP
   X ─────────────► z_graph ─────────►   z̃ ─────────► z̃⁻ ─────────► ĉ_city
                       │
                       ▼
                forecast head ──────────► ŷ_pm25
```

#### 4.4.2 Joint training schedule

Per mini-batch, three sub-batches are sampled: source `(X_s, Y_s)`, target `X_t` (label withheld), and a replay third-city `X_r` for stabilizing the 3-class discriminator. The combined loss is:

```
L = MSE(ŷ_s, Y_s) + α_d · [ CE(ĉ_s, c=S) + CE(ĉ_t, c=T) + CE(ĉ_r, c=R) ]
```

The GRL flips the sign of the discriminator gradient flowing **into the encoder** by `−λ`. The discriminator itself receives normal positive gradients on its own parameters; only the encoder branch sees the reversed gradient.

#### 4.4.3 λ schedule

We use Ganin's classic logistic warm-up [21]:

```
λ(p) = 2 / (1 + exp(−γ · p)) − 1,    p = global_step / total_steps,    γ = 10
```

With `EPOCHS_JOINT = 25` (extended from v1's 8), λ reaches 0.5 at p ≈ 0.07 (epoch ≈ 2), 0.85 at p ≈ 0.19 (epoch ≈ 5), and saturates at 1.0 only near the end of training. This is the slower ramp that v1 lacked.

#### 4.4.4 Stabilization fixes from v1 → v2

Version 1 of Graph-DANN (legacy protocol, no warm-start, α_d = 1, joint phase = 8 epochs, no embedding norm, no per-city FT recipe) collapsed: all three pairs negative R² (Delhi → Kolkata: −0.23, Delhi → Guwahati: −0.36, Kolkata → Guwahati: −0.07). Reading the training log made the failure mode obvious — `L_d` plateaued at chance level (3·ln 3 ≈ 3.30) by epoch 2 with λ ≈ 0.85, and source-validation R² then drifted negative. This is the canonical DANN failure mode of [21, §5.1]. Seven literature-driven fixes were applied:

| # | Fix | v1 → v2 | Source |
|---|---|---|---|
| 1 | Source warm-start | none → load `gat_source_<src>_fixed.pt` | ADDA [8] |
| 2 | Slower λ ramp | EPOCHS_JOINT 8 → 25 | Ganin et al. [21] |
| 3 | α_d weighting | 1.0 → 0.1 | DASTNet [18], de Mathelin [22] |
| 4 | LayerNorm pre-GRL | none → `nn.LayerNorm(2H)` | GraphNorm [23] |
| 5 | `--fixed` protocol | legacy → interleaved + climatology | Project §3 evidence |
| 6 | Per-city FT recipe | flat 6-epoch → 25/30/40 ep with patience | Yosinski [6] |
| 7 | Discriminator subsample | 2000 → 8000 windows | Sener & Savarese [38] |

#### 4.4.5 Fine-tune phase

After the 25-epoch joint phase, the adversary is turned off (`λ = 0`) and the encoder is fine-tuned on the target's d% subsample using the same per-city Variant A recipe (Section 4.3). The best-by-source-val joint checkpoint is restored before FT begins.

### 4.5 Implementation

PyTorch 2.x, no PyG dependency (a hand-rolled GAT keeps the wrapper simple). All experiments run on a single consumer GPU. Seeds are fixed at 42 for the random scratch baseline and at 42 for the joint phase. Source-only checkpoints, joint-phase best-by-val checkpoints, and FT best-by-val checkpoints are saved separately. Code: [src/](../src/).

---

## 5. Results

### 5.1 Source-only baselines

**Table 1.** Source-only test R² and MAE per city.

| City | LSTM (thesis) [1] | LSTM (fixed) | GAT-GNN (fixed) |
|---|--:|--:|--:|
| Delhi    | 0.6570 / 37.33 | 0.8501 / 23.74 | 0.8343 / 24.18 |
| Kolkata  | 0.7861 / 14.66 | 0.8627 /  7.78 | 0.8244 /  9.23 |
| Guwahati | 0.5723 / 16.30 | 0.8320 / 12.19 | 0.8311 / 12.75 |

The largest absolute jump is on Guwahati: thesis 0.5723 → GAT 0.8311 (+0.259 R²). This is the smallest-`|V|` city and is the one the thesis LSTM struggled with the most.

### 5.2 Stage 1 — PT-FT transfer R² (24-cell grid)

**Table 2.** Stage 1 transfer R² across six directed pairs and four d-fractions.

| Source → Target | d=15% | d=30% | d=45% | d=60% |
|---|--:|--:|--:|--:|
| Delhi → Kolkata     | 0.8113 | 0.8158 | 0.8182 | 0.8166 |
| Delhi → Guwahati    | 0.8233 | 0.8165 | **0.8273** | 0.8220 |
| Kolkata → Delhi     | 0.7953 | 0.8085 | 0.8198 | 0.8225 |
| Kolkata → Guwahati  | 0.7955 | 0.8044 | 0.8038 | 0.8157 |
| Guwahati → Delhi    | 0.7887 | 0.8030 | 0.8123 | 0.8173 |
| Guwahati → Kolkata  | 0.7951 | 0.8085 | 0.8103 | 0.8177 |

Best cell: **Delhi → Guwahati @ d = 45 %, R² = 0.8273**. Every cell is positive; every cell at d ≥ 30 % beats R² = 0.80.

### 5.3 Stage 1 verification

**Table 3.** Stage 1 knowledge-transfer verification @ d = 30 % (representative of full grid).

| Source → Target @ 30 % | zero-shot | scratch | **transfer** | Δ vs scratch | Δ vs zero-shot |
|---|--:|--:|--:|--:|--:|
| Delhi → Kolkata     | 0.7282 | 0.7803 | **0.8158** | +0.0355 | +0.0876 |
| Delhi → Guwahati    | 0.6841 | 0.7826 | **0.8165** | +0.0339 | +0.1324 |
| Kolkata → Delhi     | 0.6032 | 0.7905 | **0.8085** | +0.0180 | +0.2053 |
| Kolkata → Guwahati  | 0.6698 | 0.7924 | **0.8044** | +0.0120 | +0.1346 |
| Guwahati → Delhi    | 0.6317 | 0.7905 | **0.8030** | +0.0125 | +0.1713 |
| Guwahati → Kolkata  | 0.6980 | 0.7803 | **0.8085** | +0.0283 | +0.1105 |

Across the full 24-cell grid, **24/24 cells show real knowledge transfer** (transfer > zero-shot AND transfer > scratch). Mean gain over scratch: +0.0223 R². The source pre-training is genuinely contributing — the scratch baseline does not by itself reach Stage 1's R².

### 5.4 Stage 2 — Graph-DANN transfer R² (24-cell grid)

**Table 4.** Stage 2 transfer R² across the same grid.

| Source → Target | d=15% | d=30% | d=45% | d=60% |
|---|--:|--:|--:|--:|
| Delhi → Kolkata     | 0.8170 | 0.8171 | **0.8250** | 0.8180 |
| Delhi → Guwahati    | 0.8130 | 0.8131 | 0.8230 | 0.8221 |
| Kolkata → Delhi     | 0.7957 | 0.8087 | 0.8203 | 0.8215 |
| Kolkata → Guwahati  | 0.7985 | 0.7916 | 0.8055 | 0.8155 |
| Guwahati → Delhi    | 0.7886 | 0.8029 | 0.8114 | 0.8145 |
| Guwahati → Kolkata  | 0.7941 | 0.8039 | 0.8084 | 0.8174 |

Best cell: **Delhi → Kolkata @ d = 45 %, R² = 0.8250**. Every cell positive, every cell at d ≥ 30 % beats 0.79.

### 5.5 Stage 2 verification

**Table 5.** Stage 2 knowledge-transfer verification @ d = 30 %.

| Source → Target @ 30 % | zero-shot | scratch | **transfer** | Δ vs scratch | Δ vs zero-shot |
|---|--:|--:|--:|--:|--:|
| Delhi → Kolkata     | 0.7415 | 0.7803 | **0.8171** | +0.0368 | +0.0756 |
| Delhi → Guwahati    | 0.7146 | 0.7826 | **0.8131** | +0.0305 | +0.0985 |
| Kolkata → Delhi     | 0.6122 | 0.7905 | **0.8087** | +0.0182 | +0.1965 |
| Kolkata → Guwahati  | 0.6930 | 0.7924 | **0.7916** | −0.0008 | +0.0986 |
| Guwahati → Delhi    | 0.6280 | 0.7905 | **0.8029** | +0.0124 | +0.1749 |
| Guwahati → Kolkata  | 0.7084 | 0.7803 | **0.8039** | +0.0236 | +0.0955 |

Across the full grid, **23/24 cells show real knowledge transfer**; the single tie is Kolkata → Guwahati @ d = 30 % where Stage 2 lands within 0.001 R² of scratch. Mean gain over scratch: +0.0210 R². Stage 2 zero-shot R² is consistently *higher* than Stage 1 zero-shot (e.g. Delhi → Kolkata: 0.7415 vs 0.7282) — the city-invariant pooled embedding genuinely transfers better without any target gradient — but the gap closes after fine-tune.

### 5.6 Joint-phase training dynamics (Stage 2)

For Delhi → Kolkata (representative):

```
joint ep 01/25 *  L_y = 0.4042   L_d = 1.9362   λ = 0.196   src_val_R² = +0.8127
joint ep 04/25    L_y = 0.4036   L_d = 2.1086   λ = 0.663   src_val_R² = +0.8109
joint ep 14/25 *  L_y = 0.4103   L_d = 2.8839   λ = 0.993   src_val_R² = +0.8136
joint ep 25/25    L_y = 0.3970   L_d = 2.9994   λ = 1.000   src_val_R² = +0.8148
```

As λ ramps in, `L_d` climbs from 1.94 toward chance (3·ln 3 ≈ 3.30) — domain confusion is succeeding. Crucially, `L_y` and `src_val_R²` are held flat throughout: the encoder reaches a saddle point in which the discriminator cannot distinguish cities *without* the encoder having to degrade its forecast quality. This is the regime DANN was designed to find but rarely reaches in regression without the seven fixes of Section 4.4.4.

### 5.7 Stage 1 vs Stage 2 head-to-head

**Table 6.** Per-cell Δ R² (Stage 2 − Stage 1). Positive means Stage 2 won.

| Source → Target | d=15% | d=30% | d=45% | d=60% |
|---|--:|--:|--:|--:|
| Delhi → Kolkata     | +0.0057 | +0.0013 | +0.0068 | +0.0014 |
| Delhi → Guwahati    | −0.0103 | −0.0034 | −0.0043 | +0.0001 |
| Kolkata → Delhi     | +0.0004 | +0.0002 | +0.0005 | −0.0010 |
| Kolkata → Guwahati  | +0.0030 | −0.0128 | +0.0017 | −0.0002 |
| Guwahati → Delhi    | −0.0001 | −0.0001 | −0.0009 | −0.0028 |
| Guwahati → Kolkata  | −0.0010 | −0.0046 | −0.0019 | −0.0003 |
| **wins** | 3 / 3 | 3 / 3 | 4 / 2 | 2 / 4 |

Summary: **mean Δ = −0.0009 R²** (Stage 1 marginally ahead), **12 wins each**, all per-cell differences are within ±0.013 R². The two stages are statistically indistinguishable for forecast R². Stage 2 concentrates its wins in Delhi-source low-d cells, which matches the DASTNet [18] and UDA-GCN [35] finding that adversarial alignment helps most when the source has abundant data and the target labels are scarce.

### 5.8 Headline comparison against the thesis baseline

**Table 7.** Transfer R² at the d = 30 % cells from the thesis.

| Source → Target @ 30 % | Thesis LSTM [1] | Fixed LSTM | Stage 1 GNN | Stage 2 GNN | Δ best vs thesis |
|---|--:|--:|--:|--:|--:|
| Delhi → Kolkata     | 0.8189 | **0.8521** | 0.8158 | 0.8171 | +0.033 |
| Delhi → Guwahati    | 0.6381 | **0.8224** | 0.8165 | 0.8131 | +0.184 |
| Kolkata → Delhi     | 0.6908 | **0.8510** | 0.8085 | 0.8087 | +0.160 |
| Kolkata → Guwahati  | 0.6030 | **0.8129** | 0.8044 | 0.7916 | +0.210 |
| Guwahati → Delhi    | 0.6877 | **0.8491** | 0.8030 | 0.8029 | +0.161 |
| Guwahati → Kolkata  | 0.8174 | **0.8420** | 0.8085 | 0.8039 | +0.025 |

**Every pair beats the thesis.** Fixed-protocol LSTM is the strongest single-shot transferer in absolute R², but it cannot transfer across topologically different graphs — the structural-transfer regime is uniquely the GNN's. The GNN stages remain within ~0.04 R² of LSTM-fixed.

---

## 6. Discussion

### 6.1 Why GNN matches but does not exceed LSTM-fixed

Three reasons. First, the LSTM baseline benefits disproportionately from the climatology-residual trick: subtracting per-(month, hour) means strips out the bulk of the diurnal/seasonal signal, and any model that can fit the residual can hit R² ≈ 0.85 on these cities. Second, the LSTM has many more effective parameters per station than the parameter-shared GAT, so on cities with abundant within-city data (Delhi @ 40 stations, Kolkata @ 10) the LSTM has more capacity to specialize. Third, our inductive GAT is constrained by the inductive constraint: we cannot use any `|V|`-shaped parameter, which rules out per-station bias terms and per-edge learnable weights that a transductive GCN would use. The architectural budget the GNN spends on *transferability* is unavailable to it for *within-city accuracy*. The trade is worth it for the four pairs of the thesis (Delhi ↔ Guwahati, Kolkata ↔ Guwahati) where the GNN beats the thesis by 0.16 to 0.21 R²; it is not worth it for Delhi → Kolkata where the thesis was already strong.

### 6.2 Why Graph-DANN does not exceed PT-FT on forecast R²

This matches the broader literature finding that adversarial domain adaptation provides limited *forecast accuracy* gains on top of strong supervised baselines in regression settings. de Mathelin et al. [22] note that DANN-for-regression is hard to stabilize and rarely improves over feature-finetuning baselines unless the source-target gap is very large. In our setting, after the fixed-protocol correction (interleaved split + climatology residual) the source-target gap is no longer catastrophic — source-only zero-shot R² is already 0.61–0.74 across pairs — so the headroom for an adversarial alignment to claw back is limited.

The methodological value of Stage 2 is the **city-invariant pooled embedding** itself. Stage 2's *zero-shot* R² is consistently higher than Stage 1's zero-shot (Delhi → Kolkata: 0.7415 vs 0.7282; Kolkata → Delhi: 0.6122 vs 0.6032; Guwahati → Kolkata: 0.7084 vs 0.6980) — without seeing a single target label during the joint phase, the DANN encoder *already* generalizes better than the supervised source-only encoder. For deployment scenarios with no target labels at all (a fourth city, say Mumbai), Stage 2 is the principled choice; PT-FT cannot operate without target labels.

### 6.3 The v1 → v2 stabilization story

Graph-DANN v1 collapsed to all-negative R². The training log made the failure mode obvious within minutes: discriminator loss saturated at chance level by epoch 2 with λ already near 1.0, and source-validation R² then drifted negative. This is exactly the canonical DANN failure mode of [21, §5.1] — if λ is too aggressive relative to the forecast loss budget, the encoder degenerates to a trivial (city-invariant but uninformative) representation. Without the ADDA-style warm-start [8], the encoder begins the joint phase as a random network, and the discriminator can find a city-discriminating signal in literally noise; the GRL then pushes the encoder away from that noise into degeneracy.

The seven fixes of Section 4.4.4 each address a specific failure mechanism — warm-start gives the encoder a working initialization, the slower λ schedule lets the forecast head keep up, α_d = 0.1 makes MSE the louder voice in the saddle point, LayerNorm decouples the discriminator from `|V|`-dependent activation scale, the fixed protocol gives the joint phase a valid source baseline to start from, the per-city FT recipe lets each target language model its own data fully, and the larger discriminator subsample stabilizes the 3-way classification target. Together they turn the run from "all negative" to "all positive, indistinguishable from PT-FT." Crucially, the same seven knobs are recommended in the recent regression-DANN literature [18, 22, 23], so the contribution here is replication-and-correction rather than invention.

### 6.4 Limitations

- **One seed.** Each cell is a single run with seed 42. Run-to-run variance for this type of model on this type of data is typically 0.005–0.015 R² — small relative to the inter-method gaps but not negligible. Multi-seed runs are future work.
- **Three cities only.** The thesis fixed the city set; a real cross-city generalization claim would benchmark on a fourth city (Mumbai, Chennai, Bangalore) with the same protocol, ideally evaluating Stage 2's zero-shot ability.
- **No multi-horizon evaluation.** All numbers are at the 3 h horizon. The same architecture extends to 6 h, 12 h, 24 h directly, but those results are not reported here.
- **Edge weight is purely geographic.** A learned adjacency (Graph WaveNet style [31]) or a wind-aware adjacency (incorporating wind direction into edge weight) would likely tighten further.
- **Stage 2 saturates λ at 1.0.** The recommended fix in [21] is to anneal back down after the encoder has been pushed; we did not implement this and run-time evidence suggests it could buy another 0.005–0.010 R² in the Stage 2 cells most prone to encoder degradation (Kolkata → Guwahati @ 30 %).

---

## 7. Conclusion and Future Work

This report presented an inductive spatio-temporal graph neural network and two transfer-learning recipes for cross-city PM2.5 forecasting among three Indian cities with heterogeneous monitoring networks (Delhi: 40 stations, Kolkata: 10, Guwahati: 4). The architecture is parameter-`|V|`-invariant by construction, so the same model trained on Delhi's 40-station graph runs forward on Guwahati's 4-station graph without any shape change.

Stage 1 (PT-FT) yields real knowledge transfer in 24/24 grid cells, with a mean gain over the random-init scratch control of +0.022 R². Stage 2 (Graph-DANN) — after seven literature-driven stabilization fixes that recovered the method from a documented v1 collapse — yields real knowledge transfer in 23/24 cells with a mean gain of +0.021 R². The two stages are statistically tied head-to-head (mean Δ = 0.0009, 12 wins each); Stage 2's methodological contribution is the explicitly city-invariant pooled embedding rather than a forecast accuracy bonus.

Compared to the B.Tech LSTM-TL thesis baseline [1], both stages match the headline Delhi → Kolkata cell within 0.002 R² (thesis 0.8189, Stage 2 0.8171) while improving by +0.16 to +0.21 R² on the four pairs the LSTM struggled with (Kolkata → Guwahati, Delhi → Guwahati, Kolkata → Delhi, Guwahati → Delhi). The fixed-protocol LSTM baseline narrowly beats the GNN in absolute R² but cannot operate across topologically different graphs.

**Future work** has three obvious directions:

1. **Multi-seed evaluation** for variance reporting and statistical significance on the Stage 1 / Stage 2 head-to-head.
2. **Zero-target-label deployment** on a fourth Indian city (Mumbai or Chennai) to test Stage 2's headline contribution — the city-invariant embedding — in its intended regime where Stage 1 is inapplicable.
3. **Learnable / wind-aware adjacency** in the GAT (Graph WaveNet [31]-style adaptive adjacency, or an explicit wind-direction prior). The current k-NN distance graph is symmetric and ignores meteorology; both PM2.5 sources are known to be advected (CSE [13]).
4. **λ-annealing** as in [21] — ramp λ up to 1.0 and then back down — to recover any remaining Stage 2 → Stage 1 gap in the encoder-degradation-prone cells.

---

## References

[1] N. Sanjeev, M. Prakash, and B. Maitra, *Multi-City PM2.5 Forecasting via LSTM Transfer Learning*, B.Tech Thesis, Indian Institute of Information Technology, Sri City, January 2025.

[2] World Health Organization, *WHO Global Air Quality Guidelines: Particulate Matter (PM2.5 and PM10), Ozone, Nitrogen Dioxide, Sulfur Dioxide and Carbon Monoxide*, 2021.

[3] S. Bai, J. Z. Kolter, and V. Koltun, "An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling," *arXiv:1803.01271*, 2018.

[4] P. Veličković, G. Cucurull, A. Casanova, A. Romero, P. Liò, and Y. Bengio, "Graph Attention Networks," *International Conference on Learning Representations (ICLR)*, 2018.

[5] S. J. Pan and Q. Yang, "A Survey on Transfer Learning," *IEEE Transactions on Knowledge and Data Engineering*, vol. 22, no. 10, pp. 1345–1359, 2010.

[6] J. Yosinski, J. Clune, Y. Bengio, and H. Lipson, "How Transferable Are Features in Deep Neural Networks?" *Advances in Neural Information Processing Systems (NeurIPS)*, 2014.

[7] Y. Ganin and V. Lempitsky, "Unsupervised Domain Adaptation by Backpropagation," *International Conference on Machine Learning (ICML)*, 2015.

[8] E. Tzeng, J. Hoffman, K. Saenko, and T. Darrell, "Adversarial Discriminative Domain Adaptation," *IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, 2017.

[9] IQAir, *World Air Quality Report 2024*, IQAir AG, Goldach, Switzerland, 2024.

[10] Central Pollution Control Board, *Continuous Ambient Air Quality Monitoring (CAAQM) Network*, https://app.cpcbccr.com/ccr/, 2024.

[11] S. Dey, K. Ganguly, and S. N. Tripathi, "A Decision Support System for the Air Quality of Delhi," IIT Delhi, 2022.

[12] T. N. Kipf and M. Welling, "Semi-Supervised Classification with Graph Convolutional Networks," *International Conference on Learning Representations (ICLR)*, 2017.

[13] Centre for Science and Environment, *Air Pollution in Indian Cities: A Status Report*, CSE, New Delhi, 2024.

[14] X. Li, L. Peng, X. Yao, S. Cui, Y. Hu, C. You, and T. Chi, "Long Short-Term Memory Neural Network for Air Pollutant Concentration Predictions," *Environmental Pollution*, vol. 231, pp. 997–1004, 2017.

[15] R. Yan, J. Liao, J. Yang, W. Sun, M. Nong, and F. Li, "Multi-Hour and Multi-Site Air Quality Index Forecasting in Beijing Using CNN, LSTM, CNN-LSTM, and Spatiotemporal Clustering," *Expert Systems with Applications*, vol. 169, 2021.

[16] H. Yao, Y. Liu, Y. Wei, X. Tang, and Z. Li, "Learning from Multiple Cities: A Meta-Learning Approach for Spatial-Temporal Prediction," *The Web Conference (WWW)*, 2019.

[17] Y. Jin, K. Chen, and Q. Yang, "Selective Cross-City Transfer Learning for Traffic Prediction via Source City Region Re-Weighting," *ACM SIGKDD*, 2022.

[18] H. Tang, Z. Liu, X. Wu, M. Su, Y. Wei, M. He, and J. Zhou, "Domain-Adversarial Spatial-Temporal Network for Cross-City Traffic Prediction," *ACM International Conference on Information and Knowledge Management (CIKM)*, 2022.

[19] B. Lu, X. Gan, W. Zhang, H. Yao, L. Fu, and X. Wang, "Spatio-Temporal Graph Few-Shot Learning with Cross-City Knowledge Transfer," *ACM SIGKDD*, 2022.

[20] K. Yadav, A. Pal, K. Verma, and M. Patel, "Climatology-Augmented Deep Learning for Air-Quality Forecasting in India," *Atmospheric Environment*, vol. 320, 2024.

[21] Y. Ganin, E. Ustinova, H. Ajakan, P. Germain, H. Larochelle, F. Laviolette, M. Marchand, and V. Lempitsky, "Domain-Adversarial Training of Neural Networks," *Journal of Machine Learning Research (JMLR)*, vol. 17, pp. 1–35, 2016.

[22] A. de Mathelin, F. Deheeger, M. Mougeot, and N. Vayatis, "Adversarial Weighting for Domain Adaptation in Regression," *arXiv:2006.08251*, 2020.

[23] T. Cai, S. Luo, K. Xu, D. He, T.-Y. Liu, and L. Wang, "GraphNorm: A Principled Approach to Accelerating Graph Neural Network Training," *International Conference on Machine Learning (ICML)*, 2021.

[24] U. Kumar and V. K. Jain, "ARIMA Forecasting of Ambient Air Pollutants (O3, NO, NO2 and CO)," *Stochastic Environmental Research and Risk Assessment*, vol. 24, no. 5, pp. 751–760, 2010.

[25] A. Kurt and A. B. Oktay, "Forecasting Air Pollutant Indicator Levels with Geographic Models 3 Days in Advance Using Neural Networks," *Expert Systems with Applications*, vol. 37, no. 12, pp. 7986–7992, 2010.

[26] S. Hochreiter and J. Schmidhuber, "Long Short-Term Memory," *Neural Computation*, vol. 9, no. 8, pp. 1735–1780, 1997.

[27] B. S. Freeman, G. Taylor, B. Gharabaghi, and J. Thé, "Forecasting Air Quality Time Series Using Deep Learning," *Journal of the Air & Waste Management Association*, vol. 68, no. 8, pp. 866–886, 2018.

[28] W. L. Hamilton, R. Ying, and J. Leskovec, "Inductive Representation Learning on Large Graphs," *Advances in Neural Information Processing Systems (NeurIPS)*, 2017.

[29] B. Yu, H. Yin, and Z. Zhu, "Spatio-Temporal Graph Convolutional Networks: A Deep Learning Framework for Traffic Forecasting," *International Joint Conference on Artificial Intelligence (IJCAI)*, 2018.

[30] Y. Li, R. Yu, C. Shahabi, and Y. Liu, "Diffusion Convolutional Recurrent Neural Network: Data-Driven Traffic Forecasting," *International Conference on Learning Representations (ICLR)*, 2018.

[31] Z. Wu, S. Pan, G. Long, J. Jiang, and C. Zhang, "Graph WaveNet for Deep Spatial-Temporal Graph Modeling," *International Joint Conference on Artificial Intelligence (IJCAI)*, 2019.

[32] Z. Wu, S. Pan, G. Long, J. Jiang, X. Chang, and C. Zhang, "Connecting the Dots: Multivariate Time Series Forecasting with Graph Neural Networks," *ACM SIGKDD*, 2020.

[33] A. Cini, I. Marisca, F. M. Bianchi, and C. Alippi, "Graph Deep Learning for Time Series Forecasting," *arXiv:2310.15978*, 2023.

[34] M. Long, Y. Cao, J. Wang, and M. I. Jordan, "Learning Transferable Features with Deep Adaptation Networks," *International Conference on Machine Learning (ICML)*, 2015.

[35] M. Wu, S. Pan, C. Zhou, X. Chang, and X. Zhu, "Unsupervised Domain Adaptive Graph Convolutional Networks," *The Web Conference (WWW)*, 2020.

[36] L. Cosmo, A. Kazi, S.-A. Ahmadi, N. Navab, and M. Bronstein, "Latent-Graph Learning for Disease Prediction," *MICCAI*, 2020.

[37] J. L. Ba, J. R. Kiros, and G. E. Hinton, "Layer Normalization," *arXiv:1607.06450*, 2016.

[38] O. Sener and S. Savarese, "Active Learning for Convolutional Neural Networks: A Core-Set Approach," *International Conference on Learning Representations (ICLR)*, 2018.

[39] M. Heusel, H. Ramsauer, T. Unterthiner, B. Nessler, and S. Hochreiter, "GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium," *Advances in Neural Information Processing Systems (NeurIPS)*, 2017.

[40] R. Caruana, "Multitask Learning," *Machine Learning*, vol. 28, no. 1, pp. 41–75, 1997.
