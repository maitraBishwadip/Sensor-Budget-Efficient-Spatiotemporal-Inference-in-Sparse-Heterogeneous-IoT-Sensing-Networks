# Graph Neural Network–based Transfer Learning for PM2.5 Forecasting across Indian Cities under Heterogeneous Station Topologies

**A Consolidated Research Proposal**

| Field | Value |
|-------|-------|
| Primary author | Bishwadip Maitra |
| B.Tech co-authors (Stage I) | Chappati Teja Sanjeev, Bhaskarla Bhanu Prakash |
| B.Tech supervisor | Dr. Mainak Thakur, IIIT Sricity |
| Stage I (preliminary, completed) | LSTM Transfer Learning Framework — B.Tech thesis, submitted 05 Jan 2025 |
| Stage II (proposed) | Graph-DANN Transfer Learning Framework — this proposal |
| Target outcome | One peer-reviewed conference / journal publication unifying both stages |
| Proposal date | May 2026 |

---

## 1. Abstract

PM2.5 forecasting in Indian cities is constrained by data scarcity: Delhi has 39 monitoring stations under the CPCB network, Kolkata has 10, Guwahati has only 4. In our B.Tech thesis (Stage I, January 2025), we demonstrated that **station-agnostic LSTM transfer learning** strictly improves PM2.5 forecasting accuracy in data-scarce target cities over from-scratch single-city LSTM training, with a best target-city result of **R² = 0.8189 / MAE = 13.32 µg/m³** on Delhi → Kolkata at 30% target fine-tuning. That pipeline, however, treats every station as an independent time-series and discards the spatial dependencies between stations that govern PM2.5 advection. This proposal (Stage II) extends the established TL framework to **Graph Neural Networks**, addressing the central new challenge that LSTM-TL avoids by construction: the three cities induce graphs of different sizes (|V| = 4, 10, 39), different topologies, and disjoint node identities, so the graph structure itself changes when the target city changes. We propose a **topology-invariant inductive ST-GNN with adversarial graph-level domain adaptation (Graph-DANN)** that decouples model parameters from |V| via inductive aggregators and size-invariant graph readout, then aligns city feature distributions via a gradient-reversed city classifier. The Stage II protocol is identical to Stage I (same CPCB splits, same fine-tune fractions d ∈ {15, 30, 45, 60}%, same metrics), so the resulting numbers are one-to-one comparable to thesis Table 5.2. We hypothesize that Graph-DANN strictly improves both R² and MAE in every (source, target) cell, with statistical significance at p < 0.05 under the Diebold–Mariano test.

---

## 2. Introduction

India was the third most polluted country worldwide in 2023, with an average PM2.5 of 54.4 µg/m³ — well above WHO interim targets. PM2.5 is a regulated criteria pollutant linked to cardiovascular and respiratory morbidity, with documented impacts on the Indo-Gangetic plain and the northeastern states. Continuous monitoring is the prerequisite for any forecasting or mitigation system; however, station density in India is highly uneven. The Central Pollution Control Board (CPCB) operates 39 stations in Delhi, 10 in Kolkata, and only 4 in Guwahati — a 10× gap between the most and least-instrumented cities in our scope. Cities with sparse monitoring are exactly the cities where forecasting is most needed and least feasible from local data alone.

Transfer learning offers a principled remedy. In Stage I of this work (B.Tech thesis, Jan 2025) we demonstrated that pre-training an LSTM on a data-rich source city and fine-tuning on a small fraction d% of the target city's data strictly outperforms from-scratch single-city training. The best result, R² = 0.8189 / MAE = 13.32 µg/m³ on Delhi → Kolkata at d = 30%, established that TL is a viable strategy for Indian PM2.5 forecasting. The mechanism, however, is fundamentally limited: LSTMs see each station as an independent univariate sequence with no awareness of spatial coupling, which is exactly the regime PM2.5 lives in (wind-borne advection, regional emission corridors, urban-scale dispersion).

Spatio-temporal Graph Neural Networks (ST-GNNs) explicitly model the spatial graph. Recent ST-GNN work on air quality (PM2.5-GNN [W1], GAGNN [W2], HighAir [W3], AirFormer [W4]) reports 10–30% improvements over RNN/LSTM baselines on the same metrics we use. But all of those works train and test on a *single fixed graph* (a single national network in China). They do not transfer across cities with different station counts, and the published cross-city transfer learning literature on spatio-temporal data (RegionTrans [C2], CrossTReS [C7], TransGTR [C5], DASTNet [C6]) targets *traffic* on Chinese road networks, not air quality, and not on Indian data. No prior work combines (a) ST-GNN, (b) cross-city transfer learning, (c) heterogeneous |V|, and (d) Indian PM2.5 monitoring.

The central new challenge in Stage II is what we call the **intercity graph reconstruction problem**: when the source-pretrained GNN is applied to the target city, the graph G_T = (V_T, E_T) does not match G_S = (V_S, E_S) in any of |V|, |E|, node identity, or geographic extent. Vanilla (transductive) GNNs cannot transfer because their parameters are tied to a particular adjacency matrix. The proposal's methodological core is to *construct* a GNN whose parameters are demonstrably decoupled from |V|, and to align its representations across cities via adversarial domain adaptation.

**The story of this paper is therefore a clean two-stage progression by the same author on the same dataset under the same protocol:** Stage I establishes that TL is useful for PM2.5 in Indian cities (LSTM baseline). Stage II shows that incorporating the spatial graph and aligning the cross-city distribution makes TL substantially better.

---

## 3. Related Work

We organize the literature by the *mechanism family* it contributes to solving the intercity graph reconstruction problem. This taxonomy is original to this proposal and is what reviewers should engage with first; everything that follows in §6 (Stage II methodology) composes pieces from these families.

### 3.1 Mechanism family 1 — Inductive GNN aggregators (the prerequisite)

A GNN can transfer across graphs only if its parameters do not depend on the source graph's nodes. *Transductive* GCN variants (Kipf & Welling) store per-node embeddings and so cannot transfer. *Inductive* GNNs parameterize the *aggregator function* over a neighborhood and so apply to any graph:

- **GraphSAGE** [T4] (Hamilton et al., NeurIPS 2017) — neighborhood sampling + aggregator MLP.
- **GAT** [T5] (Veličković et al., ICLR 2018) — attention-weighted neighborhood aggregation.
- **GIN** [T13] (Xu et al., ICLR 2019) — Weisfeiler–Lehman-equivalent expressive power, inductive.

For air-quality forecasting with heterogeneous |V|, this family is *necessary but not sufficient*: choosing GAT/GraphSAGE removes the parameter-shape obstacle but does not by itself align cross-city distributions or yield the right adjacency for the target.

### 3.2 Mechanism family 2 — Adaptive / learned graph structure

When the target city's adjacency is unknown or unreliable, the GNN must learn it. Three sub-approaches:

- **Fully-learned adjacency** — Graph WaveNet [B3] learns `A = softmax(ReLU(E_1 E_2ᵀ))` with per-node embeddings E. *Non-transferable* in pure form.
- **Feature-induced adjacency** — MTGNN [B4] uses feature-conditioned graph learning. The decisive cross-city extension is to compute `E(x_v) = MLP(node-features_v)` so the embedding generator is *shared* across cities. **TransGTR** [C5] (Jin et al., KDD 2023) is the cleanest realization of this idea for cross-city traffic forecasting and is the direct prior art for our Stage II graph construction.
- **Domain-knowledge prior adjacency** — PM2.5-GNN [W1] hard-codes wind-direction edges. Always transferable (rule, not parameters) but rigid.

### 3.3 Mechanism family 3 — Cross-city spatio-temporal transfer

This is the family our Stage II Variant B belongs to:

- **RegionTrans** [C2] — region similarity matching before transfer.
- **MetaST** [T8] — meta-learn an initialization over many cities for fast target adaptation.
- **CrossTReS** [C7] — re-weight source regions by predicted target usefulness.
- **ST-GFSL** [C3] — generate city-specific parameters from city-level meta-knowledge; few-shot.
- **DASTNet** [C6] (Tang et al., CIKM 2022) — *Domain Adversarial Spatial-Temporal Network*. DANN-style adversarial city classifier over an ST-GNN. **The closest methodological prior art to our Variant B.** Our novelty is to transplant this recipe to PM2.5 under |V|-heterogeneity and compose it with the TransGTR-style inductive adjacency.

### 3.4 Mechanism family 4 — Theoretical transferability guarantees

These results explain *why* any GNN can transfer at all:

- **Graphon NNs** [T11] (Ruiz, Chamon & Ribeiro, NeurIPS 2020) — GNN outputs converge across graphs sampled from the same graphon. Justifies pre-training on a large source (Delhi) and transferring to a smaller target (Guwahati) when both can be modeled as samples from a similar urban-air-quality graphon.
- **Spectral GNN transferability** [T12] (Levie et al., JMLR 2021) — bounds transfer error by spectral perturbation.
- **EGI** [T2] (Zhu et al., NeurIPS 2021) — transferability depends on local ego-graph distributions, not global isomorphism. Most directly applicable: cities differ globally but their local wind-dispersion neighborhoods are similar.

### 3.5 Mechanism family 5 — Graph normalization for size invariance

- **GraphNorm** [G1] (Cai et al., ICML 2021) — per-graph normalization, explicit fix for the size-induced activation drift.
- **PairNorm** [G2] (Zhao & Akoglu, ICLR 2020) — keeps total pairwise distance constant across layers; mitigates oversmoothing, which is worse for small target graphs.

### 3.6 Air quality forecasting baselines

- **PM2.5-GNN** [W1] (Wang et al., SIGSPATIAL 2020) — domain-knowledge enhanced GNN with wind-aware message passing.
- **GAGNN** [W2] (Chen et al., 2021) — group-aware nationwide forecasting.
- **HighAir** [W3] (Shao et al., 2021) — hierarchical city-level + station-level graphs.
- **AirFormer** [W4] (Liang et al., AAAI 2023) — current SOTA on nationwide Chinese air-quality forecasting; we cite as the contemporary benchmark.
- **Bedi et al.** [Bedi24] (Urban Climate, 2024) — LSTM for PM2.5 during critical episodes in Delhi (cited in our B.Tech thesis).

### 3.7 Stage I (B.Tech thesis) literature

The thesis surveyed three closely related TL works: deep TL for crop yield prediction with GPP data (Khan et al., 2024 [Thesis-1]), TL for ozone forecasting in the Alpine region (Sangiorgio & Guariso, EMS 2024 [Thesis-2]), and the TrEnOS-ELMK ensemble TL framework (Ye & Dai, 2018 [Thesis-3]). Stage II subsumes these citations and adds the GNN-TL literature catalogued above.

### 3.8 Research gap

To our knowledge, **no prior work** combines: (i) ST-GNN architecture, (ii) cross-city transfer learning, (iii) heterogeneous station counts (|V| ranging 4–39), (iv) Indian PM2.5 monitoring. This combination is what Stage II contributes.

---

## 4. Problem Formulation

Let S denote a *source* city with station set V_S, observation tensor X_S ∈ ℝ^{|V_S| × T × F} (T time steps, F features per station per step), and a derived graph G_S = (V_S, E_S). Let T denote a *target* city with V_T, X_T, G_T, and a fine-tuning sub-sample X_T^{(d%)} of size d% of the target training partition. Crucially, |V_S| ≠ |V_T|, the station identities are disjoint, and the geographic extents differ.

**Stage I goal (established).** Learn a forecasting function f_θ^{LSTM} such that, after pre-training on X_S and fine-tuning on X_T^{(d%)}, it achieves lower MAE on the target test set than a from-scratch model trained only on X_T^{(d%)}. *Achieved* in B.Tech thesis with best result R² = 0.8189 / MAE = 13.32 µg/m³ (Delhi → Kolkata, d = 30%).

**Stage II goal (proposed).** Learn a forecasting function f_θ^{GNN-TL} such that, under the *identical* protocol (same X_S, same X_T^{(d%)}, same test partition, same metrics, same d ∈ {15, 30, 45, 60}%), it strictly outperforms f_θ^{LSTM} on every (source, target) cell of thesis Table 5.2.

**Sub-problems (consequences of cross-graph transfer):**

| ID | Sub-problem | Why it matters |
|----|-------------|----------------|
| P1 | Graph construction invariant to \|V\| | Without it, GNN weights are tied to one topology |
| P2 | Inductive node embeddings (no per-station parameters) | Required for unseen stations in target city |
| P3 | Feature-distribution mismatch between cities | Delhi winter dynamics ≠ Guwahati biomass dynamics |
| P4 | Few-shot adaptation when target ≤ 4 stations × few months | Standard fine-tune may overfit |
| P5 | Temporal misalignment (Delhi 2021–22; KOL/GUW 2022–23) | Confounds seasonal transfer |

---

## 5. Stage I — LSTM Transfer Learning (preliminary work, completed)

*This section reproduces and consolidates the key findings of the B.Tech thesis (Maitra, Sanjeev & Prakash, IIIT Sricity, 05 Jan 2025). It is included here so the proposal stands alone without the thesis PDF. Numbers are quoted verbatim from thesis Tables 5.1 and 5.2.*

### 5.1 Data

Three Indian cities, CPCB monitoring stations, observations at a 3-hour interval:
- Delhi: 39 stations, January 2021 – December 2022.
- Kolkata: 10 stations, January 2022 – December 2023.
- Guwahati: 4 stations, January 2022 – December 2023.
- Per-station features: PM2.5 (target), Relative Humidity (RH), Air Temperature (AT), Wind Speed (WS), Wind Direction (WD).

### 5.2 Preprocessing

1. **Missing-value imputation:** Inverse Distance Weighting (IDW) for spatial interpolation across nearby stations and Kalman filtering for temporal smoothing.
2. **Feature engineering:** sin/cos decomposition of wind direction; one-hot encoding of hour-of-day and month-of-year.
3. **Normalization:** StandardScaler (z-score) per feature.

### 5.3 Model

A multi-layer LSTM with dropout, fully-connected output head, MSE loss, Adam optimizer. Sequence-to-sequence: 24 h history → next 3 h PM2.5 forecast per station.

### 5.4 Transfer learning protocol

- Pre-train the LSTM on the *entirety* of the source city's training partition (M_K for Kolkata-source, M_G for Guwahati-source, M_D for Delhi-source).
- Fine-tune on d% ∈ {15, 30, 45, 60} of the target city's training partition with a reduced learning rate.
- Evaluate on the target city's held-out test partition.
- Repeat for all six (source, target) ordered pairs.

### 5.5 Stage I results

**Source-only models (Table 5.1 of thesis, R² / MAE in µg/m³):**

| Model | R² | MAE |
|-------|-----|-----|
| M_K (Kolkata) | 0.7861 | 14.6633 |
| M_G (Guwahati) | 0.5723 | 16.3012 |
| M_D (Delhi) | 0.6570 | 37.3301 |

**Transfer learning matrix (Table 5.2 of thesis, R² scores at each d%; best MAE in last column):**

| Source | Target | 15% | 30% | 45% | 60% | Best MAE |
|--------|--------|-----|-----|-----|-----|----------|
| Kolkata | Guwahati | 0.598 | 0.603 | 0.5823 | **0.6271** | 15.1105 |
| Kolkata | Delhi | **0.6939** | 0.6908 | 0.6881 | 0.687 | 35.7143 |
| Guwahati | Kolkata | 0.8098 | **0.8174** | 0.8164 | 0.8019 | 13.5344 |
| Guwahati | Delhi | **0.6995** | 0.6877 | 0.6815 | 0.6837 | 35.91 |
| **Delhi** | **Kolkata** | 0.7437 | **0.8189** | 0.8129 | 0.8051 | **13.3222** |
| Delhi | Guwahati | 0.5797 | **0.6381** | 0.621 | 0.6133 | 14.5957 |

The **headline cell** is Delhi → Kolkata at d = 30%: R² = 0.8189, MAE = 13.3222 µg/m³. This is the number Stage II must beat.

### 5.6 Stage I findings

1. Transfer learning *strictly improves* over single-city training for every target city (compare every cell of Table 5.2 with its corresponding Table 5.1 row).
2. Fine-tune fractions in {30%, 45%} are typically optimal; 60% does not always help (likely a sign of catastrophic forgetting at higher d% — relevant to Stage II's layer-freezing design).
3. Delhi is the hardest target regardless of source — its winter pollution dynamics are not well captured by any source.
4. The pipeline ignores inter-station spatial coupling — *this is exactly the gap Stage II closes*.

---

## 6. Stage II — Graph-DANN Transfer Learning (proposed extension)

### 6.1 Pipeline overview

```
   ┌─────────── Source Cities (Delhi, Kolkata) ───────────┐
   │ Station Coords + Meteorology + PM2.5 (2021–2023)     │
   └─────────────────────┬────────────────────────────────┘
                         │  Stage 0: reuse thesis preprocessing
                         │  (IDW + Kalman + cyclic time encoding)
                         ▼
            ┌──────────────────────────┐
   Stage 1: │ Topology-Invariant       │ ◄── k-NN, wind-aware, hybrid+learnable
            │ Graph Construction       │
            └─────────────┬────────────┘
                          ▼
            ┌──────────────────────────┐
   Stage 2: │ Inductive ST-GNN         │ ◄── GAT/GraphSAGE + TCN
            │ backbone f_θ             │     (no per-node weights)
            └─────────────┬────────────┘
                          ▼
            ┌──────────────────────────┐
   Stage 3: │ Graph-DANN (headline)    │ ◄── GRL + city discriminator
            │ + Variant A / C ablation │
            └─────────────┬────────────┘
                          ▼
            ┌──────────────────────────┐
   Stage 4: │ Target city d% fine-tune │
            │     + evaluate           │
            └──────────────────────────┘
```

### 6.2 Topology-invariant graph construction (solves P1, P2)

Three constructions to ablate:

1. **k-NN distance graph** — each station connected to its k nearest by Haversine distance; edge weight `w_ij = exp(−d_ij² / σ²)`. Choose k so average degree is comparable across cities (e.g., k = min(3, |V|−1) so even Guwahati's 4-station graph has consistent average degree).
2. **Wind-advection directed graph** — `w_ij = max(0, cos(θ_wind − θ_ij)) · exp(−d_ij / λ)`; PM2.5-GNN [W1]'s domain knowledge.
3. **Hybrid + learnable adjacency residual (primary)** — start from the wind graph and add a learnable, low-rank residual following TransGTR [C5]: `A_final = A_wind + softmax(E_1 E_2ᵀ)` where E_1, E_2 ∈ ℝ^{|V| × d} are produced by an **MLP over node features** (lat, lon, altitude, urban-density proxy). This makes them inductive and not per-station parameters — the key trick that makes the learnable adjacency transferable to a city of different |V|.

### 6.3 Inductive ST-GNN backbone (solves P2)

`Input → (Temporal Conv) → (GAT layer ×L) → (Temporal Conv) → Linear → Output`.

- GAT [T5] or GraphSAGE [T4] for spatial aggregation. Both parameterize *functions of neighborhoods*, not per-node embeddings.
- No node-identity embeddings; all node features are physical/meteorological.
- Temporal block: dilated TCN (Graph WaveNet style); preferred over LSTM for parallelism.
- Output head: per-node next-h-step PM2.5 (h = 3 primary, 24 secondary).

Every learnable parameter is either (i) a shared aggregator function or (ii) a feature-projection MLP. None depend on |V|. The same θ runs on a 4-node graph or a 39-node graph.

### 6.4 Variant B — Graph-DANN (headline contribution)

Adversarial domain adaptation on top of the ST-GNN encoder, following Ganin & Lempitsky [F1] adapted to graphs per UDA-GCN [T3] and DASTNet [C6]:

```
                                              ┌── h_y → ŷ_PM2.5  (regression loss L_y)
GNN encoder f_θ → graph-pool z ∈ ℝ^d  ────────┤
                                              └── GRL → h_d → {Delhi, Kolkata, Guwahati}
                                                                 (classification loss L_d)
```

- **Graph-level pool** — mean + max readout over node embeddings (size-invariant; |V| does not affect d).
- **Gradient Reversal Layer (GRL)** — encoder gradient is `−λ · ∂L_d/∂θ`, so f_θ is pushed to make z indiscriminable across cities while h_y still predicts well.
- **Discriminator h_d** — 3-class MLP over {Delhi, Kolkata, Guwahati}. Uses unlabeled target data freely.
- **Adversarial weight schedule** — `λ(p) = 2/(1 + exp(−γp)) − 1` with p = training progress fraction (Ganin's recipe).
- **Joint loss** — `L = L_y(ŷ, y) − λ · L_d(d̂, d_city)` (sign flip via GRL).

**Layer-wise fine-tune in Variant B.** During target d% fine-tune: freeze the first GAT layer for the first 20% of fine-tune steps (preserve transferred local-aggregation); unfreeze progressively. Discriminator h_d stays active throughout with target-domain examples now labeled "target city".

### 6.5 Variants A and C (ablations only)

- *Variant A* — pre-train + fine-tune of f_θ, no discriminator. Closest analog to thesis Table 5.2. Isolates the *architecture* gain from the *adversarial-alignment* gain.
- *Variant C* — Reptile meta-learning [F3] over the two source cities. Single ablation row to demonstrate that with K = 2 source cities the meta gain is marginal — preempts a reviewer question.

### 6.6 Heterogeneous-graph handling techniques layered into the pipeline

- **GraphNorm** [G1] before each GNN layer — reduces sensitivity to |V|.
- **Subgraph sampling during source training** (GraphSAGE-style) — even on Delhi's 39-node graph, train on sampled subgraphs of size ≈ |V_target| to simulate target conditions.
- **Cross-city feature standardization** — z-score per feature using source statistics, refit on target during fine-tune.
- **Temporal alignment** — re-index to month-of-year + hour-of-day; subtract per-city long-run seasonal mean before the model; add back at prediction time.

---

## 7. Experimental Design

### 7.1 Data and splits

Reuse the thesis CPCB dataset (Delhi 2021–22, Kolkata/Guwahati 2022–23) and its IDW + Kalman cleaned outputs *unchanged*. Same train/val/test temporal split, same 3 h primary horizon (24 h secondary).

### 7.2 TL protocol specification (verbatim with thesis Table 5.2)

This protocol is *pinned* to the thesis so the result tables are one-to-one comparable.

1. **Source-only pre-training.** Train f_θ on the entirety of source S's train + val partition; use S's validation set for early stopping; save f_θ^{(S)}.
2. **Target fine-tune set.** Sub-sample d% ∈ {15, 30, 45, 60} of target T's training partition, chronologically stratified.
3. **Fine-tuning.** Load f_θ^{(S)}; run a fixed number of gradient steps (3 000) on the d% target sample at 0.1× the pre-train LR; early-stop on T's validation set. For Variant B, the discriminator h_d sees both source mini-batches (replayed) and target mini-batches throughout.
4. **Evaluation.** Target hold-out test partition only. R² and MAE primary; RMSE and MAPE secondary.
5. **Zero-shot row (d = 0%).** Skip step 3; evaluate f_θ^{(S)} directly.
6. **Six ordered (source, target) pairs**, matching the layout of thesis Table 5.2.

**Consistency guards.** IDW refit uses *only* the train partition to prevent leakage into val/test. Single random seed surfaced in the config. "Epoch" is normalized to gradient steps (3 000) so d% does not change the optimization budget. The discriminator never sees target-test or target-val examples.

### 7.3 Baselines

1. LSTM-TL (thesis Table 5.2) — copy numbers verbatim.
2. City-specific LSTM (thesis Table 5.1).
3. City-specific ST-GNN — isolates architecture from transfer.
4. Zero-shot ST-GNN transfer — d = 0%.
5. Variant A — ST-GNN + pre-train + fine-tune.
6. **Variant B (headline)** — ST-GNN + Graph-DANN.
7. Variant C — ST-GNN + Reptile meta-learn.
8. External SOTA: PM2.5-GNN [W1] re-trained Delhi-only and transferred zero-shot.

### 7.4 Metrics

R², MAE primary; RMSE, MAPE secondary. Per-horizon (3 h and 24 h) and per-station error breakdown.

### 7.5 Ablations

1. Graph construction: k-NN vs wind-aware vs hybrid+learnable.
2. GNN backbone: GAT vs GraphSAGE vs DCRNN vs MTGNN.
3. Transfer strategy: A vs B vs C.
4. Fine-tune fraction d% ∈ {0, 15, 30, 45, 60}.
5. Layer-freezing schedule.
6. With / without GraphNorm.

### 7.6 Statistical rigor

5 seeds per configuration; mean ± std. Diebold–Mariano test [E1] on 24 h-ahead forecast residuals of Variant B vs LSTM-TL must reject the null at p < 0.05.

---

## 8. Expected Results and Hypotheses

### 8.1 The results table reviewers will look at first

The Stage II results table is a one-to-one replacement of thesis Table 5.2. For each (source, target, d%) cell we will report a *triple*: (Stage I LSTM-TL value from thesis, Variant A ST-GNN value, Variant B Graph-DANN value).

Template (numbers to be filled by Phase 4 of the timeline):

| Source | Target | d% | R² LSTM-TL | R² Variant A | R² Variant B | MAE LSTM-TL | MAE Var. A | MAE Var. B |
|--------|--------|----|-----------:|-------------:|-------------:|------------:|-----------:|-----------:|
| Delhi | Kolkata | 15 | 0.7437 | TBD | TBD | TBD | TBD | TBD |
| Delhi | Kolkata | 30 | **0.8189** | TBD | TBD | **13.3222** | TBD | TBD |
| Delhi | Kolkata | 45 | 0.8129 | TBD | TBD | TBD | TBD | TBD |
| Delhi | Kolkata | 60 | 0.8051 | TBD | TBD | TBD | TBD | TBD |
| Delhi | Guwahati | 15 | 0.5797 | TBD | TBD | TBD | TBD | TBD |
| Delhi | Guwahati | 30 | 0.6381 | TBD | TBD | **14.5957** | TBD | TBD |
| Delhi | Guwahati | 45 | 0.621 | TBD | TBD | TBD | TBD | TBD |
| Delhi | Guwahati | 60 | 0.6133 | TBD | TBD | TBD | TBD | TBD |
| ... | ... | ... | ... | ... | ... | ... | ... | ... |

(Full table will have 24 rows: 6 source-target pairs × 4 d% values.)

### 8.2 Hypotheses

**H1 (architecture gain).** Variant A R² > thesis LSTM-TL R² for every cell (architecture alone beats LSTM, regardless of TL strategy).

**H2 (alignment gain).** Variant B R² > Variant A R² for every cell, with the largest gain on Guwahati-as-target (the smallest |V|, where cross-city distribution alignment matters most).

**H3 (headline cell).** Variant B at Delhi → Kolkata 30% achieves R² ≥ 0.85 and MAE ≤ 12.5 µg/m³ — a clear, quotable improvement over thesis 0.8189 / 13.32.

**H4 (statistical significance).** Variant B − LSTM-TL residual differences reject the DM null at p < 0.05 on the 24 h horizon for at least 4 of 6 (source, target) pairs.

**H5 (ablation parsimony).** The hybrid + learnable adjacency (§6.2 strategy 3) beats both k-NN and wind-only, *and* GraphNorm is necessary for the gain to be stable across seeds. If H5 fails — i.e., k-NN alone is competitive — we report this honestly as a "simpler graph suffices" finding.

### 8.3 Failure cases we anticipate and what they mean

- If H2 fails (Variant B does not beat Variant A): the adversarial alignment is not the right knob; the contribution becomes *the inductive ST-GNN architecture itself* and Variant A becomes the headline.
- If H1 fails (ST-GNN does not beat LSTM-TL): would indicate that the |V|=4 target is too small for any graph model to help; we would discuss this honestly and the paper becomes a negative result + lessons.
- If H4 fails (no statistical significance): we report effect sizes and confidence intervals, not p-values; this is a common outcome of small Indian-city datasets and reviewers will accept it if the practical-significance argument is made.

---

## 9. Risks and Mitigations

| Risk | Likelihood | Mitigation |
|------|------------|------------|
| Guwahati \|V\|=4 too small for any GNN | Medium | Grid-cell virtual nodes via IDW for a denser pseudo-graph |
| Negative transfer (TL hurts target) | Medium | Per-epoch target validation; early stop on negative transfer |
| Compute budget for full ablation grid | Medium | Smallest backbone first (GAT-2L + 2-layer TCN); scale only winning configs |
| Learnable adjacency degenerates at \|V\|=4 | Low–Medium | Fall back to wind-aware static graph for very small target |
| Temporal misalignment confounds attribution | High | AdaRNN [C4] for temporal DA, or restrict to overlapping months |
| DANN training instability (saddle / collapse) | Medium | λ-warmup schedule [F1]; gradient clipping; discriminator capacity smaller than encoder |

---

## 10. Timeline (6 months, publication-targeted)

| Phase | Weeks | Deliverable | Gate |
|-------|-------|-------------|------|
| 0. Setup | 1 | Repo skeleton; port thesis preprocessing; PyG/DGL environment | Re-reproduce thesis Table 5.1 |
| 1. Single-city ST-GNN | 2–3 | One ST-GNN per city matches/beats thesis Table 5.1 | M_K R² ≥ 0.7861 |
| 2. Topology invariance | 4 | Same θ runs forward on \|V\|=4 and \|V\|=39 | Forward-pass test passes |
| 3. Variant A | 5–7 | Direct comparison row vs thesis Table 5.2 | Variant A ≥ thesis on ≥ 1 cell |
| 4. **Variant B (headline)** | 8–11 | DANN training stable, λ tuned; full d% sweep | Variant B > Variant A on Guwahati |
| 5. Ablations | 12–15 | All §7.5 rows | All ablation rows filled |
| 6. Statistics + figures | 16–18 | 5-seed runs, DM tests, figures, tables | DM p < 0.05 on headline cell |
| 7. Writing + open-source | 19–24 | Manuscript, ArXiv, GitHub release | Submission |

**Target venues, in order of fit:** *Environmental Modelling & Software*, *Urban Climate*, *Atmospheric Environment*, KDD ADS / IJCAI applied / AAAI AI for Social Impact, *Knowledge-Based Systems*.

---

## 11. References

### Must-read first (in order)

**Pillar 1 — How adversarial domain adaptation works**
- [F1] Ganin, Y., & Lempitsky, V. (2015). *Unsupervised Domain Adaptation by Backpropagation.* ICML.
- [T3] Wu, M., Pan, S., Zhou, C., Chang, X., & Zhu, X. (2020). *Unsupervised Domain Adaptive Graph Convolutional Networks.* WWW.

**Pillar 2 — GNNs in air quality**
- [W1] Wang, S., Li, Y., Zhang, J., Meng, Q., Meng, L., & Gao, F. (2020). *PM2.5-GNN: A Domain Knowledge Enhanced Graph Neural Network for PM2.5 Forecasting.* SIGSPATIAL.
- [W2] Chen, L., Xu, J., Wu, B., Qian, Y., Du, Z., Li, Y., & Zhang, Y. (2021). *Group-Aware Graph Neural Network for Nationwide City Air Quality Forecasting.* (GAGNN). arXiv:2108.12238.
- [W4] Liang, Y., Xia, Y., Ke, S., Wang, Y., Wen, Q., Zhang, J., Zheng, Y., & Zimmermann, R. (2023). *AirFormer: Predicting Nationwide Air Quality in China with Transformers.* AAAI.

**Pillar 3 — Why cross-graph transfer works**
- [T2] Zhu, Q., Yang, C., Xu, Y., Wang, H., Zhang, C., & Han, J. (2021). *Transfer Learning of Graph Neural Networks with Ego-graph Information Maximization.* NeurIPS.
- [C7] Jin, Y., Chen, K., & Yang, Q. (2022). *Selective Cross-City Transfer Learning for Traffic Prediction via Source City Region Re-Weighting.* KDD (CrossTReS).

**Pillar 4 — Direct methodological prior art for Stage II**
- [C5] Jin, Y., Chen, K., & Yang, Q. (2023). *Transferable Graph Structure Learning for Graph-Based Traffic Forecasting Across Cities.* KDD (TransGTR).
- [C6] Tang, Y., Qu, A., Chow, A. H. F., Lam, W. H. K., Wong, S. C., & Ma, W. (2022). *Domain Adversarial Spatial-Temporal Network: A Transferable Framework for Short-term Traffic Forecasting across Cities.* CIKM (DASTNet).

### Spatio-Temporal GNN backbones
- [B1] Yu, B., Yin, H., & Zhu, Z. (2018). *Spatio-Temporal Graph Convolutional Networks.* IJCAI.
- [B2] Li, Y., Yu, R., Shahabi, C., & Liu, Y. (2018). *Diffusion Convolutional Recurrent Neural Network: Data-Driven Traffic Forecasting.* ICLR (DCRNN).
- [B3] Wu, Z., Pan, S., Long, G., Jiang, J., & Zhang, C. (2019). *Graph WaveNet for Deep Spatial-Temporal Graph Modeling.* IJCAI.
- [B4] Wu, Z., Pan, S., Long, G., Jiang, J., Chang, X., & Zhang, C. (2020). *Connecting the Dots: Multivariate Time Series Forecasting with Graph Neural Networks.* KDD (MTGNN).
- [B5] Bai, L., Yao, L., Li, C., Wang, X., & Wang, C. (2020). *Adaptive Graph Convolutional Recurrent Network for Traffic Forecasting.* NeurIPS (AGCRN).
- [B6] Guo, S., Lin, Y., Feng, N., Song, C., & Wan, H. (2019). *Attention Based Spatial-Temporal Graph Convolutional Networks for Traffic Flow Forecasting.* AAAI (ASTGCN).
- [B7] Zheng, C., Fan, X., Wang, C., & Qi, J. (2020). *GMAN: A Graph Multi-Attention Network for Traffic Prediction.* AAAI.

### Air-quality forecasting
- [W3] Shao, X., et al. (2021). *HighAir: A Hierarchical Graph Neural Network-Based Air Quality Forecasting Method.* arXiv:2101.04264.
- [W5] Han, J., Liu, H., Zhu, H., Xiong, H., & Dou, D. (2021). *Joint Air Quality and Weather Prediction Based on Multi-Adversarial Spatiotemporal Networks.* AAAI.
- [W6] Zheng, Y., Yi, X., Li, M., Li, R., Shan, Z., Chang, E., & Li, T. (2015). *Forecasting Fine-Grained Air Quality Based on Big Data.* KDD.
- [Bedi24] Bedi, S., Katiyar, A., Krishnan, N. A., & Kota, S. H. (2024). *Utilizing LSTM models to predict PM2.5 levels during critical episodes in Delhi.* Urban Climate, 53, 101835.

### Inductive GNNs and pre-training
- [T1] Hu, W., Liu, B., Gomes, J., Zitnik, M., Liang, P., Pande, V., & Leskovec, J. (2020). *Strategies for Pre-training Graph Neural Networks.* ICLR.
- [T4] Hamilton, W. L., Ying, R., & Leskovec, J. (2017). *Inductive Representation Learning on Large Graphs.* NeurIPS (GraphSAGE).
- [T5] Veličković, P., Cucurull, G., Casanova, A., Romero, A., Liò, P., & Bengio, Y. (2018). *Graph Attention Networks.* ICLR (GAT).
- [T13] Xu, K., Hu, W., Leskovec, J., & Jegelka, S. (2019). *How Powerful are Graph Neural Networks?* ICLR (GIN).

### Theoretical transferability
- [T11] Ruiz, L., Chamon, L. F. O., & Ribeiro, A. (2020). *Graphon Neural Networks and the Transferability of Graph Neural Networks.* NeurIPS.
- [T12] Levie, R., Huang, W., Bucci, L., Bronstein, M., & Kutyniok, G. (2021). *Transferability of Spectral Graph Convolutional Neural Networks.* JMLR, 22(272).

### Cross-city / region transfer
- [C1] Wei, Y., Zheng, Y., & Yang, Q. (2016). *Transfer Knowledge between Cities.* KDD.
- [C2] Wang, L., Geng, X., Ma, X., Liu, F., & Yang, Q. (2018). *Cross-City Transfer Learning for Deep Spatio-Temporal Prediction.* (RegionTrans).
- [C3] Lu, B., Gan, X., Zhang, W., Yao, H., Fu, L., & Wang, X. (2022). *Spatio-Temporal Graph Few-Shot Learning with Cross-City Knowledge Transfer.* KDD (ST-GFSL).
- [C4] Du, Y., Wang, J., Feng, W., Pan, S., Qin, T., Xu, R., & Wang, C. (2021). *AdaRNN: Adaptive Learning and Forecasting for Time Series.* CIKM.
- [T6] Wang, L., Geng, X., Ma, X., Liu, F., & Yang, Q. (2019). *Cross-City Transfer Learning for Deep Spatio-Temporal Prediction.* IJCAI.
- [T8] Yao, H., Liu, Y., Wei, Y., Tang, X., & Li, Z. (2019). *Learning from Multiple Cities: A Meta-Learning Approach for Spatial-Temporal Prediction.* WWW (MetaST).

### Graph normalization and training stability
- [G1] Cai, T., Luo, S., Xu, K., He, D., Liu, T.-Y., & Wang, L. (2021). *GraphNorm: A Principled Approach to Accelerating Graph Neural Network Training.* ICML.
- [G2] Zhao, L., & Akoglu, L. (2020). *PairNorm: Tackling Oversmoothing in GNNs.* ICLR.

### Domain-adaptation foundations
- [F2] Finn, C., Abbeel, P., & Levine, S. (2017). *Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks.* ICML (MAML).
- [F3] Nichol, A., Achiam, J., & Schulman, J. (2018). *On First-Order Meta-Learning Algorithms.* arXiv:1803.02999 (Reptile).

### Surveys
- [S1] Wu, Z., Pan, S., Chen, F., Long, G., Zhang, C., & Yu, P. S. (2020). *A Comprehensive Survey on Graph Neural Networks.* IEEE TNNLS.
- [S2] Jin, M., Koh, H. Y., Wen, Q., Zambon, D., Alippi, C., Webb, G. I., King, I., & Pan, S. (2023). *A Survey on Graph Neural Networks for Time Series.* arXiv:2307.03759.
- [S3] Sahili, Z. A., & Awad, M. (2023). *Spatio-Temporal Graph Neural Networks: A Survey.* arXiv:2301.10569.
- [T10] Shi, B., et al. (2024). *A Survey on Graph Domain Adaptation.* arXiv:2402.00904.

### Statistical testing
- [E1] Diebold, F. X., & Mariano, R. S. (1995). *Comparing Predictive Accuracy.* Journal of Business & Economic Statistics.

### Stage I (B.Tech thesis literature, retained)
- [Thesis-1] Khan, S. N., Li, D., & Maimaitijiang, M. (2024). *Using gross primary production data and deep transfer learning for crop yield prediction in the US Corn Belt.* International Journal of Applied Earth Observation and Geoinformation, 131, 103965.
- [Thesis-2] Sangiorgio, M., & Guariso, G. (2024). *Transfer learning in environmental data-driven models: A study of ozone forecast in the Alpine region.* Environmental Modelling & Software, 177, 106048.
- [Thesis-3] Ye, R., & Dai, Q. (2018). *A novel transfer learning framework for time series forecasting.* Knowledge-Based Systems, 156, 74–99.

---

## Appendix A — B.Tech Thesis Numerical Baselines (preserved verbatim)

**Source document:** "Transfer Learning Framework for PM2.5 Forecasting in Indian Cities" — Chappati Teja Sanjeev, Bhaskarla Bhanu Prakash, Bishwadip Maitra; IIIT Sricity, 05 Jan 2025.

**Data:** Delhi (39 stations, Jan 2021 – Dec 2022), Kolkata (10 stations, Jan 2022 – Dec 2023), Guwahati (4 stations, Jan 2022 – Dec 2023). 3-hour cadence. Features: PM2.5, RH, AT, WS, WD.

**Preprocessing:** IDW spatial interpolation + Kalman temporal smoothing + sin/cos wind direction + one-hot hour-of-day and month-of-year + StandardScaler.

**Architecture:** Multi-layer LSTM with dropout, MSE loss, Adam, seq2seq (24 h history → next 3 h forecast).

**Table A.1 — Source-only LSTM (Thesis Table 5.1).**

| Model | R² | MAE (µg/m³) |
|-------|------|-------------|
| M_K (Kolkata) | 0.7861 | 14.6633 |
| M_G (Guwahati) | 0.5723 | 16.3012 |
| M_D (Delhi) | 0.6570 | 37.3301 |

**Table A.2 — LSTM-TL fine-tuned (Thesis Table 5.2).**

| Source | Target | 15% | 30% | 45% | 60% | MAE (best) |
|--------|--------|-----|-----|-----|-----|-----------|
| Kolkata | Kolkata | 0.7861 | — | — | — | 14.6633 |
| Kolkata | Guwahati | 0.598 | 0.603 | 0.5823 | 0.6271 | 15.1105 |
| Kolkata | Delhi | 0.6939 | 0.6908 | 0.6881 | 0.687 | 35.7143 |
| Guwahati | Guwahati | 0.5723 | — | — | — | 16.3012 |
| Guwahati | Kolkata | 0.8098 | 0.8174 | 0.8164 | 0.8019 | 13.5344 |
| Guwahati | Delhi | 0.6995 | 0.6877 | 0.6815 | 0.6837 | 35.91 |
| Delhi | Delhi | 0.657 | — | — | — | 37.3301 |
| **Delhi** | **Kolkata** | 0.7437 | **0.8189** | 0.8129 | 0.8051 | **13.3222** |
| Delhi | Guwahati | 0.5797 | 0.6381 | 0.621 | 0.6133 | 14.5957 |

**The single Stage I result Stage II must beat:** Delhi → Kolkata at d = 30%, **R² = 0.8189 / MAE = 13.32 µg/m³**.
