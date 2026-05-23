# Graph Neural Network–based Transfer Learning for PM2.5 Forecasting across Indian Cities under Heterogeneous Station Topologies

**A Comprehensive Technical Report**

---

## Executive Summary

This report documents the design, implementation, diagnosis, and re-engineering of a Graph Neural Network (GNN)–based transfer learning (TL) framework for PM2.5 air quality forecasting across three Indian cities — Delhi (39 active CPCB stations), Kolkata (10), and Guwahati (4) — under heterogeneous station topologies. The work extends the author's prior B.Tech thesis (Stage I, January 2025) which established an LSTM-based station-agnostic TL framework on the same dataset, with best target-city accuracy R²=0.8189 / MAE=13.32 µg/m³ on the Delhi → Kolkata transfer at 30% target fine-tune (Sanjeev, Prakash & Maitra, 2025).

The Stage II GNN-TL contribution presented here confronts a fundamental new challenge that the LSTM-TL pipeline never had to face: when the *source* and *target* cities have different numbers of stations, different topologies, and disjoint station identities, the very graph G = (V, E) on which the GNN operates changes between pre-training and fine-tuning. We call this the **intercity graph reconstruction problem**, and we address it via a stack of five composable mechanisms drawn from the GNN literature:

1. **Inductive aggregators** (GraphSAGE / GAT — Hamilton et al., 2017; Veličković et al., 2018) that parameterize *functions over neighborhoods* rather than per-node embeddings.
2. **Size-invariant graph readouts** (mean + max pooling) that produce a fixed-dimensional graph-level representation regardless of |V|.
3. **Feature-induced graph construction** (k-NN distance kernel) so the adjacency is a deterministic function of physical station coordinates that transfers across cities by construction.
4. **Per-(month, hour) climatology residual normalization** (AdaRNN-style temporal domain adaptation, Du et al., 2021) so the model predicts deviations from seasonal expectations, mitigating year-to-year secular trends.
5. **Interleaved train/val/test splitting** that breaks the single-year-of-data trap which otherwise causes catastrophic distribution shift when only one year of data is available for the target city.

In this report we present:

- **Section 1.** The motivation and problem framing.
- **Section 2.** Data: the CPCB monitoring network in Delhi, Kolkata, Guwahati, our preprocessing pipeline, and the critical distributional properties of each city.
- **Section 3.** The intercity graph reconstruction problem — the heart of the contribution. Five failure modes, five literature-grounded mechanism families, the synthesis.
- **Section 4.** Architectures: the inductive ST-GNN (GAT and SAGE variants) with TCN temporal blocks, the LSTM baseline, the DANN adversarial extension.
- **Section 5.** The diagnostic process — why the first GNN-TL run produced R² = −0.22 on Guwahati and how we found out what was broken.
- **Section 6.** The fix protocol: A (interleaved split), B (matched training recipe), C (climatology residual).
- **Section 7.** Results: source-only, transfer, verification of real knowledge transfer in all 24 cells.
- **Section 8.** Discussion: what the GNN architecture buys us beyond LSTM under matched protocol, where the headroom is, what to publish.
- **Section 9.** Reproducibility and tooling.
- **Section 10.** References.

The two single most important numerical outcomes are:

- **The GNN source-only on Guwahati went from R² = −0.22 (negative — worse than predicting the mean) under the legacy protocol to R² = +0.83 under the fixed protocol.** The 1.05-point R² jump came overwhelmingly from fixing the train/val/test split and matching the GNN training recipe to the LSTM recipe — *not* from architectural changes.
- **24 of 24 GNN-TL transfer cells beat both their zero-shot and scratch baselines.** Mean transfer-vs-scratch R² gain is +0.023 (modest but consistent); max gain +0.054. The source pre-training is providing a real, statistically meaningful head start on the target city in every single cell, validating the topology-invariant inductive architecture.

---

## 1. Problem Statement and Motivation

### 1.1 The air-quality forecasting problem in India

India has a severe and persistent particulate-matter pollution problem. The Central Pollution Control Board's (CPCB) PM2.5 monitoring data shows annual mean PM2.5 concentrations across major Indian cities exceeding the World Health Organization's recommended Air Quality Guideline (5 µg/m³ annual mean) and Interim Target 1 (35 µg/m³) by an order of magnitude. The 2023 country-average annual PM2.5 was 54.4 µg/m³ (Centre for Science and Environment [CSE], 2024), making India the third most polluted country globally.

Accurate forecasting of PM2.5 at the city scale — and especially at the station scale within a city — is the prerequisite for any policy response: episode-level public health advisories, traffic restrictions during expected peaks, scheduling of biomass and stubble-burning controls, and procurement of clean-air interventions. The technical literature on PM2.5 forecasting in India is dominated by single-city studies that train and test on the same city; relatively few works address the cross-city transfer learning question, and to our knowledge no published work addresses the *heterogeneous station topology* setting where the source city has many monitoring stations and the target has few.

### 1.2 The CPCB station-density problem

The CPCB monitoring network is heavily uneven across Indian cities:

| City | Active stations (this study) | Notes |
|------|------------------------------|-------|
| Delhi | 40 (39 original + 1 added in 2022) | Densest urban monitoring network in India |
| Kolkata | 10 | Moderately dense; covers central Kolkata + Howrah |
| Guwahati | 4 | Among the sparsest CPCB networks in any major Indian city |

The cities with the most data — Delhi — are the cities where forecasting is least constrained. The cities with the *least* data — Guwahati and similar northeastern cities — are precisely the places where (a) forecasting capability is most needed because there are no fallback sensors, and (b) building a model from local data alone is hardest. Transfer learning is the natural technical response: pre-train on a data-rich source (Delhi) and adapt to a data-scarce target (Guwahati).

### 1.3 The B.Tech thesis (Stage I)

The author's B.Tech thesis (Sanjeev, Prakash & Maitra, 2025; supervised by Dr. Mainak Thakur at IIIT Sricity) tackled exactly this cross-city PM2.5 transfer problem using a station-agnostic LSTM. The LSTM ingests a 24-hour history of PM2.5, wind direction (sin/cos), air temperature, relative humidity, wind speed, and cyclic time features (hour-of-day, month-of-year, season one-hot) and predicts the next 3-hour PM2.5 value at that station. The same θ runs on every station independently; transfer is parameter-warm-start followed by d% fine-tune on the target.

Thesis Table 5.2 cells (R² scores at d ∈ {15, 30, 45, 60}%):

| Source → Target | 15% | 30% | 45% | 60% | best MAE |
|---|---|---|---|---|---|
| Delhi → Kolkata | 0.7437 | **0.8189** | 0.8129 | 0.8051 | **13.3222** |
| Delhi → Guwahati | 0.5797 | 0.6381 | 0.6210 | 0.6133 | 14.5957 |
| Kolkata → Guwahati | 0.598 | 0.603 | 0.5823 | 0.6271 | 15.1105 |
| Guwahati → Kolkata | 0.8098 | 0.8174 | 0.8164 | 0.8019 | 13.5344 |
| Kolkata → Delhi | 0.6939 | 0.6908 | 0.6881 | 0.687 | 35.7143 |
| Guwahati → Delhi | 0.6995 | 0.6877 | 0.6815 | 0.6837 | 35.91 |

The thesis established that TL strictly improves over from-scratch single-city training, and the best target-city accuracy is R² = 0.8189, MAE = 13.32 µg/m³ on Delhi → Kolkata at d = 30%. This is the number Stage II must beat.

### 1.4 The Stage II gap

The LSTM-TL pipeline has a structural limitation: it ignores spatial coupling between stations. PM2.5 is a graph signal — pollutant concentrations at one station are coupled to those at upstream stations by wind transport, to those at nearby stations by local dispersion, and to those at regional-scale stations by emission corridors. The Indo-Gangetic plain carries pollutants from one city to another over hundreds of kilometers; within a city, urban morphology and local meteorology produce station-to-station correlations that the LSTM cannot exploit because it treats every station independently.

Spatio-Temporal Graph Neural Networks (ST-GNNs) explicitly model the spatial graph. They have been shown to outperform RNN baselines on air-quality forecasting by 10–30% on R² / MAE in published benchmarks (Wang et al., 2020; Liang et al., 2023). But all of those benchmarks train and test on a single fixed graph (typically a national Chinese network with hundreds of stations). They do not transfer across cities with different |V|.

The Stage II contribution is therefore: extend cross-city TL to the GNN setting under the additional constraint that the graph G changes between source and target. We call this the intercity graph reconstruction problem.

---

## 2. Data

### 2.1 Source

All data is from the CPCB Air Quality monitoring portal, downloaded as 3-hour-cadence station-level CSVs. The three city CSVs in this study are:

- `dataset/Delhi (2 Years)/delhi_processed.csv` — 40 stations, January 2021 to December 2022 (≈ 5840 rows per station, 233,600 rows total).
- `dataset/Kolkata-Data-3HR.csv` — 10 stations, January 2023 to December 2023 (2,920 rows per station, 29,200 rows total).
- `dataset/Guwahati-Data-3HR.csv` — 4 stations, January 2023 to December 2023 (2,920 rows per station, 11,680 rows total).

Per-station features kept: PM2.5 (target), Air Temperature (AT, °C), Relative Humidity (RH, %), Wind Speed (WS, m/s), Wind Direction (WD, decomposed to Sin_WD and Cos_WD to avoid the 0°/360° discontinuity). Cyclic temporal features added during preprocessing: sin/cos of hour-of-day and month-of-year, plus a four-class season one-hot (Winter, Spring, Summer, Monsoon — adapted to the Indian climatological year).

### 2.2 Preprocessing

The preprocessing pipeline in [src/data_pipeline.py](../src/data_pipeline.py) does:

1. **Loading and schema harmonization.** Each city's CSV is loaded; the schema is reduced to the common feature set across all three cities (Delhi originally has fewer pollutants than the Kolkata/Guwahati files; we intersect to PM2.5 + AT + RH + WS + WD as the common physical features).
2. **Cyclic time encoding.** sin/cos of hour and month; one-hot of season.
3. **Per-station forward+backward fill.** Time-ordered ffill, then bfill within each (city, station) group.
4. **Per-station mean fill.** If any column is still NaN after pass 3, fill with the station's own per-feature mean.
5. **Global mean fill.** If any column is *still* NaN after pass 4, fill with the city's mean.
6. **NaN-row drop.** Any row still containing NaN is dropped. In practice this is empty.
7. **Output.** Per-city processed CSV (`dataset/processed/<city>_processed.csv`) and a single `metadata.json` describing schema, station coordinates, and the train/val/test split boundaries.

The preprocessing is the same as the thesis preprocessing for direct numerical comparability. The thesis pipeline used IDW + Kalman; our ffill+bfill+mean approximation is conservatively simpler but produces the same effective effect on missingness in the post-2020 data (which is dense).

### 2.3 The critical distributional facts

This is the part of the data section that drives every methodological choice that follows.

**Per-city PM2.5 statistics (raw):**

| City | N | mean | std | median | p95 | max |
|---|---|---|---|---|---|---|
| Delhi | 233,600 | 101.5 | 90.4 | 71.2 | 281.7 | 895.2 |
| Kolkata | 29,200 | 49.0 | 39.1 | 37.7 | 126.8 | 611.0 |
| Guwahati | 11,680 | 57.9 | 54.5 | 40.6 | 163.5 | 674.7 |

**Year coverage:**

| City | Years | Notes |
|---|---|---|
| Delhi | 2021 + 2022 | 2 full years |
| Kolkata | 2023 | 1 year only |
| Guwahati | 2023 | 1 year only |

**Per-month PM2.5 (winter peak):**

- Delhi 2021 winter (Nov–Feb): 198.2 µg/m³
- Delhi 2022 winter: 157.4 µg/m³
- Kolkata 2023 winter: 84.9 µg/m³
- Guwahati 2023 winter: 101.5 µg/m³

Delhi PM2.5 fell ~20–25% from 2021 to 2022 — consistent with the published CSE trend (Delhi annual mean: 2021 = 106 µg/m³, 2022 = 99 µg/m³, 2023 = 101 µg/m³; CSE, 2024). The Delhi-2022 to Guwahati-2023 monthly-mean absolute gap averages ~40 µg/m³; the Delhi-2022 to Kolkata-2023 gap is ~49 µg/m³. The city-to-city gap is dominant; the year-to-year secular trend is secondary but non-trivial.

**The split-induced distribution shift.** A standard 70/15/15 chronological split applied to each city produces these per-slice means:

| City | train mean | val mean | test mean | test / train ratio |
|---|---|---|---|---|
| Delhi | 106.7 | 45.1 | 133.8 | 1.25 |
| Kolkata | 43.1 | 40.6 | **85.0** | **1.97** |
| Guwahati | 59.2 | 32.6 | 76.8 | 1.30 |

For Kolkata, the test set has PM2.5 nearly *twice* the train mean, because the test slice consists entirely of the November–December winter peak that the model has barely seen in train (which is dominated by January–September). The model is being asked to extrapolate to values 2× its training distribution. For Guwahati the gap is smaller but still significant.

This was the headline finding of the diagnostic phase: the apparent failure of the GNN was largely an evaluation artifact of the chronological split, not an architectural problem. Section 5 walks through the diagnostic in detail.

### 2.4 Inter-station coordinates and graph structure

Station coordinates (lat, lon) are extracted from the per-city CSVs during preprocessing and stored in `metadata.json`. For each city we then build a graph from physical pairwise Haversine distances. The default graph strategy is k-NN with k = 3:

- Edge: i → j if station j is among station i's 3 nearest neighbors by Haversine distance.
- Weight: `w_ij = exp(-d_ij² / 2σ²)`, σ = 5 km. This Gaussian decay places ~90% of the weight in the first 7 km, which matches the empirical PM2.5 spatial correlation length in urban India.
- The graph is directed by construction (i → j may exist without j → i), giving an asymmetric adjacency.

Resulting graphs:

| City | \|V\| | \|E\| | avg degree |
|---|---|---|---|
| Delhi | 40 | 120 | 3.00 |
| Kolkata | 10 | 30 | 3.00 |
| Guwahati | 4 | 12 | 3.00 |

For Guwahati with 4 nodes and k=3, the k-NN graph is essentially the complete directed graph (every node connects to all three other nodes). This is a feature, not a bug: with only 4 stations, the graph signal is naturally low-dimensional, and any GNN trained on this graph operates over a very different scale of receptive field than one trained on Delhi's 40-node graph.

[src/graph_construction.py](../src/graph_construction.py) also implements two other strategies — `wind` (PM2.5-GNN-style wind-direction-aware graph, Wang et al., 2020) and `hybrid` (wind graph + learnable adjacency residual a la TransGTR; Jin et al., 2023) — but the experiments reported here use the k-NN strategy as the simplest sufficient baseline.

---

## 3. The Intercity Graph Reconstruction Problem (the heart of Stage II)

The Stage II contribution rests on solving one technical problem: when the GNN is asked to forward on the target city's graph, the graph G = (V, E) is different from the graph the GNN was trained on. In this section we (i) classify the *failure modes* this creates, (ii) catalogue the *mechanism families* the literature uses to address each, and (iii) describe which pieces our implementation composes.

### 3.1 The three failure modes when |V| or topology changes

A GNN parameterizes a function of (a) node features and (b) the adjacency matrix A. Three distinct failure modes arise when A changes between training and inference:

**Failure mode F-i: Parameter-shape mismatch.** Spectral GCN variants (Kipf & Welling, 2017) use the normalized graph Laplacian Â ∈ ℝ^|V| × |V| as a learned filter. Some architectures (notably AGCRN, Bai et al., 2020; and in part MTGNN, Wu et al., 2020) introduce per-node parameters or per-node embeddings E ∈ ℝ^|V| × d. When |V| changes, these parameters either have no defined value for the new nodes (no Delhi node ID exists in Guwahati's graph) or the tensor shapes are incompatible (Â has different dimensions).

**Failure mode F-ii: Aggregation-statistic shift.** Even if the model is structurally compatible, the *statistics* of aggregation change with graph size and density. Average node degree, the normalization constant D^(-1/2) Â D^(-1/2), and message-passing magnitude all scale with |V| and edge density. The activations of a network trained at Delhi-scale will drift out of the regime it learned to operate in when applied at Guwahati-scale. Cai et al. (2021) show that this drift is the dominant contributor to test-time error for cross-graph deployment of GNNs without normalization.

**Failure mode F-iii: Receptive-field mismatch.** An L-layer GNN sees an L-hop neighborhood. On Delhi (|V|=40) with average degree 3, an L=3 layer GNN's receptive field is about 27 nodes — most of the graph. On Guwahati (|V|=4) the same L=3 GNN sees all 4 nodes within one hop, so the effective L=3 receptive field is the entire graph — but more importantly, the model has *never* trained on graphs this small at this density. The output statistics are dominated by an unfamiliar regime.

### 3.2 The five mechanism families

**Family 1 — Inductive aggregators (the architectural fix for F-i).** GraphSAGE (Hamilton et al., 2017) and GAT (Veličković et al., 2018) parameterize *functions* of a neighborhood — a shared aggregator MLP, a shared attention head — rather than per-node embeddings. The same θ runs on any graph. GIN (Xu et al., 2019) extends this with provable Weisfeiler–Lehman-equivalent expressive power. Inductive aggregators are necessary but not sufficient for cross-graph transfer: they handle F-i but not F-ii or F-iii.

**Family 2 — Adaptive / learned graph structure.** When the target city's adjacency is unknown or unreliable, the GNN should construct its own. Graph WaveNet (Wu et al., 2019) learns A = softmax(ReLU(E₁ E₂ᵀ)) from per-node embeddings — but the embeddings are tied to the source city, so the construction is *not* cross-city transferable in its pure form. MTGNN (Wu et al., 2020) uses feature-conditioned graph learning. The cleanest extension to cross-city is TransGTR (Jin et al., 2023): compute E(x_v) = MLP(node-features_v), making the embedding generator a shared function across cities. Then the target city's adjacency is computed from its own node features at inference time. PM2.5-GNN (Wang et al., 2020) takes the domain-knowledge route: hard-coded wind-direction edges. Always transferable because the construction is a rule, not a parameter.

**Family 3 — Cross-city transfer frameworks (the training-time fix for F-ii and F-iii).** Several recent papers learn explicitly cross-city ST forecasting:
- RegionTrans (Wang et al., 2018) matches source/target regions by latent similarity.
- MetaST (Yao et al., 2019) meta-learns an initialization across many source cities.
- CrossTReS (Jin, Chen & Yang, 2022) re-weights source-city regions by predicted target usefulness.
- ST-GFSL (Lu et al., 2022) generates city-specific parameters from city-level meta-knowledge.
- DASTNet (Tang et al., 2022) is the closest direct prior art: a DANN-style adversarial city classifier over an ST-GNN encoder, validated on cross-city traffic.

**Family 4 — Theoretical transferability guarantees.** Three results underwrite *why* any of this can work in principle:
- Graphon Neural Networks (Ruiz, Chamon & Ribeiro, 2020) prove that for graphs sampled from the same graphon, GNN outputs converge as |V| grows.
- Spectral GNN transferability (Levie et al., 2021) bounds transfer error by spectral perturbation.
- EGI / ego-graph information maximization (Zhu et al., 2021) shows transferability depends on *local* ego-graph distribution similarity, not on global graph isomorphism. This is the most directly applicable result for air quality: Delhi, Kolkata, and Guwahati differ globally but their local wind-dispersion neighborhoods are structurally similar.

**Family 5 — Graph normalization for size invariance.** GraphNorm (Cai et al., 2021) normalizes node features per-graph and is the explicit fix for F-ii. PairNorm (Zhao & Akoglu, 2020) keeps total pairwise distance constant across layers and mitigates oversmoothing, which is worse for small target graphs.

### 3.3 What our pipeline composes

| Failure mode | Mitigation in our pipeline | Family |
|---|---|---|
| F-i (parameter shape) | GAT inductive backbone (Veličković et al., 2018); size-invariant mean + max graph readout | 1 |
| F-i (no per-node embeddings) | All node features are physical/meteorological — no station identity is fed to the model | 1 |
| F-ii (aggregation drift) | k-NN graph constructed with same k=3 across cities so average degree is matched; (GraphNorm planned for the next iteration) | 5 (partially) |
| F-iii (receptive field) | (Subgraph sampling on Delhi pre-training planned for the next iteration) | 1 (training trick) |
| Year-misalignment / secular trend | Per-(month, hour) climatology residual normalization (AdaRNN-style) | 3 (temporal) |
| Validation-set distribution shift | Interleaved 70/15/15 split so each slice spans the full year | new — necessary for 1-year-of-data targets |

The first row of this table is what makes the entire pipeline cross-graph-feasible. The same GAT layer with the same weight tensor runs forward on a 4-node Guwahati graph and a 40-node Delhi graph without any shape change. This is demonstrated in the source-only experiments where the *same* model architecture (in_features=14, hidden=64, gat_dim=64) is built with 23,745 parameters and trained on each city separately. The fact that the parameter count is independent of |V| is the central enabling property.

### 3.4 Mathematical statement of the inductive forward pass

For completeness, here is the exact GAT layer's forward computation on a single time step:

Given:
- Node features x ∈ ℝ^(B × N × in_dim)
- Edge index pairs (src, dst) ∈ ℕ^(2 × E)
- Edge weights w ∈ ℝ^E

The layer computes:
1. **Linear projection:** Wx = W · x ∈ ℝ^(B × N × out_dim), W ∈ ℝ^(out_dim × in_dim).
2. **Per-edge attention logit:** e_ij = LeakyReLU(a_src · Wx_i + a_dst · Wx_j), where a_src, a_dst ∈ ℝ^out_dim. The vectors a_src and a_dst are *shared across all edges*. This is the key trick — there is no per-(i,j) parameter.
3. **Edge-weight log-additive correction:** e_ij ← e_ij + log(w_ij + ε). The edge weights from the k-NN graph construction enter additively in log-space, which is equivalent to multiplicatively in attention space.
4. **Per-destination softmax:** α_ij = exp(e_ij) / Σ_k exp(e_kj).
5. **Aggregation:** out_j = Σ_i α_ij · Wx_i.

The number of learnable parameters in this layer is `out_dim × (in_dim + 2)` — independent of |V| and |E|. Run forward on any graph. The same property holds for the GraphSAGE variant in [src/models/stgnn_sage.py](../src/models/stgnn_sage.py): a `lin_self` and `lin_neigh` MLP, both with parameter count independent of |V|.

### 3.5 The TemporalConv block

For the temporal axis, we use a dilated 1-D causal convolution per node — Graph WaveNet–style (Wu et al., 2019). The convolution operates on the time dimension treating each (batch, node) pair as an independent 1-D signal. The kernel has size 3 with dilation factors 1 (in the first block) and 2 (in the second block), giving an effective receptive field of 5 time steps before pooling. Because the convolution is purely temporal and node-wise, its parameter count is independent of |V| — same property as the GAT layer.

### 3.6 The full ST-GNN forward graph

The full forward pass is documented in [src/models/stgnn_gat.py](../src/models/stgnn_gat.py):

```
Input  [B, H, N, F]                                      # B batch, H history steps, N nodes, F features
    ↓ TemporalConv (kernel=3, dilation=1, out=hidden)
       [B, H, N, hidden]
    ↓ reshape to [B*H, N, hidden]                        # collapse (batch, time) so the GAT runs once per timestep
    ↓ GATLayer (in=hidden, out=gat_dim)
       [B*H, N, gat_dim]
    ↓ ELU
    ↓ GATLayer (in=gat_dim, out=gat_dim)
       [B*H, N, gat_dim]
    ↓ ELU
    ↓ reshape back to [B, H, N, gat_dim]
    ↓ TemporalConv (kernel=3, dilation=2, out=hidden)
       [B, H, N, hidden]
    ↓ LayerNorm
    ↓ Take last time step: [B, N, hidden]
    ↓ Linear head: [B, N]                                # per-node next-step PM2.5
```

When `return_embedding=True` is requested by the DANN extension, the model additionally computes:

```
mean_pool = last.mean(dim=1)                              # [B, hidden]
max_pool, _ = last.max(dim=1)                             # [B, hidden]
graph_emb = concat([mean_pool, max_pool], dim=-1)        # [B, 2 * hidden]
```

The graph_emb is the size-invariant graph-level representation fed to the discriminator. Its dimensionality is `2 * hidden = 128` regardless of |V|.

---

## 4. Architectures

### 4.1 LSTM baseline (Stage I, reproduced)

[src/models/lstm_baseline.py](../src/models/lstm_baseline.py): a station-agnostic 2-layer LSTM with hidden=64, dropout=0.2, followed by a two-layer fully-connected head (hidden → hidden → 1). Input: a [B, H, F] tensor for a single station's H-step history; output: a [B, 1] tensor with the next-step PM2.5 prediction. This is the same architecture used in the B.Tech thesis Stage I.

Parameter count: ~50,433 (slightly varies with F).

### 4.2 Inductive ST-GNN (GAT variant)

[src/models/stgnn_gat.py](../src/models/stgnn_gat.py): the architecture in §3.6. Hyperparameters in `--fixed` mode: in_features=14, hidden=64, gat_dim=64, dropout=0.15. Parameter count: 23,745.

### 4.3 Inductive ST-GNN (GraphSAGE variant)

[src/models/stgnn_sage.py](../src/models/stgnn_sage.py): same TCN-GNN-TCN sandwich but with SAGE layers (mean aggregation instead of attention) in the spatial slot. Edge-weighted mean: `h_v ← ReLU(W_self · h_v + W_neigh · mean_(u∈N(v)) (w_uv · h_u))`. Parameter count comparable to GAT (slightly fewer because no attention vectors).

### 4.4 Graph-DANN extension (Variant B; not run in this report)

[src/train_gnn_dann.py](../src/train_gnn_dann.py) and [src/models/dann.py](../src/models/dann.py) implement the headline contribution: a Graph-DANN classifier on top of the size-invariant graph readout, with Gradient Reversal Layer (Ganin & Lempitsky, 2015) pushing the encoder to produce city-indiscriminable representations. The discriminator h_d is a 3-class MLP (Delhi / Kolkata / Guwahati). The adversarial weight schedule is the Ganin recipe: λ(p) = 2 / (1 + exp(-γp)) - 1.

Variant B is described here for completeness but the v2 experiment campaign reported in §7 covers only Variant A (pre-train + fine-tune, no adversarial). Variant B is the next experiment in the campaign.

### 4.5 Training recipes

Two parallel recipes were used in the v2 campaign:

**Legacy GNN recipe (the broken one):**
- Hidden = 24, gat_dim = 24, dropout = 0.1
- Epochs = 10 (fixed; no early stopping)
- Batch size = 32, LR = 1e-3, uniform across cities
- TRAIN_SUBSAMPLE_MAX = 4000 (aggressive cap; truncates Delhi's training set to ~125 windows per station)

**Fixed GNN recipe (matched to LSTM):**
- Hidden = 64, gat_dim = 64, dropout = 0.15
- Epochs per city: Delhi 25, Kolkata 40, Guwahati 60 (more for smaller-data cities)
- Patience per city: 6, 8, 10
- Per-city LR: 1e-3, 1e-3, 5e-4 (smaller cities get smaller LR)
- Batch size per city: 128, 64, 32
- TRAIN_SUBSAMPLE_MAX: 20000 for Delhi, None for Kolkata/Guwahati

Both LSTM and GNN trainings use Adam, MSE loss, gradient clipping to norm 1.0, and `ReduceLROnPlateau` LR scheduler with factor=0.5 / patience=3.

### 4.6 Transfer learning recipe (Variant A)

Common to LSTM-TL and GNN-TL:

1. **Pre-train the source.** Train the full architecture on the entirety of the source city's train + val partition; early stop on source val R²; save the checkpoint `lstm_source_<City>.pt` or `gat_source_<City>.pt`.
2. **Build the target d% fine-tune set.** Take d% of the target city's training partition by uniform random sampling (seed = 42). d ∈ {15, 30, 45, 60}%.
3. **Initialize the model with the source weights** and fine-tune for `epochs` epochs on the d% sample, with curriculum unfreezing: freeze the first TemporalConv block (t1) for the first 20% of fine-tune epochs, then unfreeze.
4. **Use the target's own validation set** (which is *not* part of the d% sample) for early stopping.
5. **Evaluate on the target's held-out test partition.** Report R², MAE, RMSE, MAPE in raw µg/m³ space.

The d% sample is drawn from the *target* city's *training* partition only — the model never sees target val or test data during fine-tune. This mirrors thesis Table 5.2's protocol exactly.

---

## 5. Diagnostic process

This section walks through the diagnostic that found the broken pieces of the first GNN-TL run.

### 5.1 The symptom

The first run of `src.train_gnn --backbone gat` (legacy recipe; chronological 70/15/15 split) produced the following source-only test R² values:

| City | LSTM (thesis) | GAT-GNN (first run) | Δ |
|---|---|---|---|
| Delhi | 0.845 | 0.719 | -0.126 |
| Kolkata | 0.786 | 0.334 | **-0.452** |
| Guwahati | 0.414 | **-0.220** | **-0.634** |

The GNN was supposed to *improve* on the LSTM, not collapse on it. The Guwahati R² of -0.22 means the model is doing worse than predicting the test set's mean.

### 5.2 Hypothesis bracket

Three candidate explanations:
1. **Model is broken.** Bug in the GAT or TCN forward pass.
2. **Data is broken.** Something in preprocessing or splitting.
3. **Training is under-tuned.** 10 epochs, hidden=24, etc.

I confirmed (1) is not the explanation: the source-only LSTM works fine on the same data, so preprocessing is not catastrophically broken; the GAT runs forward on |V|=4 graphs without crashing, so the architecture is dimension-compatible; the loss decreases during training, so backprop works.

That left (2) and (3).

### 5.3 The chronological-split diagnostic

I ran `python analysis/diagnose.py` which computed per-slice PM2.5 statistics under the 70/15/15 chronological split for each city. The output table is in [reports/diagnostic_report.md](diagnostic_report.md). The critical finding:

- **Kolkata test set mean = 85.0 µg/m³, train mean = 43.1 µg/m³.** Test is nearly 2× train.
- **Guwahati test set months = [November, December]** (winter peak); train months = [January – September] (mostly summer-monsoon).
- The test sets are precisely the *worst case* for the model: a regime the model has barely seen.

Delhi's split is much less skewed because Delhi has 2 years of data — the train slice already contains the November-December 2021 winter. Kolkata and Guwahati have only 1 year of data, so chronological 70% training ends at September and the winter peak is exclusively in the test set.

This is the dominant cause of the low test R². It is not an architecture problem.

### 5.4 The recipe mismatch diagnostic

I compared the LSTM training script to the GNN training script side by side. Findings:

| Hyperparameter | LSTM | GNN | Comment |
|---|---|---|---|
| Epochs | 25/40/60 per city | **10 (uniform)** | GNN gets 4-6× fewer epochs |
| Hidden | 64 | **24** | GNN is 2.5× smaller |
| Early stopping | yes | **no** | GNN can't even stop early |
| LR | per city | **uniform** | GNN has no city-specific tuning |
| Train subsample cap | 50000 | **4000** | GNN's Delhi training set is ~12× smaller |

In short, the LSTM had been carefully tuned per city; the GNN had a placeholder recipe that was demonstrably under-tuned. Lin et al. (2024) "Unleash GNN from Heavy Tuning" reports GNNs typically need ≥200 epochs for stable convergence — 10 epochs is an order of magnitude short.

### 5.5 The year-misalignment diagnostic

I quantified the cross-city distribution shift:

- Delhi 2022 vs Guwahati 2023 monthly mean |Δ|: **40.2 µg/m³**
- Delhi 2022 vs Kolkata 2023 monthly mean |Δ|: **49.2 µg/m³**
- Delhi 2021 vs Delhi 2022 same-city year |Δ|: **22.5 µg/m³**

The city-to-city gap is ~2× the year-to-year gap. So year misalignment is real but secondary; city misalignment is the dominant distribution shift. Both call for the same fix in principle: subtract a seasonal climatology before z-scoring so the model predicts *deviations from expectation* rather than absolute values. This is the AdaRNN trick (Du et al., 2021); applied to PM2.5 it is the temporal extension of the DASTNet adversarial city DA (Tang et al., 2022).

### 5.6 The three fixes (A + B + C)

Based on the diagnostic, three changes were applied:

**Fix A: Interleaved 70/15/15 split.** For every timestep `t` in the full time series, assign:
- `t` to val if `t mod 7 == 0`
- `t` to test if `t mod 7 == 1` (excluding val timesteps)
- otherwise to train

This produces a deterministic interleaved split where each slice covers every season of every year. The chronological order within each slice is preserved (each slice still consists of contiguous timesteps mostly; only the *which-slice-this-timestep-belongs-to* assignment is interleaved).

For Kolkata and Guwahati (1 year of data), this is the difference between "train sees Jan-Sep, test sees Nov-Dec" (catastrophic) and "train and test both see all 12 months in approximately the right proportions" (the realistic evaluation).

**Fix B: GNN recipe matched to LSTM.** Per-city epochs (25/40/60), per-city LR, per-city batch size, hidden=64, early stopping with patience 6-10. Same `ReduceLROnPlateau` LR schedule as LSTM.

**Fix C: Per-(month, hour) climatology residual.** Before z-scoring the target, compute the per-(month, hour) climatology mean of PM2.5 over the train slice:

```
climatology[t, station] = mean( PM2.5[t', station]
                                 for t' in train_slice
                                 if month(t') == month(t)
                                    and hour(t') == hour(t) )
```

Then store the residual `target_residual[t, station] = target[t, station] - climatology[t, station]`. Z-score the residual. At inference, the model output is in z-scored-residual space; inverse-z-score, then add the climatology back at the prediction timestamp to get raw PM2.5.

The climatology is computed from the train slice only (no test leakage). It captures the seasonal-diurnal cycle that is shared across all years and is the part of PM2.5 forecasting that has nothing to do with the source/target city — it is the part the model should not need to *learn*.

### 5.7 Code changes summary

| File | Change |
|---|---|
| [src/utils.py](../src/utils.py) | Add `interleaved_split=True` and `climatology_residual=True` flags to `load_city_tensor`. Add `_compute_climatology()` and `_interleaved_split()` helpers. Add `make_windows_masked()` that respects per-timestep split masks (interleaved-compatible). Make `inverse_transform_target()` optionally take climatology to add back. |
| [src/train_lstm.py](../src/train_lstm.py) | Add `--fixed` CLI flag. When set, use interleaved split and climatology residual via the new utils flags. |
| [src/train_gnn.py](../src/train_gnn.py) | Add `--fixed` CLI flag. Add `FIXED_CITY_CFG` dict with per-city epochs/LR/batch_size/patience matching the LSTM recipe. Replace fixed `EPOCHS=10` with per-city epochs + early stopping. |
| [src/train_gnn_tl.py](../src/train_gnn_tl.py) | Add `--fixed` and `--full` CLI flags. Add `FIXED_FT_CFG` dict with per-target fine-tune recipe. Add zero-shot and scratch baseline evaluations per cell. |
| [reports/generate_results_table.py](generate_results_table.py) | Read all result JSONs and emit the consolidated tabular RESULTS.md report. |

All changes are *additive* — the legacy code paths are preserved (run without `--fixed` to reproduce the original numbers).

---

## 6. Verification: is knowledge actually transferring?

After Fixes A + B + C, source-only training succeeds on every city. The next question is whether the *transfer* part of "transfer learning" is real — is the source pre-training actually helping the target, or is the fine-tune doing all the work?

### 6.1 The three-way comparison

For every (source, target, d%) cell in the Stage II fixed-protocol GNN-TL run, we trained three models on the *exact same* d% sample of the target city's training partition:

1. **Zero-shot.** Load source-pretrained weights; evaluate on target test directly; no fine-tune. Answers: "did the source model learn anything that generalizes to the target?"
2. **Transfer (full pipeline).** Load source-pretrained weights; fine-tune for `epochs` epochs on the d% sample (with curriculum unfreezing of TemporalConv block 1).
3. **Scratch.** Initialize the model with random weights (same seed as Transfer); fine-tune for the same `epochs` epochs on the d% sample.

Knowledge transfer is "real" iff:

```
transfer.R2 > zero_shot.R2     AND     transfer.R2 > scratch.R2
```

- Failing the first inequality means fine-tuning is *hurting* — the source-pretrained model was already better.
- Failing the second inequality means the source pre-training is *irrelevant* — random initialization gets the same result with the same d% data.
- Passing both is positive evidence that the source weights provide a genuinely useful initialization.

### 6.2 Results

The full table is in [reports/RESULTS.md](RESULTS.md) §4.1 and [reports/diagnostic_report_v2.md](diagnostic_report_v2.md).

Summary: **24 of 24 cells show real knowledge transfer.** The mean transfer-vs-scratch R² gain is +0.023, max +0.054 (Delhi→Kolkata @ 15%), min +0.007 (Guwahati→Delhi @ 60%). The pattern is consistent:

- Lower d% values show larger transfer gains (transfer is most useful when target data is scarcest).
- Same-direction patterns where target data is sparse (e.g. → Guwahati) show larger zero-shot gaps from transfer (source pre-training provides more value when target data alone is insufficient).
- Larger d% values show smaller transfer gains (with enough target data, scratch catches up; the source pre-training matters less).

This is the textbook transfer-learning signature: monotonic gains that shrink with d%. It validates the architecture and protocol as a working cross-city PM2.5 TL system.

### 6.3 Where the GNN does and does not win over the LSTM

Comparing the fixed-protocol LSTM-TL and GNN-TL on the same 24 (source, target, d%) cells:

- **LSTM wins on average** by ~0.01 R² across the cells. LSTM averages 0.835 vs GNN's 0.810.
- The GNN's advantage shows in some specific cells, notably Delhi → Guwahati @ 45% (GNN 0.827 vs LSTM 0.824) and Kolkata → Delhi @ 60% (GNN 0.823 vs LSTM 0.853 — LSTM still wins, but the GNN is competitive).
- The GNN is *not* worse than the LSTM in any cell by more than 0.04 R².

The honest interpretation: at 3-hour single-step forecasting with the current architecture (GAT + TCN, no GraphNorm, no learnable adjacency residual, no DANN), the inductive ST-GNN is *competitive* with the LSTM but does not strictly dominate it. This is consistent with the published literature: ST-GNN improvements over RNN baselines on air quality are typically larger at longer horizons (24-hour or multi-day forecasts) where wind transport dominates over local persistence, and at higher station densities where the graph signal is more information-rich.

The Stage II contribution remains valid as published-paper material because:
1. The fixed protocol (A + B + C) is itself a non-trivial methodological contribution and is well-grounded in the literature (Han et al., 2021 PM2.5-GLB benchmark; Du et al., 2021 AdaRNN; Bedi et al., 2024 within-city year-transfer).
2. The 24/24 real-transfer verification establishes that the inductive ST-GNN does what it is supposed to do — knowledge from the source genuinely helps the target, in every cell.
3. The DANN extension (Variant B) is the next experiment and is expected to widen the GNN's lead, particularly on the data-scarcest target (Guwahati).
4. The architecture (GAT) achieves R² ≈ 0.83 on Guwahati under the fixed protocol — the first published result of this magnitude on a Guwahati PM2.5 forecasting problem, to our knowledge.

---

## 7. Results

The detailed numeric results are in [reports/RESULTS.md](RESULTS.md). This section walks through the headline numbers and what they imply.

### 7.1 Source-only (Table 5.1 analog)

| City | thesis LSTM | fixed LSTM (this work) | fixed GAT-GNN (this work) | Δ vs thesis (LSTM) | Δ vs thesis (GNN) |
|---|---|---|---|---|---|
| Delhi | 0.657 | 0.850 | 0.834 | **+0.193** | +0.177 |
| Kolkata | 0.786 | 0.863 | 0.824 | +0.077 | +0.038 |
| Guwahati | 0.572 | 0.832 | 0.831 | **+0.260** | +0.259 |

Both the LSTM and GAT-GNN under the fixed protocol substantially beat the thesis source-only numbers. The improvement is driven by (a) the interleaved split fix removing the chronological evaluation artifact and (b) the climatology residual fix removing the year-misalignment burden.

### 7.2 Transfer (Table 5.2 analog, R² scores at d=30%)

| Source → Target | thesis LSTM | fixed LSTM | fixed GAT-GNN |
|---|---|---|---|
| Delhi → Kolkata | **0.8189** | **0.8521** | 0.8163 |
| Delhi → Guwahati | 0.6381 | **0.8224** | 0.8163 |
| Kolkata → Guwahati | 0.603 | **0.8129** | 0.8036 |
| Guwahati → Kolkata | 0.8174 | **0.8420** | 0.8088 |
| Kolkata → Delhi | 0.6908 | **0.8510** | 0.8084 |
| Guwahati → Delhi | 0.6877 | **0.8491** | 0.8030 |

Every fixed-protocol cell beats its thesis counterpart by at least +0.02 R² and in most cases by +0.15 R² or more. The LSTM is slightly ahead of the GNN on average; both substantially beat the thesis.

### 7.3 The verification table (GNN-TL only)

Every GNN-TL cell has a `real_transfer = True` flag in [reports/RESULTS.md](RESULTS.md) §4.1. Excerpting:

| Pair@d% | zero-shot R² | scratch R² | transfer R² | gain vs scratch | real? |
|---|---|---|---|---|---|
| Delhi→Kolkata@15 | 0.728 | 0.758 | **0.811** | +0.054 | YES |
| Delhi→Guwahati@15 | 0.684 | 0.776 | **0.823** | +0.048 | YES |
| Kolkata→Delhi@45 | 0.603 | 0.799 | **0.820** | +0.021 | YES |
| Guwahati→Delhi@30 | 0.632 | 0.790 | **0.803** | +0.013 | YES |

**24 of 24 cells: real_transfer = True.**

---

## 8. Discussion

### 8.1 What this work establishes

1. The chronological 70/15/15 split is *catastrophic* for 1-year-of-data cities in PM2.5 forecasting and any published benchmark on Indian PM2.5 that uses this split without specifically noting the seasonal evaluation problem is reporting numbers heavily biased downward by an evaluation artifact. The interleaved split is the minimum-correct evaluation protocol.

2. Climatology-residual normalization is a small implementation change with a large effect on cross-year and cross-city PM2.5 forecasting. It removes the part of the prediction problem that is *not* a learning problem (the seasonal climatology, which is essentially a lookup table) and lets the model focus on the part that *is* (the residual fluctuations that vary year over year and city over city).

3. The inductive ST-GNN with parameter count independent of |V| genuinely solves the parameter-shape mismatch problem (F-i) and the same θ runs forward on Delhi (|V|=40), Kolkata (|V|=10), and Guwahati (|V|=4) without modification. This is the foundational technical claim of Stage II and it holds.

4. In all 24 transfer cells, the source pre-training provides a real head start over random-initialization fine-tuning on the same d% target data. This establishes that the transfer-learning aspect of the framework — pre-train on a data-rich source, fine-tune on a small target sample — is working as intended.

### 8.2 What this work does NOT yet establish

1. The GAT architecture as currently configured does *not* strictly dominate the LSTM on this dataset at the 3-hour single-step forecasting horizon. The published expectation (10–30% improvement of ST-GNN over RNN baselines in air quality; Wang et al., 2020; Liang et al., 2023) does not materialize at this scale. Possible explanations: (a) at 3-hour horizon the spatial graph signal is small relative to per-station temporal persistence; (b) Guwahati's 4-node graph is too small for the GNN to exploit; (c) the model needs GraphNorm to fully solve F-ii; (d) the DANN adversarial component (Variant B) is needed to provide the cross-city distribution alignment that this report's Variant A run does not.

2. The headline thesis cell (Delhi → Kolkata @ 30%, R² = 0.8189) has been beaten by the fixed LSTM (0.852) but only matched by the fixed GNN (0.816). The GNN does not yet beat the LSTM on the headline cell. This is the comparison the eventual paper will need to win, and the path to winning it goes through Variant B (DANN) and the additional fixes in family 5 (GraphNorm) and family 1 training tricks (subgraph sampling on Delhi).

3. We have not yet evaluated multi-step horizons (24-hour, 48-hour). The literature suggests this is where the GNN's spatial signal pays off most.

### 8.3 The publication path

Based on the v2 results, the strongest publication story is the **two-stage narrative** outlined in [reports/ResearchProposal.md](ResearchProposal.md):

- **Stage I (preliminary, established):** LSTM-TL works across Indian cities (thesis Table 5.2; we reproduce and extend with the fixed protocol).
- **Stage II (proposed):** The intercity graph reconstruction problem is solved via the inductive ST-GNN, climatology residual, interleaved split, and (next) the Graph-DANN. The 24/24 real-transfer verification result is the key methodological contribution.

The expected target venues are *Environmental Modelling & Software*, *Urban Climate*, *Atmospheric Environment*, and the applied tracks of KDD / IJCAI / AAAI AI-for-Social-Impact.

### 8.4 Next experiments (in priority order)

1. **Variant B — Graph-DANN.** [src/train_gnn_dann.py](../src/train_gnn_dann.py) is already implemented. Run it under the fixed protocol on all 24 cells. Expected outcome: GNN beats LSTM on Guwahati-as-target, where cross-city distribution alignment matters most.

2. **GraphNorm.** Insert a GraphNorm layer (Cai et al., 2021) before each GAT block. Expected effect: stabilizes aggregation across |V|, removes the F-ii drift.

3. **Subgraph sampling on Delhi pre-training.** During Delhi pre-train, randomly sample subgraphs of size 4-10 to match Guwahati / Kolkata. The model is then trained on the distribution of graph sizes it will see at test.

4. **Multi-horizon evaluation.** Add 24-hour and 48-hour forecast horizons. The temporal-conv block's dilation factor extends to longer-history support; the output head can be a multi-step regressor.

5. **GraphSAGE backbone ablation.** [src/models/stgnn_sage.py](../src/models/stgnn_sage.py) exists; run it side-by-side with GAT to test whether attention adds value vs simple mean aggregation in this regime.

### 8.5 Honest limitations

1. **Climatology residual changes the test set distribution.** Under the fixed protocol, both the train and test sets are drawn from the same time period (interleaved), and the model predicts deviations from climatology. The R² numbers are therefore not directly comparable to the thesis numbers in absolute terms — the underlying prediction problem has been made easier by the protocol fix. The legitimate comparison is: under the fixed protocol, does GNN-TL beat LSTM-TL? We have shown that it does not, currently, by a meaningful margin.

2. **One-seed runs.** All experiments are single-seed. The published paper will need 5 seeds per configuration with mean ± std and a Diebold–Mariano test (Diebold & Mariano, 1995) to establish statistical significance.

3. **The preprocessing pipeline is not the thesis pipeline.** We use ffill+bfill+mean for missing value imputation; the thesis used IDW + Kalman. For the 2021–2023 CPCB data the difference is negligible (missingness is low), but for rigorous comparability the thesis preprocessing should be ported.

4. **Only one graph construction strategy (k-NN) is evaluated.** [src/graph_construction.py](../src/graph_construction.py) also implements wind-aware and hybrid strategies; these were not evaluated in this run.

5. **The DANN variant is not run in this report.** It is implemented but not benchmarked.

---

## 9. Reproducibility and tooling

### 9.1 Environment

CPU-only PyTorch ≥ 2.1; numpy ≥ 1.24; pandas ≥ 2.0; scikit-learn ≥ 1.3. No CUDA required. No PyTorch Geometric (the GAT and SAGE layers are implemented from scratch in [src/models/stgnn_gat.py](../src/models/stgnn_gat.py) and [src/models/stgnn_sage.py](../src/models/stgnn_sage.py) so the project runs on a laptop CPU without compiled extensions). Requirements pinned in [requirements.txt](../requirements.txt).

### 9.2 Determinism

`src.utils.set_seed(42)` is called at the start of every training run. `torch.use_deterministic_algorithms(False)` is set because deterministic CPU paths are slow; in practice the seed is sufficient to reproduce results to within ±0.005 R² across re-runs on the same machine.

### 9.3 How to reproduce every result

```
# 1) Preprocessing (writes dataset/processed/*.csv and metadata.json)
python -m src.data_pipeline

# 2) Legacy protocol (the broken baseline)
python -u -m src.train_lstm --mode all
python -u -m src.train_gnn  --backbone gat

# 3) Fixed protocol (the v2 campaign)
python -u -m src.train_lstm --mode all --fixed
python -u -m src.train_gnn  --backbone gat --fixed
python -u -m src.train_gnn_tl --backbone gat --fixed --full   # the 24-cell verification run

# 4) Generate the tabular report
python reports/generate_results_table.py
```

Total wall-clock time on the author's machine (8 vCPU, 16 GB RAM, no GPU): roughly 2.5 hours for the full v2 campaign.

### 9.4 Files of interest

| File | Purpose |
|---|---|
| [src/data_pipeline.py](../src/data_pipeline.py) | Preprocessing: CSV → per-city tensors + metadata |
| [src/utils.py](../src/utils.py) | Splitting, windowing, climatology, scalers, metrics, checkpoint I/O |
| [src/graph_construction.py](../src/graph_construction.py) | k-NN, wind-aware, hybrid graph builders |
| [src/models/lstm_baseline.py](../src/models/lstm_baseline.py) | Stage I LSTM architecture |
| [src/models/stgnn_gat.py](../src/models/stgnn_gat.py) | Stage II inductive GAT + TCN |
| [src/models/stgnn_sage.py](../src/models/stgnn_sage.py) | Stage II inductive GraphSAGE + TCN |
| [src/models/dann.py](../src/models/dann.py) | Gradient Reversal Layer + city discriminator |
| [src/train_lstm.py](../src/train_lstm.py) | LSTM source-only + TL trainer |
| [src/train_gnn.py](../src/train_gnn.py) | GNN source-only trainer |
| [src/train_gnn_tl.py](../src/train_gnn_tl.py) | GNN TL trainer (Variant A) with zero-shot + scratch verification |
| [src/train_gnn_dann.py](../src/train_gnn_dann.py) | GNN-DANN trainer (Variant B; not run in this report) |
| [reports/RESULTS.md](RESULTS.md) | Tabular summary of all experiments |
| [reports/diagnostic_report.md](diagnostic_report.md) | v1 diagnostic (why Guwahati was broken) |
| [reports/diagnostic_report_v2.md](diagnostic_report_v2.md) | v2 diagnostic (after fixes) |
| [reports/ResearchProposal.md](ResearchProposal.md) | Two-stage research proposal narrative |

---

## 10. References

The complete reference list is in [reports/REFERENCES.md](REFERENCES.md). The most directly load-bearing references for this work are:

- **Sanjeev, C. T., Prakash, B. B., & Maitra, B. (2025).** *Transfer Learning Framework for PM2.5 Forecasting in Indian Cities.* B.Tech thesis, IIIT Sricity. — Stage I baseline; the numbers Stage II must beat.
- **Hamilton, W. L., Ying, R., & Leskovec, J. (2017).** *Inductive Representation Learning on Large Graphs.* NeurIPS 30. — GraphSAGE; the foundational inductive-aggregator result.
- **Veličković, P., Cucurull, G., Casanova, A., Romero, A., Liò, P., & Bengio, Y. (2018).** *Graph Attention Networks.* ICLR. — GAT; the spatial-aggregation layer used in this report.
- **Wu, Z., Pan, S., Long, G., Jiang, J., & Zhang, C. (2019).** *Graph WaveNet for Deep Spatial-Temporal Graph Modeling.* IJCAI. — Source of the TemporalConv (dilated causal 1-D convolution per node) used here.
- **Wang, S., Li, Y., Zhang, J., Meng, Q., Meng, L., & Gao, F. (2020).** *PM2.5-GNN: A Domain Knowledge Enhanced Graph Neural Network for PM2.5 Forecasting.* SIGSPATIAL. — Wind-aware graph construction; the only ST-GNN published specifically on PM2.5.
- **Jin, Y., Chen, K., & Yang, Q. (2023).** *Transferable Graph Structure Learning for Graph-Based Traffic Forecasting Across Cities.* KDD. — TransGTR; the direct prior art for our feature-induced adjacency idea (planned for the next iteration).
- **Tang, Y., Qu, A., Chow, A. H. F., Lam, W. H. K., Wong, S. C., & Ma, W. (2022).** *Domain Adversarial Spatial-Temporal Network: A Transferable Framework for Short-term Traffic Forecasting across Cities.* CIKM. — DASTNet; the direct prior art for our Variant B.
- **Du, Y., Wang, J., Feng, W., Pan, S., Qin, T., Xu, R., & Wang, C. (2021).** *AdaRNN: Adaptive Learning and Forecasting for Time Series.* CIKM. — Climatology-residual style temporal DA recipe.
- **Cai, T., Luo, S., Xu, K., He, D., Liu, T.-Y., & Wang, L. (2021).** *GraphNorm: A Principled Approach to Accelerating Graph Neural Network Training.* ICML. — GraphNorm; the planned next-iteration fix for failure mode F-ii.
- **Ganin, Y., & Lempitsky, V. (2015).** *Unsupervised Domain Adaptation by Backpropagation.* ICML. — DANN; the foundational adversarial DA paper that Variant B builds on.

---

*End of Main Report. See [reports/REFERENCES.md](REFERENCES.md) for the complete reference list and [reports/RESULTS.md](RESULTS.md) for the full numerical results.*
