# Inductive Graph Neural Networks with Adversarial Domain Adaptation for Cross-City PM2.5 Forecasting under Heterogeneous Station Topologies

**A Comprehensive Technical Report**

---

## Contents

- [Executive summary](#executive-summary)
- [1. Problem and motivation](#1-problem-and-motivation)
- [2. Data](#2-data)
- [3. The cross-city graph problem](#3-the-cross-city-graph-problem)
- [4. Models — how each one works](#4-models--how-each-one-works)
  - [4.1 The forecasting task, stated once](#41-the-forecasting-task-stated-once)
  - [4.2 The LSTM baseline](#42-the-lstm-baseline)
  - [4.3 The inductive ST-GNN (GAT + TCN)](#43-the-inductive-st-gnn-gat--tcn)
  - [4.4 GNN transfer learning — the two-phase pipeline](#44-gnn-transfer-learning--the-two-phase-pipeline)
- [5. Experimental protocol](#5-experimental-protocol)
- [6. Results](#6-results)
- [7. Discussion](#7-discussion)
- [8. Reproducibility](#8-reproducibility)
- [9. Suggested plots and figures](#9-suggested-plots-and-figures)
- [10. References](#10-references)

---

## Executive summary

This report documents an inductive **Graph Neural Network (GNN)** framework for **transfer learning (TL)** of PM2.5 air-quality forecasts across three Indian cities whose monitoring networks differ in size by an order of magnitude — **Delhi (40 CPCB stations), Kolkata (10), and Guwahati (4)**. The cities that most need forecasting (sparse-sensor cities such as Guwahati) are exactly the cities where a model cannot be trained from local data alone. Transfer learning is the natural response: learn on a data-rich source city, adapt to a data-scarce target.

The technical obstacle that distinguishes *graph-based* cross-city transfer from sequence-based transfer is simple to state and hard to solve: a GNN operates on a graph `G = (V, E)`, but `V` and `E` change when the city changes. A model that has any weight whose *shape* depends on the number of stations `|V|` cannot be moved from a 40-node city to a 4-node city at all. We solve this with an **inductive** spatio-temporal GNN whose entire parameter set is independent of `|V|`, so the *same* trained encoder runs forward, unchanged, on Delhi's 40-node graph and on Guwahati's 4-node graph.

On top of this encoder we study two complementary transfer mechanisms, forming a **two-phase GNN-TL pipeline**:

- **Phase 1 — Pre-train + Fine-tune (PT-FT):** train the encoder on the source city, then fine-tune it on a small `d %` slice of the target city's data.
- **Phase 2 — Graph-DANN (adversarial domain adaptation):** additionally force the encoder's pooled graph embedding to be *statistically indistinguishable* across cities (via a gradient-reversal city classifier) before fine-tuning, yielding an explicitly city-invariant representation.

> Phase 1 and Phase 2 here correspond to "Stage 1 (Variant A)" and "Stage 2 (Variant B)" in [RESULTS.md](RESULTS.md) and [PAPER_DRAFT.md](PAPER_DRAFT.md) — same models, descriptive names.

**Headline results (all on held-out target test data, final protocol).** Across the full grid of six ordered city pairs × four target-data fractions `d ∈ {15, 30, 45, 60} %` (24 cells per phase):

- Every source-only forecaster reaches **R² ≈ 0.82–0.86** on all three cities, including the 4-station Guwahati graph.
- Under a strict **three-way verification** (transfer must beat *both* a zero-shot baseline *and* a from-scratch baseline trained on the same target slice), **Phase 1 shows real knowledge transfer in 24/24 cells** and **Phase 2 in 23/24 cells**.
- The two phases are **statistically tied** on raw forecast accuracy (mean R² difference 0.0009; 12 wins each). Phase 2's distinct contribution is a *representation guarantee* — a city-invariant embedding that Phase 1 does not provide — which is the basis for zero-target-label and multi-source deployment.
- A fixed-protocol LSTM baseline is the strongest single-shot transferer in absolute R², but the GNN phases stay within ≈ 0.04 R² **and** uniquely solve the structural problem the LSTM cannot: the same architecture operates across `|V| ∈ {4, 10, 40}` with no parameter-shape change.

The report is self-contained: §4 explains each model from first principles (plain language, then the mathematics, then the literature), with particular depth on the GNN and the two-phase transfer pipeline, since those are the contribution. §9 proposes the figure set for a submission.

> **Scope note.** All numbers in this report are from the final, corrected protocol (interleaved 70/15/15 split + per-(month, hour) climatology-residual target + per-city training recipe). Earlier exploratory runs are not reported here. Prior LSTM transfer-learning work on these cities (Sanjeev, Prakash & Maitra, 2025) is treated as related work in §1 and the bibliography only; it is not used as a results anchor.

---

## 1. Problem and motivation

### 1.1 PM2.5 forecasting in India

Fine particulate matter (PM2.5) is the world's leading environmental health risk. In India the burden is acute: the 2023 World Air Quality Report ranked India the third-most-polluted country, and annual-mean PM2.5 in major Indian cities exceeds the World Health Organization guideline (5 µg/m³ annual mean) by an order of magnitude (CSE, 2024; WHO, 2021). Short-horizon forecasts at 1–6 hour lead times are operationally critical: they trigger graded-response actions, inform school-closure and traffic-restriction decisions, and drive personal-exposure advisories with demonstrable health benefits.

### 1.2 The data-scarcity bottleneck

The Central Pollution Control Board (CPCB) Continuous Ambient Air Quality Monitoring (CAAQM) network is the only public real-time PM2.5 feed for most Indian cities, and its coverage is highly unequal:

| City | Active stations `|V|` | Period | Climate regime |
|---|:--:|---|---|
| Delhi | 40 | Jan 2021 – Dec 2022 | Indo-Gangetic plain; severe winter inversion |
| Kolkata | 10 | 2023 | Coastal Gangetic delta; humid tropical |
| Guwahati | 4 | 2023 | Brahmaputra valley; pre-monsoon dust + monsoon |

The cities with the densest networks (Delhi) are where forecasting is *least* constrained; the cities with the sparsest networks (Guwahati and similar northeastern cities) are where forecasting is *most* needed and *hardest* to do from local data alone. Many secondary cities have one station or none. This long-tailed distribution means "train a deep model per city" is feasible only for Delhi and a handful of Tier-1 cities.

**Transfer learning** is the natural remedy: pre-train on a data-rich source and adapt to a data-scarce target with a small fine-tune set.

### 1.3 Why a graph model, and why transfer is hard for it

PM2.5 is a *spatial* signal. Pollutant concentrations at one station are coupled to upstream stations by wind transport, to nearby stations by local dispersion, and to regional stations by emission corridors. A station-independent model (e.g. one LSTM applied to each station in isolation) cannot exploit this coupling. Spatio-temporal GNNs (ST-GNNs) model the monitoring network as a graph and learn a single spatial operator that propagates information along edges (Yu, Yin & Zhu, 2018; Wu et al., 2019; Wang et al., 2020).

But every published ST-GNN air-quality benchmark trains and tests on a *single fixed graph*. Cross-city transfer adds a constraint they never face: the graph `G = (V, E)` is different for the source and the target. The source has 40 nodes and one topology; the target has 4 nodes and a different topology, with disjoint station identities. This is the **cross-city graph problem** (§3), and solving it is the core contribution.

### 1.4 Related work (where prior approaches sit, including LSTM-TL)

- **Air-quality GNNs.** PM2.5-GNN (Wang et al., 2020) introduced wind-aware directed edges for nationwide Chinese PM2.5; GAGNN (Chen et al., 2021) added group-aware aggregation; AirFormer (Liang et al., 2023) used a stochastic transformer. None transfer across cities of different station counts.
- **Cross-city ST transfer.** RegionTrans (Wang et al., 2019), MetaST (Yao et al., 2019), ST-GFSL (Lu et al., 2022), CrossTReS (Jin et al., 2022), TransGTR (Jin et al., 2023), and DASTNet (Tang et al., 2022) study cross-city transfer for *traffic* on relatively homogeneous road networks. DASTNet is the closest precedent for adversarial cross-city transfer; none treats order-of-magnitude `|V|` heterogeneity as a first-class concern.
- **Inductive GNNs.** GraphSAGE (Hamilton et al., 2017) and GAT (Veličković et al., 2018) parameterize *functions over neighborhoods* rather than per-node embeddings, making cross-graph reuse possible; transductive GCNs (Kipf & Welling, 2017) and adaptive-adjacency models (Graph WaveNet, Wu et al., 2019; AGCRN, Bai et al., 2020) do not.
- **Sequence-based PM2.5 transfer in India.** Station-independent LSTM transfer learning for cross-city PM2.5 in these three cities was studied by Sanjeev, Prakash & Maitra (2025) and, for cross-year Delhi PM2.5 with attention, by Yadav et al. (2024). These motivate the LSTM baseline we implement and compare against in §4.2, and the climatology-residual target normalization in §5.

---

## 2. Data

### 2.1 Source and features

All data is CPCB CAAQM station-level PM2.5 at 3-hour cadence. We intersect the feature schema across the three cities to a common physical set and add cyclic temporal encodings, giving **F = 14 features** per (station, timestep):

| Group | Features |
|---|---|
| Pollutant (target) | PM2.5 |
| Meteorology | Air temperature (AT), relative humidity (RH), wind speed (WS), wind direction as `Sin_WD`, `Cos_WD` |
| Temporal (cyclic) | sin/cos of hour-of-day; sin/cos of month-of-year |
| Season (one-hot) | Winter, Spring, Summer, Monsoon (India-adapted) |

Wind direction is decomposed to sine/cosine to avoid the 0°/360° discontinuity. Season one-hots are aligned to the Indian climatological year.

### 2.2 Preprocessing

The pipeline ([src/data_pipeline.py](../src/data_pipeline.py)) performs schema harmonization, cyclic time encoding, and missing-value handling by per-station forward-then-backward fill, then per-station mean, then city mean, then dropping any residual NaN row (empty in practice on the dense post-2020 CPCB feeds). Outputs are per-city processed CSVs plus a `metadata.json` with station coordinates and split boundaries. The imputation strategy and its (negligible) leakage implications are audited in [AUDIT.md](AUDIT.md).

### 2.3 Distributional facts that drive the methodology

PM2.5 concentration differs sharply across the three cities — the **city-to-city gap dominates** every other source of variation:

| City | mean | std | median | p95 |
|---|--:|--:|--:|--:|
| Delhi | 101.5 | 90.4 | 71.2 | 281.7 |
| Kolkata | 49.0 | 39.1 | 37.7 | 126.8 |
| Guwahati | 57.9 | 54.5 | 40.6 | 163.5 |

Two facts shape the protocol of §5:

1. **Strong seasonal-diurnal structure.** PM2.5 has a large, predictable winter peak and a daily cycle. Much of the "forecasting" problem is just reproducing this climatology, which is essentially a lookup table — not something a model should have to *learn*.
2. **Kolkata and Guwahati have only one year of data.** A naïve chronological split would put the entire winter peak in the test partition (the model trains on Jan–Sep, tests on Nov–Dec), forcing the model to extrapolate to a regime it never saw. This is an evaluation artifact, not a property of the forecasting problem. §5.1 describes the interleaved split that removes it.

### 2.4 Graph construction

Station coordinates (lat, lon) are used to build each city's graph from physical pairwise great-circle (Haversine) distances. The default strategy is **k-NN with k = 3** and Gaussian-decay edge weights:

```
edge i → j  iff  j is among i's 3 nearest neighbours by Haversine distance
weight      w_ij = exp( − d_ij² / (2σ²) ),   σ = 5 km
```

The Gaussian kernel places ≈ 90 % of the weight within the first 7 km, matching the empirical PM2.5 spatial-correlation length in urban India. The construction is a *rule over coordinates*, so it transfers across cities by definition — no learned per-city structure.

| City | `|V|` | `|E|` | avg degree |
|---|--:|--:|--:|
| Delhi | 40 | 120 | 3.00 |
| Kolkata | 10 | 30 | 3.00 |
| Guwahati | 4 | 12 | 3.00 |

With k = 3, the same average degree (3.00) is held across cities, which keeps the aggregation statistics comparable. [src/graph_construction.py](../src/graph_construction.py) also implements a wind-aware directed graph (PM2.5-GNN style; used in ablation) and a hybrid learnable-residual graph; the experiments here use k-NN as the simplest sufficient baseline.

---

## 3. The cross-city graph problem

A GNN computes a function of (a) node features and (b) the adjacency matrix `A`. When `A` changes between training (source) and inference (target), three distinct things can break:

- **F-i — Parameter-shape mismatch.** If any weight has a shape tied to `|V|` (e.g. a per-node embedding table `E ∈ ℝ^{|V|×d}`, as in AGCRN, Bai et al. 2020; or a learned `|V|×|V|` adjacency, as in Graph WaveNet, Wu et al. 2019), it has no value for the target's nodes and the tensor shapes are simply incompatible. The model literally cannot load.
- **F-ii — Aggregation-statistic shift.** Even with compatible shapes, the *statistics* of message passing (degree, normalization constants, magnitude) scale with graph size and density; a model trained at Delhi scale drifts out of its learned operating regime at Guwahati scale (Cai et al., 2021).
- **F-iii — Receptive-field mismatch.** An `L`-layer GNN sees an `L`-hop neighborhood. On Guwahati's 4-node graph, `L = 2` already covers the whole graph — a regime the model never trained on at Delhi's 40-node scale.

**The fix used here is to remove the dependence on `|V|` from the parameters entirely** (an *inductive* architecture, §4.3), and to hold the average degree constant across cities (k = 3, §2.4). Inductivity dissolves F-i and largely controls F-ii/F-iii in our regime; the theoretical justification for "the same operator generalizes across graphs of different size drawn from a similar process" is the graphon-transferability result of Ruiz, Chamon & Ribeiro (2020) and the spectral analysis of Levie et al. (2021). The empirical evidence is the 24/24 transfer pass rate in §6.

---

## 4. Models — how each one works

This section is the heart of the report. For each model we give three layers of explanation: an **intuition** (plain language), the **mathematics**, and the **literature** it draws on. The GNN (§4.3) and the two-phase GNN-TL pipeline (§4.4) are treated in the most depth.

### 4.1 The forecasting task, stated once

Let `c ∈ {Delhi, Kolkata, Guwahati}` index a city with `N_c` stations. Using a history window of `H = 8` timesteps (24 hours at 3-hour cadence), we predict PM2.5 one step ahead (`horizon = 1`, i.e. 3 hours) at *every* station:

```
f_θ : ℝ^{H × N_c × F}  →  ℝ^{N_c}
```

Crucially, all models below realize `f_θ` so that the parameter count of `θ` is **independent of `N_c`** — the property that makes cross-city transfer mechanically possible. Targets are predicted in **z-scored climatology-residual space** (§5.2); metrics are reported after inverting the z-score and adding the climatology back, in raw µg/m³.

### 4.2 The LSTM baseline

**Intuition.** Treat each monitoring station as an independent time series. A recurrent network reads the last 24 hours of that station's features hour-by-hour, maintaining a memory of the recent trajectory, and predicts the next value. The *same* network weights are applied to every station, so the model is "station-agnostic": transferring it to a new city just means running the same recurrent function on the new city's stations. What it cannot do is let one station's reading inform another's — there is no spatial coupling.

**Mathematics.** A 2-layer LSTM (Hochreiter & Schmidhuber, 1997) processes the per-station sequence `x_1, …, x_H ∈ ℝ^F`. At each step the LSTM cell updates a hidden state `h_t` and cell state `c_t` through input, forget, and output gates:

```
i_t = σ(W_i [h_{t-1}, x_t] + b_i)        (input gate)
f_t = σ(W_f [h_{t-1}, x_t] + b_f)        (forget gate)
o_t = σ(W_o [h_{t-1}, x_t] + b_o)        (output gate)
g_t = tanh(W_g [h_{t-1}, x_t] + b_g)     (candidate)
c_t = f_t ⊙ c_{t-1} + i_t ⊙ g_t          (cell update)
h_t = o_t ⊙ tanh(c_t)                    (hidden update)
```

The final hidden state `h_H` is mapped to the prediction by a two-layer head `ŷ = W_2 · ReLU(W_1 h_H)`. Hidden size 64, dropout 0.2; ≈ 50 k parameters. Because the weights act on a single station's sequence and are shared across stations, the parameter count does not depend on `N_c`.

**Literature.** This is the classic sequence-to-one recurrent forecaster. Station-independent LSTM transfer learning for these three cities was studied by Sanjeev, Prakash & Maitra (2025); cross-year attention-LSTM PM2.5 transfer by Yadav et al. (2024). We implement it as the apples-to-apples baseline for the GNN: same data, same fixed protocol, same per-city recipe.

### 4.3 The inductive ST-GNN (GAT + TCN)

This is the spatial model. It is a four-block sandwich — **TCN → GAT ×2 → TCN → Linear** — in the family of STGCN (Yu, Yin & Zhu, 2018), Graph WaveNet (Wu et al., 2019), and MTGNN (Wu et al., 2020), kept deliberately small (~25 k parameters, 23,745 in the fixed config) so it runs on a laptop CPU. Implementation: [src/models/stgnn_gat.py](../src/models/stgnn_gat.py).

#### 4.3.1 Intuition

Think of two interleaved operations:

- **"Look back in time" (temporal convolution, TCN):** for each station independently, slide a small causal filter over the last few hours to summarize the recent trend (rising? falling? spiking?). "Causal" means the filter only ever looks at the past, never the future.
- **"Look around in space" (graph attention, GAT):** at each timestep, let each station update its state by a *weighted average* of its neighbours' states, where the model *learns how much to trust each neighbour*. A downwind station learns to weight an upwind neighbour heavily; a station behind a hill learns to ignore a geographically-close but meteorologically-disconnected one.

Stacking time-then-space-then-time lets the model answer "given the recent local trend *and* what the relevant neighbours are doing, what comes next?"

The key design property: **none of the learned weights know how many stations there are.** The temporal filter is shared across stations; the attention mechanism is a shared function of *pairs* of station features. So the exact same trained weights run on 40 stations or 4.

#### 4.3.2 The temporal block (TCN)

A 1-D causal convolution of kernel size 3 is applied independently per node along the time axis, with left-padding `(k−1)·d` and a causal trim so no future timestep leaks into the present. Two blocks use dilation `d = 1` and `d = 2`; the second covers an effective receptive field of 5–7 of the 8 history steps. This is the TCN primitive of Bai, Kolter & Koltun (2018) and the temporal half of Graph WaveNet (Wu et al., 2019). Its parameter count is `(in · out · 3)` — independent of `|V|`.

#### 4.3.3 The spatial block (edge-weighted GAT)

For each directed edge `i → j`, the attention logit is

```
e_ij = LeakyReLU( ⟨a_src, W x_i⟩ + ⟨a_dst, W x_j⟩ ) + log(w_ij + ε)
```

where `W ∈ ℝ^{out×in}` is a shared linear projection and `a_src, a_dst ∈ ℝ^{out}` are shared per-endpoint attention vectors (the additive-attention formulation of Veličković et al., 2018). The term `log(w_ij)` injects the external k-NN Gaussian edge weight *log-additively*, which is equivalent to a multiplicative prior inside the softmax. A per-destination softmax normalizes the logits,

```
α_ij = exp(e_ij) / Σ_{k ∈ N(j)} exp(e_kj),
```

and the node update is the attention-weighted aggregation

```
h_j = Σ_{i ∈ N(j)} α_ij · (W x_i).
```

**Why this is inductive.** The number of learnable parameters in the layer is `out·(in + 2)` — it counts the entries of `W`, `a_src`, `a_dst` and *nothing else*. There is **no** per-node parameter and **no** `|V|×|V|` term. The same weight tensor therefore runs forward on any graph regardless of `|V|` or `|E|`. This is the concrete resolution of failure mode F-i (§3).

#### 4.3.4 The full forward pass

```
Input  [B, H, N, F]                       # batch, history, nodes, features
  ↓ TemporalConv (k=3, dilation=1)         → [B, H, N, hidden]
  ↓ reshape to [B·H, N, hidden]            # run the GAT once per timestep
  ↓ GATLayer (hidden → gat_dim) → ELU      → [B·H, N, gat_dim]
  ↓ GATLayer (gat_dim → gat_dim) → ELU     → [B·H, N, gat_dim]
  ↓ reshape back to [B, H, N, gat_dim]
  ↓ TemporalConv (k=3, dilation=2)         → [B, H, N, hidden]
  ↓ LayerNorm
  ↓ take last timestep                     → [B, N, hidden]
  ↓ Linear head                            → [B, N]   # next-step PM2.5 per station
```

Fixed config: `in = 14`, `hidden = 64`, `gat_dim = 64`, dropout 0.15, single attention head.

#### 4.3.5 The size-invariant graph readout (used by Phase 2)

When a *graph-level* vector is needed (for the adversarial discriminator in §4.4.4), the last-step hidden state is pooled over the node axis by concatenating mean and max pooling:

```
z = [ mean_n h_T[:, n, :]  ‖  max_n h_T[:, n, :] ]  ∈  ℝ^{B × 2·hidden}
```

Mean+max pooling collapses an arbitrary number of nodes to a *fixed* `2·hidden = 128`-dimensional vector. This is the standard GIN-style readout (Xu et al., 2019) and is exactly the primitive that lets a single discriminator compare cities whose `|V|` differs by 10×.

#### 4.3.6 Why it transfers across `|V|` — the theory

Every learnable shape depends only on `(F, hidden, gat_dim)`, never on `N_c` (enumerated in the parameter-shape table of [ARCHITECTURE.md](ARCHITECTURE.md)). So a single checkpoint loads onto Delhi's 40-node graph and Guwahati's 4-node graph without modification. *Why should the same operator behave sensibly on a different-sized graph?* The graphon-transferability theorem (Ruiz, Chamon & Ribeiro, 2020) shows that for graphs sampled from a common generating process (a "graphon"), an inductive GNN's outputs converge as `|V|` grows; the spectral-transferability bound of Levie et al. (2021) bounds the transfer error by spectral perturbation. The three CPCB k-NN station graphs are plausibly sampled from a similar urban-airshed process, which is the formal license for the empirical transfer we observe in §6.

### 4.4 GNN transfer learning — the two-phase pipeline

Both phases share the **same source-only pre-training** (§4.4.1) and the **same inductive encoder** (§4.3). They differ only in *how they adapt to the target city*. This is the central methodological contribution and is explained in the most detail.

#### 4.4.1 Phase 0 — source-only pre-training

For each city the encoder is trained on that city's full training partition with MSE loss in z-scored climatology-residual space, Adam optimizer, weight decay `1e-5`, gradient-clip norm 1.0, `ReduceLROnPlateau` on validation R², and best-by-validation checkpointing. This produces three source checkpoints (`gat_source_<City>.pt`) that both transfer phases start from.

#### 4.4.2 Phase 1 — Pre-train + Fine-tune (PT-FT)

**Intuition.** Knowledge learned on the data-rich source is stored in the encoder's weights. To adapt to a new city, *initialize* the target model with those weights instead of random numbers, then nudge them with a little target data. The early temporal feature-extractor (`t1`) is "frozen" for the first 20 % of fine-tune epochs — we keep the generic recent-trend detector fixed while the rest of the network re-aims at the new city, then unfreeze everything. This is the standard "transfer the low-level features, adapt the high-level ones" recipe.

**Mechanics.** Load source weights → freeze `t1` for the first 20 % of epochs (Yosinski et al., 2014) → fine-tune on a `d %` random sample of the target's *training* windows (`d ∈ {15, 30, 45, 60} %`, seed 42) → early-stop on the target's *own validation* set (disjoint from the `d %` sample) → evaluate on the target's held-out test partition. The fine-tune never sees target validation or test data. Implementation: [src/train_gnn_tl.py](../src/train_gnn_tl.py).

**Literature.** Supervised pre-train-then-fine-tune for GNNs (Hu et al., 2020); discriminative layer-freezing (Yosinski et al., 2014).

#### 4.4.3 The three-way verification protocol (how we prove transfer is *real*)

A transfer number alone cannot tell you whether the source *helped* — maybe fine-tuning on `d %` of target data would have worked just as well from scratch. To close that loophole, for **every** `(source, target, d %)` cell we train three models on the *identical* target sample and evaluate all three on the same target test set:

```
zero-shot : load source weights, NO fine-tune          → R²_zs
transfer  : load source weights, fine-tune on d%        → R²_tl   (the headline)
scratch   : random init, fine-tune on the same d%       → R²_sc
```

Transfer is declared **real** iff `R²_tl > R²_zs` AND `R²_tl > R²_sc`. Failing the first means fine-tuning *hurt*; failing the second means the source was *irrelevant*. Passing both is positive evidence that the source weights are a genuinely useful initialization. This protocol is applied to both phases (§6.3).

#### 4.4.4 Phase 2 — Graph-DANN (adversarial domain adaptation)

**Intuition.** PT-FT transfers *weights*, but those weights still encode source-specific statistics (Delhi's pollution looks different from Guwahati's). Phase 2 goes further: it forces the encoder to produce a pooled graph embedding whose *distribution looks the same regardless of which city it came from*. The trick is a small "referee" network — a city classifier — that tries to guess the source city from the embedding. A **gradient-reversal layer** sits between the encoder and the referee: during backpropagation it flips the sign of the referee's gradients, so the encoder is trained to *fool* the referee. When the referee can no longer tell the cities apart, the embedding is "city-invariant." Only then do we fine-tune on the target.

**Mathematics.** A 3-class city discriminator `h_d` sits on the size-invariant pooled embedding `z` (§4.3.5) behind a Gradient Reversal Layer (GRL; Ganin & Lempitsky, 2015). The joint-phase loss per mini-batch is

```
L = MSE(ŷ, y_PM2.5)  +  α_d · [ CE(ĉ_src) + CE(ĉ_tgt) + CE(ĉ_third) ]
```

processed as a three-city round-robin (one source batch, one target batch with PM2.5 labels withheld, one third-city replay batch), each on its own graph. The GRL multiplies gradients flowing into the encoder by `−λ`, so the encoder *maximizes* the discriminator's confusion while the discriminator *minimizes* its own error — a minimax game. The adversarial weight follows the Ganin warm-up schedule

```
λ(p) = 2 / (1 + exp(−γ·p)) − 1,   p = step / total_steps,   γ = 10,
```

which starts at 0 (the forecaster learns first) and saturates near 1 by the end of the joint phase (`EPOCHS_JOINT = 25`). The GRL is then switched off (`λ = 0`) and the encoder is fine-tuned on the target's `d %` labelled windows with the Phase-1 recipe. Implementation: [src/train_gnn_dann.py](../src/train_gnn_dann.py), [src/models/dann.py](../src/models/dann.py).

**Stabilization.** Adversarial training of *regression* models is delicate (de Mathelin et al., 2020). The working configuration uses: a source-pretrained warm-start (Tzeng et al., 2017, ADDA), discriminator-loss reweighting `α_d = 0.1` so the cross-entropy and MSE are comparable in magnitude, a slow `λ` schedule, and LayerNorm on the pooled embedding before the GRL (decoupling the discriminator from `|V|`-dependent activation scale — a GraphNorm-style fix, Cai et al., 2021). These choices and their effects are tabulated in [PAPER_DRAFT.md §7.7](PAPER_DRAFT.md).

**Literature.** DANN / gradient reversal (Ganin & Lempitsky, 2015; Ganin et al., 2016); ADDA warm-start (Tzeng et al., 2017); cross-city adversarial ST transfer for traffic (DASTNet, Tang et al., 2022); regression-DANN caveats (de Mathelin et al., 2020); GIN readout (Xu et al., 2019).

**Why Phase 2 over Phase 1.** On raw forecast accuracy the two are tied (§6). Phase 2's distinct value is a **representation guarantee**: a city-invariant pooled embedding. That guarantee is what enables (a) *zero-target-label deployment* — a brand-new city for which only sensor features (no PM2.5 labels) exist can be aligned through the joint phase, which PT-FT cannot do by construction; and (b) *multi-source aggregation* — the discriminator simply becomes a `k`-class problem. These are the natural extensions.

---

## 5. Experimental protocol

### 5.1 Interleaved 70/15/15 split

Because Kolkata and Guwahati have only one year of data, a chronological-tail test partition would be winter-only and unrepresentative (§2.3). We therefore use a deterministic **interleaved** split: every 7th timestep goes to validation (offset 0), every 7th to test (offset 1, disjoint from val), the remainder to train. Each split then spans every season of every year in roughly the right proportions. The same protocol is applied to *all* methods (LSTM and GNN) so any protocol effect cancels in method-vs-method comparison. The trade-off — that an interleaved (k-fold-style) split inflates absolute R² versus a chronological-block split because adjacent 3-hour steps are autocorrelated — is acknowledged and bounded in [AUDIT.md](AUDIT.md) (asymptotically valid for stationary AR processes after climatology removal; Bergmeir, Hyndman & Koo, 2018).

### 5.2 Per-(month, hour) climatology-residual target

A large, predictable part of PM2.5 is the seasonal-diurnal climatology (§2.3). We remove it so the model learns only the *residual fluctuations*. From the **train slice only**, compute the per-(month, hour) climatology mean of PM2.5; subtract it to form a residual; z-score the residual; the model predicts in residual space; at inference, invert the z-score and add the climatology back at the prediction timestamp. Computing climatology and scalers on the train slice only keeps the protocol leakage-free (verified in [AUDIT.md](AUDIT.md)). This is the AdaRNN-style temporal-DA idea (Du et al., 2021) applied to PM2.5 (cf. Yadav et al., 2024).

### 5.3 Training recipe (per city)

| Stage | City | Epochs | Batch | LR | Patience |
|---|---|--:|--:|--:|--:|
| Source-only GAT-GNN | Delhi / Kolkata / Guwahati | 25 / 40 / 60 | 128 / 64 / 32 | 1e-3 / 1e-3 / 5e-4 | 6 / 8 / 10 |
| Source-only LSTM | Delhi / Kolkata / Guwahati | 25 / 40 / 60 | 512 / 256 / 128 | 1e-3 / 1e-3 / 5e-4 | 6 / 8 / 10 |
| Fine-tune (PT-FT & DANN) | Delhi / Kolkata / Guwahati | 25 / 30 / 40 | 128 / 64 / 32 | 2e-4 / 2e-4 / 1e-4 | 6 / 8 / 10 |
| Graph-DANN joint phase | — | 25 | 32 per city | 5e-4 | — |

Common to all: Adam, weight decay 1e-5, gradient clip 1.0, dropout 0.15, hidden = gat_dim = 64, single GAT head, `t1` frozen for 20 % of fine-tune epochs, `ReduceLROnPlateau(factor=0.5, patience=3)`, seed 42. Smaller-data cities get more epochs, smaller learning rates, and smaller batches.

---

## 6. Results

All metrics are on the **target city's held-out test partition** in raw µg/m³ space. Per phase, the grid is 6 ordered (source, target) pairs × 4 `d %` values = 24 cells. The full per-cell tables are in [RESULTS.md](RESULTS.md); this section gives the headline numbers.

### 6.1 Source-only forecasts

| City | LSTM (R² / MAE) | GAT-GNN (R² / MAE) |
|---|--:|--:|
| Delhi | **0.8501** / 23.74 | 0.8343 / 24.18 |
| Kolkata | **0.8627** / 7.78 | 0.8244 / 9.23 |
| Guwahati | **0.8320** / 12.19 | 0.8311 / 12.75 |

Every source-only model reaches R² ≈ 0.82–0.86 on all three cities, **including the 4-station Guwahati graph** — the demonstration that the inductive encoder operates correctly at every `|V|`. The LSTM is marginally ahead on absolute fit (~0.02 R²); the GNN's advantage is structural transfer (§6.4), not raw single-city fit.

### 6.2 Transfer grid (R² at d = 30 %)

| Source → Target | LSTM-fixed | Phase 1 (PT-FT) | Phase 2 (Graph-DANN) |
|---|--:|--:|--:|
| Delhi → Kolkata | **0.8521** | 0.8158 | 0.8171 |
| Delhi → Guwahati | **0.8224** | 0.8165 | 0.8131 |
| Kolkata → Delhi | **0.8510** | 0.8085 | 0.8087 |
| Kolkata → Guwahati | **0.8129** | 0.8044 | 0.7916 |
| Guwahati → Delhi | **0.8491** | 0.8030 | 0.8029 |
| Guwahati → Kolkata | **0.8420** | 0.8085 | 0.8039 |

Best Phase 1 cell across all `d`: **Delhi → Guwahati @ 45 %, R² = 0.8273**. Best Phase 2 cell: **Delhi → Kolkata @ 45 %, R² = 0.8250**. (Full `d ∈ {15,30,45,60}` grids in [RESULTS.md §2.2, §3.1.1, §3.2.1](RESULTS.md).)

### 6.3 Three-way verification (does the source actually help?)

At `d = 30 %`, comparing transfer against zero-shot and scratch:

| Source → Target | zero-shot | scratch | **Phase 1 transfer** | real? |
|---|--:|--:|--:|:--:|
| Delhi → Kolkata | 0.7282 | 0.7803 | **0.8158** | ✓ |
| Delhi → Guwahati | 0.6841 | 0.7826 | **0.8165** | ✓ |
| Kolkata → Delhi | 0.6032 | 0.7905 | **0.8085** | ✓ |
| Kolkata → Guwahati | 0.6698 | 0.7924 | **0.8044** | ✓ |
| Guwahati → Delhi | 0.6317 | 0.7905 | **0.8030** | ✓ |
| Guwahati → Kolkata | 0.6980 | 0.7803 | **0.8085** | ✓ |

- **Phase 1: 24/24 cells show real knowledge transfer** (mean gain over scratch +0.0223 R²).
- **Phase 2: 23/24 cells show real knowledge transfer** (mean gain +0.0210 R²; the lone exception, Kolkata → Guwahati @ 30 %, ties scratch within 0.001 R²).

The transfer gains are largest at small `d` (transfer helps most when target data is scarcest) and shrink as `d` grows (scratch catches up) — the textbook transfer-learning signature.

### 6.4 Phase 1 vs Phase 2, and the cross-method picture

Across all 24 cells the two GNN phases are **tied 12–12**, mean R² difference 0.0009 — statistically indistinguishable, consistent with the regression-DANN literature (de Mathelin et al., 2020). Phase 2's wins concentrate in Delhi-source, low-`d` cells, matching the cross-city UDA finding that adversarial alignment helps most when source data is abundant and target labels are scarce (Tang et al., 2022).

The LSTM-fixed baseline is the strongest single-shot transferer in absolute R² (≈ +0.03 over the GNN phases on average), but the GNN phases stay within ≈ 0.04 R² **and** uniquely solve the structural-transfer problem: one architecture, no parameter-shape change, across `|V| ∈ {4, 10, 40}` — a property the station-independent LSTM cannot have. Full error metrics (MAE/RMSE/MAPE) per cell are in [RESULTS.md §3.1.2, §3.2.2](RESULTS.md).

---

## 7. Discussion

**What is established.**
1. An inductive ST-GNN with `|V|`-independent parameters genuinely solves the parameter-shape problem (F-i): the same encoder runs on 40-, 10-, and 4-station graphs and reaches R² ≈ 0.83 on all three source-only.
2. In 24/24 (PT-FT) and 23/24 (Graph-DANN) cells the source pre-training provides a real head start over from-scratch training on the same target slice — the transfer-learning premise holds in every direction tested.
3. The interleaved split + climatology-residual protocol is a defensible, literature-grounded evaluation for one-year-of-data cities, and is applied uniformly so cross-method comparisons are fair.

**What is not (yet) established.**
1. The GAT, as configured, does not strictly *dominate* the LSTM on absolute single-step accuracy at this 3-hour horizon. The published ST-GNN-over-RNN advantage tends to appear at longer horizons (where wind transport dominates persistence) and higher station densities (richer graph signal) — both untested here.
2. Phase 2 does not beat Phase 1 on accuracy; its value is the representation guarantee, which this report demonstrates structurally but does not yet exercise in a zero-target-label experiment.

**Honest limitations.** Single random seed (no variance bars or Diebold–Mariano test yet); three cities is a small `n`; only single-step horizon; only the k-NN graph evaluated; the interleaved-split absolute-R² inflation is acknowledged (a chronological-block sensitivity analysis is the top follow-up). These are detailed alongside threats to validity in [PAPER_DRAFT.md §9](PAPER_DRAFT.md) and the leakage analysis in [AUDIT.md](AUDIT.md).

**Operational implication.** A city with ≤ 10 CPCB stations need not train a dedicated deep forecaster: a Delhi-pretrained GAT-GNN fine-tuned on `d = 30 %` of the new city's data (~600 windows for a one-year deployment, ~10 min CPU) recovers R² within ≈ 0.04 of a fully-supervised model — a deployable SOP for Tier-2/Tier-3 Indian cities, extensible to multiple sources via the Graph-DANN multi-class discriminator.

---

## 8. Reproducibility

CPU-only PyTorch ≥ 2.1; NumPy, pandas, scikit-learn. **No CUDA, no PyTorch Geometric** — the GAT and SAGE layers are implemented from scratch in [src/models/](../src/models/) so the project runs on a laptop CPU. `set_seed(42)` at the start of every run; results reproduce to within ±0.005 R² across re-runs on the same machine.

```bash
# 1) Preprocess CPCB CSVs → dataset/processed/*.csv + metadata.json
python -m src.data_pipeline

# 2) Final protocol (interleaved split + climatology residual + per-city recipe)
python -u -m src.train_lstm    --mode all --fixed              # LSTM baseline (source-only + TL)
python -u -m src.train_gnn     --backbone gat --fixed          # GNN source-only
python -u -m src.train_gnn_tl  --backbone gat --fixed --full   # Phase 1: 24-cell PT-FT + verification
python -u -m src.train_gnn_dann --backbone gat --fixed         # Phase 2: 24-cell Graph-DANN

# 3) Regenerate the tabular report
python reports/generate_results_table.py
```

Full grid wall-clock on 8 vCPU / 16 GB / no GPU: ≈ 8 hours end-to-end. Files of interest are mapped in the [README](../README.md) and [ARCHITECTURE.md](ARCHITECTURE.md).

---

## 9. Suggested plots and figures

The following figure set is recommended for a submission. Several can be generated from the existing EDA scripts in [visualisations/EDA_Vis/](../visualisations/EDA_Vis/); the rest need a small results-plotting script over the JSONs in [results/](../results/).

**Data / context figures (mostly already produced by `visualisations/EDA_Vis/`):**

1. **Station maps of the three cities**, side by side, annotated with `|V|` and the k-NN graph edges overlaid — makes the 40 ↔ 10 ↔ 4 heterogeneity immediate. (`01_station_maps_geo.py`, `08_knn_graph_visualization.py`)
2. **PM2.5 distribution violins per city** (raw scale) — shows the dominant city-to-city distribution gap that motivates domain adaptation. (`03_distribution_violins.py`)
3. **Seasonal × diurnal heatmap of PM2.5 per city** — visual justification for the climatology-residual target. (`04_seasonal_diurnal_heatmaps.py`)
4. **Wind rose per city** — supports the wind-aware-graph ablation and the spatial-coupling argument. (`05_wind_rose.py`)
5. **Feature correlation matrix** — sanity check on the 14-feature schema. (`06_correlation_matrix.py`)

**Method figures (new schematics):**

6. **Encoder block diagram** — `TCN → GAT×2 → TCN → LayerNorm → Linear`, annotated with tensor shapes and a callout that *no block's parameters depend on `|V|`*.
7. **Cross-`|V|` schematic** — the single encoder drawn once, with arrows to the 40-, 10-, and 4-node graphs, captioned "same weights, three graphs."
8. **Graph-DANN diagram** — encoder → mean+max readout → GRL → 3-class city discriminator, with the three-city round-robin minibatch and the `λ(p)` warm-up curve as an inset.

**Results figures (new, from the result JSONs):**

9. **Transfer-R² heatmaps** (6 pairs × 4 `d %`), one panel per method (LSTM-fixed, Phase 1, Phase 2) — single-glance comparison of the full grid.
10. **Three-way verification grouped bar chart** at `d = 30 %`: zero-shot vs scratch vs transfer per pair, with the "real transfer" pass marked — the falsifiability evidence.
11. **Transfer gain vs `d %`** line plot (transfer − scratch, averaged over pairs) — shows gains shrinking as target data grows (the TL signature).
12. **Phase 1 vs Phase 2 paired scatter** across all 24 cells with the `y = x` line — visualizes the 12–12 tie.
13. **Predicted vs observed time series** for one representative target (e.g. Delhi → Guwahati @ 30 %) over a held-out winter week — qualitative, reviewer-friendly evidence the forecasts track reality.
14. *(optional)* **Attention-weight map** on a target city: edges shaded by learned `α_ij` for a high-pollution timestep — interpretability of what the GNN learned spatially.

---

## 10. References

The complete, topic-organized bibliography is in [REFERENCES.md](REFERENCES.md). The most load-bearing references for this report:

- **Hamilton, Ying & Leskovec (2017)** — GraphSAGE; the inductive-aggregator foundation.
- **Veličković et al. (2018)** — GAT; the spatial-attention layer used here.
- **Wu et al. (2019)** — Graph WaveNet; source of the dilated causal per-node TCN.
- **Wang et al. (2020)** — PM2.5-GNN; the only ST-GNN published specifically on PM2.5; wind-aware graph prior.
- **Tang et al. (2022)** — DASTNet; the cross-city adversarial precedent Phase 2 adapts.
- **Ganin & Lempitsky (2015)** — DANN / gradient reversal; the adversarial-DA foundation.
- **Du et al. (2021)** — AdaRNN; the climatology-residual temporal-DA idea.
- **Ruiz, Chamon & Ribeiro (2020)** / **Levie et al. (2021)** — graphon / spectral transferability; why a single operator generalizes across `|V|`.
- **Xu et al. (2019)** — GIN; the mean+max graph readout.
- **Cai et al. (2021)** — GraphNorm; the pre-GRL normalization rationale.

---

*End of report. See [RESULTS.md](RESULTS.md) for full per-cell numbers, [ARCHITECTURE.md](ARCHITECTURE.md) for the implementation deep-dive, [AUDIT.md](AUDIT.md) for the leakage/overfitting audit, and [PAPER_DRAFT.md](PAPER_DRAFT.md) for the journal-style manuscript.*
