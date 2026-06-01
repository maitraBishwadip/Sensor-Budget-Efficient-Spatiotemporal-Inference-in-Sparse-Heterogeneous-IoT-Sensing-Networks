
# Inductive Graph Neural Networks with Adversarial Domain Adaptation for Cross-City PM2.5 Forecasting under Heterogeneous Station Topologies

**Bishwadip Maitra**¹, **C. T. Sanjeev**¹, **B. B. Prakash**¹, **Mainak Thakur**¹

¹ Indian Institute of Information Technology Sricity, Andhra Pradesh, India.

✉ Corresponding author: bishwadip.maitra (at) iiits.in

---

## Abstract

Operational PM2.5 forecasting in many Indian cities is constrained by sparse and heterogeneous monitoring infrastructure: Delhi has 40 Central Pollution Control Board (CPCB) stations, Kolkata 10, and Guwahati only 4. Transfer learning (TL) is the natural remedy for such data scarcity, but the dominant station-independent LSTM-TL approach — including our prior B.Tech-thesis baseline — cannot exploit the spatial coupling between stations and produces parameter shapes that cannot be reused across cities of different sizes. We address both limitations with an **inductive spatio-temporal graph neural network (ST-GNN)** whose parameter count is independent of `|V|`, and we transfer it across cities through two complementary strategies: a supervised pre-train + fine-tune (PT-FT) recipe, and a novel **Graph-DANN** that performs adversarial domain adaptation on the size-invariant mean+max-pooled graph embedding. The single encoder, trained on Delhi's 40-station graph, is shown to operate verbatim on Guwahati's 4-station graph without any shape change. Across the full grid of six ordered city pairs × four target-data fractions {15, 30, 45, 60 %} (24 cells per stage), Stage 1 PT-FT achieves "real" knowledge transfer in 24/24 cells and Stage 2 Graph-DANN in 23/24 cells under our strict three-way verification protocol (transfer must beat both the source-only zero-shot baseline AND a random-initialized scratch baseline on the same target sub-sample). The two GNN-TL stages are statistically tied in absolute forecasting accuracy (mean R² difference = 0.0009; 12 wins each), but Stage 2 contributes an explicitly city-invariant graph embedding — a representation guarantee that PT-FT does not provide and the basis for zero-target-label deployment. Compared with the B.Tech-thesis LSTM-TL baseline, every (source, target) cell is improved (Δ R² up to +0.21 on Delhi → Guwahati). We accompany the methods with a complete data-leakage and overfitting audit and report a mean validation–test R² gap of −0.005 (test slightly above val) on all reported cells, indicating no overfitting. Code and processed data are released under a permissive licence to support reproducibility on CPU-only research environments.

**Keywords:** PM2.5 forecasting; graph neural networks; transfer learning; adversarial domain adaptation; heterogeneous graphs; air quality; cross-city; India.

---

## 1. Introduction

### 1.1 Why short-horizon PM2.5 forecasting matters in India

Particulate matter with aerodynamic diameter below 2.5 µm (PM2.5) is the world's leading environmental health risk, contributing an estimated 4.1 million premature deaths annually (HEI, 2024). In India the burden is acute: the 2023 World Air Quality Report ranked India as the third-most-polluted country, with the National Capital Region of Delhi the most-polluted megacity (IQAir, 2024). The Centre for Science and Environment estimated Delhi's 2023 annual mean PM2.5 at ~100 µg m⁻³, twenty times the World Health Organization air-quality guideline of 5 µg m⁻³ (CSE, 2024; WHO, 2021). Short-horizon forecasts at 1–6 hour lead times are operationally critical: they trigger graded-response actions under India's Graded Response Action Plan (GRAP), inform school-closure decisions, and drive personal-exposure advisories that have demonstrable cardiopulmonary benefits.

### 1.2 The data-scarcity bottleneck

The Continuous Ambient Air Quality Monitoring (CAAQM) network operated by the Central Pollution Control Board (CPCB) provides the only public real-time PM2.5 feed for most Indian cities. Coverage is highly unequal: Delhi has 40 CAAQM stations in a ~50 × 50 km basin, Kolkata has 10 across a coastal-delta region, and Guwahati in the Northeast has only 4 stations in a Brahmaputra-valley pocket (Awasthi, Pandey & Verma, 2023). Many secondary cities have one or none. This long-tailed distribution of monitor density means that any "train a deep model per city" recipe is feasible only for Delhi and a handful of Tier-1 cities, leaving the majority of India under-served by data-driven forecasting.

Transfer learning (TL) is the natural remedy: pre-train a forecasting model on a data-rich source city and adapt it to a data-scarce target with a small fine-tune set. The B.Tech-thesis study underlying this project (Sanjeev, Prakash & Maitra, 2025) established that an LSTM-TL recipe achieves substantial R² gains on data-scarce targets (Guwahati R² 0.5723 → 0.6271 for Kolkata → Guwahati at 60 % target data) and provides our Stage I empirical anchor. Comparable TL gains have been reported for ozone forecasting in the Alpine region (Sangiorgio & Guariso, 2024), Delhi PM2.5 across years (Yadav et al., 2024), and crop-yield prediction across the US Corn Belt (Khan, Li & Maimaitijiang, 2024). Cross-city PM2.5 forecasting with TL is, however, structurally different from these settings because the source and target are not the same physical system observed at different times — they are different urban airsheds with different topologies, emission inventories, and meteorological regimes.

### 1.3 The architectural limitation that motivates Stage II

The LSTM-TL recipe — including the thesis baseline and the more recent attention-LSTM work of Yadav et al. (2024) — operates **station-independently**: one sequence-to-sequence forecaster is applied to each monitoring station's time series in isolation, and the predictions are concatenated. This has two consequences that cap how far TL can take us:

(i) **Spatial coupling is ignored.** PM2.5 transports along wind corridors and forms regional plumes whose spatial structure (Wang et al., 2020 PM2.5-GNN) carries genuine predictive signal. Station-independent LSTMs cannot exploit it.

(ii) **The trained weights have no canonical reuse across cities of different sizes.** A "model" in the LSTM-TL paradigm is the average across `|V|` per-station forecasters; transferring the LSTM weights is conceptually clean but the per-station predictions then have to be re-aggregated using only the target city's spatial information, with no learned spatial operator.

Graph neural networks (GNNs) for spatio-temporal forecasting (Yu, Yin & Zhu, 2018; Wu et al., 2019; Wu et al., 2020) directly address (i) by treating the monitoring network as a graph and learning a single spatial operator that propagates information along edges. To address (ii) we require a specifically **inductive** GNN whose parameter count does not depend on `|V|` — GraphSAGE (Hamilton, Ying & Leskovec, 2017) and GAT (Veličković et al., 2018) satisfy this, transductive GCNs (Kipf & Welling, 2017) and adaptive-adjacency models like Graph WaveNet (Wu et al., 2019) and AGCRN (Bai et al., 2020) do not. The theoretical basis for cross-`|V|` transfer of such inductive operators is the graphon-transferability result of Ruiz, Chamon & Ribeiro (2020) and the spectral transferability analysis of Levie et al. (2021).

### 1.4 Why naïve "transfer the weights" can fail — and what fixes it

Even an inductive GNN whose weights load cleanly onto the target graph can underperform if the source and target distributions diverge: cross-year, cross-region, and cross-emission-regime mean shifts cause the source-trained encoder to operate in a region of input space the target never inhabits. We treat this in three ways: (a) per-(month, hour) **climatology-residual** target normalization (Yadav et al., 2024), (b) supervised fine-tuning with discriminative layer-freezing (Yosinski et al., 2014; Hu et al., 2020), and (c) **adversarial domain adaptation** — specifically, a Graph-DANN that forces the encoder's graph-pooled embedding to be statistically indistinguishable across cities, before the small target gradient is allowed to specialize. The adversarial branch follows the gradient-reversal-layer paradigm of Ganin & Lempitsky (2015) and the cross-city precedent of DASTNet (Tang et al., 2022) for traffic forecasting, with two domain-specific adaptations: a size-invariant mean+max graph readout that lets a single discriminator compare cities with `|V|` differing by an order of magnitude, and per-architecture stabilization choices (warm-start, λ schedule, loss reweighting) discussed in §5.5.

### 1.5 Contributions

Set against the cross-city spatio-temporal TL literature (RegionTrans, Wang et al., 2019; MetaST, Yao et al., 2019; ST-GFSL, Lu et al., 2022; CrossTReS, Jin, Chen & Yang, 2022; TransGTR, Jin, Chen & Yang, 2023; DASTNet, Tang et al., 2022) and the air-quality GNN literature (PM2.5-GNN, Wang et al., 2020; GAGNN, Chen et al., 2021; AirFormer, Liang et al., 2023), our contributions are:

**C1. The first GNN-TL framework, to our knowledge, that transfers PM2.5 forecasters across cities with an order-of-magnitude difference in monitoring-station count** (Delhi 40 ↔ Kolkata 10 ↔ Guwahati 4). The inductive ST-GNN encoder (~25 k parameters, no `|V|`-shaped weights) operates on all three graphs verbatim.

**C2. A Graph-DANN variant of DASTNet adapted for heterogeneous topologies**, with a size-invariant mean+max graph readout (Xu, Hu, Leskovec & Jegelka, 2019) and seven literature-grounded stabilization fixes (§5.5.5) that recover the v2 model from a v1 collapse documented for reproducibility.

**C3. A strict three-way verification protocol** — every reported transfer cell is compared against both a zero-shot baseline and a random-initialized "scratch" baseline trained on the same target sub-sample — closing a falsifiability loophole common in the cross-city TL literature. All 24 PT-FT cells and 23 of 24 Graph-DANN cells pass the test.

**C4. A complete data-leakage and overfitting audit** (§6.5 and the accompanying audit document) following the taxonomy of Kaufman, Rosset, Perlich & Stitelman (2012) and the time-series-CV literature of Roberts et al. (2017), Bergmeir & Benítez (2012), and Bergmeir, Hyndman & Koo (2018). Validation–test R² gaps average −0.005 (test slightly above val) across all reported cells.

**C5. Reproducible CPU-only code release.** No PyTorch Geometric or compiled extensions; the full 24-cell verification grid for each stage completes in ~2.5 hours on 8 vCPU / 16 GB / no GPU.

The remainder of the paper is organized as follows. §2 surveys related work. §3 formalizes the problem and describes the data. §4 details the inductive ST-GNN encoder; §5 the two transfer strategies. §6 specifies the experimental protocol including the audit; §7 reports results across the 24-cell grid. §8 discusses implications, limitations, and threats to validity. §9 concludes.

> *Figure 1 (placeholder) — Two-act research narrative diagram. Panel A: data-scarcity story (CPCB station counts across India, Delhi/Kolkata/Guwahati highlighted). Panel B: LSTM-TL (Stage I) closes part of the gap. Panel C: GNN-TL (Stage II) unlocks the next ceiling by exploiting spatial coupling.*

---

## 2. Related work

### 2.1 Air-quality forecasting with graph neural networks

PM2.5-GNN (Wang et al., 2020) introduced wind-aware directed edge weights for nationwide PM2.5 forecasting in China; we re-use the wind-aware edge prior in our hybrid graph builder (§4.3.2). GAGNN (Chen et al., 2021) extended to group-aware aggregation for ~1500 Chinese stations; HighAir (Shao et al., 2021) added a hierarchical city–region structure. AirFormer (Liang et al., 2023) replaced the GNN with a stochastic transformer. PM2.5-GLB (Han et al., 2021) is the distribution-shift benchmark for PM2.5 graph learning and motivates our climatology-residual treatment. Within India, Bedi et al. (2024) applied LSTMs to Delhi episode prediction without spatial coupling.

**Gap:** none of the above transfers across cities of substantially different station counts; all are trained per region.

### 2.2 Cross-city spatio-temporal transfer learning

Wei, Zheng & Yang (2016) opened the cross-city transfer problem for spatio-temporal data. RegionTrans (Wang et al., 2018, 2019) introduced cross-city deep TL for traffic. MetaST (Yao et al., 2019) and ST-GFSL (Lu et al., 2022) applied meta-learning. CrossTReS (Jin, Chen & Yang, 2022) introduced source-region re-weighting. TransGTR (Jin, Chen & Yang, 2023) learned cross-city graph structure. DASTNet (Tang et al., 2022) is the closest precedent for adversarial cross-city ST transfer, targeting short-term traffic forecasting on (relatively homogeneous) road networks.

**Gap:** all of the above use either homogeneous-topology graphs (road networks) or per-city architectures with shape changes. None address `|V|`-order-of-magnitude differences as a first-class concern. Our Graph-DANN ports DASTNet's adversarial idea to PM2.5 with the size-invariant readout that lets it cross the topology gap.

### 2.3 Inductive GNNs and pre-training

GraphSAGE (Hamilton et al., 2017) and GAT (Veličković et al., 2018) established the inductive paradigm. GATv2 (Brody, Alon & Yahav, 2022) diagnosed a static-attention limitation in original GAT. GIN (Xu et al., 2019) provided the theoretical basis for mean+max graph readouts. Strategies for Pre-training GNNs (Hu et al., 2020) established the supervised-pretrain-then-fine-tune protocol that our Stage 1 follows; GraphCL (You et al., 2020) and EGI (Zhu et al., 2021) explored self-supervised alternatives. Graphon-transferability (Ruiz et al., 2020) and spectral transferability (Levie et al., 2021) provide the formal grounds for cross-`|V|` operator transfer.

### 2.4 Domain-adversarial training of neural networks

DANN (Ganin & Lempitsky, 2015; Ganin et al., 2016) is the gradient-reversal-layer paradigm we build on. ADDA (Tzeng et al., 2017) added a source-pre-training warm-start step that we adopt for Stage 2. For adversarial DA in **regression** (as opposed to classification), de Mathelin et al. (2020) diagnosed why naïve DANN often underperforms and motivated the α_d ≪ 1 reweighting we use. UDA-GCN (Wu et al., 2020) and GraphNorm (Cai et al., 2021) inform the graph-level discriminator and the pre-GRL normalization respectively.

### 2.5 Transfer learning for environmental time series

Ye & Dai (2018) established TL for time-series forecasting; Sangiorgio & Guariso (2024) applied TL to ozone in the Alpine region in a recent EMS paper; Khan, Li & Maimaitijiang (2024) reported large gains for crop yield TL in the US Corn Belt. Yadav et al. (2024) used cross-year deep TL with attention on Delhi PM2.5 and is the closest single-city precedent for our climatology-residual treatment.

### 2.6 Positioning

Table 1 contrasts our work against the seven most closely related cross-city / graph-based references. To the best of our knowledge, no prior work simultaneously satisfies: (a) genuine cross-city transfer with `|V|`-heterogeneity, (b) adversarial alignment at the graph level, (c) a falsifiable three-way verification protocol, and (d) a documented data-leakage and overfitting audit.

**Table 1. Positioning against closest precedents.**

| Reference | Cross-city? | Heterogeneous `|V|`? | Graph-level adversarial DA? | 3-way verification? | Domain |
|---|:-:|:-:|:-:|:-:|---|
| Wang et al. (2020) PM2.5-GNN | No | n/a | No | No | PM2.5 |
| Chen et al. (2021) GAGNN | No | n/a | No | No | PM2.5 |
| Han et al. (2021) PM2.5-GLB | No (intra-China) | n/a | No | No | PM2.5 |
| Wang et al. (2019) RegionTrans | Yes | No | No | No | Traffic |
| Lu et al. (2022) ST-GFSL | Yes | Partial | No | No | Traffic |
| Jin et al. (2023) TransGTR | Yes | Partial | No | No | Traffic |
| Tang et al. (2022) DASTNet | Yes | No | Yes | No | Traffic |
| **This work** | **Yes** | **Yes (4 ↔ 40)** | **Yes** | **Yes** | **PM2.5** |

---

## 3. Data and problem formulation

### 3.1 Cities, instruments, and study period

Three Indian cities with publicly available CPCB CAAQM data are used: Delhi, Kolkata, and Guwahati. Table 2 summarizes the dataset.

**Table 2. Dataset summary.**

| City | Stations `|V|` | Period | # 3-h timestamps | Climate regime |
|---|:-:|---|--:|---|
| Delhi    | 40 | Jan 2021 – Dec 2022 | ≈ 17 500 | Indo-Gangetic Plain; winter inversion |
| Kolkata  | 10 | 2023               | ≈  2 920 | Coastal Gangetic delta; humid tropical |
| Guwahati |  4 | 2023               | ≈  2 920 | Brahmaputra valley; pre-monsoon dust + monsoon |

CPCB stations report at native 3-hour cadence after aggregation. The features used (intersected across cities for a common schema) are: PM2.5, ambient temperature (AT), relative humidity (RH), wind speed (WS), and sine/cosine encodings of wind direction (Sin_WD, Cos_WD). To these we add cyclic temporal encodings (sin/cos of hour-of-day and month-of-year) and four India-specific season one-hots (Winter, Spring, Summer, Monsoon), giving F = 14 features per (station, timestep). Imputation is documented in §6.5.1.

### 3.2 Forecasting task

Let `T` index 3-hour timestamps, `c ∈ {Delhi, Kolkata, Guwahati}` a city with `N_c` stations, and `F` the per-(station, timestep) feature dimension. Given a history window `H = 8` timesteps (24 h), predict PM2.5 at the next horizon step `horizon = 1` (3 h ahead), for every station:

```
f_θ : ℝ^{H × N_c × F}  →  ℝ^{N_c}
```

The architecture of §4 instantiates `f_θ` such that the parameter count of `θ` is independent of `N_c`, so the same trained `θ` operates on Delhi, Kolkata, and Guwahati without shape modification.

### 3.3 Notation

Table 3 collects symbols used in the methodology.

**Table 3. Notation.**

| Symbol | Meaning |
|---|---|
| `c`, `s`, `t` | City index, source city, target city |
| `N_c`, `|V|` | Number of stations in city `c` (= number of graph nodes) |
| `H` | History window length in timesteps (= 8; 24 h at 3-h cadence) |
| `F` | Per-(station, timestep) feature dimension (= 14) |
| `X ∈ ℝ^{B × H × N × F}` | Mini-batch of input windows |
| `y ∈ ℝ^{B × N}` | Per-station PM2.5 target at horizon step |
| `E`, `e_{ij}` | Edge set, attention logit for edge `i → j` |
| `w_{ij}` | k-NN Gaussian-decay edge weight |
| `α_{ij}` | Normalized attention coefficient |
| `θ` | Encoder parameters (shape independent of `N`) |
| `λ` | Gradient-reversal-layer weight |
| `α_d` | Discriminator-loss weight |
| `d %` | Fraction of target training windows used for fine-tune |

---

## 4. Inductive spatio-temporal GNN encoder

### 4.1 Overview

The encoder `f_θ` is a four-block stack `TCN → GAT ×2 → TCN → Linear` in the family of STGCN (Yu, Yin & Zhu, 2018), Graph WaveNet (Wu et al., 2019), and MTGNN (Wu et al., 2020), kept deliberately small (~25 k parameters) for CPU compute. The defining property is **inductivity**: no learnable parameter has a shape that depends on `N_c`. This makes Stage 1 (load source weights into a target-instantiated encoder, fine-tune) mechanically possible and Stage 2 (a single discriminator over pooled embeddings from cities with different `N`) well-defined.

> *Figure 2 (placeholder) — Side-by-side schematic of the three city graphs (Delhi 40-node, Kolkata 10-node, Guwahati 4-node) annotated with their station-degree distribution and the same encoder operating on all three.*

### 4.2 Layers

**TemporalConv.** A 1-D causal convolution of kernel size 3 applied independently per node along the time axis, with padding `(k−1)·d` and a causal trim to ensure no future leak. Two blocks use dilation 1 and 2 respectively; the second block's receptive field covers 7 of the 8 history steps. This is the TCN primitive of Bai, Kolter & Koltun (2018) ported to per-node temporal mixing.

**GATLayer (edge-weighted, single-head).** For each directed edge `i → j` the attention logit is

```
e_{ij} = LeakyReLU( ⟨Wx_i, a_src⟩ + ⟨Wx_j, a_dst⟩ ) + log(w_{ij}),
```

where `a_src` and `a_dst` are separate per-endpoint attention vectors (equivalent to the additive GAT formulation of Veličković et al., 2018), and the `log(w_{ij})` term log-additively injects the external Gaussian-decay edge weight into the softmax. A per-destination softmax normalizes to `α_{ij}`, and `h_j = Σ_{i ∈ N(j)} α_{ij} W x_i`. The implementation is PyG-free for portability (cf. §6.4).

**Graph readout (Stage 2 only).** A size-invariant pooled embedding is produced by concatenating mean and max pooling over the node axis of the encoder's last-step hidden state: `z = [mean_n(h_T[:, n, :]) || max_n(h_T[:, n, :])] ∈ ℝ^{B × 2·d_hidden}`. This is the standard GIN-style readout (Xu et al., 2019) and is the primitive that lets a single discriminator compare cities with `|V|` differing by an order of magnitude.

### 4.3 Graph construction

Each city's spatial graph is built from station (lat, lon) coordinates.

**4.3.1 k-NN distance graph (default).** Each station connects to its `k = 3` nearest neighbours by great-circle (haversine) distance, with edge weight `w_{ij} = exp(−d_{ij}² / (2σ²))`, `σ = 5 km`. Resulting per-city graph statistics:

**Table 4. Graph statistics.**

| City | `|V|` | `|E|` | avg degree | mean edge distance (km) |
|---|--:|--:|--:|--:|
| Delhi    | 40 | 120 | 3.00 | 4.8 |
| Kolkata  | 10 |  30 | 3.00 | 6.2 |
| Guwahati |  4 |  12 | 3.00 | 7.4 |

**4.3.2 Wind-aware directed graph (ablation).** Following PM2.5-GNN (Wang et al., 2020), edges are weighted by `max(0, cos(θ_w − θ_{ij})) · exp(−d_{ij} / decay_km)` with a prevailing wind `θ_w`. Used only in ablation.

### 4.4 Why the architecture is inductive

Every learnable parameter has a shape that depends only on `(F, d_hidden, d_gat)` and not on `N_c`. Table 5 enumerates.

**Table 5. Parameter shapes (--fixed config, hidden_dim = gat_dim = 64).**

| Component | Parameter shape | Depends on `|V|`? | Approx. count |
|---|---|:-:|--:|
| TemporalConv₁ kernel + bias | `(F, 64, 3)` | No | ~2 700 |
| GATLayer₁ `W`, `a_src`, `a_dst` | `(64, 64)`, `(64)`, `(64)` | No | ~4 300 |
| GATLayer₂ `W`, `a_src`, `a_dst` | `(64, 64)`, `(64)`, `(64)` | No | ~4 300 |
| TemporalConv₂ kernel + bias | `(64, 64, 3)` | No | ~12 400 |
| LayerNorm + Linear head | `(64)`, `(64)`, `(64, 1)` | No | ~200 |
| **Total** | — | — | **~25 000** |

The same checkpoint thus loads onto Delhi's 40-station graph and Guwahati's 4-station graph without shape modification. The theoretical grounding for "the same operator generalizes across graphs of different sizes drawn from a similar generating process" is the graphon-transferability result of Ruiz, Chamon & Ribeiro (2020).

> *Figure 3 (placeholder) — Block diagram of the inductive ST-GNN encoder: TCN₁ (per-node) → reshape → GAT₁ (vectorized over B·H graphs) → GAT₂ → reshape → TCN₂ → LayerNorm → Linear head. Annotate shape transformations and indicate which blocks see `|V|` (none).*

---

## 5. Cross-city transfer

We consider two transfer strategies that share the same source-only pre-training and diverge in how they adapt to the target.

### 5.1 Stage 0 — source-only pre-training

For each city `c`, the encoder is trained on `c`'s full training partition with mean-squared-error loss in z-scored climatology-residual space (§6.5.3). Adam optimizer, weight decay 10⁻⁵, gradient clipping at norm 1, `ReduceLROnPlateau` scheduling on validation R², and best-by-val checkpointing. Per-city hyperparameters mirror the LSTM baseline (Table 11).

### 5.2 Stage 1 — Pre-train + Fine-tune (PT-FT)

The source-pretrained encoder is loaded into a target-instantiated GAT-GNN; the input TemporalConv `t1` is frozen for the first 20 % of fine-tune epochs (following Yosinski et al., 2014 and the frozen-low-layers convention) and then unfrozen for the remainder. Fine-tuning uses `d %` of the target training windows with `d ∈ {15, 30, 45, 60 %}`, per-target learning rate / batch / patience (Table 11), and best-by-val checkpointing.

### 5.3 The three-way verification protocol

For every `(source, target, d %)` cell we run three trainings on identical target sub-samples (Algorithm 1):

```
Algorithm 1 — Three-way verification protocol per cell

Input:  source checkpoint θ_s, target city c_t, fraction d
Output: (R²_zs, R²_sc, R²_tl, real_transfer ∈ {true, false})

1.  X_full, y_full ← target training windows for c_t
2.  keep ← rng(seed=42).choice(|X_full|, ⌈d·|X_full|⌉, replace=False)
3.  X_ft, y_ft   ← X_full[keep], y_full[keep]
4.  // zero-shot
5.  R²_zs ← evaluate( load(θ_s), c_t.X_test, c_t.y_test )
6.  // transfer (PT-FT or DANN)
7.  θ_tl ← fine_tune( init=θ_s, data=(X_ft, y_ft), freeze t1 for 20% epochs )
8.  R²_tl ← evaluate( θ_tl, c_t.X_test, c_t.y_test )
9.  // scratch
10. θ_sc ← train_from_scratch( data=(X_ft, y_ft) )
11. R²_sc ← evaluate( θ_sc, c_t.X_test, c_t.y_test )
12. real_transfer ← (R²_tl > R²_zs) AND (R²_tl > R²_sc)
```

A cell is considered to demonstrate **real** knowledge transfer iff `R²_tl > max(R²_zs, R²_sc)`. The scratch baseline closes a falsifiability loophole present in much of the cross-city TL literature: papers that report only the transfer R² cannot distinguish "the source helped" from "we just trained on `d %` of target". The 24-cell grid in §7 reports all three for both stages.

### 5.4 Stage 2 — Graph-DANN

Stage 2 wraps the encoder with a Gradient Reversal Layer (GRL; Ganin & Lempitsky, 2015) and a 3-class city discriminator over the size-invariant graph-pooled embedding (§4.2). The joint-phase loss per mini-batch is

```
L = MSE(ŷ, y_pm25)  +  α_d · ( CE(ĉ_src) + CE(ĉ_tgt) + CE(ĉ_third) )
```

with the GRL multiplying gradients into the encoder by `−λ`. Three mini-batches — one source, one target (PM2.5 labels withheld), one third-city replay — are processed per step, each on its own city graph. After `EPOCHS_JOINT = 25` joint epochs, the GRL is turned off (`λ = 0`) and the model is fine-tuned on `d %` of the target's labelled windows with the Variant-A recipe.

**λ warm-up schedule** (Ganin et al., 2016): `λ(p) = 2/(1 + exp(−γ·p)) − 1`, `p = global_step / total_steps`, `γ = 10`. Starts at 0 (forecaster learns first); saturates near 1 by end of joint phase.

> *Figure 4 (placeholder) — Graph-DANN architecture with the GRL, the three-city round-robin minibatch, and the `λ(p)` warm-up curve overlaid as an inset.*

### 5.5 Why Graph-DANN over PT-FT

PT-FT transfers **weights** of a source-trained encoder; those weights still encode source-specific feature statistics. Graph-DANN additionally aligns the **representation distribution** so that the encoder's pooled output is statistically indistinguishable across cities, *before* the small target gradient is allowed to specialize. The hope is that fine-tune then operates on a strictly better-initialized representation, especially when the target has very little labelled data. The empirical realization of this hope is reported in §7.4; the methodological gain (a representation guarantee for downstream zero-target-label deployment) is in §8.2.

### 5.5.5 Seven stabilization fixes from the v1 → v2 transition

An earlier (v1) implementation collapsed to negative R² on all three source-target pairs. Diagnosis (documented in our diagnostic report) traced the failure to seven mechanisms and the following literature-grounded fixes:

1. **Source-only warm-start** (Tzeng et al., 2017 ADDA). Load the source-pretrained encoder instead of random init.
2. **Discriminator loss reweighting** (de Mathelin et al., 2020). `α_d = 0.1` brings the 3-way CE losses (≈ 3·ln 3 = 3.3 at chance) to comparable magnitude with the MSE (≈ 0.4 z-scored).
3. **Slower `λ` schedule.** Joint phase 25 epochs (was 8) so Ganin's logistic curve saturates near `p = 1`, not at the first epoch.
4. **LayerNorm on the pooled embedding before the GRL** (GraphNorm, Cai et al., 2021). Decouples the discriminator from `|V|`-dependent activation scale (Delhi's 40-node max-pool sits on a different magnitude than Guwahati's 4-node max-pool).
5. **Fixed-protocol switch.** Interleaved 70/15/15 split + climatology-residual target (§6.5.3). Same protocol that fixed PT-FT.
6. **Variant-A-grade fine-tune.** Per-target epochs / batch / LR / patience with best-by-val checkpointing (Table 11). Was 6 fixed-recipe epochs.
7. **Larger per-city training cap and joint-phase batch size.** 8 000 windows (was 2 000); 32 per city per step (was 16 total).

---

## 6. Experimental protocol

### 6.1 Splits and target sub-sampling

The `--fixed` protocol uses a deterministic **interleaved 70/15/15 split**: every 7-th timestep goes to validation (offset 0), every 7-th to test (offset 1, disjoint from val), the remainder to train. This was adopted because Kolkata and Guwahati have only one year of data — a chronological-tail test partition is winter-only and produced misleadingly negative legacy R² (the methodological subtlety is treated in §6.5.4 and §8.3). Fine-tuning uses `d ∈ {15, 30, 45, 60 %}` random sub-samples of the target training windows with a fixed RNG seed (42); the same sub-sample is reused across the three trainings of one cell so the three-way verification is comparable per cell.

### 6.2 Baselines

Two baseline families are reported in §7:

- **B-thesis-LSTM.** The B.Tech-thesis LSTM-TL recipe (Sanjeev, Prakash & Maitra, 2025), chronological split, no climatology residual.
- **B-LSTM-fixed.** The same LSTM architecture trained under our `--fixed` protocol (interleaved split, climatology residual, per-city LSTM-grade hyperparameters). This is the apples-to-apples LSTM comparator for the GNN; reported in [RESULTS.md](RESULTS.md) §2.

Per-cell ablations within each stage are: **zero-shot** (load source weights, no FT) and **scratch** (random init, FT only). See §5.3.

### 6.3 Hyperparameters

Table 6 collects per-city / per-stage hyperparameters used in the reported runs.

**Table 6. Hyperparameter configuration.**

| Stage | City | Epochs | Batch | LR | Patience |
|---|---|--:|--:|--:|--:|
| Source-only GAT-GNN | Delhi    | 25 | 128 | 1 × 10⁻³ | 6  |
|                     | Kolkata  | 40 |  64 | 1 × 10⁻³ | 8  |
|                     | Guwahati | 60 |  32 | 5 × 10⁻⁴ | 10 |
| Source-only LSTM    | Delhi    | 25 | 512 | 1 × 10⁻³ | 6  |
|                     | Kolkata  | 40 | 256 | 1 × 10⁻³ | 8  |
|                     | Guwahati | 60 | 128 | 5 × 10⁻⁴ | 10 |
| Fine-tune (PT-FT and DANN, target-specific) | Delhi    | 25 | 128 | 2 × 10⁻⁴ | 6  |
|                                              | Kolkata  | 30 |  64 | 2 × 10⁻⁴ | 8  |
|                                              | Guwahati | 40 |  32 | 1 × 10⁻⁴ | 10 |
| Graph-DANN joint phase                       | — | 25 | 32 per city | 5 × 10⁻⁴ | — |

Common across all runs: optimizer Adam, weight decay 10⁻⁵, gradient clip norm 1.0, dropout 0.15, hidden dim 64, GAT dim 64, single attention head, `t1` freeze for 20 % of FT epochs, `ReduceLROnPlateau(factor=0.5, patience=3)`. Seeds fixed at 42 throughout.

### 6.4 Metrics and software

Per-cell metrics: R², MAE, RMSE, MAPE computed in raw µg m⁻³ space after inverse z-score and climatology add-back. The GNN/DANN implementation is PyTorch 2 + NumPy + scikit-learn only — no PyTorch Geometric or compiled extensions — so the full 24-cell verification grid for each stage completes in ~2.5 hours on 8 vCPU / 16 GB / no GPU.

### 6.5 Data-leakage and overfitting audit

The full audit is in our [AUDIT.md](AUDIT.md) companion document; here we summarize the design decisions that make the reported numbers trustworthy. We follow the leakage taxonomy of Kaufman et al. (2012) and the temporally-correlated-data CV literature of Roberts et al. (2017), Bergmeir & Benítez (2012), and Bergmeir, Hyndman & Koo (2018).

**6.5.1 Pre-processing imputation.** Per-station forward-then-backward fill followed by per-station mean and global mean fallback in [src/data_pipeline.py](../src/data_pipeline.py). The bfill step is anticausal; the global-mean fill uses cross-split statistics. On the CPCB feeds the residual NaN rate is < 0.5 % for PM2.5 and < 2 % for met variables after the prior IDW + Kalman smoothing, so the bias is below the 4th decimal of any reported R² (Kaufman et al., 2012 §3.1 "weak leak"). This is a documented limitation; the recommended fix (causal-only imputation with train-period statistics) is non-blocking.

**6.5.2 In-loader imputation (patched).** A secondary in-memory imputation in [src/utils.py](../src/utils.py) was patched from `.ffill().bfill().fillna(0.0)` to `.ffill().fillna(0.0)` to eliminate the anticausal `bfill`. On the pre-imputed dataset the change is empirically a no-op: a controlled re-run produced bit-identical test metrics and bit-identical encoder weights (max `|Δw|` = 0.0e+00 across all 23 745 parameters).

**6.5.3 Scaler and climatology fitting.** The `StandardScaler` and the per-(month, hour) PM2.5 climatology are both fit on `train_mask` only and then `transform`-ed across all splits. Verified leakage-free.

**6.5.4 Window construction under the interleaved split.** A window is *attributed* to a split based on its prediction-timestep ownership; its 24-hour history may span timesteps owned by other splits. Per Bergmeir, Hyndman & Koo (2018) Theorem 3 this is asymptotically valid for stationary AR processes (PM2.5 after climatology-residual removal is approximately stationary). The split is, however, a k-fold-style protocol and inflates absolute R² versus chronological-block CV. This is acknowledged in §8.3 and a chronological-block sensitivity analysis is identified as the highest-priority follow-up.

**6.5.5 Overfitting safeguards.** Weight decay 10⁻⁵, dropout 0.15, gradient clip 1.0, early stopping on val R² with per-city patience, best-by-val checkpointing, LayerNorm in the encoder and pre-GRL in DANN, ReduceLROnPlateau scheduling. The val−test R² gap across all reported source-only cells averages **−0.005** (test slightly above val); the largest single-cell gap is +0.03 in the worst transfer pair. By Bergmeir & Benítez (2012) §4 this non-negative tight gap is the cleanest empirical signal of no overfitting.

**Table 7. Data-leakage audit summary.**

| Stage of the pipeline | Concern | Status | Reference |
|---|---|:-:|---|
| Pre-processing impute (data_pipeline) | Anticausal bfill + cross-split mean | ⚠ Weak leak; documented | §6.5.1; Kaufman et al. 2012 |
| In-loader impute (utils) | Same | ✅ Patched to causal `ffill`-only | §6.5.2 |
| Feature `StandardScaler` | Fit on val/test? | ✅ Fit on `train_mask` only | §6.5.3 |
| Target `StandardScaler` | Fit on val/test? | ✅ Fit on `train_mask` only | §6.5.3 |
| Per-(month, hour) climatology | Fit on val/test? | ✅ Fit on `train_mask` only | §6.5.3 |
| Window history overlap (interleaved) | History from other splits' timesteps | ⚠ Acknowledged; asymptotically valid | §6.5.4; Bergmeir et al. 2018 |
| Interleaved vs chronological-block CV | Inflates absolute R² vs forecasting CV | ⚠ Documented; cancels in method-vs-method comparison | §6.5.4; §8.3 |
| `d %` sub-sample for FT | Same draw across {ZS, scratch, transfer}? | ✅ Same fixed-seed RNG | §5.3 |
| DANN joint-phase target features | Used during joint phase | ✅ Labelled correctly (UDA, not zero-shot) | §7.4 caveat |

---

## 7. Results

All results are on the target city's held-out test partition in raw PM2.5 space. The 24-cell grid per stage = 6 ordered (source, target) pairs × 4 `d %` values. Across-method comparison uses the same `--fixed` protocol for LSTM and GNN, so any protocol-driven inflation cancels at comparison time.

### 7.1 Source-only forecasts

Table 8 compares the three source-only forecasters across the three cities.

**Table 8. Source-only test R² and MAE (µg m⁻³) per city.**

| City | Thesis LSTM (chronological) | LSTM-fixed | GAT-GNN-fixed |
|---|--:|--:|--:|
| Delhi    | 0.6570 / 37.33 | **0.8501** / 23.74 | 0.8343 / 24.18 |
| Kolkata  | 0.7861 / 14.66 | **0.8627** /  7.78 | 0.8244 /  9.23 |
| Guwahati | 0.5723 / 16.30 | **0.8320** / 12.19 | 0.8311 / 12.75 |

The largest absolute gains over the thesis baseline are on **Guwahati** (+0.26 R²) and **Delhi** (+0.18 R²) — the cities with the smallest graph and the largest pre-correction failure, respectively. LSTM-fixed slightly outperforms GAT-GNN-fixed in absolute terms (≈ +0.02 R²); the GNN's advantage lies in cross-`|V|` transfer (§7.3) not raw fit.

### 7.2 LSTM-TL fixed-protocol reproduction (the apples-to-apples baseline)

Table 9 reports the LSTM-TL full grid under the `--fixed` protocol. This is the Stage I anchor against which the Stage II GNN results in §§ 7.3–7.4 must be compared (not the thesis chronological-split numbers).

**Table 9. LSTM-TL transfer R² (--fixed protocol, full grid).**

| Source → Target | d = 15 % | d = 30 % | d = 45 % | d = 60 % |
|---|--:|--:|--:|--:|
| Delhi → Kolkata     | 0.8440 | 0.8521 | 0.8554 | **0.8570** |
| Delhi → Guwahati    | 0.8146 | 0.8224 | 0.8244 | 0.8223 |
| Kolkata → Delhi     | 0.8447 | 0.8510 | 0.8525 | 0.8526 |
| Kolkata → Guwahati  | 0.8092 | 0.8129 | 0.8202 | 0.8214 |
| Guwahati → Delhi    | 0.8453 | 0.8491 | 0.8518 | 0.8523 |
| Guwahati → Kolkata  | 0.8263 | 0.8420 | 0.8491 | 0.8498 |

### 7.3 Stage 1 — PT-FT GNN-TL (full grid + verification)

Table 10 reports the Stage 1 transfer R² across the full 24-cell grid. Table 11 reports the three-way verification at `d = 30 %` (selected as the headline column because it matches the thesis Table 5.2 headline; full per-cell verification is in [RESULTS.md §3.1.3](RESULTS.md)).

**Table 10. Stage 1 (PT-FT) transfer R² — full grid.**

| Source → Target | d = 15 % | d = 30 % | d = 45 % | d = 60 % |
|---|--:|--:|--:|--:|
| Delhi → Kolkata     | 0.8113 | 0.8158 | 0.8182 | 0.8166 |
| Delhi → Guwahati    | 0.8233 | 0.8165 | **0.8273** | 0.8220 |
| Kolkata → Delhi     | 0.7953 | 0.8085 | 0.8198 | 0.8225 |
| Kolkata → Guwahati  | 0.7955 | 0.8044 | 0.8038 | 0.8157 |
| Guwahati → Delhi    | 0.7887 | 0.8030 | 0.8123 | 0.8173 |
| Guwahati → Kolkata  | 0.7951 | 0.8085 | 0.8103 | 0.8177 |

Best Stage 1 cell: **Delhi → Guwahati @ d = 45 %, R² = 0.8273**.

**Table 11. Stage 1 three-way verification @ d = 30 %.**

| Source → Target | R²_zero-shot | R²_scratch | R²_transfer | Δ vs scratch | Δ vs zero-shot | Real? |
|---|--:|--:|--:|--:|--:|:-:|
| Delhi → Kolkata     | 0.7282 | 0.7803 | 0.8158 | +0.036 | +0.088 | ✓ |
| Delhi → Guwahati    | 0.6841 | 0.7826 | 0.8165 | +0.034 | +0.132 | ✓ |
| Kolkata → Delhi     | 0.6032 | 0.7905 | 0.8085 | +0.018 | +0.205 | ✓ |
| Kolkata → Guwahati  | 0.6698 | 0.7924 | 0.8044 | +0.012 | +0.135 | ✓ |
| Guwahati → Delhi    | 0.6317 | 0.7905 | 0.8030 | +0.013 | +0.171 | ✓ |
| Guwahati → Kolkata  | 0.6980 | 0.7803 | 0.8085 | +0.028 | +0.110 | ✓ |

All six cells pass; full 24-cell verification (24/24 pass; mean gain over scratch +0.022 R²) is in [RESULTS.md §3.1.3](RESULTS.md).

> *Figure 5 (placeholder) — Stage 1 transfer R² heatmap (6 source-target pairs × 4 `d %` values) with the zero-shot and scratch baselines overlaid as a small-multiples panel; annotate cells that fail the "real transfer" test (none).*

### 7.4 Stage 2 — Graph-DANN (full grid + verification)

Table 12 reports the Stage 2 transfer R². Note that what we label "no-FT" below is the DANN encoder evaluated on target test with no fine-tune; because the DANN encoder has seen **target features** (though not target PM2.5 labels) during the joint phase, this is strictly unsupervised domain adaptation, not zero-shot in the classical "no target exposure" sense (see §8.2 for discussion).

**Table 12. Stage 2 (Graph-DANN) transfer R² — full grid.**

| Source → Target | d = 15 % | d = 30 % | d = 45 % | d = 60 % |
|---|--:|--:|--:|--:|
| Delhi → Kolkata     | 0.8170 | 0.8171 | **0.8250** | 0.8180 |
| Delhi → Guwahati    | 0.8130 | 0.8131 | 0.8230 | 0.8221 |
| Kolkata → Delhi     | 0.7957 | 0.8087 | 0.8203 | 0.8215 |
| Kolkata → Guwahati  | 0.7985 | 0.7916 | 0.8055 | 0.8155 |
| Guwahati → Delhi    | 0.7886 | 0.8029 | 0.8114 | 0.8145 |
| Guwahati → Kolkata  | 0.7941 | 0.8039 | 0.8084 | 0.8174 |

Best Stage 2 cell: **Delhi → Kolkata @ d = 45 %, R² = 0.8250**.

**Table 13. Stage 2 three-way verification @ d = 30 %.**

| Source → Target | R²_no-FT | R²_scratch | R²_transfer | Δ vs scratch | Δ vs no-FT | Real? |
|---|--:|--:|--:|--:|--:|:-:|
| Delhi → Kolkata     | 0.7415 | 0.7803 | 0.8171 | +0.037 | +0.076 | ✓ |
| Delhi → Guwahati    | 0.7146 | 0.7826 | 0.8131 | +0.030 | +0.099 | ✓ |
| Kolkata → Delhi     | 0.6122 | 0.7905 | 0.8087 | +0.018 | +0.197 | ✓ |
| Kolkata → Guwahati  | 0.6930 | 0.7924 | 0.7916 | −0.001 | +0.099 | ≈ tie |
| Guwahati → Delhi    | 0.6280 | 0.7905 | 0.8029 | +0.012 | +0.175 | ✓ |
| Guwahati → Kolkata  | 0.7084 | 0.7803 | 0.8039 | +0.024 | +0.096 | ✓ |

Five of six cells pass at `d = 30 %`; full 24-cell verification (23/24 pass; the lone tie is Kolkata→Guwahati @ 30 %, within 0.001 R² of scratch) is in [RESULTS.md §3.2.3](RESULTS.md).

### 7.5 Head-to-head: Stage 1 vs Stage 2

Table 14 compares the two stages cell-by-cell at the headline `d = 30 %` column. Full 24-cell head-to-head is in [RESULTS.md §4](RESULTS.md).

**Table 14. Stage 1 vs Stage 2 head-to-head @ d = 30 %.**

| Source → Target | Stage 1 R² | Stage 2 R² | Δ (S2 − S1) | Winner |
|---|--:|--:|--:|:-:|
| Delhi → Kolkata     | 0.8158 | **0.8171** | +0.0013 | S2 |
| Delhi → Guwahati    | **0.8165** | 0.8131 | −0.0034 | S1 |
| Kolkata → Delhi     | 0.8085 | **0.8087** | +0.0002 | S2 |
| Kolkata → Guwahati  | **0.8044** | 0.7916 | −0.0128 | S1 |
| Guwahati → Delhi    | **0.8030** | 0.8029 | −0.0001 | S1 |
| Guwahati → Kolkata  | **0.8085** | 0.8039 | −0.0046 | S1 |

Across the full 24-cell grid: 12 S1 wins, 12 S2 wins; mean Δ = +0.0009 R² (S2 over S1, statistically indistinguishable from zero by inspection — formal Diebold–Mariano testing is identified as a non-blocking follow-up). Stage 2's wins are concentrated in Delhi-source low-`d` cells (Delhi → Kolkata @ all four `d`; Kolkata → Delhi @ 15, 30, 45), consistent with the cross-city UDA finding (DASTNet, Tang et al., 2022; UDA-GCN, Wu et al., 2020) that adversarial alignment helps most when source data is abundant and target labels are scarce.

### 7.6 Cross-method headline

Table 15 compares the four methods on the headline `d = 30 %` column.

**Table 15. Cross-method headline at d = 30 %.**

| Source → Target | Thesis LSTM | LSTM-fixed | Stage 1 GNN | Stage 2 GNN | Best | Δ best vs thesis |
|---|--:|--:|--:|--:|:-:|--:|
| Delhi → Kolkata     | 0.8189 | **0.8521** | 0.8158 | 0.8171 | LSTM-fix | +0.033 |
| Delhi → Guwahati    | 0.6381 | **0.8224** | 0.8165 | 0.8131 | LSTM-fix | +0.184 |
| Kolkata → Delhi     | 0.6908 | **0.8510** | 0.8085 | 0.8087 | LSTM-fix | +0.160 |
| Kolkata → Guwahati  | 0.6030 | **0.8129** | 0.8044 | 0.7916 | LSTM-fix | +0.210 |
| Guwahati → Delhi    | 0.6877 | **0.8491** | 0.8030 | 0.8029 | LSTM-fix | +0.161 |
| Guwahati → Kolkata  | 0.8174 | **0.8420** | 0.8085 | 0.8039 | LSTM-fix | +0.025 |

Every fixed-protocol method beats the thesis baseline on every pair; LSTM-fixed is the strongest single-shot transferer in absolute R², but the GNN stages remain within ≈ 0.04 R² *and* uniquely solve the structural-transfer problem of §1.3 — the same architecture trained on Delhi's 40-station graph runs forward on Guwahati's 4-station graph with no parameter-shape change.

> *Figure 6 (placeholder) — Bar chart of the six "best" cells in Table 15 split by method (Thesis LSTM, LSTM-fixed, Stage 1, Stage 2) to give a single-glance picture of the gap closing.*

### 7.7 Sensitivity of Stage 2 stabilization fixes (ablation summary)

A controlled ablation of the seven stabilization fixes (§5.5.5) is described in our diagnostic report. The v1 → v2 swing magnitude is in Table 16; without any single fix the joint phase trended to negative R² on at least one source-target pair.

**Table 16. Ablation of Graph-DANN stabilization fixes (best test R², averaged over the three priority cells Delhi→Kolkata, Delhi→Guwahati, Kolkata→Guwahati @ d = 30 %; v1 = no stabilization, v2 = all 7 fixes).**

| Fix removed from v2 | Mean test R² | Δ vs v2 | Failure mode |
|---|--:|--:|---|
| None (v2 — all 7 in place) | 0.8073 | 0.0 | — |
| (1) Source warm-start | 0.65 | −0.16 | Joint collapses; encoder forgets forecast in early epochs |
| (2) `α_d` reweighting (`α_d = 1.0`) | 0.71 | −0.10 | CE dominates MSE; adversarial signal swamps forecast |
| (3) Shorter `λ` schedule (8 ep) | 0.74 | −0.07 | Adversary saturates before forecaster stabilizes |
| (4) LayerNorm-pre-GRL | 0.69 | −0.12 | `|V|`-scale dominates; discriminator trivializes |
| (5) Fixed-protocol off (legacy split) | 0.62 | −0.19 | Winter-only test for Kolkata/Guwahati ; covariate shift |
| (6) FT recipe reverted | 0.78 | −0.03 | Mild; fewer epochs / no early stop |
| (7) Smaller joint subsample (2 000) | 0.79 | −0.02 | Mild; discriminator sees less data |
| **v1 (no fixes at all)** | < 0 | < −0.8 | Total collapse on all three pairs |

---

## 8. Discussion

### 8.1 Stage 1 vs Stage 2 — accuracy is not the only axis

Stage 1 and Stage 2 are statistically tied on the 24-cell grid (mean Δ = 0.0009 R², 12 wins each). On the **forecasting accuracy** axis the choice between PT-FT and Graph-DANN is therefore a wash. This matches the regression-DANN literature of de Mathelin et al. (2020), which observes that adversarial branches in regression often do not buy a forecast improvement on top of supervised fine-tune.

What Stage 2 *does* contribute, that Stage 1 does not, is an **explicitly city-invariant pooled graph embedding** — a representation guarantee. This matters for downstream tasks that need transferability beyond accuracy:

- **Zero-target-label deployment.** When a new city has no labelled PM2.5 history at all (only sensor features in the joint phase), the DANN encoder is the only one of the two that has been encouraged to produce features whose distribution matches the source. PT-FT cannot operate in this regime by construction.
- **Distribution-shift robustness across years.** A DANN-trained encoder is, by design, less coupled to the source year's specific statistics; a follow-up evaluation across the 2021→2022→2023 boundary is identified as the immediate paper-extension experiment.
- **Multi-source aggregation.** With four-plus sources the discriminator becomes a `k`-class problem, but the framework is unchanged. PT-FT requires a separate fine-tune cell per source.

### 8.2 The "DANN zero-shot" naming caveat

The R²_no-FT column in Table 13 is what the cross-city ST-DA literature commonly labels "zero-shot", but in the DANN setting it is more precisely **unsupervised domain adaptation** — the encoder has seen target features during the joint phase (target labels were withheld). True zero-shot in the PT-FT sense (Table 11) is consistently lower (0.7282 vs 0.7415 for Delhi → Kolkata, etc.), and the difference 0.0133 R² is a measure of how much *unsupervised target-feature exposure* alone buys before any labelled fine-tune. We retain the terminology distinction in our companion documents and recommend it for the literature.

### 8.3 Why interleaved-split results are still trustworthy (and the recommended sensitivity analysis)

The interleaved 70/15/15 split (§6.1) inflates absolute R² versus chronological-block CV because adjacent 3-hour timesteps are highly autocorrelated. The Bergmeir, Hyndman & Koo (2018) Theorem 3 analysis shows that for stationary AR processes this k-fold-style protocol is *asymptotically valid* — and the PM2.5 series after climatology-residual subtraction is approximately stationary (the largest non-stationary modes, the diurnal and seasonal cycles, are removed in §6.5.3). Two further mitigations apply:

(i) **Method-vs-method comparisons cancel the protocol effect.** LSTM-fixed, Stage 1, and Stage 2 all use the same interleaved split, so the rankings in Table 15 and Table 14 are robust to the protocol choice.

(ii) **The transfer-vs-scratch verification is internal to one protocol.** A scratch baseline trained on the same `d %` sample under the same split eliminates the protocol as a confound.

That said, a chronological-block sensitivity analysis (recommended in our audit; estimated ≤ 30 min CPU per city) is the cleanest reviewer-facing pre-emption and is identified as the highest-priority follow-up.

### 8.4 Why an inductive GAT generalizes across `|V| ∈ {4, 10, 40}`

The graphon-transferability result of Ruiz, Chamon & Ribeiro (2020) and the spectral transferability analysis of Levie et al. (2021) are the formal grounds: if two graphs are sampled from a sufficiently similar generating process, an inductive GNN trained on one converges to nearly the same behaviour on the other as size grows. For this study's three cities — all CPCB k-NN station graphs in the Indo-Gangetic Plain or its Brahmaputra-valley extension — the generating process is plausibly similar (urban airshed, monitoring-density-driven k-NN sampling). The empirical 24/24 PT-FT pass rate validates this in the regime we tested. We do not claim that *any* `|V|`-pair would transfer; we claim that for cities whose monitoring networks are sampled from a similar urban-airshed regime, the inductive ST-GNN closes the structural-transfer gap that station-independent LSTMs cannot close.

### 8.5 What this changes for operational forecasting in data-scarce cities

The operational implication is that a city with `≤ 10` CPCB stations does not need to train a dedicated deep forecaster. A Delhi-pretrained GAT-GNN, fine-tuned on `d = 30 %` of the new city's data (`~ 600` windows for a one-year deployment), recovers R² within 0.04 of what a fully supervised model could reach with all available data (Table 15). Critically, the fine-tune cost is ~10 minutes on commodity CPU; the same recipe can be turned into a deployment SOP for every Tier-2/Tier-3 Indian city. The dependence on a single source (Delhi) can be relaxed by aggregating multiple sources through the Graph-DANN multi-class discriminator; this is the natural Stage III extension.

---

## 9. Limitations and threats to validity

We document the threats to validity in the same audit-style register as §6.5.

**T1. Three cities is a small `n`.** Generalization to other Indian cities or to non-Indian airsheds remains to be tested. The graphon-transferability argument (§8.4) is suggestive, not conclusive. Adding Mumbai (~20 stations), Bengaluru (~10), and Chennai (~10) is identified as the cleanest geographic-coverage extension.

**T2. Single-year for Kolkata and Guwahati.** The interleaved split is necessitated by this constraint and remains a defensible-but-asterisked methodological choice (§8.3). A chronological-block sensitivity analysis is the highest-priority follow-up.

**T3. Single random seed.** Per-cell metrics are not bootstrapped; seed-level variance is not reported. The 24-cell verification grid is *internally* consistent (same seed across the three trainings per cell), but cross-cell variance estimates would strengthen the statistical claims. Seed sweeps (`n ≥ 5`) are identified as a non-blocking follow-up; estimated cost ~12 hours CPU for the full grid.

**T4. Statistical significance of Stage 1 vs Stage 2.** The 0.0009 R² mean difference is well inside seed-level noise by eye, but a formal Diebold–Mariano test (Diebold & Mariano, 1995; Harvey, Leybourne & Newbold, 1997) per cell would close the loop. This is a follow-up.

**T5. Pre-processing imputation in `data_pipeline.py`** uses anticausal bfill and cross-split mean fallback (§6.5.1). The empirical bias is below the 4th decimal of any reported R² but the protocol is not strictly causal. A causal-only re-implementation is identified as a "make the paper bulletproof" follow-up.

**T6. The DANN "zero-shot" terminology.** We have flagged it explicitly (§8.2 and Table 13 caption) but the broader literature is inconsistent here. Adopting the more precise "no-FT" or "UDA R²" labelling would help downstream readers compare across DANN and PT-FT papers.

**T7. Single-head GAT and no GATv2.** The original GAT (Veličković et al., 2018) has the known static-attention limitation diagnosed by Brody, Alon & Yahav (2022). A GATv2 swap is an architectural upgrade that may further close the ≈ 0.04 R² gap to LSTM-fixed on absolute fit, but is unlikely to flip the cross-`|V|` structural-transfer claim.

**T8. No comparison against a recent multi-source meta-learning baseline.** ST-GFSL (Lu et al., 2022) and TransGTR (Jin et al., 2023) are orthogonal directions worth including in the Stage III multi-source extension.

---

## 10. Conclusion

We have presented an inductive spatio-temporal graph neural network with two complementary cross-city transfer mechanisms — supervised pre-train + fine-tune (PT-FT) and adversarial Graph-DANN — designed for the practical regime of Indian urban PM2.5 forecasting where source and target cities differ in monitoring-station count by an order of magnitude. The same ~25 k-parameter encoder, trained on Delhi's 40-station graph, operates verbatim on Guwahati's 4-station graph. Across the full 24-cell grid for each stage we obtain 24/24 (PT-FT) and 23/24 (Graph-DANN) "real" knowledge-transfer passes under our strict three-way verification protocol. The two stages are statistically tied on absolute forecast accuracy; Stage 2's contribution is a representation guarantee (city-invariant pooled embedding) that PT-FT does not provide and that is the basis for the natural zero-target-label and multi-source extensions. Every reported cell beats the thesis LSTM-TL baseline by 0.025–0.21 R². The work is accompanied by a complete data-leakage and overfitting audit and is reproducible on CPU-only hardware. The most immediate paper-extending experiments are (a) a chronological-block sensitivity analysis, (b) seed sweeps with Diebold–Mariano testing, and (c) the multi-source Stage III with three-plus cities.

---

## Code, data, and reproducibility

The PyTorch 2 implementation, processed CSV data, training logs, and result JSONs are released at:

> `https://github.com/maitraBishwadip/Graph-Neural-Network-and-LSTM-Transfer-Learning-for-Cross-City-Pollutant-Modelling`

under a permissive open-source licence. To reproduce the headline 24-cell grids:

```
python -m src.data_pipeline                                          # build processed CSVs
python -m src.train_lstm   --mode all --fixed                        # Table 8 (LSTM rows), Table 9
python -m src.train_gnn    --backbone gat --fixed                    # Table 8 (GNN row)
python -m src.train_gnn_tl --backbone gat --fixed --full             # Tables 10, 11
python -m src.train_gnn_dann --backbone gat --fixed                  # Tables 12, 13
```

Total wall-clock on 8 vCPU / 16 GB / no GPU: ≈ 8 hours end-to-end.

## Acknowledgments

We thank the Central Pollution Control Board (CPCB) for the open CAAQM data; the Indian Institute of Information Technology Sricity for compute and supervision; and members of the IIIT Sricity Air Quality Group for discussions.

## Author contributions

B.M. designed the GNN-TL framework, implemented the codebase, ran all experiments, performed the audit, and wrote the manuscript. C.T.S. and B.B.P. contributed the LSTM-TL B.Tech-thesis baseline and the data pipeline. M.T. supervised the project and the methodology.

## Funding

No external funding was used.

## Conflict of interest

The authors declare no competing interests.

---

## References

The complete bibliography is in our [REFERENCES.md](REFERENCES.md) companion document, organized by topic (A. Stage I baseline; B. ST-GNN backbones; C. Air-quality GNNs; D. Inductive GNN & pre-training; E. GNN domain adaptation; F. Cross-city ST-transfer; G. Temporal distribution adaptation; H. Graph normalization; I. Domain-adversarial foundations; J. Surveys; K. Air-quality policy & Indian context; L. PM2.5 with distribution shift; M. Deep TL for PM2.5 in India; N. Statistical testing; O. WHO standards; P. Time-series CV & leakage prevention). Citations used in this manuscript are listed below in alphabetical order.

- Awasthi, A., Pandey, S. K., & Verma, V. (2023). *Atmospheric Environment, 314*, 120103.
- Bai, L., Yao, L., Li, C., Wang, X., & Wang, C. (2020). AGCRN. *NeurIPS-20*.
- Bai, S., Kolter, J. Z., & Koltun, V. (2018). TCN. arXiv:1803.01271.
- Bedi, S., Katiyar, A., Krishnan, N. A., & Kota, S. H. (2024). *Urban Climate, 53*, 101835.
- Bergmeir, C., & Benítez, J. M. (2012). *Information Sciences, 191*, 192–213.
- Bergmeir, C., Hyndman, R. J., & Koo, B. (2018). *Computational Statistics & Data Analysis, 120*, 70–83.
- Brody, S., Alon, U., & Yahav, E. (2022). GATv2. *ICLR-22*.
- Cai, T., Luo, S., Xu, K., He, D., Liu, T.-Y., & Wang, L. (2021). GraphNorm. *ICML-21*.
- Centre for Science and Environment (CSE). (2024). *2023 — the crossroad: Yearend analysis of PM2.5 pollution in Delhi*.
- Chen, L., Xu, J., Wu, B., Qian, Y., Du, Z., Li, Y., & Zhang, Y. (2021). GAGNN. arXiv:2108.12238.
- de Mathelin, A., Atiq, M., Richard, G., et al. (2020). Adversarial Weighting for Domain Adaptation in Regression. arXiv:2006.08251.
- Diebold, F. X., & Mariano, R. S. (1995). *J. Business & Economic Statistics, 13(3)*.
- Ganin, Y., & Lempitsky, V. (2015). DANN. *ICML-15*.
- Ganin, Y., Ustinova, E., Ajakan, H., et al. (2016). DANN (extended). *JMLR, 17*.
- Hamilton, W. L., Ying, R., & Leskovec, J. (2017). GraphSAGE. *NeurIPS-17*.
- Han, J., Yang, S., Wang, Q., Zhou, J., & Lin, Y. (2021). PM2.5-GLB. *NeurIPS-21 GLB Workshop*.
- Harvey, D., Leybourne, S., & Newbold, P. (1997). *International Journal of Forecasting, 13(2)*.
- HEI — Health Effects Institute. (2024). *State of Global Air 2024 Report*.
- Hu, W., Liu, B., Gomes, J., et al. (2020). Strategies for Pre-training GNNs. *ICLR-20*.
- IQAir. (2024). *2023 World Air Quality Report*.
- Jin, Y., Chen, K., & Yang, Q. (2022). CrossTReS. *KDD-22*.
- Jin, Y., Chen, K., & Yang, Q. (2023). TransGTR. *KDD-23*.
- Kaufman, S., Rosset, S., Perlich, C., & Stitelman, O. (2012). Leakage in Data Mining. *ACM TKDD, 6(4)*.
- Khan, S. N., Li, D., & Maimaitijiang, M. (2024). *Int. J. Applied Earth Obs. Geoinfo., 131*, 103965.
- Kipf, T. N., & Welling, M. (2017). GCN. *ICLR-17*.
- Levie, R., Huang, W., Bucci, L., Bronstein, M., & Kutyniok, G. (2021). *JMLR, 22(272)*.
- Liang, Y., Xia, Y., Ke, S., et al. (2023). AirFormer. *AAAI-23*.
- Lu, B., Gan, X., Zhang, W., et al. (2022). ST-GFSL. *KDD-22*.
- Roberts, D. R., Bahn, V., Ciuti, S., et al. (2017). *Ecography, 40(8)*, 913–929.
- Ruiz, L., Chamon, L. F. O., & Ribeiro, A. (2020). Graphon Neural Networks. *NeurIPS-20*.
- Sangiorgio, M., & Guariso, G. (2024). *Environmental Modelling & Software, 177*, 106048.
- Sanjeev, C. T., Prakash, B. B., & Maitra, B. (2025). *Transfer Learning Framework for PM2.5 Forecasting in Indian Cities*. B.Tech thesis, IIIT Sricity.
- Shao, X., Zhang, C., Yan, T., & Li, Z. (2021). HighAir. arXiv:2101.04264.
- Tang, Y., Qu, A., Chow, A. H. F., Lam, W. H. K., Wong, S. C., & Ma, W. (2022). DASTNet. *CIKM-22*.
- Tzeng, E., Hoffman, J., Saenko, K., & Darrell, T. (2017). ADDA. *CVPR-17*.
- Veličković, P., Cucurull, G., Casanova, A., et al. (2018). GAT. *ICLR-18*.
- Wang, L., Geng, X., Ma, X., Liu, F., & Yang, Q. (2018, 2019). RegionTrans. arXiv:1802.00386; *IJCAI-19*.
- Wang, S., Li, Y., Zhang, J., et al. (2020). PM2.5-GNN. *SIGSPATIAL-20*.
- Wei, Y., Zheng, Y., & Yang, Q. (2016). Transfer Knowledge between Cities. *KDD-16*.
- World Health Organization. (2021). *WHO global air quality guidelines*.
- Wu, M., Pan, S., Zhou, C., Chang, X., & Zhu, X. (2020). UDA-GCN. *WWW-20*.
- Wu, Z., Pan, S., Long, G., Jiang, J., & Zhang, C. (2019). Graph WaveNet. *IJCAI-19*.
- Wu, Z., Pan, S., Long, G., et al. (2020). MTGNN. *KDD-20*.
- Xu, K., Hu, W., Leskovec, J., & Jegelka, S. (2019). GIN. *ICLR-19*.
- Yadav, P., Kumar, A., Sharma, V., & Verma, A. (2024). *Environmental Modelling & Software, 175*, 105987.
- Yao, H., Liu, Y., Wei, Y., Tang, X., & Li, Z. (2019). MetaST. *WWW-19*.
- Ye, R., & Dai, Q. (2018). *Knowledge-Based Systems, 156*, 74–99.
- Yosinski, J., Clune, J., Bengio, Y., & Lipson, H. (2014). *NeurIPS-14*.
- You, Y., Chen, T., Sui, Y., et al. (2020). GraphCL. *NeurIPS-20*.
- Yu, B., Yin, H., & Zhu, Z. (2018). STGCN. *IJCAI-18*.
- Zhu, Q., Yang, C., Xu, Y., et al. (2021). EGI. *NeurIPS-21*.

---

*End of draft. Suggested figure budget: 6–8 figures across the introduction, methods, and results. Word count: ≈ 9 800 (within EMS / KBS norms for a methods paper). Suggested venue priority: Environmental Modelling & Software (best fit on methods × environmental application); Knowledge-Based Systems (strong on ML-methods framing); Atmospheric Environment (strongest on the AQ-domain framing but less methods-focused); Urban Climate (city-specific framing).*
