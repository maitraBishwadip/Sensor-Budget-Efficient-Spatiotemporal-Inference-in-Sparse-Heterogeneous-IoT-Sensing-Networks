# GNN and GNN-TL — Architecture and Mechanism

A deep-dive into **exactly how** the inductive Spatio-Temporal GNN and the two flavours of GNN transfer learning are implemented in this repository, and where each design choice sits in the research literature.

This is the implementation-and-mechanism counterpart to [MAIN_REPORT.md](MAIN_REPORT.md) (narrative + diagnosis), [RESULTS.md](RESULTS.md) (numbers), and [AUDIT.md](AUDIT.md) (data-leakage & overfitting audit). Bibliography is in [REFERENCES.md](REFERENCES.md).

---

## Table of contents

0. [Plain-language walkthrough — how does this actually work?](#0-plain-language-walkthrough--how-does-this-actually-work)
1. [The forecasting problem](#1-the-forecasting-problem)
2. [Inductive ST-GNN architecture](#2-inductive-st-gnn-architecture)
3. [Graph construction](#3-graph-construction)
4. [Variant A — Pre-train + Fine-tune (PT-FT)](#4-variant-a--pre-train--fine-tune-pt-ft)
5. [Variant B — Graph-DANN (adversarial domain adaptation)](#5-variant-b--graph-dann-adversarial-domain-adaptation)
6. [Distribution-shift treatment (the `--fixed` protocol)](#6-distribution-shift-treatment-the---fixed-protocol)
7. [Where this sits in the research literature](#7-where-this-sits-in-the-research-literature)
8. [What is novel here](#8-what-is-novel-here-vs-the-precedents)

---

## 0. Plain-language walkthrough — how does this actually work?

This section walks through the **specific problem this project tackles** — forecasting 3-hour-ahead PM2.5 across three Indian cities — and explains the GNN and the transfer mechanism from first principles. The single most important question — **"how do you transfer a GNN from a 40-station megacity (Delhi) to a 4-station Northeast hill city (Guwahati) when the graphs are completely different?"** — is answered in detail in §0.4 and §0.7.

### 0.1 The problem, in concrete terms

India has about **350 CPCB Continuous Ambient Air Quality Monitoring (CAAQM) stations** spread very unevenly: Delhi has 40 in one ~50 km × 50 km basin, Kolkata has 10 across a coastal-plain delta, and Guwahati in Assam has only 4 in a Brahmaputra-valley pocket surrounded by hills. WHO's 5 µg/m³ annual PM2.5 guideline (WHO, 2021) is exceeded by a factor of 10–25× in all three (IQAir, 2024). Prior station-independent **LSTM-TL** for cross-city PM2.5 (Sanjeev, Prakash & Maitra, 2025) transfers reasonably well between similar-sized cities but degrades sharply when the target has very few stations — the **Guwahati problem** — and, being station-independent, cannot exploit spatial coupling. That motivates the graph approach here.

The exact forecasting task is:

```
Given:    history window  H = 8 time steps  =  24 hours  (3-hour cadence, CPCB native)
          N_c stations    (40 Delhi  /  10 Kolkata  /  4 Guwahati)
          F = 14 features per (station, timestep)
          → PM2.5, AT (ambient temperature), RH (relative humidity), WS (wind speed),
             sin/cos wind direction, sin/cos hour-of-day, sin/cos month-of-year,
             one-hot season {Winter, Spring, Summer, Monsoon}
Predict:  PM2.5 at every station,  3 hours ahead          (horizon = 1 step)
```

So one training example is a tensor `[H=8, N_c, F=14]` and one label is a vector `[N_c]`.

Two structural difficulties make this harder than a standard time-series forecast:

1. **Topological heterogeneity.** Delhi's 40-station graph and Guwahati's 4-station graph cannot share an architecture that has a `|V|`-shaped parameter anywhere. This is failure mode F-i (parameter-shape mismatch).
2. **Cross-city distribution shift.** Delhi has annual-mean PM2.5 ≈ 100 µg/m³ dominated by winter biomass + vehicular smog (CSE, 2024); Kolkata is a humid delta with eastern-IGP transport patterns; Guwahati has lower absolute concentrations but a steep monsoon-cycle. Naïve transfer fails because the **target distribution doesn't look like the source**.

This project's design addresses (1) with an **inductive ST-GNN** (Hamilton et al., NeurIPS-17; Veličković et al., ICLR-18) and (2) with two complementary transfer strategies: pre-train + fine-tune (Yadav et al., 2024; Hu et al., ICLR-20) and adversarial graph-level domain adaptation (Ganin & Lempitsky, ICML-15; Tang et al., CIKM-22).

### 0.2 The three city graphs, side by side

Edges are 3-nearest-neighbour, weighted by `exp(−d²/(2σ²))` with `σ = 5 km`. Visually:

```
 ┌────────────────────── DELHI (source candidate) ──────────────────────────┐
 │                                                                          │
 │    N = 40 stations    E = 120 directed edges    avg degree = 3.00        │
 │    Spread:  ~50 km E-W × ~40 km N-S  across the NCT                      │
 │    Year of record:    Jan 2021 – Dec 2022                                │
 │    Climate regime:    Indo-Gangetic Plain, winter inversion-driven       │
 │                                                                          │
 │           Bawana ●─────● Alipur ───● Burari Cr.───● Sonia Vihar          │
 │              │  ╲    ╱  │           │              │                     │
 │           DTU ●─────● Ashok Vihar ● Jahangirpuri ● Vivek Vihar           │
 │              │       │            │              │                      │
 │            ...37 more stations connecting across the metro...           │
 │                                                                          │
 └──────────────────────────────────────────────────────────────────────────┘

 ┌─────────────────────── KOLKATA  ───────────────────────┐
 │                                                        │
 │   N = 10    E = 30   avg degree = 3.00                 │
 │   Coastal Gangetic delta; Bay of Bengal influence      │
 │   Year of record: 2023                                 │
 │                                                        │
 │   Ballygunge ●───● Fort William ───● Victoria          │
 │        │     ╲ ╱       │                               │
 │   Bidhannagar ●────● Rabindra Bharati ● Jadavpur       │
 │        │           │              │                    │
 │     ... 4 more ...                                     │
 │                                                        │
 └────────────────────────────────────────────────────────┘

 ┌───────── GUWAHATI  (the hardest target) ────────┐
 │                                                  │
 │   N = 4    E = 12   avg degree = 3.00            │
 │   Brahmaputra valley; hill-surrounded basin      │
 │   Year of record: 2023                           │
 │                                                  │
 │   LGB Airport ●──────● Pan Bazaar                │
 │         │   ╲    ╱      │                        │
 │   Railway Col. ●──────● IIT Guwahati             │
 │                                                  │
 │   (k=3 on |V|=4 is essentially a complete digraph)│
 │                                                  │
 └──────────────────────────────────────────────────┘
```

The whole point of the project is that **the same model architecture and weights operate on all three of these graphs without modification.**

### 0.3 The GNN, step by step on a Delhi example

Each forward pass takes one window `X ∈ ℝ^{H × N × F}` and produces `ŷ ∈ ℝ^N`. The model alternates **temporal** and **spatial** processing — a pattern shared by STGCN (Yu, Yin & Zhu, IJCAI-18), Graph WaveNet (Wu et al., IJCAI-19), and MTGNN (Wu et al., KDD-20).

```
                  ONE TRAINING EXAMPLE — DELHI WINDOW
                  ───────────────────────────────────

   X[t, n, f]        ← shape [8, 40, 14]
       │              "the last 24 h at all 40 Delhi stations"
       │
       ▼
 ┌─────────────────────────────────────────────────────────────┐
 │  STEP 1 — TemporalConv₁  (kernel=3, dilation=1, F → 64)     │
 │                                                             │
 │   For each station n independently, slide a 3-wide          │
 │   filter along the 8-step time axis. After this every       │
 │   (n, t) cell holds a 64-dim summary of "what just          │
 │   happened at this station over the last few hours".        │
 │                                                             │
 │   Shape after:  [8, 40, 64]                                 │
 └─────────────────────────────────────────────────────────────┘
       │
       ▼
 ┌─────────────────────────────────────────────────────────────┐
 │  STEP 2 — Vectorize spatial pass                            │
 │                                                             │
 │   Flatten (batch, time) → one batch dim, so we have         │
 │   8 graph instances stacked together: [8·B, 40, 64].        │
 │   The GAT layer runs once over all of them in parallel.     │
 └─────────────────────────────────────────────────────────────┘
       │
       ▼
 ┌─────────────────────────────────────────────────────────────┐
 │  STEP 3 — GATLayer₁  →  ELU                                 │
 │                                                             │
 │   For every Delhi station n, look at its 3 nearest          │
 │   neighbours (e.g. ITO's neighbours = CRRI, JLN, NSUT)      │
 │   and form a weighted mix of their feature vectors.         │
 │                                                             │
 │       e_{ij} = LeakyReLU(〈Wxᵢ, a_src〉 + 〈Wx_j, a_dst〉)   │
 │                  + log(w_{ij})                              │
 │                                                             │
 │       α_{ij} = softmax over neighbours-of-j (e_{ij})        │
 │       h_j    = Σᵢ α_{ij} · Wxᵢ                              │
 │                                                             │
 │   The Gaussian k-NN edge weight enters log-additively       │
 │   into the unnormalized attention, so geographically        │
 │   closer neighbours get a head start in the softmax         │
 │   but the model can still down-weight them if needed        │
 │   (e.g. when ITO and CRRI sit on opposite sides of a        │
 │   pollution plume edge).                                    │
 │                                                             │
 │   This is the GAT formulation of Veličković et al. (2018),  │
 │   single-head, with explicit edge-weight injection — the    │
 │   same spirit as PM2.5-GNN's wind-aware weights             │
 │   (Wang et al., SIGSPATIAL-20).                             │
 └─────────────────────────────────────────────────────────────┘
       │
       ▼
 ┌─────────────────────────────────────────────────────────────┐
 │  STEP 4 — GATLayer₂  →  ELU                                 │
 │                                                             │
 │   Second hop. Now ITO's vector has been influenced by       │
 │   its neighbours' neighbours — effectively a 6-station      │
 │   receptive field, enough to span ~10–15 km in Delhi.       │
 └─────────────────────────────────────────────────────────────┘
       │
       ▼
 ┌─────────────────────────────────────────────────────────────┐
 │  STEP 5 — Reshape back to [B, 8, 40, 64]                    │
 │           TemporalConv₂  (kernel=3, dilation=2, 64 → 64)    │
 │                                                             │
 │   Second temporal pass with dilation 2 → receptive field    │
 │   of 7 of the 8 time steps. The model can now correlate     │
 │   "early-morning rising trend" with "current concentration  │
 │   at neighbours" — a TCN-style dilated stack                │
 │   (Bai, Kolter & Koltun, 2018).                             │
 └─────────────────────────────────────────────────────────────┘
       │
       ▼
 ┌─────────────────────────────────────────────────────────────┐
 │  STEP 6 — LayerNorm  →  take last time step  →  Linear head │
 │                                                             │
 │   We only care about the prediction t+1, so we grab the     │
 │   feature at t = 8 (the last step). One 64 → 1 linear       │
 │   head maps it to a single scalar per station.              │
 │                                                             │
 │   Output:   ŷ ∈ ℝ^{40}   →  one PM2.5 forecast per station  │
 └─────────────────────────────────────────────────────────────┘

   Loss = MSE(ŷ, y_true_PM25)   (in z-scored residual space, see §6)
```

Total parameters in `--fixed` mode: ≈ 25,000 — most of which sits in TCN-2 (12k) and the two GAT projections (8k combined). **Crucially, look at where those parameters live in shape-space:**

| component | what it learns | shape | depends on `N`? |
|---|---|---|---|
| TCN-1 kernel | how PM2.5 + met evolves over 24 h | `[F, 64, 3]` = `[14, 64, 3]` | **no** |
| GAT `W` | how to project one node's vector | `[64, 64]` | **no** |
| GAT `a_src, a_dst` | how to score one edge endpoint | `[64]` each | **no** |
| TCN-2 kernel | dilated temporal mixing | `[64, 64, 3]` | **no** |
| Head | per-station regression | `[64, 1]` | **no** |

Nothing — **literally not one weight** — is a function of how many stations the city has. That is exactly why the same model file can be loaded onto Guwahati's 4-station graph and produce 4 valid forecasts.

### 0.4 Why "the graph is an input, not a parameter" is the whole game

This is the conceptual hinge of the entire project. The contrast that makes it concrete:

```
   ┌─────────────────────────────────────────────────────────────────┐
   │   ❌  TRANSDUCTIVE GNN  (e.g. vanilla GCN with learned A)       │
   ├─────────────────────────────────────────────────────────────────┤
   │                                                                 │
   │   weights = {  W₁, W₂, …, A_{40×40}  }                          │
   │                              ▲                                  │
   │                              └─ adjacency baked into params     │
   │                                                                 │
   │   Train on Delhi:    A is 40×40 → fits Delhi.                   │
   │   Load on Guwahati:  Guwahati's A is 4×4 → SHAPE MISMATCH.      │
   │   Outcome: cannot transfer without surgery.                     │
   │                                                                 │
   └─────────────────────────────────────────────────────────────────┘

   ┌─────────────────────────────────────────────────────────────────┐
   │   ✅  INDUCTIVE GNN  (GAT / GraphSAGE — this project)           │
   ├─────────────────────────────────────────────────────────────────┤
   │                                                                 │
   │   weights = {  W₁, a_src, a_dst, W₂, …  }     — no |V| anywhere │
   │   forward(x, edge_index, edge_weight)   ← graph is an INPUT     │
   │                                                                 │
   │   Train on Delhi:    pass Delhi's (ei_D, ew_D), 40 nodes works. │
   │   Load on Guwahati:  pass Guwahati's (ei_G, ew_G), 4 nodes      │
   │                      works just as well — same weights.         │
   │   Outcome: drop-in transfer with no shape changes.              │
   │                                                                 │
   └─────────────────────────────────────────────────────────────────┘
```

The theoretical grounding for "the same trained operator generalizes across graphs of different sizes" is the **graphon-transferability** result of Ruiz, Chamon & Ribeiro (NeurIPS-20) and the **spectral transferability** results of Levie et al. (JMLR-21). In plain English: if two graphs are sampled from a sufficiently similar "graph distribution" (informally, similar spatial process), an inductive GNN trained on one converges to nearly the same behaviour on the other as the sizes grow.

The relevance to **this project's three cities**: all three are city-scale k-NN graphs of CPCB monitoring stations in the Indo-Gangetic Plain or its Brahmaputra-valley extension. Same underlying generating process; different realizations. Exactly the regime where inductive transfer should work.

### 0.5 What "training on Delhi" actually means — Stage 1 (source-only pre-train)

Before any transfer, we train one GNN per source city. Concretely, for Delhi:

```
   1.  Load delhi_processed.csv          → tensor of shape [T, 40, 14]
                                          T ≈ 17,500 timestamps × 2 years
   2.  Build the Delhi k-NN graph        → (edge_index, edge_weight)
                                          120 edges, weighted by Gaussian decay
   3.  Make sliding windows              → X: [num_windows, 8, 40, 14]
                                            Y: [num_windows, 40]
   4.  Split (interleaved 70/15/15)      → spreads val/test across all seasons
   5.  Subtract per-(month, hour) climatology fit on the train split only
       ──── (Yadav et al., 2024 residual-target idea)
   6.  Z-score features and target       → fit scaler on train only
   7.  Train for 25 epochs, batch 128, LR 1e-3, Adam + ReduceLROnPlateau,
       early stopping on val R²          → save models/gnn/gat_source_Delhi_fixed.pt
   8.  Test → reported as "Delhi source-only" in RESULTS.md §3.1
```

This produces a model whose weights have absorbed:
- the typical Delhi diurnal cycle (rush-hour peaks at 9 am / 7 pm IST),
- the winter inversion regime (high stagnant PM2.5 episodes Nov–Feb),
- the spatial correlation structure of the 40-station network (e.g. ITO and CRRI Mathura Road covary strongly because they are 4 km apart in the same wind corridor).

The same procedure runs on Kolkata (2023, 10 stations) and Guwahati (2023, 4 stations), each producing its own source checkpoint. The three checkpoints are interchangeable — they all use **the same model class with the same shape**.

### 0.6 Variant A — Pre-train + Fine-tune, with the **Delhi → Guwahati** case fully spelled out

This is the simpler of the two transfer strategies. The seven-step recipe in [src/train_gnn_tl.py](../src/train_gnn_tl.py):

```
   ┌──────────────────────────────────────────────────────────────────┐
   │   Variant A:  source = Delhi    target = Guwahati    d% = 30%    │
   └──────────────────────────────────────────────────────────────────┘

   ① Load the Delhi-pretrained checkpoint
        models/gnn/gat_source_Delhi_fixed.pt
        ↓
        weights = {TCN-1, GAT-1, GAT-2, TCN-2, Head} — all shape-free in |V|
        ↓
        Instantiate a fresh STGNN_GAT(in_features=14, hidden_dim=64, gat_dim=64)
        Load weights — no shape errors, because none of them care about |V|

   ② Build the Guwahati graph
        coords = [(26.10, 91.58), (26.14, 91.74), (26.19, 91.69), (26.20, 91.66)]
        → haversine distances, k=3 → edge_index_G, edge_weight_G

   ③ Take 30% of Guwahati's training windows
        Guwahati train = ~1,800 windows of shape [8, 4, 14]
        d=30% sub-sample → 540 windows

   ④ Freeze TCN-1 for the first 20% of epochs (= 8 of 40 epochs)
        Reason: protect the low-level temporal feature extractor from
        being trampled by the small Guwahati gradient. The same logic
        as Yadav et al. (2024), and standard ImageNet TL practice
        (Yosinski et al., 2014).

   ⑤ Fine-tune for 40 epochs at LR = 1e-4
        For each batch:
            xb has shape [32, 8, 4, 14]               ← Guwahati window
            forward = model(xb, edge_index_G, edge_weight_G)
                        ↑ feeds the GUWAHATI graph to the same weights
            loss = MSE(forward, y_guwahati)
            backward, Adam step (Adam with weight decay 1e-5)

   ⑥ Early stop on Guwahati val R²
        Save best-by-val checkpoint to
        models/gnn_tl/gat_tl_Delhi_to_Guwahati_d30_fixed.pt

   ⑦ Evaluate on Guwahati's held-out test partition (raw µg/m³ space)
        Report R², MAE, RMSE, MAPE.
```

Visually, the difference from Stage 1 is just which graph is plugged in:

```
  Stage 1 (Delhi source-only)              Variant A fine-tune (Delhi → Guwahati)
  ─────────────────────────────            ──────────────────────────────────────

         Delhi window                              Guwahati window
            [8,40,14]                                  [8,4,14]
               │                                          │
               ▼                                          ▼
    ┌──────────────────────┐                  ┌──────────────────────┐
    │   STGNN_GAT          │                  │   STGNN_GAT          │
    │   ──────────────     │                  │   ──────────────     │
    │   weights θ_random   │  ──── train ──►  │   weights θ_Delhi    │
    │                      │                  │                      │
    │   edges = ei_Delhi   │                  │   edges = ei_Guwa    │  ← only this changes
    │   |E| = 120          │                  │   |E| = 12           │
    │   |V| = 40           │                  │   |V| = 4            │
    └──────────────────────┘                  └──────────────────────┘
               │                                          │
               ▼                                          ▼
            ŷ ∈ ℝ⁴⁰                                    ŷ ∈ ℝ⁴
            train on Delhi                            fine-tune on Guwahati 30%
            → save θ_Delhi                            → measure test R² on Guwahati
```

### 0.7 The three-way verification — was the transfer real?

A common failure mode in transfer-learning papers is to report only `transfer.R²` and claim success without checking against the trivial baseline of "just train from scratch on the target." This project runs **three trainings per cell** on identical data:

```
   ┌──────────────────────────────────────────────────────────────────────┐
   │   For every (source, target, d%) cell:                               │
   │                                                                      │
   │   ┌──────────────┐    ┌──────────────┐    ┌────────────────────┐     │
   │   │ ZERO-SHOT    │    │  SCRATCH     │    │  TRANSFER          │     │
   │   │              │    │              │    │                    │     │
   │   │ Delhi weights│    │ Random init  │    │ Delhi weights      │     │
   │   │ no FT        │    │ FT on d%     │    │ FT on d%           │     │
   │   │ → R²_zs      │    │ → R²_sc      │    │ → R²_tl            │     │
   │   └──────────────┘    └──────────────┘    └────────────────────┘     │
   │                                                                      │
   │   "Real transfer" iff   R²_tl > R²_zs   AND   R²_tl > R²_sc          │
   └──────────────────────────────────────────────────────────────────────┘
```

`--full` runs **24 cells** = 6 pairs × 4 d-fractions {15, 30, 45, 60}%. All 24 passed (RESULTS.md §4). This is uncommon in cross-city ST-transfer literature (RegionTrans, MetaST, ST-GFSL, CrossTReS, TransGTR all skip the scratch baseline at matched-d), which is part of why this project's claims are defensible.

### 0.8 Variant B — Graph-DANN, the adversarial trick (v2 stabilized; transfer R² ≈ 0.81 on all three pairs — see [RESULTS.md §3.4](RESULTS.md))

PT-FT in §0.6 transfers **weights**. Graph-DANN goes further: it tries to make the encoder's **internal representation distribution** statistically indistinguishable across cities, *before* the small target gradient is even allowed to specialize. The idea, due to Ganin & Lempitsky (ICML-15) and formalized in Ganin et al. (JMLR-16):

> "If a separate classifier cannot tell which city an embedding came from, then the embedding must be city-invariant — and a city-invariant embedding is, by construction, transferable."

The architecture wraps the STGNN_GAT encoder with two additions: a **graph-level pooling** to get a fixed-size city descriptor, and a **Gradient Reversal Layer (GRL)** feeding a city classifier.

```
                          GRAPH-DANN  (Variant B)
                          ──────────────────────

      Delhi window   Kolkata window   Guwahati window
        [B,8,40,14]    [B,8,10,14]      [B,8,4,14]
              │              │               │
              │  same encoder weights θ      │
              ▼              ▼               ▼
       ┌──────────────────────────────────────────┐
       │     STGNN_GAT encoder                    │
       │     ────────────────                     │
       │     called 3× per step, one per city,    │
       │     each with that city's edge_index     │
       └──────────────────────────────────────────┘
              │              │               │
              ▼              ▼               ▼
          last-step      last-step       last-step
          [B,40,64]      [B,10,64]       [B,4,64]
              │              │               │
              ▼              ▼               ▼
       ┌──────────────────────────────────────────┐
       │   mean ⊕ max pool over node dim          │
       │                                          │
       │   → all three produce  z ∈ ℝ^{B × 128}   │   ← size-invariant!
       └──────────────────────────────────────────┘
              │              │               │
              │              │               │
              ▼  (Delhi only)│               │
       ┌─────────────┐       │               │
       │ Forecast    │       │               │
       │ head        │       │               │
       │ ŷ ∈ ℝ^{B×N} │       │               │
       └─────────────┘       │               │
                             │               │
              ┌──────────────┴───────────────┘
              ▼
       ┌──────────────────────────────────────────┐
       │           GRL ( λ )                      │
       │                                          │
       │   forward:  identity                     │
       │   backward: ∇ → −λ · ∇                   │   ← the adversarial sign flip
       └──────────────────────────────────────────┘
              │
              ▼
       ┌──────────────────────────────────────────┐
       │   CityDiscriminator  128 → 64 → 64 → 3   │
       │       (ReLU + Dropout 0.2)               │
       │                                          │
       │   trains to predict:  {Delhi, Kol, Guw}  │
       └──────────────────────────────────────────┘

      Total loss per step:
         L = MSE(ŷ_Delhi, y_Delhi)              ← source-supervised
            + CE(ĉ_Delhi, "Delhi")              ← city loss, source
            + CE(ĉ_Guw,    "Guwahati")          ← city loss, target (no PM2.5 used!)
            + CE(ĉ_Kol,    "Kolkata")           ← city loss, third-city replay
```

The mechanism in plain English:

1. **The discriminator** sees a 128-dim graph descriptor and tries to predict the city. Its parameters get **normal positive** gradients — it learns to classify well.
2. **The encoder** sees, through the GRL, **negated** versions of those gradients. So when the discriminator's gradient says "to be more Delhi-like, move this way", the encoder is told "move the *opposite* way". The encoder learns to produce embeddings that *fool* the discriminator.
3. **At the same time**, the encoder receives a normal positive gradient from the **forecast head** (on Delhi-labelled batches), so it cannot just degenerate to a useless representation — it must still encode enough to predict PM2.5.

The equilibrium of this saddle-point game is an encoder that produces **city-invariant features that are still predictive of PM2.5**. Exactly the representation we want for transfer.

**λ warm-up** (Ganin et al., 2016): `λ(p) = 2/(1+exp(−10p)) − 1`, where `p ∈ [0,1]` is training progress. Starts at 0 (encoder focuses on forecasting), ramps to 1 (full adversarial pressure once the forecaster works). Without warm-up the adversary destabilizes the early encoder.

**Three-city round-robin** (one of this project's specific design choices, [train_gnn_dann.py:166-220](../src/train_gnn_dann.py#L166-L220)): in each training step the encoder is called **three times** — once on a Delhi mini-batch (graph = `ei_Delhi`), once on a Kolkata mini-batch (graph = `ei_Kolkata`), once on a Guwahati mini-batch (graph = `ei_Guwahati`). All three feed the same city discriminator. The third (replay) city stabilizes adversarial training and prevents the encoder from learning a trivial source-vs-target binary boundary.

**Phase 2 — target fine-tune.** After 8 epochs of joint training, the GRL is turned off (`λ = 0`) and the model fine-tunes for 6 epochs on `d%` of the target's PM2.5 labels — same recipe as Variant A's step ⑤–⑦.

### 0.9 The single answer to "how do you transfer when the graphs are different?"

Combining everything above:

```
   ┌──────────────────────────────────────────────────────────────────┐
   │                                                                  │
   │   The encoder weights θ never see |V|. They only see:            │
   │                                                                  │
   │     • one node's feature vector  (F → 64)                        │
   │     • one edge's source/dest pair (a_src, a_dst)                 │
   │     • one neighbour's projected vector (W·x)                     │
   │     • one time step's window      (TCN kernel of size 3)         │
   │                                                                  │
   │   The graph (edge_index, edge_weight) is an INPUT to forward,    │
   │   not a parameter. The graph-pooling step (mean ⊕ max over N)    │
   │   collapses the node dim before any |V|-sensitive layer.         │
   │                                                                  │
   │   ⟹  the SAME weights produce valid forecasts                    │
   │      on Delhi (N=40), Kolkata (N=10), and Guwahati (N=4),        │
   │      with the only per-city ingredient being that city's         │
   │      (edge_index, edge_weight) tensors.                          │
   │                                                                  │
   │   Variant A exploits this by loading source weights into a       │
   │   target-instantiated encoder and fine-tuning on d% target data. │
   │                                                                  │
   │   Variant B exploits this further by forcing the pooled          │
   │   embedding to be statistically indistinguishable across the     │
   │   three cities via an adversarial city classifier through a GRL. │
   │                                                                  │
   └──────────────────────────────────────────────────────────────────┘
```

### 0.10 References for the concepts used in §0

| concept | citation |
|---|---|
| GAT message passing | Veličković et al., ICLR-18 |
| Inductive GNN philosophy | Hamilton, Ying & Leskovec, NeurIPS-17 |
| Mean + max graph pooling | Xu, Hu, Leskovec & Jegelka (GIN), ICLR-19 |
| TCN dilated temporal stack | Bai, Kolter & Koltun, 2018; Wu et al. (Graph WaveNet), IJCAI-19 |
| Spatio-temporal sandwich | Yu, Yin & Zhu (STGCN), IJCAI-18; Wu et al. (MTGNN), KDD-20 |
| Wind-aware edge prior | Wang et al. (PM2.5-GNN), SIGSPATIAL-20 |
| Inductive transferability | Ruiz, Chamon & Ribeiro, NeurIPS-20; Levie et al., JMLR-21 |
| Pre-train + fine-tune (GNN) | Hu et al., ICLR-20 |
| Frozen-layer FT for PM2.5 | Yadav et al., 2024 |
| Climatology residual | Yadav et al., 2024; long NWP tradition |
| Gradient Reversal + DANN | Ganin & Lempitsky, ICML-15; Ganin et al., JMLR-16 |
| Cross-city adversarial ST transfer | Tang et al. (DASTNet), CIKM-22; Wu et al. (UDA-GCN), WWW-20 |
| Station-independent LSTM-TL for cross-city PM2.5 | Sanjeev, Prakash & Maitra, 2025; Yadav et al., 2024 |
| Indian PM2.5 context | CSE, 2024; IQAir World Air Quality Report, 2024; WHO Air Quality Guidelines, 2021 |

Full bibliography in [REFERENCES.md](REFERENCES.md).

---

## 1. The forecasting problem

For each city `c` we have `N_c` CPCB CAAQM stations sampled at 3-hour cadence with `F` features (PM2.5, PM10, NO/NO₂/NOₓ, CO, SO₂, O₃, met variables). The forecasting task is:

> Given a history window of `H = 8` time steps (24 h) of `[H, N_c, F]` features, predict PM2.5 at horizon `t + H + horizon − 1` for **every station** in city `c` (`horizon = 1` ⇒ +3 h ahead). Output shape: `[N_c]`.

The transfer-learning challenge has two faces:

1. **Topological heterogeneity** — `N_Delhi = 40`, `N_Kolkata = 10`, `N_Guwahati = 4`. Any model whose parameter count depends on `N` cannot be transferred across cities without surgery. This is failure mode F-i (parameter-shape mismatch).
2. **Distribution shift** — cross-year and cross-region differences in PM2.5 climatology, met regimes, and emission sources cause large covariate + label shift between source and target.

The architecture in §2 tackles (1) by being **inductive** at the spatial axis. The training protocol in §6 tackles (2) via climatology-residual normalization (Yadav et al., 2024) and adversarial domain adaptation (Ganin & Lempitsky, 2015).

---

## 2. Inductive ST-GNN architecture

The base model in [src/models/stgnn_gat.py](../src/models/stgnn_gat.py) (and the SAGE ablation in [src/models/stgnn_sage.py](../src/models/stgnn_sage.py)) is a `TCN → GNN ×2 → TCN → Linear` sandwich, in the same family as **STGCN** (Yu, Yin & Zhu, IJCAI-18) and **Graph WaveNet** (Wu et al., IJCAI-19) but kept deliberately small for CPU compute.

```
Input  X  ∈ ℝ^{B × H × N × F}
        │
        ▼
   TemporalConv₁     1-D causal dilated conv per node, kernel=3, dilation=1
   (F → hidden_dim)
        │
        ▼
   reshape to [B·H, N, hidden_dim]            ── batch dim absorbs time
        │
        ▼
   GATLayer₁  →  ELU                          attention on the spatial graph
   (hidden_dim → gat_dim)
        │
        ▼
   GATLayer₂  →  ELU
   (gat_dim   → gat_dim)
        │
        ▼
   reshape back to [B, H, N, gat_dim]
        │
        ▼
   TemporalConv₂     kernel=3, dilation=2     receptive field over 5 steps
   (gat_dim → hidden_dim)
        │
        ▼
   LayerNorm
        │
        ▼
   take last time step:  z_T ∈ ℝ^{B × N × hidden_dim}
        │
   ┌────┴────┐
   │         │
   ▼         ▼
  Head       Graph pool (mean ⊕ max)  ──►  z_graph ∈ ℝ^{B × 2·hidden_dim}
  Linear                                   (used by DANN — see §5)
  (hidden_dim → 1)
   │
   ▼
  ŷ ∈ ℝ^{B × N}     PM2.5 (residual) at t+H+horizon−1, per station
```

### 2.1 Why this is inductive

Every learnable parameter (`W` in GAT/SAGE, the TCN kernels, LayerNorm γ/β, the head) has a shape that depends only on `(in_features, hidden_dim, gat_dim)` — **never on `N`**. The same checkpoint runs forward on a 4-node Guwahati graph or a 40-node Delhi graph with zero shape change. This is the property that makes Variant A's `Delhi(40) → Guwahati(4)` transfer mechanically possible at all.

See **GraphSAGE** (Hamilton, Ying & Leskovec, NeurIPS-17) and **GAT** (Veličković et al., ICLR-18) for the inductive-attention rationale this borrows from. The theoretical grounding for transferability across graphs of different sizes is in **graphon neural networks** (Ruiz, Chamon & Ribeiro, NeurIPS-20) and **spectral transferability** (Levie et al., JMLR-21).

### 2.2 GATLayer — single-head, edge-weighted, from scratch

PyG-free, CPU-only implementation in [stgnn_gat.py:29-85](../src/models/stgnn_gat.py#L29-L85).

For each directed edge `(i → j)` the attention score uses **separate source/destination attention vectors** (an additive variant of the original GAT formulation):

```
e_{ij} = LeakyReLU(  〈Wx_i, a_src〉 + 〈Wx_j, a_dst〉  ) + log(w_{ij})
```

The `log(w_{ij})` term log-additively folds the external edge weight (k-NN Gaussian-decay distance kernel; see §3) into the unnormalized attention logits, so attention can up- or down-weight the geographic prior. A per-destination softmax then normalizes over each node's incoming edges:

```
α_{ij}  =  softmax_{i ∈ N(j)}(e_{ij})
h_j     =  Σ_{i ∈ N(j)}  α_{ij} · Wx_i
```

Implementation notes:

- The per-destination softmax is implemented as `scatter_add_` over the destination index — no PyG, no `torch_scatter`, no compiled extensions ([stgnn_gat.py:71-85](../src/models/stgnn_gat.py#L71-L85)).
- Both GAT layers operate on `[B·H, N, ·]` so the spatial pass executes **once per (batch, time-step) instance**, vectorized across `B·H` graphs ([stgnn_gat.py:160-165](../src/models/stgnn_gat.py#L160-L165)). This gave a ~Hx CPU speedup over the naïve per-time-step loop.

### 2.3 GraphSAGE ablation

[stgnn_sage.py](../src/models/stgnn_sage.py) keeps the architecture but replaces attention with **edge-weighted mean aggregation**:

```
h_v ← ReLU(  W_self · h_v  +  W_neigh · mean_{u ∈ N(v)}( w_{uv} · h_u )  )
```

This is the GraphSAGE-mean variant of Hamilton et al. (NeurIPS-17). Selected via `--backbone sage`.

### 2.4 Temporal block — dilated 1-D causal conv

Each `TemporalConv` ([stgnn_gat.py:88-111](../src/models/stgnn_gat.py#L88-L111)) applies a 1-D conv along the time axis **independently per node**, with `padding=(k−1)·dilation` followed by a causal trim `[..., :H]` so the output never sees future time steps. Two blocks are stacked with `dilation = 1, 2`, so the second block sees a receptive field of `1 + (3−1)·1 + (3−1)·2 = 7` time steps over the 8-step history.

This is the **dilated-causal-conv idea** used in TCN (Bai et al., 2018) and Graph WaveNet. It is preferred over a per-node RNN here because (a) at H=8 a small TCN gives equivalent receptive field with fewer ops, and (b) it is straightforwardly vectorized across stations.

### 2.5 Forward pass — parameter counts

For the `--fixed` configuration (hidden_dim = 64, gat_dim = 64) the model has roughly:

| component | ~params |
|---|--:|
| TCN₁ (F · 64 · 3) + bias  | ~2.2k (F≈11) |
| GATLayer₁ (W: 64 · 64) + a_src, a_dst | ~4.3k |
| GATLayer₂ (W: 64 · 64) + a_src, a_dst | ~4.3k |
| TCN₂ (64 · 64 · 3) + bias | ~12.4k |
| LayerNorm + Linear head   | ~0.2k |
| **Total**                 | **~25k** |

By design, **this does not scale with `N`**. Delhi (40) and Guwahati (4) share the same encoder.

### 2.6 The graph-level embedding (for DANN)

When called with `return_embedding=True`, the encoder also produces a size-invariant graph descriptor by concatenating **mean and max pooling** over the `N` nodes of the last-time-step hidden state:

```
z_graph = [ mean_n(z_T[:, n, :])  ‖  max_n(z_T[:, n, :]) ]   ∈ ℝ^{B × 2·hidden_dim}
```

This is the standard graph-readout used in **GIN** (Xu, Hu, Leskovec & Jegelka, ICLR-19) and the inductive-pretraining literature (**Strategies for Pre-training GNNs**, Hu et al., ICLR-20). Crucially, mean+max pool is **size-invariant**: Delhi and Guwahati produce vectors of the same shape, which is what makes a single 3-class city discriminator possible.

---

## 3. Graph construction

Per-city spatial graphs are built in [src/graph_construction.py](../src/graph_construction.py) from CPCB station coordinates. Three families are supported via the `strategy` argument:

### 3.1 k-NN distance graph (default, `strategy="knn"`, `k=3`)

For each station `i`, connect to its `k = 3` nearest neighbours by **great-circle (haversine) distance** in km, with edge weights from a Gaussian decay:

```
w_{ij}  =  exp( − d_{ij}² / (2·σ²) ),   σ = 5 km
```

This produces directed edges `i → j` (not symmetrized) with `|E| = k · |V|`:

| City | `|V|` | `|E|` | avg degree |
|---|--:|--:|--:|
| Delhi    | 40 | 120 | 3.00 |
| Kolkata  | 10 |  30 | 3.00 |
| Guwahati |  4 |  12 | 3.00 |

(Guwahati with `k=3` and `|V|=4` is effectively a complete digraph minus self-loops.)

### 3.2 Wind-aware directed graph (`strategy="wind"`)

For a prevailing wind direction `θ_w` (degrees), each ordered pair `(i, j)` gets

```
w_{ij}  =  max(0, cos(θ_w − θ_{ij})) · exp(−d_{ij} / decay_km)
```

where `θ_{ij}` is the geographic bearing from `i` to `j`. Edges outside the wind cone get zero weight and are dropped.

This is the same physical prior used in **PM2.5-GNN** (Wang et al., SIGSPATIAL-20) — PM2.5 transports along the wind, so upwind neighbours carry more information than downwind ones at the same distance.

### 3.3 Hybrid

Currently emits the wind graph; the docstring reserves room for a learnable adjacency residual on top (à la **Graph WaveNet** / **MTGNN** adaptive adjacency, Wu et al., 2020). Not used in the headline experiments.

---

## 4. Variant A — Pre-train + Fine-tune (PT-FT)

Implemented in [src/train_gnn_tl.py](../src/train_gnn_tl.py). The novelty here is not the recipe — that's textbook transfer — but the **three-way verification** in every cell.

### 4.1 The protocol

For each `(source, target, d%)` cell we run three trainings on **identical** target subsamples:

| training | initialization | fine-tune data | meaning |
|---|---|---|---|
| **Zero-shot** | source checkpoint | none | "did the source learn anything that generalizes to the target?" |
| **Scratch**   | random            | d% target | "what does the same architecture learn from d% alone?" |
| **Transfer**  | source checkpoint | d% target | "does source pre-training help over scratch?" |

Knowledge transfer is declared **real** iff

```
transfer.R²  >  zero_shot.R²    AND    transfer.R²  >  scratch.R²
```

The `--full` flag runs all 6 ordered pairs × 4 `d`-values = **24 cells**. All 24 passed the test (see [RESULTS.md §4](RESULTS.md)).

This is uncommon in the cross-city TL literature: most papers report the transfer R² alone without the scratch baseline at the same `d`%, so positive transfer claims are not falsifiable. This project's 24/24 verification is precisely about closing that loophole.

### 4.2 Layer freezing

For the first `FREEZE_FRAC = 20%` of fine-tune epochs the first temporal block `t1` is frozen ([train_gnn_tl.py:70-80](../src/train_gnn_tl.py#L70-L80)). This protects the low-level feature extractor from being overwritten by the small target gradient before higher layers have specialized — the same logic as the frozen-layer fine-tune in **Yadav et al. (2024)** on Delhi PM2.5, and the standard practice in ImageNet TL (**Yosinski et al., 2014**, "How transferable are features...").

### 4.3 Per-city fine-tune recipe

[train_gnn_tl.py:47-51](../src/train_gnn_tl.py#L47-L51) mirrors the LSTM fine-tune configuration:

| target | epochs | batch | LR | patience |
|---|--:|--:|--:|--:|
| Delhi    | 25 | 128 | 2e-4 | 6  |
| Kolkata  | 30 |  64 | 2e-4 | 8  |
| Guwahati | 40 |  32 | 1e-4 | 10 |

Smaller cities → smaller batch, more epochs, lower LR — the standard regularization-budget tradeoff for low-data targets.

### 4.4 Optimization

Adam, `weight_decay = 1e-5`, `ReduceLROnPlateau(factor=0.5, patience=3)` on val R², gradient clipping at norm 1.0, MSE loss in z-scored residual space. Best-by-val checkpoint is restored before the final test evaluation. See [train_gnn_tl.py:54-106](../src/train_gnn_tl.py#L54-L106).

---

## 5. Variant B — Graph-DANN (adversarial domain adaptation)

Implemented in [src/models/dann.py](../src/models/dann.py) and trained by [src/train_gnn_dann.py](../src/train_gnn_dann.py). This is the project's principled contribution over plain PT-FT.

> ✅ **Run status (May 2026, v2, full grid):** Graph-DANN has been stabilized and run on the full 6 pairs × 4 d-fractions = **24 cells** under the `--fixed` protocol. Every cell is positive (min +0.7886, max +0.8250); all 24 beat zero-shot; 23/24 beat the random-init+target-FT scratch baseline (the lone tie is Kolkata→Guwahati @30%). Mean Δ vs Variant A (PT-FT) is −0.0009 R² — statistically indistinguishable, 12 wins each — see [RESULTS.md §3.4](RESULTS.md). Seven stabilization fixes were applied (ADDA-style warm-start, longer λ ramp, `α_d=0.1` loss reweighting, LayerNorm pre-GRL, fixed protocol, per-city FT recipe, larger discriminator subsample). The v1 failure log and root-cause analysis remain in [RESULTS.md §2.3](RESULTS.md) as a documented negative-then-fixed result.

### 5.1 The wrapper

`GraphDANN` ([dann.py:58-92](../src/models/dann.py#L58-L92)) composes an ST-GNN encoder with:

1. A **Gradient Reversal Layer (GRL)** ([dann.py:22-36](../src/models/dann.py#L22-L36)) — identity on forward, multiplies gradients by `−λ` on backward.
2. A **3-class city discriminator MLP** ([dann.py:39-55](../src/models/dann.py#L39-L55)) operating on the size-invariant `z_graph` from §2.6: `2H → 64 → 64 → 3` with ReLU + Dropout(0.2).

```
        encoder                  GRL(λ)             CityDiscriminator
   x ──────────►  z_graph  ─────────────────────►   MLP  ──►  ĉ_city
                     │
                     ▼
                  forecast head ──►  ŷ_pm25
```

### 5.2 The loss

Per mini-batch the optimizer minimizes

```
L  =  L_forecast(ŷ, y_pm25)  +  L_city(ĉ, c_city)
   =  MSE(ŷ, y)                 +  CE(ĉ, c)
```

with the GRL multiplying gradients flowing **into the encoder** through the discriminator branch by `−λ`. This is the **Domain-Adversarial Neural Network** of Ganin & Lempitsky (ICML-15) / Ganin et al. (JMLR-16), applied at the graph-pooled embedding instead of an image embedding.

The effective optimization is a saddle point:

- The **discriminator** wants to maximize city classification accuracy: it gets normal positive gradients on its own parameters.
- The **encoder** wants to fool the discriminator (and minimize forecast MSE): it receives negative discriminator gradients via the GRL, pushing `z_graph` toward a city-invariant manifold.

### 5.3 λ schedule

[dann.py:95-98](../src/models/dann.py#L95-L98) implements Ganin's classic warm-up:

```
λ(p)  =  2 / (1 + exp(−γ·p))  −  1,    p = global_step / total_steps,   γ = 10
```

`λ` starts near 0 (let the forecaster learn first), ramps smoothly to 1 over the course of joint training. This avoids destabilizing the encoder with strong adversarial gradients before its forecast head has formed.

### 5.4 Mini-batch composition — the three-city round-robin

[train_gnn_dann.py:90-112](../src/train_gnn_dann.py#L90-L112) round-robins one mini-batch per city per step:

- **Source city batch**: contributes `L_forecast` (supervised PM2.5) AND `L_city`.
- **Target city batch**: PM2.5 labels are **withheld** from `L_forecast`; only `L_city` is computed. This is the unsupervised-domain-adaptation regime — the target's PM2.5 is *not* used during joint training, only its features and city label.
- **Third (replay) city batch**: contributes `L_city` only. Including the third city in the discriminator's view stabilizes the adversarial training and prevents the encoder from learning a trivial source-vs-target binary boundary.

Each city is batched on **its own graph** — `edges = {Delhi: ei_D, Kolkata: ei_K, Guwahati: ei_G}` are precomputed in [train_gnn_dann.py:81-87](../src/train_gnn_dann.py#L81-L87) and the same encoder is invoked three times per step with three different `(edge_index, edge_weight)` pairs. **This is only possible because the encoder is inductive (§2.1).**

### 5.5 Phase 2 — target fine-tune

After `EPOCHS_JOINT = 8` joint epochs, the model fine-tunes for `EPOCHS_FT = 6` epochs on `d%` of the target's training windows with the **GRL turned off** (`λ = 0`). This is the standard DANN protocol: adversarial alignment provides the initialization; supervised fine-tune specializes for the target.

### 5.6 Mechanism summary — why this should help over PT-FT

Pre-train + fine-tune transfers the **weights** of a source-trained encoder, but those weights encode source-specific feature statistics. Adversarial DA additionally aligns the **representation distribution** so that the encoder's pooled output is statistically indistinguishable across cities, *before* the small target gradient is allowed to specialize. The hope is that fine-tune then operates on a strictly better-initialized representation, especially when the target has very little labelled data.

---

## 6. Distribution-shift treatment (the `--fixed` protocol)

Three diagnosed fixes are bundled behind `--fixed`. The legacy chronological-split numbers stay on disk for direct comparison. These are not architectural changes, but they are essential to make the architecture's headline numbers reproducible and honest.

1. **Interleaved split** ([utils.py:147-163](../src/utils.py#L147-L163)) — every `Nth` timestep goes to val/test rather than the chronological-tail split, so Kolkata/Guwahati's single-year test set spans every season instead of being winter-only. This eliminates the seasonal label-shift artifact that crushed Guwahati's legacy R² to −0.22. **Methodological note:** this is k-fold-style splitting for a forecasting task; it is asymptotically valid for stationary AR processes (Bergmeir, Hyndman & Koo, CSDA 2018) — which the climatology-residual signal approximately is — but inflates absolute R² versus a chronological-block split. See [AUDIT.md §3.2](AUDIT.md) for the full discussion and the chronological-block sensitivity-analysis recommendation.
2. **Climatology residual** ([utils.py:119-144](../src/utils.py#L119-L144)) — fit per-(month, hour) mean PM2.5 on the train mask only, subtract it from `y` before z-scoring. The model predicts **deviations from the seasonal-diurnal expectation**; at inference, climatology is added back ([utils.py:376-387](../src/utils.py#L376-L387)). This is the same residual-target idea used in Yadav et al. (2024) on Delhi PM2.5 across years. Verified leakage-free in [AUDIT.md §2.4](AUDIT.md).
3. **Matched GNN recipe** ([train_gnn.py:76-84](../src/train_gnn.py#L76-L84)) — hidden_dim 24 → 64, per-city epochs and LR matched to the LSTM, early stopping with `ReduceLROnPlateau`. Pre-fix, the GNN was hyperparameter-starved relative to the LSTM baseline.

After these fixes, the GAT-GNN source-only and the LSTM-fix source-only are within ≈ 0.04 R² on all three cities ([RESULTS.md §3.1](RESULTS.md)). The mean val−test R² gap across all reported cells is **−0.005** (test slightly above val), indicating no overfitting ([AUDIT.md §5](AUDIT.md)).

### 6.1 In-loader imputation — causal `ffill` only

A causal-imputation fix landed in [utils.py:212-222](../src/utils.py#L212-L222): the in-memory safety-net imputation used to be `.ffill().bfill().fillna(0.0)`; the anticausal `bfill()` step has been removed so that val/test NaNs cannot be filled from future-train observations. The upstream `data_pipeline.py` had already imputed the same cells (so this is a documentation-and-bulletproofing fix, not a results-changing one), but the loader-level fix means that any future re-run of `data_pipeline.py` with a fully causal imputer (recommended in [AUDIT.md §6.2 R-5](AUDIT.md)) will still flow through a leakage-free loader. The fix is documented in [AUDIT.md §2.2](AUDIT.md).

---

## 7. Where this sits in the research literature

| concept | this project | precedent |
|---|---|---|
| GAT spatial layer | from-scratch, single-head, edge-weighted | Veličković et al. (ICLR-18) |
| GraphSAGE-mean ablation | from-scratch, edge-weighted | Hamilton, Ying & Leskovec (NeurIPS-17) |
| TCN + GNN sandwich | TCN → GAT×2 → TCN → Linear | STGCN (Yu et al., IJCAI-18); Graph WaveNet (Wu et al., IJCAI-19); MTGNN (Wu et al., KDD-20) |
| Wind-aware edge weights | `cos(θ_w − θ_{ij}) · exp(−d/decay)` prior | PM2.5-GNN (Wang et al., SIGSPATIAL-20) |
| Inductive cross-graph transfer | same params, different `|V|` | Ruiz, Chamon & Ribeiro (NeurIPS-20) on graphon transferability; Levie et al. (JMLR-21) on spectral transferability |
| Pre-train + fine-tune protocol | source-only checkpoint → d% FT | Hu et al. (ICLR-20) on GNN pre-training; Yadav et al. (2024) on Delhi PM2.5 |
| Graph-level adversarial DA | GRL on mean+max pooled embedding | Ganin & Lempitsky (ICML-15); UDA-GCN (Wu et al., WWW-20); DASTNet for cross-city traffic (Tang et al., CIKM-22); EGI (Zhu et al., NeurIPS-21) |
| λ warm-up schedule | `2/(1+exp(−γp)) − 1` | Ganin et al. (JMLR-16) |
| Climatology residual | per-(month, hour) mean, fit on train | Yadav et al. (2024); long line of NWP literature |
| Cross-city ST-prediction | source → target with d% target data | RegionTrans (Wang et al., IJCAI-19); MetaST (Yao et al., WWW-19); ST-GFSL (Lu et al., KDD-22); CrossTReS (Jin et al., KDD-22); TransGTR (Jin et al., KDD-23) |
| Three-way verification (zero-shot vs scratch vs transfer) | per-cell in [src/train_gnn_tl.py](../src/train_gnn_tl.py) | uncommon in the literature |
| Station-independent LSTM-TL (related work) | cross-city PM2.5 LSTM transfer | Sanjeev, Prakash & Maitra (2025); Yadav et al. (2024) |

Full bibliography in [REFERENCES.md](REFERENCES.md).

### 7.1 Adjacent precedents worth flagging explicitly

- **PM2.5-GNN** (Wang et al., SIGSPATIAL-20) is the closest air-quality-specific GNN. It introduces the wind-aware edge weighting reused here, but does not address cross-city transfer.
- **DASTNet** (Tang et al., CIKM-22) is the closest cross-city adversarial DA. It targets traffic forecasting on (relatively homogeneous) road networks; this project ports the idea to air quality with much more heterogeneous topologies.
- **ST-GFSL** (Lu et al., KDD-22) and **TransGTR** (Jin et al., KDD-23) explore meta-learning and learned graph structure for cross-city ST transfer — orthogonal alternatives to adversarial alignment.
- **PM2.5-GLB** (Han et al., NeurIPS-21 GLB Workshop) is the relevant distribution-shift benchmark for PM2.5 graph learning, which motivates the climatology-residual treatment in §6.

---

## 8. What is novel here vs. the precedents

1. **Graph-DANN at the city scale with heterogeneous topology.** DASTNet (Tang et al., 2022) is the closest precedent for adversarial cross-city ST transfer, but operates on homogeneous road-network embeddings. Here the discriminator sees graph-pooled embeddings from cities with `|V|` differing by **an order of magnitude** (40 vs 4), and the mean+max pooling is what makes that comparable.
2. **Three-way verification protocol.** Zero-shot + scratch + transfer in every cell, with a hard "transfer must beat both" rule. The cross-city TL literature very rarely reports the scratch baseline at the same `d%`, so positive transfer claims are not falsifiable. This project's 24/24 verification is precisely about closing that loophole.
3. **Per-(month, hour) climatology residual on PM2.5 transfer.** Yadav et al. (2024) use the residual idea within Delhi across years; this project applies it across cities, which is what lifts Guwahati R² from −0.22 (legacy) to +0.83 (fixed).
4. **From-scratch CPU-only GAT/SAGE implementation.** No `torch_geometric`, no compiled extensions. The full 24-cell verification finishes in ≈ 2.5 hours on 8 vCPU / 16 GB / no GPU. Reproducibility for low-resource research environments was an explicit design goal.

---

## 9. File map

| concern | file |
|---|---|
| GAT layer + STGNN_GAT | [src/models/stgnn_gat.py](../src/models/stgnn_gat.py) |
| SAGE layer + STGNN_SAGE | [src/models/stgnn_sage.py](../src/models/stgnn_sage.py) |
| Gradient Reversal + GraphDANN + λ schedule | [src/models/dann.py](../src/models/dann.py) |
| k-NN / wind / hybrid graph builders | [src/graph_construction.py](../src/graph_construction.py) |
| Source-only GNN training | [src/train_gnn.py](../src/train_gnn.py) |
| Variant A (PT-FT + 3-way verification) | [src/train_gnn_tl.py](../src/train_gnn_tl.py) |
| Variant B (Graph-DANN) | [src/train_gnn_dann.py](../src/train_gnn_dann.py) |
| Interleaved split, climatology residual, windowing, metrics | [src/utils.py](../src/utils.py) |

---

*Companion documents: [MAIN_REPORT.md](MAIN_REPORT.md) (narrative), [RESULTS.md](RESULTS.md) (numbers), [AUDIT.md](AUDIT.md) (data-leakage & overfitting audit), [PAPER_DRAFT.md](PAPER_DRAFT.md) (journal-style manuscript), [REFERENCES.md](REFERENCES.md) (bibliography).*
