# GNN and GNN-TL — Architecture and Mechanism

A deep-dive into **exactly how** the inductive Spatio-Temporal GNN and the two flavours of GNN transfer learning are implemented in this repository, and where each design choice sits in the research literature.

This is the implementation-and-mechanism counterpart to [MAIN_REPORT.md](MAIN_REPORT.md) (narrative + diagnosis) and [RESULTS.md](RESULTS.md) (numbers). Bibliography is in [REFERENCES.md](REFERENCES.md).

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

Before the technical sections, here is the whole pipeline in plain English. The single most important question — **"how do you transfer a GNN from a 40-station city to a 4-station city when the graphs are completely different?"** — is answered in §0.3 and §0.4.

### 0.1 What the GNN does, in one paragraph

Each city is a set of monitoring stations sitting at fixed coordinates. We connect each station to its 3 nearest neighbours with edges weighted by how close they are (closer = stronger edge). That gives us a **graph**: nodes = stations, edges = "this station is geographically near that one." At every 3-hour time step each station reports a bunch of measurements (PM2.5, wind, temperature, etc.). The GNN's job is: **given the last 24 hours of readings at every station, predict the next 3-hour PM2.5 at every station**.

How does it do that? Two ingredients, alternated:
- **Temporal block (TCN)** — for each station individually, look at its past 24 hours and squeeze the time pattern into a feature vector. This is just a 1-D convolution along time.
- **Spatial block (GAT)** — for each station, look at its neighbours in the graph and mix their feature vectors in. "Mix" is weighted by an attention score the model learns: a neighbour that is upwind during a smog event gets more weight than one downwind.

After alternating these blocks twice (TCN → GAT → GAT → TCN), the model takes the last time-step's per-station feature vector and runs a tiny linear head on it to spit out one number per station — the predicted PM2.5.

### 0.2 Step by step inside one forward pass

For a single training example (24 h history × `N` stations × `F` features):

1. **TCN-1** — slides a small temporal filter over every station's past 24 hours, producing a fresh 64-dim feature per (station, time step).
2. **GAT-1** — for every time step, every station computes an attention score with each of its 3 neighbours, softmax-normalizes those scores, and replaces its own vector with a weighted sum of neighbour vectors. The edge weights from the k-NN distance kernel are folded in additively (closer neighbours get extra attention "for free").
3. **GAT-2** — same again. After two GAT hops, each station's vector has been influenced by neighbours-of-neighbours (≈ 6-station receptive field).
4. **TCN-2** — slides another temporal filter (now with dilation = 2 so it sees a wider time window).
5. **LayerNorm + head** — grab the feature vector at the *last* time step, run a linear `64 → 1`. One PM2.5 number per station. Done.

Loss: mean squared error between the predicted PM2.5 and the truth, averaged over all stations and all training windows.

### 0.3 The transfer-learning question, made concrete

You train this GNN on **Delhi (40 stations, 120 edges)**. The model has weights `W₁, W₂, …` for the TCN kernels, the GAT projections, and the head — about **25,000 parameters total**. Now you want to use that knowledge for **Guwahati (4 stations, 12 edges)**.

The obvious worry: "Delhi's graph has 40 nodes, Guwahati's has 4. The trained model is shaped for Delhi. How can the same weights possibly run on Guwahati?"

**Answer: they can, because not a single weight in this model has a size that depends on the number of stations.** Look at what the model actually stores:
- TCN kernel — a 3-wide filter over the time axis. **Time is the same for both cities.**
- GAT weight matrix `W` — projects a 64-dim node vector to a 64-dim node vector. **Same shape for any node.**
- GAT attention vectors `a_src, a_dst` — 64-dim each. **Same.**
- Linear head — `64 → 1`. **Same.**

Nowhere is there a `40 × 40` adjacency matrix being learned, or a per-station embedding table. The model is **inductive**: it operates on *one node at a time*, talking to *whoever happens to be its neighbour in the graph you hand it*. The graph is an **input**, not a parameter.

So the same model file:
- on Delhi, runs forward and produces 40 numbers (one per Delhi station);
- on Guwahati, runs forward on the Guwahati graph and produces 4 numbers (one per Guwahati station).

**No shape change, no surgery, no re-initialization.** This is the whole point of using GAT/GraphSAGE and not, say, a fixed-adjacency Chebyshev GCN. (Spelled out in detail in §2.1 and §2.5.)

### 0.4 Variant A — Pre-train + Fine-tune, step by step

This is the simpler of the two transfer strategies. Imagine you want **Delhi → Guwahati** transfer.

1. **Train on Delhi.** Run the GNN on the full Delhi dataset for 25 epochs. Save the checkpoint to `models/gnn/gat_source_Delhi_fixed.pt`. This is the "source model" — its weights have learned what PM2.5 dynamics look like in a North Indian winter.
2. **Build the Guwahati graph.** Compute haversine distances between Guwahati's 4 stations, build the k-NN edges, get `edge_index` and `edge_weight` for Guwahati. **This is a different graph from Delhi's** — but that's fine, the model accepts any graph as input.
3. **Take a small slice of Guwahati training data.** Pick `d% = 30%` of Guwahati's training windows.
4. **Load the Delhi-trained weights into a fresh model instance.** Because nothing is shaped for 40 stations, the load just works. Now you have a model that has Delhi-style understanding of PM2.5 but is about to look at Guwahati's graph.
5. **Freeze the first temporal block for the first 20% of epochs.** This protects the low-level feature extractor from being trampled by the small Guwahati gradient. After 20%, unfreeze.
6. **Fine-tune.** Run gradient descent on the Guwahati 30% subsample for 40 epochs (Adam, LR = 1e-4, early stopping on val R²). The forward pass uses the **Guwahati graph**; the loss uses **Guwahati PM2.5 truth**. The weights drift from "Delhi-shaped" to "Delhi-pretrained-then-Guwahati-specialized".
7. **Evaluate on Guwahati's held-out test partition.** Done.

To check whether the Delhi pre-training *actually helped*, every fine-tune cell is run **three times** on the same data slice:
- **Zero-shot** — Delhi weights, no fine-tuning. Just evaluate on Guwahati. Did Delhi knowledge transfer at all?
- **Scratch** — random init, fine-tune on Guwahati 30%. What does Guwahati 30% alone teach the model?
- **Transfer** — Delhi weights, fine-tune on Guwahati 30%. The actual TL result.

If `transfer > scratch` and `transfer > zero_shot`, the source pre-training is genuinely helping; if not, it is irrelevant or hurting. All 24 cells in this project passed this test.

### 0.5 Variant B — Graph-DANN, the adversarial trick, step by step

PT-FT transfers **weights**. Graph-DANN goes further: it tries to make the *encoder's internal representation* statistically indistinguishable across cities, *before* the small target gradient is even allowed to specialize. The idea (Ganin & Lempitsky, 2015) is:

> "If a separate classifier cannot tell which city an embedding came from, then the embedding must be city-invariant — and a city-invariant embedding is by definition transferable."

Mechanism:

1. **Wrap the GNN encoder.** Same GNN as before, but now after the last spatial block we also do a **graph-level pooling**: compute the mean and max of node features over all `N` stations, concatenate. **This pooling is the magic — it produces a fixed-size vector (128-dim) regardless of `N`**, so Delhi (40 nodes), Kolkata (10), and Guwahati (4) all produce comparable city-level descriptors.
2. **Attach a city classifier.** A small MLP `128 → 64 → 64 → 3` that tries to predict "which of the 3 cities did this pooled embedding come from?"
3. **Insert a Gradient Reversal Layer (GRL) between the encoder and the classifier.** On the forward pass, the GRL does nothing. On the backward pass, it **multiplies gradients by `−λ`**. So the city classifier still learns normally, but the gradients it sends back to the encoder are **flipped in sign**.
4. **Joint training loop.** For each step:
   a. Sample a mini-batch from the **source city (Delhi)**. Run the encoder on Delhi's graph. Compute the forecast loss (Delhi PM2.5 is labelled) AND the city loss (label = "Delhi").
   b. Sample a mini-batch from the **target city (Guwahati)**. Run the encoder on Guwahati's graph. **Do not use Guwahati PM2.5 labels.** Only compute the city loss (label = "Guwahati").
   c. Sample a mini-batch from the **third city (Kolkata)**. Run on Kolkata's graph. City loss only (label = "Kolkata"). The third city stabilizes training.
   d. Sum everything: `L = MSE(forecast) + CE(city) + CE(city) + CE(city)`. Backprop.

   Because of the GRL, the encoder gets two kinds of gradient updates:
   - From the forecast loss: "make Delhi's PM2.5 predictions accurate."
   - From the city loss (reversed): "make the pooled embedding *less* distinguishable across cities."

   The encoder is being pulled toward representations that are good for forecasting **AND** indistinguishable across cities. Whichever subset of features survives both pressures is, by construction, transferable.
5. **λ warm-up.** `λ` starts near 0 and ramps to 1 smoothly over training (`λ(p) = 2/(1+exp(−10·p)) − 1`). Early in training the encoder is allowed to focus on forecasting; the adversarial pressure kicks in later, once a useful representation already exists.
6. **Phase 2 — target fine-tune.** After joint training, **turn off the adversary** (`λ = 0`) and fine-tune on `d%` of the target's PM2.5 labels, exactly like Variant A's step 6.

### 0.6 Why the same encoder can be re-used across three different graphs in one mini-batch (the crux)

Inside one Graph-DANN training step the same encoder is called **three times** — once per city — each time with a **different `edge_index` and `edge_weight`** ([train_gnn_dann.py:81-87](../src/train_gnn_dann.py#L81-L87)). This works because:

- The encoder takes `(x, edge_index, edge_weight)` as inputs. The graph is **passed in**, not built into the weights.
- The attention computation inside GAT iterates over whatever edges you hand it — 120 for Delhi, 30 for Kolkata, 12 for Guwahati — using the same `W`, `a_src`, `a_dst` parameters each time.
- The pooling at the end (mean + max over all nodes) flattens out the `N` dimension, so the city descriptor is always 128-dim no matter how many stations there were.

That is the **whole** structural answer to "how do you transfer when the graph is different?" The model never has any weight that knows or cares how many stations there are. It only has weights for: *processing one time-series*, *transforming one node's vector*, and *deciding how much to attend to one neighbour*. Those operations are universal across cities; the graph is just the local wiring that tells the model who-talks-to-whom in this particular city.

### 0.7 Tiny summary table

| step | input | weights touched | output |
|---|---|---|---|
| TCN-1 | `[B, H, N, F]` | 3-wide temporal kernels (per-node) | `[B, H, N, 64]` |
| GAT-1 | `[B·H, N, 64]` + edges | `W`, `a_src`, `a_dst` (per-node, per-edge) | `[B·H, N, 64]` |
| GAT-2 | `[B·H, N, 64]` + edges | same shape as above | `[B·H, N, 64]` |
| TCN-2 | `[B, H, N, 64]` | 3-wide dilated kernels | `[B, H, N, 64]` |
| Head | last-step `[B, N, 64]` | `64 → 1` linear | `[B, N]` PM2.5 forecast |
| (DANN only) Pool | last-step `[B, N, 64]` | none — just mean & max | `[B, 128]` city embedding |
| (DANN only) GRL + Disc | `[B, 128]` | `128 → 64 → 64 → 3` MLP | `[B, 3]` city logits |

None of the weight shapes have `N` in them. **That is the entire reason transfer across heterogeneous city graphs is possible.**

---

## 1. The forecasting problem

For each city `c` we have `N_c` CPCB CAAQM stations sampled at 3-hour cadence with `F` features (PM2.5, PM10, NO/NO₂/NOₓ, CO, SO₂, O₃, met variables). The forecasting task is:

> Given a history window of `H = 8` time steps (24 h) of `[H, N_c, F]` features, predict PM2.5 at horizon `t + H + horizon − 1` for **every station** in city `c` (`horizon = 1` ⇒ +3 h ahead). Output shape: `[N_c]`.

The transfer-learning challenge has two faces:

1. **Topological heterogeneity** — `N_Delhi = 40`, `N_Kolkata = 10`, `N_Guwahati = 4`. Any model whose parameter count depends on `N` cannot be transferred across cities without surgery. This is failure mode F-i in the thesis post-mortem.
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

1. **Interleaved split** ([utils.py:147-163](../src/utils.py#L147-L163)) — every `Nth` timestep goes to val/test rather than the chronological-tail split, so Kolkata/Guwahati's single-year test set spans every season instead of being winter-only. This eliminates the seasonal label-shift artifact that crushed Guwahati's legacy R² to −0.22.
2. **Climatology residual** ([utils.py:119-144](../src/utils.py#L119-L144)) — fit per-(month, hour) mean PM2.5 on the train mask only, subtract it from `y` before z-scoring. The model predicts **deviations from the seasonal-diurnal expectation**; at inference, climatology is added back ([utils.py:376-387](../src/utils.py#L376-L387)). This is the same residual-target idea used in Yadav et al. (2024) on Delhi PM2.5 across years.
3. **Matched GNN recipe** ([train_gnn.py:76-84](../src/train_gnn.py#L76-L84)) — hidden_dim 24 → 64, per-city epochs and LR matched to the LSTM, early stopping with `ReduceLROnPlateau`. Pre-fix, the GNN was hyperparameter-starved relative to the LSTM baseline.

After these fixes, the GAT-GNN source-only and the LSTM-fix source-only are within ≈ 0.04 R² on all three cities ([RESULTS.md §3.1](RESULTS.md)).

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
| Stage-I baseline anchor | LSTM-TL B.Tech thesis | Sanjeev, Prakash & Maitra (2025) |

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

*Companion documents: [MAIN_REPORT.md](MAIN_REPORT.md) (narrative), [RESULTS.md](RESULTS.md) (numbers), [REFERENCES.md](REFERENCES.md) (bibliography), [ResearchProposal.md](ResearchProposal.md) (publication framing).*
