# Literature Review + Positioning — Sensor-Sparsity as the Problem: Physics-Informed GNN Transfer Learning over Sparse, Heterogeneous IoT Sensor Networks

> **Venue pivot (2026-06-10):** target is now **IEEE Internet of Things Journal (IoT-J)** / an IEEE IoT venue, **not** TGRS. The problem is reframed from an *environmental / geoscience* problem (PM2.5 air-quality forecasting) to a **sensor-sparsity problem**: how to maintain spatio-temporal inference accuracy from *few* sensor nodes, and how to transfer a model across *heterogeneous deployments* with different node counts. **PM2.5 / CPCB CAAQM is the real-world IoT sensor-network testbed, not the subject.** Frame every experiment around sensor sparsity.

Purpose: ground the IoT reframing and the **physics-informed Variant E** before committing experiments, and decide what "physics-informed" means for **sparse, irregular point-sensor** inference (4–40 nodes), not the dense-grid PDE setting most PINNs assume. Cross-refs: [MAIN_REPORT.md](MAIN_REPORT.md), [PINN_PHYSICS.md](PINN_PHYSICS.md), [REFERENCES.md](REFERENCES.md), [`src/graph_construction.py`](../src/graph_construction.py).

---

## 1. The problem — sparse, heterogeneous IoT sensor networks

IoT sensing deployments are **sparse and heterogeneous by economics, not by choice**: each node costs hardware, power, calibration, and maintenance, so real deployments have *few* nodes, and different sites have *different* node counts and topologies. Three hard problems follow, all IoT-native:

1. **Sparse-deployment inference.** Can you forecast the monitored field accurately from a handful of nodes, when the spatial field is under-sampled?
2. **Cross-deployment (cold-start) transfer.** When a *new* sparse deployment comes online with no usable history, can it inherit a model trained on an existing *dense* deployment — despite a different node count `|V|` and geometry — without a long data-collection delay?
3. **Reliability under node loss.** Nodes fail (battery, comms, fouling). Does accuracy degrade *gracefully* as nodes drop out, or collapse?

The research question (IoT-J framing):

> *For a target inference accuracy, how few sensor nodes does a deployment need — and how much of that need can be removed by (a) spatial coupling across nodes (GNN), (b) transfer from a dense deployment (TL), and (c) a physical transport prior (PINN)?*

This is a **sensor-budget / deployment-cost** question, squarely in scope for IoT-J, and it connects to **graph-signal sampling theory** and **sensor placement** rather than to atmospheric science.

---

## 2. The IoT testbed and the evidence already in hand

**CPCB CAAQM is a real heterogeneous IoT deployment.** The Central Pollution Control Board's Continuous Ambient Air Quality Monitoring network is exactly a networked, always-on sensor fleet streaming at fixed cadence — three deployments of order-of-magnitude different density: **Delhi 40 → Kolkata 10 → Guwahati 4 nodes**. PM2.5 is simply the signal; the contribution is about the *sparsity of the node set*. (IoT-based air-quality monitoring is an established IoT-J topic — see §5.4 — so this bridge is well-precedented, not a stretch.)

**The sparsity evidence is already in our results.** From [`results/gnn_tl/variantA_gat_fixed.json`](../results/gnn_tl/variantA_gat_fixed.json) (GNN-TL, pre-train + fine-tune):

| Deployment (`|V|`) | Source deployment | Cold-start (zero-shot) R² | After transfer R² | MAE (µg/m³) |
|---|---|---|---|---|
| **Guwahati (4 nodes)** | Delhi | 0.684 | **0.827** (@45%) | 20.5 → 13.4 |
| **Guwahati (4 nodes)** | Kolkata | 0.670 | **0.816** (@60%) | 21.8 → 13.0 |
| Kolkata (10) | Delhi | 0.728 | 0.818 (@45%) | 14.0 → 10.0 |
| Delhi (40) | Guwahati | 0.632 | 0.817 (@60%) | 38.4 → 25.3 |

**Money observation:** the *sparsest* deployment (4 nodes) gets among the largest transfer lifts (+0.13–0.15 R²). A station-independent LSTM cannot do this — it has no channel to share signal across nodes, so a 4-node deployment stays 4 isolated series. Spatial coupling (GNN) + transfer converts a sparse node set into a forecastable network. We then test whether transport *physics* is the underlying mechanism (§7) and find it is **not** — the transfer itself is the mechanism (the physics prior is a rigorous negative).

---

## 3. Mechanism: why GNN + TL + physics, in IoT terms

- **Spatial coupling (GNN) = compensate for under-coverage.** Message passing lets a sparse deployment's nodes share information. The inductive encoder (`STGNN_GAT`, ~25k params, **no `|V|`-shaped weights, CPU-only**) reuses one *operator* across `|V| ∈ {4,10,40}` — i.e., one model serves any deployment size without re-architecting. (This is also an **edge/TinyML** selling point: lightweight, no per-deployment retraining.)
- **Transfer (TL) = solve cold-start.** A new sparse deployment inherits the dense deployment's spatial operator and temporal dynamics instead of waiting to collect its own history.
- **Physics (PINN) = model-based virtual sensing.** GNN+TL still learn the transport operator *purely from data*; with 4 nodes the field is under-determined. A physical transport prior constrains the solution so the same nodes identify a better field — classic "physics priors cut the data/sensors you need."

---

## 4. Theoretical backbone — "how few sensors suffice" (graph signal sampling)

The principled IoT-J grounding for the whole paper: **graph signal processing (GSP) sampling theory** answers *how many, and which, nodes are needed to reconstruct a signal living on a graph*.

- A **bandlimited / smooth graph signal** can be reconstructed from a sampling set far smaller than `|V|`; the minimum set size is governed by the signal's spectral bandwidth (Chen et al., *Discrete Signal Processing on Graphs: Sampling Theory*, IEEE TSP 2015; Anis et al., *Efficient Sampling Set Selection for Bandlimited Graph Signals*, IEEE TSP 2016; Ortega et al., *Graph Signal Processing: Overview, Challenges, Applications*, Proc. IEEE 2018).
- **Sensor placement under economic constraint** is the deployment-cost version of the same problem (Joshi & Boyd, *Sensor Selection via Convex Optimization*, IEEE TSP 2009; Krause et al., *Near-Optimal Sensor Placements in Gaussian Processes*, JMLR 2008; *Dynamic Sensor Placement Based on Sampling Theory for Graph Signals*, 2022).

**Why this matters for us:** it turns "fewer sensors suffice" from a hopeful claim into a *condition* — few nodes suffice **when the field is smooth/low-rank on the graph**. Our advection–diffusion prior is precisely a mechanism that *makes the learned field smooth in the transport sense*, so §4 (theory) and §7 (our physics term) are two views of the same idea. This is the bridge a reviewer will reward.

---

## 5. Prior art

### 5.1 Sparse-sensor STGNNs and cold-start transfer (the core IoT lineage)
- STGNNs compensate for sparse coverage via spatial correlation, but "rely on dense historical data, often missing in newly deployed networks → motivating transfer from data-rich source to data-sparse targets" — the literature's own statement of our cold-start contribution.
- **AGSTN** — *Attention-adjusted Graph Spatio-Temporal Networks for Short-term Urban Sensor Value Forecasting* (arXiv:2101.12465) — urban IoT sensor-value forecasting on a graph.
- **Contextualized ST-graph forecasting for sparse geospatial sensor networks** (*Expert Systems with Applications*, 2025) — explicitly "sparse sensor networks."
- **Transfer-oriented ST-graph frameworks** (e.g., *Pruning for Generalization*, 2026) — transferable ST-GNNs across networks.
- Cross-city urban-sensing transfer precedents already in REFERENCES.md §F (Wei et al. KDD'16; DASTNet; ST-GFSL; CrossTReS; TransGTR) — recast here as **cross-deployment** transfer.

### 5.2 Sensor placement / selection (deployment-cost axis)
Joshi & Boyd 2009; Krause et al. 2008; graph-sampling placement (2022). Gives us a principled choice of *which* `k` nodes to keep when we down-sample dense deployments (§8), and a cost-framed baseline.

### 5.3 Physics-informed / virtual sensing for *sparse* sensors
- **PA-DGN — Physics-aware Difference Graph Networks for Sparsely-Observed Dynamics** (Seo, Meng & Liu, ICLR 2020) — physics-difference operators on graphs from **sparse** observations. Canonical "physics + graph + sparse sensors."
- **Physic_GNN — hydrogen-jet diffusion from sparsely-distributed sensors** (arXiv:2308.12621, 2023); **PIGNN auto-encoder for urban wind fields from sparse sensors** (Gao et al., *CACAIE* 2024) — both reconstruct fields from sparse IoT-style sensors.
- **FieldFormer** (arXiv:2510.03589, 2025): classic PINNs "assume **dense, continuous input spaces**… degraded performance on **sparse or irregular** real-world data" → our justification for a **graph-discretized** transport prior, not a collocation-grid PINN.

### 5.4 Domain instances (application prior art — demote, don't lead with)
- **IoT air-quality monitoring** (Sensors/MDPI 2025 *Innovations in AQ Monitoring: Sensors, IoT*; IoT-WSN for air pollution w/ fractional-order Kalman, Sensors) — establishes AQ-as-IoT; supports the testbed framing.
- **AirPhyNet** (arXiv:2402.03784, 2024) and **TransNet** (*npj Clean Air*, 2026) — GNN + advection-diffusion for PM2.5, **single-deployment, no cross-deployment transfer, no `|V|` heterogeneity, no sensor-budget analysis.** These prove "GNN-PINN-for-PM2.5" is *not itself* novel; they are application instances we transcend, not competitors on the IoT problem.

### 5.5 The open gap (our intersection)
> physics-informed (transport) prior **×** cross-**deployment** transfer **×** order-of-magnitude node-count heterogeneity (4↔10↔40) **×** an explicit **sensor-budget / failure-robustness** evaluation.

No prior work occupies this cell. Novelty is the **combination + the IoT sensor-sparsity framing**, never "we add physics to a GNN."

---

## 6. Positioning for IoT-J + the conditions of (cold-start) transfer

- **Title/abstract** lead with **sparse, heterogeneous sensor networks** and **sensor-budget efficiency**; PM2.5 named as the evaluation testbed; **GNN-TL (pre-train + fine-tune)** is the transfer mechanism.
- **Physics is a property of the base model; transfer runs on top** → orthogonal-axes experiment **{data-only GNN-TL, physics-informed GNN-TL}**. Headline object = the **conditions map** `Δ = R²(physics) − R²(data-only)` over `|V|` and `d%`.
- **Headline contribution = conditions under which a sparse deployment can inherit a dense one** (cold-start transferability), explained, not just measured:
  1. **Physical universality → why transfer crosses `|V|`.** Advection–diffusion is the *same law* in every deployment; only the source/boundary term `S_i` is site-specific, so the transferred object (transport dynamics) is deployment-invariant. Principled answer to "why should a 40-node model work on a 4-node site?"
  2. **Field smoothness → when few nodes suffice** (ties directly to §4 GSP sampling): few nodes reconstruct a transport-dominated, graph-smooth field; a source-dominated spiky field (Delhi hotspots) needs more — explaining why Guwahati transfers to R²≈0.83 while Delhi keeps MAE≈25–35. `L_phys` becomes a **regime diagnostic**.
  3. **When the prior pays off:** strongest at sparse `|V|`, low label budget `d%`, transport-dominated regimes; ≈0 on dense, source-dominated deployments.
- **Anticipate the AirPhyNet reviewer** in related work: "unlike single-deployment GNN-PINNs, we use the transport residual as a **transferable** prior across deployments of different `|V|` and evaluate it as a **sensor-budget** mechanism."

---

## 7. The formulation (physics-informed base, transfer on top)

**Why not a vanilla PINN:** 4–40 *point* nodes, no continuous field; classic PINNs need ∇C at dense collocation points (FieldFormer caveat §5.3). Discretize the transport PDE **on the sensor graph** — feasible precisely because operators live on edges.

**E1 — soft physics-residual regularizer (recommended), a composable flag, not a new model.** Keep `STGNN_GAT` unchanged; add one loss term so the whole protocol, the `|V|`-invariance, and the transfer story carry over unchanged and stay apples-to-apples:

```
r_i = (Ĉ_i(t+Δ) − C_i(t))/Δt        # temporal derivative, from history window + forecast
      + (A_wind · C)_i               # advection — directed wind graph (wind_aware_graph, already built)
      + κ · (L · C)_i                # diffusion — graph Laplacian L = D − W of knn_graph
      + γ · C_i  −  S_i              # linear deposition/loss − source
L_phys  = mean_{i,t} ‖r_i‖²
L_total = L_data + λ_phys · L_phys    # κ, γ learnable; S_i learned or ≈0 at the 3-h horizon
```

`A_wind` and `L` come from existing builders in [`src/graph_construction.py`](../src/graph_construction.py); `λ_phys` is swept as a single regularization weight. Realized as `--physics --lambda_phys` on the GNN-TL trainer.

**E2 — physics-structured neural-ODE block (AirPhyNet-style).** More novel but a larger rebuild, collides more with AirPhyNet → defer to future work.

---

## 8. Experiments — all framed around sensor sparsity (next session)

**Structural experiment** = GNN-TL **with vs without the physics term** on the identical protocol. Read-outs:

1. **Sensor-budget curve (hero figure).** Accuracy (R²/MAE) vs **number of active nodes**, curves for LSTM-TL / GNN-TL / GNN-PINN-TL. Three real points (4/10/40); **down-sample Delhi & Kolkata** to `k ∈ {4,6,8,…}` to fill it. *Choose which `k` nodes by graph-sampling placement (§4/§5.2) vs random* — placement matters is itself a result. Claim: physics shifts the budget curve left (same accuracy, fewer nodes).
2. **Node-dropout / failure robustness (IoT-native).** At *inference*, randomly drop `p%` of nodes (battery/comms failure) and measure degradation. Hypothesis: GNN degrades gracefully (neighbours compensate), physics further; station-independent LSTM loses those nodes outright. This is an IoT reliability story no AQ-GNN paper reports.
3. **Cold-start label-budget curve.** Accuracy vs target label fraction `d% ∈ {15,30,45,60}`. Physics should lift the *low-`d%`* end most (prior substitutes for labels).
4. **Conditions map.** `Δ(physics − data-only)` as a heatmap over `|V|` × `d%` — the "explain the conditions" deliverable.
5. **Physics-consistency diagnostic.** Report `L_phys` per variant — transport-consistency where data-only is not; doubles as the regime indicator of §6.

**Decision rule**: keep the physics term if it beats its non-physics twin in mean ΔR² **or** wins on the sparsest `|V|` / lowest `d%` / node-dropout. A flat-overall-but-sparse-positive result *is* the sensor-sparsity finding. A clean null ("transport prior does not help dense deployments but recovers accuracy under node sparsity/failure") is itself citable.

---

## 9. Citations to add

| Theme | Citation | Status |
|---|---|---|
| **GSP sampling theory (how few nodes)** | Chen et al., *DSP on Graphs: Sampling Theory*, IEEE TSP 2015 | canonical |
| GSP sampling set selection | Anis et al., IEEE TSP 2016 | canonical |
| GSP overview | Ortega et al., *Proc. IEEE* 2018 | canonical |
| Sensor selection (cost) | Joshi & Boyd, IEEE TSP 2009 | canonical |
| Sensor placement (GP) | Krause, Singh & Guestrin, JMLR 2008 | canonical |
| Graph-sampling placement | *Dynamic Sensor Placement … Graph Signals*, 2022 | verified |
| Sparse-sensor urban STGNN | AGSTN, arXiv:2101.12465 | verified |
| Sparse geospatial sensor STGNN | *Contextualized ST-graph …*, ESWA 2025 | verified |
| IoT AQ monitoring (testbed bridge) | *Innovations in AQ Monitoring: Sensors, IoT*, Sensors 2025 | verified |
| **Physics + graph + sparse sensors** | PA-DGN (Seo, Meng & Liu), ICLR 2020 | verified |
| PIGNN sparse sensors | arXiv:2308.12621 (2023); Gao et al., *CACAIE* 2024 | verified |
| Vanilla-PINN-on-sparse caveat | FieldFormer, arXiv:2510.03589 (2025) | verified |
| PINN foundation / review | Raissi et al. 2019; Karniadakis et al. 2021 | canonical |
| Domain prior art (AQ GNN-PINN) | AirPhyNet arXiv:2402.03784; TransNet *npj Clean Air* 2026 | verified |
| PINN + transfer learning | arXiv:2502.00782; arXiv:2401.02810 | verified |

New REFERENCES.md sections: **"Q. Physics-informed / physics-guided spatiotemporal models"** and **"R. Sparse sensing, graph signal sampling, and sensor placement (IoT)."** Verify the full AirPhyNet author string before final bib.

---

## 10. Risks / honest caveats

- **Reframing must be thorough, not cosmetic.** An IoT-J reviewer will reject buzzword-swapped geoscience. The intro, related work, and *experiments* (node-dropout, sensor-budget, placement) must be genuinely IoT — hence §8's new experiments, not just relabelled tables.
- **Novelty:** GNN-PINN-for-PM2.5 already exists → novelty = combination + sensor-sparsity framing, stated explicitly.
- **Identifiability:** with 4 nodes the graph operators are coarse; κ, γ weakly identified → sweep `λ_phys`, consider per-deployment κ, γ.
- **Source term `S_i`:** emissions unobserved; assume ≈0 at 3-h horizon or learn a small per-node bias; document in limitations.
- The consolidated [`MAIN_REPORT.md`](MAIN_REPORT.md) is the IoT-rooted narrative; this document supplies the sensor-sparsity positioning + literature for it.

---

### Sources (this session's searches)
- Sparse-sensor STGNN / cold-start transfer: https://arxiv.org/html/2511.05179 · https://arxiv.org/pdf/2602.04153 · https://arxiv.org/pdf/2101.12465 (AGSTN) · https://www.sciencedirect.com/science/article/pii/S0957417425023978
- GSP sampling / sensor placement: https://arxiv.org/html/2211.04019v4 · https://pmc.ncbi.nlm.nih.gov/articles/PMC7922557/
- IoT air-quality monitoring (testbed): https://www.mdpi.com/1424-8220/25/7/2070 · https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8398794/
- Physics-informed / sparse-sensor: https://www.semanticscholar.org/paper/Physics-aware-Difference-Graph-Networks-for-Seo-Meng/f3819aee0294502d8705d001066789b493ee3792 · https://arxiv.org/pdf/2308.12621 · https://onlinelibrary.wiley.com/doi/10.1111/mice.13147 · https://arxiv.org/pdf/2510.03589
- AQ GNN-PINN prior art: https://arxiv.org/abs/2402.03784 (AirPhyNet) · https://www.nature.com/articles/s44407-026-00052-x (TransNet)
- PINN foundation / TL: https://arxiv.org/html/2502.00782v1 · https://www.pnnl.gov/explainer-articles/physics-informed-machine-learning
