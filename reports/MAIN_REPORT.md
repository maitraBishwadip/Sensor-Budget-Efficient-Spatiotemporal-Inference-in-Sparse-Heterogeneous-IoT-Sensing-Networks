# Transfer Learning over Sparse, Heterogeneous Sensor Networks for PM2.5 Forecasting — A Coherent Study

*Single consolidated technical report. Companion docs: math verification [MATH_AUDIT.md](MATH_AUDIT.md);
physics derivation [PINN_PHYSICS.md](PINN_PHYSICS.md); physics ablation write-up
[PHYSICS_ABLATION_SECTION.md](PHYSICS_ABLATION_SECTION.md); sensor-sparsity positioning + literature
[SENSOR_EFFICIENCY_PINN_LITERATURE.md](SENSOR_EFFICIENCY_PINN_LITERATURE.md); data-leakage audit
[AUDIT.md](AUDIT.md); bibliography [REFERENCES.md](REFERENCES.md). All numbers from `results/`.*

---

## 0. Summary (what is true)

We study PM2.5 forecasting over three real, order-of-magnitude-different ground-sensor deployments
(Delhi 40 / Kolkata 10 / Guwahati 4 nodes) as a **sparse-IoT-sensor-network** problem. Four honest findings:

1. **Transfer learning is the decisive lever.** A data-sparse deployment inherits a strong forecaster from
   a data-rich one; both LSTM and GNN transfer well (source-only R² 0.83–0.86, maintained under transfer).
2. **A station-independent LSTM is as good as or better than the GNN for *forecasting* at sensored nodes.**
   Spatial coupling does not improve short-horizon forecasting at these stations — confirmed across two GNN
   temporal encoders (TCN and GRU).
3. **The GNN's one genuine advantage is *virtual sensing*** — estimating PM2.5 at **unsensored** locations by
   borrowing from neighbours, which a station-independent LSTM cannot do. It **beats a met-only LSTM at every
   configuration** (+0.08 to +0.16 R² at the unsensored nodes).
4. **A physics-informed transport prior (advection–diffusion) does not help** — a rigorous, multi-seed
   negative; any tiny gain is generic graph smoothing, not the wind.

This is a careful "what actually helps for sparse PM2.5 sensor networks" study, not a "GNN wins" paper.

---

## 1. Problem and framing

Ground PM2.5 sensors are costly, so real deployments are **sparse and heterogeneous**: few nodes, and
different sites have different node counts/topologies. Two questions follow: (i) can a forecaster be
*transferred* from a dense deployment to a sparse one despite different `|V|`? and (ii) can we estimate PM2.5
where there is **no** sensor? CPCB's CAAQM network is the testbed — Delhi 40, Kolkata 10, Guwahati 4 nodes —
a real heterogeneous IoT deployment. PM2.5 is the signal; **sensor sparsity is the problem.**

## 2. Data and protocol

CPCB CAAQM, 3-hour cadence; features PM2.5, AT, RH, WS, Sin/Cos wind-direction (+ cyclic/season encodings);
forecast horizon = one 3-h step from 24 h (8-step) history. **Fixed protocol** (used throughout):
interleaved train/val/test split (each split spans the full year) + per-(month,hour) **climatology-residual**
target (model predicts deviation from the seasonal–diurnal mean; climatology added back at evaluation).
All metrics are computed in **raw µg/m³**. Scalers/climatology are fit on train only (leakage audit in
[AUDIT.md](AUDIT.md); every equation verified in [MATH_AUDIT.md](MATH_AUDIT.md)).

## 3. Methods

- **LSTM baseline** — station-independent 2-layer LSTM (hidden 64), applied per station ([src/models/lstm_baseline.py](../src/models/lstm_baseline.py)).
- **Inductive ST-GNN** — GAT spatial layers + temporal mixing; parameter count independent of `|V|`, so one
  model serves any deployment size. Two temporal encoders compared: **TCN** (dilated causal conv,
  [src/models/stgnn_gat.py](../src/models/stgnn_gat.py)) and **GRU** (per-node recurrent, `STGNN_GAT_GRU`).
- **Transfer learning (PT-FT)** — pre-train on source, fine-tune on d% of the target
  ([src/train_lstm.py](../src/train_lstm.py), [src/train_gnn_tl.py](../src/train_gnn_tl.py)).
- **Virtual sensing** — split nodes into k sensored / N−k unsensored; mask the PM2.5 input at unsensored nodes
  (meteorology available everywhere); train with masked reconstruction; **evaluate at the unsensored nodes**
  ([src/train_gnn_vsense.py](../src/train_gnn_vsense.py)). LSTM baseline: met-only (no PM2.5 input), since a
  station-independent model has no neighbours to borrow from ([analysis/lstm_vsense_baseline.py](../analysis/lstm_vsense_baseline.py)).
- **GNN-PINN** — a soft graph advection–diffusion–reaction residual added to the loss ([PINN_PHYSICS.md](PINN_PHYSICS.md)).

## 4. Results

### 4.1 Source-only forecasting — LSTM ≥ GNN

| City (`\|V\|`) | LSTM | GNN (GAT+TCN) | GNN (GAT+GRU) |
|---|---|---|---|
| Delhi (40) | **0.850** | 0.834 | 0.830 |
| Kolkata (10) | **0.863** | 0.824 | 0.815 |
| Guwahati (4) | **0.832** | 0.831 | 0.809 |

The GNN does not beat the LSTM, and **swapping the GNN's temporal encoder (TCN→GRU) does not help** (it is
slightly worse). The deficit is therefore *not* temporal capacity — spatial coupling simply adds no forecast
value for short-horizon PM2.5 at these sparse, far-apart stations.

### 4.2 Transfer (best R² over d ∈ {15,30,45,60}%) — TL works; LSTM-TL ≥ GNN-TL

| Source → Target | LSTM-TL | GNN-TL |
|---|---|---|
| Delhi → Kolkata | **0.857** | 0.818 |
| Kolkata → Delhi | **0.853** | 0.819 |
| Guwahati → Delhi | **0.852** | 0.817 |
| Guwahati → Kolkata | **0.850** | 0.813 |
| Delhi → Guwahati | 0.824 | 0.827 (tie) |
| Kolkata → Guwahati | **0.821** | 0.816 |

Transfer maintains strong accuracy on data-scarce targets (the headline positive), but the **LSTM-TL is the
stronger transfer model** — consistent with §4.1.

### 4.3 Virtual sensing at unsensored nodes — the GNN's genuine advantage

Both models are trained on the **k sensored** stations and scored on the **identical N−k unsensored**
stations (same seed-0 split). The GNN borrows PM2.5 from neighbours via the graph; the station-independent
LSTM can use only local meteorology (no PM2.5 input, no neighbours). R² at the held-out unsensored nodes
(data-only models; [analysis/lstm_vsense_peru.py](../analysis/lstm_vsense_peru.py), `results/gnn_vsense/`):

| Config (predict unsensored) | LSTM met-only @`U` | GNN (neighbours) @`U` | GNN − LSTM |
|---|---|---|---|
| Delhi — 16 sensors → 24 unsensored | 0.612 | **0.770** | **+0.158** |
| Delhi — 8 sensors → 32 unsensored | 0.596 | **0.726** | **+0.130** |
| Kolkata — 4 sensors → 6 unsensored | 0.647 | **0.730** | **+0.083** |

The GNN beats the met-only LSTM at virtual sensing **across every configuration** (+0.08 to +0.16 R²) — a
clear, consistent advantage that a station-independent model architecturally cannot match, because only the
graph lets a node inherit signal from its neighbours. This is the GNN's distinct, demonstrable contribution.
*(This is the rigorous per-`U` comparison: an earlier approximation that scored the LSTM on the full test set
inflated it to ~0.67–0.76 and suggested a Kolkata tie; training on the sensored set and scoring on the exact
unsensored set removes that artifact and the GNN wins everywhere.)*

### 4.4 Physics-informed prior — a rigorous negative

Adding the advection–diffusion residual **never produces a robust, significant improvement** across four
regimes (forecasting at full density; forecasting under node-dropping; single-seed virtual sensing;
multi-seed virtual sensing). Multi-seed ablation (Kolkata k=4, n=5): ΔR²_both = +0.0034 ± 0.0048, **p≈0.19
(not significant)**; advection-only ≈ diffusion-only (paired p≈0.88), and on Delhi advection-only *hurts*.
Any marginal gain is **generic graph smoothing, not transport physics**. Full write-up:
[PHYSICS_ABLATION_SECTION.md](PHYSICS_ABLATION_SECTION.md); derivation: [PINN_PHYSICS.md](PINN_PHYSICS.md).

## 5. Discussion — what actually helps for sparse PM2.5 sensor networks

- **Transfer learning: yes, decisively.** Data-scarce deployments inherit strong forecasters; this is the
  robust positive result and the practical recommendation.
- **Spatial coupling (GNN): not for forecasting; clearly for virtual sensing.** At 3-hour horizon the
  signal at one station is dominated by its own recent history; neighbouring stations (often tens of km away)
  add little for *forecasting*. But for the genuinely spatial task — estimating PM2.5 where there is no
  sensor — the graph is decisive: the GNN beats a met-only LSTM by +0.08 to +0.16 R² at unsensored nodes.
- **Transport physics: no.** A first-order advection–diffusion prior does not beat data-driven learning or
  generic smoothing here; we report this as a controlled negative rather than omit it.

## 6. Limitations and honest threats to validity

1. Three cities, one country, 3-hour cadence, single horizon — generalization untested elsewhere.
2. The virtual-sensing comparison is the rigorous per-`U` evaluation (LSTM trained on the sensored set, scored
   on the exact unsensored set), but was run for a limited set of (city, k) configurations (Delhi k=8/16,
   Kolkata k=4); Guwahati (4 nodes) is too small to hold out a meaningful unsensored set.
3. The GNN was tuned moderately (two encoders, fixed graph); a larger architecture/graph search *might* close
   the forecasting gap, but two encoders both losing to LSTM make a large reversal unlikely.
4. The physics prior is a *soft* residual; a hard dynamical coupling (neural-ODE/operator) might behave
   differently — left to future work.

## 7. Conclusion

For PM2.5 forecasting over sparse, heterogeneous sensor networks, **transfer learning is the lever that
matters**; a well-tuned station-independent LSTM is a strong baseline that spatial graph coupling does not
beat for forecasting and that transport physics does not improve. The graph's distinct, honest value is
**virtual sensing** — estimating PM2.5 at unsensored locations — a capability the LSTM lacks, with a clear and
consistent advantage (+0.08 to +0.16 R²) across all tested deployments. The contribution of this study is a careful, well-controlled
accounting of *what helps and what does not*, with two ruled-out hypotheses (spatial-coupling-for-forecasting,
transport-physics) reported as the rigorous negatives they are.
