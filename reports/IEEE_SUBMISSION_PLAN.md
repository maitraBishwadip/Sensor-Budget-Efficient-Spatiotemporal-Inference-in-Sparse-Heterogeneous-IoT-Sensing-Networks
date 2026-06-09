# IEEE TGRS Submission Plan — GNN-TL PM2.5 Paper (finalized)

**Venue:** IEEE Transactions on Geoscience and Remote Sensing (TGRS) · **hard cap 10 pp, two-column.**
**Source prose:** [reports/PAPER_DRAFT.md](PAPER_DRAFT.md) (publication-grade; work = conversion + triage, not new writing).
**Template:** [IEEE-Transactions-LaTeX2e-templates-and-instructions/](../IEEE-Transactions-LaTeX2e-templates-and-instructions/).
**This revision adds:** (1) a TGRS-grade *Study Area / Instruments / Data* and *Methodology* structure, (2) a **finalized 6-figure visualization plan**, designed fresh to TGRS conventions.

---

## A. How TGRS papers are structured (and what we adopt)

TGRS is an empirical geoscience/remote-sensing journal. Reviewers expect, in this order:

1. **Introduction** — problem + gap + contributions (numbered).
2. **Study Area and Data** — a dedicated, rigorous section: *where*, *what instruments*, *what measurements*, *how processed*. **This is the section we are currently weakest on** and must be strengthened (see §B).
3. **Methodology** — the *proposed method*, anchored by a full-width framework figure + per-module detail + equations.
4. **Experimental Setup** — splits, baselines, hyperparameters, metrics, implementation.
5. **Results and Analysis** — quantitative tables + figures, **including a predicted-vs-observed / forecast-fidelity figure** (geoscience reviewers want to see the model actually track the signal).
6. **Discussion / Limitations / Conclusion.**

Our 10 sections map onto this cleanly; the main change vs. the markdown draft is **promoting Data+Instruments to a first-class section** and **tightening Methodology into a clean module-by-module spine**.

---

## B. Section III — *Study Area, Instruments, and Dataset* (make it airtight)

This is the part you asked to make "completely clear." Required content, sub-by-sub:

**III-A Study area.** Three CPCB cities spanning distinct airshed regimes — Delhi (Indo-Gangetic Plain, winter inversion, ~100 µg/m³ annual mean), Kolkata (coastal Gangetic delta, humid tropical), Guwahati (Brahmaputra valley, pre-monsoon dust + monsoon washout). Geographic context → **Fig. 1**. State WGS-84 datum and per-city bounding boxes (derivable from [metadata.json](../dataset/processed/metadata.json)).

**III-B Instruments and data acquisition.** *← content gap to fill.* Specify:
- Network: CPCB **Continuous Ambient Air Quality Monitoring (CAAQM)** stations.
- **Sensor / measurement principle** for each variable — PM2.5 (beta-attenuation monitor, BAM-1020-class, µg/m³); AT/RH (thermo-hygrometer); WS/WD (anemometer/vane). Give nominal accuracy & sampling. *(The draft never states the instrument; TGRS requires it.)*
- Acquisition: source portal, download window, native cadence → aggregated to **3-hourly**, timezone (IST), units, QA/QC (CPCB validation flags).
- Coverage: **Delhi 39, Kolkata 10, Guwahati 4** stations. **Fix the 39-vs-40 inconsistency** — metadata + thesis baseline say 39; the draft/RESULTS say 40. Pick the true number and use it everywhere.
- Period & length: Delhi Jan 2021–Dec 2022 (5 840 × 3-h records/station); Kolkata & Guwahati 2023 (2 920 each). Per-station record counts are in metadata.

**III-C Preprocessing & feature engineering.** IDW spatial interpolation + Kalman temporal smoothing → causal imputation (ffill + train-period mean fallback) → **F = 14 features**: PM2.5, AT, RH, WS, sin/cos WD, sin/cos hour, sin/cos month, 4 season one-hots (exact list in metadata `feature_cols`). StandardScaler & per-(month,hour) climatology fit on **train mask only**; climatology-residual target.

**III-D Graph construction.** k-NN (k = 3) on haversine distance, Gaussian edge weight `w_ij = exp(−d_ij²/2σ²)`, σ = 5 km; per-city |V|/|E| = 39/117·, 10/30, 4/12 (recompute |E| for the corrected |V|). Visualized in **Fig. 1** insets.

**III-E Problem formulation.** `f_θ : ℝ^{H×N×F} → ℝ^{N}`, H = 8 (24 h) → horizon 1 (+3 h); θ shape independent of N (inductive).

---

## C. Section IV — *Methodology* (clean module spine)

| Sub | Content | Equations | Figure |
|---|---|---|---|
| IV-A | Framework overview | — | **Fig. 2** |
| IV-B | Inductive ST-GNN encoder (TCN, edge-weighted GAT, mean+max readout) + inductivity (param-shape table) | (1) forecasting map, (4) GAT logit, (5) softmax+aggregate, (6) readout, (7) edge weight | **Fig. 3** |
| IV-C | Station-independent LSTM-TL baseline | (2) LSTM gates, (3) head | — |
| IV-D | Stage 0 source pre-training | — | — |
| IV-E | Stage 1 PT-FT (layer-freeze recipe) | — | — |
| IV-F | Stage 2 Graph-DANN (GRL, 3-way discriminator, λ schedule) | (8) joint loss, (9) λ(p) | **Fig. 4** |
| IV-G | Three-way verification protocol | Algorithm 1 | (proof → Fig. 5) |

Section V (Experimental Setup) carries splits, d%-subsampling, hyperparameter table, metrics (R²/MAE/RMSE/MAPE in raw µg/m³), CPU-only implementation, and the 1-paragraph leakage-audit summary (full audit → supplementary).

---

## D. FINALIZED VISUALIZATION PLAN — 6 high-impact figures

Designed fresh to TGRS norms. **3 double-column (`figure*`) + 3 single-column** keeps the float load within a 10-page budget. Real coordinates/metrics already exist for all but the forecast-fidelity panels (Fig. 6), which need saved predictions.

### Fig. 1 — Study area & monitoring-network heterogeneity  · `figure*` (double) · **NEW**
The canonical TGRS opener. A geographic map of northern India locating Delhi, Kolkata, Guwahati (real lat/lon), with **three per-city insets** showing actual CPCB station positions + k-NN (k=3) edges, each labelled |V| = 39/10/4. Annotate the **~10× density gap** and per-city mean PM2.5 to motivate regime differences. *Replaces* the current schematic `fig1_motivation` + `fig2_city_graphs` with one real map. Build: state/coastline outline (cartopy/contextily, or a light India shapefile) + scatter from metadata; offline fallback = clean lat/lon panels with state outline + scale bar.
**Why it's impactful:** one figure delivers study-area context, instrument geolocation, *and* the core problem (heterogeneous |V|).

### Fig. 2 — Proposed cross-city transfer framework  · `figure*` (double) · **UPGRADE**
End-to-end "proposed method" overview: data & instruments → preprocessing (F=14, causal impute, climatology-residual, interleaved 70/15/15) → per-city k-NN graph → **shared inductive ST-GNN encoder** → two parallel transfer branches (**Stage 1 PT-FT** / **Stage 2 Graph-DANN**) → three-way verification → target-test evaluation. Declutter the existing `fig_pipeline`; make the two branches visually parallel.
**Why:** every TGRS method paper has this; it orients the reader before the module detail.

### Fig. 3 — Inductive ST-GNN encoder & |V|-independence  · single col · **MERGE/UPGRADE**
Layer stack `TCN₁→GAT×2→TCN₂→LayerNorm→Linear` with tensor shapes `[B,H,N,F]→[B,N]`; **inset (a)** edge-weighted GAT message-passing (node j aggregates α-weighted neighbours), **inset (b)** size-invariant mean+max readout into the same head for any N; banner: *"no parameter's shape depends on |V| → the same ~25k-θ runs on N = 39/10/4."* Merges `fig3_encoder_blocks` + the useful half of `fig_gnn_blocks`.
**Why:** makes the central novelty (inductivity) visually undeniable in one panel.

### Fig. 4 — Graph-DANN adversarial domain adaptation  · single col · **UPGRADE**
Round-robin 3-city minibatch → shared encoder → readout `z` → {forecast head MSE} and {GRL ×(−λ) → 3-way city discriminator CE}; λ(p) warm-up inset. The headline Stage-2 contribution. Upgrade of `fig4_graph_dann`.
**Why:** the paper's novel mechanism deserves one clean schematic.

### Fig. 5 — Transfer performance & three-way verification  · `figure*` (double) · **UPGRADE + NEW panel**
**(a)** heatmap small-multiples — zero-shot vs scratch vs transfer across the 24-cell grid (visually proves transfer > both, 24/24 for PT-FT); **(b)** **R²-vs-data-fraction curves** (d = 15/30/45/60 %) for representative pairs with transfer/scratch/zero-shot lines — the data-efficiency story. Merges `fig5_stage1_heatmaps` with a new data-fraction plot (data already in `make_figures.py` arrays).
**Why:** fuses the verification rigor (contribution C3) with the data-scarcity narrative.

### Fig. 6 — Forecast fidelity: predicted vs. observed  · single col (or `figure*`) · **NEW** *(the highest-value addition)*
**(a)** predicted-vs-observed density/scatter for the hero cell (Delhi→Kolkata @30 %), 1:1 line, annotated R²/MAE; **(b)** observed-vs-predicted PM2.5 **time series** for a representative target station over a test window, showing captured pollution peaks. Optional **(c)** cross-method bars (LSTM/Stage1/Stage2 @30 %).
**Why:** this is what the current figure set *lacks* and what TGRS reviewers most want — proof the model tracks real PM2.5, not just aggregate R². **Dependency:** needs saved prediction arrays; check `results/` for dumped preds, else a ~10-min eval run on the hero cell exports them.

**Dropped from main body** (→ supplementary): `fig_lstm_arch` (LSTM detail — describe in text + eqs), the standalone `fig6_headline_bars` (folded into Fig. 6c or supp), full per-cell heatmaps beyond Fig. 5.

---

## E. Length triage (unchanged target: ≤ 10 pp)

- **Figures:** 6 (above). **Tables main body ≈ 6:** dataset+graph (merge), parameter-shapes, hyperparameters, source-only, consolidated transfer @30% (LSTM/S1/S2/best + verify summary), positioning. **→ supplementary:** full 24-cell grids, per-cell verification A1–A3, audit table, 7-fix ablation.
- **Prose −~30%:** compress §5.5.5 (7 fixes → pointer), §6.5 audit (→ 1 para), §8 discussion (→ 3 points), §9 limits (→ short list).
- Page budget ≈ Intro 1.3 / Related 0.8 / Data 0.9 / Method 1.7 / Setup 0.7 / Results 1.8 / Disc+Concl 1.0 / Refs 1.0 ≈ **10 pp**. Validate by actual compile.

---

## F. IEEE house-style conversions (applied throughout)

Numbered `\cite`/BibTeX (`IEEEtran.bst`) replacing ~55 author-year cites · code-fence math → numbered `equation`/`align` (eqs 1–9) · Algorithm 1 → `algorithm`/`algorithmic` float · markdown tables → `table`/`table*`, caption-before, title-case, Roman numerals · figures → `figure`/`figure*`, image→`\caption`→`\label`, "Fig. N", **use the PDF (vector) figure files** · abstract ≤ 250 w, no math symbols/citations · `\IEEEPARstart` drop cap · Refs & Acknowledgment unnumbered · soft `\ref`/`\eqref`. TGRS keywords (geoscience-leaning): *air quality, PM2.5 forecasting, spatio-temporal graph neural networks, domain adaptation, transfer learning, environmental monitoring networks.*

---

## G. Content gaps to close before submission (flagged)
1. **Instrument specifications** (III-B) — add CPCB sensor model/measurement principle/accuracy. *Content addition.*
2. **39 vs 40 Delhi stations** — reconcile across draft, RESULTS, figures (metadata = 39).
3. **Fig. 6 predictions** — confirm saved predicted-vs-observed arrays exist in `results/`, else run a short eval export.
4. **TGRS scope framing** — foreground the ground-monitoring-network geoscience angle in abstract/intro/cover letter (TGRS skews satellite RS).

---

## H. Execution phases (on approval)
1. Scaffold `paper/main.tex` (preamble, title/author/`\thanks`, de-symbolized abstract ≤250 w, keywords) + `refs.bib` skeleton → compiles empty.
2. Sections III–IV with eqs 1–9 + Algorithm 1; wire Fig. 2–4.
3. Render Fig. 1, 5, 6 (fresh) ; build Sections I–II, V–X + the 6 main tables.
4. Convert all citations to `\cite`; finalize `refs.bib`.
5. Fit pass: compile, **measure page count**, push overflow to `supplementary.tex`, trim to ≤ 10 pp.

*Plan saved for review. Figures can be regenerated fresh — the metadata + result arrays needed for Fig. 1–5 are present; only Fig. 6 needs a prediction export.*
