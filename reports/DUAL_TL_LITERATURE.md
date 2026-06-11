# Literature Review — Dual Transfer Learning for Cross-City PM2.5 Regression

Purpose: ground the proposed "dual transfer learning" extension in the literature **before** committing experiments, and decide what "dual" should concretely mean for our *regression* problem (PM2.5 forecasting), not the classification setting most DA methods were built for.

---

## 1. The question
Graph-DANN (our Stage 2) aligns only the **marginal** embedding distribution `P(z)` and empirically **ties** PT-FT (mean ΔR² = 0.0009); a linear probe shows it only weakly reduces city-separability (0.90→0.84). Is there a principled "dual" technique that aligns more than the marginal and actually improves accuracy? The literature gives a clear, and partly cautionary, answer.

## 2. Lineage: what "dual transfer learning" means
- **Dual Transfer Learning (DTL)** — Long, Wang, Ding, Cheng, Zhang, Wang, *SDM 2012*. The namesake. Key idea: transfer methods usually bridge domains through **either** the marginal `P(x)` **or** the conditional `P(y|x)`; DTL learns **both simultaneously** and exploits the *duality* between them, arguing a single bridge is sub-optimal. → This is the conceptual root of our "dual = marginal + conditional" framing.
- Distinct from the NLP "dual learning" line (e.g., *Dual Transfer Learning for NMT with Marginal Distribution Regularization*, AAAI 2019), which is about forward/backward task duality — **not** what we mean.

## 3. Marginal vs. conditional vs. joint alignment (the method families)
- **DANN** (Ganin & Lempitsky 2015; Ganin et al. JMLR 2016) — adversarial **marginal** feature alignment via gradient reversal. *Our Stage 2 is the graph version of this.*
- **JAN** — Joint Adaptation Networks (Long et al., *ICML 2017*) — aligns the **joint** distribution of multiple layers' activations via a joint MMD (JMMD). First-class "more-than-marginal" alignment.
- **CDAN** — Conditional Adversarial Domain Adaptation (Long, Cao, Wang, Jordan, *NeurIPS 2018*). Conditions the domain discriminator on classifier predictions via a **multilinear map** `f ⊗ g` (feature × prediction), capturing the cross-covariance the concatenation misses; adds entropy conditioning for transferability. *Built for classification (multimodal class structure).* → This is the basis of the pilot variant we implemented (`GraphDualDANN`).
- **DSAN** — Deep Subdomain Adaptation Network (Zhu, Zhuang et al., *IEEE TNNLS 2021*). Non-adversarial **class-conditional (subdomain)** alignment via **LMMD** (Local MMD), weighting samples by (pseudo-)class. Shows global alignment "obscures discriminative structure"; subdomain alignment fixes it.

## 4. The regression caveat — the most important finding ⚠️
Almost all of §3 targets **classification**. Our task is **regression**, and two papers show this matters fundamentally:
- **RSD** — Representation Subspace Distance for Domain Adaptation **Regression** (Chen, Wang, Wang, Long, *ICML 2021*). Central result: *"classification is robust to feature scaling but **regression is not**, and aligning distributions of deep representations **alters feature scale and impedes** domain adaptation regression."* RSD instead aligns the **orthogonal bases / subspaces** of the representation (Grassmann-manifold geometry), which is scale-free, plus a bases-mismatch penalty.
- **de Mathelin et al. 2020** (already cited) — adversarial DA in regression underperforms unless the discriminator loss is down-weighted (α_d ≪ 1).

**Implication:** our Graph-DANN tie is *exactly what theory predicts* — marginal distribution alignment is the wrong tool for regression. A "dual" technique that simply **adds another distribution-alignment term** (plain CDAN-style conditional adversary) may *also* fail to help, because it is still distribution alignment. The regression-correct move is **subspace / geometry alignment (RSD)** and/or **conditioning that is scale-robust**.

## 5. Cross-city / spatio-temporal / air-quality precedents (TGRS relevance)
- **DASTNet** (Tang et al., *CIKM 2022*, already cited) — adversarial cross-city ST transfer for traffic; marginal alignment on (homogeneous) road graphs.
- **Cross-Mode Knowledge Adaptation via Domain-Adversarial GNNs** (bike-sharing, arXiv 2022) — domain-adversarial GNN for cross-domain ST demand; another marginal-adversarial graph precedent.
- **Robust ST Traffic Forecasting w/ Reinforced Dynamic Adversarial Training** (2023) — adversarial training adapted to ST (not DA per se, but shows static adversarial DA doesn't transfer directly to ST).
- **Multi-city PM2.5 GNNs (2023–2025):** Dynamic Global–Local ST Graph for multi-city PM2.5 (*Remote Sensing*, 2025); spatially-attentive cluster GNN for city PM2.5 (2023). These do multi-city modelling but **not** transfer across heterogeneous `|V|`, and **none** do conditional/dual alignment for regression. → confirms our gap is open.

## 6. Gap & positioning
No prior work combines: (a) cross-city transfer under order-of-magnitude `|V|` heterogeneity, (b) alignment **beyond the marginal** that is **regression-appropriate**, on (c) PM2.5. "Dual TL" is therefore a genuine contribution **if** we choose a regression-correct second alignment, not a naive conditional adversary.

## 7. Implications for the experiment design (grounded plan)
Run a controlled ladder of alignment objectives on the identical 24-cell protocol, so the comparison is one-to-one:
1. **PT-FT** (no alignment) — have it.
2. **Graph-DANN** (marginal adversarial) — have it (ties; the negative control that motivates the rest).
3. **Dual-CDAN** (marginal + conditional **adversarial**, `z⊗ĝ`) — **DROPPED (2026-06-11): the Dual-CDAN experiment (Variant C) was never completed (12/24 cells) and its code (`train_gnn_dual.py`, `GraphDualDANN`, `results/gnn_dual/`, `models/gnn_dual/`) was removed.** Hypothesis from §4 (may tie, still distribution alignment) is left untested; RSD (variant 4) is the retained dual method.
4. **Dual-RSD** (marginal alignment **+ RSD subspace alignment**, the regression-correct term) — the variant §4 predicts should actually move accuracy. **Recommended as the headline dual method.**
5. *(optional)* **Dual-LMMD** (forecast-conditioned LMMD, DSAN-style adapted to regression by binning the predicted PM2.5 into quantile "subdomains").

Decision rule: keep whichever of (3)–(5) passes 3-way verification **and** improves mean ΔR² over Graph-DANN; if none beats PT-FT, report the honest null and frame dual-TL as "conditional/subspace alignment does not overcome the regression-scale obstacle here," which is itself a citable finding consistent with RSD.

## 8. New citations to add to the manuscript
- Long et al., *Dual Transfer Learning*, SDM 2012 — names the paradigm.
- Long, Cao, Wang, Jordan, *Conditional Adversarial Domain Adaptation (CDAN)*, NeurIPS 2018.
- Chen, Wang, Wang, Long, *Representation Subspace Distance for DA Regression (RSD)*, ICML 2021 — **the regression caveat**.
- Long et al., *Joint Adaptation Networks (JAN)*, ICML 2017.
- Zhu, Zhuang et al., *Deep Subdomain Adaptation Network (DSAN/LMMD)*, IEEE TNNLS 2021.
- (have) de Mathelin et al. 2020; Tang et al. DASTNet 2022; Ganin et al. 2016.

---

### Sources
- Dual Transfer Learning (SDM 2012): http://web.cs.ucla.edu/~weiwang/paper/SDM12.pdf
- CDAN (NeurIPS 2018): https://arxiv.org/abs/1705.10667
- RSD — DA Regression (ICML 2021): https://proceedings.mlr.press/v139/chen21u/chen21u.pdf ; code https://github.com/thuml/Domain-Adaptation-Regression
- DSAN / LMMD (TNNLS 2021): https://ar5iv.labs.arxiv.org/html/2106.09388
- DASTNet (CIKM 2022): https://arxiv.org/pdf/2202.03630
- Cross-Mode domain-adversarial GNN (2022): https://arxiv.org/pdf/2211.08903
- Multi-city PM2.5 ST graph (Remote Sensing 2025): https://www.mdpi.com/2072-4292/17/16/2750
