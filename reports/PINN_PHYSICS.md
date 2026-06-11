# The Physics Embedded in Variant E (GNN-PINN-TL) — Governing Equations and Significance

This document fixes *exactly* what physics the PINN term encodes, the mathematical and physical
significance of each piece, and the discrete operators we penalize. It is the spec the
implementation (`src/models/physics.py`, `src/train_gnn_pinn.py`) must match, and the source for
the paper's methods section. Decisions locked with the author (2026-06-10): **full unsteady
advection–diffusion–reaction (ADR)**, evaluated in **physical concentration units with real
Δt = 3 h**, with a **measured, time-varying wind** advection operator.

---

## 1. Governing equation — atmospheric scalar transport

PM2.5 concentration $C(\mathbf x, t)$ [µg/m³] in the boundary layer obeys the mass-conservation
(continuity) equation for a scalar pollutant:

$$\frac{\partial C}{\partial t} \;+\; \nabla\!\cdot(\mathbf u\,C) \;=\; \nabla\!\cdot(K\nabla C) \;+\; S \;-\; \lambda C$$

Under boundary-layer approximations $\nabla\!\cdot\mathbf u \approx 0$ (incompressible flow) and
constant eddy diffusivity $K$:

$$\frac{\partial C}{\partial t} + \mathbf u\!\cdot\!\nabla C = K\nabla^2 C + S - \lambda C$$

| Term | Physical significance | Mathematical character |
|------|----------------------|------------------------|
| $\partial_t C$ | local storage (accumulation/depletion) | first-order time derivative |
| $\mathbf u\!\cdot\!\nabla C$ | **advection** — wind transports pollution upwind→downwind (e.g. crop-residue smoke into Delhi) | first-order, hyperbolic, **directional / antisymmetric** |
| $K\nabla^2 C$ | **turbulent diffusion** — eddies mix high→low concentration | second-order, parabolic, **smoothing / dissipative** |
| $S$ | **emissions** (traffic, industry, biomass) | non-negative source forcing |
| $\lambda C$ | **deposition / loss** — dry+wet removal, settling, first-order chemistry | linear sink, $\lambda>0$ |

This is the Eulerian dispersion model underlying operational chemical-transport models (CMAQ,
WRF-Chem, CAMx). Grounding the regularizer in this equation — rather than a generic graph-smoothness
penalty — is what makes the method defensibly *physics-informed*, and it is the same advection +
diffusion operator family used by PM2.5-GNN, AirPhyNet, and TransNet (which we transcend via transfer
and the sensor-sparsity framing).

---

## 2. State variable — climatology-residual concentration in physical units

We evaluate the ADR on $C'_i(t) = C_i(t) - \bar C_i(t)$ [µg/m³], the deviation of PM2.5 from the
per-(month, hour) seasonal–diurnal climatology $\bar C$ (fit on the train slice; `utils._compute_climatology`).

**Why this state, not raw $C$:** it satisfies the author's "physical units, real Δt" requirement
*and* removes the quasi-steady mean emission pattern. Mean emissions and their long-run
transport/deposition balance are absorbed into $\bar C$, so on the residual the source is negligible:
$$S' \approx 0.$$
This is the principled justification for dropping the (unobservable) emission term — without it, a raw-µg/m³
residual would be systematically violated wherever emissions are strong (all of Delhi).

The network already predicts exactly this residual (target z-scored on $C-\bar C$), so:
- Predicted next-step residual: $\hat C'_i(t{+}\Delta t) = \sigma_y\,\hat y_i + \mu_y$  (inverse target z-score; **no** climatology add-back).
- Observed current residual: $C'^{\mathrm{obs}}_i(t) = \sigma_y\,y^{\mathrm{prev}}_i + \mu_y$, with $y^{\mathrm{prev}}$ the z-scored target at the last history step. Both pass through the *same* `target_scaler`, so units are consistent µg/m³.

---

## 3. Continuous operators → operators on the sensor graph

Point sensors at irregular locations ⇒ discretize on the graph $G=(V,E)$ (never a collocation grid;
this is the FieldFormer argument for sparse/irregular data).

**Diffusion → graph Laplacian (static).** The discrete analogue of $-\nabla^2$ is $L=D-W$ with
Gaussian-distance weights $W_{ij}=\exp(-d_{ij}^2/2\sigma^2)$ (`knn_graph`). So $K\nabla^2C \mapsto -K\,LC$, i.e.
$$(L\,C')_i = \sum_{j} W_{ij}\,(C'_i - C'_j).$$
$L\mathbf 1 = 0$ ⇒ diffusion conserves total mass exactly; $L$ is PSD (dissipative). Spectral-graph
theory makes this the rigorous discrete diffusion (cf. graph heat equation; GRAND).

**Advection → measured-wind upwind flux divergence (time-varying).** Read measured wind at the last
history step: speed $v_i(t)$ (feature WS) and direction from $(\mathrm{Sin\_WD},\mathrm{Cos\_WD})$,
$\phi_i^{\mathrm{WD}} = \operatorname{atan2}(\sin,\cos)$. Wind blows *toward* $\phi_i^{\mathrm{flow}} = \phi_i^{\mathrm{WD}}+180^\circ$
(meteorological WD is the direction it comes *from*). For edge $i\to j$ with geographic bearing $\theta_{ij}$ and
distance $d_{ij}$ (haversine), the along-edge outflow speed is
$$u_{ij}(t) = \big[\,v_i(t)\,\cos(\phi_i^{\mathrm{flow}} - \theta_{ij})\,\big]_+ \big/\, d_{ij},$$
and the first-order **upwind** advection operator is
$$(\mathcal A(t)\,C')_i = \sum_{j} u_{ij}(t)\,(C'_i - C'_j).$$
Node $i$ is pulled toward its downwind neighbours in proportion to the wind component toward them and
the concentration difference — a first-order, directional (upwind-weighted) approximation of
$\nabla\!\cdot(\mathbf uC)$. **Unlike the diffusion Laplacian, this directed-difference operator is _not_
mass-conserving** ($\sum_i(\mathcal A C')_i \neq 0$ in general, since the wind graph's in- and out-degrees
differ): it is a soft *directional transport prior*, not a finite-volume flux scheme. A fully conservative
(antisymmetric edge-flux) discretization is left to future work; for a soft residual penalty the directional
structure is what matters. The operator is **driven by the sensors' own measured wind** — a genuine
"physics-informed by the IoT network" property, not an assumed prevailing wind.

---

## 4. The PINN residual (forward Euler) and loss

Explicit one-step (forward-Euler) discretization with Δt = 3 h; spatial operators act on the
**observed** current field (fully known at all nodes at $t$), and only $\hat C'(t{+}\Delta t)$ is the prediction:

$$\boxed{\,R_i \;=\; \frac{\hat C'_i(t{+}\Delta t) - C'^{\mathrm{obs}}_i(t)}{\Delta t} \;+\; \alpha\,(\mathcal A(t)\,C'^{\mathrm{obs}})_i \;+\; K\,(L\,C'^{\mathrm{obs}})_i \;+\; \lambda\,C'^{\mathrm{obs}}_i\,}$$

$$\mathcal L_{\mathrm{phys}} = \frac{1}{B\,|V|}\sum_{b,i} \Big(\frac{R_{b,i}}{\sigma_y}\Big)^2, \qquad \mathcal L_{\mathrm{total}} = \mathcal L_{\mathrm{data}} + \lambda_{\mathrm{phys}}\,\mathcal L_{\mathrm{phys}}.$$

The residual is **non-dimensionalized by the characteristic concentration scale** $\sigma_y$ (the target std). $R$ is built in physical µg/m³ (so $\alpha,K,\lambda$ stay physical), but the *penalty* is scaled to be comparable to the dimensionless z-scored data MSE — otherwise $\mathcal L_{\mathrm{phys}}\sim10^3$ µg²/m⁶ swamps $\mathcal L_{\mathrm{data}}\sim0.4$ and the fine-tune minimizes transport residual instead of forecast error. This also makes $\lambda_{\mathrm{phys}}$ **city-invariant** (Delhi and Guwahati have very different $\sigma_y$).

**Interpretation.** The physics term requires that the one-step change the model *forecasts* equals the
advective + diffusive + deposition tendency of the *observed* current field. It is a soft, differentiable,
explicit transport-consistency constraint.

**Significance for "more value from fewer sensors."** Where nodes are sparse the spatial field is
under-determined; $\mathcal L_{\mathrm{phys}}$ forces the predicted field into the transport-smooth (low graph-bandwidth)
subspace that few nodes can reconstruct (graph-sampling theory, §4 of `SENSOR_EFFICIENCY_PINN_LITERATURE.md`).
The prior should therefore help most on the sparsest deployment (Guwahati, $|V|=4$) and the lowest label budget.

**Learnable physical coefficients** (all softplus-parameterized to enforce positivity — the physical prior
that diffusion smooths and deposition decays, neither running backwards):
- $\alpha \ge 0$ — effective advection scale (absorbs km / m·s⁻¹ / h unit constants),
- $K \ge 0$ — effective eddy diffusivity,
- $\lambda \ge 0$ — deposition / relaxation rate.

These are *identified from data*; reporting the learned $\alpha, K, \lambda$ (and per-city variants as an
ablation) is a paper result in itself.

---

## 5. Assumptions and honest caveats (for the limitations section)

1. **$S' \approx 0$** — emissions are quasi-steady and absorbed into the climatology; valid for the
   residual at the 3-h horizon, weaker during sharp emission events (Diwali, fires).
2. **Incompressible flow** $\nabla\!\cdot\mathbf u\approx 0$ — standard boundary-layer approximation.
3. **First-order upwind advection** — stable but diffusive, and the graph-difference form is
   **not mass-conserving** (only the diffusion Laplacian satisfies $\mathbf 1^\top L = 0$); it acts as a
   soft *directional transport* prior rather than a conservative flux scheme.
4. **Constant (scalar) $K,\lambda$ per run** — homogeneous turbulence/deposition; per-city or
   wind-dependent $K$ is a future refinement.
5. **Δt and edge lengths** are carried as physical constants (3 h, km); residual unit constants are
   absorbed by the learnable $\alpha,K,\lambda$, so reported coefficients are *effective* (not raw SI).
6. **Single horizon** (one 3-h step) → a single forward-Euler step; multi-step rollouts are future work.

---

## 6. What this changes vs the first draft

The initial `physics.py` used a *steady-state, static-wind, mass-non-conserving* smoothing term
($I-\mathrm{rownorm}(A_{\mathrm{wind}})$). This spec replaces it with the **full unsteady forward-Euler ADR
residual in physical (climatology-residual µg/m³) units, with a measured-wind upwind advection operator and a
mass-conserving graph Laplacian diffusion** — matching the author's locked choices and the governing physics above.
