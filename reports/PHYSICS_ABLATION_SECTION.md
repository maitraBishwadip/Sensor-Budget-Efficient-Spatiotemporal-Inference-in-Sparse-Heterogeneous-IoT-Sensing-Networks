# Manuscript subsection (draft) — Physics-informed ablation: a ruled-out hypothesis

*Drop-in for the results/discussion of the IoT-J manuscript. Full derivation lives in
`reports/PINN_PHYSICS.md` (→ supplementary). Numbers from `results/gnn_pinn/*` and
`results/gnn_vsense/vsense_ablation.json`. Citations use the keys already in `reports/REFERENCES.md`
plus the physics block in `reports/SENSOR_EFFICIENCY_PINN_LITERATURE.md` §9.*

---

## X. Does a transport-physics prior explain sensor efficiency? A negative result

### X.1 Hypothesis

The transfer results (§[transfer]) show that a sparse deployment inherits strong accuracy from a
data-rich one — a four-node network is forecast at R²≈0.83 (§[Guwahati]). A natural mechanistic
explanation is *physics*: fine particulate matter obeys an advection–diffusion–reaction (ADR)
transport law, so if few sensors suffice it may be because the concentration field is low-dimensional
under transport, and explicitly encoding that law should further improve estimation — most of all where
sensors are sparse. This is the premise of physics-informed learning [Raissi 2019; Karniadakis 2021]
and of recent physics-guided air-quality GNNs [AirPhyNet; TransNet]. We test it directly, and find it
does **not** hold for our setting: a transport prior does not improve estimation beyond generic graph
smoothing, and its genuinely physical (wind-advection) component is neutral-to-harmful.

### X.2 The prior

We add a soft ADR residual on the predicted climatology-residual field `Ĉ′` (in µg/m³), discretized on
the sensor graph (operators live on edges, not a collocation grid — the regime where classic PINNs
degrade on sparse/irregular data [FieldFormer]):

R_i = (Ĉ′_i(t+Δt) − C′_i(t))/Δt  +  α(𝒜(t)C′)_i  +  K(LC′)_i  +  λ C′_i ,   L_phys = ⟨(R_i/σ_y)²⟩,

with **advection** 𝒜(t) a first-order upwind operator built from the **measured** wind (available at all
nodes), **diffusion** L the mass-conserving graph Laplacian, and α,K,λ ≥ 0 learnable effective
coefficients. The training loss is `L_data + λ_phys·L_phys`; `λ_phys = 0` recovers the data-only model.
Full derivation: supplementary / `PINN_PHYSICS.md`. The construction parallels physics-aware graph
operators for sparsely-observed dynamics [PA-DGN] and graph-PDE diffusion [GRAND].

### X.3 Protocol

We evaluate the prior in four regimes of increasing rigor, all on the identical encoder and splits:
(i) **forecasting at retained sensors**, full density; (ii) **forecasting under node-dropping** (the
sensor-budget sweep, target stripped to k∈{4,8,16,…} active nodes); (iii) **virtual sensing** —
estimating PM2.5 at the held-out *unsensored* nodes (PM2.5 input masked there, meteorology retained);
and (iv) **virtual sensing with multi-seed significance and a wind-vs-smoothness ablation**
(advection-only vs diffusion-only vs both), reporting paired per-split ΔR² over five (Kolkata) / three
(Delhi) random sensor placements.

### X.4 Results

**Forecasting (i–ii).** Adding the prior never helps. At retained sensors it monotonically *reduces*
accuracy as `λ_phys` grows (up to −0.03 R² on the densest target); under node-dropping the degradation
is merely milder on sparser graphs (e.g. −0.005 vs −0.011 R² at k=4 vs k=10) but is negative at every
budget. This is consistent with the regime being *data-determined*: with full PM2.5 supervision the
field is already identified, so a soft PDE constraint only restricts the hypothesis space.

**Virtual sensing (iii–iv).** Estimating unsensored nodes is the only regime where the field is genuinely
under-determined, and here the prior yields a small **but non-significant** mean improvement
(Kolkata, k=4: ΔR² = +0.0034 ± 0.0048 over five placements, t=1.6, **p=0.19**). The ablation attributes
this entirely to generic smoothing:

| Term (λ_phys=0.2) | Kolkata k=4 (n=5) ΔR² (p) | Delhi k=16 (n=3) ΔR² (p) |
|---|---|---|
| both (adv+diff) | +0.0034 (0.19) | +0.0013 (0.29) |
| advection-only | +0.0023 (0.14) | **−0.0019 (0.31)** |
| diffusion-only | +0.0024 (**0.05**) | +0.0019 (0.55) |
| **advection − diffusion (paired)** | −0.0001 (**0.88**) | −0.0038 (t=−2.44, 0.14) |

Only **diffusion** approaches significance (p=0.05); the **advection** (wind) term is statistically
indistinguishable from diffusion on Kolkata (paired p=0.88) and *inferior* on Delhi (advection−diffusion
t=−2.44). In other words, the marginal benefit is **generic graph smoothing — equivalent to a Tikhonov /
Kriging prior — not transport physics**.

### X.5 Conclusion

For short-horizon PM2.5 estimation over sparse, heterogeneous sensor networks, an explicit
advection–diffusion prior does not improve accuracy beyond graph smoothing, and its wind-driven
advection component — the genuinely physics-informed part — provides no reliable benefit and can degrade
results. We therefore attribute the observed sensor efficiency to the **learned cross-deployment
transfer** (§[transfer]), not to an explicit transport mechanism. This rules out a natural and
frequently-raised hypothesis and localizes the source of the gains. The finding is consistent with
physics-informed theory — priors help under *under-determination*, whereas here supervised data and
graph coupling already identify the field — and suggests that, for this task, transferable data-driven
spatial structure subsumes the first-order transport operator. Whether a *hard* dynamical coupling
(neural-ODE/operator-style, as in [AirPhyNet]) rather than a soft residual would behave differently is
left to future work.
