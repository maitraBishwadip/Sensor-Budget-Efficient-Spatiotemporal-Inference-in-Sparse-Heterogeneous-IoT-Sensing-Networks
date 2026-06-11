# Mathematical Audit — LSTM → GNN variants → PINN

Verification that every reported equation/claim matches the *implemented* math, end to end. Each row:
the claim, where it lives in code, and the verdict. Two substantive issues were found and are tracked in §3.

Conventions: target is the **z-scored climatology-residual** PM2.5 (`utils.load_city_tensor`); all metrics are
computed in **raw µg/m³** after inverse z-score + climatology add-back (`utils.inverse_transform_target`,
`train_gnn.evaluate_gnn`).

---

## 1. Verified correct

| Component | Claim | Code | Verdict |
|---|---|---|---|
| **Metrics** | R²=1−SS_res/SS_tot, MAE, RMSE, MAPE (ε=1) | `utils.py:48–77` | ✓ standard |
| **Climatology residual** | per-(month,hour) mean on train slice; target = raw−clim; added back at eval | `utils.py:119–144`, `381–392` | ✓ correct, leak-free (train-only fit) |
| **Windowing** | X=[W,H,N,F], target at t+H+horizon−1; split owns the *prediction* step | `utils.py:285–354` | ✓ causal, no future leak |
| **LSTM baseline** | per-station seq→1; out[:,−1]→MLP; applied per station & concatenated | `lstm_baseline.py:44–48` | ✓ matches thesis Variant-A |
| **GAT attention** | e_ij=LeakyReLU(⟨Wx_i,a_src⟩+⟨Wx_j,a_dst⟩)+log w_ij; per-dest softmax; h_j=Σ_i α_ij Wx_i | `stgnn_gat.py:50–85` | ✓ matches PAPER_DRAFT §4.2 eq, additive GAT (Veličković 2018) |
| **TCN** | dilated causal conv, pad (k−1)·d, right-trim to length H | `stgnn_gat.py:88–111` | ✓ causal |
| **SAGE** | h_v=ReLU(W_self h_v + W_neigh·Σw_uv h_u/Σw_uv) | `stgnn_sage.py:30–54` | ✓ weighted mean-aggregator (Hamilton 2017) |
| **Graph build** | haversine; bearing=atan2(...); kNN w=exp(−d²/2σ²); wind w=max(0,cos(θ_w−θ_ij))·exp(−d/decay) | `graph_construction.py` | ✓ standard formulas |
| **GRL** | forward identity, backward −λ·grad | `dann.py:22–36` | ✓ Ganin & Lempitsky 2015 |
| **DANN loss** | L=MSE+α_d·(CE_src+CE_tgt+CE_third); GRL injects −λ into encoder | `train_gnn_dann.py:304–310` | ✓ matches PAPER_DRAFT §5.4 eq |
| **λ schedule** | λ(p)=2/(1+e^{−γp})−1, γ=10, p=step/total | `dann.py:168–171`, `train_gnn_dann.py:172–177` | ✓ Ganin 2016 |
| **CDAN (Variant C)** | multilinear T=z⊗g, g=MLP([mean ŷ,std ŷ]); marginal+conditional GRL | (removed) | ✓ form was correct, but **Variant C / Dual-CDAN REMOVED 2026-06-11** (incomplete experiment: 12/24 cells) |
| **PINN ADR residual** | R=(Ĉ′−C′_obs)/Δt+α·adv+K·diff+λ·C′_obs, signs from ∂C/∂t+u·∇C=K∇²C+S−λC with ∇²↔−L; ÷σ_y | `physics.py PhysicsADR`, `train_gnn_pinn.py` | ✓ signs internally consistent; non-dim by σ_y matches doc |
| **Diffusion mass conservation** | (LC)_i=Σ_j W_ij(C_i−C_j), 1ᵀL=0 | `physics.py apply_diffusion` | ✓ verified numerically (Σ_i(LC)_i≈0) |
| **C′_obs reconstruction** | σ_y·y_prev+μ_y from windowed previous target (same scaler ⇒ consistent µg/m³) | `train_gnn_pinn.make_windows_phys` | ✓ |
| **Measured-wind operator** | φ_flow=WD+180°; u_ij=[v_i cos(φ_flow−θ_ij)]_+/d_ij from inverse-scaled WS/Sin_WD/Cos_WD (ch 3/4/5) | `physics.apply_advection`, `train_gnn_pinn._wind_from_batch` | ✓ matches feature schema |
| **SpatialPhysics (vsense)** | R=α·adv+K·diff on predicted field; no temporal/deposition (unsensored have no C_obs) | `physics.SpatialPhysics` | ✓ |
| **Evaluation space** | inverse z-score + climatology add-back → metrics in µg/m³ | `train_gnn.evaluate_gnn:173–203` | ✓ |

---

## 2. Experiment-protocol math (verified)

- **PT-FT (Variant A)**: load source ckpt → fine-tune on d% target windows; 3-way verification real_transfer = (transfer>zero_shot) ∧ (transfer>scratch). `train_gnn_tl.py`. ✓
- **Variant E pilot/node-drop/vsense deltas**: λ_phys=0 path is byte-identical to the data-only model (confirmed empirically — λ=0 reproduces Variant A), so ΔR² isolates the physics term. ✓
- **Ablation stats**: paired per-seed ΔR²; one-sample t vs 0 and paired t (adv vs diff) via `scipy.stats`. `analysis/vsense_ablation.py`. ✓ (small-n caveat noted in-paper).

---

## 3. Issues found

### 3.1 Advection mass-conservation — FIXED (was an overstated claim)
`PINN_PHYSICS.md` previously claimed the upwind advection operator was mass-conserving "up to the upwind
correction." **It is not:** `(𝒜C′)_i = Σ_j u_ij(C′_i−C′_j)` with outflow-only weights has unequal in/out
degrees, so `Σ_i(𝒜C′)_i ≠ 0`. Only the diffusion Laplacian conserves mass. Corrected in `PINN_PHYSICS.md`
§3 and §5 to state the advection is a **non-conservative directional transport prior** (a conservative
finite-volume flux scheme is future work). `PHYSICS_ABLATION_SECTION.md` was already accurate (attributes
conservation only to diffusion). **No code change and no results affected** — the operator is unchanged; only
the description was wrong.

### 3.2 RSD (Variant D) is a *simplified* RSD — DESCRIPTION must match implementation
`train_gnn_rsd.rsd_loss` implements only the **principal-angle (Grassmann projection) subspace distance**,
`‖sin Θ‖₁` where cos Θ = svdvals(Q_sᵀQ_t), and **omits the bases-mismatch penalty (BMP)** of Chen et al.
(ICML 2021). Two specifics:
- the `alpha_bmp` argument is defined but **unused** (dead parameter implying a BMP term that isn't computed);
- the norm is **L1** of the sine vector, not the Frobenius/L2 of the RSD paper.

`DUAL_TL_LITERATURE.md` §4 describes the *full* RSD (with BMP). **Action required before the manuscript
claims "we implement RSD":** describe Variant D as *"a Grassmann principal-angle subspace-alignment penalty
(‖sin Θ‖₁) adapted from RSD; the bases-mismatch term is omitted."* Either (a) reword the methods text and
delete the dead `alpha_bmp` param, or (b) add the BMP term and **re-run Variant D** (changes results). Do not
silently keep both the unused param and the "RSD" label. *(The Variant-D numbers themselves are valid for what
the code computes; only the name/description needs to be precise.)*

**Resolved (2026-06-11, option a):** `train_gnn_rsd.py` module + `rsd_loss` docstrings reworded to the precise
"Grassmann principal-angle subspace penalty ‖sin Θ‖₁, BMP omitted, L1 not Frobenius"; the dead `alpha_bmp`
parameter was removed. Results unchanged (the loss math is identical — only naming/description fixed).

---

## 4. Notes (not errors)
- GAT §1 "antisymmetric" descriptor in PINN_PHYSICS.md §1 refers to the **continuous** operator u·∇ (skew-adjoint), which is correct; the discrete operator is clarified in §3.
- RSD subspace dimension = joint mini-batch size (32) by construction; valid, just a design choice.
- Diffusion sign: K∇²C ↦ −K·LC, moved to residual LHS as +K·LC — consistent throughout.
