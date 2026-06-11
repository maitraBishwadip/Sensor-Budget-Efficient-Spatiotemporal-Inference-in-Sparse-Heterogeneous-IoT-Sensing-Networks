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
| **GAT attention** | e_ij=LeakyReLU(⟨Wx_i,a_src⟩+⟨Wx_j,a_dst⟩)+log w_ij; per-dest softmax; h_j=Σ_i α_ij Wx_i | `stgnn_gat.py:50–85` | ✓ additive GAT (Veličković 2018) |
| **TCN** | dilated causal conv, pad (k−1)·d, right-trim to length H | `stgnn_gat.py:88–111` | ✓ causal |
| **GAT+GRU (ST-GNN)** | GAT spatial per step → per-node GRU over time → last → head; param count `\|V\|`-independent | `stgnn_gat.py STGNN_GAT_GRU` | ✓ standard spatial-then-temporal ST-GNN |
| **SAGE** | h_v=ReLU(W_self h_v + W_neigh·Σw_uv h_u/Σw_uv) | `stgnn_sage.py:30–54` | ✓ weighted mean-aggregator (Hamilton 2017) |
| **Graph build** | haversine; bearing=atan2(...); kNN w=exp(−d²/2σ²); wind w=max(0,cos(θ_w−θ_ij))·exp(−d/decay) | `graph_construction.py` | ✓ standard formulas |
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

### 3.2 RSD (Variant D) and Graph-DANN — REMOVED
The dual domain-adaptation line (Graph-DANN, Dual-CDAN, RSD) was removed 2026-06-11 as an incomplete,
off-story branch (the project's coherent spine is LSTM/GNN baselines → LSTM-TL/GNN-TL → GNN-PINN/sensor-sparsity).
The audit of those components — including the earlier RSD-fidelity finding (it computed only a Grassmann
principal-angle ‖sin Θ‖₁ penalty, omitting the bases-mismatch term) — is therefore moot. The retained transfer
method is **GNN-TL (pre-train + fine-tune)**.

---

## 4. Notes (not errors)
- "Antisymmetric" descriptor in PINN_PHYSICS.md §1 refers to the **continuous** operator u·∇ (skew-adjoint), which is correct; the discrete operator is clarified in §3.
- Diffusion sign: K∇²C ↦ −K·LC, moved to residual LHS as +K·LC — consistent throughout.
