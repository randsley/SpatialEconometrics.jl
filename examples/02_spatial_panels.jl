# ═══════════════════════════════════════════════════════════════════════
# Example 2: Static & Dynamic Spatial Panels
# ═══════════════════════════════════════════════════════════════════════
#
# Demonstrates:
#   - Static panel SAR/SDM with Lee-Yu and Direct QML
#   - Individual, time, and two-way fixed effects
#   - Dynamic panel with bias correction
#   - Short-run and long-run spatial multipliers
#   - Formula interface for panels

using SpatialEconometrics
using Random, LinearAlgebra, SparseArrays


# ─── Generate panel data ─────────────────────────────────────────────

rng = MersenneTwister(123)
L = 12; N = L * L; T_per = 8; NT = N * T_per
W = make_rook_weights(L)

X = randn(rng, NT, 2)
β_true = [1.5, -0.8]
ρ_true = 0.45
μ_i = randn(rng, N)         # individual fixed effects
ξ_t = randn(rng, T_per)     # time fixed effects

S_lu = lu(sparse(1.0I, N, N) - ρ_true * W)
y = zeros(NT)
for t in 1:T_per
    idx = ((t - 1) * N + 1):(t * N)
    y[idx] = S_lu \ (μ_i .+ ξ_t[t] .+ X[idx, :] * β_true .+ 0.5 .* randn(rng, N))
end

# ─── Static panel: Lee-Yu with two-way FE ────────────────────────────

println("═══ Static Panel SAR — Lee-Yu Transformation (Two-Way FE) ═══")
m_leeyu = fit_spatial_panel(SAR(), TwoWayFE(), W, X, y, N, T_per;
    approach = LeeYuTransformation(),
    varnames = ["investment", "trade"])
println(m_leeyu)
println()

# ─── Static panel: Direct QML with bias correction ───────────────────

println("═══ Static Panel SAR — Direct QML (Individual FE) ═══")
m_dqml = fit_spatial_panel(SAR(), IndivFE(), W, X, y, N, T_per;
    approach = DirectQML(bias_correct=true),
    varnames = ["investment", "trade"])
println(m_dqml)
println()

# ─── Spatial effects from panel ──────────────────────────────────────

println("═══ Spatial Effects ═══")
println(spatial_effects(m_leeyu))
println()

# ─── HAC-robust inference ────────────────────────────────────────────

println("═══ Driscoll-Kraay Robust Inference ═══")
dk = DriscollKraayVCOV(max_lag=3, kernel=BartlettKernel())
println(robust_coeftable(m_leeyu, dk))
println()

# ─── Formula interface ───────────────────────────────────────────────

println("═══ Formula Interface — SDM Panel ═══")
region_id = zeros(Int, NT); year_id = zeros(Int, NT)
for t in 1:T_per
    idx = ((t - 1) * N + 1):(t * N)
    region_id[idx] .= 1:N
    year_id[idx] .= 2020 + t
end
shuf = sortperm(1:NT; by = i -> (region_id[i], year_id[i]))
tbl = (region=region_id[shuf], year=year_id[shuf], y=y[shuf],
       investment=X[shuf, 1], trade=X[shuf, 2])

m_dsl = fit_spatial_panel(
    @formula(y ~ investment + trade + slag(investment + trade) + fe(region) + fe(year)),
    tbl, W; panel=:region, time=:year)
println(m_dsl)
println()

# ═══════════════════════════════════════════════════════════════════════
# Dynamic Spatial Panel
# ═══════════════════════════════════════════════════════════════════════

println("═══ Dynamic Spatial Panel ═══")
T_dyn = 10; NT_dyn = N * T_dyn
X_dyn = randn(rng, NT_dyn, 1)
γ_true = 0.3; η_true = 0.15; ρ_dyn = 0.35

y_dyn = zeros(NT_dyn)
y_dyn[1:N] = S_lu \ (μ_i .+ X_dyn[1:N, :] * [0.8] .+ 0.4 .* randn(rng, N))
for t in 2:T_dyn
    idx = ((t - 1) * N + 1):(t * N)
    idx_prev = ((t - 2) * N + 1):((t - 1) * N)
    y_dyn[idx] = S_lu \ (μ_i .+ γ_true .* y_dyn[idx_prev] .+ η_true .* (W * y_dyn[idx_prev]) .+ X_dyn[idx, :] * [0.8] .+ 0.4 .* randn(rng, N))
end

m_dyn = fit_dynamic_spatial_panel(SAR(), IndivFE(), W, X_dyn, y_dyn, N, T_dyn;
    varnames=["productivity"])
println(m_dyn)

println("\n═══ Dynamic Spatial Effects ═══")
de = dynamic_spatial_effects(m_dyn)
println(de)
