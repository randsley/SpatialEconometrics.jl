# ═══════════════════════════════════════════════════════════════════════
# Example 8: HAC-Robust Inference (Driscoll-Kraay & Conley)
# ═══════════════════════════════════════════════════════════════════════
#
# Demonstrates:
#   - Driscoll-Kraay SCC standard errors for panel data
#   - Conley spatio-temporal HAC standard errors
#   - Comparing baseline vs robust standard errors
#   - Different kernel specifications
#   - HAC-robust spatial effects

using SpatialEconometrics
using Random, LinearAlgebra, SparseArrays

# ─── Generate panel data with heteroskedastic, clustered errors ───────

rng = MersenneTwister(707)
L = 10; N = L * L; T_per = 10
W = make_rook_weights(L)


X = randn(rng, N * T_per, 2)
μ_i = randn(rng, N)
ρ_true = 0.4
S_lu = lu(sparse(1.0I, N, N) - ρ_true * W)

y = zeros(N * T_per)
for t in 1:T_per
    idx = ((t - 1) * N + 1):(t * N)
    # Heteroskedastic errors: variance depends on X
    het_scale = 0.3 .+ 0.5 .* abs.(X[idx, 1])
    y[idx] = S_lu \ (μ_i .+ X[idx, :] * [1.0, -0.5] .+ het_scale .* randn(rng, N))
end

# ─── Fit baseline model ──────────────────────────────────────────────

m = fit_spatial_panel(SAR(), TwoWayFE(), W, X, y, N, T_per;
    varnames = ["investment", "trade"])
println("═══ Baseline Model ═══")
println(m)
println()

# ─── Driscoll-Kraay standard errors ──────────────────────────────────

println("═══ Driscoll-Kraay (Bartlett kernel, auto lag) ═══")
dk = DriscollKraayVCOV(kernel=BartlettKernel())
tab_dk = robust_coeftable(m, dk)
println(tab_dk)
println()

println("═══ Driscoll-Kraay (Parzen kernel, 5 lags) ═══")
dk_parzen = DriscollKraayVCOV(max_lag=5, kernel=ParzenKernel())
tab_dk2 = robust_coeftable(m, dk_parzen)
println(tab_dk2)
println()

# ─── Conley spatial HAC ──────────────────────────────────────────────

# Create coordinates on a grid
coords = zeros(N, 2)
for r in 1:L, c in 1:L
    i = (r - 1) * L + c
    coords[i, :] = [Float64(r), Float64(c)]
end

println("═══ Conley Spatial HAC (cutoff=3.0, 2 time lags) ═══")
conley = ConleySpatialHAC(coords, 3.0;
    max_time_lag=2,
    space_kernel=BartlettKernel(),
    time_kernel=BartlettKernel())
tab_con = robust_coeftable(m, conley)
println(tab_con)
println()

# ─── HAC-robust spatial effects ──────────────────────────────────────

println("═══ Spatial Effects: Baseline vs Driscoll-Kraay ═══")
eff_base = spatial_effects(m)
eff_dk = spatial_effects(m, dk)

for i in eachindex(eff_base.varnames)
    println("$(eff_base.varnames[i]):")
    println("  Direct:   $(round(eff_base.direct[i]; digits=4)) (SE base: $(round(eff_base.direct_se[i]; digits=4)), SE DK: $(round(eff_dk.direct_se[i]; digits=4)))")
    println("  Indirect: $(round(eff_base.indirect[i]; digits=4)) (SE base: $(round(eff_base.indirect_se[i]; digits=4)), SE DK: $(round(eff_dk.indirect_se[i]; digits=4)))")
end
