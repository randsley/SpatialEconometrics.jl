### A Pluto.jl notebook ###
# v0.20.4

using Markdown
using InteractiveUtils

# ╔═╡ a1b2c3d4
begin
    import Pkg
    Pkg.activate(joinpath(@__DIR__, ".."))
    Pkg.instantiate()
    using SpatialEconometrics
    using Random, LinearAlgebra, SparseArrays, Statistics
    md"**Setup complete** — SpatialEconometrics.jl loaded"
end

# ╔═╡ b2c3d4e5
md"""
# 🗺️ Interactive Spatial Econometrics Explorer

Adjust the sliders below to change the data-generating process and see how spatial models respond in real time.
"""

# ╔═╡ c3d4e5f6
md"## DGP Parameters"

# ╔═╡ d4e5f6a7
md"""
**Grid size** (L × L units): $(@bind L_slider html"<input type='range' min='8' max='25' value='12' step='1'><span id='L_val'></span>")

**Spatial autocorrelation** (ρ): $(@bind ρ_slider html"<input type='range' min='-50' max='90' value='45' step='5'><span id='rho_val'></span>")

**Error std dev** (σ): $(@bind σ_slider html"<input type='range' min='10' max='200' value='50' step='10'><span id='sig_val'></span>")

**Random seed**: $(@bind seed_slider html"<input type='range' min='1' max='100' value='42' step='1'>")
"""

# ╔═╡ e5f6a7b8
begin
    L_val = L_slider
    ρ_val = ρ_slider / 100.0
    σ_val = σ_slider / 100.0
    seed_val = seed_slider

    md"""
    | Parameter | Value |
    |-----------|-------|
    | Grid | $(L_val) × $(L_val) = $(L_val^2) units |
    | ρ | $(ρ_val) |
    | σ | $(σ_val) |
    | Seed | $(seed_val) |
    """
end

# ╔═╡ f6a7b8c9
begin
    rng = MersenneTwister(seed_val)
    W = make_rook_weights(L_val)
    N = size(W, 1)
    X = randn(rng, N, 2)
    β_true = [2.0, 1.5, -1.0]

    y = (I - ρ_val * W) \ (β_true[1] .+ X * β_true[2:3] .+ σ_val .* randn(rng, N))

    Markdown.parse("**Data generated**: N=$(N), mean(y)=$(round(mean(y); digits=2)), std(y)=$(round(std(y); digits=2))")
end

# ╔═╡ a7b8c9d0
md"## Diagnostics"

# ╔═╡ b8c9d0e1
begin
    mr = moran_global(y, W)
    gc = geary_c(y, W)
    lms = lm_tests(y, hcat(ones(N), X), W)

    md"""
    | Test | Statistic | p-value | Significant? |
    |------|-----------|---------|:---:|
    | **Moran's I** | $(round(mr.I_stat; digits=4)) | $(round(mr.p_value; digits=4)) | $(mr.p_value < 0.05 ? "✅" : "❌") |
    | **Geary's C** | $(round(gc.C_stat; digits=4)) | $(round(gc.p_value; digits=4)) | $(gc.p_value < 0.05 ? "✅" : "❌") |
    | **LM-Lag** | $(round(lms.lag.stat; digits=2)) | $(round(lms.lag.p_value; digits=4)) | $(lms.lag.p_value < 0.05 ? "✅" : "❌") |
    | **LM-Error** | $(round(lms.error.stat; digits=2)) | $(round(lms.error.p_value; digits=4)) | $(lms.error.p_value < 0.05 ? "✅" : "❌") |
    | **Robust LM-Lag** | $(round(lms.robust_lag.stat; digits=2)) | $(round(lms.robust_lag.p_value; digits=4)) | $(lms.robust_lag.p_value < 0.05 ? "✅" : "❌") |
    | **Robust LM-Error** | $(round(lms.robust_error.stat; digits=2)) | $(round(lms.robust_error.p_value; digits=4)) | $(lms.robust_error.p_value < 0.05 ? "✅" : "❌") |
    """
end

# ╔═╡ c9d0e1f2
md"## Model Estimation"

# ╔═╡ d0e1f2a3
begin
    const CS = SpatialEconometrics.CrossSection
    m_sar = fit_spatial(CS.SAR(), W, X, y; method=CS.ChebyshevApprox(order=8), varnames=["x1", "x2"])

    md"""
    ### SAR Results

    | Parameter | True | Estimated | SE |
    |-----------|------|-----------|-----|
    | ρ | $(ρ_val) | $(round(m_sar.rho; digits=4)) | $(round(m_sar.rho_se; digits=4)) |
    | β₁ | $(β_true[2]) | $(round(m_sar.coef[2]; digits=4)) | $(round(sqrt(max(0, m_sar.vcov[2,2])); digits=4)) |
    | β₂ | $(β_true[3]) | $(round(m_sar.coef[3]; digits=4)) | $(round(sqrt(max(0, m_sar.vcov[3,3])); digits=4)) |

    **Log-likelihood**: $(round(m_sar.loglik; digits=2)) | **σ²**: $(round(m_sar.sigma2; digits=4))
    """
end

# ╔═╡ e1f2a3b4
md"## Spatial Effects"

# ╔═╡ f2a3b4c5
begin
    eff = spatial_effects(m_sar)

    md"""
    | Variable | Direct | Indirect | Total |
    |----------|--------|----------|-------|
    | **x1** | $(round(eff.direct[1]; digits=4)) ± $(round(eff.direct_se[1]; digits=4)) | $(round(eff.indirect[1]; digits=4)) ± $(round(eff.indirect_se[1]; digits=4)) | $(round(eff.total[1]; digits=4)) ± $(round(eff.total_se[1]; digits=4)) |
    | **x2** | $(round(eff.direct[2]; digits=4)) ± $(round(eff.direct_se[2]; digits=4)) | $(round(eff.indirect[2]; digits=4)) ± $(round(eff.indirect_se[2]; digits=4)) | $(round(eff.total[2]; digits=4)) ± $(round(eff.total_se[2]; digits=4)) |

    The **spatial multiplier** is $1/(1-ρ) = $(round(1/(1-m_sar.rho); digits=3))$ — each unit of direct effect
    generates $(round(1/(1-m_sar.rho) - 1; digits=3)) additional units through spatial feedback.
    """
end

# ╔═╡ a3b4c5d6
md"""
---
## 💡 Try This

1. **Set ρ = 0**: Watch Moran's I become insignificant and direct ≈ total effects
2. **Set ρ = 0.8**: See massive spatial multipliers and tight coefficient recovery
3. **Set ρ negative**: Observe negative spatial autocorrelation (checkerboard pattern)
4. **Increase σ**: Watch SEs grow and coverage deteriorate
5. **Small grid (L=8)**: See higher estimation variance from fewer observations
"""

# ╔═╡ Cell order:
# ╟─a1b2c3d4
# ╟─b2c3d4e5
# ╟─c3d4e5f6
# ╟─d4e5f6a7
# ╟─e5f6a7b8
# ╠═f6a7b8c9
# ╟─a7b8c9d0
# ╟─b8c9d0e1
# ╟─c9d0e1f2
# ╟─d0e1f2a3
# ╟─e1f2a3b4
# ╟─f2a3b4c5
# ╟─a3b4c5d6
