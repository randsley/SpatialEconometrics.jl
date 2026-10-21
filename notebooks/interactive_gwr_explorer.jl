### A Pluto.jl notebook ###
# v0.20.4

using Markdown
using InteractiveUtils

# ╔═╡ aa11bb22
begin
    import Pkg
    Pkg.activate(joinpath(@__DIR__, ".."))
    Pkg.instantiate()
    using SpatialEconometrics
    using Random, LinearAlgebra, Statistics
    md"**Setup complete**"
end

# ╔═╡ bb22cc33
md"""
# 🌍 Interactive GWR Explorer

Explore how Geographically Weighted Regression captures spatially varying relationships.
Adjust parameters to see how bandwidth, kernel choice, and spatial variation affect local coefficients.
"""

# ╔═╡ cc33dd44
md"## Data Generation Parameters"

# ╔═╡ dd44ee55
md"""
**Number of observations**: $(@bind N_obs html"<input type='range' min='100' max='500' value='200' step='50'>")

**Spatial variation strength** (slope of β₁ across x-coordinate): $(@bind var_strength html"<input type='range' min='0' max='80' value='40' step='5'>")

**Error noise** (σ): $(@bind noise_level html"<input type='range' min='10' max='200' value='50' step='10'>")

**Seed**: $(@bind gwr_seed html"<input type='range' min='1' max='100' value='77' step='1'>")
"""

# ╔═╡ ee55ff66
begin
    n_obs = N_obs
    β_slope = var_strength / 100.0
    σ_gwr = noise_level / 100.0

    md"""
    | Setting | Value |
    |---------|-------|
    | N | $(n_obs) |
    | β₁ spatial slope | $(β_slope) (range: $(round(1.0; digits=1)) to $(round(1.0 + β_slope * 10; digits=1))) |
    | σ | $(σ_gwr) |
    """
end

# ╔═╡ ff66aa77
begin
    rng_gwr = MersenneTwister(gwr_seed)
    coords = hcat(rand(rng_gwr, n_obs) .* 10, rand(rng_gwr, n_obs) .* 10)
    X_gwr = randn(rng_gwr, n_obs, 2)

    β1_local = 1.0 .+ β_slope .* coords[:, 1]
    β2_const = fill(-0.5, n_obs)
    y_gwr = 2.0 .+ β1_local .* X_gwr[:, 1] .+ β2_const .* X_gwr[:, 2] .+ σ_gwr .* randn(rng_gwr, n_obs)

    Markdown.parse("**Data**: $(n_obs) observations, β₁ ∈ [$(round(minimum(β1_local); digits=2)), $(round(maximum(β1_local); digits=2))], β₂ = -0.5 (constant)")
end

# ╔═╡ aa77bb88
md"""
## GWR Settings

**Bandwidth (k neighbors)**: $(@bind bw_k html"<input type='range' min='15' max='200' value='50' step='5'>")

**Kernel**: $(@bind kernel_choice html"<select><option value='bisquare'>Bisquare</option><option value='gaussian'>Gaussian</option><option value='exponential'>Exponential</option></select>")
"""

# ╔═╡ bb88cc99
begin
    kern = kernel_choice == "bisquare" ? BisquareKernel() :
           kernel_choice == "gaussian" ? GaussianKernel() : ExponentialKernel()

    m_gwr = fit_gwr(X_gwr, y_gwr, coords;
        bandwidth = AdaptiveBandwidth(bw_k),
        kernel = kern,
        varnames = ["variable_x1", "constant_x2"])

    Markdown.parse("""
    ### GWR Results (k=$(bw_k), kernel=$(kernel_choice))

    | Metric | Value |
    |--------|-------|
    | R² | $(round(m_gwr.R2; digits=4)) |
    | AICc | $(round(m_gwr.AICc; digits=1)) |
    | σ² | $(round(m_gwr.sigma2; digits=4)) |
    | tr(S) | $(round(m_gwr.hat_trace; digits=1)) (effective DOF) |

    **Local coefficient variation:**

    | Variable | Mean β | SD β | Min β | Max β |
    |----------|--------|------|-------|-------|
    | x1 (varying) | $(round(mean(m_gwr.coef_local[:,2]); digits=3)) | $(round(std(m_gwr.coef_local[:,2]); digits=3)) | $(round(minimum(m_gwr.coef_local[:,2]); digits=3)) | $(round(maximum(m_gwr.coef_local[:,2]); digits=3)) |
    | x2 (constant) | $(round(mean(m_gwr.coef_local[:,3]); digits=3)) | $(round(std(m_gwr.coef_local[:,3]); digits=3)) | $(round(minimum(m_gwr.coef_local[:,3]); digits=3)) | $(round(maximum(m_gwr.coef_local[:,3]); digits=3)) |

    **Correlation**(true β₁, estimated β₁) = **$(round(cor(β1_local, m_gwr.coef_local[:, 2]); digits=3))**
    """)
end

# ╔═╡ cc99ddaa
md"""
## MGWR Comparison

MGWR automatically selects different bandwidths for each variable.
"""

# ╔═╡ ddaaeebb
begin
    m_mgwr = fit_mgwr(X_gwr, y_gwr, coords;
        kernel = BisquareKernel(),
        varnames = ["variable_x1", "constant_x2"])

    md"""
    ### MGWR Variable-Specific Bandwidths

    | Variable | Bandwidth (k) | SD of local β | Interpretation |
    |----------|:---:|:---:|----------------|
    | (Intercept) | $(round(Int, m_mgwr.bandwidths[1])) | $(round(std(m_mgwr.coef_local[:,1]); digits=3)) | — |
    | x1 (varying) | $(round(Int, m_mgwr.bandwidths[2])) | $(round(std(m_mgwr.coef_local[:,2]); digits=3)) | Smaller BW → local process ✓ |
    | x2 (constant) | $(round(Int, m_mgwr.bandwidths[3])) | $(round(std(m_mgwr.coef_local[:,3]); digits=3)) | Larger BW → global process ✓ |

    **MGWR R²**: $(round(m_mgwr.R2; digits=4)) | **Converged**: $(m_mgwr.converged) in $(m_mgwr.iterations) iterations
    """
end

# ╔═╡ eebbffcc
md"""
---
## 💡 Try This

1. **Set spatial variation = 0**: Both variables are now global → GWR wastes DOF, R² may drop
2. **Set spatial variation = 80**: Strong local variation → small bandwidth excels, large bandwidth misses it
3. **Very small bandwidth (k=15)**: Overfitting — high R² but noisy local coefficients
4. **Very large bandwidth (k=200)**: Underfitting — approaches OLS, loses local detail
5. **High noise (σ=2.0)**: Hard for GWR to detect spatial variation → correlation drops
6. **Compare kernels**: Bisquare has compact support (sharp cutoff), Gaussian is smooth
"""

# ╔═╡ Cell order:
# ╟─aa11bb22
# ╟─bb22cc33
# ╟─cc33dd44
# ╟─dd44ee55
# ╟─ee55ff66
# ╠═ff66aa77
# ╟─aa77bb88
# ╟─bb88cc99
# ╟─cc99ddaa
# ╟─ddaaeebb
# ╟─eebbffcc
