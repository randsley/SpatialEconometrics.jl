### A Pluto.jl notebook ###
# v0.20.4

using Markdown
using InteractiveUtils

# ╔═╡ d2b00001
begin
    import Pkg
    Pkg.activate(joinpath(@__DIR__, ".."))
    Pkg.instantiate()
    using SpatialEconometrics
    using Random, LinearAlgebra, SparseArrays, Statistics
    import StatsAPI: coef, stderror, nobs
    md"**Setup complete** — SpatialEconometrics.jl loaded"
end

# ╔═╡ d2b00002
md"""
# 🎯 Interactive Spatial Probit Explorer

Binary outcomes with spatial dependence: land use change, technology adoption, voting,
firm entry and exit. Both estimators here target the same latent model

```math
y^* = \\rho W y^* + X\\beta + \\varepsilon, \\qquad y = \\mathbf{1}(y^* > 0)
```

with `σ² = 1` fixed for identification.

The interesting comparison is **speed against precision**. The linearized GMM estimator
is fast and needs no MCMC; the Gibbs sampler is slower but markedly more precise on `ρ`.
"""

# ╔═╡ d2b00003
md"## DGP Parameters"

# ╔═╡ d2b00004
md"""
**Grid size** (L × L units): $(@bind L_sl html"<input type='range' min='8' max='18' value='14' step='1'>")

**Spatial autocorrelation** (ρ): $(@bind rho_sl html"<input type='range' min='0' max='80' value='40' step='5'>")

**β₁**: $(@bind b1_sl html"<input type='range' min='-150' max='150' value='50' step='10'>")

**β₂**: $(@bind b2_sl html"<input type='range' min='-150' max='150' value='-70' step='10'>")

**Random seed**: $(@bind seed_sl html"<input type='range' min='1' max='60' value='14' step='1'>")
"""

# ╔═╡ d2b00005
begin
    Lv = L_sl
    rho_true = rho_sl / 100.0
    b_true = [b1_sl / 100.0, b2_sl / 100.0]
    seedv = seed_sl

    W = make_rook_weights(Lv)
    Nv = size(W, 1)
    IN = sparse(1.0I, Nv, Nv)

    rngv = MersenneTwister(seedv)
    Xv = randn(rngv, Nv, 2)
    y_star = (IN - rho_true * W) \ (Xv * b_true .+ randn(rngv, Nv))
    yv = Float64.(y_star .> 0)
    share = mean(yv)

    md"""
    | Parameter | Value |
    |-----------|-------|
    | Grid | $(Lv) × $(Lv) = $(Nv) units |
    | ρ (true) | $(rho_true) |
    | β (true) | $(b_true) |
    | Share y = 1 | $(round(100 * share; digits=1))% |
    | Seed | $(seedv) |
    """
end

# ╔═╡ d2b00006
md"""
!!! warning "Watch the outcome share"
    If the share of ones drifts near 0% or 100%, there is almost no information left in
    the binary outcome and **both** estimators will struggle. Keep it roughly between
    20% and 80% for a fair comparison.
"""

# ╔═╡ d2b00007
md"""
## Linearized GMM (Klier & McMillen)

A standard probit supplies the `ρ = 0` starting value, the moment conditions are
linearized around it, and the result is estimated by 2SLS with `[X, WX, W²X]`
instruments plus a heteroskedasticity-robust sandwich.
"""

# ╔═╡ d2b00008
begin
    m_gmm = fit_spatial_probit(W, Xv, yv; varnames = ["x1", "x2"])
    se_gmm = stderror(m_gmm)

    md"""
    | Parameter | Estimate | Std. Err | True |
    |---|---|---|---|
    | Intercept | $(round(coef(m_gmm)[1]; digits=4)) | $(round(se_gmm[1]; digits=4)) | — |
    | x1 | $(round(coef(m_gmm)[2]; digits=4)) | $(round(se_gmm[2]; digits=4)) | $(b_true[1]) |
    | x2 | $(round(coef(m_gmm)[3]; digits=4)) | $(round(se_gmm[3]; digits=4)) | $(b_true[2]) |
    | **ρ** | **$(round(m_gmm.rho; digits=4))** | | **$(rho_true)** |

    Error in ρ̂: $(round(abs(m_gmm.rho - rho_true); digits=4))
    """
end

# ╔═╡ d2b00009
md"""
!!! danger "One draw is a coarse reading"
    Linearizing around `ρ = 0` trades precision for speed. At `N = 196` this
    estimator's `ρ̂` has a standard deviation of roughly **0.25** — about 2.5× the Gibbs
    sampler's. It is approximately *unbiased*, so a single small-sample estimate that
    looks far off is usually sampling noise rather than a defect. Move the seed slider
    and watch `ρ̂` wander.
"""

# ╔═╡ d2b0000a
md"""
## Bayesian (Albert–Chib Gibbs)

Each sweep draws the latent `y*` one site at a time from its truncated normal
conditional, draws `β` from its conjugate normal conditional, then updates `ρ` by
random-walk Metropolis.
"""

# ╔═╡ d2b0000b
md"""
**Draws**: $(@bind ndraw_sl html"<input type='range' min='400' max='3000' value='1200' step='200'>")

**Metropolis proposal SD for ρ**: $(@bind psd_sl html"<input type='range' min='2' max='60' value='20' step='2'>")
"""

# ╔═╡ d2b0000c
begin
    nd = ndraw_sl
    psd = psd_sl / 100.0
    m_bay = fit_bayesian_spatial_probit(W, Xv, yv;
        n_samples = nd, burnin = nd ÷ 4,
        rho_proposal_sd = psd,
        varnames = ["x1", "x2"], rng = MersenneTwister(99))
    se_bay = stderror(m_bay)

    acc_note = if m_bay.accept_rate < 0.15
        "⚠️ acceptance too LOW — the proposal SD is too large, the chain is sticking"
    elseif m_bay.accept_rate > 0.6
        "⚠️ acceptance too HIGH — the proposal SD is too small, the chain crawls"
    else
        "✅ acceptance in the healthy 0.2–0.5 band"
    end

    md"""
    | Parameter | Posterior mean | Posterior SD | True |
    |---|---|---|---|
    | Intercept | $(round(coef(m_bay)[1]; digits=4)) | $(round(se_bay[1]; digits=4)) | — |
    | x1 | $(round(coef(m_bay)[2]; digits=4)) | $(round(se_bay[2]; digits=4)) | $(b_true[1]) |
    | x2 | $(round(coef(m_bay)[3]; digits=4)) | $(round(se_bay[3]; digits=4)) | $(b_true[2]) |
    | **ρ** | **$(round(m_bay.rho; digits=4))** | | **$(rho_true)** |

    Draws: $(nd) (burn-in $(nd ÷ 4)) · proposal SD $(psd)

    ρ acceptance rate: **$(round(m_bay.accept_rate; digits=3))** — $(acc_note)

    Error in ρ̂: $(round(abs(m_bay.rho - rho_true); digits=4))
    """
end

# ╔═╡ d2b0000d
md"## Head to Head"

# ╔═╡ d2b0000e
begin
    e_gmm = abs(m_gmm.rho - rho_true)
    e_bay = abs(m_bay.rho - rho_true)
    winner = e_bay < e_gmm ? "Gibbs sampler" : "GMM"

    md"""
    | | ρ̂ | abs error | β̂₁ | β̂₂ |
    |---|---|---|---|---|
    | True | $(rho_true) | — | $(b_true[1]) | $(b_true[2]) |
    | GMM | $(round(m_gmm.rho; digits=4)) | $(round(e_gmm; digits=4)) | $(round(coef(m_gmm)[2]; digits=3)) | $(round(coef(m_gmm)[3]; digits=3)) |
    | Gibbs | $(round(m_bay.rho; digits=4)) | $(round(e_bay; digits=4)) | $(round(coef(m_bay)[2]; digits=3)) | $(round(coef(m_bay)[3]; digits=3)) |

    Closer on this draw: **$(winner)**

    One draw proves nothing — the point is the *distribution* across seeds. Sweep it below.
    """
end

# ╔═╡ d2b0000f
md"""
## Dispersion Across Seeds

This is the honest comparison: refit both estimators on fresh draws and compare the
spread of `ρ̂`, not any single value. (The Gibbs sampler is the slow part, so this uses
short chains.)
"""

# ╔═╡ d2b00010
md"""
**Number of seeds**: $(@bind nseed_sl html"<input type='range' min='3' max='12' value='6' step='1'>")
"""

# ╔═╡ d2b00011
begin
    rg = Float64[]
    rb = Float64[]
    for s in 1:nseed_sl
        rs = MersenneTwister(1000 + s)
        Xs = randn(rs, Nv, 2)
        ys_star = (IN - rho_true * W) \ (Xs * b_true .+ randn(rs, Nv))
        ys = Float64.(ys_star .> 0)
        # Degenerate outcomes carry no information; skip them.
        (all(ys .== 0) || all(ys .== 1)) && continue
        try
            push!(rg, fit_spatial_probit(W, Xs, ys).rho)
        catch
        end
        try
            push!(rb, fit_bayesian_spatial_probit(W, Xs, ys;
                n_samples = 600, burnin = 200,
                rng = MersenneTwister(7)).rho)
        catch
        end
    end

    summ(v) = isempty(v) ? "—" :
        string(round(mean(v); digits=3), " ± ", round(std(v); digits=3))

    md"""
    Over $(nseed_sl) seeds at N = $(Nv), true ρ = $(rho_true):

    | Estimator | mean ρ̂ ± SD | n |
    |---|---|---|
    | GMM | $(summ(rg)) | $(length(rg)) |
    | Gibbs | $(summ(rb)) | $(length(rb)) |

    The Gibbs SD should be materially smaller. Both means should sit near
    $(rho_true) — that is what "approximately unbiased" means, and it is a claim
    about the mean, not about any one estimate.
    """
end

# ╔═╡ d2b00012
md"""
## Marginal Effects

For a spatial probit the derivative of `P(y = 1)` with respect to `x` runs through
`(I − ρW)⁻¹`, so a change at one site propagates to every other site.
"""

# ╔═╡ d2b00013
begin
    eff = marginal_effects(m_gmm)
    md"""
    | Variable | Direct | Indirect | Total |
    |---|---|---|---|
    | x1 | $(round(eff.direct[1]; digits=4)) | $(round(eff.indirect[1]; digits=4)) | $(round(eff.total[1]; digits=4)) |
    | x2 | $(round(eff.direct[2]; digits=4)) | $(round(eff.indirect[2]; digits=4)) | $(round(eff.total[2]; digits=4)) |

    *Direct* is the own-site effect, *indirect* the spillover onto other sites. As ρ → 0
    the indirect column should collapse toward zero.
    """
end

# ╔═╡ d2b00014
md"""
---
## 💡 Try This

1. **Set ρ = 0**: the indirect marginal effects should collapse to roughly zero, and
   both estimators should agree closely.
2. **Set ρ = 0.8**: the latent scale ridge makes `ρ` weakly identified at small `N`.
   Watch both estimators struggle, and the GMM one struggle more.
3. **Drag the proposal SD to its minimum**: the acceptance rate shoots up past 0.9 and
   the chain barely moves — high acceptance is a *bad* sign, not a good one.
4. **Drag the proposal SD to its maximum**: acceptance collapses and the chain sticks.
   The default of 0.2 targets the 0.2–0.5 band.
5. **Push β₁ and β₂ toward ±1.5**: the outcome share saturates near 0% or 100% and
   everything degrades, because a near-constant binary outcome carries little
   information.
6. **Shrink the grid to 8 × 8**: only 64 units, and the seed-sweep spread widens sharply
   for both estimators.
"""

# ╔═╡ Cell order:
# ╟─d2b00001
# ╟─d2b00002
# ╟─d2b00003
# ╟─d2b00004
# ╟─d2b00005
# ╟─d2b00006
# ╟─d2b00007
# ╠═d2b00008
# ╟─d2b00009
# ╟─d2b0000a
# ╟─d2b0000b
# ╠═d2b0000c
# ╟─d2b0000d
# ╠═d2b0000e
# ╟─d2b0000f
# ╟─d2b00010
# ╠═d2b00011
# ╟─d2b00012
# ╠═d2b00013
# ╟─d2b00014
