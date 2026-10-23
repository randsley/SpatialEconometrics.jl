### A Pluto.jl notebook ###
# v0.20.4

using Markdown
using InteractiveUtils

# ╔═╡ a3c00001
begin
    import Pkg
    Pkg.activate(joinpath(@__DIR__, ".."))
    Pkg.instantiate()
    using SpatialEconometrics
    using Random, LinearAlgebra, SparseArrays, Statistics
    import StatsAPI: coef, stderror, nobs, loglikelihood
    md"**Setup complete** — SpatialEconometrics.jl loaded"
end

# ╔═╡ a3c00002
md"""
# 🧭 Interactive Higher-Order & Regime Explorer

Six specifications, two spatial parameters, and one recurring question: **which model
does the data actually want?**

| Model | Lag in y | Lag in X | Correlated errors |
|---|---|---|---|
| SAR | ✓ | | |
| SLX | | ✓ | |
| SDM | ✓ | ✓ | |
| SEM | | | ✓ |
| SAC / SARAR | ✓ | | ✓ |
| SDEM | | ✓ | ✓ |

Pick a true DGP below, then watch how every specification responds — including the
wrong ones.
"""

# ╔═╡ a3c00003
md"## The Data-Generating Process"

# ╔═╡ a3c00004
md"""
**True DGP**: $(@bind dgp_sl html"<select><option value='SAR'>SAR — lag in y</option><option value='SLX'>SLX — lag in X</option><option value='SDM' selected>SDM — lag in y and X</option><option value='SAC'>SAC — lag in y + errors</option><option value='SDEM'>SDEM — lag in X + errors</option></select>")

**Grid size** (L × L): $(@bind L_sl html"<input type='range' min='10' max='20' value='16' step='1'>")

**ρ** (lag in y): $(@bind rho_sl html"<input type='range' min='-60' max='85' value='55' step='5'>")

**λ** (error correlation): $(@bind lam_sl html"<input type='range' min='-60' max='85' value='40' step='5'>")

**σ** (error std dev): $(@bind sig_sl html"<input type='range' min='10' max='150' value='50' step='10'>")

**Seed**: $(@bind seed_sl html"<input type='range' min='1' max='60' value='24' step='1'>")
"""

# ╔═╡ a3c00005
begin
    Lv = L_sl
    rho_t = rho_sl / 100.0
    lam_t = lam_sl / 100.0
    sig_t = sig_sl / 100.0

    W = make_rook_weights(Lv)
    Nv = size(W, 1)
    IN = sparse(1.0I, Nv, Nv)

    rngv = MersenneTwister(seed_sl)
    Xv = randn(rngv, Nv, 2)
    b_t = [1.5, -0.8]
    th_t = [0.6, -0.3]

    dgp_obj = if dgp_sl == "SAR"
        SAR_DGP(rho_t, b_t, sig_t)
    elseif dgp_sl == "SLX"
        SLX_DGP(b_t, th_t, sig_t)
    elseif dgp_sl == "SDM"
        SDM_DGP(rho_t, b_t, th_t, sig_t)
    elseif dgp_sl == "SAC"
        SAC_DGP(rho_t, lam_t, b_t, sig_t)
    else
        SDEM_DGP(lam_t, b_t, th_t, sig_t)
    end

    yv, _ = simulate_spatial(dgp_obj, W, Xv; rng = rngv)

    used = if dgp_sl == "SAR"
        "ρ = $(rho_t)"
    elseif dgp_sl == "SLX"
        "θ = $(th_t) (no ρ, no λ)"
    elseif dgp_sl == "SDM"
        "ρ = $(rho_t), θ = $(th_t)"
    elseif dgp_sl == "SAC"
        "ρ = $(rho_t), λ = $(lam_t)"
    else
        "λ = $(lam_t), θ = $(th_t)"
    end

    md"""
    | Setting | Value |
    |---|---|
    | DGP | **$(dgp_sl)** |
    | Grid | $(Lv) × $(Lv) = $(Nv) units |
    | Parameters in play | $(used) |
    | β (true) | $(b_t) |
    | σ | $(sig_t) |

    Note which sliders actually bind: an SLX DGP ignores both ρ and λ, so moving them
    changes nothing.
    """
end

# ╔═╡ a3c00006
md"## Pre-Estimation Diagnostics"

# ╔═╡ a3c00007
begin
    Xo = hcat(ones(Nv), Xv)
    res_ols = yv .- Xo * (Xo \ yv)
    mi = moran_global(res_ols, W)
    lms = lm_tests(yv, Xo, W)

    md"""
    On the OLS residuals:

    | Test | Statistic | p-value |
    |---|---|---|
    | Moran's I | $(round(mi.I_stat; digits=4)) | $(round(mi.p_value; digits=5)) |
    | LM-lag | $(round(lms.lag.stat; digits=3)) | $(round(lms.lag.p_value; digits=5)) |
    | LM-error | $(round(lms.error.stat; digits=3)) | $(round(lms.error.p_value; digits=5)) |
    | Robust LM-lag | $(round(lms.robust_lag.stat; digits=3)) | $(round(lms.robust_lag.p_value; digits=5)) |
    | Robust LM-error | $(round(lms.robust_error.stat; digits=3)) | $(round(lms.robust_error.p_value; digits=5)) |

    The **robust** variants are the ones to read when both plain tests reject: each is
    designed to stay valid in the presence of the other form of dependence.
    """
end

# ╔═╡ a3c00008
md"""
## Fitting Every Specification

AIC is `-2·logLik + 2k`, so lower is better and the extra parameters of the richer
models have to earn their place.
"""

# ╔═╡ a3c00009
begin
    fits = Any[]
    push!(fits, ("SLX", fit_slx(W, Xv, yv; varnames = ["x1", "x2"])))
    push!(fits, ("SAR", fit_spatial(SAR(), W, Xv, yv; varnames = ["x1", "x2"])))
    push!(fits, ("SDM", fit_spatial(SDM(), W, Xv, yv; varnames = ["x1", "x2"])))
    push!(fits, ("SDEM", fit_sdem(W, Xv, yv; varnames = ["x1", "x2"])))
    push!(fits, ("SAC", fit_sac(W, W, Xv, yv; varnames = ["x1", "x2"])))

    scored = map(fits) do (nm, m)
        k = length(coef(m))
        ll = loglikelihood(m)
        (nm, m, k, ll, -2ll + 2k)
    end
    best_aic = minimum(t -> t[5], scored)

    rows = String[]
    push!(rows, "| Model | k | logLik | AIC | |")
    push!(rows, "|---|---|---|---|---|")
    for (nm, _, k, ll, aic) in scored
        mark = aic == best_aic ? "🏆 best" : ""
        push!(rows, "| $(nm) | $(k) | $(round(ll; digits=2)) | $(round(aic; digits=2)) | $(mark) |")
    end
    Markdown.parse(join(rows, "\n"))
end

# ╔═╡ a3c0000a
begin
    best_name = scored[argmin([t[5] for t in scored])][1]
    verdict = best_name == dgp_sl ?
        "✅ AIC picked **$(best_name)**, which matches the true DGP." :
        "⚠️ AIC picked **$(best_name)** but the truth is **$(dgp_sl)**. Model selection is a finite-sample gamble — nearby specifications are often nearly indistinguishable, especially at small N or weak spatial parameters."
    md"$(verdict)"
end

# ╔═╡ a3c0000b
md"## Parameter Recovery in the Richest Model (SAC)"

# ╔═╡ a3c0000c
begin
    m_sac = scored[findfirst(t -> t[1] == "SAC", scored)][2]
    se_sac = stderror(m_sac)

    md"""
    SAC estimates both spatial parameters at once, so it is the natural place to see
    whether a lag, an error process, or both are present.

    | Parameter | Estimate | Std. Err |
    |---|---|---|
    | ρ (lag) | $(round(m_sac.rho; digits=4)) | |
    | λ (error) | $(round(m_sac.lambda; digits=4)) | |
    | x1 | $(round(coef(m_sac)[2]; digits=4)) | $(round(se_sac[2]; digits=4)) |
    | x2 | $(round(coef(m_sac)[3]; digits=4)) | $(round(se_sac[3]; digits=4)) |

    True β = $(b_t). For a SAC DGP the targets are ρ = $(rho_t) and λ = $(lam_t); for
    any other DGP at least one of the two is zero in truth, and SAC should say so.
    """
end

# ╔═╡ a3c0000d
md"""
---
# Spatial Regimes

A different kind of heterogeneity: the coefficients themselves differ across regions,
while a single spatial process spans the whole map.
"""

# ╔═╡ a3c0000e
md"""
**Slope contrast between regimes**: $(@bind contrast_sl html"<input type='range' min='0' max='200' value='170' step='10'>")

**Minimum region size for SKATER**: $(@bind minsz_sl html"<input type='range' min='2' max='60' value='20' step='2'>")
"""

# ╔═╡ a3c0000f
begin
    contrast = contrast_sl / 100.0
    labels_true = [i <= Nv ÷ 2 ? 1 : 2 for i in 1:Nv]
    bs = [1.0, 2.0, -0.5]
    bn = [1.0, 2.0 - contrast, -0.5 - contrast]

    eta = similar(Xv[:, 1])
    for i in 1:Nv
        b = labels_true[i] == 1 ? bs : bn
        eta[i] = b[1] + Xv[i, 1] * b[2] + Xv[i, 2] * b[3]
    end
    y_reg = (IN - 0.45 * W) \ (eta .+ 0.4 .* randn(MersenneTwister(seed_sl + 1), Nv))

    m_pool = fit_spatial(SAR(), W, Xv, y_reg; varnames = ["x1", "x2"])
    m_reg = fit_spatial_regime(W, Xv, y_reg, ExogenousRegimes(labels_true);
        varnames = ["x1", "x2"])
    chow = regime_chow_test(m_pool, m_reg)

    md"""
    Regime A slopes: $(bs[2:3]) · Regime B slopes: $(round.(bn[2:3]; digits=2))

    | | logLik | k |
    |---|---|---|
    | Pooled SAR | $(round(loglikelihood(m_pool); digits=2)) | $(length(coef(m_pool))) |
    | Regime SAR | $(round(loglikelihood(m_reg); digits=2)) | $(length(coef(m_reg))) |

    **Chow test**: LR = $(round(chow.stat; digits=3)) on $(chow.df) df,
    p = $(round(chow.p_value; sigdigits=4))

    $(chow.p_value < 0.05 ? "✅ Rejects coefficient stability — the regimes matter." : "➖ Cannot reject stability; at this contrast the pooled model is adequate.")
    """
end

# ╔═╡ a3c00010
md"""
### Discovering Regimes with SKATER

When the break is unknown, cluster on the covariates *subject to spatial contiguity*.
SKATER partitions a minimum spanning tree, choosing each cut to maximise the reduction
in within-group sum of squared deviations, so every region is a connected area of the
map.
"""

# ╔═╡ a3c00011
begin
    signal = hcat(Float64.(labels_true) .+ 0.3 .* randn(MersenneTwister(seed_sl + 2), Nv),
                  Xv[:, 1])
    disc = skater_regimes(W, signal, 2; min_size = minsz_sl)
    sizes = [count(==(r), disc.labels) for r in 1:2]
    agree = mean(disc.labels .== labels_true)
    m_disc = fit_spatial_regime(W, Xv, y_reg, disc; varnames = ["x1", "x2"])

    md"""
    | | Value |
    |---|---|
    | Region sizes | $(sizes) |
    | Agreement with the true split | $(round(100 * max(agree, 1 - agree); digits=1))% |
    | logLik, discovered regimes | $(round(loglikelihood(m_disc); digits=2)) |
    | logLik, true regimes | $(round(loglikelihood(m_reg); digits=2)) |
    | logLik, pooled | $(round(loglikelihood(m_pool); digits=2)) |

    Cluster labels are arbitrary, so agreement is reported as the better of the two
    matchings. `min_size` is what keeps a region from shrinking to a handful of units,
    which would leave the regime regression rank-deficient.
    """
end

# ╔═╡ a3c00012
md"""
---
## 💡 Try This

1. **Set the DGP to SLX and ρ to anything**: nothing changes, because an SLX process has
   no endogenous lag. Then check that AIC still prefers a model without ρ.
2. **Set the DGP to SAC with ρ = 0.55 and λ = 0.4**: SAC should win on AIC, since it is
   the only specification that admits both channels.
3. **Shrink ρ and λ toward 0**: the AIC ranking becomes unstable and the "best" model
   flips between seeds. That is the honest finite-sample message — model selection
   cannot distinguish specifications the data barely identify.
4. **Read the robust LM tests** as you switch between SAR and SEM DGPs: the robust
   variants should point at the right channel where the plain ones both reject.
5. **Drag the slope contrast to 0**: the two regimes become identical and the Chow test
   should stop rejecting. Raise it again and watch the LR statistic climb.
6. **Drop `min_size` to 2**: SKATER is now free to carve off tiny regions. Watch the
   region sizes and ask whether you would trust a regression fitted to the smaller one.
"""

# ╔═╡ Cell order:
# ╟─a3c00001
# ╟─a3c00002
# ╟─a3c00003
# ╟─a3c00004
# ╟─a3c00005
# ╟─a3c00006
# ╠═a3c00007
# ╟─a3c00008
# ╠═a3c00009
# ╟─a3c0000a
# ╟─a3c0000b
# ╠═a3c0000c
# ╟─a3c0000d
# ╟─a3c0000e
# ╠═a3c0000f
# ╟─a3c00010
# ╠═a3c00011
# ╟─a3c00012
