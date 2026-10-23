### A Pluto.jl notebook ###
# v0.20.4

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook
# outside of Pluto, the following 'mock version' of @bind gives bound variables
# a default value (instead of an error).
macro bind(def, element)
    #! format: off
    quote
        local iv = try Base.loaded_modules[Base.PkgId(Base.UUID("6e696c72-6542-2067-7265-42206c756150"), "AbstractPlutoDingetjes")].Bonds.initial_value catch; b -> missing; end
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : iv(el)
        el
    end
    #! format: on
end

# ╔═╡ 1a001001
begin
    import Pkg
    Pkg.activate(joinpath(@__DIR__, ".."))
    Pkg.instantiate()
    using SpatialEconometrics
    using LinearAlgebra, SparseArrays, Statistics, Random, Printf
    md"**Setup complete** — SpatialEconometrics.jl loaded"
end

# ╔═╡ 1a002002
md"""
# The LM battery as a decision rule

Moran's I tells you that residuals are spatially autocorrelated. It does not tell
you *why*, and the two leading explanations call for different models:

| | Model | What is dependent |
|---|---|---|
| **Lag** | $y = \rho W y + X\beta + \varepsilon$ | the outcome itself |
| **Error** | $y = X\beta + u$, $u = \lambda W u + \varepsilon$ | only the disturbance |

The distinction matters. Under a lag process a change in one unit's $x$ propagates
to its neighbours, so the coefficients are not the marginal effects. Under an
error process $\beta$ keeps its usual meaning and the problem is confined to
inference.

Anselin's Lagrange Multiplier battery decides between them **before** you fit
anything, using only OLS residuals. This notebook lets you generate data from a
process you choose and watch the battery try to recover it.
"""

# ╔═╡ 1a003003
md"""
## Choose the true process

**Lag dependence** ρ: $(@bind rho_pct html"<input type='range' min='0' max='80' value='50' step='5'>")

**Error dependence** λ: $(@bind lam_pct html"<input type='range' min='0' max='80' value='0' step='5'>")

**Lattice side** (N = side²): $(@bind side html"<input type='range' min='8' max='22' value='14' step='2'>")

**Noise** σ: $(@bind sig_pct html"<input type='range' min='50' max='300' value='100' step='25'>")

**Seed**: $(@bind lm_seed html"<input type='range' min='1' max='60' value='42' step='1'>")
"""

# ╔═╡ 1a004004
begin
    ρ_true = rho_pct / 100.0
    λ_true = lam_pct / 100.0
    σ_true = sig_pct / 100.0

    W_lm = make_rook_weights(side)
    N_lm = size(W_lm, 1)
    Wd_lm = Matrix(W_lm)

    rng_lm = MersenneTwister(lm_seed)
    X_lm = hcat(randn(rng_lm, N_lm), randn(rng_lm, N_lm))
    β_lm = [1.5, -0.8]

    ε_lm = σ_true .* randn(rng_lm, N_lm)
    u_lm = λ_true == 0 ? ε_lm : (I - λ_true * Wd_lm) \ ε_lm
    y_lm = ρ_true == 0 ? X_lm * β_lm .+ u_lm :
                         (I - ρ_true * Wd_lm) \ (X_lm * β_lm .+ u_lm)

    Xc_lm = hcat(ones(N_lm), X_lm)
    md"Generated $(N_lm) observations with ρ = $(ρ_true), λ = $(λ_true), σ = $(σ_true)"
end

# ╔═╡ 1a005005
md"## What the battery says"

# ╔═╡ 1a006006
begin
    lms_lm = lm_tests(y_lm, Xc_lm, W_lm)
    mark(p) = p < 0.001 ? "***" : p < 0.01 ? "**" : p < 0.05 ? "*" : "—"

    rows = [("LM-Lag", lms_lm.lag), ("LM-Error", lms_lm.error),
            ("**Robust LM-Lag**", lms_lm.robust_lag),
            ("**Robust LM-Error**", lms_lm.robust_error)]

    body = join([@sprintf("| %s | %.3f | %.4f | %s |", nm, t.stat, t.p_value, mark(t.p_value))
                 for (nm, t) in rows], "\n")
    Markdown.parse("| Test | Statistic | p | |\n|---|---:|---:|---|\n" * body)
end

# ╔═╡ 1a007007
begin
    α_lm = 0.05
    pl = lms_lm.lag.p_value < α_lm
    pe = lms_lm.error.p_value < α_lm
    rl = lms_lm.robust_lag.p_value < α_lm
    re = lms_lm.robust_error.p_value < α_lm

    verdict =
        !pl && !pe ? "**Neither plain test rejects.** Keep OLS — there is no evidence of spatial dependence of either kind." :
        rl && !re  ? "**Robust lag only.** Fit a spatial lag model: `fit_spatial(SAR(), W, X, y)`." :
        re && !rl  ? "**Robust error only.** Fit a spatial error model: `fit_spatial_error(W, X, y)`." :
        rl && re   ? "**Both robust tests reject.** Consider SARAR, or suspect the mean specification — an omitted spatially patterned regressor shows up here." :
                     "**Plain tests reject but neither robust one does.** Inconclusive: there is dependence, but at this sample size the robust tests cannot say which kind."

    Markdown.parse("### Verdict\n\n" * verdict)
end

# ╔═╡ 1a008008
md"""
## Why four tests and not two

The plain tests are each computed under the assumption that the *other* kind of
dependence is absent. That assumption is doing real work: a spatial error process
inflates LM-Lag, and a lag process inflates LM-Error. Set ρ high and λ to zero, and
watch LM-Error reject anyway. It is not detecting an error process; it is
responding to one that is not there.

The robust versions correct each statistic for the presence of the other, which is
why they are the ones that decide. The price is power. Shrink the lattice to 8×8
and set ρ to 0.25: the plain tests will often still reject while the robust pair
goes quiet. That is the honest answer — at 64 observations the data cannot tell
the two processes apart — and it is worth more than a confident wrong one.

**How the four quantities relate.** With $e$ the OLS residuals,
$T = \mathrm{tr}(W'W + W^2)$, $d_\rho = e'Wy/\hat\sigma^2$,
$d_\lambda = e'We/\hat\sigma^2$ and $J$ the information term for $\rho$:

$$\text{LM-Lag} = \frac{d_\rho^2}{J}, \qquad \text{LM-Error} = \frac{d_\lambda^2}{T}$$

$$\text{Robust LM-Lag} = \frac{(d_\rho - d_\lambda)^2}{J - T}, \qquad
  \text{Robust LM-Error} = \frac{(d_\lambda - T d_\rho / J)^2}{T - T^2/J}$$

The subtraction in each robust numerator *is* the correction. Note that the two
plain tests differ in both numerator and denominator — they are genuinely
different statistics, not one statistic under two names.
"""

# ╔═╡ 1a009009
md"""
---
## Try this

1. **ρ = 0.5, λ = 0.** Both plain tests reject; only the robust lag does. This is
   the case the battery handles best.
2. **ρ = 0, λ = 0.5.** Now the robust *error* test fires and the robust lag test
   goes quiet, even though LM-Lag rejected. The correction earns its keep here.
3. **ρ = 0.4, λ = 0.4.** Both robust tests reject. The battery is telling you that
   neither single-source model will do.
4. **ρ = 0, λ = 0.** Rejections now happen about 5% of the time. Step through the
   seeds and count them — that is what a correctly sized test looks like.
5. **Shrink the lattice to 8 and set ρ = 0.25.** Watch the robust tests lose power
   before the plain ones do, and decide what you would write in a paper.
6. **Raise σ to 3.0.** How large does ρ have to be before the battery finds it?
   This is the practical question behind every specification search.
"""

# ╔═╡ Cell order:
# ╟─1a001001
# ╟─1a002002
# ╟─1a003003
# ╠═1a004004
# ╟─1a005005
# ╠═1a006006
# ╟─1a007007
# ╟─1a008008
# ╟─1a009009
