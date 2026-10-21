### A Pluto.jl notebook ###
# v0.20.4

using Markdown
using InteractiveUtils

# ╔═╡ f1a00001
begin
    import Pkg
    Pkg.activate(joinpath(@__DIR__, ".."))
    Pkg.instantiate()
    using SpatialEconometrics
    using Random, LinearAlgebra, SparseArrays, Statistics
    import StatsAPI: coef, nobs, loglikelihood
    md"**Setup complete** — SpatialEconometrics.jl loaded"
end

# ╔═╡ f1a00002
md"""
# 🔎 Interactive Spatial Filtering Explorer

Eigenvector spatial filtering (ESF) absorbs spatial autocorrelation into synthetic
regressors built from eigenvectors of the doubly-centred weight matrix. This notebook
lets you feel the central trade-off: **how many eigenvectors to keep**.

Too few and spatial structure leaks into the residuals. Too many and the model starts
absorbing noise. The notebook shows you both failure modes and the diagnostic that
tells them apart.
"""

# ╔═╡ f1a00003
md"## DGP Parameters"

# ╔═╡ f1a00004
md"""
**Grid size** (L × L units): $(@bind L_sl html"<input type='range' min='8' max='18' value='12' step='1'>")

**Spatial autocorrelation** (ρ): $(@bind rho_sl html"<input type='range' min='0' max='90' value='60' step='5'>")

**Error std dev** (σ): $(@bind sig_sl html"<input type='range' min='10' max='150' value='50' step='10'>")

**Random seed**: $(@bind seed_sl html"<input type='range' min='1' max='60' value='5' step='1'>")
"""

# ╔═╡ f1a00005
begin
    Lv = L_sl
    rhov = rho_sl / 100.0
    sigv = sig_sl / 100.0
    seedv = seed_sl

    W = make_rook_weights(Lv)
    Nv = size(W, 1)
    IN = sparse(1.0I, Nv, Nv)

    rngv = MersenneTwister(seedv)
    Xv = randn(rngv, Nv, 2)
    yv = (IN - rhov * W) \ (1.0 .+ Xv * [1.5, -0.8] .+ sigv .* randn(rngv, Nv))

    md"""
    | Parameter | Value |
    |-----------|-------|
    | Grid | $(Lv) × $(Lv) = $(Nv) units |
    | ρ | $(rhov) |
    | σ | $(sigv) |
    | Seed | $(seedv) |
    """
end

# ╔═╡ f1a00006
md"""
## The Eigenvector Basis

Eigenvector `i` of `Ω = M·Wsym·M` has Moran coefficient `(N/S₀)·λᵢ`, so the leading
eigenvectors are exactly the strongest positive-autocorrelation map patterns.

Three identities hold **exactly**, because the basis is constructed on the orthogonal
complement of the constant vector rather than by decomposing `Ω` directly:
"""

# ╔═╡ f1a00007
begin
    se_full = spatial_eigenvectors(W)
    emp_mc = [moran_coefficient(se_full.vectors[:, i], W) for i in 1:length(se_full.values)]
    id1 = maximum(abs.(emp_mc .- se_full.moran))
    id2 = maximum(abs.(transpose(se_full.vectors) * se_full.vectors - I))
    id3 = maximum(abs.(vec(sum(se_full.vectors, dims=1))))

    md"""
    | Identity | Residual |
    |----------|----------|
    | `MCᵢ = (N/S₀)·λᵢ` | $(round(id1; sigdigits=3)) |
    | `E'E = I` | $(round(id2; sigdigits=3)) |
    | `Σᵢ eᵢ = 0` (no collinearity with the intercept) | $(round(id3; sigdigits=3)) |

    Basis size: $(length(se_full.values)) = N − 1 eigenvectors, with
    $(count(>(1e-10), se_full.values)) having positive eigenvalues.
    """
end

# ╔═╡ f1a00008
md"""
## Stepwise ESF

Eigenvectors are added greedily until the residual Moran coefficient falls inside
`moran_tol` of zero, or the cap is reached.
"""

# ╔═╡ f1a00009
md"""
**Max eigenvectors**: $(@bind maxev_sl html"<input type='range' min='1' max='30' value='15' step='1'>")

**Selection criterion**: $(@bind crit_sl html"<select><option value='moran'>residual Moran</option><option value='aic'>AIC</option></select>")
"""

# ╔═╡ f1a0000a
begin
    m_esf = fit_esf(Xv, yv, W; family = :gaussian,
                    max_eigenvectors = maxev_sl,
                    criterion = Symbol(crit_sl),
                    varnames = ["x1", "x2"])
    sel = selected_eigenvectors(m_esf)
    Xo = hcat(ones(Nv), Xv)
    b_ols = Xo \ yv

    md"""
    **Selected $(length(sel.indices)) of $(maxev_sl) allowed** (criterion: `$(crit_sl)`)

    | | Intercept | x1 | x2 |
    |---|---|---|---|
    | OLS | $(round(b_ols[1]; digits=3)) | $(round(b_ols[2]; digits=3)) | $(round(b_ols[3]; digits=3)) |
    | ESF | $(round(coef(m_esf)[1]; digits=3)) | $(round(coef(m_esf)[2]; digits=3)) | $(round(coef(m_esf)[3]; digits=3)) |

    Residual Moran coefficient: **$(round(sel.moran_initial; digits=4)) → $(round(sel.moran_final; digits=4))**

    Selected indices: $(sel.indices)
    """
end

# ╔═╡ f1a0000b
md"""
!!! tip "Try setting ρ = 0"
    With no spatial autocorrelation to remove, ESF should select very few
    eigenvectors — often none. **Selecting zero is a correct outcome, not a
    failure**: it means the residuals were already clean.
"""

# ╔═╡ f1a0000c
md"""
## Random-Effects ESF

Stepwise selection introduces model-selection variance. RE-ESF instead keeps the whole
screened pool and *shrinks* every coefficient, governed by one variance ratio
`θ = τ²/σ²` estimated by profile REML.

The slider below screens the pool to `MCᵢ/max(MC) ≥ ratio`.
"""

# ╔═╡ f1a0000d
md"""
**min\\_moran\\_ratio**: $(@bind ratio_sl html"<input type='range' min='0' max='90' value='25' step='5'>")
"""

# ╔═╡ f1a0000e
begin
    ratio_v = ratio_sl / 100.0
    se_r = spatial_eigenvectors(W; min_moran_ratio = ratio_v)
    m_re = fit_re_esf(Xv, yv, W; eigen_basis = se_r, varnames = ["x1", "x2"])
    gap = m_re.moran_final - m_re.moran_reference

    verdict = if m_re.moran_final > 0.05
        "⚠️ **Under-filtering** — the residual is still positively autocorrelated. The pool is too small to absorb the structure."
    elseif gap < -0.15
        "⚠️ **Absorbing noise** — the residual is far below its reference and `edf` is a large share of N."
    else
        "✅ **Healthy** — the residual sits near its reference, which is where it belongs."
    end

    md"""
    | Quantity | Value |
    |----------|-------|
    | Pool size (L) | $(length(m_re.lambda)) |
    | θ = τ²/σ² | $(round(m_re.theta; digits=3)) |
    | τ² | $(round(m_re.tau2; digits=4)) |
    | σ² | $(round(m_re.sigma2; digits=4)) |
    | Effective df | $(round(m_re.edf; digits=1)) of N = $(Nv) |
    | Residual Moran | $(round(m_re.moran_final; digits=4)) |
    | **Complement reference** | **$(round(m_re.moran_reference; digits=4))** |
    | Gap (final − reference) | $(round(gap; digits=4)) |

    $(verdict)

    β̂ = $(round.(coef(m_re); digits=3))
    """
end

# ╔═╡ f1a0000f
md"""
### Why the residual Moran coefficient is negative

This is the part that trips people up. For RE-ESF the reported residual Moran
coefficient is for the *conditional* residual `y − Xβ̂ − Eγ̂`, and **it is supposed to
be substantially negative. Do not compare it against zero.**

Subtracting `Eγ̂` strips out the positively autocorrelated directions, so what remains
lives mostly in the orthogonal complement of `span(E)` — and a vector drawn at *random*
from that complement is itself strongly negatively autocorrelated. That expectation has
a closed form, because the `N−1` centred eigenvalues sum to `tr(M·Wsym·M) = −S₀/N`:

```math
\\frac{N}{S_0}\\cdot\\frac{-S_0/N - \\sum_l \\lambda_l}{N - 1 - L}
```

It is reported as `moran_reference`. The cell below checks it by Monte Carlo.
"""

# ╔═╡ f1a00010
begin
    pos_r = findall(>(1e-10), se_r.values)
    E_r = se_r.vectors[:, pos_r]

    function rand_perp(s)
        v = randn(MersenneTwister(s), Nv)
        v .-= mean(v)
        return v - E_r * (transpose(E_r) * v)
    end
    mc_sim = mean(moran_coefficient(rand_perp(s), W) for s in 1:200)

    md"""
    | | Moran coefficient |
    |---|---|
    | Closed form (`moran_reference`) | $(round(m_re.moran_reference; digits=4)) |
    | Monte Carlo, 200 random vectors in the complement | $(round(mc_sim; digits=4)) |
    | Actual conditional residual | $(round(m_re.moran_final; digits=4)) |

    The first two agree to Monte Carlo error. The third is the number to judge, and it
    should land *near or slightly above* the reference.
    """
end

# ╔═╡ f1a00011
md"## Sweeping the Pool Size"

# ╔═╡ f1a00012
begin
    rows = String[]
    push!(rows, "| ratio | pool | θ | edf | resid MC | reference | gap |")
    push!(rows, "|---|---|---|---|---|---|---|")
    for r in (0.0, 0.1, 0.25, 0.5, 0.75)
        seb = spatial_eigenvectors(W; min_moran_ratio = r)
        mb = fit_re_esf(Xv, yv, W; eigen_basis = seb)
        g = mb.moran_final - mb.moran_reference
        push!(rows, string("| ", r, " | ", length(mb.lambda), " | ",
                           round(mb.theta; digits=3), " | ",
                           round(mb.edf; digits=1), " | ",
                           round(mb.moran_final; digits=4), " | ",
                           round(mb.moran_reference; digits=4), " | ",
                           round(g; digits=3), " |"))
    end
    Markdown.parse(join(rows, "\n"))
end

# ╔═╡ f1a00013
md"""
---
## 💡 Try This

1. **Set ρ = 0.9 and ratio = 0.75**: the pool is too small to absorb the structure.
   Residual Moran goes **positive** — genuine leftover autocorrelation, i.e.
   under-filtering.
2. **Set ratio = 0** on a small grid: every positive eigenvector is retained, `edf`
   climbs toward N, and the residual drops far below its reference. That is noise
   absorption.
3. **Set ρ = 0**: watch ESF select almost nothing and `θ` collapse toward zero. With no
   spatial signal there is nothing for either method to do.
4. **Hold ρ fixed and raise the grid size**: `θ` stabilises and β̂ tightens, since the
   shrinkage parameter is better identified with more units.
5. **Compare the two β̂ rows** in the stepwise section as you raise `max_eigenvectors`:
   ESF and OLS agree closely on the slopes here, because filtering mainly repairs
   *inference*, not point estimates, when the covariates are spatially unstructured.
"""

# ╔═╡ Cell order:
# ╟─f1a00001
# ╟─f1a00002
# ╟─f1a00003
# ╟─f1a00004
# ╟─f1a00005
# ╟─f1a00006
# ╠═f1a00007
# ╟─f1a00008
# ╟─f1a00009
# ╠═f1a0000a
# ╟─f1a0000b
# ╟─f1a0000c
# ╟─f1a0000d
# ╠═f1a0000e
# ╟─f1a0000f
# ╠═f1a00010
# ╟─f1a00011
# ╠═f1a00012
# ╟─f1a00013
