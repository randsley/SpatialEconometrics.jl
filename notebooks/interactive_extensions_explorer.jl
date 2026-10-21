### A Pluto.jl notebook ###
# v0.20.4

using Markdown
using InteractiveUtils

# ╔═╡ a1b2c3d4
begin
    import Pkg
    # Both feature sets here are package extensions: Makie and Turing are
    # weak dependencies, so they are not in the package's own Project.toml.
    # Build an environment that carries all three. The first run downloads
    # Turing and CairoMakie, which takes a few minutes.
    Pkg.activate(mktempdir())
    Pkg.develop(path = joinpath(@__DIR__, ".."))
    Pkg.add(["CairoMakie", "Turing", "Distributions"])

    using SpatialEconometrics
    using CairoMakie, Turing, Distributions
    using Random, LinearAlgebra, SparseArrays, Statistics
    import StatsAPI: coef, coefnames, stderror, residuals

    CS = SpatialEconometrics.CrossSection
    md"**Setup complete** — Makie and Turing extensions loaded"
end

# ╔═╡ b2c3d4e5
md"""
# 📊 Interactive Extensions Explorer

`SpatialEconometrics.jl` ships two optional feature sets as Julia package
extensions, and this notebook drives both:

* **`SpatialEconometricsMakieExt`** — five plotting functions, triggered by `Makie`
* **`SpatialEconometricsTuringExt`** — three Bayesian estimators, triggered by `Turing` + `Distributions`

Neither extension exports anything. All eight names are declared and
exported by `SpatialEconometrics` itself, and the extensions add methods to
them — nothing ever writes `using SpatialEconometricsMakieExt`, so a
function an extension declared on its own would be unreachable.
"""

# ╔═╡ c3d4e5f6
begin
    _exts = [
        (:SpatialEconometricsMakieExt, "Makie", string(pkgversion(Makie))),
        (:SpatialEconometricsTuringExt, "Turing", string(pkgversion(Turing))),
    ]
    _rows = join(["| `$e` | $p $v | $(isnothing(Base.get_extension(SpatialEconometrics, e)) ? "❌ not loaded" : "✅ loaded")  |"
                  for (e, p, v) in _exts], "\n")
    Markdown.parse("""
    | Extension | Trigger | Status |
    |---|---|---|
    $_rows
    """)
end

# ╔═╡ d4e5f6a7
md"## Data Generation"

# ╔═╡ e5f6a7b8
md"""
**Grid side length** (N = L²): $(@bind ext_L html"<input type='range' min='8' max='18' value='12' step='1'>")

**Spatial dependence ρ** (×100): $(@bind ext_rho html"<input type='range' min='-80' max='90' value='55' step='5'>")

**Error noise σ** (×100): $(@bind ext_sigma html"<input type='range' min='10' max='150' value='50' step='10'>")

**Seed**: $(@bind ext_seed html"<input type='range' min='1' max='50' value='8' step='1'>")
"""

# ╔═╡ f6a7b8c9
begin
    L_ext = ext_L
    ρ_ext = ext_rho / 100.0
    σ_ext = ext_sigma / 100.0

    N_ext = L_ext^2
    W_ext = make_rook_weights(L_ext)
    coords_ext = hcat(repeat(1.0:L_ext, outer = L_ext),
                      repeat(1.0:L_ext, inner = L_ext))

    rng_ext = MersenneTwister(ext_seed)
    X_ext = randn(rng_ext, N_ext, 2)
    y_ext = (sparse(1.0I, N_ext, N_ext) - ρ_ext * W_ext) \
            (1.0 .+ X_ext * [1.5, -0.8] .+ σ_ext .* randn(rng_ext, N_ext))

    m_ext = fit_spatial(CS.SAR(), W_ext, X_ext, y_ext;
                        varnames = ["income", "density"])

    Markdown.parse("""
    **N = $N_ext** on a $(L_ext) × $(L_ext) rook grid, true ρ = $(round(ρ_ext; digits=2)), σ = $(round(σ_ext; digits=2))

    SAR fit: ρ̂ = **$(round(m_ext.rho; digits=4))**,
    β̂ = $(round.(coef(m_ext)[2:3]; digits=4))
    """)
end

# ╔═╡ a7b8c9da
md"""
## 1 · Coefficient Plot

`level` is a real parameter, not decoration: the critical value is the
normal quantile, so moving the slider genuinely changes the interval drawn.
A value outside `(0, 1)` raises an `ArgumentError`.

**Confidence level** (%): $(@bind ext_level html"<input type='range' min='50' max='99' value='95' step='1'>")
"""

# ╔═╡ b8c9dae1
plot_coefficients(m_ext; level = ext_level / 100)

# ╔═╡ c9dae1f2
md"""
## 2 · Moran Scatterplot

The slope the plot draws is Moran's I whenever `W` is row-standardized.
Worth checking rather than asserting — and worth looking at the residuals
too, because a fitted SAR should have absorbed the dependence that the raw
series carries.
"""

# ╔═╡ dae1f2a3
begin
    I_raw = moran_global(y_ext, W_ext).I_stat
    I_res = moran_global(residuals(m_ext), W_ext).I_stat
    Markdown.parse("""
    Moran's I on raw `y`: **$(round(I_raw; digits=4))** ·
    on the SAR residuals: **$(round(I_res; digits=4))**
    """)
end

# ╔═╡ e1f2a3b4
plot_moran_scatter(y_ext, W_ext)

# ╔═╡ f2a3b4c5
plot_moran_scatter(residuals(m_ext), W_ext)

# ╔═╡ a3b4c5d6
md"""
## 3 · LISA Cluster Map

Local Moran's I per unit, with a permutation test behind the significance
colouring. `cluster_type` holds Symbols, `:ns` for the units the test does
not call significant.

**Permutations**: $(@bind ext_nperm html"<input type='range' min='99' max='999' value='499' step='100'>")
"""

# ╔═╡ b4c5d6e7
begin
    lisa_ext = moran_local(y_ext, W_ext; n_perm = ext_nperm,
                           rng = MersenneTwister(ext_seed))
    counts_ext = join(["$ct: **$(count(==(ct), lisa_ext.cluster_type))**"
                       for ct in (:HH, :LL, :HL, :LH, :ns)], " · ")
    Markdown.parse(counts_ext)
end

# ╔═╡ c5d6e7f8
plot_local_moran(lisa_ext, coords_ext)

# ╔═╡ d6e7f8a9
md"""
## 4 · GWR Local Coefficient Map

Here the DGP is given a coefficient that genuinely varies over space, so the
map has something to show. `β_income` rises with the x-coordinate;
`β_density` is constant, and its map should look like noise around −0.8.

**Spatial gradient in β_income** (×100): $(@bind ext_grad html"<input type='range' min='0' max='30' value='15' step='5'>")

**Adaptive bandwidth** (neighbours): $(@bind ext_bw html"<input type='range' min='15' max='80' value='30' step='5'>")
"""

# ╔═╡ e7f8a9ba
begin
    β1_surface = 0.5 .+ (ext_grad / 100) .* coords_ext[:, 1]
    y_gwr_ext = 1.0 .+ X_ext[:, 1] .* β1_surface .+ X_ext[:, 2] .* (-0.8) .+
                0.4 .* randn(MersenneTwister(ext_seed + 1), N_ext)
    g_ext = fit_gwr(X_ext, y_gwr_ext, coords_ext;
                    bandwidth = AdaptiveBandwidth(ext_bw),
                    varnames = ["income", "density"])

    i_inc = findfirst(==("income"), String.(g_ext.coefnames))
    loc_inc = g_ext.coef_local[:, i_inc]
    corr_ext = ext_grad == 0 ? NaN : cor(loc_inc, β1_surface)

    Markdown.parse("""
    Local β_income spans **$(round(minimum(loc_inc); digits=3)) … $(round(maximum(loc_inc); digits=3))**
    (true surface: $(round(minimum(β1_surface); digits=3)) … $(round(maximum(β1_surface); digits=3)))

    Correlation with the true surface: **$(round(corr_ext; digits=4))**
    """)
end

# ╔═╡ f8a9bacb
plot_gwr_coefficients(g_ext, "income")

# ╔═╡ a9bacbdc
plot_gwr_coefficients(g_ext, "density")

# ╔═╡ bacbdced
md"""
## 5 · Direct, Indirect and Total Effects

In a SAR a change in `x` spills through `(I - ρW)⁻¹` to every other unit,
so the total effect exceeds the coefficient you estimated. The indirect
column is what this chart exists to show.
"""

# ╔═╡ cbdcedfa
begin
    eff_ext = spatial_effects(m_ext)
    eff_rows = join(["| $(eff_ext.varnames[i]) | $(round(eff_ext.direct[i]; digits=4)) | $(round(eff_ext.indirect[i]; digits=4)) | $(round(eff_ext.total[i]; digits=4)) |"
                     for i in eachindex(eff_ext.varnames)], "\n")
    Markdown.parse("""
    | variable | direct | indirect | total |
    |---|---|---|---|
    $eff_rows
    """)
end

# ╔═╡ dcedfaab
plot_spatial_effects(eff_ext)

# ╔═╡ edfaabbc
md"""
---
## 6 · Bayesian Estimation (Turing)

Sampling is deliberately off by default: NUTS on a few hundred units takes
tens of seconds, and Pluto would re-run it on every slider move.

**Run the sampler**: $(@bind ext_sample html"<input type='checkbox'>")

**Model**: $(@bind ext_model html"<select><option value='SAR'>SAR</option><option value='SEM'>SEM</option><option value='SDM'>SDM</option></select>")

**Draws per chain**: $(@bind ext_draws html"<input type='range' min='200' max='2000' value='600' step='200'>")

**Chains**: $(@bind ext_chains html"<input type='range' min='1' max='4' value='2' step='1'>")
"""

# ╔═╡ faabbccd
begin
    if !ext_sample
        bayes_ext = nothing
        Markdown.parse("""
        !!! note "Sampler idle"
            Tick **Run the sampler** above to draw a posterior. With the
            current settings that is $(ext_draws) draws × $(ext_chains)
            chain(s) on N = $(N_ext).
        """)
    else
        sampler_fn = ext_model == "SAR" ? bayesian_sar :
                     ext_model == "SEM" ? bayesian_sem : bayesian_sdm
        bayes_ext = sampler_fn(W_ext, X_ext, y_ext;
                               n_samples = ext_draws, n_chains = ext_chains,
                               varnames = ["income", "density"],
                               rng = MersenneTwister(ext_seed))
        Markdown.parse("**Sampled** — $(ext_model), $(ext_draws) × $(ext_chains)")
    end
end

# ╔═╡ aabbccde
begin
    if isnothing(bayes_ext)
        md"_(no posterior yet)_"
    else
        # The posterior and the MLE answer the same question on the same
        # data, with priors this diffuse. Disagreement would mean one of
        # them is wrong, which makes this the most useful check to run.
        mle_nm = String.(coefnames(m_ext))
        mle_se = stderror(m_ext)
        rows = String[]
        for (i, nm) in enumerate(bayes_ext.coefnames)
            j = findfirst(==(nm), mle_nm)
            mv = isnothing(j) ? "—" : string(round(coef(m_ext)[j]; digits = 4))
            ms = isnothing(j) ? "—" : string(round(mle_se[j]; digits = 4))
            push!(rows, "| $nm | $(round(bayes_ext.posterior_mean[i]; digits=4)) | $(round(bayes_ext.posterior_std[i]; digits=4)) | $mv | $ms |")
        end
        Markdown.parse("""
        **$(bayes_ext.model_type)** · N = $(bayes_ext.N)

        | parameter | post. mean | post. SD | SAR MLE | MLE SE |
        |---|---|---|---|---|
        $(join(rows, "\n"))

        The MLE column is the SAR fit throughout, so it only lines up when
        the model above is also SAR.
        """)
    end
end

# ╔═╡ bbccdeef
md"""
### The posterior of the spatial parameter

A chain is a joint posterior, so interval statements need no normal
approximation and no delta method. Scalar indexing — `chain[:ρ]` — happens
to agree across Turing's two chain back ends; indexing a vector-valued
parameter does not, which is why `posterior_mean` and `posterior_std` are
the stable way in.
"""

# ╔═╡ ccdeeffa
begin
    if isnothing(bayes_ext)
        md"_(no posterior yet)_"
    else
        sp_sym = bayes_ext.model_type == :SEM ? :λ : :ρ
        draws = vec(collect(bayes_ext.chain[sp_sym]))
        srt = sort(draws)
        qq(p) = srt[clamp(round(Int, p * length(srt)), 1, length(srt))]

        fig_post = Figure(size = (640, 380))
        ax_post = Axis(fig_post[1, 1], xlabel = "$sp_sym",
                       ylabel = "posterior draws",
                       title = "Posterior of $sp_sym  ($(length(draws)) draws)")
        hist!(ax_post, draws; bins = 40, color = (:steelblue, 0.7))
        vlines!(ax_post, [ρ_ext]; color = :black, linewidth = 2,
                linestyle = :dash, label = "true")
        vlines!(ax_post, [mean(draws)]; color = :red, linewidth = 2,
                label = "posterior mean")
        vlines!(ax_post, [qq(0.025), qq(0.975)]; color = :orange,
                linewidth = 1, label = "95% credible")
        axislegend(ax_post; position = :lt)
        fig_post
    end
end

# ╔═╡ deeffaab
begin
    if isnothing(bayes_ext)
        md"_(no posterior yet)_"
    else
        sp2 = bayes_ext.model_type == :SEM ? :λ : :ρ
        d2 = vec(collect(bayes_ext.chain[sp2]))
        s2 = sort(d2)
        q2(p) = s2[clamp(round(Int, p * length(s2)), 1, length(s2))]
        Markdown.parse("""
        * posterior mean: **$(round(mean(d2); digits=4))** (true $(round(ρ_ext; digits=2)))
        * 95% credible region: **[$(round(q2(0.025); digits=4)), $(round(q2(0.975); digits=4))]**
        * P($sp2 > 0) = **$(round(mean(d2 .> 0); digits=4))**
        * P($sp2 > 0.4) = **$(round(mean(d2 .> 0.4); digits=4))**

        That last line is the kind of statement a point estimate and a
        standard error cannot make without assuming normality.
        """)
    end
end

# ╔═╡ effaabbc
md"""
---
## 💡 Try This

1. **Set ρ = 0** and look at the Moran scatterplot: the slope collapses and the LISA map goes almost entirely `:ns`.
2. **Push ρ to 0.9**: the indirect effects swamp the direct ones — that is `(I - ρW)⁻¹` amplifying every shock.
3. **Move the confidence level from 50% to 99%**: the intervals widen by the ratio of normal quantiles, 0.674 → 2.576.
4. **Set the GWR gradient to 0**: both coefficients are global now, GWR spends degrees of freedom to discover nothing, and the correlation with the "true surface" becomes meaningless.
5. **Shrink the adaptive bandwidth to 15**: local coefficients get noisy — overfitting, visible as speckle in the map.
6. **Raise σ to 1.5** and sample: the posterior of ρ widens and may no longer exclude zero, while the MLE's standard error grows to match.
7. **Switch the Bayesian model to SEM** while the data are SAR: the MLE column no longer lines up, and λ̂ is not estimating the same thing as ρ.
8. **Drop to 200 draws**: the posterior mean still lands near the MLE, but the credible region gets visibly ragged — Monte Carlo error, not disagreement.
"""

# ╔═╡ Cell order:
# ╟─a1b2c3d4
# ╟─b2c3d4e5
# ╟─c3d4e5f6
# ╟─d4e5f6a7
# ╟─e5f6a7b8
# ╟─f6a7b8c9
# ╟─a7b8c9da
# ╠═b8c9dae1
# ╟─c9dae1f2
# ╟─dae1f2a3
# ╠═e1f2a3b4
# ╠═f2a3b4c5
# ╟─a3b4c5d6
# ╟─b4c5d6e7
# ╠═c5d6e7f8
# ╟─d6e7f8a9
# ╟─e7f8a9ba
# ╠═f8a9bacb
# ╠═a9bacbdc
# ╟─bacbdced
# ╟─cbdcedfa
# ╠═dcedfaab
# ╟─edfaabbc
# ╟─faabbccd
# ╟─aabbccde
# ╟─bbccdeef
# ╟─ccdeeffa
# ╟─deeffaab
# ╟─effaabbc
