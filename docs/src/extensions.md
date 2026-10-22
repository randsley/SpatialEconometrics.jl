# Package Extensions

Two optional feature sets ship as Julia package extensions. They are not
installed with `SpatialEconometrics`; each one activates automatically once
its trigger packages are in the session.

| Extension | Trigger packages | Provides |
|---|---|---|
| `SpatialEconometricsMakieExt` | `Makie` | `plot_coefficients`, `plot_moran_scatter`, `plot_local_moran`, `plot_gwr_coefficients`, `plot_spatial_effects` |
| `SpatialEconometricsTuringExt` | `Turing` **and** `Distributions` | `bayesian_sar`, `bayesian_sdm`, `bayesian_sem` |

## How the entry points work

All eight function names are declared and exported by `SpatialEconometrics`
itself; the extensions only add methods to them. This matters in practice:
nothing ever writes `using SpatialEconometricsMakieExt`, so a function an
extension declared on its own would be unreachable no matter how it was
exported. You therefore never need to import anything but the parent:

```julia
using SpatialEconometrics
using CairoMakie            # the Makie methods appear here
plot_coefficients(m)
```

Calling one of them before its trigger packages are loaded raises an error
that names the package to load, rather than a bare `MethodError`:

```julia
julia> using SpatialEconometrics

julia> plot_coefficients(m)
ERROR: plot_coefficients requires Makie: run `using Makie` to load the
SpatialEconometricsMakieExt extension, then call it again.
```

To check whether an extension is active:

```julia
Base.get_extension(SpatialEconometrics, :SpatialEconometricsMakieExt) !== nothing
```

## Visualization (Makie)

Install a backend rather than `Makie` itself — `CairoMakie` for files,
`GLMakie` for an interactive window:

```julia
using Pkg; Pkg.add("CairoMakie")
```

Every function returns a `Makie.Figure`, so the results compose with the
rest of Makie (`save`, further axes, custom layouts).

```julia
using SpatialEconometrics, CairoMakie

m = fit_spatial(SAR(), W, X, y; varnames = ["income", "density"])

save("coefs.png", plot_coefficients(m))              # 95% CI by default
save("coefs99.png", plot_coefficients(m; level = 0.99))
save("moran.png",  plot_moran_scatter(y, W))
save("lisa.png",   plot_local_moran(moran_local(y, W), coords))
save("gwr.png",    plot_gwr_coefficients(m_gwr, "income"))
save("effects.png", plot_spatial_effects(spatial_effects(m)))
```

### `plot_coefficients(m; level = 0.95)`

Horizontal coefficient plot with confidence intervals, for any
`StatsAPI.RegressionModel`. The critical value is the normal quantile for
`level`, so changing `level` genuinely changes the interval drawn. A `level`
outside `(0, 1)` raises an `ArgumentError`.

### `plot_moran_scatter(y, W)`

Standardized `y` against its spatial lag `Wy`, with the fitted slope drawn
through it. When `W` is row-standardized that slope *is* Moran's I, so it
should match `moran_global(y, W).I_stat`.

### `plot_local_moran(lisa, coords)`

LISA cluster map, colouring HH, LL, HL, LH and non-significant units.
`coords` must be an `N × 2` matrix describing the same units as `lisa`; a
mismatch raises a `DimensionMismatch` rather than plotting the shorter of
the two.

### `plot_gwr_coefficients(m, varname)`

Maps one GWR local coefficient over the estimation coordinates. `varname`
must be one of `m.coefnames`; an unknown name raises an error listing the
names available.

### `plot_spatial_effects(effects)`

Grouped bar chart of the direct, indirect and total effects from
`spatial_effects`.

See `examples/18_visualization.jl` for a runnable walk-through, and
`notebooks/interactive_extensions_explorer.jl` for a Pluto version.

## Bayesian estimation (Turing)

```julia
using Pkg; Pkg.add(["Turing", "Distributions"])
```

Three models are sampled by NUTS:

```julia
using SpatialEconometrics, Turing, Distributions

r = bayesian_sar(W, X, y; n_samples = 1000, n_chains = 2,
                 varnames = ["income", "density"], rng = MersenneTwister(11))
r = bayesian_sem(W, X, y)    # dependence in the errors
r = bayesian_sdm(W, X, y)    # design augmented with WX
```

| Keyword | Default | Meaning |
|---|---|---|
| `n_samples` | `1000` | Post-warmup draws per chain |
| `n_chains` | `2` | Chains; `MCMCThreads` is used only when Julia has threads |
| `varnames` | `nothing` | Slope names; defaults to `x1, x2, …` |
| `rng` | `Random.default_rng()` | Pass a seeded RNG for a reproducible run |

An intercept is added for you, so `X` holds the covariates only.

### Priors

```
β ~ MvNormal(0, 10I)
ρ, λ ~ Uniform(-0.99, 0.99)
σ ~ truncated(Normal(0, 5), 0, Inf)
```

These are deliberately diffuse, which is why the posterior mean lands on the
maximum-likelihood estimate for the same data — the most useful check to run
on a fit. They are not offered as a considered prior for any particular
application, and they are not currently configurable.

### The result

`BayesianSpatialResult`, a subtype of
[`AbstractBayesianSpatialResult`](@ref) — which the parent package declares,
so your own code can dispatch on it without depending on Turing.

| Field | Contents |
|---|---|
| `chain` | The raw chain object from Turing |
| `model_type` | `:SAR`, `:SEM` or `:SDM` |
| `coefnames` | Parameter names, aligned with the two vectors below |
| `posterior_mean` | Posterior means |
| `posterior_std` | Posterior standard deviations |
| `W`, `N` | The weight matrix and sample size used |

`coefnames` runs intercept, slopes, then the spatial parameter and `σ`. For
`bayesian_sdm` the `WX` terms appear between the slopes and `ρ`, prefixed
`W_`:

```
["(Intercept)", "income", "density", "W_income", "W_density", "ρ", "σ"]
```

!!! note "The `chain` field is not version-stable"
    Turing returns an `MCMCChains.Chains` before v0.45 and a
    `FlexiChains.FlexiChain` from v0.45 on. The two index differently for
    vector-valued parameters: `chain[Symbol("β[1]")]` on the former,
    `chain[:β]` returning one vector per draw on the latter. Scalar
    indexing — `chain[:ρ]` — happens to agree. `posterior_mean` and
    `posterior_std` are stable across both, so prefer them unless you need
    the draws themselves.

```@docs
AbstractBayesianSpatialResult
```

### Why it is affordable

Each likelihood evaluation needs `log|I - ρW|`, and NUTS asks for thousands
of them at a new `ρ` every time. A dense factorization per draw would be
prohibitive, so the extension uses the Chebyshev approximation
(`SpatialEconometrics.CrossSection.chebyshev_logdet`), which precomputes traces once and
then costs a polynomial evaluation per draw. At order 12 it agrees with the
exact log-determinant to several decimal places across the whole parameter
space, so this is the likelihood rather than a stand-in for it.

See `examples/19_bayesian.jl` for a runnable walk-through that checks the
posterior against the MLE.

## Tested versions

The compatibility bounds in `Project.toml` are wide, but the extension
testsets are run against both ends of each range:

| Package | Verified |
|---|---|
| `Makie` | 0.22.10 and 0.24.16 |
| `Turing` | 0.44.5 (MCMCChains) and 0.49.0 (FlexiChains) |

Testsets 25 and 26 of the suite cover the extensions. They skip themselves
when the trigger packages are absent, so a plain `Pkg.test()` does not have
to build Makie and Turing; the `extensions` CI job installs them and runs
the suite again with both live.
