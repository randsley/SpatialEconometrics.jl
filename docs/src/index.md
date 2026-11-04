# SpatialEconometrics.jl

```@raw html
<p class="se-tagline">
Spatial and structural econometrics in Julia — cross-section, panel and
higher-order models, diagnostics and weight-matrix construction, built on
sparse linear algebra, multiple dispatch and automatic differentiation.
</p>
```

```@raw html
<div class="se-badges">
  <span class="se-badge se-badge-accent">v0.3.0</span>
  <span class="se-badge">27 submodules</span>
  <span class="se-badge">3,415 tests</span>
  <span class="se-badge">Julia 1.10+</span>
  <span class="se-badge">MIT</span>
</div>
```

## What's here

```@raw html
<div class="se-grid">
  <div class="se-card">
    <h3>Cross-sectional models</h3>
    <ul>
      <li>SAR, SDM, SEM by concentrated ML</li>
      <li>SLX, SAC/SARAR, SDEM — separate lag and error weights</li>
      <li>Exact LU or Chebyshev log-determinant</li>
      <li><code>spatial_effects</code>: direct, indirect, total</li>
    </ul>
  </div>
  <div class="se-card">
    <h3>Panel models</h3>
    <ul>
      <li>Static spatial panels, FE/RE, Lee–Yu bias correction</li>
      <li>Dynamic panels by Direct QML</li>
      <li>GS2SLS / SARAR panels with endogenous regressors</li>
      <li>Arellano–Bond and Blundell–Bond system GMM</li>
    </ul>
  </div>
  <div class="se-card">
    <h3>Heterogeneity</h3>
    <ul>
      <li>GWR and MGWR, adaptive or fixed bandwidth</li>
      <li>Spatial regimes with SKATER regionalization</li>
      <li>Spatial probit, ordered probit, marginal effects</li>
      <li>Eigenvector spatial filtering and RE-ESF</li>
    </ul>
  </div>
  <div class="se-card">
    <h3>Flows and calibration</h3>
    <ul>
      <li>Gravity models: PPML and negative binomial</li>
      <li>Origin, destination and bilateral fixed effects</li>
      <li>Multilateral resistance terms</li>
      <li>GRAS balancing and Leontief multipliers</li>
    </ul>
  </div>
  <div class="se-card">
    <h3>Diagnostics</h3>
    <ul>
      <li>Moran's I (global and local), Geary's C</li>
      <li>LM battery: lag, error, and both robust forms</li>
      <li>Koley–Bera Durbin score tests</li>
      <li>Panel Hausman, Chow, joint LM, panel Moran</li>
    </ul>
  </div>
  <div class="se-card">
    <h3>Inference and output</h3>
    <ul>
      <li>Driscoll–Kraay SCC and Conley spatial HAC</li>
      <li>AIC/BIC, LR, Vuong, common-factor and Wald tests</li>
      <li>Coefficient and side-by-side regression tables</li>
      <li>LaTeX, CSV and Markdown export</li>
    </ul>
  </div>
</div>
```

Two optional extensions load themselves when their packages are present:
Turing.jl for Bayesian spatial models, and Makie.jl for LISA maps and
coefficient surfaces. See [Package Extensions](@ref).

## Installation

```julia
using Pkg
Pkg.add(url = "https://github.com/randsley/SpatialEconometrics.jl")
```

## A worked example

Simulate a lag process on a 15×15 rook lattice, then work the way you would
with real data: ask what kind of dependence is there, fit the specification
the tests point to, and read off the effects.

```@example quickstart
using SpatialEconometrics, Random

W = row_standardize(make_rook_weights(15))   # 15×15 lattice → 225 units
X = reshape(randn(MersenneTwister(20261030), 225), 225, 1)
y, _ = simulate_spatial(SAR_DGP(0.6, [2.0], 1.0), W, X; rng = MersenneTwister(1))

nothing # hide
```

**Which dependence?** The LM battery separates a lag process from an error
process. The two robust statistics are the ones to read: each conditions on
the other form of dependence, so only one of them should fire.

```@example quickstart
lt = lm_tests(y, hcat(ones(225), X), W)

print(lt.lag)
print(lt.error)
print(lt.robust_lag)
print(lt.robust_error)
```

The robust lag test is decisive and the robust error test is not, which is
the signature of a lag process — correctly, since that is how the data were
generated.

**Fit it.**

```@example quickstart
m = fit_spatial(SAR(), W, X, y; varnames = ["income"])
coeftable(m)
```

**Read the effects.** With `ρ ≠ 0` a coefficient is not a marginal effect: a
change in one unit's `x` propagates through the lattice and back. The direct,
indirect and total decomposition is what you report.

```@example quickstart
spatial_effects(m)
```

## Choosing a model

| If the dependence is in… | the model is | entry point |
|---|---|---|
| the outcome, `y = ρWy + Xβ + ε` | SAR | `fit_spatial(SAR(), …)` |
| the disturbance, `u = λWu + ε` | SEM | `fit_spatial_error(…)` |
| the outcome and the covariates | SDM | `fit_spatial(SDM(), …)` |
| the covariates alone | SLX | `fit_slx(…)` |
| the outcome and the disturbance | SAC / SARAR | `fit_sac(…)` |
| the covariates and the disturbance | SDEM | `fit_sdem(…)` |
| coefficients that vary over space | GWR / MGWR | `fit_gwr(…)`, `fit_mgwr(…)` |
| coefficients that vary by regime | regimes | `fit_spatial_regime(…)` |

[Getting Started](@ref) walks through weights, diagnostics and estimation in
order; the [API Reference](@ref) documents every exported name.

## Contents

```@contents
Pages = [
    "getting_started.md",
    "weights.md",
    "models/cross_sectional.md",
    "models/panel.md",
    "models/dynamic_panel.md",
    "models/sem.md",
    "models/gwr.md",
    "models/gravity.md",
    "models/higher_order.md",
    "models/regime.md",
    "models/discrete_choice.md",
    "models/filtering.md",
    "diagnostics/spatial.md",
    "diagnostics/panel.md",
    "diagnostics/model_selection.md",
    "robust.md",
    "simulation.md",
    "tables.md",
    "extensions.md",
    "api.md",
]
Depth = 1
```
