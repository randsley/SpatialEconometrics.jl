# SpatialEconometrics.jl

A comprehensive spatial econometrics package for Julia, providing maximum likelihood estimation of spatial regression models, panel data methods, diagnostics, and weight matrix construction.

## Features

- **Cross-sectional models:** SAR, SDM, SEM, SARAR via ML with exact or Chebyshev log-determinant
- **Panel models:** Static and dynamic spatial panels with FE/RE, Lee-Yu bias correction, Direct QML
- **GS2SLS:** Generalized spatial two-stage least squares with IV/endogenous regressors
- **System GMM:** Arellano-Bond / Blundell-Bond for dynamic panels
- **GWR & MGWR:** Geographically weighted regression with adaptive/fixed bandwidths and KDTree
- **Gravity models:** PPML and Negative Binomial with origin/destination/bilateral fixed effects
- **Higher-order models:** SLX, SAC/SARAR with separate lag and error weights, SDEM
- **Spatial regimes:** Regime-varying coefficients with SKATER regionalization and a Chow-type LR test
- **Discrete choice:** Spatial probit by linearized GMM and Albert-Chib Gibbs, ordered probit, marginal effects
- **Spatial filtering:** Eigenvector spatial filtering (ESF) for Gaussian/Poisson/logit/negbin, and RE-ESF by profile REML
- **Spatial diagnostics:** Moran's I (global & LISA), Geary's C, LM tests (lag, error, robust)
- **Panel diagnostics:** Spatial Hausman, Chow, joint LM tests, panel Moran's I
- **Model selection:** AIC/BIC, LR test, Vuong test, common factor test, Wald test
- **Weight matrices:** k-NN, distance-decay, distance-band, block-diagonal, standardization, GAL/GWT I/O
- **Robust inference:** Driscoll-Kraay SCC, Conley spatial HAC
- **Simulation:** Six spatial DGPs, Monte Carlo experiments, power curves
- **Output:** Tables.jl coefficient tables, side-by-side regression tables, LaTeX/CSV/Markdown export
- **Bayesian extension:** Turing.jl integration for Bayesian spatial models
- **Visualization extension:** Makie.jl plotting for LISA maps and coefficient surfaces

## Installation

```julia
using Pkg
Pkg.add("SpatialEconometrics")
```

## Quick Example

```julia
using SpatialEconometrics

# Create spatial weights
W = make_rook_weights(20)

# Fit a spatial autoregressive model
m = fit_spatial(CrossSection.SAR(), W, X, y)

# Diagnostics
mr = moran_global(residuals(m), W)
lm = lm_tests(y, hcat(ones(length(y)), X), W)

# Model comparison
println(model_comparison_table(m_sar, m_sdm, m_sem; names=["SAR","SDM","SEM"]))
```

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
```
