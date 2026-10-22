# API Reference

Every exported name, grouped by the part of the package it
belongs to. Variant and fixed-effects arguments are the shared
tags under *Variants & Specification Tags*, usable unqualified
in any call.

Submodules are not exported, so nothing here needs a module
prefix; see [Getting Started](getting_started.md) for the
reasoning.

- [Variants & Specification Tags](api/tags.md) — Shared tag types.
- [Cross-Sectional Models](api/cross_section.md) — Cross-sectional SAR and SDM by concentrated maximum likelihood, and the effects decomposition that makes their coefficients interpretable.
- [Static Spatial Panels](api/panel.md) — Static spatial panels with fixed effects.
- [Dynamic Spatial Panels](api/dynamic_panel.md) — Panels carrying both a time lag and a spatial lag, with bias correction.
- [GS2SLS / SARAR Panels](api/gs2sls.md) — Instrumental-variables estimation of SARAR panels with endogenous regressors.
- [Dynamic Panel GMM](api/gmm.md) — Arellano-Bond and Blundell-Bond GMM for dynamic panels.
- [Spatial Error Models](api/sem.md) — Models placing the dependence in the disturbance rather than in `y`.
- [Higher-Order Models](api/higher_order.md) — SLX, SAC/SARAR and SDEM specifications.
- [Spatial Regime Models](api/regime.md) — Coefficients that differ across groups of units, with SKATER regionalization.
- [Spatial Discrete Choice](api/discrete_choice.md) — Spatial probit and ordered probit, by linearized GMM and by Gibbs sampling.
- [Spatial Filtering](api/filtering.md) — Eigenvector spatial filtering (ESF) and its random-effects variant.
- [Geographically Weighted Regression](api/gwr.md) — GWR and multiscale GWR: coefficient surfaces rather than single coefficients.
- [Gravity & Flow Models](api/gravity.md) — PPML and negative binomial gravity models with high-dimensional fixed effects.
- [Spatial Diagnostics](api/diagnostics.md) — Moran's I, Geary's c, LISA and the Anselin LM specification battery.
- [Panel Diagnostics](api/panel_diagnostics.md) — Hausman, Chow and joint LM tests for panels.
- [Model Selection](api/model_selection.md) — Information criteria and tests for choosing between specifications.
- [Weight Matrices](api/weights.md) — Constructing, standardizing, inspecting and reading weight matrices.
- [Robust Inference](api/robust.md) — Driscoll-Kraay and Conley spatial HAC covariance estimators.
- [Simulation & Power Analysis](api/simulation.md) — Six data-generating processes, Monte Carlo experiments and power curves.
- [Calibration & Input-Output](api/calibration.md) — GRAS and cross-entropy matrix balancing, and SAM multipliers.
- [Formula Interface](api/formula.md) — The `@formula` terms this package adds, and panel reordering.
- [Output & Tables](api/tables.md) — Coefficient tables and LaTeX / CSV / Markdown export.
- [Package-Level & Extensions](api/package_level.md) — functions on the package itself and the extension entry points.

`@formula` is re-exported from
[StatsModels.jl](https://juliastats.org/StatsModels.jl/stable/);
see [`slag`](@ref), [`fe`](@ref) and [`endo`](@ref) for the
spatial terms this package adds to it.
