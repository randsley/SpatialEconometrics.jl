# Choosing a model

Find the row that matches the data and the question, then open the
docstring of the entry point (`api/INDEX.md` names its file) before
fitting. The docstrings state each estimator's assumptions and limits, and
those decide whether the result can be reported. The notes here say what to
check first; they do not replace the docstring.

## Contents

- Cross-section, continuous outcome
- Cross-section, other outcomes
- Static panels
- Dynamic panels
- Common factors and cross-sectional dependence
- Space-time systems
- Policy evaluation and causal designs
- Networks and peers
- Flows between places
- Semiparametric and varying coefficients
- Spatial dependence in the variance
- Choosing `W` by the data

## Cross-section, continuous outcome

One observation per unit. Run the LM battery first (SKILL.md, step 2).

| The dependence is in… | Model | Entry point |
|---|---|---|
| the outcome: `y = ρWy + Xβ + ε` | SAR | `fit_spatial(SAR(), W, X, y)` |
| the outcome and neighbours' covariates | SDM | `fit_spatial(SDM(), W, X, y)` |
| the disturbance: `u = λWu + ε` | SEM | `fit_spatial_error(W, X, y)`; `estimator = :gmm` if the error variance may differ across units |
| neighbours' covariates only | SLX | `fit_slx(W, X, y)` (OLS) |
| the outcome and the disturbance | SAC / SARAR | `fit_sac(W_lag, W_error, X, y)` |
| neighbours' covariates and the disturbance | SDEM | `fit_sdem(W, X, y)` |

- **SDM nests SAR, SLX and SEM** (SEM through the common-factor
  restriction `θ = −ρβ`), so it is a safe general model when the tests are
  ambiguous. `common_factor_test` tests SDM against SEM; `lr_test` tests
  the nested restrictions.
- **Report `spatial_effects(m)`** for SAR and SDM. For SEM, SLX and SDEM,
  `β` (and `θ` for the neighbours' covariates) can be read directly,
  because no `(I − ρW)⁻¹` multiplies them.
- **`method`.** The default `ChebyshevApprox()` is fast, and the reported
  `ρ̂` and log-likelihood are polished on the exact log-determinant.
  `ExactLU()` is exact throughout and needed for any search beyond
  `|ρ| ≥ 1`. `fit_sac` searches two parameters and is not polished, so
  pass `ExactLU()` when its exact log-likelihood matters (for example in
  an LR test against a polished fit).
- **Coefficients that vary over space:** `fit_gwr(X, y, coords)` (one
  bandwidth) or `fit_mgwr` (one bandwidth per covariate). Note the argument
  order: `X, y, coords`, not `W`. `fit_mgwr` warns when its back-fitting
  does not settle; see `warnings-and-errors.md`.
- **Coefficients that differ by known or found regimes:**
  `fit_spatial_regime(W, X, y, regimes)`, with `regime_chow_test` to test
  equality, and `skater_regimes` to find contiguous regimes from the data.
- **Filtering instead of modelling:** `fit_esf(X, y, W)` adds selected
  eigenvectors of `W` as regressors so the other coefficients can be read as
  aspatial; `fit_re_esf` treats them as random effects.

## Cross-section, other outcomes

- **Binary:** `fit_spatial_probit(W, X, y)` (GMM), or
  `fit_bayesian_spatial_probit` (Gibbs). `marginal_effects` gives the
  effects; probit coefficients are on the latent scale.
- **Ordered:** `fit_spatial_ordered_probit(W, X, y)`.
- **Counts or flows between places:** see "Flows between places".

## Static panels

`N` units over `T` periods, stacked with units varying fastest (SKILL.md).

- **SAR or SDM with fixed effects:**
  `fit_spatial_panel(SAR(), TwoWayFE(), W, X, y, N, T)`. `fe_spec` is
  `NoFE()`, `IndivFE()`, `TimeFE()` or `TwoWayFE()`. The default
  `LeeYuTransformation()` removes the incidental-parameter bias that
  individual effects cause in `σ²` and `ρ` when `T` is short.
- **SEM panel:** the panel method of `fit_spatial_error`.
- **Endogenous regressors, or SARAR in a panel:** `fit_gs2sls_panel`. If the error variance may differ across units (by size, or by neighbour count), pass `errors = :heteroskedastic` (Kelejian & Prucha 2010 moments, cluster-robust covariance, Hansen's J).
- **Testing:** `panel_lm_tests`, `panel_moran`, `spatial_hausman_test` (FE
  against RE), `spatial_chow_test`.

## Dynamic panels

A time lag of `y` as well as a spatial lag. The choice turns on `T`.

| Situation | Entry point |
|---|---|
| `T` moderate to long, fixed effects | `fit_dynamic_spatial_panel(variant, fe_spec, W, X, y, N, T)`, bias-corrected QML |
| `T` short and fixed, `N` large, correlated random effects | `fit_cre_dynamic_spatial_panel` (Li & Yang 2021) |
| `T` short, `N` large, common shocks (factors) | `fit_common_shock_dynamic_panel` (Bai & Chen 2026) |
| no spatial lag, standard dynamic panel | `fit_dynamic_gmm(DifferenceGMM() or SystemGMM(), …)` |

- Effects for dynamic models come from `dynamic_spatial_effects`, which
  separates the short run from the long run.
- The short-`T` estimators assume errors identically distributed across
  units; their docstrings say what heteroskedasticity across units does.
  Read them before using either.
- `fit_common_shock_dynamic_panel` needs the number of factors `r`, which
  its theory does not choose. Its docstring reports a bias in `β̂` under the
  default `Q` when the regressors are serially correlated, and how to avoid
  it.

## Common factors and cross-sectional dependence

Spatial dependence and common shocks (business cycles, policy changes)
both make units co-move, and a spatial lag fitted to data with a common
factor absorbs the factor into `ρ̂`. Check before trusting `ρ̂` in a panel:

1. `pesaran_cd(resid, N, T)` and `alpha_exponent`, or both at once with
   `dependence_verdict(resid, N, T)`, which gives a recommendation.
2. If a strong factor is present: `fit_sp_cce(y, X, W, N, T)` (spatial
   common correlated effects) or `fit_interactive_fe(y, X, W, N, T)`
   (factors estimated by principal components). `fit_sp_cce` warns when
   `ρ̂` was absorbing a factor.

## Space-time systems

- **Space-time autoregression:** `fit_star(y, X, W, N, T)`;
  `space_time_irf`, `spatial_forecast`.
- **Structural spatial VAR** (several variables per location, a causal
  ordering): `fit_spvar(Y, W, N, T; ordering)`; check identification with
  `spvar_order_condition`; responses with `space_time_stir` and `spvar_irf`;
  `lr_spatial_interaction` tests it against `fit_nested_panel_var`.
- Forecasts from a non-stationary fit warn: see `warnings-and-errors.md`.

## Policy evaluation and causal designs

- **Did a policy spill over to untreated neighbours?** Build the exposure
  (`exposure`, `ring_exposure`), run `check_interference` first (it says
  whether the design can identify a spillover at all), then
  `fit_spatial_did(y, treat, W)`.
  - `channels = :split` gives the treated and the controls separate
    spillover coefficients (Chagas et al. 2016); `spillover_equality`
    tests whether one would do. In a split fit report
    `att_total_treated`, not `att_direct`, which is imprecise when the
    treated are clustered.
  - If the outcome's errors are likely spatially correlated (they usually
    are), the default OLS intervals for the spillovers and the total effect
    under-cover, whatever `vcov`. Use `errors = :spatial` (ML, assumes a
    constant error variance) or `errors = :spatial_gmm` (Kelejian & Prucha
    GMM, robust to unequal variances; in a staggered design it is the
    package's own extension of their cross-section theory, so say so).
- **One or a few treated units:** `fit_synthetic_control(Y, treated, T0;
  W or coords)`, which screens donors for spillover contamination;
  `placebo_test`, `spillover_gaps`.
- **Shift-share (Bartik) designs:** `shift_share_design`, then
  `fit_shift_share` (AKM standard errors by default), with
  `rotemberg_weights`, `balance_test`, `exposure_concentration` and
  `shift_share_moran` as checks.

## Networks and peers

- **Peer effects** (linear-in-means): check `peer_identification` first,
  since the reflection problem can make the endogenous and contextual
  effects inseparable. Then `fit_peer_effects(y, X, G)`.
- **Several networks at once:** `fit_multiplex_sar(y, X, [W1, W2, …])`.
- **`W` from an adjacency matrix:** `network_weights(A)`; groups:
  `group_weights`; `network_moran` for dependence on a network.

## Flows between places

`fit_gravity(PoissonGravity(), flows, X, origin_id, dest_id; fe =
BilateralFE())`: PPML by default, `NegBinGravity()` for overdispersion;
origin, destination and bilateral fixed effects; `multilateral_resistance`.
Zero flows are kept, as PPML allows.

## Semiparametric and varying coefficients

- **A nonlinear effect of one covariate:** `fit_partially_linear_sar(y, X,
  z, W)` (Su & Jin 2010). Check `spectral_overlap` first: when the smoother
  can reproduce `W`'s dominant spatial modes, the smooth and `ρ` compete
  and `ρ̂` is biased. The docstring explains the check.
- **No log-determinant:** `closed_form_rho` (Smirnov 2022). It has no
  consistency proof, and its standard errors are unreliable; read its
  docstring before reporting one.
- **No log-determinant, with a large basis for an unknown surface `f`:**
  `root_rho(y, X, W; basis = B)` (Jin & Lee 2012). Its moment has mean
  zero for any basis size, so it is not biased by `K/n` as
  `closed_form_rho` is. Report its test-inversion set `cs`: it stays valid
  when `ρ` is weakly identified, where the Wald interval does not. `β` has
  no standard errors. Do not choose the basis size by GCV; report `ρ̂`
  across a few sizes. For a penalised basis, pass
  `tensor_pspline(u, v; k)...` and a small `lambda`: there is no default,
  because the shrinkage of `f` biases `ρ̂` and both intervals as `λ` grows.
  Report `ρ̂` across a few values of `λ`, with `r.df`.
- **Coefficients that drift over time in a panel:** `fit_tv_coef_sar`
  (Liang, Gao & Gong 2022). `ρ` stays constant; only `β` varies.

## Spatial dependence in the variance

Volatility that spreads between neighbours (returns, residuals):
`fit_spatial_log_garch(y, W; Z)` (Sato & Matsuda 2021).
- `y` must be mean-zero (returns or residuals of a mean model) and nonzero.
- Its quasi-likelihood has no maximum as `λ → 1`, so some samples have no
  estimate at all: the fit throws rather than return the edge. See
  `warnings-and-errors.md`.
- `α` has no standard error. Without `Z`, `λ` and `ρ` are separated only by
  the variance part and are estimated imprecisely; `λ + ρ` is estimated
  much better than either.
- Unknown locations: `covariate_weights` builds `W` from covariates. It
  biases `ρ̂` towards zero, more data does not cure it, and its quality
  cannot be checked from the data. Say so when reporting.

## Choosing `W` by the data

`select_knn_w(coords, y, X; k_range)` refits at each `k` and minimises an
information criterion. `bayes_spec_probs` and `sdem_roadmap` compare
specifications and weight matrices by marginal likelihood. Report that `W`
was selected, since the reported SEs do not account for the selection.
