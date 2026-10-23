# Model Selection & Comparison

## Information Criteria

```julia
spatial_aic(m)  # Akaike Information Criterion
spatial_bic(m)  # Bayesian Information Criterion
```

## Likelihood Ratio Test

For nested models (e.g., SAR nested in SDM):

```julia
lr = lr_test(m_restricted, m_unrestricted)
# lr.stat — LR statistic (χ²)
# lr.df — degrees of freedom
# lr.p_value — reject H₀ → unrestricted model preferred
```

## Vuong Test

For non-nested models (e.g., SAR vs SEM):

```julia
vt = vuong_test(m1, m2)
# vt.stat — Vuong statistic (positive favors m1)
# vt.preference — :model1, :model2, or :indistinguishable
```

## Common Factor Test

Tests whether SDM simplifies to SEM (H₀: θ + ρβ = 0):

```julia
cft = common_factor_test(beta_slope, theta, rho, vcov_full)
# cft.reject — true → SDM preferred over SEM
```

## Wald Test

General linear hypothesis Rβ = r:

```julia
R = zeros(1, length(coef(m)))
R[1, 2] = 1.0  # test H₀: β₁ = 0
wt = wald_test(m, R, [0.0])
```

## Comparison Table

```julia
tbl = model_comparison_table(m_sar, m_sdm, m_sem; names=["SAR","SDM","SEM"])
println(tbl)
```

## Choosing `k` for a knn `W`

Everything above compares specifications on a `W` you have already chosen.
`select_knn_w` chooses the `W` — specifically `k`, when `W` comes from
[`knn_weights`](@ref) — by refitting across a range of candidates and
minimising an information criterion.

```julia
sel = select_knn_w(coords, y, X; k_range = 2:25)   # X without an intercept
println(sel)                                        # the curve and its winner
m = fit_spatial(SAR(), sel.best_W, X, y)            # refit on the chosen W
```

The criterion is not monotone in `k`, which is why there is something to
search: a `W` that is too sparse misses real neighbours, and one that is too
dense dilutes them. On a 200-unit draw generated from a knn `W` at `k = 6`,
AIC falls from 312.7 at `k = 2` to 294.3 at `k = 6` and then climbs to 329.1
at `k = 20`.

`k` matters because `ρ̂` moves with it, monotonically and over a wide range —
0.305 at `k = 2`, 0.482 at the true `k = 6`, 0.630 at `k = 20`, against a
true 0.5 on that same draw. Too sparse understates the dependence; too dense
overstates it.

How often the search finds the generating `k`, over 300 draws with `k`
searched over `2:20` (so 5.3% by chance):

| `N` | `k̂ = 6` | within ±1 | within ±2 |
|---|---|---|---|
| 100 | 42.3% | 68.3% | 84.0% |
| 200 | 57.3% | 86.0% | 95.7% |
| 400 | 77.0% | 95.0% | 99.7% |

The median `k̂` is 6 at every `N`. Read that as the search being well
targeted but not sharp: at `N = 100` it misses the exact `k` more often than
it finds it, and the honest use of the result is the whole curve plus a
sensitivity check, not the argmin alone. `KnnSelectionResult` returns every
candidate's criterion for that reason.

!!! note "AIC is not penalising anything here"
    With the variant and `X` held fixed, the number of estimated parameters
    does not change with `k`, so AIC, BIC and the raw log-likelihood rank the
    candidates in exactly the same order. The criterion is reported so the
    numbers line up with [`spatial_aic`](@ref) elsewhere; its penalty only
    starts to matter once the comparison runs across variants as well.

!!! warning
    A candidate whose fit throws, or whose `ρ̂` lands on the edge of
    `rho_bounds`, is marked `converged = false`, given a `NaN` criterion and
    excluded from the argmin — a failed fit must not win by having no number
    to beat. The criteria are only comparable because `y` and the sample are
    identical across candidates, which `select_knn_w` checks rather than
    assumes.
