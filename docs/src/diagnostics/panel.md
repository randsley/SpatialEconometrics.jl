# Panel Diagnostics

## Spatial Hausman Test

Compare two spatial panel specifications; a rejection favours the first.

```julia
ht = spatial_hausman_test(m_fe, m_pooled)
# ht.stat      — Hausman statistic
# ht.df        — number of coefficients compared
# ht.coefnames — which coefficients those were
# ht.p_value   — reject H₀ → the first specification is preferred
```

Coefficients are matched **by name**, using the names the two models share
in the order they appear in the first model. `ht.coefnames` records what was
compared, so the contrast is auditable.

This matters because the specifications need not carry the same terms. A
`NoFE` fit has an intercept that an FE fit does not:

```julia
coefnames(m_fe)      # ["income", "density", "ρ"]
coefnames(m_pooled)  # ["(Intercept)", "income", "density", "ρ"]
```

The intercept is dropped, and `income` is compared with `income`. Matching
positionally instead would contrast `income` with the intercept and build a
large statistic out of unrelated parameters — a decisive-looking rejection
that means nothing.

Pass `names` to restrict the comparison; an unknown name raises an error
rather than silently shrinking it.

```julia
ht = spatial_hausman_test(m_fe, m_pooled; names = ["income", "density"])
```

!!! note "No spatial random-effects estimator"
    The package has no spatial RE estimator, so the textbook FE-vs-RE
    contrast is not available. The same machinery applies to any two
    specifications sharing coefficient names, such as individual against
    two-way effects, or fixed effects against pooled.

## Spatial Chow Test

Test for structural breaks across time periods:

```julia
ct = spatial_chow_test(y, X, W, N, T, break_period)
# ct.stat — F-statistic
# ct.p_value — reject H₀ → structural break exists
```

## Joint LM Tests for Panels

Simultaneously test for spatial lag, spatial error, and random effects (Baltagi, Song & Koh 2003):

```julia
plm = panel_lm_tests(y, X_with_intercept, W, N, T)
# plm.joint_stat — joint test (χ², df=3)
# plm.lag_stat   — marginal spatial lag test
# plm.error_stat — marginal spatial error test
# plm.re_stat    — marginal random effects test
```

## Panel Moran's I

Period-by-period and pooled Moran statistics on panel residuals:

```julia
pm = panel_moran(residuals(m), W, N, T)
# pm.period_results — vector of per-period MoranResult
# pm.pooled — meta-analytic pooled statistic
```
