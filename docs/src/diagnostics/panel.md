# Panel Diagnostics

## Spatial Hausman Test

Choose between spatial FE and RE specifications:

```julia
ht = spatial_hausman_test(m_fe, m_re)
# ht.stat — Hausman statistic
# ht.p_value — reject H₀ → FE preferred
```

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
