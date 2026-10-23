# Spatial Diagnostics

## Moran's I (Global)

Tests for global spatial autocorrelation:

```julia
mr = moran_global(y, W)
# mr.I_stat — Moran's I statistic
# mr.z_stat — z-score
# mr.p_value — p-value (permutation-based)
```

## LISA (Local Moran's I)

Identifies local clusters and spatial outliers:

```julia
lisa = moran_local(y, W; n_perm=999)
# lisa.local_I — local I for each observation
# lisa.cluster_type — :HH, :LL, :HL, :LH, or :ns
# lisa.pseudo_p — pseudo p-values
```

## Geary's C

Alternative global measure (inversely related to Moran's I):

```julia
gc = geary_c(y, W)
# gc.C_stat < 1 → positive autocorrelation
# gc.C_stat > 1 → negative autocorrelation
```

## LM Tests

Choose between spatial lag and spatial error specifications:

```julia
lm = lm_tests(y, X_with_intercept, W)
# lm.lag       — LM test for spatial lag
# lm.error     — LM test for spatial error
# lm.lag_robust   — Robust LM lag (controlling for error)
# lm.error_robust — Robust LM error (controlling for lag)
```

**Decision rule (Anselin 2005):**
1. If neither LM-lag nor LM-error is significant → OLS is fine
2. If only one is significant → use that specification
3. If both significant → compare robust versions
4. If both robust versions significant → consider SDM

## Durbin Score Tests

Step 4 above says "consider SDM" without saying how. This battery is the
answer: Rao score tests for `y = rho*W*y + X*beta + W*X_d*theta + eps`, all
computed from the same OLS residuals, so the whole specification question
costs one regression.

```julia
t = sdm_tests(y, X_with_intercept, W)
# t.joint         — rho = 0 AND theta = 0, chi2(k+1); run this first
# t.lag           — rho = 0 given theta = 0 (this is lm_test_lag)
# t.robust_lag    — rho = 0, robust to local theta != 0
# t.durbin        — theta = 0 given rho = 0, chi2(k)
# t.robust_durbin — theta = 0, robust to local rho != 0
# t.terms         — theta_j = 0 one covariate at a time, chi2(1) each
```

Each is a [`ScoreTestResult`](@ref), which unlike `LMTestResult` carries its
`df`.

**Decision rule (Koley & Bera 2023):**

1. If `joint` does not reject → OLS is adequate, and nothing below is worth
   reading
2. Only `robust_lag` rejects → a lag, not Durbin terms: fit `SAR`
3. Only `robust_durbin` rejects → Durbin terms, not a lag: fit `SLX`
4. Both reject → fit `SDM`

Read the robust pair, not the plain pair: under a lag alone, `durbin` rejects
16% of the time at the 5% level, where `robust_durbin` rejects 5.1%. The
`sdm_tests` docstring has the measured size and power tables.

!!! note "Notation"
    Koley & Bera write `lambda` for the lag and `gamma` for the Durbin terms.
    This package reserves `lambda` for the spatial error coefficient, so read
    `lambda -> rho` and `gamma -> theta`.

!!! warning
    `X` must already include its intercept column, as `lm_tests` requires.
    The lagged intercept is dropped from the Durbin block automatically:
    `W*1 = 1` under row standardisation makes it collinear with the intercept.
