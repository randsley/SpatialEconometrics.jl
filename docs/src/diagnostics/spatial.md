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
