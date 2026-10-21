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
