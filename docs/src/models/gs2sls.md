# GS2SLS / SARAR Panels

For panels carrying a spatial lag in `y` *and* spatial dependence in the
disturbance, with regressors that are themselves endogenous. Maximum
likelihood is not available here, so estimation is by generalized spatial
two-stage least squares: instruments for the endogenous regressors, spatial
instruments for `Wy`, and a GMM step for the error parameter.

## Specification

```math
y_{it} = \rho W y_{it} + X_{it} \beta + u_{it},
\qquad u_{it} = \lambda W u_{it} + \varepsilon_{it}
```

Two parameters, and they mean different things: `rho` is the spatial lag
in the dependent variable, so it induces spillovers in `y`; `lambda` is
dependence among the disturbances, which affects efficiency and standard
errors but carries no spillover interpretation.

## Estimation

```julia
m = fit_gs2sls_panel(TwoWayFE(), W, X_exog, X_endog, Z_iv, y, N, T;
    exog_names = ["x1", "x2"],
    endog_names = ["tax_rate"])
# m.rho:    spatial lag parameter
# m.lambda: spatial error parameter
```

`Z_iv` holds the external instruments for the columns of `X_endog`, and must
have at least as many columns as `X_endog` for the model to be identified.

## Formula DSL

`endo(x, z1 + z2)` marks `x` as endogenous and names its instruments, so the
exogenous block, the endogenous block and the instrument block do not have to
be assembled by hand:

```julia
m = fit_gs2sls_panel(
    @formula(y ~ x1 + endo(tax, z1 + z2) + fe(region) + fe(year)),
    table, W; panel = :region, time = :year
)
```

See [Formula Interface](../formula.md) for the spatial terms, and
[Spatial Error Models](sem.md) for the case where there is no lag in `y`.
