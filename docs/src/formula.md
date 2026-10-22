# Formula Interface

Every estimator that takes matrices also takes a `@formula`, re-exported from
[StatsModels.jl](https://juliastats.org/StatsModels.jl/stable/), plus three
spatial terms this package adds. The formula methods take a `table` — any
Tables.jl-compatible source, such as a `DataFrame` — and build the design
matrices, the spatial lags and the fixed-effects projection from the column
names, so the stacking order does not have to be managed by hand.

## The spatial terms

| Term | Meaning |
|---|---|
| `slag(x1 + x2)` | Spatially lagged regressors `W*x`, giving a Durbin specification |
| `fe(region)` | Absorb fixed effects for a grouping column |
| `endo(x, z1 + z2)` | Mark `x` endogenous and name its instruments |

`fe` appearing twice, as `fe(region) + fe(year)`, is how a two-way
specification is written; the panel and time keywords then name which columns
carry the panel and period identifiers.

## Examples

```julia
# SDM panel with two-way FE
m = fit_spatial_panel(
    @formula(y ~ x1 + x2 + slag(x1 + x2) + fe(region) + fe(year)),
    table, W; panel = :region, time = :year)

# GS2SLS with an endogenous regressor and its instruments
m = fit_gs2sls_panel(
    @formula(y ~ x1 + endo(tax, z1 + z2) + fe(region) + fe(year)),
    table, W; panel = :region, time = :year)

# Cross-section SDM
m = fit_spatial(
    @formula(y ~ x1 + x2 + slag(x1 + x2)),
    table, W)

# SEM, cross-section
m = fit_spatial_error(@formula(y ~ x1 + x2 + fe(region)), table, W)

# SEM panel
m = fit_spatial_error(@formula(y ~ x1 + x2 + fe(region) + fe(year)),
    table, W; panel = :region, time = :year)

# Gravity
m = fit_gravity(PoissonGravity(),
    @formula(flows ~ log_dist + common_lang), table;
    origin = :exporter, dest = :importer)
```

## Panel ordering

The matrix interfaces expect a panel stacked unit-within-period, that is
`[unit₁t₁, unit₂t₁, …, unitₙt₁, unit₁t₂, …]`. Data rarely arrives that way,
and a silently mis-stacked panel produces plausible-looking nonsense rather
than an error, so the reordering is available directly:

```julia
perm, N, T_per = extract_panel_permutation(table, :region, :year)
```

It returns the permutation together with the panel dimensions it inferred,
and throws an `ArgumentError` on an unbalanced panel rather than guessing.
The formula methods apply this themselves.
