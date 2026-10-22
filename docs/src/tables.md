# Output & Tables

## Coefficient Tables

Create Tables.jl-compatible coefficient tables from any fitted model:

```julia
m = fit_spatial(SAR(), W, X, y)
ct = coeftable(m; level=0.95)
```

The result implements `Tables.jl` and can be consumed by DataFrames, CSV, etc.

## Export Formats

### LaTeX

```julia
latex_str = to_latex(ct)
```

### CSV

```julia
csv_str = to_csv(ct)
```

### Markdown

```julia
md_str = to_markdown(ct)
```

## Summary Statistics

```julia
ss = summary_stats(m; model_type="SAR")
# Reports: N, K, Log-Lik, AIC, BIC, σ², R²
```

## Multi-Model Regression Table

Stargazer-style side-by-side comparison:

```julia
m_sar = fit_spatial(SAR(), W, X, y)
m_sem = fit_spatial_error(W, X, y)
println(regression_table(m_sar, m_sem; model_names=["SAR", "SEM"]))
```

## API

See the full [API Reference](@ref) for `coeftable`, `summary_stats`, `to_latex`, `to_csv`, `to_markdown`, and `regression_table`.
