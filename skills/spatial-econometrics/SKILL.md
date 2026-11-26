---
name: spatial-econometrics
description: Fit and interpret spatial econometric models in Julia with SpatialEconometrics.jl — spatial lag (SAR), error (SEM), Durbin (SDM), SLX, SAC, panels (static, dynamic, short-T, common factors), GWR/MGWR, spatial probit, gravity, spatial DiD and synthetic control, shift-share, peer effects, STAR/SpVAR, spatial log-GARCH, and the weight matrices, LM tests, Moran's I and robust (Conley, Driscoll-Kraay) inference around them. Use this skill whenever the user wants to analyse data where units interact with their neighbours, whether regions, municipalities, countries, firms in a network or grid cells. That includes choosing between spatial models, building or checking a W matrix, testing for spatial autocorrelation, computing direct and indirect (spillover) effects, or reading a warning from one of these fits, even if they never name the package or say "spatial econometrics".
---

# SpatialEconometrics.jl

A Julia package for spatial and structural econometrics: 33 submodules, one
umbrella `using SpatialEconometrics`. Its docstrings are careful about what
each estimator does and does not support, and this skill is built to get
that care into your analysis rather than leave it in the docs.

## Before the first fit

**Is it installed?** Check with `julia -e 'using SpatialEconometrics'`. If
not, it is not in the General registry yet, so install from GitHub:

```julia
using Pkg
Pkg.add(url = "https://github.com/randsley/SpatialEconometrics.jl", rev = "v0.8.0")
Pkg.add("StatsAPI")     # coef, stderror, vcov, coefnames, loglikelihood
```

Julia 1.10 or later. Put the analysis in its own project folder
(`julia --project=.`) rather than the global environment when the user has
one.

**Work in script files, not one-liners.** Each `julia` call pays the package
load and compile time again: tens of seconds for the first fit. Write the
analysis to a `.jl` file, run it once with `julia --project=. analysis.jl`,
and extend the file rather than issuing many small `julia -e` calls. If the
user runs a REPL of their own, give them the code to paste instead.

## The workflow

Most analyses follow these steps. Skip the ones that don't apply, but don't
reorder them: each one tells you what the next should be.

1. **Build `W` and check it.** `spatial_connectivity(W)` before anything
   else. Islands, disconnected components and non-row-standardised `W` all
   change what the estimators mean. Read `references/weights.md` when you
   build a `W` from coordinates, contiguity files or covariates.
2. **Look for dependence.** `moran_global(resid_or_y, W)`, then the LM
   battery: `lm_tests(y, hcat(ones(N), X), W)`. Note that `lm_tests` wants
   the intercept column and the `fit_*` functions do not (they add it).
   Read the two **robust** statistics: each conditions on the other kind of
   dependence, so ideally only one fires. If both do, consider SDM, SAC or
   SDEM; `sdm_tests` covers the Durbin forms.
3. **Choose and fit.** See `references/choosing-a-model.md`, which maps the
   data and the question to an entry point.
4. **Read every warning.** A spatial fit that warns is telling you its
   estimate or its standard error does not mean what it normally would. Do
   not drop the warning from your report. `references/warnings-and-errors.md`
   says what each one means and what to do.
5. **Report effects, not coefficients, for lag models.** In SAR, SDM and
   their panel versions, `β` is not a marginal effect: a change in one
   unit's `x` feeds back through `(I − ρW)⁻¹`. Report
   `spatial_effects(m)` (direct, indirect, total, with delta-method SEs).
   For dynamic panels use `dynamic_spatial_effects` (short and long run).
6. **Choose the standard errors.** The default covariance assumes the
   model's own error structure. For residual dependence the model does not
   capture, `robust_vcov(m, ConleySpatialHAC(...))` (distance-decayed) or
   `robust_vcov(m, DriscollKraayVCOV(...))` (panels: cross-sectional plus
   serial). `spatial_effects(m, spec)` takes the same spec.
7. **Compare and tabulate.** `spatial_aic`, `spatial_bic`, `lr_test`,
   `vuong_test`, `common_factor_test` (SDM against SEM); then
   `regression_table(m1, m2, …)` and `to_latex` / `to_markdown` / `to_csv`.

## Conventions that cause silent errors

These don't throw. They give wrong answers, so check them in every script.

- **Panel stacking.** Panel `X` and `y` are stacked with **units varying
  fastest** within each period: all `N` units for period 1, then all `N`
  for period 2. `W` is the `N × N` cross-sectional matrix, never the stacked
  `NT × NT` one. A wrong order misaligns every unit with its neighbours
  and nothing errors.
- **Row order of `W` against the data.** `read_gal(path; ids = table_ids)`
  aligns `W` to the table's order; without `ids` it uses the file's record
  order. When building `W` from coordinates, build it from the same rows,
  in the same order, as `y` and `X`.
- **Intercepts.** `X` for `fit_spatial`, `fit_spatial_error`,
  `fit_spatial_panel` and most other fits excludes the intercept. Passing a
  column of ones throws a rank error naming the column, which is the good
  case; a constant column under a different name is caught the same way.
- **`ρ` is only comparable across `W`s that are scaled alike.** A
  row-standardised `W` puts `ρ` in roughly `(−1, 1)`; a binary or
  distance-weighted `W` does not.

## Reporting

- Give `ρ̂` (or `λ̂`) with its SE, the log-likelihood, `N`, how `W` was
  built (type, `k` or cutoff, standardisation, islands), and which
  covariance the SEs use.
- For lag models, report the effects table, not `β̂` alone, and say that
  the indirect effect is the spillover.
- If a fit warned, say so and say what it means for the number it
  concerns. An estimate on a search bound has no valid SE.
- Do not claim more than the estimator supports. Several docstrings state
  limits (no SE for log-GARCH `α`, no consistency proof for the closed-form
  `ρ`, a proxy `W` that biases `ρ̂`). Check the docstring before
  interpreting an unusual result.

## Where to look

| Need | Read |
|---|---|
| Which estimator fits the data and question | `references/choosing-a-model.md` |
| Building and checking `W` | `references/weights.md` |
| A warning or error from a fit | `references/warnings-and-errors.md` |
| A function's signature, arguments and caveats | `references/api/INDEX.md`, then the submodule file it names |

The `references/api/` files are the package's own docstrings, regenerated
from the source for each release, so they are the authority on signatures
and keyword names. Grep them rather than guessing a keyword:
`grep -n "fit_sac" references/api/*.md`. Inside Julia, `?fit_sac` shows the
same text.
