# Short-T Dynamic Panel with Correlated Random Effects

[`fit_dynamic_spatial_panel`](@ref) corrects the incidental-parameter bias
of a dynamic spatial panel with the Yu, de Jong & Lee device, which needs a
long `T`. Many panels are short: a handful of years across many regions or
firms. For those, [`fit_cre_dynamic_spatial_panel`](@ref) implements Li &
Yang's (2021) M-estimator, which is consistent with `T` fixed and `n → ∞`.
This page keeps Li & Yang's symbols, since the coefficient names follow
them: `n` is the number of units and `ρ` the time lag (see
[Notation](@ref)).

## The model

```math
y_t = ρy_{t-1} + λ_1W_1y_t + λ_2W_2y_{t-1} + X_tβ + Zγ + μ + α_t1_n + u_t,\quad
u_t = λ_3W_3u_t + v_t,
```

for `t = 1, …, T`, with the individual effects **correlated random effects**
in Mundlak's form, `μ = X̄π + ε`, where `X̄` is the mean of `X_0, …, X_T`.
The time effects `α_t` enter as period dummies. `W_2` (the space-time lag)
and `W_3` (the spatial error) are optional: pass `nothing` to drop `λ₂` or
`λ₃` together with its estimating equation.

Mundlak's device is what lets time-invariant regressors `Z` stay in the model:
under fixed effects they would be swept out with `μ`.

## Why `m` is required

The panel is observed from period 0, but the process started earlier. Li &
Yang assume it started `m` periods before period 0 from an exogenous
`y_{−m}`. Then `y_0` carries `ε`, and with `T` fixed the conditional QMLE,
which treats `y_0` as given, is inconsistent.

The M-estimator corrects each biased component of the conditional score by
its expectation (Li & Yang's Lemma 2.1). The corrections involve `y_0` only
through `m`. So `m` is a required keyword with no default:

```julia
fit = fit_cre_dynamic_spatial_panel(y, X, Z, W1, N, T + 1; m = 10,
                                    W2 = W2, W3 = W3, compute_cqmle = true)
```

- `m = 0` says `y_0` is exogenous. The conditional likelihood is then exact,
  the corrections vanish, and the M-estimator **is** the conditional QMLE.
  The test suite checks this identity.
- For a stationary process, `𝓑ᵐ = (B₁⁻¹B₂)ᵐ` dies out geometrically, so any
  large `m` gives nearly the same corrections. Li & Yang report robustness to
  a wrong `m` in that sense. A small `m` is a different assumption about
  `y_0`. In example 31, `m = 2` moves `ρ̂` by .003, a tenth of its
  standard error.

## Data layout

As in [`fit_dynamic_spatial_panel`](@ref): `y` and `X` stack `n` units (the argument `N`) within
each of `T_total = T + 1` periods, **period 0 first**. Period 0 supplies `y_0`
and enters `X̄`; estimation uses periods `1..T`. `Z` is an `n × q` matrix of
time-invariant regressors without an intercept, or `nothing`.

A column of `X` that does not vary over time duplicates its own mean `X̄`.
The fit refuses that design as rank deficient. Put such a variable in `Z`.

## Estimation

`β` and `σ²_v` are concentrated out, and the adjusted quasi-score is solved
in `δ = (φ, ρ, λ₁[, λ₂][, λ₃])`, where `φ = σ²_ε/σ²_v`. No `nT × nT` matrix
is formed. `Ω⁻¹` and `log|Ω|` have closed forms through `J̄_T = J_T/T`, and
the corrections are traces of `n × n` products, through powers of `𝓑`.

Two points about solving it.

- **The adjusted score can have more than one root.** Started from the
  conditional QMLE, the solver can find a root near it, not near the truth.
  The start is therefore 2SLS on first differences, which removes `μ`. The
  instruments are `y_{t−2}` with its spatial lags, `ΔX_t`, `ΔX_{t−1}` and
  their spatial lags. `φ` and `λ₃` start from moment estimates on the
  residuals.
- **`φ = ∞` is a false root.** As `φ` grows, `Ω⁻¹` loses its between-unit
  part and the `φ` score tends to zero, so a solver can drift towards
  `φ = ∞`. The solver therefore works with the score in `log φ`, which tends
  to `−n/2`, by Levenberg–Marquardt in `(log φ, ρ, λ)`.

A poor start for `φ` can still lead the solver towards `φ = ∞`. If the
solver does not reach a root, it is run once more from the same start with
`φ = 1`. If that fails too, the fit throws: it never returns a non-root.

## Standard errors

The covariance is the sandwich `Ĵ⁻¹Γ̂Ĵ⁻¹'`, with `Ĵ` the Jacobian of the
adjusted score and `Γ̂` an estimate of `Var(S*(ψ₀))`.

`Γ̂` is **not** Li & Yang's §3 hybrid estimator. It adapts the corrected
plug-in route of Li, Meng & Yang (2026), written for the fixed-effects
version of this model:

1. Split `y_0 = η + C₀ε`, where `η` is independent of the in-sample shocks
   `u = (ε; v_1; …; v_T)`. Given `η`, every score component is
   `u'Au + a(η)'u + c`, so `Γ = E Var(S*|η)` has a closed form in the shock
   variances and third and fourth moments.
2. `η` is unobserved. It is replaced by its plug-in through
   `ε̂ = E[ε|e]`, and the bias that plug-in adds, a trace in
   `Var(ε − ε̂)`, is subtracted.
3. The third and fourth moments of `ε` and `v` are estimated from the within
   and between parts of the residuals.

At the true parameters and moments, `Γ̂` is unbiased for the exact
`Var(S*)`. The test suite checks this against a brute-force covariance of the
linear-quadratic forms over all shocks, pre-sample ones included. Without the
bias correction in step 2, the same check fails.

`vcov_hessian` keeps `Ĵ⁻¹` for comparison. It is not a valid covariance when
`m > 0`, because the adjusted score is not an information-equal score.

**Assumptions.** The errors may be non-normal but must be homoskedastic. Li &
Yang's Supplementary Appendix (§II) shows that heteroskedasticity in `ε` or
`v` enters the score corrections themselves, so it breaks the estimator and
not only the variance.

**Cost.** The estimator needs dense `n × n` algebra. `Γ̂` forms dense
`n(T+1) × n(T+1)` matrices, one for each of `σ²`, `φ`, `ρ` and the `λ`s, so
its memory grows as `(n(T+1))²`: by count, about 1.4 GB at `n = 1000`,
`T = 3` before temporaries (not measured).

## Monte Carlo

`validation/mc_cre_sdpd.jl` replicates Li & Yang's Table 1 design: DGP1,
`T = 3`, `m = 10`, queen `W₁ = W₃`, group-interaction `W₂`, `X_t` as in
their footnote 7. It has 1000 replications at `n = 100` and 500 at
`n = 200`, and normal, normal-mixture and centred `χ²₃` errors. The output
is `validation/mc_cre_sdpd_output.txt`. Selected rows, normal errors (mean,
sd in brackets):

| `n` | | CQMLE | M-estimator | Li & Yang's M-est. | mean SE | coverage |
|---|---|---|---|---|---|---|
| 100 | `ρ = .3` | .351 (.034) | .301 (.030) | .3014 (.033) | .032 | .958 |
| 100 | `π = 1` | .858 (.108) | .995 (.100) | .9940 (.106) | .102 | .963 |
| 100 | `φ = 1` | .764 (.241) | .974 (.277) | .9813 (.295) | .278 | .925 |
| 100 | `λ₃ = .2` | .155 (.122) | .145 (.126) | .1521 (.121) | .125 | .941 |
| 200 | `ρ = .3` | .351 (.025) | .300 (.021) | .3010 (.024) | .022 | .956 |
| 200 | `φ = 1` | .785 (.177) | 1.002 (.202) | .9919 (.208) | .198 | .942 |
| 200 | `λ₃ = .2` | .188 (.085) | .177 (.087) | .1761 (.085) | .088 | .954 |

- **The CQMLE does not improve with `n`.** Its `ρ̂` stays at .351, and
  `π̂` and `φ̂` stay low. The M-estimator removes these biases at both sizes,
  and its means agree with Li & Yang's to within about one Monte Carlo
  standard error.
- **`λ₃` stays biased down at `T = 3`**, by .055 at `n = 100` and .023 at
  `n = 200`, as in Li & Yang's table. The M-estimator does not correct it.
- **Coverage of the 95% sandwich intervals** is between .92 and .97 for
  `β`, `ρ` and the `λ`s in all six cells. For `σ²_v` and `φ` it is
  .89–.93 at `n = 100` and .92–.94 at `n = 200`: these are the slowest to
  reach their asymptotics, and Li & Yang's own robust SEs are low for them
  by a similar amount.
- **The normality-based SEs fail under non-normal errors.** With `χ²₃`
  errors, the intervals from `vcov_hessian` cover `σ²_v` 79% of the time at
  `n = 100` and 81% at `n = 200`: more data does not help. The sandwich
  covers 89% and 94%.

All 4500 fits succeeded. An earlier run lost 5, each with an error rather
than a wrong estimate; `docs/AUDIT.md` gives the causes, the fixes and the
full table.

## References

- Li, L., & Yang, Z. (2021). Spatial dynamic panel data models with
  correlated random effects. *Journal of Econometrics*, 221(2), 424–454.
- Li, L., Meng, X., & Yang, Z. (2026). Dynamic spatial panel data models with
  fixed effects: alternative approaches and application. *Spatial Economic
  Analysis*, accepted.
- Mundlak, Y. (1978). On the pooling of time series and cross section data.
  *Econometrica*, 46(1), 69–85.
