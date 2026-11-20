# Root Estimator of ρ

[`root_rho`](@ref) estimates `ρ` in

```math
y = ρWy + Xβ + f + ε
```

where `f` is an unknown smooth nuisance, approximated in the span of a basis
`B` that you supply: for example, a tensor spline or hat basis on the
locations. Like [`closed_form_rho`](@ref), it needs no log-determinant: `ρ̂`
is a root of a quadratic. Unlike it, it stays unbiased when the basis is
large, and it reports an interval that stays valid when `ρ` is weakly
identified.

```julia
r = root_rho(y, X, W; basis = B)
r.rho, r.se, r.ci_wald, r.cs
```

## The moment

Write `y₀(ρ) = y − ρWy` and `M = I − P_[X,B]`. Smirnov's closed form solves
`(My₀)′W(My₀) = 0`, a quadratic in `ρ`. At the true `ρ` its mean is
`tr(MWMΣ)`, which is not zero. It is of order `K/n`, where `K` is the
number of columns of `[X, B]`, so the estimator is consistent only when the
basis is small against `n`. A basis that approximates a free surface well
enough for valid inference is not small: at `K = 191` and `n = 900` the plain
closed form is biased by about −0.5 (`research/trace_cfe/`).

Jin & Lee (2012) replace `W` by a matrix with a zero diagonal:

```math
g(ρ) = y_0(ρ)'\,C\,y_0(ρ), \qquad C = (A − Δ)M,\quad
Δ = \mathrm{Diag}(AM)\,\mathrm{Diag}(M)^{-1}.
```

Then `diag C = 0` and `C[X, B] = 0`, so `E g(ρ₀) = tr(CΣ) = 0` for any
diagonal `Σ`: the moment is exact under heteroskedasticity of unknown form,
whatever `K` is. `root_rho` takes two steps, as in Jin & Lee's REsd:

1. `A = W′` gives the first-step root `ρ̃` (`r.rho_first`);
2. `A = G(ρ̃)′`, with `G(r) = W(I − rW)⁻¹`, gives `ρ̂` (`r.rho`).

The second step has a smaller variance. Each root is the one at which
`g′(ρ) < 0`; the two roots of a quadratic have slopes of opposite sign, so
that rule picks exactly one.

## Two intervals

**The Wald interval** is `ρ̂ ± z·se`, with `se = √V̂(ρ̂)/|g′(ρ̂)|`. `V̂` is
an HC2 variance (`ûᵢ²/Mᵢᵢ`) with a correction for the estimated
`P_[X,B]y₀` in the linear part of the moment.

**The test-inversion set** is

```math
CS = \{ρ : g(ρ)^2 ≤ z^2\,\hat V(ρ)\},
```

an Anderson–Rubin-type set. It needs the moment's CLT at the true `ρ` only,
not a non-vanishing slope, so it stays valid when the data say little about
`ρ`. Then it is wide, and it may reach a bound of `rho_bounds`
(`r.cs_touches_bound`) or split into more than one interval (`r.cs_split`).
`V̂(ρ)` is an exact quartic in `ρ`, so the endpoints are exact roots, not
grid points. **Report `CS`.** Where the two intervals agree, nothing is lost;
where they differ, the Wald interval is the one that under-covers.

If the moment has no admissible root at either step, `root_found` is false,
`ρ̂` and the Wald interval are `NaN`, and `CS` is still returned.

## Choosing the basis

The intervals are valid only when the approximation error of `f` is small
against the sampling noise. That needs a basis **larger** than the one that
fits `f` best:

- Do not choose `K` by GCV or by cross-validated fit of `f`. In the research
  Monte Carlo those choices over-smoothed for inference.
- Report `ρ̂` and `CS` across a few basis sizes, as `examples/34_root_rho.jl`
  does. A `ρ̂` that drifts as `K` grows is a sign that the smaller bases
  leave part of `f` in the residual.
- The coverage was measured for `K/n` up to 0.30; beyond that `root_rho`
  warns.
- A unit whose leverage in `[X, B]` exceeds 0.99 makes `Δ` ill-posed, and the
  fit is refused. Use fewer basis functions where the data are sparse.
- A hat or B-spline basis sums to one, so do not also put an intercept in `X`.

`β̂` is reported as a point estimate from least squares of `y − ρ̂Wy` on
`[X, B]`, **without standard errors**: the theory covers `ρ` only.

## Cost

Everything is dense: about four `n × n` matrices. `n` is capped at
`max_dense_n = 4000` by default (about 0.5 GB); raise it if the memory is
there. With the largest basis of the Monte Carlo (`K/n` ≈ 0.19), one
replication (a `root_rho` fit plus a `closed_form_rho` fit) took 0.12 s at
`n = 900` and 0.35 s at `n = 1600`.

## What is proved and what is measured

Jin & Lee prove consistency and asymptotic normality for a design of fixed
dimension. The results for a basis that grows with `n` are in
`research/trace_cfe/THEORY.md`: a CLT for the moment with a dense `M`
(Theorem 1), the first step's plug-in having no first-order effect
(Proposition 1), the consistency of HC2 `V̂` (Proposition 2), and the
validity of `CS` without identification (Proposition 6). **No one else has
checked these proofs.**

## What the Monte Carlo shows

`validation/mc_root_rho.jl` uses 108 cells of 1000 replications each. The
cells cross:

- `W`: kNN-6, rook, kNN-20;
- `n` = 400, 900, 1600;
- three basis sizes with `K/n` ≈ 0.05, 0.11, 0.19;
- smooth or rough variances;
- normal or standardised `χ²₃` errors.

Throughout, `ρ₀ = 0.4`. The Monte Carlo SE of a coverage near 0.95 is 0.007.

| | smallest | largest |
|---|---|---|
| coverage of `CS` | 0.935 | 0.975 |
| coverage of the Wald interval | 0.931 | 0.975 |
| `|bias|` of `ρ̂` | 0.000 | 0.060 |
| bias of `closed_form_rho` on the same draws | −1.76 | −0.05 |
| `|ρ̂ − ρ̂_research|` (first draw of each cell) | 0 | 2.5e-14 |

- **The plain closed form fails.** Its bias grows with `K/n` and with the
  density of `W`. At kNN-20 with `K/n` = 0.19 it is about −1.7: the estimate
  sits at the lower bound.
- **`root_rho` stays close to unbiased.** Its largest bias, −0.06, is in the
  weakest cell: kNN-20 at `n = 400` with `K = 82`, where the sd of `ρ̂` is
  0.30.
- **What the weak cells look like.** In the kNN-20 cells at `n = 400`:
  - up to 71 of 1000 samples had no admissible root;
  - `CS` reached a bound of `rho_bounds` in up to 740 of 1000 samples, and
    split into two intervals in up to 17;
  - `CS` still covered at 0.935–0.967.

  In these cells `CS` gives a wide set where the data say little about `ρ`.
  In every other cell, `CS` and the Wald interval agree closely.

## References

- Anderson, T. W., & Rubin, H. (1949). Estimation of the parameters of a
  single equation in a complete system of stochastic equations. *Annals of
  Mathematical Statistics*, 20(1), 46–63.
- Jin, F., & Lee, L.-F. (2012). Approximated likelihood and root estimators
  for spatial interaction in spatial autoregressive models. *Regional
  Science and Urban Economics*, 42(3), 446–458.
- Smirnov, O. A. (2022). Closed-form estimator for spatial autoregressive
  models. *Geographical Analysis*, 54(1), 3–31.
- Stock, J. H., & Wright, J. H. (2000). GMM with weak identification.
  *Econometrica*, 68(5), 1055–1096.
