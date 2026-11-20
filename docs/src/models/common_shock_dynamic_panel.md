# Short-T Dynamic Panel with Common Shocks

[`fit_cre_dynamic_spatial_panel`](@ref) handles a short dynamic spatial panel
whose individual effects are time-invariant. Often the units also respond, each
in its own way, to shocks that hit them all at once: a recession, a policy
change, a commodity price. [`fit_common_shock_dynamic_panel`](@ref)
implements Bai & Chen's (2026) conditional QML for that case, with `T` fixed
and `N → ∞`.

!!! warning "A preprint"
    Bai & Chen (2026) is an arXiv preprint (2608.22499v1), not yet
    refereed. The package checks its formulas against independent
    computations (below), but the paper's theory is as stated by its authors.

## The model

```math
y_t = ρWy_t + φy_{t-1} + X_tβ + 1_Nδ_t + Λf_t + ε_t, \quad t = 1, …, T,
```

with `r` common factors `f_t` and unit loadings `Λ = (λ_1, …, λ_N)'`. The
time effects `δ_t` are free, so `X` has no intercept. The idiosyncratic
errors `ε_it` are independent across units and over time, with variances
`σ²_t` that may change over time but not across units.

With `T` fixed, the `N` loadings cannot be estimated one by one: that is the
incidental-parameter problem behind Nickell bias. Bai & Chen project them on
observed controls instead:

```math
λ_i = Az_i + η_i, \quad
z_i = [(Q_X)_i, (WQ_X)_i, …, (W^LQ_X)_i, (y_0)_i, (Wy_0)_i, …, (W^Ly_0)_i],
```

the **spatially enriched random loadings** (SERL). `Q_X` summarises each
unit's regressors (by default its mean of `X` over periods `1..T`), and `L`
is the **SERL order**. The residual loadings `η_i` have mean zero and a common
covariance `Σ_η` given the data. Only `A`, `Σ_η` and the factor path
`F = (f_1, …, f_T)'` are estimated, never a parameter per unit.

`F` is identified up to rotation, so it is normalised to `F = [I_r; F_2]`. Its
top `r × r` block must be full rank. Bai & Chen's Remark 3 (a constant first
factor for time-invariant heterogeneity) is not implemented.

## What `L` assumes

The likelihood is conditional on `y_0`. It is right only when
`E(λ_i | y_0, X, W) = Az_i` for the chosen `L` (Bai & Chen's Assumption 2).

If `y_0` was itself produced by a long-running spatial process, it carries
the neighbours' loadings through `(I − ρW)⁻¹`, and no finite `L` captures that
exactly. The paper says so (§2.5): a finite `L` is then an approximation.
Its Table 4 measures the cost at `T = 5`, `ρ = 0.5`:

| `N` | coverage of `ρ`, `L = 0` | coverage of `ρ`, `L = 1` |
|---|---|---|
| 500 | .642 | .918 |
| 1000 | .312 | .935 |

At `L = 0` coverage **falls** as `N` grows, the signature of an inconsistent
estimator. At `L = 1` it does not fall between the two sizes, but two sizes
cannot show that it never will. This package's Monte Carlo (below) carries
the comparison to `N = 4000`.

The same assumption applies to `Q_X`: the default unit mean of `X` can leave
part of `λ_i` behind when the regressors are serially correlated with the
loadings. The Monte Carlo below measures that case too.

`L` defaults to 1. `r` is required and has no default: the theory takes it as
known and gives no rule for choosing it.

## Data layout

As in [`fit_dynamic_spatial_panel`](@ref): `y` and `X` stack `N` units within
each of `T_total = T + 1` periods, **period 0 first**. Period 0 supplies
`y_0`; `X`'s period-0 rows are not used. `Q` replaces the default `Q_X`, and
`controls` replaces the whole of `z_i`. The controls are centred and scaled
before estimation, so `A` is reported in standard-deviation units.

```julia
ev = eigvals(Matrix(W))   # once per W
fit = fit_common_shock_dynamic_panel(y, X, W, N, T + 1; r = 2, L = 1,
                                     eigenvalues = ev)
```

## Estimation

The criterion is the Gaussian quasi log-likelihood conditional on `y_0`,
eq. (17). Its only Jacobian term is `T log|I − ρW|`, evaluated from the
eigenvalues of `W`. They are computed once per fit, at `O(N³)` cost, or
passed in with `eigenvalues` when many panels share one `W`.

1. **Start** (Appendix C): per-period OLS at `ρ = 0`, then principal
   components of the residuals for `F` and the loadings.
2. **Algorithm 1**: given `ρ`, alternate the closed-form updates of `δ` and
   `A`, the ECME updates of `Σ_η`, `σ²_t` and `F`, and GLS for `(φ, β)`;
   then maximise over `ρ` with the rest held fixed, and repeat.
3. **Newton polish**: Newton steps on the full parameter vector, with step
   halving, until the largest `|∂ℓ/∂α|/N` is below `tol` (default `1e-8`).
   If that is not reached, the fit throws: it never returns a point that does
   not solve the first-order conditions.

## Standard errors

The score of `ρ` (and of `φ`) for one unit depends on the other units'
errors through `(I − ρW)⁻¹`, so the unit scores are not independent, and their
plain outer product misses cross-unit terms. Bai & Chen rearrange each pair
term onto the later unit of the pair, which makes the contributions
`g_i` a martingale-difference array (Appendix D). The covariance of the score
is then estimated by their eq. (54), quoted in their notation:

```math
\widehat Ω_M(α) := \frac{1}{N}\sum_{i=1}^{N} g^†_i(α)\,g^†_i(α)' + \frac{1}{N}\sum_{i=1}^{N}\sum_{j<i} v_{ij}(α)\,v_{ij}(α)'.
```

Their `α` is the full parameter vector. The estimator is the one-unit
parts' outer products plus the outer products of every pair term. The pair
terms are symmetric in `i` and `j`, so `Ω̂_M` does not depend on the order
of the units, although each `g_i` does.

The reported `vcov` is the `θ = (ρ, φ, β)` block of the sandwich
`Ĥ⁻¹Ω̂Ĥ⁻¹`, with the pair part scaled by `N/(N − p)` (eq. 23, the
paper's preferred version; `p` counts all parameters). `vcov_uncorrected`
leaves the factor out. The `θ` block equals the profiled covariance of §3.4.
Both allow non-normal `η_i` and `ε_i`.

**Cost.** The pair part involves all `N(N−1)/2` pairs. Because the blocks of
`(I_N⊗R_T(φ) − ρW⊗I_T)⁻¹` are `φᵏ(I − ρW)^{−(k+1)}`, the pair terms come out
of `N × N` matrix products with no loop over pairs, in `O(TN³)` time and
about eight `N × N` matrices of memory: about 1 GB at `N = 4000`, by count, not measured.

## How it is checked

The test suite (testset 43) compares values with values:

- the log-likelihood with one built from scratch over all `NT` observations,
  with `log|det|` of the full `NT × NT` Jacobian;
- the analytic score and Hessian with automatic differentiation;
- the pair matrices with the explicit `T × T` blocks of
  `(W⊗I_T)C⁻¹` and `(I_N⊗L_T)C⁻¹`, and `Σ_i g_i` with the score;
- `Ω̂_M` under a relabelling of the units;
- **`E[Ω̂_M(α_0)] = Var(Σ_i g_i(α_0))`**, by simulation with skewed `χ²`
  errors and the data held fixed. Equations (52)–(53) make this exact in
  finite samples, so it is a null-distribution test of the variance. The
  plain outer product of the unit scores fails the same test.

## Monte Carlo

`validation/mc_serl_sdpd.jl` runs Bai & Chen's long-running design (§6.2.1)
at `T = 5`, `ρ = 0.5`, with 500 panels per `N`, each fitted at `L = 0` and
`L = 1`:
- `W` is 8-nearest-neighbour;
- the regressors are `x_it = 0.5λ_i + √0.75 h_it`, with `h` an AR(1);
- `y` runs 100 periods before `y_0`.

Intervals are 95%. The Monte Carlo standard error of a coverage near .95 is
about .010.

| `N` | `L` | bias `ρ` | sd `ρ` | cov `ρ` | cov `φ` | cov `β₁` | cov `β₂` |
|---|---|---|---|---|---|---|---|
| 500 | 0 | −.018 | .018 | .823 | .950 | .926 | .944 |
| 500 | 1 | −.006 | .019 | .942 | .948 | .920 | .952 |
| 1000 | 0 | −.015 | .012 | .780 | .948 | .948 | .918 |
| 1000 | 1 | −.001 | .012 | .964 | .966 | .952 | .918 |
| 2000 | 0 | −.014 | .009 | .640 | .898 | .942 | .914 |
| 2000 | 1 | +.001 | .010 | .926 | .932 | .924 | .914 |
| 4000 | 0 | −.013 | .007 | .398 | .866 | .886 | .860 |
| 4000 | 1 | +.002 | .007 | .936 | .936 | .888 | .854 |

All 4000 fits converged except 3 at `N = 500`, `L = 0`. Those 3 threw errors
rather than return a non-solution.

- **At `L = 0`, `ρ̂` is inconsistent**, as the paper finds. Its bias stays
  near −.014 while its sd falls from .018 to .007, so coverage falls from .82
  to .40.
  The fall is slower than in the paper's Table 4 (.642 and .312 at `N = 500`
  and 1000). Our network and factor path are drawn differently from theirs.
- **At `L = 1`, `ρ̂` holds** up to `N = 4000`: bias at most .006, coverage .93
  to .96.
- **Coverage of `β` falls with `N` at both `L`.** At `N = 4000` it is .85 to
  .89, and SERL order does not explain it. `validation/mc_serl_sdpd_diagnostics.jl`
  separates the possible causes (200 panels each):

  | design | `N` | fit | bias `β₁` | bias `β₂` | SE/sd `β₁` | SE/sd `β₂` | cov `β₁` | cov `β₂` |
  |---|---|---|---|---|---|---|---|---|
  | baseline (§6.1.1) | 4000 | `L = 0` | −.001 | +.001 | 0.97 | 0.98 | .950 | .955 |
  | long-running | 4000 | `L = 1` | +.008 | −.009 | 1.00 | 0.94 | .925 | .865 |
  | long-running | 4000 | `L = 2` | +.008 | −.009 | 1.00 | 0.94 | .925 | .865 |
  | long-running | 2000 | `L = 1`, `Q` = means | +.009 | −.010 | 1.04 | 1.08 | .935 | .930 |
  | long-running | 2000 | `L = 1`, `Q` = every period | −.001 | +.001 | 1.02 | 1.07 | .955 | .965 |

  - **The standard errors are right.** In the baseline design the model is
    correctly specified at `L = 0`, and mean SE / sd is 0.94 to 1.03 for all
    four parameters, with coverage .940 to .955. In the long-running design,
    SE / sd for `β` is close to 1 as well.
  - **The problem is a bias in `β̂`, and `L` does not cause it.** `L = 2` gives
    the same bias as `L = 1`, to four decimals.
  - **The cause is the default `Q_X`.** Here the regressors' deviations from
    `0.5λ_i` are an AR(1). The best predictor of `λ_i` from `x_i1, …, x_iT`
    then weights the periods unequally, and the unit mean leaves part of
    `λ_i` in `η_i`, correlated with `X`. Passing every period of `x` as `Q`
    reduces the bias to within its Monte Carlo error, about .0012.

The bias is fixed while the sd shrinks like `1/√N`, so it matters only at
large `N`. When the regressors may be serially correlated with the loadings,
pass `Q = hcat(X_1, …, X_T)`. This costs `KT` controls per spatial order
instead of `K`.

## References

- Bai, J., & Chen, J. C. (2026). Fixed-T dynamic spatial panel model with
  common shocks. arXiv:2608.22499v1 [econ.EM]. Preprint.
- Li, L., & Yang, Z. (2021). Spatial dynamic panel data models with
  correlated random effects. *Journal of Econometrics*, 221(2), 424–454.
