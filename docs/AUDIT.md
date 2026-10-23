# Correctness audit

Status ledger for the systematic audit of every exported estimator and test.
Started 2026-10-23.

## Why this exists

Four separate correctness bugs surfaced in one week — two in the LM test
battery, one in the GWR hat-matrix trace, one set in the gravity estimator.
None was exotic, and the suite was green throughout. The common cause was the
same every time: **the tests checked signs, ranges and thresholds instead of
values against a known answer.**

Measured over `test/runtests.jl` as it stood at the start of the audit:

| | count |
|---|---|
| `@test` assertions | 510 |
| that pin a value (`isapprox` / `≈`) | 75 (15%) |
| that touch a standard error or `vcov` | 21 |
| …of those, that check anything but `size`/`length` | 1 (and it compares two internal code paths) |
| that touch a log-likelihood | 7 (all `isfinite` or `< 0`) |
| that check a p-value | by `0 ≤ p ≤ 1` or direction only |

Most of the 75 value-pinning assertions are Monte-Carlo recovery checks with
`atol` between 0.10 and 0.35. A tolerance that wide passes a materially wrong
formula: the GWR trace bug inverted the wrong factor and still recovered
every coefficient.

Testsets with **no** value-pinning assertion at all: 4, 5, 7, 8, 13, 14, 16,
17, 18, 20, 24, 25.

## Method

For each exported estimator and test, check the headline numbers — not just
the point estimates but the **variance, the log-likelihood and the test
statistic** — against one of:

1. a closed form derived independently,
2. a published number for a standard dataset,
3. a reference implementation written from the textbook formula without
   looking at the package source,
4. an exact but slow computation (dense linear algebra, a solved fixed point,
   a numerical Hessian) that the fast path is supposed to approximate.

A parameter-recovery check against the data-generating truth is **not**
sufficient evidence and does not close an item.

## Verified correct

Each row was checked against an independent reference, not merely reviewed.

| Item | Reference used | Result |
|---|---|---|
| `chebyshev_logdet` | dense `logabsdet(I - ρW)`, 81-cell rook lattice | error ≤ 0.013 nats at \|ρ\|=0.9 at the default order 6; ≤ 3e-3 at order 12. Adequate; default is defensible |
| `loglik`: SAR, SDM, SEM, SLX, SAC, SDEM | Gaussian likelihood written out separately, with the Jacobian terms supplied explicitly | all agree to ≤ 1.4e-14 |
| SAR `vcov` | exact expected information matrix, dense, no stochastic probes; plus a numerical Hessian | matches to ≤ 0.4% at N = 81…900. The residual gap to the numerical Hessian is the expected-vs-observed information distinction and shrinks with N as it should |
| `wald_test` | (Rβ−r)′(RVR′)⁻¹(Rβ−r) | correct |
| `common_factor_test` | delta method on θ + ρβ | Jacobian correct for the SDM coefficient layout |
| `make_rook_weights`, `make_queen_weights` | neighbour counts 2/3/4 and 3/5/8 at corner/edge/interior, row sums, pattern symmetry | correct. `make_queen_weights` previously had **no** test coverage of any kind |
| `knn_weights` | brute-force k nearest by Euclidean distance | neighbour sets match exactly for k = 3, 5 |
| `band_weights` | brute-force `d_min < d ≤ d_max` | pattern matches exactly |
| `distance_weights` | `w_ij = d_ij^-α` | max deviation 8.9e-16 |
| `vuong_test` | Vuong's ω̂ written out from per-observation log-densities on two non-nested OLS fits | **was wrong, now fixed** — see finding 1 below |
| `spatial_trace_powers` | dense `tr(Wᵐ)/N` for m = 1…12 | exact to 1.3e-15 |
| `g_traces` | dense `G = W(I−ρW)⁻¹`, at ρ = −0.6, 0.3, 0.8, with and without the constant projection | exact to 1e-13 in all six cases |
| `_pointwise_loglik` | sums to `loglikelihood(m)` for SAR, SDM, SEM, SLX, SAC, SDEM | exact to 1e-12 |

## Open findings

### 1. `vuong_test` computed the wrong statistic — FIXED

`omega` is meant to be Vuong's ω̂, the standard deviation of the
**per-observation** log-likelihood ratios m_i = ln f₁(yᵢ) − ln f₂(yᵢ):

    ω̂² = (1/n) Σ (mᵢ − m̄)²

The implementation never forms the per-observation contributions. It computes

    omega = sqrt(abs(2*(ll1 - ll2)/n - adj_diff^2) + eps())

from the aggregate log-likelihoods only. `2(ll1−ll2)/n` is twice the mean
log-likelihood difference and bears no relation to the variance of mᵢ. Three
symptoms, all reproduced on two non-nested SLX models (N = 144, where the
per-observation log-likelihood is unambiguous and sums exactly to the stored
`loglik`):

- ω̂ is wrong by a factor of 0.71 (correct 0.2292, reported 0.1634).
- the statistic is 0.9871 against a correct 0.7037, so p = 0.324 against a
  correct 0.482 — a factor of 1.5, enough to flip a conclusion.
- the quantity under the square root goes **negative** whenever model 1 fits
  worse; `abs()` hides that. As a result the statistic is not antisymmetric
  under swapping the arguments: +0.9871 and −0.9804, which should sum to zero.
  A correct Vuong statistic flips sign exactly.

The numerator and the √n scaling were right; only ω̂ was wrong.

The only existing assertion was `0 ≤ vt.p_value ≤ 1`, which the bug satisfies.

**Fix.** `SpatialModelSelection._pointwise_loglik(m)` returns the
per-observation contributions, and `vuong_test` takes ω̂ as their sample
standard deviation. The decomposition needs no per-model-type code: for
these Gaussian likelihoods the observation-wise part is written down from
`residuals(m)` and `σ² = e′e/n`, and the Jacobian is recovered as the
residual between that and the stored `loglikelihood(m)`, so the
contributions sum to it by construction. Verified: both properties hold
exactly for all six model types.

The Jacobian is then shared out evenly across observations. That is a
convention and is documented as one — `log|I − ρW|` is not separable across
observations, and under spatial dependence no split of it is. It cancels
from ω̂ entirely when the two models' Jacobians agree.

Covered by testset 13c, which pins the statistic to the independent
reference, pins its value, asserts exact antisymmetry, and rebuilds the old
ω̂ to assert it is still wrong by a margin that moves the p-value — so the
test guards the defect under any draw, not just this one.

### 2. Reported standard errors and impacts were not reproducible — FIXED

`compute_asymptotic_vcov` (SAR) uses two stochastic trace estimators:
`precompute_neumann_traces` with 250 Hutchinson probes for tr(G) and tr(G²),
and a 30-probe estimate of tr(G′G). Neither takes a seed from the caller and
both draw from the global RNG, so

- fitting the same model twice gives different standard errors (run-to-run
  spread 0.5% on the ρ standard error at N = 81, 0.4% on the intercept), and
- the fit silently consumes global RNG state, perturbing anything the user
  draws afterwards.

The point estimates were unaffected — ρ̂ was identical across eight fits.

**This was wider than the standard errors.** The same power traces are stored
on the model as `trace_powers` and read by `spatial_effects`, so the reported
**direct, indirect and total marginal effects** moved between runs too. For a
spatial model those impacts are the headline output, not a diagnostic. The
same defect was present in `fit_spatial_panel`, including its time-FE branch;
`fit_dynamic_spatial_panel` already threaded a seed.

**Fix.** Two helpers in `SpatialUtils`:

- `spatial_trace_powers(W; max_power, max_exact_n, n_probes, rng)` — exact
  from the eigenvalues of `W`, since `tr(Wᵐ) = Σᵢ λᵢᵐ` and one decomposition
  serves every power.
- `g_traces(W, ρ; project_constant, max_exact_n, n_probes, rng, μ)` — the
  three traces of `G = W(I−ρW)⁻¹` formed from `G` itself. `project_constant`
  removes the constant eigenvector for the panel estimators by projecting `G`
  on both sides, which no longer assumes `W` is row-standardised as the
  closed-form subtraction it replaces did.

Both are exact below `max_exact_n` (default 1000) and fall back to probes
drawn against a caller-supplied `rng` above it. `fit_spatial`,
`fit_spatial_panel` and `fit_dynamic_spatial_panel` gained `max_exact_n`,
`vcov_probes` and `rng` keywords.

Only `tr(GᵀG)` genuinely needed probing in the first place: it is not a
function of the eigenvalues of `W`, being invariant under neither similarity
nor transposition. The other two never did.

Cost of the exact path, measured: 0.14 s at N = 400 and 0.73 s at N = 1024,
against 0.04 s and 0.15 s for the probe path. Under a second at the default
gate, in exchange for exact and reproducible output.

Covered by testset 13d (70 assertions), which checks both helpers against
dense linear algebra, checks the SAR standard errors against the expected
information matrix written out with no probes anywhere — **the first
assertion in the suite to check a standard error against a value rather than
a size** — and asserts bit-identical variance, traces and impacts across
repeated fits, for `fit_spatial` and for `fit_spatial_panel` under both
`IndivFE` and `TwoWayFE`.

### 3. Robustness concerns, not bugs

- `common_factor_test` hardcodes coefficient positions (`beta_start = 2`,
  `rho_idx = n_params`). Correct for SDM, silently wrong for any other layout.
- `lr_test`, `wald_test` and `common_factor_test` all clamp the statistic at
  zero. That hides a convergence failure, which is exactly when the
  unrestricted log-likelihood comes out below the restricted one.
- `spatial_aic` / `spatial_bic` take k = `nobs − dof_residual`, which excludes
  σ². Consistent across models, so model *comparisons* are unaffected, but the
  absolute values differ from the usual convention by 2 and by log n.

## Queue

In priority order — unchecked variance first, since that is where the
suite is blindest and where three of the four bugs this week lived.

1. Remaining variance matrices, none ever checked against a value: SEM, SAC,
   SDEM, SLX, spatial panel (the trace inputs are now exact, but the assembled
   information matrix has not been checked), GS2SLS panel, dynamic GMM, probit.
2. Panel diagnostics — testset 14, zero value pins: `panel_lm_tests`,
   `spatial_chow_test`, `spatial_hausman_test`.
3. Spatial discrete choice — 52 assertions, 5 pins.
4. Spatial filtering, ESF and RE-ESF — 48 assertions; the pins are largely
   self-referential.
5. Dynamic panel and system GMM — testsets 3 and 5, zero to one pin.
6. Simulation (17), regimes (16), tables (18), extension entry points (24).

Done: `vuong_test` (finding 1) and the trace-based variances and impacts
(finding 2). Suite: 576 → 724 passing, none failing.

Closing an item means: an independent reference exists, it is committed as a
test in `test/runtests.jl`, and the row above moves to "Verified correct".
