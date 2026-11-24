# Trace-corrected closed-form estimator (research, not package code)

Exploratory Monte Carlo for an idea that came out of reading Geniaux (2026)
in full on 2026-11-18. ROADMAP #13 pass 2 is **on hold by decision**. Nothing
here is part of the package, exported or tested by `test/runtests.jl`.

## The idea

With a projection first stage `P_B` (the basis `B` spans the smooths of the
covariates and of the coordinates), the closed-form estimator's (CFE's)
moment at the true `ρ` has an exact mean:

    E[g_n(ρ₀)] = −σ² tr(P_B W) / n

This is the own-observation term that Geniaux removes by cross-fitting (his
Remark 7). It is O(K/n), so it is first-order once K is comparable to √n,
and a free surface makes K large. Two corrections remove it in closed form,
and each keeps `ρ̂` as the root of one quadratic:

- **Pooled:** in the moment's second factor, use `e₁ + τe₀` and `e₂ + τe₁`,
  with `τ = tr(P_B W)/(n − K)`. This is the same as replacing `W` by `W + τI`
  in the instrument, and it makes the quadratic form trace-zero (Lee 2007).
- **Exact diagonal:** use a vector `d` in place of `τ`, where `d` solves
  `(M∘M) d = diag(P_B W)` and `M = I − P_B`. Then `A = MW + MDM` has a zero
  diagonal, so the moment has mean zero under any heteroskedasticity, and its
  variance needs no third or fourth moments.

**Robust variance of the moment:** `2σ̂′(A_s∘A_s)σ̂ + Σᵢ(âᵢ² − noiseᵢ)σ̂ᵢ²`,
where:
- `σ̂ᵢ² = ûᵢ²/Mᵢᵢ`;
- `â = MW P_B y₀`;
- `noise = diag(G(Q′Σ̂Q)G′)`, with `G = MWQ` and `Q` an orthonormal basis of
  span `B`.

`ρ̂`'s standard error is that square root divided by `|b − 2ρ̂a|`.

**Prior art (checked 2026-11-18; see `ExtRef/README.md`).** Both corrections
are Jin & Lee (2012):
- the pooled `τ` is their homoskedastic root estimator with
  `P = W′ − tr(W′M)/(n − d)·I` (§3.1);
- the zero diagonal under heteroskedasticity is their §3.2, which builds it
  more simply, as `P = W′ − Diag(W′M)Diag(M)⁻¹`, without solving `(M∘M)d`.
  Their two-step REsd uses `G′(ρ̂)` in place of `W′`, and is more efficient
  than either (see Results).

Liu & Yang (2015) apply the same per-unit idea to the QML score. All of these
take a regressor matrix of fixed dimension, with `PM` bounded in row and
column sums.

What these papers do not cover (and see the comparison under Results):
- a basis whose size `K` grows with `n`, where `M` is dense and the trace term
  is first-order. The spline GMM papers (Chen, Song & Yu 2025; Lu, Tian &
  Yang 2024) avoid this regime with rate conditions such as `k²/n → 0`;
- the noise term in the robust variance, which is negligible at fixed `d`;
- the link to Geniaux's own-observation bias.

Su (2012) estimates `ρ` with linear moments only.

## Theory

`THEORY.md` holds theory step 1, a CLT for the moment when `M` is dense
(Theorem 1, proved given Hall & Heyde's martingale CLT, and the
estimator's Corollary, proved under a sign rule for the root), and a first analysis of step 2, the robust
variance, each with its numerical check. None of it has been checked by
anyone else.

## Scripts

Run from the repository root with `julia --project=. research/trace_cfe/<file>.jl`.
Every script uses fixed seeds, and its output is committed beside it as
`<file>_output.txt`.

| Script | What it measures | Run time |
|---|---|---|
| `tc_first_look.jl` | Bias of the plain CFE and the pooled correction; K = 29, 92, 191; constant variance | 35 s |
| `tc_first_look_het.jl` | The same with spatially varying variances; adds the per-unit approximation `d_i = (P_B W)_ii / M_ii` | 36 s |
| `tc_lib.jl` | Shared code: the design objects, the three estimators and the variance of the moment | — |
| `tc_null.jl` | Null-distribution check: fixed design, `f` in span `B`, errors redrawn 2,000 times; normal and χ²₃ errors, constant and varying variances, `ρ₀` = 0 and 0.4. The first block of its output is a 20-replication warm-up; ignore it | 43 s |
| `tc_decomp.jl` | Splits the robust variance's bias into its quadratic and linear parts, before the noise term was subtracted | 6 s |
| `tc_jinlee.jl` | Jin & Lee (2012, §3.2) against the exact diagonal in the concurvity design: their `P = W′ − Diag(W′M)Diag(M)⁻¹` (JL-W) and their two-step REsd with `G(ρ̂)′` (JL-G), all with the same robust variance; 16 cells at K = 100 and 4 at K = 199, 500 replications each | 15 min |
| `tc_clt.jl` | Theory step 1: the conditions of the CLT for a dense `M` (norm ratios, the `Δ` bound, Lindeberg for the linear part) and the distribution of the standardised moment, for local and global bases with K ≈ 2√n and K ≈ 0.15n, plus a power check with two complete groups. See `THEORY.md` | 45 s |
| `tc_clt_moments.jl` | Theorem 1 with `2 + δ` moments: `tc_clt.jl`'s hat-basis design with standardised `t₃` and Pareto (tail index 3) errors, which have no fourth moment, n = 400 to 4900, R = 2,000; the power check is `t₃` errors with two complete groups. See `THEORY.md` | 2 min |
| `tc_coverage_tails.jl` | Coverage of `ρ̂` (Jin & Lee's first step) under normal, χ²₃, t₃ and Pareto(3) errors, n = 400 to 2500, R = 1,000: robust, oracle (exact `V`) and homoskedastic intervals, the last as the power check. See `THEORY.md` | 80 s |
| `tc_roots.jl` | The Corollary's root selection: the discriminant's sign, how often `cfe_root`'s rule (the root nearest the OLS ratio) differs from the sign rule the proof uses, and how far the other root lies, in `tc_coverage_tails.jl`'s design with normal and t₃ errors, R = 1,000 | 25 s |
| `tc_sign.jl` | The sign of `J` (A9): 3,000 hub-and-spoke designs searched for `J < 0` (0.7% have it); on the strongest, the sign-rule root against the other root, Wald and `CS` coverage, and how often `root_rho`'s check warns, for the one-step and REsd moments; false warnings on kNN designs with `J > 0` (none in 1,000 fits) | 1.5 min |
| `tc_resd.jl` | Proposition 1: Jin & Lee's two-step REsd against the infeasible estimator with `C(ρ₀)`, n = 400 to 2500, R = 500, normal errors; the power check is a fixed wrong plug-in `C(ρ₀ + 0.3)`. One dense solve per replication to form `G(ρ̃)` | 10 min |
| `tc_vhat_lo.jl` | Theory step 2: Kline, Saggio & Sølvsten's leave-one-out variances `y₀ᵢûᵢ/Mᵢᵢ` in the robust `V̂`, against HC2 and the oracle, in `tc_vhat.jl`'s design and seeds; the quadratic and linear parts are reported separately. The first output line is a 5-replication warm-up; ignore it | about 10 min |
| `tc_vhat_lo_lin.jl` | The same check with `f` scaled so the linear part is 90% of `V`. It shows the leave-one-out quadratic part collapsing (negative from K/n = 0.28) when the signal is strong. The first output line is a warm-up | about 10 min |
| `tc_vhat_lo_hat.jl` | The strong-signal check with a tensor hat basis (leverage ≤ 0.94 up to K/n = 0.25), adding CJN's `(M∘M)⁻¹û²`; a cell whose `M∘M` is singular is skipped and says so. Run with the argument `weak` for `tc_vhat.jl`'s unscaled `f` (`tc_vhat_lo_hat_weak_output.txt`). The first output line is a warm-up | about 8 min per run |
| `tc_coverage_vhat.jl` | Coverage of `ρ̂` at the feasible root with the oracle, HC2, leave-one-out, CJN and homoskedastic `V̂`, in `tc_vhat_lo_hat.jl`'s hat-basis design at both signal strengths, `ρ₀ = 0.4`, R = 1000. Negative `V̂` counts as not covering and is reported. The first output line is a warm-up | not timed (tens of minutes) |
| `tc_cjn_bias.jl` | The exact bias of CJN's quadratic part, `4tr(ΩΛ[(MΣM)∘(MΣM)]Λ)`, against simulation, and its plug-in correction, in the hat-basis design at ρ₀, normal and `χ²₃` errors, R = 1000. The first output line is a warm-up | not timed (tens of minutes) |
| `tc_cjn_lin.jl` | The exact bias of CJN's linear part (skewness, cross and noise terms) against simulation, and E V̂/V for CJN with both parts corrected by plug-in, in the hat-basis design at ρ₀, both signal strengths, R = 1000. The first output line is a warm-up | not timed (tens of minutes) |
| `tc_coverage_cjnc.jl` | Coverage of `ρ̂` with CJN corrected for both parts' exact bias, against the oracle, HC2 and plain CJN, in `tc_coverage_vhat.jl`'s design and seeds (its columns repeat exactly). The first output line is a warm-up | 25 min |
| `tc_approx.jl` | Proposition 3: `f` outside `span B`. The exact moment bias `β = r′(W+D)r + r′r_W`, `c_β = β/√V` and `J`, against the simulated bias and coverage of `ρ̂` (HC2 `V̂`), hat basis K = 10 to 170, two surface amplitudes, R = 1000. The first output line is a warm-up | 20 s |
| `tc_smoother.jl` | Proposition 4: a tensor P-spline smoother in place of the projection, λ from 10⁻³ to 10², two surface amplitudes, R = 1000. Theorem 1's KS check, the shrinkage bias against Proposition 3, and coverage with the oracle, the general `V̂`, the projection plug-in and the idempotent shortcut moment. The first output line is a warm-up | 1.3 min |
| `tc_smoother_vhat.jl` | Proposition 4(c): the smoother's HC2 `V̂` at `ρ₀`, with and without the shrinkage `r` in `s`; the gap against its exact mean under normal errors, and the shrunk-signal shortfall `Δ_b`; power check: the first-order gap with `b` in place of `μ`. R = 500. The first output line is a warm-up | 0.4 min |
| `tc_rbc.jl` | A λ rule with a coverage guarantee: bias-corrected moments `C_λ = A_λ − P_BA_λP_B + M₀D₀M₀` (unbiased for every `f` in span(B)), λ chosen by `V̂/g′²`; exact `c` and sd, Wald coverage per λ and at the chosen λ, and the same rule on the plain smoother (power check). tc_lambda's design, R = 500 at a = 1 and 4. The first block is a warm-up | 0.6 min |
| `tc_honest.jl` | A λ rule with a coverage guarantee, II: a bias-aware interval over `F(M) = {Bθ : θ′Ωθ ≤ M²}`, with the exact worst-case bias of the linear term and a bound on the quadratic term, a debiased plug-in, and λ chosen by length. The bound against the exact bias, coverage and length per λ and at the chosen λ for `M` = 0, M*/2, M*, 2M*, 4M* (0 and M*/2 are the power checks). tc_lambda's design, R = 500 at a = 1 and 4. The first block is a warm-up | 0.4 min |
| `tc_rchoice.jl` | Choosing ℛ for Proposition 7: ℛ* against the curvature formula, the largest nearly invisible roughness, the debiased and shrunk estimates of ψ′Πψ and a lower bound, coverage and length across ℛ/ℛ* (0–16) and for plug-in rules, and the breakdown value of excluding ρ = 0.2. tc_honest's design, R = 500 at a = 1 and 4. The first block is a warm-up | 0.4 min |
| `tc_refbasis.jl` | Lemma 5 and Proposition 7′: the bias-aware interval with the class on a finer reference basis (k̃ = 30), for f ∉ span(B). The split as an identity, the oracle bound for every candidate (REsd and 12 λ), its convergence in k̃ (15–40), and coverage and length at λ̂ for the released rule against the new one, at k = 4, 5, 7, 15, a = 1 and 4, ℛ × 1 and 4. tc_honest's design, R = 300. The first block is a warm-up | 9.0 min |
| `tc_resd_approx.jl` | Proposition 1′: REsd when `f ∉ span(B)`. Part 1: the gap `ρ̂ − ρ̂*` against the infeasible root with `C(ρ₀)`, the bias against `β*/J*`, and coverage against `P(|Z + c*| ≤ 1.96)`, with `K` growing (arm A) and fixed at k = 4 (arm B, the power check), n = 400 to 2500, R = 500. Part 2: the package's REsd with a reference basis, coverage and length against the Wald interval, n = 900, R = 300 | 36 min |
| `tc_trust.jl` | Lemma 6 and Proposition 8: the joint (trust-region) bound of Lemma 5's linear and quadratic terms. (0) the solver against brute force on 200 random problems and a built hard case; (1) the oracle split of `b̃ − \|β̃\|` into bound and class, per candidate; (2) coverage and half-length at λ̂ of `b̃`, the raw `b̂_TR` and `b̂_min`, at k = 5 and 15, a = 1 and 4, ℛ̃ × 1 and 4. tc_refbasis's design, R = 300. The first block is a warm-up | 6.6 min |
| `tc_lambda.jl` | Choosing λ: GCV's choice and its coverage, a plug-in `ĉ_β(λ)` from an undersmoothed pilot against the exact `c_β`, and RMSE and coverage on a half-decade λ grid. R = 500 at a = 1 and 4. The first block is a warm-up | 1.0 min |
| `tc_coverage_robust.jl` | Step 2 for other `W` (kNN-6, kNN-20, rook, distance band), n = 400, 900, 1600 and K/n up to 0.30: oracle, HC2 and homoskedastic coverage at the feasible root, and the coverage and length of Proposition 6's `CS`, R = 1000, 96 designs (6 skipped for leverage > 0.99). Normal errors by default (`tc_coverage_robust_output.txt`); argument `chisq` gives standardised `χ²₃` errors (`tc_coverage_robust_chisq_output.txt`); argument `time` times the largest cells. The first output line is a warm-up | 27 min per law |
| `tc_ar.jl` | Proposition 6: the test-inversion set `CS = {ρ : g(ρ)² ≤ 1.96² V̂(ρ)}`, with `V̂(ρ)` an exact quartic, against the Wald interval, in the seven weakly identified cells of `tc_coverage_robust.jl` and six controls, normal and `χ²₃` errors, R = 1000; the quartic is checked against the direct `V̂`. The first output line is a warm-up | 3.7 min |
| `tc_a5.jl` | Proposition 5: the lower bounds on `‖Cs‖_F²/n`, (a) from `W`, K/n and `d̄` and (b) from the spectral overlap `π̄_w`, against the exact value, for the Chebyshev and hat bases; no simulation | not timed (short) |
| `tc_schatten.jl` | Theorem 1 without the log factor: the triangular part's operator-norm and Schatten-4 ratios for the moment's matrices (n = 400 to 2500, K ≈ 2√n) and, as the power check, an interleaved Hilbert matrix whose operator-norm ratio grows like log n. See `THEORY.md` | 35 s |
| `tc_vhat.jl` | Theory step 2: the bias of the robust variance at the true `ρ`, K/n from 0.05 to 0.39, constant, smooth and rough variances, normal and χ²₃ errors, R = 1,000. Plug-ins: oracle, HC2, Cattaneo–Jansson–Newey and a Gaussian-corrected version of it, with the homoskedastic one as the power check. The first output line is a 5-replication warm-up; ignore it. See `THEORY.md` | — |
| `tc_commute.jl` | The two zero-diagonal corrections on a torus: equal with a Fourier basis (`MW = WM`), different with a Chebyshev basis | 10 s |
| `tc_misalign.jl` | The delta-method variance of `ρ̂` at `ρ₀ = 0`, split into quadratic, linear and derivative parts: checked against simulation, along a basis-misalignment path, in the concurvity design, and with `W′` and a symmetrised `W` | 1 min |
| `tc_concurvity.jl` | `x2 = (1−ω)·smooth(u,v) + ω·noise` with free smooths of `(u, v)` and `x2`, K = 100, 500 replications per cell, against an oracle | 24 s |

`tc_first_look*.jl` are self-contained and predate `tc_lib.jl`. Their
per-unit correction is the approximation that the exact diagonal later
replaced.

## Results (n = 900, 6-NN `W`, Chebyshev bases)

**Bias of `ρ̂`, the first look:**

| K | Plain CFE | Pooled `τ`, constant variance | With varying variances: pooled `τ` | Per-unit approximation |
|---|---|---|---|---|
| 29 | −0.07 | 0.000 | — | — |
| 92 | −0.19 to −0.22 | −0.002 | −0.05 to −0.06 | −0.006 to −0.007 |
| 191 | −0.47 to −0.53 | 0.000 | −0.08 | −0.011 to −0.012 |

**Null-distribution check (`tc_null.jl`), all 8 cells:**
- the moment's mean has |z| ≤ 1.5;
- its simulated variance against the exact formula: ratio 0.94–0.98, |z| ≤ 1.9;
- the robust variance estimator against the exact formula: ratio 0.998–1.026.
  The remaining bias is conservative and comes from the quadratic plug-in;
  z reaches 7 in the constant-variance χ²₃ cell;
- **power check:** under varying variances the homoskedastic formula has a
  ratio of 0.64 (z = −160);
- coverage of the trace-corrected estimator is 0.948–0.958, and of the plain
  CFE 0.01–0.05.

Before the noise term was subtracted, the robust variance ran 3–10% high.
`tc_decomp.jl` traces the excess to the linear part: `â` carries
`MWP_B ε`, and the analytic prediction of that bias (1.661) matches the
simulation (1.668).

**Concurvity (`tc_concurvity.jl`), 16 cells:**
- the exact-diagonal correction's bias is ≤ 0.004 in every cell, including
  `ω = 0`, where `x2` is purely a function of location;
- its coverage is 0.936–0.950, with the standard errors 2–4% below the
  simulated sd, within Monte Carlo error at R = 500;
- the plain CFE has bias −0.16 to −0.34;
- the pooled `τ` leaves about −0.03 under varying variances;
- the sd is about 1.4–1.5 times the oracle's.

In this design, removing the trace term leaves no separate "population
channel".

**Against Jin & Lee (`tc_jinlee.jl`), 20 cells, corrected 2026-11-19:**
- all three estimators are unbiased (|bias| ≤ 0.008) and cover at
  0.936–0.952;
- ours against their first step (JL-W, `A = W′`): variance ratio 0.99–1.01
  at K = 100, and **1.11–1.19 at K = 199**, so ours is worse with a large basis;
- their two-step REsd (JL-G, `A = G(ρ̂)′`) has the smallest variance: 0.87–0.99
  of JL-W's, with the largest gains at `ρ₀ = 0.4`.

The first run of this script used `A = W` and `A = G(ρ̂)`: the primes were
lost in the `pdftotext` extraction. It reported a 25–40% gain for ours, which
was the cost of putting the instrument on the wrong side of an asymmetric `W`.
The corrected run replaces that output.

**Where the variance comes from (`tc_misalign.jl`, `tc_commute.jl`):**
- on a torus with a Fourier basis, `MW = WM`, and the two corrections are
  the same matrix (difference 1e-14);
- the delta-method variance `[2σ⁴tr(Cs²) + 4σ²‖Cs f‖²]/J²` matches the
  simulated sd to three digits, and in the concurvity design it predicts the
  first run's sd ratios (0.849 and 0.783, against simulated 0.84 and 0.76);
- rotating the basis away from the Fourier modes changes nothing: the ratios
  stay within 0.5% of 1;
- the first run's gap is all in the linear part. Ours uses the instrument
  `MWf`, and Jin & Lee with `A = W` use `M(W′ − Δ)f`; their cosine is 0.53
  under kNN `W`. With `A = W′`, or with a symmetrised `W`, the variance
  ratio is 0.998.

## Not settled

- A penalised smoother (`M = I − S_λ`, not idempotent) instead of a
  projection, and Geniaux's own E3 design.
- Large n: `A` is a dense n × n matrix. At n = 900 that is cheap; at Geniaux's
  n = 25,357 it would need iterative solves.
- Theory: the CLT for the moment is proved (`THEORY.md`, Theorem 1), but the
  estimator's corollary (the LLN for `g′` and the choice of root), a
  primitive form of A5 and a proof of step 2 for K/n → 0 are still owed.
- The robust variance under rough heteroskedasticity: conservative by up to
  13–17% at K/n ≈ 0.3–0.4 (`tc_vhat.jl`). Cattaneo, Jansson & Newey's
  unbiased plug-in needs leverage below 1/2 and fails here; a leave-out
  estimator (Kline, Saggio & Sølvsten 2020) is untried.

## References

- Geniaux, G. (2026). arXiv:2611.16592v1, preprint.
- Lee, L.-F. (2007). GMM and 2SLS estimation of mixed regressive, spatial
  autoregressive models. *Journal of Econometrics*, 137(2), 489–514.
  doi:10.1016/j.jeconom.2005.10.004
- Jin, F., & Lee, L.-F. (2012). Approximated likelihood and root estimators
  for spatial interaction in spatial autoregressive models. *Regional Science
  and Urban Economics*, 42(3), 446–458. doi:10.1016/j.regsciurbeco.2011.12.004
- Liu, S. F., & Yang, Z. (2015). Modified QML estimation of spatial
  autoregressive models with unknown heteroskedasticity and nonnormality.
  *Regional Science and Urban Economics*. doi:10.1016/j.regsciurbeco.2015.02.003
- Su, L. (2012). Semiparametric GMM estimation of spatial autoregressive
  models. *Journal of Econometrics*, 167(2), 543–560.
  doi:10.1016/j.jeconom.2011.09.034
- Cattaneo, M. D., Jansson, M., & Newey, W. K. (2018). Inference in linear
  regression models with many covariates and heteroscedasticity. *JASA*,
  113(523), 1350–1361. doi:10.1080/01621459.2017.1328360
- Kline, P., Saggio, R., & Sølvsten, M. (2020). Leave-out estimation of
  variance components. *Econometrica*, 88(5), 1859–1898.
  doi:10.3982/ECTA16410
- Lee, J. (2026). arXiv:2608.22706v1, preprint (buffered spatial
  cross-fitting, §2.8 and §3).
