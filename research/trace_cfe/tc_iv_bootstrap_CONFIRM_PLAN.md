# Confirmatory run of the spatial-block IV bootstrap: the plan

Written 2026-11-26 and committed **before** the run, so that the commit
order shows that nothing below was chosen after seeing its result.

## Why

The exploratory run (`tc_iv_bootstrap.jl`, output
`tc_iv_bootstrap_output.txt`, commit f2a834d; THEORY.md "A spatial-block
bootstrap for the IV") failed its rule 1 by 0.002: at 40 km blocks, the
WRE bootstrap covered ρ₀ in 0.928 of 500 draws in cell E5, against a bar
of 0.93 (MC se 0.012). Its real-data reading (rule 2) was therefore not
made. This run decides rule 1 with more precision. It is the only
confirmatory run: whatever it shows is the result, and it will not be
repeated with other seeds or sizes.

## What is run

The script `tc_iv_bootstrap.jl` as committed in f2a834d, with one code
edit, committed with this plan: it reads `IVB_SEED` (default 0, which
gives the exploratory run's seeds exactly) and adds it to its three
seeds. The run then differs from the exploratory one in two settings
only:

- `IVB_R=2000`: 2,000 placebo draws per cell (MC se of a coverage near
  0.93: 0.0057), against 500 in the exploratory run.
- `IVB_SEED=20261127`: added to every seed in the script (the placebo
  data, the placebo bootstrap signs and the real-data bootstrap signs), so
  that no draw is reused from the exploratory run.

Unchanged: the cells (B, C, D, E2, E5) and their truths, the fit (`k` = 24,
λ = 0.1 in Part 1; `k` = 24 and 16, λ = 0.1 in Part 2), the instruments
`M[WX, W²X]`, the blocks (40 km primary, 20 km reported), B = 399
bootstrap draws in Part 1 and 999 in Part 2, Rademacher signs, the
cluster-robust `t`, the grid for Part 2 (step 0.005 over [−0.7, 0.9]).

## The rules (unchanged from the exploratory run)

1. The WRE bootstrap at L = 40 km covers ρ₀ in at least **0.93** of draws
   in **each** of B, C, E2 and E5. D is left out, as before: its `ρ̂_IV`
   is biased. L = 20 km and the analytic intervals are reported, with no
   pass mark.
2. Only if rule 1 passes, the real-data reading at L = 40 km, λ = 0.1,
   for each `k` (24 and 16), from this run's Part 2:
   - the set excludes `root_rho`'s `ρ̂` (0.339 at `k` = 24, 0.380 at
     `k` = 16) ⇒ part of that `ρ̂` is not a lag (error correlation, or
     bias of the kind seen in D, about −0.075 there, which goes the other
     way);
   - it includes 0 ⇒ the data are consistent with no lag;
   - it excludes 0 ⇒ evidence of a lag that error correlation does not
     explain.

The bar stays at 0.93, although at R = 2,000 "0.95 less two MC se" would
be 0.94. The rule is not tightened, because that would also be a change
after seeing the exploratory result. Note what this means: with R = 2,000,
a cell whose true coverage is 0.925 passes with probability about 0.2,
and one at 0.935 with about 0.8.

## What is already known, and what is not

Part 2 depends on the data and on the bootstrap seed only. Its exploratory
values are known (THEORY.md): at 40 km, `k` = 24 gave [−0.200, 0.410] and
`k` = 16 gave [−0.330, 0.260]. With a new seed, the set's ends can move by
about one grid step or so. So if rule 1 passes, the reading will be about
what the exploratory run showed. The new information in this run is
rule 1.

## Run

```
IVB_R=2000 IVB_SEED=20261127 julia --project=. research/trace_cfe/tc_iv_bootstrap.jl > research/trace_cfe/tc_iv_bootstrap_confirm_output.txt
```

The result is recorded in THEORY.md under the exploratory section,
pass or fail.
