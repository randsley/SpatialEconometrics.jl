# Validation studies

Monte Carlo studies that are too slow for the test suite. Each script
measures an estimator's bias, spread and interval coverage, usually against
the paper it implements. Each one's output is committed beside it, so a
figure quoted in the docs can be traced to the run that produced it.
The last column says where each study's findings are written up: a section
of `docs/AUDIT.md`, or the README's Validation Status table.

| Script | What it measures | Output | Recorded in |
|---|---|---|---|
| `mc_sdm.jl` | Cross-section SDM: the Chebyshev log-determinant's error against an exact LU, then bias, sd and coverage of `ρ`, `β` and the direct, indirect and total effects, fitted with the exact LU and with Chebyshev order 6 | `mc_sdm_output.txt` (UTF-16, saved from a PowerShell redirect) | README, Validation Status |
| `mc_semiparametric.jl` | #13: coverage of the partially linear SAR (QML and closed-form `ρ`); Su & Jin's `Ω` under skewed errors; Geniaux's geoadditive trap with the overlap `π̄` | `mc_semiparametric_output.txt` | AUDIT, "#13 pass 1" |
| `mc_tv_coef.jl` | #13: coverage of the pointwise band for `β(τ)`, at the ends of the sample and inside it | `mc_tv_coef_output.txt` | AUDIT, "#13 pass 1" |
| `mc_cre_sdpd.jl` | #1: Li & Yang's Table 1 (CRE dynamic panel, `T = 3`): CQMLE against the M-estimator, and sandwich coverage | `mc_cre_sdpd_output.txt` | AUDIT, "Li & Yang's M-estimator" |
| `mc_serl_sdpd.jl` | #1: Bai & Chen's long-running design to `N = 4000`, at SERL order `L = 0` and `L = 1` | `mc_serl_sdpd_output.txt` | AUDIT, "Bai & Chen's common-shock panel" |
| `mc_serl_sdpd_diagnostics.jl` | Why coverage of `β` falls with `N` there: the baseline design, `L = 2`, and the choice of `Q` | `mc_serl_sdpd_diagnostics_output.txt` | same section |
| `mc_spatial_log_garch.jl` | #16: Sato & Matsuda's Table 1 with coverage (A), the intercept-only case (B), a proxy `W` (C) | `mc_spatial_log_garch_output.txt` | AUDIT, "#16" |
| `mc_spatial_log_garch_diagnostics.jl` | Why one cell of A differs from Sato & Matsuda's table: the start, the draw of `z`, a `λ ≥ 0` region | `mc_spatial_log_garch_diagnostics_output.txt` | same section |
| `mc_spatial_did_channels.jl` | #15: the two-channel spatial DiD (Chagas et al. eq. 7) by OLS and by spatial-error ML: the equality test's size and power, coverage of the four effects, the pooled model's bias; a lattice and Chagas et al.'s own design | `mc_spatial_did_channels_output.txt` | AUDIT, finding 34 |
| `mc_spatial_did_channels_diagnostics.jl` | Why the OLS intervals under-cover there: pooled against split, correlated against uncorrelated errors, five `vcov`s including wide Conley cutoffs | `mc_spatial_did_channels_diagnostics_output.txt` | same section |
| `mc_spatial_did_gmm.jl` | The heteroskedasticity-robust spatial error (Kelejian & Prucha 2010) against ML: the DiD on a lattice under three variance patterns, Chagas et al.'s staggered design (the panel extension), and a cross-section SEM with variance tied to the neighbour count | `mc_spatial_did_gmm_output.txt` | AUDIT, finding 35 |
| `mc_gs2sls_het.jl` | The heteroskedasticity-robust GS2SLS (Kelejian & Prucha 2010, endogenous regressors) against the default path: cross-sections of 300 and 1,200 units and six-period panels with unit and two-way effects, error sd constant or proportional to the neighbour count | `mc_gs2sls_het_output.txt` | AUDIT, finding 36 |
| `mc_root_rho.jl` | `root_rho` (Jin & Lee's REsd, projection on a growing basis): coverage of the Wald interval and of the test-inversion set over 108 cells (3 `W`, `n` 400–1600, `K/n` 0.05–0.19, two variance patterns, normal and `χ²₃` errors), agreement with the research code, and the plain closed form's bias on the same draws | `mc_root_rho_output.txt` | AUDIT, "`root_rho`" |
| `mc_root_rho_smoother.jl` | `root_rho` with a penalised smoother (`tensor_pspline`, the exact-diagonal moment): coverage of both intervals over 108 cells (3 `W`, `n` 400–1600, `λ` 0.01–1, two signal strengths, two variance and error laws) against the coverage predicted from the exact shrinkage bias, the shortcut moment on the same draws (power), and agreement with a dense reference | `mc_root_rho_smoother_output.txt` | AUDIT, "`root_rho` with a penalised smoother" |

## Running them

From the repository root, one at a time:

```bash
julia --project=. validation/mc_spatial_log_garch.jl
```

- **Output.** The two `mc_spatial_log_garch*` scripts, the three `mc_spatial_did_*` scripts, `mc_gs2sls_het.jl` and `mc_cre_sdpd.jl`
  write their output file themselves. The others print to the terminal, and
  their committed output is that printout, saved as the file in the table.
- **Arguments.** Each script's header says what it takes:
  - `mc_spatial_log_garch.jl` reads `R_SCALE` from the environment
    (`R_SCALE=0.05` for a pilot at 5% of the replications);
  - `mc_serl_sdpd.jl` and its diagnostics take the replications and the
    design on the command line;
  - the three `mc_spatial_did_*` scripts and `mc_gs2sls_het.jl` read `MC_REPS` from the
    environment (`MC_REPS=50` for a pilot).
- **Run time.** The longest are about two hours on one thread
  (`mc_spatial_log_garch.jl`: 7,069 s), so run them in a terminal you can
  leave.
- **Reproducibility.** Each script seeds its own generators, so a rerun
  with the same Julia version reproduces its output exactly. Julia 1.10 and
  1.13 draw different random streams, so the figures differ between them,
  within Monte Carlo error.

## Adding one

A new estimator gets a study here before its results are quoted. The study:
- states its design in the script's header comment;
- writes or saves its output as `<name>_output.txt`;
- reports failed fits instead of dropping them silently;
- has its findings written up in `docs/AUDIT.md`, including what it does
  not settle.
