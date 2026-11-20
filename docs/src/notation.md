# Notation

The manual uses one set of symbols for the package's own text. This page
lists them, then lists each page that departs from them, and why.

!!! note "Quotes use the notation of their source"
    A theorem, formula or sentence quoted from a paper keeps that paper's
    words and symbols exactly, even where a symbol means something else in
    this manual. A line after the quote maps its symbols to the manual's.
    A page that implements one paper's model may also keep that paper's
    symbols throughout; the second table below lists those pages.

## The core symbols

| Symbol | Meaning | Result field or keyword |
|---|---|---|
| `y` | the outcome: an `n`-vector, or `y_t` / `y_{it}` in a panel | |
| `X`, `β` | the regressors and their coefficients | `beta`, `coef` |
| `W` | the spatial weight matrix, `n × n` (`N × N` in a panel), zero diagonal | |
| `ρ` | the coefficient on the spatial lag `Wy` | `rho` |
| `λ` | the coefficient of the spatial error, `u = λWu + ε` | `lambda` |
| `θ` | the coefficients on `WX` (SLX, SDM, SDEM) | |
| `u` | a disturbance before the spatial error filter | |
| `ε`, `σ²` | the innovations and their variance | `sigma2` |
| `Σ` | `Diag(σᵢ²)`, the variances under heteroskedasticity | |
| `γ`, `η` | in a dynamic panel: the coefficients on `y_{t−1}` and `Wy_{t−1}` | `gamma`, `eta` |
| `μᵢ`, `ξ_t` | unit and time effects | |
| `N`, `T` | the number of spatial units and of periods | the argument `N`, `T` |
| `n` | the number of observations: `n = N` in a cross-section | |
| `i, j`; `t` | indices of units; of periods | |
| `I`, `1_n` | the identity; the vector of ones | |
| `k` | the number of columns of `X` | |
| `M` | an annihilator `I − P`: residual-maker of a projection or a centring | |

## Pages with their own symbols

These pages define the symbols below where they use them. The table maps
each one to the core symbols.

| Page | Symbol there | Meaning | Core symbol | Why |
|---|---|---|---|---|
| [Short-T Dynamic Panel (CRE)](models/cre_dynamic_panel.md) | `ρ`; `λ₁`, `λ₂`, `λ₃` | time lag; spatial lag, space-time lag, spatial error | `γ`; `ρ`, `η`, `λ` | Li & Yang (2021); the coefficient names (`"rho"`, `"lambda1"`) follow it |
| | `φ`; `μ`; `α_t`; `n` | variance ratio; CRE effects; time effects; units | —; `μᵢ`; `ξ_t`; `N` | Li & Yang (2021) |
| [Short-T Dynamic Panel (Common Shocks)](models/common_shock_dynamic_panel.md) | `φ`; `δ_t`; `Λ`, `λᵢ`, `f_t` | time lag; time effects; factor loadings and factors | `γ`; `ξ_t`; — | Bai & Chen (2026); the coefficient names (`"phi"`, `"delta_t"`) follow it |
| [Common Factors & Spatio-Temporal](models/common_factors.md) | `λᵢ`, `fₜ`; `α`; `φ` | factor loadings and factors; the exponent of cross-sectional dependence; STAR coefficients | —; —; — | the factor and CD-test literature |
| [Spatial VAR](models/spvar.md) | `C_h`, `Φ`, `Λ = diag(λ)` | structural lag matrices; spatial coefficients; error variances | —; —; `σ²` | Di Giacinto (2010) |
| [Networks](models/networks.md) | `β`; `γ`; `δ`; `G` | endogenous peer effect; own characteristics; contextual effect; the network | `ρ`; `β`; `θ`; `W` | Bramoullé, Djebbari & Fortin (2009); `fit.beta` is the peer effect |
| [Spatial log-GARCH](models/spatial_garch.md) | `λ`; `ρ`; `δ` | coefficient on `W log h`; on `W log y²`; on `Z` | —; —; `β` | Sato & Matsuda (2021); `m.lambda`, `m.rho` follow it |
| [Spatial Filtering](models/filtering.md) | `λᵢ`; `γ`; `τ²`, `θ = τ²/σ²` | eigenvalues; eigenvector coefficients; their variance, its ratio to `σ²` | —; —; — | the ESF literature (Griffith; Murakami & Griffith) |
| [Causal Inference](models/causal.md) | `τ`; `δ`; `D`; `E = WD` | direct effect; spillover effect; treatment; exposure | — | difference-in-differences usage |
| [Semiparametric Models](models/semiparametric.md) | `m(z)`; `S`; `τ = t/T`; `e_t` | the unknown function; the smoother; rescaled time; errors | —; —; —; `ε` | Su & Jin (2010); Liang, Gao & Gong (2022); `tv.tau` |
| [Root Estimator of ρ](models/root_rho.md) | `λ`; `A`, `C`; `S`, `Ω̃` | the smoothing penalty; moment matrices; the smoother and its penalty | —; —; — | the keyword `lambda =`; the derivations are in `research/trace_cfe/THEORY.md` |
| [Discrete Choice](models/discrete_choice.md) | `y*`; `τ_j` | the latent outcome; the ordered-probit thresholds | — | |
| [Gravity Models](models/gravity.md) | `F_{ij}`; `αᵢ`, `γⱼ` | flows; origin and destination effects | —; `μᵢ`-like effects | the gravity literature |

Three symbols mean different things on different pages, by design:

- `λ`: the spatial error coefficient (the core meaning), Li & Yang's lag
  coefficients, factor loadings, eigenvalues in spatial filtering, and the
  smoothing penalty of `root_rho`. Each page defines it where it uses it.
- `G`: `W(I − ρW)⁻¹` in the root estimator, `R(λ)⁻¹W` in the log-GARCH
  scores, and the network itself on the networks page.
- `Ω`: the penalty of a smoother, `MWᵤM` in spatial filtering, and the
  middle of a sandwich variance (`Ω̂`).
