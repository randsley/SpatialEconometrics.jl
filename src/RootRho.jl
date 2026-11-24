"""
    RootRho

Jin & Lee's (2012) root estimator of `ρ` in a spatial autoregression,
with a first stage that projects on a basis whose size may grow with `n`.

```math
y = ρWy + Xβ + f + ε
```

`f` is a smooth nuisance approximated in the span of a basis `B` that the
user supplies (a tensor spline or hat basis on the locations, for example).
The estimator needs no log-determinant: `ρ̂` is a root of a quadratic.

The trace correction is what makes this work when `B` is large. With the
projection `M = I − P_[X,B]`, Smirnov's plain closed form has a moment whose
mean at the true `ρ` is of order `K/n` (`K` the number of columns of
`[X, B]`), so it is biased, and badly so once a free surface makes `K`
large; [`closed_form_rho`](@ref SpatialEconometrics.SemiparametricSpatial.closed_form_rho) is that estimator. Jin & Lee's moment
`y₀′Cy₀` uses a `C` with a zero diagonal and `C[X, B] = 0`, so its mean is
exactly zero whatever the heteroskedasticity.

# Inference

Two intervals are reported for `ρ`:
- the **Wald** interval `ρ̂ ± z√V̂/|g′(ρ̂)|`, with the HC2 variance `V̂`
  that includes a noise term for the estimated `P_[X,B]y₀`;
- the **test-inversion set** `CS = {ρ : g(ρ)² ≤ z²V̂(ρ)}`, an
  Anderson–Rubin-type set. It does not need `ρ` to be identified: where the
  data say little about `ρ`, it is wide, and it may reach the bounds, while
  the Wald interval under-covers. Report it.

# With a penalised smoother

Given a `penalty` and `lambda`, the first stage is the penalised smoother
`S = Z(Z′Z + λΩ)⁻¹Z′` (only the basis is penalised), and `M = I − S` is no
longer idempotent. Then the moment is the one-step exact-diagonal moment
`y₀′Ay₀` with `A = MW + MDM`, `(M∘M)d = −diag(MW)`, so `diag A = 0`, and the
HC2 plug-in is `ûᵢ²/(M²)ᵢᵢ` (THEORY.md, Proposition 4). Jin & Lee's two-step
is not used there: its plug-in result needs `Mf = 0`, which a smoother
breaks by shrinking `f`. The smoother's shrinkage biases `ρ̂` and both
intervals, so smooth **less** than a good fit of `f` would.
[`tensor_pspline`](@ref) builds a tensor P-spline basis and penalty.

The growing-basis results (a CLT for the moment with a dense `M`, the
consistency of the HC2 `V̂`, the validity of `CS`, and the penalised
smoother, and its bias-aware interval) are proved in
`research/trace_cfe/THEORY.md` (Theorem 1, Propositions 1, 1′, 2, 4, 6,
6″, 7 and 7′, and Lemmas 4, 5 and 7–10). They have not been checked by
anyone else. They are measured in that directory and in
`validation/mc_root_rho.jl`, `validation/mc_root_rho_smoother.jl` and
`validation/mc_root_rho_hc2_worst.jl`.

# References

- Jin, F., & Lee, L.-F. (2012). Approximated likelihood and root estimators
  for spatial interaction in spatial autoregressive models. *Regional
  Science and Urban Economics*, 42(3), 446–458.
- Smirnov, O. A. (2022). A closed-form consistent estimator for linear
  models with spatial dependence. *Geographical Analysis*, 54(1), 3–31.
  https://doi.org/10.1111/gean.12268
- Anderson, T. W., & Rubin, H. (1949). Estimation of the parameters of a
  single equation in a complete system of stochastic equations. *Annals of
  Mathematical Statistics*, 20(1), 46–63.
"""
module RootRho

using LinearAlgebra
using SparseArrays
using Printf

import ..SpatialUtils: check_design_rank, _check_spatial_bounds, ExactLU, _normcdf
import ..SpatialDiscreteChoice: _norminv

export RootRhoResult, root_rho, tensor_pspline, basis_roughness

"""
    RootRhoResult

The result of [`root_rho`](@ref).

# Fields

- `rho` — the root `ρ̂`: REsd's with a projection, the exact-diagonal
  moment's with a penalised smoother (`NaN` when no admissible root exists;
  see `root_found`).
- `rho_first` — Jin & Lee's first-step root `ρ̃` (`NaN` if none, and always
  `NaN` with a penalised smoother, which takes one step).
- `se` — `√V̂(ρ̂)/|g′(ρ̂)|`, with the HC2 `V̂`.
- `ci_wald` — `ρ̂ ± z·se`.
- `cs` — the test-inversion set, as a vector of closed intervals within
  `rho_bounds`. Usually one interval; empty if the moment rejects every
  `ρ` in the bounds.
- `cs_touches_bound`, `cs_split` — whether `cs` reaches a bound, and whether
  it is more than one interval.
- `level` — the confidence level of both.
- `beta` — the coefficients of `X` from least squares of `y − ρ̂Wy` on
  `[X, B]`. **No standard errors**: the theory covers `ρ` only.
- `varnames` — the names of `beta`.
- `n`, `K` — the sample size and the number of columns of `[X, B]`.
- `smoother` — `:projection` or `:penalised`.
- `lambda` — the penalty's `λ` (`0` for a projection).
- `df` — the effective degrees of freedom `tr S` of the first stage (`K`
  for a projection).
- `max_leverage` — `maxᵢ Sᵢᵢ` of the first stage (`hᵢᵢ` for a projection).
- `a`, `b`, `c` — the final moment `g(ρ) = c − bρ + aρ²`.
- `vpoly` — `V̂(ρ) = Σₖ vpoly[k+1] ρᵏ`, a quartic.
- `vhat` — `V̂(ρ̂)`.
- `root_found` — `false` when the moment has no admissible root at either
  step. Then `rho`, `se` and `ci_wald` are `NaN`, and `cs` is still valid.
- `rho_bounds`.
- `vhat_worst_bias` — for a projection, the worst case of the HC2 `V̂`'s
  relative bias `(E V̂ − V)/V` at `ρ₀`, over symmetric homoskedastic error
  laws (THEORY.md, Lemma 8). It is computed from `W`, `[X, B]` and the
  final moment matrix alone, and is reached by Rademacher (`±σ`) errors at
  `f = 0`. The test-inversion set then covers at least
  `2Φ(z√(1 + vhat_worst_bias)) − 1` for large `n` (Proposition 6″). It is
  `0` when the bias cannot be negative, `-Inf` when it has no worst case,
  and `NaN` for a penalised smoother, where Lemma 8 does not apply. The
  HC2 `V̂` is not consistent when `K/n` does not tend to 0, and this is the
  price.
- `roughness` — the bound `ℛ` given for the bias-aware interval (`NaN`
  when none was asked for).
- `bias_bound` — `b̄`, the bound on the moment's bias `|E g(ρ₀)|` over the
  roughness class (0 for a projection; `NaN` when not asked for).
- `ci_bias_aware` — `ρ̂ ± cv(b̄/√V̂)·se`, with `cv(t)` the `level` quantile of
  `|N(t, 1)|` (`NaN` when not asked for).
- `lambda_grid`, `halflength_grid` — with a vector of `lambda`s, every
  candidate (the projection as `λ = 0` first) and the half-length of its
  bias-aware interval; the result is the shortest. Empty otherwise.
- `reference_K` — the number of columns of `[X, reference basis]` when the
  roughness class is stated on a `reference` basis (Lemma 5); 0 otherwise.
- `basis_grid`, `basis_index` — with a vector of bases, the basis of each
  candidate in `lambda_grid` (an index into the vector), and the index of
  the chosen one. Empty and 0 otherwise.
"""
struct RootRhoResult{T<:AbstractFloat}
    rho::T
    rho_first::T
    se::T
    ci_wald::Tuple{T, T}
    cs::Vector{Tuple{T, T}}
    cs_touches_bound::Bool
    cs_split::Bool
    level::T
    beta::Vector{T}
    varnames::Vector{String}
    n::Int
    K::Int
    smoother::Symbol
    lambda::T
    df::T
    max_leverage::T
    a::T
    b::T
    c::T
    vpoly::NTuple{5, T}
    vhat::T
    root_found::Bool
    rho_bounds::Tuple{T, T}
    vhat_worst_bias::T
    roughness::T
    bias_bound::T
    ci_bias_aware::Tuple{T, T}
    lambda_grid::Vector{T}
    halflength_grid::Vector{T}
    reference_K::Int
    basis_grid::Vector{Int}
    basis_index::Int
end

# The fields of a result without the bias-aware interval.
_no_bias(::Type{T}) where {T} = (T(NaN), T(NaN), (T(NaN), T(NaN)), T[], T[], 0, Int[], 0)

# A copy of r with the bias-aware fields replaced.
function _with_bias(r::RootRhoResult{T}, ℛ, bbar, ci, grid, hl;
                    refK::Int = r.reference_K, bgrid::Vector{Int} = r.basis_grid,
                    bidx::Int = r.basis_index) where {T}
    f = [getfield(r, k) for k in fieldnames(RootRhoResult)]
    f[end-7:end] = Any[T(ℛ), T(bbar), (T(ci[1]), T(ci[2])), Vector{T}(grid), Vector{T}(hl), refK,
                       bgrid, bidx]
    return RootRhoResult{T}(f...)
end

function Base.show(io::IO, r::RootRhoResult)
    if r.smoother === :penalised
        println(io, "Root estimator of ρ (exact-diagonal moment), penalised smoother on [X, B]")
        println(io, "──────────────────────────────────────────────────────────────────────────")
        @printf(io, "n = %d, K = %d, λ = %.4g, df = tr S = %.1f (df/n = %.3f), max Sᵢᵢ %.3f\n",
                r.n, r.K, r.lambda, r.df, r.df / r.n, r.max_leverage)
    else
        println(io, "Root estimator of ρ (Jin & Lee 2012, REsd), projection on [X, B]")
        println(io, "────────────────────────────────────────────────────────────────")
        @printf(io, "n = %d, K = %d (K/n = %.3f), max leverage %.3f\n", r.n, r.K, r.K / r.n,
                r.max_leverage)
    end
    pct = round(100 * r.level; digits = 1)
    if r.root_found && r.smoother === :penalised
        @printf(io, "ρ̂ = %.4f   SE %.4f\n", r.rho, r.se)
        @printf(io, "Wald %g%% interval:            [%.4f, %.4f]\n", pct, r.ci_wald...)
    elseif r.root_found
        @printf(io, "ρ̂ = %.4f   SE %.4f   (first step ρ̃ = %.4f)\n", r.rho, r.se, r.rho_first)
        @printf(io, "Wald %g%% interval:            [%.4f, %.4f]\n", pct, r.ci_wald...)
    else
        println(io, "No admissible root: the data do not identify ρ through this moment.")
        println(io, "The test-inversion set below is still valid.")
    end
    if !isnan(r.roughness) && r.root_found
        @printf(io, "Bias-aware %g%% interval:      [%.4f, %.4f]   (ℛ = %.4g, b̄/√V̂ = %.3f)\n",
                pct, r.ci_bias_aware..., r.roughness, r.bias_bound / sqrt(r.vhat))
        r.reference_K > 0 && @printf(io,
            "  ℛ on a reference basis with K̃ = %d columns (Lemma 5)\n", r.reference_K)
        if r.basis_index > 0
            @printf(io, "  basis %d of %d (K = %d) and λ chosen by the interval's length from %d candidates\n",
                    r.basis_index, maximum(r.basis_grid), r.K, length(r.lambda_grid))
        elseif !isempty(r.lambda_grid)
            @printf(io, "  λ chosen by the interval's length from %d candidates (λ = 0 is the projection)\n",
                    length(r.lambda_grid))
        end
    end
    if isempty(r.cs)
        @printf(io, "Test-inversion %g%% set: empty (every ρ in [%.2f, %.2f] is rejected)\n",
                pct, r.rho_bounds...)
    else
        s = join((@sprintf("[%.4f, %.4f]", lo, hi) for (lo, hi) in r.cs), " ∪ ")
        @printf(io, "Test-inversion %g%% set:       %s\n", pct, s)
        r.cs_touches_bound && println(io,
            "  The set reaches a bound of rho_bounds: ρ is weakly identified.")
    end
    if r.vhat_worst_bias == -Inf
        println(io, "  HC2 V̂: no worst case over symmetric error laws (its bias falls as the kurtosis grows)")
    elseif !isnan(r.vhat_worst_bias)
        @printf(io, "  HC2 V̂: worst-case bias %.1f%% (symmetric homoskedastic errors), so the set covers ≥ %.1f%% for large n\n",
                100 * r.vhat_worst_bias, 100 * _coverage_given_bias(r.vhat_worst_bias, r.level))
    end
    if !isempty(r.beta)
        println(io, "β (point estimates only; no standard errors):")
        for (nm, b) in zip(r.varnames, r.beta)
            @printf(io, "  %-12s %10.4f\n", nm, b)
        end
    end
end

# C = (A − Δ)M with Δ = Diag(AM)Diag(M)⁻¹, for a dense A and M = I − QQ′.
# diag C = 0 and CQ = 0 exactly.
function _moment_matrix(A::Matrix{T}, Q::Matrix{T}, Mii::Vector{T}) where {T}
    AM = A .- (A * Q) * Q'
    δ = diag(AM) ./ Mii
    C = AM
    @inbounds for i in eachindex(δ)
        C[i, i] -= δ[i]
    end
    C .+= (δ .* Q) * Q'
    return C
end

# The moment g(ρ) = y₀(ρ)′C y₀(ρ) = c − bρ + aρ², and the root at which
# g′ < 0 (the sign rule), or `nothing` if it is not real or not in bounds.
function _quadratic(C::Matrix{T}, y::Vector{T}, Wy::Vector{T}) where {T}
    Cy = C * y; CWy = C * Wy
    return dot(Wy, CWy), dot(y, CWy) + dot(Wy, Cy), dot(y, Cy)
end

function _sign_rule_root(a::T, b::T, c::T, bounds) where {T}
    if iszero(a)
        b > 0 || return nothing
        ρ = c / b
    else
        D = b^2 - 4a * c
        D > 0 || return nothing
        ρ = (b - sqrt(D)) / (2a)     # g′(ρ) = −√D < 0 at this root
    end
    return bounds[1] < ρ < bounds[2] ? ρ : nothing
end

# A preliminary 2SLS estimate of ρ, with instruments [Z, WZ], used only to read
# the sign of J = −E g′(ρ₀), which the sign rule assumes positive (THEORY.md,
# A9). `nothing` when WZ adds nothing to span Z, or when the instruments span
# every direction (2SLS would then be OLS, which is not consistent).
function _prelim_2sls(y::Vector{T}, Wy::Vector{T}, Z::Matrix{T},
                      Ws::SparseMatrixCSC{T}) where {T}
    n, K = size(Z)
    F = qr(hcat(Z, Ws * Z), ColumnNorm())
    d = abs.(diag(F.R))
    r = count(>(1e-10 * d[1]), d)
    K < r < n || return nothing
    Qh = Matrix(F.Q)[:, 1:r]
    Xh = hcat(Wy, Z)
    ρ = ((Qh * (Qh' * Xh)) \ y)[1]
    return isfinite(ρ) ? ρ : nothing
end

# Warn when g′ = −b + 2aρ, for the moment that gave ρ̂, is positive at the
# preliminary estimate: the sign rule then picks the wrong root if J < 0. As g′
# is linear, this is the same as the preliminary estimate lying nearer the
# other root. Only the final moment matters: REsd's first step only chooses
# C(ρ̃), and the moment at ρ₀ is valid for any choice.
function _check_slope_sign(ρp, a, b)
    ρp === nothing && return false
    bad = -b + 2a * ρp > 0
    bad && @warn "root_rho: the moment's slope g′ is positive at a preliminary 2SLS " *
        "estimate (ρ = $(round(ρp; digits = 3))). The root rule takes the root where " *
        "g′ < 0, which assumes J = −E g′(ρ₀) > 0, so ρ̂ and its Wald interval may be the " *
        "wrong root. The test-inversion set does not use the root rule and stays valid."
    return bad
end

# V̂(ρ) for the moment with C, as the coefficients of a quartic in ρ. The HC2
# plug-in sᵢ(ρ) = ûᵢ(ρ)²/Mᵢᵢ is quadratic in ρ, â(ρ) = (C + C′)P y₀(ρ) is
# linear, and the noise term diag(H(Q′diag(s)Q)H′), H = (C + C′)Q, is linear
# in s:  V̂ = 2s′(Cs∘Cs)s + Σᵢ (âᵢ² − noiseᵢ) sᵢ.
function _vhat_poly(C::Matrix{T}, Q::Matrix{T}, Mii::Vector{T}, y::Vector{T},
                    Wy::Vector{T}) where {T}
    Csum = C .+ C'
    Ω = (Csum ./ 2) .^ 2
    H = Csum * Q
    e0 = y .- Q * (Q' * y); e1 = Wy .- Q * (Q' * Wy)
    S = (e0 .^ 2 ./ Mii, -2 .* e0 .* e1 ./ Mii, e1 .^ 2 ./ Mii)
    α0 = Csum * (y .- e0); α1 = -(Csum * (Wy .- e1))
    noise(s) = vec(sum((H * (Q' * (s .* Q))) .* H; dims = 2))
    N = map(noise, S)
    ΩS = map(s -> Ω * s, S)
    Acoef = (α0 .^ 2 .- N[1], 2 .* α0 .* α1 .- N[2], α1 .^ 2 .- N[3])
    p = zeros(T, 5)
    for k in 0:2, l in 0:2
        p[k + l + 1] += 2 * dot(S[k + 1], ΩS[l + 1]) + dot(Acoef[k + 1], S[l + 1])
    end
    return (p[1], p[2], p[3], p[4], p[5])
end

# The same quartic for a general symmetric first stage S, M = I − S (not
# idempotent): sᵢ(ρ) = ûᵢ(ρ)²/mᵢ with û = My₀ and mᵢ = (M²)ᵢᵢ; â(ρ) = Hy₀(ρ)
# with H = (A + A′)S; the noise term diag(H diag(s) H′) = (H∘H)s
# (THEORY.md, Proposition 4).
function _vhat_poly_general(A::Matrix{T}, H::Matrix{T}, M::Matrix{T}, m::Vector{T},
                            y::Vector{T}, Wy::Vector{T}) where {T}
    Ω = ((A .+ A') ./ 2) .^ 2
    H2 = H .^ 2
    e0 = M * y; e1 = M * Wy
    S = (e0 .^ 2 ./ m, -2 .* e0 .* e1 ./ m, e1 .^ 2 ./ m)
    α0 = H * y; α1 = -(H * Wy)
    N = map(s -> H2 * s, S)
    ΩS = map(s -> Ω * s, S)
    Acoef = (α0 .^ 2 .- N[1], 2 .* α0 .* α1 .- N[2], α1 .^ 2 .- N[3])
    p = zeros(T, 5)
    for k in 0:2, l in 0:2
        p[k + l + 1] += 2 * dot(S[k + 1], ΩS[l + 1]) + dot(Acoef[k + 1], S[l + 1])
    end
    return (p[1], p[2], p[3], p[4], p[5])
end

# Lemma 8 (THEORY.md): the worst case of the HC2 V̂'s relative bias at ρ₀ over
# symmetric homoskedastic error laws, β_min = min(0, Q₂ + L₅ᴳ − 2Σₖcₖ)/(2Σᵢⱼ Ωᵢⱼ),
# reached by Rademacher (±σ) errors at f = 0. With uₖ = (Mᵢₖ²/Mᵢᵢ)ᵢ (the
# columns of U) and H = (C + C′)P:
#   Q₂ = 4Σᵢⱼ ΩᵢⱼMᵢⱼ²/(MᵢᵢMⱼⱼ),  L₅ᴳ = −2Σᵢⱼ Hᵢⱼ²Mᵢⱼ²/(MᵢᵢMⱼⱼ),
#   Σₖcₖ = 2Σₖuₖ′Ωuₖ + Σᵢₖ Hᵢₖ²Mᵢₖ²/Mᵢᵢ − Σₖuₖ′(H∘H)uₖ.
# −Inf when Σₖcₖ < 0: the bias then falls without limit as the kurtosis grows.
# Two n × n × n products and four n × n matrices.
function _hc2_worst_bias(C::Matrix{T}, Q::Matrix{T}, Mii::Vector{T}) where {T}
    n = size(C, 1)
    Ω = C .+ C'
    H = (Ω * Q) * Q'
    Ω .= (Ω ./ 2) .^ 2
    H .= H .^ 2
    U = Q * Q'
    @inbounds for j in 1:n, i in 1:n
        m = (i == j ? one(T) : zero(T)) - U[i, j]
        U[i, j] = m^2 / Mii[i]
    end
    q2 = zero(T); l5 = zero(T); t3 = zero(T)
    @inbounds for j in 1:n, i in 1:n
        q2 += Ω[i, j] * U[i, j] / Mii[j]
        l5 += H[i, j] * U[i, j] / Mii[j]
        t3 += H[i, j] * U[i, j]
    end
    B = Ω * U
    c1 = dot(U, B)
    mul!(B, H, U)
    sumc = 2c1 + t3 - dot(U, B)
    sumc < 0 && return T(-Inf)
    return min(zero(T), 4q2 - 2l5 - 2sumc) / (2 * sum(Ω))
end

# The coverage that a relative bias β of V̂ implies for the test-inversion set
# at level `level`: 2Φ(z√(1 + β)) − 1.
_coverage_given_bias(β::Real, level::Real) =
    β <= -1 ? zero(β) : 2 * _normcdf(_norminv(1 - (1 - level) / 2) * sqrt(1 + β)) - 1

# The exact-diagonal moment matrix A = MW + MDM, (M∘M)d = −diag(MW), for a
# symmetric M. diag A = 0 exactly: diag(MDM) = (M∘M)d needs only that M is
# symmetric. The moment must be formed from A itself: the residual-vector
# shortcut (My₀)′(MWy₀ + DMy₀) is y₀′(M²W + MDM)y₀, which differs when M² ≠ M.
_exact_diag_moment(M::Matrix{T}, W::AbstractMatrix{T}) where {T} = _exact_diag_moment_d(M, W)[1]
function _exact_diag_moment_d(M::Matrix{T}, W::AbstractMatrix{T}) where {T}
    MW = M * W
    d = (M .* M) \ (-diag(MW))
    MW .+= M * (d .* M)
    return MW, d
end

# ── The bias-aware interval (THEORY.md, Lemma 4 and Proposition 7) ──────────
#
# Over F(ℛ) = {f = Zψ : ψ′Πψ ≤ ℛ²}, Π the penalty on the basis block of Z, the
# smoother's bias in the moment, β = f′Af, satisfies |β| ≤ b̄ = ℛ√(v′Π⁺v) + ℛ²χ̄,
# with v = (MZ)′Wf and χ̄ the largest |eigenvalue| of Π^{+1/2}(MZ)′D(MZ)Π^{+1/2}.
# v is estimated from the projection: v̂ = (MZ)′W P_Z y₀(ρ̂₀), with ρ̂₀ Jin &
# Lee's first-step root, and v̂′Π⁺v̂ is debiased by Σᵢ ŝ₀ᵢ‖Υeᵢ‖²,
# Υ = Π^{+1/2}(MZ)′W P_Z, ŝ₀ the projection's HC2 plug-in.

# cv(t): the 1 − α quantile of |N(t, 1)|, by bisection.
function _cv_bias(t::Real, α::Real)
    t = abs(Float64(t)); lo, hi = 0.0, t + 10.0
    for _ in 1:100
        c = (lo + hi) / 2
        _normcdf(c - t) - _normcdf(-c - t) < 1 - α ? (lo = c) : (hi = c)
    end
    return (lo + hi) / 2
end

# The projection pilot: its basis Q, Q′y₀(ρ̂₀) and the HC2 plug-in ŝ₀.
function _bias_pilot(y::Vector{T}, Wy::Vector{T}, Z::Matrix{T}, Ws::SparseMatrixCSC{T},
                     bnds) where {T}
    K = size(Z, 2)
    Q = Matrix(qr(Z).Q)[:, 1:K]
    Mii = 1 .- vec(sum(abs2, Q; dims = 2))
    nbad = count(<(0.01), Mii)
    nbad > 0 && throw(ArgumentError(
        "root_rho: the bias-aware interval uses the projection on [X, basis] as a pilot, " *
        "and $nbad unit(s) have leverage above 0.99 there; use fewer basis functions " *
        "where the data are sparse"))
    C1 = _moment_matrix(Matrix(Ws'), Q, Mii)
    ρ0 = _sign_rule_root(_quadratic(C1, y, Wy)..., bnds)
    ρ0 === nothing && return nothing
    y0 = y .- ρ0 .* Wy
    qy = Q' * y0
    s0 = (y0 .- Q * qy) .^ 2 ./ Mii
    return (; Q, qy, s0)
end

# The ℛ-free parts of the bound, for a linear coefficient ṽ = Tm·f_B and a
# quadratic matrix Qq: √(debiased v̂′Π⁺v̂), with v̂ = Tm·P_Z y₀(ρ̂₀), and χ̄, the
# largest |eigenvalue| of N·sym(Qq)·N′.
function _bias_lin_chi(Tm::AbstractMatrix{T}, Qq::AbstractMatrix{T}, N::AbstractMatrix{T},
                       pilot) where {T}
    R1 = N * (Tm * pilot.Q)
    colsq = vec(sum(abs2, R1 * pilot.Q'; dims = 1))  # ‖Υeᵢ‖²
    nv = R1 * pilot.qy
    lin = sqrt(max(dot(nv, nv) - dot(colsq, pilot.s0), zero(T)))
    χ = maximum(abs, eigvals(Symmetric(N * ((Qq .+ Qq') ./ 2) * N')))
    return lin, χ
end

# Lemma 4 (the class on [X, basis]) or Lemma 5 (on a reference basis Z̃) for the
# smoother A = MW + MDM; M = I − S_λ.
function _bias_parts(M::Matrix{T}, d::Vector{T}, Z::Matrix{T}, Ws::SparseMatrixCSC{T},
                     bias) where {T}
    if bias.Zt === nothing
        MZ = M * Z
        return _bias_lin_chi(Matrix((Ws' * MZ)'), MZ' * (d .* MZ), bias.N, bias.pilot)
    end
    MZt = M * bias.Zt
    Qq = MZt' * (Ws * bias.MZZt)
    Qq .+= MZt' * (d .* MZt)
    return _bias_lin_chi(Matrix((Ws' * MZt)'), Qq, bias.Nt, bias.pilot)
end

# Lemma 5(c): REsd, C = EM, on a reference basis: ṽ = (CZ̃)′f_B, Q̃ = (MZ̃)′CZ̃.
function _bias_parts_resd(C::Matrix{T}, bias) where {T}
    CZt = C * bias.Zt
    return _bias_lin_chi(Matrix(CZt'), bias.MZZt' * CZt, bias.Nt, bias.pilot)
end

# The bias-aware interval ρ̂ ± cv(b̄/√V̂)·se; with b̄ = 0, exactly the Wald interval.
function _bias_ci(ρ::T, se::T, V::T, bbar::T, α::T) where {T}
    (V > 0 && isfinite(se)) || return (T(NaN), T(NaN))
    c = iszero(bbar) ? T(_norminv(1 - α / 2)) : T(_cv_bias(bbar / sqrt(V), α))
    return (ρ - c * se, ρ + c * se)
end

# Π^{+1/2} on range(Π) and an orthonormal basis of null(Π).
function _penalty_parts(Π::Matrix{T}) where {T}
    E = eigen(Symmetric(Π))
    tol = 1e-10 * max(maximum(abs, E.values), floatmin(T))
    keep = E.values .> tol
    return Diagonal(1 ./ sqrt.(E.values[keep])) * Matrix(E.vectors[:, keep]'),
           Matrix(E.vectors[:, .!keep])
end

"""
    basis_roughness(f_ref, basis, penalty; X = nothing) -> ℛ

The roughness `√(ψ′Πψ)` of a reference surface `f_ref` (a vector over the
`n` units) in a basis and penalty: `ψ` is the least-squares fit of `f_ref` on
`[X, basis]`, and `Π` applies `penalty` to the basis block and leaves `X`
unpenalised, as [`root_rho`](@ref) does. Use it to set `roughness`: choose
an `f_ref` at least as rough as `f` is believed to be. The data cannot
choose `ℛ` (`research/trace_cfe/THEORY.md`, "Choosing ℛ"), and too small
an `ℛ` costs coverage, so err on the large side.

For a tensor hat basis with knot spacing `h` and a second-difference
penalty, `ψ′Πψ ≈ h²∫(f_uu² + f_vv²)`, so `ℛ` is also a bound on the mean
squared curvature (measured ratio 1.10 in `tc_rchoice.jl`).

    basis_roughness(g, u, v; k, order = 2) -> ℛ

The roughness of a reference surface `g(u, v)` (a function) in
[`tensor_pspline`](@ref)`(u, v; k, order)`: `ψ` holds `g` at the knots, in
the basis's column order, so `Bψ` is the hat interpolant of `g`. Use this
form for a `reference` basis in [`root_rho`](@ref): a fine reference has
more columns than there are units, and the least-squares fit above is then
not defined. As `k` grows, `(k − 1)·ℛ` settles (to within 2% between
`k` = 15 and 40 in `tc_refbasis.jl`), so the class does not depend on the
reference grid once it is fine. Unpenalised functions (bilinear surfaces,
for `order = 2`) give 0.
"""
function basis_roughness(f_ref::AbstractVector{<:Real}, basis::AbstractMatrix{<:Real},
                         penalty::AbstractMatrix{<:Real}; X = nothing)
    n, kb = size(basis)
    length(f_ref) == n || throw(DimensionMismatch(
        "basis_roughness: f_ref has $(length(f_ref)) entries, basis has $n rows"))
    size(penalty) == (kb, kb) || throw(DimensionMismatch(
        "basis_roughness: penalty is $(size(penalty)), expected ($kb, $kb)"))
    Xm = X === nothing ? zeros(n, 0) : Matrix{Float64}(X)
    size(Xm, 1) == n || throw(DimensionMismatch("basis_roughness: X has $(size(Xm, 1)) rows"))
    kx = size(Xm, 2)
    ψ = hcat(Xm, Matrix{Float64}(basis)) \ Vector{Float64}(f_ref)
    ψb = ψ[(kx + 1):end]
    return sqrt(max(dot(ψb, Matrix{Float64}(penalty) * ψb), 0.0))
end

function basis_roughness(g::Function, u::AbstractVector{<:Real}, v::AbstractVector{<:Real};
                         k::Integer, order::Integer = 2)
    _check_tensor_args(u, v, k, order)
    tu = range(extrema(Vector{Float64}(u))...; length = k)
    tv = range(extrema(Vector{Float64}(v))...; length = k)
    ψ = [Float64(g(tu[i], tv[j])) for i in 1:k for j in 1:k]
    all(isfinite, ψ) || throw(ArgumentError("basis_roughness: g is not finite at every knot"))
    # ψ′(P ⊗ I + I ⊗ P)ψ as ‖ΨD′‖² + ‖DΨ‖², with Ψ = reshape(ψ, k, k): the
    # differences of an unpenalised surface are then rounding-sized, and so is
    # their norm, where the quadratic form's √ would amplify rounding to ~1e-7.
    D = _difference_matrix(Int(k), Int(order)); Ψ = reshape(ψ, Int(k), Int(k))
    return sqrt(sum(abs2, Ψ * D') + sum(abs2, D * Ψ))
end

function _check_tensor_args(u, v, k, order)
    length(u) == length(v) || throw(DimensionMismatch(
        "tensor_pspline: u has $(length(u)) entries, v has $(length(v))"))
    order >= 1 || throw(ArgumentError("tensor_pspline: order must be at least 1, got $order"))
    k >= order + 2 || throw(ArgumentError(
        "tensor_pspline: k = $k knots is too few for order $order; need k ≥ $(order + 2)"))
    return nothing
end

# D, the order-th difference matrix on k knots.
function _difference_matrix(k::Int, order::Int)
    D = Matrix{Float64}(I, k, k)
    for _ in 1:order
        D = D[2:end, :] .- D[1:(end - 1), :]
    end
    return D
end

# P ⊗ I + I ⊗ P with P = D′D.
function _tensor_penalty(k::Int, order::Int)
    D = _difference_matrix(k, order)
    P = D' * D; Ik = Matrix{Float64}(I, k, k)
    return kron(P, Ik) .+ kron(Ik, P)
end

# Piecewise-linear "hat" functions on k equally spaced knots over the range of x.
function _hat_basis(x::Vector{T}, k::Int) where {T}
    lo, hi = extrema(x)
    hi > lo || throw(ArgumentError("tensor_pspline: a coordinate is constant"))
    t = range(lo, hi; length = k); w = (hi - lo) / (k - 1)
    return hcat([max.(zero(T), one(T) .- abs.(x .- t[j]) ./ w) for j in 1:k]...)
end

"""
    tensor_pspline(u, v; k, order = 2) -> (; basis, penalty)

A tensor-product P-spline on two coordinates, for the `basis` and `penalty`
arguments of [`root_rho`](@ref):

```julia
r = root_rho(y, X, W; tensor_pspline(u, v; k = 15)..., lambda = 1.0)
```

- `basis` — `n × k²`: the products of piecewise-linear "hat" functions on
  `k` equally spaced knots over the range of `u` and of `v`. Column
  `(i − 1)k + j` is `hᵢ(u)·hⱼ(v)`. The rows sum to one, so do not also put
  an intercept in `X`.
- `penalty` — `k² × k²`: `P ⊗ I + I ⊗ P` with `P = D′D` and `D` the
  `order`-th difference matrix: a difference penalty along each coordinate.
  It is symmetric and positive semi-definite. With `order = 2` it leaves
  four functions unpenalised: those that are linear in each coordinate's
  knot index, `1, i, j, ij`.

`k` must be at least `order + 2`. Used without `penalty`, `basis` is the
projection basis of the default `root_rho`.
"""
function tensor_pspline(u::AbstractVector{<:Real}, v::AbstractVector{<:Real};
                        k::Integer, order::Integer = 2)
    _check_tensor_args(u, v, k, order)
    Hu = _hat_basis(Vector{Float64}(u), Int(k)); Hv = _hat_basis(Vector{Float64}(v), Int(k))
    B = hcat([Hu[:, i] .* Hv[:, j] for i in 1:k for j in 1:k]...)
    return (; basis = B, penalty = _tensor_penalty(Int(k), Int(order)))
end

_polyval(p, t) = foldr((ck, acc) -> ck + t * acc, p; init = zero(t))

# Real roots of the polynomial Σₖ p[k+1]tᵏ, from the eigenvalues of its
# companion matrix. Leading coefficients that are negligible against the
# largest one are dropped.
function _real_roots(p::Vector{T}) where {T}
    scale = maximum(abs, p)
    iszero(scale) && return T[]
    q = copy(p)
    while length(q) > 1 && abs(q[end]) <= 1e-13 * scale
        pop!(q)
    end
    d = length(q) - 1
    d == 0 && return T[]
    d == 1 && return [-q[1] / q[2]]
    Cm = zeros(T, d, d)
    for i in 2:d
        Cm[i, i - 1] = one(T)
    end
    Cm[:, d] .= -q[1:d] ./ q[end]
    λ = eigvals(Cm)
    return sort!([real(z) for z in λ if abs(imag(z)) <= 1e-8 * (1 + abs(z))])
end

# The set {ρ ∈ bounds : g(ρ)² − z²V̂(ρ) ≤ 0} as closed intervals.
function _inversion_set(a::T, b::T, c::T, vp::NTuple{5, T}, z::T, bounds) where {T}
    g = (c, -b, a)
    g2 = zeros(T, 5)
    for i in 0:2, j in 0:2
        g2[i + j + 1] += g[i + 1] * g[j + 1]
    end
    q = g2 .- z^2 .* collect(vp)
    lo, hi = T(bounds[1]), T(bounds[2])
    pts = vcat(lo, filter(t -> lo < t < hi, _real_roots(q)), hi)
    out = Tuple{T, T}[]
    for k in 1:(length(pts) - 1)
        u, v = pts[k], pts[k + 1]
        u == v && continue
        _polyval(q, (u + v) / 2) <= 0 || continue
        if !isempty(out) && out[end][2] == u
            out[end] = (out[end][1], v)
        else
            push!(out, (u, v))
        end
    end
    return out
end

"""
    root_rho(y, X, W; basis = nothing, penalty = nothing, lambda = nothing,
             roughness = nothing, bonferroni = false, reference = nothing,
             varnames = nothing, level = 0.95, rho_bounds = (-0.99, 0.99),
             max_dense_n = 4000)

Jin & Lee's (2012) two-step root estimator of `ρ` in
`y = ρWy + Xβ + f + ε`, with the first stage a projection on
`Z = [X, basis]`. The moment is

```math
g(ρ) = y_0(ρ)'\\,C\\,y_0(ρ),\\quad y_0(ρ) = y − ρWy,\\quad
C = (A − Δ)M,\\; Δ = \\mathrm{Diag}(AM)\\,\\mathrm{Diag}(M)^{-1},
```

with `M = I − P_Z`, so `diag C = 0` and `CZ = 0`, and `E g(ρ₀) = 0` under
any heteroskedasticity. The first step uses `A = W′` and gives `ρ̃`; the
second (their REsd) uses `A = G(ρ̃)′`, `G(r) = W(I − rW)⁻¹`, and gives `ρ̂`.
Each is the root of a quadratic at which `g′ < 0`.

Two intervals are returned (see [`RootRhoResult`](@ref)): the Wald interval
with the HC2 `V̂`, and the test-inversion set `{ρ : g(ρ)² ≤ z²V̂(ρ)}`,
which stays valid when `ρ` is weakly identified. `V̂(ρ)` is a quartic in `ρ`,
so the set's endpoints are exact.

**When `K/n` is not small.** The HC2 `V̂` is not consistent when `K/n`
tends to `θ > 0`. Its relative bias at `ρ₀` has an explicit limit
(THEORY.md, Lemma 7), and the test-inversion set covers
`2Φ(z√(1 + β_V)) − 1` for large `n` (Proposition 6″). That needs a local
basis, one whose projection draws on a bounded total weight of the units
(Lemma 10), such as [`tensor_pspline`](@ref). With homoskedastic symmetric
errors, the bias is at least the worst case over error laws, which is
computed from the design alone (Lemma 8) and reported as
`vhat_worst_bias`, for a projection. It is `O(K/n)` (Lemma 9). It was −0.4%
to −2.7% at `K/n` = 0.06–0.40 in the designs measured, so the set's
coverage stays at or above about 94.7%. With heteroskedastic or skewed
errors the bias has no such design-only bound. It was −1% to +20% where
measured. The computation adds two `n × n × n` products, about 1.8 s at
n = 4000.

# Arguments

- `X` — regressors whose `β` is reported (point estimates only). May have
  zero columns. Include the intercept here or in `basis`, not both: a hat
  or B-spline basis already sums to one.
- `W` — a spatial weights matrix with a zero diagonal, row-standardised so
  that `rho_bounds` lies inside `(−1, 1)`.
- `basis` — the `n × K_B` basis for `f`. With `reference` and `roughness`,
  also a vector of bases, among which the shortest interval chooses (see
  "Choosing the basis").
- `penalty`, `lambda` — a symmetric positive semi-definite `K_B × K_B`
  penalty `Ω` on the basis coefficients, and its weight `λ > 0`. Give both
  or neither. [`tensor_pspline`](@ref) returns a basis and a penalty.
  With `roughness`, `lambda` may also be a vector (see below). With a
  vector of bases, `penalty` is a vector too, one per basis, and `lambda`
  applies to each.
- `roughness` — a bound `ℛ` on the roughness `√(ψ′Πψ)` of `f = [X, basis]ψ`
  (`Π` the penalty on the basis block), for the bias-aware interval.
  [`basis_roughness`](@ref) computes it from a reference surface.
- `bonferroni` — with a vector of `lambda`s or of bases, use level `1 − α/L`
  for each of the `L` candidates, so that coverage holds for any choice
  among them.
- `reference` — a finer basis and penalty `(; basis, penalty)`, as
  [`tensor_pspline`](@ref) returns, on which `roughness` states the class,
  so that `f` need not lie in `span(basis)` (see below). Then `roughness`
  may be given without `penalty`, for the projection alone.

# With a penalised smoother

The first stage is `S = Z(Z′Z + λΩ̃)⁻¹Z′`, where `Ω̃` applies `Ω` to the
basis and leaves `X` unpenalised. `M = I − S` is not idempotent, so a
different moment is used, in one step:

```math
g(ρ) = y_0(ρ)'\\,A\\,y_0(ρ),\\quad A = MW + MDM,\\quad (M∘M)\\,d = −\\mathrm{diag}(MW),
```

so `diag A = 0` and `E g(ρ₀) = f′Af`, the bias from the smoother's
shrinkage of `f` (THEORY.md, Proposition 4). The HC2 plug-in is
`ûᵢ²/(M²)ᵢᵢ`, and the noise term uses `H = (A + A′)S`. Jin & Lee's
two-step is not used here: its plug-in result needs `Mf = 0`.

**Choose a small `λ`.** The shrinkage bias grows with `λ`, and it makes
both intervals under-cover. GCV and the RMSE-optimal `λ` were too large for
valid inference in `research/trace_cfe/tc_lambda.jl`, so `lambda` has no
default. Report `ρ̂` across a few values of `λ`, with `r.df = tr S`. A
`df/n` above 0.30 gives a warning, and units with `Sᵢᵢ` above 0.99 are
refused.

# The bias-aware interval

The smoother's shrinkage biases the moment by `β = f′Af`, which the Wald
interval and the test-inversion set ignore. Given `roughness = ℛ`, the
result also holds the interval

```math
ρ̂ ± \\mathrm{cv}(\\bar b/\\sqrt{\\hat V})\\,\\mathrm{se},\\qquad
\\bar b = ℛ\\sqrt{v'Π^+v} + ℛ^2\\bar χ,
```

where `b̄` bounds `|β|` for every `f = [X, basis]ψ` with `ψ′Πψ ≤ ℛ²`
(THEORY.md, Lemma 4), `v = (MZ)′Wf` is estimated from the projection (at Jin
& Lee's first-step root, and debiased), `χ̄` bounds the quadratic part, and
`cv(t)` is the `level` quantile of `|N(t, 1)|`. Its coverage holds uniformly
over that class, whatever `λ` (Proposition 7), so `λ` no longer has to be
small for the interval to be valid. With `ℛ = 0` it is the Wald interval.

With a **vector** of `lambda`s, each is fitted, the projection is added as
`λ = 0` (its bias bound is 0), and the result is the candidate with the
shortest bias-aware interval; `lambda_grid` and `halflength_grid` record
them all. Proposition 7 covers that choice when one candidate is clearly the
shortest; `bonferroni = true` covers any choice, at a cost in length (about
46% at `b̄ = 0` with 12 candidates).

**`ℛ` is an assumption, not an estimate.** The data cannot certify an upper
bound on the roughness, and every rule that estimated it under-covered
(`research/trace_cfe/tc_rchoice.jl`). State it through a reference surface
at least as rough as `f` is believed to be ([`basis_roughness`](@ref)), and
err on the large side: a large `ℛ` moves the choice towards the projection
and costs at most the gain, a small one costs coverage. In
`tc_honest.jl`'s design (n = 900), with `ℛ` the true roughness, the chosen
interval covered 0.954 and 0.964 and was 53% and 33% shorter than the
projection's; with `ℛ` four times too large, it covered 0.982 and 0.970
and was still 35% and 7% shorter. Report the result across a few `ℛ`.

**With a reference basis.** The class above contains only `f ∈ span(Z)`,
and gives the projection `b̄ = 0`. A real `f` leaves a remainder that
biases every candidate, the projection too. `reference = (; basis = B̃,
penalty = Π̃)` states the class on a finer basis,
`{f = [X, B̃]ψ̃ : ψ̃′Π̃ψ̃ ≤ ℛ̃²}`, while the estimator keeps `basis`. The bound
`b̃` then covers the remainder (THEORY.md, Lemma 5 and Proposition 7′). It
needs the reference's unpenalised functions to be unpenalised in
`[X, basis]` (tensor hat bases on the same coordinates, with the same
penalty `order`), which is checked. State `ℛ̃` with
`basis_roughness(g, u, v; k)` for a reference surface `g`:

```julia
ref = tensor_pspline(u, v; k = 30)
ℛ̃ = basis_roughness((s, t) -> 2sin(2πs)cos(πt), u, v; k = 30)
r = root_rho(y, X, W; ps..., lambda = 10.0 .^ (-3:0.5:2), roughness = ℛ̃, reference = ref)
```

Use it whenever `basis` may be too coarse for `f`. In
`research/trace_cfe/tc_refbasis.jl` (n = 900), the class without a reference
covered 0.050, 0.717 and 0.923 with 4, 5 and 7 knots a side and a rough `f`;
with the reference, every cell covered 0.960–1.000. The price is length,
and it is small only for a fine `basis`: with 15 knots and `ℛ̃` true, 0% and
8% longer than without; with `ℛ̃` four times too large, 7% and 115%; with
coarse bases, 1.1 to 10 times. So the projection is no longer free of the
bound, and with a reference the rule can do worse than the projection's
Wald interval, which is then not valid. Because the class no longer depends
on `basis`, results across bases are comparable, and the shortest interval
was at the finest basis tried. The projection candidate is REsd, which
THEORY.md's Proposition 1′ covers when the largest remainder `‖Mf‖` in the
class is small against `√n` (Proposition 7′(vi)). Each candidate costs two more `n × n × K̃` products
and a `K̃ × K̃` eigenproblem: a grid of 13 took about 3.5 s at n = 900,
K̃ = 901.

# Choosing the basis

The interval is valid only if the approximation error of `f` is small
against the sampling noise (THEORY.md, Proposition 3: `β = o(√n)`). That
calls for a **larger** basis than a good fit of `f` would: GCV and the
RMSE-optimal size over-smooth for inference (`research/trace_cfe/`,
`tc_lambda.jl`). Report `ρ̂` across a few basis sizes. The coverage of both
intervals was measured for `K/n` up to 0.30; a larger `K` gives a warning.
Units with leverage above 0.99 make the correction ill-posed and are
refused.

**Chosen by the data, with a reference basis.** Once the class is stated
on `reference`, it does not depend on `basis`, and the bias-aware intervals
of different bases bound the same thing. So `basis` may be a vector, and
the shortest interval over every (basis, λ) pair is returned (THEORY.md,
Corollary 7′); `basis_grid` and `basis_index` record the choice, and
`bonferroni = true` counts every pair. Each basis needs Lemma 5's
condition (R), and costs one fit per `λ`:

```julia
ref = tensor_pspline(u, v; k = 30)
pss = [tensor_pspline(u, v; k) for k in (5, 10, 15)]
r = root_rho(y, X, W; basis = [p.basis for p in pss], penalty = [p.penalty for p in pss],
             lambda = 10.0 .^ (-3:0.5:2), roughness = ℛ̃, reference = ref)
```

Without a reference this is refused: the class `F(ℛ)` then depends on the
basis, and a coarse basis would look short only because its class is
small.

# The root rule

`g` is a quadratic in `ρ`, and `ρ̂` is the root at which `g′ < 0`. This
assumes `J = −E g′(ρ₀) > 0` (THEORY.md, A9). If `J < 0`, the other root is
the consistent one, and the rule picks a root at a distance of about
`|J/g₂|` from `ρ₀` (`g₂` the quadratic coefficient), which does not shrink
with `n`. A 2SLS estimate (instruments `[X, basis, W·[X, basis]]`) is used
to check the sign: if `g′` is positive there, a warning is given. The
test-inversion set does not use the rule and stays valid.

`J < 0` is rare. In a search over 3,000 hub-and-spoke designs, 0.7% had it;
the one in testset 47g gives a sign-rule root biased by 0.06 (2 SD) with the
smoother. In 1,000 fits on kNN designs (`n` 400–900, both paths) the check
never fired. REsd's second step has `J > 0` in the designs measured, even
when the first step's `J` is negative, so on the projection path the
warning is rarer still.

# Cost

Everything is dense: about four `n × n` matrices, so `n` is capped at
`max_dense_n` (4000 by default, about 0.5 GB). The penalised smoother holds
about six and solves one dense `n × n` system for `d`.

!!! note "Proved, but not checked by anyone else"
    The results for a basis that grows with `n` (a CLT for the moment, the
    consistency of `V̂`, the validity of the test-inversion set, and the
    coverage of the bias-aware interval, also on a reference basis and with
    the basis chosen by the data) are in
    `research/trace_cfe/THEORY.md`. The published theory (Jin & Lee 2012)
    assumes a design of fixed dimension.
"""
function root_rho(y::AbstractVector{<:Real}, X::AbstractMatrix{<:Real}, W::AbstractMatrix{<:Real};
                  basis::Union{Nothing, AbstractMatrix{<:Real},
                               AbstractVector{<:AbstractMatrix{<:Real}}} = nothing,
                  penalty::Union{Nothing, AbstractMatrix{<:Real},
                                 AbstractVector{<:AbstractMatrix{<:Real}}} = nothing,
                  lambda::Union{Nothing, Real, AbstractVector{<:Real}} = nothing,
                  roughness::Union{Nothing, Real} = nothing, bonferroni::Bool = false,
                  reference = nothing, kw...)
    basis isa AbstractVector || penalty isa AbstractVector{<:AbstractMatrix} && throw(ArgumentError(
        "root_rho: a vector of penalties needs a vector of bases, one per penalty"))
    basis isa AbstractVector ||
        return _root_rho_one(y, X, W; basis, penalty, lambda, roughness, bonferroni, reference, kw...)
    # A grid of bases (THEORY.md, Corollary 7′): the class is stated on `reference`, so it
    # does not depend on the basis, and the intervals of all bases can be compared.
    J = length(basis)
    J > 0 || throw(ArgumentError("root_rho: the vector of bases is empty"))
    (reference !== nothing && roughness !== nothing) || throw(ArgumentError(
        "root_rho: a vector of bases needs `reference` and `roughness`. Without a reference " *
        "the roughness class depends on the basis, so the intervals of different bases " *
        "bound different things and cannot be compared"))
    pens = if penalty === nothing
        fill(nothing, J)
    elseif penalty isa AbstractVector{<:AbstractMatrix}
        length(penalty) == J || throw(DimensionMismatch(
            "root_rho: $(length(penalty)) penalties for $J bases"))
        penalty
    else
        throw(ArgumentError("root_rho: with a vector of bases, give a vector of penalties, one per basis"))
    end
    per = penalty === nothing ? 1 : lambda isa AbstractVector ? length(lambda) + 1 : 1
    nb = bonferroni ? J * per : 0
    fits = [_root_rho_one(y, X, W; basis = basis[j], penalty = pens[j], lambda, roughness,
                          reference, nbonf = nb, kw...) for j in 1:J]
    T = Float64
    grid = T[]; hl = T[]; bg = Int[]
    for (j, r) in enumerate(fits)
        if isempty(r.lambda_grid)
            push!(grid, r.lambda)
            push!(hl, r.root_found && isfinite(r.ci_bias_aware[1]) ?
                      (r.ci_bias_aware[2] - r.ci_bias_aware[1]) / 2 : T(Inf))
            push!(bg, j)
        else
            append!(grid, r.lambda_grid); append!(hl, r.halflength_grid)
            append!(bg, fill(j, length(r.lambda_grid)))
        end
    end
    best = [minimum(hl[bg .== j]) for j in 1:J]
    j = all(isinf, best) ? J : argmin(best)
    r = fits[j]
    return _with_bias(r, roughness, r.bias_bound, r.ci_bias_aware, grid, hl; bgrid = bg, bidx = j)
end

# One basis. `nbonf > 0` splits α among that many candidates (a grid of bases).
function _root_rho_one(y::AbstractVector{<:Real}, X::AbstractMatrix{<:Real}, W::AbstractMatrix{<:Real};
                       basis::Union{Nothing, AbstractMatrix{<:Real}} = nothing,
                       penalty::Union{Nothing, AbstractMatrix{<:Real}} = nothing,
                       lambda::Union{Nothing, Real, AbstractVector{<:Real}} = nothing,
                       roughness::Union{Nothing, Real} = nothing, bonferroni::Bool = false,
                       nbonf::Int = 0, reference = nothing, varnames = nothing,
                       level::Real = 0.95, rho_bounds = (-0.99, 0.99),
                       max_dense_n::Integer = 4000)
    T = Float64
    n = length(y)
    size(X, 1) == n || throw(DimensionMismatch("root_rho: y has $n rows, X has $(size(X, 1))"))
    size(W) == (n, n) || throw(DimensionMismatch("root_rho: W is $(size(W)), expected ($n, $n)"))
    basis === nothing || size(basis, 1) == n || throw(DimensionMismatch(
        "root_rho: y has $n rows, basis has $(size(basis, 1))"))
    0 < level < 1 || throw(ArgumentError("root_rho: level must be in (0, 1), got $level"))
    (penalty === nothing) == (lambda === nothing) || throw(ArgumentError(
        "root_rho: give both `penalty` and `lambda`, or neither. There is no default λ: " *
        "GCV and the RMSE-optimal λ smooth too much for valid inference"))
    penalised = penalty !== nothing
    roughness === nothing || penalised || reference !== nothing || throw(ArgumentError(
        "root_rho: `roughness` needs a penalty and lambda, or a `reference` basis: with f in " *
        "span(basis), a projection has no bias, and its Wald interval is the bias-aware one"))
    reference === nothing || roughness !== nothing || throw(ArgumentError(
        "root_rho: `reference` states the roughness class of `roughness`; give both"))
    reference === nothing || basis !== nothing || throw(ArgumentError(
        "root_rho: `reference` needs a `basis` for the estimator"))
    Pref = reference === nothing ? nothing : _check_reference(reference, n)
    roughness === nothing || (isfinite(roughness) && roughness >= 0) || throw(ArgumentError(
        "root_rho: roughness must be finite and non-negative, got $roughness"))
    lambda isa AbstractVector && roughness === nothing && throw(ArgumentError(
        "root_rho: a vector of lambdas needs `roughness`: λ is chosen by the length of the " *
        "bias-aware interval, and without a roughness bound there is none"))
    bonferroni && !(lambda isa AbstractVector) && throw(ArgumentError(
        "root_rho: `bonferroni` applies only to a vector of lambdas"))
    if penalised
        basis === nothing && throw(ArgumentError("root_rho: a penalty needs a basis"))
        lambda isa AbstractVector && isempty(lambda) && throw(ArgumentError(
            "root_rho: the vector of lambdas is empty"))
        all(l -> isfinite(l) && l > 0, lambda) || throw(ArgumentError(
            "root_rho: lambda must be positive and finite, got $lambda; for λ = 0 omit " *
            "penalty and lambda (the projection)"))
        kb = size(basis, 2)
        size(penalty) == (kb, kb) || throw(DimensionMismatch(
            "root_rho: penalty is $(size(penalty)), expected ($kb, $kb) for the basis"))
        Pm = Matrix{Float64}(penalty)
        sc = max(maximum(abs, Pm), floatmin(Float64))
        maximum(abs, Pm .- Pm') <= 1e-10 * sc || throw(ArgumentError(
            "root_rho: penalty must be symmetric"))
        eigmin(Symmetric(Pm)) >= -1e-10 * sc || throw(ArgumentError(
            "root_rho: penalty must be positive semi-definite"))
    end
    n <= max_dense_n || throw(ArgumentError(
        "root_rho: n = $n exceeds max_dense_n = $max_dense_n. The estimator holds about " *
        "four dense n × n matrices ($(round(4 * 8 * n^2 / 2^30; digits = 1)) GiB here); " *
        "raise max_dense_n if that fits in memory"))
    any(!iszero, diag(W)) && throw(ArgumentError(
        "root_rho: W must have a zero diagonal"))
    Ws = sparse(T.(W))
    _check_spatial_bounds(Ws, rho_bounds, ExactLU(); caller = "root_rho")
    bnds = (T(rho_bounds[1]), T(rho_bounds[2]))

    Xm = Matrix{T}(X)
    kx = size(Xm, 2)
    vnames = varnames === nothing ? ["x$j" for j in 1:kx] : String.(collect(varnames))
    length(vnames) == kx || throw(ArgumentError(
        "root_rho: $(length(vnames)) varnames for $kx columns of X"))
    Bm = basis === nothing ? zeros(T, n, 0) : Matrix{T}(basis)
    Z = hcat(Xm, Bm)
    K = size(Z, 2)
    K > 0 || throw(ArgumentError("root_rho: X and basis together have no columns"))
    check_design_rank(Z, vcat(vnames, ["basis$j" for j in 1:size(Bm, 2)]), "root_rho";
        context = "A hat or B-spline basis sums to one, so it is collinear with an " *
                  "intercept in X: drop one of them.")
    yv = Vector{T}(y)
    penalised || roughness !== nothing ||
        return _root_rho_projection(yv, Z, Ws, kx, vnames, T(level), bnds)
    roughness === nothing && return _root_rho_penalised(yv, Z, Ws, kx, Pm, T(lambda), vnames,
                                                        T(level), bnds, nothing)
    # The bias-aware interval: one λ, or λ chosen by length with the projection as λ = 0.
    Wy = Ws * yv
    pilot = _bias_pilot(yv, Wy, Z, Ws, bnds)
    pilot === nothing && throw(ArgumentError(
        "root_rho: the projection pilot has no admissible root, so the bias bound cannot " *
        "be estimated; the test-inversion sets of root_rho without `roughness` still apply"))
    N, N0 = if penalised
        Πt = zeros(T, K, K)
        Πt[(kx + 1):K, (kx + 1):K] .= Pm
        _penalty_parts(Πt)
    else
        zeros(T, 0, K), Matrix{T}(I, K, K)           # a projection: everything is free
    end
    refK = 0
    Zt = Nt = MZZt = nothing
    if reference !== nothing
        Zt = hcat(Xm, Matrix{T}(reference.basis))
        refK = size(Zt, 2)
        Πtt = zeros(T, refK, refK)
        Πtt[(kx + 1):refK, (kx + 1):refK] .= Pref
        Nt, N0t = _penalty_parts(Πtt)
        _check_reference_null(Zt * N0t, Z * N0)
        refK > 2n && @warn "root_rho: the reference basis has $refK columns for n = $n; " *
            "each candidate costs O(n²K̃ + K̃³)"
        MZZt = Zt .- pilot.Q * (pilot.Q' * Zt)
    end
    α0 = 1 - T(level)
    if !penalised                                   # the projection alone, with a reference
        proj, C = _root_rho_projection_C(yv, Z, Ws, kx, vnames, T(level), bnds)
        bias = (; ℛ = T(roughness), N, pilot, α = α0 / max(nbonf, 1), Zt, Nt, MZZt, refK)
        return _projection_with_bias(proj, C, bias)
    end
    λs = lambda isa AbstractVector ? T.(collect(lambda)) : [T(lambda)]
    ncand = lambda isa AbstractVector ? length(λs) + 1 : 1
    α = α0 / (nbonf > 0 ? nbonf : bonferroni ? ncand : 1)
    bias = (; ℛ = T(roughness), N, pilot, α, Zt, Nt, MZZt, refK)
    fits = [_root_rho_penalised(yv, Z, Ws, kx, Pm, λ, vnames, T(level), bnds, bias) for λ in λs]
    lambda isa AbstractVector || return fits[1]
    proj = try
        _root_rho_projection_C(yv, Z, Ws, kx, vnames, T(level), bnds)
    catch e
        e isa ArgumentError || rethrow()
        @warn "root_rho: the projection (λ = 0) is left out of the candidates: $(e.msg)"
        nothing
    end
    cands = RootRhoResult{T}[]
    proj === nothing || push!(cands, _projection_with_bias(proj..., bias))
    append!(cands, fits)
    grid = [c.lambda for c in cands]
    hl = [c.root_found && isfinite(c.ci_bias_aware[1]) ?
          (c.ci_bias_aware[2] - c.ci_bias_aware[1]) / 2 : T(Inf) for c in cands]
    all(isinf, hl) && return _with_bias(cands[end], roughness, cands[end].bias_bound,
                                        cands[end].ci_bias_aware, grid, hl)
    j = argmin(hl)
    return _with_bias(cands[j], roughness, cands[j].bias_bound, cands[j].ci_bias_aware, grid, hl)
end

# The projection (REsd) as a bias-aware candidate: b̄ = 0 when f ∈ span(Z) is
# assumed (Lemma 4's class), Lemma 5(c) on a reference basis.
function _projection_with_bias(proj::RootRhoResult{T}, C::Matrix{T}, bias) where {T}
    proj.root_found || return _with_bias(proj, bias.ℛ, T(NaN), (T(NaN), T(NaN)), T[], T[];
                                         refK = bias.refK)
    bbar = if bias.Zt === nothing
        zero(T)
    else
        lin, χ = _bias_parts_resd(C, bias)
        bias.ℛ * lin + bias.ℛ^2 * χ
    end
    return _with_bias(proj, bias.ℛ, bbar, _bias_ci(proj.rho, proj.se, proj.vhat, bbar, bias.α),
                      T[], T[]; refK = bias.refK)
end

# `reference = (; basis, penalty)`: the checks applied to `penalty`, and its matrix.
function _check_reference(reference, n::Int)
    (hasproperty(reference, :basis) && hasproperty(reference, :penalty)) || throw(ArgumentError(
        "root_rho: `reference` must have fields `basis` and `penalty`, as tensor_pspline returns"))
    Bt = reference.basis; Pt = reference.penalty
    size(Bt, 1) == n || throw(DimensionMismatch(
        "root_rho: y has $n rows, reference.basis has $(size(Bt, 1))"))
    kt = size(Bt, 2)
    size(Pt) == (kt, kt) || throw(DimensionMismatch(
        "root_rho: reference.penalty is $(size(Pt)), expected ($kt, $kt)"))
    P = Matrix{Float64}(Pt)
    sc = max(maximum(abs, P), floatmin(Float64))
    maximum(abs, P .- P') <= 1e-10 * sc || throw(ArgumentError(
        "root_rho: reference.penalty must be symmetric"))
    eigmin(Symmetric(P)) >= -1e-10 * sc || throw(ArgumentError(
        "root_rho: reference.penalty must be positive semi-definite"))
    return P
end

# Lemma 5's condition (R): the reference's unpenalised functions must be
# unpenalised functions of [X, basis] (all of span[X, basis] for a projection).
function _check_reference_null(Yt::Matrix{T}, Y::Matrix{T}) where {T}
    size(Yt, 2) == 0 && return nothing
    res = Yt .- Y * (Y \ Yt)
    rel = norm(res) / max(norm(Yt), floatmin(T))
    rel <= 1e-8 || throw(ArgumentError(
        "root_rho: the reference basis leaves unpenalised functions that [X, basis] penalises " *
        "or cannot represent (relative residual $(round(rel; sigdigits = 3))), so Lemma 5's " *
        "condition B̃·null(Π̃) ⊂ B·null(Π) fails; use the same box and penalty order for both"))
    return nothing
end

# The projection on Z: Jin & Lee's two steps (REsd).
_root_rho_projection(args...) = first(_root_rho_projection_C(args...))

# The same, with the final moment matrix C, for the bias bound (Lemma 5(c)).
function _root_rho_projection_C(yv::Vector{T}, Z::Matrix{T}, Ws::SparseMatrixCSC{T}, kx::Int,
                                vnames::Vector{String}, level::T, bnds::Tuple{T, T}) where {T}
    n, K = size(Z)
    Q = Matrix(qr(Z).Q)[:, 1:K]
    Mii = 1 .- vec(sum(abs2, Q; dims = 2))
    h = 1 .- Mii
    nbad = count(>(0.99), h)
    nbad > 0 && throw(ArgumentError(
        "root_rho: $nbad unit(s) have leverage above 0.99 in the projection on [X, basis] " *
        "(max $(round(maximum(h); digits = 4))). The trace correction divides by 1 − hᵢᵢ " *
        "and is ill-posed there; use a basis with fewer columns where the data are sparse"))
    K / n > 0.30 && @warn "root_rho: K/n = $(round(K / n; digits = 3)) is above 0.30, " *
        "beyond the range where the intervals' coverage was measured"

    Wy = Ws * yv
    z = T(_norminv(1 - (1 - level) / 2))
    Wt = Matrix(Ws')

    # First step: A = W′
    C1 = _moment_matrix(Wt, Q, Mii)
    a1, b1, c1 = _quadratic(C1, yv, Wy)
    ρ1 = _sign_rule_root(a1, b1, c1, bnds)

    C, a, b, c, ρ = C1, a1, b1, c1, nothing
    if ρ1 !== nothing
        # REsd: A = G(ρ̃)′ = (I − ρ̃W′)⁻¹W′
        Gt = lu(sparse(one(T) * I, n, n) - ρ1 * sparse(Ws')) \ Wt
        C2 = _moment_matrix(Gt, Q, Mii)
        a2, b2, c2 = _quadratic(C2, yv, Wy)
        ρ2 = _sign_rule_root(a2, b2, c2, bnds)
        C, a, b, c = C2, a2, b2, c2
        ρ = ρ2
    end
    ρ === nothing || _check_slope_sign(_prelim_2sls(yv, Wy, Z, Ws), a, b)
    vp = _vhat_poly(C, Q, Mii, yv, Wy)
    cs = _inversion_set(a, b, c, vp, z, bnds)
    touches = !isempty(cs) && (cs[1][1] == bnds[1] || cs[end][2] == bnds[2])
    βw = _hc2_worst_bias(C, Q, Mii)

    if ρ === nothing
        nanT = T(NaN)
        return RootRhoResult{T}(nanT, ρ1 === nothing ? nanT : ρ1, nanT, (nanT, nanT), cs,
            touches, length(cs) > 1, T(level), fill(nanT, kx), vnames, n, K, :projection,
            zero(T), T(K), maximum(h), a, b, c, vp, nanT, false, bnds, βw, _no_bias(T)...), C
    end
    V = _polyval(vp, ρ)
    se = V > 0 ? sqrt(V) / abs(b - 2ρ * a) : T(NaN)
    β = kx > 0 ? (Z \ (yv .- ρ .* Wy))[1:kx] : T[]
    return RootRhoResult{T}(ρ, ρ1, se, (ρ - z * se, ρ + z * se), cs, touches,
        length(cs) > 1, T(level), β, vnames, n, K, :projection, zero(T), T(K), maximum(h),
        a, b, c, vp, V, true, bnds, βw, _no_bias(T)...), C
end

# The penalised smoother: S = Z(Z′Z + λΩ̃)⁻¹Z′, the exact-diagonal moment in
# one step, and the general HC2 V̂ (THEORY.md, Proposition 4).
function _root_rho_penalised(y::Vector{T}, Z::Matrix{T}, Ws::SparseMatrixCSC{T}, kx::Int,
                             Ω::Matrix{T}, λ::T, vnames::Vector{String}, level::T,
                             bnds::Tuple{T, T}, bias) where {T}
    n, K = size(Z)
    Ωt = zeros(T, K, K)
    Ωt[(kx + 1):K, (kx + 1):K] .= Ω
    F = cholesky(Symmetric(Z' * Z .+ λ .* Ωt))
    S = Z * (F \ Z')
    S .= (S .+ S') ./ 2
    Sii = diag(S)
    nbad = count(>(0.99), Sii)
    nbad > 0 && throw(ArgumentError(
        "root_rho: $nbad unit(s) have Sᵢᵢ above 0.99 in the penalised smoother " *
        "(max $(round(maximum(Sii); digits = 4))). The correction divides by (M∘M) and " *
        "is ill-posed there; use a larger lambda or fewer basis functions where the " *
        "data are sparse"))
    df = sum(Sii)
    df / n > 0.30 && @warn "root_rho: df/n = tr S/n = $(round(df / n; digits = 3)) is " *
        "above 0.30, beyond the range where the intervals' coverage was measured"
    M = Matrix{T}(I, n, n) .- S
    m = vec(sum(abs2, M; dims = 1))                  # (M²)ᵢᵢ, M symmetric
    A, d = _exact_diag_moment_d(M, Ws)
    H = (A .+ A') * S
    Wy = Ws * y
    z = T(_norminv(1 - (1 - level) / 2))
    a, b, c = _quadratic(A, y, Wy)
    ρ = _sign_rule_root(a, b, c, bnds)
    ρ === nothing || _check_slope_sign(_prelim_2sls(y, Wy, Z, Ws), a, b)
    vp = _vhat_poly_general(A, H, M, m, y, Wy)
    cs = _inversion_set(a, b, c, vp, z, bnds)
    touches = !isempty(cs) && (cs[1][1] == bnds[1] || cs[end][2] == bnds[2])
    nanT = T(NaN)
    if ρ === nothing
        r = RootRhoResult{T}(nanT, nanT, nanT, (nanT, nanT), cs, touches, length(cs) > 1,
            level, fill(nanT, kx), vnames, n, K, :penalised, λ, df, maximum(Sii), a, b, c,
            vp, nanT, false, bnds, nanT, _no_bias(T)...)
        return bias === nothing ? r : _with_bias(r, bias.ℛ, nanT, (nanT, nanT), T[], T[];
                                                 refK = bias.refK)
    end
    V = _polyval(vp, ρ)
    se = V > 0 ? sqrt(V) / abs(b - 2ρ * a) : nanT
    β = kx > 0 ? (F \ (Z' * (y .- ρ .* Wy)))[1:kx] : T[]
    r = RootRhoResult{T}(ρ, nanT, se, (ρ - z * se, ρ + z * se), cs, touches,
        length(cs) > 1, level, β, vnames, n, K, :penalised, λ, df, maximum(Sii), a, b, c,
        vp, V, true, bnds, nanT, _no_bias(T)...)
    bias === nothing && return r
    lin, χ = _bias_parts(M, d, Z, Ws, bias)
    bbar = bias.ℛ * lin + bias.ℛ^2 * χ
    return _with_bias(r, bias.ℛ, bbar, _bias_ci(ρ, se, V, bbar, bias.α), T[], T[];
                      refK = bias.refK)
end

end # module
