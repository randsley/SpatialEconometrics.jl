"""
    SemiparametricSpatial

Spatial autoregressions with an unknown smooth component — roadmap #13.

```math
y = ρWy + Xβ + m(z) + ε
```

`m` is left unspecified and estimated by a local-polynomial smoother on
`z`. Three things live here, and they share one first stage:

1. [`fit_partially_linear_sar`](@ref): Su & Jin's (2010) profile
   quasi-maximum likelihood. The smoother matrix `S` depends on `z` and the
   bandwidth only, not on `ρ`, so after residualising on `(I − S)` the
   concentrated likelihood has exactly the shape `fit_spatial` maximises.
   It **keeps** the log-determinant.
2. [`closed_form_rho`](@ref): Smirnov's (2022) closed-form `ρ`, computed
   from the same residualised `y`, `Wy` and `W²y`. No log-determinant.
3. [`spectral_overlap`](@ref): Geniaux's (2026) diagnostic of how much of
   `W`'s dominant spatial modes the smoother can reproduce. A smoother that
   reproduces them competes with `ρWy` for the same variation — the
   "geoadditive trap" — and `ρ` loses its identification.

# What the overlap check can and cannot say

A small overlap says only that the smoother does not compete with `ρWy`
for `W`'s dominant modes, **given that `W` is the true interaction
structure**. That is statistical identification of `ρ`. It is not causal
identification of a spillover: the endogenous and the correlated effects
stay confounded (Manski's reflection problem), and in a cross-section
spatial heterogeneity and spillovers are observationally equivalent, so a
large overlap cannot say which allocation of the variation is true either
(Gibbons & Overman 2012; Debarsy & Le Gallo 2024). Nothing in this module
makes `ρ̂`, or the impacts computed from it, causal.

# References

- Su, L., & Jin, S. (2010). Profile quasi-maximum likelihood estimation of
  partially linear spatial autoregressive models. *Journal of
  Econometrics*, 157(1), 18–33.
- Smirnov, O. A. (2022). Closed-form estimator for spatial autoregressive
  models. *Geographical Analysis*, 54(1), 3–31.
- Geniaux, G. (2026). Flexible nonlinear spatial autoregressive models
  under spatial confounding. arXiv:2611.16592 (preprint).
- Lee, L.-F. (2004). Asymptotic distributions of quasi-maximum likelihood
  estimators for spatial autoregressive models. *Econometrica*, 72(6),
  1899–1925.
- Gibbons, S., & Overman, H. G. (2012). Mostly pointless spatial
  econometrics? *Journal of Regional Science*, 52(2), 172–191.
- Debarsy, N., & Le Gallo, J. (2024). Identification of spatial
  spillovers: Do's and dont's. LEM working paper DT2024-03.
"""
module SemiparametricSpatial

using LinearAlgebra
using SparseArrays
using Statistics
using Random
using Printf

import StatsAPI
import StatsAPI: coef, vcov, stderror, residuals, loglikelihood, nobs,
                 coefnames, fitted

using NearestNeighbors: inrange, knn

import ..SpatialUtils: LogDetMethod, ExactLU, ChebyshevApprox,
       precompute_chebyshev_traces, chebyshev_logdet, g_traces,
       brent_maximize, check_design_rank, _normcdf, _print_coef_table
import ..GWREstimation: _build_tree
import ..SpatialFiltering: spatial_eigenvectors

export local_polynomial_smoother,
       SpectralOverlapResult, spectral_overlap,
       PartiallyLinearSARModel, fit_partially_linear_sar,
       ClosedFormRhoResult, closed_form_rho,
       TVCoefSARModel, fit_tv_coef_sar

# ═══════════════════════════════════════════════════════════════════════
# The smoother
# ═══════════════════════════════════════════════════════════════════════

# Standardised Epanechnikov kernel, the one Su & Jin (2010, §6) use: unit
# variance, support |u| ≤ √5. The constant 3/(4√5) cancels in every local
# fit, so it is dropped.
const _EPAN_RADIUS = sqrt(5.0)
@inline _epan(u2::T) where {T} = u2 < T(5) ? one(T) - u2 / T(5) : zero(T)

"""
    _standardise(z)

Each column of `z` divided by its sample standard deviation, so one scalar
bandwidth means the same thing in every direction. Columns are not centred:
the local fits are translation-invariant.
"""
function _standardise(z::AbstractMatrix{T}) where {T}
    s = vec(std(z; dims=1))
    any(iszero, s) && throw(ArgumentError(
        "a column of z is constant, so it carries nothing for a smoother " *
        "to fit; drop it"))
    return z ./ permutedims(s), s
end

# Number of local-polynomial terms of total degree ≤ r in q variables.
_n_terms(q::Int, r::Int) = binomial(q + r, r)

function _local_design!(D::AbstractMatrix{T}, Zs::AbstractMatrix{T},
                        idx::AbstractVector{Int}, i::Int, h::T,
                        r::Int) where {T}
    q = size(Zs, 2)
    @inbounds for (t, j) in enumerate(idx)
        D[t, 1] = one(T)
        c = 1
        for a in 1:q
            c += 1
            D[t, c] = (Zs[j, a] - Zs[i, a]) / h
        end
        if r >= 2
            for a in 1:q, b in a:q
                c += 1
                D[t, c] = D[t, 1 + a] * D[t, 1 + b]
            end
        end
    end
    return D
end

"""
    local_polynomial_smoother(z, h; order = 1)

The `n×n` smoother matrix `S` of an order-`r` local-polynomial regression
on `z`, so that the fitted values of any `v` regressed on `z` are `S * v`.

Row `i` is `e₁'(D'KD)⁻¹D'K`, with `D` the local design `[1, (z_j − z_i)/h,
…]` up to total degree `order` and `K` the standardised Epanechnikov
kernel weights, as in Su & Jin (2010, eq. 2.6 and §6).

`z` is an `n×q` matrix (a vector is one column); each column is divided by
its standard deviation first, so `h` is in standard-deviation units. The
kernel has compact support of radius `√5 h`, which their Assumption 5
requires.

`order = 1` (local linear) is enough for `q ≤ 3`; their Assumption 5 needs
`order > q/2 − 1`, so `q ≥ 4` needs `order = 2`, and larger `q` is refused.
Only orders 1 and 2 are offered: order 0 (Nadaraya–Watson) has the boundary
bias Su & Jin exclude it for.

Local-polynomial smoothers reproduce polynomials of degree `≤ order`
exactly, so `S * ones(n) == ones(n)` to rounding. That is why an intercept
must not be in `X` for [`fit_partially_linear_sar`](@ref): `m` absorbs it.

**Windows that hold too few points are widened.** A compact kernel at a
fixed `h` can leave an observation in the tail of `z` with too few points in
its window to fit the local polynomial. That observation's window is then
widened just past its `2p + 1`-th nearest neighbour, `p` being the number of
local-polynomial terms. Su & Jin's conditions have `nh^q → ∞`, so the share
of widened windows vanishes as `n` grows; in finite samples it is reported
as `n_widened` on [`fit_partially_linear_sar`](@ref)'s result. Without this
guard, 221 of 500 fits failed at `n = 400` in the Monte Carlo of
`docs/AUDIT.md`. An `ArgumentError` is thrown only if a local design is
singular even after widening.

Dense: `O(n²)` memory, intended for `n` up to a few thousand.
"""
function local_polynomial_smoother(z::AbstractVecOrMat{<:Real}, h::Real;
                                   order::Int = 1)
    Z = z isa AbstractVector ? reshape(float.(z), :, 1) : float.(z)
    T = eltype(Z)
    n, q = size(Z)
    _check_smoother_args(q, order)
    h > 0 || throw(ArgumentError("bandwidth h must be positive, got $h"))
    Zs, _ = _standardise(Z)
    return first(_smoother(Zs, T(h), order))
end

function _check_smoother_args(q::Int, order::Int)
    order in (1, 2) || throw(ArgumentError("order must be 1 or 2, got $order"))
    order > q / 2 - 1 || throw(ArgumentError(
        "order = $order is too low for q = $q smoothing variables: Su & Jin's " *
        "Assumption 5 needs order > q/2 - 1"))
    q <= 4 || throw(ArgumentError(
        "q = $q smoothing variables is past what a local polynomial can fit " *
        "(the curse of dimensionality); Su & Jin expect small q"))
    return nothing
end

function _smoother(Zs::Matrix{T}, h::T, r::Int) where {T}
    n, q = size(Zs)
    p = _n_terms(q, r)
    m_min = min(2p + 1, n)          # points a window must hold
    tree = _build_tree(Zs)
    S = zeros(T, n, n)
    e1 = zeros(T, p); e1[1] = one(T)
    n_widened = 0
    for i in 1:n
        zi = Zs[i, :]
        hi = h
        idx = inrange(tree, zi, T(_EPAN_RADIUS) * hi, true)
        if length(idx) < m_min
            # Finite-sample guard: widen this window just past its m_min-th
            # nearest neighbour, so that neighbour carries positive weight.
            _, dists = knn(tree, zi, m_min, true)
            hi = T(1.05) * T(dists[end]) / T(_EPAN_RADIUS)
            idx = inrange(tree, zi, T(_EPAN_RADIUS) * hi, true)
            n_widened += 1
        end
        b = length(idx)
        D = Matrix{T}(undef, b, p)
        _local_design!(D, Zs, idx, i, hi, r)
        w = Vector{T}(undef, b)
        @inbounds for (t, j) in enumerate(idx)
            u2 = zero(T)
            for a in 1:q
                δ = (Zs[j, a] - Zs[i, a]) / hi
                u2 += δ * δ
            end
            w[t] = _epan(u2)
        end
        DW = D .* w
        F = cholesky(Symmetric(D' * DW); check = false)
        (issuccess(F) && cond(F.U)^2 < T(1e12)) || throw(ArgumentError(
            "bandwidth h = $(round(h; sigdigits=4)): the local design at " *
            "observation $i is singular or nearly so even after widening its " *
            "window to $b points; use a larger bandwidth"))
        s = DW * (F \ e1)
        @inbounds for (t, j) in enumerate(idx)
            S[i, j] = s[t]
        end
    end
    return S, n_widened
end

# ═══════════════════════════════════════════════════════════════════════
# The identification diagnostic
# ═══════════════════════════════════════════════════════════════════════

"""
    SpectralOverlapResult

The spectral projection diagnostic of Geniaux (2026, Appendix B.6).

# Fields

- `pi::Vector` — `π_k = ‖S φ_k‖` for each of the `k` dominant
  eigenvectors `φ_k`: the share of that spatial mode the smoother can
  reproduce, between 0 (none) and about 1 (all of it).
- `pibar` — the mean of `pi`, Geniaux's summary `π̄`.
- `eigenvalues::Vector` — the eigenvalues of the doubly-centred symmetrised
  `W` belonging to those modes, descending.
- `moran::Vector` — the Moran coefficient of each mode.

# This is a measure, not a test

The source gives **no threshold and no null distribution** for `π̄`; its
Lucas County application reports 0.06 under 4-nearest-neighbour weights
and 0.26 under 10-nearest-neighbour weights, and reads the second as a
configuration where identification needs scrutiny. There is accordingly no
p-value here and no verdict, and none should be attached.

A small `π̄` says the smoother does not compete with `ρWy` for `W`'s
dominant modes, given that `W` is the true interaction structure. It does
not make `ρ̂` causal; see the module docstring.
"""
struct SpectralOverlapResult{T<:AbstractFloat}
    pi::Vector{T}
    pibar::T
    eigenvalues::Vector{T}
    moran::Vector{T}
    k::Int
end

function Base.show(io::IO, r::SpectralOverlapResult)
    println(io, "Spectral overlap of the smoother with W's dominant modes")
    println(io, "─────────────────────────────────────────────────────────")
    @printf(io, "modes k = %d    mean overlap π̄ = %.4f    max = %.4f (mode %d)\n",
            r.k, r.pibar, maximum(r.pi), argmax(r.pi))
    println(io, "A descriptive measure with no null distribution: it flags")
    println(io, "competition between the smoother and ρWy given W, and says")
    println(io, "nothing about whether ρ is a causal spillover.")
end

"""
    spectral_overlap(S, W; k = 25)

How much of each of `W`'s `k` dominant spatial modes the smoother `S` can
reproduce: `π_j = ‖S φ_j‖` for the leading eigenvectors `φ_j` (Geniaux
2026, Appendix B.6 and Figure B.10, which uses `k = 25`).

The modes are the eigenvectors of the **doubly-centred** symmetrised `W`,
`(I − 11'/n)(W + W')/2 (I − 11'/n)`, from `spatial_eigenvectors`. Geniaux
states the eigenvectors of the symmetrised `W` itself. The centring is a
deliberate departure: every smoother here reproduces constants, and the top
eigenvector of an uncentred row-standardised `W` is nearly constant, so its
overlap would read close to 1 for any fit and say nothing. The constant is
absorbed by `m` in any case and is not a mode `ρ` competes for.

Returns a [`SpectralOverlapResult`](@ref). Dense eigendecomposition:
`O(n³)`.
"""
function spectral_overlap(S::AbstractMatrix{T}, W::AbstractSparseMatrix{T};
                          k::Int = 25) where {T<:AbstractFloat}
    n = size(W, 1)
    size(S) == (n, n) || throw(DimensionMismatch("S is $(size(S)), W is $(size(W))"))
    k >= 1 || throw(ArgumentError("k must be at least 1"))
    E = spatial_eigenvectors(W; n_candidates = min(k, n - 1))
    SE = S * E.vectors
    π = vec(sqrt.(sum(abs2, SE; dims = 1)))
    return SpectralOverlapResult{T}(π, mean(π), E.values, E.moran, length(π))
end

# ═══════════════════════════════════════════════════════════════════════
# Su & Jin (2010): profile QML
# ═══════════════════════════════════════════════════════════════════════

"""
    PartiallyLinearSARModel

A fit of `y = ρWy + Xβ + m(z) + ε` by [`fit_partially_linear_sar`](@ref).

# Fields

- `coef` — `[β; ρ]`; `coefnames` names them.
- `vcov` — their covariance matrix (see the variance note on
  [`fit_partially_linear_sar`](@ref)).
- `rho`, `beta`, `sigma2`, `loglik`.
- `m_hat::Vector` — the estimated `m(z_i)` at each observation.
- `residuals::Vector` — `y − ρ̂Wy − Xβ̂ − m̂`.
- `bandwidth` — in standard-deviation units of `z`; `bandwidth_rule` says
  how it was chosen (`:rule_of_thumb`, `:lscv` or `:user`).
- `order` — the local-polynomial order.
- `n_widened` — how many observations' kernel windows were widened because
  they held too few points at the chosen bandwidth (see
  [`local_polynomial_smoother`](@ref)). A large share means the bandwidth is
  too small for these data.
- `overlap::SpectralOverlapResult` — the identification diagnostic, computed
  on every fit.
- `S::Matrix` — the smoother, kept for [`closed_form_rho`](@ref) and for
  [`spectral_overlap`](@ref) at another `k`.
"""
struct PartiallyLinearSARModel{T<:AbstractFloat, M<:AbstractSparseMatrix{T}} <:
       StatsAPI.StatisticalModel
    coef::Vector{T}
    vcov::Matrix{T}
    coefnames::Vector{String}
    rho::T
    beta::Vector{T}
    sigma2::T
    loglik::T
    m_hat::Vector{T}
    residuals::Vector{T}
    y::Vector{T}
    X::Matrix{T}
    W::M
    S::Matrix{T}
    bandwidth::T
    bandwidth_rule::Symbol
    order::Int
    n_widened::Int
    overlap::SpectralOverlapResult{T}
end

StatsAPI.coef(m::PartiallyLinearSARModel) = m.coef
StatsAPI.vcov(m::PartiallyLinearSARModel) = m.vcov
StatsAPI.stderror(m::PartiallyLinearSARModel) = sqrt.(max.(diag(m.vcov), 0))
StatsAPI.coefnames(m::PartiallyLinearSARModel) = m.coefnames
StatsAPI.loglikelihood(m::PartiallyLinearSARModel) = m.loglik
StatsAPI.nobs(m::PartiallyLinearSARModel) = length(m.y)
StatsAPI.residuals(m::PartiallyLinearSARModel) = m.residuals
StatsAPI.fitted(m::PartiallyLinearSARModel) = m.y .- m.residuals

function Base.show(io::IO, m::PartiallyLinearSARModel)
    println(io, "═══════════════════════════════════════════════════════════════════════")
    println(io, "Partially linear SAR — profile QML (Su & Jin 2010)")
    println(io, "───────────────────────────────────────────────────────────────────────")
    _print_coef_table(io, m.coefnames, m.coef, stderror(m))
    println(io, "───────────────────────────────────────────────────────────────────────")
    @printf(io, "σ² = %.4f    Log-lik = %.2f    N = %d\n", m.sigma2, m.loglik, nobs(m))
    @printf(io, "local polynomial order %d, bandwidth %.4f sd (%s), %d window(s) widened\n",
            m.order, m.bandwidth, String(m.bandwidth_rule), m.n_widened)
    @printf(io, "spectral overlap π̄ = %.4f over %d modes — a measure, not a test;\n",
            m.overlap.pibar, m.overlap.k)
    println(io, "ρ is identified only given W, and is not thereby a causal spillover.")
    println(io, "═══════════════════════════════════════════════════════════════════════")
end

# The (I − S)-residualised pieces every estimator here works from.
struct _FirstStage{T}
    Xt::Matrix{T}          # (I − S)X
    F                       # QR of Xt, or nothing when X has no columns
    e0::Vector{T}          # M(I − S)y
    e1::Vector{T}          # M(I − S)Wy
    yt::Vector{T}          # (I − S)y
    wyt::Vector{T}         # (I − S)Wy
end

_resid(F, Xt, v) = F === nothing ? copy(v) : v .- Xt * (F \ v)

function _first_stage(S::Matrix{T}, X::Matrix{T}, y::Vector{T},
                      Wy::Vector{T}) where {T}
    IS = I - S
    Xt = IS * X
    F = size(X, 2) == 0 ? nothing : qr(Xt)
    yt = IS * y
    wyt = IS * Wy
    return _FirstStage{T}(Xt, F, _resid(F, Xt, yt), _resid(F, Xt, wyt), yt, wyt)
end

function _profile(fs::_FirstStage{T}, W, method::LogDetMethod,
                  rho_bounds) where {T}
    n = length(fs.e0)
    a00 = dot(fs.e0, fs.e0); a01 = dot(fs.e0, fs.e1); a11 = dot(fs.e1, fs.e1)
    if method isa ChebyshevApprox
        τ = precompute_chebyshev_traces(W, method)
        logdet_fn = ρ -> chebyshev_logdet(ρ, τ)
    else
        Isp = sparse(one(T) * I, n, n)
        logdet_fn = ρ -> first(logabsdet(lu(Isp - ρ * W)))
    end
    c = -T(0.5) * n * (log(T(2π)) + one(T))
    ll(ρ) = c + logdet_fn(ρ) - T(0.5) * n * log((a00 - 2ρ * a01 + ρ^2 * a11) / n)
    ρ̂, llmax = brent_maximize(ll, T(rho_bounds[1]), T(rho_bounds[2]))
    return ρ̂, llmax
end

_beta(fs::_FirstStage{T}, ρ) where {T} =
    fs.F === nothing ? T[] : fs.F \ (fs.yt .- ρ .* fs.wyt)

"""
    _lscv(r, Zs, order, h0)

Least-squares cross-validation of a local-polynomial regression of `r` on
`z`: the leave-one-out residual of a linear smoother is
`(r_i − (Sr)_i) / (1 − S_ii)`. Searched on a log grid of 21 points from
`h0/2` to `4h0`; bandwidths whose windows are too small are skipped.
"""
function _lscv(r::Vector{T}, Zs::Matrix{T}, order::Int, h0::T) where {T}
    best_h = h0; best = T(Inf)
    for h in h0 .* exp.(range(log(T(0.5)), log(T(4)), length = 21))
        S = try
            first(_smoother(Zs, h, order))
        catch err
            err isa ArgumentError || rethrow()
            continue
        end
        d = one(T) .- diag(S)
        any(<=(sqrt(eps(T))), d) && continue
        cv = mean(abs2, (r .- S * r) ./ d)
        if cv < best
            best = cv; best_h = h
        end
    end
    isfinite(best) || throw(ArgumentError(
        "no bandwidth on the cross-validation grid gives a well-posed smoother"))
    return best_h
end

"""
    fit_partially_linear_sar(y, X, z, W; bandwidth = :rule_of_thumb, order = 1,
                             method = ChebyshevApprox(order = 6),
                             rho_bounds = (-0.995, 0.995), overlap_k = 25,
                             varnames = nothing, max_exact_n = 1000,
                             vcov_probes = 250, rng = Random.default_rng())

Su & Jin's (2010) profile quasi-maximum likelihood estimator of

```math
y = ρWy + Xβ + m(z) + ε,
```

with `m` unknown and `ε` i.i.d.

For fixed `ρ`, `m` is estimated by a local-polynomial regression of
`(I − ρW)y − Xβ` on `z` ([`local_polynomial_smoother`](@ref)). The smoother
`S` does not depend on `ρ`, so `β(ρ)` is least squares of `(I − S)(I − ρW)y`
on `(I − S)X`, `σ²(ρ)` is a quadratic in `ρ`, and `ρ̂` maximises

```math
\\ell(ρ) = -\\tfrac{n}{2}\\log σ^2(ρ) + \\log|I - ρW| + \\text{const}.
```

The log-determinant is kept: `method` chooses [`ChebyshevApprox`](@ref) (the
default, as in `fit_spatial`) or [`ExactLU`](@ref). Then
`m̂ = S((I − ρ̂W)y − Xβ̂)`.

# Arguments

- `X` must **not** contain an intercept or any function of `z` alone: the
  smoother reproduces constants and absorbs both, and the design check then
  refuses the fit as rank deficient. Pass `zeros(n, 0)` for no `X`.
- `z`: `n×q` (or a vector), standardised internally.
- `bandwidth`: a number (in standard deviations of `z`), `:rule_of_thumb`
  for Su & Jin's `h = n^{−1/(q+4)}` (§6), or `:lscv` for their two-step
  recommendation: fit at the rule of thumb, then choose `h` by
  least-squares cross-validation of `y − Xβ̃ − ρ̃Wy` on `z` and refit.
- `overlap_k`: modes for the [`spectral_overlap`](@ref) diagnostic, which
  is computed on every fit.

# Variance

`vcov` is the inverse of Su & Jin's information matrix (their Theorem 4.3):
Lee's (2004) SAR information with `P = (I − S)'(I − S)` weighting the `X`
and `R = G(Xβ + m)` blocks, `G = W(I − ρW)⁻¹`. Their full sandwich adds an
`Ω` built from the third and fourth moments of `ε`; `Ω` is zero under
normality and is **not** included, so the standard errors assume normal
errors. Their coverage under non-normal errors is measured, not assumed:
see `docs/AUDIT.md`.

# Assumptions that matter

Errors must be **homoskedastic** (their Assumption 2: under
heteroskedasticity the QMLE is generally inconsistent). Rates are `√n`
when each unit has a bounded number of neighbours. Identification
(Assumption 7) needs `GXβ`, after `(I − S)` and the projection off `X`, to
stay non-degenerate — exactly what a smoother in the coordinates threatens.
Check `m.overlap`.

Returns a [`PartiallyLinearSARModel`](@ref).

# Reference

Su, L., & Jin, S. (2010). Profile quasi-maximum likelihood estimation of
partially linear spatial autoregressive models. *Journal of
Econometrics*, 157(1), 18–33.
"""
function fit_partially_linear_sar(
    y::AbstractVector{T},
    X::AbstractMatrix{T},
    z::AbstractVecOrMat{<:Real},
    W::AbstractSparseMatrix{T};
    bandwidth::Union{Real, Symbol} = :rule_of_thumb,
    order::Int = 1,
    method::LogDetMethod = ChebyshevApprox(order = 6, n_probes = 0),
    rho_bounds::Tuple{<:Real, <:Real} = (-0.995, 0.995),
    overlap_k::Int = 25,
    varnames::Union{Nothing, Vector{String}} = nothing,
    max_exact_n::Int = 1000,
    vcov_probes::Int = 250,
    rng::AbstractRNG = Random.default_rng(),
) where {T<:AbstractFloat}

    n, K = size(X)
    length(y) == n || throw(DimensionMismatch("y has $(length(y)) rows, X has $n"))
    size(W) == (n, n) || throw(DimensionMismatch("W is $(size(W)), expected ($n, $n)"))
    Z = z isa AbstractVector ? reshape(T.(z), :, 1) : T.(z)
    size(Z, 1) == n || throw(DimensionMismatch("z has $(size(Z, 1)) rows, X has $n"))
    q = size(Z, 2)
    _check_smoother_args(q, order)
    vnames = isnothing(varnames) ? ["x$i" for i in 1:K] : varnames
    length(vnames) == K || throw(ArgumentError("varnames has $(length(vnames)) names for $K columns"))

    Zs, _ = _standardise(Z)
    yv = Vector{T}(y); Xm = Matrix{T}(X)
    Wy = W * yv

    h0 = T(n)^(-one(T) / (q + 4))
    if bandwidth isa Symbol
        bandwidth in (:rule_of_thumb, :lscv) || throw(ArgumentError(
            "bandwidth must be a number, :rule_of_thumb or :lscv, got :$bandwidth"))
        h = h0
    else
        h = T(bandwidth)
        h > 0 || throw(ArgumentError("bandwidth must be positive, got $bandwidth"))
    end
    rule = bandwidth isa Symbol ? bandwidth : :user

    S, n_wid = _smoother(Zs, h, order)
    fs = _first_stage(S, Xm, yv, Wy)
    K > 0 && check_design_rank(fs.Xt, vnames, "fit_partially_linear_sar";
        context = "After (I - S) an intercept, or any column that is a " *
                  "smooth function of z, becomes zero: m(z) absorbs it.")
    ρ̂, llmax = _profile(fs, W, method, rho_bounds)

    if rule === :lscv
        β̃ = _beta(fs, ρ̂)
        r = yv .- ρ̂ .* Wy .- (K > 0 ? Xm * β̃ : zero(yv))
        h = _lscv(r, Zs, order, h0)
        S, n_wid = _smoother(Zs, h, order)
        fs = _first_stage(S, Xm, yv, Wy)
        ρ̂, llmax = _profile(fs, W, method, rho_bounds)
    end

    β̂ = _beta(fs, ρ̂)
    u = fs.e0 .- ρ̂ .* fs.e1                  # = (I − S)(y − ρ̂Wy − Xβ̂)
    σ2 = dot(u, u) / n
    lin = yv .- ρ̂ .* Wy .- (K > 0 ? Xm * β̂ : zero(yv))
    m̂ = S * lin
    res = lin .- m̂

    V = _plsar_vcov(fs.Xt, S, W, Xm, β̂, m̂, ρ̂, σ2;
                    max_exact_n = max_exact_n, n_probes = vcov_probes, rng = rng)
    ov = spectral_overlap(S, W; k = overlap_k)

    return PartiallyLinearSARModel{T, typeof(W)}(
        vcat(β̂, ρ̂), V, vcat(vnames, "rho"), ρ̂, β̂, σ2, llmax, m̂, res, yv,
        Xm, W, S, h, rule, order, n_wid, ov)
end

# Su & Jin's information (Theorem 4.3) under normality: Lee's (2004) SAR
# information with P = (I − S)'(I − S) in the X and R blocks.
function _plsar_vcov(Xt::Matrix{T}, S::Matrix{T}, W, X::Matrix{T},
                     β::Vector{T}, m̂::Vector{T}, ρ::T, σ2::T;
                     max_exact_n, n_probes, rng) where {T}
    n, K = size(X)
    A = lu(sparse(one(T) * I, n, n) - ρ * W)
    R = W * (A \ ((K > 0 ? X * β : zeros(T, n)) .+ m̂))
    Rt = R .- S * R                           # (I − S)R
    trG, trG2, trGtG = g_traces(W, ρ; max_exact_n = max_exact_n,
                                n_probes = n_probes, rng = rng)
    Info = zeros(T, K + 2, K + 2)
    Info[1:K, 1:K] .= (Xt' * Xt) ./ σ2
    Info[1:K, K + 1] .= (Xt' * Rt) ./ σ2
    Info[K + 1, 1:K] .= Info[1:K, K + 1]
    Info[K + 1, K + 1] = dot(Rt, Rt) / σ2 + trG2 + trGtG
    Info[K + 1, K + 2] = Info[K + 2, K + 1] = trG / σ2
    Info[K + 2, K + 2] = n / (2σ2^2)
    return Matrix(inv(Symmetric(Info))[1:(K + 1), 1:(K + 1)])
end

# ═══════════════════════════════════════════════════════════════════════
# Smirnov (2022) / Geniaux (2026): closed-form ρ
# ═══════════════════════════════════════════════════════════════════════

"""
    ClosedFormRhoResult

The closed-form `ρ` of [`closed_form_rho`](@ref).

# Fields

- `rho`, `beta` (least squares at `ρ̂` on the residualised design),
  `sigma2`.
- `se_classical` — Smirnov's `√(σ̂²/(b − 2ρ̂a))`. Not
  heteroskedasticity-robust, and **conservative**: Geniaux (2026,
  Proposition 2 and Appendix C.6) reports it over-states the sampling
  variance, increasingly as `ρ` grows.
- `se_hc0` — `√(Σmᵢ²)/d` with `mᵢ = (e₀ − ρ̂e₁)ᵢ(e₁ − ρ̂e₂)ᵢ`,
  `d = b − 2ρ̂a` (Geniaux §2.5). Ignores the correlation between
  neighbouring `mᵢ` and **under-covers** in his check (0.90–0.92).
- `se_shac` — adds `Σ_{i≠j} mᵢmⱼ` over pairs within graph distance two in
  `W`; close to nominal in his check.
- `a`, `b`, `c` — the quadratic `aρ² − bρ + c = 0` that was solved.
- `smoothed::Bool` — whether the first stage carried a smoother. When it
  did, see the warning on [`closed_form_rho`](@ref).
"""
struct ClosedFormRhoResult{T<:AbstractFloat}
    rho::T
    beta::Vector{T}
    sigma2::T
    se_classical::T
    se_hc0::T
    se_shac::T
    a::T
    b::T
    c::T
    smoothed::Bool
end

function Base.show(io::IO, r::ClosedFormRhoResult)
    println(io, "Closed-form ρ (Smirnov 2022; Geniaux 2026)")
    println(io, "──────────────────────────────────────────")
    @printf(io, "ρ̂ = %.4f    σ̂² = %.4f\n", r.rho, r.sigma2)
    @printf(io, "SE: classical %.4f (conservative)   HC0 %.4f   SHAC %.4f\n",
            r.se_classical, r.se_hc0, r.se_shac)
    r.smoothed && println(io,
        "First stage is a kernel smoother: no consistency proof covers this case.")
end

"""
    closed_form_rho(m::PartiallyLinearSARModel)
    closed_form_rho(y, X, W)

Smirnov's (2022) closed-form estimator of `ρ`, which needs **no
log-determinant**.

With `e₀`, `e₁`, `e₂` the residuals of `y`, `Wy` and `W²y` on the first
stage, `ρ̂` is the smaller root of the quadratic quasi-score

```math
aρ^2 - bρ + c = 0,\\quad a = e_1'e_2,\\; b = e_0'e_2 + e_1'e_1,\\; c = e_0'e_1,
```

`ρ̂ = (b − √(b² − 4ac)) / 2a` (Smirnov 2022, eqs 11–16). The quasi-score
keeps the first term of the power-series expansion of the ML score and
sets the rest to their expectation, zero — which uses `tr W = 0`, so a `W`
with a non-zero diagonal is refused.

The first method uses the partially linear model's own first stage,
`eₖ = M(I − S)Wᵏy` with `M` the projection off `(I − S)X`: the same
residuals that give its `σ̂²(ρ) = ‖e₀ − ρe₁‖²/n`. The second is Smirnov's
linear estimator, with the first stage least squares on `X` (which should
then include the intercept).

!!! warning "With a smoother, this estimator is unproven"
    Smirnov gives no consistency proof. Geniaux (2026) proves consistency
    and asymptotic normality for a **fixed** spline basis, where the
    smoother collapses onto a projection (his Lemma 1). A kernel smoother
    with a shrinking bandwidth is the sieve regime of his Remark 6, where
    rate conditions return and nothing is proved. Its coverage is measured
    in `docs/AUDIT.md`; read that before relying on any of its standard
    errors.

Returns a [`ClosedFormRhoResult`](@ref).
"""
function closed_form_rho(m::PartiallyLinearSARModel{T}) where {T}
    Wy = m.W * m.y
    fs = _first_stage(m.S, m.X, m.y, Wy)
    W2yt = (I - m.S) * (m.W * Wy)
    e2 = _resid(fs.F, fs.Xt, W2yt)
    return _cfe(fs, e2, m.W, true)
end

function closed_form_rho(y::AbstractVector{T}, X::AbstractMatrix{T},
                         W::AbstractSparseMatrix{T}) where {T<:AbstractFloat}
    n = length(y)
    size(X, 1) == n || throw(DimensionMismatch("y has $n rows, X has $(size(X, 1))"))
    Xm = Matrix{T}(X)
    yv = Vector{T}(y)
    Wy = W * yv
    F = qr(Xm)
    fs = _FirstStage{T}(Xm, F, _resid(F, Xm, yv), _resid(F, Xm, Wy), yv, Wy)
    e2 = _resid(F, Xm, W * Wy)
    return _cfe(fs, e2, W, false)
end

function _cfe(fs::_FirstStage{T}, e2::Vector{T}, W, smoothed::Bool) where {T}
    any(!iszero, diag(W)) && throw(ArgumentError(
        "closed_form_rho needs tr W = 0: the quasi-score drops terms whose " *
        "expectation is zero only when W has a zero diagonal"))
    e0, e1 = fs.e0, fs.e1
    n = length(e0)
    a = dot(e1, e2); b = dot(e0, e2) + dot(e1, e1); c = dot(e0, e1)
    disc = b^2 - 4a * c
    disc >= 0 || throw(ArgumentError(
        "the closed-form quadratic has no real root (b² − 4ac < 0)"))
    ρ = iszero(a) ? c / b : (b - sqrt(disc)) / (2a)
    u = e0 .- ρ .* e1
    σ2 = dot(u, u) / n
    d = b - 2ρ * a
    d > 0 || throw(ArgumentError(
        "the quasi-score has non-positive slope at the root (b − 2ρa ≤ 0)"))
    mi = u .* (e1 .- ρ .* e2)
    s_hc0 = sum(abs2, mi)
    # Pairs within graph distance two: the off-diagonal pattern of
    # (A + A²) with A the symmetrised adjacency of W.
    A = dropzeros(sparse(W + W'))
    fill!(nonzeros(A), one(T))
    A2 = A + A * A
    s_shac = s_hc0
    rows = rowvals(A2)
    for j in 1:n, p in nzrange(A2, j)
        i = rows[p]
        i == j && continue
        s_shac += mi[i] * mi[j]
    end
    β = _beta(fs, ρ)
    return ClosedFormRhoResult{T}(ρ, β, σ2, sqrt(σ2 / d), sqrt(s_hc0) / d,
                                  sqrt(max(s_shac, zero(T))) / d,
                                  a, b, c, smoothed)
end

# ═══════════════════════════════════════════════════════════════════════
# Liang, Gao & Gong (2022): time-varying coefficients, local-linear QML
# ═══════════════════════════════════════════════════════════════════════

# Epanechnikov kernel on [-1, 1], the one LGG use (§5). Constant dropped.
@inline _epan1(u::T) where {T} = abs(u) < one(T) ? one(T) - u * u : zero(T)

"""
    _TVSmoother

The local-linear smoother in rescaled time, held as an operator rather
than an `NT×NT` matrix. For a stacked vector `v` (N units within each of
T periods), `(Sv)_s = X_s β̂_v(τ_s)` with

    β̂_v(τ) = (I_d, 0) [Σ_t K_t M_t'M_t]⁻¹ Σ_t K_t M_t'v_t,
    M_t = [X_t, u_t X_t],  u_t = (t/T − τ)/h,  K_t = K(u_t),

LGG eq. (2.7). Only `X_t'X_t` and `X_t'v_t` enter, so applying it costs
`O(T²d²)` per column plus one pass over the data.
"""
struct _TVSmoother{T}
    X::Vector{Matrix{T}}        # X_t, N×d, t = 1..T
    K::Matrix{T}                # K[s, t] = K((t − s)/(T h))
    U::Matrix{T}                # U[s, t] = (t − s)/(T h)
    Ainv::Vector{Matrix{T}}     # inverse local Gram at τ_s, 2d×2d
    N::Int
    Tp::Int
    d::Int
end

function _tv_smoother(Xs::Vector{Matrix{T}}, h::T) where {T}
    Tp = length(Xs); N, d = size(Xs[1])
    XtX = [X' * X for X in Xs]
    K = zeros(T, Tp, Tp); U = zeros(T, Tp, Tp)
    for s in 1:Tp, t in 1:Tp
        u = T(t - s) / (Tp * h)
        U[s, t] = u; K[s, t] = _epan1(u)
    end
    Ainv = Vector{Matrix{T}}(undef, Tp)
    for s in 1:Tp
        A = zeros(T, 2d, 2d)
        for t in 1:Tp
            k = K[s, t]; iszero(k) && continue
            u = U[s, t]
            A[1:d, 1:d] .+= k .* XtX[t]
            A[1:d, d+1:2d] .+= (k * u) .* XtX[t]
            A[d+1:2d, d+1:2d] .+= (k * u * u) .* XtX[t]
        end
        A[d+1:2d, 1:d] .= A[1:d, d+1:2d]'
        # Scale-invariant conditioning check: equilibrate before judging.
        δ = sqrt.(max.(diag(A), floatmin(T)))
        Ae = A ./ (δ * δ')
        F = cholesky(Symmetric(Ae); check = false)
        (issuccess(F) && cond(F.U)^2 < T(1e12)) || throw(ArgumentError(
            "bandwidth h = $(round(h; sigdigits=4)): the local-linear design at " *
            "τ = $(round(s / Tp; digits=3)) is singular or nearly so; use a " *
            "larger bandwidth"))
        Ainv[s] = (inv(F) ./ (δ * δ'))
    end
    return _TVSmoother{T}(Xs, K, U, Ainv, N, Tp, d)
end

# β̂_v(τ_s) for every s, for each column of V (NT×m): returns d×m×T.
function _tv_beta(Sm::_TVSmoother{T}, V::AbstractMatrix{T}) where {T}
    N, Tp, d = Sm.N, Sm.Tp, Sm.d
    m = size(V, 2)
    C = [Sm.X[t]' * V[(t-1)*N+1:t*N, :] for t in 1:Tp]     # d×m each
    B = zeros(T, d, m, Tp)
    b = zeros(T, 2d, m)
    for s in 1:Tp
        fill!(b, zero(T))
        for t in 1:Tp
            k = Sm.K[s, t]; iszero(k) && continue
            b[1:d, :] .+= k .* C[t]
            b[d+1:2d, :] .+= (k * Sm.U[s, t]) .* C[t]
        end
        B[:, :, s] .= (Sm.Ainv[s] * b)[1:d, :]
    end
    return B
end

function _tv_apply(Sm::_TVSmoother{T}, V::AbstractMatrix{T}) where {T}
    B = _tv_beta(Sm, V)
    N = Sm.N
    out = similar(V)
    for s in 1:Sm.Tp
        out[(s-1)*N+1:s*N, :] .= Sm.X[s] * B[:, :, s]
    end
    return out
end
_tv_apply(Sm::_TVSmoother{T}, v::AbstractVector{T}) where {T} =
    vec(_tv_apply(Sm, reshape(v, :, 1)))

"""
    TVCoefSARModel

A fit of `Y_t = ρWY_t + X_tβ(t/T) + α + e_t` by [`fit_tv_coef_sar`](@ref).

# Fields

- `rho`, `rho_se`, `sigma2`, `loglik`.
- `beta::Matrix` — `T×d`: row `s` is `β̂(s/T)`.
- `beta_se::Matrix` — pointwise standard errors of `β̂(s/T)`, conditional
  on `ρ̂` and `α̂`: `σ̂²ΨΨ'` for the linear smoother `Ψ`. They **exclude the
  smoothing bias**, of order `h²β''(τ)` (LGG Theorem 3), so a band built
  from them covers `E β̂(τ)`, not `β(τ)`, unless `h` undersmooths.
- `tau::Vector` — the rescaled times `s/T`.
- `alpha::Vector` — fixed effects, normalised to sum to zero (LGG §2.1).
- `bandwidth` (in units of rescaled time), `bandwidth_rule`.
- `residuals`, `N`, `T`, `coefnames` (the columns of `X`).
"""
struct TVCoefSARModel{T<:AbstractFloat} <: StatsAPI.StatisticalModel
    rho::T
    rho_se::T
    sigma2::T
    loglik::T
    beta::Matrix{T}
    beta_se::Matrix{T}
    tau::Vector{T}
    alpha::Vector{T}
    residuals::Vector{T}
    bandwidth::T
    bandwidth_rule::Symbol
    N::Int
    T::Int
    coefnames::Vector{String}
end

StatsAPI.coef(m::TVCoefSARModel) = [m.rho]
StatsAPI.coefnames(m::TVCoefSARModel) = ["rho"]
StatsAPI.vcov(m::TVCoefSARModel) = fill(m.rho_se^2, 1, 1)
StatsAPI.stderror(m::TVCoefSARModel) = [m.rho_se]
StatsAPI.loglikelihood(m::TVCoefSARModel) = m.loglik
StatsAPI.nobs(m::TVCoefSARModel) = m.N * m.T
StatsAPI.residuals(m::TVCoefSARModel) = m.residuals

function Base.show(io::IO, m::TVCoefSARModel)
    println(io, "═══════════════════════════════════════════════════════════════════════")
    println(io, "SAR panel with time-varying coefficients — LLQML (Liang, Gao & Gong 2022)")
    println(io, "───────────────────────────────────────────────────────────────────────")
    _print_coef_table(io, ["rho"], [m.rho], [m.rho_se])
    println(io, "───────────────────────────────────────────────────────────────────────")
    @printf(io, "σ² = %.4f    Log-lik = %.2f    N = %d    T = %d\n",
            m.sigma2, m.loglik, m.N, m.T)
    @printf(io, "bandwidth %.4f in rescaled time (%s)\n", m.bandwidth, String(m.bandwidth_rule))
    println(io, "β(τ) at τ = first, middle, last period:")
    for (k, nm) in enumerate(m.coefnames)
        mid = cld(m.T, 2)
        @printf(io, "  %-12s %9.4f %9.4f %9.4f\n", nm,
                m.beta[1, k], m.beta[mid, k], m.beta[end, k])
    end
    println(io, "═══════════════════════════════════════════════════════════════════════")
end

# The concentrated fit at one bandwidth.
function _tv_fit(Xs, y::Vector{T}, Wy::Vector{T}, N, Tp, h::T, W,
                 method::LogDetMethod, rho_bounds) where {T}
    Sm = _tv_smoother(Xs, h)
    D0 = vcat(-ones(T, 1, N - 1), Matrix{T}(I, N - 1, N - 1))   # N×(N−1), LGG D₀
    D = repeat(D0, Tp)
    Dt = D .- _tv_apply(Sm, D)
    F = qr(Dt)
    yt = y .- _tv_apply(Sm, y)
    wyt = Wy .- _tv_apply(Sm, Wy)
    e0 = yt .- Dt * (F \ yt); e1 = wyt .- Dt * (F \ wyt)
    NT = N * Tp
    a00 = dot(e0, e0); a01 = dot(e0, e1); a11 = dot(e1, e1)
    if method isa ChebyshevApprox
        τ = precompute_chebyshev_traces(W, method)
        ld = ρ -> chebyshev_logdet(ρ, τ)
    else
        Isp = sparse(one(T) * I, N, N)
        ld = ρ -> first(logabsdet(lu(Isp - ρ * W)))
    end
    c = -T(0.5) * NT * (log(T(2π)) + one(T))
    ll(ρ) = c + Tp * ld(ρ) - T(0.5) * NT * log((a00 - 2ρ * a01 + ρ^2 * a11) / NT)
    ρ̂, llmax = brent_maximize(ll, T(rho_bounds[1]), T(rho_bounds[2]))
    α = F \ (yt .- ρ̂ .* wyt)
    return (; Sm, D, Dt, F, e0, e1, ρ̂, llmax, α)
end

# LGG Appendix A.3: leave-one-unit-out cross-validation. With pilot ρ̃ and
# α̃ (from the fit at a pilot bandwidth), β̂^(−i)(τ) is the local-linear fit
# of Z − Dα̃ on X over every unit but i; the loss is ‖M(Z − B)‖², with M
# removing each unit's time mean, so the fixed effects do not enter it.
function _tv_cv(Xs, z::Vector{T}, N, Tp, grid) where {T}
    d = size(Xs[1], 2)
    best = T(Inf); best_h = grid[1]
    Zm = reshape(z, N, Tp)
    for h in grid
        Sm = try _tv_smoother(Xs, h) catch err
            err isa ArgumentError || rethrow(); continue
        end
        # Per-unit Gram and cross-product contributions, for removal.
        loss = zero(T); ok = true
        pred = zeros(T, N, Tp)
        for s in 1:Tp
            A = zeros(T, 2d, 2d); bb = zeros(T, 2d)
            Ai = [zeros(T, 2d, 2d) for _ in 1:N]; bi = [zeros(T, 2d) for _ in 1:N]
            for t in 1:Tp
                k = Sm.K[s, t]; iszero(k) && continue
                u = Sm.U[s, t]
                X = Xs[t]
                for i in 1:N
                    x = X[i, :]
                    g = vcat(x, u .* x)
                    Ai[i] .+= k .* (g * g'); bi[i] .+= (k * Zm[i, t]) .* g
                end
            end
            for i in 1:N; A .+= Ai[i]; bb .+= bi[i]; end
            for i in 1:N
                Fi = cholesky(Symmetric(A .- Ai[i]); check = false)
                issuccess(Fi) || (ok = false; break)
                β = (Fi \ (bb .- bi[i]))[1:d]
                pred[i, s] = dot(Xs[s][i, :], β)
            end
            ok || break
        end
        ok || continue
        r = Zm .- pred
        r .-= mean(r; dims = 2)
        loss = sum(abs2, r)
        if loss < best
            best = loss; best_h = h
        end
    end
    isfinite(best) || throw(ArgumentError(
        "no bandwidth on the cross-validation grid gives a well-posed smoother"))
    return best_h
end

"""
    fit_tv_coef_sar(y, X, W, N, T; bandwidth = :cv, method = ChebyshevApprox(order = 6),
                    rho_bounds = (-0.995, 0.995), varnames = nothing,
                    max_exact_n = 1000, vcov_probes = 250, rng = Random.default_rng())

Liang, Gao & Gong's (2022) local-linear concentrated QML for the spatial
panel with **time-varying coefficients** and individual fixed effects:

```math
Y_t = ρWY_t + X_tβ(t/T) + α + e_t,\\qquad t = 1,…,T.
```

`ρ` is a **constant**; only the regressor coefficients vary, as smooth
functions of rescaled time `τ = t/T`. A time-varying `ρ` is not this model
and is not offered: no published method for it exists (ROADMAP #13).

The data are stacked as elsewhere in the package's panels: `N` units within
each of `T` periods, so `y[(t−1)N + i]` is unit `i` at time `t`. Put a
column of ones in `X` for a time-varying intercept; the fixed effects are
then normalised to sum to zero, as in the paper (§2.1).

# Method (LGG §2.2)

1. For fixed `(ρ, α)`, `β(τ)` is the local-linear kernel regression of
   `(I − ρW)Y_t − α` on `X_t` around `τ` (Epanechnikov kernel, bandwidth
   `h` in units of rescaled time). The smoother `S` does not depend on
   `(ρ, α)`.
2. Partial `α` out of the `(I − S)`-residualised data, and maximise the
   concentrated log-likelihood, which keeps `T log|I − ρW|`.
3. Update `β̂(τ)` at `(ρ̂, α̂)`.

`bandwidth` is a number, or `:cv` for the paper's leave-one-unit-out
cross-validation (Appendix A.3) on a 15-point log grid from `2/T` to `1`,
with pilot `(ρ̃, α̃)` from a fit at `h = T^{−1/5}`. (The paper takes the
pilot `ρ̃` from the time-invariant Lee–Yu model; using the model's own fit
keeps the pilot inside the model being tuned.)

# Inference

`rho_se` is the inverse of LGG's `Σ₀` under normal errors (their
Proposition 1): `R'PR/σ² + T(tr G² + tr G'G)` against `σ²`, with
`P = (I − S)'Q(I − S)`. The `Ω` term carrying the third and fourth error
moments (Theorem 2) is omitted, so the standard error assumes normality.
Rates: `√(NT)` for `ρ`, `√(NTh)` for `β(τ)`, as **both** `N` and `T` grow
(Assumption 3: `Th → ∞`, `NTh⁸ → 0`). The errors must be serially and
cross-sectionally independent and homoskedastic (Assumption 2). See
[`TVCoefSARModel`](@ref) for what `beta_se` does and does not cover.

# Reference

Liang, X., Gao, J., & Gong, X. (2022). Semiparametric spatial
autoregressive panel data model with fixed effects and time-varying
coefficients. *Journal of Business & Economic Statistics*, 40(4),
1784–1802.
"""
function fit_tv_coef_sar(
    y::AbstractVector{T},
    X::AbstractMatrix{T},
    W::AbstractSparseMatrix{T},
    N::Int,
    T_per::Int;
    bandwidth::Union{Real, Symbol} = :cv,
    method::LogDetMethod = ChebyshevApprox(order = 6, n_probes = 0),
    rho_bounds::Tuple{<:Real, <:Real} = (-0.995, 0.995),
    varnames::Union{Nothing, Vector{String}} = nothing,
    max_exact_n::Int = 1000,
    vcov_probes::Int = 250,
    rng::AbstractRNG = Random.default_rng(),
) where {T<:AbstractFloat}

    NT, d = size(X)
    NT == N * T_per || throw(DimensionMismatch("X has $NT rows, N*T = $(N * T_per)"))
    length(y) == NT || throw(DimensionMismatch("y has $(length(y)) rows, expected $NT"))
    size(W) == (N, N) || throw(DimensionMismatch("W is $(size(W)), expected ($N, $N)"))
    d >= 1 || throw(ArgumentError("X needs at least one column"))
    N >= 2 && T_per >= 3 || throw(ArgumentError("need N ≥ 2 and T ≥ 3"))
    vnames = isnothing(varnames) ? ["x$i" for i in 1:d] : varnames
    length(vnames) == d || throw(ArgumentError("varnames has $(length(vnames)) names for $d columns"))

    yv = Vector{T}(y)
    Xs = [Matrix{T}(X[(t-1)*N+1:t*N, :]) for t in 1:T_per]
    Wy = vec(W * reshape(yv, N, T_per))

    if bandwidth isa Symbol
        bandwidth === :cv || throw(ArgumentError("bandwidth must be a number or :cv"))
        h0 = T(T_per)^(-T(1) / 5)
        pilot = _tv_fit(Xs, yv, Wy, N, T_per, h0, W, method, rho_bounds)
        z = yv .- pilot.ρ̂ .* Wy .- pilot.D * pilot.α
        grid = exp.(range(log(T(2) / T_per), log(one(T)), length = 15))
        h = _tv_cv(Xs, z, N, T_per, grid)
        rule = :cv
    else
        h = T(bandwidth)
        h > 0 || throw(ArgumentError("bandwidth must be positive, got $bandwidth"))
        rule = :user
    end

    f = _tv_fit(Xs, yv, Wy, N, T_per, h, W, method, rho_bounds)
    ρ̂ = f.ρ̂
    v = yv .- ρ̂ .* Wy .- f.D * f.α                  # Y(ρ̂) − Dα̂
    B = _tv_beta(f.Sm, reshape(v, :, 1))             # d×1×T
    β = permutedims(B[:, 1, :])                      # T×d
    μ = _tv_apply(f.Sm, v) .+ f.D * f.α              # X̃β̃ + Dα̂
    res = yv .- ρ̂ .* Wy .- μ
    u = f.e0 .- ρ̂ .* f.e1
    σ2 = dot(u, u) / NT

    # LGG Σ₀ under normality, in totals rather than per-NT averages.
    A = lu(sparse(one(T) * I, N, N) - ρ̂ * W)
    R = vec(W * (A \ reshape(μ, N, T_per)))
    Rt = R .- _tv_apply(f.Sm, R)
    Rt .-= f.Dt * (f.F \ Rt)
    trG, trG2, trGtG = g_traces(W, ρ̂; max_exact_n = max_exact_n,
                                n_probes = vcov_probes, rng = rng)
    Info = [dot(Rt, Rt) / σ2 + T_per * (trG2 + trGtG)   T_per * trG / σ2;
            T_per * trG / σ2                             NT / (2σ2^2)]
    ρse = sqrt(max(inv(Info)[1, 1], zero(T)))

    # Pointwise SEs of β̂(τ_s) = Ψ_s v, conditional on (ρ̂, α̂): σ²ΨΨ'.
    βse = zeros(T, T_per, d)
    for s in 1:T_per
        V = zeros(T, d, d)
        As = f.Sm.Ainv[s]
        for t in 1:T_per
            k = f.Sm.K[s, t]; iszero(k) && continue
            Xt = Xs[t]
            G = vcat(Xt', f.Sm.U[s, t] .* Xt')       # 2d×N
            P = (As * G)[1:d, :] .* k                 # rows of Ψ_s on period t
            V .+= P * P'
        end
        βse[s, :] .= sqrt.(max.(σ2 .* diag(V), zero(T)))
    end

    α_full = vcat(-sum(f.α), f.α)
    return TVCoefSARModel{T}(ρ̂, ρse, σ2, f.llmax, β, βse,
        T.((1:T_per) ./ T_per), α_full, res, h, rule, N, T_per, vnames)
end

end # module
