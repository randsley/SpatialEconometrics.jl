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
smoother) are proved in `research/trace_cfe/THEORY.md` (Theorem 1 and
Propositions 1, 2, 4 and 6). They have not been checked by anyone else.
They are measured in that directory and in `validation/mc_root_rho.jl` and
`validation/mc_root_rho_smoother.jl`.

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

import ..SpatialUtils: check_design_rank, _check_spatial_bounds, ExactLU
import ..SpatialDiscreteChoice: _norminv

export RootRhoResult, root_rho, tensor_pspline

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
    if isempty(r.cs)
        @printf(io, "Test-inversion %g%% set: empty (every ρ in [%.2f, %.2f] is rejected)\n",
                pct, r.rho_bounds...)
    else
        s = join((@sprintf("[%.4f, %.4f]", lo, hi) for (lo, hi) in r.cs), " ∪ ")
        @printf(io, "Test-inversion %g%% set:       %s\n", pct, s)
        r.cs_touches_bound && println(io,
            "  The set reaches a bound of rho_bounds: ρ is weakly identified.")
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

# The exact-diagonal moment matrix A = MW + MDM, (M∘M)d = −diag(MW), for a
# symmetric M. diag A = 0 exactly: diag(MDM) = (M∘M)d needs only that M is
# symmetric. The moment must be formed from A itself: the residual-vector
# shortcut (My₀)′(MWy₀ + DMy₀) is y₀′(M²W + MDM)y₀, which differs when M² ≠ M.
function _exact_diag_moment(M::Matrix{T}, W::AbstractMatrix{T}) where {T}
    MW = M * W
    d = (M .* M) \ (-diag(MW))
    MW .+= M * (d .* M)
    return MW
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
    length(u) == length(v) || throw(DimensionMismatch(
        "tensor_pspline: u has $(length(u)) entries, v has $(length(v))"))
    order >= 1 || throw(ArgumentError("tensor_pspline: order must be at least 1, got $order"))
    k >= order + 2 || throw(ArgumentError(
        "tensor_pspline: k = $k knots is too few for order $order; need k ≥ $(order + 2)"))
    Hu = _hat_basis(Vector{Float64}(u), Int(k)); Hv = _hat_basis(Vector{Float64}(v), Int(k))
    B = hcat([Hu[:, i] .* Hv[:, j] for i in 1:k for j in 1:k]...)
    D = Matrix{Float64}(I, k, k)
    for _ in 1:order
        D = D[2:end, :] .- D[1:(end - 1), :]
    end
    P = D' * D; Ik = Matrix{Float64}(I, k, k)
    return (; basis = B, penalty = kron(P, Ik) .+ kron(Ik, P))
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

# Arguments

- `X` — regressors whose `β` is reported (point estimates only). May have
  zero columns. Include the intercept here or in `basis`, not both: a hat
  or B-spline basis already sums to one.
- `W` — a spatial weights matrix with a zero diagonal, row-standardised so
  that `rho_bounds` lies inside `(−1, 1)`.
- `basis` — the `n × K_B` basis for `f`.
- `penalty`, `lambda` — a symmetric positive semi-definite `K_B × K_B`
  penalty `Ω` on the basis coefficients, and its weight `λ > 0`. Give both
  or neither. [`tensor_pspline`](@ref) returns a basis and a penalty.

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

# Choosing the basis

The interval is valid only if the approximation error of `f` is small
against the sampling noise (THEORY.md, Proposition 3: `β = o(√n)`). That
calls for a **larger** basis than a good fit of `f` would: GCV and the
RMSE-optimal size over-smooth for inference (`research/trace_cfe/`,
`tc_lambda.jl`). Report `ρ̂` across a few basis sizes. The coverage of both
intervals was measured for `K/n` up to 0.30; a larger `K` gives a warning.
Units with leverage above 0.99 make the correction ill-posed and are
refused.

# Cost

Everything is dense: about four `n × n` matrices, so `n` is capped at
`max_dense_n` (4000 by default, about 0.5 GB). The penalised smoother holds
about six and solves one dense `n × n` system for `d`.

!!! note "Proved, but not checked by anyone else"
    The results for a basis that grows with `n` (a CLT for the moment, the
    consistency of `V̂`, the validity of the test-inversion set) are in
    `research/trace_cfe/THEORY.md`. The published theory (Jin & Lee 2012)
    assumes a design of fixed dimension.
"""
function root_rho(y::AbstractVector{<:Real}, X::AbstractMatrix{<:Real}, W::AbstractMatrix{<:Real};
                  basis::Union{Nothing, AbstractMatrix{<:Real}} = nothing,
                  penalty::Union{Nothing, AbstractMatrix{<:Real}} = nothing,
                  lambda::Union{Nothing, Real} = nothing,
                  varnames = nothing, level::Real = 0.95,
                  rho_bounds = (-0.99, 0.99), max_dense_n::Integer = 4000)
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
    if penalised
        basis === nothing && throw(ArgumentError("root_rho: a penalty needs a basis"))
        isfinite(lambda) && lambda > 0 || throw(ArgumentError(
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
    penalised && return _root_rho_penalised(Vector{T}(y), Z, Ws, kx, Pm, T(lambda), vnames,
                                            T(level), bnds)
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

    yv = Vector{T}(y); Wy = Ws * yv
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
    vp = _vhat_poly(C, Q, Mii, yv, Wy)
    cs = _inversion_set(a, b, c, vp, z, bnds)
    touches = !isempty(cs) && (cs[1][1] == bnds[1] || cs[end][2] == bnds[2])

    if ρ === nothing
        nanT = T(NaN)
        return RootRhoResult{T}(nanT, ρ1 === nothing ? nanT : ρ1, nanT, (nanT, nanT), cs,
            touches, length(cs) > 1, T(level), fill(nanT, kx), vnames, n, K, :projection,
            zero(T), T(K), maximum(h), a, b, c, vp, nanT, false, bnds)
    end
    V = _polyval(vp, ρ)
    se = V > 0 ? sqrt(V) / abs(b - 2ρ * a) : T(NaN)
    β = kx > 0 ? (Z \ (yv .- ρ .* Wy))[1:kx] : T[]
    return RootRhoResult{T}(ρ, ρ1, se, (ρ - z * se, ρ + z * se), cs, touches,
        length(cs) > 1, T(level), β, vnames, n, K, :projection, zero(T), T(K), maximum(h),
        a, b, c, vp, V, true, bnds)
end

# The penalised smoother: S = Z(Z′Z + λΩ̃)⁻¹Z′, the exact-diagonal moment in
# one step, and the general HC2 V̂ (THEORY.md, Proposition 4).
function _root_rho_penalised(y::Vector{T}, Z::Matrix{T}, Ws::SparseMatrixCSC{T}, kx::Int,
                             Ω::Matrix{T}, λ::T, vnames::Vector{String}, level::T,
                             bnds::Tuple{T, T}) where {T}
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
    A = _exact_diag_moment(M, Ws)
    H = (A .+ A') * S
    Wy = Ws * y
    z = T(_norminv(1 - (1 - level) / 2))
    a, b, c = _quadratic(A, y, Wy)
    ρ = _sign_rule_root(a, b, c, bnds)
    vp = _vhat_poly_general(A, H, M, m, y, Wy)
    cs = _inversion_set(a, b, c, vp, z, bnds)
    touches = !isempty(cs) && (cs[1][1] == bnds[1] || cs[end][2] == bnds[2])
    nanT = T(NaN)
    if ρ === nothing
        return RootRhoResult{T}(nanT, nanT, nanT, (nanT, nanT), cs, touches, length(cs) > 1,
            level, fill(nanT, kx), vnames, n, K, :penalised, λ, df, maximum(Sii), a, b, c,
            vp, nanT, false, bnds)
    end
    V = _polyval(vp, ρ)
    se = V > 0 ? sqrt(V) / abs(b - 2ρ * a) : nanT
    β = kx > 0 ? (F \ (Z' * (y .- ρ .* Wy)))[1:kx] : T[]
    return RootRhoResult{T}(ρ, nanT, se, (ρ - z * se, ρ + z * se), cs, touches,
        length(cs) > 1, level, β, vnames, n, K, :penalised, λ, df, maximum(Sii), a, b, c,
        vp, V, true, bnds)
end

end # module
