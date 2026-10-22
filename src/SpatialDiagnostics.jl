module SpatialDiagnostics

using LinearAlgebra
using SparseArrays
using Statistics
using Random
using Printf

import ..SpatialUtils: _normcdf

export MoranResult, LISAResult, LMTestResult, GearyCResult,
       moran_global, moran_local, geary_c,
       lm_test_lag, lm_test_error, lm_test_robust_lag, lm_test_robust_error, lm_tests

# ─── Result types ─────────────────────────────────────────────────────

"""
    MoranResult

Global Moran's I with its reference moments, returned by
[`moran_global`](@ref).

# Fields
- `I_stat`: the statistic. Positive means like values cluster together,
  negative means neighbours tend to differ.
- `E_I`: expectation under no spatial autocorrelation, `-1/(N-1)` — note
  that it is slightly negative, not zero.
- `Var_I`: variance under the same null.
- `z_stat`: `(I_stat - E_I) / sqrt(Var_I)`.
- `p_value`: two-sided.
- `method`: `:asymptotic` or `:permutation`, recording how `p_value` was
  obtained.
"""
struct MoranResult{T<:AbstractFloat}
    I_stat::T
    E_I::T
    Var_I::T
    z_stat::T
    p_value::T
    method::Symbol
end

"""
    LISAResult

Local indicators of spatial association, returned by
[`moran_local`](@ref). All four vectors are length `N` and aligned with the
units of `y`.

# Fields
- `local_I`: the local Moran statistic per unit. These sum to the global
  statistic up to a scale factor.
- `pseudo_p`: conditional-permutation p-value per unit.
- `cluster_type`: one of `:HH`, `:LL`, `:HL`, `:LH` or `:ns` — note these
  are `Symbol`s, not strings. `:HH` and `:LL` mark clusters of high and of
  low values; `:HL` and `:LH` mark spatial outliers; `:ns` marks units with
  `pseudo_p > 0.05`.
- `z_scores`: standardized `y`, the basis of the H/L classification.

!!! warning
    Each unit is tested separately, so with many units some will be
    flagged by chance. Treat the map as exploratory unless the p-values are
    adjusted for multiplicity.
"""
struct LISAResult{T<:AbstractFloat}
    local_I::Vector{T}
    pseudo_p::Vector{T}
    cluster_type::Vector{Symbol}
    z_scores::Vector{T}
end

"""
    LMTestResult

One Lagrange-multiplier specification test, returned by
[`lm_test_lag`](@ref) and its siblings.

# Fields
- `stat`: the test statistic, asymptotically chi-squared with 1 degree of
  freedom under the null of no spatial dependence.
- `p_value`: upper-tail probability.
- `test_name`: which test produced it.
"""
struct LMTestResult{T<:AbstractFloat}
    stat::T
    p_value::T
    test_name::String
end

"""
    GearyCResult

Geary's c with its reference moments, returned by [`geary_c`](@ref).

# Fields
- `C_stat`: the statistic. Its null expectation is 1; **below** 1 indicates
  positive spatial autocorrelation and above 1 negative — the opposite
  direction to Moran's I.
- `E_C`: expectation under the null, 1.
- `Var_C`: variance under the null.
- `z_stat`: `(C_stat - E_C) / sqrt(Var_C)`, so a negative `z_stat` means
  positive autocorrelation.
- `p_value`: two-sided.
"""
struct GearyCResult{T<:AbstractFloat}
    C_stat::T
    E_C::T
    Var_C::T
    z_stat::T
    p_value::T
end

# ─── Global Moran's I ────────────────────────────────────────────────

"""
    moran_global(y, W; inference = :asymptotic, n_perm = 999, rng = Random.default_rng())

Global Moran's I, the standard test for spatial autocorrelation in `y`.

```math
I = (N/S_0) * (z' W z) / (z' z)
```

where `z` is `y` in deviations from its mean and `S_0` is the sum of all
weights. When `W` is row-standardized `S_0 = N` and `I` reduces to the
slope of `Wz` on `z`, which is why it can be read off a Moran scatter plot
(`plot_moran_scatter`).

# Keywords
- `inference`: `:asymptotic` uses the normal approximation with the exact
  finite-sample variance, correcting for the kurtosis of `y`.
  `:permutation` instead randomly reassigns `y` across locations `n_perm`
  times and reports the rank of the observed statistic, which makes no
  distributional assumption and is the safer choice for small `N` or
  heavy-tailed `y`.
- `n_perm`, `rng`: used only when `inference = :permutation`.

# Returns
A [`MoranResult`](@ref).

!!! note
    The null expectation is `-1/(N-1)`, not 0, so compare `I_stat` against
    `E_I` rather than against zero — the difference matters in small
    samples.

# Examples
```julia
W = make_rook_weights(10)
moran_global(y, W)
moran_global(y, W; inference = :permutation, n_perm = 9999)

m = fit_spatial(SAR(), W, X, y)
moran_global(residuals(m), W)      # did the model absorb the dependence?
```

See also [`moran_local`](@ref), [`geary_c`](@ref), [`lm_tests`](@ref).

# References
Anselin, L. (1988). *Spatial Econometrics: Methods and Models*. Kluwer.
"""
function moran_global(
    y::AbstractVector{T},
    W::AbstractSparseMatrix{T};
    inference::Symbol=:asymptotic,
    n_perm::Int=999,
    rng::AbstractRNG=Random.default_rng()
) where {T<:AbstractFloat}
    N = length(y)
    ȳ = mean(y)
    z = y .- ȳ
    Wz = W * z

    S0 = sum(W)
    zz = dot(z, z)
    I_stat = (T(N) / S0) * dot(z, Wz) / zz
    E_I = -one(T) / (N - 1)

    if inference === :permutation
        count_ge = 0
        z_perm = similar(z)
        for _ in 1:n_perm
            randperm!(rng, z_perm, z)
            Wz_p = W * z_perm
            I_p = (T(N) / S0) * dot(z_perm, Wz_p) / dot(z_perm, z_perm)
            if abs(I_p) >= abs(I_stat)
                count_ge += 1
            end
        end
        p_value = T(count_ge + 1) / T(n_perm + 1)
        z_score = (I_stat - E_I) / max(std([I_stat]), T(1e-12))
        return MoranResult{T}(I_stat, E_I, zero(T), z_score, p_value, :permutation)
    end

    # Asymptotic variance (Cliff & Ord)
    S1 = _trace_WpWt_sum(W)
    S2 = _col_row_sum_sq(W, N)

    b2 = _kurtosis_ratio(z, N)
    Var_I = (_var_moran_asymptotic(N, S0, S1, S2, b2) - E_I^2)
    Var_I = max(Var_I, T(1e-20))
    z_stat = (I_stat - E_I) / sqrt(Var_I)
    p_value = T(2.0) * (one(T) - T(_normcdf(abs(Float64(z_stat)))))

    return MoranResult{T}(I_stat, E_I, Var_I, z_stat, p_value, :asymptotic)
end

function randperm!(rng::AbstractRNG, dest::AbstractVector, src::AbstractVector)
    copyto!(dest, src)
    n = length(dest)
    @inbounds for i in n:-1:2
        j = rand(rng, 1:i)
        dest[i], dest[j] = dest[j], dest[i]
    end
    return dest
end

function _trace_WpWt_sum(W::AbstractSparseMatrix{T}) where {T}
    # S1 = (1/2) * sum_ij (w_ij + w_ji)^2
    # For row-standardized W this simplifies but we compute generally
    rows = SparseArrays.rowvals(W)
    vals = SparseArrays.nonzeros(W)
    N = size(W, 1)
    s1 = zero(T)
    for j in 1:N
        for idx in SparseArrays.nzrange(W, j)
            i = rows[idx]
            wij = vals[idx]
            wji = _sparse_getindex(W, j, i)
            s1 += (wij + wji)^2
        end
    end
    return s1 / 2
end

function _sparse_getindex(W::SparseArrays.SparseMatrixCSC{T}, i::Int, j::Int) where {T}
    rows = SparseArrays.rowvals(W)
    vals = SparseArrays.nonzeros(W)
    for idx in SparseArrays.nzrange(W, j)
        rows[idx] == i && return vals[idx]
    end
    return zero(T)
end

function _col_row_sum_sq(W::AbstractSparseMatrix{T}, N::Int) where {T}
    # S2 = sum_i (sum_j w_ij + sum_j w_ji)^2
    row_sums = zeros(T, N)
    col_sums = zeros(T, N)
    rows = SparseArrays.rowvals(W)
    vals = SparseArrays.nonzeros(W)
    for j in 1:N
        for idx in SparseArrays.nzrange(W, j)
            i = rows[idx]
            v = vals[idx]
            row_sums[i] += v
            col_sums[j] += v
        end
    end
    s2 = zero(T)
    @inbounds for i in 1:N
        s2 += (row_sums[i] + col_sums[i])^2
    end
    return s2
end

function _kurtosis_ratio(z::AbstractVector{T}, N::Int) where {T}
    m2 = dot(z, z) / N
    m4 = sum(z .^ 4) / N
    return m4 / (m2^2)
end

function _var_moran_asymptotic(N::Int, S0::T, S1::T, S2::T, b2::T) where {T}
    n = T(N)
    A = n * ((n^2 - 3n + 3) * S1 - n * S2 + 3 * S0^2)
    D = (n - 1) * (n - 2) * (n - 3) * S0^2
    B = b2 * ((n^2 - n) * S1 - 2n * S2 + 6 * S0^2)
    return (A - B) / D
end

# ─── Local Moran's I (LISA) ──────────────────────────────────────────

"""
    moran_local(y, W; n_perm = 999, rng = Random.default_rng())

Local Moran's I per unit, with conditional-permutation inference and LISA
cluster classification.

Where [`moran_global`](@ref) asks whether there is clustering anywhere,
this asks *where* it is. Inference is conditional: unit `i`'s own value is
held fixed while the remaining `N-1` values are permuted among the other
locations, which is the appropriate null for a local statistic.

Each unit is then labelled by the sign of its own standardized value and of
its spatial lag: `:HH` and `:LL` for clusters, `:HL` and `:LH` for spatial
outliers, and `:ns` where `pseudo_p > 0.05`.

# Returns
A [`LISAResult`](@ref).

!!! warning
    These are `N` simultaneous tests, so some units will be significant by
    chance alone; the `0.05` cut used for `:ns` is unadjusted. Read the
    classification as exploratory.

# Examples
```julia
lisa = moran_local(y, W; n_perm = 9999)
count(==(:HH), lisa.cluster_type)
plot_local_moran(lisa, coords)          # needs a Makie backend
```

# References
Anselin, L. (1995). Local indicators of spatial association — LISA.
*Geographical Analysis*, 27(2).
"""
function moran_local(
    y::AbstractVector{T},
    W::AbstractSparseMatrix{T};
    n_perm::Int=999,
    rng::AbstractRNG=Random.default_rng()
) where {T<:AbstractFloat}
    N = length(y)
    ȳ = mean(y)
    z = y .- ȳ
    m2 = dot(z, z) / N

    local_I = zeros(T, N)
    Wz = W * z
    @inbounds for i in 1:N
        local_I[i] = (z[i] / m2) * Wz[i]
    end

    # Conditional permutation inference
    pseudo_p = ones(T, N)
    z_perm = copy(z)
    for i in 1:N
        count_ge = 0
        zi = z[i]
        for _ in 1:n_perm
            _conditional_permute!(z_perm, z, i, rng)
            Wz_p_i = _sparse_dot_row(W, z_perm, i)
            I_p = (zi / m2) * Wz_p_i
            if abs(I_p) >= abs(local_I[i])
                count_ge += 1
            end
        end
        pseudo_p[i] = T(count_ge + 1) / T(n_perm + 1)
    end

    # Cluster classification
    cluster_type = Vector{Symbol}(undef, N)
    z_scores = z ./ sqrt(m2)
    Wz_std = Wz ./ sqrt(m2)
    @inbounds for i in 1:N
        if pseudo_p[i] > T(0.05)
            cluster_type[i] = :ns
        elseif z_scores[i] > 0 && Wz_std[i] > 0
            cluster_type[i] = :HH
        elseif z_scores[i] < 0 && Wz_std[i] < 0
            cluster_type[i] = :LL
        elseif z_scores[i] > 0 && Wz_std[i] < 0
            cluster_type[i] = :HL
        else
            cluster_type[i] = :LH
        end
    end

    return LISAResult{T}(local_I, pseudo_p, cluster_type, z_scores)
end

function _conditional_permute!(dest::AbstractVector, src::AbstractVector, exclude::Int, rng::AbstractRNG)
    N = length(src)
    copyto!(dest, src)
    # Fisher-Yates shuffle excluding position `exclude`
    @inbounds for k in N:-1:2
        k == exclude && continue
        j = rand(rng, 1:(k))
        while j == exclude
            j = rand(rng, 1:(k))
        end
        dest[k], dest[j] = dest[j], dest[k]
    end
    return dest
end

function _sparse_dot_row(W::SparseArrays.SparseMatrixCSC{T}, x::AbstractVector{T}, i::Int) where {T}
    # Compute (W * x)[i] = sum_j W[i,j] * x[j] by scanning columns
    s = zero(T)
    rows = SparseArrays.rowvals(W)
    vals = SparseArrays.nonzeros(W)
    for j in 1:size(W, 2)
        for idx in SparseArrays.nzrange(W, j)
            if rows[idx] == i
                s += vals[idx] * x[j]
                break
            end
        end
    end
    return s
end

# ─── Geary's C ───────────────────────────────────────────────────────

"""
    geary_c(y, W)

Geary's c, a test for spatial autocorrelation built on differences between
neighbours.

```math
c = ((N-1) / (2 S_0)) * sum_ij w_ij (y_i - y_j)^2 / sum_i (y_i - ybar)^2
```

Where Moran's I is a cross-product statistic and so responds to the global
pattern, Geary's c sums squared differences across linked pairs and is
therefore more sensitive to local, short-range variation. Using both is
informative when they disagree.

# Returns
A [`GearyCResult`](@ref).

!!! note
    The scale runs the other way from Moran's I: the null expectation is 1,
    values **below** 1 indicate positive autocorrelation, and `z_stat` is
    correspondingly negative when values cluster.

# Examples
```julia
g = geary_c(y, W)
g.C_stat < 1      # positive autocorrelation
```

See also [`moran_global`](@ref).

# References
Anselin, L. (1988). *Spatial Econometrics: Methods and Models*. Kluwer.
"""
function geary_c(y::AbstractVector{T}, W::AbstractSparseMatrix{T}) where {T<:AbstractFloat}
    N = length(y)
    ȳ = mean(y)
    z = y .- ȳ
    S0 = sum(W)
    zz = dot(z, z)

    # C = [(N-1)/(2*S0)] * [sum_ij w_ij (y_i - y_j)^2] / [sum_i (y_i - ȳ)^2]
    rows = SparseArrays.rowvals(W)
    vals = SparseArrays.nonzeros(W)
    num = zero(T)
    for j in 1:N
        for idx in SparseArrays.nzrange(W, j)
            i = rows[idx]
            num += vals[idx] * (y[i] - y[j])^2
        end
    end
    C_stat = (T(N - 1) / (2 * S0)) * num / zz
    E_C = one(T)

    # Asymptotic variance under normality
    S1 = _trace_WpWt_sum(W)
    S2 = _col_row_sum_sq(W, N)
    n = T(N)
    Var_C = ((2 * S1 + S2) * (n - 1) - 4 * S0^2) / (2 * (n + 1) * S0^2)
    Var_C = max(Var_C, T(1e-20))
    z_stat = (C_stat - E_C) / sqrt(Var_C)
    p_value = T(2.0) * (one(T) - T(_normcdf(abs(Float64(z_stat)))))

    return GearyCResult{T}(C_stat, E_C, Var_C, z_stat, p_value)
end

# ─── LM Tests (Anselin 1988) ─────────────────────────────────────────

function _ols_residuals_and_sigma(y::AbstractVector{T}, X::AbstractMatrix{T}) where {T<:AbstractFloat}
    N = length(y)
    qr_X = qr(X)
    β = qr_X \ y
    e = y .- X * β
    σ2 = dot(e, e) / N
    return e, σ2, β, qr_X
end

function _trace_W2_WtW(W::AbstractSparseMatrix{T}) where {T}
    # tr(W^2 + W'W)
    N = size(W, 1)
    # tr(W^2): sum_i (W*W)[i,i]
    # tr(W'W): sum of all w_ij^2
    W2_diag = zero(T)
    WtW_trace = zero(T)
    rows = SparseArrays.rowvals(W)
    vals = SparseArrays.nonzeros(W)
    for j in 1:N
        for idx in SparseArrays.nzrange(W, j)
            WtW_trace += vals[idx]^2
        end
    end
    # tr(W^2) via column-wise computation
    for i in 1:N
        # (W^2)[i,i] = sum_k W[i,k]*W[k,i]
        s = zero(T)
        for k in 1:N
            wik = _sparse_getindex(W, i, k)
            wki = _sparse_getindex(W, k, i)
            if wik != zero(T) && wki != zero(T)
                s += wik * wki
            end
        end
        W2_diag += s
    end
    return W2_diag + WtW_trace
end

function _trace_W2_WtW_fast(W::AbstractSparseMatrix{T}) where {T}
    # tr(W^2) = sum_ij W[i,j]*W[j,i] (same as tr(W*W'))
    # tr(W'W) = sum_ij W[i,j]^2
    # Combined: tr(W^2 + W'W)
    N = size(W, 1)
    rows = SparseArrays.rowvals(W)
    vals = SparseArrays.nonzeros(W)

    # tr(W'W) = ||W||_F^2
    wtw = zero(T)
    for j in 1:N
        for idx in SparseArrays.nzrange(W, j)
            wtw += vals[idx]^2
        end
    end

    # tr(W^2): W^2[i,i] = sum_k W[i,k]*W[k,i]
    # More efficient: tr(W^2) = sum of element-wise product W .* W'
    w2 = zero(T)
    for j in 1:N
        for idx in SparseArrays.nzrange(W, j)
            i = rows[idx]
            wij = vals[idx]
            wji = _sparse_getindex(W, j, i)
            w2 += wij * wji
        end
    end
    return w2 + wtw
end

"""
    lm_test_lag(y, X, W)

Anselin's LM test for an omitted spatial lag of the dependent variable.

Fitted by OLS under the null of no spatial dependence, this tests
`H0: rho = 0` in `y = rho*W*y + X*beta + e` against the alternative that a
lag belongs in the model. The statistic is chi-squared with 1 degree of
freedom.

It is **not robust**: a spatial error process also inflates it, so a
rejection on its own does not distinguish lag from error dependence. Use
[`lm_test_robust_lag`](@ref), or [`lm_tests`](@ref) for the whole battery.
`X` must include an intercept column; none is added for you.


# Returns
An [`LMTestResult`](@ref).

# References
Anselin, L. (1988). *Spatial Econometrics: Methods and Models*. Kluwer.
"""
function lm_test_lag(y::AbstractVector{T}, X::AbstractMatrix{T}, W::AbstractSparseMatrix{T}) where {T<:AbstractFloat}
    e, σ2, _, _ = _ols_residuals_and_sigma(y, X)
    N = length(y)
    We = W * e
    D = _trace_W2_WtW_fast(W)
    RS_lag = (dot(e, We) / σ2)^2 / D
    # chi2(1) p-value: P(χ² > x) ≈ 2*(1 - Φ(√x))
    p_val = T(2.0) * (one(T) - T(_normcdf(sqrt(max(zero(T), RS_lag)))))
    return LMTestResult{T}(RS_lag, p_val, "LM-Lag")
end

"""
    lm_test_error(y, X, W)

Anselin's LM test for spatially autocorrelated errors.

Tests `H0: lambda = 0` in `y = X*beta + u`, `u = lambda*W*u + e`, from OLS
residuals. Chi-squared with 1 degree of freedom.

Like [`lm_test_lag`](@ref) it is not robust to the other form of
dependence; see [`lm_test_robust_error`](@ref) and [`lm_tests`](@ref).
`X` must include an intercept column; none is added for you.


# Returns
An [`LMTestResult`](@ref).

# References
Anselin, L. (1988). *Spatial Econometrics: Methods and Models*. Kluwer.
"""
function lm_test_error(y::AbstractVector{T}, X::AbstractMatrix{T}, W::AbstractSparseMatrix{T}) where {T<:AbstractFloat}
    e, σ2, _, _ = _ols_residuals_and_sigma(y, X)
    N = length(y)
    We = W * e
    D = _trace_W2_WtW_fast(W)
    RS_err = (dot(e, We) / σ2)^2 / D
    p_val = T(2.0) * (one(T) - T(_normcdf(sqrt(max(zero(T), RS_err)))))
    return LMTestResult{T}(RS_err, p_val, "LM-Error")
end

"""
    lm_test_robust_lag(y, X, W)

LM test for an omitted spatial lag, robust to the presence of spatial error
dependence.

This is the test to read when both plain statistics reject. It corrects
`lm_test_lag` for local misspecification of the error process, so a
rejection points at the lag specifically rather than at spatial dependence
in general.
`X` must include an intercept column; none is added for you.


# Returns
An [`LMTestResult`](@ref).

See also [`lm_tests`](@ref), [`lm_test_robust_error`](@ref).

# References
Anselin, L., Bera, A., Florax, R. & Yoon, M. (1996). Simple diagnostic
tests for spatial dependence. *Regional Science and Urban Economics*,
26(1).
"""
function lm_test_robust_lag(y::AbstractVector{T}, X::AbstractMatrix{T}, W::AbstractSparseMatrix{T}) where {T<:AbstractFloat}
    e, σ2, _, qr_X = _ols_residuals_and_sigma(y, X)
    N = length(y)
    We = W * e
    Wy = W * y
    D = _trace_W2_WtW_fast(W)

    # MX = I - X(X'X)^{-1}X'
    MWy = Wy .- X * (qr_X \ Wy)
    J_rho_rho = dot(MWy, MWy) / σ2 + D
    J_rho_lam = D
    J_lam_lam = D

    s_rho = dot(e, MWy) / σ2 + dot(e, We) / σ2
    s_lam = dot(e, We) / σ2

    # Robust LM-lag = (s_rho - s_lam)^2 / (J_rho_rho - J_rho_lam)
    numer = (s_rho - s_lam)^2
    denom = J_rho_rho - J_rho_lam
    RS_rob_lag = numer / max(denom, T(1e-12))
    p_val = T(2.0) * (one(T) - T(_normcdf(sqrt(max(zero(T), RS_rob_lag)))))
    return LMTestResult{T}(RS_rob_lag, p_val, "Robust LM-Lag")
end

"""
    lm_test_robust_error(y, X, W)

LM test for spatially autocorrelated errors, robust to the presence of an
omitted spatial lag.

The counterpart to [`lm_test_robust_lag`](@ref); together they are what
separates an error process from a lag process when both plain tests reject.
`X` must include an intercept column; none is added for you.


# Returns
An [`LMTestResult`](@ref).

# References
Anselin, L., Bera, A., Florax, R. & Yoon, M. (1996). *Regional Science and
Urban Economics*, 26(1).
"""
function lm_test_robust_error(y::AbstractVector{T}, X::AbstractMatrix{T}, W::AbstractSparseMatrix{T}) where {T<:AbstractFloat}
    e, σ2, _, qr_X = _ols_residuals_and_sigma(y, X)
    N = length(y)
    We = W * e
    Wy = W * y
    D = _trace_W2_WtW_fast(W)

    MWy = Wy .- X * (qr_X \ Wy)
    J_rho_rho = dot(MWy, MWy) / σ2 + D

    s_rho = dot(e, MWy) / σ2 + dot(e, We) / σ2
    s_lam = dot(e, We) / σ2

    # Robust LM-error = (s_lam - D * s_rho / J_rho_rho)^2 / (D - D^2 / J_rho_rho)
    numer = (s_lam - D * s_rho / J_rho_rho)^2
    denom = D - D^2 / J_rho_rho
    RS_rob_err = numer / max(denom, T(1e-12))
    p_val = T(2.0) * (one(T) - T(_normcdf(sqrt(max(zero(T), RS_rob_err)))))
    return LMTestResult{T}(RS_rob_err, p_val, "Robust LM-Error")
end

"""
    lm_tests(y, X, W) -> NamedTuple

Run the full Anselin specification-search battery and return all four tests
as `(; lag, error, robust_lag, robust_error)`.

This is the usual way to decide which spatial model to fit. The standard
reading:

| Plain tests | Robust tests | Indication |
|---|---|---|
| neither rejects | — | OLS is adequate |
| only `lag` rejects | — | fit `SAR` |
| only `error` rejects | — | fit a spatial error model |
| both reject | `robust_lag` only | fit `SAR` |
| both reject | `robust_error` only | fit a spatial error model |
| both reject | both reject | consider `SDM` or `SARAR` |

Each element is an [`LMTestResult`](@ref).

# Examples
```julia
t = lm_tests(y, hcat(ones(N), X), W)      # X needs the intercept column
t.lag.p_value, t.robust_lag.p_value

# fit_spatial does add its own intercept, so pass the bare X there
m = t.robust_lag.p_value < t.robust_error.p_value ?
        fit_spatial(SAR(), W, X, y) : fit_spatial_error(W, X, y)
```

!!! warning
    `X` must **already include** an intercept column — the OLS step is a
    bare `qr(X) \\ y` and adds nothing. Pass `hcat(ones(N), X)`, or the
    residuals are taken around zero and every statistic is wrong.

# References
Anselin, L., Bera, A., Florax, R. & Yoon, M. (1996). *Regional Science and
Urban Economics*, 26(1).
"""
function lm_tests(y::AbstractVector{T}, X::AbstractMatrix{T}, W::AbstractSparseMatrix{T}) where {T<:AbstractFloat}
    return (
        lag = lm_test_lag(y, X, W),
        error = lm_test_error(y, X, W),
        robust_lag = lm_test_robust_lag(y, X, W),
        robust_error = lm_test_robust_error(y, X, W)
    )
end

# ─── Show methods ─────────────────────────────────────────────────────

function Base.show(io::IO, r::MoranResult)
    println(io, "Global Moran's I Test ($(r.method))")
    @printf(io, "  I = %.6f    E[I] = %.6f    z = %.4f    p = %.4f\n", r.I_stat, r.E_I, r.z_stat, r.p_value)
end

function Base.show(io::IO, r::LISAResult)
    n_hh = count(==(:HH), r.cluster_type)
    n_ll = count(==(:LL), r.cluster_type)
    n_hl = count(==(:HL), r.cluster_type)
    n_lh = count(==(:LH), r.cluster_type)
    n_ns = count(==(:ns), r.cluster_type)
    println(io, "Local Moran's I (LISA) — N = $(length(r.local_I))")
    @printf(io, "  Clusters: HH=%d  LL=%d  HL=%d  LH=%d  ns=%d\n", n_hh, n_ll, n_hl, n_lh, n_ns)
end

function Base.show(io::IO, r::LMTestResult)
    stars = r.p_value < 0.001 ? "***" : r.p_value < 0.01 ? "**" : r.p_value < 0.05 ? "*" : ""
    @printf(io, "%s: stat = %.4f    p = %.4f %s\n", r.test_name, r.stat, r.p_value, stars)
end

function Base.show(io::IO, r::GearyCResult)
    @printf(io, "Geary's C Test: C = %.6f    E[C] = %.6f    z = %.4f    p = %.4f\n", r.C_stat, r.E_C, r.z_stat, r.p_value)
end

end # module
