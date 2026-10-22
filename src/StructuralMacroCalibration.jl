module StructuralMacroCalibration

using LinearAlgebra
using SparseArrays
using Printf
using ForwardDiff
using ChainRulesCore

export CalibrationResult,
       balance_gras, balance_gras_full,
       balance_generalized_ce,
       sam_leontief_multipliers

"""
    CalibrationResult

Output of a matrix-balancing problem, from
[`balance_gras_full`](@ref) or [`balance_generalized_ce`](@ref).

# Fields
- `X_balanced`: the balanced matrix, the result itself.
- `X_prior`: the input matrix it started from.
- `tau`, `sigma`: the row and column scaling multipliers.
- `W_sens`: sensitivity matrix, for propagating uncertainty in the targets
  through to the solution.
- `kl_divergence`: how far `X_balanced` moved from `X_prior`. Large values
  mean the targets were hard to reconcile with the prior.
- `max_discrepancy`: largest remaining violation of a target constraint.
- `iterations`, `converged`: solver state. **Check `converged`**, and read
  `max_discrepancy` even when it is `true`.
"""
struct CalibrationResult{T<:AbstractFloat}
    X_balanced::Matrix{T}
    X_prior::Matrix{T}
    tau::Vector{T}
    sigma::Vector{T}
    W_sens::Matrix{T}
    kl_divergence::T
    max_discrepancy::T
    iterations::Int
    converged::Bool
end

function solve_gras_primal_dual(
    X0::AbstractMatrix{T},
    u::AbstractVector{T},
    v::AbstractVector{T};
    tol::T=T(1e-11),
    max_coord_iter::Int=25,
    max_newton_iter::Int=20
) where {T<:AbstractFloat}

    M, N = size(X0)
    sum_u = sum(u)
    sum_v = sum(v)
    if abs(sum_u - sum_v) > T(1e-6) * max(one(T), abs(sum_u))
        throw(ArgumentError("Inconsistent macro totals: sum(u) = $sum_u != sum(v) = $sum_v"))
    end

    P = zeros(T, M, N)
    Neg = zeros(T, M, N)
    @inbounds for j in 1:N, i in 1:M
        val = X0[i, j]
        if val >= zero(T); P[i, j] = val; else; Neg[i, j] = -val; end
    end

    r = ones(T, M)
    s = ones(T, N)

    for _ in 1:max_coord_iter
        @inbounds for i in 1:M
            pi_sum = zero(T); ni_sum = zero(T)
            for j in 1:N
                pi_sum += P[i, j] * s[j]
                ni_sum += Neg[i, j] / s[j]
            end
            if pi_sum > zero(T)
                r[i] = (u[i] + sqrt(u[i]^2 + 4 * pi_sum * ni_sum)) / (2 * pi_sum)
            elseif ni_sum > zero(T) && u[i] < zero(T)
                r[i] = -ni_sum / u[i]
            end
        end

        @inbounds for j in 1:N
            pj_sum = zero(T); nj_sum = zero(T)
            for i in 1:M
                pj_sum += P[i, j] * r[i]
                nj_sum += Neg[i, j] / r[i]
            end
            if pj_sum > zero(T)
                s[j] = (v[j] + sqrt(v[j]^2 + 4 * pj_sum * nj_sum)) / (2 * pj_sum)
            elseif nj_sum > zero(T) && v[j] < zero(T)
                s[j] = -nj_sum / v[j]
            end
        end
    end

    s_N = s[N]
    τ = log.(r .* s_N)
    σ = log.(s ./ s_N)
    σ[N] = zero(T)

    X_bal = zeros(T, M, N)
    W_sens = zeros(T, M, N)
    F_res = zeros(T, M + N)
    J_kkt = zeros(T, M + N, M + N)

    converged = false
    total_iter = max_coord_iter
    max_err = T(Inf)

    for iter in 1:max_newton_iter
        @inbounds for j in 1:N, i in 1:M
            θ_ij = τ[i] + σ[j]
            e_pos = exp(θ_ij)
            e_neg = exp(-θ_ij)
            p_term = P[i, j] * e_pos
            n_term = Neg[i, j] * e_neg
            X_bal[i, j] = p_term - n_term
            W_sens[i, j] = p_term + n_term
        end

        @inbounds for i in 1:M
            row_s = zero(T)
            for j in 1:N; row_s += X_bal[i, j]; end
            F_res[i] = row_s - u[i]
        end
        @inbounds for j in 1:N
            col_s = zero(T)
            for i in 1:M; col_s += X_bal[i, j]; end
            F_res[M + j] = col_s - v[j]
        end

        max_err = maximum(abs, F_res)
        if max_err <= tol; converged = true; break; end
        total_iter += 1

        fill!(J_kkt, zero(T))
        @inbounds for i in 1:M
            w_row = zero(T)
            for j in 1:N
                wij = W_sens[i, j]
                w_row += wij
                J_kkt[i, M + j] = wij
                J_kkt[M + j, i] = wij
            end
            J_kkt[i, i] = w_row + T(1e-12)
        end
        @inbounds for j in 1:N
            w_col = zero(T)
            for i in 1:M; w_col += W_sens[i, j]; end
            J_kkt[M + j, M + j] = w_col + T(1e-12)
        end

        dim_red = M + N - 1
        @views J_red = J_kkt[1:dim_red, 1:dim_red]
        @views F_red = F_res[1:dim_red]

        Δ = -(Symmetric(J_red) \ F_red)
        step = min(one(T), T(3.0) / max(maximum(abs, Δ), T(1e-12)))

        @inbounds for i in 1:M; τ[i] += step * Δ[i]; end
        @inbounds for j in 1:(N - 1); σ[j] += step * Δ[M + j]; end
        σ[N] = zero(T)
    end

    kl_div = zero(T)
    @inbounds for j in 1:N, i in 1:M
        x_abs = abs(X_bal[i, j])
        x0_abs = abs(X0[i, j])
        if x_abs > zero(T) && x0_abs > zero(T)
            kl_div += x_abs * log(x_abs / x0_abs) - (x_abs - x0_abs)
        end
    end

    return CalibrationResult{T}(
        X_bal, Matrix{T}(X0), τ, σ, W_sens,
        kl_div, max_err, total_iter, converged
    )
end

"""
    balance_gras_full(X0, u, v; tol = 1e-11) -> CalibrationResult

Balance a matrix to given row and column totals by GRAS, returning the full
diagnostics.

GRAS (Generalized RAS) extends biproportional scaling to matrices
containing **negative** entries, which ordinary RAS cannot handle — the
reason it is the standard tool for social accounting matrices and
input-output tables, where negatives legitimately occur.

The solution is the matrix closest to `X0` in a cross-entropy sense that
satisfies the targets, so the prior's structure is preserved as far as the
constraints allow. Sign patterns and zeros in `X0` are respected: a zero
cell stays zero.

# Arguments
- `X0`: prior matrix.
- `u`: target row sums.
- `v`: target column sums.

Totals must be consistent (`sum(u) == sum(v)`) for a solution to exist.

# Returns
A [`CalibrationResult`](@ref). Use [`balance_gras`](@ref) when you want the
balanced matrix alone.

!!! note
    Differentiable: the solution can be propagated through automatic
    differentiation, so a balanced matrix can sit inside a larger
    calibration or estimation problem.

# References
Junius, T. & Oosterhaven, J. (2003). *Economic Systems Research*, 15(1).
"""
function balance_gras_full(X0::AbstractMatrix{T}, u::AbstractVector{T}, v::AbstractVector{T}; kwargs...) where {T<:AbstractFloat}
    return solve_gras_primal_dual(X0, u, v; kwargs...)
end

"""
    balance_gras(X0, u, v; tol = 1e-11) -> Matrix

Balance a matrix to target row sums `u` and column sums `v` by GRAS,
returning just the balanced matrix.

A convenience wrapper over [`balance_gras_full`](@ref), which returns the
multipliers, sensitivities and convergence diagnostics as well. Prefer the
full version when the result matters, since this one discards the
information that would tell you the solve went badly.

# Examples
```julia
X = balance_gras(X_prior, row_targets, col_targets)
```
"""
function balance_gras(X0::AbstractMatrix{T}, u::AbstractVector{T}, v::AbstractVector{T}; tol::Float64=1e-11) where {T<:AbstractFloat}
    res = solve_gras_primal_dual(X0, u, v; tol=T(tol))
    return res.X_balanced
end

function solve_kkt_adjoint(W_sens::Matrix{T}, rhs_u::AbstractVector{T}, rhs_v::AbstractVector{T}) where {T<:AbstractFloat}
    M, N = size(W_sens)
    dim_red = M + N - 1
    J_red = zeros(T, dim_red, dim_red)

    @inbounds for i in 1:M
        w_row = zero(T)
        for j in 1:N
            wij = W_sens[i, j]
            w_row += wij
            if j < N; J_red[i, M + j] = wij; J_red[M + j, i] = wij; end
        end
        J_red[i, i] = w_row + T(1e-12)
    end

    @inbounds for j in 1:(N - 1)
        w_col = zero(T)
        for i in 1:M; w_col += W_sens[i, j]; end
        J_red[M + j, M + j] = w_col + T(1e-12)
    end

    rhs_red = vcat(rhs_u, @view rhs_v[1:(N - 1)])
    sol = Symmetric(J_red) \ rhs_red

    λ_τ = sol[1:M]
    λ_σ = vcat(sol[(M + 1):dim_red], zero(T))
    return λ_τ, λ_σ
end

function ChainRulesCore.rrule(::typeof(balance_gras), X0::AbstractMatrix{T}, u::AbstractVector{T}, v::AbstractVector{T}; tol::Float64=1e-11) where {T<:AbstractFloat}
    res = solve_gras_primal_dual(X0, u, v; tol=T(tol))
    X_star = res.X_balanced; W_sens = res.W_sens; τ = res.tau; σ = res.sigma

    function balance_gras_pullback(ΔX_raw)
        ΔX = unthunk(ΔX_raw)
        M, N = size(X_star)
        rhs_u = zeros(T, M); rhs_v = zeros(T, N)
        @inbounds for j in 1:N, i in 1:M
            w_dx = W_sens[i, j] * ΔX[i, j]
            rhs_u[i] += w_dx; rhs_v[j] += w_dx
        end
        λ_τ, λ_σ = solve_kkt_adjoint(W_sens, rhs_u, rhs_v)
        Δu = copy(λ_τ); Δv = copy(λ_σ)
        ΔX0 = zeros(T, M, N)
        @inbounds for j in 1:N, i in 1:M
            θ_ij = τ[i] + σ[j]
            scale_ij = X0[i, j] >= zero(T) ? exp(θ_ij) : exp(-θ_ij)
            ΔX0[i, j] = (ΔX[i, j] - λ_τ[i] - λ_σ[j]) * scale_ij
        end
        return NoTangent(), ΔX0, Δu, Δv
    end
    return X_star, balance_gras_pullback
end

function balance_gras(X0::AbstractMatrix{T1}, u::AbstractVector{T2}, v::AbstractVector{T3}; tol::Float64=1e-11) where {T1<:Real, T2<:Real, T3<:Real}
    R = promote_type(T1, T2, T3)
    if !(R <: ForwardDiff.Dual)
        return balance_gras(convert(Matrix{R}, X0), convert(Vector{R}, u), convert(Vector{R}, v); tol=tol)
    end
    V = ForwardDiff.valtype(R); N_partials = ForwardDiff.npartials(R)
    X0_val = ForwardDiff.value.(X0); u_val = ForwardDiff.value.(u); v_val = ForwardDiff.value.(v)
    res = solve_gras_primal_dual(X0_val, u_val, v_val; tol=V(tol))
    X_star = res.X_balanced; W_sens = res.W_sens; τ = res.tau; σ = res.sigma
    M, N = size(X_star)

    dim_red = M + N - 1
    J_red = zeros(V, dim_red, dim_red)
    @inbounds for i in 1:M
        w_row = zero(V)
        for j in 1:N
            wij = W_sens[i, j]
            w_row += wij
            if j < N; J_red[i, M + j] = wij; J_red[M + j, i] = wij; end
        end
        J_red[i, i] = w_row + V(1e-12)
    end
    @inbounds for j in 1:(N - 1)
        w_col = zero(V)
        for i in 1:M; w_col += W_sens[i, j]; end
        J_red[M + j, M + j] = w_col + V(1e-12)
    end
    # `ldlt` has no dense-Symmetric method (LinearAlgebra defines it only for
    # SymTridiagonal and sparse CHOLMOD matrices), so this path previously threw
    # a MethodError for every ForwardDiff/dual call. bunchkaufman is the dense
    # symmetric LDL' factorization and handles the indefinite case too.
    J_fact = bunchkaufman(Symmetric(J_red))

    rhs_mat = zeros(V, dim_red, N_partials)
    dX0_direct = Array{V, 3}(undef, M, N, N_partials)
    @inbounds for j in 1:N, i in 1:M
        θ_ij = τ[i] + σ[j]
        scale_ij = X0_val[i, j] >= zero(V) ? exp(θ_ij) : exp(-θ_ij)
        for p in 1:N_partials
            dx0_p = T1 <: ForwardDiff.Dual ? ForwardDiff.partials(X0[i, j], p) : zero(V)
            dx_dir = scale_ij * dx0_p
            dX0_direct[i, j, p] = dx_dir
            rhs_mat[i, p] -= dx_dir
            if j < N; rhs_mat[M + j, p] -= dx_dir; end
        end
    end

    @inbounds for i in 1:M, p in 1:N_partials
        du_p = T2 <: ForwardDiff.Dual ? ForwardDiff.partials(u[i], p) : zero(V)
        rhs_mat[i, p] += du_p
    end
    @inbounds for j in 1:(N - 1), p in 1:N_partials
        dv_p = T3 <: ForwardDiff.Dual ? ForwardDiff.partials(v[j], p) : zero(V)
        rhs_mat[M + j, p] += dv_p
    end

    d_mult = J_fact \ rhs_mat
    X_dual = Matrix{R}(undef, M, N)
    @inbounds for j in 1:N, i in 1:M
        wij = W_sens[i, j]
        part_tuple = ntuple(N_partials) do p
            dτ_ip = d_mult[i, p]
            dσ_jp = j < N ? d_mult[M + j, p] : zero(V)
            return dX0_direct[i, j, p] + wij * (dτ_ip + dσ_jp)
        end
        X_dual[i, j] = R(X_star[i, j], ForwardDiff.Partials(part_tuple))
    end
    return X_dual
end

"""
    balance_generalized_ce(X0, G, b, sigma_b; tol = 1e-10, maxiter = 40) -> CalibrationResult

Balance a matrix under general linear constraints with uncertain targets, by
cross-entropy minimization.

More flexible than [`balance_gras_full`](@ref) in two ways. Constraints are
arbitrary linear combinations `G*vec(X) = b` rather than just row and column
sums, so aggregates spanning several cells can be targeted. And each target
carries a standard deviation `sigma_b`, so constraints are matched as
closely as their reliability warrants rather than exactly — which is what
makes an over-determined or mutually inconsistent set of targets solvable
at all.

# Arguments
- `X0`: prior matrix.
- `G`: constraint matrix acting on `vec(X)`.
- `b`: target values.
- `sigma_b`: standard deviation per target. Smaller means a harder
  constraint; set one very small to make a target effectively exact.

# Returns
A [`CalibrationResult`](@ref).

# References
Golan, A., Judge, G. & Robinson, S. (1994). Recovering information from
incomplete or partial multisectoral economic data. *Review of Economics and
Statistics*, 76(3).
"""
function balance_generalized_ce(X0::AbstractMatrix{T}, G::AbstractMatrix{T}, b::AbstractVector{T}, sigma_b::AbstractVector{T}; tol::T=T(1e-10), maxiter::Int=40) where {T<:AbstractFloat}
    M, N = size(X0); MN = M * N; K = length(b)
    x0_vec = vec(copy(X0)); p_vec = max.(x0_vec, zero(T)); n_vec = max.(-x0_vec, zero(T))
    var_b = sigma_b .^ 2; λ = zeros(T, K); x_curr = similar(x0_vec); w_curr = similar(x0_vec)

    for _ in 1:maxiter
        θ = G' * λ
        @inbounds for idx in 1:MN
            ep = exp(θ[idx]); en = exp(-θ[idx])
            x_curr[idx] = p_vec[idx] * ep - n_vec[idx] * en
            w_curr[idx] = p_vec[idx] * ep + n_vec[idx] * en
        end
        F_dual = G * x_curr .+ (var_b .* λ) .- b
        maximum(abs, F_dual) <= tol && break
        GW = G .* w_curr'; H_dual = GW * G' + Diagonal(var_b .+ T(1e-12))
        Δλ = -(Symmetric(H_dual) \ F_dual)
        λ .+= min(one(T), T(3.0) / max(maximum(abs, Δλ), T(1e-12))) .* Δλ
    end
    return reshape(x_curr, M, N), -(var_b .* λ), λ
end

"""
    sam_leontief_multipliers(X_sam, endogenous_idx)

Leontief multipliers from a social accounting matrix.

Forms the column-coefficient matrix over the accounts named by
`endogenous_idx`, then inverts `(I - A)` to give the total effect of a unit
of exogenous injection on each endogenous account — direct plus all
induced rounds.

# Arguments
- `X_sam`: a **balanced** SAM. Balance it first with
  [`balance_gras`](@ref); multipliers from an unbalanced matrix are
  meaningless.
- `endogenous_idx`: indices of the accounts treated as endogenous. The rest
  are the exogenous injections, and this split determines the multipliers
  entirely.

!!! warning
    `(I - A)` must be invertible, which needs the endogenous block to be
    productive. A near-singular inverse shows up as implausibly large
    multipliers rather than an error.
"""
function sam_leontief_multipliers(X_sam::AbstractMatrix{T}, endogenous_idx::AbstractVector{Int}) where {T<:Real}
    col_totals = vec(sum(X_sam; dims=1))
    X_endog = X_sam[endogenous_idx, endogenous_idx]
    y_endog = col_totals[endogenous_idx]
    A_n = X_endog ./ y_endog'
    K_endog = length(endogenous_idx)
    M_a = (Matrix{T}(I, K_endog, K_endog) - A_n) \ Matrix{T}(I, K_endog, K_endog)
    return M_a, A_n
end

function Base.show(io::IO, r::CalibrationResult)
    M, N = size(r.X_balanced)
    println(io, "═══════════════════════════════════════════════════════════════════════")
    println(io, "GRAS Calibration Result — $(M)×$(N) Matrix")
    println(io, "───────────────────────────────────────────────────────────────────────")
    Printf.@printf(io, "Converged: %s    Iterations: %d\n", r.converged ? "Yes" : "No", r.iterations)
    Printf.@printf(io, "KL divergence: %.6e    Max discrepancy: %.6e\n", r.kl_divergence, r.max_discrepancy)
    println(io, "═══════════════════════════════════════════════════════════════════════")
end

end # module
