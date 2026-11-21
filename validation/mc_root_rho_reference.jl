# Monte Carlo for root_rho's bias-aware interval on a reference basis (THEORY.md, Lemma 5 and
# Proposition 7′), through the released function, with λ chosen by the interval's length.
#
# The research check (research/trace_cfe/tc_refbasis.jl) used its own code, hat functions on
# [0, 1] and tc_lib's kNN. Here the bases come from tensor_pspline (knots over the range of the
# data) and W from knn_weights, so this measures the package, not a copy of the research numbers.
#
# Design: n = 900 random locations, kNN-6, x unpenalised as X, f = x + a·sin(2πu)cos(πv),
# heteroskedastic normal errors, ρ₀ = 0.4. Estimation basis tensor_pspline(u, v; k), the λ grid
# 10^(-3:0.5:2) plus the projection (REsd); reference tensor_pspline(u, v; k = 30), ℛ̃* =
# basis_roughness(g, u, v; k = 30) for the true surface g. R = 300 per cell. Cells:
#   1. k = 5,  a = 4, ℛ̃*        2. k = 15, a = 4, ℛ̃*       3. k = 15, a = 1, ℛ̃*
#   4. k = 15, a = 4, 4ℛ̃*       5. (power) k = 5, a = 4, no reference, ℛ from the
#      least-squares fit of f on [X, basis] (the released rule without `reference`).
# Pass criterion (set before the run): cells 1–4 cover within 0.93–1.00; cell 5 below 0.90.
# Also: on the first draw of cell 1, the projection's b̃ against a dense reimplementation of
# Lemma 5(c), to 1e-8 relative.
#
# Run:    julia --project=. validation/mc_root_rho_reference.jl [R]   (R = 300 by default)
#         julia --project=. validation/mc_root_rho_reference.jl time  (cost per fit)
# Output: validation/mc_root_rho_reference_output.txt
using SpatialEconometrics, LinearAlgebra, SparseArrays, Random, Statistics, Printf

const λS = 10.0 .^ (-3:0.5:2)

function design(; n = 900, seed = 5)
    rng = MersenneTwister(seed)
    P = rand(rng, n, 2); W = knn_weights(P, 6); u, v = P[:, 1], P[:, 2]
    x = randn(rng, n)
    σ2 = exp.(3 .* (u .+ v .- 1)); σ2 ./= mean(σ2)
    (; n, W, u, v, X = reshape(x, :, 1), σ2, Fac = lu(sparse(I - 0.4 * W)),
       ref = tensor_pspline(u, v; k = 30))
end

# Lemma 5(c) for the projection alone, dense, at the package's own first-step root.
function dense_resd_bound(d, B, y, ρ̃, ℛ)
    Wd = Matrix(d.W); Z = hcat(d.X, B); K = size(Z, 2); Zt = hcat(d.X, d.ref.basis)
    Q = Matrix(qr(Z).Q)[:, 1:K]; Pz = Q * Q'; M0 = I - Pz; Mii = diag(M0)
    jlE(A) = A - Diagonal(diag(A * M0) ./ Mii)
    Wy = Wd * y; y0 = y .- ρ̃ .* Wy; s0 = (M0 * y0) .^ 2 ./ Mii; fh = Pz * y0
    Πt = zeros(size(Zt, 2), size(Zt, 2)); Πt[2:end, 2:end] = d.ref.penalty
    E = eigen(Symmetric(Πt)); keep = E.values .> 1e-10 * maximum(E.values); U = E.vectors[:, keep]
    Πp = U * Diagonal(1 ./ E.values[keep]) * U'; Πh = U * Diagonal(1 ./ sqrt.(E.values[keep])) * U'
    C = jlE((I - ρ̃ * Wd') \ Wd') * M0; Tm = (C * Zt)'; Qq = (M0 * Zt)' * C * Zt
    Υ = Πh * Tm * Pz
    q = dot(Tm * fh, Πp * (Tm * fh)) - sum(s0 .* vec(sum(abs2, Υ; dims = 1)))
    χ = maximum(abs, eigvals(Symmetric(Πh * ((Qq + Qq') / 2) * Πh)))
    ℛ * sqrt(max(q, 0.0)) + ℛ^2 * χ
end

function cell(d, k, amp, mult, R; reference = true, seed = 11, check = false)
    g(s, t) = amp * sin(2π * s) * cos(π * t)
    f = d.X[:, 1] .+ g.(d.u, d.v)
    ps = tensor_pspline(d.u, d.v; k)
    ℛ = reference ? mult * basis_roughness(g, d.u, d.v; k = 30) :
                    mult * basis_roughness(f, ps.basis, ps.penalty; X = d.X)
    rng = MersenneTwister(seed + 10k + round(Int, amp))
    cov = 0; hl = Float64[]; e = Float64[]; picks = Dict{Float64, Int}(); m = 0; gap = NaN
    for r in 1:R
        y = d.Fac \ (f .+ sqrt.(d.σ2) .* randn(rng, d.n))
        if check && r == 1
            p = root_rho(y, d.X, d.W; basis = ps.basis, roughness = ℛ, reference = d.ref)
            gap = abs(p.bias_bound / dense_resd_bound(d, ps.basis, y, p.rho_first, ℛ) - 1)
        end
        gr = reference ?
            root_rho(y, d.X, d.W; ps..., lambda = λS, roughness = ℛ, reference = d.ref) :
            root_rho(y, d.X, d.W; ps..., lambda = λS, roughness = ℛ)
        gr.root_found && isfinite(gr.ci_bias_aware[1]) || continue
        m += 1; lo, hi = gr.ci_bias_aware
        cov += lo <= 0.4 <= hi; push!(hl, (hi - lo) / 2); push!(e, gr.rho - 0.4)
        picks[gr.lambda] = get(picks, gr.lambda, 0) + 1
    end
    (; ℛ, cov = cov / m, m, hl = mean(hl), bias = mean(e), sd = std(e), picks, gap)
end

function main(R)
    d = design()
    println("root_rho's bias-aware interval on a reference basis (Lemma 5, Proposition 7′); Julia $VERSION")
    println("n = $(d.n), kNN-6, reference tensor_pspline(k = 30) (K̃ = $(size(d.ref.basis, 2) + 1)), λ grid 10^(-3:0.5:2) plus REsd, R = $R\n")
    pass = true; power = true
    cells = ((5, 4.0, 1.0, true), (15, 4.0, 1.0, true), (15, 1.0, 1.0, true), (15, 4.0, 4.0, true),
             (5, 4.0, 1.0, false))
    for (j, (k, amp, mult, ref)) in enumerate(cells)
        t0 = time()
        c = cell(d, k, amp, mult, R; reference = ref, check = j == 1)
        ok = ref ? 0.93 <= c.cov <= 1.0 : c.cov < 0.90
        ref ? (pass &= ok) : (power &= ok)
        pk = join(["$(l == 0 ? "REsd" : round(l, sigdigits = 3)):$(c.picks[l])" for l in sort(collect(keys(c.picks)))], " ")
        @printf("%d. k = %2d, a = %.0f, %s ℛ = %.3f: coverage %.3f (%d fits), half-length %.4f, bias %+.4f, sd %.4f | %s | %.1f min\n",
                j, k, amp, ref ? @sprintf("reference, %.0f ℛ̃*,", mult) : "no reference, LS",
                c.ℛ, c.cov, c.m, c.hl, c.bias, c.sd, ok ? "ok" : "FAIL", (time() - t0) / 60)
        println("    picks (λ: count) ", pk)
        j == 1 && @printf("    first draw: the projection's b̃ against the dense Lemma 5(c), relative gap %.1e (%s)\n",
                          c.gap, c.gap < 1e-8 ? "ok" : "FAIL")
        j == 1 && (pass &= c.gap < 1e-8)
    end
    println("\nPass criterion (cells 1–4 within 0.93–1.00, and the first-draw check): ", pass ? "met" : "NOT met")
    println("Power check (cell 5 below 0.90): ", power ? "met" : "NOT met")
end

if !isempty(ARGS) && ARGS[1] == "time"
    d = design(); ps = tensor_pspline(d.u, d.v; k = 15); g(s, t) = 4sin(2π * s) * cos(π * t)
    y = d.Fac \ (d.X[:, 1] .+ g.(d.u, d.v) .+ sqrt.(d.σ2) .* randn(MersenneTwister(1), d.n))
    ℛ = basis_roughness(g, d.u, d.v; k = 30)
    root_rho(y, d.X, d.W; ps..., lambda = λS, roughness = ℛ, reference = d.ref)
    t = @elapsed root_rho(y, d.X, d.W; ps..., lambda = λS, roughness = ℛ, reference = d.ref)
    @printf("one fit over the grid with a reference: %.2f s; 5 cells × R = 300: about %.0f min\n",
            t, 5 * 300 * t / 60)
else
    main(isempty(ARGS) ? 300 : parse(Int, ARGS[1]))
end
