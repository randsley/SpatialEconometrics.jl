# Monte Carlo for root_rho with the basis chosen by the interval's length (THEORY.md, Corollary 7′),
# through the released function.
#
# Design: validation/mc_root_rho_reference.jl's (n = 900 random locations, kNN-6, x unpenalised as X,
# f = x + a·sin(2πu)cos(πv), heteroskedastic normal errors, ρ₀ = 0.4). Bases tensor_pspline(u, v; k)
# for k = 5, 10, 15, each with the λ grid 10^(-3:0.5:2) plus the projection (REsd): 39 candidates.
# Reference tensor_pspline(u, v; k = 30), ℛ̃* = basis_roughness(g, u, v; k = 30). R = 200 per cell.
# Cells:
#   1. a = 4, ℛ̃*     2. a = 1, ℛ̃*     3. a = 4, 4ℛ̃*     4. a = 4, ℛ̃*, bonferroni = true
#   5. (power) a = 4, no reference: each basis with its own class, ℛ_k from the least-squares fit
#      of f on [X, basis_k], and the shortest interval over the three bases (what the refused call
#      would do).
# Pass criterion (set before the run): cells 1–4 cover within 0.93–1.00; cell 5 below 0.90.
#
# Run:    julia --project=. validation/mc_root_rho_basis_grid.jl [R]   (R = 200 by default)
# Output: validation/mc_root_rho_basis_grid_output.txt
using SpatialEconometrics, LinearAlgebra, SparseArrays, Random, Statistics, Printf

const λS = 10.0 .^ (-3:0.5:2)
const KS = (5, 10, 15)

function design(; n = 900, seed = 5)
    rng = MersenneTwister(seed)
    P = rand(rng, n, 2); W = knn_weights(P, 6); u, v = P[:, 1], P[:, 2]
    x = randn(rng, n)
    σ2 = exp.(3 .* (u .+ v .- 1)); σ2 ./= mean(σ2)
    (; n, W, u, v, X = reshape(x, :, 1), σ2, Fac = lu(sparse(I - 0.4 * W)),
       ref = tensor_pspline(u, v; k = 30), pss = [tensor_pspline(u, v; k) for k in KS])
end

hlen(r) = r.root_found && isfinite(r.ci_bias_aware[1]) ? (r.ci_bias_aware[2] - r.ci_bias_aware[1]) / 2 : Inf

function cell(d, amp, mult, R; reference = true, bonferroni = false, seed = 21)
    g(s, t) = amp * sin(2π * s) * cos(π * t)
    f = d.X[:, 1] .+ g.(d.u, d.v)
    Bs = [p.basis for p in d.pss]; Ps = [p.penalty for p in d.pss]
    ℛ = mult * basis_roughness(g, d.u, d.v; k = 30)
    ℛk = [basis_roughness(f, p.basis, p.penalty; X = d.X) for p in d.pss]
    rng = MersenneTwister(seed + round(Int, 10amp) + round(Int, mult))
    cov = 0; hl = Float64[]; e = Float64[]; picks = zeros(Int, length(KS)); m = 0
    for _ in 1:R
        y = d.Fac \ (f .+ sqrt.(d.σ2) .* randn(rng, d.n))
        r, j = if reference
            gr = root_rho(y, d.X, d.W; basis = Bs, penalty = Ps, lambda = λS, roughness = ℛ,
                          reference = d.ref, bonferroni)
            gr, gr.basis_index
        else
            fits = [root_rho(y, d.X, d.W; basis = Bs[k], penalty = Ps[k], lambda = λS, roughness = ℛk[k])
                    for k in eachindex(Bs)]
            jj = argmin(hlen.(fits)); fits[jj], jj
        end
        isfinite(hlen(r)) || continue
        m += 1; lo, hi = r.ci_bias_aware
        cov += lo <= 0.4 <= hi; push!(hl, (hi - lo) / 2); push!(e, r.rho - 0.4); picks[j] += 1
    end
    (; ℛ, ℛk, cov = cov / m, m, hl = mean(hl), bias = mean(e), sd = std(e), picks)
end

function main(R)
    d = design()
    println("root_rho with the basis chosen by the interval's length (Corollary 7′); Julia $VERSION")
    println("n = $(d.n), kNN-6, bases k = $(join(KS, ", ")), λ grid 10^(-3:0.5:2) plus REsd for each, reference k̃ = 30, R = $R\n")
    t = @elapsed cell(d, 4.0, 1.0, 1)
    @printf("one fit over the three bases with a reference: %.1f s (with compilation)\n\n", t)
    pass = true; power = true
    cells = ((4.0, 1.0, true, false), (1.0, 1.0, true, false), (4.0, 4.0, true, false),
             (4.0, 1.0, true, true), (4.0, 1.0, false, false))
    for (j, (amp, mult, ref, bon)) in enumerate(cells)
        t0 = time()
        c = cell(d, amp, mult, R; reference = ref, bonferroni = bon)
        ok = ref ? 0.93 <= c.cov <= 1.0 : c.cov < 0.90
        ref ? (pass &= ok) : (power &= ok)
        what = ref ? @sprintf("reference, %.0f ℛ̃* (ℛ̃ = %.3f)%s", mult, c.ℛ, bon ? ", Bonferroni over 39" : "") :
                     "no reference, LS ℛ_k = " * join([@sprintf("%.2f", r) for r in c.ℛk], "/")
        @printf("%d. a = %.0f, %s: coverage %.3f (%d fits), half-length %.4f, bias %+.4f, sd %.4f | %s | %.1f min\n",
                j, amp, what, c.cov, c.m, c.hl, c.bias, c.sd, ok ? "ok" : "FAIL", (time() - t0) / 60)
        println("    basis chosen (k: count) ", join(["$(KS[i]):$(c.picks[i])" for i in eachindex(KS)], " "))
    end
    println("\nPass criterion (cells 1–4 within 0.93–1.00): ", pass ? "met" : "NOT met")
    println("Power check (cell 5 below 0.90): ", power ? "met" : "NOT met")
end

main(isempty(ARGS) ? 200 : parse(Int, ARGS[1]))
