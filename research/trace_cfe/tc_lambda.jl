# Choosing λ (THEORY.md, "Still to do" item 4): GCV, and whether c = β/√V can be estimated.
#
# Proposition 3 and 4: the interval is valid when c(λ) = β(λ)/√V(λ) is small, and sd(ρ̂) grows as λ falls
# (tc_smoother.jl: 0.058 at λ = 1, 0.122 at λ = 0.001), so there is a trade-off. Two questions:
# (1) GCV. A pilot fit at λ_p = 10⁻³ gives ρ̂_p and y₀p = y − ρ̂_p Wy; GCV(λ) = n‖M_λ y₀p‖²/(n − tr S_λ)²
#     picks λ_G on the grid; the TC-CFE at λ_G with the general V̂ gives coverage. Does GCV over-smooth?
# (2) A plug-in ĉ(λ) = (f̃′A_λ f̃ − tr(A_λ S_p Σ̂ S_p))/√V̂_λ with f̃ = S_p y₀p and Σ̂ = diag(û_p²/(M_p²)ᵢᵢ):
#     its mean and sd across replications against the exact c(λ). If its sd is of order 1, no plug-in
#     rule can tell c = 0.3 from c = 0; it can only flag a large c.
# Also reported per λ: the RMSE of ρ̂ and the coverage, so the RMSE-best λ and the coverage-valid range
# can be read off. Same design as tc_smoother.jl (n = 900, tensor P-spline, kNN W), R = 500.
include(joinpath(@__DIR__, "tc_lib.jl"))
using Printf, Distributions

hat(x, k) = (t = range(0, 1, length = k); hcat([max.(0, 1 .- abs.(x .- t[j]) .* (k - 1)) for j in 1:k]...))
hatbasis(u, v, k) = (Hu = hat(u, k); Hv = hat(v, k); hcat([Hu[:, i] .* Hv[:, j] for i in 1:k for j in 1:k]...))

function run(; n = 900, kh = 15, amp = 1.0, R = 500, ρ0 = 0.4, seed = 5,
               λs = 10.0 .^ (-3:0.5:2))
    rng = MersenneTwister(seed)
    P = rand(rng, n, 2); W = knnW(P, 6); u, v = P[:, 1], P[:, 2]; Wd = Matrix(W)
    x = randn(rng, n)
    B = hcat(hatbasis(u, v, kh), x)
    D2 = diff(diff(Matrix(1.0I, kh, kh), dims = 1), dims = 1)
    Ωh = kron(D2' * D2, Matrix(1.0I, kh, kh)) + kron(Matrix(1.0I, kh, kh), D2' * D2)
    Ωp = zeros(kh^2 + 1, kh^2 + 1); Ωp[1:kh^2, 1:kh^2] = Ωh
    f = amp .* sin.(2π .* u) .* cos.(π .* v) .+ x
    σ2 = exp.(3 .* (u .+ v .- 1)); σ2 ./= mean(σ2)
    L = length(λs)
    Ss = Vector{Matrix{Float64}}(undef, L); As = similar(Ss); As2s = similar(Ss); Hs = similar(Ss)
    m2s = Vector{Vector{Float64}}(undef, L); cs = zeros(L); dfs = zeros(L)
    for (j, λ) in enumerate(λs)
        S = B * ((B' * B + λ * Ωp) \ B'); S = (S + S') / 2
        M = I - S; d = (M .* M) \ (-diag(M * Wd)); A = M * Wd + M * Diagonal(d) * M
        Asym = (A + A') / 2
        Ss[j] = S; As[j] = A; As2s[j] = Asym .^ 2; Hs[j] = (A + A') * S; m2s[j] = diag(M * M)
        b = (A + A') * f
        cs[j] = dot(f, A * f) / sqrt(2 * dot(σ2, As2s[j] * σ2) + sum(b .^ 2 .* σ2)); dfs[j] = tr(S)
    end
    jp = 1                                                  # pilot: the smallest λ
    Sp = Ss[jp]
    dSAS = [diag(Sp * As[j] * Sp) for j in 1:L]             # tr(A S_p Σ̂ S_p) = ŝ′ diag(S_p A S_p)
    Fac = lu(sparse(I - ρ0 * W))
    vh(j, y0, s) = (H = Hs[j]; â = H * y0; noise = vec(sum((H .^ 2) .* s', dims = 2));
                    2 * dot(s, As2s[j] * s) + sum((â .^ 2 .- noise) .* s))
    ρh = zeros(R, L); cov = zeros(Bool, R, L); chat = zeros(R, L); jG = zeros(Int, R)
    y0p = zeros(n); sp = zeros(n)
    for r in 1:R
        y = Fac \ (f .+ sqrt.(σ2) .* randn(rng, n)); Wy = W * y
        for j in 1:L
            A = As[j]
            ρ, a, bq = cfe_root(y, A * y, Wy, A * Wy); ρh[r, j] = ρ
            y0 = y .- ρ .* Wy; û = y0 .- Ss[j] * y0
            Vg = vh(j, y0, û .^ 2 ./ m2s[j])
            cov[r, j] = Vg > 0 && abs(ρ - ρ0) <= 1.96 * sqrt(Vg) / abs(bq - 2ρ * a)
            if j == jp
                y0p .= y0; sp .= û .^ 2 ./ m2s[j]
            end
        end
        ft = Sp * y0p
        gcv = [n * sum(abs2, y0p .- Ss[j] * y0p) / (n - dfs[j])^2 for j in 1:L]
        jG[r] = argmin(gcv)
        for j in 1:L
            A = As[j]
            tcorr = dot(sp, dSAS[j])
            chat[r, j] = (dot(ft, A * ft) - tcorr) / sqrt(max(vh(j, y0p, sp), 1e-12))
        end
    end
    @printf("amp %.0f (pilot λ = %.0e, R = %d)\n", amp, λs[jp], R)
    for j in 1:L
        e = ρh[:, j] .- ρ0
        @printf("  λ %8.4f tr S %6.1f | c %+7.3f ĉ mean %+7.3f sd %.3f | ρ̂ bias %+.4f sd %.4f RMSE %.4f | coverage %.3f | GCV picks %3d\n",
                λs[j], dfs[j], cs[j], mean(chat[:, j]), std(chat[:, j]), mean(e), std(e), sqrt(mean(e .^ 2)),
                mean(cov[:, j]), count(==(j), jG))
    end
    covG = mean(cov[r, jG[r]] for r in 1:R); eG = [ρh[r, jG[r]] - ρ0 for r in 1:R]
    @printf("  GCV choice: coverage %.3f, ρ̂ bias %+.4f sd %.4f RMSE %.4f\n\n", covG, mean(eG), std(eG), sqrt(mean(eG .^ 2)))
end

run(; R = 3, λs = [1e-3, 1.0]); println("(warm-up above)\n")
t0 = time()
for amp in (1.0, 4.0)
    run(; amp)
end
println("total $(round((time() - t0) / 60, digits = 1)) min")
