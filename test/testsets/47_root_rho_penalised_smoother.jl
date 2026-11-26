@testset "47. root_rho with a penalised smoother" begin
    _progress("47. root_rho with a penalised smoother")
    RR = SpatialEconometrics.RootRho
    # Stratified locations: one per cell of a g × g grid, so no knot cell is empty.
    function design47(g, k, kw, seed)
        rng = MersenneTwister(seed); n = g^2
        P = hcat(repeat(0:g-1, inner = g), repeat(0:g-1, outer = g)) .+ rand(rng, n, 2); P ./= g
        (; n, P, W = knn_weights(P, kw), x = randn(rng, n),
           ps = tensor_pspline(P[:, 1], P[:, 2]; k), rng)
    end
    function draw47(d, ρ0, rng; amp = 1.0)
        f = 0.5 .* d.x .+ amp .* sin.(2π .* d.P[:, 1]) .* cos.(π .* d.P[:, 2])
        σ2 = exp.(3 .* (d.P[:, 1] .+ d.P[:, 2] .- 1)); σ2 ./= mean(σ2)
        lu(sparse(I - ρ0 * d.W)) \ (f .+ sqrt.(σ2) .* randn(rng, d.n))
    end
    # The dense reference: S from the penalised normal equations, A = MW + MDM.
    function ref47(d, λ)
        Z = hcat(d.x, d.ps.basis); K = size(Z, 2)
        Ωt = zeros(K, K); Ωt[2:end, 2:end] = d.ps.penalty
        G = Z' * Z + λ * Ωt
        S = Z * (G \ Z'); S = (S + S') / 2; M = I - S; Wd = Matrix(d.W)
        dd = (M .* M) \ (-diag(M * Wd))
        A = M * Wd + M * Diagonal(dd) * M
        (; Z, G, S, M, A, H = (A + A') * S, m = diag(M * M))
    end

    @testset "47a. tensor_pspline" begin
        rng = MersenneTwister(47_001); u = rand(rng, 200); v = rand(rng, 200)
        k = 6; ps = tensor_pspline(u, v; k)
        @test size(ps.basis) == (200, 36) && size(ps.penalty) == (36, 36)
        @test maximum(abs, sum(ps.basis; dims = 2) .- 1) < 1e-12
        @test all(>=(0), ps.basis)
        # Column (i − 1)k + j is hᵢ(u)·hⱼ(v), checked at one unit.
        i1 = argmin(u)
        hu = max.(0, 1 .- abs.(u[i1] .- range(extrema(u)...; length = k)) ./ ((maximum(u) - minimum(u)) / (k - 1)))
        hv = max.(0, 1 .- abs.(v[i1] .- range(extrema(v)...; length = k)) ./ ((maximum(v) - minimum(v)) / (k - 1)))
        @test ps.basis[i1, :] ≈ vec([hu[i] * hv[j] for j in 1:k, i in 1:k]) atol = 1e-12
        Ω = ps.penalty
        @test Ω == Ω' && eigmin(Symmetric(Ω)) > -1e-10
        # Order 2 leaves 1, i, j and ij unpenalised, and nothing else.
        ci, cj = repeat(1.0:k, inner = k), repeat(1.0:k, outer = k)
        for c in (ones(k^2), ci, cj, ci .* cj)
            @test maximum(abs, Ω * c) < 1e-10
        end
        @test maximum(abs, Ω * (ci .^ 2)) > 1
        @test rank(Ω) == k^2 - 4
        @test rank(tensor_pspline(u, v; k, order = 1).penalty) == k^2 - 1
        @test_throws "too few" tensor_pspline(u, v; k = 3)
        @test_throws DimensionMismatch tensor_pspline(u, v[1:end-1]; k)
        @test_throws "constant" tensor_pspline(u, fill(0.5, 200); k)
    end

    d = design47(12, 6, 6, 47_002)                      # n = 144, K = 37
    y = draw47(d, 0.4, d.rng); Wy = d.W * y
    X1 = reshape(d.x, :, 1); λ = 0.3
    r = root_rho(y, X1, d.W; d.ps..., lambda = λ, varnames = ["x"])
    R0 = ref47(d, λ)
    function vhat_ref(ρ)
        y0 = y - ρ * Wy; s = (R0.M * y0) .^ 2 ./ R0.m
        2 * s' * (((R0.A + R0.A') ./ 2) .^ 2) * s +
            sum(((R0.H * y0) .^ 2 .- vec(sum(R0.H .^ 2 .* s'; dims = 2))) .* s)
    end

    @testset "47b. against a dense reimplementation" begin
        @test maximum(abs, diag(RR._exact_diag_moment(Matrix(R0.M), d.W))) < 1e-12
        @test maximum(abs, RR._exact_diag_moment(Matrix(R0.M), d.W) .- R0.A) < 1e-10
        a, b, c = Wy' * R0.A * Wy, y' * R0.A * Wy + Wy' * R0.A * y, y' * R0.A * y
        ρ = (b - sqrt(b^2 - 4a * c)) / (2a)
        @test r.smoother === :penalised && r.lambda == λ && isnan(r.rho_first)
        @test [r.a, r.b, r.c] ≈ [a, b, c] rtol = 1e-10
        @test r.rho ≈ ρ atol = 1e-10
        @test r.se ≈ sqrt(vhat_ref(ρ)) / abs(b - 2ρ * a) atol = 1e-10
        @test r.beta ≈ (R0.G \ (R0.Z' * (y - ρ * Wy)))[1:1] atol = 1e-10
        @test r.df ≈ tr(R0.S) atol = 1e-10
        @test r.max_leverage ≈ maximum(diag(R0.S)) atol = 1e-12
        for t in (-0.5, 0.1, 0.7)
            @test RR._polyval(r.vpoly, t) ≈ vhat_ref(t) rtol = 1e-10
        end
        # With a projection basis (M² = M) the general V̂ is the projection's V̂.
        Q = Matrix(qr(R0.Z).Q)[:, 1:size(R0.Z, 2)]; Mp = I - Q * Q'
        Cp = RR._moment_matrix(Matrix(Matrix(d.W)'), Q, diag(Mp))
        @test collect(RR._vhat_poly_general(Cp, (Cp + Cp') * Q * Q', Matrix(Mp), diag(Mp), y, Wy)) ≈
              collect(RR._vhat_poly(Cp, Q, diag(Mp), y, Wy)) rtol = 1e-9
    end

    @testset "47c. the test-inversion set's endpoints" begin
        z = 1.959963984540054
        q(t) = (r.c - r.b * t + r.a * t^2)^2 - z^2 * RR._polyval(r.vpoly, t)
        sc = z^2 * RR._polyval(r.vpoly, r.rho)
        # A fixed number of assertions, whatever the number of intervals or
        # whether the set touches a bound: a guarded @test that does not run
        # leaves no trace but the count, and the count differs by RNG stream.
        @test !isempty(r.cs)
        ends = [e for (lo, hi) in r.cs for e in (lo, hi)]
        @test any(e ∉ r.rho_bounds for e in ends)     # the next check is not vacuous
        @test all(e in r.rho_bounds || abs(q(e)) < 1e-8 * sc for e in ends)
        @test all(all(q(t) <= 0 for t in range(lo, hi; length = 9)[2:end-1]) for (lo, hi) in r.cs)
        @test all(lo == r.rho_bounds[1] || q(lo - 1e-4) > 0 for (lo, hi) in r.cs)
        @test all(hi == r.rho_bounds[2] || q(hi + 1e-4) > 0 for (lo, hi) in r.cs)
        @test any(lo <= r.rho <= hi for (lo, hi) in r.cs)
    end

    @testset "47d. the moment's mean is f′Af exactly (power: the shortcut moment)" begin
        # n = 100, k = 5, λ = 0.1, heteroskedastic normal errors, at ρ₀.
        # f₀ = 0.5x + B·(coefficients the penalty leaves alone), so Sf₀ = f₀ and
        # E g(ρ₀) = 0. f₁ = f₀ + a smooth surface the smoother shrinks, so
        # E g(ρ₀) = f₁′Af₁ ≠ 0 (Proposition 4). The shortcut moment
        # y₀′(M²W + MDM)y₀ has a nonzero mean for f₀ and fails the same test.
        e = design47(10, 5, 6, 7); n = e.n; Re = ref47(e, 0.1)
        Wd = Matrix(e.W); M = Re.M; A = Re.A
        Acut = M * M * Wd + M * Diagonal((M .* M) \ (-diag(M * Wd))) * M
        σ2 = exp.(3 .* (e.P[:, 1] .+ e.P[:, 2] .- 1)); σ2 ./= mean(σ2); Σ = Diagonal(σ2)
        ci, cj = repeat(1.0:5, inner = 5), repeat(1.0:5, outer = 5)
        f0 = 0.5 .* e.x .+ e.ps.basis * (1 .+ 0.3 .* ci .- 0.2 .* cj .+ 0.1 .* ci .* cj)
        f1 = f0 .+ 2 .* sin.(2π .* e.P[:, 1]) .* cos.(π .* e.P[:, 2])
        @test maximum(abs, M * f0) < 1e-12 && abs(tr(A * Σ)) < 1e-12
        Vx(C, f) = (Cs = (C + C') / 2; 2 * tr(Cs * Σ * Cs * Σ) + dot((C + C') * f, σ2 .* ((C + C') * f)))
        V0, V1, Vc = Vx(A, f0), Vx(A, f1), Vx(Acut, f0); β1 = dot(f1, A * f1)
        # The shortcut's exact mean, standardised, is far from rounding.
        @test abs(dot(f0, Acut * f0) + tr(Acut * Σ)) / sqrt(Vc) > 0.2
        rng = MersenneTwister(47_004); R = 4000
        z0 = zeros(R); z1 = zeros(R); zc = zeros(R)
        for k in 1:R
            ε = sqrt.(σ2) .* randn(rng, n)
            z0[k] = dot(f0 + ε, A * (f0 + ε)) / sqrt(V0)
            z1[k] = (dot(f1 + ε, A * (f1 + ε)) - β1) / sqrt(V1)
            zc[k] = dot(f0 + ε, Acut * (f0 + ε)) / sqrt(Vc)
        end
        @test abs(mean(z0)) < 4 / sqrt(R)
        @test abs(var(z0) - 1) < 4 * sqrt(2 / R)
        @test abs(mean(z1)) < 4 / sqrt(R)
        @test abs(var(z1) - 1) < 4 * sqrt(2 / R)
        @test abs(mean(zc)) > 4 / sqrt(R)                 # power: the shortcut moment
        @test abs(mean(z1) + β1 / sqrt(V1)) > 4 / sqrt(R)  # power: f₁′Af₁ is not negligible
    end

    @testset "47e. recovery at n = 900" begin
        e = design47(30, 12, 6, 47_005)                  # K = 145
        ye = draw47(e, 0.4, e.rng)
        re = root_rho(ye, reshape(e.x, :, 1), e.W; e.ps..., lambda = 0.01)
        @test abs(re.rho - 0.4) < 4 * re.se
        @test re.se < 0.15
        @test abs(re.beta[1] - 0.5) < 0.15
        @test re.K == 145 && re.n == 900 && re.root_found && re.df < re.K
    end

    @testset "47f. guards" begin
        B = d.ps.basis; Ω = d.ps.penalty
        @test_throws "give both" root_rho(y, X1, d.W; basis = B, penalty = Ω)
        @test_throws "give both" root_rho(y, X1, d.W; basis = B, lambda = 1.0)
        @test_throws "needs a basis" root_rho(y, X1, d.W; penalty = Ω, lambda = 1.0)
        @test_throws "positive and finite" root_rho(y, X1, d.W; basis = B, penalty = Ω, lambda = 0.0)
        @test_throws "positive and finite" root_rho(y, X1, d.W; basis = B, penalty = Ω, lambda = Inf)
        @test_throws DimensionMismatch root_rho(y, X1, d.W; basis = B, penalty = Ω[1:end-1, 1:end-1], lambda = 1.0)
        Ωa = copy(Ω); Ωa[1, 2] += 1
        @test_throws "symmetric" root_rho(y, X1, d.W; basis = B, penalty = Ωa, lambda = 1.0)
        @test_throws "semi-definite" root_rho(y, X1, d.W; basis = B, penalty = -Ω, lambda = 1.0)
        # An unpenalised indicator column gives Sᵢᵢ = 1 at unit 5.
        e1 = zeros(d.n); e1[5] = 1
        Ωe = zeros(size(Ω, 1) + 1, size(Ω, 1) + 1); Ωe[1:end-1, 1:end-1] = Ω
        @test_throws "Sᵢᵢ above 0.99" root_rho(y, X1, d.W; basis = hcat(B, e1), penalty = Ωe, lambda = 1.0)
        f8 = tensor_pspline(d.P[:, 1], d.P[:, 2]; k = 8)       # 64 columns at n = 144
        @test_logs (:warn, r"df/n") root_rho(y, X1, d.W; f8..., lambda = 0.01)
        @test occursin("penalised smoother", sprint(show, r)) && occursin("df = tr S", sprint(show, r))
        @test !occursin("first step", sprint(show, r))
    end

    @testset "47g. the sign rule's check: J < 0 warns (power: J > 0 does not)" begin
        # A hub design with J = −E g′(ρ₀) < 0. Found by a search over 3,000
        # hub designs (0.7% had J < 0); built from formulas, so the design and
        # J are the same on every Julia version. Six hubs point to fixed units,
        # every other unit to hub mod1(4i, 6); f lies in the span of Z, and the
        # smoother is the projection (penalty 0), so Mf = 0.
        n = 100; Wh = zeros(n, n)
        for (h, t) in enumerate((35, 4, 37, 96, 15, 9)); Wh[h, t] = 1; end
        for i in 7:n; Wh[i, mod1(4i, 6)] = 1; end
        Wh = sparse(Wh); xh = sin.(1.28 .* (1:n)); Zh = hcat(ones(n), xh, xh .^ 2)
        fh = Zh * [-4.2, -3.9, 4.2]; ρh = -0.9
        Qh = Matrix(qr(Zh).Q)[:, 1:3]; Mh = I - Qh * Qh'
        Ah = RR._exact_diag_moment(Matrix(Mh), Wh); Ahs = (Ah + Ah') / 2
        Gh = Matrix(Wh) / (I - ρh * Matrix(Wh))
        Jh = 2 * (dot(fh, Gh' * Ahs * fh) + tr(Gh' * Ahs)) / n
        @test Jh < -20                                   # −24.33, exact
        yh = (I - ρh * Matrix(Wh)) \ (fh .+ randn(MersenneTwister(1), n))
        fit_h = @test_logs (:warn, r"slope g′ is positive") match_mode = :any root_rho(
            yh, Zh[:, 2:3], Wh; basis = Zh[:, 1:1], penalty = zeros(1, 1), lambda = 1.0)
        # Why it matters: the sign rule took the inconsistent root; the other
        # root is the one near ρ₀ (0.009 against 0.063 on 1.13, 0.001 against
        # 0.055 on 1.10). ρ₀ is still in the test-inversion set.
        other = (fit_h.b + sqrt(fit_h.b^2 - 4fit_h.a * fit_h.c)) / (2fit_h.a)
        @test abs(other - ρh) < abs(fit_h.rho - ρh) / 3
        @test any(lo <= ρh <= hi for (lo, hi) in fit_h.cs)
        # No preliminary estimate when WZ adds nothing to span Z (W1 = 1).
        @test RR._prelim_2sls(yh, Wh * yh, ones(n, 1), Wh) === nothing
        # Power: on this testset's design (J > 0) neither path logs anything.
        @test_logs root_rho(y, X1, d.W; d.ps..., lambda = λ)
        @test_logs root_rho(y, X1, d.W; basis = d.ps.basis)
    end

    @testset "47h. an irregular region: a rank-deficient basis" begin
        # Units fill the triangle u + v ≤ 1 of the unit square, so the hats
        # above the diagonal have no unit in their support and [x, B] is
        # rank-deficient. The smoother needs only Z′Z + λΩ̃ invertible (found
        # on the Portuguese municipalities, research/trace_cfe/tc_portugal.jl).
        rng = MersenneTwister(47_008); g = 24
        P = hcat(repeat(0:g-1, inner = g), repeat(0:g-1, outer = g)) .+ rand(rng, g^2, 2); P ./= g
        P = P[P[:, 1] .+ P[:, 2] .<= 1, :]
        P = vcat(P, [0.0 1.0; 1.0 0.0])        # the box's corners, so the knots span [0, 1]²
        n = size(P, 1); Wt = knn_weights(P, 6); x = randn(rng, n)
        ps = tensor_pspline(P[:, 1], P[:, 2]; k = 10)
        Z = hcat(x, ps.basis); K = size(Z, 2)
        @test rank(Z) < K - 10                   # many columns are empty or dependent
        @test count(iszero, vec(maximum(ps.basis; dims = 1))) > 10
        f = 0.5 .* x .+ sin.(2π .* P[:, 1]) .* cos.(π .* P[:, 2])
        yt = lu(sparse(I - 0.4 * Wt)) \ (f .+ randn(rng, n))
        Wyt = Wt * yt; λt = 1.0
        rt = root_rho(yt, reshape(x, :, 1), Wt; ps..., lambda = λt)
        # Against a dense reimplementation (as 47b), with G = Z′Z + λΩ̃ positive definite.
        Ωt = zeros(K, K); Ωt[2:end, 2:end] = ps.penalty
        G = Z' * Z + λt * Ωt
        @test isposdef(Symmetric(G))
        S = Z * (G \ Z'); S = (S + S') / 2; M = I - S
        dd = (M .* M) \ (-diag(M * Matrix(Wt)))
        A = M * Matrix(Wt) + M * Diagonal(dd) * M
        a, b, c = Wyt' * A * Wyt, yt' * A * Wyt + Wyt' * A * yt, yt' * A * yt
        @test [rt.a, rt.b, rt.c] ≈ [a, b, c] rtol = 1e-9
        @test rt.rho ≈ (b - sqrt(b^2 - 4a * c)) / (2a) atol = 1e-9
        @test rt.df ≈ tr(S) atol = 1e-9
        @test rt.max_leverage < 0.99 && !isempty(rt.cs)
        # A zero column whose penalty is decoupled (Ω entry 1, no cross terms)
        # adds a coefficient the penalty sets to 0, so the fit cannot change.
        kb = size(ps.basis, 2)
        Ωz = zeros(kb + 1, kb + 1); Ωz[1:kb, 1:kb] = ps.penalty; Ωz[end, end] = 1
        rz = root_rho(yt, reshape(x, :, 1), Wt; basis = hcat(ps.basis, zeros(n)), penalty = Ωz, lambda = λt)
        @test rz.rho ≈ rt.rho atol = 1e-9
        @test rz.se ≈ rt.se atol = 1e-9
        @test collect(rz.vpoly) ≈ collect(rt.vpoly) rtol = 1e-8
        # The preliminary 2SLS uses span(Z) only, so dropping the empty columns
        # cannot change it.
        used = vec(maximum(ps.basis; dims = 1)) .> 0
        p_all = RR._prelim_2sls(yt, Wyt, Z, Wt)
        p_used = RR._prelim_2sls(yt, Wyt, hcat(x, ps.basis[:, used]), Wt)
        @test p_all !== nothing
        @test p_all ≈ p_used atol = 1e-8
        # Power: the projection on the same basis is still refused, and the
        # message names the cause and the remedy.
        @test_throws "penalised smoother" root_rho(yt, reshape(x, :, 1), Wt; basis = ps.basis)
        # An intercept in X is collinear with the constant, which the penalty
        # leaves free: Z′Z + λΩ̃ is singular.
        @test_throws "intercept" root_rho(yt, hcat(ones(n), x), Wt; ps..., lambda = λt)
        # X itself must have full rank: it is not penalised.
        @test_throws "not penalised" root_rho(yt, hcat(x, 2 .* x), Wt; ps..., lambda = λt)
    end

    @testset "47i. the positive-definiteness check does not depend on units" begin
        # The check compared the eigenvalues of Z′Z + λΩ̃ themselves, so a column of X in large
        # units (a percentage, a price) made a well-conditioned G look singular (AUDIT 38, found
        # on the Dutch districts). The fit is invariant to the scale of X's columns, and so must
        # the check be.
        rng = MersenneTwister(47_009); g = 16; n = g^2
        P = hcat(repeat(0:g-1, inner = g), repeat(0:g-1, outer = g)) .+ rand(rng, n, 2); P ./= g
        Wi = knn_weights(P, 6); x = randn(rng, n)
        ps = tensor_pspline(P[:, 1], P[:, 2]; k = 8)
        f = 0.5 .* x .+ sin.(2π .* P[:, 1]) .* cos.(π .* P[:, 2])
        yi = lu(sparse(I - 0.4 * Wi)) \ (f .+ randn(rng, n))
        c = 1e4; λi = 0.01
        Z = hcat(c .* x, ps.basis); K = size(Z, 2)
        Ωt = zeros(K, K); Ωt[2:end, 2:end] = ps.penalty
        ev = eigvals(Symmetric(Z' * Z + λi * Ωt))
        @test ev[1] < 1e-10 * ev[end]           # power: the old rule refused this design
        @test isposdef(Symmetric(Z' * Z + λi * Ωt))
        r1 = root_rho(yi, reshape(x, :, 1), Wi; ps..., lambda = λi)
        rc = root_rho(yi, reshape(c .* x, :, 1), Wi; ps..., lambda = λi)
        @test rc.rho ≈ r1.rho rtol = 1e-8
        @test rc.se ≈ r1.se rtol = 1e-8
        @test rc.vhat ≈ r1.vhat rtol = 1e-8
        @test length(rc.cs) == length(r1.cs) && all(isapprox.(first.(rc.cs), first.(r1.cs); atol = 1e-6))
        @test rc.beta[1] * c ≈ r1.beta[1] rtol = 1e-6
        # The genuine case is still refused at any scale: an intercept with the hat basis.
        @test_throws "intercept" root_rho(yi, hcat(fill(c, n), x), Wi; ps..., lambda = λi)
    end
end
