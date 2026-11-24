@testset "51. root_rho: the HC2 V̂'s worst-case bias (Lemma 8)" begin
    _progress("51. root_rho: the HC2 V̂'s worst-case bias (Lemma 8)")
    # Testset 49's design (n = 196, kNN-6), with a tensor P-spline basis of k = 6 (K = 37).
    RR = SpatialEconometrics.RootRho
    rng = MersenneTwister(49_001); g = 14; n = g^2
    P = hcat(repeat(0:g-1, inner = g), repeat(0:g-1, outer = g)) .+ rand(rng, n, 2); P ./= g
    u, v = P[:, 1], P[:, 2]; W = knn_weights(P, 6); Wd = Matrix(W)
    x = randn(rng, n); X = reshape(x, :, 1)
    gs(s, t) = 2 * sin(2π * s) * cos(π * t)
    f = 0.5 .* x .+ gs.(u, v)
    y = lu(sparse(I - 0.4 * W)) \ (f .+ randn(rng, n))
    ps = tensor_pspline(u, v; k = 6); B = ps.basis
    Z = hcat(x, B); Q = Matrix(qr(Z).Q)[:, 1:size(Z, 2)]
    Mii = 1 .- vec(sum(abs2, Q; dims = 2))

    # Lemma 8 written out densely, from M, Ω and H directly (THEORY.md).
    function worst_dense(C)
        M = I - Q * Q'; Ω = ((C + C') ./ 2) .^ 2; H2 = ((C + C') * (Q * Q')) .^ 2; M2 = M .^ 2
        Q2 = 4 * sum(Ω .* M2 ./ (Mii .* Mii')); L5G = -2 * sum(H2 .* M2 ./ (Mii .* Mii'))
        U = M2 ./ Mii
        c = vec(2 .* sum(U .* (Ω * U); dims = 1) .+ sum(H2 .* M2 ./ Mii; dims = 1) .-
                sum(U .* (H2 * U); dims = 1))
        return min(0, Q2 + L5G - 2sum(c)) / (2sum(Ω)), sum(c)
    end

    @testset "51a. root_rho reports Lemma 8 at its final moment matrix" begin
        r = root_rho(y, X, W; basis = B)
        @test r.root_found
        # REsd's final C is the moment matrix of G(ρ̃)′, rebuilt here independently.
        Gt = Matrix((I - r.rho_first * Wd)' \ Wd')
        M = I - Q * Q'
        C = (Gt .- Diagonal(diag(Gt * M) ./ Mii)) * M
        βd, sc = worst_dense(C)
        @test sc > 0                                     # (b)'s condition holds here
        @test r.vhat_worst_bias ≈ βd rtol = 1e-9
        @test -0.1 < r.vhat_worst_bias < 0               # negative, and small at K/n = 0.19
        # The same value with the bias-aware interval on a reference basis.
        ref = tensor_pspline(u, v; k = 10); ℛ = basis_roughness(gs, u, v; k = 10)
        rb = root_rho(y, X, W; basis = B, roughness = ℛ, reference = ref)
        @test rb.vhat_worst_bias == r.vhat_worst_bias
        # Lemma 8 does not apply to a penalised smoother.
        rp = root_rho(y, X, W; basis = B, penalty = ps.penalty, lambda = 1.0)
        @test isnan(rp.vhat_worst_bias)
    end

    @testset "51b. Rademacher errors at f = 0 reach it (Monte Carlo, R = 20000)" begin
        C = RR._moment_matrix(Matrix((I - 0.4 * Wd)' \ Wd'), Q, Mii)
        β = RR._hc2_worst_bias(C, Q, Mii)
        @test β ≈ worst_dense(C)[1] rtol = 1e-9
        V = 2 * sum(((C + C') ./ 2) .^ 2)               # Var g at f = 0, σ² = 1
        mc = MersenneTwister(51_001); R = 20000
        r = [RR._polyval(RR._vhat_poly(C, Q, Mii, rand(mc, (-1.0, 1.0), n), zeros(n)), 0.0) / V
             for _ in 1:R]
        se = std(r) / sqrt(R)
        @test abs(mean(r) - (1 + β)) < 4se               # E V̂/V = 1 + β_min
        @test abs(mean(r) - 1) > 5se                     # power: V̂ is biased here
    end

    @testset "51c. the coverage it implies" begin
        # The package's Φ is Abramowitz & Stegun 26.2.17 (error under 7.5e-8), enough for a
        # printed percentage.
        @test RR._coverage_given_bias(0.0, 0.95) ≈ 0.95 atol = 1e-6
        @test RR._coverage_given_bias(-0.02, 0.95) ≈
              2 * cdf(Normal(), quantile(Normal(), 0.975) * sqrt(0.98)) - 1 atol = 1e-6
        @test RR._coverage_given_bias(-0.02, 0.95) < 0.95 - 0.002   # power: 0.1145 × 0.02
        @test RR._coverage_given_bias(-0.02, 0.90) < 0.90
        @test RR._coverage_given_bias(-1.0, 0.95) == 0.0
    end

    @testset "51d. show" begin
        r = root_rho(y, X, W; basis = B)
        s = sprint(show, r)
        @test occursin("HC2 V̂: worst-case bias", s)
        @test occursin(string(round(100 * RR._coverage_given_bias(r.vhat_worst_bias, 0.95); digits = 1), "%"), s)
        rp = root_rho(y, X, W; basis = B, penalty = ps.penalty, lambda = 1.0)
        @test !occursin("worst-case bias", sprint(show, rp))
    end
end
