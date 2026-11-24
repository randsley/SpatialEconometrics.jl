@testset "52. root_rho: the HC2 V̂'s worst-case bias with skewed errors (Lemma 12)" begin
    _progress("52. root_rho: the HC2 V̂'s worst-case bias with skewed errors (Lemma 12)")
    # Testset 51's layout (n = 196) with kNN-10 and a tensor P-spline of k = 5 (K = 26), a
    # design where skewness doubles the worst case (testset 51's design has worst γ = 0).
    RR = SpatialEconometrics.RootRho
    rng = MersenneTwister(49_001); g = 14; n = g^2
    P = hcat(repeat(0:g-1, inner = g), repeat(0:g-1, outer = g)) .+ rand(rng, n, 2); P ./= g
    u, v = P[:, 1], P[:, 2]; W = knn_weights(P, 10); Wd = Matrix(W)
    x = randn(rng, n); X = reshape(x, :, 1)
    gs(s, t) = 2 * sin(2π * s) * cos(π * t)
    y = lu(sparse(I - 0.4 * W)) \ (0.5 .* x .+ gs.(u, v) .+ randn(rng, n))
    B = tensor_pspline(u, v; k = 5).basis
    Z = hcat(x, B); Q = Matrix(qr(Z).Q)[:, 1:size(Z, 2)]
    Mii = 1 .- vec(sum(abs2, Q; dims = 2))
    γ̄ = 2.0

    # Lemma 12 densely: its terms from M, Ω and H, the projection on range(C′Z) by pinv, and
    # the minimum over (γ, ‖b‖) by brute force on a grid, not by the eigenvalue formula.
    function parts(C)
        M = I - Q * Q'; Ω = ((C + C') ./ 2) .^ 2; H = (C + C') * (Q * Q'); M2 = M .^ 2
        H2 = H .^ 2; DD = Mii .* Mii'; U = M2 ./ Mii
        A = 4 * sum(Ω .* M2 ./ DD) - 2 * sum(H2 .* M2 ./ DD)
        sc = sum(2 .* sum(U .* (Ω * U); dims = 1) .+ sum(H2 .* M2 ./ Mii; dims = 1) .-
                 sum(U .* (H2 * U); dims = 1))
        t1 = vec(sum(H .* U; dims = 2)); CZ = C' * Z; pt = CZ * (pinv(CZ) * t1)
        (; A, sc, t1, pt, τ = norm(pt), V0 = 2 * sum(Ω), CZ)
    end
    function brute(p)
        ss = sqrt(p.V0) .* 10 .^ range(-4, 4, length = 20001)
        lo = Inf; γb = 0.0; sb = 0.0
        for γ in range(0, γ̄, length = 401)
            a = p.A - 2p.sc + γ^2 * p.sc
            for s in ss
                val = (a - 2γ * p.τ * s) / (p.V0 + s^2)
                val < lo && (lo = val; γb = γ; sb = s)
            end
        end
        lo, γb, sb
    end

    @testset "52a. root_rho reports Lemma 12 at its final moment matrix" begin
        r0 = root_rho(y, X, W; basis = B)
        r2 = root_rho(y, X, W; basis = B, skewness = γ̄)
        @test r0.root_found && r2.root_found
        Gt = Matrix((I - r2.rho_first * Wd)' \ Wd')
        M = I - Q * Q'
        p = parts((Gt .- Diagonal(diag(Gt * M) ./ Mii)) * M)
        βb, γb, _ = brute(p)
        @test r2.vhat_worst_bias ≈ βb rtol = 1e-5
        @test γb > 0                                     # skewness bites in this design
        βsym = min(0, p.A - 2p.sc) / p.V0
        @test r0.vhat_worst_bias ≈ βsym rtol = 1e-9      # skewness = 0 is Lemma 8
        @test r2.vhat_worst_bias < 1.5 * r0.vhat_worst_bias          # power: much worse
        @test r2.vhat_worst_bias >= βsym - γ̄ * p.τ / sqrt(p.V0) - 1e-12   # Lemma 12(c)
        @test r0.skewness == 0 && r2.skewness == γ̄
        # A diagnostic only: the estimate and the intervals are unchanged.
        @test r2.rho == r0.rho && r2.cs == r0.cs && r2.ci_wald == r0.ci_wald
        rp = root_rho(y, X, W; basis = B, penalty = tensor_pspline(u, v; k = 5).penalty,
                      lambda = 1.0, skewness = γ̄)
        @test isnan(rp.vhat_worst_bias) && isnan(rp.skewness)
        @test_throws ArgumentError root_rho(y, X, W; basis = B, skewness = -1)
        @test_throws ArgumentError root_rho(y, X, W; basis = B, skewness = Inf)
    end

    @testset "52b. two-point errors at the worst f reach it (Monte Carlo, R = 20000)" begin
        C = RR._moment_matrix(Matrix((I - 0.4 * Wd)' \ Wd'), Q, Mii)
        p = parts(C)
        βb, γ, s = brute(p)
        @test RR._hc2_worst_bias(C, Q, Mii, γ̄) ≈ βb rtol = 1e-5
        b = -s .* p.pt ./ p.τ                            # the worst signal C′f
        fz = Z * (p.CZ \ b); bb = C' * fz                # an f in span[X, B] with C′f = b
        @test norm(bb - b) <= 1e-8 * norm(b)
        V = p.V0 + sum(abs2, bb)
        pred = 1 + (p.A + p.sc * (γ^2 - 2) + 2γ * dot(bb, p.t1)) / V
        @test pred - 1 ≈ βb rtol = 1e-6
        # The two-point law of skewness γ: excess kurtosis γ² − 2, Pearson's bound.
        q = (1 - γ / sqrt(γ^2 + 4)) / 2; hi = sqrt((1 - q) / q); lo = -sqrt(q / (1 - q))
        mc = MersenneTwister(52_001); R = 20000
        r = [RR._polyval(RR._vhat_poly(C, Q, Mii, fz .+ [rand(mc) < q ? hi : lo for _ in 1:n],
                                       zeros(n)), 0.0) / V for _ in 1:R]
        se = std(r) / sqrt(R)
        @test abs(mean(r) - pred) < 4se                  # E V̂/V = 1 + β_skew
        # power: without the skewness term L₂ the prediction fails
        @test abs(mean(r) - (1 + (p.A + p.sc * (γ^2 - 2)) / V)) > 5se
    end

    @testset "52c. show" begin
        s = sprint(show, root_rho(y, X, W; basis = B, skewness = γ̄))
        @test occursin("|skewness| ≤ 2", s)
        @test !occursin("symmetric", s)
        @test occursin("symmetric homoskedastic errors", sprint(show, root_rho(y, X, W; basis = B)))
    end
end
