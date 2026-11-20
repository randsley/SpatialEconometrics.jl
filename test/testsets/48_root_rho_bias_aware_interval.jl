@testset "48. root_rho: the bias-aware interval" begin
    _progress("48. root_rho: the bias-aware interval")
    RR = SpatialEconometrics.RootRho
    # Testset 47's stratified design (one location per cell of a g × g grid).
    function design48(g, k, kw, seed)
        rng = MersenneTwister(seed); n = g^2
        P = hcat(repeat(0:g-1, inner = g), repeat(0:g-1, outer = g)) .+ rand(rng, n, 2); P ./= g
        (; n, P, W = knn_weights(P, kw), x = randn(rng, n),
           ps = tensor_pspline(P[:, 1], P[:, 2]; k), rng)
    end
    surf48(d; amp = 2.0) = 0.5 .* d.x .+ amp .* sin.(2π .* d.P[:, 1]) .* cos.(π .* d.P[:, 2])
    function draw48(d, f, ρ0, rng)
        σ2 = exp.(3 .* (d.P[:, 1] .+ d.P[:, 2] .- 1)); σ2 ./= mean(σ2)
        lu(sparse(I - ρ0 * d.W)) \ (f .+ sqrt.(σ2) .* randn(rng, d.n))
    end
    d = design48(14, 6, 6, 48_001)
    X = reshape(d.x, :, 1); B = d.ps.basis; Ω = d.ps.penalty
    Z = hcat(d.x, B); K = size(Z, 2); n = d.n; Wd = Matrix(d.W)
    Πt = zeros(K, K); Πt[2:end, 2:end] = Ω
    f = surf48(d)
    y = draw48(d, f, 0.4, d.rng)
    ℛ = basis_roughness(f, B, Ω; X)

    # An independent dense reference of Lemma 4's bound and Proposition 7's interval.
    function ref48(λ, ℛv; α = 0.05)
        Q = Matrix(qr(Z).Q)[:, 1:K]; M0 = I - Q * Q'; Mii = diag(M0)
        Δ = Diagonal(diag(Wd' * M0) ./ Mii); C1 = (Wd' - Δ) * M0       # Jin & Lee's first step
        Wy = Wd * y
        a1 = dot(Wy, C1 * Wy); b1 = dot(y, C1 * Wy) + dot(Wy, C1 * y); c1 = dot(y, C1 * y)
        r1 = [(b1 - s * sqrt(b1^2 - 4a1 * c1)) / (2a1) for s in (1, -1)]
        ρ0 = only(filter(r -> -0.99 < r < 0.99 && -b1 + 2a1 * r < 0, r1))
        y0 = y .- ρ0 .* Wy; s0 = (M0 * y0) .^ 2 ./ Mii; fh = Q * (Q' * y0)
        S = Z * ((Z' * Z + λ * Πt) \ Z'); S = (S + S') / 2; M = I - S
        dd = (M .* M) \ (-diag(M * Wd)); Dg = Diagonal(dd)
        Tm = (M * Z)' * Wd
        E = eigen(Symmetric(Πt)); keep = E.values .> 1e-10 * maximum(E.values)
        Uk = E.vectors[:, keep]; Πp = Uk * Diagonal(1 ./ E.values[keep]) * Uk'
        Πph = Uk * Diagonal(1 ./ sqrt.(E.values[keep])) * Uk'
        Υ = Πph * Tm * Q * Q'
        q = dot(Tm * fh, Πp * (Tm * fh)) - sum(s0 .* vec(sum(abs2, Υ; dims = 1)))
        χ = maximum(abs, eigvals(Symmetric(Πph * ((M * Z)' * Dg * (M * Z)) * Πph)))
        bbar = ℛv * sqrt(max(q, 0.0)) + ℛv^2 * χ
        (; bbar, χ, M, Dg, Tm, Πp, Πph,
           cv = t -> find_zero(c -> cdf(Normal(), c - t) - cdf(Normal(), -c - t) - (1 - α),
                               (0.0, t + 10.0)))
    end
    find_zero(g, (lo, hi)) = (for _ in 1:200; m = (lo + hi) / 2; g(m) < 0 ? (lo = m) : (hi = m); end; (lo + hi) / 2)

    @testset "48a. the package against an independent reference" begin
        for λ in (0.1, 1.0)
            r = root_rho(y, X, d.W; basis = B, penalty = Ω, lambda = λ, roughness = ℛ)
            ref = ref48(λ, ℛ)
            @test r.root_found
            @test r.bias_bound ≈ ref.bbar rtol = 1e-8
            h = ref.cv(ref.bbar / sqrt(r.vhat)) * r.se
            @test r.ci_bias_aware[1] ≈ r.rho - h atol = 1e-6
            @test r.ci_bias_aware[2] ≈ r.rho + h atol = 1e-6
            @test r.roughness == ℛ
        end
    end

    @testset "48b. Lemma 4: the bound holds for f in span(Z), and needs its linear term" begin
        ψ = Z \ f; fz = Z * ψ; ℛz = sqrt(dot(ψ, Πt * ψ))
        for λ in (0.1, 1.0, 10.0)
            ref = ref48(λ, ℛz)
            A = ref.M * Wd + ref.M * ref.Dg * ref.M
            β = dot(fz, A * fz)
            v = ref.Tm * fz
            oracle = ℛz * sqrt(dot(v, ref.Πp * v)) + ℛz^2 * ref.χ
            @test abs(β) <= oracle
            # Power: the quadratic term alone does not bound β.
            @test abs(β) > ℛz^2 * ref.χ
        end
    end

    @testset "48c. ℛ = 0 gives the Wald interval; a larger ℛ a wider interval" begin
        r0 = root_rho(y, X, d.W; basis = B, penalty = Ω, lambda = 1.0, roughness = 0.0)
        @test r0.bias_bound == 0
        @test r0.ci_bias_aware[1] ≈ r0.ci_wald[1] atol = 1e-6
        @test r0.ci_bias_aware[2] ≈ r0.ci_wald[2] atol = 1e-6
        r1 = root_rho(y, X, d.W; basis = B, penalty = Ω, lambda = 1.0, roughness = ℛ)
        r2 = root_rho(y, X, d.W; basis = B, penalty = Ω, lambda = 1.0, roughness = 4ℛ)
        w(r) = r.ci_bias_aware[2] - r.ci_bias_aware[1]
        @test w(r0) < w(r1) < w(r2)
        @test r1.rho == r0.rho && r1.ci_wald == r0.ci_wald      # the rest is unchanged
    end

    @testset "48d. λ chosen by length, with the projection as λ = 0" begin
        λs = [0.01, 0.1, 1.0, 10.0]
        g = root_rho(y, X, d.W; basis = B, penalty = Ω, lambda = λs, roughness = ℛ)
        @test g.lambda_grid == vcat(0.0, λs)
        @test length(g.halflength_grid) == 5
        @test (g.ci_bias_aware[2] - g.ci_bias_aware[1]) / 2 ≈ minimum(g.halflength_grid)
        @test g.lambda == g.lambda_grid[argmin(g.halflength_grid)]
        # The chosen candidate equals the single-λ fit at that λ.
        if g.lambda > 0
            s = root_rho(y, X, d.W; basis = B, penalty = Ω, lambda = g.lambda, roughness = ℛ)
            @test s.ci_bias_aware == g.ci_bias_aware
        else
            p = root_rho(y, X, d.W; basis = B)
            @test p.ci_wald == g.ci_bias_aware
        end
        # A very large ℛ makes every smoother's bound dominate: the projection wins.
        big = root_rho(y, X, d.W; basis = B, penalty = Ω, lambda = λs, roughness = 1e4ℛ)
        @test big.lambda == 0 && big.smoother === :projection && big.bias_bound == 0
        # Bonferroni lengthens every candidate.
        bf = root_rho(y, X, d.W; basis = B, penalty = Ω, lambda = λs, roughness = ℛ,
                      bonferroni = true)
        @test all(bf.halflength_grid .> g.halflength_grid)
    end

    @testset "48e. guards" begin
        @test_throws ArgumentError root_rho(y, X, d.W; basis = B, roughness = ℛ)
        @test_throws ArgumentError root_rho(y, X, d.W; basis = B, penalty = Ω, lambda = 1.0,
                                            roughness = -1.0)
        @test_throws ArgumentError root_rho(y, X, d.W; basis = B, penalty = Ω, lambda = [0.1, 1.0])
        @test_throws ArgumentError root_rho(y, X, d.W; basis = B, penalty = Ω, lambda = 1.0,
                                            roughness = ℛ, bonferroni = true)
        @test_throws ArgumentError root_rho(y, X, d.W; basis = B, penalty = Ω, lambda = [0.1, 0.0],
                                            roughness = ℛ)
    end

    @testset "48f. basis_roughness" begin
        ψ = randn(MersenneTwister(48_002), K)
        @test basis_roughness(Z * ψ, B, Ω; X) ≈ sqrt(dot(ψ[2:end], Ω * ψ[2:end])) rtol = 1e-8
        # The penalty's null space (here a plane in the coordinates) and X are free.
        # √ of a quadratic form at rounding level: 2e-15 gives 4.8e-8 (Julia 1.10).
        plane = 1 .+ 2 .* d.P[:, 1] .- d.P[:, 2] .+ 3 .* d.x
        @test basis_roughness(plane, B, Ω; X) < 1e-6
        # Power: a curved surface is not.
        @test basis_roughness(f, B, Ω; X) > 1
        @test_throws DimensionMismatch basis_roughness(f[1:end-1], B, Ω)
    end

    @testset "48g. show" begin
        r = root_rho(y, X, d.W; basis = B, penalty = Ω, lambda = 1.0, roughness = ℛ)
        s = sprint(show, r)
        @test occursin("Bias-aware 95% interval", s)
        @test !occursin("Bias-aware", sprint(show, root_rho(y, X, d.W; basis = B, penalty = Ω,
                                                             lambda = 1.0)))
    end
end
