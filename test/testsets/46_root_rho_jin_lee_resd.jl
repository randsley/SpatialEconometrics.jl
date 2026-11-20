@testset "46. root_rho: Jin & Lee's REsd with a growing basis" begin
    _progress("46. root_rho: Jin & Lee's REsd with a growing basis")
    RR = SpatialEconometrics.RootRho
    hat46(x, k) = (t = range(0, 1, length = k);
                   hcat([max.(0, 1 .- abs.(x .- t[j]) .* (k - 1)) for j in 1:k]...))
    hatbasis46(u, v, k) = (Hu = hat46(u, k); Hv = hat46(v, k);
                           hcat([Hu[:, i] .* Hv[:, j] for i in 1:k for j in 1:k]...))
    # Stratified locations: one per cell of a g × g grid, so no basis cell is empty.
    function design46(g, kh, kw, seed)
        rng = MersenneTwister(seed); n = g^2
        P = hcat(repeat(0:g-1, inner = g), repeat(0:g-1, outer = g)) .+ rand(rng, n, 2); P ./= g
        (; n, P, W = knn_weights(P, kw), x = randn(rng, n), B = hatbasis46(P[:, 1], P[:, 2], kh), rng)
    end
    # The DGP: f in span[x, B], heteroskedastic, ρ₀ = 0.4.
    function draw46(d, ρ0, rng)
        f = 0.5 .* d.x .+ sin.(2π .* d.P[:, 1]) .* cos.(π .* d.P[:, 2])
        σ2 = exp.(3 .* (d.P[:, 1] .+ d.P[:, 2] .- 1)); σ2 ./= mean(σ2)
        lu(sparse(I - ρ0 * d.W)) \ (f .+ sqrt.(σ2) .* randn(rng, d.n))
    end

    d = design46(12, 4, 6, 46_001)                     # n = 144, K = 17
    y = draw46(d, 0.4, d.rng); Wy = d.W * y; Wd = Matrix(d.W)
    X1 = reshape(d.x, :, 1)
    r = root_rho(y, X1, d.W; basis = d.B, varnames = ["x"])
    Z = hcat(d.x, d.B)
    Md = I - Z * ((Z' * Z) \ Z'); Pd = I - Md          # dense, from the normal equations
    Mii = diag(Md)
    Q = Matrix(qr(Z).Q)[:, 1:size(Z, 2)]

    # The reference: C = (AM − Diag(AM)Diag(M)⁻¹M), formed densely from Md.
    Cref(A) = A * Md - Diagonal(diag(A * Md) ./ Mii) * Md
    function root_ref(C)
        a, b, c = Wy' * C * Wy, y' * C * Wy + Wy' * C * y, y' * C * y
        (b - sqrt(b^2 - 4a * c)) / (2a), a, b, c
    end
    function vhat_ref(C, ρ)
        y0 = y - ρ * Wy; u = Md * y0; s = u .^ 2 ./ Mii; Cs = C + C'
        â = Cs * (Pd * y0)
        noise = diag(Cs * Pd * Diagonal(s) * Pd * Cs')
        2 * s' * ((Cs ./ 2) .^ 2) * s + sum((â .^ 2 .- noise) .* s)
    end

    @testset "46a. the moment matrices: zero diagonal, CZ = 0" begin
        for A in (Matrix(Wd'), Matrix((Wd / (I - r.rho_first * Wd))'))
            C = RR._moment_matrix(A, Q, 1 .- vec(sum(abs2, Q; dims = 2)))
            @test maximum(abs, diag(C)) < 1e-12
            @test maximum(abs, C * Z) < 1e-12
            @test maximum(abs, C - Cref(A)) < 1e-10
        end
    end

    @testset "46b. against a dense reimplementation" begin
        ρ1, = root_ref(Cref(Matrix(Wd')))
        C2 = Cref(Matrix((Wd / (I - ρ1 * Wd))'))
        ρ2, a, b, c = root_ref(C2)
        @test r.rho_first ≈ ρ1 atol = 1e-10
        @test r.rho ≈ ρ2 atol = 1e-10
        @test [r.a, r.b, r.c] ≈ [a, b, c] rtol = 1e-10
        @test r.se ≈ sqrt(vhat_ref(C2, ρ2)) / abs(b - 2ρ2 * a) atol = 1e-10
        @test r.ci_wald[2] - r.ci_wald[1] ≈ 2 * 1.959963984540054 * r.se atol = 1e-6
        @test r.beta ≈ (Z \ (y - ρ2 * Wy))[1:1] atol = 1e-10
        # The quartic V̂(ρ) against the direct V̂ at three values of ρ.
        for ρ in (-0.5, 0.1, 0.7)
            @test RR._polyval(r.vpoly, ρ) ≈ vhat_ref(C2, ρ) rtol = 1e-10
        end
    end

    @testset "46c. the test-inversion set's endpoints" begin
        z = 1.959963984540054
        q(ρ) = (r.c - r.b * ρ + r.a * ρ^2)^2 - z^2 * RR._polyval(r.vpoly, ρ)
        sc = z^2 * RR._polyval(r.vpoly, r.rho)          # the scale of q near ρ̂
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
        # The root finder on a quartic with known roots −0.5, 0.2, 0.3, 0.8.
        pk = [1.0]
        for t in (-0.5, 0.2, 0.3, 0.8)                   # pk = Π (ρ − t), ascending powers
            pk = vcat(0.0, pk) .- t .* vcat(pk, 0.0)
        end
        @test RR._real_roots(pk) ≈ [-0.5, 0.2, 0.3, 0.8] atol = 1e-10
        @test RR._inversion_set(0.0, 0.0, 0.0, (1.0, 0.0, 0.0, 0.0, 0.0), 1.96, (-0.99, 0.99)) ==
              [(-0.99, 0.99)]
    end

    @testset "46d. the moment has mean exactly zero at ρ₀ (power: the plain moment)" begin
        # Fixed design, n = 100, K = 17, f in span[x, B], heteroskedastic,
        # normal errors. With the exact variance of g(ρ₀), the standardised
        # moment has mean 0 and variance 1. The plain moment y₀′MWMy₀ has
        # mean tr(MWMΣ) ≠ 0 and fails the same test.
        e = design46(10, 4, 6, 7)
        Ze = hcat(e.x, e.B); Qe = Matrix(qr(Ze).Q)[:, 1:size(Ze, 2)]
        Me = I - Qe * Qe'; We = Matrix(e.W)
        C = RR._moment_matrix(Matrix(We'), Qe, diag(Me))
        f = Qe * (Qe' * (sin.(2π .* e.P[:, 1]) .* cos.(π .* e.P[:, 2]) .+ e.x))
        σ2 = exp.(3 .* (e.P[:, 1] .+ e.P[:, 2] .- 1)); σ2 ./= mean(σ2)
        @test abs(f' * C * f) < 1e-12 && abs(tr(C * Diagonal(σ2))) < 1e-12
        Σ = Diagonal(σ2)
        Vexact(C) = (Cs = (C + C') / 2; 2 * tr(Cs * Σ * Cs * Σ) + dot((C + C') * f, σ2 .* ((C + C') * f)))
        Cp = Me * We * Me
        V, Vp = Vexact(C), Vexact(Cp)
        rng = MersenneTwister(46_004); R = 300
        zs = zeros(R); zp = zeros(R)
        for k in 1:R
            y0 = f .+ sqrt.(σ2) .* randn(rng, e.n)
            zs[k] = dot(y0, C * y0) / sqrt(V); zp[k] = dot(y0, Cp * y0) / sqrt(Vp)
        end
        @test abs(mean(zs)) < 4 / sqrt(R)
        @test abs(var(zs) - 1) < 4 * sqrt(2 / R)
        @test abs(mean(zp)) > 4 / sqrt(R)                # power
    end

    @testset "46e. recovery at n = 900" begin
        e = design46(30, 7, 6, 46_005)                   # K = 50
        ye = draw46(e, 0.4, e.rng)
        re = root_rho(ye, reshape(e.x, :, 1), e.W; basis = e.B)
        @test abs(re.rho - 0.4) < 4 * re.se
        @test re.se < 0.15
        @test abs(re.beta[1] - 0.5) < 0.15
        @test re.K == 50 && re.n == 900 && re.root_found
    end

    @testset "46f. guards and the no-root path" begin
        Wbad = copy(Wd); Wbad[1, 1] = 0.1
        @test_throws "zero diagonal" root_rho(y, X1, Wbad; basis = d.B)
        @test_throws "max_dense_n" root_rho(y, X1, d.W; basis = d.B, max_dense_n = 100)
        @test_throws "linear combination" root_rho(y, hcat(ones(d.n), d.x), d.W; basis = d.B)  # hat basis sums to one
        e1 = zeros(d.n); e1[5] = 1                       # leverage 1 at unit 5
        @test_throws "leverage above 0.99" root_rho(y, X1, d.W; basis = hcat(d.B, e1))
        @test_throws ArgumentError root_rho(y, X1, d.W; basis = d.B, level = 1.2)
        @test_throws DimensionMismatch root_rho(y[1:end-1], X1, d.W; basis = d.B)
        @test_logs (:warn, r"above 0.30") root_rho(y, X1, d.W; basis = hatbasis46(d.P[:, 1], d.P[:, 2], 7))
        # Pure noise with kNN-20 at n = 100: the first-step moment has no real
        # root. ρ is reported as NaN, and the test-inversion set is still returned.
        e = design46(10, 3, 20, 11)
        rn = root_rho(randn(e.rng, e.n), reshape(e.x, :, 1), e.W; basis = e.B)
        @test !rn.root_found && isnan(rn.rho) && isnan(rn.rho_first) && isnan(rn.se)
        @test !isempty(rn.cs)
        @test occursin("No admissible root", sprint(show, rn))
        @test occursin("Test-inversion", sprint(show, r)) && occursin("no standard errors", sprint(show, r))
    end
end
