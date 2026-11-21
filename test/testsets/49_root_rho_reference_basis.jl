@testset "49. root_rho: the bias-aware interval on a reference basis" begin
    _progress("49. root_rho: the bias-aware interval on a reference basis")
    # Testset 48's stratified design, with a coarse estimation basis (k = 4) and a
    # finer reference (k̃ = 10) on the same box, so f is not in span(Z).
    rng = MersenneTwister(49_001); g = 14; n = g^2
    P = hcat(repeat(0:g-1, inner = g), repeat(0:g-1, outer = g)) .+ rand(rng, n, 2); P ./= g
    u, v = P[:, 1], P[:, 2]; W = knn_weights(P, 6); Wd = Matrix(W)
    x = randn(rng, n); X = reshape(x, :, 1)
    gs(s, t) = 2 * sin(2π * s) * cos(π * t)
    ps = tensor_pspline(u, v; k = 4); B = ps.basis; Ω = ps.penalty
    ref = tensor_pspline(u, v; k = 10); Bt = ref.basis
    Z = hcat(x, B); K = size(Z, 2); Zt = hcat(x, Bt); Kt = size(Zt, 2)
    Πt = zeros(K, K); Πt[2:end, 2:end] = Ω
    Πtt = zeros(Kt, Kt); Πtt[2:end, 2:end] = ref.penalty
    f = 0.5 .* x .+ gs.(u, v)
    σ2 = exp.(3 .* (u .+ v .- 1)); σ2 ./= mean(σ2)
    y = lu(sparse(I - 0.4 * W)) \ (f .+ sqrt.(σ2) .* randn(rng, n))
    ℛ = basis_roughness(gs, u, v; k = 10)

    # ── An independent dense reimplementation of Lemma 5 (tc_refbasis.jl's formulas) ──
    Q = Matrix(qr(Z).Q)[:, 1:K]; Pz = Q * Q'; M0 = I - Pz; Mii = diag(M0)
    jlE(A) = A - Diagonal(diag(A * M0) ./ Mii)                     # C = E·M0
    Wy = Wd * y
    quad(C) = (dot(Wy, C * Wy), dot(y, C * Wy) + dot(Wy, C * y), dot(y, C * y))
    function root(a, b, c)
        r = [(b - s * sqrt(b^2 - 4a * c)) / (2a) for s in (1, -1)]
        only(filter(t -> -0.99 < t < 0.99 && -b + 2a * t < 0, r))
    end
    ρ̃ = root(quad(jlE(Wd') * M0)...)
    y0 = y .- ρ̃ .* Wy; s0 = (M0 * y0) .^ 2 ./ Mii; fh = Pz * y0
    E = eigen(Symmetric(Πtt)); keep = E.values .> 1e-10 * maximum(E.values)
    Uk = E.vectors[:, keep]; N0t = E.vectors[:, .!keep]
    Πp = Uk * Diagonal(1 ./ E.values[keep]) * Uk'; Πph = Uk * Diagonal(1 ./ sqrt.(E.values[keep])) * Uk'
    MZZt = M0 * Zt
    function bound(Tm, Qq, ℛv)                                      # debiased plug-in
        Υ = Πph * Tm * Pz
        q = dot(Tm * fh, Πp * (Tm * fh)) - sum(s0 .* vec(sum(abs2, Υ; dims = 1)))
        χ = maximum(abs, eigvals(Symmetric(Πph * ((Qq + Qq') / 2) * Πph)))
        ℛv * sqrt(max(q, 0.0)) + ℛv^2 * χ, χ
    end
    function smoother(λ)
        S = Z * ((Z' * Z + λ * Πt) \ Z'); S = (S + S') / 2; M = I - S
        dd = (M .* M) \ (-diag(M * Wd)); Dg = Diagonal(dd)
        MZt = M * Zt
        (; A = M * Wd + M * Dg * M, M, Dg, Tm = MZt' * Wd, Qq = MZt' * Wd * MZZt + MZt' * Dg * MZt)
    end
    cvf(t; α = 0.05) = (lo = 0.0; hi = t + 10.0;
        for _ in 1:200; c = (lo + hi) / 2;
            cdf(Normal(), c - t) - cdf(Normal(), -c - t) < 1 - α ? (lo = c) : (hi = c); end; (lo + hi) / 2)

    @testset "49a. the package against the reference implementation" begin
        for λ in (0.1, 1.0)
            r = root_rho(y, X, W; ps..., lambda = λ, roughness = ℛ, reference = ref)
            o = smoother(λ); b, _ = bound(o.Tm, o.Qq, ℛ)
            @test r.bias_bound ≈ b rtol = 1e-8
            h = cvf(b / sqrt(r.vhat)) * r.se
            @test r.ci_bias_aware[2] - r.rho ≈ h atol = 1e-6
            @test r.reference_K == Kt
        end
        p = root_rho(y, X, W; basis = B, roughness = ℛ, reference = ref)
        @test p.rho_first ≈ ρ̃ atol = 1e-10
        A2 = (I - p.rho_first * Wd') \ Wd'; C = jlE(A2) * M0
        bR, _ = bound((C * Zt)', MZZt' * C * Zt, ℛ)
        @test p.bias_bound ≈ bR rtol = 1e-8
        @test p.ci_bias_aware[2] - p.rho ≈ cvf(bR / sqrt(p.vhat)) * p.se atol = 1e-6
    end

    @testset "49b. Lemma 5(e): the estimation basis as its own reference is Lemma 4" begin
        r4 = root_rho(y, X, W; ps..., lambda = 1.0, roughness = 1.0)
        r5 = root_rho(y, X, W; ps..., lambda = 1.0, roughness = 1.0, reference = ps)
        @test r5.bias_bound ≈ r4.bias_bound rtol = 1e-8
        p5 = root_rho(y, X, W; basis = B, roughness = 1.0, reference = ps)
        @test p5.bias_bound < 1e-8 * sqrt(p5.vhat)
        @test p5.ci_bias_aware[1] ≈ p5.ci_wald[1] atol = 1e-6
    end

    @testset "49c. Lemma 5 against the exact bias of an f̃ in the reference class" begin
        tk = range(extrema(u)..., length = 10); tl = range(extrema(v)..., length = 10)
        ψt = vcat(0.5, [gs(tk[i], tl[j]) for i in 1:10 for j in 1:10])
        ft = Zt * ψt; ℛt = sqrt(dot(ψt, Πtt * ψt))
        @test ℛt ≈ ℛ rtol = 1e-12
        ψ1 = ψt - N0t * (N0t' * ψt)
        # Lemma 4 on Z, with ℛ from the least-squares fit of f̃ on Z: what the class
        # without a reference would assume.
        ψz = Z \ ft; ℛz = sqrt(dot(ψz, Πt * ψz))
        Ez = eigen(Symmetric(Πt)); kz = Ez.values .> 1e-10 * maximum(Ez.values)
        Πpz = Ez.vectors[:, kz] * Diagonal(1 ./ Ez.values[kz]) * Ez.vectors[:, kz]'
        Πphz = Ez.vectors[:, kz] * Diagonal(1 ./ sqrt.(Ez.values[kz])) * Ez.vectors[:, kz]'
        exceeded = 0
        for λ in (0.01, 0.1, 1.0)
            o = smoother(λ); β = dot(ft, o.A * ft); vt = o.Tm * (Pz * ft)
            @test abs(β - (dot(ψ1, vt) + dot(ψ1, o.Qq * ψ1))) <= 1e-10 * max(abs(β), 1.0)
            _, χ = bound(o.Tm, o.Qq, ℛt)
            @test abs(β) <= ℛt * sqrt(dot(vt, Πp * vt)) + ℛt^2 * χ
            MZ = o.M * Z; vz = (MZ)' * Wd * ft
            χz = maximum(abs, eigvals(Symmetric(Πphz * (MZ' * o.Dg * MZ) * Πphz)))
            exceeded += abs(β) > ℛz * sqrt(dot(vz, Πpz * vz)) + ℛz^2 * χz
        end
        # Power: Lemma 4's bound misses the remainder at the smallest λ at least.
        @test exceeded >= 1
        C = jlE((I - 0.4 * Wd') \ Wd') * M0
        βR = dot(ft, C * ft); vR = (C * Zt)' * (Pz * ft); QR = MZZt' * C * Zt
        @test abs(βR - (dot(ψ1, vR) + dot(ψ1, QR * ψ1))) <= 1e-10 * max(abs(βR), 1.0)
        _, χR = bound((C * Zt)', QR, ℛt)
        @test abs(βR) <= ℛt * sqrt(dot(vR, Πp * vR)) + ℛt^2 * χR
        @test abs(βR) > 1e-3                                        # REsd is biased: b̄ = 0 is wrong
    end

    @testset "49d. the projection and the grid with a reference" begin
        p = root_rho(y, X, W; basis = B, roughness = ℛ, reference = ref)
        @test p.bias_bound > 0
        @test p.ci_bias_aware[2] - p.ci_bias_aware[1] > p.ci_wald[2] - p.ci_wald[1]
        gr = root_rho(y, X, W; ps..., lambda = [0.1, 1.0], roughness = ℛ, reference = ref)
        @test gr.lambda_grid == [0.0, 0.1, 1.0]
        @test gr.halflength_grid[1] ≈ (p.ci_bias_aware[2] - p.ci_bias_aware[1]) / 2
        @test (gr.ci_bias_aware[2] - gr.ci_bias_aware[1]) / 2 ≈ minimum(gr.halflength_grid)
    end

    @testset "49e. guards" begin
        @test_throws ArgumentError root_rho(y, X, W; basis = B, roughness = ℛ,
                                            reference = tensor_pspline(u, v; k = 10, order = 3))
        @test root_rho(y, X, W; basis = B, roughness = ℛ,
                       reference = tensor_pspline(u, v; k = 10, order = 2)) isa RootRhoResult
        @test_throws ArgumentError root_rho(y, X, W; basis = B, reference = ref)
        @test_throws ArgumentError root_rho(y, X, W; roughness = ℛ, reference = ref)
        @test_throws ArgumentError root_rho(y, X, W; basis = B, roughness = ℛ, reference = (; basis = Bt))
        @test_throws DimensionMismatch root_rho(y, X, W; basis = B, roughness = ℛ,
                                                reference = (; basis = Bt[1:end-1, :], penalty = ref.penalty))
    end

    @testset "49f. basis_roughness(g, u, v; k)" begin
        tk = range(extrema(u)..., length = 10); tl = range(extrema(v)..., length = 10)
        ψ = [gs(tk[i], tl[j]) for i in 1:10 for j in 1:10]
        @test ℛ ≈ sqrt(dot(ψ, ref.penalty * ψ)) rtol = 1e-12
        # The coefficients follow tensor_pspline's columns: a bilinear g is interpolated exactly.
        bl(s, t) = 1 + 2s - t + 3s * t
        @test Bt * [bl(tk[i], tl[j]) for i in 1:10 for j in 1:10] ≈ bl.(u, v) atol = 1e-12
        @test basis_roughness(bl, u, v; k = 10) < 1e-10
        @test ℛ > 1                                                 # power: a curved g is not free
        r20 = 19 * basis_roughness(gs, u, v; k = 20); r30 = 29 * basis_roughness(gs, u, v; k = 30)
        @test abs(r20 / r30 - 1) < 0.05
        @test_throws ArgumentError basis_roughness((s, t) -> NaN, u, v; k = 10)
    end

    @testset "49g. show" begin
        s = sprint(show, root_rho(y, X, W; basis = B, roughness = ℛ, reference = ref))
        @test occursin("reference basis with K̃ = $Kt", s)
    end
end
