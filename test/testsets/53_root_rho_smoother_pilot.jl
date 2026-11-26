@testset "53. root_rho: the smoother as the bias-aware interval's pilot (Lemma 13)" begin
    _progress("53. root_rho: the smoother as the bias-aware interval's pilot (Lemma 13)")
    # Testset 47h's triangle: units fill u + v ≤ 1, so the hats above the diagonal have no unit
    # and [x, B] is rank-deficient. The projection pilot of Proposition 7 cannot run there;
    # the smoother at λ_p can (THEORY.md, Lemma 13 and Proposition 7″).
    rng = MersenneTwister(53_001); g = 24
    P = hcat(repeat(0:g-1, inner = g), repeat(0:g-1, outer = g)) .+ rand(rng, g^2, 2); P ./= g
    P = P[P[:, 1] .+ P[:, 2] .<= 1, :]
    P = vcat(P, [0.0 1.0; 1.0 0.0])
    n = size(P, 1); W = knn_weights(P, 6); Wd = Matrix(W); x = randn(rng, n)
    X = reshape(x, :, 1)
    ps = tensor_pspline(P[:, 1], P[:, 2]; k = 10)
    Z = hcat(x, ps.basis); K = size(Z, 2)
    Πt = zeros(K, K); Πt[2:end, 2:end] = ps.penalty
    f = 0.5 .* x .+ sin.(2π .* P[:, 1]) .* cos.(π .* P[:, 2])
    y = lu(sparse(I - 0.4 * W)) \ (f .+ randn(rng, n))
    Wy = Wd * y
    ℛ = basis_roughness(f, ps.basis, ps.penalty; X)
    E = eigen(Symmetric(Πt)); keep = E.values .> 1e-10 * maximum(E.values)
    N = Diagonal(1 ./ sqrt.(E.values[keep])) * E.vectors[:, keep]'     # N′N = Π̃⁺

    smoother(λ) = (S = Z * ((Z' * Z + λ * Πt) \ Z'); S = (S + S') / 2; (S, I - S))
    function moment(M)
        dd = (M .* M) \ (-diag(M * Wd))
        A = M * Wd + M * Diagonal(dd) * M
        (A, dd)
    end
    function sign_root(A)
        a = dot(Wy, A * Wy); b = dot(y, A * Wy) + dot(Wy, A * y); c = dot(y, A * y)
        only(filter(r -> -0.99 < r < 0.99 && -b + 2a * r < 0,
                    [(b - s * sqrt(b^2 - 4a * c)) / (2a) for s in (1, -1)]))
    end
    # Lemma 13 and Proposition 7″, written out densely from THEORY.md.
    function ref53(λ, λp, ℛv)
        Sp, Mp = smoother(λp)
        ρp = sign_root(moment(Mp)[1])
        y0 = y .- ρp .* Wy
        m = vec(sum(abs2, Mp; dims = 1)); good = m .>= 0.01
        e = Mp * y0; s = [good[i] ? e[i]^2 / m[i] : 0.0 for i in 1:n]
        _, M = smoother(λ); _, dd = moment(M)
        T = Z' * M * Wd
        Υp = N * T * Sp
        c = vec(sum(abs2, Υp; dims = 1))
        rp = opnorm(Mp * Z * N'); ωp = opnorm(N * T * Mp * Z * N')
        c̄ = maximum(c[good] ./ m[good])
        q̂ = sum(abs2, Υp * y0) - dot(c, s) + c̄ * ℛv^2 * rp^2
        χ = maximum(abs, eigvals(Symmetric(N * ((M * Z)' * Diagonal(dd) * (M * Z)) * N')))
        (; bbar = ℛv * sqrt(max(q̂, 0.0)) + ℛv^2 * (χ + ωp), χ, ωp, rp, nleft = count(!, good),
           Mp, Sp, T)
    end

    @testset "53a. the package against an independent reference" begin
        @test rank(Z) < K - 10
        for (λ, λp) in ((1.0, 1.0), (1.0, 0.1), (10.0, 0.1))
            r = root_rho(y, X, W; ps..., lambda = λ, roughness = ℛ, pilot_lambda = λp)
            ref = ref53(λ, λp, ℛ)
            @test r.root_found
            @test r.pilot === :penalised && r.pilot_lambda == λp
            @test r.pilot_left_out == ref.nleft
            @test r.bias_bound ≈ ref.bbar rtol = 1e-8
            # The pilot's two constants are paid: the bound exceeds its ℛ²χ̄ part by more
            # than ℛ²ω_p, and ω_p is not negligible here.
            @test r.bias_bound > ℛ^2 * (ref.χ + ref.ωp)
            @test ref.ωp > 1e-3 * ref.χ
        end
        # :auto chooses the smoother here, with the smallest λ of a vector as λ_p.
        rv = root_rho(y, X, W; ps..., lambda = [0.1, 1.0, 10.0], roughness = ℛ)
        @test rv.pilot === :penalised && rv.pilot_lambda == 0.1
        @test length(rv.lambda_grid) == 3 && 0.0 ∉ rv.lambda_grid     # no projection candidate
        @test rv.bias_bound ≈ ref53(rv.lambda, 0.1, ℛ).bbar rtol = 1e-8
        @test occursin("pilot: the smoother at λ_p = 0.1", sprint(show, rv))
        # Power: the projection pilot is refused on this region.
        @test_throws ArgumentError root_rho(y, X, W; ps..., lambda = 1.0, roughness = ℛ,
                                            pilot = :projection)
    end

    @testset "53b. Lemma 13(a): the pilot's shrinkage is bounded by ℛω_p" begin
        # For f in the class, ‖NTf‖ ≤ ‖NTS_pf‖ + ℛω_p and ‖M_pf‖ ≤ ℛr_p, with equality
        # approached by the worst direction. Checked on random f and on that direction.
        ref = ref53(1.0, 0.1, 1.0)
        rng2 = MersenneTwister(53_002)
        for _ in 1:20
            ψ = randn(rng2, K)
            ψ ./= sqrt(dot(ψ, Πt * ψ))                     # ψ′Π̃ψ = 1 = ℛ²
            fz = Z * ψ
            @test norm(N * ref.T * fz) <= norm(N * ref.T * ref.Sp * fz) + ref.ωp + 1e-10
            @test norm(ref.Mp * fz) <= ref.rp + 1e-10
        end
        # The worst direction attains the second bound (it is an operator norm).
        F = svd(ref.Mp * Z * N')
        fz = Z * (N' * F.V[:, 1])
        @test norm(ref.Mp * fz) ≈ ref.rp rtol = 1e-8
        # ω_p is attained too: the pilot's shrinkage, seen through the candidate, reaches it.
        G = svd(N * ref.T * ref.Mp * Z * N')
        fw = Z * (N' * G.V[:, 1])
        @test norm(N * ref.T * ref.Mp * fw) ≈ ref.ωp rtol = 1e-8
    end

    @testset "53c. Lemma 13(d): the plug-in errs upwards in expectation" begin
        # With equal variances σ² = 1 and f in the class, E q̂_p at ρ₀ is
        # ‖Υ_pf‖² + Σ_{i∉𝒪} cᵢ + c̄ℛ²r² − Σ_{i∈𝒪} cᵢ(M_pf)ᵢ²/mᵢ ≥ ‖NTS_pf‖², exactly.
        ref = ref53(1.0, 0.1, ℛ)
        m = vec(sum(abs2, ref.Mp; dims = 1)); good = m .>= 0.01
        Υp = N * ref.T * ref.Sp; c = vec(sum(abs2, Υp; dims = 1)); c̄ = maximum(c[good] ./ m[good])
        ψ = Z \ f; fz = Z * ψ; ℛz = sqrt(dot(ψ, Πt * ψ))
        Mf = ref.Mp * fz
        Eq = sum(abs2, Υp * fz) + sum(c[.!good]) + c̄ * ℛz^2 * ref.rp^2 -
             sum(c[good] .* Mf[good] .^ 2 ./ m[good])
        @test Eq >= sum(abs2, N * ref.T * ref.Sp * fz)
        # A Monte Carlo of the same expectation at ρ₀ (no root estimated): R = 400 draws.
        rng3 = MersenneTwister(53_003); R = 400
        qs = map(1:R) do _
            y0 = fz .+ randn(rng3, n)
            e = ref.Mp * y0
            s = [good[i] ? e[i]^2 / m[i] : 0.0 for i in 1:n]
            sum(abs2, Υp * y0) - dot(c, s) + c̄ * ℛz^2 * ref.rp^2
        end
        z = (mean(qs) - Eq) / (std(qs) / sqrt(R))
        @test abs(z) < 4
    end

    @testset "53d. arguments" begin
        @test_throws ArgumentError root_rho(y, X, W; ps..., lambda = 1.0, roughness = ℛ,
                                            pilot = :bogus)
        @test_throws ArgumentError root_rho(y, X, W; ps..., lambda = 1.0, pilot = :penalised)
        @test_throws ArgumentError root_rho(y, X, W; ps..., lambda = 1.0, roughness = ℛ,
                                            pilot = :projection, pilot_lambda = 0.1)
        @test_throws ArgumentError root_rho(y, X, W; ps..., lambda = 1.0, roughness = ℛ,
                                            pilot_lambda = 0.0)
        ref = (; basis = ps.basis, penalty = ps.penalty)
        @test_throws ArgumentError root_rho(y, X, W; ps..., lambda = 1.0, roughness = ℛ,
                                            reference = ref, pilot = :penalised)
        # Without `roughness` there is no pilot.
        r0 = root_rho(y, X, W; ps..., lambda = 1.0)
        @test r0.pilot === :none && isnan(r0.pilot_lambda) && r0.pilot_left_out == 0
    end
end
