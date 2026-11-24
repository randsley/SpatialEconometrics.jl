@testset "50. root_rho: the basis chosen by the interval's length" begin
    _progress("50. root_rho: the basis chosen by the interval's length")
    # Testset 49's design: three estimation bases (k = 4, 5, 6) against one reference
    # (k̃ = 10), so the class does not depend on the basis (THEORY.md, Corollary 7′).
    rng = MersenneTwister(49_001); g = 14; n = g^2
    P = hcat(repeat(0:g-1, inner = g), repeat(0:g-1, outer = g)) .+ rand(rng, n, 2); P ./= g
    u, v = P[:, 1], P[:, 2]; W = knn_weights(P, 6)
    x = randn(rng, n); X = reshape(x, :, 1)
    gs(s, t) = 2 * sin(2π * s) * cos(π * t)
    f = 0.5 .* x .+ gs.(u, v)
    σ2 = exp.(3 .* (u .+ v .- 1)); σ2 ./= mean(σ2)
    y = lu(sparse(I - 0.4 * W)) \ (f .+ sqrt.(σ2) .* randn(rng, n))
    ref = tensor_pspline(u, v; k = 10); ℛ = basis_roughness(gs, u, v; k = 10)
    pss = [tensor_pspline(u, v; k) for k in (4, 5, 6)]
    Bs = [p.basis for p in pss]; Ps = [p.penalty for p in pss]
    λs = [0.1, 1.0]
    hlen(r) = (r.ci_bias_aware[2] - r.ci_bias_aware[1]) / 2

    @testset "50a. the grid is the union of the single-basis grids" begin
        gr = root_rho(y, X, W; basis = Bs, penalty = Ps, lambda = λs, roughness = ℛ, reference = ref)
        singles = [root_rho(y, X, W; basis = Bs[j], penalty = Ps[j], lambda = λs, roughness = ℛ,
                            reference = ref) for j in 1:3]
        @test gr.basis_grid == repeat(1:3, inner = 3)
        @test gr.lambda_grid == repeat([0.0, 0.1, 1.0], 3)
        @test gr.halflength_grid ≈ vcat([s.halflength_grid for s in singles]...) rtol = 1e-12
        j = argmin([hlen(s) for s in singles])
        @test gr.basis_index == j
        @test gr.rho == singles[j].rho
        @test gr.ci_bias_aware == singles[j].ci_bias_aware
        @test gr.K == size(Bs[j], 2) + 1
        @test hlen(gr) ≈ minimum(gr.halflength_grid)
    end

    @testset "50b. Bonferroni counts every (basis, λ) pair" begin
        gb = root_rho(y, X, W; basis = Bs, penalty = Ps, lambda = 1.0, roughness = ℛ,
                      reference = ref, bonferroni = true)
        @test gb.lambda_grid == fill(1.0, 3)
        # One candidate per basis: each at level 1 − α/3.
        sb = [root_rho(y, X, W; basis = Bs[j], penalty = Ps[j], lambda = 1.0, roughness = ℛ,
                       reference = ref, level = 1 - 0.05 / 3) for j in 1:3]
        @test gb.halflength_grid ≈ hlen.(sb) rtol = 1e-10
        g0 = root_rho(y, X, W; basis = Bs, penalty = Ps, lambda = 1.0, roughness = ℛ, reference = ref)
        @test all(gb.halflength_grid .> g0.halflength_grid)          # power: Bonferroni lengthens
        @test gb.level == 0.95                                       # the Wald level is unchanged
    end

    @testset "50c. projections alone" begin
        gp = root_rho(y, X, W; basis = Bs, roughness = ℛ, reference = ref)
        sp = [root_rho(y, X, W; basis = Bs[j], roughness = ℛ, reference = ref) for j in 1:3]
        @test gp.lambda_grid == zeros(3)
        @test gp.halflength_grid ≈ hlen.(sp) rtol = 1e-12
        @test gp.basis_index == argmin(hlen.(sp))
        # The remainder costs the coarsest basis length (Lemma 5): it is not the shortest.
        @test hlen(sp[1]) > minimum(hlen.(sp))
    end

    @testset "50d. guards" begin
        @test_throws ArgumentError root_rho(y, X, W; basis = Bs, penalty = Ps, lambda = λs, roughness = ℛ)
        @test_throws ArgumentError root_rho(y, X, W; basis = Bs, reference = ref)
        @test_throws ArgumentError root_rho(y, X, W; basis = typeof(Bs[1])[], roughness = ℛ, reference = ref)
        @test_throws DimensionMismatch root_rho(y, X, W; basis = Bs, penalty = Ps[1:2], lambda = λs,
                                                roughness = ℛ, reference = ref)
        @test_throws ArgumentError root_rho(y, X, W; basis = Bs, penalty = Ps[1], lambda = λs,
                                            roughness = ℛ, reference = ref)
        @test_throws ArgumentError root_rho(y, X, W; basis = Bs[1], penalty = Ps, lambda = λs,
                                            roughness = ℛ, reference = ref)
        # (R) per basis: a first-difference penalty leaves only constants free, the
        # reference's bilinear surfaces are not; power: the second-difference one passes.
        p1 = tensor_pspline(u, v; k = 5, order = 1)
        @test_throws ArgumentError root_rho(y, X, W; basis = [Bs[1], p1.basis], penalty = [Ps[1], p1.penalty],
                                            lambda = λs, roughness = ℛ, reference = ref)
        @test root_rho(y, X, W; basis = [Bs[1], Bs[2]], penalty = [Ps[1], Ps[2]], lambda = λs,
                       roughness = ℛ, reference = ref) isa RootRhoResult
    end

    @testset "50e. single basis results are unchanged" begin
        r = root_rho(y, X, W; basis = Bs[1], penalty = Ps[1], lambda = λs, roughness = ℛ, reference = ref)
        @test r.basis_index == 0
        @test isempty(r.basis_grid)
    end

    @testset "50f. show" begin
        gr = root_rho(y, X, W; basis = Bs, penalty = Ps, lambda = λs, roughness = ℛ, reference = ref)
        s = sprint(show, gr)
        @test occursin("basis $(gr.basis_index) of 3 (K = $(gr.K))", s)
    end
end
