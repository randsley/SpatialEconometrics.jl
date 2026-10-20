using Test
using SpatialEconometrics
using LinearAlgebra
using SparseArrays
using Statistics
using Random
using ForwardDiff
using ChainRulesCore
using StatsAPI
import StatsAPI: coef, vcov, stderror, nobs, residuals, coefnames, loglikelihood, dof_residual
using StatsModels
using Tables

@testset "SpatialEconometrics.jl Full Suite Verification" begin

    @testset "1. Cross-Sectional SAR & SDM" begin
        rng = MersenneTwister(101); W = make_rook_weights(20); N = size(W, 1)
        X = randn(rng, N, 2); β_true = [2.0, 1.5, -1.0]; θ_true = [0.8, -0.5]; ρ_true = 0.55; σ_true = 0.50
        y = (I - ρ_true * W) \ (β_true[1] .+ X * β_true[2:3] .+ (W * X) * θ_true .+ σ_true .* randn(rng, N))
        m_cheb = fit_spatial(CrossSection.SDM(), W, X, y; method=CrossSection.ChebyshevApprox(order=7))
        @test isapprox(m_cheb.rho, ρ_true; atol=0.08)
        @test length(coef(m_cheb)) == 6  # intercept + 2 β + 2 θ + ρ
        @test length(residuals(m_cheb)) == N

        m_sar = fit_spatial(CrossSection.SAR(), W, X, y; method=CrossSection.ChebyshevApprox(order=7))
        @test length(coef(m_sar)) == 4  # intercept + 2 β + ρ
    end

    @testset "2. Static Spatial Panels" begin
        rng = MersenneTwister(202); L = 12; N = L * L; T_per = 6; NT = N * T_per; W = make_rook_weights(L)
        X = randn(rng, NT, 2); β_true = [1.5, -0.8]; ρ_true = 0.45; σ_true = 0.5
        μ_i = randn(rng, N); ξ_t = randn(rng, T_per)
        S_lu = lu(sparse(1.0I, N, N) - ρ_true * W)
        y = zeros(NT)
        for t in 1:T_per
            idx = ((t - 1) * N + 1):(t * N)
            y[idx] = S_lu \ (μ_i .+ ξ_t[t] .+ X[idx, :] * β_true .+ σ_true .* randn(rng, N))
        end

        m_panel = fit_spatial_panel(StaticPanel.SAR(), StaticPanel.TwoWayFE(), W, X, y, N, T_per; approach=LeeYuTransformation())
        @test isapprox(m_panel.rho, ρ_true; atol=0.12)
        @test length(coef(m_panel)) == 3  # x1, x2, ρ
        @test nobs(m_panel) == NT
        @test length(residuals(m_panel)) == NT

        m_dqml = fit_spatial_panel(StaticPanel.SAR(), StaticPanel.IndivFE(), W, X, y, N, T_per; approach=DirectQML(bias_correct=true))
        @test length(coef(m_dqml)) == 3
    end

    @testset "3. Dynamic Spatial Panels" begin
        rng = MersenneTwister(303); L = 10; N = L * L; T_tot = 8; NT = N * T_tot; W = make_rook_weights(L)
        X = randn(rng, NT, 1); β_true = [0.8]; γ_true = 0.3; η_true = 0.15; ρ_true = 0.35; σ_true = 0.4
        μ_i = randn(rng, N)
        S_lu = lu(sparse(1.0I, N, N) - ρ_true * W)
        y = zeros(NT)
        y[1:N] = S_lu \ (μ_i .+ X[1:N, :] * β_true .+ σ_true .* randn(rng, N))
        for t in 2:T_tot
            idx = ((t - 1) * N + 1):(t * N)
            idx_prev = ((t - 2) * N + 1):((t - 1) * N)
            y[idx] = S_lu \ (μ_i .+ γ_true .* y[idx_prev] .+ η_true .* (W * y[idx_prev]) .+ X[idx, :] * β_true .+ σ_true .* randn(rng, N))
        end

        m_dyn = fit_dynamic_spatial_panel(DynamicPanel.SAR(), DynamicPanel.IndivFE(), W, X, y, N, T_tot)
        @test length(coef(m_dyn)) == 4  # y_lag, Wy_lag, x1, ρ
        @test m_dyn.T_post == T_tot - 1
        @test isapprox(m_dyn.rho, ρ_true; atol=0.15)
    end

    @testset "4. GS2SLS / SARAR Panel" begin
        rng = MersenneTwister(404); L = 10; N = L * L; T_tot = 6; NT = N * T_tot; W = make_rook_weights(L)
        X_exog = randn(rng, NT, 2); β_true = [1.2, -0.6]; ρ_true = 0.3; λ_true = 0.25
        z1 = randn(rng, NT); z2 = randn(rng, NT)
        endog = 0.5 .* z1 .- 0.3 .* z2 .+ 0.2 .* randn(rng, NT); δ_true = 0.8

        S_lu = lu(sparse(1.0I, N, N) - ρ_true * W)
        B_lu = lu(sparse(1.0I, N, N) - λ_true * W)
        y = zeros(NT)
        for t in 1:T_tot
            idx = ((t - 1) * N + 1):(t * N)
            u_t = B_lu \ (0.4 .* randn(rng, N))
            y[idx] = S_lu \ (X_exog[idx, :] * β_true .+ endog[idx] .* δ_true .+ u_t)
        end

        m_gs = fit_gs2sls_panel(GS2SLSPanel.IndivFE(), W, X_exog, reshape(endog, NT, 1), hcat(z1, z2), y, N, T_tot)
        @test length(coef(m_gs)) == 5  # ρ, exog1, exog2, endog1, λ
        @test m_gs.T_post == T_tot - 1
        @test size(vcov(m_gs)) == (5, 5)
    end

    @testset "5. System GMM (Arellano-Bond / Blundell-Bond)" begin
        rng = MersenneTwister(505); N = 80; T_tot = 7; K = 1
        γ_true = 0.4; β_true = [1.0]
        μ_i = randn(rng, N); Y = zeros(N, T_tot); X_ten = randn(rng, N, T_tot, K)
        for i in 1:N; Y[i, 1] = μ_i[i] + X_ten[i, 1, 1] * β_true[1] + randn(rng); end
        for t in 2:T_tot, i in 1:N
            Y[i, t] = μ_i[i] + γ_true * Y[i, t - 1] + X_ten[i, t, 1] * β_true[1] + 0.5 * randn(rng)
        end

        m_sys = fit_dynamic_gmm(SystemGMM(), Y, X_ten; max_lag=3)
        @test length(coef(m_sys)) == 2  # y_lag, x1
        @test m_sys.hansen_df >= 0
        @test coefnames(m_sys) == ["y_lag", "x1"]

        m_diff = fit_dynamic_gmm(DifferenceGMM(), Y, X_ten; max_lag=3)
        @test length(coef(m_diff)) == 2
    end

    @testset "6. GRAS / Cross-Entropy Calibration" begin
        rng = MersenneTwister(606)
        M_dim = 5; N_dim = 4
        X0 = randn(rng, M_dim, N_dim) .* 10
        u_true = vec(sum(X0; dims=2)); v_true = vec(sum(X0; dims=1))
        u_target = u_true .* (1 .+ 0.1 .* randn(rng, M_dim))
        adj = (sum(u_target) - sum(v_true)) / N_dim
        v_target = v_true .+ adj

        res = balance_gras_full(X0, u_target, v_target)
        @test res.converged
        @test isapprox(vec(sum(res.X_balanced; dims=2)), u_target; atol=1e-6)
        @test isapprox(vec(sum(res.X_balanced; dims=1)), v_target; atol=1e-6)
        @test res.kl_divergence >= 0
    end

    @testset "7. Driscoll-Kraay & Conley HAC" begin
        rng = MersenneTwister(707); L = 10; N = L * L; T_per = 8; W = make_rook_weights(L)
        X = randn(rng, N * T_per, 2); ρ_true = 0.4
        S_lu = lu(sparse(1.0I, N, N) - ρ_true * W)
        μ_i = randn(rng, N); y = zeros(N * T_per)
        for t in 1:T_per
            idx = ((t - 1) * N + 1):(t * N)
            y[idx] = S_lu \ (μ_i .+ X[idx, :] * [1.0, -0.5] .+ (0.3 .+ 0.5 .* abs.(randn(rng, N))) .* randn(rng, N))
        end

        m_pan = fit_spatial_panel(StaticPanel.SAR(), StaticPanel.TwoWayFE(), W, X, y, N, T_per)
        V_dk = robust_vcov(m_pan, DriscollKraayVCOV())
        @test size(V_dk) == size(vcov(m_pan))
        @test all(eigvals(Symmetric(V_dk)) .>= -1e-10)  # PSD

        coords = hcat(repeat(1.0:L, L), repeat(1.0:L, inner=L))
        V_con = robust_vcov(m_pan, ConleySpatialHAC(coords, 3.0))
        @test size(V_con) == size(vcov(m_pan))
        @test all(eigvals(Symmetric(V_con)) .>= -1e-10)
    end

    @testset "8. FormulaDSL.jl (@formula, slag, fe, endo)" begin
        rng = MersenneTwister(808); L = 15; N = L * L; T_per = 6; NT = N * T_per; W = make_rook_weights(L)
        X = randn(rng, NT, 2); β_true = [1.4, -0.9]; θ_true = [0.7, -0.35]; ρ_true = 0.45
        μ_i = randn(rng, N); ξ_t = randn(rng, T_per)

        S_lu = lu(sparse(1.0I, N, N) - ρ_true * W)
        y = zeros(Float64, NT); region_id = zeros(Int, NT); year_id = zeros(Int, NT)
        for t in 1:T_per
            idx = ((t - 1) * N + 1):(t * N)
            y[idx] = S_lu \ (μ_i .+ ξ_t[t] .+ X[idx, :] * β_true .+ (W * X[idx, :]) * θ_true .+ 0.5 .* randn(rng, N))
            region_id[idx] .= 1:N; year_id[idx]   .= 2020 + t
        end

        shuf_idx = sortperm(1:NT; by = i -> (region_id[i], year_id[i]))
        tbl = (region = region_id[shuf_idx], year = year_id[shuf_idx], y = y[shuf_idx], x1 = X[shuf_idx, 1], x2 = X[shuf_idx, 2])

        m_dsl = fit_spatial_panel(
            @formula(y ~ x1 + x2 + slag(x1 + x2) + fe(region) + fe(year)),
            tbl, W; panel=:region, time=:year, approach=LeeYuTransformation()
        )
        @test coefnames(m_dsl) == ["x1", "x2", "W_x1", "W_x2", "ρ"]

        z1 = randn(rng, NT); z2 = randn(rng, NT); tax_endog = 0.6 .* z1 .- 0.4 .* z2 .+ 0.3 .* randn(rng, NT)
        tbl_iv = (region = region_id[shuf_idx], year = year_id[shuf_idx], y = y[shuf_idx], x1 = X[shuf_idx, 1], tax = tax_endog[shuf_idx], z1 = z1[shuf_idx], z2 = z2[shuf_idx])

        m_iv_dsl = fit_gs2sls_panel(
            @formula(y ~ x1 + endo(tax, z1 + z2) + fe(region) + fe(year)),
            tbl_iv, W; panel=:region, time=:year
        )
        @test coefnames(m_iv_dsl) == ["ρ (Wy)", "x1", "tax", "λ (Wu)"]
    end

    @testset "9. Spatial Diagnostics (Moran, LM, Geary)" begin
        rng = MersenneTwister(909); W = make_rook_weights(15); N = size(W, 1)
        X = randn(rng, N, 2)
        y_sar = (I - 0.5 * W) \ (X * [1.5, -0.8] .+ 0.4 .* randn(rng, N))

        mr = moran_global(y_sar, W)
        @test mr.I_stat > 0.0
        @test mr.p_value < 0.05

        gc = geary_c(y_sar, W)
        @test gc.C_stat < 1.0  # positive autocorrelation → C < 1

        lms = lm_tests(y_sar, hcat(ones(N), X), W)
        @test lms.lag.stat > 0.0
        @test lms.lag.p_value < 0.05

        lisa = moran_local(y_sar, W; n_perm=199, rng=MersenneTwister(9091))
        @test length(lisa.local_I) == N
        @test any(lisa.cluster_type .== :HH) || any(lisa.cluster_type .== :LL)
    end

    @testset "10. Spatial Error Model (SEM)" begin
        rng = MersenneTwister(1010); W = make_rook_weights(15); N = size(W, 1)
        X = randn(rng, N, 2); β_true = [1.0, 2.0, -1.0]; λ_true = 0.55
        u = (I - λ_true * W) \ (0.5 .* randn(rng, N))
        y = β_true[1] .+ X * β_true[2:3] .+ u

        m_sem = fit_spatial_error(W, X, y)
        @test isapprox(m_sem.lambda, λ_true; atol=0.12)
        @test length(coef(m_sem)) == 4  # intercept + 2β + λ
        @test coefnames(m_sem) == ["(Intercept)", "x1", "x2", "λ"]

        # Panel SEM
        T_per = 6; NT = N * T_per
        X_pan = randn(rng, NT, 2); μ_i = randn(rng, N)
        y_pan = zeros(NT)
        B_lu = lu(sparse(1.0I, N, N) - λ_true * W)
        for t in 1:T_per
            idx = ((t - 1) * N + 1):(t * N)
            u_t = B_lu \ (0.4 .* randn(rng, N))
            y_pan[idx] = μ_i .+ X_pan[idx, :] * [1.5, -0.7] .+ u_t
        end
        m_sem_pan = fit_spatial_error(SEMMod.IndivFE(), W, X_pan, y_pan, N, T_per)
        @test isapprox(m_sem_pan.lambda, λ_true; atol=0.15)
        @test coefnames(m_sem_pan) == ["x1", "x2", "λ"]
    end

    @testset "11. GWR & MGWR" begin
        rng = MersenneTwister(1111); N = 150
        coords = hcat(rand(rng, N) .* 10, rand(rng, N) .* 10)
        X = randn(rng, N, 2)
        # Spatially varying: β1 increases with x-coordinate
        β1 = 1.0 .+ 0.3 .* coords[:, 1]
        β2 = fill(-0.5, N)  # constant
        y = 2.0 .+ β1 .* X[:, 1] .+ β2 .* X[:, 2] .+ 0.5 .* randn(rng, N)

        m_gwr = fit_gwr(X, y, coords; bandwidth=AdaptiveBandwidth(40))
        @test m_gwr.R2 > 0.5
        @test size(m_gwr.coef_local) == (N, 3)  # intercept + 2 vars
        @test length(residuals(m_gwr)) == N
        # Check that local β1 varies more than local β2
        @test std(m_gwr.coef_local[:, 2]) > std(m_gwr.coef_local[:, 3])

        m_mgwr = fit_mgwr(X, y, coords)
        @test m_mgwr.R2 > 0.5
        @test length(m_mgwr.bandwidths) == 3  # intercept + 2 vars
        @test size(m_mgwr.coef_local) == (N, 3)
    end

    @testset "12. Gravity / Flow Models (PPML & NB)" begin
        rng = MersenneTwister(1212)
        N_orig = 15; N_dest = 12; N_obs = N_orig * N_dest
        origin_id = repeat(1:N_orig, inner=N_dest)
        dest_id = repeat(1:N_dest, outer=N_orig)
        dist = rand(rng, N_obs) .* 5 .+ 0.5
        gdp_orig = rand(rng, N_obs) .* 3 .+ 1
        X = hcat(log.(dist), log.(gdp_orig))
        β_true = [-1.2, 0.8]

        # True origin/dest FE
        α_o = randn(rng, N_orig) .* 0.5
        α_d = randn(rng, N_dest) .* 0.5
        η = X * β_true
        for n in 1:N_obs
            η[n] += α_o[origin_id[n]] + α_d[dest_id[n]]
        end
        μ_true = exp.(η)
        flows = Float64[max(0.0, rand(rng) < 0.95 ? Float64(round(μ_true[n] + sqrt(μ_true[n]) * randn(rng))) : 0.0) for n in 1:N_obs]
        flows = max.(flows, 0.0)

        m_ppml = fit_gravity(PoissonGravity(), flows, X, origin_id, dest_id; varnames=["log_dist", "log_gdp"], maxiter=200)
        @test m_ppml.iterations > 0
        @test length(coef(m_ppml)) == 2
        @test coefnames(m_ppml) == ["log_dist", "log_gdp"]
        @test coef(m_ppml)[1] < 0  # distance should be negative

        m_nb = fit_gravity(NegBinGravity(), flows, X, origin_id, dest_id; varnames=["log_dist", "log_gdp"])
        @test m_nb.alpha > 0.0
        @test length(coef(m_nb)) == 2
    end

end
