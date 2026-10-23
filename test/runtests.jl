using Test
using SpatialEconometrics
using LinearAlgebra
using SparseArrays
using Statistics
using Random
using ForwardDiff
using ChainRulesCore
using StatsAPI
import StatsAPI: coef, vcov, stderror, nobs, residuals, coefnames, loglikelihood, dof_residual, predict, fitted
using StatsModels
using Tables
using NearestNeighbors

@testset "SpatialEconometrics.jl Full Suite Verification" begin

    @testset "1. Cross-Sectional SAR & SDM" begin
        rng = MersenneTwister(101); W = make_rook_weights(20); N = size(W, 1)
        X = randn(rng, N, 2); β_true = [2.0, 1.5, -1.0]; θ_true = [0.8, -0.5]; ρ_true = 0.55; σ_true = 0.50
        y = (I - ρ_true * W) \ (β_true[1] .+ X * β_true[2:3] .+ (W * X) * θ_true .+ σ_true .* randn(rng, N))
        m_cheb = fit_spatial(SDM(), W, X, y; method=ChebyshevApprox(order=7))
        @test isapprox(m_cheb.rho, ρ_true; atol=0.08)
        @test length(coef(m_cheb)) == 6  # intercept + 2 β + 2 θ + ρ
        @test length(residuals(m_cheb)) == N

        m_sar = fit_spatial(SAR(), W, X, y; method=ChebyshevApprox(order=7))
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

        m_panel = fit_spatial_panel(SAR(), TwoWayFE(), W, X, y, N, T_per; approach=LeeYuTransformation())
        @test isapprox(m_panel.rho, ρ_true; atol=0.12)
        @test length(coef(m_panel)) == 3  # x1, x2, ρ
        @test nobs(m_panel) == NT
        @test length(residuals(m_panel)) == NT

        m_dqml = fit_spatial_panel(SAR(), IndivFE(), W, X, y, N, T_per; approach=DirectQML(bias_correct=true))
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

        m_dyn = fit_dynamic_spatial_panel(SAR(), IndivFE(), W, X, y, N, T_tot)
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

        m_gs = fit_gs2sls_panel(IndivFE(), W, X_exog, reshape(endog, NT, 1), hcat(z1, z2), y, N, T_tot)
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

        m_pan = fit_spatial_panel(SAR(), TwoWayFE(), W, X, y, N, T_per)
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

        Xc = hcat(ones(N), X)
        lms = lm_tests(y_sar, Xc, W)
        @test lms.lag.stat > 0.0
        @test lms.lag.p_value < 0.05

        # LM-lag and LM-error are different statistics. They were once written
        # with the same body, so both reported the LM-error value under two
        # labels; a sign-and-threshold check passes on that. Pin both to their
        # closed forms instead.
        let qrX = qr(Xc), Wd = Matrix(W)
            b = qrX \ y_sar
            e = y_sar .- Xc * b
            s2 = dot(e, e) / N
            Tr = tr(Wd' * Wd + Wd * Wd)
            WXb = Wd * (Xc * b)
            MWXb = WXb .- Xc * (qrX \ WXb)
            Jrr = dot(MWXb, MWXb) / s2 + Tr
            d_rho = dot(e, Wd * y_sar) / s2
            d_lam = dot(e, Wd * e) / s2

            @test isapprox(lms.lag.stat, d_rho^2 / Jrr; rtol = 1e-8)
            @test isapprox(lms.error.stat, d_lam^2 / Tr; rtol = 1e-8)
            @test !isapprox(lms.lag.stat, lms.error.stat; rtol = 1e-6)

            # Anselin, Bera, Florax & Yoon (1996). The lag correction
            # SUBTRACTS the error score; an earlier version added it, which
            # made the robust lag test fire on pure SEM data.
            @test isapprox(lms.robust_lag.stat,
                           (d_rho - d_lam)^2 / (Jrr - Tr); rtol = 1e-8)
            @test isapprox(lms.robust_error.stat,
                           (d_lam - Tr * d_rho / Jrr)^2 / (Tr - Tr^2 / Jrr);
                           rtol = 1e-8)
        end

        # The battery has to separate the two processes, not merely reject.
        # On data generated with a pure error process the robust ERROR test
        # must be the one that fires, and vice versa.
        let L = 14, Wl = make_rook_weights(L), n = size(Wl, 1), Wf = Matrix(Wl)
            g = MersenneTwister(31337)
            Xd = hcat(randn(g, n), randn(g, n)); Xdc = hcat(ones(n), Xd)
            bd = [1.5, -0.8]

            y_sem = Xd * bd .+ (I - 0.5 * Wf) \ randn(g, n)
            t_sem = lm_tests(y_sem, Xdc, Wl)
            @test t_sem.robust_error.p_value < 0.05
            @test t_sem.robust_lag.p_value > 0.05

            y_lag = (I - 0.5 * Wf) \ (Xd * bd .+ randn(g, n))
            t_lag = lm_tests(y_lag, Xdc, Wl)
            @test t_lag.robust_lag.p_value < 0.05
            @test t_lag.robust_error.p_value > 0.05
        end


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
        m_sem_pan = fit_spatial_error(IndivFE(), W, X_pan, y_pan, N, T_per)
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

    @testset "12. Spatial Weights Construction" begin
        rng = MersenneTwister(1200)
        N = 100
        coords = hcat(rand(rng, N) .* 10, rand(rng, N) .* 10)

        # knn weights
        W_knn = knn_weights(coords, 5)
        @test size(W_knn) == (N, N)
        @test all(abs.(vec(sum(W_knn, dims=2)) .- 1.0) .< 1e-10)  # row-standardized

        # knn binary + symmetric
        W_knn_sym = knn_weights(coords, 4; binary=true, symmetrize=true, row_standardize=false)
        @test issymmetric(W_knn_sym) || norm(W_knn_sym - W_knn_sym', Inf) < 1e-10
        @test all(nonzeros(W_knn_sym) .== 1.0)

        # distance-decay weights
        W_dist = distance_weights(coords, 3.0; decay=:inverse)
        @test size(W_dist) == (N, N)
        sc_dist = spatial_connectivity(W_dist)
        @test sc_dist.N == N
        @test sc_dist.avg_neighbors > 0

        # bisquare decay
        W_bisq = distance_weights(coords, 4.0; decay=:bisquare, row_standardize=false)
        @test all(nonzeros(W_bisq) .>= 0.0)
        @test all(nonzeros(W_bisq) .<= 1.0)

        # distance band
        W_band = band_weights(coords, 1.0, 3.0)
        @test size(W_band) == (N, N)

        # block-diagonal
        W1 = make_rook_weights(4); W2 = make_rook_weights(5)
        W_block = block_diagonal_weights([W1, W2])
        @test size(W_block) == (16 + 25, 16 + 25)
        @test nnz(W_block) == nnz(W1) + nnz(W2)

        # standardization
        W_raw = knn_weights(coords, 3; row_standardize=false)
        W_rs = row_standardize(W_raw)
        row_sums = vec(sum(W_rs, dims=2))
        @test all(s -> s ≈ 0.0 || s ≈ 1.0, row_sums)

        W_spec = spectral_standardize(W_raw)
        @test size(W_spec) == (N, N)

        W_mm = minmax_standardize(W_raw)
        @test size(W_mm) == (N, N)

        # connectivity diagnostics
        sc = spatial_connectivity(W_knn)
        @test sc.N == N
        @test sc.n_islands == 0
        @test sc.avg_neighbors ≈ 5.0 atol=0.5
        @test sc.spectral_radius > 0

        # GAL I/O round-trip
        gal_path = tempname() * ".gal"
        ids = string.(1:N)
        W_for_gal = make_rook_weights(10)
        write_gal(gal_path, W_for_gal, string.(1:100))
        W_read, ids_read = read_gal(gal_path)
        @test size(W_read) == size(W_for_gal)
        @test length(ids_read) == 100

        # GWT I/O round-trip
        gwt_path = tempname() * ".gwt"
        write_gwt(gwt_path, W_for_gal, string.(1:100))
        W_read2, ids_read2 = read_gwt(gwt_path)
        @test size(W_read2) == size(W_for_gal)
    end

    @testset "13. Model Selection & Comparison" begin
        rng = MersenneTwister(1300); W = make_rook_weights(15); N = size(W, 1)
        X = randn(rng, N, 2)
        y = (I - 0.5 * W) \ (fill(1.0, N) .+ X * [1.5, -0.8] .+ (W * X) * [0.6, -0.3] .+ 0.5 .* randn(rng, N))

        m_sar = fit_spatial(SAR(), W, X, y)
        m_sdm = fit_spatial(SDM(), W, X, y)

        # AIC/BIC
        aic_sar = spatial_aic(m_sar); aic_sdm = spatial_aic(m_sdm)
        bic_sar = spatial_bic(m_sar); bic_sdm = spatial_bic(m_sdm)
        @test isfinite(aic_sar) && isfinite(aic_sdm)
        @test isfinite(bic_sar) && isfinite(bic_sdm)

        # LR test (SAR nested in SDM)
        lr = lr_test(m_sar, m_sdm)
        @test lr.stat >= 0.0
        @test 0.0 <= lr.p_value <= 1.0
        @test lr.df == 2  # SDM adds 2 WX terms

        # Vuong test
        m_sem = fit_spatial_error(W, X, y)
        vt = vuong_test(m_sar, m_sem)
        @test isfinite(vt.stat)
        @test 0.0 <= vt.p_value <= 1.0
        @test vt.preference in (:model1, :model2, :indistinguishable)

        # Wald test (H₀: β₂ = 0)
        R = zeros(1, length(coef(m_sar)))
        R[1, 3] = 1.0  # third coef = β₂
        r = [0.0]
        wt = wald_test(m_sar, R, r)
        @test wt.stat >= 0.0
        @test 0.0 <= wt.p_value <= 1.0

        # Common factor test
        β_sdm = coef(m_sdm)
        K = 2  # number of X variables
        beta_slope = β_sdm[2:K+1]
        theta = β_sdm[K+2:2K+1]
        rho = m_sdm.rho
        cft = common_factor_test(beta_slope, theta, rho, vcov(m_sdm))
        @test cft.stat >= 0.0
        @test 0.0 <= cft.p_value <= 1.0

        # Comparison table
        tbl = model_comparison_table(m_sar, m_sdm, m_sem; names=["SAR", "SDM", "SEM"])
        @test occursin("SAR", tbl)
        @test occursin("SDM", tbl)
        @test occursin("SEM", tbl)
        @test occursin("AIC", tbl)
    end

    @testset "14. Spatial Panel Diagnostics" begin
        rng = MersenneTwister(1400); L = 10; N = L * L; T_per = 8; NT = N * T_per
        W = make_rook_weights(L)
        X = randn(rng, NT, 2); β_true = [1.5, -0.8]; ρ_true = 0.45
        μ_i = randn(rng, N); ξ_t = randn(rng, T_per)
        S_lu = lu(sparse(1.0I, N, N) - ρ_true * W)
        y = zeros(NT)
        for t in 1:T_per
            idx = ((t-1)*N+1):(t*N)
            y[idx] = S_lu \ (μ_i .+ ξ_t[t] .+ X[idx, :] * β_true .+ 0.5 .* randn(rng, N))
        end

        # Spatial Chow test
        ct = spatial_chow_test(y, X, W, N, T_per, 4)
        @test ct.stat >= 0.0
        @test 0.0 <= ct.p_value <= 1.0
        @test ct.break_period == 4

        # Joint LM tests
        X_with_const = hcat(ones(NT), X)
        plm = panel_lm_tests(y, X_with_const, W, N, T_per)
        @test plm.joint_stat >= 0.0
        @test plm.lag_stat >= 0.0
        @test plm.error_stat >= 0.0
        @test plm.re_stat >= 0.0
        @test 0.0 <= plm.joint_p <= 1.0
        # Should detect spatial dependence
        @test plm.lag_p < 0.10 || plm.error_p < 0.10

        # Panel Moran's I
        m_pan = fit_spatial_panel(SAR(), TwoWayFE(), W, X, y, N, T_per)
        pm = panel_moran(residuals(m_pan), W, N, T_per)
        @test pm.T_periods == T_per
        @test length(pm.period_results) == T_per
        @test isfinite(pm.pooled.I_stat)

        # Hausman test (compare two panel models with different FE)
        m_fe = fit_spatial_panel(SAR(), IndivFE(), W, X, y, N, T_per)
        m_twfe = fit_spatial_panel(SAR(), TwoWayFE(), W, X, y, N, T_per)
        ht = spatial_hausman_test(m_fe, m_twfe)
        @test ht.stat >= 0.0
        @test 0.0 <= ht.p_value <= 1.0
        @test ht.df > 0
    end

    @testset "15. Higher-Order Spatial Models (SLX, SAC, SDEM)" begin
        rng = MersenneTwister(1500); W = make_rook_weights(15); N = size(W, 1)
        X = randn(rng, N, 2)

        # SLX: y = Xβ + WXθ + ε
        θ_true = [0.6, -0.3]
        y_slx = fill(1.0, N) .+ X * [1.5, -0.8] .+ (W * X) * θ_true .+ 0.4 .* randn(rng, N)
        m_slx = fit_slx(W, X, y_slx; varnames=["x1", "x2"])
        @test length(coef(m_slx)) == 5  # intercept + 2β + 2θ
        @test coefnames(m_slx) == ["(Intercept)", "x1", "x2", "W_x1", "W_x2"]
        @test length(residuals(m_slx)) == N
        @test isfinite(loglikelihood(m_slx))

        # SAC: y = ρW₁y + Xβ + u, u = λW₂u + ε
        ρ_true = 0.4; λ_true = 0.3
        IN = sparse(1.0I, N, N)
        ε_sac = 0.5 .* randn(rng, N)
        u_sac = (IN - λ_true * W) \ ε_sac
        y_sac = (IN - ρ_true * W) \ (fill(1.0, N) .+ X * [1.5, -0.8] .+ u_sac)
        m_sac = fit_sac(W, W, X, y_sac; varnames=["x1", "x2"])
        @test length(coef(m_sac)) == 5  # intercept + 2β + ρ + λ
        @test coefnames(m_sac) == ["(Intercept)", "x1", "x2", "ρ", "λ"]
        @test isfinite(loglikelihood(m_sac))
        @test nobs(m_sac) == N
        @test isapprox(m_sac.rho, ρ_true; atol=0.20)

        # SDEM: y = Xβ + WXθ + u, u = λWu + ε
        λ_sdem = 0.45
        u_sdem = (IN - λ_sdem * W) \ (0.4 .* randn(rng, N))
        y_sdem = fill(1.0, N) .+ X * [1.5, -0.8] .+ (W * X) * [0.5, -0.3] .+ u_sdem
        m_sdem = fit_sdem(W, X, y_sdem; varnames=["x1", "x2"])
        @test length(coef(m_sdem)) == 6  # intercept + 2β + 2θ + λ
        @test coefnames(m_sdem) == ["(Intercept)", "x1", "x2", "W_x1", "W_x2", "λ"]
        @test isfinite(loglikelihood(m_sdem))
        @test isapprox(m_sdem.lambda, λ_sdem; atol=0.20)

        # StatsAPI interface completeness
        for m in (m_slx, m_sac, m_sdem)
            @test length(stderror(m)) == length(coef(m))
            @test size(vcov(m)) == (length(coef(m)), length(coef(m)))
            @test dof_residual(m) == nobs(m) - length(coef(m))
        end
    end

    @testset "16. Spatial Regime Models" begin
        rng = MersenneTwister(1600); W = make_rook_weights(15); N = size(W, 1)
        X = randn(rng, N, 2); ρ_true = 0.4
        IN = sparse(1.0I, N, N)

        # Create two regimes: left half vs right half of grid
        labels = [((i - 1) % 15) < 8 ? 1 : 2 for i in 1:N]
        regimes = ExogenousRegimes(labels)

        # DGP with different coefficients per regime
        β_r1 = [2.0, 1.0, -0.5]; β_r2 = [0.5, -0.8, 1.2]
        y = zeros(N)
        for i in 1:N
            β = labels[i] == 1 ? β_r1 : β_r2
            y[i] = β[1] + X[i, :]' * β[2:3]
        end
        y .+= 0.5 .* randn(rng, N)
        y = (IN - ρ_true * W) \ y

        # Fit regime model with spatial lag
        m_reg = fit_spatial_regime(W, X, y, regimes; spatial_lag=true, varnames=["x1", "x2"])
        @test m_reg.n_regimes == 2
        @test m_reg.has_spatial_lag
        @test length(coef(m_reg)) == 7  # 2 regimes × 3 params + ρ
        @test isfinite(loglikelihood(m_reg))

        # Fit without spatial lag
        m_ols = fit_spatial_regime(W, X, y, regimes; spatial_lag=false, varnames=["x1", "x2"])
        @test !m_ols.has_spatial_lag
        @test length(coef(m_ols)) == 6  # 2 regimes × 3 params

        # Regime Chow test
        m_pooled = fit_spatial(SAR(), W, X, y)
        ct = regime_chow_test(m_pooled, m_reg)
        @test ct.stat >= 0.0
        @test 0.0 <= ct.p_value <= 1.0
        @test ct.df == length(coef(m_reg)) - length(coef(m_pooled))

        # SKATER regimes
        coords = hcat(repeat(1.0:15, 15), repeat(1.0:15, inner=15))
        sr = skater_regimes(W, coords, 3)
        @test maximum(sr.labels) == 3
        @test length(sr.labels) == N
        @test minimum(sr.labels) == 1
    end

    @testset "17. Spatial Simulation & Monte Carlo" begin
        rng = MersenneTwister(1700); L = 10; N = L * L
        W = make_rook_weights(L)

        # SAR DGP
        dgp_sar = SAR_DGP(0.5, [1.5, -0.8], 0.5)
        X = randn(rng, N, 2)
        y, _ = simulate_spatial(dgp_sar, W, X; rng=rng)
        @test length(y) == N
        @test all(isfinite, y)

        # SEM DGP
        dgp_sem = SEM_DGP(0.4, [1.0, -0.5], 0.6)
        y_sem, _ = simulate_spatial(dgp_sem, W, X; rng=rng)
        @test length(y_sem) == N

        # SDM DGP
        dgp_sdm = SDM_DGP(0.3, [1.0, -0.5], [0.4, -0.2], 0.5)
        y_sdm, _ = simulate_spatial(dgp_sdm, W, X; rng=rng)
        @test length(y_sdm) == N

        # SLX DGP
        dgp_slx = SLX_DGP([1.0, -0.5], [0.3, -0.1], 0.4)
        y_slx, _ = simulate_spatial(dgp_slx, W, X; rng=rng)
        @test length(y_slx) == N

        # SAC DGP
        dgp_sac = SAC_DGP(0.3, 0.2, [1.0, -0.5], 0.5)
        y_sac, _ = simulate_spatial(dgp_sac, W, X; rng=rng)
        @test length(y_sac) == N

        # SDEM DGP
        dgp_sdem = SDEM_DGP(0.4, [1.0, -0.5], [0.3, -0.2], 0.5)
        y_sdem, _ = simulate_spatial(dgp_sdem, W, X; rng=rng)
        @test length(y_sdem) == N

        # Monte Carlo with few reps (just checking it runs)
        mc = monte_carlo_spatial(
            SAR_DGP(0.5, [1.5, -0.8], 0.5), W, N, 2,
            (W, X, y) -> fit_spatial(SAR(), W, X, y);
            config=MonteCarloConfig(n_replications=10, seed=42)
        )
        @test length(mc.param_means) == 3  # β1, β2, ρ
        @test length(mc.param_bias) == 3
        @test all(isfinite, mc.param_rmse)
        @test haskey(mc.rejection_rates, "moran_05")
    end

    @testset "18. Spatial Tables & Output" begin
        rng = MersenneTwister(1800); W = make_rook_weights(10); N = size(W, 1)
        X = randn(rng, N, 2)
        y = (I - 0.4 * W) \ (fill(1.0, N) .+ X * [1.5, -0.8] .+ 0.5 .* randn(rng, N))

        m = fit_spatial(SAR(), W, X, y)

        # CoefTable
        ct = coeftable(m)
        @test length(ct.names) == length(coef(m))
        @test length(ct.estimate) == length(coef(m))
        @test all(isfinite, ct.z_stat)
        @test all(p -> 0.0 <= p <= 1.0, ct.p_value)
        @test all(ct.ci_lower .<= ct.ci_upper)

        # Tables.jl interface
        @test Tables.istable(typeof(ct))
        @test :name in Tables.columnnames(ct)
        @test :estimate in Tables.columnnames(ct)
        col = Tables.getcolumn(ct, :estimate)
        @test length(col) == length(coef(m))

        # Export formats
        latex = to_latex(ct)
        @test occursin("\\begin{tabular}", latex)
        @test occursin("\\end{tabular}", latex)

        csv = to_csv(ct)
        @test occursin("name,estimate,std_error", csv)
        lines = split(strip(csv), '\n')
        @test length(lines) == length(coef(m)) + 1  # header + rows

        md = to_markdown(ct)
        @test occursin("| Variable |", md)

        # Summary stats
        ss = summary_stats(m; model_type="SAR")
        @test ss.n == N
        @test ss.k == length(coef(m))
        @test isfinite(ss.aic)
        @test isfinite(ss.bic)
        @test 0.0 <= ss.r2 <= 1.0

        # Multi-model regression table
        m_sem = fit_spatial_error(W, X, y)
        rt = regression_table(m, m_sem; model_names=["SAR", "SEM"])
        @test occursin("SAR", rt)
        @test occursin("SEM", rt)
        @test occursin("N", rt)
        @test occursin("Log-Lik", rt)
    end

    @testset "19. Spatial Discrete Choice (Probit)" begin
        rng = MersenneTwister(1900); L = 14; W = make_rook_weights(L); N = size(W, 1)
        X = randn(rng, N, 2)
        X_aug = hcat(ones(N), X)
        β_true = [0.3, 1.2, -0.9]; ρ_true = 0.4
        IN = sparse(1.0I, N, N)
        y_star = (IN - ρ_true * W) \ (X_aug * β_true .+ randn(rng, N))
        y = Float64.(y_star .> 0)
        # Guard against a degenerate draw making the tests vacuous.
        @test 0.2 < mean(y) < 0.8

        # ── Linearized GMM (Klier-McMillen) ──
        m_gmm = fit_spatial_probit(W, X, y; varnames=["x1", "x2"])
        @test length(coef(m_gmm)) == 4  # intercept + 2β + ρ
        @test coefnames(m_gmm) == ["(Intercept)", "x1", "x2", "ρ"]
        @test size(vcov(m_gmm)) == (4, 4)
        @test length(stderror(m_gmm)) == 4
        @test all(isfinite, coef(m_gmm))
        @test nobs(m_gmm) == N
        @test dof_residual(m_gmm) == N - 4
        @test abs(m_gmm.rho) < 1.0
        @test isfinite(loglikelihood(m_gmm))
        @test loglikelihood(m_gmm) < 0.0

        # Fitted probabilities are genuine probabilities and beat a coin flip.
        probs = predict(m_gmm)
        @test length(probs) == N
        @test all(p -> 0.0 <= p <= 1.0, probs)
        @test mean((probs .>= 0.5) .== (y .== 1.0)) > 0.65
        @test isapprox(residuals(m_gmm), y .- probs; atol=1e-10)

        # Slope signs must be recovered; ρ̂ is high-variance at this N
        # (SD ≈ 0.25 over seeds), so the band is deliberately wide.
        @test coef(m_gmm)[2] > 0.0
        @test coef(m_gmm)[3] < 0.0
        @test isapprox(m_gmm.rho, ρ_true; atol=0.35)

        # ── Marginal effects ──
        eff = marginal_effects(m_gmm)
        @test length(eff.direct) == 2
        @test eff.varnames == ["x1", "x2"]
        @test all(isfinite, eff.total)
        # total = direct + indirect, by construction
        @test isapprox(eff.direct .+ eff.indirect, eff.total; atol=1e-10)
        # Effects inherit the sign of their coefficient.
        @test eff.direct[1] > 0.0 && eff.total[1] > 0.0
        @test eff.direct[2] < 0.0 && eff.total[2] < 0.0
        # With ρ > 0 the spillover runs the same way as the direct effect.
        @test eff.indirect[1] > 0.0

        # ── Bayesian Gibbs (Albert-Chib) ──
        m_bayes = fit_bayesian_spatial_probit(W, X, y; n_samples=600, burnin=300,
            varnames=["x1", "x2"], rng=MersenneTwister(1901))
        @test length(coef(m_bayes)) == 4
        @test coefnames(m_bayes) == ["(Intercept)", "x1", "x2", "ρ"]
        @test size(m_bayes.beta_draws) == (600, 3)
        @test length(m_bayes.rho_draws) == 600
        @test all(r -> -1.0 < r < 1.0, m_bayes.rho_draws)
        @test all(p -> 0.0 <= p <= 1.0, predict(m_bayes))
        # The default proposal should land in the usable acceptance band.
        @test 0.15 < m_bayes.accept_rate < 0.6
        @test coef(m_bayes)[2] > 0.0
        @test coef(m_bayes)[3] < 0.0
        @test isapprox(m_bayes.rho, ρ_true; atol=0.3)
        # Posterior SD must be positive — a stuck chain would show zero spread.
        @test std(m_bayes.rho_draws) > 1e-4

        # ── Ordered probit (3 categories, no intercept) ──
        rng_o = MersenneTwister(1902)
        z_ord = (IN - 0.35 * W) \ (X * [1.0, -0.7] .+ randn(rng_o, N))
        y_ord = [zi <= 0.0 ? 1 : (zi <= 1.2 ? 2 : 3) for zi in z_ord]
        @test all(j -> count(==(j), y_ord) >= 10, 1:3)

        m_ord = fit_spatial_ordered_probit(W, X, y_ord; n_samples=600, burnin=300,
            varnames=["x1", "x2"], rng=MersenneTwister(1903))
        @test m_ord.n_categories == 3
        @test length(coef(m_ord)) == 3  # 2β + ρ (no intercept)
        @test coefnames(m_ord) == ["x1", "x2", "ρ"]
        @test length(m_ord.cutpoints) == 2
        @test m_ord.cutpoints[1] == 0.0          # τ₁ fixed for identification
        @test m_ord.cutpoints[2] > m_ord.cutpoints[1]  # cutpoints must be ordered
        @test size(m_ord.cut_draws) == (600, 2)
        @test coef(m_ord)[1] > 0.0
        @test coef(m_ord)[2] < 0.0
        @test isapprox(m_ord.rho, 0.35; atol=0.3)
        @test 0.15 < m_ord.accept_rate < 0.6

        # ── Input validation ──
        @test_throws AssertionError fit_spatial_probit(W, X, fill(2.0, N))
        @test_throws AssertionError fit_spatial_probit(W, X, y[1:end-1])
        @test_throws AssertionError fit_spatial_ordered_probit(W, X, y_ord; n_categories=1)
    end

    @testset "20. Gravity / Flow Models (PPML & NB)" begin
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

    @testset "21. Spatial Filtering (ESF / RE-ESF)" begin
        W = make_rook_weights(11); N = size(W, 1)
        IN = sparse(1.0I, N, N)
        S0 = sum(W)

        # ── Eigenvector basis identities (all exact) ──
        se = spatial_eigenvectors(W)
        @test se.N == N
        @test length(se.values) == N - 1           # complement of the constant
        @test size(se.vectors) == (N, N - 1)
        @test issorted(se.values; rev=true)

        # Each eigenvector's own Moran coefficient equals (N/S0)*lambda.
        emp = [moran_coefficient(se.vectors[:, i], W) for i in 1:length(se.values)]
        @test maximum(abs.(emp .- se.moran)) < 1e-8
        # Orthonormal columns.
        @test maximum(abs.(transpose(se.vectors) * se.vectors - I)) < 1e-8
        # Orthogonal to the constant vector, so the intercept is never absorbed.
        @test maximum(abs.(vec(sum(se.vectors, dims=1)))) < 1e-8
        # Centred spectrum sums to tr(M*Wsym*M) = -S0/N.
        @test sum(se.values) ≈ -S0 / N atol = 1e-8

        # Screening keeps a strict subset, ordered by Moran coefficient.
        se_scr = spatial_eigenvectors(W; min_moran_ratio=0.25)
        @test length(se_scr.values) < length(se.values)
        @test minimum(se_scr.moran) >= 0.25 * maximum(se.moran) - 1e-8

        # ── Gaussian ESF on SAR-generated data ──
        rng = MersenneTwister(5)
        X = randn(rng, N, 2)
        y = (IN - 0.6 * W) \ (1.0 .+ X * [1.5, -0.8] .+ 0.5 * randn(rng, N))
        m = fit_esf(X, y, W; family=:gaussian, max_eigenvectors=15, varnames=["x1", "x2"])
        @test nobs(m) == N
        @test coefnames(m)[1] == "(Intercept)"
        @test length(coef(m)) == 3 + length(m.selected)
        @test m.n_covariates == 2
        # Filtering must leave less residual autocorrelation than OLS.
        @test abs(m.moran_final) < abs(m.moran_initial)
        @test !isempty(m.selected)
        s = selected_eigenvectors(m)
        @test s.indices == m.selected
        @test length(s.moran) == length(s.indices)
        @test allunique(s.indices)
        @test length(residuals(m)) == N
        @test fitted(m) ≈ predict(m)

        # ── Poisson ESF: slope recovery ──
        function rpois(rng, lam)
            L = exp(-lam); k = 0; p = 1.0
            while true
                p *= rand(rng)
                p <= L && return Float64(k)
                k += 1
                k > 10_000 && return Float64(k)
            end
        end
        rng = MersenneTwister(31)
        Xc = randn(rng, N, 2)
        sig = 1.2 * se_scr.vectors[:, 2] * sqrt(N) * 0.4 +
              0.8 * se_scr.vectors[:, 5] * sqrt(N) * 0.4
        eta = 0.6 .+ Xc * [0.5, -0.35] .+ sig
        yp = [rpois(rng, exp(e)) for e in eta]
        mp = fit_esf(Xc, yp, W; family=:poisson, max_eigenvectors=8,
                     eigen_basis=se_scr, varnames=["x1", "x2"])
        @test coef(mp)[2] ≈ 0.5 atol = 0.15
        @test coef(mp)[3] ≈ -0.35 atol = 0.15
        @test all(>(0.0), fitted(mp))

        # ── Logit ESF: valid probabilities ──
        yb = Float64[(1.0 / (1.0 + exp(-e))) > rand(rng) ? 1.0 : 0.0 for e in eta]
        ml = fit_esf(Xc, yb, W; family=:logit, max_eigenvectors=8,
                     eigen_basis=se_scr, varnames=["x1", "x2"])
        @test all(p -> 0.0 <= p <= 1.0, fitted(ml))
        @test isfinite(loglikelihood(ml))

        # ── Negative binomial ESF: dispersion is estimated and positive ──
        yn = Float64[]
        for e in eta
            g = sum(-log(rand(rng)) for _ in 1:2) * 0.5   # mean-1 gamma, shape 2
            push!(yn, rpois(rng, exp(e) * g))
        end
        mn = fit_esf(Xc, yn, W; family=:negbin, max_eigenvectors=8,
                     eigen_basis=se_scr, varnames=["x1", "x2"])
        @test mn.alpha > 0.0
        @test var(yn) / mean(yn) > 1.0                    # genuinely overdispersed
        @test abs(mn.moran_final) < abs(mn.moran_initial)

        # ── RE-ESF on correctly specified data ──
        pos = findall(>(1e-10), se_scr.values)
        E = se_scr.vectors[:, pos]; lam = se_scr.values[pos]; L = length(pos)
        rng = MersenneTwister(3)
        Xr = randn(rng, N, 2)
        gam = sqrt(0.8) .* sqrt.(lam) .* randn(rng, L)
        yr = 1.0 .+ Xr * [1.5, -0.8] .+ E * gam .+ 0.5 .* randn(rng, N)
        mr = fit_re_esf(Xr, yr, W; varnames=["x1", "x2"])
        @test length(coef(mr)) == 3
        @test coef(mr)[1] ≈ 1.0 atol = 0.25
        @test coef(mr)[2] ≈ 1.5 atol = 0.25
        @test coef(mr)[3] ≈ -0.8 atol = 0.25
        @test mr.sigma2 ≈ 0.25 atol = 0.15
        @test mr.theta > 0.0
        @test mr.tau2 ≈ mr.theta * mr.sigma2
        @test length(mr.gamma) == L

        # Effective df sits strictly between the fixed-effect count and N.
        @test 3.0 < mr.edf < Float64(N)

        # The conditional residual Moran is expected to be NEGATIVE: removing Eγ̂
        # strips the positive-MC directions. The benchmark is the complement's
        # expectation, not zero.
        nrest = N - 1 - L
        ref_closed = (N / S0) * ((-S0 / N) - sum(lam)) / nrest
        @test mr.moran_reference ≈ ref_closed atol = 1e-10
        # Closed form must equal the mean MC of the non-retained eigenvectors.
        keep = sortperm(se.values; rev=true)[1:L]
        rest = setdiff(1:length(se.values), keep)
        @test mr.moran_reference ≈ mean(se.moran[rest]) atol = 1e-8
        @test mr.moran_reference < 0.0
        # A healthy fit lands at or above that reference, not near zero.
        @test mr.moran_final > mr.moran_reference - 0.15
        @test mr.moran_final < 0.0
        # The random effects really did absorb the positive-MC content.
        share_before = norm(transpose(E) * (yr - hcat(ones(N), Xr) * (hcat(ones(N), Xr) \ yr)))
        share_after = norm(transpose(E) * residuals(mr))
        @test share_after < share_before

        # V₀⁻¹ = I - E D E' is a genuine inverse of I + θEΛE'.
        d = (mr.theta .* lam) ./ (1.0 .+ mr.theta .* lam)
        V0 = I + mr.theta * (E * (lam .* transpose(E)))
        V0inv = I - E * (d .* transpose(E))
        @test maximum(abs.(V0 * V0inv - I)) < 1e-8

        # ── Input validation ──
        @test_throws AssertionError fit_esf(X, y[1:end-1], W)
        @test_throws AssertionError fit_re_esf(X, y[1:end-1], W)
    end

    @testset "22. Regressions found by the example scripts" begin
        W = make_rook_weights(12); N = size(W, 1)
        IN = sparse(1.0I, N, N)
        rng = MersenneTwister(4242)
        X = randn(rng, N, 2)
        y = (IN - 0.5 * W) \ (1.0 .+ X * [1.5, -0.8] .+ 0.5 * randn(rng, N))

        # ── show() must not throw ──
        # Several modules called nobs/stderror inside Base.show without
        # importing them from StatsAPI. The suite never rendered a model, so
        # printing any of these threw UndefVarError for users.
        for m in (fit_slx(W, X, y), fit_sdem(W, X, y), fit_sac(W, W, X, y),
                  fit_esf(X, y, W; max_eigenvectors=5), fit_re_esf(X, y, W),
                  fit_spatial_regime(W, X, y,
                      ExogenousRegimes([i <= N ÷ 2 ? 1 : 2 for i in 1:N])))
            str = sprint(show, m)
            @test !isempty(str)
            @test occursin("N = ", str) || occursin("N  ", str)
        end

        # ── chi-squared survival function ──
        # The regime module used a truncated lower-incomplete-gamma series that
        # only converges for z < a+1. Past that it returned a tiny partial sum,
        # so the survival function came back near 1 and a decisive rejection was
        # reported as "fail to reject".
        RM = SpatialEconometrics.SpatialRegimeModels
        MS = SpatialEconometrics.SpatialModelSelection
        for (x, df) in [(0.5, 1), (3.841, 1), (6.635, 1), (7.815, 3),
                        (11.345, 3), (2.0, 3), (100.0, 3), (50.0, 10),
                        (1.0, 10), (499.5, 3), (1000.0, 5)]
            a = RM._chi2_sf(x, df)
            @test 0.0 <= a <= 1.0
            # Must agree with the independently written implementation.
            @test a ≈ MS._chi2_sf(x, df) rtol = 1e-8
        end
        # Reference values (R: pchisq(q, df, lower.tail=FALSE)).
        @test RM._chi2_sf(3.841, 1) ≈ 0.05 atol = 1e-4
        @test RM._chi2_sf(7.815, 3) ≈ 0.05 atol = 1e-4
        @test RM._chi2_sf(11.345, 3) ≈ 0.01 atol = 1e-4
        @test RM._chi2_sf(0.5, 1) ≈ 0.4795001 atol = 1e-6
        # The case that was silently wrong: a huge statistic must reject.
        @test RM._chi2_sf(499.5, 3) < 1e-100
        # log-gamma must stay finite where Gamma itself would overflow.
        @test isfinite(RM._chi2_sf(1000.0, 500))

        # A regime Chow test on a genuine structural break must reject.
        β_a = [1.0, 2.0, -0.5]; β_b = [1.0, 0.3, -1.6]
        labels = [i <= N ÷ 2 ? 1 : 2 for i in 1:N]
        η = similar(X[:, 1])
        for i in 1:N
            b = labels[i] == 1 ? β_a : β_b
            η[i] = b[1] + X[i, 1] * b[2] + X[i, 2] * b[3]
        end
        y_br = (IN - 0.45 * W) \ (η .+ 0.4 .* randn(MersenneTwister(9), N))
        m_pool = fit_spatial(SpatialEconometrics.SAR(), W, X, y_br)
        m_reg = fit_spatial_regime(W, X, y_br, ExogenousRegimes(labels))
        chow = regime_chow_test(m_pool, m_reg)
        @test chow.stat > 0.0
        @test chow.df == length(coef(m_reg)) - length(coef(m_pool))
        @test chow.p_value < 1e-6          # was reported as 0.998

        # ── SKATER must not produce degenerate regions ──
        # Cutting the LONGEST MST edge peels off a single outlying leaf, which
        # left the downstream regime regression rank-deficient.
        signal = hcat(Float64.(labels) .+ 0.3 .* randn(MersenneTwister(11), N), X[:, 1])
        for k in (2, 3)
            reg = skater_regimes(W, signal, k)
            sizes = [count(==(r), reg.labels) for r in 1:k]
            @test length(reg.labels) == N
            @test sort(unique(reg.labels)) == collect(1:k)
            @test minimum(sizes) >= 2        # no singleton regions
            @test sum(sizes) == N
        end
        # min_size is honoured, and the regions stay usable for a regression.
        reg20 = skater_regimes(W, signal, 2; min_size = 20)
        @test minimum([count(==(r), reg20.labels) for r in 1:2]) >= 20
        @test length(coef(fit_spatial_regime(W, X, y_br, reg20))) == 7
        @test_throws AssertionError skater_regimes(W, signal, 2; min_size = N)

        # ── Monte Carlo parameter alignment ──
        # The DGPs omit an intercept but the estimators fit one, so taking the
        # leading n_params coefficients compared the intercept against β₁ and
        # reported the off-by-one as enormous bias with a tiny SD.
        Wmc = make_rook_weights(10); Nmc = size(Wmc, 1)
        CSx = SpatialEconometrics.CrossSection
        mc = monte_carlo_spatial(
            SAR_DGP(0.5, [1.5, -0.8], 0.5), Wmc, Nmc, 2,
            (Wm, Xm, ym) -> fit_spatial(CSx.SAR(), Wm, Xm, ym);
            config = MonteCarloConfig(n_replications = 20, seed = 7))
        @test mc.n_successful == 20          # was 0 — every replication threw
        @test mc.n_replications == 20
        @test mc.param_names == ["β1", "β2", "ρ"]
        @test mc.param_means[1] ≈ 1.5 atol = 0.15
        @test mc.param_means[2] ≈ -0.8 atol = 0.15
        @test mc.param_means[3] ≈ 0.5 atol = 0.15
        @test all(abs.(mc.param_bias) .< 0.15)
        @test all(mc.param_stds .>= 0.0)
        # Diagnostics ran rather than being skipped by the estimator's `continue`.
        @test mc.rejection_rates["moran_05"] > 0.5
        @test !isempty(sprint(show, mc))

        # ── spatial_hausman_test must match coefficients by NAME ──
        # It aligned the two coefficient vectors by position, so comparing an
        # FE fit (no intercept) with a pooled fit (intercept) contrasted
        # β_income with the intercept and returned a statistic of 7.8e6: a
        # decisive-looking rejection assembled from unrelated parameters.
        let
            rngh = MersenneTwister(1717)
            Lh = 8; Nh = Lh * Lh; Th = 5; NTh = Nh * Th
            Wh = make_rook_weights(Lh)
            Sh = lu(sparse(1.0I, Nh, Nh) - 0.4 * Wh)
            Xh = randn(rngh, NTh, 2)
            yh = zeros(NTh)
            μh = 2.0 .* randn(rngh, Nh)
            for t in 1:Th
                idx = ((t - 1) * Nh + 1):(t * Nh)
                yh[idx] = Sh \ (μh .+ Xh[idx, :] * [1.0, -0.5] .+ 0.5 .* randn(rngh, Nh))
            end
            fp(fe) = fit_spatial_panel(SAR(), fe, Wh, Xh, yh, Nh, Th;
                                       varnames = ["a", "b"])
            m_feh = fp(IndivFE())
            m_pooh = fp(NoFE())

            # The premise of the bug: the two layouts genuinely differ.
            @test coefnames(m_feh) == ["a", "b", "ρ"]
            @test coefnames(m_pooh) == ["(Intercept)", "a", "b", "ρ"]

            ht = spatial_hausman_test(m_feh, m_pooh)
            # The pooled intercept has no FE counterpart, so it drops out.
            @test ht.coefnames == ["a", "b", "ρ"]
            @test ht.df == 3
            @test ht.stat >= 0.0
            @test 0.0 <= ht.p_value <= 1.0

            # Swapping the arguments compares the same set of names.
            @test spatial_hausman_test(m_pooh, m_feh).coefnames == ["a", "b", "ρ"]

            # Comparing a model with itself is the degenerate case: identical
            # coefficients, so the statistic must be zero and p must be 1.
            ht_self = spatial_hausman_test(m_feh, m_feh)
            @test ht_self.stat ≈ 0.0 atol = 1e-6
            @test ht_self.p_value ≈ 1.0 atol = 1e-8

            # `names` restricts the comparison to a chosen subset.
            ht_sl = spatial_hausman_test(m_feh, m_pooh; names = ["a", "b"])
            @test ht_sl.coefnames == ["a", "b"]
            @test ht_sl.df == 2

            # An unknown name is an error, not a silently smaller comparison.
            @test_throws ErrorException spatial_hausman_test(m_feh, m_pooh;
                                                            names = ["nope"])
            # The compared names appear in the printed output.
            @test occursin("a, b, ρ", sprint(show, ht))
        end
    end

    @testset "23. Regressions found by the full sweep" begin
        W = make_rook_weights(10); N = size(W, 1)
        IN = sparse(1.0I, N, N)
        rng = MersenneTwister(606)
        X = randn(rng, N, 2)
        y = (IN - 0.5 * W) \ (1.0 .+ X * [1.5, -0.8] .+ 0.5 * randn(rng, N))
        coords = hcat(rand(rng, N) .* 10, rand(rng, N) .* 10)
        CSx = SpatialEconometrics.CrossSection
        SPx = SpatialEconometrics.StaticPanel

        # ── robust_coeftable must be callable ──
        # It lives in the umbrella module, which has both `using StatsAPI` and
        # `using StatsModels`. That makes the bare name `vcov` ambiguous, so it
        # resolved to nothing and every call raised UndefVarError.
        m_sar = fit_spatial(CSx.SAR(), W, X, y; varnames=["x1", "x2"])
        rt = robust_coeftable(m_sar, ConleySpatialHAC(coords, 3.0))
        @test !isempty(sprint(show, rt))
        @test length(rt.coef) == length(coef(m_sar))
        @test length(rt.se_robust) == length(coef(m_sar))
        @test all(rt.se_robust .>= 0.0)
        @test all(isfinite, rt.se_robust)
        # Robust SEs differ from the model's own, but not wildly.
        @test rt.se_baseline ≈ stderror(m_sar) rtol = 1e-10
        @test all(rt.se_robust .< 50 .* rt.se_baseline)
        # t-stats must be coef / robust se.
        @test rt.z_robust ≈ coef(m_sar) ./ rt.se_robust rtol = 1e-10

        # Panel variants exercise the other robust_vcov methods.
        Lp = 7; Np = Lp * Lp; Tp = 4; NTp = Np * Tp
        Wp = make_rook_weights(Lp)
        Xp = randn(rng, NTp, 2)
        Slu = lu(sparse(1.0I, Np, Np) - 0.4 * Wp)
        mu = randn(rng, Np)
        yp = zeros(NTp)
        for t in 1:Tp
            idx = ((t-1)*Np+1):(t*Np)
            yp[idx] = Slu \ (mu .+ Xp[idx, :] * [1.5, -0.8] .+ 0.5 .* randn(rng, Np))
        end
        m_pan = fit_spatial_panel(SPx.SAR(), SPx.TwoWayFE(), Wp, Xp, yp, Np, Tp)
        coords_p = hcat(rand(rng, Np) .* 10, rand(rng, Np) .* 10)
        for spec in (DriscollKraayVCOV(), ConleySpatialHAC(coords_p, 3.0))
            rtp = robust_coeftable(m_pan, spec)
            @test !isempty(sprint(show, rtp))
            @test all(isfinite, rtp.se_robust)
            @test all(rtp.se_robust .>= 0.0)
        end

        # ── balance_gras must be differentiable ──
        # The dual path factorized with `ldlt(Symmetric(dense))`, which has no
        # method, so every ForwardDiff call through balance_gras threw.
        X0 = [10.0 -2.0 5.0; 3.0 12.0 -1.0; 6.0 4.0 9.0]
        u = [14.0, 15.0, 20.0]
        v = [18.0, 15.0, 16.0]
        v = v .* (sum(u) / sum(v))

        # GRAS pins row sums to u, so sum(X) == sum(u) and d sum(X)/du is exactly 1.
        g = ForwardDiff.gradient(uu -> sum(balance_gras(X0, uu, v)), u)
        @test all(isfinite, g)
        @test g ≈ ones(3) atol = 1e-8

        # The reverse-mode rule must agree with the primal and the same analytic gradient.
        yb, pb = ChainRulesCore.rrule(balance_gras, X0, u, v)
        @test yb ≈ balance_gras(X0, u, v) atol = 1e-10
        dvals = pb(ones(size(yb)))
        @test collect(dvals[3]) ≈ ones(3) atol = 1e-8

        # Gradient wrt X0 must match central differences.
        f(x) = sum(abs2, balance_gras(reshape(x, 3, 3), u, v))
        x0 = vec(X0)
        g_ad = ForwardDiff.gradient(f, x0)
        g_fd = similar(g_ad)
        for i in eachindex(x0)
            h = 1e-6
            xp2 = copy(x0); xp2[i] += h
            xm2 = copy(x0); xm2[i] -= h
            g_fd[i] = (f(xp2) - f(xm2)) / (2h)
        end
        @test maximum(abs.(g_ad .- g_fd) ./ max.(abs.(g_fd), 1e-8)) < 1e-5

        # ── power_analysis rates are real rates ──
        # Each count was divided by n_reps even when a diagnostic threw, so power
        # was understated and an all-failure run reported 0.0, which is
        # indistinguishable from a test having no power.
        pa = power_analysis(r -> SAR_DGP(r, [1.5, -0.8], 0.5), [6, 8], [0.0, 0.6];
                            n_reps = 30, seed = 3)
        for M in (pa.power_moran, pa.power_lm_lag, pa.power_lm_error)
            @test all(isfinite, M)                      # NaN only if nothing ran
            @test all(0.0 .<= M .<= 1.0)
            # Power must rise with rho, and the rho=0 column is the test's size.
            @test M[1, 1] < 0.35
            @test M[end, end] > M[end, 1]
        end

        # ── one chi-squared survival function, shared ──
        # There used to be three copies, which is how one of them came to be
        # wrong: a truncated lower-gamma series reported p = 0.9977 for an LR of
        # 499.5 on 3 df. They now resolve to a single definition in SpatialUtils,
        # so identity — not numerical agreement — is the invariant to pin.
        UTx = SpatialEconometrics.SpatialUtils
        MSx = SpatialEconometrics.SpatialModelSelection
        PDx = SpatialEconometrics.SpatialPanelDiagnostics
        RMx = SpatialEconometrics.SpatialRegimeModels
        @test MSx._chi2_sf === UTx._chi2_sf
        @test PDx._chi2_sf === UTx._chi2_sf
        @test RMx._chi2_sf === UTx._chi2_sf
        # PD keeps the F / incomplete-beta helpers, so its log Gamma must be the
        # shared one too rather than a fourth private copy.
        @test PDx._loggamma === UTx._loggamma
        @test PDx._f_sf(2.0, 4, 8) ≈ 0.1875 rtol = 1e-9

        for df in (1, 2, 3, 6, 10, 50), x in (0.5, 2.0, 7.815, 25.0, 100.0)
            q = UTx._chi2_sf(x, df)
            @test isfinite(q)
            @test 0.0 <= q <= 1.0
        end
        # Standard critical values, and the degenerate cases.
        @test UTx._chi2_sf(3.841, 1) ≈ 0.05 atol = 5e-5
        @test UTx._chi2_sf(7.815, 3) ≈ 0.05 atol = 5e-5
        @test UTx._chi2_sf(0.0, 3) == 1.0      # nothing to reject
        @test UTx._chi2_sf(10.0, 0) == 1.0     # no restriction to reject
        # The case that exposed the bug: a large LR must stay decisive.
        @test UTx._chi2_sf(499.5, 3) < 1e-100
        # Exact closed form at even df: Q(x; 2k) = exp(-x/2) * sum_{j<k} (x/2)^j / j!
        for (x, df) in [(2.0, 2), (5.0, 4), (10.0, 6), (3.0, 10)]
            k = df ÷ 2; z = x / 2
            want = exp(-z) * sum(z^j / factorial(j) for j in 0:(k-1))
            @test UTx._chi2_sf(x, df) ≈ want rtol = 1e-7
        end

        # ── GAL / GWT readers must preserve the unit ordering ──
        # Both readers assigned matrix indices to ids on FIRST APPEARANCE, so a
        # neighbour mentioned before its own record landed at the wrong index and
        # the matrix came back silently permuted: on a 3x3 rook grid unit 1's
        # neighbours {2, 4} were read back as {2, 3}.
        mktempdir() do dir
            Wio = make_rook_weights(5)
            idv = collect(1:size(Wio, 1))

            galp = joinpath(dir, "rt.gal")
            write_gal(galp, Wio, idv)
            Wg, idg = read_gal(galp)
            @test size(Wg) == size(Wio)
            @test idg == string.(idv)
            # GAL stores adjacency only, so check the pattern as well as the values.
            @test Set(zip(findnz(Wg)[1], findnz(Wg)[2])) ==
                  Set(zip(findnz(Wio)[1], findnz(Wio)[2]))
            @test maximum(abs.(Wg .- Wio)) < 1e-12

            gwtp = joinpath(dir, "rt.gwt")
            write_gwt(gwtp, Wio, idv)
            Ww, idw = read_gwt(gwtp)
            @test size(Ww) == size(Wio)
            @test idw == string.(idv)
            @test maximum(abs.(Ww .- Wio)) < 1e-12

            # Non-contiguous ids: GAL must map by record order, and GWT by the
            # ordering it is handed, not by order of appearance in the edge list.
            odd = [101, 7, 55, 3, 99, 42, 88, 12, 76]
            W3 = make_rook_weights(3)
            g3 = joinpath(dir, "odd.gal")
            write_gal(g3, W3, odd)
            W3r, id3 = read_gal(g3)
            @test id3 == string.(odd)
            @test maximum(abs.(W3r .- W3)) < 1e-12

            w3 = joinpath(dir, "odd.gwt")
            write_gwt(w3, W3, odd)
            W3w, _ = read_gwt(w3; ids = odd)
            @test maximum(abs.(W3w .- W3)) < 1e-12

            # A neighbour id with no record of its own is a malformed file, not
            # something to paper over with a silently invented index.
            bad = joinpath(dir, "bad.gal")
            write(bad, "0 2\n1 1\n2\n2 1\n9\n")
            @test_throws ErrorException read_gal(bad)
        end
    end

    @testset "24. Extension Entry Points" begin
        # The extension functions are declared by the parent, so
        # `using SpatialEconometrics` alone makes them reachable. An
        # extension cannot usefully export anything of its own: nothing ever
        # `using`s it, so a function it declared itself would be invisible.
        for f in (:plot_coefficients, :plot_moran_scatter, :plot_local_moran,
                  :plot_gwr_coefficients, :plot_spatial_effects,
                  :bayesian_sar, :bayesian_sdm, :bayesian_sem)
            @test isdefined(SpatialEconometrics, f)
            @test Base.isexported(SpatialEconometrics, f)
        end
        @test isdefined(SpatialEconometrics, :AbstractBayesianSpatialResult)
        @test isabstracttype(AbstractBayesianSpatialResult)

        # Calling one without its weak dependency must say which package to
        # load, rather than raise a bare MethodError.
        for (f, pkg) in ((plot_coefficients, "Makie"), (plot_moran_scatter, "Makie"),
                         (bayesian_sar, "Turing"), (bayesian_sem, "Turing"))
            err = try; f(nothing); catch e; e; end
            @test err isa ErrorException
            @test occursin(pkg, err.msg)
        end
    end

    # The remaining two testsets need the weak dependencies. They are skipped
    # when those are not installed, so a plain `Pkg.test()` still runs, and
    # they are exercised by the `extensions` CI job, which installs them.
    _HAVE_MAKIE  = !isnothing(Base.identify_package("Makie"))
    _HAVE_TURING = !isnothing(Base.identify_package("Turing")) &&
                   !isnothing(Base.identify_package("Distributions"))

    @testset "25. Makie Extension" begin
        if !_HAVE_MAKIE
            @info "Makie is not installed: skipping the Makie extension tests."
            @test_skip false
        else
            @eval using Makie
            @test !isnothing(Base.get_extension(SpatialEconometrics,
                                                :SpatialEconometricsMakieExt))

            rng = MersenneTwister(5); L = 8; N = L * L
            W = make_rook_weights(L)
            coords = hcat(repeat(1.0:L, outer = L), repeat(1.0:L, inner = L))
            X = randn(rng, N, 2)
            y = (I - 0.5 * W) \ (1.0 .+ X * [1.5, -0.8] .+ 0.5 .* randn(rng, N))
            m = fit_spatial(SAR(), W, X, y; varnames = ["income", "density"])

            @test plot_coefficients(m) isa Makie.Figure
            @test plot_moran_scatter(y, W) isa Makie.Figure
            @test plot_spatial_effects(spatial_effects(m)) isa Makie.Figure

            lisa = moran_local(y, W; n_perm = 99, rng = MersenneTwister(1))
            @test plot_local_moran(lisa, coords) isa Makie.Figure

            g = fit_gwr(X, y, coords; bandwidth = AdaptiveBandwidth(20),
                        varnames = ["income", "density"])
            @test plot_gwr_coefficients(g, "income") isa Makie.Figure

            # `level` must actually widen the interval that gets drawn.
            ci_span(f) = (seg = f.content[1].scene.plots[1][1][];
                          abs(seg[2][1] - seg[1][1]))
            @test ci_span(plot_coefficients(m; level = 0.99)) >
                  ci_span(plot_coefficients(m; level = 0.95))

            @test_throws ArgumentError plot_coefficients(m; level = 0.0)
            @test_throws ArgumentError plot_coefficients(m; level = 1.5)
            @test_throws ErrorException plot_gwr_coefficients(g, "not_a_variable")
            @test_throws DimensionMismatch plot_local_moran(lisa, coords[1:end-1, :])
        end
    end

    @testset "26. Turing Extension" begin
        if !_HAVE_TURING
            @info "Turing is not installed: skipping the Turing extension tests."
            @test_skip false
        else
            @eval using Turing, Distributions
            @test !isnothing(Base.get_extension(SpatialEconometrics,
                                                :SpatialEconometricsTuringExt))

            rng = MersenneTwister(2024); L = 10; N = L * L
            W = make_rook_weights(L)
            X = randn(rng, N, 2)
            β_true = [1.0, 1.5, -0.8]; ρ_true = 0.5; σ_true = 0.5
            Xa = hcat(ones(N), X)
            y = (I - ρ_true * W) \ (Xa * β_true .+ σ_true .* randn(rng, N))

            r = bayesian_sar(W, X, y; n_samples = 600, n_chains = 2,
                             varnames = ["x1", "x2"], rng = MersenneTwister(11))
            @test r isa AbstractBayesianSpatialResult
            @test r.coefnames == ["(Intercept)", "x1", "x2", "ρ", "σ"]
            @test length(r.posterior_mean) == 5 == length(r.posterior_std)
            @test all(isfinite, r.posterior_mean)
            @test all(>(0), r.posterior_std)
            @test r.N == N
            @test occursin("SAR", sprint(show, r))

            # The posterior has to concentrate on the truth. This is the test
            # that catches a mis-wired likelihood: a `~` whose left-hand side
            # is a local variable makes DynamicPPL sample it as one more
            # latent, and then every "posterior" comes back as its prior.
            @test isapprox(r.posterior_mean[2], β_true[2]; atol = 0.15)
            @test isapprox(r.posterior_mean[3], β_true[3]; atol = 0.15)
            @test isapprox(r.posterior_mean[4], ρ_true;    atol = 0.10)
            @test isapprox(r.posterior_mean[5], σ_true;    atol = 0.10)
            # The prior SD of β is sqrt(10) ≈ 3.16; the posterior is far tighter.
            @test all(r.posterior_std[1:3] .< 0.5)

            # ... and it has to agree with the MLE on the same data.
            mle = fit_spatial(SAR(), W, X, y)
            @test isapprox(r.posterior_mean[4], mle.rho; atol = 0.05)

            λ_true = 0.6
            y_sem = Xa * β_true .+ ((I - λ_true * W) \ (σ_true .* randn(rng, N)))
            r2 = bayesian_sem(W, X, y_sem; n_samples = 600, n_chains = 1,
                              varnames = ["x1", "x2"], rng = MersenneTwister(11))
            @test r2.coefnames == ["(Intercept)", "x1", "x2", "λ", "σ"]
            @test isapprox(r2.posterior_mean[4], λ_true; atol = 0.12)
            @test occursin("SEM", sprint(show, r2))

            θ_true = [0.6, -0.3]
            y_sdm = (I - ρ_true * W) \
                    (Xa * β_true .+ (W * X) * θ_true .+ σ_true .* randn(rng, N))
            r3 = bayesian_sdm(W, X, y_sdm; n_samples = 600, n_chains = 1,
                              varnames = ["x1", "x2"], rng = MersenneTwister(11))
            @test r3.coefnames ==
                  ["(Intercept)", "x1", "x2", "W_x1", "W_x2", "ρ", "σ"]
            @test isapprox(r3.posterior_mean[4], θ_true[1]; atol = 0.25)
            @test isapprox(r3.posterior_mean[5], θ_true[2]; atol = 0.25)

            # Default names when none are given.
            r4 = bayesian_sar(W, X, y; n_samples = 60, n_chains = 1,
                              rng = MersenneTwister(3))
            @test r4.coefnames == ["(Intercept)", "x1", "x2", "ρ", "σ"]

            # Same seed, same answer.
            r5 = bayesian_sar(W, X, y; n_samples = 60, n_chains = 1,
                              rng = MersenneTwister(3))
            @test r4.posterior_mean == r5.posterior_mean

            @test_throws DimensionMismatch bayesian_sar(W, X, y[1:end-1])
            @test_throws DimensionMismatch bayesian_sar(make_rook_weights(5), X, y)
        end
    end

end
