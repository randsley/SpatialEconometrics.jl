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
using Distributions

# A stand-in fit, so spatial_hausman_test can be handed chosen coefficient
# vectors and covariance matrices rather than whatever two panel fits happen
# to produce. Structs must be defined at top level, hence not in-testset.
struct HausMock <: StatsAPI.RegressionModel
    names::Vector{String}
    b::Vector{Float64}
    V::Matrix{Float64}
end
coefnames(m::HausMock) = m.names
coef(m::HausMock) = m.b
vcov(m::HausMock) = m.V

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

    @testset "4b. GS2SLS moments, dynamic Jacobian, panel information" begin
        GS = SpatialEconometrics.SpatialGS2SLSPanel
        L = 14; N = L * L
        W = make_rook_weights(L); Wd = Matrix(W)
        c = sum(abs2, W) / N                      # tr(W'W)/N
        @test isapprox(c, tr(transpose(Wd) * Wd) / N; rtol = 1e-12)

        # --- The Kelejian-Prucha moment conditions. Two of the three used to
        # omit their right-hand sides, so the objective drove eps'eps/N and
        # eps'W'W eps/N toward zero when their true values are sigma2 and
        # sigma2*tr(W'W)/N. Only the middle moment was valid.
        rng = MersenneTwister(400_041)
        lam_t = 0.5
        eps = randn(rng, N)
        u = (I - lam_t .* Wd) \ eps
        Wu = Wd * u; WWu = Wd * Wu
        e_true = u .- lam_t .* Wu
        @test isapprox(e_true, eps; rtol = 1e-10)
        s2 = dot(eps, eps) / N
        # the two moments the old objective wrongly set to zero
        @test isapprox(dot(eps, eps) / N, s2; rtol = 1e-12)
        @test s2 > 0.5
        @test dot(Wd * eps, Wd * eps) / N > 0.1
        # and the one that really is zero in expectation
        @test abs(dot(eps, Wd * eps) / N) < 0.15

        # The objective must be near-minimal at the truth now.
        lam_hat = GS.estimate_lambda_gmm(u, Wu, WWu, c)
        q_true = GS.gmm_objective(lam_t, u, Wu, WWu, c)
        @test GS.gmm_objective(lam_hat, u, Wu, WWu, c) <= q_true + 1e-12
        for bad in (lam_t - 0.3, lam_t + 0.3)
            @test GS.gmm_objective(bad, u, Wu, WWu, c) > q_true
        end

        # One draw is too noisy to pin -- the sampling standard deviation is
        # about 0.08 at this N -- so compare the two estimators over a set of
        # draws. The discarded objective is rebuilt here, so this stays a
        # regression test under any seed rather than pinning one number.
        function old_obj(uu, Wuu, WWuu, l)
            m11 = dot(uu, uu) / N; m12 = dot(uu, Wuu) / N
            m22 = dot(Wuu, Wuu) / N; m13 = dot(uu, WWuu) / N
            m23 = dot(Wuu, WWuu) / N; m33 = dot(WWuu, WWuu) / N
            g1 = m11 - 2l * m12 + l^2 * m22
            g2 = m12 - l * m22 - l * m13 + l^2 * m23
            g3 = m22 - 2l * m23 + l^2 * m33
            g1^2 + g2^2 + g3^2
        end
        new_err = Float64[]; old_err = Float64[]
        for _ in 1:12
            uu = (I - lam_t .* Wd) \ randn(rng, N)
            Wuu = Wd * uu; WWuu = Wd * Wuu
            push!(new_err,
                  abs(GS.estimate_lambda_gmm(uu, Wuu, WWuu, c) - lam_t))
            push!(old_err,
                  abs(argmin(l -> old_obj(uu, Wuu, WWuu, l),
                             -0.95:0.005:0.95) - lam_t))
        end
        # the old objective was biased about +0.27 at this lambda
        @test mean(old_err) > 0.15
        @test mean(new_err) < 0.08
        @test mean(new_err) < mean(old_err) / 2
        @test isapprox(mean(GS.estimate_lambda_gmm(u, Wu, WWu, c)), lam_hat)

        # --- The full fit: lambda's variance slot used to hold sigma2/NT,
        # which involves neither W nor lambda-hat, and the Sargan p-value was
        # the literal constant 1.0.
        Lp = 8; Np = Lp * Lp; T_tot = 6; NTp = Np * T_tot
        Wp = make_rook_weights(Lp)
        rng2 = MersenneTwister(400_042)
        Xe = randn(rng2, NTp, 2); z1 = randn(rng2, NTp); z2 = randn(rng2, NTp)
        en = 0.5 .* z1 .- 0.3 .* z2 .+ 0.2 .* randn(rng2, NTp)
        Sl = lu(sparse(1.0I, Np, Np) - 0.3 .* Wp)
        Bl = lu(sparse(1.0I, Np, Np) - 0.25 .* Wp)
        yp = zeros(NTp)
        for t in 1:T_tot
            idx = ((t - 1) * Np + 1):(t * Np)
            yp[idx] = Sl \ (Xe[idx, :] * [1.2, -0.6] .+ en[idx] .* 0.8 .+
                            (Bl \ (0.4 .* randn(rng2, Np))))
        end
        m_gs2 = fit_gs2sls_panel(IndivFE(), Wp, Xe, reshape(en, NTp, 1),
                                 hcat(z1, z2), yp, Np, T_tot)
        se_gs = stderror(m_gs2)
        @test all(isfinite, se_gs)
        @test se_gs[end] > 0
        @test !isapprox(se_gs[end]^2, m_gs2.sigma2 / (Np * (T_tot - 1));
                        rtol = 1e-6)
        @test 0.0 < m_gs2.sargan_pval < 1.0
        @test m_gs2.sargan_pval != 1.0
        # over-identified by one: three instruments columns beyond the
        # structural parameters is what gives the test any power
        @test m_gs2.sargan_stat > 0
        # a 120-replication Monte Carlo put every one of the five estimates
        # within 0.006 of the truth; here just pin that they are sane
        @test isapprox(m_gs2.rho, 0.3; atol = 0.1)
        @test isapprox(m_gs2.lambda, 0.25; atol = 0.15)

        # --- The dynamic panel counted log|I - rho*W| once instead of once
        # per period, which left rho-hat inconsistent: over 80 draws it sat at
        # 0.397 for T = 8 and 0.409 for T = 64 against a true 0.35, so the
        # error never shrank and coverage fell to 1.2% as the standard error
        # tightened on the wrong value. With T large the surviving bias is
        # small enough to pin.
        Ld = 7; Nd = Ld * Ld; Td = 16
        Wdp = make_rook_weights(Ld)
        rng3 = MersenneTwister(400_043)
        Xd = randn(rng3, Nd * Td, 1)
        mu_i = randn(rng3, Nd)
        Sd = lu(sparse(1.0I, Nd, Nd) - 0.35 .* Wdp)
        yd = zeros(Nd * Td)
        yd[1:Nd] = Sd \ (mu_i .+ Xd[1:Nd, :] * [0.8] .+ 0.4 .* randn(rng3, Nd))
        for t in 2:Td
            idx = ((t - 1) * Nd + 1):(t * Nd)
            ip = ((t - 2) * Nd + 1):((t - 1) * Nd)
            yd[idx] = Sd \ (mu_i .+ 0.3 .* yd[ip] .+ 0.15 .* (Wdp * yd[ip]) .+
                            Xd[idx, :] * [0.8] .+ 0.4 .* randn(rng3, Nd))
        end
        m_dyn2 = fit_dynamic_spatial_panel(SAR(), IndivFE(), Wdp, Xd, yd, Nd,
                                           Td; rng = MersenneTwister(11))
        @test isapprox(m_dyn2.rho, 0.35; atol = 0.04)
        @test isapprox(m_dyn2.gamma, 0.3; atol = 0.05)
        @test isapprox(m_dyn2.eta, 0.15; atol = 0.07)
        @test isapprox(coef(m_dyn2)[3], 0.8; atol = 0.05)

        # --- The spatial panel information matrix, against a numerical
        # Hessian of the within-transformed likelihood. Its trace inputs were
        # made exact earlier, but the assembled matrix had never been checked.
        Ls = 7; Ns = Ls * Ls; Ts = 5
        Ws = make_rook_weights(Ls); Wsd = Matrix(Ws)
        rng4 = MersenneTwister(400_044)
        Xs2 = randn(rng4, Ns * Ts, 2)
        ys2 = zeros(Ns * Ts)
        for t in 1:Ts
            idx = ((t - 1) * Ns + 1):(t * Ns)
            ys2[idx] = (I - 0.4 .* Wsd) \ (Xs2[idx, :] * [1.0, -0.5] .+
                                           randn(rng4, Ns))
        end
        m_sp = fit_spatial_panel(SAR(), IndivFE(), Ws, Xs2, ys2, Ns, Ts;
                                 method = ExactLU())
        dmv(z) = begin
            Z = reshape(copy(z), Ns, Ts); Z .-= mean(Z; dims = 2); vec(Z)
        end
        yt2 = dmv(ys2)
        Xt2 = reduce(hcat, [dmv(Xs2[:, k]) for k in 1:2])
        Te2 = Ts - 1
        f_sp = th -> begin
            bb = th[1:2]; rr = th[3]; ss = th[4]
            ss <= 0 && return -Inf
            A = I - rr .* Wsd; ee = zeros(Ns * Ts)
            for t in 1:Ts
                idx = ((t - 1) * Ns + 1):(t * Ns)
                ee[idx] = A * yt2[idx] .- Xt2[idx, :] * bb
            end
            -0.5 * Ns * Te2 * log(2pi * ss) - dot(ee, ee) / (2ss) +
                Te2 * first(logabsdet(A))
        end
        th_sp = vcat(coef(m_sp)[1:2], m_sp.rho, m_sp.sigma2)
        hh = 1e-5 .* max.(abs.(th_sp), 1.0); Hsp = zeros(4, 4)
        for i in 1:4, j in 1:4
            ei = zeros(4); ei[i] = hh[i]; ej = zeros(4); ej[j] = hh[j]
            Hsp[i, j] = (f_sp(th_sp + ei + ej) - f_sp(th_sp + ei - ej) -
                         f_sp(th_sp - ei + ej) + f_sp(th_sp - ei - ej)) /
                        (4hh[i] * hh[j])
        end
        se_sp_num = sqrt.(abs.(diag(inv(-Symmetric(Hsp)))[1:3]))
        @test isapprox(stderror(m_sp), se_sp_num; rtol = 0.06)

        # --- The regime model: rho's variance slot also held 1/N.
        Lr = 10; Nr = Lr * Lr
        Wr = make_rook_weights(Lr); Wrd = Matrix(Wr)
        rng5 = MersenneTwister(400_045)
        Xr = hcat(randn(rng5, Nr), randn(rng5, Nr))
        labels = [i <= Nr ÷ 2 ? 1 : 2 for i in 1:Nr]
        yr = (I - 0.4 .* Wrd) \ (randn(rng5, Nr) .+ Xr * [1.0, -0.5])
        m_rg = fit_spatial_regime(Wr, Xr, yr, ExogenousRegimes(labels);
                                  method = ExactLU())
        Pr = length(coef(m_rg)) - 1
        Xrg = m_rg.X_aug
        br = coef(m_rg)[1:Pr]; rr = m_rg.rho; s2r = m_rg.sigma2
        Gr = Wrd * inv(I - rr .* Wrd); Gxb = Gr * (Xrg * br)
        Ir = zeros(Pr + 2, Pr + 2)
        Ir[1:Pr, 1:Pr] = (transpose(Xrg) * Xrg) ./ s2r
        Ir[1:Pr, Pr + 1] = (transpose(Xrg) * Gxb) ./ s2r
        Ir[Pr + 1, 1:Pr] = Ir[1:Pr, Pr + 1]
        Ir[Pr + 1, Pr + 1] = dot(Gxb, Gxb) / s2r + tr(Gr * Gr) +
                             tr(transpose(Gr) * Gr)
        Ir[Pr + 1, Pr + 2] = tr(Gr) / s2r
        Ir[Pr + 2, Pr + 1] = Ir[Pr + 1, Pr + 2]
        Ir[Pr + 2, Pr + 2] = Nr / (2 * s2r^2)
        se_rg_ref = sqrt.(diag(inv(Symmetric(Ir)))[1:(Pr + 1)])
        @test isapprox(stderror(m_rg), se_rg_ref; rtol = 1e-8)
        @test !isapprox(stderror(m_rg)[end], 1 / sqrt(Nr); rtol = 1e-3)
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

    @testset "5b. Dynamic GMM: Windmeijer Assembly and Diagnostics" begin
        DG = SpatialEconometrics.DynamicPanelSystemGMM

        # y_it = ρ y_{i,t-1} + x_it β + μ_i + ε_it, burnt in so the initial
        # condition is from the stationary distribution.
        function gen_dp(rng; N = 100, Tt = 6, ρ = 0.5, b = 1.0, burn = 25)
            Y = zeros(N, Tt); Xt = zeros(N, Tt, 1)
            μ = randn(rng, N)
            for i in 1:N
                x = randn(rng, Tt + burn)
                yv = μ[i] / (1 - ρ)
                for sidx in 1:burn
                    yv = ρ * yv + b * x[sidx] + μ[i] + randn(rng)
                end
                for t in 1:Tt
                    yv = ρ * yv + b * x[burn + t] + μ[i] + randn(rng)
                    Y[i, t] = yv; Xt[i, t, 1] = x[burn + t]
                end
            end
            (Y, Xt)
        end

        # Rebuild the two-step pieces so the assembly can be checked.
        function gmm_pieces(variant, Y, Xt; max_lag = 4, collapsed = false)
            Nn = size(Xt, 1)
            Xl, yl, Zl, Hm, T_diff, Lq =
                DG.build_gmm_system(variant, Y, Xt, max_lag, collapsed)
            Pq = 1 + size(Xt, 3)
            S_ZX = zeros(Lq, Pq); S_Zy = zeros(Lq); O1 = zeros(Lq, Lq)
            for i in 1:Nn
                S_ZX .+= Zl[i]' * Xl[i]
                S_Zy .+= Zl[i]' * yl[i]
                O1 .+= Zl[i]' * Hm * Zl[i]
            end
            W1 = pinv(Symmetric(O1))
            G1i = inv(Symmetric(S_ZX' * W1 * S_ZX))
            t1 = G1i * (S_ZX' * (W1 * S_Zy))
            O2 = zeros(Lq, Lq)
            for i in 1:Nn
                z = Zl[i]' * (yl[i] .- Xl[i] * t1); O2 .+= z * z'
            end
            W2 = pinv(Symmetric(O2))
            V2 = inv(Symmetric(S_ZX' * W2 * S_ZX))
            t2 = V2 * (S_ZX' * (W2 * S_Zy))
            V1 = G1i * (S_ZX' * W1 * O2 * W1 * S_ZX) * G1i
            g2 = S_Zy .- S_ZX * t2
            uv = W2 * g2; Apre = -V2 * (S_ZX' * W2)
            psi = th -> begin
                wu = zeros(eltype(th), Lq)
                for i in 1:Nn
                    z = Zl[i]' * (yl[i] .- Xl[i] * th)
                    wu .+= z .* dot(z, uv)
                end
                Apre * wu
            end
            Dm = ForwardDiff.jacobian(psi, t1)
            (; V2, V1, D = Dm, t1, t2, L = Lq, P = Pq, g2, W2, T_diff, H = Hm)
        end

        rngd = MersenneTwister(2024)
        Yd, Xd = gen_dp(rngd)

        for variant in (DifferenceGMM(), SystemGMM())
            q = gmm_pieces(variant, Yd, Xd)
            m = fit_dynamic_gmm(variant, Yd, Xd)

            # point estimates and the uncorrected variance
            @test maximum(abs, coef(m) .- q.t2) < 1e-9
            @test maximum(abs, m.coef_onestep .- q.t1) < 1e-9
            @test maximum(abs, m.vcov_naive .- q.V2) < 1e-10
            @test m.n_instruments == q.L
            @test m.hansen_df == q.L - q.P
            @test isapprox(m.hansen_j, dot(q.g2, q.W2, q.g2); rtol = 1e-9)

            # Windmeijer (2005) eq. 4.4 / Roodman (2009) eq. 23:
            #   V_c = V_2 + D V_2 + (D V_2)' + D V_1 D'
            DV = q.D * q.V2
            V_correct = q.V2 .+ DV .+ transpose(DV) .+ (q.D * q.V1 * q.D')
            @test maximum(abs, vcov(m) .- V_correct) < 1e-10
            @test issymmetric(round.(vcov(m); digits = 12))

            # The discarded assembly put a V_1 inside the middle terms,
            # shrinking them by about two orders of magnitude and leaving the
            # corrected variance barely above the uncorrected one.
            V_old = q.V2 .+ (q.D * q.V1 * q.V2) .+ (q.V2 * q.V1 * q.D') .+
                    (q.D * q.V1 * q.D')
            @test norm(q.D * q.V2) > 50 * norm(q.D * q.V1 * q.V2)
            for j in 1:q.P
                @test sqrt(V_old[j, j]) < sqrt(V_correct[j, j])
                # and the correction must actually inflate the naive variance
                @test sqrt(V_correct[j, j]) > sqrt(q.V2[j, j])
            end
        end

        # ---- the instrument matrix -------------------------------------
        let Tt = 6, N2 = 8, ml = 4
            rg2 = MersenneTwister(5)
            Y2, X2 = gen_dp(rg2; N = N2, Tt = Tt)
            T_diff = Tt - 2

            Xl, yl, Zl, Hm, td, Lq =
                DG.build_gmm_system(DifferenceGMM(), Y2, X2, ml, false)
            @test td == T_diff
            # one column per (period, lag) pair, plus the exogenous regressor
            @test Lq == sum(min(t - 2, ml) for t in 3:Tt) + 1
            @test size(Zl[1]) == (T_diff, Lq)

            # the differenced equation and its regressors
            i = 3
            for eq in 1:T_diff
                t = eq + 2
                @test isapprox(yl[i][eq], Y2[i, t] - Y2[i, t-1]; rtol = 1e-12)
                @test isapprox(Xl[i][eq, 1], Y2[i, t-1] - Y2[i, t-2]; rtol = 1e-12)
                @test isapprox(Xl[i][eq, 2], X2[i, t, 1] - X2[i, t-1, 1];
                               rtol = 1e-12)
            end
            # equation t is instrumented by the levels y_{t-2} … y_1
            @test isapprox(Zl[i][1, 1], Y2[i, 1]; rtol = 1e-12)
            @test isapprox(Zl[i][2, 2], Y2[i, 2]; rtol = 1e-12)
            @test isapprox(Zl[i][2, 3], Y2[i, 1]; rtol = 1e-12)

            # H is the covariance of Δε up to scale: 2 on the diagonal, -1 off
            @test size(Hm) == (T_diff, T_diff)
            @test all(diag(Hm) .== 2)
            @test all(diag(Hm, 1) .== -1) && all(diag(Hm, -1) .== -1)
            @test Hm == Hm'

            # collapsing holds the count down: one column per lag, not per pair
            _, _, Zc, _, _, Lc =
                DG.build_gmm_system(DifferenceGMM(), Y2, X2, ml, true)
            @test Lc == min(ml, Tt - 2) + 1
            @test Lc < Lq
            # column s of the collapsed block holds lag s in every equation
            for eq in 1:T_diff
                t = eq + 2
                @test isapprox(Zc[i][eq, 1], Y2[i, t-2]; rtol = 1e-12)
            end

            # System GMM stacks level equations under the differenced ones,
            # instrumented by Δy_{t-1}, and gives them an identity H block.
            Xs, ys, Zs, Hs, _, Ls =
                DG.build_gmm_system(SystemGMM(), Y2, X2, ml, false)
            @test size(Zs[1], 1) == 2 * T_diff
            @test Ls == Lq + T_diff
            for eq in 1:T_diff
                row = T_diff + eq; t = eq + 2
                @test isapprox(ys[i][row], Y2[i, t]; rtol = 1e-12)
                @test isapprox(Xs[i][row, 1], Y2[i, t-1]; rtol = 1e-12)
                @test isapprox(Zs[i][row, Lq - 1 + eq],
                               Y2[i, t-1] - Y2[i, t-2]; rtol = 1e-12)
            end
            @test Hs[(T_diff+1):end, (T_diff+1):end] == I(T_diff)
            @test all(Hs[1:T_diff, (T_diff+1):end] .== 0)
        end

        # ---- the diagnostics are calibrated ----------------------------
        # Δε is MA(1) by construction, so AR(1) in the differenced residuals
        # must reject hard while AR(2) must behave like a standard normal.
        # The corrected variance must also cover better than the naive one,
        # and the gain is largest at small N, which is the point of it.
        function dp_diagnostics(variant, N, R, seed)
            rg = MersenneTwister(seed)
            a1 = Float64[]; a2 = Float64[]
            hit_c = 0; hit_n = 0; jrej = 0; njj = 0
            for _ in 1:R
                Yr, Xr = gen_dp(rg; N = N)
                mm = fit_dynamic_gmm(variant, Yr, Xr)
                isfinite(mm.ar1_z) && push!(a1, mm.ar1_z)
                isfinite(mm.ar2_z) && push!(a2, mm.ar2_z)
                abs(coef(mm)[1] - 0.5) <= 1.96 * stderror(mm)[1] && (hit_c += 1)
                abs(coef(mm)[1] - 0.5) <= 1.96 * sqrt(mm.vcov_naive[1, 1]) &&
                    (hit_n += 1)
                if mm.hansen_df > 0
                    njj += 1
                    ccdf(Chisq(mm.hansen_df), mm.hansen_j) < 0.05 && (jrej += 1)
                end
            end
            (mean(a1), mean(a2), std(a2), hit_c / R, hit_n / R, jrej / max(njj, 1))
        end

        m1, m2, s2, cov_c, cov_n, jr = dp_diagnostics(SystemGMM(), 50, 80, 4242)
        @test m1 < -2.5                 # AR(1) must reject: Δε is MA(1)
        @test abs(m2) < 0.45            # AR(2) centred
        @test 0.75 < s2 < 1.35          # AR(2) scaled like a standard normal
        @test jr < 0.15                 # Hansen J not wildly oversized
        # the Windmeijer correction earns its keep at N = 50
        @test cov_c > cov_n + 0.05
        @test cov_c > 0.85
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

        # MGWR degrees of freedom, accumulated per Yu et al. (2020). The fit
        # is additive in its components, so S = sum_k R_k and tr(S) must be
        # the sum of the per-covariate traces exactly, not just to tolerance.
        @test length(m_mgwr.enp) == 3
        @test all(isfinite, m_mgwr.enp)
        @test sum(m_mgwr.enp) == m_mgwr.hat_trace
        @test all(>=(1.0), m_mgwr.enp)      # each coefficient costs a parameter
        @test 3.0 <= m_mgwr.hat_trace <= N

        # sigma2 and AICc must now carry that penalty. Before the accumulator
        # existed, sigma2 was RSS/N and AICc was N*log(RSS/N) + N -- the
        # tr(S) = 0 case, which made MGWR beat GWR by exactly the penalty it
        # was missing. The last assertion is the one that catches a relapse.
        let RSS = sum(abs2, residuals(m_mgwr)), t = m_mgwr.hat_trace
            @test isapprox(m_mgwr.sigma2, RSS / (N - t); rtol = 1e-10)
            @test isapprox(m_mgwr.AICc,
                           N * log(RSS / N) + N * (N + t) / (N - 2 - t);
                           rtol = 1e-10)
            @test m_mgwr.AICc > N * log(RSS / N) + N + 1.0
        end

        # beta1 varies with the x-coordinate and beta2 is constant, so the
        # first should buy more effective parameters on a shorter bandwidth.
        # This is the substantive claim MGWR makes, and it reads straight off
        # the accumulator rather than off the coefficient spread.
        @test m_mgwr.enp[2] > m_mgwr.enp[3]
        @test m_mgwr.bandwidths[2] < m_mgwr.bandwidths[3]

        # The standard errors come from the coefficient operator B_k, so they
        # are finite and strictly positive everywhere once it is accumulated.
        @test all(isfinite, m_mgwr.se_local)
        @test all(>(0.0), m_mgwr.se_local)

        # Opting out must change only what is reported, never the fit: the
        # accumulator rides alongside the back-fitting loop and must not
        # perturb it. Everything it identifies is NaN rather than a number
        # that would invite comparison with a GWR AICc.
        m_noinf = fit_mgwr(X, y, coords; inference = false)
        @test m_noinf.coef_local == m_mgwr.coef_local
        @test isnan(m_noinf.hat_trace)
        @test isnan(m_noinf.AICc)
        @test all(isnan, m_noinf.enp)
        @test isapprox(m_noinf.sigma2, sum(abs2, residuals(m_noinf)) / N;
                       rtol = 1e-10)

        # Above max_inference_n the accumulator is skipped with a warning,
        # not an error: K+1 dense N x N operators is a memory ceiling, not a
        # reason to refuse to fit.
        m_gated = @test_logs (:warn, r"degrees-of-freedom accumulator") match_mode=:any fit_mgwr(X, y, coords; max_inference_n = 10)
        @test isnan(m_gated.hat_trace)
        @test m_gated.R2 > 0.5

        # tr(S) is an effective parameter count, so it lives in [K, N].
        # It was once formed with R^{-1} in place of R^{-T}, which pushed it
        # past N on a badly scaled X and then collapsed the N - tr(S) divisor
        # in sigma2 onto its floor of 1, leaving sigma2 equal to RSS.
        @test 3.0 <= m_gwr.hat_trace <= N
        @test isapprox(m_gwr.sigma2,
                       sum(abs2, residuals(m_gwr)) / (N - m_gwr.hat_trace);
                       rtol = 1e-10)

        # As the bandwidth grows every weight tends to one, so GWR collapses
        # onto OLS: tr(S) -> K and the local coefficients become the global
        # ones. This pins the trace to a value known in closed form.
        let Xc = hcat(ones(N), X), b_ols = Xc \ y
            m_wide = fit_gwr(X, y, coords; bandwidth = FixedBandwidth(1.0e4))
            @test isapprox(m_wide.hat_trace, 3.0; atol = 1e-4)
            @test isapprox(vec(mean(m_wide.coef_local, dims = 1)), b_ols; rtol = 1e-6)
            @test all(isapprox.(m_wide.coef_local[1, :], b_ols; rtol = 1e-6))
        end
    end

    @testset "11b. Bandwidth selection & back-fitting convergence" begin
        rng = MersenneTwister(2211); N = 45
        coords = hcat(rand(rng, N) .* 10, rand(rng, N) .* 10)
        X = randn(rng, N, 2)
        y = 2.0 .+ (1.0 .+ 0.3 .* coords[:, 1]) .* X[:, 1] .-
            0.5 .* X[:, 2] .+ 0.5 .* randn(rng, N)

        # The adaptive bandwidth asks the KD-tree for k+1 neighbours, which
        # throws at k = N -- and k_max = min(N, 500) put exactly that value
        # in the search range. It survived only because the old golden-section
        # search never happened to evaluate the endpoint.
        @test fit_gwr(X, y, coords; bandwidth = AdaptiveBandwidth(N)) isa GWRModel

        # Over a span this short the selector scans every candidate, so it
        # must return the exact grid minimum. What it replaced was a
        # golden-section search, valid only on a unimodal objective; the AICc
        # curve in k is not unimodal (eleven sign changes on Columbus), and
        # there that search returned k = 43 where the minimum is k = 23,
        # 2.3 AICc units worse.
        m_sel = fit_gwr(X, y, coords)
        @test 5 <= m_sel.bandwidth <= N - 1
        @test all(k -> fit_gwr(X, y, coords;
                               bandwidth = AdaptiveBandwidth(k)).AICc >=
                       m_sel.AICc - 1e-9, 5:(N - 1))

        # Back-fitting runs in two phases and `converged` requires both: the
        # bandwidths settling and the coefficients reaching tol. Cut it short
        # and there is no fixed operator to take a trace of, so the reported
        # degrees of freedom are NaN rather than whatever the half-built
        # accumulator happened to hold.
        m_short = @test_logs (:warn,) match_mode=:any fit_mgwr(X, y, coords; max_iter = 2)
        @test !m_short.converged
        @test isnan(m_short.hat_trace)

        # Columbus is the case where back-fitting genuinely fails to
        # converge: INC and HOVAL are spatially correlated with each other,
        # so the additive components compete and the contraction factor is
        # about 0.94 per sweep. It needs ~140 sweeps; max_iter used to stop
        # at 50 and return a fit that was not a fixed point of anything.
        # @test_logs with no expected entry is the real assertion here: it
        # fails if any of the three convergence warnings fire.
        csv = joinpath(@__DIR__, "..", "notebooks", "data", "columbus.csv")
        if isfile(csv)
            # Parsed by hand rather than with DelimitedFiles, which is no
            # longer an unconditional stdlib and is not a test dependency.
            lines = filter(!isempty, strip.(readlines(csv)))
            cnames = string.(split(lines[1], ','))
            rows = [parse.(Float64, split(l, ',')) for l in lines[2:end]]
            getcol(c) = [r[findfirst(==(c), cnames)] for r in rows]
            yc = getcol("CRIME")
            Xc = hcat(getcol("INC"), getcol("HOVAL"))
            cc = hcat(getcol("X"), getcol("Y"))

            mc = @test_logs fit_mgwr(Xc, yc, cc)
            @test mc.converged
            @test mc.iterations > 50
            @test all(isfinite, mc.enp)
            @test sum(mc.enp) == mc.hat_trace
            @test isapprox(mc.sigma2,
                           sum(abs2, residuals(mc)) /
                           (length(yc) - mc.hat_trace); rtol = 1e-10)
        end
    end

    @testset "11c. MGWR degrees of freedom against a solved fixed point" begin
        # The back-fitting fixed point is a linear system, so at small N it
        # can be solved outright and both degrees-of-freedom paths checked
        # against the answer instead of against each other. Every bug found
        # in this part of the package was a formula that no test compared to
        # a closed form, so this is the test that should have existed.
        #
        #   f_k = S^(k) (y - sum_{j != k} f_j)
        #   => R_k + S^(k) sum_{j != k} R_j = S^(k)
        #
        # which is one (NK x NK) solve with N right-hand sides.
        GW = SpatialEconometrics.GWREstimation

        rng = MersenneTwister(3311); N = 30
        coords = hcat(rand(rng, N) .* 10, rand(rng, N) .* 10)
        X = randn(rng, N, 2)
        y = 2.0 .+ (1.0 .+ 0.3 .* coords[:, 1]) .* X[:, 1] .-
            0.5 .* X[:, 2] .+ 0.5 .* randn(rng, N)
        X_aug = hcat(ones(N), X)
        K = 3
        kern = BisquareKernel()
        tree = GW._build_tree(coords)

        m_ex = fit_mgwr(X, y, coords; tol = 1e-8, max_iter = 4000)
        @test m_ex.converged
        @test m_ex.df_method === :exact

        Sk = map(1:K) do k
            bw = AdaptiveBandwidth(round(Int, m_ex.bandwidths[k]))
            xk = X_aug[:, k]
            S = zeros(N, N)
            for i in 1:N
                w = GW._get_weights(tree, coords, i, bw, kern)
                den = sum(w[j] * xk[j]^2 for j in 1:N)
                den <= 1e-12 && continue
                for j in 1:N
                    S[i, j] = xk[i] * w[j] * xk[j] / den
                end
            end
            S
        end

        A = zeros(N * K, N * K); Bm = zeros(N * K, N)
        for k in 1:K
            rows = ((k - 1) * N + 1):(k * N)
            A[rows, rows] = Matrix(I, N, N)
            for j in 1:K
                j == k && continue
                A[rows, ((j - 1) * N + 1):(j * N)] = Sk[k]
            end
            Bm[rows, :] = Sk[k]
        end
        Rst = A \ Bm
        truth = [tr(Rst[((k - 1) * N + 1):(k * N), :]) for k in 1:K]

        # The solved operator must reproduce the fit it is the operator of.
        f_hat = zeros(N)
        for k in 1:K
            f_hat .+= Rst[((k - 1) * N + 1):(k * N), :] * y
        end
        @test isapprox(f_hat, m_ex.fitted; atol = 1e-6)

        # The accumulator must match it to the tolerance it was run at.
        @test isapprox(m_ex.enp, truth; atol = 1e-4)
        @test isapprox(m_ex.hat_trace, sum(truth); atol = 1e-4)
        @test all(iszero, m_ex.enp_se)       # exact, so no Monte Carlo error

        # The probe estimator must agree with the same truth, within its own
        # reported error. 4 SEs rather than 2: against a solved fixed point
        # these z-scores have standard deviation about 1.04 at 200 probes
        # (1.28 at 50), because a quadratic form in Rademacher variables is
        # not Gaussian and its variance is estimated from the same probes.
        m_st = fit_mgwr(X, y, coords; inference = :stochastic,
                        rng = MersenneTwister(606), tol = 1e-8, max_iter = 4000)
        @test m_st.df_method === :stochastic
        @test all(isfinite, m_st.enp_se)
        @test all(>(0.0), m_st.enp_se)
        @test all(abs.(m_st.enp .- truth) .<= 4 .* m_st.enp_se)
        @test abs(m_st.hat_trace - sum(truth)) <= 4 * sqrt(sum(abs2, m_st.enp_se))

        # Whatever is asked of inference, the fit must come out the same: the
        # degrees of freedom are read off the frozen smoother afterwards and
        # cannot feed back into the back-fitting.
        m_no = fit_mgwr(X, y, coords; inference = false, tol = 1e-8, max_iter = 4000)
        @test m_no.df_method === :none
        @test m_no.coef_local == m_ex.coef_local
        @test m_st.coef_local == m_ex.coef_local
        @test all(isnan, m_no.enp_se)

        # Memory, not principle, is what gates the exact path, so the
        # stochastic one must still work where the exact one gives up.
        m_gate = @test_logs (:warn,) match_mode=:any fit_mgwr(X, y, coords; max_inference_n = 5)
        @test m_gate.df_method === :none
        m_sg = fit_mgwr(X, y, coords; inference = :stochastic, max_inference_n = 5,
                        rng = MersenneTwister(77))
        @test m_sg.df_method === :stochastic
        @test isfinite(m_sg.hat_trace)

        @test_throws ArgumentError fit_mgwr(X, y, coords; inference = :exact)
    end

    @testset "11d. Gravity: FE sweeping and NB dispersion" begin
        rng = MersenneTwister(5150)
        No, Nd = 12, 10; N = No * Nd
        o = repeat(1:No, inner = Nd)
        d = repeat(1:Nd, outer = No)
        ao = randn(rng, No) .* 0.5
        ad = randn(rng, Nd) .* 0.4
        # log_distance deliberately picks up the origin effect, so the
        # regressors are NOT orthogonal to the fixed effects. That is what
        # exposes a sweep applied to y but not to X: with independent
        # regressors the omission is almost invisible.
        logd = rand(rng, N) .* 2 .+ 1 .+ 1.5 .* [ao[o[n]] for n in 1:N]
        lang = Float64.(rand(rng, N) .< 0.3)
        Xg = hcat(logd, lang)
        eta = Xg * [-0.8, 0.5] .+ 2.0
        for n in 1:N; eta[n] += ao[o[n]] + ad[d[n]]; end
        mu_true = exp.(eta)
        yg = Float64[max(0.0, round(mu_true[n] + sqrt(mu_true[n]) *
                                    randn(rng))) for n in 1:N]

        # A two-way within PPML written out independently, as the reference.
        # The point of the rewrite is that the response and every regressor
        # pass through the SAME projection; the previous code swept y with a
        # single pass of independent origin and destination means (not the
        # two-way projection, and double-counting the overall level) while
        # sweeping X by alternating projection, and diverged to ~1e50.
        function dm2!(z, o, d, No, Nd, w)
            for _ in 1:500
                ch = 0.0
                for (ids, Ng) in ((o, No), (d, Nd))
                    num = zeros(Ng); den = zeros(Ng)
                    for n in eachindex(z)
                        num[ids[n]] += w[n] * z[n]; den[ids[n]] += w[n]
                    end
                    for g in 1:Ng; den[g] > 0 && (num[g] /= den[g]); end
                    for n in eachindex(z)
                        z[n] -= num[ids[n]]
                        ch = max(ch, abs(num[ids[n]]))
                    end
                end
                ch < 1e-13 && break
            end
            z
        end
        function ppml_ref(y, X, o, d, No, Nd)
            Nn, Kk = size(X)
            b = zeros(Kk); mu = fill(max(mean(y), 0.1), Nn)
            for _ in 1:300
                bo = copy(b)
                et = log.(max.(mu, 1e-12))
                z = et .+ (y .- mu) ./ max.(mu, 1e-12)
                w = max.(mu, 1e-12)
                zd = dm2!(copy(z), o, d, No, Nd, w)
                Xd = reduce(hcat, [dm2!(copy(X[:, k]), o, d, No, Nd, w)
                                   for k in 1:Kk])
                Ws = Diagonal(sqrt.(w))
                b = (Ws * Xd) \ (Ws * zd)
                r = z .- X * b
                fo = zeros(No); fd = zeros(Nd)
                for _ in 1:500
                    ch = 0.0
                    for (ids, Ng, f) in ((o, No, fo), (d, Nd, fd))
                        num = zeros(Ng); den = zeros(Ng)
                        for n in 1:Nn
                            e = r[n] - fo[o[n]] - fd[d[n]]
                            num[ids[n]] += w[n] * e; den[ids[n]] += w[n]
                        end
                        for g in 1:Ng
                            den[g] > 0 && (f[g] += num[g] / den[g];
                                           ch = max(ch, abs(num[g] / den[g])))
                        end
                    end
                    ch < 1e-13 && break
                end
                en = X * b
                for n in 1:Nn; en[n] += fo[o[n]] + fd[d[n]]; end
                mu = exp.(clamp.(en, -30, 30))
                maximum(abs, b .- bo) < 1e-10 && break
            end
            b
        end

        m_bi = fit_gravity(PoissonGravity(), yg, Xg, o, d; fe = BilateralFE(),
                           maxiter = 300, tol = 1e-10)
        @test m_bi.converged
        @test m_bi.iterations < 50          # it used to exhaust 200 and diverge
        @test all(isfinite, m_bi.coef)
        @test isapprox(m_bi.coef, ppml_ref(yg, Xg, o, d, No, Nd); rtol = 1e-6)

        # All three sweeps must converge, and the one-way ones must now
        # demean the regressors too.
        for f in (OriginFE(), DestinationFE())
            m = fit_gravity(PoissonGravity(), yg, Xg, o, d; fe = f,
                            maxiter = 300, tol = 1e-10)
            @test m.converged
            @test all(isfinite, m.coef)
        end

        # The recovered effects must reproduce the fit exactly: that is the
        # check that _solve_fe! inverts the sweep. Independent one-pass means
        # per side do not, because both are measured from the same residual
        # and their sum counts the level twice.
        mr = multilateral_resistance(m_bi)
        eta_hat = Xg * m_bi.coef .+
                  [mr.outward[o[n]] + mr.inward[d[n]] for n in 1:N]
        @test isapprox(exp.(eta_hat), m_bi.fitted; rtol = 1e-8)

        # A regressor with no within variation is unidentified, and saying so
        # beats returning whatever the least squares happens to pick.
        X_absorbed = hcat(Xg[:, 1], Float64[o[n] for n in 1:N])
        @test_throws ArgumentError fit_gravity(PoissonGravity(), yg, X_absorbed,
                                               o, d; fe = OriginFE())

        # The negative binomial has to differ from Poisson. It used to compute
        # the NB weights, discard them, refit the identical Poisson model each
        # pass and report Poisson's vcov -- so its coefficients were Poisson's
        # and its standard errors were too.
        m_p = fit_gravity(PoissonGravity(), yg, Xg, o, d; maxiter = 300, tol = 1e-10)
        m_nb = fit_gravity(NegBinGravity(), yg, Xg, o, d; maxiter = 300, tol = 1e-10)
        @test m_nb.converged
        @test m_nb.alpha > 0.0
        @test isfinite(m_nb.alpha)
        @test m_nb.vcov != m_p.vcov
        # Overdispersion is what Poisson understates, so NB errors are wider.
        @test all(diag(m_nb.vcov) .>= diag(m_p.vcov) .- 1e-12)
        # PPML stays consistent under overdispersion, so the coefficients
        # should agree even though the weights and errors do not.
        @test isapprox(m_nb.coef, m_p.coef; rtol = 0.25)

        # alpha = 0 through the shared core must be Poisson exactly.
        @test isapprox(m_p.coef,
                       fit_gravity(PoissonGravity(), yg, Xg, o, d;
                                   fe = BilateralFE(), maxiter = 300,
                                   tol = 1e-10).coef; rtol = 1e-12)

        # Deviance is now 2*sum[y log(y/mu) - (y - mu)], not 2*(ll - null_ll).
        dev_cf = 2 * sum(yg[n] > 0 ?
                         yg[n] * log(yg[n] / m_p.fitted[n]) -
                         (yg[n] - m_p.fitted[n]) : m_p.fitted[n]
                         for n in 1:N)
        @test isapprox(m_p.deviance, dev_cf; rtol = 1e-10)
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

    @testset "13b. Log-likelihoods against an independent Gaussian density" begin
        # Every one of these was previously asserted only to be finite. The
        # concentrated likelihood each estimator maximises is rewritten here
        # straight from the density, with the filtered residual formed
        # explicitly, so a wrong constant, a dropped Jacobian or the wrong
        # filter would all show up.
        rng = MersenneTwister(1_300_013)
        L = 9; N = L * L
        W = make_rook_weights(L); Wd = Matrix(W)
        X = hcat(randn(rng, N), randn(rng, N)); Xa = hcat(ones(N), X)
        beta = [1.0, 2.0, -1.0]
        y_lag = (I - 0.5 .* Wd) \ (Xa * beta .+ randn(rng, N))
        y_err = Xa * beta .+ ((I - 0.4 .* Wd) \ randn(rng, N))

        ld(A) = first(logabsdet(A))
        gll(e, s2, jac) = -0.5N * log(2pi) - 0.5N * log(s2) -
                          dot(e, e) / (2s2) + jac
        Xsdm = hcat(ones(N), X, Wd * X)

        m_sar = fit_spatial(SAR(), W, X, y_lag; method = ExactLU())
        e_sar = (I - m_sar.rho .* Wd) * y_lag .- Xa * coef(m_sar)[1:3]
        @test isapprox(loglikelihood(m_sar),
                       gll(e_sar, m_sar.sigma2, ld(I - m_sar.rho .* Wd));
                       rtol = 1e-12)
        @test isapprox(m_sar.sigma2, dot(e_sar, e_sar) / N; rtol = 1e-12)

        m_sdm = fit_spatial(SDM(), W, X, y_lag; method = ExactLU())
        e_sdm = (I - m_sdm.rho .* Wd) * y_lag .- Xsdm * coef(m_sdm)[1:5]
        @test isapprox(loglikelihood(m_sdm),
                       gll(e_sdm, m_sdm.sigma2, ld(I - m_sdm.rho .* Wd));
                       rtol = 1e-12)

        m_sem = fit_spatial_error(W, X, y_err; method = ExactLU())
        e_sem = (I - m_sem.lambda .* Wd) * (y_err .- Xa * coef(m_sem)[1:3])
        @test isapprox(loglikelihood(m_sem),
                       gll(e_sem, m_sem.sigma2, ld(I - m_sem.lambda .* Wd));
                       rtol = 1e-12)

        m_slx2 = fit_slx(W, X, y_lag)
        e_slx2 = y_lag .- Xsdm * coef(m_slx2)[1:5]
        @test isapprox(loglikelihood(m_slx2), gll(e_slx2, m_slx2.sigma2, 0.0);
                       rtol = 1e-12)

        # SAC carries two Jacobians, and its filter is (I-lambda*W2)(I-rho*W1).
        m_sac2 = fit_sac(W, W, X, y_lag; method = ExactLU())
        A = I - m_sac2.rho .* Wd; B = I - m_sac2.lambda .* Wd
        e_sac2 = B * (A * y_lag .- Xa * coef(m_sac2)[1:3])
        @test isapprox(loglikelihood(m_sac2),
                       gll(e_sac2, m_sac2.sigma2, ld(A) + ld(B)); rtol = 1e-12)

        m_sdem2 = fit_sdem(W, X, y_err; method = ExactLU())
        e_sdem2 = (I - m_sdem2.lambda .* Wd) *
                  (y_err .- Xsdm * coef(m_sdem2)[1:5])
        @test isapprox(loglikelihood(m_sdem2),
                       gll(e_sdem2, m_sdem2.sigma2,
                           ld(I - m_sdem2.lambda .* Wd)); rtol = 1e-12)

        # residuals(m) must BE the residual that enters the likelihood, and the
        # ML variance must be e-prime-e over n. _pointwise_loglik needs both.
        for (m, jac) in ((m_sar, ld(I - m_sar.rho .* Wd)),
                         (m_sdm, ld(I - m_sdm.rho .* Wd)),
                         (m_sem, ld(I - m_sem.lambda .* Wd)),
                         (m_slx2, 0.0),
                         (m_sac2, ld(A) + ld(B)),
                         (m_sdem2, ld(I - m_sdem2.lambda .* Wd)))
            e = residuals(m)
            @test isapprox(sum(abs2, e) / N, m.sigma2; rtol = 1e-10)
            @test isapprox(loglikelihood(m), gll(e, m.sigma2, jac); rtol = 1e-12)
        end
    end

    @testset "13c. Vuong: omega is a variance of pointwise ratios" begin
        # vuong_test built omega out of the AGGREGATE log-likelihoods, as
        # sqrt(abs(2(ll1-ll2)/n - adj^2)). That is twice a mean difference, not
        # a variance: it came out 0.71x the right size here, giving p = 0.324
        # against a correct 0.482. And because the quantity under the root goes
        # negative when model 1 fits worse, the abs() left the statistic
        # non-antisymmetric in its arguments. Only 0 <= p <= 1 was asserted.
        MS = SpatialEconometrics.SpatialModelSelection
        rng = MersenneTwister(1_300_014)
        L = 12; N = L * L
        W1 = make_rook_weights(L); W2 = make_queen_weights(L)
        W1d = Matrix(W1); W2d = Matrix(W2)
        X = hcat(randn(rng, N), randn(rng, N))
        y = 1.0 .+ X * [2.0, -1.0] .+ 0.6 .* (W1d * X[:, 1]) .+ randn(rng, N)

        # Non-nested: same response, different W. Both are plain OLS, so the
        # per-observation log-density is unambiguous and can be written out.
        m1 = fit_slx(W1, X, y); m2 = fit_slx(W2, X, y)

        function ll_i(m, Wd)
            Xa = hcat(ones(N), X, Wd * X)
            e = y .- Xa * coef(m)[1:size(Xa, 2)]
            s2 = m.sigma2
            -0.5 .* log(2pi * s2) .- (e .^ 2) ./ (2s2)
        end
        l1 = ll_i(m1, W1d); l2 = ll_i(m2, W2d)
        @test isapprox(sum(l1), loglikelihood(m1); rtol = 1e-12)
        @test isapprox(sum(l2), loglikelihood(m2); rtol = 1e-12)
        @test isapprox(MS._pointwise_loglik(m1), l1; rtol = 1e-10)

        # The contributions have to sum to the stored log-likelihood for every
        # model type, Jacobian and all. That is what lets one routine serve
        # them without knowing which spatial structure produced the fit.
        y_lag = (I - 0.5 .* W1d) \ (hcat(ones(N), X) * [1.0, 2.0, -1.0] .+
                                    randn(rng, N))
        for m in (fit_spatial(SAR(), W1, X, y_lag; method = ExactLU()),
                  fit_spatial(SDM(), W1, X, y_lag; method = ExactLU()),
                  fit_spatial_error(W1, X, y_lag; method = ExactLU()),
                  fit_sac(W1, W1, X, y_lag; method = ExactLU()),
                  fit_sdem(W1, X, y_lag; method = ExactLU()),
                  m1)
            @test length(MS._pointwise_loglik(m)) == N
            @test isapprox(sum(MS._pointwise_loglik(m)), loglikelihood(m);
                           rtol = 1e-12)
        end

        k1 = N - dof_residual(m1); k2 = N - dof_residual(m2)
        mi = l1 .- l2
        omega_ref = sqrt(sum(abs2, mi .- mean(mi)) / N)
        z_ref = ((loglikelihood(m1) - loglikelihood(m2)) - (k1 - k2)) /
                (sqrt(N) * omega_ref)

        vt = vuong_test(m1, m2)
        @test isapprox(vt.stat, z_ref; rtol = 1e-10)
        @test isapprox(vt.stat, 1.87685263; atol = 1e-6)
        @test isapprox(vt.p_value, 0.06053817; atol = 1e-7)

        # Guard the specific defect rather than only the value, so this stays
        # a regression test under any draw: rebuild the old omega and check it
        # is both wrong and wrong by a margin that moves the p-value.
        adj_old = ((loglikelihood(m1) - loglikelihood(m2)) - (k1 - k2)) / N
        omega_old = sqrt(abs(2.0 * (loglikelihood(m1) - loglikelihood(m2)) / N -
                             adj_old^2) + eps())
        @test !isapprox(omega_old, omega_ref; rtol = 0.05)
        stat_old = sqrt(N) * adj_old / omega_old
        @test !isapprox(stat_old, vt.stat; rtol = 0.05)

        # A Vuong statistic flips sign exactly when its arguments swap. The old
        # one gave +0.9871 against -0.9804.
        vt_rev = vuong_test(m2, m1)
        @test isapprox(vt.stat, -vt_rev.stat; rtol = 1e-12)
        @test isapprox(vt.p_value, vt_rev.p_value; rtol = 1e-12)

        # omega is a standard deviation, so it must not collapse towards zero
        # just because the two fits are close.
        @test omega_ref > 0.1
        @test 0.0 <= vt.p_value <= 1.0
    end

    @testset "13d. Trace-based variances are exact and reproducible" begin
        # tr(G), tr(G^2) and tr(G-transpose-G) were estimated by unseeded
        # Hutchinson probes off the global RNG: 250 for the power traces, 30
        # for the last one. So fitting the same model twice gave different
        # standard errors (0.5% apart on the rho SE at N = 81), the fit quietly
        # consumed global RNG state, and since spatial_effects reads the same
        # power traces, the reported impacts moved between runs as well. Of the
        # 21 assertions that touched a standard error, 20 checked only size.
        SU = SpatialEconometrics.SpatialUtils
        L = 8; N = L * L
        W = make_rook_weights(L); Wd = Matrix(W)
        rng = MersenneTwister(1_300_015)
        X = hcat(randn(rng, N), randn(rng, N)); Xa = hcat(ones(N), X)
        y = (I - 0.5 .* Wd) \ (Xa * [1.0, 2.0, -1.0] .+ randn(rng, N))

        # Power traces against dense powers of W.
        mu = SU.spatial_trace_powers(W; max_power = 12, max_exact_n = 10_000)
        @test mu[1] == 1.0
        Pm = Matrix{Float64}(I, N, N)
        for m in 1:12
            Pm = Pm * Wd
            @test isapprox(mu[m + 1], tr(Pm) / N; atol = 1e-12)
        end

        # The three G traces against a dense G = W(I - rho*W)^-1, in both the
        # plain and the constant-projected form the panel estimators need.
        for rho in (-0.6, 0.3, 0.8), pc in (false, true)
            G = Wd * inv(I - rho .* Wd)
            if pc
                Q = I - ones(N, N) ./ N
                G = Q * G * Q
            end
            g = SU.g_traces(W, rho; project_constant = pc, max_exact_n = 10_000)
            @test isapprox(g[1], tr(G); atol = 1e-10)
            @test isapprox(g[2], tr(G * G); atol = 1e-9)
            @test isapprox(g[3], sum(abs2, G); atol = 1e-9)
        end

        # Above the size gate the probe path runs instead. Seeded, it has to
        # repeat exactly, and it has to land near the exact value.
        g_ex = SU.g_traces(W, 0.5; max_exact_n = 10_000)
        g_p1 = SU.g_traces(W, 0.5; max_exact_n = 0, n_probes = 400,
                           rng = MersenneTwister(7))
        g_p2 = SU.g_traces(W, 0.5; max_exact_n = 0, n_probes = 400,
                           rng = MersenneTwister(7))
        @test g_p1 == g_p2
        @test isapprox(g_p1[3], g_ex[3]; rtol = 0.15)

        # The SAR variance against the expected information matrix, written out
        # densely with no probes anywhere. This is the first assertion in the
        # suite to check a standard error against a value rather than a size.
        m = fit_spatial(SAR(), W, X, y; method = ExactLU())
        b = coef(m)[1:3]; rho_h = m.rho; s2 = m.sigma2; P = 3
        G = Wd * inv(I - rho_h .* Wd); Gxb = G * (Xa * b)
        Ie = zeros(P + 2, P + 2)
        Ie[1:P, 1:P] = (transpose(Xa) * Xa) ./ s2
        Ie[1:P, P + 1] = (transpose(Xa) * Gxb) ./ s2
        Ie[P + 1, 1:P] = Ie[1:P, P + 1]
        Ie[P + 1, P + 1] = dot(Gxb, Gxb) / s2 + tr(transpose(G) * G) + tr(G * G)
        Ie[P + 1, P + 2] = tr(G) / s2
        Ie[P + 2, P + 1] = Ie[P + 1, P + 2]
        Ie[P + 2, P + 2] = N / (2 * s2^2)
        se_ref = sqrt.(diag(inv(Symmetric(Ie)))[1:(P + 1)])
        @test isapprox(stderror(m), se_ref; rtol = 1e-8)
        @test isapprox(m.rho_se, se_ref[P + 1]; rtol = 1e-8)

        # Same data, same call: bit-identical variance, traces and impacts.
        fits = [fit_spatial(SAR(), W, X, y; method = ExactLU()) for _ in 1:4]
        for f in fits
            @test vcov(f) == vcov(fits[1])
            @test f.rho_se == fits[1].rho_se
            @test f.trace_powers == fits[1].trace_powers
        end
        sdm = [fit_spatial(SDM(), W, X, y; method = ExactLU()) for _ in 1:3]
        effs = [spatial_effects(f) for f in sdm]
        for e in effs
            @test e.direct == effs[1].direct
            @test e.indirect == effs[1].indirect
            @test e.indirect_se == effs[1].indirect_se
        end

        # The panel estimators shared the defect, including the time-FE branch
        # that projects out the constant eigenvector.
        Np = 36; Tp = 6
        Wp = make_rook_weights(6)
        rng2 = MersenneTwister(1_300_016)
        Xp = randn(rng2, Np * Tp, 2)
        yp = Xp * [1.0, -0.5] .+ randn(rng2, Np * Tp)
        for fe in (IndivFE(), TwoWayFE())
            ps = [fit_spatial_panel(SAR(), fe, Wp, Xp, yp, Np, Tp) for _ in 1:3]
            for f in ps
                @test vcov(f) == vcov(ps[1])
                @test f.rho_se == ps[1].rho_se
            end
            @test all(isfinite, stderror(ps[1]))
        end
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
        # IndivFE against TwoWayFE is a near-cancellation, so the variance
        # difference often has no positive direction at all and the test is
        # not computable. df reports how many directions were used.
        @test 0 <= ht.df <= 3
        if ht.df == 0
            @test isnan(ht.stat) && isnan(ht.p_value)
        else
            @test ht.stat >= 0.0
            @test 0.0 <= ht.p_value <= 1.0
        end
    end

    @testset "14b. Panel Diagnostics: Scale, Degrees of Freedom, Degeneracy" begin
        rng = MersenneTwister(1414)
        L = 6; N = L * L; T_per = 5; NT = N * T_per
        W = make_rook_weights(L)
        X = hcat(ones(NT), randn(rng, NT, 2))
        β = [1.0, 2.0, -1.0]
        y = X * β .+ randn(rng, NT)

        A = kron(sparse(1.0I, T_per, T_per), W)
        # τ = tr(A²) + tr(A′A), the variance of the spatial error score
        τ = T_per * (sum(abs2, W) + sum(W .* transpose(W)))
        β_ols = X \ y; e = y .- X * β_ols; σ² = dot(e, e) / NT
        M = I - X * ((X' * X) \ X')
        d_ρ = dot(e, A * y) / σ²
        d_λ = dot(e, A * e) / σ²
        Aŷ = A * (X * β_ols)
        D = dot(Aŷ, M * Aŷ) / σ² + τ

        plm = panel_lm_tests(y, X, W, N, T_per)

        # Each component must be the squared score over its own variance.
        @test isapprox(plm.error_stat, d_λ^2 / τ; rtol = 1e-10)
        @test isapprox(plm.lag_stat, d_ρ^2 / D; rtol = 1e-10)

        # The discarded denominator used the lagged *observed* y and dropped
        # the trace term, which makes it smaller and the statistic larger.
        J_old = dot(A * y, M * (A * y)) / σ²
        @test J_old < D
        @test d_ρ^2 / J_old > plm.lag_stat

        # The joint test is the quadratic form in the inverse information, not
        # the sum of the marginals: d_ρ and d_λ are correlated, Cov = τ.
        LM_sp = (τ * d_ρ^2 - 2 * τ * d_ρ * d_λ + D * d_λ^2) / (τ * (D - τ))
        @test isapprox(plm.joint_stat, LM_sp + plm.re_stat; rtol = 1e-10)
        @test plm.joint_df == 3
        @test !isapprox(plm.joint_stat,
                        plm.lag_stat + plm.error_stat + plm.re_stat; rtol = 1e-6)

        # ---- scale under the null ----------------------------------------
        # Each component is χ²(1) and the joint χ²(3), so the means are 1 and
        # 3. The Monte-Carlo standard error of a χ²(1) mean at R = 300 is
        # 0.082, and of a χ²(3) mean 0.141.
        function lm_null_means(seed, R)
            rg = MersenneTwister(seed)
            acc = zeros(5)
            for _ in 1:R
                Xr = hcat(ones(NT), randn(rg, NT, 2))
                yr = Xr * β .+ randn(rg, NT)
                t = panel_lm_tests(yr, Xr, W, N, T_per)
                br = Xr \ yr; er = yr .- Xr * br; s2 = dot(er, er) / NT
                Mr = I - Xr * ((Xr' * Xr) \ Xr')
                Ay = A * yr
                old = (dot(er, Ay) / s2)^2 / (dot(Ay, Mr * Ay) / s2)
                acc .+= (t.lag_stat, t.error_stat, t.re_stat, t.joint_stat, old)
            end
            acc ./ R
        end
        m_lag, m_err, m_re, m_joint, m_old = lm_null_means(9090, 300)
        @test 0.78 < m_lag < 1.25
        @test 0.78 < m_err < 1.25
        @test 0.78 < m_re < 1.25
        @test 2.4 < m_joint < 3.6
        # the discarded lag statistic sits clearly above its corrected form
        @test m_old > m_lag + 0.08

        # ---- Chow degrees of freedom -------------------------------------
        # The sub-period fits estimate the K slopes *and* a spatial lag
        # coefficient, so the break imposes K + 1 restrictions. Counting only
        # K roughly doubles the size of the test.
        K = size(X, 2)
        ct = spatial_chow_test(y, X, W, N, T_per, 3)
        @test ct.df == K + 1

        function chow_rej(seed, R; ρ = 0.4, brk = 3, break_size = 0.0)
            rg = MersenneTwister(seed)
            S_lu = lu(sparse(1.0I, N, N) - ρ * W)
            n_rej = 0
            for _ in 1:R
                Xr = hcat(ones(NT), randn(rg, NT, 2))
                yr = zeros(NT)
                for t in 1:T_per
                    idx = ((t-1)*N+1):(t*N)
                    bt = t <= brk ? β : β .+ [0.0, break_size, 0.0]
                    yr[idx] = S_lu \ (Xr[idx, :] * bt .+ randn(rg, N))
                end
                spatial_chow_test(yr, Xr, W, N, T_per, brk).p_value < 0.05 &&
                    (n_rej += 1)
            end
            n_rej / R
        end
        # correctly sized with no break, and powerful against one
        @test chow_rej(515, 300) < 0.10
        @test chow_rej(616, 150; break_size = 0.6) > 0.60

        # ---- Hausman: no fabricated statistic ----------------------------
        nm = ["x1", "x2", "ρ"]
        b_fe = [1.00, 2.00, 0.40]
        b_re = [1.05, 1.96, 0.42]
        d = b_fe .- b_re
        V_re = [0.0100 0.0010 0.0005
                0.0010 0.0120 0.0008
                0.0005 0.0008 0.0090]

        # A genuine efficiency ordering: the difference is positive definite
        # and the statistic is the textbook form on all three directions.
        V_fe = V_re + 0.004 * Matrix(1.0I, 3, 3)
        h = spatial_hausman_test(HausMock(nm, b_fe, V_fe),
                                 HausMock(nm, b_re, V_re))
        @test h.df == 3
        @test isapprox(h.stat, dot(d, (V_fe - V_re) \ d); rtol = 1e-8)
        @test 0.0 <= h.p_value <= 1.0

        # Reversed arguments: nothing positive to invert, so the test says so
        # rather than returning a number.
        h_rev = spatial_hausman_test(HausMock(nm, b_re, V_re),
                                     HausMock(nm, b_fe, V_fe))
        @test h_rev.df == 0
        @test isnan(h_rev.stat)
        @test isnan(h_rev.p_value)

        # One clearly positive direction, the other two a near-cancellation of
        # indeterminate sign -- the common case for two panel fits that differ
        # only in their fixed effects. The old code shifted every eigenvalue up
        # to force definiteness, which left the smallest at the shift itself,
        # so the inverse carried an eigenvalue of 1e6 and the statistic was
        # whatever that noise direction happened to be.
        Q = Matrix(qr(randn(MersenneTwister(77), 3, 3)).Q)
        V_part = Q * Diagonal([0.004, -1.0e-5, -2.0e-5]) * Q'
        h_part = spatial_hausman_test(HausMock(nm, b_fe, V_re + V_part),
                                      HausMock(nm, b_re, V_re))
        @test h_part.df == 1
        ev = eigen(Symmetric(V_part))
        j = argmax(ev.values)
        z = dot(ev.vectors[:, j], d)
        @test isapprox(h_part.stat, z^2 / ev.values[j]; rtol = 1e-6)
        shift = -minimum(ev.values) + 1.0e-6
        stat_ridge = dot(d, Symmetric(V_part + shift * I) \ d)
        @test stat_ridge > 50 * h_part.stat

        # ---- panel_moran: the pooled row is internally consistent ----
        # pooled.I_stat is an average over T_per periods, so the variance
        # beside it must be the variance of that average. It carried the
        # single-period variance, too large by a factor of T_per.
        u = randn(MersenneTwister(2323), NT)
        pm = panel_moran(u, W, N, T_per)
        one_period = moran_global(u[1:N], W)
        @test length(pm.period_results) == T_per
        @test pm.T_periods == T_per
        @test isapprox(pm.pooled.I_stat,
                       mean(r.I_stat for r in pm.period_results); rtol = 1e-12)
        @test isapprox(pm.pooled.E_I, one_period.E_I; rtol = 1e-12)
        @test isapprox(pm.pooled.Var_I, one_period.Var_I / T_per; rtol = 1e-12)
        # the combined z is Stouffer's, not I_stat standardised by Var_I
        @test isapprox(pm.pooled.z_stat,
                       mean(r.z_stat for r in pm.period_results) * sqrt(T_per);
                       rtol = 1e-12)

        # The variance of the average must track the spread of the average,
        # not the spread of a single period. Over 400 independent draws the
        # two differ by a factor of about T_per.
        function moran_spread(seed, R)
            rg = MersenneTwister(seed)
            avg = zeros(R); first_p = zeros(R)
            for r in 1:R
                q = panel_moran(randn(rg, NT), W, N, T_per)
                avg[r] = q.pooled.I_stat
                first_p[r] = q.period_results[1].I_stat
            end
            (var(avg), var(first_p))
        end
        v_avg, v_one = moran_spread(2424, 400)
        @test 2.5 < v_one / v_avg < 8.0
        @test 0.5 < pm.pooled.Var_I / v_avg < 2.0
        @test pm.pooled.Var_I < one_period.Var_I
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

    @testset "15b. Variances of SLX, SEM, SDEM and SAC" begin
        # None of these had ever been checked against a value. SAC and SDEM
        # turned out not to compute a variance for their spatial parameters at
        # all: the diagonal slots held 1/N, so every reported standard error
        # for rho and lambda was exactly 1/sqrt(N) whatever the data said.
        rng = MersenneTwister(1_500_021)
        L = 12; N = L * L
        W = make_rook_weights(L); Wd = Matrix(W)
        X = hcat(randn(rng, N), randn(rng, N)); Xa = hcat(ones(N), X)
        u = (I - 0.4 .* Wd) \ randn(rng, N)
        y = (I - 0.5 .* Wd) \ (Xa * [1.0, 2.0, -1.0] .+ u)

        function numhess(f, th)
            n = length(th); h = 1e-5 .* max.(abs.(th), 1.0); H = zeros(n, n)
            for i in 1:n, j in 1:n
                ei = zeros(n); ei[i] = h[i]; ej = zeros(n); ej[j] = h[j]
                H[i, j] = (f(th + ei + ej) - f(th + ei - ej) -
                           f(th - ei + ej) + f(th - ei - ej)) / (4h[i] * h[j])
            end
            H
        end
        ld(A) = first(logabsdet(A))

        # --- SLX is plain OLS, so the variance is exactly knowable. It used
        # the ML sigma2 = RSS/N while dof_residual reported N - P, which
        # understated every standard error by sqrt((N-P)/N).
        m_slx3 = fit_slx(W, X, y)
        Xs = hcat(ones(N), X, Wd * X); Ps = size(Xs, 2)
        e_slx3 = y .- Xs * (Xs \ y)
        V_ols = (dot(e_slx3, e_slx3) / (N - Ps)) .* inv(Symmetric(Xs' * Xs))
        @test isapprox(vcov(m_slx3), V_ols; rtol = 1e-10)
        @test isapprox(stderror(m_slx3), sqrt.(diag(V_ols)); rtol = 1e-10)
        @test dof_residual(m_slx3) == N - Ps
        # the ML variance is a different matrix, and smaller
        V_ml = (dot(e_slx3, e_slx3) / N) .* inv(Symmetric(Xs' * Xs))
        @test !isapprox(vcov(m_slx3), V_ml; rtol = 1e-6)
        @test all(diag(vcov(m_slx3)) .> diag(V_ml))
        # sigma2 itself must stay the ML value, so the likelihood is unchanged
        @test isapprox(m_slx3.sigma2, dot(e_slx3, e_slx3) / N; rtol = 1e-10)

        # --- SEM against the expected information matrix, written out here.
        # In the error model beta is asymptotically independent of (lambda,
        # sigma2), so the matrix is block diagonal.
        y_err = Xa * [1.0, 2.0, -1.0] .+ ((I - 0.4 .* Wd) \ randn(rng, N))
        m_sem3 = fit_spatial_error(W, X, y_err; method = ExactLU())
        P = 3
        B = I - m_sem3.lambda .* Wd; GB = Wd * inv(B)
        Xf = B * Xa; s2 = m_sem3.sigma2
        Ie = zeros(P + 2, P + 2)
        Ie[1:P, 1:P] = (transpose(Xf) * Xf) ./ s2
        Ie[P + 1, P + 1] = tr(GB * GB) + tr(transpose(GB) * GB)
        Ie[P + 1, P + 2] = tr(GB) / s2; Ie[P + 2, P + 1] = Ie[P + 1, P + 2]
        Ie[P + 2, P + 2] = N / (2 * s2^2)
        se_sem_ref = sqrt.(diag(inv(Symmetric(Ie)))[1:(P + 1)])
        @test isapprox(stderror(m_sem3), se_sem_ref; rtol = 1e-8)
        @test isapprox(m_sem3.lambda_se, se_sem_ref[end]; rtol = 1e-8)
        # lambda-s standard error used 30 unseeded probes and moved 8% between
        # identical fits; beta-s, built from Xf-prime-Xf, never did.
        sems = [fit_spatial_error(W, X, y_err; method = ExactLU()) for _ in 1:4]
        for mm in sems
            @test vcov(mm) == vcov(sems[1])
        end

        # --- SDEM is an error model in augmented regressors, so the same
        # expected information applies with X replaced by [1 X WX].
        Xd = hcat(ones(N), X, Wd * X)
        y_sd = Xd * [1.0, 2.0, -1.0, 0.5, -0.3] .+
               ((I - 0.4 .* Wd) \ randn(rng, N))
        m_sdem3 = fit_sdem(W, X, y_sd; method = ExactLU())
        Pd = 5
        Bd2 = I - m_sdem3.lambda .* Wd; GBd = Wd * inv(Bd2)
        s2d = m_sdem3.sigma2
        Id = zeros(Pd + 2, Pd + 2)
        Id[1:Pd, 1:Pd] = (transpose(Bd2 * Xd) * (Bd2 * Xd)) ./ s2d
        Id[Pd + 1, Pd + 1] = tr(GBd * GBd) + tr(transpose(GBd) * GBd)
        Id[Pd + 1, Pd + 2] = tr(GBd) / s2d
        Id[Pd + 2, Pd + 1] = Id[Pd + 1, Pd + 2]
        Id[Pd + 2, Pd + 2] = N / (2 * s2d^2)
        se_sdem_ref = sqrt.(diag(inv(Symmetric(Id)))[1:(Pd + 1)])
        @test isapprox(stderror(m_sdem3)[end], se_sdem_ref[end]; rtol = 1e-8)
        @test !isapprox(stderror(m_sdem3)[end], 1 / sqrt(N); rtol = 1e-3)

        # --- SAC: the joint information for (rho, lambda). Checked against a
        # numerical Hessian of the full log-likelihood, which shares none of
        # the algebra. Expected and observed information differ in finite
        # samples, so the spatial parameters get a loose tolerance; separately
        # a 400-replication Monte Carlo at N = 196 put the reported standard
        # errors at 0.94 (rho), 0.96 (lambda) and 0.98 (beta1) of the sampling
        # standard deviation, with 93-94% coverage of the nominal 95%.
        m_sac3 = fit_sac(W, W, X, y; method = ExactLU())
        f_sac = th -> begin
            bb = th[1:P]; rr = th[P + 1]; ll2 = th[P + 2]; ss = th[P + 3]
            ss <= 0 && return -Inf
            A2 = I - rr .* Wd; B2 = I - ll2 .* Wd
            e2 = B2 * (A2 * y .- Xa * bb)
            -0.5N * log(2pi) - 0.5N * log(ss) - dot(e2, e2) / (2ss) +
                ld(A2) + ld(B2)
        end
        Hs = numhess(f_sac, vcat(coef(m_sac3)[1:P], m_sac3.rho,
                                 m_sac3.lambda, m_sac3.sigma2))
        se_sac_num = sqrt.(diag(inv(-Symmetric(Hs)))[1:(P + 2)])
        @test isapprox(stderror(m_sac3)[1:P], se_sac_num[1:P]; rtol = 0.06)
        @test isapprox(stderror(m_sac3)[P + 1], se_sac_num[P + 1]; rtol = 0.2)
        @test isapprox(stderror(m_sac3)[P + 2], se_sac_num[P + 2]; rtol = 0.2)

        # The old answer was 1/sqrt(N) for both, so pin that it is gone and
        # that the standard errors actually respond to the data.
        @test !isapprox(stderror(m_sac3)[P + 1], 1 / sqrt(N); rtol = 1e-3)
        @test !isapprox(stderror(m_sac3)[P + 2], 1 / sqrt(N); rtol = 1e-3)
        @test stderror(m_sac3)[P + 1] != stderror(m_sac3)[P + 2]
        @test size(vcov(m_sac3)) == (P + 2, P + 2)
        @test all(isfinite, vcov(m_sac3))
        @test issymmetric(round.(vcov(m_sac3); digits = 12))

        y2 = (I - 0.1 .* Wd) \ (Xa * [1.0, 2.0, -1.0] .+
                                ((I - 0.7 .* Wd) \ randn(rng, N)))
        m_sac4 = fit_sac(W, W, X, y2; method = ExactLU())
        @test stderror(m_sac4)[P + 1] != stderror(m_sac3)[P + 1]
        @test stderror(m_sac4)[P + 2] != stderror(m_sac3)[P + 2]

        # Above the size gate the probe path runs instead: seeded it must
        # repeat, and it must land near the exact answer.
        a1 = fit_sac(W, W, X, y; method = ExactLU(), max_exact_n = 0,
                     vcov_probes = 600, rng = MersenneTwister(9))
        a2 = fit_sac(W, W, X, y; method = ExactLU(), max_exact_n = 0,
                     vcov_probes = 600, rng = MersenneTwister(9))
        @test vcov(a1) == vcov(a2)
        @test isapprox(stderror(a1)[P + 1], stderror(m_sac3)[P + 1]; rtol = 0.2)
        @test isapprox(stderror(a1)[P + 2], stderror(m_sac3)[P + 2]; rtol = 0.2)
        # the probe path must not disturb the fit itself
        @test a1.rho == m_sac3.rho
        @test a1.lambda == m_sac3.lambda
        @test coef(a1)[1:P] == coef(m_sac3)[1:P]
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

    @testset "21b. Spatial Filtering: Basis Identities and ESF Variance" begin
        W = make_rook_weights(8); N = size(W, 1); S0 = sum(W)
        Wd = Matrix(W); Wsym = 0.5 .* (Wd .+ Wd')
        Mc = I - ones(N, N) / N
        Om = Mc * Wsym * Mc

        # ---- the eigenvector basis -----------------------------------
        se = spatial_eigenvectors(W)
        E = se.vectors
        @test size(E) == (N, N - 1)          # the constant is excluded
        @test issorted(se.values; rev = true)
        @test maximum(abs, E' * E - I) < 1e-12
        @test maximum(abs, vec(sum(E, dims = 1))) < 1e-12

        # Each column is an eigenvector of Ω = M W_sym M, and the full
        # spectrum of Ω is these eigenvalues plus one zero for the constant.
        @test maximum(norm(Om * E[:, i] - se.values[i] * E[:, i])
                      for i in 1:(N-1)) < 1e-10
        @test maximum(abs, sort(eigvals(Symmetric(Om))) .-
                           sort(vcat(0.0, se.values))) < 1e-10

        # MC_i = (N/S₀)λ_i, and must agree with the package's own statistic.
        @test maximum(abs, se.moran .- (N / S0) .* se.values) < 1e-14
        @test maximum(abs, [moran_coefficient(E[:, i], W) for i in 1:(N-1)] .-
                           se.moran) < 1e-10

        # tr(M W_sym M) = -S₀/N, because W has a zero diagonal and 1'W1 = S₀.
        # So the Moran coefficients over the complete basis sum to exactly -1
        # and average to -1/(N-1), which is E[I]. This is the identity
        # fit_re_esf's reference Moran coefficient is built on.
        @test isapprox(sum(se.values), -S0 / N; atol = 1e-10)
        @test isapprox(sum(se.moran), -1.0; atol = 1e-10)
        @test isapprox(mean(se.moran), -1 / (N - 1); atol = 1e-12)

        # min_moran_ratio screens the pool; n_candidates truncates it.
        se_scr = spatial_eigenvectors(W; min_moran_ratio = 0.25)
        @test all(se_scr.moran ./ maximum(se.moran) .>= 0.25 - 1e-12)
        @test size(spatial_eigenvectors(W; n_candidates = 7).vectors, 2) == 7

        # ---- the Gaussian ESF covariance -----------------------------
        rng = MersenneTwister(4242)
        X = randn(rng, N, 2); β = [1.0, 2.0, -1.0]
        y = (I - 0.6 * Wd) \ (β[1] .+ X * β[2:3] .+ 0.5 .* randn(rng, N))
        m = fit_esf(X, y, W)
        Xf = m.X_aug; P = length(coef(m))
        @test P == 1 + 2 + length(m.selected)
        @test dof_residual(m) == N - P

        # A Gaussian ESF fit is OLS on [1 X E_selected], so the covariance is
        # s²(X'X)⁻¹ with the *unbiased* s². Dividing by N understates every
        # standard error by exactly sqrt((N - P)/N).
        r = y .- Xf * (Xf \ y)
        @test maximum(abs, coef(m) .- (Xf \ y)) < 1e-10
        V_unb = (dot(r, r) / (N - P)) .* inv(Symmetric(Xf' * Xf))
        @test maximum(abs, stderror(m) ./ sqrt.(diag(V_unb)) .- 1) < 1e-10
        # the discarded version, rebuilt, is short by that factor
        V_ml = (dot(r, r) / N) .* inv(Symmetric(Xf' * Xf))
        @test isapprox(sqrt(diag(V_ml)[2] / diag(V_unb)[2]), sqrt((N - P) / N);
                       rtol = 1e-10)
        @test sqrt.(diag(V_ml))[2] < stderror(m)[2]
        # sigma2 stays the ML value, because the log-likelihood uses it
        @test isapprox(m.sigma2, dot(r, r) / N; rtol = 1e-10)

        # ---- RE-ESF against dense algebra ----------------------------
        mr = fit_re_esf(X, y, W)
        Er = mr.E; λr = mr.lambda; θ = mr.theta; σ2 = mr.sigma2
        Xa = mr.X_aug; Pr = size(Xa, 2); Lr = length(λr)
        V0 = I + θ .* (Er * Diagonal(λr) * Er')
        d = (θ .* λr) ./ (1 .+ θ .* λr)

        # Woodbury: (I + θEΛE')⁻¹ = I - E diag(θλ/(1+θλ)) E', since E'E = I.
        @test maximum(abs, (I - Er * Diagonal(d) * Er') - inv(V0)) < 1e-12
        @test isapprox(sum(log.(1 .+ θ .* λr)), logdet(V0); rtol = 1e-12)

        β_gls = (Xa' * (V0 \ Xa)) \ (Xa' * (V0 \ y))
        @test maximum(abs, coef(mr) .- β_gls) < 1e-10
        @test maximum(abs, vcov(mr) .- σ2 .* inv(Xa' * (V0 \ Xa))) < 1e-10

        r_gls = y .- Xa * β_gls
        rss = dot(r_gls, V0 \ r_gls)
        # unbiased, unlike the ESF branch above before its fix
        @test isapprox(σ2, rss / (N - Pr); rtol = 1e-10)
        @test !isapprox(σ2, rss / N; rtol = 1e-6)
        @test isapprox(mr.tau2, θ * σ2; rtol = 1e-12)

        # BLUP γ̂ = θΛE'V₀⁻¹r collapses to d .* (E'r).
        @test maximum(abs, mr.gamma .-
                      θ .* (λr .* (Er' * (V0 \ r_gls)))) < 1e-10
        @test maximum(abs, fitted(mr) .- (Xa * coef(mr) .+ Er * mr.gamma)) < 1e-10

        # The reference Moran coefficient averages the non-retained
        # directions, using Σλ over the complete basis = -S₀/N.
        @test isapprox(mr.moran_reference,
                       (N / S0) * ((-S0 / N) - sum(λr)) / (N - 1 - Lr);
                       rtol = 1e-10)
        @test isapprox(loglikelihood(mr),
                       -0.5 * (N * log(2π * σ2) + logdet(V0) + rss / σ2);
                       rtol = 1e-10)

        # REML: Brent must not be beaten by a grid over the same bracket.
        function reml_dense(lt)
            t = exp(lt); Vt = I + t .* (Er * Diagonal(λr) * Er')
            Vi = inv(Vt); XtX = Xa' * Vi * Xa; b = XtX \ (Xa' * Vi * y)
            rr = dot(y .- Xa * b, Vi * (y .- Xa * b))
            -0.5 * (logdet(Vt) + (N - Pr) * log(rr / (N - Pr)) + logdet(XtX))
        end
        @test reml_dense(log(θ)) >=
              maximum(reml_dense(g) for g in range(-12, 8; length = 801)) - 1e-8

        # ---- RE-ESF recovers its own variance components --------------
        function reesf_recovery(seed, R; tau2 = 0.5, sig = 0.5)
            eb = spatial_eigenvectors(W; min_moran_ratio = 0.25)
            ps = findall(>(1e-10), eb.values)
            Eb = eb.vectors[:, ps]; lb = eb.values[ps]
            rg = MersenneTwister(seed)
            t2 = zeros(R); s2 = zeros(R); cov_hit = 0
            for i in 1:R
                Xr = randn(rg, N, 2)
                g = sqrt(tau2) .* (sqrt.(lb) .* randn(rg, length(lb)))
                yr = β[1] .+ Xr * β[2:3] .+ Eb * g .+ sig .* randn(rg, N)
                mm = fit_re_esf(Xr, yr, W; eigen_basis = eb)
                t2[i] = mm.tau2; s2[i] = mm.sigma2
                abs(coef(mm)[2] - β[2]) <= 1.96 * stderror(mm)[2] &&
                    (cov_hit += 1)
            end
            (mean(t2), mean(s2), cov_hit / R)
        end
        mt2, ms2, cvg = reesf_recovery(99, 120)
        @test 0.38 < mt2 < 0.65        # true 0.5
        @test 0.21 < ms2 < 0.30        # true 0.25
        @test cvg > 0.88               # nominal 0.95
    end

    @testset "21c. ESF GLM Families against Independent Densities" begin
        SF = SpatialEconometrics.SpatialFiltering
        rng = MersenneTwister(606)
        n = 300
        Xg = hcat(ones(n), randn(rng, n), randn(rng, n))
        bg = [0.4, 0.6, -0.3]
        ηg = Xg * bg
        μ_log = exp.(ηg)
        μ_logit = 1 ./ (1 .+ exp.(-ηg))
        α_t = 0.5; r_nb = 1 / α_t

        y_p = Float64[rand(rng, Poisson(m)) for m in μ_log]
        y_b = Float64[rand(rng, Bernoulli(m)) for m in μ_logit]
        y_n = Float64[rand(rng, NegativeBinomial(r_nb, r_nb / (r_nb + m)))
                      for m in μ_log]

        # ---- the log-likelihoods are the textbook densities -----------
        @test isapprox(SF._glm_loglik(:poisson, y_p, μ_log, 0.0, 0.0),
                       sum(logpdf(Poisson(μ_log[i]), Int(y_p[i])) for i in 1:n);
                       rtol = 1e-10)
        @test isapprox(SF._glm_loglik(:logit, y_b, μ_logit, 0.0, 0.0),
                       sum(logpdf(Bernoulli(μ_logit[i]), Bool(y_b[i])) for i in 1:n);
                       rtol = 1e-10)
        @test isapprox(SF._glm_loglik(:negbin, y_n, μ_log, α_t, 0.0),
                       sum(logpdf(NegativeBinomial(r_nb, r_nb / (r_nb + μ_log[i])),
                                  Int(y_n[i])) for i in 1:n);
                       rtol = 1e-10)
        # the Gaussian branch is the concentrated form, -N/2(log2π + logσ² + 1)
        @test isapprox(SF._glm_loglik(:gaussian, y_p, μ_log, 0.0, 2.0),
                       -0.5 * n * (log(2π) + log(2.0) + 1); rtol = 1e-12)

        # ---- IRLS reaches the actual maximum --------------------------
        # Differentiate an independently written log-likelihood at the IRLS
        # answer: the score must vanish and a Newton step must not move.
        nll_p(v) = -sum(logpdf(Poisson(exp(dot(view(Xg, i, :), v))), Int(y_p[i]))
                        for i in 1:n)
        nll_b(v) = -sum(logpdf(Bernoulli(1 / (1 + exp(-dot(view(Xg, i, :), v)))),
                               Bool(y_b[i])) for i in 1:n)
        nll_n(v) = -sum(logpdf(NegativeBinomial(r_nb,
                            r_nb / (r_nb + exp(dot(view(Xg, i, :), v)))),
                            Int(y_n[i])) for i in 1:n)

        for (fam, yv, nll, al) in ((:poisson, y_p, nll_p, 0.0),
                                   (:logit, y_b, nll_b, 0.0),
                                   (:negbin, y_n, nll_n, α_t))
            bh, μh, wh, conv = SF._fit_glm(Xg, yv, fam; α = al)
            @test conv
            g = ForwardDiff.gradient(nll, bh)
            @test maximum(abs, g) < 1e-6
            @test norm(ForwardDiff.hessian(nll, bh) \ g) < 1e-8
            # fitted means are the inverse link of Xβ̂
            η̂ = Xg * bh
            exp_μ = fam === :logit ? 1 ./ (1 .+ exp.(-η̂)) : exp.(η̂)
            @test maximum(abs, μh .- exp_μ) < 1e-10
        end

        # The log and logit links are canonical for the Poisson and the
        # Bernoulli, so the IRLS weights reproduce the observed Hessian
        # exactly. The log link is NOT canonical for NB2, so there the
        # working weights give the expected information instead and the two
        # genuinely differ -- which is why vcov uses X′WX and not the Hessian.
        for (fam, yv, nll, al) in ((:poisson, y_p, nll_p, 0.0),
                                   (:logit, y_b, nll_b, 0.0))
            bh, _, wh, _ = SF._fit_glm(Xg, yv, fam; α = al)
            H = ForwardDiff.hessian(nll, bh)
            @test maximum(abs, (H .- Xg' * (wh .* Xg)) ./ abs.(H)) < 1e-10
        end
        let
            bh, _, wh, _ = SF._fit_glm(Xg, y_n, :negbin; α = α_t)
            H = ForwardDiff.hessian(nll_n, bh)
            @test maximum(abs, (H .- Xg' * (wh .* Xg)) ./ abs.(H)) > 0.1
        end

        # ---- Pearson residuals use the right variance function --------
        μv = [0.5, 2.0, 4.0]; yv3 = [1.0, 1.0, 5.0]
        @test SF._pearson_resid(:gaussian, yv3, μv, 0.0) ≈ yv3 .- μv
        @test SF._pearson_resid(:poisson, yv3, μv, 0.0) ≈ (yv3 .- μv) ./ sqrt.(μv)
        @test SF._pearson_resid(:negbin, yv3, μv, 0.5) ≈
              (yv3 .- μv) ./ sqrt.(μv .+ 0.5 .* μv .^ 2)
        let p3 = [0.2, 0.5, 0.9], y3 = [0.0, 1.0, 1.0]
            @test SF._pearson_resid(:logit, y3, p3, 0.0) ≈
                  (y3 .- p3) ./ sqrt.(p3 .* (1 .- p3))
        end

        # ---- the dispersion update ------------------------------------
        # For NB2, Var(y) = μ + αμ², so ((y-μ)² - μ)/μ² has mean α. The
        # average is over the residual degrees of freedom, not N: μ̂ is
        # fitted, so dividing by N biases α downwards.
        y_od = [3.0, 6.0, 1.0]          # overdispersed, so α̂ > 0
        raw_terms = ((y_od .- μv) .^ 2 .- μv) ./ μv .^ 2
        @test sum(raw_terms) > 0
        @test isapprox(SF._update_negbin_alpha(y_od, μv),
                       sum(raw_terms) / 3; rtol = 1e-12)
        @test isapprox(SF._update_negbin_alpha(y_od, μv; n_par = 2),
                       sum(raw_terms) / 1; rtol = 1e-12)
        # dividing by the residual df, not N, raises the estimate
        @test SF._update_negbin_alpha(y_od, μv; n_par = 2) >
              SF._update_negbin_alpha(y_od, μv)

        # Underdispersed data makes the moment estimate negative, which is
        # not a dispersion; the floor keeps it positive rather than
        # handing a negative variance to the working weights.
        @test sum(((yv3 .- μv) .^ 2 .- μv) ./ μv .^ 2) < 0
        @test SF._update_negbin_alpha(yv3, μv) > 0
        @test SF._update_negbin_alpha([1.0, 1.0], [1.0, 1.0]) > 0

        # α is consistent: the bias shrinks with N. 1500 observations with no
        # spatial structure should land within 10% of the truth.
        function alpha_at(nn, R, seed)
            rg = MersenneTwister(seed); acc = 0.0
            for _ in 1:R
                Xn = hcat(ones(nn), randn(rg, nn))
                mu = exp.(Xn * [0.4, 0.6])
                yn = Float64[rand(rg, NegativeBinomial(r_nb, r_nb / (r_nb + m)))
                             for m in mu]
                a = 0.0
                for _ in 1:30
                    _, mh, _, _ = SF._fit_glm(Xn, yn, :negbin; α = a)
                    an = SF._update_negbin_alpha(yn, mh; n_par = size(Xn, 2))
                    abs(an - a) < 1e-10 && (a = an; break)
                    a = an
                end
                acc += a
            end
            acc / R
        end
        @test isapprox(alpha_at(1500, 40, 2001), α_t; rtol = 0.10)

        # ---- coverage for the canonical families ----------------------
        # Poisson and logit report the inverse expected information, which is
        # the right variance for a canonical link; NB2 runs mildly low.
        function esf_coverage(family, R, seed)
            Wl = make_rook_weights(9); Nl = size(Wl, 1)
            eb = spatial_eigenvectors(Wl; min_moran_ratio = 0.25)
            ps = findall(>(1e-10), eb.values)
            Esp = eb.vectors[:, ps[1:min(6, length(ps))]]
            rg = MersenneTwister(seed); hit = 0
            for _ in 1:R
                Xr = randn(rg, Nl, 2)
                et = 0.4 .+ Xr * [0.6, -0.3] .+ 0.8 .* (Esp * randn(rg, size(Esp, 2)))
                yr = if family === :poisson
                    Float64[rand(rg, Poisson(exp(clamp(e, -20, 5)))) for e in et]
                else
                    Float64[rand(rg, Bernoulli(1 / (1 + exp(-e)))) for e in et]
                end
                mm = fit_esf(Xr, yr, Wl; family = family, eigen_basis = eb,
                             max_eigenvectors = 10)
                abs(coef(mm)[2] - 0.6) <= 1.96 * stderror(mm)[2] && (hit += 1)
            end
            hit / R
        end
        @test esf_coverage(:poisson, 120, 1234) > 0.87
        @test esf_coverage(:logit, 120, 1234) > 0.87
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

            # The NoFE fit is the less precise one -- an unmodelled
            # individual effect inflates σ̂² and so every standard error --
            # so it goes first, and V_fe - V_re is positive definite.
            ht = spatial_hausman_test(m_pooh, m_feh)
            # The pooled intercept has no FE counterpart, so it drops out.
            @test ht.coefnames == ["a", "b", "ρ"]
            @test ht.df == 3
            @test ht.stat >= 0.0
            @test 0.0 <= ht.p_value <= 1.0

            # Swapping the arguments compares the same set of names, but
            # the variance difference is then negative definite in every
            # direction, so there is nothing to invert and the test says
            # so instead of returning a number.
            ht_bad = spatial_hausman_test(m_feh, m_pooh)
            @test ht_bad.coefnames == ["a", "b", "ρ"]
            @test ht_bad.df == 0
            @test isnan(ht_bad.stat)
            @test all(eigvals(Symmetric(vcov(m_feh)[1:3, 1:3] .-
                                        vcov(m_pooh)[2:4, 2:4])) .< 0)

            # Comparing a model with itself is the degenerate case: identical
            # coefficients, so the statistic must be zero and p must be 1.
            ht_self = spatial_hausman_test(m_feh, m_feh)
            @test ht_self.stat ≈ 0.0 atol = 1e-6
            @test ht_self.p_value ≈ 1.0 atol = 1e-8

            # `names` restricts the comparison to a chosen subset.
            ht_sl = spatial_hausman_test(m_pooh, m_feh; names = ["a", "b"])
            @test ht_sl.coefnames == ["a", "b"]
            @test ht_sl.df == 2

            # An unknown name is an error, not a silently smaller comparison.
            @test_throws ErrorException spatial_hausman_test(m_pooh, m_feh;
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
