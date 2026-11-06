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
# Estimators for testset 17b. SimShortMock reports no spatial parameter, so
# monte_carlo_spatial has fewer coefficients than the DGP has parameters;
# SimFullMock reports one, standing in for a real fit where only the
# DGP-side diagnostics are under test. Structs must be top level.
struct SimShortMock <: StatsAPI.RegressionModel
    b::Vector{Float64}
end
coef(m::SimShortMock) = m.b
coefnames(m::SimShortMock) = ["(Intercept)", "x1", "x2"]

struct SimFullMock <: StatsAPI.RegressionModel
    b::Vector{Float64}
end
coef(m::SimFullMock) = m.b
coefnames(m::SimFullMock) = ["(Intercept)", "x1", "x2", "ρ"]

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

    @testset "4c. GS2SLS λ first-stage correction (finding 24)" begin
        GS = SpatialEconometrics.SpatialGS2SLSPanel
        SU = SpatialEconometrics.SpatialUtils
        L = 8; N = L * L; T_tot = 6; NT = N * T_tot
        W = make_rook_weights(L)
        λ_true = 0.25; ρ_true = 0.3

        # One panel, built the way testset 4 builds its own.
        function gen(rg)
            Xe = randn(rg, NT, 2); z1 = randn(rg, NT); z2 = randn(rg, NT)
            en = 0.5 .* z1 .- 0.3 .* z2 .+ 0.2 .* randn(rg, NT)
            Sl = lu(sparse(1.0I, N, N) - ρ_true .* W)
            Bl = lu(sparse(1.0I, N, N) - λ_true .* W)
            y = zeros(NT)
            for t in 1:T_tot
                idx = ((t - 1) * N + 1):(t * N)
                y[idx] = Sl \ (Xe[idx, :] * [1.2, -0.6] .+ en[idx] .* 0.8 .+
                               (Bl \ (0.4 .* randn(rg, N))))
            end
            return Xe, en, z1, z2, y
        end

        # Rebuild the first stage so the corrected and uncorrected standard
        # errors can be compared on identical inputs.
        function first_stage(Xe, en, z1, z2, y)
            T_post = T_tot - 1
            idx = (N + 1):NT
            y_p = y[idx]; Xe_p = Xe[idx, :]
            en_p = reshape(en[idx], :, 1); Zi_p = hcat(z1, z2)[idx, :]
            Wy_p = SU.panel_spatial_lag!(similar(y_p), W, y_p, N, T_post)
            fe = IndivFE()
            y_t = SU.within_transform!(copy(y_p), N, T_post, fe)
            Wy_t = SU.within_transform!(copy(Wy_p), N, T_post, fe)
            Xe_t = SU.within_transform(Xe_p, N, T_post, fe)
            En_t = SU.within_transform(en_p, N, T_post, fe)
            Zi_t = SU.within_transform(Zi_p, N, T_post, fe)
            WXe = SU.panel_spatial_lag(W, Xe_t, N, T_post)
            H = hcat(Xe_t, WXe, SU.panel_spatial_lag(W, WXe, N, T_post),
                     Zi_t, SU.panel_spatial_lag(W, Zi_t, N, T_post))
            Psi_endog = hcat(Wy_t, En_t)
            HHi = inv(Symmetric(H' * H))
            Psi_hat = H * (HHi * (H' * Psi_endog))
            Psi_full = hcat(Psi_hat[:, 1], Xe_t, Psi_hat[:, 2:end])
            Psi_raw = hcat(Wy_t, Xe_t, En_t)
            θ = (Psi_full' * Psi_raw) \ (Psi_full' * y_t)
            u = y_t .- Psi_raw * θ
            Wu = SU.panel_spatial_lag!(similar(u), W, u, N, T_post)
            WWu = SU.panel_spatial_lag!(similar(u), W, Wu, N, T_post)
            lam = GS.estimate_lambda_gmm(u, Wu, WWu, sum(abs2, W) / N)
            return lam, u, Wu, Psi_full, Psi_raw, T_post
        end

        lam, u, Wu, Psi_full, Psi_raw, T_post = first_stage(gen(MersenneTwister(4400))...)

        se_plain = GS._lambda_gmm_se(lam, u, Wu, W, N, T_post)
        se_corr = GS._lambda_gmm_se(lam, u, Wu, W, N, T_post;
                                    Psi_full=Psi_full, Psi_raw=Psi_raw)
        @test isfinite(se_plain) && se_plain > 0
        @test isfinite(se_corr) && se_corr > 0
        # The correction adds a positive semi-definite term, so it can only
        # widen the interval -- never narrow it.
        @test se_corr > se_plain
        # …and it is a correction, not a rescaling: the same order of magnitude.
        @test se_corr < 2.0 * se_plain

        # The added term itself must be symmetric and positive semi-definite.
        e_f = u .- lam .* Wu
        σ2 = dot(e_f, e_f) / (N * T_post)
        c = sum(abs2, W) / N
        FS = GS._first_stage_term(lam, e_f, W, Psi_full, Psi_raw, N, T_post,
                                  c, σ2)
        @test size(FS) == (2, 2)
        @test isapprox(FS, FS'; atol=1e-14 * max(1.0, maximum(abs, FS)))
        @test all(eigvals(Symmetric(FS)) .>= -1e-12)

        # Omitting the first-stage arguments must reproduce the uncorrected
        # value exactly -- that path is what checks the moment algebra against
        # known disturbances.
        @test GS._lambda_gmm_se(lam, u, Wu, W, N, T_post) == se_plain
        # A first stage with no columns adds nothing.
        @test GS._lambda_gmm_se(lam, u, Wu, W, N, T_post;
                                Psi_full=zeros(N * T_post, 0),
                                Psi_raw=zeros(N * T_post, 0)) == se_plain

        # The full fit must use the corrected value.
        Xe, en, z1, z2, y = gen(MersenneTwister(4400))
        m = fit_gs2sls_panel(IndivFE(), W, Xe, reshape(en, NT, 1),
                             hcat(z1, z2), y, N, T_tot)
        j = findfirst(n -> occursin("λ", n), coefnames(m))
        @test !isnothing(j)
        @test isapprox(stderror(m)[j], se_corr; rtol=1e-8)
        @test stderror(m)[j] > se_plain

        # ── Calibration: the corrected interval must cover near nominal ──
        # Before the correction the ratio of reported standard error to
        # sampling variability sat at 0.82-0.86 at every size, and coverage at
        # 88.8-92.0%. Pin the direction of travel with a short run.
        lam_hat = Float64[]; lam_se = Float64[]; hit = 0
        rg = MersenneTwister(4401)
        for r in 1:40
            Xe_r, en_r, z1_r, z2_r, y_r = gen(rg)
            try
                mr = fit_gs2sls_panel(IndivFE(), W, Xe_r, reshape(en_r, NT, 1),
                                      hcat(z1_r, z2_r), y_r, N, T_tot)
                jr = findfirst(n -> occursin("λ", n), coefnames(mr))
                isnothing(jr) && continue
                isfinite(stderror(mr)[jr]) || continue
                push!(lam_hat, coef(mr)[jr]); push!(lam_se, stderror(mr)[jr])
                abs(coef(mr)[jr] - λ_true) <= 1.96 * stderror(mr)[jr] &&
                    (hit += 1)
            catch
                continue
            end
        end
        @test length(lam_hat) >= 30
        # λ̂ is consistent at this size.
        @test abs(mean(lam_hat) - λ_true) < 0.06
        # The reported standard error now tracks the sampling variability.
        @test mean(lam_se) / std(lam_hat) > 0.85
        # It stays conservative rather than optimistic.
        @test mean(lam_se) / std(lam_hat) < 1.25
        @test hit / length(lam_hat) > 0.80
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

        # ── the scale of the sandwich, pinned exactly ──
        # A sandwich covariance is equivariant in the scale of y: multiply y
        # by c and every beta standard error must scale by exactly c, while
        # rho's is unchanged because rho is dimensionless. This held only
        # by accident of sigma2 ~ 1. `extract_spatial_jacobian_scores`
        # returned `sigma2 .* vcov_qml` as the bread, but `vcov_qml` is
        # already sigma2 * (Z'Z)^-1 and the meat is built from Z'e with no
        # 1/sigma2, so the bread carried sigma2 twice over: the variance
        # came out scaled by sigma2^4 and the standard error by sigma2^2.
        # Against a 200-replication sampling distribution the robust errors
        # were four times too small; with sigma2 > 1 they are too large
        # instead. Every assertion above passes under that bug -- they test
        # shape, symmetry and positive-semidefiniteness, all of which a
        # misscaled sandwich keeps -- so the scale needs pinning on its own.
        # On the tolerance, which is measured rather than guessed. The
        # equivariance is exact in exact arithmetic, but rho comes from a
        # numerical search that lands differently per Julia version, so the
        # two fits are not bit-identical: rho differs by 2e-11 between them on
        # 1.13.3 and by 4e-8 on 1.10.13, and the Driscoll-Kraay standard error
        # amplifies the latter to 6.3e-7 relative. An earlier rtol of 1e-8 was
        # chosen because it passed on 1.13, where the worst deviation is
        # 2.6e-11; it failed on 1.10 in CI. That is the argument for measuring
        # a margin instead of taking the tightest number that happens to pass.
        # 1e-4 sits 160x above the worst observed deviation and roughly 1e6
        # below the relative deviation of 80 to 240 that the bug produced, so
        # it discriminates with room at both ends.
        c_scale = 3.0
        m_scaled = fit_spatial_panel(SAR(), TwoWayFE(), W, X, c_scale .* y, N, T_per)
        for spec in (DriscollKraayVCOV(), ConleySpatialHAC(coords, 3.0))
            se_1 = sqrt.(diag(robust_vcov(m_pan, spec)))
            se_c = sqrt.(diag(robust_vcov(m_scaled, spec)))
            @test se_c[1:2] ≈ c_scale .* se_1[1:2] rtol = 1e-4
            @test se_c[end] ≈ se_1[end] rtol = 1e-4
        end

        # ── and pinned against the sandwich built by hand ──
        # Rebuilt from the formula and the model's own public fields:
        # Z = [X_tilde, W(I-rho*W)^-1 X_tilde*beta], meat = sum_t (Z_t'e_t)(Z_t'e_t)',
        # and bread = vcov / sigma2. The bread is the QML covariance de-scaled,
        # not (Z'Z)^-1: that is deliberate, and a first version of this test
        # asserted (Z'Z)^-1 and failed. The two agree on the beta block, but
        # the QML information matrix's rho block carries the log-determinant
        # trace terms, which (Z'Z)^-1 knows nothing about, and because that
        # matrix is inverted jointly its beta block is not (Z'Z)^-1 either --
        # measured 1.7% away, the Schur complement of the beta-rho cross
        # terms. Using the model's own covariance as bread is what makes the
        # rho row of the robust table mean anything. max_lag = 0 and no
        # small-sample correction leave the meat as H*H' with nothing folded
        # in, so this is exact.
        beta_hat = coef(m_pan)[1:2]
        Slu_h = lu(sparse(1.0I, N, N) - m_pan.rho * W)
        Xb_h = m_pan.X_tilde * beta_hat
        GXb_h = vec(W * (Slu_h \ reshape(Xb_h, N, T_per)))
        Z_h = hcat(m_pan.X_tilde, GXb_h)
        H_h = zeros(size(Z_h, 2), T_per)
        for t in 1:T_per
            idx = ((t - 1) * N + 1):(t * N)
            H_h[:, t] = Z_h[idx, :]' * m_pan.residuals[idx]
        end
        bread_h = vcov(m_pan) ./ m_pan.sigma2
        V_hand = bread_h * (H_h * H_h') * bread_h'
        V_got = robust_vcov(m_pan, DriscollKraayVCOV(max_lag = 0,
                                                     small_sample_correct = false))
        @test V_got ≈ V_hand rtol = 1e-10
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
        # No literal for vt.stat here. randn(::MersenneTwister, n) is only
        # reproducible within a Julia minor version, so the same seed gives
        # different data on 1.10 and on 1.13 -- X sums to -32.82 against
        # -19.63 -- and this statistic came out 0.114 against 1.877. Both
        # agreed with z_ref to 1e-10, which is the assertion that carries the
        # finding: the statistic is whatever the data say, and what is being
        # pinned is that it equals an independently recomputed reference.
        @test 0 <= vt.p_value <= 1

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


    @testset "17b. Simulation DGPs and Monte Carlo bookkeeping (finding 22)" begin
        L = 5; W = make_rook_weights(L); N = size(W, 1); Wd = Matrix(W)
        X = randn(MersenneTwister(77), N, 2)
        b = [1.0, -0.5]; th = [0.4, 0.2]; s = 0.8; ρ = 0.5; λ = 0.3
        Ai = inv(I - ρ * Wd); Bi = inv(I - λ * Wd)

        # ── Each DGP against its closed-form mean and covariance ──
        # With X held fixed, y is multivariate normal with a known mean and
        # covariance for every one of the six processes. A sign error or a
        # swapped A/B filter shows up immediately here.
        cases = [
            ("SAR",  SAR_DGP(ρ, b, s),      Ai * (X * b),            s^2 * Ai * Ai'),
            ("SEM",  SEM_DGP(λ, b, s),      X * b,                   s^2 * Bi * Bi'),
            ("SDM",  SDM_DGP(ρ, b, th, s),  Ai * (X * b + Wd * X * th), s^2 * Ai * Ai'),
            ("SLX",  SLX_DGP(b, th, s),     X * b + Wd * X * th,     s^2 * Matrix(I, N, N)),
            ("SAC",  SAC_DGP(ρ, λ, b, s),   Ai * (X * b),            s^2 * Ai * Bi * Bi' * Ai'),
            ("SDEM", SDEM_DGP(λ, b, th, s), X * b + Wd * X * th,     s^2 * Bi * Bi'),
        ]
        R = 8_000
        for (nm, dgp, m_want, V_want) in cases
            rg = MersenneTwister(4242)
            acc = zeros(N); M2 = zeros(N, N)
            # X is passed through untouched, so the caller can reuse it.
            @test simulate_spatial(dgp, W, X; rng=rg)[2] === X
            for _ in 1:R
                y, _ = simulate_spatial(dgp, W, X; rng=rg)
                acc .+= y; M2 .+= y * y'
            end
            m_hat = acc ./ R
            V_hat = M2 ./ R .- m_hat * m_hat'
            # Mean within a few Monte Carlo standard errors.
            mc_se = sqrt.(diag(V_want) ./ R)
            @test all(abs.(m_hat .- m_want) .<= 4.5 .* mc_se)
            # Covariance within 10% of the largest entry.
            @test maximum(abs.(V_hat .- V_want)) < 0.10 * maximum(abs.(V_want))
        end

        # σ scales the standard deviation, not the variance: doubling σ must
        # double the spread of y about its mean.
        rg_a = MersenneTwister(99); rg_b = MersenneTwister(99)
        ya = [simulate_spatial(SAR_DGP(ρ, b, 1.0), W, X; rng=rg_a)[1] for _ in 1:3000]
        yb = [simulate_spatial(SAR_DGP(ρ, b, 2.0), W, X; rng=rg_b)[1] for _ in 1:3000]
        sa = std(reduce(hcat, ya), dims=2); sb = std(reduce(hcat, yb), dims=2)
        @test isapprox(sb ./ sa, fill(2.0, N); rtol=0.08)

        # ── Monte Carlo bookkeeping ──
        Wm = make_rook_weights(8); Nm = size(Wm, 1)
        CSx = SpatialEconometrics.CrossSection
        dgp_m = SAR_DGP(0.4, [1.0, -0.5], 1.0)
        mc = monte_carlo_spatial(dgp_m, Wm, Nm, 2,
            (w, x, y) -> fit_spatial(CSx.SAR(), w, x, y);
            config=MonteCarloConfig(n_replications=60, seed=5))
        @test mc.n_successful == 60
        @test all(isfinite, mc.param_means)
        # bias is mean minus truth, exactly.
        @test isapprox(mc.param_bias, mc.param_means .- [1.0, -0.5, 0.4]; rtol=1e-12)
        # RMSE combines bias and spread; it can never be below |bias|.
        @test all(mc.param_rmse .>= abs.(mc.param_bias) .- 1e-12)
        @test isapprox(mc.param_rmse,
                       sqrt.(mc.param_bias .^ 2 .+ mc.param_stds .^ 2); rtol=1e-12)
        # A fixed seed must give a reproducible experiment.
        mc2 = monte_carlo_spatial(dgp_m, Wm, Nm, 2,
            (w, x, y) -> fit_spatial(CSx.SAR(), w, x, y);
            config=MonteCarloConfig(n_replications=60, seed=5))
        @test mc.param_means == mc2.param_means

        # ── Finding 22: a short coefficient vector must not read as zero ──
        # SimShortMock reports an intercept and two slopes and no spatial
        # parameter, so the DGP's ρ row has nothing behind it.
        mc_s = @test_logs (:warn, r"trailing 1 row") match_mode=:any monte_carlo_spatial(
            dgp_m, Wm, Nm, 2,
            (w, x, y) -> SimShortMock(hcat(ones(Nm), x) \ y);
            config=MonteCarloConfig(n_replications=30, seed=3))
        @test mc_s.n_successful == 30            # the fits did succeed
        @test all(isfinite, mc_s.param_means[1:2])   # β rows are real estimates
        @test isnan(mc_s.param_means[3])         # ρ was never estimated
        @test isnan(mc_s.param_stds[3])
        @test isnan(mc_s.param_bias[3])
        @test isnan(mc_s.param_rmse[3])
        # The β rows must be unaffected by the NaN beside them.
        @test isapprox(mc_s.param_means[1], 1.0; atol=0.2)
        @test isapprox(mc_s.param_means[2], -0.5; atol=0.2)

        # ── Every replication failing is an error, not an empty table ──
        @test_throws ErrorException monte_carlo_spatial(
            dgp_m, Wm, Nm, 2, (w, x, y) -> error("no fit");
            config=MonteCarloConfig(n_replications=5, seed=1))

        # ── Diagnostic rejection rates are sized under the null ──
        # The three OLS diagnostics are computed from the simulated data and do
        # not depend on the estimator, so a cheap mock keeps this fast.
        mc_null = monte_carlo_spatial(SAR_DGP(0.0, [1.0, -0.5], 1.0), Wm, Nm, 2,
            (w, x, y) -> SimFullMock(vcat(hcat(ones(Nm), x) \ y, 0.0));
            config=MonteCarloConfig(n_replications=300, seed=21))
        for k in ("moran_05", "lm_lag_05", "lm_error_05")
            @test 0.0 <= mc_null.rejection_rates[k] <= 0.12   # nominal 0.05
        end
        # …and have power against a genuine lag process.
        mc_alt = monte_carlo_spatial(SAR_DGP(0.5, [1.0, -0.5], 1.0), Wm, Nm, 2,
            (w, x, y) -> SimFullMock(vcat(hcat(ones(Nm), x) \ y, 0.0));
            config=MonteCarloConfig(n_replications=300, seed=22))
        @test mc_alt.rejection_rates["lm_lag_05"] > 0.5
        @test mc_alt.rejection_rates["lm_lag_05"] > mc_null.rejection_rates["lm_lag_05"]
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


    @testset "18b. Table exporters escape and honour digits (finding 23)" begin
        ST = SpatialEconometrics.SpatialTables
        W = make_rook_weights(8); N = size(W, 1)
        rg = MersenneTwister(31); X = randn(rg, N, 2)
        lab = [i <= N ÷ 2 ? 1 : 2 for i in 1:N]
        y = (I - 0.3 * Matrix(W)) \ (1.0 .+ X * [1.0, -0.5] .+ 0.6 .* randn(rg, N))
        # Regime names carry underscores, and ρ is Greek — between them they
        # exercise every escape path the exporters need.
        m = fit_spatial_regime(W, X, y, ExogenousRegimes(lab); varnames=["x1", "x2"])
        ct = coeftable(m)
        @test any(nm -> occursin("_", nm), ct.names)
        @test "ρ" in ct.names

        # Count the pipes a Markdown renderer treats as cell separators, i.e.
        # those not escaped by a preceding backslash.
        function sep_pipes(line::AbstractString)
            n = 0
            prev = ' '
            for c in line
                c == '|' && prev != '\\' && (n += 1)
                prev = c
            end
            return n
        end

        # ── to_markdown: every row must declare the same cells ──
        md = to_markdown(ct)
        rows = filter(!isempty, split(strip(md), '\n'))
        @test length(rows) == length(ct.names) + 2        # header + rule + body
        want = sep_pipes(rows[2])                          # the separator row
        @test want == 7                                    # 6 columns
        @test all(r -> sep_pipes(r) == want, rows)
        # The header's P>|z| must be escaped, not bare.
        @test occursin("P>\\|z\\|", rows[1])
        # A name containing a pipe must not break its row either.
        ct_pipe = ST.CoefTable{Float64}(["a|b"], [1.0], [0.1], [10.0], [0.0],
                                        [0.8], [1.2], 0.95)
        rows_p = filter(!isempty, split(strip(to_markdown(ct_pipe)), '\n'))
        @test all(r -> sep_pipes(r) == want, rows_p)

        # ── to_latex: no bare specials, Greek in math mode ──
        tex = to_latex(ct)
        body = filter(l -> occursin("&", l) && !occursin("Variable", l),
                      split(tex, '\n'))
        @test length(body) == length(ct.names)
        for l in body
            nm = strip(first(split(l, '&')))
            # Every underscore in a name must be backslash-escaped.
            for (i, c) in enumerate(nm)
                c == '_' && @test i > 1 && nm[prevind(nm, i)] == '\\'
            end
        end
        @test occursin("\$\\rho\$", tex)       # ρ became $\rho$
        @test !occursin("ρ", tex)              # and no raw Greek is left
        @test occursin("\\begin{tabular}", tex) && occursin("\\end{tabular}", tex)
        @test ST._escape_latex("a_b&c%d#e") == "a\\_b\\&c\\%d\\#e"
        @test ST._escape_latex("plain") == "plain"

        # ── to_csv: RFC 4180 quoting ──
        ct_c = ST.CoefTable{Float64}(["a,b", "q\"r", "plain"], [1.0, 2.0, 3.0],
            [0.1, 0.2, 0.3], [10.0, 10.0, 10.0], [0.0, 0.0, 0.0],
            [0.8, 1.6, 2.4], [1.2, 2.4, 3.6], 0.95)
        csv = to_csv(ct_c)
        # A minimal RFC 4180 field splitter: quotes toggle, commas inside them
        # are data. Each row must come out at exactly the 7-column schema.
        function rfc_fields(line::AbstractString)
            fields = String[]; buf = IOBuffer(); inq = false; i = 1
            chars = collect(line)
            while i <= length(chars)
                c = chars[i]
                if inq
                    if c == '"'
                        if i < length(chars) && chars[i + 1] == '"'
                            print(buf, '"'); i += 1
                        else
                            inq = false
                        end
                    else
                        print(buf, c)
                    end
                elseif c == '"'
                    inq = true
                elseif c == ','
                    push!(fields, String(take!(buf)))
                else
                    print(buf, c)
                end
                i += 1
            end
            push!(fields, String(take!(buf)))
            return fields
        end
        lines = filter(!isempty, split(strip(csv), '\n'))
        @test length(lines) == 4                       # header + 3 rows
        @test all(l -> length(rfc_fields(l)) == 7, lines)
        # …and the names survive the round trip intact.
        @test [rfc_fields(l)[1] for l in lines[2:end]] == ["a,b", "q\"r", "plain"]
        @test ST._csv_field("plain") == "plain"        # untouched when safe

        # ── `digits` actually reaches the output ──
        # Count the decimals in the estimate column at two settings.
        dec(cell) = (parts = split(strip(cell), '.'); length(parts) == 2 ?
                     length(filter(isdigit, parts[2])) : 0)
        md2 = to_markdown(ct; digits=2); md6 = to_markdown(ct; digits=6)
        @test md2 != md6
        cell2 = split(filter(!isempty, split(strip(md2), '\n'))[3], '|')[3]
        cell6 = split(filter(!isempty, split(strip(md6), '\n'))[3], '|')[3]
        @test dec(cell2) == 2
        @test dec(cell6) == 6
        tex2 = to_latex(ct; digits=2); tex6 = to_latex(ct; digits=6)
        @test tex2 != tex6
        @test dec(split(first(filter(l -> occursin("&", l) &&
                 !occursin("Variable", l), split(tex2, '\n'))), '&')[2]) == 2
        @test dec(split(first(filter(l -> occursin("&", l) &&
                 !occursin("Variable", l), split(tex6, '\n'))), '&')[2]) == 6
        @test ST._fmt(1.23456, 0) == "1"
        @test ST._fmt(1.23456, 3) == "1.235"

        # ── regression_table matches coefficients by name, not position ──
        # SAR carries ρ and SEM carries λ, so neither name appears in both
        # fits. Each must show exactly one number on its row; borrowing the
        # neighbouring model's figure would put two there.
        m_sar = fit_spatial(SpatialEconometrics.CrossSection.SAR(), W, X, y)
        m_sem = fit_spatial_error(W, X, y)
        rt = regression_table(m_sar, m_sem; model_names=["SAR", "SEM"])
        @test occursin("SAR", rt) && occursin("SEM", rt)
        rt_lines = split(rt, '
')
        nnum(l) = length(collect(eachmatch(r"-?\d+\.\d+", l)))
        rho_line = first(filter(l -> startswith(l, "ρ"), rt_lines))
        lam_line = first(filter(l -> startswith(l, "λ"), rt_lines))
        @test nnum(rho_line) == 1      # SAR only
        @test nnum(lam_line) == 1      # SEM only
        # A name both models share must show both numbers.
        int_line = first(filter(l -> startswith(l, "(Intercept)"), rt_lines))
        @test nnum(int_line) == 2
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


    @testset "19b. Probit Gibbs internals (audit finding 21)" begin
        DC = SpatialEconometrics.SpatialDiscreteChoice
        L = 6; W = make_rook_weights(L); N = size(W, 1)
        Wf = SparseMatrixCSC{Float64, Int}(W)
        IN = sparse(1.0I, N, N)

        # ── `_rho_logpost` against dense log|A| - ½r'r ──
        # The Metropolis step for ρ is only as good as this kernel. Compare it
        # to the textbook expression formed with a dense determinant.
        rng = MersenneTwister(4100)
        z_t = randn(rng, N); Xb_t = randn(rng, N)
        for ρ in (-0.8, -0.3, 0.0, 0.25, 0.6, 0.9)
            Ad = Matrix(IN - ρ * Wf)
            r = Ad * z_t .- Xb_t
            want = logabsdet(Ad)[1] - 0.5 * dot(r, r)
            @test isapprox(DC._rho_logpost(ρ, Wf, z_t, Xb_t, N), want; rtol=1e-12)
        end
        # Singular A must fall back to -Inf rather than throw: ρ = 1 on a
        # row-standardised W puts a zero eigenvalue in A exactly.
        lp_sing = DC._rho_logpost(1.0, Wf, z_t, Xb_t, N)
        @test !isnan(lp_sing) && lp_sing < Inf

        # ── `_rand_truncnorm` against Distributions.truncated ──
        # Four configurations, including a mean two SD outside the kept region,
        # which is where a naive rejection sampler would stall or bias.
        for (mu, sd, lo, hi) in ((0.0, 1.0, 0.0, Inf), (0.0, 1.0, -Inf, 0.0),
                                 (-2.0, 1.0, 0.0, Inf), (1.5, 0.5, -1.0, 2.0))
            d = truncated(Normal(mu, sd), lo == -Inf ? -Inf : lo,
                                          hi ==  Inf ?  Inf : hi)
            rg = MersenneTwister(4101)
            draws = [DC._rand_truncnorm(rg, mu, sd, lo, hi) for _ in 1:40_000]
            @test all(v -> lo <= v <= hi, draws)
            @test isapprox(mean(draws), mean(d); atol=6e-3 + 0.02 * std(d))
            @test isapprox(std(draws), std(d); rtol=0.05)
        end

        # ── `_sweep_latent!` has the right stationary distribution ──
        # Untruncated, the Gibbs sweep over the precision-parameterised
        # conditionals must leave N(μ, H⁻¹) invariant. H = A'A for the SAR
        # latent, so H⁻¹ = (A'A)⁻¹ is the implied latent covariance.
        ρ_s = 0.4
        A_s = IN - ρ_s * Wf
        H_s = SparseMatrixCSC{Float64, Int}(A_s' * A_s)
        Σ_s = inv(Matrix(H_s))
        mu_s = collect(range(-1.0, 1.0; length=N))
        lo_s = fill(-Inf, N); hi_s = fill(Inf, N)
        z_s = copy(mu_s)
        rg_s = MersenneTwister(4102)
        n_keep = 30_000
        acc_m = zeros(N); acc_v = zeros(N)
        for s in 1:(2000 + n_keep)
            DC._sweep_latent!(z_s, H_s, mu_s, lo_s, hi_s, rg_s)
            if s > 2000
                acc_m .+= z_s
                acc_v .+= (z_s .- mu_s) .^ 2
            end
        end
        acc_m ./= n_keep; acc_v ./= n_keep
        # Mean reproduces μ, variance reproduces diag(H⁻¹).
        @test maximum(abs.(acc_m .- mu_s)) < 0.10
        @test isapprox(acc_v, diag(Σ_s); rtol=0.08)
        # The conditional variance the sweep uses is 1/H_ii exactly, not a
        # marginal — check that against the matrix it was handed.
        @test all(i -> H_s[i, i] > 0.0, 1:N)

        # ── Truncation is enforced in the real sweep ──
        y_b = Float64.(rand(MersenneTwister(4103), N) .< 0.5)
        lo_b = [y_b[i] == 1.0 ? 0.0 : -Inf for i in 1:N]
        hi_b = [y_b[i] == 1.0 ? Inf : 0.0 for i in 1:N]
        z_b = [y_b[i] == 1.0 ? 0.5 : -0.5 for i in 1:N]
        for _ in 1:50
            DC._sweep_latent!(z_b, H_s, mu_s, lo_b, hi_b, MersenneTwister(4104))
        end
        @test all(i -> (y_b[i] == 1.0) == (z_b[i] >= 0.0), 1:N)

        # ── β conditional is the conjugate posterior ──
        # b₁ = B₁X'Az with B₁ = (X'X + I/v)⁻¹; the draw adds chol(B₁).L·randn,
        # so the sampled covariance is B₁ and not B₁ᵀB₁ or its inverse.
        Xd = [ones(N) randn(MersenneTwister(4105), N, 2)]
        v_prior = 100.0
        B1 = inv(Symmetric(Xd' * Xd + (1 / v_prior) * I))
        Lb = cholesky(Symmetric(0.5 .* (B1 .+ B1'))).L
        @test isapprox(Lb * Lb', B1; rtol=1e-10)

        # ── Ordered probit: identification and Albert-Chib cutpoint bounds ──
        rng_o = MersenneTwister(4106)
        X_o = randn(rng_o, N, 2)
        z_o = (Matrix(IN) - 0.3 * Matrix(Wf)) \ (X_o * [1.0, -0.7] .+ randn(rng_o, N))
        y_o = [zi <= 0.0 ? 1 : (zi <= 1.1 ? 2 : 3) for zi in z_o]
        if all(j -> count(==(j), y_o) >= 5, 1:3)
            m_o = fit_spatial_ordered_probit(Wf, X_o, y_o; n_samples=500,
                burnin=250, varnames=["x1", "x2"], rng=MersenneTwister(4107))
            # τ₁ = 0 in every draw: it is fixed, not sampled, so no intercept
            # is needed and the scale is pinned by σ² = 1.
            @test all(iszero, m_o.cut_draws[:, 1])
            @test m_o.cutpoints[1] == 0.0
            # Cutpoints stay ordered in every retained draw.
            @test all(s -> m_o.cut_draws[s, 2] > m_o.cut_draws[s, 1],
                      1:size(m_o.cut_draws, 1))
            @test !("(Intercept)" in coefnames(m_o))
        end

        # ── Consistency: both samplers' bias shrinks with N ──
        # Finding 21 measured the small-N posterior bias; pin the direction of
        # travel so a future change cannot quietly make it worse. Single draws
        # at two sizes, same generating parameters.
        function bprobit_err(Lg, seed)
            Wg = make_rook_weights(Lg); Ng = size(Wg, 1)
            rg = MersenneTwister(seed)
            Xg = randn(rg, Ng, 2)
            Sg = inv(Matrix(sparse(1.0I, Ng, Ng) - 0.4 * Matrix(Wg)))
            yg = Float64.(Sg * (0.3 .+ Xg * [1.0, -0.6] .+ randn(rg, Ng)) .> 0)
            mg = fit_bayesian_spatial_probit(Wg, Xg, yg; n_samples=800,
                burnin=400, rng=MersenneTwister(seed + 11))
            return (abs(coef(mg)[2] - 1.0), mg.accept_rate)
        end
        # Averaged over ten seeds, not read off one. Measured over seeds
        # 4200:4209, the N = 196 fit is the closer one in 8 of 10 draws on
        # Julia 1.13 and 6 of 10 on 1.10, so the single-seed version of this
        # test was close to a coin flip -- it passed on seed 4200 by luck and
        # failed once the Julia version changed the draws. Shrinking bias is a
        # statement about the sampling distribution, so it is tested on a
        # summary of it. Medians: 0.911 -> 0.153 on 1.13, 0.198 -> 0.127 on
        # 1.10.
        seeds = 4200:4209
        es = Float64[]; as = Float64[]; el = Float64[]; al = Float64[]
        for sd in seeds
            a, b = bprobit_err(6, sd);  push!(es, a); push!(as, b)
            a, b = bprobit_err(14, sd); push!(el, a); push!(al, b)
        end
        # Every chain must sit in a usable acceptance band. Measured range
        # across both Julia versions and all ten seeds: 0.200 to 0.664. The
        # old 0.15-0.6 band was not only version-fragile, it was seed-fragile
        # on the version it was written on -- 1.13 reaches 0.630 at seed 4202.
        @test all(a -> 0.15 < a < 0.75, as)
        @test all(a -> 0.15 < a < 0.75, al)
        # The larger N must be the closer one on |β₁| in the median.
        @test median(el) < median(es)
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
            # The claim is the contrast with the 1e-10 above, so the threshold
            # is set orders of magnitude away from both sides rather than at
            # the value one draw produced. Measured across five seeds on two
            # Julia versions the discrepancy runs 0.026 to 0.616, while the
            # canonical links stay near 1e-15; the old 0.1 bound sat inside
            # that spread and failed at 0.076.
            @test maximum(abs, (H .- Xg' * (wh .* Xg)) ./ abs.(H)) > 1e-3
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

        # ---- digamma and trigamma against closed forms ---------------
        # ψ(1) = -γ, ψ(1/2) = -γ - 2log2, ψ(n) = -γ + Σ_{k<n} 1/k;
        # ψ₁(1) = π²/6, ψ₁(1/2) = π²/2, ψ₁(n) = π²/6 - Σ_{k<n} 1/k².
        let γ = 0.5772156649015328606
            @test isapprox(SF._digamma(1.0), -γ; rtol = 1e-13)
            @test isapprox(SF._digamma(0.5), -γ - 2log(2); rtol = 1e-13)
            @test isapprox(SF._trigamma(1.0), π^2 / 6; rtol = 1e-12)
            @test isapprox(SF._trigamma(0.5), π^2 / 2; rtol = 1e-12)
            for m in 2:8
                @test isapprox(SF._digamma(Float64(m)),
                               -γ + sum(1 / k for k in 1:(m - 1)); rtol = 1e-13)
                @test isapprox(SF._trigamma(Float64(m)),
                               π^2 / 6 - sum(1 / k^2 for k in 1:(m - 1)); rtol = 1e-12)
            end
            # the recurrences, across the switch from recursion to the series
            for x in (0.01, 0.37, 1.0, 4.5, 9.3, 9.999, 10.0, 25.0, 1e3)
                @test isapprox(SF._digamma(x + 1) - SF._digamma(x), 1 / x; rtol = 1e-11)
                @test isapprox(SF._trigamma(x + 1) - SF._trigamma(x),
                               -1 / x^2; rtol = 1e-10)
            end
        end

        # ---- the dispersion by maximum likelihood --------------------
        # Score for α at fixed μ, written in θ = 1/α, with ∂ℓ/∂α = -θ²s(θ).
        # Differentiating the Distributions log-likelihood in α must agree.
        ll_a(a) = sum(logpdf(NegativeBinomial(1 / a, (1 / a) / ((1 / a) + μ_log[i])),
                             Int(y_n[i])) for i in 1:n)
        for a in (0.2, 0.5, 0.6, 1.3)
            sc, _ = SF._negbin_theta_score(y_n, μ_log, 1 / a)
            @test isapprox(-(1 / a)^2 * sc, ForwardDiff.derivative(ll_a, a);
                           rtol = 1e-10)
        end
        # the reported derivative in θ is the derivative of the score
        for a in (0.3, 0.9)
            θa = 1 / a
            sc, dsc = SF._negbin_theta_score(y_n, μ_log, θa)
            h = 1e-5 * θa
            num = (SF._negbin_theta_score(y_n, μ_log, θa + h)[1] -
                   SF._negbin_theta_score(y_n, μ_log, θa - h)[1]) / (2h)
            @test isapprox(dsc, num; rtol = 1e-6)
            @test dsc < 0                      # the score falls in θ, rises in α
        end

        # The MLE is a root of that score, and it is not merely close to the
        # best point of a 2001-point grid over the same range -- it is better.
        α_ml = SF._negbin_alpha_mle(y_n, μ_log)
        @test abs(SF._negbin_theta_score(y_n, μ_log, 1 / α_ml)[1]) < 1e-8
        let grid = range(0.05, 2.0; length = 2001), vals = [ll_a(a) for a in grid]
            @test abs(α_ml - grid[argmax(vals)]) < 2e-3   # grid spacing is 1e-3
            @test ll_a(α_ml) >= maximum(vals) - 1e-9
        end
        # and it beats the moment estimate on its own criterion
        @test ll_a(α_ml) > ll_a(SF._update_negbin_alpha(y_n, μ_log; n_par = 3))
        # starting value must not change the answer
        for a0 in (1e-4, 0.05, 1.0, 20.0)
            @test isapprox(SF._negbin_alpha_mle(y_n, μ_log; α_start = a0), α_ml;
                           rtol = 1e-8)
        end

        # Underdispersed data maximises the likelihood at the Poisson limit,
        # so α̂ sits on the floor rather than going negative.
        @test SF._negbin_alpha_mle(yv3, μv) ≈ 1e-6
        @test SF._negbin_alpha_mle([1.0, 1.0], [1.0, 1.0]) ≈ 1e-6
        # the boundary test is the Poisson overdispersion score ½Σ[(y-μ)²-y]
        @test sum((yv3 .- μv) .^ 2 .- yv3) < 0
        @test sum((y_od .- μv) .^ 2 .- y_od) > 0
        @test SF._negbin_alpha_mle(y_od, μv) > 1e-6

        # α is consistent: the bias shrinks with N. 1500 observations with no
        # spatial structure should land within 10% of the truth.
        function alpha_at(nn, R, seed; mode = :moments)
            rg = MersenneTwister(seed); acc = 0.0
            for _ in 1:R
                Xn = hcat(ones(nn), randn(rg, nn))
                mu = exp.(Xn * [0.4, 0.6])
                yn = Float64[rand(rg, NegativeBinomial(r_nb, r_nb / (r_nb + m)))
                             for m in mu]
                a = 0.0
                for _ in 1:30
                    _, mh, _, _ = SF._fit_glm(Xn, yn, :negbin; α = a)
                    an = mode === :mle ?
                        SF._negbin_alpha_mle(yn, mh; α_start = a) :
                        SF._update_negbin_alpha(yn, mh; n_par = size(Xn, 2))
                    abs(an - a) < 1e-10 && (a = an; break)
                    a = an
                end
                acc += a
            end
            acc / R
        end
        @test isapprox(alpha_at(1500, 40, 2001), α_t; rtol = 0.10)

        # The MLE is consistent too, and at N = 1500 the two schemes agree:
        # the gap between them is a small-sample effect, not a different
        # estimand.
        @test isapprox(alpha_at(1500, 40, 2001; mode = :mle), α_t; rtol = 0.10)
        @test isapprox(alpha_at(1500, 40, 2001; mode = :mle),
                       alpha_at(1500, 40, 2001); rtol = 0.05)

        # The MLE is the more precise of the two, which is the actual reason
        # to prefer it: on the 60 samples below its root mean squared error
        # around the true α is 0.200 against the moment estimate's 0.249,
        # and over 3000 samples at this N it is 0.199 against 0.285, with a
        # median bias of -8.7% against -16.2%. On the fit_esf coverage
        # design below the moment estimate also lands on the positivity
        # floor for 5.6% of fits against the MLE's 0.7%, and those fits
        # cover 13-19 points worse than the rest.
        function alpha_rmse(nn, R, seed, mode)
            rg = MersenneTwister(seed); acc = 0.0
            for _ in 1:R
                Xn = hcat(ones(nn), randn(rg, nn))
                mu = exp.(Xn * [0.4, 0.6])
                yn = Float64[rand(rg, NegativeBinomial(r_nb, r_nb / (r_nb + m)))
                             for m in mu]
                a = 0.0
                for _ in 1:30
                    _, mh, _, _ = SF._fit_glm(Xn, yn, :negbin; α = a)
                    an = mode === :mle ?
                        SF._negbin_alpha_mle(yn, mh; α_start = a) :
                        SF._update_negbin_alpha(yn, mh; n_par = size(Xn, 2))
                    abs(an - a) < 1e-10 && (a = an; break)
                    a = an
                end
                acc += (a - α_t)^2
            end
            sqrt(acc / R)
        end
        @test alpha_rmse(81, 60, 3001, :mle) <
              0.9 * alpha_rmse(81, 60, 3001, :moments)

        # ---- the dispersion= switch inside fit_esf -------------------
        let Wd = make_rook_weights(8), Nd = size(Wd, 1)
            rgd = MersenneTwister(515)
            Xd = randn(rgd, Nd, 2)
            ηd = 0.3 .+ Xd * [0.5, -0.4]
            yd = Float64[rand(rgd, NegativeBinomial(r_nb, r_nb / (r_nb + exp(e))))
                         for e in ηd]
            m_ml = fit_esf(Xd, yd, Wd; family = :negbin, max_eigenvectors = 4)
            m_mm = fit_esf(Xd, yd, Wd; family = :negbin, max_eigenvectors = 4,
                           dispersion = :moments)
            # :mle is the default
            @test m_ml.alpha == fit_esf(Xd, yd, Wd; family = :negbin,
                                        max_eigenvectors = 4,
                                        dispersion = :mle).alpha
            # the two schemes differ, and :mle has the higher likelihood at
            # the dispersion it reports
            @test m_ml.alpha != m_mm.alpha
            s_ml = abs(SF._negbin_theta_score(yd, m_ml.fitted, 1 / m_ml.alpha)[1])
            s_mm = abs(SF._negbin_theta_score(yd, m_mm.fitted, 1 / m_mm.alpha)[1])
            @test s_ml < 1e-4
            @test s_mm > 100 * s_ml
            @test_throws AssertionError fit_esf(Xd, yd, Wd; family = :negbin,
                                                dispersion = :mom)
            # dispersion= is inert for the other three families
            for fam in (:gaussian, :poisson)
                yf = fam === :gaussian ? Xd * [1.0, -0.5] .+ randn(rgd, Nd) :
                     Float64[rand(rgd, Poisson(exp(e))) for e in ηd]
                a = fit_esf(Xd, yf, Wd; family = fam, max_eigenvectors = 3)
                b = fit_esf(Xd, yf, Wd; family = fam, max_eigenvectors = 3,
                            dispersion = :moments)
                @test coef(a) == coef(b)
                @test a.alpha == b.alpha == 0.0
            end
        end

        # ---- coverage for the canonical families ----------------------
        # Poisson and logit report the inverse expected information, which is
        # the right variance for a canonical link; NB2 runs mildly low.
        function esf_coverage(family, R, seed; dispersion = :mle)
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
                elseif family === :negbin
                    Float64[rand(rg, NegativeBinomial(r_nb,
                        r_nb / (r_nb + exp(clamp(e, -20, 5))))) for e in et]
                else
                    Float64[rand(rg, Bernoulli(1 / (1 + exp(-e)))) for e in et]
                end
                mm = fit_esf(Xr, yr, Wl; family = family, eigen_basis = eb,
                             max_eigenvectors = 10, dispersion = dispersion)
                abs(coef(mm)[2] - 0.6) <= 1.96 * stderror(mm)[2] && (hit += 1)
            end
            hit / R
        end
        @test esf_coverage(:poisson, 120, 1234) > 0.87
        @test esf_coverage(:logit, 120, 1234) > 0.87
        # The negative binomial reaches the canonical families' coverage once
        # the dispersion is its own MLE. Over 6000 replications it runs at
        # 93.3% against the Poisson's 93.1% on this design; with the moment
        # estimate it was 92.2%. Neither reaches 95%: what is left is the
        # post-selection effect that the fit_esf docstring documents, not the
        # dispersion.
        @test esf_coverage(:negbin, 120, 1234) > 0.87
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
        # The cross-section path shares the misscaled bread described in
        # testset 7, so it gets the same exact scale check: scaling y by c
        # scales every coefficient's robust standard error by exactly c and
        # leaves rho's alone.
        m_sar_c = fit_spatial(CSx.SAR(), W, X, 3.0 .* y; varnames=["x1", "x2"])
        se_cs_1 = rt.se_robust
        se_cs_c = robust_coeftable(m_sar_c, ConleySpatialHAC(coords, 3.0)).se_robust
        @test se_cs_c[1:(end - 1)] ≈ 3.0 .* se_cs_1[1:(end - 1)] rtol = 1e-4
        @test se_cs_c[end] ≈ se_cs_1[end] rtol = 1e-4

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
        SSx = SpatialEconometrics.SpatialShiftShare
        NWx = SpatialEconometrics.SpatialNetworks
        @test MSx._chi2_sf === UTx._chi2_sf
        @test PDx._chi2_sf === UTx._chi2_sf
        @test RMx._chi2_sf === UTx._chi2_sf
        # A fourth copy did get added, in SpatialShiftShare, and this test did
        # not catch it because it was spelled `_chisq_sf`. Identity is pinned
        # by module here rather than by name, so a renamed copy fails too.
        @test SSx._chi2_sf === UTx._chi2_sf
        @test NWx._chi2_sf === UTx._chi2_sf
        @test !isdefined(SSx, :_chisq_sf)
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


    @testset "27. Durbin Score Tests (Koley & Bera)" begin
        SD = SpatialEconometrics.SpatialDiagnostics
        chi2sf = SpatialEconometrics.SpatialUtils._chi2_sf

        # ── 27a. shapes, degrees of freedom, and the guards ───────────────
        @testset "27a. structure" begin
            rng = MersenneTwister(2701)
            W = make_rook_weights(7); N = size(W, 1)
            X = hcat(ones(N), randn(rng, N, 2))
            y = X * [1.0, 0.5, -0.3] .+ randn(rng, N)

            t = sdm_tests(y, X, W)
            @test t.joint.df == 3          # rho plus two Durbin terms
            @test t.robust_lag.df == 1
            @test t.durbin.df == 2
            @test t.robust_durbin.df == 2
            @test length(t.terms) == 2
            @test all(s -> s.df == 1, t.terms)
            for s in (t.joint, t.robust_lag, t.durbin, t.robust_durbin, t.terms...)
                @test s.stat >= 0
                @test 0 <= s.p_value <= 1
                @test s isa ScoreTestResult
            end
            # the names carry the column the term belongs to
            @test t.terms[1].test_name == "RS theta[x2] | rest"
            @test t.terms[2].test_name == "RS theta[x3] | rest"
            @test occursin("df", sprint(show, t.joint))

            # X with no non-constant column has no W*X_d block to test:
            # W * 1 = 1 for a row-standardised W, so the constant drops out.
            @test_throws ArgumentError lm_test_sdm(y, ones(N, 1), W)
            @test_throws ArgumentError lm_test_durbin(y, ones(N, 1), W)
            @test_throws DimensionMismatch lm_test_sdm(y[1:end-1], X, W)

            # a duplicated covariate makes the Durbin block singular, and the
            # per-term statistics must say so rather than return a ridged
            # number (finding 16).
            Xdup = hcat(ones(N), X[:, 2], X[:, 2])
            @test_throws ArgumentError lm_test_durbin_terms(y, Xdup, W)
        end

        # ── 27b. RS_rho IS the existing LM-lag statistic ──────────────────
        # The two are the same algebra reached by different code paths, so
        # they must agree to machine precision. Anything else means one of
        # them is wrong.
        @testset "27b. reduction to lm_test_lag" begin
            rng = MersenneTwister(2702)
            W = make_rook_weights(7); N = size(W, 1)
            X = hcat(ones(N), randn(rng, N, 2))
            y = X * [1.0, 0.5, -0.3] .+ randn(rng, N)

            c = SD._sdm_score_components(y, X, W)
            @test c.d_rho^2 / c.J_rr ≈ lm_test_lag(y, X, W).stat rtol = 1e-14

            # The block inverse identity: the per-term formula applied at the
            # rho position equals the Bera-Yoon robust lag statistic.
            Jinv = inv(Symmetric(c.J)); u = Jinv * c.d
            @test u[1]^2 / Jinv[1, 1] ≈
                  lm_test_robust_lag_durbin(y, X, W).stat rtol = 1e-12

            # The theta block carries no determinant term; only rho's does.
            @test c.J_rr > c.Tr
            @test c.J[1, 1] == c.J_rr
            @test c.J[2:end, 2:end] ≈ Matrix(c.J_tt)
        end

        # ── 27c. the analytic scores against ForwardDiff ──────────────────
        # The SDM log-likelihood written here from the model, not from the
        # package source:
        #   y = rho*W*y + X*beta + W*X_d*theta + eps,  eps ~ N(0, sigma2 I)
        @testset "27c. scores against automatic differentiation" begin
            rng = MersenneTwister(2703)
            W = make_rook_weights(7); N = size(W, 1)
            Wden = Matrix(W)
            X = hcat(ones(N), randn(rng, N, 2))
            y = X * [1.0, 0.5, -0.3] .+ randn(rng, N)

            function sdm_loglik(p)
                K = size(X, 2)
                rho = p[1]; theta = p[2:3]; beta = p[4:3+K]; s2 = p[end]
                r = (I - rho * Wden) * y .- X * beta .- (Wden * X[:, 2:3]) * theta
                return -0.5 * N * (log(2pi) + log(s2)) +
                       logabsdet(I - rho * Wden)[1] - dot(r, r) / (2 * s2)
            end

            c = SD._sdm_score_components(y, X, W)
            bhat = X \ y
            e = y .- X * bhat
            p0 = vcat(0.0, 0.0, 0.0, bhat, dot(e, e) / N)
            g = ForwardDiff.gradient(sdm_loglik, p0)

            @test c.d_rho ≈ g[1] rtol = 1e-10
            @test c.d_theta[1] ≈ g[2] rtol = 1e-10
            @test c.d_theta[2] ≈ g[3] rtol = 1e-10
            # the nuisance scores vanish at the OLS null, which is what makes
            # these the right scores to test with
            @test maximum(abs, g[4:6]) < 1e-10
            @test abs(g[end]) < 1e-10
        end

        # ── 27d. the null distribution ────────────────────────────────────
        # The audit's standing gate: a score statistic is believed only once
        # its null distribution has been measured. Each statistic must match
        # its chi-squared reference in mean (df), variance (2df) and size at
        # three levels. Measured over 20000 replications this session:
        #
        #   statistic        df   mean    var    p<.10  p<.05  p<.01
        #   RS joint          3   3.054   6.050  0.103  0.051  0.010
        #   RS* rho | theta   1   0.978   1.953  0.097  0.047  0.010
        #   RS theta          2   2.076   4.075  0.107  0.054  0.009
        #   RS* theta | rho   2   2.065   4.077  0.107  0.052  0.010
        #   RS theta[1]       1   1.022   1.956  0.104  0.052  0.009
        #
        # The suite runs 3000, so the bounds below are about three standard
        # errors wide. They are still far tighter than the 1.3x-17x
        # mis-scalings of findings 13-16.
        @testset "27d. null distribution, 3000 replications" begin
            R = 3000
            rng = MersenneTwister(2704)
            W = make_rook_weights(9); N = size(W, 1)
            X = hcat(ones(N), randn(rng, N, 2))
            beta = [1.0, 0.5, -0.3]

            S = [Float64[] for _ in 1:5]
            for _ in 1:R
                y = X * beta .+ randn(rng, N)
                t = sdm_tests(y, X, W)
                for (i, s) in enumerate((t.joint, t.robust_lag, t.durbin,
                                         t.robust_durbin, t.terms[1]))
                    push!(S[i], s.stat)
                end
            end

            for (i, df) in enumerate((3, 1, 2, 2, 1))
                v = S[i]
                @test mean(v) ≈ df atol = 0.25               # E[chi2] = df
                @test var(v) ≈ 2df rtol = 0.20               # Var[chi2] = 2df
                @test 0.075 <= mean(x -> chi2sf(x, df) < 0.10, v) <= 0.130
                @test 0.035 <= mean(x -> chi2sf(x, df) < 0.05, v) <= 0.068
                @test 0.004 <= mean(x -> chi2sf(x, df) < 0.01, v) <= 0.019
            end
        end

        # ── 27e. the robust forms do what they claim ──────────────────────
        # This is the whole point of the Bera-Yoon construction and it is
        # sharp: under a lag with no Durbin terms, the plain RS_theta picks up
        # the lag and over-rejects, while RS*_theta holds its nominal size --
        # and symmetrically for LM-lag against RS*_rho under SLX data.
        # Measured over 2000 replications this session:
        #
        #   data        RS theta  RS* theta | LM-lag  RS* rho
        #   SAR(0.5)      0.161     0.051   |  0.926   0.887
        #   SLX           0.716     0.703   |  0.070   0.052
        #   SDM(0.4)      0.772     0.448   |  0.870   0.651
        @testset "27e. robustness to the other parameter" begin
            R = 1000
            rng = MersenneTwister(2705)
            W = make_rook_weights(9); N = size(W, 1)
            X = hcat(ones(N), randn(rng, N, 2))

            function rejection(dgp)
                out = zeros(Int, 4)
                for _ in 1:R
                    y, _ = simulate_spatial(dgp, W, X[:, 2:end]; rng = rng)
                    t = sdm_tests(y, X, W)
                    t.durbin.p_value        < 0.05 && (out[1] += 1)
                    t.robust_durbin.p_value < 0.05 && (out[2] += 1)
                    t.lag.p_value           < 0.05 && (out[3] += 1)
                    t.robust_lag.p_value    < 0.05 && (out[4] += 1)
                end
                out ./ R
            end

            # a lag, no Durbin terms
            r = rejection(SAR_DGP(0.5, [0.5, -0.3], 1.0))
            @test r[1] > 0.10                  # RS theta is fooled by the lag
            @test 0.03 <= r[2] <= 0.08         # RS* theta is not
            @test r[3] > 0.80 && r[4] > 0.80   # both lag tests have power

            # Durbin terms, no lag
            r = rejection(SLX_DGP([0.5, -0.3], [0.6, -0.5], 1.0))
            @test r[1] > 0.50 && r[2] > 0.50   # both Durbin tests have power
            @test r[3] > r[4]                  # LM-lag is fooled by W*X
            @test 0.03 <= r[4] <= 0.08         # RS* rho is not

            # both present: everything should reject
            r = rejection(SDM_DGP(0.4, [0.5, -0.3], [0.5, -0.4], 1.0))
            @test all(x -> x > 0.30, r)
        end
    end


    @testset "28. knn W Selection by AIC" begin
        # The generating design throughout: coordinates uniform on the unit
        # square, y from a SAR on the knn W at k = 6, rho = 0.5.
        K_STAR = 6
        RHO = 0.5
        function knn_draw(rng, N)
            coords = rand(rng, N, 2)
            X = randn(rng, N, 2)
            y, _ = simulate_spatial(SAR_DGP(RHO, [1.0, -0.5], 0.5),
                                    knn_weights(coords, K_STAR), X; rng = rng)
            return coords, X, y
        end

        # ── 28a. structure and the guards ─────────────────────────────────
        @testset "28a. structure" begin
            rng = MersenneTwister(2801)
            coords, X, y = knn_draw(rng, 60)
            sel = select_knn_w(coords, y, X; k_range = 2:10)

            @test sel isa KnnSelectionResult
            @test sel.k_values == collect(2:10)
            @test length(sel.aic) == length(sel.k_values)
            @test length(sel.bic) == length(sel.k_values)
            @test length(sel.loglik) == length(sel.k_values)
            @test length(sel.rho) == length(sel.k_values)
            @test length(sel.converged) == length(sel.k_values)
            @test sel.criterion === :aic
            @test sel.best_k in sel.k_values
            @test size(sel.best_W) == (60, 60)
            @test occursin("Selected", sprint(show, sel))

            # best_W must be exactly the knn W at best_k, not a near miss
            @test sel.best_W ≈ knn_weights(coords, sel.best_k)

            # knn cannot leave an isolate: every unit has exactly k neighbours
            @test all(iszero, sel.n_islands)

            # the criteria must be the ones spatial_aic/spatial_bic would give
            i = findfirst(==(sel.best_k), sel.k_values)
            m = fit_spatial(SAR(), sel.best_W, X, y)
            @test sel.aic[i] ≈ spatial_aic(m) rtol = 1e-8
            @test sel.bic[i] ≈ spatial_bic(m) rtol = 1e-8
            @test sel.loglik[i] ≈ loglikelihood(m) rtol = 1e-8
            @test sel.rho[i] ≈ m.rho rtol = 1e-8

            # guards
            @test_throws ArgumentError select_knn_w(coords, y, X; k_range = 60:80)
            @test_throws ArgumentError select_knn_w(coords, y, X; criterion = :foo)
            @test_throws DimensionMismatch select_knn_w(coords, y[1:end-1], X)
            @test_throws DimensionMismatch select_knn_w(coords, y, X[1:end-1, :])

            # a range that only partly exceeds N is clipped, not rejected
            @test select_knn_w(coords, y, X; k_range = 55:70).k_values ==
                  collect(55:59)
        end

        # ── 28b. the curve, and why there is something to search ──────────
        # Measured on one N = 200 draw: AIC 312.7 at k = 2, down to 294.3 at
        # the true k = 6, up to 329.1 at k = 20 — a 34.9-point spread with an
        # interior minimum. The non-monotonicity is the paper's point: too
        # sparse a W misses real neighbours, too dense a W dilutes them.
        @testset "28b. the AIC curve is not monotone" begin
            rng = MersenneTwister(5150)
            coords, X, y = knn_draw(rng, 200)
            sel = select_knn_w(coords, y, X; k_range = 2:20)

            @test all(sel.converged)
            # NOT sel.best_k == K_STAR. 28c measures that at 57% of draws at
            # this N, so pinning it on one draw asserts something the
            # measurement says fails about two draws in five -- it passed on
            # one platform and reported k = 7 on another. The recovery claim
            # belongs to the rate in 28c; what a single draw can carry is the
            # shape of the curve.
            d = diff(sel.aic)
            @test !all(>(0), d) && !all(<(0), d)        # not monotone
            @test maximum(sel.aic) - minimum(sel.aic) > 10.0
            # the minimum is interior, not at either end of the range
            @test sel.best_k != first(sel.k_values)
            @test sel.best_k != last(sel.k_values)

            # rho-hat drifts upward with k: a denser W spreads the same
            # dependence over more neighbours, and rho absorbs it. Counted as
            # upward steps rather than as a correlation with k, because
            # rho-hat is concave in k -- it climbs fast and then flattens, so
            # Pearson against raw k measures the curvature as much as the
            # monotonicity (it is 0.947 here, against 17 of 18 steps up).
            @test count(>(0), diff(sel.rho)) >= length(sel.rho) - 2
            @test sel.rho[findfirst(==(2), sel.k_values)] < RHO
            @test sel.rho[findfirst(==(20), sel.k_values)] > RHO

            # AIC and BIC rank identically here, because with the variant and
            # X fixed the parameter count does not change with k. Documented
            # rather than relied on: it is why the criterion choice is inert
            # in this search.
            selb = select_knn_w(coords, y, X; k_range = 2:20, criterion = :bic)
            @test selb.best_k == sel.best_k
            @test sortperm(sel.aic) == sortperm(selb.bic)
            @test selb.criterion === :bic

            # SDM lags X as well, so the regressors change with k too
            seld = select_knn_w(coords, y, X; k_range = 2:12, variant = SDM())
            @test all(seld.converged)
            @test seld.best_k in 2:12
        end

        # ── 28c. does the search find the generating k? ───────────────────
        # Measured over 300 draws with k searched over 2:20 (5.3% by chance):
        # k-hat is exactly 6 in 42.3% / 57.3% / 77.0% of draws at N = 100 /
        # 200 / 400, and within +/-1 in 68.3% / 86.0% / 95.0%. The median
        # k-hat is 6 at every N. The suite runs the cheap corner of that:
        # N = 200 over 2:12, 40 draws, where three independent seeds gave
        # exact rates of 0.45-0.625 and within-1 rates of 0.825-0.875
        # against a 9.1% chance rate.
        @testset "28c. recovery of the generating k, 40 draws" begin
            R = 40
            rng = MersenneTwister(2803)
            hits = Int[]
            better = 0
            for _ in 1:R
                coords, X, y = knn_draw(rng, 200)
                sel = select_knn_w(coords, y, X; k_range = 2:12)
                push!(hits, sel.best_k)
                # and the point of selecting at all: it must beat a fixed,
                # wrong k on |rho-hat - rho|. At k = 2 the fixed choice is
                # badly biased (-35.1% over 300 draws), so this is nearly
                # always true -- 98.3% there, asserted loosely here.
                r_sel = fit_spatial(SAR(), sel.best_W, X, y).rho
                r_2 = fit_spatial(SAR(), knn_weights(coords, 2), X, y).rho
                abs(r_sel - RHO) < abs(r_2 - RHO) && (better += 1)
            end

            @test count(==(K_STAR), hits) / R >= 0.25          # chance: 0.091
            @test count(k -> abs(k - K_STAR) <= 1, hits) / R >= 0.55
            @test median(hits) == K_STAR
            @test better / R >= 0.80
        end

        # ── 28d. what the wrong k costs, on one draw ──────────────────────
        # The across-draw version, 300 draws at N = 200, is in the
        # select_knn_w docstring: mean rho-hat runs 0.325 at k = 2 (-35.1%
        # bias) through 0.493 at the true k = 6 (-1.4%) to 0.646 at k = 20
        # (+29.1%), while selecting gives 0.502 (+0.5%). The suite checks the
        # direction and ordering on a single draw, which is cheap and still
        # falsifiable.
        @testset "28d. rho-hat moves with k" begin
            rng = MersenneTwister(2804)
            coords, X, y = knn_draw(rng, 200)
            r(k) = fit_spatial(SAR(), knn_weights(coords, k), X, y).rho
            r2, r6, r20 = r(2), r(K_STAR), r(20)
            # The ordering is the property, and it is scale-free: rho-hat
            # rises with k whatever the draw. Absolute margins against the
            # true rho are not -- `r20 > RHO + 0.05` held on one platform and
            # gave 0.5404 on another, because where the whole sequence sits
            # is one draw's sampling noise. The level claims are made across
            # draws in the docstring's bias table, not here.
            @test r2 < r6 < r20                 # monotone in k, both sides
            @test r20 - r2 > 0.05               # and the drift is not trivial
            @test r2 < RHO                      # too sparse understates
        end
    end

    @testset "29. Identification guards on the design matrix" begin
        SU = SpatialEconometrics.SpatialUtils

        # Two estimators used to meet a rank-deficient design the same way
        # the audit kept finding: `fit_spatial` died inside LAPACK with
        # "Zero pivot found at index 2", which says nothing about the design,
        # and `fit_gravity` returned coefficients of order 1e12 that cancel
        # against each other with a plausible pseudo-R2 beside them. Neither
        # is a numerical hiccup: the coefficient is not identified, and the
        # estimator should say which column is the problem.

        @testset "29a. rank_deficient_columns against known ranks" begin
            # Deterministic designs throughout: a pinned random draw is not
            # a reference, and the streams differ across Julia versions.
            N = 40
            v1 = Float64[sin(i / 3) for i in 1:N]
            v2 = Float64[cos(i / 5) for i in 1:N]
            v3 = Float64[i % 7 for i in 1:N]

            @test isempty(SU.rank_deficient_columns(hcat(v1, v2, v3)))

            # Which column of a dependent set gets deferred is pivoting's
            # choice -- it keeps the largest-norm ones -- so the claim is
            # that exactly one goes, and that dropping it restores full
            # rank. Naming an index would be asserting LAPACK's ordering.
            for A in (hcat(v1, v2, v1), hcat(v1, v2, 2 .* v1 .- 3 .* v2))
                bad = SU.rank_deficient_columns(A)
                @test length(bad) == 1
                keep = setdiff(1:3, bad)
                @test isempty(SU.rank_deficient_columns(A[:, keep]))
                @test rank(A) == 2
            end

            # An intercept duplicated, which is what a caller who adds their
            # own constant column hands to fit_spatial.
            dup_const = hcat(ones(N), ones(N), v1)
            @test length(SU.rank_deficient_columns(dup_const)) == 1

            # Fewer rows than columns: the excess cannot be independent.
            wide = hcat(v1[1:3], v2[1:3], v3[1:3], v1[1:3] .+ v2[1:3])
            @test length(SU.rank_deficient_columns(wide)) == 1

            @test SU.rank_deficient_columns(zeros(N, 2)) == [1, 2]
            @test isempty(SU.rank_deficient_columns(zeros(N, 0)))

            # False positives are the thing to fear in a guard that throws,
            # so: a design that is merely awkward must pass. Correlation of
            # 0.9999 between two columns is identified, and so is a column
            # scaled by 1e6 -- the test is on the conditioning of the span,
            # not on how big the numbers are.
            near = v1 .+ 1e-4 .* v2
            @test cor(v1, near) > 0.9999
            @test isempty(SU.rank_deficient_columns(hcat(v1, near, v3)))
            @test isempty(SU.rank_deficient_columns(hcat(1e6 .* v1, v2, v3)))
        end

        @testset "29b. fit_spatial rejects a design it cannot identify" begin
            W = row_standardize(make_rook_weights(8))
            N = size(W, 1)
            x = Float64[sin(i / 4) for i in 1:N]
            X = reshape(x, N, 1)
            # y built from the model directly, with a deterministic
            # disturbance, so every number here is reproducible on any
            # version.
            eps = Float64[0.3 * cos(i / 2) for i in 1:N]
            rho_true, beta_true = 0.5, 1.5
            y = (I - rho_true * W) \ (x .* beta_true .+ eps)

            # The caller supplied their own intercept, and one is added.
            @test_throws ArgumentError fit_spatial(SAR(), W, hcat(ones(N), x), y)
            # A duplicated covariate.
            @test_throws ArgumentError fit_spatial(SAR(), W, hcat(x, x), y)
            # A third column that is a combination of the first two.
            x2 = Float64[i % 5 for i in 1:N]
            @test_throws ArgumentError fit_spatial(
                SAR(), W, hcat(x, x2, x .- 2 .* x2), y)
            # SDM builds W*X into the design, so a duplicate there is caught
            # on the augmented matrix rather than on what the caller passed.
            @test_throws ArgumentError fit_spatial(SDM(), W, hcat(x, x), y)

            # The message has to name the problem, or it is no better than
            # the LAPACK one it replaces.
            err = try
                fit_spatial(SAR(), W, hcat(ones(N), x), y); nothing
            catch e
                sprint(showerror, e)
            end
            @test occursin("not identified", err)
            @test occursin("intercept", err)
            @test occursin("fit_spatial:", err)

            # And a design that is fine stays fine: the guard is a
            # precondition, not a change to the estimator.
            m = fit_spatial(SAR(), W, X, y)
            @test isapprox(m.rho, rho_true; atol = 0.2)
            @test isapprox(m.coef[2], beta_true; atol = 0.2)
        end

        @testset "29c. fit_gravity rejects what the fixed effects absorb" begin
            No, Nd = 10, 8
            N = No * Nd
            o = repeat(1:No, inner = Nd)
            d = repeat(1:Nd, outer = No)
            # Deterministic flows from a known log-link DGP, so the control
            # below is a statement about this test's data and not a draw.
            # Both covariates have to vary within origin AND within
            # destination, or the fixed effects absorb one and the guard
            # fires on the control itself. Modular functions of the pair
            # index do that without an RNG.
            logd = Float64[1.0 + ((o[n] * d[n]) % 11) / 4 for n in 1:N]
            lang = Float64[((o[n] + 2 * d[n]) % 5 == 0) ? 1.0 : 0.0 for n in 1:N]
            Xg = hcat(logd, lang)
            ao = Float64[0.3 * sin(i) for i in 1:No]
            ad = Float64[0.2 * cos(i) for i in 1:Nd]
            eta = Xg * [-0.9, 0.4] .+ 3.0
            for n in 1:N; eta[n] += ao[o[n]] + ad[d[n]]; end
            yg = round.(exp.(eta))

            # Control first: the guard must not block a design that is fine.
            m = fit_gravity(PoissonGravity(), yg, Xg, o, d;
                            fe = BilateralFE(), maxiter = 300, tol = 1e-12)
            @test m.converged
            @test isapprox(m.coef[1], -0.9; atol = 0.05)
            @test isapprox(m.coef[2], 0.4; atol = 0.05)

            # A duplicated regressor used to reach LAPACK and throw
            # SingularException from inside the solve.
            @test_throws ArgumentError fit_gravity(
                PoissonGravity(), yg, hcat(logd, logd), o, d; fe = BilateralFE())

            # The quiet one, and the reason the per-column check was not
            # enough: each column survives demeaning on its own, but their
            # sum is a pure origin-level term that the fixed effects absorb.
            # Before the guard this returned coefficients of order 1e12.
            u = Float64[0.5 + ((3i) % 13) / 7 for i in 1:N]
            orig_level = Float64[o[n] for n in 1:N]
            pair = hcat(u, orig_level .- u)
            @test maximum(abs, pair[:, 1]) > 1e-6     # neither column is
            @test maximum(abs, pair[:, 2]) > 1e-6     # absorbed on its own
            @test_throws ArgumentError fit_gravity(
                PoissonGravity(), yg, pair, o, d; fe = OriginFE())

            # The negative binomial goes through the same core, so it is
            # guarded too.
            @test_throws ArgumentError fit_gravity(
                NegBinGravity(), yg, hcat(logd, logd), o, d; fe = BilateralFE())
        end

        @testset "29d. the same guard at every other entry point" begin
            # Before this sweep, nine of these died inside LAPACK with a
            # message about a zero pivot and six returned quietly: the
            # minimum-norm solve splits the unidentified coefficient across
            # the dependent columns, so fit_spatial_error handed back 0.8833
            # twice where the truth was one coefficient of 1.77, with NaN
            # standard errors beside it.
            W = row_standardize(make_rook_weights(8))
            N = size(W, 1)
            x = Float64[sin(i / 4) for i in 1:N]
            eps = Float64[0.2 * cos(i / 3) for i in 1:N]
            y = (I - 0.4 * W) \ (1.0 .+ 2.0 .* x .+ eps)
            Xb = hcat(x, x)                       # the duplicated column
            yb = Float64.(y .> median(y))
            yo = Int.(1 .+ (y .> median(y)) .+ (y .> quantile(y, 0.75)))
            coords = hcat(repeat(1.0:8.0, inner = 8), repeat(1.0:8.0, outer = 8))
            labels = [i <= N ÷ 2 ? 1 : 2 for i in 1:N]

            @test_throws ArgumentError fit_spatial_error(W, Xb, y)
            @test_throws ArgumentError fit_slx(W, Xb, y)
            @test_throws ArgumentError fit_sac(W, W, Xb, y; method = ExactLU())
            @test_throws ArgumentError fit_sdem(W, Xb, y; method = ExactLU())
            @test_throws ArgumentError fit_spatial_regime(W, Xb, y,
                ExogenousRegimes(labels))
            @test_throws ArgumentError fit_gwr(Xb, y, coords;
                bandwidth = AdaptiveBandwidth(30))
            @test_throws ArgumentError fit_mgwr(Xb, y, coords; inference = false)
            @test_throws ArgumentError fit_spatial_probit(W, Xb, yb)
            @test_throws ArgumentError fit_bayesian_spatial_probit(W, Xb, yb;
                n_samples = 20, burnin = 5)
            @test_throws ArgumentError fit_spatial_ordered_probit(W, Xb, yo;
                n_samples = 20, burnin = 5, n_categories = 3)
            @test_throws ArgumentError fit_esf(Xb, y, W; family = :gaussian,
                max_eigenvectors = 5)
            @test_throws ArgumentError fit_re_esf(Xb, y, W)

            # Panels. The design that gets solved is the within-transformed
            # one, so that is what the guard has to look at.
            L = 5; Np = L * L; Tp = 4; NT = Np * Tp
            Wp = row_standardize(make_rook_weights(L))
            xp = Float64[sin(i / 5) for i in 1:NT]
            Sp = lu(sparse(1.0I, Np, Np) - 0.3 * Wp)
            mu = Float64[0.4 * cos(i) for i in 1:Np]
            yp = zeros(NT)
            for t in 1:Tp
                idx = ((t - 1) * Np + 1):(t * Np)
                yp[idx] = Sp \ (mu .+ 1.2 .* xp[idx] .+
                                Float64[0.1 * sin(i + t) for i in 1:Np])
            end
            Xpb = hcat(xp, xp)

            @test_throws ArgumentError fit_spatial_panel(SAR(), IndivFE(), Wp,
                Xpb, yp, Np, Tp)
            @test_throws ArgumentError fit_spatial_error(IndivFE(), Wp, Xpb,
                yp, Np, Tp)
            @test_throws ArgumentError fit_dynamic_spatial_panel(SAR(),
                IndivFE(), Wp, Xpb, yp, Np, Tp)
            en = Float64[0.3 * cos(i / 2) for i in 1:NT]
            z1 = Float64[sin(i / 7) for i in 1:NT]
            z2 = Float64[cos(i / 11) for i in 1:NT]
            @test_throws ArgumentError fit_gs2sls_panel(IndivFE(), Wp, Xpb,
                reshape(en, NT, 1), hcat(z1, z2), yp, Np, Tp)
            # Duplicated instruments are unidentified too, and they fail in a
            # different place: inv(H'H), before the structural solve.
            @test_throws ArgumentError fit_gs2sls_panel(IndivFE(), Wp,
                reshape(xp, NT, 1), reshape(en, NT, 1), hcat(z1, z1), yp, Np, Tp)

            # Not a duplicate at all: a regressor that is constant within
            # each unit, which individual effects sweep out entirely. This
            # one returned a coefficient before -- the column the caller
            # passed has plenty of variation, it is only the transformed
            # design that is short a rank.
            xinv = repeat(Float64[0.7 * sin(i) for i in 1:Np], Tp)
            @test_throws ArgumentError fit_spatial_panel(SAR(), IndivFE(), Wp,
                hcat(xp, xinv), yp, Np, Tp)
            # and it is fine without those fixed effects, which is the proof
            # that the guard is reading the transform and not the column.
            @test fit_spatial_panel(SAR(), NoFE(), Wp, hcat(xp, xinv), yp,
                                    Np, Tp) isa SpatialPanelModel

            # Dynamic GMM carries the lagged response beside the covariates.
            Ng, Tg = 25, 6
            Xg1 = Float64[sin(i / 3 + t) for i in 1:Ng, t in 1:Tg]
            Xten = zeros(Ng, Tg, 2)
            Xten[:, :, 1] = Xg1
            Xten[:, :, 2] = Xg1
            Yg = zeros(Ng, Tg)
            Yg[:, 1] = Float64[0.5 * cos(i) for i in 1:Ng]
            for t in 2:Tg
                Yg[:, t] = 0.4 .* Yg[:, t - 1] .+ 0.8 .* Xg1[:, t] .+
                           Float64[0.05 * sin(i * t) for i in 1:Ng]
            end
            @test_throws ArgumentError fit_dynamic_gmm(SystemGMM(), Yg, Xten;
                                                       max_lag = 3)
        end
    end

    @testset "30. Bayesian specification comparison" begin
        # Deterministic data throughout, since `randn(::MersenneTwister)` is
        # not promised to give the same numbers across Julia releases and
        # this testset asserts which specification wins. The disturbances
        # need their own small generator rather than a smooth function of
        # the index: on a lattice, `sin(i)` is spatially autocorrelated --
        # consecutive indices are neighbours -- so a design whose errors are
        # built that way is not the independent-error case it looks like,
        # and the specification comparison correctly refuses to call it one.
        # An LCG with Box-Muller gives a stream that is both independent of
        # the lattice and identical on every version.
        function det_normals(n::Int; seed::UInt64 = 0x2545f4914f6cdd1d)
            s = seed
            nxt = () -> (s = s * 0x5851f42d4c957f2d + 0x14057b7ef767814f;
                         Float64((s >> 11) & 0x001fffffffffffff) / 9007199254740992.0)
            out = Vector{Float64}(undef, n); i = 1
            while i <= n
                u1 = max(nxt(), 1e-12); u2 = nxt()
                r = sqrt(-2 * log(u1)); th = 2pi * u2
                out[i] = r * cos(th); i += 1
                i <= n && (out[i] = r * sin(th); i += 1)
            end
            return out
        end

        W5 = row_standardize(make_rook_weights(5))
        N5 = size(W5, 1)
        X5 = hcat(Float64[sin(i / 2) for i in 1:N5],
                  Float64[cos(i / 3) for i in 1:N5])
        e5 = 0.4 .* det_normals(N5)
        y5 = (I - 0.5 * W5) \ (1.0 .+ X5 * [1.5, -0.8] .+ (W5 * X5) * [0.7, 0.0] .+ e5)

        # The same construction on a 16 x 16 lattice, where the posterior
        # over the spatial parameter is sharp and interior rather than
        # spread across the whole prior support.
        W16 = row_standardize(make_rook_weights(16))
        N16 = size(W16, 1)
        # Covariates from the same stream: a smooth function of the index
        # would make `W*X` nearly collinear with `X`, which is a hard
        # Durbin design rather than the ordinary one this is meant to be.
        X16 = hcat(det_normals(N16; seed = 0xd1b54a32d192ed03),
                   det_normals(N16; seed = 0x27bb2ee687b0b0fd))
        e16 = 0.5 .* det_normals(N16; seed = 0x9e3779b97f4a7c15)
        mu16 = X16 * [1.0, -0.8] .+ (W16 * X16) * [0.7, 0.0]
        y16_sdm = (I - 0.6 * W16) \ (mu16 .+ e16)

        @testset "30a. the marginal against an exact closed form" begin
            # With k = 3 columns and N = 25 observations the gamma term is
            # Gamma(11) = 10!, so the whole constant can be written out
            # exactly and compared against the implementation's Lanczos
            # logarithm and its QR determinant. This is the part that is
            # easy to get wrong and impossible to notice: dropping
            # |Z'Z|^(-1/2) or Gamma((n-k)/2) leaves a quantity that still
            # ranks two specifications of equal width correctly, and
            # mis-ranks every comparison between designs of different width.
            Z = hcat(ones(N5), X5)
            m = N5 - size(Z, 2)
            @test m == 22
            S = sum(abs2, y5 .- Z * (Z \ y5))
            ref = -0.5 * m * log(pi) - 0.5 * log(det(Z' * Z)) +
                  log(factorial(big(10))) - 0.5 * m * log(S)
            @test log_marginal_likelihood(:ols, y5, X5, W5) ≈ Float64(ref) atol = 1e-10

            # And the Durbin design, where |Z'Z| is what penalises the two
            # extra columns: k = 5, so the gamma term is Gamma(10) = 9!.
            Zd = hcat(ones(N5), X5, W5 * X5)
            md = N5 - size(Zd, 2)
            Sd = sum(abs2, y5 .- Zd * (Zd \ y5))
            refd = -0.5 * md * log(pi) - 0.5 * log(det(Zd' * Zd)) +
                   log(factorial(big(9))) - 0.5 * md * log(Sd)
            @test log_marginal_likelihood(:slx, y5, X5, W5) ≈ Float64(refd) atol = 1e-10
        end

        @testset "30b. the error family against an independent integral" begin
            # An independently written :sem marginal: dense log|I - lambda W|
            # by `logabsdet`, |Z'Z| by `det` of the filtered cross-product,
            # and the trapezoid written out longhand. The point of the check
            # is the determinant term, which moves with lambda for an error
            # model and is constant for a lag model -- dropping it is a
            # mistake that only shows up when error specifications are
            # compared against lag ones.
            Z = hcat(ones(N5), X5)
            k = size(Z, 2); m = N5 - k
            ng = 21; a, b = -0.9, 0.9
            grid = range(a, b; length = ng); h = step(grid)
            Wd = Matrix(W5)
            l = Float64[]
            for lam in grid
                F = I - lam * Wd
                Zf = F * Z; yf = F * y5
                Sf = sum(abs2, yf .- Zf * (Zf \ yf))
                push!(l, first(logabsdet(F)) - 0.5 * m * log(pi) -
                         0.5 * log(det(Zf' * Zf)) + log(factorial(big(10)) |> Float64) -
                         0.5 * m * log(Sf) - log(b - a))
            end
            mx = maximum(l)
            w = [i == 1 || i == ng ? 0.5 : 1.0 for i in 1:ng]
            ref = mx + log(sum(w .* exp.(l .- mx)) * h)
            @test log_marginal_likelihood(:sem, y5, X5, W5;
                rho_bounds = (-0.9, 0.9), n_grid = ng) ≈ ref atol = 1e-9
        end

        @testset "30c. the no-parameter specifications are the limit" begin
            # :ols and :slx are the lambda = 0 members of the families they
            # belong to, so integrating over a vanishing interval around
            # zero has to reproduce their closed forms exactly. This pins
            # the quadrature's normalisation against the analytic branch:
            # a prior that did not integrate to one over `rho_bounds` would
            # shift one and not the other.
            for (flat, family) in ((:ols, :sar), (:ols, :sem),
                                   (:slx, :sdm), (:slx, :sdem))
                @test log_marginal_likelihood(flat, y5, X5, W5) ≈
                      log_marginal_likelihood(family, y5, X5, W5;
                          rho_bounds = (-1e-8, 1e-8), n_grid = 4) atol = 1e-8
            end
        end

        @testset "30d. quadrature and log-determinant settings" begin
            # Where the integrand rises and falls away to nothing inside the
            # interval, the trapezoid rule converges geometrically rather
            # than as O(h^2), because every Euler-Maclaurin boundary term
            # vanishes. The two lag specifications on this sample are that
            # case -- the integrand is 39 and 66 logs below its peak at the
            # ends of the support -- and 100 nodes is already at machine
            # precision.
            for s in (:sar, :sdm)
                v = [log_marginal_likelihood(s, y16_sdm, X16, W16; n_grid = g)
                     for g in (100, 200, 800)]
                @test maximum(abs.(v .- v[3])) < 1e-9
            end
            # The decay is a property of the sample, not of the rule, and it
            # fails in exactly the case worth knowing about: an error
            # specification fitted to lag data wants its parameter at the
            # edge of the stationary interval, so the integrand is only 8
            # logs down at rho = 0.995, the boundary term survives, and the
            # rate falls back to O(h^2). The same thing happens to every
            # specification at N = 25, where the posterior is wide enough to
            # be truncated by the prior's support. Both are harmless -- the
            # absolute error at the default 200 nodes stays below 1e-4,
            # three orders under the gaps that separate specifications --
            # but refining has to reduce it, which a quadrature that had
            # stopped depending on its grid would not do.
            for (s, y, X, W, g3) in ((:sem, y16_sdm, X16, W16, 800),
                                     (:sdem, y16_sdm, X16, W16, 800),
                                     (:sar, y5, X5, W5, 6400),
                                     (:sdm, y5, X5, W5, 6400),
                                     (:sem, y5, X5, W5, 6400),
                                     (:sdem, y5, X5, W5, 6400))
                v = [log_marginal_likelihood(s, y, X, W; n_grid = ng)
                     for ng in (100, 200, g3)]
                @test abs(v[2] - v[3]) < abs(v[1] - v[3]) < 1e-3
                @test 2.5 < (v[1] - v[3]) / (v[2] - v[3]) < 5.0
            end
            # Chebyshev log-determinants with deterministic traces are an
            # approximation to log|I - rho W|, so the marginal inherits that
            # error -- about 1e-5 at order 16 here -- and nothing worse.
            for s in (:sar, :sdm)
                @test log_marginal_likelihood(s, y5, X5, W5; method = ExactLU()) ≈
                      log_marginal_likelihood(s, y5, X5, W5;
                          method = ChebyshevApprox(order = 16, n_probes = 0)) atol = 1e-4
            end
        end

        @testset "30e. posterior probabilities" begin
            r = bayes_spec_probs(y5, X5, [W5])
            @test sum(r.prob) ≈ 1.0
            @test sum(r.prob_spec) ≈ 1.0
            @test sum(r.prob_w) ≈ 1.0
            @test r.prob_w[1] ≈ 1.0
            @test r.best[1] == r.specs[argmax(r.prob_spec)]
            @test r.prob_global + r.prob_local +
                  r.prob_spec[findfirst(==(:ols), r.specs)] ≈ 1.0
            # The probabilities are the normalised marginals, nothing else.
            lm = [log_marginal_likelihood(s, y5, X5, W5) for s in r.specs]
            p = exp.(lm .- maximum(lm)); p ./= sum(p)
            @test r.prob_spec ≈ p atol = 1e-10
            @test vec(r.log_marginal) ≈ lm atol = 1e-10

            # Two weight matrices: the margins have to be the row and column
            # sums of the same table, and the winner its largest cell.
            W5b = row_standardize(make_queen_weights(5))
            r2 = bayes_spec_probs(y5, X5, [W5, W5b]; w_names = ["rook", "queen"])
            @test size(r2.prob) == (6, 2)
            @test sum(r2.prob) ≈ 1.0
            @test r2.prob_spec ≈ vec(sum(r2.prob; dims = 2))
            @test r2.prob_w ≈ vec(sum(r2.prob; dims = 1))
            idx = argmax(r2.prob)
            @test r2.best == (r2.specs[idx[1]], r2.w_names[idx[2]])
            @test occursin("P(global)", sprint(show, r2))
        end

        @testset "30f. the generating specification is recovered" begin
            # The two structures that are actually hard to tell apart:
            # global spillovers through y against local ones through the
            # disturbance, both carrying the same Durbin terms. One draw
            # proves nothing on its own -- `docs/AUDIT.md` records the
            # selection rates over repeated samples -- but it pins the
            # winner on a sample that cannot change.
            r_sdm = bayes_spec_probs(y16_sdm, X16, [W16])
            @test r_sdm.best[1] == :sdm
            @test r_sdm.prob_spec[findfirst(==(:sdm), r_sdm.specs)] > 0.9
            @test r_sdm.prob_global > 0.9

            y_sdem = mu16 .+ ((I - 0.6 * W16) \ e16)
            r_sdem = bayes_spec_probs(y_sdem, X16, [W16])
            @test r_sdem.best[1] == :sdem
            @test r_sdem.prob_spec[findfirst(==(:sdem), r_sdem.specs)] > 0.9
            @test r_sdem.prob_local > 0.9

            # Independent errors and no feedback: the SLX specification,
            # which both Durbin alternatives nest at their parameter's zero.
            r_slx = bayes_spec_probs(mu16 .+ e16, X16, [W16])
            @test r_slx.best[1] == :slx

            # And no spillovers at all, where the Durbin block is noise.
            r_ols = bayes_spec_probs(X16 * [1.0, -0.8] .+ e16, X16, [W16])
            @test r_ols.best[1] == :ols
        end

        @testset "30g. the prior on the spatial parameter does not wash out" begin
            # A marginal likelihood is not a likelihood ratio: the prior is
            # over a parameter the flat specifications do not have, so
            # widening its support penalises the spatial ones and leaves
            # :ols and :slx untouched. The size of the penalty is exactly
            # the width ratio whenever the posterior mass sits inside both
            # intervals, which for `y5` -- generated at rho = 0.5 -- it
            # does: halving the support must add log 2 and nothing else.
            wide = log_marginal_likelihood(:sar, y5, X5, W5;
                                           rho_bounds = (-0.995, 0.995))
            half = log_marginal_likelihood(:sar, y5, X5, W5;
                                           rho_bounds = (0.0, 0.995))
            # The residual is the posterior mass below zero that the half
            # interval drops, which at N = 25 is not quite nothing.
            @test half - wide ≈ log(1.99 / 0.995) atol = 1e-5
            for s in (:ols, :slx)
                @test log_marginal_likelihood(s, y5, X5, W5;
                          rho_bounds = (-0.995, 0.995)) ==
                      log_marginal_likelihood(s, y5, X5, W5;
                          rho_bounds = (0.0, 0.5))
            end
            # A Beta prior concentrates the same support differently, so it
            # has to produce a different answer -- if it did not, the prior
            # argument would not be reaching the integrand.
            bp = bayes_spec_probs(y5, X5, [W5]; prior = :beta, beta_shape = 3.0)
            un = bayes_spec_probs(y5, X5, [W5])
            @test bp.prob_spec != un.prob_spec
            @test bp.prior === :beta
        end

        @testset "30h. the SDEM roadmap" begin
            W5b = row_standardize(make_queen_weights(5))
            r = sdem_roadmap(y5, X5, [W5, W5b]; w_names = ["rook", "queen"])
            @test r.specs == [:sdem, :slx, :sem]
            @test size(r.score) == (2, 3)
            @test r.criterion === :marginal
            for i in 1:2
                row = r.score[i, :]
                o = sortperm(row; rev = true)
                @test r.best_spec[i] == r.specs[o[1]]
                @test r.margin[i] ≈ row[o[1]] - row[o[2]]
                @test r.score[i, :] ≈ [log_marginal_likelihood(s, y5, X5, [W5, W5b][i])
                                       for s in r.specs] atol = 1e-10
            end
            flat = argmax(r.score)
            @test r.best_w == r.w_names[flat[1]]
            @test r.best_overall == r.specs[flat[2]]
            @test occursin("roadmap", sprint(show, r))

            # AIC and BIC are minimised, not maximised, and the search has
            # to read them that way; BIC penalises the wider design harder,
            # so the two need not agree, but each must agree with itself.
            for c in (:aic, :bic)
                rc = sdem_roadmap(y5, X5, [W5, W5b]; criterion = c,
                                  w_names = ["rook", "queen"])
                @test rc.criterion === c
                for i in 1:2
                    o = sortperm(rc.score[i, :])
                    @test rc.best_spec[i] == rc.specs[o[1]]
                    @test rc.margin[i] ≈ rc.score[i, o[2]] - rc.score[i, o[1]]
                end
                @test rc.best_overall == rc.specs[argmin(rc.score)[2]]
            end
            # One specification leaves nothing to compare against.
            r1 = sdem_roadmap(y5, X5, [W5]; specs = [:sdem])
            @test r1.margin == [Inf]
            @test r1.best_overall === :sdem
        end

        @testset "30i. what the entry points reject" begin
            @test_throws ArgumentError log_marginal_likelihood(:sarar, y5, X5, W5)
            @test_throws ArgumentError log_marginal_likelihood(:ols, y5, X5, W5;
                                                               prior = :normal)
            @test_throws ArgumentError log_marginal_likelihood(:sar, y5, X5, W5;
                                                               rho_bounds = (0.9, -0.9))
            @test_throws ArgumentError bayes_spec_probs(y5, X5, [W5]; specs = Symbol[])
            @test_throws ArgumentError bayes_spec_probs(y5, X5,
                                           SparseMatrixCSC{Float64, Int}[])
            @test_throws ArgumentError bayes_spec_probs(y5, X5, [W5];
                                                        w_names = ["a", "b"])
            @test_throws ArgumentError sdem_roadmap(y5, X5, [W5]; criterion = :mse)

            # The identification guard of testset 29 reaches this module too,
            # and a Durbin specification can be deficient when the plain one
            # is not: `W*X` collapses into the span of `X` whenever `W` maps
            # the covariates onto themselves.
            Xc = hcat(ones(N5) .* 2.0, Float64[sin(i / 2) for i in 1:N5])
            err = try
                log_marginal_likelihood(:ols, y5, Xc, W5); nothing
            catch e; e end
            @test err isa ArgumentError
            @test occursin("not identified", err.msg)
            @test occursin("log_marginal_likelihood:", err.msg)

            # n > k, or there is no marginal to speak of.
            W2 = row_standardize(make_rook_weights(2))
            X2 = hcat(Float64[sin(i) for i in 1:4], Float64[cos(i) for i in 1:4],
                      Float64[sin(2i) for i in 1:4])
            @test_throws ArgumentError log_marginal_likelihood(:ols,
                Float64[1.0, 2.0, 3.0, 4.0], X2, W2)
        end
    end

    @testset "31. Spatio-temporal principal components" begin
        SF = SpatialEconometrics.SpatialFiltering
        SD = SpatialEconometrics.SpatialDiagnostics

        # The reference's own grid: 2002-2018 observed at the midpoints of
        # seventeen unit intervals, expanded on [0, 17] in five Fourier
        # functions per feature.
        tsq = collect(0.5:1.0:16.5)
        IV = (0.0, 17.0)
        Wq = row_standardize(make_rook_weights(6))
        Nq = size(Wq, 1)

        # Deterministic data, for the reason recorded in testset 30: a
        # stand-in for a draw has to be independent of the lattice.
        function det_normals31(n::Int; seed::UInt64 = 0x853c49e6748fea9b)
            s = seed
            nxt = () -> (s = s * 0x5851f42d4c957f2d + 0x14057b7ef767814f;
                         Float64((s >> 11) & 0x001fffffffffffff) / 9007199254740992.0)
            out = Vector{Float64}(undef, n); i = 1
            while i <= n
                u1 = max(nxt(), 1e-12); u2 = nxt()
                r = sqrt(-2 * log(u1)); th = 2pi * u2
                out[i] = r * cos(th); i += 1
                i <= n && (out[i] = r * sin(th); i += 1)
            end
            return out
        end

        pq, ntq = 3, length(tsq)
        noise = det_normals31(Nq * pq * ntq)
        Xq = Array{Float64}(undef, Nq, pq, ntq)
        let c = 0
            for i in 1:Nq, j in 1:pq, t in 1:ntq
                c += 1
                Xq[i, j, t] = noise[c] * 0.4 + sin(tsq[t] / 3) * (i % 5) / 3 +
                              cos(tsq[t] / 5) * j / 2
            end
        end

        @testset "31a. the Fourier basis is orthonormal" begin
            # The construction rests on the basis being orthonormal with
            # respect to the integral over the interval: that is what makes
            # the functional inner product <u, X> equal the dot product of
            # coefficient vectors, and so what makes gamma'S gamma the
            # variance of the functional principal component.
            ng = 100001
            tg = collect(range(IV[1], IV[2]; length = ng))
            P = fourier_basis(tg, 5; interval = IV)
            h = (IV[2] - IV[1]) / (ng - 1)
            w = [k == 1 || k == ng ? 0.5 : 1.0 for k in 1:ng]
            G = [sum(w .* P[:, i] .* P[:, j]) * h for i in 1:5, j in 1:5]
            @test maximum(abs.(G - I)) < 1e-12

            # Least squares in that basis is exact on a function already in
            # its span, which is the identity the expansion step needs.
            Pt = fourier_basis(tsq, 5; interval = IV)
            ctrue = [1.3, -0.7, 2.1, 0.4, -1.6]
            @test Pt \ (Pt * ctrue) ≈ ctrue atol = 1e-12

            # Padding the interval is not cosmetic. A Fourier basis is
            # periodic, so expanding on exactly [t_1, t_T] forces the fit to
            # take the same value at both ends; the default pads by half a
            # step, which is what [0, 17] is for this grid.
            # For this grid the default padding gives exactly the [0, 17]
            # the reference uses, which is the point of the default.
            @test fourier_basis(tsq, 5) ≈ fourier_basis(tsq, 5; interval = IV)
            # Unpadded, the first and last rows coincide -- the fit is
            # forced to return to where it started, whatever the data do.
            Pn = fourier_basis(tsq, 5; interval = (first(tsq), last(tsq)))
            @test Pn[1, :] ≈ Pn[end, :]
            Pp = fourier_basis(tsq, 5; interval = IV)
            @test !isapprox(Pp[1, :], Pp[end, :])
            @test_throws ArgumentError fourier_basis(tsq, 0)
            @test_throws ArgumentError fourier_basis([1.0], 1)
            @test_throws ArgumentError fourier_basis([3.0, 1.0, 2.0], 2)
        end

        @testset "31b. y'B*y is Geary's numerator" begin
            # Expanding sum_ij w_ij (y_i - y_j)^2 into a quadratic form is
            # the step that turns the ratio into an eigenproblem, so it is
            # checked against the raw double sum rather than assumed.
            y = det_normals31(Nq; seed = 0x9e3779b97f4a7c15)
            Wd = Matrix(Wq)
            raw = sum(Wd[i, j] * (y[i] - y[j])^2 for i in 1:Nq, j in 1:Nq)
            G = SF._geary_matrix(Wq)
            @test dot(y, G * y) ≈ raw atol = 1e-10
            @test dot(y, ((G + G') ./ 2) * y) ≈ raw atol = 1e-10
            # and the ratio it builds is the package's own Geary's c
            @test (Nq - 1) / (2 * sum(Wq)) * dot(y, G * y) / sum(abs2, y .- mean(y)) ≈
                  SD.geary_c(y, Wq).C_stat atol = 1e-12
        end

        @testset "31c. every eigenvalue is its own objective" begin
            # The claim the whole method rests on: the eigenvalue of E is
            # not a proxy for the quantity being maximised, it IS that
            # quantity -- the functional principal component's variance
            # times the spatial autocorrelation of the scores it produces.
            # Both factors are recomputed here from other modules, so this
            # fails if either the scaling of E or the centring of A is
            # wrong.
            for meas in (:geary, :moran)
                m = stpca(Xq, Wq; measure = meas, n_basis = 5, times = tsq,
                          interval = IV)
                @test length(m.values) == pq * 5
                @test issorted(m.values; rev = true)
                @test m.vectors' * m.vectors ≈ I atol = 1e-10
                for k in 1:length(m.values)
                    y = m.scores[:, k]
                    @test mean(y) ≈ 0.0 atol = 1e-10
                    ac = meas === :geary ? SD.geary_c(y, Wq).C_stat :
                                           moran_coefficient(y, Wq)
                    var = sum(abs2, y) / Nq
                    @test m.autocorr[k] ≈ ac atol = 1e-10
                    @test m.variance[k] ≈ var atol = 1e-10
                    @test m.values[k] ≈ var * ac atol = 1e-9
                end
                @test sum(abs, m.proportion) ≈ 1.0
            end
        end

        @testset "31d. the sign structure of the two measures" begin
            mg = stpca(Xq, Wq; measure = :geary, times = tsq, interval = IV)
            mm = stpca(Xq, Wq; measure = :moran, times = tsq, interval = IV)
            # E is a variance times a Geary ratio, and neither factor can be
            # negative, so E is non-negative definite. This is a property of
            # the construction, not of the sample.
            @test minimum(mg.values) > -1e-10
            @test all(>=(0.0), mg.autocorr)
            # Moran's I is a cross-product and can be negative, which is the
            # global/local split the eigenvalue signs carry. On data with
            # both structures present both signs appear.
            @test minimum(mm.values) < 0.0
            @test maximum(mm.values) > 0.0
            @test sign.(mm.values) == sign.(mm.autocorr)
        end

        @testset "31e. scale and rank behaviour" begin
            m1 = stpca(Xq, Wq; times = tsq, interval = IV)
            m3 = stpca(3.0 .* Xq, Wq; times = tsq, interval = IV)
            # Variance scales with c^2 and the ratios are scale free, so the
            # whole spectrum scales by c^2 and the directions do not move.
            @test m3.values ≈ 9 .* m1.values atol = 1e-8 * maximum(m1.values)
            @test abs.(m3.vectors' * m1.vectors) ≈ I atol = 1e-6
            @test m3.autocorr ≈ m1.autocorr atol = 1e-8

            # Adding a constant to a unit's series moves only the basis
            # coefficient of the constant function, and the coefficients are
            # centred, so a shift common to every unit changes nothing.
            @test stpca(Xq .+ 7.0, Wq; times = tsq, interval = IV).values ≈
                  m1.values atol = 1e-8 * maximum(m1.values)

            # B = p * n_basis directions, but at most n - 1 can carry
            # variance when there are fewer units than coefficients.
            Xsmall = Xq[1:4, :, :]
            Wsmall = row_standardize(sparse(Float64[0 1 0 0; 1 0 1 0;
                                                    0 1 0 1; 0 0 1 0]))
            ms = stpca(Xsmall, Wsmall; times = tsq, interval = IV)
            @test length(ms.values) == pq * 5
            @test count(>(1e-8 * maximum(ms.values)), ms.values) <= 3
        end

        @testset "31f. the single-feature method and feature shares" begin
            Xone = Xq[:, 1, :]
            m2d = stpca(Xone, Wq; times = tsq, interval = IV)
            m3d = stpca(reshape(Xone, Nq, 1, ntq), Wq; times = tsq, interval = IV)
            @test m2d.values ≈ m3d.values
            @test m2d.n_features == 1
            # One feature takes the whole of every component.
            @test feature_contributions(m2d, 1) ≈ [1.0]

            m = stpca(Xq, Wq; times = tsq, interval = IV, varnames = ["a", "b", "c"])
            for k in (1, 2, length(m.values))
                fc = feature_contributions(m, k)
                @test length(fc) == pq
                @test sum(fc) ≈ 1.0
                @test all(>=(0.0), fc)
            end
            # The shares are areas under |u_kj|, so refining the grid moves
            # them only by the trapezoid's own error.
            # |u_kj| has a corner wherever u_kj crosses zero, so the
            # trapezoid converges more slowly than it would on a smooth
            # integrand; the shares are stable to about 1e-5, which is
            # four orders finer than any difference between features.
            @test feature_contributions(m, 1; n_grid = 4096) ≈
                  feature_contributions(m, 1; n_grid = 512) atol = 1e-4
            # A feature that is identically zero contributes nothing.
            Xz = copy(Xq); Xz[:, 3, :] .= 0.0
            mz = stpca(Xz, Wq; times = tsq, interval = IV)
            @test feature_contributions(mz, 1)[3] ≈ 0.0 atol = 1e-10

            # The weighting function is the direction read back as a
            # function of time, so its coefficients are the eigenvector's.
            U = component_weight(m, 1, tsq)
            @test size(U) == (ntq, pq)
            Pt = fourier_basis(tsq, 5; interval = IV)
            @test U[:, 2] ≈ Pt * m.vectors[6:10, 1] atol = 1e-12
            @test m.varnames == ["a", "b", "c"]
        end

        @testset "31g. planted structure lands where it should" begin
            # One draw pins what the Monte Carlo in docs/AUDIT.md measures
            # as a rate: a smooth field is the leading :moran component and
            # never the leading :geary one.
            se = spatial_eigenvectors(Wq)
            smooth = se.vectors[:, 1]
            @test moran_coefficient(smooth, Wq) > 0.8
            prof = sin.(2pi .* tsq ./ 17)
            Xp = copy(Xq)
            for i in 1:Nq
                Xp[i, 1, :] .+= 3.0 * smooth[i] .* prof
            end
            mm = stpca(Xp, Wq; measure = :moran, times = tsq, interval = IV)
            mg = stpca(Xp, Wq; measure = :geary, times = tsq, interval = IV)
            cm = [abs(cor(mm.scores[:, k], smooth)) for k in 1:length(mm.values)]
            cg = [abs(cor(mg.scores[:, k], smooth)) for k in 1:length(mg.values)]
            @test argmax(cm) == 1
            # The planted field is spread across the spectrum rather than
            # captured whole -- the Monte Carlo median of this correlation
            # is about 0.56 at this N -- so the claim is that the leading
            # component is where most of it goes, not that it takes it all.
            @test cm[1] > 0.5
            @test cm[1] > 1.5 * maximum(cm[2:end])
            @test argmax(cg) != 1

            # The sign groups: :moran splits global from local, :geary
            # cannot, which is the construction and not the sample.
            gm = component_groups(mm)
            @test !isempty(gm.positive) && !isempty(gm.negative)
            @test all(>(0.0), mm.values[gm.positive])
            @test all(<(0.0), mm.values[gm.negative])
            @test issorted(mm.values[gm.positive]; rev = true)
            @test issorted(abs.(mm.values[gm.negative]); rev = true)
            @test 1 in gm.positive
            @test isempty(component_groups(mg).negative)
            # and the planted feature is the one the component reads
            @test argmax(feature_contributions(mm, 1)) == 1
        end

        @testset "31h. what stpca rejects" begin
            @test_throws ArgumentError stpca(Xq, Wq; measure = :getis)
            @test_throws DimensionMismatch stpca(Xq, row_standardize(make_rook_weights(5)))
            @test_throws DimensionMismatch stpca(Xq, Wq; times = tsq[1:5])
            # more basis functions than observation times is not identified
            @test_throws ArgumentError stpca(Xq, Wq; n_basis = ntq + 1, times = tsq)
            Xbad = copy(Xq); Xbad[1, 1, 1] = NaN
            @test_throws ArgumentError stpca(Xbad, Wq; times = tsq)
            @test_throws ArgumentError stpca(Xq, Wq; times = tsq, varnames = ["a"])
            @test_throws ArgumentError stpca(Xq, spzeros(Nq, Nq); times = tsq)
            m = stpca(Xq, Wq; times = tsq, interval = IV)
            @test_throws ArgumentError component_weight(m, 0, tsq)
            @test_throws ArgumentError component_weight(m, 999, tsq)
            @test_throws ArgumentError feature_contributions(m, 1; n_grid = 1)
            @test occursin("Geary", sprint(show, m))
            @test occursin("Moran", sprint(show,
                stpca(Xq, Wq; measure = :moran, times = tsq, interval = IV)))
        end
    end

    @testset "32. Performance invariants" begin
        # Correctness tests do not notice an accidental quadratic: the old
        # `row_standardize!` rescanned every nonzero in the matrix once per
        # row, so it was O(N * nnz) rather than O(nnz), and every assertion
        # about its output passed throughout. What catches that is a bound
        # on how long it may take.

        @testset "32a. row_standardize is linear in the nonzeros" begin
            for side in (20, 40)
                W = SparseMatrixCSC{Float64, Int}(make_rook_weights(side))
                Wr = row_standardize(W)
                rs = vec(sum(Wr; dims = 2))
                @test all(x -> isapprox(x, 1.0; atol = 1e-12), rs)
                @test nnz(Wr) == nnz(W)
                @test W == SparseMatrixCSC{Float64, Int}(make_rook_weights(side))
            end

            # A row with no neighbours stays zero rather than dividing by it.
            Wgap = sparse([1, 1, 2], [2, 3, 1], [1.0, 2.0, 3.0], 4, 4)
            Wg = row_standardize(Wgap)
            @test vec(sum(Wg; dims = 2)) ≈ [1.0, 1.0, 0.0, 0.0]

            # The row sum is of absolute values, so a signed W normalises by
            # magnitude -- which is what the original did, and what the
            # linear rewrite has to keep doing.
            Wneg = sparse([1, 2, 2], [2, 1, 3], [-1.0, 2.0, -3.0], 3, 3)
            @test nonzeros(row_standardize(Wneg)) ≈ [2 / 5, -1.0, -3 / 5]

            # The timing bound. At N = 8100 the quadratic version took 310 ms
            # and the linear one takes 0.19 ms, so 50 ms sits 260x above the
            # new cost and 6x below the old. That margin is what makes a
            # wall-clock assertion safe to run on a shared CI runner; a
            # tighter bound would be measuring the runner, not the code.
            Wbig = SparseMatrixCSC{Float64, Int}(make_rook_weights(90))
            @test size(Wbig, 1) == 8100
            row_standardize(Wbig)                      # compile first
            t = @elapsed row_standardize(Wbig)
            @test t < 0.05
        end

        @testset "32b. the probes-contiguous trace kernel" begin
            SU = SpatialEconometrics.SpatialUtils

            # The kernel is a hand-written gather, so it gets checked against
            # the thing it replaced rather than only against itself: the same
            # probe block, multiplied by `W` with `mul!`, and the same traces
            # read off by `dot`. Agreement has to be at roundoff and no
            # better, since the two sum the outer loop in different orders.
            for side in (6, 11), p in (1, 4, 37)
                W = SparseMatrixCSC{Float64, Int}(make_rook_weights(side))
                N = size(W, 1)
                mp = 12
                got = SU.precompute_neumann_traces(W; max_power = mp,
                                                   n_probes = p,
                                                   rng = MersenneTwister(4))
                # Reference: draw the probes column-major exactly as the
                # documented contract says, then walk the powers densely.
                V0 = Matrix{Float64}(undef, N, p)
                rand!(MersenneTwister(4), V0, (-1.0, 1.0))
                want = zeros(mp + 1)
                want[1] = 1.0
                want[2] = tr(W) / N
                Vc = W * V0
                for m in 2:mp
                    Vc = W * Vc
                    want[m + 1] = dot(V0, Vc) / (N * p)
                end
                @test got ≈ want rtol = 1e-10
                @test got[1] == 1.0
                @test got[2] == tr(W) / N
            end

            # The probe set is part of the contract: a caller who passes a
            # seed gets the column-major Rademacher draw, not whatever
            # layout happens to be fastest inside. If that ever changes,
            # every reported impact and standard error moves by a Monte
            # Carlo error, which is exactly the surprise this pins down.
            V = Matrix{Float64}(undef, 9, 5)
            rand!(MersenneTwister(13), V, (-1.0, 1.0))
            @test SU._rademacher_probes(Float64, 9, 5, MersenneTwister(13)) ==
                  permutedims(V)
            @test all(x -> abs(x) == 1.0, V)

            # Same seed, same answer, bit for bit -- the reason the kernel
            # reduces over a per-row array in index order instead of letting
            # threads accumulate into one scalar.
            W2 = SparseMatrixCSC{Float64, Int}(make_rook_weights(14))
            a = SU.precompute_neumann_traces(W2; max_power = 20, n_probes = 8,
                                             rng = MersenneTwister(1))
            b = SU.precompute_neumann_traces(W2; max_power = 20, n_probes = 8,
                                             rng = MersenneTwister(1))
            @test all(a .=== b)

            # An asymmetric W, checked the same way. Worth stating why this
            # case needs the reference comparison and not a cleverer test:
            # tr((Wᵀ)ᵐ) = tr(Wᵐ) for every m, so a gather that walked the
            # matrix the wrong way round would still be an unbiased
            # estimator of the right traces and no test of the trace values
            # alone could see it. What pins the direction is agreeing with
            # `mul!(V, W, ...)` realisation for realisation.
            Wa = SparseMatrixCSC{Float64, Int}(
                sparse([1, 2, 3, 1, 4, 4], [2, 3, 1, 3, 1, 2],
                       [0.5, 0.25, 0.75, 0.5, 0.3, 0.7], 4, 4))
            @test !issymmetric(Matrix(Wa))
            p = 16
            got = SU.precompute_neumann_traces(Wa; max_power = 9, n_probes = p,
                                               rng = MersenneTwister(6))
            V0 = Matrix{Float64}(undef, 4, p)
            rand!(MersenneTwister(6), V0, (-1.0, 1.0))
            want = zeros(10)
            want[1] = 1.0
            want[2] = tr(Wa) / 4
            Vc = Wa * V0
            for m in 2:9
                Vc = Wa * Vc
                want[m + 1] = dot(V0, Vc) / (4 * p)
            end
            @test got ≈ want rtol = 1e-10
            # And it would not have agreed with the transpose: same probes,
            # a different realisation, even though the expectation is equal.
            Vt = permutedims(Matrix(Wa)) * V0
            Vt = permutedims(Matrix(Wa)) * Vt
            @test !isapprox(dot(V0, Vt) / (4 * p), got[4]; rtol = 1e-6)
        end

        @testset "32c. threaded GWR writes disjoint rows" begin
            # The three local loops in `fit_gwr` were threaded. Each writes
            # only row `i` of its output, and the reductions that follow --
            # `tr(S)` and `RSS` -- are serial passes afterwards, so the fit
            # is the same whatever the thread count. A test cannot vary the
            # thread count, so what it checks instead is the property that
            # makes that true: the threaded result equals the same arithmetic
            # done serially, here by rebuilding one local regression by hand.
            side = 12
            N = side^2
            coords = hcat(repeat(1:side, side), repeat(1:side, inner = side)) .* 1.0
            lcg = Ref(UInt64(20251030))
            function draw()
                lcg[] = 6364136223846793005 * lcg[] + 1442695040888963407
                return Float64(lcg[] >> 11) / Float64(UInt64(1) << 53) * 2 - 1
            end
            X = reshape([draw() for _ in 1:(2N)], N, 2)
            y = 1.0 .+ X * [1.0, -0.5] .+ 0.3 .* [draw() for _ in 1:N]

            m = fit_gwr(X, y, coords; bandwidth = FixedBandwidth(4.0),
                        kernel = GaussianKernel())
            @test size(m.coef_local) == (N, 3)
            @test all(isfinite, m.coef_local)
            @test all(isfinite, m.se_local)
            @test all(x -> x > 0, m.se_local)

            # Row 7 of coef_local, rebuilt serially from the definition.
            i = 7
            Xa = hcat(ones(N), X)
            # Note the truncation: `_get_weights` evaluates the kernel only
            # at the points `inrange` returns, so a Gaussian kernel is cut
            # off hard at one bandwidth rather than decaying over all N. The
            # reference has to do the same or it is weighting a different
            # neighbourhood. That cut-off is the estimator's own existing
            # choice, not something this pass introduced.
            dist = [sqrt(sum((coords[i, :] .- coords[j, :]) .^ 2)) for j in 1:N]
            w = [d <= 4.0 ? exp(-0.5 * (d / 4.0)^2) : 0.0 for d in dist]
            @test count(!iszero, w) < N
            Wh = Diagonal(sqrt.(w))
            beta = qr(Wh * Xa) \ (Wh * y)
            @test m.coef_local[i, :] ≈ beta rtol = 1e-10

            # And local R-squared, which had its two loop-invariant vectors
            # hoisted out of the threaded loop. Same number, computed once.
            tss = dot(w, (y .- mean(y)) .^ 2)
            rss = dot(w, m.residuals .^ 2)
            @test m.local_R2[i] ≈ 1 - rss / tss rtol = 1e-10

            # `_euclid` replaced `sqrt(sum((a .- b) .^ 2))`, which it has to
            # match exactly and not merely closely, since it feeds the kernel.
            GW = SpatialEconometrics.GWREstimation
            for (a, b) in ((1, 2), (5, 99), (40, 40), (1, N))
                @test GW._euclid(coords, a, b) ===
                      sqrt(sum((coords[a, :] .- coords[b, :]) .^ 2))
            end
            # Three columns too: GWR asserts two, but the helper is generic.
            c3 = [0.5 -1.25 2.0; 3.0 0.125 -4.5; -1.0 1.0 1.0]
            @test GW._euclid(c3, 1, 2) ≈ sqrt(sum((c3[1, :] .- c3[2, :]) .^ 2))
        end

        @testset "32d. knn_weights keeps k non-self neighbours" begin
            # The selection used to re-derive, per candidate, how many
            # non-self neighbours the list held -- an allocating `filter` and
            # a `count` over a fresh slice inside the inner loop. A running
            # counter decides the same thing; the rule is what the test pins.
            grid(side) = hcat(repeat(1:side, side), repeat(1:side, inner = side)) .* 1.0
            for side in (7, 20), k in (1, 3, 12)
                W = knn_weights(grid(side), k; row_standardize = false)
                N = side^2
                @test size(W) == (N, N)
                @test all(==(k), vec(sum(W .!= 0; dims = 2)))
                @test all(iszero, diag(W))
                @test nnz(W) == N * k
            end

            # Coincident points: self can be absent from its own neighbour
            # list, which is the case the old `length(filter(...)) >= k`
            # guard existed to handle. Six points stacked at the origin.
            dup = vcat(zeros(6, 2), Float64[1 0; 0 1; 2 2; -1 -1; 3 0; 0 3])
            for k in (2, 5)
                W = knn_weights(dup, k; row_standardize = false)
                @test all(==(k), vec(sum(W .!= 0; dims = 2)))
                @test all(iszero, diag(W))
            end

            # binary, symmetrize and row_standardize still compose.
            Wb = knn_weights(grid(10), 4; binary = true, row_standardize = false)
            @test all(==(1.0), nonzeros(Wb))
            Wr = knn_weights(grid(10), 4)
            @test all(x -> isapprox(x, 1.0; atol = 1e-12), vec(sum(Wr; dims = 2)))
            Wsym = knn_weights(grid(10), 4; symmetrize = true, row_standardize = false)
            @test Wsym == permutedims(Wsym)
        end

        @testset "32e. the log-determinant sweep reuses its factorisation" begin
            SMS = SpatialEconometrics.SpatialModelSelection
            W = SparseMatrixCSC{Float64, Int}(make_rook_weights(14))
            N = size(W, 1)
            Isp = sparse(1.0I, N, N)

            # The closure writes values into a fixed pattern and calls `lu!`
            # instead of building `I - rho*W` and factorising from scratch.
            # Exactness is the point of this branch, so the standard here is
            # bit-identical and not merely close.
            f = SMS._logdet_closure(W, N, ExactLU(), typemax(Int))
            for rho in (-0.995, -0.4, 0.0, 0.17, 0.5, 0.9, 0.995)
                @test f(rho) === first(logabsdet(lu(Isp - rho * W)))
            end
            # Called repeatedly and out of order: the value buffer must be
            # rewritten each time, not accumulated into.
            @test f(0.5) === f(0.5)
            @test f(0.3) === first(logabsdet(lu(Isp - 0.3 * W)))

            # The pattern helper must hold the structural union of I and W,
            # including positions where `I - W` would cancel to exactly zero.
            Wone = sparse([1, 2, 2, 3], [2, 1, 2, 3], [1.0, 0.5, 1.0, 1.0], 3, 3)
            A, iv, wv = SMS._affine_pattern(Wone, 3)
            @test nnz(A) == 5          # three diagonal, plus (1,2) and (2,1)
            nzv = nonzeros(A)
            for rho in (0.0, 0.25, -0.75)
                for t in eachindex(nzv)
                    nzv[t] = iv[t] - rho * wv[t]
                end
                @test Matrix(A) ≈ Matrix(1.0I(3) - rho * Wone)
            end
        end

        @testset "32f. the Chebyshev recursion has a memory ceiling" begin
            SU = SpatialEconometrics.SpatialUtils
            grid(side) = hcat(repeat(1:side, side), repeat(1:side, inner = side)) .* 1.0
            unbounded = typemax(Int) ÷ 4

            for (Wraw, q) in ((make_rook_weights(20), 6),
                              (make_rook_weights(20), 12),
                              (row_standardize(knn_weights(grid(14), 12)), 10))
                Ws = SparseMatrixCSC{Float64, Int}(Wraw)
                m = ChebyshevApprox(order = q, n_probes = 0)
                ref = SU.precompute_chebyshev_traces(Ws, m; max_fill_bytes = unbounded)
                # A ceiling of zero forces the blocked path from order 2 on.
                blocked = SU.precompute_chebyshev_traces(Ws, m; max_fill_bytes = 0)
                @test length(blocked) == q + 1
                @test blocked[1] == size(Ws, 1)       # tr(T_0) = tr(I) = N
                @test blocked[2] == tr(Ws)
                # Same traces to roundoff: the two sum over columns in
                # different groupings, so this is reordering rather than a
                # different quantity.
                @test blocked ≈ ref rtol = 1e-10
                # What actually matters is the log-determinant it feeds.
                for rho in (-0.9, -0.25, 0.0, 0.45, 0.9)
                    @test SU.chebyshev_logdet(rho, blocked) ≈
                          SU.chebyshev_logdet(rho, ref) atol = 1e-8
                end
                # A generous ceiling must leave the fast path untouched, bit
                # for bit: the default is not allowed to perturb anything.
                @test all(SU.precompute_chebyshev_traces(Ws, m) .=== ref)
            end

            # Block size: at least 16 columns, never more than N, otherwise
            # as many as three dense N-vectors per column allow.
            @test SU._cheb_block_size(1000, 0, Float64) == 16
            @test SU._cheb_block_size(1000, typemax(Int) ÷ 4, Float64) == 1000
            @test SU._cheb_block_size(1000, 3 * 1000 * 8 * 50, Float64) == 50

            # A block size that does not divide N must leave no column out:
            # the trace is a sum over all N, so a short final block would
            # show up as a wrong answer rather than as an error.
            W11 = SparseMatrixCSC{Float64, Int}(make_rook_weights(11))   # N = 121
            m5 = ChebyshevApprox(order = 5, n_probes = 0)
            @test SU.precompute_chebyshev_traces(W11, m5; max_fill_bytes = 0) ≈
                  SU.precompute_chebyshev_traces(W11, m5;
                                                 max_fill_bytes = unbounded) rtol = 1e-10

            # The probe branch is untouched by any of this.
            a = SU.precompute_chebyshev_traces(W11,
                    ChebyshevApprox(order = 6, n_probes = 8, rng = MersenneTwister(5));
                    max_fill_bytes = 0)
            b = SU.precompute_chebyshev_traces(W11,
                    ChebyshevApprox(order = 6, n_probes = 8, rng = MersenneTwister(5)))
            @test all(a .=== b)
        end
    end

    @testset "33. GWR compact local design" begin
        # Every local regression used to build a dense `N x K` weighted
        # design when only the `b` rows inside the kernel's support carry
        # anything. The compact form is the same arithmetic on the rows that
        # matter, so what these tests pin is the equivalence -- against the
        # dense form kept in `_get_weights` for exactly this purpose -- and
        # the properties the compact form has to have for that to hold.
        GW = SpatialEconometrics.GWREstimation

        grid(side) = hcat(repeat(1:side, side), repeat(1:side, inner = side)) .* 1.0

        # The dense arithmetic, verbatim as it stood before the change.
        function dense_fit(X, y, coords, tree, bw, kernel)
            N, K = size(X)
            coef = zeros(N, K); hat = zeros(N); res = zeros(N); fit = zeros(N)
            for i in 1:N
                w = GW._get_weights(tree, coords, i, bw, kernel)
                Ws = Diagonal(sqrt.(w))
                F = qr(Ws * X)
                b = F \ (Ws * y)
                coef[i, :] = b
                fit[i] = dot(X[i, :], b)
                res[i] = y[i] - fit[i]
                hat[i] = sum(abs2, F.R' \ (X[i, :] .* sqrt(w[i])))
            end
            return coef, res, fit, sum(hat), dot(res, res)
        end

        @testset "33a. the support is what the dense vector was" begin
            coords = grid(9)
            N = size(coords, 1)
            tree = GW._build_tree(coords)
            for bw in (FixedBandwidth(2.5), FixedBandwidth(50.0),
                       AdaptiveBandwidth(5), AdaptiveBandwidth(N - 1)),
                kern in (GaussianKernel(), BisquareKernel(), ExponentialKernel())

                sc = GW._support_scratch(Float64)
                for i in (1, 13, N)
                    n, h_i, w_self = GW._local_support!(sc, tree, coords, i, bw, kern)
                    w = GW._get_weights(tree, coords, i, bw, kern)

                    # Ascending index order is a contract, not an accident:
                    # it is what makes the compact design's row order both
                    # reproducible and equal to the dense path's.
                    @test issorted(view(sc.idx, 1:n))
                    @test allunique(view(sc.idx, 1:n))

                    # Scattering the support back must reproduce the dense
                    # vector exactly -- same values, same positions, and
                    # nothing nonzero outside the support.
                    scattered = zeros(N)
                    for t in 1:n
                        scattered[sc.idx[t]] = sc.wv[t]
                    end
                    @test scattered == w

                    # `w_self` is computed as the kernel at distance zero
                    # rather than looked up, so it has to agree bit for bit.
                    @test w_self === w[i]
                    @test h_i > 0
                end
            end
        end

        @testset "33b. compact equals dense, across kernels and bandwidths" begin
            worst = 0.0
            for side in (8, 12), kern in (GaussianKernel(), BisquareKernel(),
                                          ExponentialKernel())
                N = side^2
                coords = grid(side)
                lcg = Ref(UInt64(6103))
                function draw()
                    lcg[] = 6364136223846793005 * lcg[] + 1442695040888963407
                    return Float64(lcg[] >> 11) / Float64(UInt64(1) << 53) * 2 - 1
                end
                X = hcat(ones(N), reshape([draw() for _ in 1:(2N)], N, 2))
                y = 1.0 .+ X[:, 2:3] * [1.0, -0.5] .+ 0.3 .* [draw() for _ in 1:N]
                tree = GW._build_tree(coords)

                for bw in (FixedBandwidth(2.5), FixedBandwidth(6.0),
                           AdaptiveBandwidth(6), AdaptiveBandwidth(N - 1))
                    dc, dr, df, dtrS, dRSS = dense_fit(X, y, coords, tree, bw, kern)
                    cc, cr, cf, ctrS, cRSS = GW._gwr_fit_at_bandwidth(
                        X, y, coords, tree, bw, kern)

                    @test cc ≈ dc rtol = 1e-10
                    @test cf ≈ df rtol = 1e-10
                    @test ctrS ≈ dtrS rtol = 1e-10
                    @test cRSS ≈ dRSS rtol = 1e-10
                    worst = max(worst, maximum(abs.(cc .- dc) ./ max.(abs.(dc), 1e-300)))
                end
            end
            # The stated standard. A Householder QR on `b x K` is not the
            # same arithmetic as on the same matrix padded with `N - b` zero
            # rows, so this is roundoff and not equality -- but it is
            # roundoff, and the bound says so rather than leaving it open.
            @test worst < 1e-11
        end

        @testset "33c. the bandwidth search still picks the same k" begin
            # The one discrete risk: `_select_bandwidth_aicc` compares AICc
            # with `<`, so a near-tie could flip the chosen bandwidth. The
            # margin to the runner-up is ~0.04 AICc units, about 1e11 eps,
            # which no roundoff can cross -- this pins that it does not.
            for side in (8, 11)
                N = side^2
                coords = grid(side)
                lcg = Ref(UInt64(99173))
                function draw()
                    lcg[] = 6364136223846793005 * lcg[] + 1442695040888963407
                    return Float64(lcg[] >> 11) / Float64(UInt64(1) << 53) * 2 - 1
                end
                X = reshape([draw() for _ in 1:(2N)], N, 2)
                y = 1.0 .+ X * [1.0, -0.5] .+ 0.3 .* [draw() for _ in 1:N]
                Xa = hcat(ones(N), X)
                tree = GW._build_tree(coords)

                bw = GW._select_bandwidth_aicc(Xa, y, coords, tree,
                                               GaussianKernel(), AdaptiveBandwidth)
                # Rebuild the AICc curve from the dense arithmetic and check
                # the argmin agrees exactly, not approximately.
                kmin = clamp(max(4, 3 + 2), 1, N - 1)
                kmax = clamp(min(N - 1, 500), kmin, N - 1)
                aiccs = [GW._gwr_fit_at_bandwidth(Xa, y, coords, tree,
                             AdaptiveBandwidth(k), GaussianKernel())[8]
                         for k in kmin:kmax]
                best = (kmin:kmax)[argmin(aiccs)]
                @test bw.k == best
                # And the margin really is enormous, which is why this is safe.
                srt = sort(aiccs)
                @test srt[2] - srt[1] > 1e-6
            end
        end

        @testset "33d. edge cases the compact form has to survive" begin
            coords = grid(7)
            N = size(coords, 1)
            tree = GW._build_tree(coords)
            lcg = Ref(UInt64(4441))
            function draw()
                lcg[] = 6364136223846793005 * lcg[] + 1442695040888963407
                return Float64(lcg[] >> 11) / Float64(UInt64(1) << 53) * 2 - 1
            end
            X = hcat(ones(N), reshape([draw() for _ in 1:(2N)], N, 2))
            y = [draw() for _ in 1:N]

            # A bandwidth wide enough to cover everything: the compact path
            # must reduce to the dense one, support of exactly N.
            sc = GW._support_scratch(Float64)
            n, _, _ = GW._local_support!(sc, tree, coords, 1, FixedBandwidth(1000.0),
                                         GaussianKernel())
            @test n == N
            @test view(sc.idx, 1:n) == 1:N
            dc, _, _, dtrS, _ = dense_fit(X, y, coords, tree,
                                          FixedBandwidth(1000.0), GaussianKernel())
            cc, _, _, ctrS, _ = GW._gwr_fit_at_bandwidth(X, y, coords, tree,
                                          FixedBandwidth(1000.0), GaussianKernel())
            @test cc ≈ dc rtol = 1e-10
            @test ctrS ≈ dtrS rtol = 1e-10

            # Coincident points, where a point need not appear first in its
            # own neighbour list and `w_self` cannot be found by position.
            dup = vcat(zeros(5, 2), grid(5))
            Nd = size(dup, 1)
            td = GW._build_tree(dup)
            scd = GW._support_scratch(Float64)
            for i in 1:5
                n, _, w_self = GW._local_support!(scd, td, dup, i,
                                                  FixedBandwidth(1.5), BisquareKernel())
                @test n >= 5
                @test w_self === GW._get_weights(td, dup, i, FixedBandwidth(1.5),
                                                 BisquareKernel())[i]
            end

            # A Bisquare kernel is exactly zero at `d == h`, and `inrange`
            # returns that point, so the support may legitimately contain
            # zero weights. Nothing downstream may divide by that.
            scz = GW._support_scratch(Float64)
            n, _, _ = GW._local_support!(scz, tree, coords, 25,
                                         FixedBandwidth(2.0), BisquareKernel())
            @test any(iszero, view(scz.wv, 1:n))
        end

        @testset "33e. the chunk helper covers 1:N exactly once" begin
            for N in (0, 1, 7, 64, 100), nc in (1, 3, 8, 64)
                hits = zeros(Int, max(N, 1))
                GW._tforeach_chunks(N; nchunks = nc) do rows
                    for i in rows
                        hits[i] += 1
                    end
                end
                @test all(==(1), view(hits, 1:N))
                N == 0 && @test true          # must not error on an empty range
            end
        end

        @testset "33f. fit_gwr and fit_mgwr end to end" begin
            side = 10
            N = side^2
            coords = grid(side)
            lcg = Ref(UInt64(77001))
            function draw()
                lcg[] = 6364136223846793005 * lcg[] + 1442695040888963407
                return Float64(lcg[] >> 11) / Float64(UInt64(1) << 53) * 2 - 1
            end
            X = reshape([draw() for _ in 1:(2N)], N, 2)
            y = 1.0 .+ X * [1.0, -0.5] .+ 0.3 .* [draw() for _ in 1:N]

            m = fit_gwr(X, y, coords; bandwidth = FixedBandwidth(3.0),
                        kernel = GaussianKernel())
            @test size(m.coef_local) == (N, 3)
            @test all(isfinite, m.coef_local)
            @test all(x -> x > 0, m.se_local)
            @test all(isfinite, m.local_R2)
            @test 0 < m.hat_trace < N

            # se_local against the dense form it replaced.
            tree = GW._build_tree(coords)
            Xa = hcat(ones(N), X)
            for i in (1, 50, N)
                w = GW._get_weights(tree, coords, i, FixedBandwidth(3.0), GaussianKernel())
                Xw = Diagonal(sqrt.(w)) * Xa
                inv_k = inv(Symmetric(Xw' * Xw))
                for k in 1:3
                    @test m.se_local[i, k] ≈
                          sqrt(max(0.0, m.sigma2 * inv_k[k, k])) rtol = 1e-9
                end
                # and local R², which sums over the support
                tss = dot(w, (y .- mean(y)) .^ 2)
                rss = dot(w, m.residuals .^ 2)
                @test m.local_R2[i] ≈ 1 - rss / tss rtol = 1e-10
            end

            mm = fit_mgwr(X, y, coords; inference = false, max_iter = 8)
            @test size(mm.coef_local) == (N, 3)
            @test all(isfinite, mm.coef_local)
            @test all(isfinite, mm.se_local)
            @test length(mm.bandwidths) == 3

            # `_mgwr_se_approx` replaced two byte-identical blocks; check it
            # against the formula those blocks computed.
            for i in (1, 37), k in (1, 3)
                bw_k = AdaptiveBandwidth(round(Int, mm.bandwidths[k]))
                w = GW._get_weights(tree, coords, i, bw_k, BisquareKernel())
                xw = Xa[:, k] .* sqrt.(w)
                denom = dot(xw, xw)
                want = denom > 1e-12 ? sqrt(mm.sigma2 * w[i] / denom) : 0.0
                @test mm.se_local[i, k] ≈ want rtol = 1e-9
            end
        end
    end


    @testset "34. Spatial causal inference" begin
        # The verification standard here is algebra, not a recovery
        # tolerance: the two-period design first-differences to an exact
        # OLS problem, so with no noise the estimator must return the
        # generating parameters to machine precision. Anything looser would
        # not distinguish a correct decomposition from a nearly correct one,
        # which is the failure shape of audit findings 13-16.
        SC = SpatialEconometrics.SpatialCausal

        grid(side) = hcat(repeat(1:side, side), repeat(1:side, inner = side)) .* 1.0

        # A contiguous treated block: exposure then varies among controls
        # (some border the block, most do not), which is what identifies a
        # spillover.
        function block_design(side, lo, hi)
            N = side^2
            coords = grid(side)
            D = zeros(N)
            for i in 1:N
                r, c = coords[i, 1], coords[i, 2]
                (lo <= r <= hi && lo <= c <= hi) && (D[i] = 1.0)
            end
            return N, coords, D
        end

        @testset "34a. exposure mapping is what it claims to be" begin
            side = 8
            N, coords, D = block_design(side, 3, 5)
            W = make_rook_weights(side)
            Wt = row_standardize(W)

            em = exposure(D, W)
            @test size(em.E) == (N, 1)
            @test em.order == 1
            @test em.labels == ["WD"]
            @test em.standardized

            # Row-standardized: exposure is the fraction of neighbours
            # treated, so it is bounded and matches a hand computation.
            @test all(0.0 .<= em.E[:, 1] .<= 1.0)
            for i in (1, 20, N)
                nbrs = findall(!iszero, W[i, :])
                @test em.E[i, 1] ≈ sum(D[nbrs] .* Wt[i, nbrs]) rtol = 1e-14
            end
            @test em.E[:, 1] ≈ Wt * D rtol = 1e-14

            # Unstandardized gives a count, not a fraction.
            emc = exposure(D, W; standardize = false)
            @test emc.E[:, 1] ≈ W * D rtol = 1e-14
            @test !emc.standardized

            # Second order is W² D, and it is a genuinely different column.
            em2 = exposure(D, W; order = 2)
            @test size(em2.E) == (N, 2)
            @test em2.labels == ["WD", "W2D"]
            @test em2.E[:, 1] ≈ Wt * D rtol = 1e-14
            @test em2.E[:, 2] ≈ Wt * (Wt * D) rtol = 1e-14
            @test !(em2.E[:, 1] ≈ em2.E[:, 2])

            # A non-binary treatment makes the direct/indirect split
            # meaningless, and degenerate designs have nothing to estimate.
            @test_throws ArgumentError exposure([0.0, 0.5, 1.0], make_rook_weights(2)[1:3, 1:3])
            @test_throws ArgumentError exposure(zeros(N), W)
            @test_throws ArgumentError exposure(ones(N), W)
            @test_throws DimensionMismatch exposure(zeros(N + 1), W)
            @test_throws ArgumentError exposure(D, W; order = 0)
        end

        @testset "34b. interference report finds what identifies δ" begin
            side = 10
            N, coords, D = block_design(side, 4, 6)
            W = make_rook_weights(side)
            em = exposure(D, W)
            rep = check_interference(em)

            @test rep.n_treated == count(isone, D)
            @test rep.n_control == N - rep.n_treated
            @test rep.n_treated + rep.n_control == N
            @test rep.identified
            # A block in a 10x10 grid leaves plenty of untouched controls.
            @test rep.n_control_unexposed > 0
            @test rep.exposure_var_control > 0
            @test rep.mean_exposure_treated > rep.mean_exposure_control

            E = em.E[:, 1]
            @test rep.mean_exposure_treated ≈ mean(E[D .== 1.0]) rtol = 1e-14
            @test rep.mean_exposure_control ≈ mean(E[D .== 0.0]) rtol = 1e-14
            @test rep.exposure_cor ≈ cor(D, E) rtol = 1e-12
            @test occursin("identified", rep.message)

            # No control has a treated neighbour: δ has no baseline, and
            # the report must say so rather than leave it to be discovered.
            Wfar = make_rook_weights(side)
            Dlone = zeros(N); Dlone[1] = 1.0
            # Remove the treated unit's links so nothing is exposed at all.
            Wiso = sparse(Wfar)
            Wiso[1, :] .= 0.0; Wiso[:, 1] .= 0.0
            dropzeros!(Wiso)
            em_iso = exposure(Dlone, Wiso)
            rep_iso = check_interference(em_iso)
            @test !rep_iso.identified
            @test rep_iso.n_control_unexposed == rep_iso.n_control
            @test occursin("no control", lowercase(rep_iso.message))

            # show must not error on either branch
            @test occursin("Interference", sprint(show, rep))
            @test occursin("identified", sprint(show, rep_iso))
        end

        @testset "34c. two-period DiD is exact, not approximate" begin
            side = 12
            N, coords, D = block_design(side, 3, 6)
            W = make_rook_weights(side)
            em = exposure(D, W)
            E = em.E[:, 1]

            γ, τ, δ = 0.7, 2.5, -1.25
            μ = collect(range(-1.0, 1.0, length = N))

            y = zeros(N, 2)
            y[:, 1] .= μ
            y[:, 2] .= μ .+ γ .+ τ .* D .+ δ .* E

            m = fit_spatial_did(y, D, W; vcov = :ols)

            Ē_T = mean(E[D .== 1.0])
            Ē_C = mean(E[D .== 0.0])

            # The generating parameters, to machine precision.
            @test m.att_direct ≈ τ rtol = 1e-12
            @test m.att_spill_treated ≈ δ * Ē_T rtol = 1e-12
            @test m.att_spill_control ≈ δ * Ē_C rtol = 1e-12
            @test m.att_total_treated ≈ τ + δ * Ē_T rtol = 1e-12
            @test m.coef[1] ≈ γ rtol = 1e-12
            @test m.coef[2] ≈ τ rtol = 1e-12
            @test m.coef[3] ≈ δ rtol = 1e-12
            @test norm(m.residuals) < 1e-10

            # The decomposition is an identity, so it holds exactly rather
            # than to a tolerance -- both are the same linear form in θ̂.
            @test m.att_total_treated == m.att_direct + m.att_spill_treated

            # First-differencing removes the unit effect, so an arbitrary
            # change to it cannot move any reported effect.
            y2 = copy(y)
            shift = collect(range(-50.0, 50.0, length = N))
            y2[:, 1] .+= shift
            y2[:, 2] .+= shift
            m2 = fit_spatial_did(y2, D, W; vcov = :ols)
            @test m2.att_direct ≈ m.att_direct rtol = 1e-12
            @test m2.att_spill_control ≈ m.att_spill_control rtol = 1e-12

            # StatsAPI surface and show
            @test nobs(m) == N
            @test length(coef(m)) == 3
            @test size(vcov(m)) == (3, 3)
            @test coefnames(m) == ["(Intercept)", "D", "WD"]
            @test dof_residual(m) == N - 3
            @test length(stderror(m)) == 3
            @test occursin("Difference-in-Differences", sprint(show, m))
            @test m.design === :two_period
            @test m.n_units == N && m.n_periods == 2

            # Second-order exposure recovers both rings exactly.
            δ2 = 0.4
            em2 = exposure(D, W; order = 2)
            y3 = zeros(N, 2)
            y3[:, 1] .= μ
            y3[:, 2] .= μ .+ γ .+ τ .* D .+ δ .* em2.E[:, 1] .+ δ2 .* em2.E[:, 2]
            m3 = fit_spatial_did(y3, D, W; vcov = :ols, order = 2)
            @test m3.coef[2] ≈ τ rtol = 1e-10
            @test m3.coef[3] ≈ δ rtol = 1e-10
            @test m3.coef[4] ≈ δ2 rtol = 1e-10
            @test m3.coefnames == ["(Intercept)", "D", "WD", "W2D"]
            # The reported spillover sums over both rings.
            want_t = δ * mean(em2.E[D .== 1.0, 1]) + δ2 * mean(em2.E[D .== 1.0, 2])
            @test m3.att_spill_treated ≈ want_t rtol = 1e-10
        end

        @testset "34d. staggered rollout, exact under homogeneity" begin
            side = 10
            N = side^2
            T_per = 8
            coords = grid(side)
            W = make_rook_weights(side)
            Wt = row_standardize(W)

            adopt = fill(0, N)
            for i in 1:N
                c = coords[i, 2]
                if c <= 2.5;     adopt[i] = 3
                elseif c <= 5.0; adopt[i] = 5
                elseif c <= 7.5; adopt[i] = 7
                end
            end
            Dpath = zeros(N, T_per)
            for i in 1:N, t in 1:T_per
                adopt[i] != 0 && t >= adopt[i] && (Dpath[i, t] = 1.0)
            end

            μ = collect(range(-1.0, 1.0, length = N))
            λ = [0.0, 0.3, 0.1, -0.2, 0.4, 0.25, -0.1, 0.5]
            τ, δ = 2.0, -0.8

            y = zeros(N, T_per)
            for i in 1:N, t in 1:T_per
                E_it = dot(view(Wt, i, :), view(Dpath, :, t))
                y[i, t] = μ[i] + λ[t] + τ * Dpath[i, t] + δ * E_it
            end

            m = fit_spatial_did(y, Dpath, W; vcov = :ols)
            @test m.design === :staggered
            @test m.n_periods == T_per
            # The within transform absorbs μ_i and γ_t exactly, so with
            # homogeneous effects this is as exact as the two-period path.
            @test m.att_direct ≈ τ rtol = 1e-12
            @test m.coef[2] ≈ δ rtol = 1e-12
            @test norm(m.residuals) < 1e-10
            # No intercept: the two-way transform already removed it, and a
            # column of ones would be collinear with the transform.
            @test m.coefnames == ["D", "WD"]
            @test nobs(m) == N * T_per

            # Unit and period effects are absorbed, so shifting either
            # cannot move the estimate.
            y2 = copy(y)
            for i in 1:N, t in 1:T_per
                y2[i, t] += 10.0 * μ[i] - 3.0 * λ[t]
            end
            m2 = fit_spatial_did(y2, Dpath, W; vcov = :ols)
            @test m2.att_direct ≈ τ rtol = 1e-10
            @test m2.coef[2] ≈ δ rtol = 1e-10

            # A length-N vector cannot describe an 8-period design, and
            # saying so beats silently treating it as two periods.
            @test_throws ArgumentError fit_spatial_did(y, zeros(N), W; vcov = :ols)
            @test_throws ArgumentError fit_spatial_did(y[:, 1:1], zeros(N), W; vcov = :ols)
            @test_throws DimensionMismatch fit_spatial_did(y, zeros(N, T_per + 1), W; vcov = :ols)
            @test_throws ArgumentError fit_spatial_did(y, fill(0.5, N, T_per), W; vcov = :ols)
        end

        @testset "34e. staggered TWFE under heterogeneity, as documented" begin
            # The docstring warns that this estimator does not recover the
            # sample-weighted ATT when cohort effects differ, and quotes
            # numbers. These are those numbers: if the implementation
            # changes the warning has to change with it, which is the point
            # of pinning them.
            side = 10
            N = side^2
            T_per = 8
            coords = grid(side)
            W = make_rook_weights(side)
            μ = collect(range(-1.0, 1.0, length = N))
            λ = [0.0, 0.3, 0.1, -0.2, 0.4, 0.25, -0.1, 0.5]
            coh = Dict(3 => 1.0, 5 => 2.0, 7 => 3.0)

            function run_case(layout, profile)
                adopt = fill(0, N)
                for i in 1:N
                    r, c = Int(coords[i, 1]), Int(coords[i, 2])
                    if layout === :bands
                        c <= 2 ? (adopt[i] = 3) : c <= 4 ? (adopt[i] = 5) :
                            c <= 6 ? (adopt[i] = 7) : nothing
                    else
                        k = mod(r + c, 4)
                        k == 0 ? (adopt[i] = 3) : k == 1 ? (adopt[i] = 5) :
                            k == 2 ? (adopt[i] = 7) : nothing
                    end
                end
                Dp = zeros(N, T_per)
                for i in 1:N, t in 1:T_per
                    adopt[i] != 0 && t >= adopt[i] && (Dp[i, t] = 1.0)
                end
                y = zeros(N, T_per)
                num, den, lo, hi = 0.0, 0.0, Inf, -Inf
                for i in 1:N, t in 1:T_per
                    eff = 0.0
                    if adopt[i] != 0 && t >= adopt[i]
                        eff = profile === :constant ? coh[adopt[i]] :
                              coh[adopt[i]] * (t - adopt[i] + 1)
                        num += eff; den += 1
                        lo = min(lo, eff); hi = max(hi, eff)
                    end
                    y[i, t] = μ[i] + λ[t] + eff
                end
                m = fit_spatial_did(y, Dp, W; vcov = :ols)
                return m.att_direct, num / den, lo, hi
            end

            # No noise, so every gap is bias rather than sampling error.
            est_bc, att_bc, _, _ = run_case(:bands, :constant)
            est_bd, att_bd, lo_bd, hi_bd = run_case(:bands, :dynamic)
            est_ic, att_ic, _, _ = run_case(:interleaved, :constant)
            est_id, att_id, _, _ = run_case(:interleaved, :dynamic)

            @test att_bc ≈ 1.6667 atol = 1e-3
            @test est_bc ≈ 2.7805 atol = 1e-3
            @test att_bd ≈ 4.1667 atol = 1e-3
            @test est_bd ≈ 1.8049 atol = 1e-3
            @test att_ic ≈ 1.6622 atol = 1e-3
            @test est_ic ≈ 2.0000 atol = 1e-3
            @test att_id ≈ 4.1554 atol = 1e-3
            @test est_id ≈ 3.4397 atol = 1e-3

            # The documented qualitative claims, which are what a reader
            # takes away: the bias is large, its sign is not predictable,
            # and spatially contiguous cohorts are the worst case.
            @test est_bc > att_bc            # too high with constant effects
            @test est_bd < att_bd            # too low with growing ones
            @test abs(est_bd - att_bd) / att_bd > 0.5
            @test abs(est_bd - att_bd) > abs(est_id - att_id)
            # Inside the hull in every case -- a weak guarantee, since the
            # hull is [1, 8] here, and the docstring says so.
            @test lo_bd <= est_bd <= hi_bd
            @test (hi_bd - lo_bd) >= 7.0
        end

        @testset "34f. covariance estimators and their coverage" begin
            side = 12
            N, coords, D = block_design(side, 3, 6)
            W = make_rook_weights(side)
            em = exposure(D, W)
            E = em.E[:, 1]
            γ, τ, δ = 0.7, 2.5, -1.25
            μ = collect(range(-1.0, 1.0, length = N))

            rng = MersenneTwister(4242)
            y = zeros(N, 2)
            y[:, 1] .= μ .+ randn(rng, N)
            y[:, 2] .= μ .+ γ .+ τ .* D .+ δ .* E .+ randn(rng, N)

            # Every estimator must produce a usable, symmetric, PSD
            # covariance and a positive standard error on each effect.
            m_ols = fit_spatial_did(y, D, W; vcov = :ols)
            m_clu = fit_spatial_did(y, D, W; vcov = :cluster)
            m_con = fit_spatial_did(y, D, W; vcov = :conley,
                                    coords = coords, dist_cutoff = 2.5)
            for m in (m_ols, m_clu, m_con)
                V = vcov(m)
                @test maximum(abs.(V .- V')) < 1e-10
                @test all(diag(V) .>= 0)
                @test minimum(eigvals(Symmetric(V))) > -1e-10
                @test all(>(0), (m.att_direct_se, m.att_spill_treated_se,
                                 m.att_spill_control_se, m.att_total_treated_se))
                # Point estimates do not depend on the covariance.
                @test m.att_direct ≈ m_ols.att_direct rtol = 1e-12
            end
            @test occursin("cluster", m_clu.vcov_name)
            @test occursin("Conley", m_con.vcov_name)
            @test occursin("OLS", m_ols.vcov_name)

            # Default is :cluster, on measured coverage rather than
            # convention -- see the fit_spatial_did docstring.
            m_def = fit_spatial_did(y, D, W)
            @test m_def.vcov ≈ m_clu.vcov rtol = 1e-12

            # Clustering on unit in a first-differenced cross-section is a
            # singleton-cluster design, so it coincides with the
            # heteroskedasticity-robust form. Checked directly, because the
            # docstring says so.
            Z = hcat(ones(N), D, E)
            r = m_clu.residuals
            bread = inv(Symmetric(Z' * Z))
            meat = zeros(3, 3)
            for i in 1:N
                zi = Z[i, :]
                meat .+= (zi * zi') .* r[i]^2
            end
            c_adj = N / (N - 1) * (N - 1) / (N - 3)
            @test m_clu.vcov ≈ bread * (c_adj .* meat) * bread rtol = 1e-10

            # :conley without what it needs must say what is missing.
            @test_throws ArgumentError fit_spatial_did(y, D, W; vcov = :conley)
            @test_throws ArgumentError fit_spatial_did(y, D, W; vcov = :conley,
                                                       coords = coords)
            @test_throws ArgumentError fit_spatial_did(y, D, W; vcov = :nonsense)
            # Driscoll-Kraay needs more than one period; a differenced
            # two-period design has one.
            @test_throws ArgumentError fit_spatial_did(y, D, W;
                                                       vcov = DriscollKraayVCOV())

            # A spec instance is accepted as well as a symbol.
            m_spec = fit_spatial_did(y, D, W;
                                     vcov = ConleySpatialHAC(coords, 2.5; max_time_lag = 0))
            @test m_spec.vcov ≈ m_con.vcov rtol = 1e-10

            # Coverage, measured. These are the figures the docstring
            # quotes; the reps are kept modest so the suite stays quick, so
            # the bands are wide enough to be stable but still fail if an
            # estimator breaks.
            ρ_err = 0.6
            A = sparse(1.0I, N, N) - ρ_err * W
            Afac = lu(A)
            z95 = 1.959963984540054
            nreps = 300
            cov_c = 0; cov_o = 0; cov_n15 = 0; cov_n60 = 0
            rng2 = MersenneTwister(9091)
            for _ in 1:nreps
                u0 = Afac \ randn(rng2, N)
                u1 = Afac \ randn(rng2, N)
                yy = zeros(N, 2)
                yy[:, 1] .= μ .+ u0
                yy[:, 2] .= μ .+ γ .+ τ .* D .+ δ .* E .+ u1
                mc = fit_spatial_did(yy, D, W; vcov = :cluster)
                abs(mc.att_direct - τ) <= z95 * mc.att_direct_se && (cov_c += 1)
                mo = fit_spatial_did(yy, D, W; vcov = :ols)
                abs(mo.att_direct - τ) <= z95 * mo.att_direct_se && (cov_o += 1)
                m15 = fit_spatial_did(yy, D, W; vcov = :conley,
                                      coords = coords, dist_cutoff = 1.5)
                abs(m15.att_direct - τ) <= z95 * m15.att_direct_se && (cov_n15 += 1)
                m60 = fit_spatial_did(yy, D, W; vcov = :conley,
                                      coords = coords, dist_cutoff = 6.0)
                abs(m60.att_direct - τ) <= z95 * m60.att_direct_se && (cov_n60 += 1)
            end
            # Near nominal.
            @test 0.90 <= cov_c / nreps <= 0.99
            @test 0.90 <= cov_o / nreps <= 0.99
            @test 0.89 <= cov_n15 / nreps <= 0.99
            # The documented failure: a wide Conley cutoff under-covers
            # badly, and by more than the narrow one.
            @test cov_n60 / nreps < 0.92
            @test cov_n60 < cov_n15
        end

        @testset "34g. the identification guard actually fires" begin
            side = 8
            N = side^2
            coords = grid(side)
            W = make_rook_weights(side)

            # Treat a checkerboard: every control borders a treated unit
            # and exposure among controls is constant, so δ is absorbed by
            # the intercept. The design is rank deficient and must be
            # refused by name rather than silently fitted.
            D = zeros(N)
            for i in 1:N
                isodd(Int(coords[i, 1]) + Int(coords[i, 2])) && (D[i] = 1.0)
            end
            em = exposure(D, W)
            rep = check_interference(em)
            # On a rook lattice a checkerboard makes every neighbour of a
            # control treated, so exposure is 1 for controls and 0 for the
            # treated -- exactly collinear with D.
            @test rep.n_control_unexposed == 0
            @test all(em.E[D .== 0.0, 1] .≈ 1.0)

            μ = collect(range(-1.0, 1.0, length = N))
            y = zeros(N, 2)
            y[:, 1] .= μ
            y[:, 2] .= μ .+ 0.7 .+ 2.5 .* D
            err = nothing
            try
                fit_spatial_did(y, D, W; vcov = :ols)
            catch e
                err = e
            end
            @test err isa ArgumentError
            @test occursin("not identified", err.msg)
            @test occursin("fit_spatial_did", err.msg)

            # A design with no unexposed control but genuine variation is
            # still identified, and the report says what the contrast is.
            Wq = make_queen_weights(side)
            Dq = zeros(N)
            for i in 1:N
                r, c = coords[i, 1], coords[i, 2]
                (3 <= r <= 4 && 3 <= c <= 4) && (Dq[i] = 1.0)
            end
            repq = check_interference(exposure(Dq, Wq))
            @test repq.identified
        end

        @testset "34h. ring exposure reads a band, not a neighbourhood" begin
            side = 10
            N, coords, D = block_design(side, 4, 6)

            near = ring_exposure(D, coords, 0.0, 1.5)
            far = ring_exposure(D, coords, 1.5, 3.0)
            @test size(near.E) == (N, 1)
            @test !(near.E[:, 1] ≈ far.E[:, 1])

            # The near ring is rook-and-diagonal contiguity, so it must
            # agree with a queen weight matrix's exposure.
            Wq = make_queen_weights(side)
            @test near.E[:, 1] ≈ exposure(D, Wq).E[:, 1] rtol = 1e-12

            # A band that reaches nothing leaves exposure at zero, and the
            # interference report must call that unidentified rather than
            # returning a spillover built on no variation.
            none = ring_exposure(D, coords, 20.0, 25.0)
            @test all(iszero, none.E)
            @test !check_interference(none).identified

            @test_throws DimensionMismatch ring_exposure(zeros(N + 1), coords, 0.0, 1.5)
        end

        @testset "34i. vcov_cluster on its own terms" begin
            # The estimator underneath, checked against the formula rather
            # than through the DiD. A covariance is a second-order quantity,
            # which is where audit findings 1-7 lived.
            rng = MersenneTwister(31337)
            n, K, G = 120, 3, 12
            Z = hcat(ones(n), randn(rng, n, K - 1))
            r = randn(rng, n)
            groups = repeat(1:G, inner = div(n, G))

            spec = ClusterRobustVCOV(groups)
            V = vcov_cluster(Z, r, spec)

            bread = inv(Symmetric(Z' * Z))
            S = zeros(K, G)
            for i in 1:n
                S[:, groups[i]] .+= Z[i, :] .* r[i]
            end
            c_adj = G / (G - 1) * (n - 1) / (n - K)
            @test V ≈ bread * (c_adj .* (S * S')) * bread rtol = 1e-12
            @test maximum(abs.(V .- V')) < 1e-10
            @test minimum(eigvals(Symmetric(V))) > -1e-10

            # Without the correction the scaling is exactly c_adj smaller.
            V0 = vcov_cluster(Z, r, ClusterRobustVCOV(groups; dof_correction = false))
            @test V0 ≈ V ./ c_adj rtol = 1e-12

            # Singleton clusters reduce to the heteroskedasticity-robust
            # form, which is the trap the docstring names.
            Vsing = vcov_cluster(Z, r, ClusterRobustVCOV(1:n))
            meat = zeros(K, K)
            for i in 1:n
                zi = Z[i, :]
                meat .+= (zi * zi') .* r[i]^2
            end
            c_hc = n / (n - 1) * (n - 1) / (n - K)
            @test Vsing ≈ bread * (c_hc .* meat) * bread rtol = 1e-12

            # Labels need not be integers or contiguous.
            glab = [string("g", g) for g in groups]
            @test vcov_cluster(Z, r, ClusterRobustVCOV(glab)) ≈ V rtol = 1e-12

            # Group order must not matter: permuting rows permutes nothing.
            perm = randperm(MersenneTwister(5), n)
            Vp = vcov_cluster(Z[perm, :], r[perm], ClusterRobustVCOV(groups[perm]))
            @test Vp ≈ V rtol = 1e-10

            @test_throws DimensionMismatch vcov_cluster(Z, r[1:(n - 1)], spec)
            @test_throws DimensionMismatch vcov_cluster(Z, r, ClusterRobustVCOV(groups[1:(n - 1)]))
            @test_throws ArgumentError ClusterRobustVCOV(Int[])
            # One group, and a saturated design, both divide by zero under
            # the correction and are named rather than returning Inf.
            @test_throws ArgumentError vcov_cluster(Z, r, ClusterRobustVCOV(ones(Int, n)))
            @test_throws ArgumentError vcov_cluster(Z[1:K, :], r[1:K],
                                                    ClusterRobustVCOV(1:K))
        end

        @testset "34j. covariates enter without disturbing the effects" begin
            side = 10
            N, coords, D = block_design(side, 3, 6)
            W = make_rook_weights(side)
            em = exposure(D, W)
            E = em.E[:, 1]
            γ, τ, δ = 0.7, 2.5, -1.25
            μ = collect(range(-1.0, 1.0, length = N))
            βx = [1.5, -0.75]
            rng = MersenneTwister(5)

            # Two-period: X is N x K x 2, and differencing it is part of
            # the transform rather than something the caller must do.
            X = randn(rng, N, 2, 2)
            y = zeros(N, 2)
            y[:, 1] .= μ .+ X[:, :, 1] * βx
            y[:, 2] .= μ .+ γ .+ τ .* D .+ δ .* E .+ X[:, :, 2] * βx

            m = fit_spatial_did(y, D, W; X = X, vcov = :ols)
            @test m.coefnames == ["(Intercept)", "D", "WD", "ΔX1", "ΔX2"]
            @test m.coef[2] ≈ τ rtol = 1e-11
            @test m.coef[3] ≈ δ rtol = 1e-11
            @test m.coef[4] ≈ βx[1] rtol = 1e-11
            @test m.coef[5] ≈ βx[2] rtol = 1e-11
            @test norm(m.residuals) < 1e-10
            # The effects are still read off the same slots, so adding
            # covariates must not shift which coefficient is which.
            @test m.att_direct ≈ τ rtol = 1e-11
            @test m.att_spill_control ≈ δ * mean(E[D .== 0.0]) rtol = 1e-11

            # A time-invariant covariate differences to exactly zero. It is
            # refused with the reason, rather than silently dropped or left
            # to surface as a rank deficiency the caller has to decode.
            err = nothing
            try
                fit_spatial_did(y, D, W; X = randn(rng, N, 2), vcov = :ols)
            catch e
                err = e
            end
            @test err isa ArgumentError
            @test occursin("time-invariant", err.msg)
            @test_throws ArgumentError fit_spatial_did(y, D, W;
                                                       X = randn(rng, N, 2, 3), vcov = :ols)

            # Staggered: X is N x K x T and enters the within transform.
            T_per = 5
            Wt = row_standardize(W)
            adopt = fill(0, N)
            for i in 1:N
                c = coords[i, 2]
                c <= 3 ? (adopt[i] = 2) : c <= 6 ? (adopt[i] = 4) : nothing
            end
            Dp = zeros(N, T_per)
            for i in 1:N, t in 1:T_per
                adopt[i] != 0 && t >= adopt[i] && (Dp[i, t] = 1.0)
            end
            Xs = randn(rng, N, 2, T_per)
            λ = [0.0, 0.2, -0.1, 0.3, 0.15]
            ys = zeros(N, T_per)
            for i in 1:N, t in 1:T_per
                E_it = dot(view(Wt, i, :), view(Dp, :, t))
                ys[i, t] = μ[i] + λ[t] + τ * Dp[i, t] + δ * E_it +
                           dot(view(Xs, i, :, t), βx)
            end
            ms = fit_spatial_did(ys, Dp, W; X = Xs, vcov = :ols)
            @test ms.coefnames == ["D", "WD", "X1", "X2"]
            @test ms.coef[1] ≈ τ rtol = 1e-11
            @test ms.coef[2] ≈ δ rtol = 1e-11
            @test ms.coef[3] ≈ βx[1] rtol = 1e-11
            @test ms.coef[4] ≈ βx[2] rtol = 1e-11
            @test norm(ms.residuals) < 1e-10

            @test_throws DimensionMismatch fit_spatial_did(ys, Dp, W;
                                                           X = randn(rng, N, 2, T_per + 1),
                                                           vcov = :ols)
            @test_throws ArgumentError fit_spatial_did(ys, Dp, W;
                                                       X = randn(rng, N, 2), vcov = :ols)
        end
    end
    @testset "35. Spatial synthetic control" begin
        # The gate here is the same one testset 34 uses: algebra where the
        # quantity is algebraic. The donor weights solve a convex program,
        # so the reference is not a tolerance on recovery but the exact
        # optimum computed independently by enumerating every support --
        # which is feasible for a small donor pool and is the only check
        # that distinguishes a converged solver from a nearly converged
        # one.
        SC = SpatialEconometrics.SpatialCausal

        scgrid(side) = hcat(repeat(1:side, side), repeat(1:side, inner = side)) .* 1.0

        # Exact optimum of  min ||x1 - X0 w||^2 + lam d'w  s.t. w >= 0,
        # sum(w) = 1, by enumerating all 2^J - 1 supports and solving the
        # equality-constrained least squares on each. O(2^J), so J stays
        # small; written out here rather than reusing anything from src so
        # that it is a genuinely independent reference.
        function scm_brute(X0::Matrix{Float64}, x1::Vector{Float64},
                           d::Vector{Float64}, lam::Float64)
            J = size(X0, 2)
            best_w = Float64[]; best_f = Inf
            objf(w) = sum(abs2, x1 .- X0 * w) + lam * dot(d, w)
            for mask in 1:(2^J - 1)
                S = [j for j in 1:J if (mask >> (j - 1)) & 1 == 1]
                k = length(S)
                Xs = X0[:, S]
                A = [2 .* (Xs' * Xs)  ones(k); ones(1, k)  0.0]
                rank(A) < k + 1 && continue
                sol = nothing
                try
                    sol = A \ vcat(2 .* (Xs' * x1) .- lam .* d[S], 1.0)
                catch
                    continue
                end
                v = sol[1:k]
                all(>=(-1e-11), v) || continue
                w = zeros(J); w[S] .= max.(v, 0.0)
                sw = sum(w); sw > 0 && (w ./= sw)
                f = objf(w)
                f < best_f - 1e-14 && (best_f = f; best_w = w)
            end
            return best_w, best_f
        end

        @testset "35a. simplex projection" begin
            # The projection is characterised by subtracting one common
            # threshold and clipping: so on the support, v - w must be
            # constant, and nowhere may w go negative.
            rng = MersenneTwister(3501)
            worst_sum = 0.0; worst_neg = 0.0; worst_spread = 0.0
            for _ in 1:3000
                J = rand(rng, 1:12)
                v = randn(rng, J) .* exp(randn(rng) * 2)
                w = copy(v); SC._project_simplex!(w)
                worst_sum = max(worst_sum, abs(sum(w) - 1.0))
                worst_neg = max(worst_neg, -minimum(w))
                sup = w .> 0
                if any(sup) && !all(sup)
                    c = (v .- w)[sup]
                    worst_spread = max(worst_spread, maximum(c) - minimum(c))
                end
            end
            @test worst_sum < 1e-13
            @test worst_neg == 0.0
            @test worst_spread < 1e-13

            # Against brute force in low dimension: no randomly drawn
            # simplex point may be closer to v than the projection is.
            rng2 = MersenneTwister(3502)
            worst_excess = -Inf
            for _ in 1:400
                J = rand(rng2, 2:4)
                v = randn(rng2, J) .* 2
                w = copy(v); SC._project_simplex!(w)
                dw = sum(abs2, v .- w)
                best = Inf
                for _ in 1:2000
                    c = rand(rng2, J); c ./= sum(c)
                    best = min(best, sum(abs2, v .- c))
                end
                worst_excess = max(worst_excess, dw - best)
            end
            @test worst_excess < 0.0

            # Degenerate inputs. A single donor must take all the weight,
            # and a point already on the simplex must be returned
            # unchanged.
            w1 = [-5.0]; SC._project_simplex!(w1)
            @test w1 == [1.0]
            w2 = [0.2, 0.3, 0.5]; w2c = copy(w2)
            SC._project_simplex!(w2)
            @test maximum(abs, w2 .- w2c) < 1e-15
            @test_throws ArgumentError SC._project_simplex!(Float64[])
        end

        @testset "35b. solver against the exact optimum" begin
            # J <= T0, so X0 has full column rank and the optimum is
            # unique: the weights themselves must agree, not merely the
            # objective.
            rng = MersenneTwister(3503)
            max_obj = 0.0; max_w = 0.0; max_kkt = 0.0; max_viol = 0.0
            nprob = 0
            for _ in 1:120
                J = rand(rng, 3:6)
                T0 = rand(rng, (J + 2):(J + 8))
                X0 = randn(rng, T0, J)
                x1 = randn(rng, T0)
                d = abs.(randn(rng, J))
                lam = rand(rng) < 0.5 ? 0.0 : rand(rng) * 2.0
                w, f, _, kkt, viol = SC._solve_pensynth(X0, x1, d, lam)
                wb, fb = scm_brute(X0, x1, d, lam)
                isempty(wb) && continue
                nprob += 1
                max_obj = max(max_obj, f - fb)
                max_w = max(max_w, maximum(abs, w .- wb))
                max_kkt = max(max_kkt, kkt); max_viol = max(max_viol, viol)
            end
            @test nprob > 100
            @test max_obj < 1e-12        # measured 3.55e-15
            @test max_w < 1e-10          # measured 1.31e-14
            @test max_kkt < 1e-12        # measured 9.68e-15
            @test max_viol == 0.0

            # J > T0 is the realistic case -- more donors than
            # pre-periods -- and there the optimum is a face, not a point.
            # Only the objective is well defined, which is itself the
            # argument for the penalty.
            rng2 = MersenneTwister(3504)
            max_obj2 = 0.0; n2 = 0
            for _ in 1:80
                T0 = rand(rng2, 2:4); J = rand(rng2, 5:7)
                X0 = randn(rng2, T0, J); x1 = randn(rng2, T0)
                d = abs.(randn(rng2, J))
                lam = rand(rng2) < 0.5 ? 0.0 : rand(rng2) * 1.5
                w, f, = SC._solve_pensynth(X0, x1, d, lam)
                wb, fb = scm_brute(X0, x1, d, lam)
                isempty(wb) && continue
                n2 += 1
                max_obj2 = max(max_obj2, f - fb)
            end
            @test n2 > 60
            @test max_obj2 < 1e-12       # measured 1.78e-15
        end

        @testset "35c. exact convex combination" begin
            # The treated unit's path IS 0.6 * donor_a + 0.4 * donor_b at
            # every period, so with no noise the weights are identified
            # and the gap is zero throughout. Machine precision, not a
            # recovery tolerance.
            side = 8; N = side^2; T_per = 20; T0 = 14
            W = make_rook_weights(side); coords = scgrid(side)
            rng = MersenneTwister(3505)
            Y = randn(rng, N, T_per)
            a, b = 50, 61
            Y[1, :] .= 0.6 .* Y[a, :] .+ 0.4 .* Y[b, :]

            scm = fit_synthetic_control(Y, 1, T0; W = W, coords = coords)
            qa = findfirst(==(a), scm.donors)
            qb = findfirst(==(b), scm.donors)
            @test qa !== nothing && qb !== nothing
            @test scm.weights[1, qa] ≈ 0.6 atol = 1e-10
            @test scm.weights[1, qb] ≈ 0.4 atol = 1e-10
            truth = [j == a ? 0.6 : j == b ? 0.4 : 0.0 for j in scm.donors]
            @test maximum(abs, scm.weights[1, :] .- truth) < 1e-10
            @test abs(sum(scm.weights[1, :]) - 1.0) < 1e-13
            @test minimum(scm.weights) >= 0.0
            @test maximum(abs, scm.gaps) < 1e-9
            @test abs(scm.att) < 1e-9
            @test scm.kkt_gap[1] < 1e-12
            @test scm.kkt_violation[1] < 1e-12

            # The gap is a definition, so it holds exactly rather than to
            # a tolerance.
            syn = (Y[scm.donors, :])' * scm.weights[1, :]
            @test maximum(abs, scm.gaps[1, :] .- (Y[1, :] .- syn)) == 0.0
            @test maximum(abs, scm.synthetic[1, :] .- syn) == 0.0
            # residuals() is the pre-period slice of the gaps.
            @test size(StatsAPI.residuals(scm)) == (1, T0)
            @test maximum(abs, StatsAPI.residuals(scm) .- scm.gaps[:, 1:T0]) == 0.0
            @test StatsAPI.coef(scm) === donor_weights(scm)
            @test StatsAPI.coefnames(scm) == ["unit $(j)" for j in scm.donors]
            @test StatsAPI.nobs(scm) == T0
            # No standard errors exist for this estimator, and the type
            # must not pretend otherwise.
            @test_throws MethodError StatsAPI.vcov(scm)
            @test_throws MethodError StatsAPI.stderror(scm)
        end

        @testset "35d. penalty limits" begin
            rng = MersenneTwister(3506)
            T0 = 10; J = 6
            X0 = randn(rng, T0, J); x1 = randn(rng, T0)
            d = [5.0, 1.0, 9.0, 3.0, 7.0, 4.0]

            # lam = 0 is the unpenalised program.
            w0, f0, = SC._solve_pensynth(X0, x1, d, 0.0)
            wn, fn, = SC._solve_pensynth(X0, x1, zeros(J), 0.0)
            @test maximum(abs, w0 .- wn) < 1e-10
            @test abs(f0 - fn) < 1e-12

            # As lam grows the support must collapse into argmin d. With
            # distinct distances that is the single nearest donor.
            for lam in (50.0, 1e4, 1e8)
                w, = SC._solve_pensynth(X0, x1, d, lam)
                @test argmax(w) == argmin(d)
                @test w[argmin(d)] ≈ 1.0 atol = 1e-12
            end

            # With ties -- the usual case on a lattice, where several
            # donors sit at the same distance -- the support must lie
            # within the tied set, which is the general statement.
            rng2 = MersenneTwister(3507)
            bad = 0
            for _ in 1:40
                Jq = rand(rng2, 4:8); T0q = rand(rng2, 5:10)
                X0q = randn(rng2, T0q, Jq); x1q = randn(rng2, T0q)
                dq = abs.(randn(rng2, Jq)) .+ 0.1
                wq, = SC._solve_pensynth(X0q, x1q, dq, 1e8)
                dmin = minimum(dq)
                all(q -> dq[q] <= dmin + 1e-9, findall(>(1e-9), wq)) || (bad += 1)
            end
            @test bad == 0

            # Monotonicity of the penalty term: a larger lam cannot buy a
            # larger distance-weighted average.
            side = 8; N = side^2; T_per = 16; T0b = 12
            W = make_rook_weights(side); coords = scgrid(side)
            rng3 = MersenneTwister(3508)
            Yb = randn(rng3, N, T_per)
            prev = Inf
            for lam in (0.0, 0.05, 0.5, 5.0)
                s = fit_synthetic_control(Yb, 28, T0b; W = W, coords = coords,
                                          penalty = lam)
                dn = [sqrt(sum(abs2, coords[28, :] .- coords[j, :]))
                      for j in s.donors]
                avg = dot(dn, s.weights[1, :])
                @test avg <= prev + 1e-8
                prev = avg
            end
        end

        @testset "35e. donor pool screening" begin
            side = 8; N = side^2; T_per = 16; T0 = 12
            W = make_rook_weights(side); coords = scgrid(side)
            rng = MersenneTwister(3509)
            Y = randn(rng, N, T_per)

            # Unit 28 sits at (4,4) on the lattice; its rook neighbours are
            # 20, 27, 29 and 36 and none of them may be a donor.
            s = fit_synthetic_control(Y, 28, T0; W = W, coords = coords)
            @test sort(s.excluded_neighbours) == [20, 27, 29, 36]
            @test !(28 in s.donors)
            @test isempty(intersect(s.donors, s.excluded_neighbours))
            @test length(s.donors) == N - 1 - 4
            @test s.donor_pool_rule === :exclude_neighbours

            # :all keeps every untreated unit, neighbours included, and
            # records no exclusions.
            sa = fit_synthetic_control(Y, 28, T0; donor_pool = :all,
                                       penalty_metric = :none)
            @test length(sa.donors) == N - 1
            @test isempty(sa.excluded_neighbours)
            @test !(28 in sa.donors)

            # An explicit pool is honoured, and the treated unit is removed
            # from it even if the caller left it in.
            se = fit_synthetic_control(Y, 28, T0; donor_pool = [1, 2, 3, 28],
                                       penalty_metric = :none)
            @test se.donors == [1, 2, 3]
            @test se.donor_pool_rule === :explicit

            # Several treated units: none of them may be a donor, and the
            # exclusion set is the union of their neighbourhoods.
            sm = fit_synthetic_control(Y, [28, 30], T0; W = W, coords = coords)
            @test isempty(intersect(sm.donors, [28, 30]))
            @test 29 in sm.excluded_neighbours      # neighbours both
            @test 22 in sm.excluded_neighbours      # neighbours 30 only
            @test size(sm.weights) == (2, length(sm.donors))

            # The default needs W, and says so rather than silently
            # falling back to a pool that includes contaminated units.
            err = nothing
            try
                fit_synthetic_control(Y, 28, T0)
            catch e
                err = e
            end
            @test err isa ArgumentError
            @test occursin("needs `W`", err.msg)

            # Guards.
            @test_throws ArgumentError fit_synthetic_control(Y, 28, T0;
                                                             donor_pool = :nonsense)
            @test_throws ArgumentError fit_synthetic_control(Y, 28, T_per)
            @test_throws ArgumentError fit_synthetic_control(Y, 28, 0; W = W,
                                                             coords = coords)
            @test_throws ArgumentError fit_synthetic_control(Y, N + 1, T0; W = W,
                                                             coords = coords)
            @test_throws ArgumentError fit_synthetic_control(Y, 28, T0; W = W,
                                                             coords = coords,
                                                             penalty = -1.0)
            @test_throws ArgumentError fit_synthetic_control(Y, [28, 28], T0; W = W,
                                                             coords = coords)
            # :spatial penalty without coords.
            @test_throws ArgumentError fit_synthetic_control(Y, 28, T0; W = W,
                                                             penalty = 0.5)
            # Treating every unit leaves no donors at all.
            @test_throws ArgumentError fit_synthetic_control(Y, collect(1:N), T0;
                                                             donor_pool = :all)
        end

        @testset "35f. donor-pool contamination, exactly" begin
            # The point of screening the donor pool, made algebraic rather
            # than statistical. The spillover lands only in the
            # post-treatment periods, so the weights -- fitted on the
            # pre-period -- are untouched by it. The contamination a
            # neighbour carries into the synthetic control is therefore
            # exactly its weight times its exposure times the spillover,
            # and the bias follows without a tolerance.
            #
            # The design makes the trap unavoidable: the treated unit has a
            # path that is an exact half-and-half combination of two of its
            # own neighbours, which are in turn exact copies of two far
            # corners. An unscreened fit can reach a perfect pre-period fit
            # through the neighbours, and does.
            side = 9; N = side^2; T_per = 20; T0 = 14
            W = make_rook_weights(side); coords = scgrid(side)
            Wt = SpatialEconometrics.row_standardize(W)
            i = 41                               # (5,5): interior, 4 neighbours
            nbs = findall(!iszero, Matrix(W)[i, :])
            @test nbs == [32, 40, 42, 50]
            nb1, nb2 = nbs[1], nbs[2]
            a, b = 1, 9                          # far corners, exposure 0
            tau = 3.0
            D = zeros(N); D[i] = 1.0
            E = Wt * D
            @test E[nb1] == 0.25 && E[nb2] == 0.25
            @test E[a] == 0.0 && E[b] == 0.0 && E[i] == 0.0

            rng = MersenneTwister(3510)
            base = randn(rng, N, T_per)
            base[nb1, :] .= base[a, :]
            base[nb2, :] .= base[b, :]
            base[i, :]   .= 0.5 .* base[a, :] .+ 0.5 .* base[b, :]

            # The bias is linear in the spillover, with slope
            # -(weight on contaminated donors) * exposure. Pinned at four
            # spillover sizes, each to machine precision.
            for spill in (0.0, 1.0, 2.0, 4.0)
                Y = copy(base)
                for t in (T0 + 1):T_per
                    Y[i, t] += tau
                    Y[:, t] .+= spill .* E
                end
                s_all = fit_synthetic_control(Y, i, T0; donor_pool = :all,
                                              coords = coords, penalty = 0.01)
                wd = Dict(j => s_all.weights[1, q]
                          for (q, j) in enumerate(s_all.donors))
                # It reaches a perfect pre-period fit through the two
                # contaminated neighbours, taking half from each.
                @test s_all.rmspe_pre[1] < 1e-12
                @test wd[nb1] ≈ 0.5 atol = 1e-9
                @test wd[nb2] ≈ 0.5 atol = 1e-9
                # And is wrong by exactly the contamination it absorbed.
                contam = sum(s_all.weights[1, q] * spill * E[j]
                             for (q, j) in enumerate(s_all.donors))
                @test contam ≈ spill * 0.25 atol = 1e-9
                @test s_all.att ≈ tau - contam atol = 1e-12
                @test s_all.att ≈ tau - spill * 0.25 atol = 1e-9
            end

            # The screened fit cannot touch the neighbours, so it is
            # right -- and its pre-period fit is *worse*. A good
            # pre-period fit is not evidence that a synthetic control is
            # the right one, which is the reason this testset exists.
            spill = 2.0
            Y = copy(base)
            for t in (T0 + 1):T_per
                Y[i, t] += tau
                Y[:, t] .+= spill .* E
            end
            s_screen = fit_synthetic_control(Y, i, T0; W = W, coords = coords,
                                             penalty = 0.01)
            s_all = fit_synthetic_control(Y, i, T0; donor_pool = :all,
                                          coords = coords, penalty = 0.01)
            @test sort(s_screen.excluded_neighbours) == nbs
            @test abs(s_screen.att - tau) < 0.01
            @test s_all.att ≈ tau - 0.5 atol = 1e-9
            @test abs(s_all.att - tau) > 0.49
            @test s_all.rmspe_pre[1] < s_screen.rmspe_pre[1]

            # The weights are fitted pre-period, so adding a post-period
            # spillover cannot change them at all. This is what makes the
            # identity above exact rather than approximate.
            Y_nospill = copy(base)
            for t in (T0 + 1):T_per
                Y_nospill[i, t] += tau
            end
            s_ns = fit_synthetic_control(Y_nospill, i, T0; donor_pool = :all,
                                         coords = coords, penalty = 0.01)
            @test maximum(abs, s_ns.weights .- s_all.weights) == 0.0
            @test s_ns.att ≈ tau atol = 1e-9
            @test (s_all.att - s_ns.att) ≈ -0.5 atol = 1e-9

            # The spillover onto the excluded neighbours is recoverable
            # from the same clean pool -- but only for the neighbours the
            # pool can actually fit, and the pre-period RMSPE says which
            # those are before any gap is read.
            #
            # Units nb1 and nb2 are exact copies of the far corners a and
            # b, which are in the clean pool, so their synthetic controls
            # are exact and their gaps are the spillover alone. The other
            # two neighbours have no such counterpart, fit badly, and
            # their gaps are not the spillover. That is why
            # `SpilloverGaps` carries `rmspe_pre` per unit rather than
            # only the average.
            s_exact = fit_synthetic_control(Y, i, T0; W = W, coords = coords,
                                            penalty = 0.0)
            sg = spillover_gaps(s_exact, Y; W = W, coords = coords)
            @test sort(sg.units) == nbs
            @test length(sg.units) == 4
            @test size(sg.gaps) == (4, T_per)
            @test maximum(abs, sg.spillover_path .-
                               vec(mean(sg.gaps; dims = 1))) < 1e-14
            @test sg.spillover ≈ mean(sg.spillover_path[(T0 + 1):T_per]) atol = 1e-14

            good = [q for (q, u) in enumerate(sg.units) if u in (nb1, nb2)]
            poor = [q for (q, u) in enumerate(sg.units) if !(u in (nb1, nb2))]
            @test length(good) == 2 && length(poor) == 2
            # The two that can be fitted are fitted exactly, and return
            # the spillover to machine precision.
            for q in good
                @test sg.rmspe_pre[q] < 1e-13
                @test maximum(abs, sg.gaps[q, 1:T0]) < 1e-13
                @test sg.gaps[q, T_per] ≈ spill * 0.25 atol = 1e-10
                @test mean(sg.gaps[q, (T0 + 1):T_per]) ≈ spill * 0.25 atol = 1e-10
            end
            # The two that cannot are off by more than the spillover
            # itself, and their pre-period fit is three orders of
            # magnitude worse -- visible before the gap is looked at.
            for q in poor
                @test sg.rmspe_pre[q] > 0.4
                @test abs(sg.gaps[q, T_per] - spill * 0.25) > 0.5
            end
            @test minimum(sg.rmspe_pre[poor]) > 1e10 * maximum(sg.rmspe_pre[good])
            # So the unweighted average across all four is pulled off the
            # truth, by much more than the well-fitted pair are. The fix
            # is the user reading `rmspe_pre`, not the function silently
            # dropping units.
            #
            # This aggregate is deliberately bounded rather than pinned:
            # it depends on the two gaps that cannot be fitted, and those
            # follow the `randn(::MersenneTwister)` stream, which differs
            # between Julia minor versions. Measured 0.5452 on 1.13.2 and
            # 0.5887 on 1.10.12 — the well-fitted pair returns exactly
            # 0.500000000000 on both, which is why that is what gets
            # pinned above and this does not.
            @test 0.52 < sg.spillover < 0.62
            @test abs(sg.spillover - spill * 0.25) >
                  10 * abs(mean(sg.gaps[good, (T0 + 1):T_per]) - spill * 0.25)
            # Asking for spillovers when nothing was excluded is an error,
            # not an empty result that would read as "no spillover".
            @test_throws ArgumentError spillover_gaps(s_all, Y)
            @test_throws DimensionMismatch spillover_gaps(s_screen,
                                                          Y[:, 1:(T_per - 1)])
        end

        @testset "35g. aggregation over treated units" begin
            side = 8; N = side^2; T_per = 18; T0 = 13
            W = make_rook_weights(side); coords = scgrid(side)
            rng = MersenneTwister(3511)
            Y = randn(rng, N, T_per)
            tr = [11, 30, 54]
            s = fit_synthetic_control(Y, tr, T0; W = W, coords = coords,
                                      penalty = 0.05)
            @test s.treated == tr
            @test size(s.weights) == (3, length(s.donors))
            # Each row is a point on the simplex.
            for k in 1:3
                @test abs(sum(s.weights[k, :]) - 1.0) < 1e-13
                @test minimum(s.weights[k, :]) >= 0.0
            end
            # The ATT path is the mean of the per-unit gaps, exactly, and
            # the ATT is its post-period mean.
            @test maximum(abs, s.att_path .- vec(mean(s.gaps; dims = 1))) == 0.0
            @test s.att ≈ mean(s.att_path[(T0 + 1):T_per]) atol = 1e-15
            # Per-unit gaps are per-unit definitions.
            for k in 1:3
                syn = (Y[s.donors, :])' * s.weights[k, :]
                @test maximum(abs, s.gaps[k, :] .- (Y[tr[k], :] .- syn)) == 0.0
                @test s.rmspe_pre[k] ≈ sqrt(mean(abs2, s.gaps[k, 1:T0])) atol = 1e-15
                @test s.rmspe_post[k] ≈
                      sqrt(mean(abs2, s.gaps[k, (T0 + 1):T_per])) atol = 1e-15
                @test s.ratio[k] ≈ s.rmspe_post[k] / s.rmspe_pre[k] atol = 1e-13
            end
            @test StatsAPI.nobs(s) == 3 * T0

            # A 0/1 indicator and an index vector must select the same
            # units and give the same answer.
            ind = zeros(N); ind[tr] .= 1.0
            s2 = fit_synthetic_control(Y, ind, T0; W = W, coords = coords,
                                       penalty = 0.05)
            @test s2.treated == tr
            @test maximum(abs, s2.weights .- s.weights) == 0.0
            bind = falses(N); bind[tr] .= true
            s3 = fit_synthetic_control(Y, bind, T0; W = W, coords = coords,
                                       penalty = 0.05)
            @test s3.treated == tr
        end

        @testset "35h. predictors and standardisation" begin
            side = 8; N = side^2; T_per = 16; T0 = 12
            W = make_rook_weights(side); coords = scgrid(side)
            rng = MersenneTwister(3512)
            Y = randn(rng, N, T_per)

            # An explicit predictor matrix is used in place of the
            # pre-period path, and changes the number of fitted moments.
            P = randn(rng, N, 4)
            sp = fit_synthetic_control(Y, 28, T0; W = W, coords = coords,
                                       predictors = P)
            @test StatsAPI.nobs(sp) == 4
            @test size(sp.weights) == (1, length(sp.donors))

            # Standardisation must be scale-free: multiplying one
            # predictor by a constant cannot change the weights once each
            # is scaled to unit variance, and must change them when it is
            # not.
            P2 = copy(P); P2[:, 1] .*= 1000.0
            s_std_a = fit_synthetic_control(Y, 28, T0; W = W, coords = coords,
                                            predictors = P, standardize = true)
            s_std_b = fit_synthetic_control(Y, 28, T0; W = W, coords = coords,
                                            predictors = P2, standardize = true)
            @test maximum(abs, s_std_a.weights .- s_std_b.weights) < 1e-8
            s_raw_a = fit_synthetic_control(Y, 28, T0; W = W, coords = coords,
                                            predictors = P, standardize = false)
            s_raw_b = fit_synthetic_control(Y, 28, T0; W = W, coords = coords,
                                            predictors = P2, standardize = false)
            @test maximum(abs, s_raw_a.weights .- s_raw_b.weights) > 1e-3

            # Covariate-distance penalty is available and differs from the
            # spatial one.
            sc = fit_synthetic_control(Y, 28, T0; W = W, coords = coords,
                                       penalty = 0.2,
                                       penalty_metric = :covariate)
            @test sc.penalty_metric === :covariate
            @test abs(sum(sc.weights[1, :]) - 1.0) < 1e-13
            # penalty = 0 records :none whatever the metric asked for,
            # since no distance is consulted.
            sz = fit_synthetic_control(Y, 28, T0; W = W, coords = coords,
                                       penalty = 0.0,
                                       penalty_metric = :spatial)
            @test sz.penalty_metric === :none

            @test_throws ArgumentError fit_synthetic_control(Y, 28, T0; W = W,
                                                             coords = coords,
                                                             penalty = 0.2,
                                                             penalty_metric = :bogus)
            @test_throws DimensionMismatch fit_synthetic_control(Y, 28, T0; W = W,
                                                                 coords = coords,
                                                                 predictors = randn(rng, N + 1, 3))
            # Haversine is accepted for lon/lat coordinates.
            sh = fit_synthetic_control(Y, 28, T0; W = W, coords = coords,
                                       penalty = 0.01,
                                       metric = HaversineDistance())
            @test abs(sum(sh.weights[1, :]) - 1.0) < 1e-13
        end

        @testset "35i. placebo test mechanics" begin
            side = 7; N = side^2; T_per = 14; T0 = 10
            W = make_rook_weights(side); coords = scgrid(side)
            rng = MersenneTwister(3513)
            F = randn(rng, 2, T_per); L = rand(rng, N, 2)
            Y = L * F .+ 0.05 .* randn(rng, N, T_per)
            i = 25
            for t in (T0 + 1):T_per
                Y[i, t] += 4.0
            end
            s = fit_synthetic_control(Y, i, T0; W = W, coords = coords)
            J = length(s.donors)

            pt = placebo_test(s, Y; W = W, coords = coords)
            # One treated unit and J donors gives exactly J assignments,
            # so the test is exact and the p-value lives on a grid of
            # 1/(J+1).
            @test pt.exact
            @test pt.n_placebo == J
            @test pt.n_possible == Float64(J)
            @test pt.m_treated == 1
            @test pt.n_donors == J
            @test pt.min_attainable_p ≈ 1 / (J + 1) atol = 1e-15
            @test length(pt.ratio_placebo) == J
            @test pt.ratio_observed == s.ratio_aggregate
            # The p-value is the rank formula, recomputed here rather than
            # taken on trust.
            @test pt.p_value ≈
                  (1 + count(>=(pt.ratio_observed), pt.ratio_placebo)) / (1 + J) atol = 1e-15
            @test pt.p_value >= pt.min_attainable_p
            # A planted effect this large should land at the floor.
            @test pt.p_value ≈ pt.min_attainable_p atol = 1e-15

            # Sampling path: cap the assignments below the number
            # possible and the test reports itself as inexact.
            sm = fit_synthetic_control(Y, [25, 27], T0; W = W, coords = coords)
            Jm = length(sm.donors)
            pts = placebo_test(sm, Y; n_placebo = 50, rng = MersenneTwister(1),
                               W = W, coords = coords)
            @test !pts.exact
            @test pts.n_placebo == 50
            @test pts.n_possible ≈ Jm * (Jm - 1) / 2 rtol = 1e-12
            @test pts.min_attainable_p ≈ 1 / 51 atol = 1e-15
            # Seeded, so it repeats exactly.
            pts2 = placebo_test(sm, Y; n_placebo = 50, rng = MersenneTwister(1),
                                W = W, coords = coords)
            @test pts2.p_value == pts.p_value
            @test maximum(abs, pts2.ratio_placebo .- pts.ratio_placebo) == 0.0
            # And a different seed generally gives a different sample.
            pts3 = placebo_test(sm, Y; n_placebo = 50, rng = MersenneTwister(9),
                                W = W, coords = coords)
            @test pts3.n_placebo == 50

            # _n_subsets against binomial, including the overflow guard.
            @test SC._n_subsets(10, 3) ≈ 120.0 atol = 1e-9
            @test SC._n_subsets(5, 0) == 1.0
            @test SC._n_subsets(5, 6) == 0.0
            @test SC._n_subsets(44, 1) == 44.0
            @test isfinite(SC._n_subsets(2000, 900)) == false
            # _enumerate_subsets really enumerates, without repeats.
            su = SC._enumerate_subsets([3, 7, 9, 11], 2)
            @test length(su) == 6
            @test length(unique(su)) == 6
            @test all(v -> length(v) == 2 && issorted(v), su)
            @test [3, 7] in su && [9, 11] in su

            @test_throws DimensionMismatch placebo_test(s, Y[:, 1:(T_per - 1)];
                                                        W = W, coords = coords)
            @test_throws ArgumentError placebo_test(s, Y; n_placebo = 0,
                                                    W = W, coords = coords)
            # No clean donor to permute against.
            stiny = fit_synthetic_control(Y, 1, T0; donor_pool = [2],
                                          penalty_metric = :none)
            @test_throws ArgumentError placebo_test(stiny, Y; donor_pool = [2])
        end

        @testset "35j. placebo size under the null" begin
            # A p-value is a test statistic, so the standing rule in
            # docs/AUDIT.md applies: its null distribution is measured,
            # not assumed. Here a short version of the study recorded in
            # the ledger -- enough replications to catch a grossly
            # miscalibrated test without making the suite slow.
            side = 6; N = side^2; T_per = 12; T0 = 9
            W = make_rook_weights(side); coords = scgrid(side)
            rng = MersenneTwister(3514)
            ps = Float64[]
            minp = NaN
            for _ in 1:60
                F = randn(rng, 2, T_per); L = rand(rng, N, 2)
                Y = L * F .+ 0.3 .* randn(rng, N, T_per)   # no effect at all
                s = fit_synthetic_control(Y, 15, T0; W = W, coords = coords)
                pt = placebo_test(s, Y; W = W, coords = coords)
                push!(ps, pt.p_value)
                minp = pt.min_attainable_p
            end
            # Every p-value is a valid probability on the attainable grid.
            @test all(p -> minp - 1e-12 <= p <= 1.0 + 1e-12, ps)
            # With no effect the test must not reject wholesale. This is a
            # loose bound by design: the precise size, and the fact that
            # it is not nominal, is measured in docs/AUDIT.md rather than
            # asserted here.
            @test count(<=(0.10), ps) / length(ps) < 0.5
            @test mean(ps) > 0.1
        end
    end

    @testset "36. Spatial shift-share diagnostics" begin
        # Three of the four quantities in this module are exact algebraic
        # identities published as such -- the Borusyak-Hull-Jaravel
        # shock-level equivalence, the Goldsmith-Pinkham-Sorkin-Swift
        # decomposition, and the reduction of the Adao-Kolesar-Morales
        # variance to a cluster-robust one under full specialisation. Those
        # are checked against independent computation at machine precision,
        # because an identity that holds to 1e-13 and one that holds to
        # 1e-3 are different claims and only the first is the published
        # one. The two test statistics owe a null-distribution Monte Carlo
        # under the standing rule, and their measured tables live in
        # docs/AUDIT.md; what is asserted here is the loose bound that
        # would catch a regression.
        SS = SpatialEconometrics.SpatialShiftShare

        # Shares with a tunable concentration across shocks. Rows sum to
        # one, which is the complete-shares case.
        function ss_shares(rng, N, K; concentration = 0.0)
            a = exp.(concentration .* randn(rng, K))
            W = zeros(N, K)
            for i in 1:N
                row = a .* (0.2 .+ rand(rng, K))
                W[i, :] = row ./ sum(row)
            end
            return W
        end

        # Residuals carrying a sector-mix component: regions with similar
        # industry mixes are correlated regardless of geography, which is
        # the dependence the AKM standard error exists for. `mix` is the
        # ratio of sectoral to idiosyncratic standard deviation; the
        # sqrt(K) rescaling keeps that ratio fixed as K grows, since W*eta
        # has standard deviation about sqrt(1/K) when rows sum to one.
        function ss_resid(rng, W; mix = 3.0, nu = 1.0)
            N, K = size(W)
            return W * (randn(rng, K) .* (mix * sqrt(K))) .+ randn(rng, N) .* nu
        end

        @testset "36a. design validation" begin
            rng = MersenneTwister(4601)
            W = ss_shares(rng, 40, 6)
            g = randn(rng, 6)
            d = shift_share_design(W, g)

            @test size(d.W) == (40, 6)
            @test d.X ≈ W * g
            @test d.complete_shares
            @test d.s_bar ≈ vec(sum(W; dims = 1)) ./ 40
            # An intercept is always present, and nothing else by default.
            @test size(d.Z) == (40, 1)
            @test all(==(1.0), d.Z)
            # Controls are appended after the intercept.
            C = randn(rng, 40, 2)
            dc = shift_share_design(W, g; controls = C)
            @test size(dc.Z) == (40, 3)
            @test dc.Z[:, 2:3] ≈ C

            # Each refusal is by name rather than a downstream failure.
            @test_throws DimensionMismatch shift_share_design(W, randn(rng, 5))
            @test_throws ArgumentError shift_share_design(W[:, 1:1], g[1:1])
            @test_throws ArgumentError shift_share_design(fill(NaN, 40, 6), g)
            @test_throws ArgumentError shift_share_design(W, fill(Inf, 6))
            # An all-zero share column would divide by zero on aggregation.
            Wz = copy(W); Wz[:, 3] .= 0.0
            @test_throws ArgumentError shift_share_design(Wz, g)
            # Collinear controls are caught by the shared rank check.
            @test_throws ArgumentError shift_share_design(W, g;
                controls = hcat(C[:, 1], C[:, 1]))

            # Incomplete shares warn and report, rather than silently
            # adding the exposure sum as a control.
            Wi = W .* 0.5; Wi[1, :] .*= 0.5
            di = @test_logs (:warn, r"do not sum to a constant") match_mode = :any begin
                shift_share_design(Wi, g)
            end
            @test !di.complete_shares
            @test di.exposure_sum[1] < di.exposure_sum[2]
        end

        @testset "36b. shock-level equivalence, exactly" begin
            # Borusyak, Hull & Jaravel (2022) eq. 4: the regional estimate
            # equals an exposure-weighted estimate formed from K
            # shock-level observations. This is an identity, so the gate is
            # machine precision, and it is checked with controls present
            # since the residualisation is where it could go wrong.
            worst_direct = 0.0; worst_iv = 0.0; worst_fn6 = 0.0
            for rep in 1:60
                rng = MersenneTwister(4700 + rep)
                N = rand(rng, 40:90); K = rand(rng, 5:15)
                W = ss_shares(rng, N, K)
                g = randn(rng, K)
                d = shift_share_design(W, g; controls = randn(rng, N, 2))
                y = randn(rng, N); t = d.X .* 0.7 .+ randn(rng, N)

                f = fit_shift_share(y, d)
                a = shock_aggregate(d, y)
                worst_direct = max(worst_direct,
                    abs(a.beta_equiv - f.beta) / max(1e-12, abs(f.beta)))
                @test !a.is_iv

                fiv = fit_shift_share(y, d; treatment = t)
                aiv = shock_aggregate(d, y; treatment = t)
                worst_iv = max(worst_iv,
                    abs(aiv.beta_equiv - fiv.beta) / max(1e-12, abs(fiv.beta)))
                @test aiv.is_iv
                # Footnote 6: the exposure-weighted aggregates of a
                # residual vector have weighted mean exactly zero.
                worst_fn6 = max(worst_fn6, abs(dot(a.s_bar, a.y_bar)))
            end
            @test worst_direct < 1e-10
            @test worst_iv < 1e-10
            @test worst_fn6 < 1e-12
            # Measured: 2.29e-13, 2.26e-13 and 2.04e-16.
            @test worst_direct < 1e-11
            @test worst_fn6 < 1e-14
        end

        @testset "36c. Rotemberg decomposition, exactly" begin
            # Goldsmith-Pinkham, Sorkin & Swift (2020) Proposition 3. Both
            # the unit sum of the weights and the decomposition itself are
            # identities.
            worst_sum = 0.0; worst_dec = 0.0; saw_negative = false
            for rep in 1:60
                rng = MersenneTwister(4800 + rep)
                N = rand(rng, 40:90); K = rand(rng, 5:15)
                W = ss_shares(rng, N, K)
                d = shift_share_design(W, randn(rng, K);
                                       controls = randn(rng, N, 2))
                y = randn(rng, N); t = d.X .* 0.7 .+ randn(rng, N)

                r = rotemberg_weights(y, t, d)
                f = fit_shift_share(y, d; treatment = t)
                worst_sum = max(worst_sum, abs(r.alpha_sum - 1))
                worst_dec = max(worst_dec,
                    abs(r.beta_overall - f.beta) / max(1e-12, abs(f.beta)))
                r.n_negative > 0 && (saw_negative = true)
                @test length(r.alpha) == K
                @test r.order[1] == argmax(abs.(r.alpha))
            end
            @test worst_sum < 1e-10
            @test worst_dec < 1e-10
            # Measured: 1.42e-14 and 2.64e-13.
            @test worst_sum < 1e-12
            # Negative weights are expected, not a defect: the paper says
            # so. If they never appeared the test would be checking a
            # design too tame to be representative.
            @test saw_negative
        end

        @testset "36d. AKM reduces to clustering, exactly" begin
            # Under full specialisation -- every region in exactly one
            # sector -- Adao, Kolesar & Morales note that their variance
            # becomes the cluster-robust variance clustering on that
            # sector. Checked against the package's own vcov_cluster with
            # the degrees-of-freedom correction off, so the reference is
            # independent code rather than a rearrangement of the same
            # expression.
            worst = 0.0
            for rep in 1:40
                rng = MersenneTwister(4900 + rep)
                N = rand(rng, 40:80); K = rand(rng, 4:8)
                assign = rand(rng, 1:K, N)
                for k in 1:K; assign[k] = k; end   # every sector used
                W = zeros(N, K)
                for i in 1:N; W[i, assign[i]] = 1.0; end
                d = shift_share_design(W, randn(rng, K))
                y = d.X .* 1.5 .+ randn(rng, N)

                f = fit_shift_share(y, d; vcov = :akm)
                spec = SpatialEconometrics.ClusterRobustVCOV(assign;
                                                             dof_correction = false)
                Vc = SpatialEconometrics.vcov_cluster(
                    reshape(f.X_tilde, :, 1), residuals(f), spec)
                worst = max(worst, abs(f.se^2 - Vc[1, 1]) / max(1e-300, Vc[1, 1]))
            end
            @test worst < 1e-10
            # Measured: 2.97e-15.
            @test worst < 1e-12
        end

        @testset "36e. the shock-level projection" begin
            # Step 2 of their Remark 5 computes X_ring by regressing the
            # residualised regressor on the share matrix. It is NOT the
            # shock vector g: the intercept in Z demeans the regressor
            # first. Pinning the relationship is what proves the step is
            # implemented as published rather than shortcut to g.
            worst_ne = 0.0; worst_sq = 0.0; worst_c = 0.0
            for rep in 1:40
                rng = MersenneTwister(5000 + rep)
                K = rand(rng, 3:8)
                g = randn(rng, K)

                # Square and invertible: the projection is exact.
                Wsq = rand(rng, K, K) .+ 0.5
                Wsq ./= sum(Wsq; dims = 2)
                dsq = shift_share_design(Wsq, g)
                fsq = fit_shift_share(randn(rng, K), dsq)
                worst_sq = max(worst_sq,
                    maximum(abs.(Wsq * fsq.X_ring .- fsq.X_tilde)) /
                    max(1e-12, maximum(abs, fsq.X_tilde)))

                # Tall, with shares summing to c. Then W*(1/c) = 1, so the
                # projection is g shifted by the regressor's mean over c.
                N = rand(rng, 30:70)
                c = rand(rng) * 2 + 0.5
                W = ss_shares(rng, N, K) .* c
                d = shift_share_design(W, g)
                f = fit_shift_share(randn(rng, N), d)
                ne = W' * (W * f.X_ring .- f.X_tilde)
                worst_ne = max(worst_ne, maximum(abs, ne) /
                    max(1e-12, maximum(abs, W' * f.X_tilde)))
                pred = g .- mean(d.X) / c
                worst_c = max(worst_c,
                    maximum(abs.(f.X_ring .- pred)) / maximum(abs, pred))
            end
            @test worst_ne < 1e-10
            @test worst_sq < 1e-10
            @test worst_c < 1e-10
            # Measured: 2.99e-15, 2.28e-15, 1.85e-15.
            @test worst_c < 1e-12

            # More shocks than regions leaves the projection
            # underdetermined, and more than one shock with the same
            # exposure pattern leaves it non-unique. Both are refused by
            # name rather than returning a least-norm vector.
            rng = MersenneTwister(5101)
            Wwide = ss_shares(rng, 6, 10)
            dwide = shift_share_design(Wwide, randn(rng, 10))
            @test_throws ArgumentError fit_shift_share(randn(rng, 6), dwide)
            # Renormalise after duplicating the column: overwriting it
            # changes each row's sum, which would trip the incomplete-shares
            # warning for an unrelated reason. Dividing both copies by the
            # same row sum leaves them identical, so the rank guard is still
            # what fires.
            Wdup = ss_shares(rng, 40, 5); Wdup[:, 2] = Wdup[:, 1]
            Wdup ./= sum(Wdup; dims = 2)
            @test Wdup[:, 1] == Wdup[:, 2]
            ddup = shift_share_design(Wdup, randn(rng, 5))
            @test_throws ArgumentError fit_shift_share(randn(rng, 40), ddup)
        end

        @testset "36f. exposure concentration" begin
            # Uniform exposure gives exactly K effective shocks, up to the
            # representability of 1/K.
            for K in (4, 10, 50)
                d = shift_share_design(fill(1.0 / K, 30, K),
                                       randn(MersenneTwister(K), K))
                c = exposure_concentration(d)
                @test c.n_shocks == K
                @test isapprox(c.effective_shocks, K; rtol = 1e-12)
                @test isapprox(c.hhi, 1 / K; rtol = 1e-12)
                @test isapprox(c.max_share, 1 / K; rtol = 1e-12)
            end

            # Concentrating exposure on one shock drives the effective
            # count monotonically toward one.
            rng = MersenneTwister(5201)
            effs = Float64[]
            for frac in (0.2, 0.5, 0.9, 0.99)
                K = 20
                W = fill((1 - frac) / (K - 1), 50, K); W[:, 1] .= frac
                c = exposure_concentration(shift_share_design(W, randn(rng, K)))
                push!(effs, c.effective_shocks)
                @test isapprox(c.max_share, frac; rtol = 1e-10)
                @test c.order[1] == 1
            end
            @test issorted(effs; rev = true)
            @test effs[end] < 1.1
            @test effs[1] > 5

            # The normalisation makes the measure scale-free: multiplying
            # every share by a constant cannot change concentration.
            W = ss_shares(MersenneTwister(5202), 40, 8)
            c1 = exposure_concentration(shift_share_design(W, ones(8)))
            c2 = exposure_concentration(shift_share_design(W .* 3.0, ones(8)))
            @test isapprox(c1.hhi, c2.hhi; rtol = 1e-12)
        end

        @testset "36g. balance test mechanics" begin
            rng = MersenneTwister(5301)
            N = 80; K = 14
            W = ss_shares(rng, N, K)
            d = shift_share_design(W, randn(rng, K))

            # One characteristic: the joint Wald statistic is the squared
            # robust t exactly, which is the internal consistency check on
            # the cross-equation covariance.
            worst = 0.0
            for rep in 1:40
                rngr = MersenneTwister(5400 + rep)
                dd = shift_share_design(W, randn(rngr, K))
                b = balance_test(dd, randn(rngr, N))
                worst = max(worst, abs(b.joint_chi2 - b.t_stats[1]^2) /
                                   max(1e-12, b.t_stats[1]^2))
                @test b.joint_df == 1
                @test b.n_shocks == K
                @test 0.0 <= b.p_values[1] <= 1.0
            end
            @test worst < 1e-10
            # Measured: 1.09e-15.
            @test worst < 1e-12

            # Several characteristics at once, with names carried through.
            R = randn(rng, N, 3)
            b3 = balance_test(d, R; names = ["pretrend", "income", "share65"])
            @test length(b3.coefs) == 3
            @test b3.names == ["pretrend", "income", "share65"]
            @test b3.joint_df == 3
            @test all(p -> 0.0 <= p <= 1.0, b3.p_values)
            @test 0.0 <= b3.joint_p <= 1.0

            # A characteristic built to correlate with the shocks must be
            # detected: the test has power, not merely size.
            dpow = shift_share_design(W, randn(MersenneTwister(5501), K))
            rpow = dpow.X .* 2.0 .+ randn(MersenneTwister(5502), N) .* 0.1
            bpow = balance_test(dpow, rpow)
            @test bpow.p_values[1] < 0.01

            # Shock-level controls take one row per shock, and the wrong
            # orientation is refused by name.
            bq = balance_test(d, randn(rng, N); controls = randn(rng, K, 2))
            @test bq.n_shocks == K
            @test_throws DimensionMismatch balance_test(d, randn(rng, N);
                                                        controls = randn(rng, N, 2))
            @test_throws DimensionMismatch balance_test(d, randn(rng, N + 1))
            @test_throws DimensionMismatch balance_test(d, randn(rng, N);
                                                        names = ["a", "b"])
        end

        @testset "36h. chi-square tail against known critical values" begin
            # The upper tail comes from the shared `SpatialUtils._chi2_sf`
            # rather than from SpecialFunctions, so it is checked against the
            # published 5% critical values. This module briefly carried its
            # own closed-form copy, which leaned on `_normcdf` and so was
            # accurate only to ~1.5e-7 on odd df; the shared version is good
            # to 6e-9 there, which is why the tolerance below can be tight.
            for (x, k) in ((3.841459, 1), (5.991465, 2), (7.814728, 3),
                           (9.487729, 4), (11.070498, 5), (18.307038, 10),
                           (31.410433, 20))
                @test isapprox(SS._chi2_sf(x, k), 0.05; atol = 1e-7)
            end
            @test SS._chi2_sf(0.0, 3) == 1.0
            @test SS._chi2_sf(-1.0, 3) == 1.0
            @test SS._chi2_sf(1e4, 5) < 1e-10
        end

        @testset "36i. AKM against HC1 when the dependence is sectoral" begin
            # The reason this module exists. With a sector-mix component in
            # the residuals, heteroskedasticity-robust standard errors are
            # badly undersized and the AKM standard error is not. Measured
            # tables are in docs/AUDIT.md; the assertions here are the
            # loose bounds that would catch a regression without pinning a
            # Monte Carlo rate.
            rng = MersenneTwister(5601)
            N = 300; K = 100
            W = ss_shares(rng, N, K)
            eps = ss_resid(rng, W; mix = 3.0)
            groups = [argmax(@view W[i, :]) for i in 1:N]
            beta = 1.0

            reps = 150
            cov_akm = 0; cov_hc1 = 0; cov_cl = 0
            for _ in 1:reps
                g = randn(rng, K)
                d = shift_share_design(W, g)
                y = d.X .* beta .+ eps
                fa = fit_shift_share(y, d; vcov = :akm)
                fh = fit_shift_share(y, d; vcov = :hc1)
                spec = SpatialEconometrics.ClusterRobustVCOV(groups)
                secl = sqrt(max(0.0, SpatialEconometrics.vcov_cluster(
                    reshape(fa.X_tilde, :, 1), residuals(fa), spec)[1, 1]))
                # The point estimate does not depend on the covariance.
                @test fa.beta == fh.beta
                abs(fa.beta - beta) <= 1.96 * fa.se && (cov_akm += 1)
                abs(fa.beta - beta) <= 1.96 * fh.se && (cov_hc1 += 1)
                abs(fa.beta - beta) <= 1.96 * secl && (cov_cl += 1)
            end
            # Measured over 2000 reps: AKM 0.936, HC1 0.732, cluster 0.720.
            @test cov_akm / reps > 0.85
            @test cov_hc1 / reps < 0.88
            @test cov_akm > cov_hc1
            # Clustering on the dominant sector is not a substitute for the
            # AKM correction; it tracks HC1 rather than fixing it.
            @test cov_cl / reps < 0.88

            # The AKM standard error is larger than HC1 here, which is the
            # whole content of the correction on this design.
            g = randn(rng, K)
            d = shift_share_design(W, g)
            y = d.X .* beta .+ eps
            @test fit_shift_share(y, d; vcov = :akm).se >
                  fit_shift_share(y, d; vcov = :hc1).se

            # Interface: a 1x1 covariance, consistent accessors, and no
            # pretence of more coefficients than there are.
            f = fit_shift_share(y, d)
            @test size(vcov(f)) == (1, 1)
            @test coef(f) == [f.beta]
            @test stderror(f) == [f.se]
            @test vcov(f)[1, 1] ≈ f.se^2
            @test nobs(f) == N
            @test length(coefnames(f)) == 1
            @test length(residuals(f)) == N
            @test dof_residual(f) == N - size(d.Z, 2) - 1
            @test_throws ArgumentError fit_shift_share(y, d; vcov = :bogus)
            @test_throws DimensionMismatch fit_shift_share(randn(rng, N + 1), d)
        end

        @testset "36j. balance test size under the null" begin
            # The standing rule: a test statistic owes a null-distribution
            # Monte Carlo. The precise rates, and the fact that they are
            # nominal only when the effective number of shocks is large,
            # are recorded in docs/AUDIT.md. Asserted here is the bound
            # that would catch the test breaking outright.
            rng = MersenneTwister(5701)
            N = 300; K = 100
            W = ss_shares(rng, N, K)
            reps = 150
            rej05 = 0; ps = Float64[]
            for _ in 1:reps
                g = randn(rng, K)
                d = shift_share_design(W, g)
                # independent of g, but sectorally structured
                r = ss_resid(rng, W; mix = 1.0)
                b = balance_test(d, r)
                push!(ps, b.p_values[1])
                b.p_values[1] <= 0.05 && (rej05 += 1)
            end
            # Measured over 2000 reps at 100 effective shocks: 0.056.
            @test rej05 / reps < 0.15
            @test mean(ps) > 0.3
            @test all(p -> 0.0 <= p <= 1.0, ps)
            # A p-value distribution concentrated near zero or one would
            # both be wrong; under the null it should be roughly uniform.
            @test 0.2 < count(<=(0.5), ps) / reps < 0.8
        end

        @testset "36k. spatial correlation of shares and regressor" begin
            # Moran's I reused from SpatialDiagnostics. Industries cluster
            # geographically, so spatially correlated shares are the normal
            # case rather than a problem -- what matters is that a spatial
            # correction is the wrong tool here, which the docstring says
            # and this testset documents by construction.
            side = 8; N = side * side; K = 6
            coords = hcat(repeat(1:side, side), repeat(1:side, inner = side)) .* 1.0
            Wsp = make_queen_weights(side)
            rng = MersenneTwister(5801)

            # Build shares that vary smoothly in space, so the exposure
            # pattern is genuinely autocorrelated. The centres must be
            # distinct: drawing them independently lets two industries land
            # on the same cell, which makes two share columns identical and
            # the shock-level projection non-unique -- which is exactly what
            # fit_shift_share refuses, so it would be a defect in this DGP
            # rather than a finding.
            W = zeros(N, K)
            centres = randperm(rng, N)[1:K]
            for k in 1:K
                c = coords[centres[k], :]
                W[:, k] = [exp(-0.05 * sum(abs2, coords[i, :] .- c))
                           for i in 1:N] .+ 0.05
            end
            W ./= sum(W; dims = 2)
            # Guard the guard: the design this testset builds must be the
            # full-rank one, or the assertions below are vacuous.
            @test rank(W) == K
            d = shift_share_design(W, randn(rng, K))
            y = d.X .* 1.0 .+ randn(rng, N)
            f = fit_shift_share(y, d)

            m = shift_share_moran(d, Wsp)
            @test length(m.share_I) == K
            @test length(m.share_p) == K
            @test isnothing(m.residual)
            @test 1 <= m.worst_share <= K
            @test all(I -> -2.0 <= I <= 2.0, filter(isfinite, m.share_I))
            @test all(p -> 0.0 <= p <= 1.0, m.share_p)
            # Smoothly varying shares are positively autocorrelated.
            @test m.share_I[m.worst_share] > 0.0

            mf = shift_share_moran(d, Wsp; fit = f)
            @test !isnothing(mf.residual)
            @test mf.regressor.I_stat ≈ m.regressor.I_stat
            # Passing a weight matrix of the wrong size is refused by name,
            # since confusing the spatial weights with the exposure shares
            # is the obvious mistake here.
            @test_throws DimensionMismatch shift_share_moran(d,
                make_queen_weights(side + 1))
        end
    end

    @testset "37. Spatial econometrics for networks" begin
        # Bramoulle, Djebbari and Fortin (2009) is unusually good to test
        # against: their identification results are statements about the
        # rank of integer-structured matrices, so the gates here are exact
        # integers and machine-precision identities rather than tolerances.
        # Every verdict asserted below is one the paper states in words,
        # and several of the algebraic identities are printed in it.
        NW = SpatialEconometrics.SpatialNetworks

        # Erdos-Renyi graph with no isolates, which the rank results for the
        # non-identification half assume.
        #
        # Isolates are repaired rather than rejected. At n = 150 and density
        # 0.02 the chance that a draw happens to have none is about 1 in
        # 2,000, so rejection sampling here does not terminate in any sane
        # number of tries -- and burns a 150-node sparse build on each one.
        # Attaching each isolate to one random other node is O(isolates),
        # always succeeds, and changes the density by less than a rounding.
        function nw_er(rng, n, d)
            e = [(i, j) for i in 1:n for j in (i + 1):n if rand(rng) < d]
            deg = zeros(Int, n)
            for (i, j) in e
                deg[i] += 1; deg[j] += 1
            end
            for i in 1:n
                if deg[i] == 0
                    j = rand(rng, 1:(n - 1))
                    j >= i && (j += 1)
                    push!(e, (i, j))
                    deg[i] += 1; deg[j] += 1
                end
            end
            return network_weights(e, n)
        end

        nw_star(n) = network_weights([(1, j) for j in 2:n], n)
        nw_line(n) = network_weights([(i, i + 1) for i in 1:(n - 1)], n)

        @testset "37a. network weight construction" begin
            A = [0 1 1; 1 0 0; 1 0 0]
            G = network_weights(A)
            @test size(G) == (3, 3)
            @test vec(sum(G, dims = 2)) ≈ ones(3)
            @test G[1, 2] ≈ 0.5
            @test G[2, 1] ≈ 1.0
            # :none leaves the adjacency alone.
            @test network_weights(A; normalize = :none) == sparse(Float64.(A))

            # An edge list must be told how many units there are, because
            # isolates never appear in one.
            Ge = network_weights([(1, 2), (1, 3)], 4)
            @test size(Ge) == (4, 4)
            @test iszero(Ge[4, :])
            @test Matrix(Ge[1:3, 1:3]) ≈ Matrix(G)

            # Directed graphs are supported and are not symmetrised.
            Gd = network_weights([(1, 2), (2, 3)], 3; directed = true)
            @test Gd[1, 2] ≈ 1.0
            @test Gd[2, 1] == 0.0
            @test iszero(Gd[3, :])

            # Weighted edges, and duplicates summed.
            Gw = network_weights([(1, 2, 3.0), (1, 3, 1.0)], 3)
            @test Gw[1, 2] ≈ 0.75
            Gdup = network_weights([(1, 2), (1, 2), (1, 3)], 3; normalize = :none)
            @test Gdup[1, 2] ≈ 2.0

            # Named failures rather than silent degeneracy.
            @test_throws DimensionMismatch network_weights(zeros(2, 3))
            @test_throws ArgumentError network_weights([0 1; 1 1])      # self-link
            @test_throws ArgumentError network_weights([0 -1; -1 0])    # negative
            @test_throws ArgumentError network_weights(A; normalize = :bogus)
            @test_throws ArgumentError network_weights([(1, 9)], 3)     # out of range
            @test_throws ArgumentError network_weights([(1, 2, -1.0)], 3)
            # A self-link is allowed when it is asked for by name.
            @test network_weights([0 1; 1 1]; allow_self = true)[2, 2] ≈ 0.5
        end

        @testset "37b. the published algebraic identities" begin
            # Manski (1993): the group mean INCLUDES the unit, and then
            # G^2 = G exactly -- whatever the group sizes are. The paper
            # states this as the reason his model is not identified.
            for sizes in ([5, 5, 5], [4, 5, 6], [3, 7, 11])
                G = group_weights(sizes; include_self = true)
                @test maximum(abs, Matrix(G * G - G)) < 1e-15
            end

            # Moffitt (2001) / Lee (2007): the mean EXCLUDES the unit, and
            # within a block of size s the paper gives
            #   G^2 = I/(s-1) + ((s-2)/(s-1)) G.
            for s in (3, 5, 8)
                G = group_weights([s, s, s])
                n = 3s
                pred = (1 / (s - 1)) * Matrix(1.0I, n, n) +
                       ((s - 2) / (s - 1)) * Matrix(G)
                @test maximum(abs, Matrix(G * G) - pred) < 1e-15
            end

            # Their page 47, two groups of sizes s1 and s2: G^3 is an exact
            # combination of I, G and G^2, which is why two group sizes do
            # not suffice under network fixed effects.
            for (s1, s2) in ((4, 7), (3, 5), (6, 11))
                G = Matrix(group_weights([s1, s2]))
                n = s1 + s2
                d = (s1 - 1) * (s2 - 1)
                pred = (1 / d) * Matrix(1.0I, n, n) +
                       ((s1 + s2 - 3) / d) * G +
                       ((s1 * s2 - 2 * (s1 + s2) + 3) / d) * (G * G)
                @test maximum(abs, G * G * G - pred) < 1e-14
            end

            # Complete bipartite graphs, of which the star is one, satisfy
            # G^3 = G. The paper names them as the leading failure of
            # Proposition 4.
            for n in (5, 9, 14)
                G = Matrix(nw_star(n))
                @test maximum(abs, G * G * G - G) < 1e-15
            end
            Gb = Matrix(network_weights([(i, j) for i in 1:4 for j in 5:9], 9))
            @test maximum(abs, Gb * Gb * Gb - Gb) < 1e-15
        end

        @testset "37c. the rank condition, verdict by verdict" begin
            # Proposition 2: equal group sizes are not identified; at least
            # two different sizes are.
            @test !peer_identification(group_weights([5, 5, 5])).identified
            @test !peer_identification(group_weights([8, 8])).identified
            @test !peer_identification(group_weights([3, 3, 3, 3])).identified
            @test peer_identification(group_weights([4, 6])).identified
            @test peer_identification(group_weights([3, 9])).identified
            @test peer_identification(group_weights([4, 5, 6])).identified

            # Manski's convention fails whatever the sizes, since G^2 = G.
            for sizes in ([5, 5, 5], [4, 5, 6], [3, 7, 11])
                id = peer_identification(group_weights(sizes; include_self = true))
                @test !id.identified
                @test id.rank == 2
            end

            # The ranks themselves, not just the verdicts.
            @test peer_identification(group_weights([5, 5, 5])).rank == 2
            @test peer_identification(group_weights([4, 6])).rank == 3

            # Proposition 4 is strictly more demanding: two group sizes give
            # rank 3 where 4 is needed, and three sizes give 4. This is the
            # paper's "three different classroom sizes are now necessary".
            for sizes in ([4, 6], [3, 9], [5, 12])
                @test peer_identification(group_weights(sizes)).identified
                id4 = peer_identification(group_weights(sizes); fixed_effects = true)
                @test id4.rank == 3
                @test !id4.identified
                @test id4.required == 4
            end
            for sizes in ([4, 6, 9], [3, 5, 7], [2, 5, 11, 13])
                id4 = peer_identification(group_weights(sizes); fixed_effects = true)
                @test id4.rank == 4
                @test id4.identified
            end

            # Stars pass Proposition 1 and fail Proposition 4, since
            # G^3 = G but G^2 differs from both I and G.
            for n in (5, 9, 14)
                @test peer_identification(nw_star(n)).identified
                @test !peer_identification(nw_star(n); fixed_effects = true).identified
            end

            # Intransitive triads identify: the paper's own line example.
            @test peer_identification(nw_line(8)).identified
            @test peer_identification(nw_line(8); fixed_effects = true).identified
        end

        @testset "37d. the rank tolerance is measured, not assumed" begin
            # The Gram route never forms the n^2 x m matrix, at the cost of
            # squaring the condition number. Its singular values must match
            # a dense SVD of the explicit matrix.
            let rng = MersenneTwister(4113), worst = 0.0
                for G in (group_weights([4, 6, 9]), group_weights([5, 5, 5]),
                          nw_star(9), nw_line(12), nw_er(rng, 60, 0.08))
                    n = size(G, 1)
                    mats = Any[sparse(1.0I, n, n), G, G * G, G * G * G]
                    sg = NW._vec_svals(mats)
                    sd = svdvals(hcat((vec(Matrix(m)) for m in mats)...))
                    worst = max(worst, maximum(abs.(sg .- sd)) / sd[1])
                end
                # Degenerate designs sit at the sqrt(eps) noise floor; that
                # is the whole reason the tolerance is where it is.
                @test worst < 1e-7
            end

            # The regression this guards. Two equal groups of 8 are NOT
            # identified (Proposition 2), but the spurious singular value
            # the Gram route produces for them is 1.97e-8 -- above a 1e-8
            # tolerance, which therefore reports them as identified. The
            # default was moved to 1e-6 on that measurement.
            @test NW.DEFAULT_RANK_TOL == 1e-6
            @test !peer_identification(group_weights([8, 8])).identified
            @test peer_identification(group_weights([8, 8]); tol = 1e-8).identified
            @test peer_identification(group_weights([8, 8])).gap > 1e-8
            @test peer_identification(group_weights([8, 8])).gap < 1e-6

            # Identified structures sit far above the tolerance, which is
            # the other half of the separation.
            for G in (group_weights([4, 6]), nw_line(12), nw_star(9))
                @test peer_identification(G).gap > 1e-4
            end
        end

        @testset "37e. diameter and Corollary 1" begin
            # Corollary 1: a diameter of 3 or more is sufficient for
            # identification under network fixed effects.
            #
            # Two deliberate choices here, both of them scar tissue.
            #
            # First, the corollary is an implication, and it is asserted as
            # one rather than as a `@test` behind an `if` guard. A guarded
            # assertion that fails its guard does not fail -- it does not
            # run, the suite stays green, and the only trace is that the
            # assertion count is one lower. That is strictly worse than a
            # red test.
            #
            # Second, every structure in the tuple is deterministic. An
            # Erdos-Renyi draw stood here and was the thing that found the
            # problem: `rand(rng)` over a MersenneTwister does not produce
            # the same edge set on every Julia version -- the same seed gave
            # 326 edges on 1.13.2 and 288 on 1.10.12 -- and at 288 the draw
            # came out disconnected, so on 1.10 the guard failed and the
            # assertion silently vanished. A 4x4 grid (diameter 6) keeps a
            # structure that is neither a line nor a ring without betting
            # the gate on an RNG stream.
            let grid4 = network_weights(
                    vcat([((r - 1) * 4 + c, (r - 1) * 4 + c + 1)
                          for r in 1:4 for c in 1:3],
                         [((r - 1) * 4 + c, r * 4 + c)
                          for r in 1:3 for c in 1:4]), 16)
                for G in (nw_line(8), nw_line(30), grid4,
                          network_weights(vcat([(i, i + 1) for i in 1:9],
                                               [(10, 1)]), 10))
                    id = peer_identification(G; fixed_effects = true)
                    @test !(id.diameter >= 3 && id.connected) || id.identified
                end
                # ... and the grid genuinely enters the corollary rather
                # than satisfying the implication vacuously.
                let id = peer_identification(grid4; fixed_effects = true)
                    @test id.connected
                    @test id.diameter == 6
                end
            end
            # The star has diameter 2 and fails, which is the paper's own
            # illustration of where the corollary stops.
            let id = peer_identification(nw_star(9); fixed_effects = true)
                @test id.diameter == 2
                @test id.connected
                @test !id.identified
            end
            @test nw_line(8) |> G -> peer_identification(G).diameter == 7
            # Disconnected structures are flagged rather than reported as
            # though every pair were reachable.
            @test !peer_identification(group_weights([4, 6])).connected
            # Opting out returns a sentinel, not a wrong number.
            @test peer_identification(nw_line(8); compute_diameter = false).diameter == -1
        end

        @testset "37f. 2SLS against explicit normal equations" begin
            # The estimator is checked against the formula as printed in
            # their Section 4, written out independently here.
            let rng = MersenneTwister(31337)
                n = 240
                G = nw_er(rng, n, 0.02)
                Gd = Matrix(G)
                X = randn(rng, n, 2)
                y = (Matrix(1.0I, n, n) - 0.4 * Gd) \
                    (1.0 .+ X * [1.0, -0.5] .+ Gd * X * [0.8, 0.3] .+ randn(rng, n))

                Gy = Gd * y; GX = Gd * X; G2X = Gd * GX
                Xr = hcat(ones(n), Gy, X, GX)
                S  = hcat(ones(n), X, GX, G2X)
                P  = S * inv(S'S) * S'
                th = inv(Xr' * P * Xr) * (Xr' * P * y)

                f = fit_peer_effects(y, X, G)
                @test maximum(abs, coef(f) .- th) < 1e-10
                @test f.beta ≈ th[2]
                @test f.alpha ≈ th[1]
                @test f.gamma ≈ th[3:4]
                @test f.delta ≈ th[5:6]
                @test length(residuals(f)) == n
                @test nobs(f) == n

                # The fixed-effects branch is their eq. (10): the within
                # local transformation (I - G) applied throughout, with
                # S = [(I-G)X (I-G)GX (I-G)G^2X] as the paper writes it.
                M  = Matrix(1.0I, n, n) - Gd
                Xrf = hcat(M * Gy, M * X, M * GX)
                Sf  = hcat(M * X, M * GX, M * G2X)
                Pf  = Sf * inv(Sf'Sf) * Sf'
                thf = inv(Xrf' * Pf * Xrf) * (Xrf' * Pf * (M * y))

                ff = fit_peer_effects(y, X, G; fixed_effects = true)
                @test maximum(abs, coef(ff) .- thf) < 1e-10
                @test ff.alpha === nothing        # swept out by the transform
                @test ff.beta ≈ thf[1]
            end
        end

        @testset "37g. the identification check is not optional" begin
            let rng = MersenneTwister(9)
                G = group_weights([6, 6, 6])        # equal sizes: not identified
                n = size(G, 1)
                X = randn(rng, n, 2)
                y = randn(rng, n)
                # The fit refuses, and says why rather than failing inside
                # the linear algebra.
                @test_throws ArgumentError fit_peer_effects(y, X, G)

                # `check = false` does not rescue an exactly degenerate
                # network, and that is Bramoulle, Djebbari and Fortin's
                # Appendix B rather than a limitation of the override: when
                # I, G and G^2 are dependent, E(Gy|x) is perfectly collinear
                # with (iota, x, Gx), so the instrument block is collinear
                # with the included regressors and no estimate exists. The
                # design-rank guard is what fires, naming the instruments.
                @test_throws ArgumentError fit_peer_effects(y, X, G;
                                                            check = false)
                # The equivalence itself, both directions. E(Gy|x) is the
                # expected endogenous regressor, so the systematic part of
                # the reduced form with the error dropped.
                nw_EGy = function (Gm, Xm)
                    nm = size(Gm, 1)
                    sys = 1.0 .+ Xm * [1.0, -0.5] .+ Gm * Xm * [0.8, 0.3]
                    return Gm * ((Matrix(1.0I, nm, nm) - 0.4 * Gm) \ sys)
                end
                # Dependent  =>  E(Gy|x) collinear with (iota, x, Gx).
                let Gd = Matrix(G), E = nw_EGy(Matrix(G), X)
                    Z = hcat(ones(n), X, Gd * X)
                    @test maximum(abs, E .- Z * (Z \ E)) < 1e-12
                end
                # Independent  =>  it is not collinear.
                let Gi = group_weights([4, 6])
                    ni = size(Gi, 1)
                    Xi = randn(rng, ni, 2)
                    Gid = Matrix(Gi)
                    E = nw_EGy(Gid, Xi)
                    Zi = hcat(ones(ni), Xi, Gid * Xi)
                    @test peer_identification(Gi).identified
                    @test maximum(abs, E .- Zi * (Zi \ E)) > 1e-6
                end

                # The override does work where it is meant to: a network
                # whose design still has rank.
                let Gn = nw_er(rng, 120, 0.03)
                    nn = size(Gn, 1)
                    Xn = randn(rng, nn, 2)
                    Gnd = Matrix(Gn)
                    sys = 1.0 .+ Xn * [1.0, -0.5] .+ Gnd * Xn * [0.8, 0.3]
                    yn = (Matrix(1.0I, nn, nn) - 0.4 * Gnd) \ (sys .+ randn(rng, nn))
                    fo = fit_peer_effects(yn, Xn, Gn; check = false)
                    @test isfinite(fo.beta)
                    @test fo.ident.identified
                end
                # Two group sizes pass Proposition 1 but not Proposition 4,
                # so the same network is accepted one way and refused the
                # other -- the clearest statement that the fixed-effects
                # model asks for more.
                G2 = group_weights([5, 9])
                n2 = size(G2, 1)
                X2 = randn(rng, n2, 2); y2 = randn(rng, n2)
                @test fit_peer_effects(y2, X2, G2) isa PeerEffectsFit
                @test_throws ArgumentError fit_peer_effects(y2, X2, G2;
                                                            fixed_effects = true)
            end
        end

        @testset "37h. first-stage strength and the vcov options" begin
            let rng = MersenneTwister(606)
                n = 260
                Gsp = nw_er(rng, n, 0.015)      # sparse: strong instruments
                Gdn = nw_er(rng, n, 0.20)       # dense: weak ones
                mk = G -> begin
                    Gd = Matrix(G); X = randn(rng, n, 2)
                    y = (Matrix(1.0I, n, n) - 0.4 * Gd) \
                        (1.0 .+ X * [1.0, -0.5] .+ Gd * X * [0.8, 0.3] .+ randn(rng, n))
                    (y, X)
                end
                ys, Xs = mk(Gsp); yd, Xd = mk(Gdn)
                fs = fit_peer_effects(ys, Xs, Gsp; check = false)
                fd = fit_peer_effects(yd, Xd, Gdn; check = false)
                # The direction is the finding; the magnitudes live in
                # docs/AUDIT.md.
                @test fs.first_stage_F > fd.first_stage_F
                @test fs.first_stage_F > 10

                # The three covariance options differ but share a point
                # estimate: only the uncertainty changes.
                fh = fit_peer_effects(ys, Xs, Gsp; check = false,
                                      vcov = :homoskedastic)
                @test coef(fh) ≈ coef(fs)
                @test stderror(fh) != stderror(fs)
                @test all(>(0), stderror(fs))
                @test all(>(0), stderror(fh))
                @test size(vcov(fs)) == (length(coef(fs)), length(coef(fs)))

                # :transform corrects for the correlation the within-local
                # transformation induces, so it is meaningless without it
                # and is refused by name.
                @test_throws ArgumentError fit_peer_effects(ys, Xs, Gsp;
                                                            vcov = :transform)
                ft = fit_peer_effects(ys, Xs, Gsp; check = false,
                                      fixed_effects = true, vcov = :transform)
                @test all(>(0), stderror(ft))
                @test_throws ArgumentError fit_peer_effects(ys, Xs, Gsp;
                                                            vcov = :bogus)
                @test_throws ArgumentError fit_peer_effects(ys, Xs, Gsp;
                                                            instrument_order = 4)
                @test_throws DimensionMismatch fit_peer_effects(ys, Xs[1:10, :], Gsp)
                @test_throws DimensionMismatch fit_peer_effects(ys, Xs, Gsp;
                                                                varnames = ["a"])
            end
        end

        @testset "37i. Sargan mechanics" begin
            let rng = MersenneTwister(1212)
                n = 240
                G = nw_er(rng, n, 0.02); Gd = Matrix(G)
                X = randn(rng, n, 2)
                y = (Matrix(1.0I, n, n) - 0.4 * Gd) \
                    (1.0 .+ X * [1.0, -0.5] .+ Gd * X * [0.8, 0.3] .+ randn(rng, n))

                f2 = fit_peer_effects(y, X, G; instrument_order = 2)
                f3 = fit_peer_effects(y, X, G; instrument_order = 3)
                # K excluded instruments against one endogenous regressor
                # leaves K - 1 over-identifying restrictions.
                @test f2.sargan_df == 1
                @test f3.sargan_df == 3
                @test 0 <= f2.sargan_p <= 1
                @test f2.sargan >= 0
                # The statistic agrees with its own definition, computed here.
                let S = hcat(ones(n), X, Gd * X, Gd * Gd * X),
                    u = residuals(f2)
                    @test f2.sargan ≈ dot(u, S * (S \ u)) / (dot(u, u) / n) rtol = 1e-10
                end

                # With one characteristic the model is exactly identified at
                # order 2, and a Sargan test does not exist. Reporting df 0
                # beats reporting a p-value of 1 as though it meant something.
                f1 = fit_peer_effects(y, X[:, 1:1], G; instrument_order = 2)
                @test f1.sargan_df == 0
                @test f1.sargan == 0
            end
        end

        @testset "37j. multiplex reduces to the audited single-channel fit" begin
            let rng = MersenneTwister(8080)
                n = 150
                G = nw_er(rng, n, 0.06)
                X = randn(rng, n, 2)
                y = (Matrix(1.0I, n, n) - 0.5 * Matrix(G)) \
                    (2.0 .+ X * [1.0, -0.7] .+ randn(rng, n))

                m1 = fit_multiplex_sar(y, X, [G])
                sar = fit_spatial(SAR(), G, X, y; method = ExactLU(),
                                  max_exact_n = 10_000)
                # The reference is independently written, already-audited
                # code, not a rearrangement of the same expression. The
                # maximised log-likelihood is the sharp comparison; rho and
                # beta agree to the Brent tolerance both searches use.
                @test abs(m1.loglik - sar.loglik) < 1e-10
                @test abs(m1.rho[1] - sar.rho) < 1e-7
                @test maximum(abs, m1.beta .- sar.coef[1:(end - 1)]) < 1e-6
                @test abs(m1.rho_se[1] - sar.rho_se) < 1e-7
                @test maximum(abs, m1.beta_se .-
                              sqrt.(diag(sar.vcov))[1:(end - 1)]) < 1e-6
                @test m1.converged
                @test m1.channel_rank == 1
                @test nobs(m1) == n
                @test length(coefnames(m1)) == length(coef(m1))
            end
        end

        @testset "37k. multiplex coordinate ascent against a grid" begin
            # Coordinate ascent is exact along each axis but can stall on a
            # ridge. The profile likelihood is written out independently
            # here and maximised by brute force over the stationary region.
            let rng = MersenneTwister(2024)
                n = 120
                W1 = nw_er(rng, n, 0.05)
                W2 = nw_er(rng, n, 0.05)
                X = randn(rng, n, 2)
                A = Matrix(1.0I, n, n) - 0.35 * Matrix(W1) - 0.25 * Matrix(W2)
                y = A \ (1.0 .+ X * [1.0, -0.6] .+ randn(rng, n))

                Xa = hcat(ones(n), X)
                function prof(r1, r2)
                    abs(r1) + abs(r2) >= 0.995 && return -Inf
                    B = Matrix(1.0I, n, n) - r1 * Matrix(W1) - r2 * Matrix(W2)
                    e = B * y
                    u = e .- Xa * (Xa \ e)
                    s2 = dot(u, u) / n
                    return -0.5n * (log(2π) + 1) + logabsdet(B)[1] - 0.5n * log(s2)
                end

                m = fit_multiplex_sar(y, X, [W1, W2])
                step = 0.01
                best = -Inf
                for r1 in -0.95:step:0.95, r2 in -0.95:step:0.95
                    v = prof(r1, r2)
                    v > best && (best = v)
                end
                @test m.converged
                # The ascent must not be beaten by the grid beyond what the
                # grid spacing can resolve.
                @test m.loglik >= best - 1e-6
                @test abs(m.rho[1] - 0.35) < 0.25
                @test abs(m.rho[2] - 0.25) < 0.25
                @test m.loglik ≈ prof(m.rho[1], m.rho[2]) rtol = 1e-10
                @test m.channel_rank == 2
                @test sum(abs, m.rho) < 0.995
            end
        end

        @testset "37l. multiplex named failures" begin
            let rng = MersenneTwister(606)
                n = 60
                W1 = nw_er(rng, n, 0.08)
                X = randn(rng, n, 2); y = randn(rng, n)
                # Two identical channels cannot have their rhos told apart,
                # which is the same algebra as the peer-effects rank
                # condition and is refused the same way.
                @test_throws ArgumentError fit_multiplex_sar(y, X, [W1, W1])
                @test_throws ArgumentError fit_multiplex_sar(y, X, [W1, 2.0 .* W1])
                @test_throws ArgumentError fit_multiplex_sar(y, X, [])
                @test_throws DimensionMismatch fit_multiplex_sar(y, X,
                    [W1, nw_er(rng, n + 1, 0.08)])
                @test_throws ArgumentError fit_multiplex_sar(y, X, [W1];
                                                             rho_bound = 1.5)
                # An intercept is added, so one in X is a rank failure
                # named by column rather than a singular matrix.
                @test_throws ArgumentError fit_multiplex_sar(y, hcat(ones(n), X), [W1])
            end
        end

        @testset "37m. residual dependence after a fit" begin
            let rng = MersenneTwister(4004)
                n = 200
                G = nw_er(rng, n, 0.02); Gd = Matrix(G)
                X = randn(rng, n, 2)
                y = (Matrix(1.0I, n, n) - 0.4 * Gd) \
                    (1.0 .+ X * [1.0, -0.5] .+ Gd * X * [0.8, 0.3] .+ randn(rng, n))
                f = fit_peer_effects(y, X, G)
                m = network_moran(f, G)
                @test m isa MoranResult
                @test isfinite(m.I_stat)
                @test -1.5 <= m.I_stat <= 1.5
                # A different channel may be passed, which is the point:
                # it asks whether dependence survives along a route the
                # model did not include.
                other = nw_er(rng, n, 0.02)
                @test network_moran(f, other) isa MoranResult
                @test_throws DimensionMismatch network_moran(f, nw_er(rng, n + 5, 0.02))
            end
        end

        @testset "37n. show methods run" begin
            let rng = MersenneTwister(5150), io = IOBuffer()
                n = 150
                G = nw_er(rng, n, 0.02); Gd = Matrix(G)
                X = randn(rng, n, 2)
                y = (Matrix(1.0I, n, n) - 0.4 * Gd) \
                    (1.0 .+ X * [1.0, -0.5] .+ Gd * X * [0.8, 0.3] .+ randn(rng, n))
                show(io, peer_identification(G))
                show(io, peer_identification(group_weights([5, 5, 5])))
                show(io, fit_peer_effects(y, X, G))
                show(io, fit_multiplex_sar(y, X, [G]))
                s = String(take!(io))
                @test occursin("NOT IDENTIFIED", s)
                @test occursin("reflection problem", s)
                @test occursin("first-stage F", s)
            end
        end
    end

    # Row-standardised rook-contiguity grid, used by testsets 38 and 39.
    # Defined once here rather than inside each: a grid W is the standard
    # panel test fixture and both rows need the same one.
    function cf_grid_W(nx, ny)
        n = nx * ny
        I_ = Int[]; J_ = Int[]
        idx(i, j) = (j - 1) * nx + i
        for i in 1:nx, j in 1:ny, (di, dj) in ((1, 0), (-1, 0), (0, 1), (0, -1))
            a, b = i + di, j + dj
            if 1 <= a <= nx && 1 <= b <= ny
                push!(I_, idx(i, j)); push!(J_, idx(a, b))
            end
        end
        A = sparse(I_, J_, ones(length(I_)), n, n)
        return sparse(A ./ sum(A, dims = 2))
    end

    @testset "38. Weak vs strong cross-sectional dependence" begin
        # Pesaran (2015) and Bailey, Kapetanios & Pesaran (2016) state their
        # results as equations and as thresholds, so most gates here are
        # machine-precision identities or exact integers rather than bands.
        #
        # Every assertion is unconditional. Where a claim is conditional it
        # is asserted as an implication, `@test !guard || claim`, never as a
        # `@test` behind an `if` -- a guarded test that fails its guard does
        # not fail, it silently disappears and the only trace is the
        # assertion count. See the `37e` entry in docs/AUDIT.md.
        CF = SpatialEconometrics.SpatialCommonFactors

        # A panel whose error dependence has exponent alpha: a factor loads
        # on M = N^alpha of the N units, which is both papers' construction.
        function cf_panel(rng, N, T, alpha; loading_mean = 1.0)
            E = randn(rng, N, T)
            if alpha > 0
                M = max(1, round(Int, N^alpha))
                f = randn(rng, T)
                lam = loading_mean .+ 0.2 .* randn(rng, M)
                @views E[1:M, :] .+= lam .* transpose(f)
            end
            return E
        end

        @testset "38a. CD and CD_lm against the printed equations" begin
            # Pesaran (2015) eqs (2), (5), (9), (10), written out literally.
            function by_definition(E)
                N, T = size(E)
                rho = zeros(N, N)
                for i in 1:N, j in 1:N
                    num = 0.0
                    for t in 1:T; num += E[i, t] * E[j, t]; end
                    d1 = 0.0; for t in 1:T; d1 += E[i, t]^2; end
                    d2 = 0.0; for t in 1:T; d2 += E[j, t]^2; end
                    rho[i, j] = num / (sqrt(d1) * sqrt(d2))
                end
                acc = 0.0; acc_lm = 0.0; acc_abs = 0.0
                for i in 1:(N - 1), j in (i + 1):N
                    acc += rho[i, j]
                    acc_abs += abs(rho[i, j])
                    acc_lm += T * rho[i, j]^2 - 1.0
                end
                np = N * (N - 1) ÷ 2
                rb = acc / np
                return (cd = sqrt(T * N * (N - 1) / 2) * rb,
                        lm = sqrt(1.0 / (N * (N - 1))) * acc_lm,
                        rho_bar = rb, abs_rho_bar = acc_abs / np)
            end

            rng = MersenneTwister(3_801)
            for (N, T) in ((5, 10), (12, 8), (25, 20))
                for alpha in (0.0, 0.9)
                    E = cf_panel(rng, N, T, alpha)
                    ref = by_definition(E)
                    rc = pesaran_cd(vec(E), N, T; variant = :cd)
                    rl = pesaran_cd(vec(E), N, T; variant = :lm)
                    @test isapprox(rc.stat, ref.cd; rtol = 1e-10, atol = 1e-12)
                    @test isapprox(rl.stat, ref.lm; rtol = 1e-10, atol = 1e-12)
                    @test isapprox(rc.rho_bar, ref.rho_bar; rtol = 1e-10, atol = 1e-12)
                    @test isapprox(rc.abs_rho_bar, ref.abs_rho_bar;
                                   rtol = 1e-10, atol = 1e-12)
                end
            end

            # The matrix method is the vector method.
            let E = randn(MersenneTwister(1), 20, 15)
                @test pesaran_cd(vec(E), 20, 15).stat == pesaran_cd(E).stat
            end

            # p-value is two-sided normal, and symmetric in the statistic.
            let E = cf_panel(MersenneTwister(2), 30, 40, 0.9)
                r = pesaran_cd(vec(E), 30, 40)
                @test isapprox(r.p_value,
                               2 * (1 - SpatialEconometrics.SpatialUtils._normcdf(abs(r.stat)));
                               rtol = 1e-12)
                @test 0.0 <= r.p_value <= 1.0
            end
        end

        @testset "38b. the independence benchmark for mean |corr|" begin
            # Under independence |rho_hat| is half-normal with sd 1/sqrt(T),
            # so E|rho_hat| = sqrt(2/(pi T)). The field `abs_rho_bar_null`
            # is that value and the measured average must land on it.
            rng = MersenneTwister(3_802)
            for T in (30, 60, 120)
                N = 40
                acc = 0.0; reps = 60
                for _ in 1:reps
                    E = randn(rng, N, T)
                    acc += pesaran_cd(vec(E), N, T).abs_rho_bar
                end
                m = acc / reps
                null = sqrt(2 / (pi * T))
                @test isapprox(pesaran_cd(vec(randn(rng, N, T)), N, T).abs_rho_bar_null,
                               null; rtol = 1e-12)
                # 60 reps of an average over 780 pairs: tight.
                @test isapprox(m, null; rtol = 0.05)
            end
        end

        @testset "38c. eq (12): the implicit null is weak dependence" begin
            # The reported threshold is (2 - eps)/4 with eps = ln(T)/ln(N).
            rng = MersenneTwister(3_803)
            for (N, T) in ((100, 50), (50, 50), (200, 20))
                r = pesaran_cd(vec(randn(rng, N, T)), N, T)
                eps_rate = log(T) / log(N)
                @test isapprox(r.implicit_null_alpha, (2 - eps_rate) / 4;
                               rtol = 1e-12)
                # It is a number in (0, 1/2] for any sane panel shape.
                @test 0.0 < r.implicit_null_alpha <= 0.5
            end
        end

        @testset "38d. alpha: the published formulas" begin
            # eqs (13), (14), (15), recomputed directly.
            rng = MersenneTwister(3_804)
            N, T = 80, 60
            X = cf_panel(rng, N, T, 0.95)
            # Replicate the standardisation and eq (13)/(14) by hand.
            M = copy(X)
            for i in 1:N
                mu = mean(@view M[i, :]); s = std(@view M[i, :])
                M[i, :] = (M[i, :] .- mu) ./ s
            end
            xbar_t = vec(mean(M, dims = 1))
            s2 = sum((xbar_t .- mean(xbar_t)) .^ 2) / T
            a_ref = 1 + 0.5 * log(s2) / log(N)

            r = alpha_exponent(vec(X), N, T; bias_correct = false)
            @test isapprox(r.sigma2_xbar, s2; rtol = 1e-10)
            @test isapprox(r.alpha_raw, a_ref; rtol = 1e-10)
            @test isapprox(r.alpha, a_ref; rtol = 1e-10)
            @test r.kappa2 == 1.0

            # kappa2 shifts alpha by exactly (1/2) ln(kappa2)/ln(N).
            for k2 in (0.25, 0.5, 2.0)
                rk = alpha_exponent(vec(X), N, T; bias_correct = false, kappa2 = k2)
                @test isapprox(rk.alpha, a_ref - 0.5 * log(k2) / log(N);
                               rtol = 1e-10)
            end

            # The bias correction only ever subtracts.
            rb = alpha_exponent(vec(X), N, T; bias_correct = true)
            @test rb.alpha <= rb.alpha_raw
            @test isapprox(rb.alpha_raw, a_ref; rtol = 1e-10)
        end

        @testset "38e. alpha: scale invariance, exactly" begin
            rng = MersenneTwister(3_805)
            N, T = 60, 50
            X = cf_panel(rng, N, T, 0.95)
            a1 = alpha_exponent(vec(X), N, T).alpha
            X2 = copy(X)
            for i in 1:N; X2[i, :] .*= (0.1 + 10 * rand(rng)); end
            a2 = alpha_exponent(vec(X2), N, T).alpha
            # Standardisation makes this exact, not approximate.
            @test isapprox(a1, a2; rtol = 1e-12, atol = 1e-12)
            # Shifting a series is absorbed too.
            X3 = copy(X)
            for i in 1:N; X3[i, :] .+= 5 * randn(rng); end
            @test isapprox(alpha_exponent(vec(X3), N, T).alpha, a1;
                           rtol = 1e-12, atol = 1e-12)
            # Without standardisation the scale matters, which is the point
            # of having the option at all.
            b1 = alpha_exponent(vec(X), N, T; standardize = false).alpha
            b2 = alpha_exponent(vec(X2), N, T; standardize = false).alpha
            @test !isapprox(b1, b2; atol = 1e-6)
        end

        @testset "38f. alpha: the two thresholds are enforced" begin
            rng = MersenneTwister(3_806)
            # alpha <= 1/2 is NOT IDENTIFIED: must throw, not return.
            @test_throws ArgumentError alpha_exponent(vec(randn(rng, 80, 100)), 80, 100)
            let e = try
                        alpha_exponent(vec(randn(rng, 80, 100)), 80, 100); nothing
                    catch err; err end
                @test e isa ArgumentError
                @test occursin("NOT IDENTIFIED", sprint(showerror, e))
            end
            # A strong factor is identified, and inference is valid there.
            let X = cf_panel(rng, 100, 80, 1.0)
                r = alpha_exponent(vec(X), 100, 80)
                @test r.alpha > 0.5
                # implication, not a guarded test: inference_valid is
                # equivalent to alpha > 4/7 by construction.
                @test r.inference_valid == (r.alpha > 4 / 7)
                @test !r.inference_valid || isfinite(r.se)
                @test !r.inference_valid || (r.ci[1] < r.alpha < r.ci[2])
                @test r.inference_valid || isnan(r.se)
            end
        end

        @testset "38g. alpha: named failures" begin
            rng = MersenneTwister(3_807)
            X = cf_panel(rng, 40, 30, 0.95)
            @test_throws ArgumentError alpha_exponent(vec(X), 2, 30)     # N < 3
            @test_throws ArgumentError alpha_exponent(vec(X[:, 1:1]), 40, 1)  # T < 2
            @test_throws ArgumentError alpha_exponent(vec(X), 40, 30; kappa2 = 0.0)
            @test_throws ArgumentError alpha_exponent(vec(X), 40, 30; kappa2 = -1.0)
            # A constant unit cannot be standardised.
            let Xc = copy(X); Xc[3, :] .= 2.0
                @test_throws ArgumentError alpha_exponent(vec(Xc), 40, 30)
            end
            # non-finite input
            let Xn = copy(X); Xn[1, 1] = NaN
                @test_throws ArgumentError alpha_exponent(vec(Xn), 40, 30)
            end
        end

        @testset "38h. CD: named failures" begin
            rng = MersenneTwister(3_808)
            E = randn(rng, 20, 15)
            @test_throws ArgumentError pesaran_cd(vec(E), 20, 15; variant = :nope)
            @test_throws ArgumentError pesaran_cd(vec(E[1:2, :]), 2, 15)
            @test_throws DimensionMismatch pesaran_cd(vec(E), 20, 14)
            let En = copy(E); En[1, 1] = Inf
                @test_throws ArgumentError pesaran_cd(vec(En), 20, 15)
            end
            # A perfectly fitted unit has no residual variation, so its
            # correlations are undefined and that is said rather than
            # producing a NaN statistic.
            let Ez = copy(E); Ez[4, :] .= 0.0
                @test_throws ArgumentError pesaran_cd(vec(Ez), 20, 15)
            end
        end

        @testset "38i. the verdict ties the two together" begin
            rng = MersenneTwister(3_809)
            # Independent: weak, and alpha unidentified.
            let v = dependence_verdict(vec(randn(rng, 60, 80)), 60, 80)
                @test v.kind === :unidentified
                @test v.alpha === nothing
                @test !isempty(v.recommendation)
            end
            # Strong one-signed factor: strong, alpha near 1.
            #
            # `loading_mean = 2.0` is deliberate and the margin is asserted
            # rather than just the verdict. At `loading_mean = 1.0` the
            # estimate is alpha-hat = 0.9093 (sd 0.0109 over 400 draws),
            # which is 0.85 sd above the 0.9 threshold the `:strong` branch
            # uses -- so 17% of draws come out `:semi_weak` and the test
            # passed or failed by RNG stream. It failed on Julia 1.10.13
            # while passing on 1.13.3, which is how it was found. The cause
            # is not a defect: `kappa2 = 1` biases alpha-hat down by
            # (1/2)ln(kappa2)/ln(N) under standardisation, documented on
            # `alpha_exponent`. At `loading_mean = 2.0` the minimum over
            # 400 draws is 0.9516, so the claim is a property of the design
            # and not of the draw.
            let v = dependence_verdict(
                        vec(cf_panel(rng, 60, 80, 1.0; loading_mean = 2.0)),
                        60, 80)
                @test v.kind === :strong
                @test v.alpha !== nothing
                @test v.alpha.alpha > 0.93
                @test v.cd.p_value < 0.05
            end
            # Cancelling loadings: the configuration where BOTH published
            # diagnostics fail. The verdict must still call it strong, on
            # the absolute average.
            let N = 40, T = 60
                f = randn(rng, T)
                lam = [i <= N ÷ 2 ? 1.0 : -1.0 for i in 1:N]
                E = lam .* transpose(f) .+ 0.5 .* randn(rng, N, T)
                v = dependence_verdict(vec(E), N, T)
                @test v.kind === :strong
                # alpha is refused here -- it is the measured outcome, and
                # it is why the verdict cannot rely on alpha alone.
                @test v.alpha === nothing
                @test v.cd.abs_rho_bar > 2 * v.cd.abs_rho_bar_null
                @test abs(v.cd.rho_bar) < 0.05
                @test occursin("cancel", v.recommendation)
            end
        end

        @testset "38j. spatial CCE" begin
            rng = MersenneTwister(3_810)
            W = cf_grid_W(6, 5); N = size(W, 1); T = 25
            Im = Matrix(1.0I, N, N)
            rho, beta = 0.4, [1.0, -0.5]
            Ainv = inv(Im - rho * Matrix(W))
            X = randn(rng, N * T, 2)
            f = randn(rng, T); lam = 1.0 .+ randn(rng, N)
            y = Vector{Float64}(undef, N * T)
            for t in 1:T
                r = ((t - 1) * N + 1):(t * N)
                y[r] = Ainv * (X[r, :] * beta .+ lam .* f[t] .+ randn(rng, N))
            end

            m = fit_sp_cce(y, X, W, N, T)
            @test m isa SpCCEModel
            @test length(coef(m)) == 3
            @test coefnames(m)[1] == "rho"
            @test length(stderror(m)) == 3
            @test all(isfinite, coef(m))
            @test all(>(0), stderror(m))
            @test nobs(m) == N * T
            @test m.n_augment == 3           # ybar + 2 Xbar columns
            @test m.augment == [:y, :X]
            @test isfinite(m.sigma2) && m.sigma2 > 0
            @test length(residuals(m)) == N * T
            @test isfinite(m.first_stage_F) && m.first_stage_F > 0
            # The estimator exists to recover rho under a factor; on one
            # draw ask only that it is in the right region and far better
            # than the unadjusted fit's known +0.28 bias.
            @test abs(m.rho - rho) < 0.2

            # Augmentation set changes the projected column count.
            let m2 = fit_sp_cce(y, X, W, N, T; augment = [:y])
                @test m2.n_augment == 1
            end
            let m3 = fit_sp_cce(y, X, W, N, T; augment = [:y, :X, :Wy])
                @test m3.n_augment == 4
            end

            # Mean group returns per-unit slopes and a dispersion vcov.
            let mg = fit_sp_cce(y, X, W, N, T; mean_group = true)
                @test mg.coef_mg !== nothing
                @test size(mg.coef_mg, 2) == 2
                @test length(mg.coef_mg_mean) == 2
                @test size(mg.vcov_mg) == (2, 2)
                @test all(>=(0), diag(mg.vcov_mg))
                @test isposdef(mg.vcov_mg + 1e-12I)
            end

            # Named failures.
            @test_throws ArgumentError fit_sp_cce(y, X, W, N, T; augment = Symbol[])
            @test_throws ArgumentError fit_sp_cce(y, X, W, N, T; augment = [:bogus])
            @test_throws DimensionMismatch fit_sp_cce(y, X, W, N, T + 1)
            # Augmentation wider than T is refused rather than fitted on
            # negative degrees of freedom.
            @test_throws ArgumentError fit_sp_cce(y[1:(N * 3)], X[1:(N * 3), :],
                                                  W, N, 3;
                                                  augment = [:y, :X, :Wy])
            # TimeFE is redundant with the augmentation and says so.
            @test_logs (:warn,) match_mode = :any fit_sp_cce(y, X, W, N, T;
                                                             fe = TimeFE())

            @test occursin("Common Correlated Effects", sprint(show, m))
            @test occursin("First-stage F", sprint(show, m))
        end

        @testset "38k. a time effect is a factor with equal loadings" begin
            # Exact, and the sharpest statement of what this module adds:
            # TimeFE removes a homogeneous-loading factor to machine
            # precision and leaves a heterogeneous one essentially intact.
            SU = SpatialEconometrics.SpatialUtils
            rng = MersenneTwister(3_811)
            N, T = 30, 25
            f = randn(rng, T)
            hom = vec(ones(N) * transpose(f))
            @test maximum(abs, SU.within_transform!(copy(hom), N, T, TimeFE())) < 1e-12

            lam = randn(rng, N)
            het = vec(lam * transpose(f))
            after = SU.within_transform!(copy(het), N, T, TimeFE())
            @test maximum(abs, after) > 0.1 * maximum(abs, het)
            # Almost all of the variance survives.
            @test var(after) / var(het) > 0.9
        end

        @testset "38l. show methods run" begin
            rng = MersenneTwister(3_812)
            E = cf_panel(rng, 50, 60, 1.0)
            @test occursin("Pesaran CD", sprint(show, pesaran_cd(vec(E), 50, 60)))
            @test occursin("Breusch", sprint(show, pesaran_cd(vec(E), 50, 60;
                                                              variant = :lm)))
            @test occursin("Exponent", sprint(show, alpha_exponent(vec(E), 50, 60)))
            @test occursin("Verdict", sprint(show, dependence_verdict(vec(E), 50, 60)))
        end
    end

    @testset "39. Spatio-temporal models" begin
        # As above: implications, never guarded tests.
        ST = SpatialEconometrics.SpatioTemporalModels

        @testset "39a. STAR against explicit normal equations" begin
            rng = MersenneTwister(3_901)
            W = cf_grid_W(5, 4); N = size(W, 1); T = 18
            Wd = Matrix(W)
            X = randn(rng, N * T, 2)
            y = Vector{Float64}(undef, N * T)
            y[1:N] = randn(rng, N)
            for t in 2:T
                prev = y[((t - 2) * N + 1):((t - 1) * N)]
                r = ((t - 1) * N + 1):(t * N)
                y[r] = 0.3 .* prev .+ 0.25 .* (Wd * prev) .+
                       X[r, :] * [0.8, -0.4] .+ randn(rng, N)
            end

            m = fit_star(y, X, W, N, T; spatial_lags = 1, temporal_lags = 1)
            @test m isa STARModel
            @test m.T_eff == T - 1
            @test size(m.phi) == (2, 1)
            @test length(m.beta) == 2
            @test nobs(m) == N * (T - 1)

            # Rebuild the design and solve the normal equations directly.
            Teff = T - 1
            D = Matrix{Float64}(undef, N * Teff, 5)
            yv = Vector{Float64}(undef, N * Teff)
            for t in 1:Teff
                src = ((t - 1) * N + 1):(t * N)
                dst = ((t - 1) * N + 1):(t * N)
                tgt = (t * N + 1):((t + 1) * N)
                D[dst, 1] = y[src]
                D[dst, 2] = Wd * y[src]
                D[dst, 3] = X[tgt, 1]
                D[dst, 4] = X[tgt, 2]
                D[dst, 5] .= 1.0
                yv[dst] = y[tgt]
            end
            theta = (D' * D) \ (D' * yv)
            @test isapprox(m.coef, theta; rtol = 1e-9, atol = 1e-11)
            @test isapprox(residuals(m), yv .- D * theta;
                           rtol = 1e-9, atol = 1e-11)
            s2 = sum(abs2, yv .- D * theta) / (N * Teff - 5)
            @test isapprox(m.sigma2, s2; rtol = 1e-10)
            @test isapprox(vcov(m), s2 .* inv(Symmetric(D' * D));
                           rtol = 1e-8, atol = 1e-12)

            # phi is laid out as documented: row k+1 multiplies W^k.
            @test m.phi[1, 1] == m.coef[1]
            @test m.phi[2, 1] == m.coef[2]
            @test "y(t-1)" in coefnames(m)
            @test "W^1 y(t-1)" in coefnames(m)
        end

        @testset "39b. STAR: S = 0 is a non-spatial panel AR" begin
            rng = MersenneTwister(3_902)
            W = cf_grid_W(4, 4); N = size(W, 1); T = 20
            X = randn(rng, N * T, 1)
            y = randn(rng, N * T)
            m = fit_star(y, X, W, N, T; spatial_lags = 0, temporal_lags = 2)
            @test size(m.phi) == (1, 2)
            @test length(m.coef) == 2 + 1 + 1   # two AR, one x, intercept
            # With S = 0 the design never touches W, so changing W cannot
            # change the fit.
            W2 = cf_grid_W(4, 4) .* 0.5
            m2 = fit_star(y, X, sparse(W2), N, T;
                          spatial_lags = 0, temporal_lags = 2)
            @test isapprox(m.coef, m2.coef; rtol = 1e-12, atol = 1e-14)
        end

        @testset "39c. STAR: named failures and the rank guard" begin
            rng = MersenneTwister(3_903)
            W = cf_grid_W(4, 4); N = size(W, 1); T = 12
            X = randn(rng, N * T, 1); y = randn(rng, N * T)
            @test_throws ArgumentError fit_star(y, X, W, N, T; temporal_lags = 0)
            @test_throws ArgumentError fit_star(y, X, W, N, T; spatial_lags = -1)
            @test_throws ArgumentError fit_star(y, X, W, N, T; temporal_lags = T)
            @test_throws DimensionMismatch fit_star(y, X, W, N, T + 1)
            let yn = copy(y); yn[1] = NaN
                @test_throws ArgumentError fit_star(yn, X, W, N, T)
            end
            # A complete graph gives W^2 proportional to W, so the spatial
            # powers are dependent and raising S must be refused rather
            # than failing inside LAPACK.
            let n = 8
                A = ones(n, n) - Matrix(1.0I, n, n)
                Wc = sparse(A ./ sum(A, dims = 2))
                Xc = randn(rng, n * T, 1); yc = randn(rng, n * T)
                @test_throws ArgumentError fit_star(yc, Xc, Wc, n, T;
                                                     spatial_lags = 3,
                                                     temporal_lags = 1)
            end
        end

        @testset "39d. the IRF's long run is the spatial multiplier" begin
            rng = MersenneTwister(3_904)
            W = cf_grid_W(5, 4); N = size(W, 1); T = 20
            Wd = Matrix(W); Im = Matrix(1.0I, N, N)
            X = randn(rng, N * T, 1)
            y = Vector{Float64}(undef, N * T)
            y[1:N] = randn(rng, N)
            for t in 2:T
                prev = y[((t - 2) * N + 1):((t - 1) * N)]
                r = ((t - 1) * N + 1):(t * N)
                y[r] = 0.25 .* prev .+ 0.2 .* (Wd * prev) .+
                       X[r, :] * [0.5] .+ randn(rng, N)
            end
            m = fit_star(y, X, W, N, T; spatial_lags = 1, temporal_lags = 1)
            @test m.stationary
            @test m.spectral_radius < 1

            irf = space_time_irf(m; horizon = 300)
            @test irf.psi[1] == Matrix(1.0I, N, N)
            A1 = m.phi[1, 1] .* Im .+ m.phi[2, 1] .* Wd
            acc = zeros(N, N)
            for h in 0:300; acc .+= irf.psi[h + 1]; end
            @test isapprox(acc, inv(Im - A1); rtol = 1e-8, atol = 1e-10)

            cum = space_time_irf(m; horizon = 300, cumulative = true)
            @test isapprox(cum.total[end], sum(inv(Im - A1)) / N;
                           rtol = 1e-8, atol = 1e-10)
            # own + spillover == total, at every horizon, by construction.
            @test all(isapprox.(cum.own .+ cum.spillover, cum.total;
                                rtol = 1e-10, atol = 1e-12))
            # Horizon 0 is purely own: a shock has not yet propagated.
            @test isapprox(irf.own[1], 1.0; rtol = 1e-12)
            @test isapprox(irf.spillover[1], 0.0; atol = 1e-12)

            # store_matrices = false keeps the summaries only.
            let r2 = space_time_irf(m; horizon = 10, store_matrices = false)
                @test isempty(r2.psi)
                @test length(r2.total) == 11
            end
            @test_throws ArgumentError space_time_irf(m; horizon = -1)
            @test occursin("Impulse Response", sprint(show, irf))
        end

        @testset "39e. interactive FE reduces to the audited panel" begin
            # n_factors = 0 removes the factor step entirely, so the
            # estimator must be the package's own SAR panel. The reference
            # is independently written, already-audited code.
            rng = MersenneTwister(3_905)
            W = cf_grid_W(6, 5); N = size(W, 1); T = 20
            Im = Matrix(1.0I, N, N)
            X = randn(rng, N * T, 2)
            Ainv = inv(Im - 0.4 * Matrix(W))
            y = Vector{Float64}(undef, N * T)
            for t in 1:T
                r = ((t - 1) * N + 1):(t * N)
                y[r] = Ainv * (X[r, :] * [1.0, -0.5] .+ randn(rng, N))
            end

            # fit_spatial_panel prepends an intercept under NoFE, so the
            # comparable design carries one explicitly.
            ref = fit_spatial_panel(SAR(), NoFE(), W, X, y, N, T;
                                    approach = DirectQML(), method = ExactLU())
            m0 = fit_interactive_fe(y, hcat(ones(N * T), X), W, N, T;
                                    n_factors = 0)
            # brent_maximize's own tolerance is 1e-8, which is the binding
            # constraint here rather than anything structural.
            @test isapprox(m0.rho, ref.rho; atol = 1e-6)
            @test isapprox(m0.beta, ref.coef[1:3]; atol = 1e-6)
            @test isapprox(m0.loglik, ref.loglik; rtol = 1e-9)
            @test m0.n_factors == 0
            @test size(m0.factors) == (T, 0)
            @test m0.converged
            @test factor_projection(m0) == zeros(T, T)
        end

        @testset "39f. factors are identified only up to rotation" begin
            rng = MersenneTwister(3_906)
            W = cf_grid_W(6, 5); N = size(W, 1); T = 20
            Im = Matrix(1.0I, N, N)
            Ainv = inv(Im - 0.4 * Matrix(W))
            X = randn(rng, N * T, 2)
            lam = randn(rng, N); f = randn(rng, T)
            y = Vector{Float64}(undef, N * T)
            for t in 1:T
                r = ((t - 1) * N + 1):(t * N)
                y[r] = Ainv * (X[r, :] * [1.0, -0.5] .+ lam .* f[t] .+
                               randn(rng, N))
            end

            m1 = fit_interactive_fe(y, X, W, N, T; n_factors = 1,
                                    rng = MersenneTwister(1))
            m2 = fit_interactive_fe(y, X, W, N, T; n_factors = 1, n_starts = 4,
                                    rng = MersenneTwister(999))
            # rho and beta are identified.
            @test isapprox(m1.rho, m2.rho; atol = 1e-6)
            @test isapprox(m1.beta, m2.beta; atol = 1e-6)
            # The projection is identified; the factors themselves are not,
            # so only the projection is pinned.
            @test isapprox(factor_projection(m1), factor_projection(m2);
                           atol = 1e-6)
            # A sign flip is the simplest rotation and leaves the
            # projection exactly alone.
            let F = m1.factors, Ff = -m1.factors
                P = F * ((F' * F) \ F')
                Pf = Ff * ((Ff' * Ff) \ Ff')
                @test isapprox(P, Pf; rtol = 1e-12, atol = 1e-14)
            end
            # Normalisation F'F/T = I_r.
            @test isapprox(m1.factors' * m1.factors ./ T, Matrix(1.0I, 1, 1);
                           rtol = 1e-10)
            # The projection is idempotent and symmetric, as a projection.
            let P = factor_projection(m1)
                @test isapprox(P * P, P; rtol = 1e-9, atol = 1e-11)
                @test isapprox(P, P'; rtol = 1e-12, atol = 1e-14)
                @test isapprox(tr(P), 1.0; rtol = 1e-8)   # rank 1
            end
        end

        @testset "39g. interactive FE: structure and named failures" begin
            rng = MersenneTwister(3_907)
            W = cf_grid_W(5, 4); N = size(W, 1); T = 18
            X = randn(rng, N * T, 2); y = randn(rng, N * T)

            m = fit_interactive_fe(y, X, W, N, T; n_factors = 2)
            @test m.n_factors == 2
            @test size(m.factors) == (T, 2)
            @test size(m.loadings) == (N, 2)
            @test length(coef(m)) == 3
            @test coefnames(m)[1] == "rho"
            @test all(isfinite, coef(m))
            @test all(>(0), stderror(m))
            @test length(residuals(m)) == N * T
            @test isapprox(tr(factor_projection(m)), 2.0; rtol = 1e-8)

            @test_throws ArgumentError fit_interactive_fe(y, X, W, N, T;
                                                           n_factors = -1)
            # Saturating the factor space is refused.
            @test_throws ArgumentError fit_interactive_fe(y, X, W, N, T;
                                                           n_factors = min(N, T))
            @test_throws ArgumentError fit_interactive_fe(y, X, W, N, T;
                                                           n_starts = 0)
            @test_throws DimensionMismatch fit_interactive_fe(y, X, W, N, T + 1)
            let yn = copy(y); yn[2] = NaN
                @test_throws ArgumentError fit_interactive_fe(yn, X, W, N, T)
            end
            @test occursin("Interactive Fixed Effects", sprint(show, m))
            @test occursin("rotation", sprint(show, m))
        end

        @testset "39h. forecasting" begin
            rng = MersenneTwister(3_908)
            W = cf_grid_W(5, 4); N = size(W, 1); T = 20
            Wd = Matrix(W)
            X = randn(rng, N * T, 1)
            y = Vector{Float64}(undef, N * T)
            y[1:N] = randn(rng, N)
            for t in 2:T
                prev = y[((t - 2) * N + 1):((t - 1) * N)]
                r = ((t - 1) * N + 1):(t * N)
                y[r] = 0.3 .* prev .+ 0.2 .* (Wd * prev) .+
                       X[r, :] * [0.6] .+ randn(rng, N)
            end
            m = fit_star(y, X, W, N, T; spatial_lags = 1, temporal_lags = 1)

            Xf = randn(rng, N * 4, 1)
            hist = y[((T - 1) * N + 1):(T * N)]
            fc = spatial_forecast(m, Xf; h = 4, y_history = hist)
            @test size(fc) == (N, 4)
            @test all(isfinite, fc)

            # The one-step forecast IS the model's own prediction, exactly.
            A1 = m.phi[1, 1] .* Matrix(1.0I, N, N) .+ m.phi[2, 1] .* Wd
            # beta excludes the intercept, which is kept in its own field.
            @test length(m.beta) == size(X, 2)
            @test m.has_intercept
            @test m.intercept == m.coef[end]
            c = m.intercept
            manual = A1 * hist .+ Xf[1:N, :] * m.beta .+ c
            @test isapprox(fc[:, 1], manual; rtol = 1e-10, atol = 1e-12)
            # And the second step iterates on the first, not on the data.
            manual2 = A1 * fc[:, 1] .+ Xf[(N + 1):(2N), :] * m.beta .+ c
            @test isapprox(fc[:, 2], manual2; rtol = 1e-10, atol = 1e-12)

            @test_throws ArgumentError spatial_forecast(m, Xf; h = 0,
                                                         y_history = hist)
            @test_throws ArgumentError spatial_forecast(m, Xf; h = 4)
            @test_throws DimensionMismatch spatial_forecast(m, Xf; h = 4,
                                                             y_history = hist[1:3])
            @test_throws DimensionMismatch spatial_forecast(m, Xf; h = 3,
                                                             y_history = hist)
        end

        @testset "39i. the IFE multiplier matches the static one" begin
            # With no temporal dynamics a shock propagates within the period
            # and stops, so the response is (I - rho W)^-1 and nothing after.
            rng = MersenneTwister(3_909)
            W = cf_grid_W(5, 4); N = size(W, 1); T = 18
            X = randn(rng, N * T, 1); y = randn(rng, N * T)
            m = fit_interactive_fe(y, X, W, N, T; n_factors = 1)
            irf = space_time_irf(m, W; horizon = 5)
            @test isapprox(irf.psi[1],
                           inv(Matrix(1.0I, N, N) - m.rho * Matrix(W));
                           rtol = 1e-10)
            @test all(iszero, irf.psi[2])
            @test all(iszero, irf.total[2:end])
            # Cumulative therefore flattens after horizon 0.
            let c = space_time_irf(m, W; horizon = 5, cumulative = true)
                @test all(isapprox.(c.total[2:end], c.total[1];
                                    rtol = 1e-12, atol = 1e-14))
            end
            @test_throws DimensionMismatch space_time_irf(m, cf_grid_W(3, 3);
                                                           horizon = 2)
        end

        # ── 39j ───────────────────────────────────────────────
        # Pfeifer & Deutsch define the spatial lag operator on order-k
        # NEIGHBOURHOODS, not powers of W: w_ij^(k) is "nonzero only if
        # sites i and j are kth order neighbors". `fit_star` has always
        # computed powers. Both are now available; the default is still
        # powers, so these tests pin (i) that the two agree exactly where
        # the conventions agree, and (ii) that they genuinely differ where
        # they do not -- an option that quietly made no difference would be
        # worse than no option.
        @testset "39j. powers of W versus order-k neighbours" begin
            rng = MersenneTwister(3_910)
            W = cf_grid_W(5, 4); N = size(W, 1); T = 40
            Wd = Matrix(W)
            X = randn(rng, N * T, 2)
            y = Vector{Float64}(undef, N * T)
            y[1:N] = randn(rng, N)
            for t in 2:T
                prev = y[((t - 2) * N + 1):((t - 1) * N)]
                r = ((t - 1) * N + 1):(t * N)
                y[r] = 0.30 .* prev .+ 0.25 .* (Wd * prev) .+
                       X[r, :] * [0.8, -0.4] .+ randn(rng, N)
            end

            # At S = 1 the conventions coincide by definition: W^(1) IS the
            # row-standardised first-order neighbour matrix. Identical to
            # the last bit, not merely close -- a tolerance here would hide
            # the case where :orders rebuilds W slightly differently.
            m1p = fit_star(y, X, W, N, T; spatial_lags = 1,
                           spatial_weights = :powers)
            m1o = fit_star(y, X, W, N, T; spatial_lags = 1,
                           spatial_weights = :orders)
            @test m1p.coef == m1o.coef
            @test m1p.loglik == m1o.loglik
            @test m1p.spectral_radius == m1o.spectral_radius
            @test Matrix(m1o.Wk[2]) == Wd
            @test m1o.spatial_weights === :orders
            @test m1p.spatial_weights === :powers

            # S = 0 touches neither, so it cannot depend on the choice.
            @test fit_star(y, X, W, N, T; spatial_lags = 0,
                           spatial_weights = :orders).coef ==
                  fit_star(y, X, W, N, T; spatial_lags = 0,
                           spatial_weights = :powers).coef

            m2p = fit_star(y, X, W, N, T; spatial_lags = 2,
                           spatial_weights = :powers)
            m2o = fit_star(y, X, W, N, T; spatial_lags = 2,
                           spatial_weights = :orders)

            # The structural difference. On a rook lattice there are no
            # triangles, so every length-2 walk between distinct sites is a
            # distance-2 pair and the ENTIRE discrepancy is the diagonal:
            # W^2 puts weight on the unit itself and calls it a spatial lag.
            let A2p = Matrix(m2p.Wk[3]), A2o = Matrix(m2o.Wk[3])
                @test count(!iszero, diag(A2p)) == N      # all 20
                @test count(!iszero, diag(A2o)) == 0
                @test count(!iszero, A2p) == count(!iszero, A2o) + N
                # off the diagonal the supports agree here, which is what
                # isolates the diagonal as the whole of the difference
                offp = [i == j ? false : A2p[i, j] != 0
                        for i in 1:N, j in 1:N]
                @test offp == (A2o .!= 0)
                # ... and each is row-standardised in its own right
                @test all(isapprox.(sum(A2o, dims = 2), 1.0; atol = 1e-12))
            end

            # So the fits must differ, and the coefficient on the order-2
            # term is the one that moves.
            @test m2p.coef != m2o.coef
            @test abs(m2p.phi[3, 1] - m2o.phi[3, 1]) > 0.01
            @test "W^2 y(t-1)" in coefnames(m2p)
            @test "W⁽2⁾ y(t-1)" in coefnames(m2o)
            @test !("W^2 y(t-1)" in coefnames(m2o))
            @test occursin("order-k neighbours", sprint(show, m2o))
            @test occursin("powers of W", sprint(show, m2p))

            # Everything built on the fit must use the matrices the fit
            # used. The long-run multiplier is the gate: if the IRF
            # recomputed powers while the design used orders, this breaks.
            for m in (m2p, m2o)
                S = m.spatial_lags
                irf = space_time_irf(m; horizon = 500)
                @test irf.psi[1] == Matrix(1.0I, N, N)
                A1 = ST._star_A(m.phi, m.Wk, N, S, 1)
                @test maximum(abs, sum(irf.psi) .-
                              inv(Matrix(1.0I, N, N) - A1)) < 1e-12
            end

            # L = 2 as well, so the companion block structure is exercised
            # under the orders convention too.
            let mL = fit_star(y, X, W, N, T; spatial_lags = 2,
                              temporal_lags = 2, spatial_weights = :orders)
                @test mL.stationary
                Asum = ST._star_A(mL.phi, mL.Wk, N, 2, 1) .+
                       ST._star_A(mL.phi, mL.Wk, N, 2, 2)
                @test maximum(abs,
                        sum(space_time_irf(mL; horizon = 500).psi) .-
                        inv(Matrix(1.0I, N, N) - Asum)) < 1e-12
            end

            # Pfeifer & Deutsch's stationarity condition is determinantal:
            # every x solving det[x^L I - sum_l A_l x^(L-l)] = 0 must lie
            # inside the unit circle. This package reports a companion
            # spectral radius instead. Assert they are the same statement
            # by evaluating the paper's determinant AT the companion
            # eigenvalues -- it must vanish.
            let mL = fit_star(y, X, W, N, T; spatial_lags = 2,
                              temporal_lags = 2, spatial_weights = :orders),
                L = 2
                As = [ST._star_A(mL.phi, mL.Wk, N, 2, l) for l in 1:L]
                C = zeros(L * N, L * N)
                for l in 1:L
                    C[1:N, ((l - 1) * N + 1):(l * N)] = As[l]
                end
                C[(N + 1):(2N), 1:N] = Matrix(1.0I, N, N)
                ev = eigvals(C)
                @test isapprox(maximum(abs, ev), mL.spectral_radius;
                               rtol = 1e-10)
                worst = 0.0
                for x in ev
                    M = ComplexF64(x)^L * Matrix{ComplexF64}(I, N, N)
                    for l in 1:L
                        M .-= ComplexF64(x)^(L - l) .* As[l]
                    end
                    worst = max(worst, abs(det(M)))
                end
                @test worst < 1e-20
            end

            # Named failures specific to the option.
            @test_throws ArgumentError fit_star(y, X, W, N, T;
                spatial_weights = :order)
            @test_throws ArgumentError fit_star(y, X, W, N, T;
                spatial_weights = :neighbours)
            # Past the graph's diameter an order matrix is empty, and that
            # is refused with the diameter rather than fitted as zeros.
            # The 5x4 rook lattice has diameter 7.
            let e = try
                        fit_star(y, X, W, N, T; spatial_lags = 20,
                                 spatial_weights = :orders)
                        nothing
                    catch err
                        err
                    end
                @test e isa ArgumentError
                @test occursin("diameter is 7", sprint(showerror, e))
            end
            # Under :powers that same S fails for a different reason --
            # rank deficiency, not emptiness -- so the two paths must not
            # be reporting each other's diagnosis.
            @test_throws ArgumentError fit_star(y, X, W, N, T;
                spatial_lags = 20, spatial_weights = :powers)
        end
    end

    # ═══════════════════════════════════════════════════════════════════
    # 40. Spatial VAR — Di Giacinto (2010)
    # ═══════════════════════════════════════════════════════════════════

    @testset "40. Spatial VAR" begin
        SV = SpatialEconometrics.SpVAR

        # A ring is regular, so row-standardising keeps W⁽¹⁾ symmetric —
        # which the outward/inward STIR gate needs. See 40g.
        function sv_ring(n)
            I_ = Int[]; J_ = Int[]
            for i in 1:n
                push!(I_, i); push!(J_, mod1(i + 1, n))
                push!(I_, i); push!(J_, mod1(i - 1, n))
            end
            return sparse(I_, J_, ones(2n), n, n)
        end

        function sv_path(n)
            I_ = Int[]; J_ = Int[]
            for i in 1:(n - 1)
                push!(I_, i); push!(J_, i + 1)
                push!(I_, i + 1); push!(J_, i)
            end
            return sparse(I_, J_, ones(2 * (n - 1)), n, n)
        end

        SV_R1, SV_R2 = 0.35, -0.20
        SV_PS = (0.40, 0.15)
        SV_A = ((0.30, 0.10), (0.15, -0.05), (-0.10, 0.08), (0.25, 0.12))

        # A homogeneous SpVAR(1,1) with K = 2, built from the same W⁽ˡ⁾ the
        # estimator will use. `spatial = false` zeroes every spatial
        # coefficient, giving the nested panel VAR of eqs. (15)–(18).
        function sv_sim(rng, Wl, N, T; burn = 150, spatial = true)
            Wd = Matrix(Wl[2]); Im = Matrix(1.0I, N, N); Z = zeros(N, N)
            C0, C1 = if spatial
                ([Im - SV_R1 * Wd  Z;
                  -(SV_PS[1] * Im + SV_PS[2] * Wd)  Im - SV_R2 * Wd],
                 [SV_A[1][1] * Im + SV_A[1][2] * Wd   SV_A[2][1] * Im + SV_A[2][2] * Wd;
                  SV_A[3][1] * Im + SV_A[3][2] * Wd   SV_A[4][1] * Im + SV_A[4][2] * Wd])
            else
                ([Im Z; -SV_PS[1] * Im Im],
                 [SV_A[1][1] * Im  SV_A[2][1] * Im;
                  SV_A[3][1] * Im  SV_A[4][1] * Im])
            end
            F = lu(C0); n = 2N
            y = zeros(n); Y = Matrix{Float64}(undef, N * T, 2)
            for t in 1:(T + burn)
                y = F \ (C1 * y + randn(rng, n))
                if t > burn
                    tt = t - burn
                    for i in 1:N
                        Y[(tt - 1) * N + i, 1] = y[i]
                        Y[(tt - 1) * N + i, 2] = y[N + i]
                    end
                end
            end
            return Y, C0, C1
        end

        @testset "40a. order-l weights are not powers of W" begin
            # Hand-checkable on a path: order l has exactly 2(n-l) ordered
            # pairs, since |i-j| = l for i = 1..n-l in each direction.
            for n in (5, 9, 12)
                G = sv_path(n)
                for l in 1:(n - 1)
                    Wl = order_weights(G, l; row_standardize = false)
                    @test nnz(Wl) == 2 * (n - l)
                    @test all(iszero, diag(Matrix(Wl)))
                end
                # The orders partition every ordered pair exactly once.
                tot = sum(nnz(order_weights(G, l; row_standardize = false))
                          for l in 1:(n - 1))
                @test tot == n * (n - 1)
            end
            # order 0 is the identity, whatever W is.
            @test order_weights(sv_path(7), 0) == sparse(1.0I, 7, 7)

            # And the trap: W² is a different object. On a 4x4 rook lattice
            # it is non-zero on the diagonal and has more non-zeros.
            let nx = 4, ny = 4, I_ = Int[], J_ = Int[]
                idx(i, j) = (j - 1) * nx + i
                for i in 1:nx, j in 1:ny, (di, dj) in ((1, 0), (-1, 0), (0, 1), (0, -1))
                    a, b = i + di, j + dj
                    if 1 <= a <= nx && 1 <= b <= ny
                        push!(I_, idx(i, j)); push!(J_, idx(a, b))
                    end
                end
                A = sparse(I_, J_, ones(length(I_)), 16, 16)
                P = A * A
                O = order_weights(A, 2; row_standardize = false)
                @test count(!iszero, diag(Matrix(P))) == 16
                @test count(!iszero, diag(Matrix(O))) == 0
                @test nnz(P) > nnz(O)
                # W² keeps first-order neighbours where triangles exist; on
                # a bipartite lattice there are none, so test the claim on a
                # graph that has one.
                let G = sparse([1, 2, 1, 3, 2, 3, 1, 4],
                               [2, 1, 3, 1, 3, 2, 4, 1], ones(8), 4, 4),
                    Gp = G * G,
                    G2 = order_weights(G, 2; row_standardize = false)
                    kept_p = count(ij -> G[ij...] != 0 && Gp[ij...] != 0,
                                   [(i, j) for i in 1:4, j in 1:4 if i != j])
                    kept_o = count(ij -> G[ij...] != 0 && G2[ij...] != 0,
                                   [(i, j) for i in 1:4, j in 1:4 if i != j])
                    @test kept_p == 6
                    @test kept_o == 0
                end
            end
            # A disconnected pair has no order-2 link, so the list refuses
            # rather than handing back a structurally empty W⁽²⁾.
            @test_throws ArgumentError order_weights_list(
                sparse([1, 2], [2, 1], ones(2), 2, 2), 2)
            @test_throws ArgumentError order_weights(sv_path(5), -1)
        end

        # `binary = false` weights each order-l link by 1/d, where d is
        # the number of distinct SHORTEST paths. On a rook lattice every
        # shortest path is monotone, so d between (1,1) and (a+1,b+1) is
        # exactly the binomial C(a+b, a) -- an independent closed form,
        # not a tolerance. Checked because the keyword is documented, and
        # a documented keyword no run has exercised is the thing this
        # project's standard exists to prevent.
        @testset "40a2. binary = false counts shortest paths" begin
            G = let nx = 4, ny = 4
                n = nx * ny
                I_ = Int[]
                J_ = Int[]
                idx(i, j) = (j - 1) * nx + i
                for i in 1:nx, j in 1:ny,
                    (di, dj) in ((1, 0), (-1, 0), (0, 1), (0, -1))
                    a, b = i + di, j + dj
                    if 1 <= a <= nx && 1 <= b <= ny
                        push!(I_, idx(i, j))
                        push!(J_, idx(a, b))
                    end
                end
                sparse(I_, J_, ones(length(I_)), n, n)
            end
            gidx(i, j) = (j - 1) * 4 + i
            nchoosek(n, k) = factorial(n) ÷ (factorial(k) * factorial(n - k))
            for (a, b) in ((1, 1), (2, 1), (2, 2), (3, 1), (3, 2), (3, 3))
                Wl = order_weights(G, a + b; row_standardize = false,
                                   binary = false)
                @test isapprox(Wl[gidx(1, 1), gidx(a + 1, b + 1)],
                               1 / nchoosek(a + b, a); atol = 1e-12)
            end
            # A 4-ring: the opposite unit is reached by exactly two
            # shortest paths, so the weight is 1/2 and there are N of them.
            let W2 = order_weights(sv_ring(4), 2; row_standardize = false,
                                   binary = false)
                @test isapprox(W2[1, 3], 0.5; atol = 1e-14)
                @test nnz(W2) == 4
            end
            # A 6-ring: order 2 is single-path, order 3 is the diameter and
            # is reached two ways.
            let W2 = order_weights(sv_ring(6), 2; row_standardize = false,
                                   binary = false),
                W3 = order_weights(sv_ring(6), 3; row_standardize = false,
                                   binary = false)
                @test isapprox(W2[1, 3], 1.0; atol = 1e-14)
                @test isapprox(W2[1, 5], 1.0; atol = 1e-14)
                @test isapprox(W3[1, 4], 0.5; atol = 1e-14)
            end
            # The two settings differ only in the VALUES: same pattern,
            # and `binary = true` is all ones before standardisation.
            for l in 1:6
                Wb = order_weights(G, l; row_standardize = false,
                                   binary = true)
                Wp = order_weights(G, l; row_standardize = false,
                                   binary = false)
                @test nnz(Wb) == nnz(Wp)
                @test Wb.rowval == Wp.rowval
                @test all(isone, nonzeros(Wb))
            end
        end

        @testset "40b. the order condition reproduces the paper" begin
            # Di Giacinto's own worked numbers, for the heterogeneous
            # parameterisation his eq. (13) counts: K=2 needs N≥3 at s=1
            # and N≥4 at s=2. Integer bounds, not tolerances.
            for (K, s, want) in ((2, 1, 3), (2, 2, 4))
                @test spvar_order_condition(want, K, s;
                                            homogeneous = false).n_min == want
                @test spvar_order_condition(want, K, s;
                                            homogeneous = false).satisfied
                @test !spvar_order_condition(want - 1, K, s;
                                             homogeneous = false).satisfied
            end
            # The free-coefficient count derived here, N(s+1)K(K-1)/2 + NsK,
            # must equal the paper's printed NK(K-1)/2 + sNK(K+1)/2.
            for N in (3, 5, 12, 40), K in (2, 3, 4), s in (1, 2, 3)
                @test SV._n_free_c0(N, K, s, false) ==
                      div(N * K * (K - 1), 2) + div(s * N * K * (K + 1), 2)
            end
            # The bound falls as K rises, which is the paper's remark.
            @test spvar_order_condition(3, 3, 2; homogeneous = false).n_min <=
                  spvar_order_condition(3, 2, 2; homogeneous = false).n_min
            # Homogeneous coefficients do not scale with N, so the
            # condition binds only for tiny N.
            @test spvar_order_condition(4, 2, 1; homogeneous = true).satisfied
            @test spvar_order_condition(50, 2, 1; homogeneous = true).n_free <
                  spvar_order_condition(50, 2, 1; homogeneous = false).n_free
            @test_throws ArgumentError spvar_order_condition(0, 2, 1)
            @test occursin("order condition",
                           sprint(show, spvar_order_condition(12, 2, 1)))
        end

        @testset "40c. |C₀| factors over the diagonal blocks" begin
            # This identity (from block triangularity) is what makes the
            # concentrated likelihood separable, so it is pinned exactly.
            rng = MersenneTwister(40_301)
            N, T = 10, 200
            Wl = order_weights_list(sv_ring(N), 1)
            Y, = sv_sim(rng, Wl, N, T)
            m = fit_spvar(Y, Wl[2], N, T; ordering = [1, 2], s = 1, Wl = Wl)
            blk(k) = ((k - 1) * N + 1):(k * N)
            full = logabsdet(m.C0)[1]
            parts = sum(logabsdet(m.C0[blk(k), blk(k)])[1] for k in 1:m.K)
            @test isapprox(full, parts; rtol = 1e-12, atol = 1e-12)
            # C₀ is block LOWER triangular: the (1,2) block is exactly zero.
            @test all(iszero, m.C0[blk(1), blk(2)])
            @test !all(iszero, m.C0[blk(2), blk(1)])
        end

        @testset "40d. the concentrated likelihood is separable" begin
            rng = MersenneTwister(40_401)
            N, T = 10, 200
            Wl = order_weights_list(sv_ring(N), 1)
            Y, = sv_sim(rng, Wl, N, T)
            m = fit_spvar(Y, Wl[2], N, T; ordering = [1, 2], s = 1, Wl = Wl)
            cst = -(N * m.K * m.T_eff) / 2 * log(2π) - (N * m.K * m.T_eff) / 2
            @test isapprox(sum(m.loglik_eq) + cst, m.loglik;
                           rtol = 1e-12, atol = 1e-10)
            @test length(m.loglik_eq) == m.K
        end

        @testset "40e. s = 0 is the nested panel VAR" begin
            # fit_nested_panel_var is written independently of fit_spvar —
            # no shared solver — so this is a real cross-check, not a
            # tautology. With s = 0 every W⁽ˡ⁾ block collapses to W⁽⁰⁾ = I,
            # A⁽⁰⁾ₖₖ = I and the Jacobian vanishes: eqs. (15)-(18) exactly.
            rng = MersenneTwister(40_501)
            N, T = 9, 180
            Wl = order_weights_list(sv_ring(N), 1)
            Y, = sv_sim(rng, Wl, N, T)
            m0 = fit_spvar(Y, Wl[2], N, T; ordering = [1, 2], s = 0)
            nv = fit_nested_panel_var(Y, N, T; ordering = [1, 2],
                                      homogeneous = true)
            @test isapprox(m0.loglik, nv.loglik; rtol = 1e-10, atol = 1e-10)
            for k in 1:2
                @test isapprox(m0.theta[k], vec(nv.coefs[k]);
                               rtol = 1e-8, atol = 1e-10)
            end
            @test all(isempty, m0.rho)
            # No Jacobian at s = 0: log|C₀| = 0 because C₀ is unit
            # triangular in the blocks.
            @test isapprox(logabsdet(m0.C0)[1], 0.0; atol = 1e-12)
            # The heterogeneous nested VAR has MORE parameters, so it is not
            # nested in the homogeneous SpVAR and must not be LR-tested
            # against it. Recorded as an assertion so the claim is pinned.
            let nvh = fit_nested_panel_var(Y, N, T; ordering = [1, 2],
                                           homogeneous = false)
                @test nvh.n_params > nv.n_params
                @test nvh.loglik >= nv.loglik
            end
            @test occursin("Nested panel VAR", sprint(show, nv))
        end

        @testset "40f. IRF sums to the long-run multiplier" begin
            rng = MersenneTwister(40_601)
            N, T = 10, 200
            Wl = order_weights_list(sv_ring(N), 1)
            Y, = sv_sim(rng, Wl, N, T)
            m = fit_spvar(Y, Wl[2], N, T; ordering = [1, 2], s = 1, Wl = Wl)
            @test m.stationary
            psi = spvar_irf(m; horizon = 400)
            @test psi[1] ≈ inv(m.C0)
            acc = zeros(2N, 2N)
            for P in psi
                acc .+= P
            end
            @test isapprox(acc, inv(m.C0 - m.C[1]); rtol = 1e-8, atol = 1e-10)
            # Convergence is monotone in the horizon, not accidental.
            errs = Float64[]
            for H in (5, 20, 80)
                a = zeros(2N, 2N)
                for P in spvar_irf(m; horizon = H)
                    a .+= P
                end
                push!(errs, maximum(abs, a .- inv(m.C0 - m.C[1])))
            end
            @test issorted(errs; rev = true)
            @test_throws ArgumentError spvar_irf(m; horizon = -1)
        end

        @testset "40g. outward and inward STIR" begin
            # They coincide exactly when W⁽ˡ⁾ is symmetric and K = 1, since
            # the blocks are then commuting polynomials in W. A RING keeps
            # that under row standardisation; a PATH does not, and that is
            # asserted too rather than assumed.
            rng = MersenneTwister(40_701)
            N, T = 12, 300
            for (regular, Wl) in ((true, order_weights_list(sv_ring(N), 1)),
                                  (false, order_weights_list(sv_path(N), 1)))
                W1 = Wl[2]
                asym = maximum(abs, Matrix(W1) .- Matrix(W1)')
                @test (asym <= 1e-12) == regular
                Wd = Matrix(W1); Im = Matrix(1.0I, N, N)
                C0 = Im - 0.3 * Wd; C1 = 0.3 * Im + 0.1 * Wd
                F = lu(C0); y = zeros(N)
                Y1 = Matrix{Float64}(undef, N * T, 1)
                for t in 1:(T + 150)
                    y = F \ (C1 * y + randn(rng, N))
                    if t > 150
                        for i in 1:N
                            Y1[(t - 151) * N + i, 1] = y[i]
                        end
                    end
                end
                m1 = fit_spvar(Y1, W1, N, T; ordering = [1], s = 1, Wl = Wl)
                st = space_time_stir(m1; horizon = 6)
                gap = maximum(abs, st.outward .- st.inward)
                if regular
                    @test gap < 1e-10
                else
                    @test gap > 1e-4
                end
            end
            # Order 0 uses W⁽⁰⁾ = I, so the outward STIR there is the mean
            # diagonal of the response block: the own-location effect.
            let Wl = order_weights_list(sv_ring(10), 1)
                Y, = sv_sim(MersenneTwister(40_702), Wl, 10, 200)
                m = fit_spvar(Y, Wl[2], 10, 200; ordering = [1, 2], s = 1,
                              Wl = Wl)
                st = space_time_stir(m; horizon = 3)
                psi = spvar_irf(m; horizon = 3)
                blk(k) = ((k - 1) * 10 + 1):(k * 10)
                for h in 0:3, k in 1:2, r in 1:2
                    @test isapprox(st.outward[h + 1, 1, k, r],
                        mean(diag(psi[h + 1][blk(k), blk(r)]));
                        rtol = 1e-10, atol = 1e-12)
                end
                # Cumulative accumulates; horizon 0 is unchanged.
                let sc = space_time_stir(m; horizon = 5, cumulative = true)
                    @test sc.outward[1, :, :, :] ≈ st.outward[1, :, :, :]
                    @test isapprox(sc.outward[3, 2, 1, 1],
                        sum(st.outward[1:3, 2, 1, 1]);
                        rtol = 1e-10, atol = 1e-12)
                end
                # store_local averages to the global version, eqs. (31)-(32).
                let sl = space_time_stir(m; horizon = 2, store_local = true)
                    @test size(sl.local_outward, 5) == 10
                    for h in 0:2, k in 1:2, r in 1:2, l in 1:2
                        @test isapprox(sl.outward[h + 1, l, k, r],
                            mean(sl.local_outward[h + 1, l, k, r, :]);
                            rtol = 1e-10, atol = 1e-12)
                    end
                end
                @test occursin("STIR", sprint(show, st))
            end
        end

        @testset "40h. forecasts, and y_history is required" begin
            rng = MersenneTwister(40_801)
            N, T, H = 10, 250, 4
            Wl = order_weights_list(sv_ring(N), 1)
            Y, = sv_sim(rng, Wl, N, T + H)
            m = fit_spvar(Y[1:(N * T), :], Wl[2], N, T; ordering = [1, 2],
                          s = 1, Wl = Wl)
            hist = Y[(N * (T - m.p) + 1):(N * T), :]
            f = spvar_forecast(m, H; y_history = hist)
            @test size(f) == (N * H, 2)
            @test all(isfinite, f)
            # Beats a no-change forecast on this stationary DGP.
            act = Y[(N * T + 1):(N * (T + H)), :]
            naive = repeat(hist, H, 1)
            @test sqrt(mean(abs2, f .- act)) < sqrt(mean(abs2, naive .- act))
            @test_throws ArgumentError spvar_forecast(m, H)
            @test_throws ArgumentError spvar_forecast(m, 0;
                                                      y_history = hist)
            @test_throws DimensionMismatch spvar_forecast(m, H;
                y_history = hist[:, 1:1])
            # One step ahead is exactly the reduced form applied to history.
            let f1 = spvar_forecast(m, 1; y_history = hist),
                Ct = lu(m.C0) \ m.C[1],
                v = Vector{Float64}(undef, 2N)
                for k in 1:2, i in 1:N
                    v[(k - 1) * N + i] =
                        hist[i, m.ordering[k]] - m.means[i, k]
                end
                pred = Ct * v
                for k in 1:2, i in 1:N
                    @test isapprox(f1[i, m.ordering[k]],
                        pred[(k - 1) * N + i] + m.means[i, k];
                        rtol = 1e-10, atol = 1e-12)
                end
            end
        end

        @testset "40i. the ordering is an assumption, and it binds" begin
            rng = MersenneTwister(40_901)
            N, T = 12, 300
            Wl = order_weights_list(sv_ring(N), 1)
            Y, = sv_sim(rng, Wl, N, T)
            # No default: identification of C₀ rests on it.
            @test_throws ArgumentError fit_spvar(Y, Wl[2], N, T)
            @test_throws ArgumentError fit_spvar(Y, Wl[2], N, T;
                                                 ordering = [1, 1])
            @test_throws ArgumentError fit_spvar(Y, Wl[2], N, T;
                                                 ordering = [1, 2, 3])
            # Names work, and agree with the equivalent integer ordering.
            let ma = fit_spvar(Y, Wl[2], N, T; ordering = [2, 1], s = 1,
                               Wl = Wl, varnames = ["g", "y"]),
                mb = fit_spvar(Y, Wl[2], N, T; ordering = ["y", "g"], s = 1,
                               Wl = Wl, varnames = ["g", "y"])
                @test ma.ordering == mb.ordering == [2, 1]
                @test isapprox(ma.loglik, mb.loglik; rtol = 1e-12)
                @test ma.varnames == ["y", "g"]
            end
            @test_throws ArgumentError fit_spvar(Y, Wl[2], N, T;
                ordering = ["g", "nope"], varnames = ["g", "y"])
            # Reversing it changes the answer. The magnitude is measured in
            # docs/AUDIT.md; here only that it is not a rounding error.
            let m1 = fit_spvar(Y, Wl[2], N, T; ordering = [1, 2], s = 1,
                               Wl = Wl),
                m2 = fit_spvar(Y, Wl[2], N, T; ordering = [2, 1], s = 1,
                               Wl = Wl)
                @test abs(m1.loglik - m2.loglik) > 1.0
                s1 = space_time_stir(m1; horizon = 8)
                s2 = space_time_stir(m2; horizon = 8)
                @test maximum(abs, s1.outward) != maximum(abs, s2.outward)
            end
        end

        @testset "40j. refusals are named" begin
            rng = MersenneTwister(41_001)
            N, T = 10, 120
            Wl = order_weights_list(sv_ring(N), 1)
            Y, = sv_sim(rng, Wl, N, T)
            W1 = Wl[2]
            @test_throws DimensionMismatch fit_spvar(Y, W1, N, T + 1;
                                                      ordering = [1, 2])
            @test_throws DimensionMismatch fit_spvar(Y, sv_ring(N + 1), N, T;
                                                      ordering = [1, 2])
            @test_throws ArgumentError fit_spvar(Y, W1, N, T;
                                                  ordering = [1, 2], p = 0)
            @test_throws ArgumentError fit_spvar(Y, W1, N, T;
                                                  ordering = [1, 2], s = -1)
            # T_per > p + 1, with Y shaped to match so the row check
            # does not fire first and mask it.
            @test_throws ArgumentError fit_spvar(Y[1:(N * 2), :], W1, N, 2;
                                                  ordering = [1, 2])
            @test_throws DimensionMismatch fit_spvar(Y, W1, N, T;
                ordering = [1, 2], s = 1, Wl = Wl[1:1])
            let Ybad = copy(Y); Ybad[1, 1] = Inf
                @test_throws ArgumentError fit_spvar(Ybad, W1, N, T;
                                                      ordering = [1, 2])
            end
            @test_throws DimensionMismatch fit_spvar(Y, W1, N, T;
                ordering = [1, 2], varnames = ["only one"])
            # The order condition refuses before any estimation happens.
            @test_throws ArgumentError fit_spvar(randn(2 * 40, 2),
                sparse([1, 2], [2, 1], ones(2), 2, 2), 2, 40;
                ordering = [1, 2], s = 2)
            # s = 0 has nothing to test for spatial interaction.
            let m0 = fit_spvar(Y, W1, N, T; ordering = [1, 2], s = 0)
                @test_throws ArgumentError lr_spatial_interaction(m0, Y, W1)
            end
            let m = fit_spvar(Y, W1, N, T; ordering = [1, 2], s = 1, Wl = Wl)
                @test_throws DimensionMismatch lr_spatial_interaction(
                    m, Y[:, 1:1], W1)
            end
        end

        @testset "40k. the LR test and its degrees of freedom" begin
            rng = MersenneTwister(41_101)
            N, T = 12, 300
            Wl = order_weights_list(sv_ring(N), 1)
            Y, = sv_sim(rng, Wl, N, T)
            m = fit_spvar(Y, Wl[2], N, T; ordering = [1, 2], s = 1, Wl = Wl)
            lr = lr_spatial_interaction(m, Y, Wl[2])
            # df counted from the parameter vectors must match the formula
            # s(K(K+1)/2 + K²p) derived from the block structure.
            @test lr.df == 1 * (div(2 * 3, 2) + 4 * 1)
            @test lr.stat > 0
            @test lr.loglik_full >= lr.loglik_restricted
            @test 0 <= lr.p_value <= 1
            # On a DGP with real spatial interaction it rejects decisively.
            @test lr.p_value < 1e-6
            # And the formula holds across shapes.
            for (K, p, s) in ((2, 1, 1), (2, 2, 1), (3, 1, 1))
                nf = sum(s + s * (k - 1) + s * K * p for k in 1:K)
                @test nf == s * (div(K * (K + 1), 2) + K^2 * p)
            end
        end

        # ── 40m ────────────────────────────────────────────────────────
        # The K = 1 reductions. `fit_spvar` absorbs μ by demeaning each
        # unit's LEVEL series once, over t = p+1..T_per, which is what keeps
        # (I − ρW)ỹ exact when s ≥ 1: the spatial lag of a demeaned series
        # is not the demeaned spatial lag, so demeaning the constructed
        # columns separately would break the structural form.
        #
        # At s = 0 there is no spatial term, and the exact profiled FIML
        # instead demeans the lagged and current sub-series by their OWN
        # means, which differ by (y_T − y_1)/(T−1). So the two disagree in
        # finite samples. The rate was measured rather than guessed, against
        # an independently written profiled FIML, N = 10, φ = 0.55,
        # heteroskedastic λ ∈ [0.5, 1.5]:
        #
        #   T  =  150  300  600  1200  2400  4800
        #   gap = 6.23e-5 1.45e-5 4.42e-6 1.11e-6 3.90e-7 5.06e-8
        #
        # The ratio per doubling averages about 4, not 2, and T²·gap is
        # roughly constant in 1.2–2.3 while T·gap falls steadily. So the
        # two ways of concentrating μ agree to **second** order, not first
        # -- a stronger result than the O(1/T) first guessed, and the reason
        # this is a documented property rather than a defect.
        @testset "40m. the K = 1 reductions" begin
            function sv_ar1(rng, N, T, phi, lam)
                mu = randn(rng, N)
                y = zeros(N)
                Y = Matrix{Float64}(undef, N * T, 1)
                for t in 1:(T + 200)
                    y = mu .+ phi .* y .+ sqrt.(lam) .* randn(rng, N)
                    if t > 200
                        Y[((t - 201) * N + 1):((t - 200) * N), 1] = y
                    end
                end
                return Y
            end
            # An independently written exact profiled FIML: μ concentrated
            # by the likelihood's own first-order condition, λ by eq (21).
            function sv_ar1_fiml(Y, N, T; iters = 4000)
                yv = reshape(view(Y, :, 1), N, T)
                yl = yv[:, 1:(T - 1)]
                yc = yv[:, 2:T]
                lam = ones(N)
                b = 0.0
                for _ in 1:iters
                    num = 0.0
                    den = 0.0
                    for i in 1:N
                        xi = yl[i, :] .- mean(yl[i, :])
                        zi = yc[i, :] .- mean(yc[i, :])
                        num += sum(xi .* zi) / lam[i]
                        den += sum(xi .* xi) / lam[i]
                    end
                    b = num / den
                    for i in 1:N
                        a = mean(yc[i, :]) - b * mean(yl[i, :])
                        r = yc[i, :] .- a .- b .* yl[i, :]
                        lam[i] = sum(abs2, r) / (T - 1)
                    end
                end
                return b, lam
            end

            Nr = 10
            Wr = sv_ring(Nr)
            lamr = collect(range(0.5, 1.5; length = Nr))
            gaps = Float64[]
            for Tr in (300, 1200)
                Yr = sv_ar1(MersenneTwister(4041), Nr, Tr, 0.55, lamr)
                mr = fit_spvar(Yr, Wr, Nr, Tr; ordering = [1], p = 1, s = 0,
                               varnames = ["y"])
                br, lmr = sv_ar1_fiml(Yr, Nr, Tr)
                @test mr.converged
                # Both are estimating the same φ, so they must agree to
                # well inside sampling error at either T.
                @test abs(br - mr.theta[1][1]) < 1e-4
                @test maximum(abs, lmr .- vec(mr.lambda)) < 1e-4
                push!(gaps, abs(br - mr.theta[1][1]))
            end
            # Assert the IMPLICATION of second-order agreement: a fourfold
            # increase in T must shrink the gap by far more than fourfold.
            # Measured ratio 13.1 against the 5 asserted, so the margin is
            # 2.6x -- checked rather than assumed, per testset 38i.
            @test gaps[1] / gaps[2] > 5.0

            # s = 1, K = 1: the single-equation spatial case recovers its
            # own parameters. This is the specification `fit_star` cannot
            # express, since it has a contemporaneous Wyₜ.
            let Tr = 300
                WLr = order_weights_list(Wr, 1)
                W1r = Matrix(WLr[2])
                rng = MersenneTwister(4042)
                Fr = lu(Matrix(1.0I, Nr, Nr) .- 0.40 .* W1r)
                mur = randn(rng, Nr)
                y = zeros(Nr)
                Y2 = Matrix{Float64}(undef, Nr * Tr, 1)
                for t in 1:(Tr + 200)
                    y = Fr \ (mur .+ 0.30 .* y .+ 0.10 .* (W1r * y) .+
                              randn(rng, Nr))
                    if t > 200
                        Y2[((t - 201) * Nr + 1):((t - 200) * Nr), 1] = y
                    end
                end
                m2 = fit_spvar(Y2, Wr, Nr, Tr; ordering = [1], p = 1, s = 1,
                               Wl = WLr, varnames = ["y"])
                @test m2.converged
                @test m2.K == 1
                @test isapprox(m2.rho[1][1], 0.40; atol = 0.06)
                @test isapprox(m2.theta[1][1], 0.30; atol = 0.05)
                @test isapprox(m2.theta[1][2], 0.10; atol = 0.05)
                @test m2.stationary
                # At K = 1 the structural and reduced forms differ only by
                # the Jacobian, so |C₀| is a single determinant.
                @test isapprox(logabsdet(m2.C0)[1],
                               logabsdet(Matrix(1.0I, Nr, Nr) .-
                                         m2.rho[1][1] .* W1r)[1];
                               atol = 1e-10)

                # `fit_dynamic_spatial_panel` builds its own time lag, so
                # handing it y(t−1) as a regressor is exactly collinear
                # with that lag and it refuses. Pinned, because the refusal
                # is the reason these two estimators cannot be compared
                # coefficient by coefficient.
                yv = reshape(view(Y2, :, 1), Nr, Tr)
                @test_throws ArgumentError fit_dynamic_spatial_panel(
                    SAR(), IndivFE(), Wr,
                    reshape(vec(yv[:, 1:(Tr - 1)]), :, 1),
                    vec(yv[:, 2:Tr]), Nr, Tr - 1)
            end
        end

        # ── 40n ────────────────────────────────────────────────────────
        # p > 1. Every other subset here runs at the p = 1 default, so the
        # multi-lag path -- the loop over C_h, the (t-2) coefficient names,
        # the Kp.N companion matrix, and the longer history a forecast
        # needs -- was shipping unexercised. The companion matrix is the
        # part that matters: at p = 1 it IS the reduced-form matrix, so its
        # block structure is only non-trivial from p = 2.
        @testset "40n. p > 1, and rho_bounds" begin
            Np, Tp = 12, 600
            Wp = sv_ring(Np)
            WLp = order_weights_list(Wp, 1)
            W1p = Matrix(WLp[2])
            Imp = Matrix(1.0I, Np, Np)
            Zp = zeros(Np, Np)
            C0p = [Imp - 0.30 * W1p            Zp;
                   -(0.35 * Imp + 0.12 * W1p)  Imp + 0.18 * W1p]
            C1p = [0.25 * Imp + 0.08 * W1p   0.12 * Imp - 0.04 * W1p;
                   -0.08 * Imp + 0.06 * W1p  0.20 * Imp + 0.10 * W1p]
            C2p = [-0.12 * Imp + 0.05 * W1p  0.06 * Imp;
                   0.05 * Imp                -0.10 * Imp + 0.04 * W1p]
            Fp = lu(C0p)
            # Assert the DGP is stationary rather than assuming it: the
            # companion form at p = 2 is 2Kp.N square.
            let comp = [Fp \ C1p  Fp \ C2p;
                        Matrix(1.0I, 2Np, 2Np)  zeros(2Np, 2Np)]
                @test maximum(abs, eigvals(comp)) < 1
            end
            rngp = MersenneTwister(2026)
            y1 = zeros(2Np)
            y2 = zeros(2Np)
            Yp = Matrix{Float64}(undef, Np * Tp, 2)
            for t in 1:(Tp + 300)
                yy = Fp \ (C1p * y1 + C2p * y2 + randn(rngp, 2Np))
                y2 = y1
                y1 = yy
                if t > 300
                    k = t - 300
                    for i in 1:Np
                        Yp[(k - 1) * Np + i, 1] = yy[i]
                        Yp[(k - 1) * Np + i, 2] = yy[Np + i]
                    end
                end
            end

            mp = fit_spvar(Yp, Wp, Np, Tp; ordering = [1, 2], p = 2, s = 1,
                           Wl = WLp, varnames = ["a", "b"])
            @test mp.converged
            @test mp.p == 2
            @test length(mp.C) == 2
            @test mp.T_eff == Tp - 2
            @test mp.stationary
            @test isapprox(mp.rho[1][1], 0.30; atol = 0.05)
            @test isapprox(mp.rho[2][1], -0.18; atol = 0.05)
            # The second lag's coefficients must be named, and named with
            # the right lag index -- the obvious way for a multi-lag design
            # to be silently wrong is to label the wrong column.
            @test any(contains("(t-2)"), mp.coefnames_eq[1])
            @test any(contains("(t-1)"), mp.coefnames_eq[1])
            @test count(contains("(t-2)"), mp.coefnames_eq[1]) ==
                  count(contains("(t-1)"), mp.coefnames_eq[1])

            # The long-run multiplier now has TWO lag matrices in it, so
            # this fails if the C_h loop drops or double-counts one.
            let psi = spvar_irf(mp; horizon = 600)
                acc = zeros(2Np, 2Np)
                for P in psi
                    acc .+= P
                end
                @test maximum(abs, acc .- inv(mp.C0 - mp.C[1] - mp.C[2])) <
                      1e-12
            end

            # A p = 2 forecast needs two periods of history, and must
            # refuse one rather than quietly padding.
            let hist2 = Yp[(end - 2Np + 1):end, :]
                f = spvar_forecast(mp, 3; y_history = hist2)
                @test size(f) == (Np * 3, 2)
                @test all(isfinite, f)
            end
            @test_throws DimensionMismatch spvar_forecast(mp, 3;
                y_history = Yp[(end - Np + 1):end, :])

            # `rho_bounds` is a documented keyword, so it is exercised:
            # bounded away from the unconstrained optimum, the estimate
            # must sit ON the boundary rather than outside it.
            let mb = fit_spvar(Yp, Wp, Np, Tp; ordering = [1, 2], p = 2,
                               s = 1, Wl = WLp, rho_bounds = (0.5, 0.9))
                @test mb.rho[1][1] >= 0.5 - 1e-8
                @test mb.rho[1][1] <= 0.9 + 1e-8
                @test isapprox(mb.rho[1][1], 0.5; atol = 1e-4)
                # and a bounded fit is worse, since the truth is outside
                @test mb.loglik < mp.loglik
            end
        end

        @testset "40l. accessors and show" begin
            rng = MersenneTwister(41_201)
            N, T = 10, 200
            Wl = order_weights_list(sv_ring(N), 1)
            Y, = sv_sim(rng, Wl, N, T)
            m = fit_spvar(Y, Wl[2], N, T; ordering = [1, 2], s = 1, Wl = Wl,
                          varnames = ["g", "y"])
            @test nobs(m) == N * m.T_eff
            @test loglikelihood(m) == m.loglik
            @test size(residuals(m)) == (N * m.T_eff, 2)
            @test length(coef(m)) == length(coefnames(m))
            @test length(coef(m)) == length(stderror(m))
            @test size(vcov(m), 1) == length(coef(m))
            @test all(>=(0), stderror(m))
            @test issymmetric(vcov(m))
            @test size(m.lambda) == (N, 2)
            @test all(>(0), m.lambda)
            @test size(m.means) == (N, 2)
            @test m.T_eff == T - m.p
            @test m.homogeneous
            @test occursin("Spatial VAR", sprint(show, m))
            @test occursin("g", sprint(show, m))
            # Residual variances recover the unit structural variances the
            # DGP used, as T grows: the DGP draws u ~ N(0, I).
            @test 0.6 < mean(m.lambda) < 1.5
        end
    end
end
