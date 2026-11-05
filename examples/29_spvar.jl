# ---------------------------------------------------------------------------
# 29. Spatial VAR: the causal ordering is the answer
#
# `fit_star` models ONE variable through space and time. A regional system
# has several, and the question a policy maker actually asks — "does a
# fiscal shock in region i raise output in region j, and after how long?" —
# needs the system.
#
# Identifying that system means ordering the variables contemporaneously.
# Di Giacinto (2010) orders the VARIABLES but not the locations, because a
# recursive causal chain through space would say region 3 can move region 7
# within the period but never the reverse, and spatial interaction is
# bilateral. `ordering` therefore has no default in `fit_spvar`: it is the
# one thing the data cannot settle, so the user has to say it out loud.
#
# This file is about what that choice costs when it is made wrongly, about
# the difference between the order-l weights the model needs and the powers
# of W that look like them, and about the asymptotic regime, which runs the
# opposite way from every other panel estimator in the package.
#
# Run: julia --project=. examples/29_spvar.jl
# ---------------------------------------------------------------------------

using SpatialEconometrics
using LinearAlgebra, SparseArrays, Random, Statistics, Printf
import StatsAPI: stderror, coef, coefnames

const RNG = MersenneTwister(20_261_105)

# A ring of regions: every region has two neighbours, the graph is regular,
# and the order-l sets are easy to reason about by hand.
function ring_W(n)
    I_ = Int[]; J_ = Int[]
    for i in 1:n
        push!(I_, i); push!(J_, mod1(i + 1, n))
        push!(I_, i); push!(J_, mod1(i - 1, n))
    end
    return sparse(I_, J_, ones(2n), n, n)
end

# A homogeneous SpVAR(1,1) with K = 2: g is public spending, y is output.
#
#   C0 = [ I - r1*W            0        ]      g does not respond to y
#        [ -(p0*I + p1*W)    I - r2*W   ]      within the period
#   C1 = the four lagged blocks, each b0*I + b1*W
#
# The TRUTH here is that spending moves first: a shock to g reaches y in the
# same period, both at home (p0) and across the border (p1), while a shock
# to y does not reach g until next period. That is the ordering [g, y].
const TRUTH = (r1 = 0.35, r2 = -0.20, p0 = 0.40, p1 = 0.15,
               a = ((0.30, 0.10), (0.15, -0.05), (-0.10, 0.08), (0.25, 0.12)))

function simulate_spvar(rng, Wl, N, T; burn = 200)
    Wd = Matrix(Wl[2]); Im = Matrix(1.0I, N, N); Z = zeros(N, N)
    t = TRUTH
    C0 = [Im - t.r1 * Wd              Z;
          -(t.p0 * Im + t.p1 * Wd)    Im - t.r2 * Wd]
    C1 = [t.a[1][1] * Im + t.a[1][2] * Wd   t.a[2][1] * Im + t.a[2][2] * Wd;
          t.a[3][1] * Im + t.a[3][2] * Wd   t.a[4][1] * Im + t.a[4][2] * Wd]
    F = lu(C0)
    rad = maximum(abs, eigvals(F \ C1))
    rad < 1 || error("the DGP is explosive, spectral radius $rad")
    n = 2N
    y = zeros(n)
    Y = Matrix{Float64}(undef, N * T, 2)
    for tt in 1:(T + burn)
        y = F \ (C1 * y + randn(rng, n))
        if tt > burn
            k = tt - burn
            for i in 1:N
                Y[(k - 1) * N + i, 1] = y[i]
                Y[(k - 1) * N + i, 2] = y[N + i]
            end
        end
    end
    return Y, rad
end

const N, T = 14, 400
const W = ring_W(N)
# Build the order-l list ONCE and hand the same matrices to the simulator
# and the estimator, so the comparison is about estimation, not about two
# different standardizations of W.
const WL = order_weights_list(W, 1)
const Y, RAD = simulate_spvar(RNG, WL, N, T)

println("="^74)
println("29. Spatial VAR: the causal ordering is the answer")
println("="^74)
@printf("Ring of N = %d regions, T = %d periods, K = 2 variables (g, y).\n",
        N, T)
@printf("DGP spectral radius %.4f.\n", RAD)

# ───────────────────────────────────────────────────────────────────────
# 1. The order condition comes before the estimate
# ───────────────────────────────────────────────────────────────────────
println()
println("-"^74)
println("1. The order condition: N >= s + (s+K)/K")
println("-"^74)
println("Counting equations against free coefficients in C0 bounds N below.")
println("It is an ORDER condition: necessary, not sufficient.")
println()
for (K, s) in ((2, 1), (2, 2), (3, 1), (3, 2))
    oc = spvar_order_condition(N, K, s; homogeneous = false)
    ocmin = spvar_order_condition(oc.n_min, K, s; homogeneous = false)
    ocbad = spvar_order_condition(oc.n_min - 1, K, s; homogeneous = false)
    @printf("  K=%d s=%d: bound N >= %d.  at N=%d %s, at N=%d %s\n",
            K, s, oc.n_min, oc.n_min, ocmin.satisfied ? "ok" : "REFUSED",
            oc.n_min - 1, ocbad.satisfied ? "ok" : "refused")
end
println()
print("  a system below the bound is refused before estimation: ")
try
    fit_spvar(randn(2 * 60, 2), ring_W(2), 2, 60; ordering = [1, 2], s = 2)
    println("NOT REFUSED  <- wrong")
catch e
    println("yes")
end
print("  and `ordering` has no default: ")
try
    fit_spvar(Y, W, N, T)
    println("NOT REFUSED  <- wrong")
catch e
    println("refused, as it must be")
end

# ───────────────────────────────────────────────────────────────────────
# 2. The ordering, got right and got backwards
# ───────────────────────────────────────────────────────────────────────
println()
println("-"^74)
println("2. The same data, two orderings")
println("-"^74)

m_true = fit_spvar(Y, W, N, T; ordering = ["g", "y"], s = 1, Wl = WL,
                   varnames = ["g", "y"])
m_rev  = fit_spvar(Y, W, N, T; ordering = ["y", "g"], s = 1, Wl = WL,
                   varnames = ["g", "y"])

@printf("  log-likelihood   g -> y: %12.3f\n", m_true.loglik)
@printf("                   y -> g: %12.3f   (difference %.3f)\n",
        m_rev.loglik, m_true.loglik - m_rev.loglik)
println()
println("  Recovery under the TRUE ordering (truth in brackets):")
@printf("    eq g:  rho_1   = %+.4f  [%+.2f]\n", m_true.rho[1][1], TRUTH.r1)
@printf("    eq g:  g(t-1)  = %+.4f  [%+.2f]\n", m_true.theta[1][1],
        TRUTH.a[1][1])
@printf("    eq y:  rho_1   = %+.4f  [%+.2f]\n", m_true.rho[2][1], TRUTH.r2)
@printf("    eq y:  g       = %+.4f  [%+.2f]   (contemporaneous)\n",
        m_true.theta[2][1], TRUTH.p0)
@printf("    eq y:  W.g     = %+.4f  [%+.2f]   (contemporaneous, neighbours)\n",
        m_true.theta[2][2], TRUTH.p1)
println()
println("  Under the REVERSED ordering the contemporaneous effect of g on y")
println("  is not a parameter at all — the model has it running the other")
println("  way — so there is nothing to compare it against. The two fits")
println("  are not two estimates of one quantity. They answer different")
println("  questions, and only one of them is the question asked.")

# ───────────────────────────────────────────────────────────────────────
# 3. What that costs in the impulse response
# ───────────────────────────────────────────────────────────────────────
println()
println("-"^74)
println("3. The space-time impulse response, both ways round")
println("-"^74)
println("Outward STIR: a shock in region i reaching its order-l neighbours.")
println()

st_true = space_time_stir(m_true; horizon = 12)
st_rev  = space_time_stir(m_rev;  horizon = 12)

function outward(m, st, from, to, l)
    kf = findfirst(==(from), m.varnames)
    kt = findfirst(==(to), m.varnames)
    return st.outward[:, l + 1, kt, kf]
end

for l in (0, 1)
    ra = outward(m_true, st_true, "g", "y", l)
    rb = outward(m_rev,  st_rev,  "g", "y", l)
    ia, ib = argmax(abs.(ra)), argmax(abs.(rb))
    @printf("  order l = %d: response of y to a g shock\n", l)
    @printf("    g -> y: peak %+.6f at h = %d\n", ra[ia], ia - 1)
    @printf("    y -> g: peak %+.6f at h = %d\n", rb[ib], ib - 1)
    @printf("    ratio of peak magnitudes: %.2fx\n",
            abs(ra[ia]) / max(abs(rb[ib]), 1e-12))
end
println()
println("  The ordering is not a normalization. Contemporaneously it moves")
println("  the peak spillover by a factor of five, and it moves the TIMING")
println("  too: under the true ordering the order-0 response peaks on")
println("  impact, under the reversed one two periods later, because the")
println("  effect has to travel through a lag to exist at all. A paper that")
println("  does not state its ordering has not reported its result.")

# ───────────────────────────────────────────────────────────────────────
# 4. Outward is not inward
# ───────────────────────────────────────────────────────────────────────
println()
println("-"^74)
println("4. Outward and inward differ by a transpose")
println("-"^74)
println("Outward: my shock reaching my neighbours.")
println("Inward:  their shocks reaching me.")
println()
@printf("  K = 2, row-standardized ring:  max|outward - inward| = %.4e\n",
        maximum(abs, st_true.outward .- st_true.inward))
let m1 = fit_spvar(Y[:, 1:1], W, N, T; ordering = [1], s = 1, Wl = WL,
                   varnames = ["g"]),
    st1 = space_time_stir(m1; horizon = 8)
    @printf("  K = 1, same weights:           max|outward - inward| = %.4e\n",
            maximum(abs, st1.outward .- st1.inward))
end
let WLu = order_weights_list(W, 1; row_standardize = false),
    m1 = fit_spvar(Y[:, 1:1], W, N, T; ordering = [1], s = 1, Wl = WLu,
                   varnames = ["g"]),
    st1 = space_time_stir(m1; horizon = 8)
    @printf("  K = 1, UNstandardized ring:    max|outward - inward| = %.4e\n",
            maximum(abs, st1.outward .- st1.inward))
end
# A ring is regular, so row standardization preserves its symmetry. A path
# is not, and that alone separates outward from inward even at K = 1 --
# which is worth measuring rather than asserting.
let P = let n = N; I_ = Int[]; J_ = Int[]
            for i in 1:(n - 1)
                push!(I_, i); push!(J_, i + 1)
                push!(I_, i + 1); push!(J_, i)
            end
            sparse(I_, J_, ones(2 * (n - 1)), n, n)
        end
    # Row standardization and spectral standardization both bound the
    # spectrum so the DGP stays stationary. Only the second preserves
    # symmetry, and that is the whole difference being measured. A raw
    # binary path cannot be used here at all: its eigenvalues reach 2, the
    # DGP above is explosive on it, and `simulate_spvar` refuses -- which
    # is the guard doing its job, not an obstacle to work around.
    raw = order_weights(P, 1; row_standardize = false)
    for (lbl, W1p) in (("row-standardized", order_weights(P, 1)),
                       ("spectrally std.", spectral_standardize(raw)))
        WLp = [sparse(1.0I, N, N), W1p]
        asym = maximum(abs, Matrix(WLp[2]) .- Matrix(WLp[2])')
        Yp, = simulate_spvar(MersenneTwister(5150), WLp, N, T)
        m1 = fit_spvar(Yp[:, 1:1], WLp[2], N, T; ordering = [1], s = 1,
                       Wl = WLp, varnames = ["g"])
        st1 = space_time_stir(m1; horizon = 8)
        @printf("  K = 1, PATH %-17s |W-W'| = %.1e, STIR gap = %.3e\n",
                lbl * ":", asym, maximum(abs, st1.outward .- st1.inward))
    end
end
println()
println("  At K = 1 with symmetric W the two coincide, because Psi is then")
println("  a polynomial in symmetric matrices. A ring is regular, so row")
println("  standardization preserves that symmetry. A path is not: its two")
println("  end units have one neighbour where the rest have two, so row")
println("  standardization alone makes outward and inward differ. That is")
println("  not a bug, and it is not something to tolerance away.")

# ───────────────────────────────────────────────────────────────────────
# 5. Is there any spatial interaction to model?
# ───────────────────────────────────────────────────────────────────────
println()
println("-"^74)
println("5. The nested panel VAR, and the test against it")
println("-"^74)
println("Setting every spatial coefficient to zero leaves one small VAR per")
println("region. fit_nested_panel_var is written independently, which is")
println("what makes the reduction worth checking.")
println()
let m0 = fit_spvar(Y, W, N, T; ordering = ["g", "y"], s = 0,
                   varnames = ["g", "y"]),
    nv = fit_nested_panel_var(Y, N, T; ordering = ["g", "y"],
                              homogeneous = true, varnames = ["g", "y"])
    @printf("  s = 0 loglik vs nested panel VAR:  %.3e\n",
            abs(m0.loglik - nv.loglik))
    dmax = maximum(maximum(abs, m0.theta[k] .- vec(nv.coefs[k])) for k in 1:2)
    @printf("  max |coefficient difference|:      %.3e\n", dmax)
end
let lr = lr_spatial_interaction(m_true, Y, W)
    @printf("  LR = %.3f on %d df, p = %.3e\n", lr.stat, lr.df, lr.p_value)
    @printf("  df = s(K(K+1)/2 + K^2 p) = %d\n", 1 * (3 + 4))
end

# ───────────────────────────────────────────────────────────────────────
# 6. The weights are order-l, not powers of W
# ───────────────────────────────────────────────────────────────────────
println()
println("-"^74)
println("6. W^(l) is not W^l")
println("-"^74)
println("Di Giacinto's W^(l) is the matrix of order-l NEIGHBOURS. Powers of")
println("W count WALKS: they are non-zero on the diagonal, they double-count")
println("paths, and where the graph has triangles they keep first-order")
println("neighbours at order 2.")
println()
let G = let nx = 4, ny = 4
            n = nx * ny; I_ = Int[]; J_ = Int[]
            idx(i, j) = (j - 1) * nx + i
            for i in 1:nx, j in 1:ny, (di, dj) in ((1,0),(-1,0),(0,1),(0,-1))
                a, b = i + di, j + dj
                if 1 <= a <= nx && 1 <= b <= ny
                    push!(I_, idx(i, j)); push!(J_, idx(a, b))
                end
            end
            sparse(I_, J_, ones(length(I_)), n, n)
        end
    W2 = G * G
    O2 = order_weights(G, 2; row_standardize = false)
    @printf("  4x4 rook lattice, N = 16, %d first-order links\n", nnz(G))
    @printf("    W^2:                  %3d non-zeros, %2d on the diagonal\n",
            nnz(W2), count(!iszero, diag(Matrix(W2))))
    @printf("    order_weights(W, 2):  %3d non-zeros, %2d on the diagonal\n",
            nnz(O2), count(!iszero, diag(Matrix(O2))))
    tot = 0
    for l in 1:6
        Ol = order_weights(G, l; row_standardize = false)
        tot += nnz(Ol)
    end
    @printf("    orders 1..6 cover %d pairs; N(N-1) = %d\n", tot, 16 * 15)
end
println()
println("  Every ordered pair appears at exactly one order. That is the")
println("  property the model needs and the property powers do not have.")

# ───────────────────────────────────────────────────────────────────────
# 7. The asymptotics run the other way
# ───────────────────────────────────────────────────────────────────────
println()
println("-"^74)
println("7. T -> infinity at FIXED N")
println("-"^74)
println("Every other panel estimator in this package is consistent as N")
println("grows. This one is not. The bias in the dynamic coefficient g(t-1)")
println("(truth 0.30), 60 replications:")
println()
@printf("  %-22s %-12s %-12s\n", "", "bias", "RMSE")
println("  -- N fixed at 12, T growing --")
for Tn in (50, 200, 800)
    rng = MersenneTwister(2900 + Tn)
    Wl = order_weights_list(ring_W(12), 1)
    b = Float64[]
    for _ in 1:60
        Yr, = simulate_spvar(rng, Wl, 12, Tn)
        m = fit_spvar(Yr, Wl[2], 12, Tn; ordering = [1, 2], s = 1, Wl = Wl)
        push!(b, m.theta[1][1] - TRUTH.a[1][1])
    end
    @printf("  T = %-18d %+-12.4f %-12.4f\n", Tn, mean(b),
            sqrt(mean(abs2, b)))
end
println("  -- T fixed at 50, N growing --")
for Nn in (12, 48, 96)
    rng = MersenneTwister(3700 + Nn)
    Wl = order_weights_list(ring_W(Nn), 1)
    b = Float64[]
    for _ in 1:60
        Yr, = simulate_spvar(rng, Wl, Nn, 50)
        m = fit_spvar(Yr, Wl[2], Nn, 50; ordering = [1, 2], s = 1, Wl = Wl)
        push!(b, m.theta[1][1] - TRUTH.a[1][1])
    end
    @printf("  N = %-18d %+-12.4f %-12.4f\n", Nn, mean(b),
            sqrt(mean(abs2, b)))
end
println()
println("  Growing T removes the bias. Growing N does not touch it: it buys")
println("  variance, and nothing else. A wide short panel is the wrong")
println("  shape of data for this estimator, and the package says so in the")
println("  docstring rather than letting the user find out.")

# ───────────────────────────────────────────────────────────────────────
# 8. Forecasting
# ───────────────────────────────────────────────────────────────────────
println()
println("-"^74)
println("8. Forecasting the system")
println("-"^74)
let Tfit = T - 12
    Yfit = Y[1:(N * Tfit), :]
    mf = fit_spvar(Yfit, W, N, Tfit; ordering = ["g", "y"], s = 1, Wl = WL,
                   varnames = ["g", "y"])
    hist = Yfit[(end - N + 1):end, :]
    fc = spvar_forecast(mf, 12; y_history = hist)
    act = Y[(N * Tfit + 1):end, :]
    # The benchmark is each unit's IN-SAMPLE mean, not zero. `fit_spvar`
    # absorbs a per-unit intercept and `spvar_forecast` adds it back, so a
    # zero forecast would only look competitive on a DGP whose levels
    # happen to sit near zero -- which flatters the model for the wrong
    # reason and hides whether the intercepts were restored at all.
    ybar = [mean(Yfit[:, k]) for k in 1:2]
    # Averaging over all 12 horizons hides where the gain is. With a
    # companion radius of 0.74 the forecast has decayed most of the way to
    # the unconditional mean within a handful of periods, after which no
    # method can beat the mean -- so the average is dominated by horizons
    # where the model is not expected to win. Report it per horizon.
    @printf("  %-5s %-10s %-10s %-10s %-10s\n", "h",
            "g: model", "g: mean", "y: model", "y: mean")
    for h in (1, 2, 3, 4, 8, 12)
        row = ((h - 1) * N + 1):(h * N)
        vals = Float64[]
        for k in 1:2
            push!(vals, sqrt(mean(abs2, fc[row, k] .- act[row, k])))
            push!(vals, sqrt(mean(abs2, act[row, k] .- ybar[k])))
        end
        @printf("  %-5d %-10.4f %-10.4f %-10.4f %-10.4f\n", h,
                vals[1], vals[2], vals[3], vals[4])
    end
    println()
    for k in 1:2
        rm = sqrt(mean(abs2, fc[:, k] .- act[:, k]))
        rmean = sqrt(mean(abs2, act[:, k] .- ybar[k]))
        @printf("  %s: over all 12 horizons, %.4f against %.4f (%.2fx)\n",
                mf.varnames[k], rm, rmean, rmean / rm)
    end
    @printf("  companion spectral radius %.4f, stationary = %s\n",
            mf.spectral_radius, mf.stationary)
end
println()
println("  The gain is concentrated at h = 1 -- 1.20 against 1.47 for g and")
println("  1.10 against 1.29 for y -- and it is essentially gone by h = 2.")
println("  At h = 3 the model is MARGINALLY WORSE than the sample mean. All")
println("  of that follows from a companion radius of 0.74: 0.74^2 is about")
println("  0.55 and 0.74^4 about 0.30, so little of today's state survives")
println("  even a few periods, and past that point the per-horizon figures")
println("  are one draw over 14 units and mostly sampling noise.")
println()
println("  Averaged over twelve horizons this reads as 1.03x and looks like")
println("  a failure. Read per horizon it is the model doing what the")
println("  spectral radius says it can, and no more. Both cuts are printed,")
println("  because an example that shows only the flattering one is not an")
println("  example.")

println()
println("="^74)
println("Di Giacinto, V. (2010). On vector autoregressive modeling in space")
println("and time. Journal of Geographical Systems, 12(2), 125-154.")
println("doi:10.1007/s10109-010-0116-6")
println("="^74)
