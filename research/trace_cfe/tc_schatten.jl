# Theorem 1 without the log factor: is the triangular part bounded in the Schatten-4 norm?
#
# Bhansali, Giraitis & Kokoszka (2007, Lemma 2.1, via Matsaev 1961; the bound is Theorem 1 of Dokl. 139(4), 810-813) use that triangular truncation is
# bounded on the Schatten-p class for 1 < p < ∞, with a constant free of n: ‖L‖₄ ≤ C₄‖A‖₄ for the strictly
# lower-triangular part L of a symmetric A. In the operator norm the same truncation can grow like log n
# (THEORY.md, Lemma 3), and that is where Theorem 1's (log n)² came from.
#
# Two designs:
#   power check : A is the interleaved Hilbert matrix, A[2i, 2j−1] = A[2j−1, 2i] = 1/(i − j + ½), whose
#                 operator norm stays below π while its triangular part grows like log n in the operator
#                 norm. If ‖L‖₄/‖A‖₄ were also unbounded, it would show here.
#   the moment  : Cs = (C + C′)/2 for the exact-diagonal moment, from tc_lib.jl, kNN W and a Chebyshev
#                 surface basis with K ≈ 2√n, as in tc_clt.jl.
# Printed: the operator-norm ratio ‖L‖_op/‖A‖_op, the Schatten-4 ratio ‖L‖₄/‖A‖₄, and the two bounds the
# revised proof uses: ‖L‖_op²/‖A‖_F ÷ ‖A‖_op (must stay bounded) and ‖L′L‖_F/(‖A‖_op‖A‖_F) (likewise).
include(joinpath(@__DIR__, "tc_lib.jl"))
using LinearAlgebra, Random, Printf

schatten(M, p) = norm(svdvals(M), p)

function report(label, A)
    L = tril(A, -1)
    s = svdvals(L); sA = svdvals(Symmetric(A))
    opL, opA = s[1], sA[1]
    @printf("%-28s n %5d | ‖L‖op/‖A‖op %.3f | ‖L‖₄/‖A‖₄ %.3f | ‖L‖op²/(‖A‖op‖A‖F) %.4f | ‖L′L‖F/(‖A‖op‖A‖F) %.3f | ‖A‖op/‖A‖F %.4f\n",
            label, size(A, 1), opL / opA, norm(s, 4) / norm(sA, 4), opL^2 / (opA * norm(sA)),
            norm(L' * L) / (opA * norm(sA)), opA / norm(sA))
end

function hilbert_interleaved(m)
    A = zeros(2m, 2m)
    for i in 1:m, j in 1:m
        A[2i, 2j-1] = A[2j-1, 2i] = 1 / (i - j + 0.5)
    end
    A
end

println("Power check: the interleaved Hilbert matrix (‖A‖op < π for every n)")
for m in (125, 250, 500, 1000, 2000)
    report("hilbert", hilbert_interleaved(m))
end

println("\nThe moment matrices (kNN W, Chebyshev surface basis, K ≈ 2√n)")
#   two moment matrices: tc_clt.jl's Jin & Lee P = W′ − Δ with C = PM, and tc_lib.jl's exact diagonal
#   A = MW + MDM.
for n in (400, 900, 1600, 2500)
    rng = MersenneTwister(11)
    P = rand(rng, n, 2); W = knnW(P, 6)
    dsurf = max(2, round(Int, (sqrt(8 * 2sqrt(n) + 1) - 3) / 2))   # (d+1)(d+2)/2 ≈ 2√n
    B = surfbasis(P[:, 1], P[:, 2], dsurf)
    D = tc_design(B, W)
    δ = diag(D.Wd' * D.M) ./ D.Mii
    C = (D.Wd' .- Diagonal(δ)) * D.M
    report("JL Cs, K = $(D.K)", (C + C') / 2)
    report("exact-diag As, K = $(D.K)", D.As)
end
