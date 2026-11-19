# A CLT for the zero-diagonal closed-form moment when the basis grows

**Status: written 2026-11-19, and not checked by anyone else.** Each step
below says whether it is proved here, sketched, or assumed. "Proved" means
proved from the stated results of the cited works. Revised the same day after
reading de Jong (1987): Theorem 1 is now proved, not sketched, and step 2 has
a first analysis. Revised again after reading Bhansali, Giraitis & Kokoszka
(2007), Kelejian & Prucha (2001) and Hall & Heyde (1980): Theorem 1 no longer
needs a log factor, and it cites the martingale CLT as Hall & Heyde state it.
Revised a third time the same day: Theorem 1 needs only `2 + δ` moments,
and the Corollary (the estimator's limit and the choice of root) and
Proposition 1 (the two-step REsd, formerly Remark 2) are proved. Later the
same day: Proposition 2 (the robust `V̂` is consistent when K/n → 0) is
proved, and the leave-out alternatives are measured. Then Proposition 3
(approximation error) replaces the sketched Remark 1, which had missed a
first-order term, and Proposition 4 (a penalised smoother) replaces
Remark 3, and Proposition 5 gives a primitive condition for A5.
The numerical checks are in `tc_clt.jl`, `tc_schatten.jl`, `tc_clt_moments.jl`
and their output. ROADMAP #13,
research item, theory step 1.

## Setting

- `y = ρWy + f + ε`, with `n` units, `W` known, and `ε` independent.
- `B` is an `n × K` basis, `P_B` the projection on its span, `M = I − P_B`,
  and `hᵢᵢ = (P_B)ᵢᵢ` the leverages. `K` may grow with `n`.
- The moment is Jin & Lee's (2012): `g(ρ) = y₀(ρ)′ C y₀(ρ)`, with
  `y₀(ρ) = y − ρWy`, `C = PM` and `diag(C) = 0`. For the first step,
  `P = W′ − Δ` with `Δ = Diag(W′M)Diag(M)⁻¹`. The two-step REsd replaces `W′`
  by `G(ρ̃)′`; see Proposition 1.
- The root `ρ̂` solves `g(ρ̂) = 0`, a quadratic in `ρ`.
- Write `Cs = (C + C′)/2`, `Σ = Diag(σ₁², …, σₙ²)`, and `L` for the strictly
  lower-triangular part of `Cs`.

The design (`W`, `B`, `f`) is fixed. Every statement is conditional on it.

## Assumptions

- **A1.** The `εᵢ` are independent, with mean 0, variances
  `σᵢ² ∈ [σ̲², σ̄²]` where `σ̲ > 0`, and `μ̄ = supᵢ E|εᵢ|^{2+δ} < ∞` for
  some `δ > 0`. They may form a triangular array. Taking `δ ≤ 2` loses
  nothing, so write `p = 1 + δ/2 ∈ (1, 2]` and `q = 2 + δ ∈ (2, 4]`. (Until
  2026-11-19 this asked for `supᵢ E εᵢ⁴ < ∞`.)
- **A2.** `diag(W) = 0`, and `W` has row and column sums bounded in absolute
  value. So `‖W‖_op ≤ c_W`, and each column has `‖Weᵢ‖ ≤ c_W`.
  **This is the only bounded-row-sum condition. It is on `W`, not on `C` or
  `M`.**
- **A3.** The largest leverage stays below 1: `maxᵢ hᵢᵢ ≤ h̄ < 1`.
- **A4.** At first, `f ∈ span(B)`. Proposition 3 drops this.
- **A5 (non-degeneracy).** `liminf ‖Cs‖_F²/n > 0`. Proposition 5 derives
  it from `W` when K/n is small; beyond that it is assumed. Measured values
  are 0.05–0.12 in `tc_clt.jl` and 0.04–0.15 in `tc_a5.jl`.
- **A6 (Lindeberg for the linear part).** `maxᵢ bᵢ² / ‖b‖² → 0`, where
  `b = MP′f`. It holds if `maxᵢ hᵢᵢ → 0` (Lemma 2b), but the measurements
  show that this is not necessary.

For the estimator (the Corollary) only:
- **A7 (stability).** `ρ₀` is such that `supₙ ‖(I − ρ₀W)⁻¹‖_op < ∞`, so
  `G = W(I − ρ₀W)⁻¹` has `‖G‖_op ≤ c_G`. This is the usual SAR condition.
- **A8.** `‖f‖²/n` is bounded.
- **A9 (identification).** `liminf J/n > 0`, where
  `J = −E g′(ρ₀) = 2[f′G′Cs f + tr(G′CsΣ)]`. The sign is part of the
  assumption: if instead `limsup J/n < 0`, everything below holds with the
  sign rule reversed. Assumed, not derived. In `tc_roots.jl`, `J/n` is
  0.20–0.28.

For the variance estimate (Proposition 2) only:
- **A10.** `Σᵢ bᵢ²hᵢᵢ = o(n)`: the
  signal in the linear part does not sit on high-leverage units. It holds
  if `maxᵢ hᵢᵢ → 0` (then it is `o(‖b‖²)`), or if `maxᵢ |bᵢ|` is bounded and
  `K/n → 0` (then it is `O(K)`).

## Lemma 1: the moment at the true ρ (proved)

At `ρ₀`, `y₀ = f + ε`. Since `Mf = 0`, `Cf = PMf = 0`, so
`g(ρ₀) = b′ε + ε′Cε`, with `b = C′f = MP′f`. Then:
- `E g(ρ₀) = tr(CΣ) = Σᵢ Cᵢᵢσᵢ² = 0`, under any heteroskedasticity, because
  `diag(C) = 0`;
- `V := Var g(ρ₀) = 2σ′(Cs∘Cs)σ + Σᵢ bᵢ²σᵢ²`, where `σ` is the vector of
  variances. The zero diagonal removes every third- and fourth-moment term,
  including the covariance between the linear and quadratic parts, because
  `E εᵢεⱼεₖ ≠ 0` only when `i = j = k`.

`tc_jinlee.jl` checks this exact variance against simulation.

## Lemma 2: the moment matrix is bounded although M is dense (proved)

(a) `‖M‖_op = 1`, so `‖C‖_op ≤ ‖P‖_op ≤ c_W + maxᵢ|Δᵢᵢ|`, and
`‖Cs‖_op ≤ ‖C‖_op`.

(b) For any `n × n` matrix `A`, `|(A′M)ᵢᵢ| = |(Aeᵢ)′(Meᵢ)| ≤ ‖A‖_op ‖Meᵢ‖`,
and `‖Meᵢ‖² = Mᵢᵢ = 1 − hᵢᵢ`. So `Δ = Diag(A′M)Diag(M)⁻¹` satisfies

    |Δᵢᵢ| ≤ ‖A‖_op / √(1 − hᵢᵢ),

which is bounded under A3. This covers REsd's `A = G(ρ)`, whose diagonal is
not zero. When `diag(A) = 0`, as for `A = W`, there is a finer bound:
`(A′M)ᵢᵢ = −(A′P_B)ᵢᵢ`, and Cauchy–Schwarz gives

    |Δᵢᵢ| ≤ ‖Aeᵢ‖ √hᵢᵢ / (1 − hᵢᵢ),

which tends to 0 when `maxᵢ hᵢᵢ → 0`.

(c) With A2 and A3, `‖C‖_op = O(1)`, and with A5,
`‖Cs‖_op / ‖Cs‖_F = O(n^{−1/2})`.

The point of the lemma: Kelejian & Prucha's (2001) CLT asks for bounded row
and column sums of the moment matrix. A dense `M` has row sums of order up to
`√(n hᵢᵢ)`, which breaks that. But the CLT argument below only needs norm
ratios, and the operator norm survives the projection. In `tc_clt.jl`,
`‖Cs‖_op` stays at 0.83–1.08 and the bound in (b) holds in every design.

## Lemma 3: the triangular part (proved)

Let `L` be the strictly lower-triangular part of a symmetric `A`. Then
`‖L‖_op ≤ ⌈log₂ n⌉ ‖A‖_op`.

*Proof.* Pad the index set to `2^m` with `m = ⌈log₂ n⌉`. At level
`k = 1, …, m`, cut the indices into consecutive blocks of size `2^{m−k+1}`,
and each block into two halves. Let `L_k` hold the entries of `A` whose row is
in the second half and whose column is in the first half of the same block.
Each pair `i > j` belongs to exactly one level: the finest one at which `i`
and `j` share a block but not a half. So `L = Σ_k L_k`. Within one level the
blocks have disjoint rows and disjoint columns, so `‖L_k‖_op` is the largest
operator norm of its sub-blocks, and no sub-block of `A` has a norm above
`‖A‖_op`. ∎

The log factor is known to be the right order for triangular truncation
(Kwapień & Pełczyński 1970; not read), and `tc_schatten.jl` shows it: for an
interleaved Hilbert matrix with `‖A‖_op < π`, `‖L‖_op/‖A‖_op` grows from 1.90
to 2.77 as n goes from 250 to 4000. In the moment's designs the ratio is only
0.67–0.69.

**Theorem 1 no longer uses this lemma.** In the Schatten-4 norm
`‖X‖₄ = (Σ sᵢ⁴)^{1/4}` (`sᵢ` the singular values), triangular truncation is
bounded with a constant free of n: `‖L‖₄ ≤ C₄‖A‖₄` (Macaev 1961, as used by
Bhansali, Giraitis & Kokoszka 2007 in their Lemma 2.1; Macaev not read). With
`‖A‖₄⁴ = ‖A²‖_F² ≤ ‖A‖_op²‖A‖_F²` this gives

    ‖L′L‖_F = ‖L‖₄² ≤ C₄²‖A‖_op‖A‖_F,     ‖L‖_op² ≤ ‖L‖₄² ≤ C₄²‖A‖_op‖A‖_F.

In `tc_schatten.jl`, `‖L‖₄/‖A‖₄` stays at 0.82–0.84 for the Hilbert matrix
while its operator-norm ratio grows, and at 0.70–0.71 for the moment's
matrices, n = 400 to 2500. The lemma is kept because its bound is elementary
and gives the operator norm at the full rate, which the Schatten bound does
not.

## Theorem 1: CLT for the moment (proved, given Hall & Heyde's Corollary 3.1, Macaev's theorem and two moment inequalities)

Assume A1, so the errors need only `2 + δ` moments. Write `σ̲, σ̄` for the
variance bounds. If

- (a) `‖Cs‖_op / ‖Cs‖_F → 0`, and
- (c′) `maxᵢ bᵢ² / V → 0`,

then `g(ρ₀)/√V → N(0, 1)`. Under A2–A5, (a) holds at rate `n^{−1/2}` by
Lemma 2. (The first version of this theorem needed
`(log n)²‖Cs‖_op²/‖Cs‖_F² → 0`, from Lemma 3, and fourth moments.)
(c′) is implied by A6, and it is weaker than A6 when the quadratic
part dominates `V`.

*Proof.* Write `c = Cs`, which is symmetric with zero diagonal, and
`uᵢ = Σ_{j<i} cᵢⱼεⱼ`, so that `u = Lε`. Then `ε′Cε = 2Σᵢ εᵢuᵢ`, and
`g = Σᵢ ξᵢ` with `ξᵢ = εᵢ(bᵢ + 2uᵢ)`. Since `uᵢ` depends only on
`ε₁, …, εᵢ₋₁`, the `ξᵢ` are martingale differences for
`Fᵢ = σ(ε₁, …, εᵢ)`. Also

    V = Σᵢ σᵢ²bᵢ² + 2Σ_{i≠j} σᵢ²σⱼ²cᵢⱼ²  ≥  max(σ̲²‖b‖², 2σ̲⁴‖Cs‖_F²).

Hall & Heyde's (1980) Corollary 3.1 (p. 58, read from the page image) needs
the two conditions below, with limit `η² = 1`. Its nesting condition (3.21)
is not needed: by the Remarks on p. 59 it can be replaced by `η²` being
measurable, which a constant trivially is (the stability of the limit is
then lost, and it is not needed here). So the errors may form a triangular
array.

1. **Conditional variance.** `S = Σᵢ E(ξᵢ²|Fᵢ₋₁) = Σᵢ σᵢ²(bᵢ + 2uᵢ)²`, with
   `E S = V`, and `S − V = 4A + 4(B − E B)`, where:
   - `A = Σᵢσᵢ²bᵢuᵢ = (L′Σb)′ε`. So
     `Var A ≤ σ̄²‖L′Σb‖² ≤ σ̄⁶‖L‖_op²‖b‖²`. By the Schatten-4 bound after
     Lemma 3, `‖L‖_op² ≤ C₄²‖Cs‖_op‖Cs‖_F`. Since
     `V² ≥ 8σ̲⁶‖b‖²‖Cs‖_F²` (from `(x + y)² ≥ 4xy`),
     `Var A/V² ≤ c‖Cs‖_op/‖Cs‖_F`.
   - `B = u′Σu = ε′Dε` with `D = L′ΣL`, which is symmetric and positive
     semidefinite. Split it as `B − E B = B₁ + B₂`, with
     `B₁ = Σ_{j≠k} Dⱼₖεⱼεₖ` and `B₂ = Σⱼ Dⱼⱼ(εⱼ² − σⱼ²)`.
     - **`B₁` needs only second moments.** Its terms are uncorrelated except
       in the pairs `{j, k} = {l, m}`, so `E B₁² = 2Σ_{j≠k} Dⱼₖ²σⱼ²σₖ² ≤
       2σ̄⁴‖D‖_F²`. Since `0 ≼ L′ΣL ≼ σ̄²L′L`,
       `‖D‖_F ≤ σ̄²‖L′L‖_F ≤ σ̄²C₄²‖Cs‖_op‖Cs‖_F`. With `V ≥ 2σ̲⁴‖Cs‖_F²`,
       `E B₁²/V² ≤ c‖Cs‖_op²/‖Cs‖_F²`.
     - **`B₂` is a sum of independent, mean-zero terms with `p` moments.**
       By von Bahr & Esseen's (1965) inequality, for `1 ≤ p ≤ 2`,
       `E|Σⱼ Xⱼ|^p ≤ 2Σⱼ E|Xⱼ|^p`. Here
       `E|εⱼ² − σⱼ²|^p ≤ 2^{p−1}(E|εⱼ|^{2p} + σⱼ^{2p}) ≤ 2^p μ̄`, by
       Jensen's inequality for the second term. So
       `E|B₂|^p ≤ 2^{p+1}μ̄ Σⱼ Dⱼⱼ^p`. Each `Dⱼⱼ = Σᵢσᵢ²Lᵢⱼ² ≤
       σ̄²‖Cs eⱼ‖² ≤ σ̄²‖Cs‖_op²`, and `Σⱼ Dⱼⱼ = tr D ≤ σ̄²‖L‖_F² =
       σ̄²‖Cs‖_F²/2`. So `Σⱼ Dⱼⱼ^p ≤ (maxⱼ Dⱼⱼ)^{p−1} Σⱼ Dⱼⱼ ≤
       σ̄^{2p}‖Cs‖_op^{2(p−1)}‖Cs‖_F²/2`, and with `V^p ≥
       (2σ̲⁴)^p‖Cs‖_F^{2p}`,
       `E|B₂|^p/V^p ≤ c (‖Cs‖_op²/‖Cs‖_F²)^{p−1} = c (‖Cs‖_op/‖Cs‖_F)^δ`.

   The constants depend only on `σ̲`, `σ̄`, `μ̄`, `δ` and `C₄`. Every bound
   tends to 0 by (a), so `A/V`, `B₁/V` and `B₂/V` tend to 0 in probability
   (Chebyshev's and Markov's inequalities), and `S/V → 1` in probability.
   The earlier proof used fourth moments here, because `Var(ε′Dε)` contains
   `Σⱼ Dⱼⱼ² Var(εⱼ²)`, and in the Lyapunov step below, which was of order 4.
2. **Lyapunov of order `q = 2 + δ`, which implies the conditional
   Lindeberg condition.** For `Xᵢ = ξᵢ/√V` and any `η > 0`,
   `E Σᵢ E[Xᵢ² 1{|Xᵢ| > η} | Fᵢ₋₁] = Σᵢ E[Xᵢ² 1{|Xᵢ| > η}] ≤
   η^{−δ} Σᵢ E|ξᵢ|^q / V^{q/2}`, so if the right side tends to 0 the
   conditional sum tends to 0 in `L¹`, hence in probability.

   By independence of `εᵢ` and `uᵢ`,
   `E|ξᵢ|^q = E|εᵢ|^q · E|bᵢ + 2uᵢ|^q ≤ μ̄ 2^{q−1}(|bᵢ|^q + 2^q E|uᵢ|^q)`.
   By Rosenthal's (1970) inequality for the independent, mean-zero terms of
   `uᵢ = Σ_{j<i} cᵢⱼεⱼ`, with `rᵢ² = Σ_{j<i} cᵢⱼ²`,

       E|uᵢ|^q ≤ C_q [Σⱼ |cᵢⱼ|^q E|εⱼ|^q + (Σⱼ cᵢⱼ²σⱼ²)^{q/2}] ≤ C_q (μ̄ + σ̄^q) rᵢ^q,

   because `Σⱼ|cᵢⱼ|^q ≤ (Σⱼcᵢⱼ²)^{q/2}` for `q ≥ 2`. Then:
   - `Σᵢ|bᵢ|^q ≤ (maxᵢbᵢ²)^{δ/2}‖b‖² ≤ (maxᵢbᵢ²)^{δ/2} V/σ̲²`, so this part
     of the ratio is at most `c (maxᵢbᵢ²/V)^{δ/2}`;
   - `Σᵢ rᵢ^q ≤ (maxᵢ rᵢ²)^{δ/2} Σᵢ rᵢ² ≤ ‖Cs‖_op^δ ‖Cs‖_F²/2`, because a
     row norm is at most the operator norm, and with
     `V^{q/2} ≥ (2σ̲⁴)^{q/2}‖Cs‖_F^q` this part is at most
     `c (‖Cs‖_op/‖Cs‖_F)^δ`.

   So `Σᵢ E|ξᵢ|^q/V^{q/2} ≤ c₂[(maxᵢbᵢ²/V)^{δ/2} + (‖Cs‖_op/‖Cs‖_F)^δ]`,
   which tends to 0 by (c′) and (a). ∎

**The cost of fewer moments is the rate, not the conditions.** Every bound
above is a power of the same two ratios as before, `‖Cs‖_op/‖Cs‖_F` and
`maxᵢbᵢ²/V`. The bounds that used fourth moments (`B₂` and Lyapunov) now
have exponents `δ` and `δ/2` where they had 2 and 1. So with
`δ` small the normal approximation should improve slowly in n, and
`tc_clt_moments.jl` shows that (see the numerical check below).

**Against Kelejian & Prucha (2001)** (Theorem 1, Assumptions 1–3 and the
proof in their appendix, read 2026-11-19 from the page images). Their CLT
covers the same linear-quadratic form with independent, heteroskedastic
errors, and they too use a martingale CLT (their Theorem A.1, from Gänssler &
Stute 1977, a special case of Hall & Heyde's Corollary 3.1). Their
conditions differ from ours in three places:
- they need **bounded absolute row sums** of `A`, `sup Σᵢ|aᵢⱼ| < ∞`, which a
  dense `M` breaks; Theorem 1 replaces this with (a);
- they need `n⁻¹σ²_Q ≥ c > 0`, the rate `n`; Theorem 1 needs no rate;
- they need only `2 + η` moments when `aᵢᵢ = 0` (their Assumption 3(a)), and
  `4 + η` otherwise; Theorem 1 now matches them, with `2 + δ` (the
  zero diagonal is what allows it: a nonzero `Cᵢᵢ` would bring
  `Σᵢ Cᵢᵢ(εᵢ² − σᵢ²)` into `g`, whose variance needs fourth moments). No
  truncation is needed, because the zero diagonal leaves the `εᵢ²` only
  inside the conditional variance, where a law of large numbers with `p`
  moments is enough.

**Against Lee (2004)** (as restated, with its proof, in Appendix 1A of
chapter 1 of Lee 2023, held in `ExtRef/`). It extends Kelejian & Prucha to entries of
uniform order `1/hₙ` and a variance of order `n/hₙ`, with
`h_n^{1+2/δ}/n → 0`. It still assumes bounded row and column sums, so it
does not cover a dense `M` either.

**What Bhansali, Giraitis & Kokoszka (2007) add** (read 2026-11-19, from the
page images). Their Theorem 2.1 is for a quadratic form alone, in i.i.d.
variables, with a nonzero diagonal allowed. Under finite fourth moments it
needs `Δₙ = (‖Ã′Ã‖_F + maxₜ Σₛ aₜₛ²)/‖A‖_F² → 0`, with `Ã` the strictly
lower-triangular part, which their Lemma 2.1 shows is implied by
`‖A‖_op/‖A‖_F → 0`. With the diagonal of order `o(‖A‖_F²)` it needs only
`2 + δ` moments. It does not cover a linear part or heteroskedastic errors,
so it does not contain Theorem 1. Its Lemma 2.1 is what removes the log
factor here.

**What de Jong (1987) adds** (read 2026-11-19, from the page images).
His Theorem 5.2 treats the quadratic part alone: `Σ_{i≠j} aᵢⱼXᵢXⱼ`, with the
`Xᵢ` independent and not necessarily identically distributed, `EXᵢ = 0`,
`EXᵢ² = 1` and `aᵢᵢ = 0`. It gives normality under three conditions:
- (1) `K(n)⁴σ(n)⁻² maxᵢ Σⱼaᵢⱼ² → 0` for some sequence `K(n)`;
- (2) `maxᵢ EXᵢ²1{|Xᵢ| > K(n)} → 0`;
- (3) `σ(n)⁻² maxᵢ μᵢ² → 0`, where the `μᵢ` are the eigenvalues of `(aᵢⱼ)`.

With `Xᵢ = εᵢ/σᵢ` and `aᵢⱼ = σᵢσⱼcᵢⱼ`, condition (3) is our condition (a),
and (1)–(2) hold with uniformly bounded `2 + δ`
moments. So **the quadratic part alone needs no fourth
moments.** The martingale proof above is needed for the linear and quadratic
parts jointly, which de Jong's theorem does not cover.

## Corollary: the estimator (proved, given Theorem 1)

Assume A1–A9, and choose `ρ̂` by the **sign rule**: of the two roots of
`g`, take the one at which `g′(ρ̂) < 0`. Then, with probability tending to
1, `g` has two distinct real roots and the sign rule picks exactly one, and

    J (ρ̂ − ρ₀) / √V → N(0, 1),

so `ρ̂ − ρ₀ = O_p(n^{−1/2})`. Moreover `|g′(ρ̂)|/J → 1` in probability, so
the feasible standard error `√V̂ / |g′(ρ̂)|` is valid whenever `V̂/V → 1`.
Proposition 2 proves that for the coded `V̂` when K/n → 0 (under A10 and
four moments); for K/n → c > 0 it is open.

*Proof.* Write `η = f + ε = y₀(ρ₀)` and `t = ρ − ρ₀`. Since `y = (I − ρ₀W)⁻¹η`,
`Wy = Gη`, and `y₀(ρ) = η − tGη`. So `g` is exactly a quadratic in `t`:

    g(ρ₀ + t) = g₀ + g₁t + a t²,   g₀ = g(ρ₀),   g₁ = g′(ρ₀) = −2η′G′Csη,   a = η′G′CsGη.

(In the code, `g(ρ) = c − bρ + aρ²`, so `g′(ρ) = −(b − 2ρa)`.)

1. **Orders.** Under A2, A3 and A7, `‖Cs‖_op` and `‖G‖_op` are bounded
   (Lemma 2), and `‖Cs‖_F² ≤ n‖Cs‖_op²`.
   - `g₀ = O_p(√n)`: Theorem 1 gives `g₀ = O_p(√V)`, and
     `V ≤ 2σ̄⁴‖Cs‖_F² + σ̄²‖C′f‖² = O(n)` by A8.
   - `a = O_p(n)`: `|a| ≤ ‖Cs‖_op‖G‖_op²‖η‖²`, and `E‖η‖² = ‖f‖² + Σσᵢ² = O(n)`,
     so Markov's inequality applies. Only second moments are used.
   - `g₁ = −J + o_p(n)`. Since `E ε = 0`, `E g₁ = −2[f′G′Csf + tr(G′CsΣ)] = −J`.
     With `A = (G′Cs + CsG)/2`, which has `‖A‖_op ≤ c_G‖Cs‖_op` and
     `‖A‖_F ≤ c_G‖Cs‖_F = O(√n)`,
     `g₁ + J = −2[2f′Aε + Σ_{i≠j}Aᵢⱼεᵢεⱼ + Σᵢ Aᵢᵢ(εᵢ² − σᵢ²)]`. The linear
     term has variance at most `4σ̄²‖A‖_op²‖f‖² = O(n)`, and the off-diagonal
     term has second moment at most `2σ̄⁴‖A‖_F² = O(n)`: both are `O_p(√n)`.
     The diagonal term is a sum of independent, mean-zero terms with `p`
     moments, so by von Bahr & Esseen
     `E|Σᵢ Aᵢᵢ(εᵢ² − σᵢ²)|^p ≤ 2^{p+1}μ̄ Σᵢ|Aᵢᵢ|^p ≤ 2^{p+1}μ̄ n‖A‖_op^p`. It is
     therefore `O_p(n^{1/p})`, which is `o_p(n)` because `p > 1`. (With fourth
     moments it would be `O_p(√n)`; `o_p(n)` is all that is needed.)
2. **The discriminant and the two roots.** `D = g₁² − 4ag₀ = J²(1 + o_p(1)) −
   O_p(n)·O_p(√n) = J²(1 + o_p(1))`, because `J² ≥ cn²` by A9. So `D > 0`
   with probability → 1. At the two roots of a quadratic, `g′` equals `+√D`
   and `−√D`, so the sign rule picks exactly one root, `t̂`, at which
   `g′(ρ₀ + t̂) = g₁ + 2at̂ = −√D = −J(1 + o_p(1))`.
3. **The chosen root is the consistent one.** From `g(ρ₀ + t̂) = 0`,
   `t̂ (g₁ + a t̂) = −g₀`. Write `g₁ + at̂ = [(g₁ + 2at̂) + g₁]/2`; both terms
   are `−J(1 + o_p(1))`, so `g₁ + at̂ = −J(1 + o_p(1))` and

       t̂ = g₀ / J · (1 + o_p(1)) = O_p(√n/n) = O_p(n^{−1/2}).

   This needs no prior consistency argument: the identity holds at the
   chosen root whatever its size, because the derivative condition fixes
   `g₁ + at̂` directly.
4. **The limit.** `J t̂/√V = (g₀/√V)(1 + o_p(1)) → N(0, 1)` by Theorem 1 and
   Slutsky's lemma. And `|g′(ρ̂)| = √D = J(1 + o_p(1))`. ∎

**The other root** is `t̃ = g₀/(a t̂)` (the product of the roots is `g₀/a`), so
`|t̃| = (J/|a|)(1 + o_p(1))`, of the order of a constant: it does not
approach `ρ₀`. If `a` is near 0 it is far away; it is never a competitor in
the limit.

**The code's rule is not the sign rule.** `cfe_root` keeps the root in
`(−1, 1)` nearest the OLS ratio `e₁′e₀/e₁′e₁`, which is not a consistent
estimator of `ρ₀`, so the proof does not cover it. `tc_roots.jl` measures
the difference in the coverage design (`ρ₀ = 0.4`, normal and `t₃` errors,
n = 400 to 2500, R = 1000 each): the discriminant was positive in all 8000
replications, the two rules disagreed once (`t₃`, n = 400), and the other
root was at least 1.37 from `ρ₀` (median about 2). The sign rule needs the
sign of `J`, which A9 assumes. It cannot be read off `g′` at either root,
since those are `±√D` by construction; it needs a consistent preliminary
estimate, such as a 2SLS fit, at which to evaluate `−g′`. A package
implementation should do that, or report both roots.

**The rate for K.** Nothing in Theorem 1 needs `K/n → 0`. A3 and A5 allow
`K/n → c ∈ [0, 1)`. This is weaker than the published rate conditions for
spline GMM, such as `k_n²/n → 0` (Lu, Tian & Yang 2024). It is possible
because the zero diagonal removes the trace bias exactly, at any `K`. Rate
conditions are otherwise needed to make that bias small. A large `K` still
costs efficiency, because `‖Cs‖_F²/n` falls as `K/n` rises (from 0.11 at
K/n = 0.04 to 0.05 at K/n = 0.15 in `tc_clt.jl`), but it does not cost
validity.

**Proposition 3 (approximation error; proved, given Theorem 1 and the
Corollary; replaces Remark 1).** Drop A4, so `f = P_Bf + r` with
`r = Mf ≠ 0`, and read A6 for `b̃ = (C + C′)f`. Write `β = f′Cf`, which is
`E g(ρ₀)` exactly, since `tr(CΣ) = 0`. Then
`(g(ρ₀) − β)/√V → N(0, 1)`, with `V` computed from `b̃`; and if
`β/√V → c`, then `J(ρ̂ − ρ₀)/√V → N(c, 1)`, so the nominal 95% interval
covers with probability `P(|Z + c| ≤ 1.96)`. The interval is valid if and
only if `β = o(√n)`.

For the coded moment, `C = A = M(W + DM)`, and `β` splits into

    β = r′(W + D)r + r′r_W,     r_W = MWP_Bf,

where `r_W` is the part of the lagged fitted surface that `B` cannot
represent. (For Jin & Lee's `C = PM`, the same with `P` in place of
`W + D` and `r_W = MP′P_Bf`.)

*Proof.* `g(ρ₀) = η′Cη` with `η = f + ε` is `β + b̃′ε + ε′Cε`. Theorem 1
covers `b̃′ε + ε′Cε` for any `b̃` satisfying A6, and its proof does not use
`Mf = 0`. In the Corollary, `Mf = 0` is not used either: `J` is defined as
`−E g′(ρ₀)` for the actual `f`, and `g₀ = O_p(√n)` needs only `β = O(√n)`.
So `t̂ = (g₀/J)(1 + o_p(1))` as before, and the limit follows. For the split,
`f′Af = (Mf)′(W + DM)f = r′W(P_Bf + r) + r′Dr`, and `r′WP_Bf = r′r_W`
because `r = Mr`. ∎

**The sketched remark kept only the first term.** That term is at most
`‖W + D‖_op‖r‖² ≤ c n‖r‖∞²`, so it is `o(√n)` when `√n‖r‖∞² → 0`; with
`‖r‖∞ = O(K^{−s/d})` for smoothness `s` in `d` dimensions, that is
`K ≫ n^{d/(4s)}`. The second term is at most `‖r‖‖r_W‖`, and `r_W` need not
be small: a covariate `x` in `B` that is rough in space has a lag `Wx` that
the basis cannot represent. In the worst case this needs
`‖r‖∞ = o(n^{−1/2})`, that is `K ≫ n^{d/(2s)}`, twice the exponent. When
`r` is smooth and `r_W` is rough, the two are nearly orthogonal and
`r′r_W` is of order `‖r‖∞ √n` instead, which needs only `‖r‖∞ → 0`. So the
binding condition is `β = o(√n)` itself, and it is a **lower** bound on K.
The robust `V̂` should stay consistent (sketched; Proposition 2's steps
are not redone here): `û` gains `r`, so `sᵢ` gains
`(rᵢ² + 2rᵢ(Mε)ᵢ)/Mᵢᵢ`, which adds `O(‖r‖² + ‖r‖√n)` to `V̂`, and that is
`o(n)` when `‖r‖∞ → 0`.

**Numerical check (`tc_approx.jl`).** n = 900, kNN `W`, a tensor hat basis
plus a covariate `x`, `f = a·sin(2πu)cos(πv) + x` with `a` = 1 and 4,
smooth variances, `ρ₀ = 0.4`, R = 1000. `β`, its two terms, `V`, `c` and
`J` are exact; the bias and coverage of `ρ̂` (HC2 `V̂`) are simulated.

| a | K | ‖r‖∞ | ‖r_W‖∞ | r′(W+D)r | r′r_W | c = β/√V | bias: predicted β/J | simulated | coverage: predicted | simulated |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 10 | 1.13 | 1.21 | 87.4 | −1.09 | 2.46 | 0.107 | 0.116 | 0.309 | 0.211 |
| 1 | 17 | 0.29 | 1.20 | 6.40 | −0.14 | 0.22 | 0.0122 | 0.0122 | 0.944 | 0.933 |
| 1 | 26 | 0.19 | 1.24 | 2.08 | −1.41 | 0.03 | 0.0014 | 0.0019 | 0.950 | 0.947 |
| 1 | 50 | 0.075 | 1.34 | 0.27 | −0.01 | 0.01 | 0.0007 | 0.0022 | 0.950 | 0.946 |
| 4 | 17 | 1.15 | 1.30 | 102.5 | 1.17 | 2.87 | 0.125 | 0.139 | 0.181 | 0.091 |
| 4 | 26 | 0.78 | 1.26 | 33.2 | −2.27 | 1.10 | 0.0527 | 0.0555 | 0.804 | 0.780 |
| 4 | 50 | 0.30 | 1.22 | 4.30 | 1.29 | 0.24 | 0.0137 | 0.0151 | 0.943 | 0.941 |
| 4 | 101 | 0.17 | 1.11 | 0.31 | 0.72 | 0.06 | 0.0040 | 0.0042 | 0.950 | 0.960 |
| 4 | 170 | 0.089 | 1.09 | 0.02 | 0.22 | 0.02 | 0.0015 | 0.0003 | 0.950 | 0.947 |

(All rows are in `tc_approx_output.txt`, including a = 1 at K = 101 and 170
and a = 4 at K = 10, where `c = 15.7`.)

- **`r_W` is not small**: `‖r_W‖∞` is 1.0–1.3 at every K, because of the
  covariate. The cross term `r′r_W` is the larger part of `β` at a = 4 and
  K ≥ 101, so the sketched remark's `r′(W+D)r` alone would understate the
  bias there by a factor of 3 to 12. It is still far below `‖r‖‖r_W‖`,
  because a smooth `r` and a rough `r_W` are nearly orthogonal.
- **The prediction holds while `c` is moderate.** For `c` ≤ 1.1 the
  simulated bias matches `β/J` (`|z|` ≤ 1.9) and the coverage matches
  `P(|Z + c| ≤ 1.96)` within 2.4 points. For `c` of 2.5 and more, the
  first-order expansion understates the bias (0.116 against 0.107) and
  overstates the coverage (0.211 against 0.309). That is outside the
  local result.
- **What the check cannot show.** Where the two forms of `β` differ
  (a = 4, K ≥ 50), the bias of `ρ̂` is 0.0003–0.015 and the two predictions
  differ by at most 0.005, against a Monte Carlo SE of 0.002–0.003, so the
  simulation cannot separate them (`z` at most 2.6). The
  split itself is exact and needs no simulation.
- **Under-smoothing is the price.** Coverage is within 2 points of 0.95
  once `c` ≤ 0.25,
  which here means K ≥ 17 at a = 1 and K ≥ 50 at a = 4: a stronger
  surface needs a larger K. Because the zero diagonal removes the trace
  bias at any K, raising K costs efficiency only (the sd rises from 0.054
  to 0.100 at a = 1 as K goes from 17 to 170), not validity.

**Proposition 1 (REsd, the two-step; proved, given the Corollary).** Write
`C(r) = (G(r)′ − Δ(r))M`, with `G(r) = W(I − rW)⁻¹` and
`Δ(r) = Diag(G(r)′M)Diag(M)⁻¹`, so `diag C(r) = 0` for every `r`. Jin &
Lee's REsd takes the first-step root `ρ̃` (with `C₁ = (W′ − Δ)M`) and then
the root `ρ̂` of `y₀(ρ)′C(ρ̃)y₀(ρ)`, by the sign rule. Let `ρ̂*` be the
infeasible root with `C(ρ₀)`. Assume A1–A9, with A5, A6 and A9 holding for both `C₁`
and `C(ρ₀)`, and Theorem 1's conditions (a) and (c′) for `C(ρ₀)`. Then
`ρ̂ − ρ̂* = o_p(n^{−1/2})`, and so `J*(ρ̂ − ρ₀)/√V* → N(0, 1)` with `J*` and
`V*` those of `C(ρ₀)`: the plug-in costs nothing in the limit.

*Proof.* Write `C_r` and `C_rr` for the derivatives in `r`, and `G = G(ρ₀)`.
1. **Uniform bounds near `ρ₀`.** With `c_G` from A7, for `|r − ρ₀| ≤ ε =
   1/(2c_G)` the Neumann series `(I − rW)⁻¹ = (I − ρ₀W)⁻¹ Σₖ ((r − ρ₀)G)ᵏ`
   gives `‖(I − rW)⁻¹‖_op ≤ 2‖(I − ρ₀W)⁻¹‖_op`, so `‖G(r)‖_op` is bounded
   on that interval. Since `∂G/∂r = G(r)²` and `∂²G/∂r² = 2G(r)³`, and `Δ` is
   linear in `G′` (`Δ[A] = Diag(AM)Diag(M)⁻¹` for `A = G′`, `(G²)′`,
   `(G³)′`), Lemma 2(b) gives `|Δ[A]ᵢᵢ| ≤ ‖A‖_op/√(1 − h̄)`. So
   `‖C(r)‖_op`, `‖C_r(r)‖_op` and `‖C_rr(r)‖_op` are bounded uniformly on the
   interval, which contains `ρ̃` with probability → 1 (the Corollary for
   `C₁`: `ρ̃ − ρ₀ = O_p(n^{−1/2})`).
2. **Orthogonality.** `Mf = 0` gives `C(r)f = 0`, and `diag C(r) = 0`, for
   every `r`. Differentiating, `C_r f = 0` and `diag C_r = 0`. So, with
   `η = f + ε`,

       η′C_r(ρ₀)η = (C_r′f)′ε + ε′C_r ε,   E η′C_r(ρ₀)η = f′C_r f + tr(C_rΣ) = 0,

   and both terms are `O_p(√n)` with second moments: the linear one has
   variance at most `σ̄²‖C_r‖_op²‖f‖² = O(n)`, and the quadratic one, whose
   matrix has a zero diagonal, has variance at most `2σ̄⁴‖C_r‖_F² = O(n)`.
   This is the Neyman orthogonality of the remark that this proposition
   replaces, now with its proof.
3. **The three coefficients.** In the Corollary's notation, with `C(ρ̃)` in
   place of a fixed `C`, the quadratic in `t = ρ − ρ₀` has coefficients
   `g₀(ρ̃) = η′C(ρ̃)η`, `g₁(ρ̃) = −2η′G′Cs(ρ̃)η` and `a(ρ̃) = η′G′Cs(ρ̃)Gη`.
   By Taylor's theorem in `r`, with `ρ̄` between `ρ̃` and `ρ₀`:
   - `g₀(ρ̃) − g₀(ρ₀) = (ρ̃ − ρ₀)η′C_r(ρ₀)η + ½(ρ̃ − ρ₀)²η′C_rr(ρ̄)η
     = O_p(n^{−1/2})O_p(√n) + O_p(n^{−1})O_p(n) = O_p(1)`, using step 2 for
     the first term and `|η′C_rr(ρ̄)η| ≤ sup‖C_rr‖_op‖η‖² = O_p(n)` for the
     second. Since `V* ≥ 2σ̲⁴‖Cs(ρ₀)‖_F² ≥ cn` by A5, this is `o_p(√V*)`.
   - `g₁(ρ̃) − g₁(ρ₀) = O_p(n^{−1/2})·O_p(n) = o_p(n)`, by the uniform bound
     on `‖C_r‖_op`; so `g₁(ρ̃) = −J* + o_p(n)`.
   - `a(ρ̃) = O_p(n)`, by the uniform bound on `‖C‖_op`.
4. **Conclusion.** These are exactly the orders the Corollary's steps 2–4
   use, with `g₀(ρ̃)/√V* = g₀(ρ₀)/√V* + o_p(1) → N(0, 1)` by Theorem 1. So
   the sign rule picks `t̂ = (g₀(ρ̃)/J*)(1 + o_p(1))`, and the infeasible
   root is `t̂* = (g₀(ρ₀)/J*)(1 + o_p(1))`. Their difference is
   `[g₀(ρ̃) − g₀(ρ₀)]/J* + o_p(√n/n) = O_p(1/n) + o_p(n^{−1/2})`. ∎

Only the mean-zero property in step 2 needs the zero diagonal and `Mf = 0`.
Without it, `(ρ̃ − ρ₀)E η′C_rη` would be of order `n^{−1/2}·n = √n`, the
same order as `g₀`, and the plug-in would change the limit.

**Numerical check (`tc_resd.jl`).** `tc_coverage_tails.jl`'s design with
normal errors, `ρ₀ = 0.4`, R = 500 per cell (Monte Carlo SE of a coverage
near 0.95: 0.010). `max|C(ρ₀)f|` and `max|diag C(ρ₀)|` are below `4e−15`,
which checks step 2's identities.

| n | K | sd(ρ̂ − ρ̂*)/sd(ρ̂*) | the same × √n | wrong plug-in `C(ρ₀ + 0.3)` | coverage, REsd | coverage, infeasible |
|---|---|---|---|---|---|---|
| 400 | 36 | 0.116 | 2.32 | 0.242 | 0.946 | 0.948 |
| 900 | 64 | 0.059 | 1.76 | 0.212 | 0.948 | 0.950 |
| 1600 | 81 | 0.036 | 1.44 | 0.203 | 0.950 | 0.952 |
| 2500 | 100 | 0.027 | 1.33 | 0.208 | 0.952 | 0.952 |

- **The plug-in effect vanishes, as the proposition says.** Relative to
  `sd(ρ̂*)`, the gap between REsd and the infeasible estimator falls by a
  factor of 4 over this range, at least as fast as `n^{−1/2}` (the scaled
  column still falls). The proof predicts `ρ̂ − ρ̂* = O_p(1/n)` plus terms of
  smaller order than `n^{−1/2}`.
- **Power check:** the fixed wrong plug-in `C(ρ₀ + 0.3)` stays at 0.20–0.24
  of `sd(ρ̂*)` at every n, so the check can tell a plug-in that matters
  from one that does not. That estimator is still nearly unbiased (bias
  −0.010 to −0.002), as any fixed `r` gives a valid moment.
- REsd's sd is 3–10% below the first step's (0.1301 against 0.1444 at
  n = 400; 0.0497 against 0.0513 at n = 2500), and its coverage with the
  robust variance and the code's root rule is 0.946–0.952.

**Proposition 4 (a penalised smoother; proved; replaces Remark 3).** Let `M = I − S` with
`S` symmetric, its eigenvalues in `[0, 1]` (a ridge- or difference-penalised
spline), `maxᵢ Sᵢᵢ ≤ s̄ < 1` in place of A3, and `M∘M` invertible. Use the
moment `g(ρ) = y₀′Ay₀` with `A = MW + MDM` and `(M∘M)d = −diag(MW)`, so
`diag A = 0` exactly; the identity `diag(MDM) = (M∘M)d` needs only `M`
symmetric. Then:
- **Lemma 2 holds with `Sᵢᵢ` in place of `hᵢᵢ`.** `‖M‖_op ≤ 1`. The
  eigenvalues of `M` lie in `[0, 1]`, so `M² ≼ M` and
  `‖Meᵢ‖² = (M²)ᵢᵢ ≤ Mᵢᵢ`; hence `|(A′M)ᵢᵢ|/Mᵢᵢ ≤ ‖A‖_op/√Mᵢᵢ`, the bound of
  (b). The finer bound uses `‖Seᵢ‖² = (S²)ᵢᵢ ≤ Sᵢᵢ` the same way.
- **Theorem 1 holds as stated**, since its proof uses only `‖Cs‖_op`,
  `‖Cs‖_F` and A5–A6, never idempotency.
- **The Corollary and Proposition 3 hold with `r = Mf`.** `Mf ≠ 0` even for
  a smooth `f` because the penalty shrinks: `r` is now shrinkage bias plus
  approximation error, and validity needs `β = f′Af = o(√n)`. Over-smoothing
  breaks it exactly as too small a K does.
- **The variance estimate needs two changes.** Under homoskedasticity
  `E ûᵢ² = σ²(M²)ᵢᵢ`, not `σ²Mᵢᵢ`, so the plug-in is `sᵢ = ûᵢ²/(M²)ᵢᵢ`.
  With `f̂ = Sy₀`, `â = (A + A′)Sy₀` and the noise term is `diag(HΣ̂H′)`
  with `H = (A + A′)S`, the general form of the projection's `G(Q′Σ̂Q)G′`.
  This `V̂` is consistent (part (c) below).

**The idempotent shortcut is wrong here.** The code forms the moment from
residual vectors as `(My₀)′(MWy₀ + D My₀)`, which is `y₀′(M²W + MDM)y₀`.
That equals `y₀′Ay₀` only when `M² = M`. For a smoother its diagonal is not
zero, and `E g(ρ₀)` picks up `tr((M²W + MDM)Σ) ≠ 0`. A TC-CFE with a
smoother must form `A` itself. (The package's smoothed `closed_form_rho` is
Smirnov's plain CFE, already marked unproven, so no shipped code is
affected.)

**Numerical check (`tc_smoother.jl`).** A tensor P-spline: a 15 × 15 hat
basis with a second-difference penalty, plus an unpenalised covariate; n =
900, kNN `W`, `f = a·sin(2πu)cos(πv) + x`, smooth variances, normal errors,
`ρ₀ = 0.4`, R = 1000. Coverage is of the 95% interval at the feasible root.

| a | λ | tr S | max Sᵢᵢ | KS of z₀ | c = β/√V | bias: predicted | simulated | coverage: predicted | oracle | general `V̂` | shortcut moment |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 0.001 | 222 | 0.93 | 0.021 | 0.003 | 0.0004 | 0.0027 | 0.950 | 0.955 | 0.951 | 0.953 |
| 1 | 0.1 | 131 | 0.75 | 0.027 | 0.014 | 0.0011 | 0.0031 | 0.950 | 0.954 | 0.948 | 0.879 |
| 1 | 1 | 62 | 0.43 | 0.030 | 0.096 | 0.0056 | 0.0073 | 0.949 | 0.947 | 0.949 | 0.903 |
| 1 | 10 | 27 | 0.19 | 0.034 | 0.659 | 0.0324 | 0.0338 | 0.899 | 0.919 | 0.900 | 0.946 |
| 1 | 100 | 12 | 0.08 | 0.033 | 1.949 | 0.0850 | 0.0904 | 0.505 | 0.553 | 0.454 | 0.725 |
| 4 | 0.1 | 131 | 0.75 | 0.021 | 0.232 | 0.0155 | 0.0175 | 0.944 | 0.950 | 0.954 | 0.863 |
| 4 | 1 | 62 | 0.43 | 0.029 | 1.594 | 0.0698 | 0.0753 | 0.643 | 0.679 | 0.663 | 0.908 |
| 4 | 10 | 27 | 0.19 | 0.029 | 8.88 | 0.202 | 0.250 | 0.000 | 0.000 | 0.000 | 0.592 |

(All 12 cells are in `tc_smoother_output.txt`.)

- **Theorem 1 holds for the smoother.** In all 12 cells, `z₀ =
  (g(ρ₀) − β)/√V` has mean within ±0.06, sd 0.98–1.01, and a KS distance
  0.016–0.034 from N(0, 1), below the 5% critical value 0.043. That includes
  λ = 0.001, where `max Sᵢᵢ` = 0.93 and `max|dᵢ|` reaches 44: `‖As‖_op` stays
  at 0.71, because a large `dᵢ` sits where `Mᵢᵢ` is small.
- **Shrinkage bias behaves as Proposition 3 says.** For `c` ≤ 0.66 the
  simulated bias matches `β/J` (`|z|` ≤ 1.0) and the general `V̂`'s coverage
  is within 1.0 point of `P(|Z + c| ≤ 1.96)`. For `c` ≥ 1.6 the first-order
  bias is too small (`z` from 3.8 up), as in `tc_approx.jl`.
- **Power checks.** The shortcut moment has `E g(ρ₀)` from −14 to +710
  instead of 0. Where the shrinkage bias is small (λ = 0.1 and 1 at a = 1,
  λ = 0.1 at a = 4) it covers 0.863–0.903, against 0.948–0.954 for the
  correct moment. Where the shrinkage bias is large, its coverage is the
  higher one (0.946, 0.908, 0.592) only because its trace bias happens to
  offset the shrinkage bias.
  The projection plug-in `ûᵢ²/Mᵢᵢ` differs from `ûᵢ²/(M²)ᵢᵢ` by at most
  1.5 points of coverage (0.939 against 0.954 at a = 4, λ = 0.1), so that
  correction matters less than the moment does.
- **Over-smoothing is what fails.** Coverage is nominal for λ ≤ 1 at
  a = 1 and λ ≤ 0.1 at a = 4. As with K, the choice that keeps the
  interval valid is the one that smooths less; a criterion such as GCV,
  which targets the fit of `f`, may choose a λ that is too large.

**Proposition 4(c) (the smoother's `V̂` is consistent; proved).** Take the
`V̂` above, with `sᵢ = ûᵢ²/mᵢ`, `mᵢ = (M²)ᵢᵢ`, evaluated at the root `ρ̂`.
Assume A1 with four moments, A2, A4–A9, Proposition 4's conditions on `S`,
`tr S/n → 0`, and for the shrinkage `r = Mf`: `‖r‖² = o(n)` and
`maxᵢ|rᵢ| ≤ c`. In place of A10 assume `Σᵢ bᵢ²(Sᵢᵢ + rᵢ²) = o(n)`. Then
`V̂/V → 1` in probability.

*Proof.* Proposition 2's three steps, with these changes.
- **`mᵢ` is bounded below**: `mᵢ = Mᵢᵢ² + Σ_{j≠i} Sᵢⱼ² ≥ (1 − s̄)²`.
- **The remainder in `sᵢ`.** At `ρ₀`, `ûᵢ = zᵢ + rᵢ` with `z = Mε`, and
  `zᵢ = Mᵢᵢεᵢ − qᵢ` with `qᵢ = Σ_{j≠i} Sᵢⱼεⱼ`. Since `(S²)ᵢᵢ ≤ Sᵢᵢ`,
  `Σ_{j≠i} Sᵢⱼ² ≤ Sᵢᵢ(1 − Sᵢᵢ)`, so `E qᵢ² ≤ σ̄²Sᵢᵢ` and
  `E qᵢ⁴ ≤ cSᵢᵢ²`. Then `sᵢ = εᵢ² + dᵢ` with
  `dᵢ = εᵢ²(Mᵢᵢ²/mᵢ − 1) + (−2Mᵢᵢεᵢqᵢ + qᵢ² + 2rᵢzᵢ + rᵢ²)/mᵢ`, and
  `|Mᵢᵢ²/mᵢ − 1| ≤ Sᵢᵢ/(1 − s̄)²`. So `E dᵢ² ≤ c(Sᵢᵢ + rᵢ²)`, using
  `|rᵢ| ≤ c`. Hence `E‖d‖² ≤ c(tr S + ‖r‖²) = o(n)` and
  `Σᵢ (E dᵢ²)^{1/2} ≤ c(√(n tr S) + √n‖r‖) = o(n)`. These are the two facts
  steps 1 and 2 used, with `tr S + ‖r‖²` in place of `K`. `Ω = As∘As` keeps a
  zero diagonal and bounded row sums, by Proposition 4's Lemma 2.
- **Step 1 (quadratic part)** goes through unchanged with this `d`.
- **Step 2 (linear part).** Now `â = μ + η` with `μ = (A + A′)Sf =
  b − (A + A′)r` and `η = Hε`. The extra mean `w = (A + A′)r` has
  `‖w‖² ≤ c‖r‖² = o(n)`, so `Σ wᵢ²sᵢ = O_p(‖r‖²)` (`E sᵢ ≤ c`) and the cross
  term with `b` is `O_p(√n · ‖r‖) = o_p(n)`. `Σ Var ηᵢ = tr(HΣH′) ≤
  c‖S‖_F² ≤ c tr S`, and the noise term's mean is at most `c‖H‖_F²`, so both
  are `O_p(tr S)`. In `Σ bᵢ²dᵢ`, the zero-diagonal quadratic form
  `−2Σ bᵢ²Mᵢᵢεᵢqᵢ/mᵢ` has variance at most `cΣᵢ bᵢ⁴Sᵢᵢ = o(n²)`. The
  linear form `2Σ bᵢ²rᵢzᵢ/mᵢ` has variance at most `cΣ bᵢ⁴rᵢ² = o(n²)`
  (A6). The rest has absolute mean at most `cΣ bᵢ²(Sᵢᵢ + rᵢ²) = o(n)`.
- **Step 3 (`ρ̂` for `ρ₀`)** is unchanged: `û(ρ̂) = û − tMWy` and
  `â(ρ̂) = â − tHWy` with `‖M‖_op, ‖H‖_op ≤ c`, and `Σ ûᵢ⁴ = O_p(n)` because
  `|rᵢ| ≤ c`. ∎

The proof hides two opposite `O(‖r‖²)` effects. The shrinkage in `ûᵢ` adds
`(2rᵢzᵢ + rᵢ²)/mᵢ` to `sᵢ`, which raises `V̂`. `â` estimates the shrunk
signal `μ`, not `b`, which lowers the linear part by
`Δ_b = Σᵢ (μᵢ² − bᵢ²)σᵢ²`. Over-smoothing therefore biases `V̂` too, not only
`ρ̂`. The condition `‖r‖² = o(n)` is separate from the Corollary's
`β = o(√n)`: `β` is an indefinite quadratic form in `r` (Proposition 3),
so neither condition implies the other.

**Numerical check (`tc_smoother_vhat.jl`).** Same design as
`tc_smoother.jl`, at `ρ₀`, R = 500. Under normal errors the mean gap between
`V̂` with the feasible `s` and with `r` removed from `s` is exact, and so is
`Δ_b`:

| a | λ | ‖r‖²/n | `V̂/V` feasible | `r` removed | 1 + Δ_b/V | gap: simulated | exact | first order with `b` |
|---|---|---|---|---|---|---|---|---|
| 1 | 0.001 | 0.0000 | 1.008 | 1.008 | 1.000 | 0.0000 | 0.0000 | 0.0000 |
| 1 | 10 | 0.0042 | 0.993 | 0.988 | 0.984 | 0.0053 | 0.0046 | 0.0043 |
| 1 | 100 | 0.052 | 0.818 | 0.778 | 0.778 | 0.0406 | 0.0396 | 0.0605 |
| 4 | 1 | 0.0016 | 1.000 | 0.998 | 0.994 | 0.0022 | 0.0018 | 0.0016 |
| 4 | 10 | 0.068 | 0.927 | 0.849 | 0.849 | 0.0779 | 0.0752 | 0.0883 |
| 4 | 100 | 0.83 | 0.550 | 0.228 | 0.229 | 0.322 | 0.321 | 1.471 |

(All 10 cells are in `tc_smoother_vhat_output.txt`.)

- **The exact gap matches the simulation** in all 10 cells, with `|z|` ≤ 2.2.
  Two cells have `z` = 2.1 and 2.2, both where the gap is largest relative
  to its SE.
- **`Δ_b` accounts for the r-removed mean** to within 0.004 where `r`
  matters; the 0.000–0.008 left at small λ is HC2's own bias. So the two
  effects explain `V̂`'s bias, and the shrunk signal is the larger one.
- **Power check.** The first-order shift with `b` in place of `μ` misses
  by 50% at a = 1, λ = 100 (0.0605 against 0.0396) and by a factor of 4.6
  at a = 4, λ = 100 (1.47 against 0.32).
- **Where `‖r‖²/n` ≤ 0.002,** `V̂/V` is 1.000–1.008, within the projection's
  HC2 range; that is λ ≤ 1 for both amplitudes. In this design `β` binds
  first: at a = 4, λ = 1, `V̂` is unbiased but coverage is 0.663
  (`tc_smoother.jl`), because `c` = 1.6.

**Choosing λ (measured; no theory).** Propositions 3 and 4 say when an
interval is valid, not how to pick λ. Two ways were tried in
`tc_lambda.jl`: GCV, and a plug-in estimate of `c` (same design as
`tc_smoother.jl`, λ on a half-decade grid from 10⁻³ to 10², R = 500).
- GCV is `n‖M_λy₀p‖²/(n − tr S_λ)²`, where `y₀p` comes from a pilot fit at
  λ = 10⁻³.
- The plug-in is `ĉ(λ) = (f̃′A_λf̃ − ŝ′diag(S_pA_λS_p))/√V̂_λ`, with
  `f̃ = S_py₀p` from the same pilot.

| a | λ | tr S | c | ĉ: mean | ĉ: sd | sd ρ̂ | RMSE ρ̂ | coverage | GCV picks (of 500) |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 0.001 | 222 | 0.003 | 0.001 | 0.03 | 0.119 | 0.119 | 0.960 | 0 |
| 1 | 0.1 | 131 | 0.014 | 0.068 | 0.84 | 0.075 | 0.075 | 0.944 | 4 |
| 1 | 1 | 62 | 0.096 | 0.246 | 1.53 | 0.059 | 0.059 | 0.944 | 46 |
| 1 | 3.16 | 41 | 0.261 | 0.446 | 1.77 | 0.054 | **0.057** | 0.944 | 82 |
| 1 | 10 | 27 | 0.659 | 0.864 | 2.02 | 0.049 | 0.062 | 0.900 | 275 |
| 1 | 31.6 | 18 | 1.331 | 1.597 | 2.35 | 0.044 | 0.080 | 0.720 | 68 |
| 4 | 0.001 | 222 | 0.031 | 0.003 | 0.03 | 0.109 | 0.110 | 0.970 | 0 |
| 4 | 0.1 | 131 | 0.232 | 0.243 | 0.87 | 0.068 | 0.071 | 0.946 | 4 |
| 4 | 0.316 | 93 | 0.598 | 0.645 | 1.34 | 0.056 | **0.068** | 0.908 | 27 |
| 4 | 1 | 62 | 1.594 | 1.677 | 1.93 | 0.045 | 0.090 | 0.648 | 167 |
| 4 | 3.16 | 41 | 4.139 | 4.261 | 2.91 | 0.034 | 0.156 | 0.028 | 302 |

With GCV's own choice, coverage is **0.888** (a = 1) and **0.320**
(a = 4). (All 22 rows are in `tc_lambda_output.txt`.)

- **GCV over-smooths for inference.** It picks λ = 3–10 at a = 1 and
  λ = 1–3 at a = 4. Those values give a good fit of `f`, but `c` is 0.3–4
  there. Under-coverage is mild for a weak surface and severe for a strong
  one. GCV must not be the default λ of a smoothed TC-CFE.
- **The RMSE-best λ is already too large for an interval.** RMSE is
  smallest where `c` ≈ 0.26 (a = 1) and 0.6 (a = 4), and coverage there is
  0.944 and 0.908. This is the usual undersmoothing result: a valid interval
  needs `c` → 0, so λ must be below the RMSE-optimal one. The price is
  precision. `sd(ρ̂)` doubles from λ = 1 to λ = 0.001.
- **`c` cannot be estimated well enough to pick λ.** While `c` ≤ 4, `ĉ`'s
  bias is at most 0.2, small beside its sd (0.645 against 0.598, 4.26
  against 4.14). Its sd,
  however, is 0.8–2.9 in the range that matters, because `f̃′A_λf̃` carries
  noise of the same order as `√V`. So it can flag a large bias (`c` ≳ 4) but
  cannot tell `c` = 0.3 from 0. Beyond `c` ≈ 4 it overshoots (30.7 against
  14.8). The cause is not checked; a likely one is the pilot's own error
  in `f̃` once `A_λ` is far from `A_p`.
- **What this leaves.** The largest λ with coverage ≥ 0.94 is about 1/3
  of GCV's modal choice at a = 1 (3.16 against 10) and 1/30 at a = 4 (0.1
  against 3.16). So the factor depends on the signal, and two designs do
  not justify a rule. An honest
  default is to undersmooth: take λ at the small end of the range where
  `V̂` is still stable, and report `sd(ρ̂)` across λ as a sensitivity
  check. The same holds for K with a projection (`tc_approx.jl`).

**Proposition 5 (a primitive condition for A5; proved).** Let `W` be
non-negative with a zero diagonal, `‖W‖_op ≤ c_W` and `‖W‖_F² ≥ κn`; a
row-standardised `W` with at most `m` neighbours per row has `κ = 1/m`.
For the exact-diagonal moment, with `d̄ = maxᵢ|dᵢ|`,

    ‖Cs‖_F²/n ≥ κ/2 − 2(K/n)(c_W + d̄)².

So A5 holds whenever `limsup K/n < κ/(4(c_W + d̄)²)`, and in particular
when `K/n → 0` with `d̄` bounded, with `liminf ‖Cs‖_F²/n ≥ κ/2`. A
sufficient condition for bounded `d̄` is `maxᵢ hᵢᵢ ≤ h̄ < 1/2`, which gives

    d̄ ≤ maxᵢ‖Weᵢ‖ √h̄ / ((1 − h̄)(1 − 2h̄)).

For Jin & Lee's `C = (W′ − Δ)M` the same holds with `max|Δᵢᵢ|` in place of
`d̄`, which Lemma 2(b) bounds under A3 alone.

**(b) The spectral form.** With `X = Ws + D = Σⱼ λⱼφⱼφⱼ′` and
`πⱼ = ‖P_Bφⱼ‖²`, the overlap of the basis with eigenvector `φⱼ`,

    ‖Cs‖_F² ≥ Σⱼ λⱼ²(1 − 2πⱼ) = ‖X‖_F² (1 − 2π̄_w),   π̄_w = Σⱼ λⱼ²πⱼ / Σⱼ λⱼ².

So A5 holds whenever the λ²-weighted mean overlap `π̄_w` stays below 1/2:
the basis must not absorb half of `X`'s energy. That condition does not
mention K/n at all; (a) is the special case that bounds every `πⱼ`-weighted
sum by `K‖X‖_op²`. It is the moment-matrix analogue of the spectral overlap
that the package reports for a smoother against `W`'s dominant modes.

*Proof of (a) and (b).* Write `Ws = (W + W′)/2` and `X = Ws + D`. Since `M` is an
orthogonal projection, `MCsM = M(Ws + D)M = MXM`, and `‖Cs‖_F ≥ ‖MCsM‖_F`.
The four pieces `MXM`, `P_BXM`, `MXP_B`, `P_BXP_B` are orthogonal in the
Frobenius inner product and sum to `X`, so
`‖MXM‖_F² ≥ ‖X‖_F² − ‖P_BX‖_F² − ‖XP_B‖_F²`. Each of the last two is at
most `‖P_B‖_F²‖X‖_op² = K‖X‖_op² ≤ K(c_W + d̄)²`. Since `diag Ws = 0`,
`‖X‖_F² = ‖Ws‖_F² + ‖D‖_F² ≥ ‖Ws‖_F²`, and
`‖Ws‖_F² = (‖W‖_F² + tr W²)/2 ≥ ‖W‖_F²/2` because `tr W² = Σᵢⱼ WᵢⱼWⱼᵢ ≥ 0`.
For a row-standardised row with at most `m` entries, `Σⱼ Wᵢⱼ² ≥ 1/m`. For
`d̄`: `(M∘M)d = −diag(MW) = diag(P_BW)`, and
`|eᵢ′P_BWeᵢ| ≤ ‖P_Beᵢ‖‖Weᵢ‖ = √hᵢᵢ‖Weᵢ‖`. Row `i` of `M∘M` has diagonal
`(1 − hᵢᵢ)²` and off-diagonal sum `Σ_{j≠i} hᵢⱼ² = hᵢᵢ(1 − hᵢᵢ)`, so its
dominance margin is `(1 − hᵢᵢ)(1 − 2hᵢᵢ)`, and
`‖(M∘M)⁻¹‖_∞ ≤ 1/((1 − h̄)(1 − 2h̄))`. For (b), the same orthogonal
split gives `‖MXM‖_F² ≥ ‖X‖_F² − 2‖P_BX‖_F²` (with `X` symmetric,
`‖XP_B‖_F = ‖P_BX‖_F`), and `‖P_BX‖_F² = tr(XP_BX) = Σⱼ λⱼ²φⱼ′P_Bφⱼ`. ∎

**How sharp it is (`tc_a5.jl`).** Exact computation, no simulation, for the
n = 900 kNN `W` (k = 6, so `κ` = 1/6 and the bound on `‖Ws‖_F²/n` is 0.083;
the actual value is 0.153) with the Chebyshev and hat bases used above:

| basis | K/n | max hᵢᵢ | ‖Cs‖_F²/n | d̄ | (a), measured ‖Ws‖_F, d̄ | (a), 1/(2m) | (a), the d̄ bound | π̄_w | (b) spectral |
|---|---|---|---|---|---|---|---|---|---|
| hat | 0.006 | 0.02 | 0.148 | 0.022 | 0.137 | 0.068 | 0.066 | 0.031 | 0.143 |
| hat | 0.019 | 0.11 | 0.134 | 0.087 | 0.093 | 0.024 | 0.003 | 0.119 | 0.117 |
| hat | 0.029 | 0.17 | 0.125 | 0.113 | 0.058 | 0 | 0 | 0.183 | 0.098 |
| hat | 0.056 | 0.38 | 0.102 | 0.184 | 0 | 0 | 0 | 0.334 | 0.052 |
| hat | 0.112 | 0.78 | 0.069 | 0.75 | 0 | 0 | 0 | 0.559 | 0 |
| hat | 0.251 | 0.94 | 0.041 | 398 | 0 | 0 | 0 | 0.931 | 0 |
| Chebyshev | 0.008 | 0.03 | 0.145 | 0.030 | 0.130 | 0.061 | 0.058 | 0.046 | 0.139 |
| Chebyshev | 0.032 | 0.41 | 0.123 | 0.159 | 0.039 | 0 | 0 | 0.188 | 0.096 |
| Chebyshev | 0.051 | 0.75 | 0.110 | 0.34 | 0 | 0 | 0 | 0.278 | 0.069 |
| Chebyshev | 0.102 | 0.98 | 0.083 | 40 | 0 | 0 | 0 | 0.935 | 0 |

- **(a) is close at small K/n and vanishes by K/n ≈ 0.03–0.05.** With the
  measured ingredients it is 0.137 against 0.148. The worst case
  `‖P_BX‖_F² ≤ K‖X‖_op²` is what loses: it treats every basis direction as
  if it met `X`'s top eigenvectors.
- **(b) is sharper and reaches about twice as far.** It is 0.143 against
  0.148 at the smallest K/n, still 0.052–0.069 at K/n ≈ 0.05 where (a) is
  zero, and it vanishes where `π̄_w` passes 1/2 (K/n ≈ 0.10–0.11 in both
  bases). A likely cause, not measured: as `d̄` grows, more of `X`'s energy
  sits on `D`'s entries at high-leverage units, which the basis covers
  almost entirely. What (b) still drops is `‖P_BXP_B‖_F² ≥ 0`.
  So A5 is proved while `π̄_w < 1/2` and stays an assumption beyond; it is
  measured at 0.04–0.15 in every design used here.
- **Leverage 1 makes `d` explode.** In the Chebyshev designs `d̄` is 40 at
  K/n = 0.10, 1.3 × 10⁴ at 0.19 and 1.9 × 10⁹ at 0.39, where `M∘M` is
  nearly singular, and at K/n = 0.39 that drives `‖Cs‖_F²/n` to 2.5. The
  exact-diagonal moment is ill-posed there, which is one more reason for
  A3 and for a basis with bounded leverage.

## Numerical check (`tc_clt.jl`)

Design: 6-NN `W`, random locations, heteroskedastic standardised `χ²₃` errors,
`R = 2000` draws per design. `z = g(ρ₀)/√V` uses the exact `V` of Lemma 1.

| Design | n | K/n | max hᵢᵢ | ‖Cs‖_op/‖Cs‖_F | skew z | exc. kurt. z | KS | size of \|z\| > 1.96 |
|---|---|---|---|---|---|---|---|---|
| hat basis, K ≈ 2√n | 400 | 0.09 | 0.43 | 0.18 | 0.41 | 1.63 | 0.041 | 0.054 |
| | 2500 | 0.04 | 0.27 | 0.06 | 0.16 | 0.05 | 0.018 | 0.057 |
| hat basis, K ≈ 0.15n | 400 | 0.16 | 0.60 | 0.19 | 0.32 | 1.33 | 0.038 | 0.050 |
| | 2500 | 0.14 | 0.76 | 0.08 | 0.14 | 0.16 | 0.030 | 0.048 |
| Chebyshev, K ≈ 2√n | 400 | 0.09 | 0.72 | 0.17 | 0.51 | 1.86 | 0.038 | 0.050 |
| | 2500 | 0.04 | 0.75 | 0.06 | 0.14 | 0.05 | 0.016 | 0.046 |
| **power check:** two complete groups | 400 | 0.09 | 0.43 | 1.00 | 3.02 | 15.3 | 0.193 | 0.041 |
| | 1600 | 0.05 | 0.37 | 1.00 | 2.75 | 11.1 | 0.185 | 0.048 |

The Monte Carlo SEs at R = 2000 are 0.055 for the skewness, 0.11 for the
excess kurtosis and 0.005 for the size. The KS 95% critical value is 0.030.
The full output, with n = 900 and 1600 and every quantity, is in
`tc_clt_output.txt`.

- **Normality improves with n in all three kNN regimes.** That includes
  K/n fixed at about 0.15, and the Chebyshev basis, whose largest leverage
  stays at 0.7–0.8. So the CLT does not need `max hᵢᵢ → 0`.
- **The skewness falls like `‖Cs‖_op/‖Cs‖_F`.** Its ratio to that quantity
  is about 2.3–2.5 across n, which is the Berry–Esseen-type rate the sketch
  predicts.
- **The bound in Lemma 2(b) holds in every design.** For example, the
  largest `|Δᵢᵢ|` is 0.148 against a bound of 0.196.
- **Power check.** With `W` made of two complete groups, `‖Cs‖_op/‖Cs‖_F`
  stays at 1, and `z` stays far from normal at every n (skewness 2.4–3.0,
  KS 0.19). So the check can detect a failing CLT. This is the
  Kelejian–Prucha case of a number of neighbours that grows with n.
- **Caveat:** the size of `|z| > 1.96` is close to 0.05 even in the power
  check. A symmetric two-sided size is not a sensitive test of normality
  here; the skewness and the KS distance are.

## Numerical check with no fourth moment (`tc_clt_moments.jl`, 2026-11-19)

`tc_clt.jl`'s hat-basis design (same seed, `W`, basis, `f` and variances),
K ≈ 2√n, R = 2000, with two error laws whose fourth moment is infinite and
whose moments of order below 3 are finite, so A1 holds for any `δ < 1`:
standardised `t₃` (symmetric) and a centred Pareto with tail index 3
(skewed). The KS 95% critical value is 0.030.

| n | `‖Cs‖_op/‖Cs‖_F` | KS, `t₃` | KS, Pareto | KS, `χ²₃` (`tc_clt.jl`) |
|---|---|---|---|---|
| 400 | 0.178 | 0.062 | 0.131 | 0.041 |
| 900 | 0.114 | 0.044 | 0.111 | 0.032 |
| 1600 | 0.081 | 0.050 | 0.097 | 0.016 |
| 2500 | 0.064 | 0.039 | 0.078 | 0.018 |
| 4900 | 0.044 | 0.038 | 0.067 | — |

- **Normality improves with n under both laws, but slowly**, as the proof's
  exponents predict. At n = 4900 neither law is within the KS critical
  value: `t₃` has levelled off near 0.04, and the Pareto law is still
  falling. Its sample skewness of `z` stays at 1–3.6, which with an
  infinite fourth moment is itself very noisy.
- The size of `|z| > 1.96` is 0.034–0.052, and the sd of `z` is
  0.93–1.10 (Pareto) and 0.98–1.03 (`t₃`).
- **Power check:** with `t₃` errors and `W` of two complete groups, where
  (a) fails, the KS distance stays at 0.18–0.19 for every n.

So the simulation is consistent with Theorem 1 under `2 + δ` moments, and
shows its practical cost: with heavy tails, n in the thousands still leaves
a visible departure from normality in the tails of `z`. This is about the
moment at the true `ρ`; `ρ̂`'s coverage is measured next.

**Coverage of `ρ̂` (`tc_coverage_tails.jl`, 2026-11-19).** The same design
with `y = (I − ρ₀W)⁻¹(f + ε)`, `ρ₀ = 0.4`, Theorem 1's estimator (Jin & Lee's
first step, `C = (W′ − Δ)M`) and R = 1000 per cell (Monte Carlo SE of a
coverage near 0.95: 0.007). Three 95% intervals share the data's delta-method
denominator: the robust plug-in variance, the exact `V` (the oracle), and the
homoskedastic plug-in (the power check).

| Errors | Robust, n = 400 → 2500 | Oracle | Homoskedastic |
|---|---|---|---|
| normal | 0.948–0.952 | 0.951–0.955 | 0.797–0.853 |
| `χ²₃` | 0.932–0.951 | 0.948–0.953 | 0.805–0.845 |
| `t₃` | 0.953–0.969 | 0.952–0.969 | 0.800–0.890 |
| Pareto, tail index 3 | 0.961–0.977 | 0.950–0.969 | 0.866–0.920 |

- **Heavy tails make the intervals conservative, not anti-conservative.** In
  every `t₃` and Pareto cell the coverage is at or above 0.95, by up to
  0.027 (Pareto, n = 900), which is about 4 Monte Carlo SEs. The oracle
  intervals return to 0.950–0.952 at n = 2500; the robust ones stay at
  0.956 (`t₃`) and 0.961 (Pareto).
- This fits what the moment showed: under heavy tails `z = g/√V` is more
  peaked than normal, so `|z| > 1.96` is rarer (its size was 0.034–0.052 in
  `tc_clt_moments.jl`). With the plug-in variance, a large `εᵢ` inflates
  both `g` and `V̂`, which thins the tails of the ratio further; that is
  probably why the robust intervals over-cover more than the oracle. This
  explanation is not checked.
- The median robust se is within 5% of the sd of `ρ̂` in every cell, and the
  bias of `ρ̂` is at most 0.017 (`χ²₃`, n = 400), small against its sd.
- **Power check:** the homoskedastic intervals under-cover in every cell,
  0.80–0.92. The gap is smaller under the Pareto law.
- Not covered: one design (hat basis, smooth variances, `ρ₀ = 0.4`), and
  two-sided coverage only, which can hide a lopsided miss under the skewed
  Pareto law.

## Theory step 2: the robust variance (first analysis, 2026-11-19)

The estimator, as coded in `tc_lib.jl` and `tc_jinlee.jl`, is

    V̂ = 2σ̂′(Cs∘Cs)σ̂ + Σᵢ (âᵢ² − noiseᵢ) σ̂ᵢ²,   σ̂ᵢ² = ûᵢ²/Mᵢᵢ,  û = My₀(ρ̂).

**The linear part.** Since `CP_B = PMP_B = 0`, `â = (C + C′)P_By₀ = b + η`, with
`η = C′P_Bε`. So `E âᵢ² = bᵢ² + Var ηᵢ`, and
`Σᵢ Var ηᵢ = tr(C′P_BΣP_BC) ≤ σ̄²‖C‖_op² K`. So without the noise term, the
linear part is biased upwards by `O(K)`, against `V ≍ n`: a relative bias of
`O(K/n)`. That matches the 3–10% excess measured at K/n of 0.1–0.2 before the
term was added. The noise term, `diag(H(Q′Σ̂Q)H′)` with `H = C′Q`, estimates
`Var ηᵢ` and removes it.

**The plug-in variances are the weak point when K/n does not vanish.**
`E ûᵢ² = Σⱼ Mᵢⱼ²σⱼ²`, with `Mᵢⱼ = −hᵢⱼ` for `j ≠ i` and
`Σ_{j≠i} hᵢⱼ² = hᵢᵢ(1 − hᵢᵢ)`. So

    E σ̂ᵢ² − σᵢ² = Σ_{j≠i} hᵢⱼ²(σⱼ² − σᵢ²) / (1 − hᵢᵢ),

which is at most `hᵢᵢ · maxⱼ|σⱼ² − σᵢ²|`. It is zero under
homoskedasticity, `O(K/n)` on average otherwise, and does not vanish if
`K/n → c > 0`. The products `σ̂ᵢ²σ̂ⱼ²` in the quadratic part add a
correlation bias of the same order: `Cov(ûᵢ², ûⱼ²)` involves
`(Σₖ MᵢₖMⱼₖσₖ²)²`, and summing `cᵢⱼ²hᵢⱼ²` over all pairs gives at most
`maxᵢⱼ cᵢⱼ² · K`, so the relative bias is `O(K/n)`. This is probably the 1–3%
conservative bias seen in `tc_null.jl`.

**The conjecture for step 2,** under A1–A6 with bounded eighth moments, which
the variance of the degree-4 terms of `V̂` needs:
- `V̂/V → 1` in probability when `K/n → 0`;
- the same when `K/n → c > 0` under homoskedasticity;
- `V̂` is inconsistent under heteroskedasticity with `K/n → c > 0`.

This is the many-covariates problem for heteroskedasticity-robust variances
(Cattaneo, Jansson & Newey 2018; Kline, Saggio & Sølvsten 2020). It suggests a
leave-out estimator of the `σᵢ²` and of the products `σᵢ²σⱼ²`, in the manner
of Kline et al., as the remedy. If it works, that would be the first part of
this project that is genuinely new.

**Numerical check (`tc_vhat.jl`, 2026-11-19).**
- Setup: n = 900, a kNN `W`, a Chebyshev surface basis plus one covariate, with
  K/n from 0.05 to 0.39, normal and `χ²₃` errors, and R = 1000. Everything is
  evaluated at the true ρ, and `E V̂/V` is reported with its `z`.
- Checks of the check: the oracle `V̂` (true `σ²`) gives 0.997–1.002 in every
  cell, and the simulated `Var g/V` agrees with 1. The homoskedastic `V̂` fails
  under heteroskedasticity (0.67–0.75 at small K/n, `z` down to −118), so the
  test has power.

Results for the current `V̂` (HC2 plug-in):

| Variances | K/n = 0.05 | 0.10 | 0.19 | 0.28 | 0.39 |
|---|---|---|---|---|---|
| homoskedastic, normal | 1.004 | 1.006 | 1.009 | 1.011 | 1.030 |
| smooth in space, normal | 1.004 | 1.006 | 0.998 | 0.962 | 0.998 |
| rough (iid log-normal), normal | 1.010 | 1.024 | 1.042 | 1.069 | 1.134 |
| rough, `χ²₃` | 1.028 | 1.058 | 1.130 | 1.167 | 1.154 |

So the conjecture holds in part:
- **Rough heteroskedasticity:** the bias grows with K/n, as predicted. It is
  upward (conservative), about 1/3 of K/n for normal errors, and `z` is 5–9
  from K/n = 0.1.
- **Smooth heteroskedasticity:** no bias is detectable. This is consistent
  with the formula, because `σⱼ² − σᵢ²` is small where `hᵢⱼ²` puts its mass.
- **Homoskedastic errors:** at most 3%.

**Cattaneo, Jansson & Newey's `(M∘M)⁻¹û²` cannot be used here.** Their
estimator needs `max hᵢᵢ < 1/2`. These designs have `max hᵢᵢ` from 0.75 to
1.0, because the Chebyshev basis has boundary leverage. At K/n ≥ 0.28 the
estimate reaches 1.1–2.7 × V. A Gaussian removal of `Cov(sᵢ, sⱼ)` makes it
worse, and even goes negative. A remedy must therefore cope with leverage near
1: Kline, Saggio & Sølvsten's leave-out with sample splitting, or a local
(hat-function) basis, whose leverage is bounded.

**Kline, Saggio & Sølvsten's leave-one-out variances (`tc_vhat_lo.jl`,
2026-11-19).** Replace the plug-in by `sᵢ = y₀ᵢûᵢ/Mᵢᵢ`, which is
`y₀ᵢ(y₀ᵢ − ŷ₋ᵢ)`, the product with the residual of a fit that leaves unit `i`
out. Since `û = Mε` and `y₀ = f + ε`, `E[y₀ᵢûᵢ] = Mᵢᵢσᵢ²`, so `E sᵢ = σᵢ²`
**exactly, for any variances and any `Mᵢᵢ > 0`**: no `(M∘M)⁻¹` is needed.
That holds unit by unit only. The products below divide by `MᵢᵢMⱼⱼ`, and
with a strong signal and leverage near 1 they ruin the quadratic part (the
strong-signal check below). The linear part's noise term is linear in `s`,
so it is unbiased too. What remains is the bias of products of dependent
estimates. For `i ≠ j`, with third moments `μ₃`,

    E[sᵢsⱼ] = σᵢ²σⱼ²(1 + Mᵢⱼ²/(MᵢᵢMⱼⱼ)) + [fᵢfⱼ(MΣM)ᵢⱼ + fᵢMᵢⱼMⱼⱼμ₃ⱼ + fⱼMᵢⱼMᵢᵢμ₃ᵢ]/(MᵢᵢMⱼⱼ),

and `âᵢ²sᵢ` in the linear part has a bias of the same kind. Kline et al.
remove these by sample splitting, which needs their condition
`P_{iℓ,1}P_{iℓ,2} = 0`; a dense projection does not satisfy it.

The check uses `tc_vhat.jl`'s design and seeds exactly, so the HC2 column is
the committed one:

| Variances, errors | HC2, K/n = 0.05 → 0.39 | leave-one-out | largest \|z\|, leave-one-out |
|---|---|---|---|
| constant, normal | 1.004–1.030 | 0.989–1.009 | 2.9 |
| smooth, normal | 0.962–1.006 | 0.995–1.009 | 1.4 |
| rough, normal | 1.010–1.134 | 0.974–1.016 | 2.4 |
| constant, `χ²₃` | 1.000–1.048 | 0.990–1.022 | 2.0 |
| smooth, `χ²₃` | 0.985–1.027 | 0.968–1.009 | 3.3 |
| rough, `χ²₃` | 1.028–1.167 | 0.995–1.030 | 2.1 |

- **It removes the rough-heteroskedasticity bias.** Where HC2 reaches 1.13
  and 1.17 with `z` of 6–9, leave-one-out is within 3% of `V`, and only one
  of the 30 cells has `|z| > 3` (smooth, `χ²₃`, K/n = 0.19: 0.968). The
  oracle stays at 0.997–1.002, and HC2 itself serves as the power check: the
  same test finds its bias at `z` up to 9.4.
- **The parts are not separately unbiased.** The linear part falls with K/n,
  to 0.84–0.87 of its true value at K/n = 0.39 with smooth variances (and
  0.95–0.97 otherwise); the quadratic part runs up to 6% high
  (rough, `χ²₃`, K/n = 0.19). The total stays within 3% because the
  quadratic part dominates `V` here, and because `f` is weak. The
  strong-signal check below shows what happens when it is not.
- **It is noisier.** At K/n = 0.39 its sd is 0.44–0.78 of `V`, against HC2's
  0.31–0.66; at small K/n the two are within 10% of each other. The effect
  on coverage is not measured.

**With a strong signal, leave-one-out fails (`tc_vhat_lo_lin.jl`).** The
same design and seeds, with `f` scaled so that the linear part is 90% of
`V`. The quadratic part's true value is unchanged; only `f` grows.

| K/n | quadratic part, leave-one-out | total, leave-one-out | total, HC2 |
|---|---|---|---|
| 0.05 | 0.80–0.98 | 0.968–1.007 | 0.990–1.003 |
| 0.10 | 0.73–0.96 | 0.957–0.994 | 0.995–1.010 |
| 0.19 | 0.69–0.96 | 0.945–0.982 | 0.994–1.010 |
| 0.28 | −0.19 to 0.16 | 0.830–0.894 | 0.990–1.035 |
| 0.39 | −42 to −16 | −3.2 to −0.8 | 0.987–1.035 |

- **The quadratic part collapses as K/n grows**, and is negative from
  K/n = 0.28, when the leverage reaches 1. This is the
  `fᵢfⱼ(MΣM)ᵢⱼ/(MᵢᵢMⱼⱼ)` term in `E[sᵢsⱼ]` above: it grows with `‖f‖²` and is
  divided by `MᵢᵢMⱼⱼ`. Its sd is up to 28 × `V` at K/n = 0.39. The total is
  10–17% low at K/n = 0.28 (`z` of −7 to −10), and negative at 0.39.
- **The linear part is close to unbiased**: 0.93–1.04 in 27 of 30 cells; at
  K/n = 0.39 two `χ²₃` cells reach 0.88 and one normal cell 1.11. The
  prediction written before the run,
  that the linear part's `2bᵢfᵢ(GQ′ΣM)ᵢᵢ/Mᵢᵢ` bias would show here at
  10–16%, was wrong. The 16% shortfall in `tc_vhat_lo.jl` came with a weak
  `f` and is not this term.
- HC2 is good here (0.987–1.035), because its quadratic part, which carries
  its rough-heteroskedasticity bias, is only 10% of `V`. The oracle stays at
  0.998–1.000, which is the power check for the method.

So neither estimator is safe everywhere. HC2 is biased up when the quadratic
part dominates and the variances are rough. Leave-one-out is biased, and
then unusable, when the signal is strong and the leverage is near 1. A
hybrid, leave-one-out in the linear part and HC2 in the quadratic part,
keeps HC2's quadratic bias. What is needed is an estimate of `σᵢ²σⱼ²` that
is free of `f`, for example a leave-two-out product, or Kline et al.'s
split, which a dense projection does not allow.

**Leverage 1 rules out any estimate free of `f`.** An estimator that does
not change when `f` moves within `span B` is a function of the maximal
invariant `My₀ = û`. If `Mᵢᵢ = 0`, row `i` of `M` is zero, since
`Mᵢᵢ = ‖Mᵢ·‖²`, so the law of `û` does not depend on `εᵢ` at all. But
`ε′Aε = û′(Wε + Dû)`, in which `εᵢ` enters through `Wε`, so `V` depends on
`σᵢ²`. No invariant estimator is then unbiased for `V`, and one that uses
`y₀` picks up `f`, as leave-one-out does. For `Mᵢᵢ` small but positive, the
information about `σᵢ²` in `û` shrinks with `Mᵢᵢ`. So step 2 needs leverage
bounded away from 1 as an assumption, not only a better estimator.

**Leverage is not the whole story for leave-one-out (`tc_vhat_lo_hat.jl`).**
The same 30 cells with a tensor hat basis (`k × k` hats plus the covariate,
`k` = 7 to 19), whose leverage stays at 0.38–0.94 up to K/n = 0.25. At
K/n = 0.40 some unit has leverage 1, `M∘M` is singular and the cell is
skipped. The script adds CJN's `(M∘M)⁻¹û²`. It runs twice: with `f` scaled
so that the linear part is 90% of `V`, and (argument `weak`) with
`tc_vhat.jl`'s `f`, where the quadratic part is 57–79% of `V`. Totals,
E V̂/V over the six variance-and-error cells:

| K/n (max leverage) | HC2, strong | HC2, weak | leave-one-out, strong | leave-one-out, weak | CJN, strong | CJN, weak |
|---|---|---|---|---|---|---|
| 0.06 (0.38) | 0.995–1.008 | 1.005–1.042 | 0.966–1.007 | 0.989–1.015 | 0.991–1.006 | 0.991–1.007 |
| 0.11 (0.78) | 0.994–1.004 | 1.004–1.068 | 0.958–0.987 | 0.972–1.005 | 0.986–1.001 | 0.981–0.989 |
| 0.19 (0.88) | 0.993–1.021 | 1.004–1.161 | 0.940–0.988 | 0.971–1.015 | 0.980–0.996 | 0.960–0.974 |
| 0.25 (0.94) | 1.001–1.023 | 1.000–1.212 | 0.925–1.002 | 0.961–1.005 | 0.966–0.992 | 0.839–0.935 |

- **Leave-one-out's strong-signal bias stays with bounded leverage.** Its
  quadratic part is 0.68–0.99 at K/n ≤ 0.19, where the leverage is at most
  0.88. This fits the `fᵢfⱼ(MΣM)ᵢⱼ/(MᵢᵢMⱼⱼ)` term: with constant variances
  `(MΣM)ᵢⱼ = −σ²Pᵢⱼ`, which for a local basis is of order K/n between
  neighbours, and `fᵢfⱼ ≈ fᵢ²` for a smooth `f`. So the relative bias is
  of order the signal-to-noise ratio times K/n, and is negative.
- **HC2's bias is in its quadratic part** (up to 1.44 of its true value), so
  it shows when that part dominates and the variances are rough or `χ²₃`
  (up to 1.21 at K/n = 0.25).
- **CJN's bias does not depend on `f`.** Its quadratic part is identical in
  the two runs (same seeds; it uses only `û`): 0.99–1.01 at K/n = 0.06,
  0.985–0.988 at 0.11, 0.96–0.98 at 0.19 and 0.88–0.94 at 0.25. Its linear
  part falls too in the weak-signal run (0.72–0.95 at K/n = 0.25), so its
  total is worst there.
- The oracle stays at 0.997–1.003 throughout, which is the power check for
  the method; the same test finds the other biases at `z` up to 17.

**Where step 2 stands.** At K/n ≤ 0.11 with leverage ≤ 0.78, CJN is within
2% of `V` at both signal strengths (0.981–1.007); HC2 is within 7%, and
leave-one-out within 4.2%. No estimator tried is within 5% everywhere at
K/n ≥ 0.19. That fits a theorem for K/n → 0 with leverage bounded away
from 1, which is the case to prove. For larger K/n, CJN's bias does not
depend on `f`, so it may be removable from `û` alone; Cattaneo, Jansson &
Newey's own correction is the place to start. The Gaussian correction tried
in `tc_vhat.jl` made things worse there, but leverage reached 1 in that
design.

**Proposition 2 (the robust `V̂` is consistent when K/n → 0; proved).**
Take the coded `V̂` with the HC2 plug-in `sᵢ = ûᵢ²/Mᵢᵢ`, evaluated at the
root `ρ̂`. Assume A1 with `μ₄ = supᵢ E εᵢ⁴ < ∞`, A2–A9, A10 and `K/n → 0`.
Then `V̂/V → 1` in probability. Neither the noise term nor any leave-out
correction is needed for this; both act only on the `O(K/n)` bias.

*Proof.* Write `Ω = Cs∘Cs` and `G₂ = (C + C′)P_B`, so that `V = 2σ²′Ωσ² +
Σᵢ bᵢ²σᵢ²` and `V̂ = 2s′Ωs + Σᵢ (âᵢ² − (G₂∘G₂ s)ᵢ)sᵢ`. Since A5 gives
`V ≥ 2σ̲⁴‖Cs‖_F² ≥ cn`, it is enough to show `V̂ − V = o_p(n)`. Two facts
carry the argument:
- **`Ω` is bounded although `Cs` is dense.** Its entries are non-negative,
  its diagonal is zero (`diag C = 0`), and its row sums are
  `Σⱼ Csᵢⱼ² = ‖Cs eᵢ‖² ≤ ‖Cs‖_op² ≤ c` (Lemma 2). So `‖Ω‖_op ≤ c` and
  `‖Ω‖_F² ≤ nc²`.
- **`sᵢ` is `εᵢ²` plus a remainder of order `hᵢᵢ`.** At `ρ₀`, with `f ∈ span
  B`, `ûᵢ = Mᵢᵢεᵢ − rᵢ` with `rᵢ = Σ_{j≠i} hᵢⱼεⱼ`. Then `sᵢ = εᵢ² + dᵢ`,
  `dᵢ = −hᵢᵢεᵢ² − 2εᵢrᵢ + rᵢ²/Mᵢᵢ`. From `Σ_{j≠i} hᵢⱼ² = hᵢᵢ(1 − hᵢᵢ)`,
  `E rᵢ² ≤ σ̄²hᵢᵢ` and `E rᵢ⁴ ≤ (3σ̄⁴ + μ₄)hᵢᵢ²`; with A3 this gives
  `E dᵢ² ≤ c hᵢᵢ`. Hence `E‖d‖² ≤ cK` and `Σᵢ (E dᵢ²)^{1/2} ≤ c Σᵢ √hᵢᵢ ≤
  c√(nK)`, by Cauchy–Schwarz and `Σᵢ hᵢᵢ = K`. Both are `o(n)` when
  `K/n → 0`.

1. **Quadratic part at `ρ₀`.** With `e = (εᵢ²)ᵢ`,

       s′Ωs − σ²′Ωσ² = (e − σ²)′Ω(e − σ²) + 2σ²′Ω(e − σ²) + 2e′Ωd + d′Ωd.

   The first term has mean zero because `Ω` has a zero diagonal and the
   `εᵢ² − σᵢ²` are independent; its variance is at most
   `2μ₄²‖Ω‖_F² = O(n)`. The second has variance at most `4μ₄‖Ωσ²‖² = O(n)`.
   So both are `O_p(√n)`. For the third, `E|e′Ωd| ≤ Σᵢⱼ Ωᵢⱼ (E εⱼ⁴)^{1/2}
   (E dᵢ²)^{1/2} ≤ c Σᵢ √hᵢᵢ = O(√(nK))`, using the row sums of `Ω`. The
   fourth is at most `‖Ω‖_op‖d‖² = O_p(K)`.
2. **Linear part at `ρ₀`.** `â = b + η`, with `η = C′P_Bε`. Split
   `Σᵢ âᵢ²sᵢ − Σᵢ bᵢ²σᵢ²` into four pieces:
   - `Σ bᵢ²(εᵢ² − σᵢ²)` has variance at most `μ₄ Σ bᵢ⁴ ≤ μ₄ maxᵢ bᵢ² ‖b‖²`,
     which is `o(n²)` by A6 and `‖b‖² = O(n)` (A8, Lemma 2).
   - `Σ bᵢ²dᵢ`. Its part `−2Σ bᵢ²εᵢrᵢ` is a quadratic form in `ε` with a
     zero diagonal and matrix entries `bᵢ²hᵢₖ`, so its variance is at most
     `4σ̄⁴ Σᵢ bᵢ⁴ hᵢᵢ = o(n²)`. Its other parts have absolute mean at most
     `c Σᵢ bᵢ² hᵢᵢ`, which is `o(n)` by A10.
   - `2Σ bᵢηᵢsᵢ ≤ 2(Σ bᵢ²sᵢ)^{1/2}(Σ ηᵢ²sᵢ)^{1/2}`, and
   - `Σ ηᵢ²sᵢ` and the noise term `s′(G₂∘G₂)s`. Their means are at most
     `c Σᵢ Var ηᵢ ≤ c‖C‖_op²K` and `c‖G₂‖_F² ≤ cK`, because
     `E ηᵢ⁴ ≤ c (Var ηᵢ)²`, `E sᵢ² ≤ c` and `E sᵢsₖ ≤ c` under four moments.
     So they are `O_p(K)`, and the cross term is `O_p(√(nK))`.
3. **Replacing `ρ₀` by `ρ̂`.** With `t = ρ̂ − ρ₀ = O_p(n^{−1/2})` (the
   Corollary) and `m = MWy`, `û(ρ̂) = û − tm` and `â(ρ̂) = â − tG₂Wy`. Here
   `‖m‖² = O_p(n)` and `‖G₂Wy‖² = O_p(n)` by A7 and A8, so the change `δa` in
   `â` has `‖δa‖ = O_p(1)`. The change in `s` is
   `δsᵢ = (−2tûᵢmᵢ + t²mᵢ²)/Mᵢᵢ`, with
   `‖δs‖ ≤ c(|t| ‖û‖₄‖m‖ + t²‖m‖²) = O_p(n^{−1/2} n^{1/4} n^{1/2}) + O_p(1)
   = O_p(n^{1/4})`, because `Σ ûᵢ⁴ = O_p(n)` under four moments. So the
   quadratic part moves by at most `‖Ω‖_op(2‖s‖‖δs‖ + ‖δs‖²) =
   O_p(n^{3/4})`. The linear part moves by terms such as
   `|t| Σ âᵢ²|ûᵢmᵢ|/Mᵢᵢ ≤ c|t|(Σ âᵢ²ûᵢ²)^{1/2}(Σ âᵢ²mᵢ²)^{1/2}`, which is
   `O_p(n^{−1/2})·O_p(√n)·o_p(n)`, since `Σ bᵢ²mᵢ² ≤ maxᵢ bᵢ² ‖m‖² = o_p(n²)`
   by A6; and `Σ (2âᵢδaᵢ + δaᵢ²)sᵢ`, which is
   `O_p(√n)·O_p(maxᵢ sᵢ)^{1/2} + O_p(maxᵢ sᵢ) = O_p(n^{3/4})`, since
   `maxᵢ sᵢ = O_p(√n)` when `E sᵢ² ≤ c`. All of these are `o_p(n)`. ∎

**What the proof says about the measurements.** The remainders are of order
`√(K/n)` relative to `V` in the worst case, and the measured bias at
K/n ≈ 0.05 (0.990–1.042 in all the designs above) is smaller than that. The
proof does not cover `K/n → c > 0`, where `Σ √hᵢᵢ ≍ n` and the bias
measured above stays. The conjecture asked for eighth moments; consistency
needs only four. The eighth would be needed for a rate, or for the variance
of `V̂`.

**The CJN version.** With `s = (M∘M)⁻¹û²` the same steps go through if
`‖(M∘M)⁻¹‖_∞` is bounded, which Cattaneo, Jansson & Newey obtain from
`maxᵢ hᵢᵢ < 1/2`; then `E s = σ²` exactly, the `dᵢ` have mean zero, and A10
is not needed. That condition is stronger than A3, and the designs above
(leverage up to 0.94) do not satisfy it, although CJN measured well there
at K/n ≤ 0.11.

**Coverage of `ρ̂` with each `V̂` (`tc_coverage_vhat.jl`).** The exact-diagonal
TC-CFE at its feasible root, with `ρ₀ = 0.4`, in `tc_vhat_lo_hat.jl`'s
hat-basis design: smooth and rough variances, normal and `χ²₃` errors, both
signal strengths, K/n from 0.06 to 0.25, R = 1000 per cell (Monte Carlo SE
0.007). A negative `V̂` counts as not covering.

| `V̂` | weak signal, K/n ≤ 0.19 | weak, K/n = 0.25 | strong signal, K/n ≤ 0.19 | strong, K/n = 0.25 |
|---|---|---|---|---|
| oracle | 0.945–0.963 | 0.950–0.961 | 0.945–0.963 | 0.953–0.964 |
| HC2 (coded) | 0.940–0.966 | 0.956–0.968 | 0.945–0.958 | 0.953–0.965 |
| leave-one-out | 0.934–0.955 | 0.937–0.950 | 0.929–0.950 | 0.906–0.936 |
| CJN | 0.932–0.953 | 0.911–0.940 | 0.943–0.957 | 0.950–0.958 |

- **HC2, the coded estimator, has the best coverage**, within 0.94–0.97
  everywhere. Its upward bias in `V̂` (up to 21% under rough variances)
  moves coverage by at most 1.6 points above the oracle's, because the
  standard error is the square root of `V̂` and the tail is thin there.
- **Leave-one-out under-covers with a strong signal** (0.906–0.936 at
  K/n = 0.25), with `V̂ < 0` in up to 15 of 1000 replications.
- **CJN under-covers at K/n = 0.25 with a weak signal** (0.911–0.940),
  with `V̂ < 0` in up to 13 replications; elsewhere it is 0.932–0.958.
- **The power check works only for smooth variances.** The homoskedastic
  `V̂` covers 0.869–0.921 there (weak signal), but 0.95–0.97 under rough
  variances: rough variances that are unrelated to location average out in
  `V`, so a pooled `σ̂²` is nearly right for this statistic. That is a fact
  about this `V`, not a failure of the check.

So for inference the conservative HC2 estimator is the safe choice in these
designs, and the estimators that remove its bias in `E V̂` pay for it in
variance and in under-coverage. A bias correction is worth building only
if it keeps HC2's coverage.

**The exact bias of CJN's quadratic part (proved; `tc_cjn_bias.jl`).** With
`s = Λû²`, `Λ = (M∘M)⁻¹` and `Ω = Cs∘Cs`, at `ρ₀`:

    E[2s′Ωs] − 2σ²′Ωσ² = 4 tr(ΩΛ[(MΣM)∘(MΣM)]Λ),   for any error law.

*Proof.* `E s = σ²`, so the left side is `2 tr(Ω Cov s)`, with
`Cov s = Λ Cov(û²) Λ`. For `û = Mε` with independent `εₖ`,
`Cov(ûᵢ², ûⱼ²) = 2(MΣM)ᵢⱼ² + Σₖ Mᵢₖ²Mⱼₖ²κₖ`, where `κₖ` is the fourth
cumulant. The second part is `(M∘M)Diag(κ)(M∘M)`, so it contributes
`Λ(M∘M)Diag(κ)(M∘M)Λ = Diag(κ)` to `Cov s`, and `tr(Ω Diag(κ)) = 0`
because `diag Ω = 0`. ∎

So the "Gaussian" correction in `tc_vhat.jl` is in fact distribution-free
for the quadratic part. It failed there because the leverage reached 1. In
the hat-basis design (R = 1000):

- **The formula is exact.** The predicted ratio `1 + bias/Vq` falls from
  0.993–0.995 at K/n = 0.06 to 0.888–0.940 at K/n = 0.25, and the simulated
  mean matches it in all 24 cells (`|z|` ≤ 1.8). The `χ²₃` cells, where
  `κ ≠ 0`, match as well as the normal ones, which checks that the cumulant
  term cancels. Power: the same test separates the uncorrected mean from 1,
  for example 0.964 at K/n = 0.19 with rough variances, a `z` of −5.4.
- **A plug-in correction works up to K/n = 0.19.** Subtracting
  `4 tr(ΩΛ[(MSM)∘(MSM)]Λ)` with `S = Diag(s)` gives 0.986–1.015 at
  K/n ≤ 0.19 (`|z|` ≤ 2.0), at no cost in sd. At K/n = 0.25, where the
  leverage reaches 0.94, it fails: 0.52–0.92, `z` from −5 to −28. There the
  plug-in `(MSM)ᵢⱼ²` carries its own bias, which involves `σₖ⁴` and so
  cannot be removed from one observation per unit.
- This corrects only the quadratic part; the next paragraph does the linear
  part.

**The exact bias of CJN's linear part (proved; `tc_cjn_lin.jl`).** At `ρ₀`,
`â = b + η` with `η = Hε`, `H = MWP_B`, and the noise term is
`noiseᵢ = Σₖ Hᵢₖ²sₖ`. Write `X = (MΣM)∘(MΣM)`. Then

    E Σᵢ(âᵢ² − noiseᵢ)sᵢ − Σᵢ bᵢ²σᵢ²
        = 2Σᵢ bᵢHᵢᵢμ₃ᵢ + 2Σᵢₖ Λᵢₖ(HΣM)ᵢₖ² − 2Σᵢ[(H∘H)ΛXΛ]ᵢᵢ,

the skewness, cross and noise terms.

*Proof.* For linear forms `ξ = α′ε` and `ζ = γ′ε` in independent `ε`,
`E[ξζ²] = Σₗ αₗγₗ²μ₃ₗ` and
`E[ξ²ζ²] = E ξ² E ζ² + 2(E ξζ)² + Σₗ αₗ²γₗ²κₗ`. With `ξ = ηᵢ`, `ζ = ûₖ`
and `Λ(M∘M) = I`:
- `E[ηᵢsᵢ] = Σₖ Λᵢₖ Σₗ HᵢₗMₖₗ²μ₃ₗ = Hᵢᵢμ₃ᵢ`;
- `E[ηᵢ²sᵢ] = σᵢ² Var ηᵢ + 2Σₖ Λᵢₖ(HΣM)ᵢₖ² + Hᵢᵢ²κᵢ`, since
  `Σₖ Λᵢₖ E ûₖ² = σᵢ²`;
- `E[noiseᵢsᵢ] = Σₖ Hᵢₖ²(σₖ²σᵢ² + Cov(sₖ, sᵢ))`, where
  `Cov s = 2ΛXΛ + Diag(κ)` from the previous proof, which gives
  `σᵢ² Var ηᵢ + 2[(H∘H)ΛXΛ]ᵢᵢ + Hᵢᵢ²κᵢ`.

The `Var ηᵢ` and `κᵢ` terms cancel between the last two. ∎

The cross term vanishes under homoskedasticity, since then
`HΣM = σ²MWP_BM = 0`. The noise term does not: `ΛXΛ = σ⁴Λ` there. The skewness term needs a
per-unit `μ₃ᵢ`, which one observation cannot estimate, so the correction
below leaves it out.

Results (hat basis, smooth and rough variances, normal and `χ²₃` errors,
both signal strengths, R = 1000, at `ρ₀`):
- **The formula is exact.** The simulated linear part matches the prediction
  in all 32 cells (`|z|` ≤ 2.0). The noise term is the large one: up to
  −0.275 of the linear part at K/n = 0.25 with smooth variances and a weak
  signal. The skewness term is at most 0.072 and the cross term at most
  0.025.
- **The full plug-in correction** subtracts both parts' bias formulas with
  `Σ` replaced by `Diag(s)`. At K/n ≤ 0.19 it gives E V̂/V of 0.984–1.013
  (`|z|` ≤ 2.1), against plain CJN's 0.960–1.007 (`z` down to −5.6), at no
  visible cost in sd.
- **At K/n = 0.25 it makes things worse:** 0.789–0.982, against plain CJN's
  0.839–0.992. The plug-ins carry second-order bias there, where the
  leverage reaches 0.94.

**Coverage with the corrected CJN (`tc_coverage_cjnc.jl`).** The same design,
seeds and estimator as `tc_coverage_vhat.jl`; its oracle, HC2 and plain-CJN
columns repeat that script's in all 32 cells, digit for digit.

| `V̂` | K/n ≤ 0.19, both signals | weak signal, K/n = 0.25 | strong signal, K/n = 0.25 |
|---|---|---|---|
| oracle | 0.945–0.963 | 0.950–0.961 | 0.953–0.964 |
| HC2 (coded) | 0.940–0.966 | 0.956–0.968 | 0.953–0.965 |
| CJN | 0.932–0.957 | 0.911–0.940 | 0.950–0.958 |
| CJN, both parts corrected | 0.935–0.958 | 0.875–0.916 | 0.948–0.956 |

The correction moves coverage by at most 0.5 points where it removes the
bias in `E V̂`, and costs up to 3.6 points, with `V̂ < 0` in up to 36 of 1000
replications, at K/n = 0.25 with a weak signal.

**Step 2's conclusion.** In these designs, the coded HC2 `V̂` is the one to
use. It is consistent when K/n → 0 (Proposition 2), and it covers
0.94–0.97 in every cell measured up to K/n = 0.25, with leverage up to 0.94.
Its upward bias in `E V̂`, up to 21% when the quadratic part dominates and
the variances are rough, costs at most 1.6 points of over-coverage. The
alternatives that remove that bias (leave-one-out, CJN, CJN corrected) do
so at the cost of variance, and under-cover where HC2 does not. The exact
bias formulas for CJN are proved and checked, but they do not give a better
interval. What these designs do not cover: K/n above 0.25, which a basis
with bounded leverage cannot reach at n = 900; other `W`; and `n` other than
900.

## Still to do

1. Theorem 1 and the Corollary are proved (2026-11-19), the Corollary
   under the sign rule for the root and the identification condition A9.
   Proposition 1 (REsd, the two-step) is proved too: the plug-in `ρ̃` costs
   nothing in the limit. Step 1 has no sketched parts left. Proposition 3
   replaces the sketched Remark 1: the approximation-error bias is
   `β = r′(W + D)r + r′r_W`, the second term was missing, and the interval
   is valid if and only if `β = o(√n)` (`tc_approx.jl`). Proposition 4
   replaces Remark 3: the theory carries over to a symmetric penalised
   smoother with `Sᵢᵢ` in place of `hᵢᵢ`, provided the moment is formed as
   `y₀′Ay₀` and not by the idempotent shortcut (`tc_smoother.jl`). Its
   `V̂` is consistent if `tr S/n → 0` and `‖Mf‖² = o(n)` (Proposition 4(c),
   `tc_smoother_vhat.jl`).
2. A5: Proposition 5 derives it from `W` for K/n → 0 and, in its spectral
   form, whenever the weighted overlap `π̄_w` of `B` with the eigenvectors
   of `Ws + D` stays below 1/2 (up to K/n ≈ 0.05–0.10 in `tc_a5.jl`).
   Still open: a condition that keeps `‖P_BXP_B‖_F²`, which the bound drops,
   and so covers the designs where `π̄_w ≥ 1/2` but A5 still holds.
3. Theory step 2: the conjecture is checked numerically (`tc_vhat.jl`),
   and so are Kline, Saggio & Sølvsten's leave-one-out variances
   (`tc_vhat_lo.jl`, `tc_vhat_lo_lin.jl`). Neither is safe everywhere: HC2
   fails when the quadratic part dominates and the variances are rough, and
   leave-one-out fails with a strong signal, even with bounded leverage
   (`tc_vhat_lo_hat.jl`). Leverage 1 rules out any estimator free of `f`.
   CJN is within 2% at K/n ≤ 0.11 with bounded leverage, at both signal
   strengths, and biased down beyond that by an amount that does not depend
   on `f`. Proposition 2 proves the coded HC2 `V̂` consistent for
   K/n → 0 under four moments and A10. Coverage is measured (`tc_coverage_vhat.jl`): HC2 covers
   0.94–0.97 in every cell, and leave-one-out and CJN under-cover at
   K/n = 0.25. CJN's quadratic-part bias has an exact, distribution-free
   formula (`tc_cjn_bias.jl`), and so does its linear part
   (`tc_cjn_lin.jl`). With both corrected, CJN is within 1.6% of `V` at
   K/n ≤ 0.19, and worse than uncorrected at 0.25. Coverage with the
   corrected `V̂` (`tc_coverage_cjnc.jl`) gains at most 0.5 points and
   loses up to 3.6 at K/n = 0.25, so HC2 stays the recommended `V̂`. Step 2
   is closed for these designs; untested are K/n above 0.25, other `W`,
   and other `n`.
4. Proposition 4(c) proves the smoother's `V̂` consistent. Choosing λ is
   measured, not solved (`tc_lambda.jl`): GCV over-smooths (coverage 0.888
   and 0.320), the RMSE-best λ is already too large, and a plug-in `ĉ` is
   too noisy (sd 0.8–2.9) to choose λ. Still open: a data-driven λ (or K)
   with a coverage guarantee, for example one targeting `c` rather than the
   fit of `f`.
5. Reading: de Jong (1987), Bhansali, Giraitis & Kokoszka (2007), Kelejian &
   Prucha (2001) and Hall & Heyde's §3.2 are read. Cattaneo, Jansson & Newey
   (2018) §3 and Kline, Saggio & Sølvsten (2020) §4.2 have been read via
   alphaXiv. Macaev's theorem (1961), von Bahr & Esseen's inequality (1965)
   and Rosenthal's inequality (1970) are used as they are usually stated,
   and have not been read.
6. Theorem 1 now needs `2 + δ` moments (done 2026-11-19). Consistency of
   `V̂` needs four (Proposition 2); eighth moments would be needed only for
   a rate or for the variance of `V̂`.

## References

All have been checked to exist (via Scite, or by resolving the DOI), except
Macaev (1961). Read for this note: de Jong (1987), §2 and §5, from the page
images of Springer's free copy; Cattaneo, Jansson & Newey (2018), §3 (arXiv
1507.02493); Kline, Saggio & Sølvsten (2020), §4.2 (arXiv 1806.01494); and,
from the page images of the copies in `ExtRef/`, Bhansali, Giraitis &
Kokoszka (2007), all of it; Kelejian & Prucha (2001), §3.1 and the proof of
Theorem 1; Hall & Heyde (1980), §3.2; and Lee (2004)'s CLT, as restated in
Lee (2023), ch. 1, §1.8.3 and Appendix 1A. The others have not been read.
- Bhansali, R. J., Giraitis, L., & Kokoszka, P. S. (2007). Convergence of
  quadratic forms with nonvanishing diagonal. *Statistics & Probability
  Letters*, 77(7), 726–734. doi:10.1016/j.spl.2006.11.007
- Cattaneo, M. D., Jansson, M., & Newey, W. K. (2018). Inference in linear
  regression models with many covariates and heteroscedasticity. *Journal of
  the American Statistical Association*, 113(523), 1350–1361.
  doi:10.1080/01621459.2017.1328360
- de Jong, P. (1987). A central limit theorem for generalized quadratic forms.
  *Probability Theory and Related Fields*, 75, 261–277. doi:10.1007/BF00354037
- Hall, P., & Heyde, C. C. (1980). *Martingale Limit Theory and Its
  Application*. Academic Press. doi:10.1016/C2013-0-10818-5
- Kelejian, H. H., & Prucha, I. R. (2001). On the asymptotic distribution of
  the Moran I test statistic with applications. *Journal of Econometrics*,
  104(2), 219–257. doi:10.1016/S0304-4076(01)00064-1
- Kline, P., Saggio, R., & Sølvsten, M. (2020). Leave-out estimation of
  variance components. *Econometrica*, 88(5), 1859–1898.
  doi:10.3982/ECTA16410
- Lee, L.-F. (2023). Spatial econometric models: model specifications and
  basic statistics. In *Spatial Econometrics: Spatial Autoregressive Models*
  (ch. 1, pp. 1–56). World Scientific. doi:10.1142/9789811270499_0001.
  The book is doi:10.1142/13253; Crossref lists Lee as its only author.
- Lee, L.-F. (2004). Asymptotic distributions of quasi-maximum likelihood
  estimators for spatial autoregressive models. *Econometrica*, 72(6),
  1899–1925. doi:10.1111/j.1468-0262.2004.00558.x
- Rosenthal, H. P. (1970). On the subspaces of Lᵖ (p > 2) spanned by
  sequences of independent random variables. *Israel Journal of
  Mathematics*, 8(3), 273–303. doi:10.1007/BF02771562
- von Bahr, B., & Esseen, C.-G. (1965). Inequalities for the rth absolute
  moment of a sum of random variables, 1 ≤ r ≤ 2. *The Annals of
  Mathematical Statistics*, 36(1), 299–303. doi:10.1214/aoms/1177700291
- Macaev, V. I. (1961). A class of completely continuous operators. *Doklady
  Akademii Nauk SSSR*, 139, 548–551 (in Russian). Cited from Bhansali et al.;
  not checked or read.
- Kwapień, S., & Pełczyński, A. (1970). The main triangle projection in matrix
  spaces and its applications. *Studia Mathematica*, 34, 43–67.
  doi:10.4064/sm-34-1-43-67

Read, and held in `ExtRef/`: Jin & Lee (2012); Lu, Tian & Yang (2024).
