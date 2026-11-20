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
Revised on 2026-11-20 after a review: a gap in
Proposition 2, step 3, is closed (the `ζ` part of `b̂`), the steps it
asked to be explicit are written out, the meaning of `H` is fixed, and the
sign of `J` is measured (`tc_sign.jl`), with a check in `root_rho`. Later
the same day: Lemma 4 bounds the smoother's bias over a roughness class, and
Proposition 7 proves the coverage of a bias-aware interval and of a λ chosen
by its length (`tc_rbc.jl`, `tc_honest.jl`); "Choosing ℛ" says how to set
its bound and why the data cannot (`tc_rchoice.jl`). Then Lemma 5 and
Proposition 7′ extend both to `f ∉ span(B)`, with the class on a finer
reference basis (`tc_refbasis.jl`).
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

## Notation

Each symbol below has one meaning throughout. A symbol used only inside one
proof is defined there and is not used elsewhere.

**Quotes use the notation of their source.** A theorem or formula quoted
from a paper keeps that paper's words and symbols exactly, as read from its
page images, even where a symbol means something else in this note. A line
after the quote maps its symbols to this note's. The paragraphs that compare
this note with other papers (Kelejian & Prucha, Lee, Bhansali et al., de
Jong, Hall & Heyde) also keep those papers' symbols, inside that paragraph
only.

`i, j, k, l` are indices, and `c` is a generic constant that may change from
line to line; named constants carry a subscript (`c_W`, `c_G`, `c_L`, `c_𝔭`, `Θ_q`).

| Symbol | Meaning |
|---|---|
| `ε`, `σᵢ²`, `Σ`, `σ²` | the errors; their variances; `Diag(σᵢ²)`; the vector `(σᵢ²)ᵢ` |
| `δ`, `p = 1 + δ/2`, `q = 2 + δ` | the moment exponent of A1, and the two exponents derived from it |
| `μ̄`, `μ₄`, `μ₃ᵢ`, `κᵢ` | moment bounds; third moments; fourth cumulants |
| `hᵢᵢ`, `h̄`, `M`, `P_B` | leverages and their bound; `I − P_B`; the projection on `B` |
| `S`, `s̄`, `λ` | a smoother (Proposition 4), its largest `Sᵢᵢ`, its penalty |
| `C`, `Cs`, `γᵢⱼ`, `Ω`, `L` | the moment matrix; its symmetric part and that part's entries; `Cs∘Cs`; the strictly lower part of `Cs` |
| `P = W′ − Δ`, `Δ` | Jin & Lee's first-step matrix and its diagonal correction |
| `d`, `D = Diag(d)`, `d̄`, `A = MW + MDM` | the exact-diagonal correction, its largest entry, and the exact-diagonal moment matrix (there `C = A`) |
| `G`, `G(ϱ)`, `C(ϱ)` | `W(I − ρ₀W)⁻¹`; the same, and the moment matrix, at a plug-in value `ϱ` (Proposition 1) |
| `t = ρ − ρ₀`, `g₀`, `g₁`, `g₂`, `Γ` | the deviation from `ρ₀`; the coefficients of `g(ρ₀ + t)`; the discriminant `g₁² − 4g₂g₀` |
| `η` | `f + ε = y₀(ρ₀)` |
| `b`, `b̂`, `ζ`, `H` | the linear part's coefficient; its estimate; that estimate's noise `ζ = Hε`, with `H = (C + C′)P_B` for a projection and `H = (A + A′)S` for a smoother (Proposition 4). For the exact-diagonal `A = MW + MDM` with a projection, `MP_B = 0` makes `H = MWP_B` |
| `V`, `V̂`, `J` | the moment's variance; its estimate; `−E g′(ρ₀)` |
| `sᵢ`, `Σ̂`, `νᵢ`, `ϑᵢ` | the plug-in variance; `Diag(s)`; its remainder `sᵢ − εᵢ²`; the leverage noise in `ûᵢ` |
| `û`, `û₀`, `û₁` | `My₀(ρ̂)`; `My`; `MWy` |
| `r = Mf`, `r_W`, `β`, `c_β` | the approximation (or shrinkage) remainder; its lagged part; the bias `f′Cf`; its size `β/√V` |
| `Π`, `ψ`, `ℛ`, `F(ℛ)` | Lemma 4 and Proposition 7: a smoother's penalty; the coefficients of `f = Bψ`; a roughness bound; the class `{Bψ : ψ′Πψ ≤ ℛ²}` |
| `v`, `T_λ`, `χ̄_λ`, `b̄`, `t̄`, `cv(t)`, `Υ_λ`, `𝒢` | `B′M_λWf`; `B′M_λW`; the quadratic part's bound; the bias bound; `b̄/√V`; the `1 − α` quantile of `\|N(t, 1)\|`; `Π^{+1/2}T_λP_B`; the grid of λ |
| `B̃`, `Π̃`, `ψ̃`, `ℛ̃`, `F̃(ℛ̃)`, `f_B` | Lemma 5: a finer reference basis and its penalty; the coefficients of `f = B̃ψ̃`; its roughness bound; the class; `P_Bf` |
| `ṽ`, `Q̃`, `χ̃`, `b̃`, `T̃`, `Υ̃`, `E` | Lemma 5: the linear coefficient; the quadratic matrix; its bound; the bias bound; `ṽ = T̃f_B`; `Π̃^{+1/2}T̃P_B`; `G(ϱ)′ − Δ(ϱ)` |
| `θ` | the limit of `K/n` |
| `κ_W`, `n_W`, `X`, `ωⱼ`, `φⱼ`, `πⱼ` | Proposition 5: `‖W‖_F² ≥ κ_W n`; neighbours per row; `Ws + D`; its eigenvalues and eigenvectors; their overlaps with `B` |
| `Y` | a generic matrix (Lemmas 2 and 3, and the bound after Lemma 3) |
| `Z` | a standard normal variable |

In the numerical checks, `a` is the amplitude of `f`, `k` the knots per side
of a hat basis, `(u, v)` the locations, `R` the replications, and `z` either
the standardised moment that the check defines or a Monte Carlo z-score; each
check says which.

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
- `V := Var g(ρ₀) = 2σ²′(Cs∘Cs)σ² + Σᵢ bᵢ²σᵢ²`, where `σ²` is the vector of
  variances. The zero diagonal removes every third- and fourth-moment term,
  including the covariance between the linear and quadratic parts, because
  `E εᵢεⱼεₖ ≠ 0` only when `i = j = k`.

`tc_jinlee.jl` checks this exact variance against simulation.

## Lemma 2: the moment matrix is bounded although M is dense (proved)

(a) `‖M‖_op = 1`, so `‖C‖_op ≤ ‖P‖_op ≤ c_W + maxᵢ|Δᵢᵢ|`, and
`‖Cs‖_op ≤ ‖C‖_op`.

(b) For any `n × n` matrix `Y`, `|(Y′M)ᵢᵢ| = |(Yeᵢ)′(Meᵢ)| ≤ ‖Y‖_op ‖Meᵢ‖`,
and `‖Meᵢ‖² = Mᵢᵢ = 1 − hᵢᵢ`. So `Δ = Diag(Y′M)Diag(M)⁻¹` satisfies

    |Δᵢᵢ| ≤ ‖Y‖_op / √(1 − hᵢᵢ),

which is bounded under A3. This covers REsd's `Y = G(ϱ)`, whose diagonal is
not zero. When `diag(Y) = 0`, as for `Y = W`, there is a finer bound:
`(Y′M)ᵢᵢ = −(Y′P_B)ᵢᵢ`, and Cauchy–Schwarz gives

    |Δᵢᵢ| ≤ ‖Yeᵢ‖ √hᵢᵢ / (1 − hᵢᵢ),

which tends to 0 when `maxᵢ hᵢᵢ → 0`.

(c) With A2 and A3, `‖C‖_op = O(1)`, and with A5,
`‖Cs‖_op / ‖Cs‖_F = O(n^{−1/2})`.

The point of the lemma: Kelejian & Prucha's (2001) CLT asks for bounded row
and column sums of the moment matrix. A dense `M` has row sums of order up to
`√(n hᵢᵢ)`, which breaks that. But the CLT argument below only needs norm
ratios, and the operator norm survives the projection. In `tc_clt.jl`,
`‖Cs‖_op` stays at 0.83–1.08 and the bound in (b) holds in every design.

## Lemma 3: the triangular part (proved)

Let `L` be the strictly lower-triangular part of a symmetric `Y`. Then
`‖L‖_op ≤ ⌈log₂ n⌉ ‖Y‖_op`.

*Proof.* Pad the index set to `2^𝔪` with `𝔪 = ⌈log₂ n⌉`. At level
`k = 1, …, 𝔪`, cut the indices into consecutive blocks of size `2^{𝔪−k+1}`,
and each block into two halves. Let `L_k` hold the entries of `Y` whose row is
in the second half and whose column is in the first half of the same block.
Each pair `i > j` belongs to exactly one level: the finest one at which `i`
and `j` share a block but not a half. So `L = Σ_k L_k`. Within one level the
blocks have disjoint rows and disjoint columns, so `‖L_k‖_op` is the largest
operator norm of its sub-blocks, and no sub-block of `Y` has a norm above
`‖Y‖_op`. ∎

The log factor is known to be the right order for triangular truncation
(Kwapień & Pełczyński 1970; not read), and `tc_schatten.jl` shows it: for an
interleaved Hilbert matrix with `‖Y‖_op < π`, `‖L‖_op/‖Y‖_op` grows from 1.90
to 2.77 as n goes from 250 to 4000. In the moment's designs the ratio is only
0.67–0.69.

**Theorem 1 no longer uses this lemma.** In the Schatten-4 norm
`‖Y‖₄ = (Σₖ ςₖ⁴)^{1/4}` (`ςₖ` the singular values), triangular truncation is
bounded with a constant free of n: `‖L‖₄ ≤ c_L‖Y‖₄` (Matsaev 1961b,
Theorem 1, read; used by Bhansali, Giraitis & Kokoszka 2007 in their Lemma
2.1, who cite Matsaev 1961a instead). With
`‖Y‖₄⁴ = ‖Y²‖_F² ≤ ‖Y‖_op²‖Y‖_F²` this gives

    ‖L′L‖_F = ‖L‖₄² ≤ c_L²‖Y‖_op‖Y‖_F,     ‖L‖_op² ≤ ‖L‖₄² ≤ c_L²‖Y‖_op‖Y‖_F.

In `tc_schatten.jl`, `‖L‖₄/‖Y‖₄` stays at 0.82–0.84 for the Hilbert matrix
while its operator-norm ratio grows, and at 0.70–0.71 for the moment's
matrices, n = 400 to 2500. The lemma is kept because its bound is elementary
and gives the operator norm at the full rate, which the Schatten bound does
not.

**What Matsaev proves, and how it gives the bound** (read 2026-11-20 from the
page images of both notes, which are in Russian). Matsaev (1961b), p. 811,
verbatim, with his symbols:

> Т е о р е м а 1. *Пусть A = G + iH (G = ½(A + A\*)) — вольтерров
> оператор. Если H ∈ 𝔖_p (1 < p < ∞), то G ∈ 𝔖_p и*
>
>     |G|_p ≤ C_p |H|_p,     (2)
>
> *где C_p — некоторая константа, зависящая лишь от p. При этом, если под
> C_p понимать наименьшую в (2) константу, то*
>
>     C_p = C_q   (p⁻¹ + q⁻¹ = 1).

(Translation: "Theorem 1. Let A = G + iH (G = ½(A + A\*)) be a Volterra
operator. If H ∈ 𝔖_p (1 < p < ∞), then G ∈ 𝔖_p and |G|_p ≤ C_p|H|_p (2),
where C_p is a constant that depends only on p. Moreover, if C_p is taken
to be the smallest constant in (2), then C_p = C_q (p⁻¹ + q⁻¹ = 1)." The
printed `C_p` in the last display carries a stray mark above the `p`, a
printing defect.) A Volterra operator is a compact operator whose spectrum
is `{0}`.

His `A`, `G`, `H`, `p`, `q`, `C_p` clash with this note's symbols, so the
derivation below writes them as `T`, `T_R`, `T_I`, `𝔭`, `𝔭′`, `c_𝔭`. In finite
dimension a strictly lower triangular matrix is nilpotent, so it is a
Volterra operator, and the theorem applies with no condition on the chain.
Take `Y` real symmetric with a zero diagonal and `L` its strictly lower
part, and set `T = 2iL`. Then `T_I = L + L′ = Y` and `T_R = i(L − L′)`, so
`‖L − L′‖_𝔭 ≤ c_𝔭‖Y‖_𝔭`, and `L = (Y + (L − L′))/2` gives
`‖L‖_𝔭 ≤ (1 + c_𝔭)/2 · ‖Y‖_𝔭`. A nonzero diagonal `Diag(Y)` costs a factor of 2
at most: taking the diagonal (a pinching) does not increase a Schatten norm,
so `‖Y − Diag(Y)‖_𝔭 ≤ 2‖Y‖_𝔭`. So the `c_L` above is at most `1 + c_𝔭` at
`𝔭 = 4`, free of n.

Two limits. (1) The note is a Doklady announcement, and it outlines its
proofs. For `1 < p < 2` the outline uses entire functions; for `p = 2` the
result is Sakhnovich's; for `p > 2`, the case used here, the note says
(p. 811): "При p > 2 теорему 1 и равенство C_p = C_q (p⁻¹ + q⁻¹ = 1)
можно вывести из случая 1 < p < 2 с помощью общей теоремы 2 из (¹)"
("For p > 2, Theorem 1 and the equality C_p = C_q can be derived from the
case 1 < p < 2 by means of the general Theorem 2 of (¹)"), where (¹) is
Gohberg & Krein (1961). That theorem was read on 2026-11-20 (p. 1036,
verbatim, with their symbols):

> Т е о р е м а 2. *Пусть 𝔖_I — одно из пространств 𝔖_p (1 < p < ∞),
> 𝔖_Π^(0), 𝔖_π, а 𝔖_II — одно из пространств 𝔖_p (1 < p ≤ ∞), 𝔖_Π^(0).
> Если пространство 𝔖_I ⊂ 𝔇 и отображается оператором 𝒥 непрерывно в
> пространство 𝔖_II, то 𝔖\*_II ⊂ 𝔇 и отображается непрерывно оператором
> 𝒥 в пространство 𝔖\*_I. Обратно, если пространство 𝔖\*_II ⊂ 𝔇
> отображается оператором 𝒥 непрерывно в пространство 𝔖\*_I и выполняется
> дополнительное условие 𝒥(𝔎) ⊂ 𝔖_II, то 𝔖_I ⊂ 𝔇 и отображается
> непрерывно оператором 𝒥 в пространство 𝔖_II.*
>
> *В обоих случаях из двух операторов 𝒥(𝔖_I → 𝔖_II) и 𝒥(𝔖\*_II → 𝔖\*_I),
> порожденных оператором 𝒥 при этих отображениях, второй является
> сопряженным к первому.*
>
> *В случае 𝔖_I = 𝔖_1 теорема остается справедливой, если в ее
> формулировке пространство 𝔖\*_I заменить на 𝔖_∞.*
>
> Точно такая же теорема имеет место для оператора 𝒮.

(Translation: "Theorem 2. Let 𝔖_I be one of the spaces 𝔖_p (1 < p < ∞),
𝔖_Π^(0), 𝔖_π, and 𝔖_II one of the spaces 𝔖_p (1 < p ≤ ∞), 𝔖_Π^(0). If the
space 𝔖_I ⊂ 𝔇 and is mapped by the operator 𝒥 continuously into the space
𝔖_II, then 𝔖\*_II ⊂ 𝔇 and is mapped continuously by 𝒥 into the space 𝔖\*_I.
Conversely, if the space 𝔖\*_II ⊂ 𝔇 is mapped by 𝒥 continuously into the
space 𝔖\*_I and the additional condition 𝒥(𝔎) ⊂ 𝔖_II holds, then 𝔖_I ⊂ 𝔇
and is mapped continuously by 𝒥 into the space 𝔖_II. In both cases, of the
two operators 𝒥(𝔖_I → 𝔖_II) and 𝒥(𝔖\*_II → 𝔖\*_I) generated by 𝒥 under
these maps, the second is the adjoint of the first. In the case 𝔖_I = 𝔖_1
the theorem remains valid if 𝔖\*_I is replaced by 𝔖_∞ in its statement.
Exactly the same theorem holds for the operator 𝒮.") Here
`𝒥(X) = ∫_𝔓 PX dP` and `𝒮(X) = i∫_𝔓 (PX dP − dP XP)`, over a continuous chain
`𝔓` of orthogonal projections `P`, and `𝔇` is their common domain (p. 1036).
With `𝔖_I = 𝔖_II = 𝔖_𝔭′` and `(𝔖_𝔭′)* = 𝔖_𝔭`, the bound for `𝔭′ = 4/3` gives
the bound for `𝔭 = 4` with the same constant.
Gohberg & Krein state it without proof, but in finite dimension it is
elementary: the strictly lower truncation is an orthogonal projection on the
entries, so it is self-adjoint in the trace pairing `⟨Y₁, Y₂⟩ = tr(Y₁′Y₂)`,
and Schatten duality gives it the same norm on `𝔖_𝔭` as on `𝔖_𝔭′`. So the
step from `𝔭 < 2` to `𝔭 = 4` needs nothing unread. (2) Matsaev (1961a), the
note Bhansali et al. cite, is about a different space. Its Theorem 1 bounds
the triangular truncation from the Matsaev ideal `𝔖_ω`
(`Σₖ ςₖ/(2k − 1) < ∞`) into the compact operators, with the sharp constant
`2/π`. It mentions `𝔖_𝔭` only in a remark, crediting the convergence of the
truncation integral, when the imaginary part is in a Schatten class of
index above 1, to Gohberg & Krein. So the
result this note uses is in 1961b, not in 1961a.

## Theorem 1: CLT for the moment (proved, given Hall & Heyde's Corollary 3.1, Matsaev's theorem and two moment inequalities)

Assume A1, so the errors need only `2 + δ` moments. Write `σ̲, σ̄` for the
variance bounds. If

- (a) `‖Cs‖_op / ‖Cs‖_F → 0`, and
- (c′) `maxᵢ bᵢ² / V → 0`,

then `g(ρ₀)/√V → N(0, 1)`. Under A2–A5, (a) holds at rate `n^{−1/2}` by
Lemma 2. (The first version of this theorem needed
`(log n)²‖Cs‖_op²/‖Cs‖_F² → 0`, from Lemma 3, and fourth moments.)
(c′) is implied by A6, and it is weaker than A6 when the quadratic
part dominates `V`.

*Proof.* Write `γᵢⱼ` for the entries of `Cs`, which is symmetric with zero diagonal, and
`wᵢ = Σ_{j<i} γᵢⱼεⱼ`, so that `w = Lε`. Then `ε′Cε = 2Σᵢ εᵢwᵢ`, and
`g = Σᵢ ξᵢ` with `ξᵢ = εᵢ(bᵢ + 2wᵢ)`. Since `wᵢ` depends only on
`ε₁, …, εᵢ₋₁`, the `ξᵢ` are martingale differences for
`Fᵢ = σ(ε₁, …, εᵢ)`. Also

    V = Σᵢ σᵢ²bᵢ² + 2Σ_{i≠j} σᵢ²σⱼ²γᵢⱼ²  ≥  max(σ̲²‖b‖², 2σ̲⁴‖Cs‖_F²).

Hall & Heyde's (1980) Corollary 3.1 (p. 58, read from the page image) needs
the two conditions below, with limit 1 (their `η² = 1`). Its nesting condition
(3.21) is not needed. Their Remarks on p. 59 say: "It is possible to replace
(3.21) by a measurability condition on η²", provided "the stability is
dropped", and "if η² is constant, then it is trivially measurable". Stability
is not needed here, so the errors may form a triangular array.

1. **Conditional variance.** `𝒱 = Σᵢ E(ξᵢ²|Fᵢ₋₁) = Σᵢ σᵢ²(bᵢ + 2wᵢ)²`, with
   `E 𝒱 = V`, and `𝒱 − V = 4𝒜 + 4(ℬ − E ℬ)`, where:
   - `𝒜 = Σᵢσᵢ²bᵢwᵢ = (L′Σb)′ε`. So
     `Var 𝒜 ≤ σ̄²‖L′Σb‖² ≤ σ̄⁶‖L‖_op²‖b‖²`. By the Schatten-4 bound after
     Lemma 3, `‖L‖_op² ≤ c_L²‖Cs‖_op‖Cs‖_F`. Since
     `V² ≥ 8σ̲⁶‖b‖²‖Cs‖_F²` (from `(x + y)² ≥ 4xy`),
     `Var 𝒜/V² ≤ c‖Cs‖_op/‖Cs‖_F`.
   - `ℬ = w′Σw = ε′Φε` with `Φ = L′ΣL`, which is symmetric and positive
     semidefinite. Split it as `ℬ − E ℬ = ℬ₁ + ℬ₂`, with
     `ℬ₁ = Σ_{j≠k} Φⱼₖεⱼεₖ` and `ℬ₂ = Σⱼ Φⱼⱼ(εⱼ² − σⱼ²)`.
     - **`ℬ₁` needs only second moments.** Its terms are uncorrelated except
       in the pairs `{j, k} = {l, m}`, so `E ℬ₁² = 2Σ_{j≠k} Φⱼₖ²σⱼ²σₖ² ≤
       2σ̄⁴‖Φ‖_F²`. Since `0 ≼ L′ΣL ≼ σ̄²L′L`,
       `‖Φ‖_F ≤ σ̄²‖L′L‖_F ≤ σ̄²c_L²‖Cs‖_op‖Cs‖_F`. With `V ≥ 2σ̲⁴‖Cs‖_F²`,
       `E ℬ₁²/V² ≤ c‖Cs‖_op²/‖Cs‖_F²`.
     - **`ℬ₂` is a sum of independent, mean-zero terms with `p` moments.**
       By von Bahr & Esseen's (1965) inequality, for `1 ≤ p ≤ 2`,
       `E|Σⱼ Ξⱼ|^p ≤ 2Σⱼ E|Ξⱼ|^p`. Here
       `E|εⱼ² − σⱼ²|^p ≤ 2^{p−1}(E|εⱼ|^{2p} + σⱼ^{2p}) ≤ 2^p μ̄`, by
       Jensen's inequality for the second term. So
       `E|ℬ₂|^p ≤ 2^{p+1}μ̄ Σⱼ Φⱼⱼ^p`. Each `Φⱼⱼ = Σᵢσᵢ²Lᵢⱼ² ≤
       σ̄²‖Cs eⱼ‖² ≤ σ̄²‖Cs‖_op²`, and `Σⱼ Φⱼⱼ = tr Φ ≤ σ̄²‖L‖_F² =
       σ̄²‖Cs‖_F²/2`. So `Σⱼ Φⱼⱼ^p ≤ (maxⱼ Φⱼⱼ)^{p−1} Σⱼ Φⱼⱼ ≤
       σ̄^{2p}‖Cs‖_op^{2(p−1)}‖Cs‖_F²/2`, and with `V^p ≥
       (2σ̲⁴)^p‖Cs‖_F^{2p}`,
       `E|ℬ₂|^p/V^p ≤ c (‖Cs‖_op²/‖Cs‖_F²)^{p−1} = c (‖Cs‖_op/‖Cs‖_F)^δ`.

   The constants depend only on `σ̲`, `σ̄`, `μ̄`, `δ` and `c_L`. Every bound
   tends to 0 by (a), so `𝒜/V`, `ℬ₁/V` and `ℬ₂/V` tend to 0 in probability
   (Chebyshev's and Markov's inequalities), and `𝒱/V → 1` in probability.
   The earlier proof used fourth moments here, because `Var(ε′Φε)` contains
   `Σⱼ Φⱼⱼ² Var(εⱼ²)`, and in the Lyapunov step below, which was of order 4.
2. **Lyapunov of order `q = 2 + δ`, which implies the conditional
   Lindeberg condition.** For `υᵢ = ξᵢ/√V` and any `τ > 0`,
   `E Σᵢ E[υᵢ² 1{|υᵢ| > τ} | Fᵢ₋₁] = Σᵢ E[υᵢ² 1{|υᵢ| > τ}] ≤
   τ^{−δ} Σᵢ E|ξᵢ|^q / V^{q/2}`, so if the right side tends to 0 the
   conditional sum tends to 0 in `L¹`, hence in probability.

   `wᵢ = Σ_{j<i} γᵢⱼεⱼ` is `Fᵢ₋₁`-measurable, and under A1 `εᵢ` is
   independent of `Fᵢ₋₁`. So `E[|ξᵢ|^q | Fᵢ₋₁] = |bᵢ + 2wᵢ|^q E|εᵢ|^q` a.s.,
   and taking expectations,
   `E|ξᵢ|^q = E|εᵢ|^q · E|bᵢ + 2wᵢ|^q ≤ μ̄ 2^{q−1}(|bᵢ|^q + 2^q E|wᵢ|^q)`.
   By Rosenthal's (1970) inequality for the independent, mean-zero terms of
   `wᵢ = Σ_{j<i} γᵢⱼεⱼ`, with `ℓᵢ² = Σ_{j<i} γᵢⱼ²`,

       E|wᵢ|^q ≤ Θ_q [Σⱼ |γᵢⱼ|^q E|εⱼ|^q + (Σⱼ γᵢⱼ²σⱼ²)^{q/2}] ≤ Θ_q (μ̄ + σ̄^q) ℓᵢ^q,

   because `Σⱼ|γᵢⱼ|^q = Σⱼ(γᵢⱼ²)^{q/2} ≤ (Σⱼγᵢⱼ²)^{q/2}` (the ℓ_{q/2} norm of
   the non-negative `γᵢⱼ²` is at most their ℓ₁ norm, as `q/2 > 1`), and
   `σⱼ^q = (σⱼ²)^{q/2} ≤ σ̄^q`. Rosenthal's inequality is used here for
   independent, mean-zero, not identically distributed summands. That is
   the form usually quoted, but the paper has not been read, so its exact
   conditions and constant are unchecked ("Still to do", item 5).
   Then:
   - `Σᵢ|bᵢ|^q ≤ (maxᵢbᵢ²)^{δ/2}‖b‖² ≤ (maxᵢbᵢ²)^{δ/2} V/σ̲²`, so this part
     of the ratio is at most `c (maxᵢbᵢ²/V)^{δ/2}`;
   - `Σᵢ ℓᵢ^q ≤ (maxᵢ ℓᵢ²)^{δ/2} Σᵢ ℓᵢ² ≤ ‖Cs‖_op^δ ‖Cs‖_F²/2`, because a
     row norm is at most the operator norm, and with
     `V^{q/2} ≥ (2σ̲⁴)^{q/2}‖Cs‖_F^q` this part is at most
     `c (‖Cs‖_op/‖Cs‖_F)^δ`.

   So `Σᵢ E|ξᵢ|^q/V^{q/2} ≤ c₂[(maxᵢbᵢ²/V)^{δ/2} + (‖Cs‖_op/‖Cs‖_F)^δ]`,
   which tends to 0 by (c′) and (a). ∎

**The cost of fewer moments is the rate, not the conditions.** Every bound
above is a power of the same two ratios as before, `‖Cs‖_op/‖Cs‖_F` and
`maxᵢbᵢ²/V`. The bounds that used fourth moments (`ℬ₂` and Lyapunov) now
have exponents `δ` and `δ/2` where they had 2 and 1. So with
`δ` small the normal approximation should improve slowly in n, and
`tc_clt_moments.jl` shows that (see the numerical check below).

**Against Kelejian & Prucha (2001)** (Theorem 1, Assumptions 1–3 and the
proof in their appendix, read 2026-11-19 from the page images; the quotes
below re-checked against them on 2026-11-20). Their CLT covers the same
linear-quadratic form, `Q_n = ε′_nA_nε_n + b′_nε_n` (their (3.1)), with
independent, heteroskedastic errors, and they too use a martingale CLT
(their Theorem A.1, from Gänssler & Stute 1977, a special case of Hall &
Heyde's Corollary 3.1). Their `A_n` (symmetric), `b_n`, `ε_{i,n}` and
`σ²_{Q_n}` are this note's `Cs`, `b`, `εᵢ` and `V`. Their conditions differ
from ours in three places:
- Assumption 2 needs bounded absolute column sums,
  "sup_{1≤j≤n,n≥1} Σ_{i=1}^n |a_{ij,n}| < ∞", which a dense `M` breaks;
  Theorem 1 replaces this with (a);
- Theorem 1 assumes "n⁻¹σ²_{Q_n} ≥ c for some c > 0", the rate `n`; our
  Theorem 1 needs no rate;
- Assumption 3 needs only "sup_{1≤i≤n,n≥1} E|ε_{i,n}|^{2+η₂} < ∞ for some
  η₂ > 0 and a_{ii,n} = 0" (part (a)), and `4 + η₂` moments otherwise (part
  (b)); Theorem 1 now matches them, with `2 + δ` (the zero diagonal is what
  allows it: a nonzero `Cᵢᵢ` would bring `Σᵢ Cᵢᵢ(εᵢ² − σᵢ²)` into `g`, whose
  variance needs fourth moments). No truncation is needed, because the zero
  diagonal leaves the `εᵢ²` only inside the conditional variance, where a
  law of large numbers with `p` moments is enough.

**Against Lee (2004)** (as restated, with its proof, in Appendix 1A of
chapter 1 of Lee 2023, held in `ExtRef/`; quotes checked against the page
image of p. 36 on 2026-11-20). Lemma 1A.1 extends Kelejian & Prucha to
large groups. It assumes that "{A_n} is a sequence of symmetric matrices
with row and column sums uniformly bounded", that "the elements of A_n are
of uniform order O(1/h_n)", that "the variance σ²_{Q_n} is O(n/h_n)", that
"the moment E(|v|^{4+2δ}) for some δ > 0 of v exists", for i.i.d. `v`, and
concludes: "If lim_{n→∞} h_n^{1+2/δ}/n = 0, then Q_n/σ_{Q_n} →D N(0, 1)".
(Lee's `δ` is his own moment exponent, not this note's.) The bounded row and
column sums still exclude a dense `M`.

**What Bhansali, Giraitis & Kokoszka (2007) add** (read 2026-11-19, from the
page images; the quotes below re-checked against them on 2026-11-20). Their
Theorem 2.1 is for a quadratic form alone, `T_n = Σ_{t,k=1}^n a_{n;tk}Z_tZ_k`
in i.i.d. `Z_t` with `EZ_k = 0`, `EZ_k² = 1`, and a nonzero diagonal is
allowed. In their notation `‖A_n‖` is the Euclidean (Frobenius) norm,
`‖A_n‖_sp` the spectral norm and `Ã_n` the strictly lower-triangular part of
`A_n`. Under "EZ_t⁴ < ∞" it needs

    Δ_n = (‖Ã′_nÃ_n‖ + max_{t=1,…,n} Σ_{s=1}^n a²_{ts}) / ‖A_n‖² → 0     (their (2.1)–(2.2)),

and their Lemma 2.1 shows "Δ_n ≤ C‖A_n‖_sp/‖A_n‖" (2.9), so `Δ_n → 0`
follows from our `‖Cs‖_op/‖Cs‖_F → 0`. Part (iii) needs only
"EZ_t^{2+δ} < ∞ (for some δ > 0) and Σ_{t=1}^n a²_{n;tt} = o(‖A_n‖²)" (2.6).
It does not cover a linear part or heteroskedastic errors, so it does not
contain Theorem 1. Its Lemma 2.1, in the form (2.8),
"‖Ã′_nÃ_n‖ ≤ C‖A_n‖_sp‖A_n‖", is what removes the log factor here.

**What de Jong (1987) adds** (read 2026-11-19, from the page images; the
quotes below re-checked against them on 2026-11-20). His Theorem 5.2 (p. 275)
treats the quadratic part alone, "W(n) = Σ_{1≤i≠j≤n} a_ij X_i X_j", "a
quadratic form in independent random variables X_i (EX_i = 0, EX_i² = 1),
with μ_1, …, μ_n the eigenvalues of the symmetric matrix (a_ij), with
vanishing diagonal elements: a_ii = 0 for all i". It concludes
"σ(n)⁻¹ W(n) →d N(0, 1)" under three conditions, for "a sequence of real
numbers K(n)":
- 1) "K(n)⁴ σ(n)⁻² max_{1≤i≤n} Σ_{1≤j≤n} a_ij² → 0";
- 2) "max_{1≤i≤n} EX_i² 1{|X_i| > K(n)} → 0";
- 3) "σ(n)⁻² max_{1≤i≤n} μ_i² → 0".

With `Xᵢ = εᵢ/σᵢ` and `aᵢⱼ = σᵢσⱼγᵢⱼ`, condition 3) is our condition (a),
and 1)–2) hold with uniformly bounded `2 + δ`
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
four moments); for K/n → θ > 0 it is open.

*Proof.* Write `η = f + ε = y₀(ρ₀)` and `t = ρ − ρ₀`. Since `y = (I − ρ₀W)⁻¹η`,
`Wy = Gη`, and `y₀(ρ) = η − tGη`. So `g` is exactly a quadratic in `t`:

    g(ρ₀ + t) = g₀ + g₁t + g₂t²,   g₀ = g(ρ₀),   g₁ = g′(ρ₀) = −2η′G′Csη,   g₂ = η′G′CsGη.

(The code writes `g(ρ) = c − bρ + aρ²` in `ρ`, so `g′(ρ) = −(b − 2ρa)`; those are the code's scalars, not `g₂`, the vector `b` or a constant.)

1. **Orders.** Under A2, A3 and A7, `‖Cs‖_op` and `‖G‖_op` are bounded
   (Lemma 2), and `‖Cs‖_F² ≤ n‖Cs‖_op²`.
   - `g₀ = O_p(√n)`: Theorem 1 gives `g₀ = O_p(√V)`, and
     `V ≤ 2σ̄⁴‖Cs‖_F² + σ̄²‖C′f‖² = O(n)` by A8.
   - `g₂ = O_p(n)`: `|g₂| ≤ ‖Cs‖_op‖G‖_op²‖η‖²`, and `E‖η‖² = ‖f‖² + Σσᵢ² = O(n)`,
     so Markov's inequality applies. Only second moments are used.
   - `g₁ = −J + o_p(n)`. Since `E ε = 0`, `E g₁ = −2[f′G′Csf + tr(G′CsΣ)] = −J`.
     With `Ψ = (G′Cs + CsG)/2`, which has `‖Ψ‖_op ≤ c_G‖Cs‖_op` and
     `‖Ψ‖_F ≤ c_G‖Cs‖_F = O(√n)`,
     `g₁ + J = −2[2f′Ψε + Σ_{i≠j}Ψᵢⱼεᵢεⱼ + Σᵢ Ψᵢᵢ(εᵢ² − σᵢ²)]`. The linear
     term has variance at most `4σ̄²‖Ψ‖_op²‖f‖² = O(n)`, and the off-diagonal
     term has second moment at most `2σ̄⁴‖Ψ‖_F² = O(n)`: both are `O_p(√n)`.
     The diagonal term is a sum of independent, mean-zero terms with `p`
     moments, so by von Bahr & Esseen
     `E|Σᵢ Ψᵢᵢ(εᵢ² − σᵢ²)|^p ≤ 2^{p+1}μ̄ Σᵢ|Ψᵢᵢ|^p ≤ 2^{p+1}μ̄ n‖Ψ‖_op^p`. It is
     therefore `O_p(n^{1/p})`, which is `o_p(n)` because `p > 1`. (With fourth
     moments it would be `O_p(√n)`; `o_p(n)` is all that is needed.)
2. **The discriminant and the two roots.** `Γ = g₁² − 4g₂g₀ = J²(1 + o_p(1)) −
   O_p(n)·O_p(√n) = J²(1 + o_p(1))`, because `J² ≥ cn²` by A9. So `Γ > 0`
   with probability → 1. At the two roots of a quadratic, `g′` equals `+√Γ`
   and `−√Γ`, so the sign rule picks exactly one root, `t̂`, at which
   `g′(ρ₀ + t̂) = g₁ + 2g₂t̂ = −√Γ = −J(1 + o_p(1))`.
3. **The chosen root is the consistent one.** From `g(ρ₀ + t̂) = 0`,
   `t̂ (g₁ + g₂t̂) = −g₀`. Write `g₁ + g₂t̂ = [(g₁ + 2g₂t̂) + g₁]/2`; both terms
   are `−J(1 + o_p(1))`, so `g₁ + g₂t̂ = −J(1 + o_p(1))` and

       t̂ = g₀ / J · (1 + o_p(1)) = O_p(√n/n) = O_p(n^{−1/2}).

   This needs no prior consistency argument: the identity holds at the
   chosen root whatever its size, because the derivative condition fixes
   `g₁ + g₂t̂` directly.
4. **The limit.** `J t̂/√V = (g₀/√V)(1 + o_p(1)) → N(0, 1)` by Theorem 1 and
   Slutsky's lemma. And `|g′(ρ̂)| = √Γ = J(1 + o_p(1))`. ∎

**The other root** is `t̃ = g₀/(g₂t̂)` (the product of the roots is `g₀/g₂`), so
`|t̃| = (J/|g₂|)(1 + o_p(1))`, of the order of a constant: it does not
approach `ρ₀`. If `g₂` is near 0 it is far away; it is never a competitor in
the limit.

**The code's rule is not the sign rule.** `cfe_root` keeps the root in
`(−1, 1)` nearest the OLS ratio `û₁′û₀/û₁′û₁` (`û₀ = My`, `û₁ = MWy`), which is not a consistent
estimator of `ρ₀`, so the proof does not cover it. `tc_roots.jl` measures
the difference in the coverage design (`ρ₀ = 0.4`, normal and `t₃` errors,
n = 400 to 2500, R = 1000 each): the discriminant was positive in all 8000
replications, the two rules disagreed once (`t₃`, n = 400), and the other
root was at least 1.37 from `ρ₀` (median about 2). (`tc_roots.jl` is a
research script; the package's `root_rho` uses the sign rule.)

**The sign of `J`.** A9 assumes `J > 0`, and under that assumption the sign
rule is feasible: of the two roots, take the one where `g′ < 0`. The data
do not show the sign of `J` at the roots, since `g′ = ±√Γ` there by
construction. To check it, `g′` must be evaluated at a consistent
preliminary estimate `ρ̌`, for example 2SLS. As `g′` is linear,
`g′(ρ̌) > 0` is the same as `ρ̌` lying nearer the other root. `root_rho`
does this check, with 2SLS on the instruments `[Z, WZ]`, and warns if it
fails.

`J < 0` can happen. `tc_sign.jl` builds 3,000 hub-and-spoke designs
(`n = 100`) from formulas; 22 (0.7%) have `J < 0` for the first-step
moment, all with hubs. In the strongest (`J/n = −24.3`), with the one-step
exact-diagonal moment (the smoother's, here with penalty 0), the sign-rule
root is biased by +0.066 (sd 0.030), and the Wald interval covers 56 of 257
draws. The other root has bias −0.003 (sd 0.016). The check warns in 256 of
300 draws, and `CS` covers 286 of 300. REsd's second step, with
`C(ρ̃) ≈ G(ρ̃)′M`, has `J > 0` there (bias +0.002, Wald 237 of 246, 2
warnings in 300), as its signal part is close to `‖MGf‖² ≥ 0`. On kNN
designs with `J > 0` (`n` 400–900, both moments, 1,000 fits) the check never
warned. Testset 47g fixes the hub design and checks the warning.

**The rate for K.** Nothing in Theorem 1 needs `K/n → 0`. A3 and A5 allow
`K/n → θ ∈ [0, 1)`. This is weaker than the published rate conditions for
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
`β/√V → c_β`, then `J(ρ̂ − ρ₀)/√V → N(c_β, 1)`, so the nominal 95% interval
covers with probability `P(|Z + c_β| ≤ 1.96)`. The interval is valid if and
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
`‖r‖∞ = O(K^{−s_f/d_f})` for smoothness `s_f` in `d_f` dimensions, that is
`K ≫ n^{d_f/(4s_f)}`. The second term is at most `‖r‖‖r_W‖`, and `r_W` need not
be small: a covariate `x` in `B` that is rough in space has a lag `Wx` that
the basis cannot represent. In the worst case this needs
`‖r‖∞ = o(n^{−1/2})`, that is `K ≫ n^{d_f/(2s_f)}`, twice the exponent. When
`r` is smooth and `r_W` is rough, the two are nearly orthogonal and
`r′r_W` is of order `‖r‖∞ √n` instead, which needs only `‖r‖∞ → 0`. So the
binding condition is `β = o(√n)` itself, and it is a **lower** bound on K.
The robust `V̂` should stay consistent (sketched; Proposition 2's steps
are not redone here): `û` gains `r`, so `sᵢ` gains
`(rᵢ² + 2rᵢ(Mε)ᵢ)/Mᵢᵢ`, which adds `O(‖r‖² + ‖r‖√n)` to `V̂`, and that is
`o(n)` when `‖r‖∞ → 0`.

**Numerical check (`tc_approx.jl`).** n = 900, kNN `W`, a tensor hat basis
plus a covariate `x`, `f = a·sin(2πu)cos(πv) + x` with `a` = 1 and 4,
smooth variances, `ρ₀ = 0.4`, R = 1000. `β`, its two terms, `V`, `c_β` and
`J` are exact; the bias and coverage of `ρ̂` (HC2 `V̂`) are simulated.

| a | K | ‖r‖∞ | ‖r_W‖∞ | r′(W+D)r | r′r_W | c_β = β/√V | bias: predicted β/J | simulated | coverage: predicted | simulated |
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
and a = 4 at K = 10, where `c_β = 15.7`.)

- **`r_W` is not small**: `‖r_W‖∞` is 1.0–1.3 at every K, because of the
  covariate. The cross term `r′r_W` is the larger part of `β` at a = 4 and
  K ≥ 101, so the sketched remark's `r′(W+D)r` alone would understate the
  bias there by a factor of 3 to 12. It is still far below `‖r‖‖r_W‖`,
  because a smooth `r` and a rough `r_W` are nearly orthogonal.
- **The prediction holds while `c_β` is moderate.** For `c_β` ≤ 1.1 the
  simulated bias matches `β/J` (`|z|` ≤ 1.9) and the coverage matches
  `P(|Z + c_β| ≤ 1.96)` within 2.4 points. For `c_β` of 2.5 and more, the
  first-order expansion understates the bias (0.116 against 0.107) and
  overstates the coverage (0.211 against 0.309). That is outside the
  local result.
- **What the check cannot show.** Where the two forms of `β` differ
  (a = 4, K ≥ 50), the bias of `ρ̂` is 0.0003–0.015 and the two predictions
  differ by at most 0.005, against a Monte Carlo SE of 0.002–0.003, so the
  simulation cannot separate them (`z` at most 2.6). The
  split itself is exact and needs no simulation.
- **Under-smoothing is the price.** Coverage is within 2 points of 0.95
  once `c_β` ≤ 0.25,
  which here means K ≥ 17 at a = 1 and K ≥ 50 at a = 4: a stronger
  surface needs a larger K. Because the zero diagonal removes the trace
  bias at any K, raising K costs efficiency only (the sd rises from 0.054
  to 0.100 at a = 1 as K goes from 17 to 170), not validity.

**Proposition 1 (REsd, the two-step; proved, given the Corollary).** Write
`C(ϱ) = (G(ϱ)′ − Δ(ϱ))M`, with `G(ϱ) = W(I − ϱW)⁻¹` and
`Δ(ϱ) = Diag(G(ϱ)′M)Diag(M)⁻¹`, so `diag C(ϱ) = 0` for every `ϱ`. Jin &
Lee's REsd takes the first-step root `ρ̃` (with `C₁ = (W′ − Δ)M`) and then
the root `ρ̂` of `y₀(ρ)′C(ρ̃)y₀(ρ)`, by the sign rule. Let `ρ̂*` be the
infeasible root with `C(ρ₀)`. Assume A1–A9, with A5, A6 and A9 holding for both `C₁`
and `C(ρ₀)`, and Theorem 1's conditions (a) and (c′) for `C(ρ₀)`. Then
`ρ̂ − ρ̂* = o_p(n^{−1/2})`, and so `J*(ρ̂ − ρ₀)/√V* → N(0, 1)` with `J*` and
`V*` those of `C(ρ₀)`: the plug-in costs nothing in the limit.

*Proof.* Write `C_ϱ` and `C_ϱϱ` for the derivatives in `ϱ`, and `G = G(ρ₀)`.
1. **Uniform bounds near `ρ₀`.** With `c_G` from A7, for `|ϱ − ρ₀| ≤
   1/(2c_G)` the Neumann series `(I − ϱW)⁻¹ = (I − ρ₀W)⁻¹ Σₖ ((ϱ − ρ₀)G)ᵏ`
   gives `‖(I − ϱW)⁻¹‖_op ≤ 2‖(I − ρ₀W)⁻¹‖_op`, so `‖G(ϱ)‖_op` is bounded
   on that interval. Since `∂G/∂ϱ = G(ϱ)²` and `∂²G/∂ϱ² = 2G(ϱ)³`, and `Δ` is
   linear in `G′` (`Δ[Y] = Diag(YM)Diag(M)⁻¹` for `Y = G′`, `(G²)′`,
   `(G³)′`), Lemma 2(b) gives `|Δ[Y]ᵢᵢ| ≤ ‖Y‖_op/√(1 − h̄)`. So
   `‖C(ϱ)‖_op`, `‖C_ϱ(ϱ)‖_op` and `‖C_ϱϱ(ϱ)‖_op` are bounded uniformly on the
   interval, which contains `ρ̃` with probability → 1 (the Corollary for
   `C₁`: `ρ̃ − ρ₀ = O_p(n^{−1/2})`).
2. **Orthogonality.** `Mf = 0` gives `C(ϱ)f = 0`, and `diag C(ϱ) = 0`, for
   every `ϱ`. Differentiating, `C_ϱ f = 0` and `diag C_ϱ = 0`. So, with
   `η = f + ε`,

       η′C_ϱ(ρ₀)η = (C_ϱ′f)′ε + ε′C_ϱ ε,   E η′C_ϱ(ρ₀)η = f′C_ϱ f + tr(C_ϱΣ) = 0,

   and both terms are `O_p(√n)` with second moments: the linear one has
   variance at most `σ̄²‖C_ϱ‖_op²‖f‖² = O(n)`, and the quadratic one, whose
   matrix has a zero diagonal, has variance at most `2σ̄⁴‖C_ϱ‖_F² = O(n)`.
   This is the Neyman orthogonality of the remark that this proposition
   replaces, now with its proof.
3. **The three coefficients.** In the Corollary's notation, with `C(ρ̃)` in
   place of a fixed `C`, the quadratic in `t = ρ − ρ₀` has coefficients
   `g₀(ρ̃) = η′C(ρ̃)η`, `g₁(ρ̃) = −2η′G′Cs(ρ̃)η` and `g₂(ρ̃) = η′G′Cs(ρ̃)Gη`.
   By Taylor's theorem in `ϱ`, with `ρ̄` between `ρ̃` and `ρ₀`:
   - `g₀(ρ̃) − g₀(ρ₀) = (ρ̃ − ρ₀)η′C_ϱ(ρ₀)η + ½(ρ̃ − ρ₀)²η′C_ϱϱ(ρ̄)η
     = O_p(n^{−1/2})O_p(√n) + O_p(n^{−1})O_p(n) = O_p(1)`, using step 2 for
     the first term and `|η′C_ϱϱ(ρ̄)η| ≤ sup‖C_ϱϱ‖_op‖η‖² = O_p(n)` for the
     second. Since `V* ≥ 2σ̲⁴‖Cs(ρ₀)‖_F² ≥ cn` by A5, this is `o_p(√V*)`.
   - `g₁(ρ̃) − g₁(ρ₀) = O_p(n^{−1/2})·O_p(n) = o_p(n)`, by the uniform bound
     on `‖C_ϱ‖_op`; so `g₁(ρ̃) = −J* + o_p(n)`.
   - `g₂(ρ̃) = O_p(n)`, by the uniform bound on `‖C‖_op`.
4. **Conclusion.** These are exactly the orders the Corollary's steps 2–4
   use, with `g₀(ρ̃)/√V* = g₀(ρ₀)/√V* + o_p(1) → N(0, 1)` by Theorem 1. So
   the sign rule picks `t̂ = (g₀(ρ̃)/J*)(1 + o_p(1))`, and the infeasible
   root is `t̂* = (g₀(ρ₀)/J*)(1 + o_p(1))`. Their difference is
   `[g₀(ρ̃) − g₀(ρ₀)]/J* + o_p(√n/n) = O_p(1/n) + o_p(n^{−1/2})`. ∎

Only the mean-zero property in step 2 needs the zero diagonal and `Mf = 0`.
Without it, `(ρ̃ − ρ₀)E η′C_ϱη` would be of order `n^{−1/2}·n = √n`, the
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
  −0.010 to −0.002), as any fixed `ϱ` gives a valid moment.
- REsd's sd is 3–10% below the first step's (0.1301 against 0.1444 at
  n = 400; 0.0497 against 0.0513 at n = 2500), and its coverage with the
  robust variance and the code's root rule is 0.946–0.952. That rule is
  `cfe_root`'s (the root nearest the OLS ratio), not the sign rule the
  Corollary covers; `tc_roots.jl` found the two rules disagree in 1 of 8000
  replications of this kind of design. The package's `root_rho` uses the
  sign rule, and validation/mc_root_rho.jl measures its coverage
  (0.931–0.975 for the Wald interval).

**Proposition 4 (a penalised smoother; proved; replaces Remark 3).** Let `M = I − S` with
`S` symmetric, its eigenvalues in `[0, 1]` (a ridge- or difference-penalised
spline), `maxᵢ Sᵢᵢ ≤ s̄ < 1` in place of A3, and `M∘M` invertible. `S` has
eigenvalue 1 on the directions the penalty leaves alone (`X` and the
penalty's null space), so `M` is singular; nothing below needs `M`
invertible, only `M∘M`, for `d`. Use the
moment `g(ρ) = y₀′Ay₀` with `A = MW + MDM` and `(M∘M)d = −diag(MW)`, so
`diag A = 0` exactly; the identity `diag(MDM) = (M∘M)d` needs only `M`
symmetric. Then:
- **Lemma 2 holds with `Sᵢᵢ` in place of `hᵢᵢ`.** `‖M‖_op ≤ 1`. The
  eigenvalues of `M` lie in `[0, 1]`, so `M² ≼ M` and
  `‖Meᵢ‖² = (M²)ᵢᵢ ≤ Mᵢᵢ`; hence `|(Y′M)ᵢᵢ|/Mᵢᵢ ≤ ‖Y‖_op/√Mᵢᵢ`, the bound of
  (b). The finer bound uses `‖Seᵢ‖² = (S²)ᵢᵢ ≤ Sᵢᵢ` the same way.
- **Theorem 1 holds as stated**, since its proof uses only `‖Cs‖_op`,
  `‖Cs‖_F` and A5–A6, never idempotency.
- **The Corollary and Proposition 3 hold with `r = Mf`.** `Mf ≠ 0` even for
  a smooth `f` because the penalty shrinks: `r` is now shrinkage bias plus
  approximation error, and validity needs `β = f′Af = o(√n)`. Over-smoothing
  breaks it exactly as too small a K does.
- **The variance estimate needs two changes.** Under homoskedasticity
  `E ûᵢ² = σ²(M²)ᵢᵢ`, not `σ²Mᵢᵢ`, so the plug-in is `sᵢ = ûᵢ²/(M²)ᵢᵢ`.
  With `f̂ = Sy₀`, `b̂ = (A + A′)Sy₀` and the noise term is `diag(HΣ̂H′)`
  with `H = (A + A′)S`. This is the notation table's `H` with the
  smoother's `A` and `S` in place of the projection's `C` and `P_B`.
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

| a | λ | tr S | max Sᵢᵢ | KS of z₀ | c_β = β/√V | bias: predicted | simulated | coverage: predicted | oracle | general `V̂` | shortcut moment |
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
  λ = 0.001, where `max Sᵢᵢ` = 0.93 and `max|dᵢ|` reaches 44: `‖Cs‖_op` stays
  at 0.71, because a large `dᵢ` sits where `Mᵢᵢ` is small.
- **Shrinkage bias behaves as Proposition 3 says.** For `c_β` ≤ 0.66 the
  simulated bias matches `β/J` (`|z|` ≤ 1.0) and the general `V̂`'s coverage
  is within 1.0 point of `P(|Z + c_β| ≤ 1.96)`. For `c_β` ≥ 1.6 the first-order
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
- **`mᵢ` is bounded below**: `mᵢ = Σₖ Mᵢₖ² ≥ Mᵢᵢ² = (1 − Sᵢᵢ)² ≥ (1 − s̄)²`.
- **The remainder in `sᵢ`.** At `ρ₀`, `ûᵢ = (Mε)ᵢ + rᵢ`, and
  `(Mε)ᵢ = Mᵢᵢεᵢ − ϑᵢ` with `ϑᵢ = Σ_{j≠i} Sᵢⱼεⱼ`. Since `(S²)ᵢᵢ ≤ Sᵢᵢ`,
  `Σ_{j≠i} Sᵢⱼ² ≤ Sᵢᵢ(1 − Sᵢᵢ)`, so `E ϑᵢ² ≤ σ̄²Sᵢᵢ` and
  `E ϑᵢ⁴ ≤ cSᵢᵢ²`. Then `sᵢ = εᵢ² + νᵢ` with
  `νᵢ = εᵢ²(Mᵢᵢ²/mᵢ − 1) + (−2Mᵢᵢεᵢϑᵢ + ϑᵢ² + 2rᵢ(Mε)ᵢ + rᵢ²)/mᵢ`, and
  `|Mᵢᵢ²/mᵢ − 1| ≤ Sᵢᵢ/(1 − s̄)²`. So `E νᵢ² ≤ c(Sᵢᵢ + rᵢ²)`, using
  `|rᵢ| ≤ c`. Hence `E‖ν‖² ≤ c(tr S + ‖r‖²) = o(n)` and
  `Σᵢ (E νᵢ²)^{1/2} ≤ c(√(n tr S) + √n‖r‖) = o(n)`. These are the two facts
  steps 1 and 2 used, with `tr S + ‖r‖²` in place of `K`. `Ω = Cs∘Cs` keeps a
  zero diagonal and bounded row sums, by Proposition 4's Lemma 2.
- **Step 1 (quadratic part)** goes through unchanged with this `ν`.
- **Step 2 (linear part).** Now `b̂ = b_S + ζ` with `b_S = (A + A′)Sf =
  b − b_r`, `b_r = (A + A′)r` and `ζ = Hε`. The extra mean `b_r` has
  `‖b_r‖² ≤ c‖r‖² = o(n)`, so `Σᵢ (b_r)ᵢ²sᵢ = O_p(‖r‖²)` (`E sᵢ ≤ c`) and the cross
  term with `b` is `O_p(√n · ‖r‖) = o_p(n)`. `Σ Var ζᵢ = tr(HΣH′) ≤
  c‖S‖_F² ≤ c tr S`, and the noise term's mean is at most `c‖H‖_F²`, so both
  are `O_p(tr S)`. In `Σ bᵢ²νᵢ`, the zero-diagonal quadratic form
  `−2Σ bᵢ²Mᵢᵢεᵢϑᵢ/mᵢ` has variance at most `cΣᵢ bᵢ⁴Sᵢᵢ = o(n²)`. The
  linear form `2Σ bᵢ²rᵢ(Mε)ᵢ/mᵢ` has variance at most `cΣ bᵢ⁴rᵢ² = o(n²)`
  (A6). The rest has absolute mean at most `cΣ bᵢ²(Sᵢᵢ + rᵢ²) = o(n)`.
- **Step 3 (`ρ̂` for `ρ₀`)** is unchanged: `û(ρ̂) = û − tMWy` and
  `b̂(ρ̂) = b̂ − tHWy` with `‖M‖_op, ‖H‖_op ≤ c`, and `Σ ûᵢ⁴ = O_p(n)` because
  `|rᵢ| ≤ c`. ∎

The proof hides two opposite `O(‖r‖²)` effects. The shrinkage in `ûᵢ` adds
`(2rᵢ(Mε)ᵢ + rᵢ²)/mᵢ` to `sᵢ`, which raises `V̂`. `b̂` estimates the shrunk
signal `b_S`, not `b`, which lowers the linear part by
`ϖ_b = Σᵢ ((b_S)ᵢ² − bᵢ²)σᵢ²`. Over-smoothing therefore biases `V̂` too, not only
`ρ̂`. The condition `‖r‖² = o(n)` is separate from the Corollary's
`β = o(√n)`: `β` is an indefinite quadratic form in `r` (Proposition 3),
so neither condition implies the other.

**Numerical check (`tc_smoother_vhat.jl`).** Same design as
`tc_smoother.jl`, at `ρ₀`, R = 500. Under normal errors the mean gap between
`V̂` with the feasible `s` and with `r` removed from `s` is exact, and so is
`ϖ_b`:

| a | λ | ‖r‖²/n | `V̂/V` feasible | `r` removed | 1 + ϖ_b/V | gap: simulated | exact | first order with `b` |
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
- **`ϖ_b` accounts for the r-removed mean** to within 0.004 where `r`
  matters; the 0.000–0.008 left at small λ is HC2's own bias. So the two
  effects explain `V̂`'s bias, and the shrunk signal is the larger one.
- **Power check.** The first-order shift with `b` in place of `b_S` misses
  by 50% at a = 1, λ = 100 (0.0605 against 0.0396) and by a factor of 4.6
  at a = 4, λ = 100 (1.47 against 0.32).
- **Where `‖r‖²/n` ≤ 0.002,** `V̂/V` is 1.000–1.008, within the projection's
  HC2 range; that is λ ≤ 1 for both amplitudes. In this design `β` binds
  first: at a = 4, λ = 1, `V̂` is unbiased but coverage is 0.663
  (`tc_smoother.jl`), because `c_β` = 1.6.

**Choosing λ (measured; no theory).** Propositions 3 and 4 say when an
interval is valid, not how to pick λ. Two ways were tried in
`tc_lambda.jl`: GCV, and a plug-in estimate of `c_β` (same design as
`tc_smoother.jl`, λ on a half-decade grid from 10⁻³ to 10², R = 500).
- GCV is `n‖M_λy₀p‖²/(n − tr S_λ)²`, where `y₀p` comes from a pilot fit at
  λ = 10⁻³.
- The plug-in is `ĉ_β(λ) = (f̃′A_λf̃ − ŝ′diag(S_pA_λS_p))/√V̂_λ`, with
  `f̃ = S_py₀p` from the same pilot.

| a | λ | tr S | c_β | ĉ_β: mean | ĉ_β: sd | sd ρ̂ | RMSE ρ̂ | coverage | GCV picks (of 500) |
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
  λ = 1–3 at a = 4. Those values give a good fit of `f`, but `c_β` is 0.3–4
  there. Under-coverage is mild for a weak surface and severe for a strong
  one. GCV must not be the default λ of a smoothed TC-CFE.
- **The RMSE-best λ is already too large for an interval.** RMSE is
  smallest where `c_β` ≈ 0.26 (a = 1) and 0.6 (a = 4), and coverage there is
  0.944 and 0.908. This is the usual undersmoothing result: a valid interval
  needs `c_β` → 0, so λ must be below the RMSE-optimal one. The price is
  precision. `sd(ρ̂)` doubles from λ = 1 to λ = 0.001.
- **`c_β` cannot be estimated well enough to pick λ.** While `c_β` ≤ 4, `ĉ_β`'s
  bias is at most 0.2, small beside its sd (0.645 against 0.598, 4.26
  against 4.14). Its sd,
  however, is 0.8–2.9 in the range that matters, because `f̃′A_λf̃` carries
  noise of the same order as `√V`. So it can flag a large bias (`c_β` ≳ 4) but
  cannot tell `c_β` = 0.3 from 0. Beyond `c_β` ≈ 4 it overshoots (30.7 against
  14.8). The cause is not checked; a likely one is the pilot's own error
  in `f̃` once `A_λ` is far from `A_p`.
- **What this leaves.** The largest λ with coverage ≥ 0.94 is about 1/3
  of GCV's modal choice at a = 1 (3.16 against 10) and 1/30 at a = 4 (0.1
  against 3.16). So the factor depends on the signal, and two designs do
  not justify a rule. An honest
  default is to undersmooth: take λ at the small end of the range where
  `V̂` is still stable, and report `sd(ρ̂)` across λ as a sensitivity
  check. The same holds for K with a projection (`tc_approx.jl`).
- **Removing the bias from the moment removes the gain too** (`tc_rbc.jl`).
  With the projection as pilot, `C_λ = A_λ − P_BA_λP_B + M₀D₀M₀`, with `D₀`
  chosen so that `diag C_λ = 0`, has `f′C_λf = 0` for every `f` in span(B)
  and every λ, and `C_0` is the projection moment. So λ can be chosen by
  `V̂/g′²` alone, and coverage holds: 0.952 and 0.956 at the chosen λ, and
  0.94–0.97 in every cell, with an exact `|c|` ≤ 0.044 although `f` is not
  in span(B). But the sd is 0.110–0.138 at every λ, against 0.127 and
  0.114 for the projection. The best member gains 3–4% over the projection
  (0.122 against 0.127 at a = 1, 0.110 against 0.114 at a = 4). The
  smoother's own sd falls to 0.058 and 0.044 at λ = 1. Power check: the
  same rule on the plain smoother picks λ = 100 in all 500 draws and
  covers 0.418 and 0.000. So among moments that are unbiased for every `f`
  in span(B), smoothing gains almost nothing here. The smoother's
  efficiency is bought with its bias. A λ with a coverage guarantee and a
  real gain therefore needs a restriction on `f` beyond `f ∈ span(B)`, such
  as a bound on `θ′Ωθ`. This is the usual finding that an adaptive interval
  cannot be honest without a smoothness bound.
- **A bias-aware interval under a roughness bound** (`tc_honest.jl`;
  Proposition 7 below proves its coverage). Over the class
  `F(ℛ) = {f = Bψ : ψ′Πψ ≤ ℛ²}`, with `Π` the penalty, the smoother's bias
  `β(λ) = f′A_λf` is at most `b̄ = ℛ√(v′Π⁺v) + ℛ²χ̄` (Lemma 4 below). Here
  `v = B′M_λWf`, and `χ̄` bounds the quadratic part. `v` is plugged in as
  `T P_B y₀(ρ̂₀)`, with `T = B′M_λW` and `ρ̂₀` the projection's root, and
  debiased by `tr(Π⁺TP_BΣ̂P_BT′)`. The interval is
  `ρ̂ ± cv(b̄/√V̂)·√V̂/|g′(ρ̂)|`, with `cv(t)` the 0.95 quantile of `|N(t, 1)|`,
  and λ̂ minimises its length. On tc_lambda's design (R = 500; `ℛ*` is the
  roughness `√(ψ*′Πψ*)` of the least-squares fit `Bψ*` of `f`; the script
  writes `Ω`, `θ` and `M` for `Π`, `ψ` and `ℛ`):
  - **The bound holds** at every λ > 0.001. At λ ≤ 0.001 it is 0, and the
    small `c` (0.003–0.031) comes from the part of `f` outside span(B).
    The linear term is 94–99% of `b̄`.
  - **The plug-in** averages 0.96–0.99 of the oracle `√(v′Π⁺v)` when
    debiased (raw: 1.06–1.26). Evaluated at each λ's own `ρ̂` instead of
    `ρ̂₀`, it fell to 0.39 at a = 4, λ = 100, and coverage to 0.006.
  - **With `ℛ = ℛ*`, coverage is 0.950–0.996 at every λ.** At λ̂ it is
    0.954 (a = 1) and 0.964 (a = 4), with a half-length of 0.117 and 0.152
    against the projection's 0.250 and 0.227: 53% and 33% shorter.
  - **Over-stating `ℛ` costs length, not coverage:** at `2ℛ*`, 0.974 and
    0.982 with half-lengths 0.134 and 0.185; at `4ℛ*`, 0.982 and 0.970 with
    0.162 and 0.210, still shorter than the projection.
  - **Power checks:** with `ℛ = 0`, the rule picks λ = 100 every time and
    covers 0.418 and 0.000; with `ℛ = ℛ*/2`, it covers 0.908 and 0.902.
  So the guarantee holds over `F(ℛ)` for the `ℛ` given, and the gain is
  real. `ℛ` is the user's assumption: it cannot be estimated without losing
  the guarantee, which is the price of honesty.

**Lemma 4 (the smoother's bias over a roughness class; proved).** Let
`S_λ = B(B′B + λΠ)⁻¹B′`, with `Π` symmetric positive semi-definite and
`B′B + λΠ` invertible; `M_λ = I − S_λ`; and `A_λ = M_λW + M_λD_λM_λ` as in
Proposition 4. Let `f = Bψ`, and split `ψ = ψ₀ + ψ₁`, with `ψ₀` the
orthogonal projection of `ψ` on `null(Π)` and `ψ₁ ∈ range(Π)`; write
`fₖ = Bψₖ`. Then:

(a) `M_λf₀ = 0`, so `β(λ) = f′A_λf = f₁′M_λWf + f₁′M_λD_λM_λf₁`.

(b) With `v = B′M_λWf`, `v ∈ range(Π)` and
`|f₁′M_λWf| = |ψ₁′v| ≤ √(ψ′Πψ) · √(v′Π⁺v)`. For a fixed `v`, equality holds
at `ψ₁ ∝ Π⁺v`.

(c) With `Q_λ = (M_λB)′D_λ(M_λB)` and `χ̄_λ` the largest `|eigenvalue|` of
`Π^{+1/2}Q_λΠ^{+1/2}`, `|f₁′M_λD_λM_λf₁| ≤ χ̄_λ ψ′Πψ`.

So for `f ∈ F(ℛ) = {Bψ : ψ′Πψ ≤ ℛ²}`,
`|β(λ)| ≤ b̄(λ) = ℛ√(v′Π⁺v) + ℛ²χ̄_λ`.

*Proof.* (a) `Πψ₀ = 0`, so `(B′B + λΠ)ψ₀ = B′Bψ₀`, and
`S_λf₀ = B(B′B + λΠ)⁻¹B′Bψ₀ = Bψ₀ = f₀`. `M_λ` is symmetric, so
`f′M_λ = f₁′M_λ`, and `M_λD_λM_λf = M_λD_λM_λf₁`.
(b) For any `ψ̃₀ ∈ null(Π)`, `ψ̃₀′v = (M_λBψ̃₀)′Wf = 0` by (a). So
`v ⊥ null(Π)`, that is `v ∈ range(Π) = range(Π^{1/2})`. As `ψ₁` is in the
same range, `ψ₁′v = (Π^{1/2}ψ₁)′(Π^{+1/2}v)`, and Cauchy–Schwarz gives the
bound, with `ψ₁′Πψ₁ = ψ′Πψ`. Equality holds when `Π^{1/2}ψ₁ ∝ Π^{+1/2}v`.
(c) `Q_λψ₀ = 0` by (a). With `z = Π^{1/2}ψ₁`, `ψ₁ = Π^{+1/2}z` and
`‖z‖² = ψ′Πψ`, so `ψ₁′Q_λψ₁ = z′Π^{+1/2}Q_λΠ^{+1/2}z` and `|ψ₁′Q_λψ₁| ≤ χ̄_λ‖z‖²`. ∎

**`b̄` is a bound for each `f`, not the worst case over `F(ℛ)`.** `v`
depends on `f`, including on `f₀`, whose size the class leaves free. The
linear term grows with `‖f₀‖`, so `sup over F(ℛ) of |β|` is infinite in
general. The bound depends on `f` only through `v`, which the data
estimate. At λ = 0, `M₀B = 0`, so `v = 0`, `Q₀ = 0` and `b̄ = 0`.

**Proposition 7 (a bias-aware interval for the smoother; proved, given
Propositions 3, 4 and 4(c), Proposition 2's proof and the Corollary).** Fix
a level `α`, a finite grid `𝒢` of values `λ ≥ 0` that contains 0 (the
projection), and a roughness bound `ℛ`. For `λ ∈ 𝒢`, let

    I_λ = ρ̂_λ ± cv(t̂_λ) √V̂_λ / |g′_λ(ρ̂_λ)|,    t̂_λ = (ℛ q̂_λ^{1/2} + ℛ²χ̄_λ) / √V̂_λ,

where `ρ̂_λ`, `V̂_λ` and `g′_λ` are those of `A_λ` (Proposition 4, the sign
rule), and `cv(t)` is the `1 − α` quantile of `|N(t, 1)|`. Here
`v̂ = T_λP_By₀(ρ̂₀)`, `T_λ = B′M_λW`, `ρ̂₀` is the projection's root, and
`Υ_λ = Π^{+1/2}T_λP_B`. `q̂_λ` is either the raw `v̂′Π⁺v̂`, or the debiased
`max(v̂′Π⁺v̂ − Σᵢ ŝ₀ᵢ‖Υ_λeᵢ‖², 0)`, with `ŝ₀` the projection's HC2 plug-in at
`ρ̂₀`. Assume, along a sequence `f = f_n ∈ F(ℛ)`:
- (i) the conditions of the Corollary and Proposition 2 for the projection
  (A4 holds, since `f ∈ span(B)`);
- (ii) for each `λ ∈ 𝒢`, the conditions of Proposition 4(c), and those of
  the Corollary for `A_λ` with `J_λ = −E g′_λ(ρ₀)` (Proposition 3);
- (iii) `t̄_λ = b̄(λ)/√V_λ = O(1)` for each `λ`;
- (iv) `ℛ‖Υ_λ‖_op/√V_λ → 0` and `ℛ‖Υ_λ‖_F/√V_λ = O(1)`; for the debiased
  `q̂_λ`, also `ℛ‖Υ_λ‖_op √K/√V_λ → 0`.

Then:
- (a) for each `λ ∈ 𝒢`, with either `q̂_λ`, `liminf P(ρ₀ ∈ I_λ) ≥ 1 − α`;
- (b) let `ℓ_λ = cv(t̄_λ)√V_λ/J_λ`, and `λ̂` minimise the half-length of
  `I_λ` over `𝒢`. If, for some `η > 0` and all large `n`, the minimiser
  `λ*_n` of `ℓ` is unique with `ℓ_λ ≥ (1 + η)ℓ_{λ*_n}` for every
  `λ ≠ λ*_n`, then with the debiased `q̂`, `P(λ̂ = λ*_n) → 1` and
  `liminf P(ρ₀ ∈ I_λ̂) ≥ 1 − α`;
- (c) without that separation, the intervals at level `α/|𝒢|` give
  `liminf P(ρ₀ ∈ I_λ̂) ≥ 1 − α` for any choice `λ̂`, data-driven or not;
- (d) if (i)–(iv) hold uniformly over `F(ℛ)` (no constant or rate in them
  depends on `f`), then (a) and (c) hold with the infimum over
  `f ∈ F(ℛ)` inside the `liminf`, and so does (b) if its separation is
  uniform too.

*Proof.* Write `Z` for a standard normal variable.
1. **The critical value.** `cv` is defined by
   `Φ(cv − t) − Φ(−cv − t) = 1 − α`. Implicit differentiation gives
   `cv′(t) = (φ(cv − t) − φ(cv + t))/(φ(cv − t) + φ(cv + t)) ∈ [0, 1)` for
   `t ≥ 0`, so `cv` is increasing and 1-Lipschitz. For a fixed `c₀ > 0`,
   `d/dc P(|Z + c| ≤ c₀) = φ(c₀ + c) − φ(c₀ − c) ≤ 0` for `c ≥ 0`, so
   `P(|Z + c| ≤ c₀)` falls as `|c|` grows.
2. **The plug-in (fixed λ; drop the subscript).** Since `f ∈ span(B)`,
   `P_Bf = f` and `Π^{+1/2}v = Υf`, so `q := v′Π⁺v = ‖Υf‖²`. Let
   `t₀ = ρ̂₀ − ρ₀ = O_p(n^{−1/2})` (Corollary, by (i)) and `η = f + ε`. Then
   `y₀(ρ̂₀) = η − t₀Gη`, and `Π^{+1/2}v̂ = Υf + Υε − t₀ΥGη`, with
   `‖t₀ΥGη‖ ≤ |t₀|‖Υ‖_op‖G‖_op‖η‖ = O_p(‖Υ‖_op)`, since `‖η‖ = O_p(√n)`.
   Expanding the square,

       v̂′Π⁺v̂ = q + 2f′Υ′Υε + ‖Υε‖² + R,   |R| ≤ O_p(‖Υ‖_op)(2√q + 2‖Υε‖ + O_p(‖Υ‖_op)).

   Write `a = ℛ√q/√V`, which is at most `t̄ = O(1)`; `u = ℛ‖Υ‖_op/√V → 0`;
   and `w = ℛ‖Υ‖_F/√V = O(1)`. All terms are scaled by `ℛ²/V`.
   - `2f′Υ′Υε` has mean 0 and variance at most `4σ̄²‖Υ‖_op²q`, so the scaled
     term has variance at most `4σ̄²u²a² → 0`.
   - `E‖Υε‖² = Σᵢ σᵢ²cᵢ ≤ σ̄²‖Υ‖_F²`, with `cᵢ = ‖Υeᵢ‖²`, so
     `ℛ‖Υε‖/√V = O_p(w)`, and the scaled `R` is `O_p(u)(2a + O_p(w) + O_p(u)) → 0`.

   So `ℛ²(v̂′Π⁺v̂ − q − ‖Υε‖²)/V → 0` in probability. As `‖Υε‖² ≥ 0`, the
   raw `q̂` satisfies `ℛ²(q − q̂)₊/V → 0`. For the debiased `q̂`, it
   remains to show `ℛ²(‖Υε‖² − Σᵢ ŝ₀ᵢcᵢ)/V → 0`. Split it into three parts.
   - `‖Υε‖² − Σᵢσᵢ²cᵢ = Σ_{i≠j}(Υ′Υ)ᵢⱼεᵢεⱼ + Σᵢcᵢ(εᵢ² − σᵢ²)` has second
     moment at most `2σ̄⁴‖Υ′Υ‖_F² + μ₄Σᵢcᵢ² ≤ c‖Υ‖_op²‖Υ‖_F²`, since
     `‖Υ′Υ‖_F ≤ ‖Υ‖_op‖Υ‖_F` and `Σcᵢ² ≤ maxᵢcᵢ · Σcᵢ ≤ ‖Υ‖_op²‖Υ‖_F²`. Scaled,
     it is `O_p(uw)`. The same bound holds for `Σᵢcᵢ(εᵢ² − σᵢ²)` alone.
   - At `ρ₀`, `ŝ₀ᵢ = εᵢ² + ν₀ᵢ` with `E ν₀ᵢ² ≤ c hᵢᵢ` (Proposition 2's proof,
     with four moments), so `E|Σᵢcᵢν₀ᵢ| ≤ (Σcᵢ²)^{1/2}(cK)^{1/2} ≤
     c‖Υ‖_op‖Υ‖_F√K`. Scaled, this is `O(uw√K) → 0` by (iv).
   - Moving from `ρ₀` to `ρ̂₀` changes `ûᵢ` by `−t₀(M₀Wy)ᵢ`. With
     `Mᵢᵢ ≥ 1 − h̄` (A3), `Σᵢ ûᵢ⁴ = O_p(n)` (Proposition 2, step 3) and
     `Σᵢ(M₀Wy)ᵢ⁴ = O_p(n)` (the same argument: each `(M₀Gη)ᵢ` is a linear
     form with bounded coefficients, under four moments), the change in `Σᵢcᵢŝ₀ᵢ` is at most
     `c|t₀|(Σcᵢ²)^{1/2}(Σᵢûᵢ²(M₀Wy)ᵢ²)^{1/2} + ct₀² maxᵢcᵢ‖M₀Wy‖² =
     O_p(‖Υ‖_op‖Υ‖_F + ‖Υ‖_op²)`. Scaled, this is `O_p(uw + u²) → 0`.

   The floor at 0 only moves `q̂` towards `q ≥ 0`. Since
   `|√x − √y| ≤ √|x − y|`, `ℛ(q^{1/2} − q̂^{1/2})₊/√V → 0` for the raw
   `q̂`, and `ℛ|q̂^{1/2} − q^{1/2}|/√V → 0` for the debiased one. With
   `V̂/V → 1` (Proposition 4(c)) and `t̄ = O(1)`, this gives
   `t̂ ≥ t̄ − o_p(1)` for the raw `q̂`, and `t̂ − t̄ → 0` for the debiased one.
3. **Coverage at a fixed λ.** Let `c_n = β/√V`. By Lemma 4, `|c_n| ≤ t̄_n`.
   Take any subsequence, and within it a further one along which
   `c_n → c` and `t̄_n → t̄`, so that `|c| ≤ t̄`. Along it, Proposition 3
   (with Proposition 4's `r = M_λf`) gives `J(ρ̂ − ρ₀)/√V → N(c, 1)`; its
   condition `β = O(√n)` holds because `|β| ≤ t̄_n√V` and `V = O(n)`. The
   Corollary gives `|g′(ρ̂)|/J → 1`. `ρ₀ ∈ I_λ` is the event

       |J(ρ̂ − ρ₀)/√V| ≤ cv(t̂) · √(V̂/V) · J/|g′(ρ̂)|,

   and by steps 1 and 2 the right side is at least `cv(t̄_n) − o_p(1)`. The
   limit law is continuous, so `liminf P(ρ₀ ∈ I_λ) ≥ P(|Z + c| ≤ cv(t̄))`,
   which is at least `P(|Z + t̄| ≤ cv(t̄)) = 1 − α` by step 1. Every
   subsequence has a further subsequence along which this holds, which
   proves (a).
4. **The choice of λ.** (b) With the debiased `q̂`, step 2 and
   Proposition 4(c) give `ĥ_λ/ℓ_λ → 1` for each `λ`, where `ĥ_λ` is the
   half-length of `I_λ`, since `cv(t̄_λ) ≥ z_{1−α/2} > 0`. The grid is finite
   and the separation is `η > 0`, so `P(λ̂ = λ*_n) → 1`. Then
   `P(ρ₀ ∈ I_λ̂) ≥ P(ρ₀ ∈ I_{λ*_n}) − P(λ̂ ≠ λ*_n)`. Along any subsequence
   on which `λ*_n` is constant, (a) applies, and the grid is finite.
   (c) `P(ρ₀ ∉ I_λ̂) ≤ Σ_{λ∈𝒢} P(ρ₀ ∉ I_λ^{(α/|𝒢|)})`, and each term has
   `limsup ≤ α/|𝒢|` by (a).
   (d) If the uniform statement failed, there would be `ε₀ > 0` and
   `f_n ∈ F(ℛ)` with `P_{f_n}(ρ₀ ∈ I) < 1 − α − ε₀` infinitely often. Along
   that sequence (i)–(iv) hold by assumption, so (a)–(c) contradict it. ∎

**What is new against Proposition 3.** Proposition 3's interval is valid
only if `c_β → 0`. Here `c_β` may stay away from zero. The price is
`cv(t̄) > z_{1−α/2}` and the assumption `ℛ`. At λ = 0, `b̄ = 0` and `I_0`
is the projection's Wald interval, so the rule never does worse than
choosing the projection, up to the noise in the choice. That holds only
under A4: when `f ∉ span(B)`, the projection is biased too (Lemma 5).

**The raw plug-in needs fewer conditions** than the debiased one: (a) and
(c) hold with it under (iv)'s first two conditions alone, because its
upward bias `‖Υε‖²` makes it conservative. Only (b) needs the debiased one.

**The conditions in `tc_honest.jl`** (`ℛ = ℛ*`, n = 900, K = 226, printed
for each λ):
- `ℛ‖Υ‖_op/√V` is at most 0.25 (a = 1) and 0.46 (a = 4);
- `ℛ‖Υ‖_F/√V` is at most 0.94 and 1.74;
- the debiased condition `ℛ‖Υ‖_op√K/√V` is at most 0.57 and 2.18 for
  λ ≤ 1, where λ̂ falls, and reaches 3.7 and 7.0 at λ = 100.

Yet the debiased plug-in averages 0.96–0.98 of the oracle at λ = 100. So
(iv)'s third condition is sufficient but loose: the Cauchy–Schwarz step for
`Σcᵢν₀ᵢ` gives away the sign structure of `ν₀`.

**What is not covered.**
- **`f ∉ span(B)`.** A remainder `f − P_Bf` adds to `β` a part that `b̄`
  does not bound, and it moves `v̂`. Lemma 5 and Proposition 7′ (below)
  cover it, by putting the class on a finer reference basis. In `tc_honest.jl`, `c` at λ ≤ 0.001 (where `b̄ ≈ 0`) is 0.003–0.031.
- **The separation in (b)** fails near ties. In `tc_honest.jl`, λ̂ spreads
  over 4–5 grid values, most of them on two or three (198, 258 and 39 of
  500 draws at λ = 1, 3.16 and 10, a = 1). So (b) does not literally apply at n = 900,
  but the selected interval still covered 0.954 and 0.964 without the
  Bonferroni step of (c). With 12 grid values, (c) would raise `z` from
  1.96 to 2.86: 46% longer where `b̄ = 0`, and less where `b̄` dominates.
- **Choosing `ℛ`** is the user's assumption. It cannot be estimated without
  losing the guarantee ("Choosing ℛ", next).

The construction is that of the bias-aware ("honest") intervals of the
nonparametric-regression literature. This note has not read those papers;
the argument above does not rely on them.

**Choosing ℛ (measured, with one remark proved).** Proposition 7 needs `ℛ`
to bound the roughness `ψ′Πψ` of `f` from above. The data cannot supply
such a bound, so `ℛ` has to be an assumption. What follows says why, and
how to state the assumption so that it costs little (`tc_rchoice.jl`, on
tc_honest's design, R = 500).

*Remark (the data cannot certify an upper bound; proved).* Let
`g = Bψ_g` with `g′Σ⁻¹g ≤ δ²`. Under normal errors, `y` has the law
`N((I − ρ₀W)⁻¹(f + g), (I − ρ₀W)⁻¹Σ(I − ρ₀W)⁻′)`, so the Kullback–Leibler
divergence between the laws under `f` and `f + g` is `g′Σ⁻¹g/2 ≤ δ²/2`, and by
Pinsker's inequality their total variation distance is at most `δ/2`. So any
statistic `ℛ̂` has `|P_f(ℛ̂ ≥ x) − P_{f+g}(ℛ̂ ≥ x)| ≤ δ/2` for every `x`. If
`ℛ̂` is a valid upper bound, with `P_h(ℛ̂ ≥ ℛ(h)) ≥ 1 − α` for every `h`, then
under `f` it must also exceed `ℛ(f + g)` with probability at least
`1 − α − δ/2`, for every such `g`. The largest `ψ_g′Πψ_g` under the
constraint is `δ²ν_Σ`, with `ν_Σ` the largest eigenvalue of `Π` against
`B′Σ⁻¹B`. In tc_rchoice's design, with `δ = 0.2` (total variation at most
0.1), that roughness is 5.39: 3.4ℛ* at a = 1 and 0.84ℛ* at a = 4. With the
sign of `g` chosen so that `ψ_f′Πψ_g ≥ 0`,
`ℛ(f + g) ≥ √(ℛ(f)² + ℛ(g)²)`. So when `f` has roughness `ℛ*`, a valid
data-based bound must exceed `√(ℛ*² + 5.39²)` with probability at least
0.85: that is 5.62 = 3.5ℛ* at a = 1, and 8.36 = 1.31ℛ* at a = 4. `ν_Σ` grows as the knots get finer, so
the gap grows with K. ∎

The remark is about certifying `ℛ`. It does not say that a perturbation
of that size destroys coverage: by the same total-variation bound, it
changes any coverage probability by at most `δ/2`.

**What the data can do, measured:**
- **The debiased estimate is useless.** `ℛ̂² = ψ̂′Πψ̂ − tr(noise)`, from
  the projection's coefficients at `ρ̂₀`, has an sd of 666ℛ*² (a = 1) and
  42ℛ*² (a = 4), and is ≤ 0 in 55% and 53% of draws. The noise term is
  the projection noise seen through `Π`, summed over 226 coefficients, and
  it dwarfs the signal.
- **Shrunk coefficients are biased, and still noisy.** From a pilot at
  λ = 1, `ℛ̂²` averages 1.39ℛ*² (sd 3.5) and 0.89ℛ*² (sd 0.46). At λ = 0.1,
  it averages 4.07ℛ*² (sd 39) and 1.09ℛ*² (sd 2.5).
- **The lower bound is uninformative.** `ℛ_L² = ℛ̂² − 1.645·se` is at most
  `ℛ*²` in 99% of draws, as it should be, but it is positive in only 1%.
- **Plug-in rules under-cover.** With λ̂ chosen as in Proposition 7, the
  coverage is:

  | ℛ from the data | coverage, a = 1 | coverage, a = 4 |
  |---|---|---|
  | `ℛ̂`, from the projection | 0.662 | 0.448 |
  | `2ℛ̂`, from the projection | 0.660 | 0.450 |
  | `ℛ̂`, pilot λ = 0.1 | 0.714 | 0.626 |
  | `ℛ̂`, pilot λ = 1 | 0.774 | 0.912 |
  | `2ℛ̂`, pilot λ = 1 | 0.772 | 0.952 |
  | `ℛ_L` | 0.424 | 0.006 |

  The projection's `ℛ̂` averages 9.4ℛ* at a = 1 and still under-covers. It
  is ≤ 0 (so `ℛ = 0`) in half the draws, and it is small exactly in the
  draws where a small bound hurts. Only `2ℛ̂` from the heavy pilot at
  a = 4 reaches 0.95, and the same rule gives 0.772 at a = 1.

**What a user can do:**
1. **State `ℛ` through a reference surface.** Choose a surface `f_ref`, in
   the same coordinates, that is at least as rough as `f` is believed to be.
   Set `ℛ = √(ψ_ref′Πψ_ref)` with `ψ_ref = B \ f_ref`. For a tensor hat
   basis with knot spacing `h` and a second-difference penalty,
   `ψ′Πψ ≈ h²∫(f_uu² + f_vv²)`: in tc_rchoice's design,
   `ℛ* = 1.10 · h√∫(f_uu² + f_vv²)` at both amplitudes. So a bound on the
   mean squared curvature over the study area is the same statement.
   (`f_ref` need not be in span(B); the least-squares fit is what enters.)
2. **Err on the large side.** The loss is asymmetric. λ = 0 is always in
   the grid with `b̄ = 0`, so a large `ℛ` sends λ̂ towards the projection
   and costs at most the gain. A small `ℛ` costs coverage. Measured at λ̂
   (projection half-lengths 0.250 at a = 1, 0.227 at a = 4):

   | ℛ/ℛ* | 0.25 | 0.5 | 0.75 | 1 | 2 | 4 | 8 | 16 |
   |---|---|---|---|---|---|---|---|---|
   | coverage, a = 1 | 0.734 | 0.908 | 0.936 | 0.954 | 0.974 | 0.982 | 0.978 | 0.970 |
   | half-length, a = 1 | 0.099 | 0.107 | 0.113 | 0.117 | 0.134 | 0.162 | 0.200 | 0.230 |
   | coverage, a = 4 | 0.648 | 0.902 | 0.950 | 0.964 | 0.982 | 0.970 | 0.972 | 0.968 |
   | half-length, a = 4 | 0.106 | 0.125 | 0.140 | 0.152 | 0.185 | 0.210 | 0.220 | 0.226 |

   With `ℛ` over-stated fourfold, the interval is still 35% shorter than
   the projection's at a = 1, and 7% shorter at a = 4. At 16-fold it is no
   longer than the projection's.
3. **Report the result across `ℛ`.** Give the interval on a grid of `ℛ`,
   and for each conclusion the largest `ℛ` at which it still holds (its
   breakdown value). In tc_rchoice's design, the projection's interval
   excludes ρ = 0.2 in 36% (a = 1) and 43% (a = 4) of draws. The bias-aware
   interval excludes it for every `ℛ` up to 64ℛ* in 29% and 39% of draws.
   Its breakdown value lies between ℛ* and 64ℛ* in another 64% and 45%,
   and below ℛ* in 7% and 16%. So a reader can see which conclusions need
   a small `ℛ`.
4. **Do not estimate `ℛ` from the data.** The remark shows that a valid
   data bound must be loose. The table above shows that plug-in rules,
   including doubled ones, under-cover.

**Lemma 5 (the bias over a class on a finer reference basis; proved).**
Lemma 4 needs `f ∈ span(B)`. Let `B̃` (n × K̃) be a second, finer basis
with a penalty `Π̃`, used only to state the class
`F̃(ℛ̃) = {B̃ψ̃ : ψ̃′Π̃ψ̃ ≤ ℛ̃²}`. The estimator still uses `B`. Assume

- (R) `B̃ · null(Π̃) ⊂ B · null(Π)`.

For tensor hat bases on the same box with second-difference penalties,
both sides are the bilinear surfaces (and `x`, if it is a column of both),
so (R) holds. Split `ψ̃ = ψ̃₀ + ψ̃₁` as in Lemma 4, and write `fₖ = B̃ψ̃ₖ`,
`f_B = P_Bf` and `M = I − P_B`. Let `C = EM` be Jin & Lee's matrix at any
`ϱ`, with `E = G(ϱ)′ − Δ(ϱ)` (Proposition 1). Then, for `f = B̃ψ̃`:

(a) `M_λf₀ = 0` for every `λ ≥ 0` (with `M₀ = M`), so `Mf = MB̃ψ̃₁`.

(b) The smoother: `β(λ) = f′A_λf = ψ̃₁′ṽ_λ + ψ̃₁′Q̃_λψ̃₁`, with
`ṽ_λ = (M_λB̃)′Wf_B` and `Q̃_λ = (M_λB̃)′W(MB̃) + (M_λB̃)′D_λ(M_λB̃)`.

(c) REsd: `β_R = f′Cf = ψ̃₁′ṽ_R + ψ̃₁′Q̃_Rψ̃₁`, with `ṽ_R = (MB̃)′E′f_B`
and `Q̃_R = (MB̃)′E(MB̃)`.

(d) In both cases `ṽ ∈ range(Π̃)`, and `Q̃ψ̃₀ = 0 = ψ̃₀′Q̃`. So
`|β| ≤ b̃ = ℛ̃√(ṽ′Π̃⁺ṽ) + ℛ̃²χ̃` for `f ∈ F̃(ℛ̃)`, with `χ̃` the largest
`|eigenvalue|` of `Π̃^{+1/2} sym(Q̃) Π̃^{+1/2}`, and `sym(Y) = (Y + Y′)/2`.

(e) With `B̃ = B` and `Π̃ = Π`, (b) is Lemma 4 and (c) gives `b̃ = 0`. At
`λ = 0`, (b) is Proposition 3's split.

*Proof.* (a) By (R), `f₀ = Bφ₀` with `Πφ₀ = 0`, so Lemma 4(a) gives
`M_λf₀ = 0`; at `λ = 0` this is `Mf₀ = 0`.
(b) Lemma 4(a)'s argument uses only `M_λf₀ = 0`, so
`β = f₁′M_λWf + f₁′M_λD_λM_λf₁`. Write `Wf = Wf_B + WMf`, and use (a)
for `Mf = MB̃ψ̃₁`.
(c) `Cf_B = 0`, since `Mf_B = 0`. So `f′Cf = (f_B + Mf)′EMf`, and (a)
gives `Mf = MB̃ψ̃₁`.
(d) For `ψ̃₀ ∈ null(Π̃)`: `ψ̃₀′ṽ_λ = (M_λf₀)′Wf_B = 0`, and
`ψ̃₀′ṽ_R = (Mf₀)′E′f_B = 0`. Each `Q̃` has `M_λB̃` or `MB̃` as its left
and its right factor, and these vanish on `ψ̃₀` by (a). The rest is
Lemma 4(b) and (c), with `ψ′Qψ = ψ′sym(Q)ψ`.
(e) `MB = 0`, so the `W` term of `Q̃_λ` vanishes, `f_B = f` and
`ṽ_λ = v`; `ṽ_R` and `Q̃_R` vanish. At `λ = 0`,
`β = r′Wf_B + r′(W + D)r` with `r = Mf`, and `r′Wf_B = r′MWP_Bf = r′r_W`. ∎

`b̃` depends on `f` only through `f_B`, which `P_By₀(ρ₀)` estimates
without bias, because `P_Br = 0`. The plug-in and its debiasing are those
of Proposition 7, with `T̃_λ = (M_λB̃)′W` or `T̃_R = (MB̃)′E′`, and
`Υ̃ = Π̃^{+1/2}T̃P_B`. **The class does not depend on the reference grid in
the limit.** With the penalty scaled by the knot spacing,
`ψ̃′Π̃ψ̃ ≈ h̃²∫(f_uu² + f_vv²)` (as in "Choosing ℛ"), so as `B̃` is refined,
`F̃` tends to a bound on the curvature integral. For the same reason `ℛ̃`
is stated by interpolating a reference surface at the knots of `B̃`, not
by least squares, which is not defined once `K̃ > n`.

**Proposition 7′ (the bias-aware interval when `f ∉ span(B)`; proved for
the smoother candidates, given Proposition 7's inputs).** Proposition 7(a)–(d)
hold for `f ∈ F̃(ℛ̃)` in place of `F(ℛ)`, with `b̃` in place of `b̄`, at every
`λ ∈ 𝒢`, including `λ = 0`, which now has `b̃ > 0`. The changes to its
assumptions are:
- (i) A4 is replaced by Proposition 3, for the projection with
  `c_β = O(1)`;
- (iv) is read with `Υ̃`;
- (v) is new: `ℛ̃‖Υ̃‖_op‖Mf‖/√V → 0`.

The REsd candidate is not covered, because Proposition 1 has not been
proved without A4 (its step 2 uses `Mf = 0`).

*Proof.* Proposition 7's proof uses `f ∈ span(B)` only in step 2, for
`Π^{+1/2}v = Υf`, and through A4 for `t₀ = O_p(n^{−1/2})`.
- Since `P_Bf = f_B`, `Π̃^{+1/2}ṽ = Υ̃f` for every `f`, and step 2's
  expansion is unchanged.
- Under Proposition 3 with `c_β = O(1)`,
  `J(ρ̂₀ − ρ₀)/√V → N(c_β, 1)`, so `t₀ = O_p(n^{−1/2})` still.
- The one new term is in the debiasing. At `ρ₀`, `û₀ = r + Mε`, so `ŝ₀ᵢ`
  gains `(rᵢ² + 2rᵢ(Mε)ᵢ)/Mᵢᵢ`. Its weighted sum is at most
  `c maxᵢcᵢ(‖r‖² + O_p(‖r‖))`, which is `O_p(‖Υ̃‖_op²‖r‖²)`. Scaled by
  `ℛ̃²/V`, this vanishes by (v). The term matters because it inflates
  `ŝ₀`, which deflates `q̂`.
- Step 3 uses Lemma 5 in place of Lemma 4, and Proposition 3 with
  `r = M_λf`. Steps 1 and 4 are unchanged. ∎

**Measured** (`tc_refbasis.jl`). The design is tc_honest's, with the
estimation basis coarsened to `k` = 4, 5, 7 or 15 knots a side. The
reference has `k̃ = 30` (`K̃ = 901 > n`). The candidates are REsd and 12
smoother λ, with R = 300. "Old" is the released rule: Lemma 4 on `B`, with
`ℛ` from the least-squares fit of `f` on `B`, and `b̄ = 0` for REsd. "New"
is Lemma 5, with `ℛ̃` from the interpolant (`ℛ̃*`, true for `f̃`, and
`‖f − f̃‖/‖f‖ ≤ 0.005`).
- **The split is exact.** (b) and (c) hold to a relative 1e-12 or better,
  at every λ, in all 8 cells. This also checks (a), which the split uses.
  `B̃·null(Π̃)` lies in `span(B)` to 1e-14.
- **The new bound holds** in all 104 cases: every candidate of every cell.
  The old bound is exceeded for REsd in all 8 cells, and at small λ when
  `k ≤ 7`.
- **It converges in `k̃`.** At `k = 5`, `a = 4`, `λ = 1`, `b̃/√V` is 15.26,
  15.55, 15.71 and 15.77 for `k̃` = 15, 20, 30 and 40, and 10.98–11.12 for
  REsd. `ℛ̃*(k̃ − 1)` is 85.0–83.0.
- **Coverage at λ̂, with mean half-lengths in brackets.** The old rule
  under-covers where the remainder matters, and the new one covers
  everywhere:

  | a | k | old, `ℛ*` | new, `ℛ̃*` | new, `4ℛ̃*` |
  |---|---|---|---|---|
  | 1 | 4 | 0.933 (0.102) | 1.000 (0.185) | 1.000 (1.181) |
  | 1 | 5 | 0.950 (0.105) | 0.990 (0.148) | 1.000 (0.630) |
  | 1 | 7 | 0.967 (0.113) | 0.983 (0.125) | 1.000 (0.305) |
  | 1 | 15 | 0.960 (0.118) | 0.960 (0.118) | 0.997 (0.174) |
  | 4 | 4 | 0.050 (0.079) | 1.000 (0.798) | 1.000 (7.206) |
  | 4 | 5 | 0.717 (0.091) | 1.000 (0.516) | 1.000 (4.545) |
  | 4 | 7 | 0.923 (0.107) | 1.000 (0.277) | 1.000 (1.960) |
  | 4 | 15 | 0.973 (0.151) | 0.980 (0.164) | 1.000 (0.432) |

- **The price is small only for a fine `B`.** At `k = 15` and `ℛ̃*`, the
  new rule is 0% and 8% longer than the old one. At `4ℛ̃*` it is 7% and
  115% longer. At `k ≤ 7` it is 1.1–10 times longer.
- **The bound is loose against this `f`.** For the smoother at small λ, `b̃`
  is 7–10 times `|β|` when `k ≤ 5`. For REsd at `k = 15` it is 0.137 and
  0.705 against `|c|` = 0.001 and 0.012. The worst case in `F̃` may spend
  the whole curvature budget on the part of `f` that `B` misses. Such
  functions are in the class: inside each cell the remainder has the
  curvature of `f` itself. So much of the gap is probably the class and
  not the bound, but that is not shown. The joint (trust-region) bound of
  the two terms would show how much is the bound.
- **A rule for K.** The new half-length can be compared across bases, and
  its minimum over `k` is at the finest `k` tried (15) in all four
  settings. This is a measured version of "undersmooth".

**Proposition 5 (a primitive condition for A5; proved).** Let `W` be
non-negative with a zero diagonal, `‖W‖_op ≤ c_W` and `‖W‖_F² ≥ κ_W n`; a
row-standardised `W` with at most `n_W` neighbours per row has `κ_W = 1/n_W`.
For the exact-diagonal moment, with `d̄ = maxᵢ|dᵢ|`,

    ‖Cs‖_F²/n ≥ κ_W/2 − 2(K/n)(c_W + d̄)².

So A5 holds whenever `limsup K/n < κ_W/(4(c_W + d̄)²)`, and in particular
when `K/n → 0` with `d̄` bounded, with `liminf ‖Cs‖_F²/n ≥ κ_W/2`. A
sufficient condition for bounded `d̄` is `maxᵢ hᵢᵢ ≤ h̄ < 1/2`, which gives

    d̄ ≤ c_W √h̄ / ((1 − h̄)(1 − 2h̄)),

using `maxᵢ‖Weᵢ‖ ≤ c_W` (A2).

For Jin & Lee's `C = (W′ − Δ)M` the same holds with `max|Δᵢᵢ|` in place of
`d̄`, which Lemma 2(b) bounds under A3 alone.

**(b) The spectral form.** With `X = Ws + D = Σⱼ ωⱼφⱼφⱼ′` and
`πⱼ = ‖P_Bφⱼ‖²`, the overlap of the basis with eigenvector `φⱼ`,

    ‖Cs‖_F² ≥ Σⱼ ωⱼ²(1 − 2πⱼ) = ‖X‖_F² (1 − 2π̄_w),   π̄_w = Σⱼ ωⱼ²πⱼ / Σⱼ ωⱼ².

So A5 holds whenever the ω²-weighted mean overlap `π̄_w` stays below 1/2:
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
For a row-standardised row with at most `n_W` entries, `Σⱼ Wᵢⱼ² ≥ 1/n_W`. For
`d̄`: `(M∘M)d = −diag(MW) = diag(P_BW)`, and
`|eᵢ′P_BWeᵢ| ≤ ‖P_Beᵢ‖‖Weᵢ‖ = √hᵢᵢ‖Weᵢ‖`. Row `i` of `M∘M` has diagonal
`(1 − hᵢᵢ)²` and off-diagonal sum `Σ_{j≠i} hᵢⱼ² = hᵢᵢ(1 − hᵢᵢ)`, so its
dominance margin is `(1 − hᵢᵢ)(1 − 2hᵢᵢ)`, and
`‖(M∘M)⁻¹‖_∞ ≤ 1/((1 − h̄)(1 − 2h̄))`. For (b), the same orthogonal
split gives `‖MXM‖_F² ≥ ‖X‖_F² − 2‖P_BX‖_F²` (with `X` symmetric,
`‖XP_B‖_F = ‖P_BX‖_F`), and `‖P_BX‖_F² = tr(XP_BX) = Σⱼ ωⱼ²φⱼ′P_Bφⱼ`. ∎

**How sharp it is (`tc_a5.jl`).** Exact computation, no simulation, for the
n = 900 kNN `W` (k = 6, so `κ_W` = 1/6 and the bound on `‖Ws‖_F²/n` is 0.083;
the actual value is 0.153) with the Chebyshev and hat bases used above:

| basis | K/n | max hᵢᵢ | ‖Cs‖_F²/n | d̄ | (a), measured ‖Ws‖_F, d̄ | (a), 1/(2n_W) | (a), the d̄ bound | π̄_w | (b) spectral |
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

    V̂ = 2s′(Cs∘Cs)s + Σᵢ (b̂ᵢ² − noiseᵢ) sᵢ,   sᵢ = ûᵢ²/Mᵢᵢ,  û = My₀(ρ̂).

**The linear part.** Since `CP_B = PMP_B = 0`, `b̂ = (C + C′)P_By₀ = b + ζ`, with
`ζ = C′P_Bε`. So `E b̂ᵢ² = bᵢ² + Var ζᵢ`, and
`Σᵢ Var ζᵢ = tr(C′P_BΣP_BC) ≤ σ̄²‖C‖_op² K`. So without the noise term, the
linear part is biased upwards by `O(K)`, against `V ≍ n`: a relative bias of
`O(K/n)`. That matches the 3–10% excess measured at K/n of 0.1–0.2 before the
term was added. The noise term, `diag(HΣ̂H′) = (H∘H)s` with `H = (C + C′)P_B` and
`Σ̂ = Diag(s)`, estimates `Var ζᵢ` and removes it.

**The plug-in variances are the weak point when K/n does not vanish.**
`E ûᵢ² = Σⱼ Mᵢⱼ²σⱼ²`, with `Mᵢⱼ = −hᵢⱼ` for `j ≠ i` and
`Σ_{j≠i} hᵢⱼ² = hᵢᵢ(1 − hᵢᵢ)`. So

    E sᵢ − σᵢ² = Σ_{j≠i} hᵢⱼ²(σⱼ² − σᵢ²) / (1 − hᵢᵢ),

which is at most `hᵢᵢ · maxⱼ|σⱼ² − σᵢ²|`. It is zero under
homoskedasticity, `O(K/n)` on average otherwise, and does not vanish if
`K/n → θ > 0`. The products `sᵢsⱼ` in the quadratic part add a
correlation bias of the same order: `Cov(ûᵢ², ûⱼ²)` involves
`(Σₖ MᵢₖMⱼₖσₖ²)²`, and summing `γᵢⱼ²hᵢⱼ²` over all pairs gives at most
`maxᵢⱼ γᵢⱼ² · K`, so the relative bias is `O(K/n)`. This is probably the 1–3%
conservative bias seen in `tc_null.jl`.

**The conjecture for step 2,** under A1–A6 with bounded eighth moments, which
the variance of the degree-4 terms of `V̂` needs:
- `V̂/V → 1` in probability when `K/n → 0`;
- the same when `K/n → θ > 0` under homoskedasticity;
- `V̂` is inconsistent under heteroskedasticity with `K/n → θ > 0`.

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

and `b̂ᵢ²sᵢ` in the linear part has a bias of the same kind. Kline et al.
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
  that the linear part's `2bᵢfᵢ(HΣM)ᵢᵢ/Mᵢᵢ` bias would show here at
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
  `(MΣM)ᵢⱼ = −σ²(P_B)ᵢⱼ`, which for a local basis is of order K/n between
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

*Proof.* Write `Ω = Cs∘Cs` and `H = (C + C′)P_B`, so that `V = 2σ²′Ωσ² +
Σᵢ bᵢ²σᵢ²` and `V̂ = 2s′Ωs + Σᵢ (b̂ᵢ² − ((H∘H)s)ᵢ)sᵢ`. Since A5 gives
`V ≥ 2σ̲⁴‖Cs‖_F² ≥ cn`, it is enough to show `V̂ − V = o_p(n)`. Two facts
carry the argument:
- **`Ω` is bounded although `Cs` is dense.** Its entries are non-negative,
  its diagonal is zero (`diag C = 0`), and its row sums are
  `Σⱼ Csᵢⱼ² = ‖Cs eᵢ‖² ≤ ‖Cs‖_op² ≤ c` (Lemma 2). So `‖Ω‖_op ≤ c` and
  `‖Ω‖_F² ≤ nc²`.
- **`sᵢ` is `εᵢ²` plus a remainder of order `hᵢᵢ`.** At `ρ₀`, with `f ∈ span
  B`, `ûᵢ = Mᵢᵢεᵢ − ϑᵢ` with `ϑᵢ = Σ_{j≠i} hᵢⱼεⱼ`. Then `sᵢ = εᵢ² + νᵢ`,
  `νᵢ = −hᵢᵢεᵢ² − 2εᵢϑᵢ + ϑᵢ²/Mᵢᵢ`. From `Σ_{j≠i} hᵢⱼ² = hᵢᵢ(1 − hᵢᵢ)`,
  `E ϑᵢ² ≤ σ̄²hᵢᵢ` and `E ϑᵢ⁴ ≤ (3σ̄⁴ + μ₄)hᵢᵢ²`; with A3 this gives
  `E νᵢ² ≤ c hᵢᵢ`. Hence `E‖ν‖² ≤ cK` and `Σᵢ (E νᵢ²)^{1/2} ≤ c Σᵢ √hᵢᵢ ≤
  c√(nK)`, by Cauchy–Schwarz and `Σᵢ hᵢᵢ = K`. Both are `o(n)` when
  `K/n → 0`.

1. **Quadratic part at `ρ₀`.** With `ε² = (εᵢ²)ᵢ`,

       s′Ωs − σ²′Ωσ² = (ε² − σ²)′Ω(ε² − σ²) + 2σ²′Ω(ε² − σ²) + 2ε²′Ων + ν′Ων.

   The first term has mean zero because `Ω` has a zero diagonal and the
   `εᵢ² − σᵢ²` are independent; its variance is at most
   `2μ₄²‖Ω‖_F² = O(n)`. The second has variance at most `4μ₄‖Ωσ²‖² = O(n)`.
   So both are `O_p(√n)`. For the third, `E|ε²′Ων| ≤ Σᵢⱼ Ωᵢⱼ (E εⱼ⁴)^{1/2}
   (E νᵢ²)^{1/2} ≤ c Σᵢ √hᵢᵢ = O(√(nK))`, using the row sums of `Ω`. The
   fourth is at most `‖Ω‖_op‖ν‖² = O_p(K)`.
2. **Linear part at `ρ₀`.** `b̂ = b + ζ`, with `ζ = C′P_Bε`. Split
   `Σᵢ b̂ᵢ²sᵢ − Σᵢ bᵢ²σᵢ²` into four pieces:
   - `Σ bᵢ²(εᵢ² − σᵢ²)` has variance at most `μ₄ Σ bᵢ⁴ ≤ μ₄ maxᵢ bᵢ² ‖b‖²`,
     which is `o(n²)` by A6 and `‖b‖² = O(n)` (A8, Lemma 2).
   - `Σ bᵢ²νᵢ`. Its part `−2Σ bᵢ²εᵢϑᵢ` is a quadratic form in `ε` with a
     zero diagonal and matrix entries `bᵢ²hᵢₖ`, so its variance is at most
     `4σ̄⁴ Σᵢ bᵢ⁴ hᵢᵢ = o(n²)`. Its other parts have absolute mean at most
     `c Σᵢ bᵢ² hᵢᵢ`, which is `o(n)` by A10.
   - `2Σ bᵢζᵢsᵢ ≤ 2(Σ bᵢ²sᵢ)^{1/2}(Σ ζᵢ²sᵢ)^{1/2}`, and
   - `Σ ζᵢ²sᵢ` and the noise term `s′(H∘H)s`. Their means are at most
     `c Σᵢ Var ζᵢ ≤ c‖C‖_op²K` and `c‖H‖_F² ≤ cK`, because
     `E ζᵢ⁴ ≤ c (Var ζᵢ)²`, `E sᵢ² ≤ c` and `E sᵢsₖ ≤ c` under four moments.
     So they are `O_p(K)`, and the cross term is `O_p(√(nK))`.
3. **Replacing `ρ₀` by `ρ̂`.** With `t = ρ̂ − ρ₀ = O_p(n^{−1/2})` (the
   Corollary) and `û₁ = MWy`, `û(ρ̂) = û − tû₁` and `b̂(ρ̂) = b̂ − tHWy`. Here
   `‖û₁‖² = O_p(n)` and `‖HWy‖² = O_p(n)` by A7 and A8, so the change
   `b̂_t = b̂(ρ̂) − b̂` has `‖b̂_t‖ = O_p(1)`. The change `s_t = s(ρ̂) − s` is
   `(s_t)ᵢ = (−2tûᵢ(û₁)ᵢ + t²(û₁)ᵢ²)/Mᵢᵢ`, with
   `‖s_t‖ ≤ c(|t| ‖û‖₄‖û₁‖ + t²‖û₁‖²) = O_p(n^{−1/2} n^{1/4} n^{1/2}) + O_p(1)
   = O_p(n^{1/4})`, because `Σ ûᵢ⁴ = O_p(n)` under four moments. So the
   quadratic part moves by at most `‖Ω‖_op(2‖s‖‖s_t‖ + ‖s_t‖²) =
   O_p(n^{3/4})`. The linear part moves by terms such as
   `|t| Σ b̂ᵢ²|ûᵢ(û₁)ᵢ|/Mᵢᵢ ≤ c|t|(Σ b̂ᵢ²ûᵢ²)^{1/2}(Σ b̂ᵢ²(û₁)ᵢ²)^{1/2}`, which is
   `O_p(n^{−1/2})·O_p(√n)·o_p(n)`, because `Σ b̂ᵢ²(û₁)ᵢ² = o_p(n²)`. To see
   this, write `b̂ = b + ζ`, so `b̂ᵢ² ≤ 2bᵢ² + 2ζᵢ²`:
   - A6 with `‖b‖² = O(n)` (A8, Lemma 2) gives
     `maxᵢ bᵢ² = (maxᵢ bᵢ²/‖b‖²)‖b‖² = o(n)`, so
     `Σ bᵢ²(û₁)ᵢ² ≤ maxᵢ bᵢ² ‖û₁‖² = o(n)·O_p(n) = o_p(n²)`;
   - `ζ = Hε` has `Var ζᵢ ≤ σ̄²‖H‖_op² ≤ c` (Lemma 2) and
     `E ζᵢ⁴ ≤ c(Var ζᵢ)²` (step 2), so `Σᵢ ζᵢ⁴ = O_p(n)` and
     `maxᵢ ζᵢ² ≤ (Σᵢ ζᵢ⁴)^{1/2} = O_p(√n)`. Hence
     `Σ ζᵢ²(û₁)ᵢ² ≤ maxᵢ ζᵢ² ‖û₁‖² = O_p(n^{3/2}) = o_p(n²)`.

   The other term is `Σ (2b̂ᵢ(b̂_t)ᵢ + (b̂_t)ᵢ²)sᵢ`, which is
   `O_p(√n)·O_p(maxᵢ sᵢ)^{1/2} + O_p(maxᵢ sᵢ) = O_p(n^{3/4})`, since
   `maxᵢ sᵢ = O_p(√n)` when `E sᵢ² ≤ c`. All of these are `o_p(n)`. ∎

**What the proof says about the measurements.** The remainders are of order
`√(K/n)` relative to `V` in the worst case, and the measured bias at
K/n ≈ 0.05 (0.990–1.042 in all the designs above) is smaller than that. The
proof does not cover `K/n → θ > 0`, where `Σ √hᵢᵢ ≍ n` and the bias
measured above stays. The conjecture asked for eighth moments; consistency
needs only four. The eighth would be needed for a rate, or for the variance
of `V̂`.

**The CJN version.** With `s = (M∘M)⁻¹û²` the same steps go through if
`‖(M∘M)⁻¹‖_∞` is bounded, which Cattaneo, Jansson & Newey obtain from
`maxᵢ hᵢᵢ < 1/2`; then `E s = σ²` exactly, the `νᵢ` have mean zero, and A10
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
  `4 tr(ΩΛ[(MΣ̂M)∘(MΣ̂M)]Λ)` with `Σ̂ = Diag(s)` gives 0.986–1.015 at
  K/n ≤ 0.19 (`|z|` ≤ 2.0), at no cost in sd. At K/n = 0.25, where the
  leverage reaches 0.94, it fails: 0.52–0.92, `z` from −5 to −28. There the
  plug-in `(MΣ̂M)ᵢⱼ²` carries its own bias, which involves `σₖ⁴` and so
  cannot be removed from one observation per unit.
- This corrects only the quadratic part; the next paragraph does the linear
  part.

**The exact bias of CJN's linear part (proved; `tc_cjn_lin.jl`).** At `ρ₀`,
`b̂ = b + ζ` with `ζ = Hε`, `H = MWP_B` (the notation table's `H` for this
check's moment `A = MW + MDM`), and the noise term is
`noiseᵢ = Σₖ Hᵢₖ²sₖ`. Write `𝒳 = (MΣM)∘(MΣM)`. Then

    E Σᵢ(b̂ᵢ² − noiseᵢ)sᵢ − Σᵢ bᵢ²σᵢ²
        = 2Σᵢ bᵢHᵢᵢμ₃ᵢ + 2Σᵢₖ Λᵢₖ(HΣM)ᵢₖ² − 2Σᵢ[(H∘H)Λ𝒳Λ]ᵢᵢ,

the skewness, cross and noise terms.

*Proof.* For linear forms `Υ₁ = α′ε` and `Υ₂ = ψ′ε` in independent `ε`,
`E[Υ₁Υ₂²] = Σₗ αₗψₗ²μ₃ₗ` and
`E[Υ₁²Υ₂²] = E Υ₁² E Υ₂² + 2(E Υ₁Υ₂)² + Σₗ αₗ²ψₗ²κₗ`. With `Υ₁ = ζᵢ`, `Υ₂ = ûₖ`
and `Λ(M∘M) = I`:
- `E[ζᵢsᵢ] = Σₖ Λᵢₖ Σₗ HᵢₗMₖₗ²μ₃ₗ = Hᵢᵢμ₃ᵢ`;
- `E[ζᵢ²sᵢ] = σᵢ² Var ζᵢ + 2Σₖ Λᵢₖ(HΣM)ᵢₖ² + Hᵢᵢ²κᵢ`, since
  `Σₖ Λᵢₖ E ûₖ² = σᵢ²`;
- `E[noiseᵢsᵢ] = Σₖ Hᵢₖ²(σₖ²σᵢ² + Cov(sₖ, sᵢ))`, where
  `Cov s = 2Λ𝒳Λ + Diag(κ)` from the previous proof, which gives
  `σᵢ² Var ζᵢ + 2[(H∘H)Λ𝒳Λ]ᵢᵢ + Hᵢᵢ²κᵢ`.

The `Var ζᵢ` and `κᵢ` terms cancel between the last two. ∎

The cross term vanishes under homoskedasticity, since then
`HΣM = σ²MWP_BM = 0`. The noise term does not: `Λ𝒳Λ = σ⁴Λ` there. The skewness term needs a
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
900. The next check covers those.

**Other `W`, other `n`, K/n up to 0.30 (`tc_coverage_robust.jl`).** The
same estimator and HC2 `V̂`, with the weak signal, normal errors, smooth and
rough variances, `ρ₀` = 0.4 and R = 1000, over:
- four `W`: kNN with k = 6 and k = 20, rook contiguity on a regular grid,
  and a distance band (inverse-distance weights within the radius that
  gives 10 neighbours on average, so the count varies);
- n = 400, 900 and 1600;
- K/n ≈ 0.05, 0.11, 0.19 and 0.30.

The locations are stratified: one per cell of a √n × √n grid, jittered
within it. With uniform locations the hat basis had units with leverage 1
at n = 1600, so these draws differ from the n = 900 design above. Six
designs with `max hᵢᵢ` > 0.99 (K/n = 0.29 at n = 900, for the three
`W` on random locations and both variances) were skipped as outside A3.
That leaves 90 cells.

| | cells | HC2 coverage | oracle coverage |
|---|---|---|---|
| `sd(ρ̂)` < 0.39 | 83 | 0.930–0.966 | 0.931–0.964 |
| `sd(ρ̂)` ≥ 0.39 | 7 | 0.852–0.942 | 0.869–0.928 |
| all | 90 | HC2 − oracle: −0.026 to +0.018 | |

- **HC2 tracks the oracle everywhere**, within −2.6 to +1.8 points. No
  `V̂` was negative. Where coverage fails, the oracle fails as well, so the
  failure lies in the first-order approximation, not in `V̂`.
- **The failures are weak identification.** All seven cells with an oracle
  coverage below 0.93 have `sd(ρ̂)` ≥ 0.39; the largest `sd(ρ̂)` among the
  other 83 is 0.35. Six are kNN-20 at n = 400 or the distance band at
  n = 400, K/n = 0.30, and one is kNN-20 at n = 1600, K/n = 0.30. There
  `‖Cs‖_F²` is 1.6–14, `ρ̂` is biased down by 0.08–0.26 (probably because
  the root is confined to (−1, 1); not checked), and the interval is
  ±0.8 to ±1.1 wide. The 0.39 split is read
  off these results, not derived. A dense `W` with small `n` is where
  this happens: kNN-20 at n = 400 has `‖Cs‖_F²/n` = 0.004–0.021, against
  0.045–0.23 for kNN-6 and rook. An interval as wide as that should be read
  as "ρ is not identified here", not as a 95% statement.
- **K/n = 0.30 is fine when the leverage stays bounded.** Rook, whose grid
  basis has `max hᵢᵢ` 0.89–0.93, covers 0.949–0.963 at K/n = 0.30 for all
  three `n`. kNN-6 at n = 1600 covers 0.936 and 0.951 there, though `d̄`
  is large (`‖Cs‖_F²/n` jumps to 0.19, `sd(ρ̂)` to 0.30 under rough
  variances).
- **n = 1600 is slightly low with kNN-6 and rough variances** (oracle
  0.931–0.939, HC2 0.936–0.944), and so is rook with smooth variances at
  K/n = 0.05 (oracle 0.932, HC2 0.930). The oracle is low by the same
  amount, so this is the first-order approximation too.
- **Power check.** The homoskedastic `V̂` covers 0.811–0.986 under smooth
  variances in the 83 identified cells, and 0.858–0.967 under rough ones.
  As above, it fails mainly where the variance has a spatial pattern.

So step 2's conclusion holds for these `W` and `n` and up to K/n = 0.30:
HC2 covers as well as the oracle does, within 2.6 points. What it cannot fix
is weak identification, which shows as a large `sd(ρ̂)` and a small
`‖Cs‖_F²`. Proposition 6 fixes that with a different interval.

**Proposition 6 (an interval that does not need identification; proved).**
Let `V̂(ρ)` be the HC2 `V̂` evaluated at a candidate `ρ` (its `sᵢ`, `b̂` and
noise term computed from `y₀(ρ)`). Define the test-inversion set

    CS = {ρ ∈ (−1, 1) : g(ρ)² ≤ 1.96² V̂(ρ)}.

Assume A1 with four moments, A2–A8, A10 and K/n → 0, but **not A9**. Then
`P(ρ₀ ∈ CS) → 0.95`.

*Proof.* `ρ₀ ∈ CS` if and only if `g(ρ₀)²/V̂(ρ₀) ≤ 1.96²`. Theorem 1 gives
`g(ρ₀)/√V → N(0, 1)` from A1–A6 alone; its proof never uses `J`. Steps 1
and 2 of Proposition 2 give `V̂(ρ₀)/V → 1` at `ρ₀` itself. They use A5
(`V ≥ cn`), A6, A8, A10 and Lemma 2, but neither A9 nor the root; step 3,
which moves from `ρ₀` to `ρ̂`, is not needed. Slutsky's lemma finishes. ∎

This is the Anderson–Rubin device (Anderson & Rubin, 1949), in the form
Stock & Wright (2000) give for GMM with weak identification: test each `ρ`
with the moment itself, and keep the `ρ` that are not rejected. (Both papers
are cited from their standard use; neither has been read for this note.) The Wald
interval `ρ̂ ± 1.96√V̂/|g′(ρ̂)|` needs `J/√V → ∞` (A9), through the
Corollary. `CS` does not, so its coverage holds however small `J` is.
`CS` is cheap to compute. `g` is quadratic in `ρ`. `V̂(ρ)` is
an exact quartic in `ρ`, because `sᵢ(ρ) = (û₀ − ρû₁)ᵢ²/Mᵢᵢ` is quadratic,
`b̂(ρ)` is linear and the noise term is linear in `s`. So `CS` is the set
where a quartic polynomial is non-negative. It can be an interval, a union
of intervals, or the whole range, and the last means the data say nothing
about `ρ`.

**Numerical check (`tc_ar.jl`).** The seven weakly identified cells of
`tc_coverage_robust.jl` and six well-identified controls, with normal and
`χ²₃` errors, R = 1000. Once per cell, the quartic is checked against `V̂`
computed directly, at `ρ̂` and `ρ₀`; it agrees to within 1.1 × 10⁻¹⁵ in
every cell.

| cells | Wald, HC2 | `CS`, HC2 | `CS`, true σ² | mean length of `CS` |
|---|---|---|---|---|
| weak, normal (7) | 0.852–0.942 | 0.956–0.978 | 0.941–0.975 | 1.05–1.46 |
| weak, `χ²₃` (7) | 0.841–0.963 | 0.938–0.981 | 0.928–0.962 | 1.04–1.45 |
| controls, normal (6) | 0.936–0.953 | 0.947–0.956 | 0.947–0.955 | 0.23–1.05 |
| controls, `χ²₃` (6) | 0.937–0.962 | 0.945–0.961 | 0.940–0.951 | 0.24–1.05 |

- **`CS` covers where the Wald interval fails.** In the weak cells the
  Wald interval covers as little as 0.841, and `CS` covers 0.938–0.981.
  The over-coverage up to 0.98 is HC2's upward bias at `ρ` far from `ρ₀`;
  with the true `σ²` it is 0.928–0.975.
- **`CS` costs nothing where identification is strong.** In the controls
  its mean length equals the Wald interval's: 0.335 against
  2 · 1.96 · sd(ρ̂) = 0.325 for kNN-6 at n = 900, and 0.230 against 0.220
  for rook. That is the power check: a set that always covered by being
  wide would fail it.
- **`CS` says when `ρ` is not identified.** In the weak cells it is 1.0–1.5
  long out of 1.98, and it reaches both ends of (−0.99, 0.99) in up to 9% of
  replications (kNN-20, n = 400, K/n = 0.30). `CS` was one interval in every
  replication of every cell, although the proposition allows a union.
- **Non-normal errors.** With `χ²₃` errors, `CS` covers 0.938–0.981 in the
  weak cells and 0.945–0.961 in the controls. The lowest, 0.938, is
  kNN-20 at n = 400, K/n = 0.30, where `‖Cs‖_F²` = 1.5.

So `CS` should be reported with, or in place of, the Wald interval. It needs
no more computation than `V̂` at three vectors, and it is valid without A9.
The weak-identification diagnostic asked for in item 3 of "Still to do" is
then not needed for validity: a wide `CS` is the diagnostic.

**`CS` over the whole grid, with normal and `χ²₃` errors
(`tc_coverage_robust.jl`, argument `chisq`).** The robustness grid was rerun
with `CS` added, under both laws. With normal errors the Wald columns
reproduce the earlier run exactly, since the seeds are the same.

| errors | cells | Wald, HC2 | Wald, oracle | `CS`, HC2 |
|---|---|---|---|---|
| normal | 83 with `sd(ρ̂)` < 0.39 | 0.930–0.966 | 0.931–0.964 | 0.932–0.992 |
| normal | 7 with `sd(ρ̂)` ≥ 0.39 | 0.852–0.942 | 0.869–0.928 | 0.956–0.978 |
| `χ²₃` | 84 with `sd(ρ̂)` < 0.39 | 0.934–0.973 | 0.933–0.963 | 0.942–0.991 |
| `χ²₃` | 6 with `sd(ρ̂)` ≥ 0.39 | 0.841–0.914 | 0.845–0.918 | 0.938–0.970 |

- **Non-normal errors change nothing.** With `χ²₃` errors the Wald interval
  fails in the same six weakly identified cells, and `CS` covers 0.938 or
  more in all 90. HC2 stays within −2.3 to +3.2 points of the oracle. One
  cell had a negative `V̂` (2 of 1000; kNN-20, n = 400, K/n = 0.04).
- **`CS` never under-covers by more than 1.8 points.** Its lowest values,
  0.932 and 0.934 with normal errors, are at n = 1600 (rook, smooth
  variances, K/n = 0.05; kNN-6, rough, K/n = 0.11), where the Wald interval
  and the oracle are low by the same amount.
- **`CS` over-covers where the leverage is near 0.95.** At n = 1600, K/n =
  0.30, rough variances, the three `W` on random locations (`max hᵢᵢ` =
  0.949) give `CS` 0.978–0.992 under both laws; rook at the same K/n
  (`max hᵢᵢ` = 0.93) gives 0.959. The likely cause, not measured here, is
  HC2's upward bias, applied at every candidate `ρ`. It is conservative,
  not invalid.
- **Length.** In the identified cells the mean length of `CS` is 0.63–1.16
  times `2 · 1.96 · sd(ρ̂)`. Below 1 means `CS` is shorter than a Wald
  interval with the true sd. A likely reason, not checked, is heavy tails
  in `ρ̂`, which inflate its sd but not `CS`.

So `CS` covers at least as well as the Wald interval in every one of the
180 cells, within Monte Carlo error (it is at most 0.5 points lower, against
an SE of 0.7), and much better where `ρ` is weakly identified.

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
   loses up to 3.6 at K/n = 0.25, so HC2 stays the recommended `V̂`. Four
   `W`, n = 400–1600 and K/n up to 0.30 (`tc_coverage_robust.jl`, 90
   cells): HC2 is within −2.6 to +1.8 points of the oracle everywhere, and
   covers 0.930–0.966 wherever `sd(ρ̂)` < 0.39. The 7 failing cells are
   weak identification (a dense `W` at n = 400), where the oracle fails
   too. Proposition 6 gives the remedy: the test-inversion set `CS` is
   valid without A9 and covers 0.938–0.981 in those cells, with normal and
   `χ²₃` errors (`tc_ar.jl`), at no cost in length where identification is
   strong. Over the whole grid with normal and `χ²₃` errors (180 cells),
   `CS` covers 0.932–0.992, and over-covers (to 0.99) only where the
   leverage is near 0.95. Open: `CS` for K/n → θ > 0, which needs
   Proposition 2's open case.
4. Proposition 4(c) proves the smoother's `V̂` consistent. Choosing λ is
   measured, not solved (`tc_lambda.jl`): GCV over-smooths (coverage 0.888
   and 0.320), the RMSE-best λ is already too large, and a plug-in `ĉ_β` is
   too noisy (sd 0.8–2.9) to choose λ. Bias-corrected moments
   (`tc_rbc.jl`) give the guarantee but gain only 3–4% over the projection.
   A bias-aware interval over `F(ℛ) = {Bψ : ψ′Πψ ≤ ℛ²}` (`tc_honest.jl`)
   covers 0.954 and 0.964 at its chosen λ with `ℛ` true, and is 33–53%
   shorter than the projection's. Proposition 7 proves its coverage at each
   λ, at a chosen λ (under a separation condition, or with Bonferroni), and
   uniformly over `F(ℛ)`. Lemma 5 and Proposition 7′ cover `f ∉ span(B)`,
   with the class on a finer reference basis (`tc_refbasis.jl`): the
   released rule covers 0.050–0.973 at a coarse `B`, the new one
   0.960–1.000, at a price that is small only for a fine `B`. Still open:
   the joint bound of the linear and quadratic terms (a trust-region
   problem) in place of their sum, which would show how much of Lemma 5's
   looseness is the bound; REsd under Proposition 7′ (Proposition 1
   without A4); a sharper form of (iv)'s third condition;
   and reading the bias-aware-interval literature (Armstrong & Kolesár) to
   compare. "Choosing ℛ" (`tc_rchoice.jl`) answers how to pick `ℛ`: state
   it through a reference surface or a curvature bound, err large, and
   report the result across `ℛ`. The data cannot certify it (a remark,
   proved), and every plug-in rule tried under-covers.
5. Reading: de Jong (1987), Bhansali, Giraitis & Kokoszka (2007), Kelejian &
   Prucha (2001) and Hall & Heyde's §3.2 are read. Cattaneo, Jansson & Newey
   (2018) §3 and Kline, Saggio & Sølvsten (2020) §4.2 have been read via
   alphaXiv. Matsaev's theorem (1961b, Theorem 1) and the duality step for
   `p > 2` (Gohberg & Krein 1961, Theorem 2) were read on 2026-11-20; the
   proof for `1 < p < 2` is outlined in the note, not given in full. Von Bahr & Esseen's inequality (1965) and
   Rosenthal's inequality (1970) are used as they are usually stated, and
   have not been read. Theorem 1 needs Rosenthal's inequality for
   independent, mean-zero, not identically distributed terms (the `εⱼ` of
   a triangular array); that is the form to confirm.
6. Theorem 1 now needs `2 + δ` moments (done 2026-11-19). Consistency of
   `V̂` needs four (Proposition 2); eighth moments would be needed only for
   a rate or for the variance of `V̂`.
7. A9's sign is assumed, not derived. `J < 0` occurs in 0.7% of the hub
   designs of `tc_sign.jl` and in none of the kNN designs measured. A
   condition on `W` and `f` that rules it out is open.

## References

All have been checked to exist (via Scite, or by resolving the DOI), except
the two Matsaev notes and Gohberg & Krein (1961), which have no DOI; they
were checked against their Math-Net.Ru records (dan25286, dan25330,
dan24853), and all three are read. Read for this
note: de Jong (1987), §2 and §5, from the page
images of Springer's free copy; Cattaneo, Jansson & Newey (2018), §3 (arXiv
1507.02493); Kline, Saggio & Sølvsten (2020), §4.2 (arXiv 1806.01494); and,
from the page images of the copies in `ExtRef/`, Bhansali, Giraitis &
Kokoszka (2007), all of it; Kelejian & Prucha (2001), §3.1 and the proof of
Theorem 1; Hall & Heyde (1980), §3.2; and Lee (2004)'s CLT, as restated in
Lee (2023), ch. 1, §1.8.3 and Appendix 1A. The others have not been read.
- Anderson, T. W., & Rubin, H. (1949). Estimation of the parameters of a
  single equation in a complete system of stochastic equations. *Annals of
  Mathematical Statistics*, 20(1), 46–63. doi:10.1214/aoms/1177730090
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
- Stock, J. H., & Wright, J. H. (2000). GMM with weak identification.
  *Econometrica*, 68(5), 1055–1096. doi:10.1111/1468-0262.00151
- von Bahr, B., & Esseen, C.-G. (1965). Inequalities for the rth absolute
  moment of a sum of random variables, 1 ≤ r ≤ 2. *The Annals of
  Mathematical Statistics*, 36(1), 299–303. doi:10.1214/aoms/1177700291
- Gohberg, I. C., & Krein, M. G. (1961). On the theory of triangular
  representations of nonselfadjoint operators. *Doklady Akademii Nauk
  SSSR*, 137(5), 1034–1037 (in Russian). Math-Net.Ru dan24853. Read
  2026-11-20; its Theorem 2 is the duality step for `p > 2`.
- Matsaev (Macaev), V. I. (1961a). A class of completely continuous
  operators. *Doklady Akademii Nauk SSSR*, 139(3), 548–551 (in Russian).
  Math-Net.Ru dan25286. Read 2026-11-20. Cited by Bhansali et al. for the
  `𝔖_p` bound, which is not in it: its theorem is on `𝔖_ω`.
- Matsaev (Macaev), V. I. (1961b). Volterra operators obtained from
  self-adjoint operators by perturbation. *Doklady Akademii Nauk SSSR*,
  139(4), 810–813 (in Russian). Math-Net.Ru dan25330. Read 2026-11-20; its
  Theorem 1 is the `𝔖_p` bound used for Theorem 1 here.
- Kwapień, S., & Pełczyński, A. (1970). The main triangle projection in matrix
  spaces and its applications. *Studia Mathematica*, 34, 43–67.
  doi:10.4064/sm-34-1-43-67

Read, and held in `ExtRef/`: Jin & Lee (2012); Lu, Tian & Yang (2024).
