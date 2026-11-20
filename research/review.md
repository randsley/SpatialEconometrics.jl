Here are the concrete mathematical and technical points to tighten before formalizing the draft:

* **Martingale measurability in Theorem 1 (Step 2)**
* You write that by independence of $\varepsilon_i$ and $w_i$, $\mathbb{E}\vert{}\xi_i\vert{}^q = \mathbb{E}\vert{}\varepsilon_i\vert{}^q \cdot \mathbb{E}\vert{}b_i + 2w_i\vert{}^q$. State explicitly that $w_i = \sum_{j < i} \gamma_{ij}\varepsilon_j$ is $\mathcal{F}_{i-1}$-measurable and $\varepsilon_i$ is independent of $\mathcal{F}_{i-1}$ under A1, so $\mathbb{E}[\vert{}\xi_i\vert{}^q \mid \mathcal{F}_{i-1}] = \vert{}b_i + 2w_i\vert{}^q \mathbb{E}\vert{}\varepsilon_i\vert{}^q$ a.s., giving the unconditional product on taking expectations.




* **Rosenthal's inequality and the $\ell_q / \ell_2$ step in Theorem 1**
* In Step 2, you bound $\sum_j \vert{}\gamma_{ij}\vert{}^q \leq (\sum_j \gamma_{ij}^2)^{q/2}$ for $q \geq 2$. Note explicitly that $q > 2 \implies q/2 > 1$, so for non-negative terms $\gamma_{ij}^2$ the $\ell_{q/2} \leq \ell_1$ inequality gives $\sum_j (\vert{}\gamma_{ij}\vert{}^2)^{q/2} \leq (\sum_j \gamma_{ij}^2)^{q/2}$.


* Also check the constant in $\mathbb{E}\vert{}w_i\vert{}^q \leq \Theta_q (\bar{\mu} + \bar{\sigma}^q)\ell_i^q$: since $\mathbb{E}\vert{}\varepsilon_j\vert{}^q$ is bounded by $\bar{\mu}$ and $\sigma_j^q \leq \bar{\sigma}^q$ (given $\sigma_j^2 \leq \bar{\sigma}^2$ and $q/2 > 1$, so $\sigma_j^q = (\sigma_j^2)^{q/2} \leq \bar{\sigma}^q$), the factor $(\bar{\mu} + \bar{\sigma}^q)$ is clean—just state $\sigma_j^q \leq \bar{\sigma}^q$ explicitly.




* **Proposition 2, Step 3 (`ρ̂` replacing `ρ₀` in $\hat{V}$)**
* In bounding $\sum_i \hat{b}_i^2 \vert{}\hat{u}_i\vert{} \vert{}(\hat{u}_1)_i\vert{} / M_{ii}$, you invoke $\sum_i b_i^2 (\hat{u}_1)_i^2 \leq (\max_i b_i^2)\Vert{}\hat{u}_1\Vert{}^2 = o_p(n^2)$ by A6 and $\Vert{}\hat{u}_1\Vert{}^2 = O_p(n)$. Because A6 is assumed for $b = MP'f$ while $\hat{u}_1 = MWy = MW(I - \rho_0 W)^{-1}(f + \varepsilon)$, spell out why $\max_i b_i^2 / \Vert{}b\Vert{}^2 \to 0$ together with $\Vert{}b\Vert{}^2 = O(n)$ (or a bound on $\max_i b_i^2$) gives $\sum_i b_i^2 (\hat{u}_1)_i^2 = o_p(n^2)$—and show how $\hat{b}_i$ is related to $b_i$ in that product.




* **Proposition 4 & 4(c): eigenvalues of $S$, lower bound on $m_i$, and $\hat{V}$**
* In Proposition 4, $M = I - S$ with $S$ symmetric having eigenvalues in $[0, 1]$, $\max_i S_{ii} \leq \bar{s} < 1$, and $M \circ M$ invertible. If $S$ has an eigenvalue equal to $1$, $M$ is singular (still symmetric PSD with eigenvalues in $[0, 1]$, so $\Vert{}M\Vert{}_{\text{op}} \leq 1$ and $M^2 \preceq M$ hold), while invertibility of $M \circ M$ is assumed for $(M \circ M)d = -\operatorname{diag}(MW)$.


* In Proposition 4(c), $m_i = (M^2)_{ii} = \sum_k M_{ik}^2 \geq M_{ii}^2 = (1 - S_{ii})^2 \geq (1 - \bar{s})^2$ follows directly from the diagonal entry of $M^2$ without needing $\sum_{j \neq i} S_{ij}^2$.


* Check the noise term in Proposition 4: in the text of Proposition 4 it is written $\operatorname{diag}(H\hat{\Sigma}H')$ with $H = (A + A')S$, whereas in the notation table $H$ is defined as $(C + C')$ times the first stage's hat matrix and in `tc_cjn_lin.jl` $H = MWP_B$. Make the smoother's $H$ explicit in both the proposition and the notation table.




* **Proposition 5(a): `max_i \|We_i\|` vs. column-sum bound in A2**
* In the bound $\bar{d} \leq \max_i \Vert{}We_i\Vert{}\sqrt{\bar{h}} / ((1 - \bar{h})(1 - 2\bar{h}))$, A2 states each column has $\Vert{}We_i\Vert{} \leq c_W$, which immediately yields $\max_i \Vert{}We_i\Vert{} \leq c_W$. Writing $c_W$ directly (or noting $\max_i \Vert{}We_i\Vert{} \leq c_W$ from A2) keeps the bound purely in terms of stated constants.




* **Root selection, sign rule, and A9**
* The Corollary establishes the asymptotic normality of $\hat{t}$ for the **sign rule** ($g'(\hat{\rho}) < 0$, assuming $\liminf J/n > 0$), while noting that `cfe_root` in the code selects the root in $(-1, 1)$ closest to $\hat{u}_1'\hat{u}_0 / \hat{u}_1'\hat{u}_1$ and that evaluating $-g'$ requires a preliminary consistent estimator rather than evaluating at the candidate roots themselves (where $g' = \pm\sqrt{\Gamma}$). Keep this distinction sharp, and flag that in `tc_resd.jl` the numerical table reports coverage with *"the code's root rule"* rather than the sign rule.




* **Unverified external inequalities**
* In "Still to do" (item 5) and "References," von Bahr & Esseen (1965) and Rosenthal (1970) are noted as not yet read, and Anderson & Rubin (1949) and Stock & Wright (2000) are cited from standard use. Reading them—especially to confirm exact conditions/constants for triangular arrays or non-identically distributed mean-zero summands in Rosenthal's inequality—is the remaining verification item on the math side.