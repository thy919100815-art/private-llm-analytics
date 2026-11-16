# Phase 3: Independent Two-Pass Read-Only PVLDB 2027 Reviewer Risk Report

- **Reviewed Manuscript**: `docs/pvlbd_2027_paper/main.tex` and compiled `docs/pvlbd_2027_paper/main.pdf` (12 pages, two-column PVLDB format).
- **Review Protocol**: Strictly **read-only** two-pass audit executed *after* compiling `main.pdf`, without modifying `main.tex` during or between the two passes so that both reviewers' independent findings can be compared directly.

---

## 1. Pass 1: Mathematical & Theoretical Correctness Review (Reviewer #1 — Theory & Privacy)

### Overall Assessment (Pass 1)
The mathematical formulation of Multi-Horizon User-Time zCDP, the closed-form data-window intersection count $n_f(s, h; \phi_f)$, the 210-day periodic shift identity, and the concavity-free 575-day finite-to-infinite reduction theorem are rigorous and self-contained. In particular, identifying that the canonical commitment curves (`Original_Planner` and `Week1_v2`) are *non-concave* in zCDP space (Proposition 1) and proving Theorem 3 using only the $h \ge 365$ affine tail eliminates a major mathematical flaw that would have invalidated concavity-based horizon reductions. Furthermore, replacing heuristic IEEE-754 `(1 - 1e-14)` dual deductions with 100-digit directed-rounding enclosure arithmetic (`DECIMAL100_DIRECTED_ROUNDING`, Theorem 4) and strictly separating `NEAR_GLOBAL_OPTIMAL_CERTIFIED` from symbolic `GLOBAL_OPTIMAL_PROVED` reflects high mathematical integrity.

### Categorized Findings (Pass 1)

#### Critical Issues (0)
- *None.* All theorem statements match their proofs, all assumptions (Assumption 1 A1–A3) are explicitly stated, and no unproved heuristic is claimed as a formal theorem.

#### Major Issues (3)
- **[M1.1] Table 1 (`tab:curves`) Anchor Value Discrepancy on `Week1_v2` at $h \in \{90, 365\}$**:
  - *Location*: Section 2.2, Table 1 (lines 125–128 of `main.tex`).
  - *Diagnosis*: The table body lists $\varepsilon(90) = 2.51 \mapsto \bar{\rho}(90) = 0.076306$ and $\varepsilon(365) = 4.41 \mapsto \bar{\rho}(365) = 0.295121$, while a footnote explains that the operational `WEEK1_V2_CURVE` code object uses $\bar{\rho}(90) = 0.084831$ and $\bar{\rho}(365) = 0.303646$ (yielding secant slopes $0.000770$ and $0.000796$). Note that both sets of values are non-concave, so Proposition 1 holds under either parameterization, but presenting two slightly different numerical anchor pairs in the same table can confuse a careful theory referee.
  - *Action for Next Revision*: Display the exact operational `WEEK1_V2_CURVE` $\bar{\rho}(h)$ values directly in the primary row of Table 1 and state their corresponding $\varepsilon(h)$ values ($2.660$ and $4.477$ at $\delta=10^{-6}$).
- **[M1.2] Conditional Scope of `DECIMAL100_DIRECTED_ROUNDING` Relative to `float64` Curve Inputs**:
  - *Location*: Section 5.2, Theorem 4 & Paragraph "Eliminating IEEE-754 Cancellation" (lines 321–342).
  - *Diagnosis*: `certify_lagrangian_dual_bound_exact()` converts the operational `float64` RHS array `b = self.rho_bars` via `Decimal.from_float(b[i])` (and lower-bounds rational $1/7, 1/30$ via `ROUND_FLOOR`). Thus, the 100-digit directed-rounding certificate is unconditional with respect to the *operational floating-point constraint system* enforced by `UnifiedPrivacyLedger`, rather than an interval enclosure of the transcendental map $\varepsilon \mapsto (\sqrt{\varepsilon + \ln(1/\delta)} - \sqrt{\ln(1/\delta)})^2$ from rational $\varepsilon$. While Section 8 (Limitation 4) and the v3 report state this condition, Theorem 4's surrounding text should state explicitly that $\mathbf{b}$ denotes the operational `float64` commitment vector.
- **[M1.3] Mixed Add/Remove vs.\ Bounded-Replacement Sensitivity Convention in Equation (11)**:
  - *Location*: Section 4.1, Equation (11) (lines 216–220).
  - *Diagnosis*: The sensitivity bound $\Delta_{\text{distinct}}^2 = m + 2k = 5 + 2(3) = 11$ combines $m = 5$ (maximum $\ell_2^2$ change in service indicators when adding/removing a user) with $2k = 6$ (maximum $\ell_2^2$ change in $k$-task indicators when replacing $k$ tasks with $k$ disjoint tasks; under pure add/remove of a user, at most $k$ task counts change by $+1$, which would give $m + k = 8$). Hence $\Delta^2 = m + 2k = 11$ is a strictly conservative upper bound under add/remove adjacency and exact under task-replacement adjacency.
  - *Action for Next Revision*: Add one sentence clarifying that $m + 2k$ covers worst-case replacement of a user's $k$-task set (and conservatively upper-bounds pure add/remove sensitivity $m + k = 8$).

#### Minor Issues (3)
- **[m1.1] Typographical Capitalization**: Section 3.2, line 194 reads `and Perturbs (\tilde{R}_t, \tilde{F}_t)` instead of `and perturbs`.
- **[m1.2] Explicit Hypothesis $W_f \ge S_f$ in Theorem 2**: The proof of Theorem 2(2) (line 273) uses $\Delta_f(s) \le S_f - 1 \le W_f + h - 2$ for all $h \ge 1$, which relies on $W_f \ge S_f$ (satisfied by $(7,7), (30,7), (90,30)$). Including $W_f \ge S_f$ in Theorem 2's formal statement makes the theorem self-contained for arbitrary calendar families.
- **[m1.3] LaTeX Wide-Table Overflow**: In `main.log`, Table 4 (`tab:baselines_v3`, lines 471–482) has an overfull `\hbox` of `257.56pt`, and Appendix Table 5 (`tab:traceability`) has `7–43pt` overfull `\texttt` entries. Wrapping Table 4 in `\resizebox{\textwidth}{!}{...}` and adding `\sloppy` or line breaks in Table 5 will eliminate all layout warnings.

---

## 2. Pass 2: Novelty, Experimental Fairness & PVLDB Systems Value Review (Reviewer #2 — Systems & Experiments)

### Overall Assessment (Pass 2)
This manuscript stands out for its statistical rigor and scientific honesty. Unlike many privacy systems papers that cherry-pick favorable query windows or use pseudo-replicated confidence intervals, this paper uses $K = 8$ independent longitudinal populations ($df = 7$) with 128-bit collision-free Common Random Numbers (CRN), applies Holm--Bonferroni and exact Wilcoxon tests, unifies post-processing ($\hat{R} = \max(0, \tilde{R}),\, \hat{F} = \min(\hat{R}, \max(0, \tilde{F}))$) across all competitive baselines, and evaluates `BudgetOnly` vs.\ `JointPlanning` on identical query request dates (`common_daily_*` and `common_sunday_*`). Most importantly, the paper transparently reports negative and non-significant findings: `30d` distinct MAPE is statistically non-significant ($p_{\text{holm}} = 1.000$), Sunday-anchored queries incur a $+3.0\text{d}$ staleness lag under $\phi_{30}=4$, `DistinctHeavy` trades $+0.38\%\text{--}+0.51\%$ higher daily additive RMSE, and `Tree-2l` has a $0.38\%$ lower mean RMSE than `SDG-MH` on 90d Requests due to clipping-bias convergence.

### Categorized Findings (Pass 2)

#### Critical Issues (0)
- *None.* Prior art is properly credited (daily Gaussian and Cauchy--Schwarz splitting are attributed to classical literature), "No Epochs, No Expiry" is explicitly distinguished from constant lifetime privacy, and competitive baseline claims are strictly confined to the shared $[1, 420]$ additive workload.

#### Major Issues (3)
- **[M2.1] Synthetic-Only Telemetry in the 420-Day End-to-End Benchmark**:
  - *Location*: Section 6.1–6.5.
  - *Diagnosis*: While the 420-day synthetic generator models workforce turnover, weekday/weekend dips ($0.25\times$), seasonal waves, and heavy-tailed LogNormal request distributions across $N_0 = 2,000$ users, PVLDB reviewers strongly prefer seeing at least one real-world trace (e.g., the 845-day WildChat dataset or a production enterprise trace) evaluated through the exact same `JointCalendarPlanner` and `ReadOnlySealedQueryService` pipeline.
- **[M2.2] Empirical Omission of Toeplitz / Matrix Factorization Baselines in Table 4**:
  - *Location*: Section 6.5, Table 4 (`tab:baselines_v3`).
  - *Diagnosis*: Sections 3.2 and 7 provide a sound theoretical explanation of why Toeplitz/banded matrix factorizations~\cite{denisov2022improved,fichtenberger2023constant,choquette2023banded} suffer cross-day sensitivity penalties under linear multi-horizon tails, and Table 4 includes `Tree-2l` and `Tree-U` with exact Honaker GLS estimation~\cite{honaker2015efficient} (which is itself a structured matrix mechanism on the dyadic tree). However, reviewers familiar with ICML/NeurIPS 2023–2025 streaming DP literature may still ask for an explicit lower-triangular Toeplitz baseline row in Table 4.
- **[M2.3] Modest Empirical Magnitude of Phase Optimization (`D` vs.\ `B`) Relative to Budget Optimization (`B` vs.\ `A`)**:
  - *Location*: Section 6.2–6.3, Tables 2 & 3.
  - *Diagnosis*: Moving from naive uncalibrated/aligned Group A to budget-optimized Group B reduces workload objective $f^*$ by $\sim 30\%$, whereas moving from Group B ($\boldsymbol{\phi}=(0,0,0)$) to Joint Planning Group D ($\boldsymbol{\phi}=(0,4,0)$) reduces $f^*$ by $1.04\%\text{--}1.12\%$ on `DistinctHeavy` and $0\%$ on `Default`. While the $1.58\text{s}$ offline solve time of Group D is negligible (making Group D strictly "free" to run before deployment), reviewers may view phase shifting as a secondary refinement compared to continuous convex budget allocation.

#### Minor Issues (2)
- **[m2.1] Single-Process Execution Scope**: Section 2.1 and Section 8 note that `UnifiedPrivacyLedger` and `RollingDistinctState` are validated within a single-node process; distributed consensus across sharded ingestion workers is not evaluated.
- **[m2.2] Fixed Top-$k$ Cap ($k=3$)**: Section 4.1 fixes $k=3$ tasks per user without an ablation over $k \in \{1, 3, 5, 10\}$ showing the sensitivity $\Delta^2 = m + 2k$ vs.\ truncation recall trade-off.

---

## 3. Side-by-Side Comparison of Pass 1 and Pass 2 Conclusions

| Dimension | Pass 1 (Mathematical & Theoretical Audit) | Pass 2 (Novelty, Fairness & PVLDB Systems Audit) | Convergence & Readiness Verdict |
|---|---|---|---|
| **Critical Flaws** | **0** (All proofs valid; non-concavity handled via affine tail; `DECIMAL100` directed rounding sound) | **0** (No overclaiming of known primitives; negative/non-significant results fully disclosed) | **Ready**: Zero blocking mathematical or integrity defects |
| **Primary Strengths** | Concavity-free 575d infinite reduction (Thm. 3); $5\times$ intersection counterexample (Thm. 1); 100-digit dual certificate (Thm. 4) | $K=8$ stratified CRN design ($df=7$); 128-bit collision-free seeds; same-query-date staleness separation; uniform post-processing | Strong alignment between theoretical guarantees and experimental protocol |
| **Top Major Risks** | M1.1 (`Week1_v2` Table 1 footnote clarity); M1.2 (`float64` RHS condition in Thm. 4); M1.3 ($m+2k$ replacement vs. add/remove note) | M2.1 (Synthetic-only 420d evaluation); M2.2 (No Toeplitz row in Table 4); M2.3 ($1.04\%\text{--}1.12\%$ modest phase gain vs. Group B) | M1.1–M1.3 are 1-paragraph text clarifications; M2.1–M2.3 are inherent scope trade-offs transparently documented in Section 8 |
| **Minor Polish Items** | m1.1 (`Perturbs` typo); m1.2 ($W_f \ge S_f$ in Thm. 2); m1.3 (Table 4/5 `\hbox` width) | m2.1 (Single-process scope); m2.2 (Fixed $k=3$ task cap) | Easily addressed in camera-ready formatting |
