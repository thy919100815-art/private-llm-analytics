# SDG-MH Prior-Art Audit, 4-Way Novelty Attribution, and P0–P2 Mathematical Verification Report

- **Repository**: `thy919100815-art/private-llm-analytics`
- **Audit Baseline**: Draft PR #8 (`paper/pvlbd-2027-prior-art-and-manuscript`, SHA `c8d3446`)
- **Revision Branch**: `fix/pvlbd-2027-math-orbits-and-vldb-template`
- **Target Venue**: PVLDB Volume 20 (VLDB 2027, Official `acmart` + `pvldb.sty` Template)

---

## 1. Independent Reproduction and Verification of P0-A, P0-B, P0-C, P0-D

### 1.1 P0-A: Unified $\bar{\rho}_{\min}$ and $(\varepsilon, \delta)$ Drift Bound (Lemma 1)
- **Issue Reproduced in PR #8 (`c8d3446`)**:
  In `c8d3446`, `main.tex` Lemma 1 computed the $\varepsilon$-drift bound from `Original_Planner` $\bar{\rho}(1) = 0.01419861$ ($\Delta\varepsilon \le 3.22 \times 10^{-11}$), while Table 1's main row for `Week1_v2` listed $\bar{\rho}(1) = 0.00894312$ ($\varepsilon(1)=0.70$), which would correspond to a slightly larger tangent drift $\Delta\varepsilon \le 4.0273 \times 10^{-11}$.
- **Independent Source-Code Verification (`calendar_accounting.py` lines 174–186, `ledger.py`)**:
  1. In actual source code (`src/private_llm_analytics/sdg_mh/calendar_accounting.py`):
     - `ORIGINAL_PLANNER_CURVE`: `horizons=(1, 7, 30, 90, 365)`, `epsilons=(0.90, 1.08, 1.49, 2.24, 4.39)`, `delta=1e-6` $\implies \bar{\rho} = (0.01419861, 0.02031995, 0.03814349, 0.08410616, 0.30235637)$. Since $\bar{\rho}(h)$ is strictly increasing in $h \ge 1$ ($\beta_{\text{tail}} = 0.00079364 > 0$), $\min_{h \ge 1} \bar{\rho}_{\text{Orig}}(h) = \bar{\rho}_{\text{Orig}}(1) = 0.01419861$, giving tangent drift bound:
       $$\Delta\varepsilon \le \left(1 + \sqrt{\frac{\ln(10^6)}{0.01419861}}\right) \cdot 10^{-12} = 3.219325 \times 10^{-11}.$$
     - `WEEK1_V2_CURVE`: `horizons=(1, 7, 30, 90, 365)`, `epsilons=(1.00, 1.20, 1.50, 2.25, 4.40)`, `delta=1e-6` $\implies \bar{\rho} = (0.01746890, 0.02498393, 0.03864426, 0.08483138, 0.30364628)$. Its minimum across all $h \ge 1$ is $\bar{\rho}_{\text{W1v2}}(1) = 0.01746890$, giving:
       $$\Delta\varepsilon \le \left(1 + \sqrt{\frac{\ln(10^6)}{0.01746890}}\right) \cdot 10^{-12} = 2.912229 \times 10^{-11}.$$
  2. Therefore, across **all commitment curves and all horizons $h \ge 1$ in actual source code**, the global minimum is $\bar{\rho}_{\min} = \bar{\rho}_{\text{Orig}}(1) = 0.01419861$, and the unified drift bound is $\Delta\varepsilon \le 3.2193 \times 10^{-11}$.
  3. Furthermore, Lemma 1 and `UnifiedPrivacyLedger.commitment_curve_drift_summary()` now explicitly report:
     - Legacy draft knot $\varepsilon(1)=0.70$ ($\bar{\rho}=0.00894312$): $\Delta\varepsilon \le 4.0273 \times 10^{-11}$;
     - Smallest individual sub-channel (`calibration.py` additive sub-channel $\rho_{1,\text{add}} = 0.00030204$): $\Delta\varepsilon \le 2.1471 \times 10^{-10}$;
     - Solver box floor ($l_{\min} = 10^{-6}$): $\Delta\varepsilon \le 3.7184 \times 10^{-9}$;
     - Universal non-singular root bound valid for all $\rho \ge 0$:
       $$\Delta\varepsilon(\rho; \tau_{\text{tol}}) \le \min\!\left\{\left(1 + \sqrt{\frac{\ln(1/\delta)}{\rho}}\right)\tau_{\text{tol}},\;\; \tau_{\text{tol}} + 2\sqrt{\tau_{\text{tol}}\ln(1/\delta)}\right\} \le 7.4338 \times 10^{-6}.$$

### 1.2 P0-B: Resolution of Table 1 `Week1_v2` Main Row vs. Footnote Contradiction
- **Issue Reproduced in PR #8 (`c8d3446`)**:
  In `c8d3446`, Table 1 listed `Week1_v2` main-row values from an obsolete Week-1 markdown draft ($\varepsilon = (0.70, 1.08, 1.49, 2.14, 4.29)$, $\bar{\rho} = (0.008943, 0.020320, 0.038143, 0.076999, 0.289614)$, $\beta_{\text{tail}} = 0.00077315$), while its footnote reported secant slopes $s_{30\to 90} = 0.00076979$ and $s_{90\to 365} = 0.00079569$ computed from the actual code curve `WEEK1_V2_CURVE` ($\varepsilon = (1.0, 1.2, 1.5, 2.25, 4.4)$).
- **Fix Applied**:
  Table 1 (`\label{tab:commitments}`) in `main.tex` now derives every value ($\varepsilon(h)$, $\bar{\rho}(h)$, adjacent secant slopes $\Delta\bar{\rho}/\Delta h$, $\beta_{\text{tail}}$, $a_{\text{tail}}$, $\bar{\rho}_{\min}$, and $\Delta\varepsilon$) directly from `ORIGINAL_PLANNER_CURVE` and `WEEK1_V2_CURVE` in `src/private_llm_analytics/sdg_mh/calendar_accounting.py`, verified by regression test `test_p0a_p0b_commitment_curve_exact_params_and_drift_bounds`.

### 1.3 P0-C: Symbolic Proof and Numerical Verification of 7 Canonical Phase Equivalence Classes (Refuting the 210-Class Claim)
- **Mathematical Proof (Proposition 1 in `main.tex` / `canonicalize_phase_orbit` in `joint_planner.py`)**:
  - The discrete phase space is $\Phi = \mathbb{Z}_7 \times \mathbb{Z}_7 \times \mathbb{Z}_{30}$ with $|\Phi| = 7 \times 7 \times 30 = 1{,}470$.
  - The common time-shift group $G = \mathbb{Z}_{210}$ ($P = \operatorname{lcm}(7, 7, 30) = 210$) acts on $\Phi$ by shifting the calendar origin by $\delta \in \mathbb{Z}_{210}$:
    $$\delta \cdot (\phi_7, \phi_{30}, \phi_{90}) = \bigl((\phi_7 - \delta) \bmod 7,\;\; (\phi_{30} - \delta) \bmod 7,\;\; (\phi_{90} - \delta) \bmod 30\bigr).$$
  - An element $\delta \in \mathbb{Z}_{210}$ stabilizes $(\phi_7, \phi_{30}, \phi_{90})$ iff $\delta \equiv 0 \pmod 7$ and $\delta \equiv 0 \pmod{30}$, which by $\operatorname{lcm}(7, 30) = 210$ holds iff $\delta \equiv 0 \pmod{210}$. Thus $\mathrm{Stab}_{\mathbb{Z}_{210}}(\boldsymbol{\phi}) = \{0\}$ for every $\boldsymbol{\phi} \in \Phi$.
  - By the **Orbit–Stabilizer Theorem**, every orbit has size $|\mathcal{O}(\boldsymbol{\phi})| = |G| / |\mathrm{Stab}(\boldsymbol{\phi})| = 210 / 1 = 210$, partitioning the $1{,}470$ phase space into **exactly $1{,}470 / 210 = 7$ canonical phase equivalence classes**.
  - By the **Chinese Remainder Theorem** ($\gcd(7, 30) = 1$, orthogonal idempotents $e_7 = 120 \equiv 1 \pmod 7,\, 120 \equiv 0 \pmod{30}$ and $e_{30} = 91 \equiv 0 \pmod 7,\, 91 \equiv 1 \pmod{30}$), for any $(\phi_7, \phi_{30}, \phi_{90}) \in \Phi$, the unique shift
    $$\delta(\boldsymbol{\phi}) = (120\,\phi_7 + 91\,\phi_{90}) \bmod 210$$
    satisfies $\delta(\boldsymbol{\phi}) \equiv \phi_7 \pmod 7$ and $\delta(\boldsymbol{\phi}) \equiv \phi_{90} \pmod{30}$, mapping $\boldsymbol{\phi}$ to its unique canonical representative:
    $$\boldsymbol{\phi}_{\mathrm{can}} = \bigl(0,\;\; (\phi_{30} - \phi_7) \bmod 7,\;\; 0\bigr) \in \mathcal{R}_7 := \bigl\{(0, d, 0) : d \in \{0, 1, 2, 3, 4, 5, 6\}\bigr\}.$$
  - Consequently, the 210-tuple slice $\{(0, \phi_{30}, \phi_{90}) : \phi_{30} \in \mathbb{Z}_7, \phi_{90} \in \mathbb{Z}_{30}\}$ only quotiented out $\mathbb{Z}_7$ instead of $\mathbb{Z}_{210}$ and repeated each of the 7 canonical orbits **30 times** (via $\delta = 91\phi_{90} \bmod 210$).
- **When $\mathbb{Z}_{210}$ Equivalence Is Broken**:
  1. External fixed calendar-date / weekday queries (e.g., Sunday-pinned executive queries $q \equiv 6 \pmod 7$, where staleness $\tau_f(q; \phi_f) = (q - \phi_f) \bmod s_f$ is not shift-invariant);
  2. Origin-anchored staleness penalties ($\lambda_{\text{stale}} > 0$) or finite startup-day utility weighting;
  3. Pinned phase business constraints (`fixed_phi_7`, `fixed_phi_30`, `fixed_phi_90`).
- **Numerical & Runtime Verification (`results/sdg_mh/canonical_7_orbits_benchmark/canonical_7_vs_210_benchmark.csv`)**:
  Across all 4 `(curve, workload)` pairs, 7-orbit canonical search (`use_canonical_7_orbits=True`) produces **identical** optimal phase, budget vector, primal objective, and `DECIMAL100` dual lower bound as 210-tuple enumeration (`use_canonical_7_orbits=False`), while cutting Group D solve time by **$19.1\times\text{--}23.2\times$** (`0.050s–0.081s` vs. `0.952s–1.701s`).

### 1.4 P0-D: Official VLDB 2027 Template Compliance (`https://github.com/vldbproceedings/VLDB-Template`)
- Replaced `\documentclass[10pt,twocolumn]{article}` with the official VLDB 2027 template files (`acmart.cls` v2.19, `pvldb.sty` `[2026/07/06 v1.0 VLDB 2027 / PVLDB Vol. 20 formatting overrides]`, `ACM-Reference-Format.bst`):
  - `\documentclass[sigconf, nonacm]{acmart}`
  - `\usepackage{pvldb}`
  - `\maketitle` + `\vldbtopmatter`
  - `\bibliographystyle{ACM-Reference-Format}`
  - Single-blind author/affiliation block using explicit placeholder fields (`[Author Name(s) Placeholder -- To Be Completed Prior to Submission]`, `[Institution / Department Affiliation Placeholder]`) without fabricating any person or organization.
  - Compiled PDF (`docs/pvlbd_2027_paper/main.pdf`) is **9 pages total** (including Appendix and References, strictly $\le 12$ pages) with **0 LaTeX errors, 0 undefined citations, and 0 Overfull `\hbox`/`\vbox` warnings**.

---

## 2. Prior-Art Comparison Table (Verified Literature Through 2026)

| Reference | Privacy Definition & Unit | Horizon Scope | Query / Output Semantics | Error Bound | Computational Cost | Relation to SDG-MH |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Dwork et al. (STOC 2010)** `10.1145/1806689.1806787`; **Chan et al. (TISSEC 2011)** `10.1145/2043621.2043626` | Event-level $(\varepsilon, 0)$ / $(\varepsilon, \delta)$ continual observation | Finite $T$ or logarithmic dyadic tree | Linear additive prefix sums $\sum_{t=1}^k x_t$ | $O(\varepsilon^{-1}\log^{1.5} T)$ | $O(\log T)$ per step | Cannot reconstruct rolling $W$-day distinct users from daily noisy leaves; violates multi-knot short-horizon $\bar{\rho}(1), \bar{\rho}(7)$ when calibrated only to $\bar{\rho}(365)$. |
| **Honaker (2015)** *Efficient Use of DP Binary Trees* | Event/window continual DP | Finite $T$ ($L = \lceil\log_2 T\rceil$) | Linear range/prefix sums via 2-level variance optimization | Constant-factor reduction over uniform tree | $O(\log T)$ per query | Included as `Tree-2l` baseline; outperforms SDG-MH on $h=365$ additive sums ($2.09\%$ vs $3.03\%$ MAPE) but $5\times$ worse at $h=1$ ($50.09\%$ vs $10.03\%$). |
| **Kellaris et al. (PVLDB 2014)** `10.14778/2732977.2732989` | $w$-event DP (single window $w$ over $t \in [1, \infty)$) | Infinite stream, single $w$ | Linear/spatial histograms per step (`W-BD`, `W-BA`) | $O(w / \varepsilon)$ under uniform division | $O(1)$ per step | Single-horizon $w$ only; SDG-MH generalizes $w$-event DP to a simultaneous multi-knot curve $\bar{\rho}(h)$ ($h \in \{1,7,30,90,365\}$ + affine tail) across mixed daily + periodic window releases. |
| **Bun & Steinke (TCC 2016)** `10.1007/978-3-662-53641-4_24` (arXiv:1605.02065) | Zero-Concentrated DP ($\rho$-zCDP) | Static / adaptive composition | Exact additive composition $\sum \rho_i$ for Gaussian mechanisms | Exact Gaussian variance $\Delta_2^2 / (2\rho)$ | $O(1)$ per composition | Direct mathematical foundation for SDG-MH's window-overlap accounting and Lemma 1 $(\varepsilon, \delta)$ conversion. |
| **Denisov et al. (NeurIPS 2022)** arXiv:2202.08312; **Fichtenberger et al. (ICML 2023)** arXiv:2202.11205 | Adaptive continual DP via matrix factorization $W = BC$ | Finite $T$ | Linear prefix / momentum queries | Optimal $O(\log T)$ matrix mechanism constants | $O(T^2)$ or Toeplitz $O(T \log T)$ | Linear additive operators only; does not handle non-additive periodic window distinct counts or multi-knot user-time curves. |
| **Choquette-Choo et al. (NeurIPS 2023)** arXiv:2306.08153; **Henzinger & Upadhyay (2024)** arXiv:2412.02840; **McMahan & Pillutla (2025)** arXiv:2504.21413 | $b$-Banded / Buffered Linear Toeplitz (BLT) matrix factorization | Finite or sliding $b$-banded stream | Linear streaming prefix sums | Near-optimal continual linear query error | $O(b)$ state / BLT streaming | State-of-the-art for continual linear additive queries; Section 6.2 of `main.tex` specifies the exact multi-horizon $b$-banded MF baseline protocol. |
| **Epasto et al. (ITCS 2023)** arXiv:2301.05605; **Ghazi et al. (ITCS 2023)** arXiv:2211.11718; **Jain et al. (NeurIPS 2023)** arXiv:2306.06723 | Continual distinct / $F_0$ estimation in sliding windows / turnstile streams | Finite $T$ | Streaming distinct counts ($F_0$) | Multiplicative / polylog $T$ additive error | Sketch-based | Theoretical streaming sketches for single-metric distinct estimation; SDG-MH instead uses exact window-level Gaussian releases scheduled via a periodic calendar under multi-horizon zCDP. |

---

## 3. Strict 4-Way Novelty Classification (P1 Requirement)

1. **(I) New Privacy Problem Formulation**:
   - *Multi-Horizon User-Time zCDP with Window-Support Overlap Charging*: Simultaneous enforcement of $\sup_{s \ge 0} \sum_{r : B_r \cap [s, s+h-1] \neq \emptyset} \rho_r \le \bar{\rho}(h)$ across a multi-knot curve $h \in \{1, 7, 30, 90, 365\}$ and an affine long-term rate $\beta_{\text{tail}}$ for $h \ge 365$, unifying daily additive sealing and periodic multi-window distinct-user releases.
2. **(II) New Mathematical Theorems & Propositions Specific to This Formulation**:
   - *Theorem 1 (210-Day Periodicity)* and *Theorem 2 (575-Day Affine-Tail Finite-to-Infinite Reduction)*: Exact reduction of infinite-horizon multi-knot verification over $(s, h) \in \mathbb{Z}_{\ge 0} \times \mathbb{Z}_{\ge 1}$ to $s \in [0, 209]$ and $h \in [1, 575]$ plus $\mathrm{slope}_{\infty}(\boldsymbol{v}) \le \beta_{\text{tail}}$, without requiring global concavity of $\bar{\rho}(h)$.
   - *Proposition 1 (Orbit–Stabilizer & CRT 7-Orbit Reduction)*: Proof that $\mathbb{Z}_{210}$ acts freely on $\Phi = \mathbb{Z}_7 \times \mathbb{Z}_7 \times \mathbb{Z}_{30}$ ($|\Phi| = 1{,}470$), partitioning $\Phi$ into **7 canonical orbits of size 210** with representatives $\{(0, (\phi_{30}-\phi_7)\bmod 7, 0)\}$ via $\delta = (120\phi_7 + 91\phi_{90}) \bmod 210$, eliminating the $30\times$ redundancy in naive $(0, \phi_{30}, \phi_{90})$ enumeration.
3. **(III) Direct Mathematical Identities / Known Tools (Must NOT Be Claimed as Standalone Theoretical Breakthroughs)**:
   - Standard Gaussian zCDP composition and zCDP-to-$(\varepsilon, \delta)$ conversion (Bun & Steinke 2016);
   - Elementary interval intersection bounds $k_{\min}, k_{\max}$ (Lemma 1);
   - Separable convex programming $\min \sum_j c_j/v_j$ over linear constraints $A\boldsymbol{v} \le \boldsymbol{b}$ and single-constraint KKT square-root allocation.
4. **(IV) System & Numerical Engineering Integration**:
   - Two-phase atomic ledger reservation (`RESERVED` $\to$ persist $\to$ `COMMITTED`);
   - Inward ray budget shrinkage $\boldsymbol{v}_{\text{cert}} = \gamma \hat{\boldsymbol{v}}$ to eliminate $\sim -3 \times 10^{-12}$ SLSQP floating-point violations;
   - 100-digit directed-rounding (`DECIMAL100_DIRECTED_ROUNDING`) Lagrangian dual lower-bound certification eliminating float64 catastrophic cancellation.
