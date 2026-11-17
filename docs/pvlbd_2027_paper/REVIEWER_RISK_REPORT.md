# Post-Revision Mathematical, Statistical, and Formatting Audit Report (PVLDB 2027)

- **Repository**: `thy919100815-art/private-llm-analytics`
- **Baseline Audited**: Draft PR #8 (`c8d3446`)
- **Revised Branch**: `fix/pvlbd-2027-math-orbits-and-vldb-template`
- **Audit Scope**: Post-revision verification of `P0-A`, `P0-B`, `P0-C`, `P0-D`, `P1`, and `P2`.

---

## 1. Summary of Issues Found in Baseline (`c8d3446`) and Post-Revision Status

| ID | Severity in `c8d3446` | Issue Description | Resolution in Current Revision | Post-Revision Status |
| :--- | :--- | :--- | :--- | :--- |
| **P0-A** | **Critical** | `main.tex` Lemma 1 used `Original_Planner` $\bar{\rho}(1)=0.01419861$ ($\Delta\varepsilon \le 3.22\times 10^{-11}$) while Table 1 listed `Week1_v2` $\bar{\rho}(1)=0.008943$ without reconciling the global minimum across all curves and horizons. | Verified all source-code curves in `calendar_accounting.py`: `Original_Planner` $\bar{\rho}_{\min} = \bar{\rho}(1) = 0.01419861$ ($\Delta\varepsilon \le 3.2193 \times 10^{-11}$) and `Week1_v2` $\bar{\rho}_{\min} = \bar{\rho}(1) = 0.01746890$ ($\Delta\varepsilon \le 2.9122 \times 10^{-11}$). Added `commitment_curve_drift_summary()` in `ledger.py` and unified tangent + non-singular root bound in Lemma 2. | **RESOLVED (Verified by Unit Tests)** |
| **P0-B** | **Critical** | `main.tex` Table 1 mixed obsolete Week-1 draft numbers ($\varepsilon=(0.70, 1.08, 1.49, 2.14, 4.29)$) in the `Week1_v2` main row with footnote secant slopes computed from actual code ($\varepsilon=(1.0, 1.2, 1.5, 2.25, 4.4)$), and Table 2 used hand-copied budget figures. | Rebuilt Table 2 (`\label{tab:commitments}`), Table 3 (`\label{tab:planner_benchmark}`), and Table 6 (`\label{tab:orbits_summary}`) directly from `calendar_accounting.py` and `JointCalendarPlanner` outputs. Zero contradiction between rows, footnotes, and code. | **RESOLVED (Verified by Unit Tests)** |
| **P0-C** | **Critical** | Paper and `JointCalendarPlanner` claimed "210 canonical phase equivalence classes" by fixing only $\phi_7=0$ in $\mathbb{Z}_7 \times \mathbb{Z}_7 \times \mathbb{Z}_{30}$, overlooking that $\gcd(7,30)=1$ implies a $30\times$ redundancy under $\mathbb{Z}_{210}$. | Proved Proposition 1 via Orbit–Stabilizer and CRT ($\delta = (120\phi_7 + 91\phi_{90}) \bmod 210$): all $1{,}470$ phases partition into **7 canonical orbits of size 210** represented by $\{(0, (\phi_{30}-\phi_7)\bmod 7, 0)\}$. Updated `JointCalendarPlanner` (`use_canonical_7_orbits=True`), cutting Group D solve time by $19\times\text{--}23\times$ (`0.050s–0.081s`). Documented all 3 symmetry-breaking conditions. | **RESOLVED (Proved & Benchmarked)** |
| **P0-D** | **Critical** | `main.tex` used `\documentclass[10pt,twocolumn]{article}` instead of the official 2027 VLDB template and had Overfull wide tables. | Migrated to official 2027 VLDB template (`acmart.cls` v2.19 + `pvldb.sty` v1.0 + `ACM-Reference-Format.bst` + `\vldbtopmatter`) with prominent single-blind author/affiliation placeholders. Compiled `main.pdf` is **9 pages total** with **0 LaTeX errors, 0 undefined citations, and 0 Overfull warnings**. | **RESOLVED (Compiled & Verified)** |
| **P1-A** | **Major (Open Reviewer Risk)** | Main 420-day experiments use synthetic enterprise telemetry due to `DATA_APPROVAL.md` governance restriction; reviewers may ask for public real-world LLM log validation. | Explicitly documented in Section 6.1 of `main.tex` with a concrete, reproducible 420-day experimental protocol on public **WildChat-1M** (`zhao2024wildchat`). | **OPEN EXTERNAL BENCHMARK RISK (Protocol Specified)** |
| **P1-B** | **Major (Open Reviewer Risk)** | Continual additive baselines (`Tree-2l`, `Tree-U`, `W-BD-30`) do not include multi-horizon $b$-banded / Toeplitz matrix factorization (`choquette2023banded`, `henzinger2024group`, `mcmahan2025blt`). | Explicitly documented in Table 1 and Section 6.2 of `main.tex` with a concrete $b$-banded lower-triangular factorization protocol calibrated to $\bar{\rho}_{\text{add}}(h)$ across $h \in [1, 575]$. | **OPEN EXTERNAL BENCHMARK RISK (Protocol Specified)** |
| **P1-C** | **Minor (Open Reviewer Risk)** | Fixed $\ell_2$ clipping ($\Delta_2 = \sqrt{11}$) on multi-cell window distinct indicators causes $\approx -40\%$ negative clipping bias on 30d/90d windows, causing lower Gaussian variance in Group D to slightly increase unclipped Distinct30 MAPE ($+4.03\%$). | Fully explained analytically (via truncated normal expectation $\mathbb{E}[\max(0, Y)]$) and empirically in Section 5.2 and Table 4 without hiding negative results. | **TRANSPARENTLY DISCLOSED** |

---

## 2. Post-Revision Audit Counts (Critical / Major / Minor)

- **Mathematical & Code-Consistency Critical Issues**: **0** (all `P0-A`, `P0-B`, `P0-C`, `P0-D` defects reproduced, proved, and resolved).
- **Formatting & Template Critical/Minor Issues**: **0** (`acmart` + `pvldb.sty` v1.0 official 2027 VLDB template; 9 pages $\le 12$ pages; 0 Overfull boxes; 0 undefined citations).
- **Open Experimental / Pre-Submission Risks (Major)**: **2**
  1. Execution of the Section 6.1 public real-data replay protocol (WildChat-1M 420-day replay) if human reviewers require non-synthetic 420-day traces beyond the frozen benchmark suite;
  2. Execution of the Section 6.2 $b$-banded / BLT matrix factorization baseline on the additive channel;
  3. Completion of real single-blind author names and institutional affiliations in `\author{...}` / `\affiliation{...}` by the human authors prior to camera-ready/submission upload.

> **Mandatory Scientific Integrity Disclaimer**:
> 1. All audits in this repository are automated/agentic technical verifications and **do not constitute external independent human peer review**.
> 2. Accordingly, we **do not declare unconditional "Submission Ready" status**: final PVLDB 2027 submission requires human author sign-off, completion of the single-blind author metadata placeholders in `main.tex`, and human judgment on whether to execute the WildChat-1M and $b$-banded MF protocols prior to submission.

---

## 3. Revision Diff Summary and Evidence Traceability Map

| Deliverable / Claim | Source Code / Data Evidence | Test / Verification Script | Manuscript Location (`docs/pvlbd_2027_paper/main.tex`) |
| :--- | :--- | :--- | :--- |
| **P0-A**: Unified $\bar{\rho}_{\min}$ & $\varepsilon$-drift bound | `src/private_llm_analytics/sdg_mh/ledger.py` (`conservative_epsilon_drift_bound`, `commitment_curve_drift_summary`) | `tests/sdg_mh/test_joint_planner.py::test_p0a_p0b_commitment_curve_exact_params_and_drift_bounds` | Section 3.3, Lemma 2 (`\label{lem:drift}`), Eq. (3), Table 2 |
| **P0-B**: Exact `Original_Planner` & `Week1_v2` parameters | `src/private_llm_analytics/sdg_mh/calendar_accounting.py` (`ORIGINAL_PLANNER_CURVE`, `WEEK1_V2_CURVE`) | `tests/sdg_mh/test_joint_planner.py::test_p0a_p0b_commitment_curve_exact_params_and_drift_bounds` | Section 3.3, Table 2 (`\label{tab:commitments}`), Section 4 |
| **P0-C**: 7 Canonical Phase Orbits via CRT & Orbit–Stabilizer | `src/private_llm_analytics/sdg_mh/joint_planner.py` (`CANONICAL_7_ORBIT_REPRESENTATIVES`, `canonicalize_phase_orbit`, `use_canonical_7_orbits`); `results/sdg_mh/canonical_7_orbits_benchmark/canonical_7_vs_210_benchmark.csv` | `tests/sdg_mh/test_joint_planner.py::test_p0c_crt_7_canonical_orbits_partition_1470_phases_and_match_210_slice`, `test_p0c_pinned_phase_breaks_z210_orbit_symmetry` | Section 5.1, Proposition 1 (`\label{prop:orbits}`), Eqs. (6)–(8), Table 3, Appendix A.1, Table 6 |
| **P0-D**: Official 2027 VLDB Template (`acmart` + `pvldb.sty`) | `docs/pvlbd_2027_paper/acmart.cls`, `pvldb.sty`, `ACM-Reference-Format.bst`, `main.tex`, `main.pdf` | `pdflatex` + `bibtex` (9 pages, 0 errors, 0 Overfull boxes) | `docs/pvlbd_2027_paper/main.tex` & `main.pdf` |
| **P1**: 4-Way Novelty Matrix & WildChat / Banded-MF Protocols | `docs/pvlbd_2027_paper/NOVELTY_AND_PRIOR_ART_AUDIT.md` | Verified against bibliography (`references.bib`) | Table 1 (`\label{tab:novelty_matrix}`), Section 2, Section 6.1–6.3 |
| **Frozen Experimental Data Integrity** | `results/calendar_planning/`, `results/sdg_mh/long_horizon_420d*` | `git diff` confirms 0 modifications to frozen result directories | Section 5, Tables 4–5 |
