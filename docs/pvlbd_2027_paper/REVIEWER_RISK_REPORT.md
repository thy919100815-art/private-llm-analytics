# Post-Revision Mathematical, Evidence-Consistency, and Formatting Audit Report (PVLDB 2027)

- **Repository**: `thy919100815-art/private-llm-analytics`
- **Baselines Audited**: Draft PR #8 (`c8d3446`) and Draft PR #9 (`5d50deb`)
- **Revised Branch**: `fix/pvlbd-2027-v3-evidence-and-pinned-orbits`
- **Audit Scope**: Full verification of v3 evidence restoration, pinned-phase residual symmetry subgroup completeness, 7-orbit CRT reduction, and official 2027 VLDB template compliance.

---

## 1. Resolution of All Evidence-Consistency and Pinned-Phase Search Defects

| ID | Severity in `5d50deb` | Defect Description | Exact Fix & Verification in Current Branch | Status |
| :--- | :--- | :--- | :--- | :--- |
| **P0-1 (v3 Evidence Restoration)** | **Critical** | `5d50deb` inadvertently replaced PR #8 (`c8d3446`)'s authentic v3 experimental sections (`2,000` users, `420` days, `K=8` seeds $\times$ `5` CRN trials, $m=5$ services, $|\mathcal{A}|=20$ tasks, $k=3$ top tasks, 128-bit seed packing) with unverified summary numbers (`180 users`, `9 cells`, `10 seeds x 10 MC`, `35%–44% 30d Distinct MAPE`, `20.05% variance drop`, `40% distinct clipping bias`). | Restored the authentic v3 system definitions, threat model, Discrete Gaussian equations, 128-bit CRN seed audit (`50,520` unique seeds, `0` collisions), and all v3 tables/figures (`Table 3`, `Table 4`, `Table 5`, `Figs. 1–4`) from PR #8 (`c8d3446`) while retaining PR #9's verified mathematical fixes (`P0-A..D`). | **RESOLVED (`174/174` Provenance Checks Passed)** |
| **P0-2 (Pinned-Phase Search Completeness)** | **Critical** | `plan_group_c()` and `plan_group_d()` previously used `p7_candidates = [0]` whenever `fixed_phi_7 is None`, even when `fixed_phi_30` was pinned. Counterexample: for `fixed_phi_30=0, fixed_phi_90=0, fixed_phi_7=None`, `(3,0,0)` is in the same $\mathbb{Z}_{210}$-orbit as `(0,4,0)` and achieves $f^* = 922.9791 < 932.6324$, which was missed by `p7_candidates=[0]`. | Implemented `build_admissible_phase_space()` in `joint_planner.py`, decomposing $G_{\text{res}} = G_7 \oplus G_{30} \le \mathbb{Z}_{210}$ where $G_7 = 30\mathbb{Z}_{210} \cong \mathbb{Z}_7$ iff **both** $\phi_7$ and $\phi_{30}$ are unpinned, and $G_{30} = 7\mathbb{Z}_{210} \cong \mathbb{Z}_{30}$ iff $\phi_{90}$ is unpinned. Added `quotient_orbits_complete` guard so incomplete searches are downgraded to `FEASIBLE_CERTIFIED` (`INCOMPLETE_PHASE_SUBSET`). | **RESOLVED (All 8 Pinning Patterns Verified)** |
| **P0-3 (Baseline Calibration Fairness)** | **Critical** | `5d50deb` reported uncalibrated `Tree-2l`, `Tree-U`, and `W-BD-30` baselines that violated short/long horizons. | Restored v3's fair competitive baseline table (`Table 5`, `audit_v3_competitive_baselines_summary.csv`) where all methods are calibrated to the exact remaining additive envelope $\bar{\rho}_{\text{add}}(h)$ across **all** $h \in [1, 420]$ with identical window-level post-processing $\hat{R}=\max(0,\tilde{R}),\, \hat{F}=\min(\hat{R},\max(0,\tilde{F}))$. | **RESOLVED** |
| **P0-4 (Automated Provenance Gate)** | **Critical** | Lack of automated check preventing unverified or mismatched numbers from entering `main.tex`. | Created `scripts/verify_manuscript_csv_provenance.py` and unit test `test_p0_manuscript_to_csv_provenance_verifier` verifying `174/174` checks across source code and frozen v1/v3 CSVs and banning non-v3 tokens. | **RESOLVED (`174/174` Passed)** |
| **P0-A/B/C/D (Retained from PR #9)** | **Critical (in `c8d3446`)** | Unified $\bar{\rho}_{\min}$ & $\varepsilon$-drift bound (`Lemma 1`), exact source-code `Original_Planner` & `Week1_v2` parameters (`Table 2`), 7-orbit CRT reduction (`Proposition 2`), and official 2027 VLDB `acmart` + `pvldb.sty` template. | Fully preserved in `main.tex` and `joint_planner.py`/`ledger.py`. Compiled `main.pdf` is **12 pages total** (including Appendix and References, $\le 12$ pages) with **0 LaTeX errors and 0 Overfull boxes**. | **RESOLVED** |

---

## 2. Post-Revision Audit Counts (Critical / Major / Minor)

- **Evidence Consistency, Mathematical & Code-Consistency Critical Issues**: **0** (all blocking evidence-consistency and pinned-phase search completeness defects are eliminated and gated by `174/174` automated provenance checks + `17/17` planner tests).
- **Formatting & Official VLDB 2027 Template Issues (Critical / Major / Minor)**: **0 / 0 / 0** (`\documentclass[sigconf, nonacm]{acmart}` + `\usepackage{pvldb}`, prominent single-blind author/affiliation placeholders, 12 pages total including Appendix & References, 0 LaTeX errors, 0 Overfull `\hbox`/`\vbox` warnings).
- **Open Pre-Submission / External Validation Items (Major)**: **2** (evaluated separately from current frozen v3 evidence):
  1. Optional external real-world telemetry validation (e.g., 420-day replay on public WildChat-1M logs) if human reviewers request non-synthetic longitudinal traces;
  2. Optional multi-horizon $b$-banded / BLT matrix factorization baseline comparison on the additive channel;
  3. Replacement of single-blind author/affiliation placeholders with real author metadata by the human authors prior to submission.

> **Scientific Integrity Statement**:
> Automated verification scripts and agentic audits verify internal mathematical proofs, code-to-CSV provenance (`174/174` checks), and LaTeX formatting compliance, but **do not substitute for independent external human peer review**. Accordingly, we do **not** declare unconditional "Submission Ready" status prior to human author review and metadata completion.
