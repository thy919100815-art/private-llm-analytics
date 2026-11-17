# SDG-MH Pre-Freeze Engineering & Experimental Evidence Audit Report (v6)

**Baseline Audited**: Draft PR #12 (`b8295243`, `fix/pvlbd-2027-v5-audit-remediation`)  
**Scope**: Reproducibility, timing boundaries, calendar origin alignment, real-data statistical scope, and manuscript provenance verification. No core privacy theory rewritten; no frozen v1–v5 CSVs modified.

## 1. P0 Public Reproducibility (Fresh Hugging Face Download)
- **Script**: `scripts/prepare_wildchat_public_manifest.py`
- **Verified Fresh-Environment Command (zero local cache, zero git blob fallback)**:
  ```powershell
  .\.venv\Scripts\python.exe scripts\prepare_wildchat_public_manifest.py --download-from-hf --no-git-fallback --cache-path "$env:TEMP\wildchat_fresh_hf_repro.parquet" --output-dir "results\sdg_mh\wildchat_repro_audit_v6"
  ```
- **Result**: Streamed 4 columns (`hashed_ip`, `timestamp`, `model`, `country`) across all **86 parquet shards** of `allenai/WildChat-4.8M` at pinned revision `c827c6df8fcf008219ffaffa4d1dd77491099367`:
  - `row_count = 3,199,860` (`row_count_matches_expected = True`)
  - `null_counts = {'hashed_ip': 0, 'timestamp': 0, 'model': 0, 'country': 217369}`
  - `logical_records_sha256 = 37144d8ddfbbf8b6633331de612e2cdb96d07aea459db166d9ddd6197101bd09` (`logical_sha256_matches_expected = True`)
  - `input_file_sha256` (`7e13956612e4...`) differs from the legacy pre-projected parquet blob (`581e68ce482d...`) due to parquet writer/compression metadata, confirming why `logical_records_sha256` is the invariant reproducibility check.

## 2. P0 Dual-Regime Additive Operator Microbenchmark (`results/sdg_mh/banded_mf_timing_audit_v6/banded_mf_timing_summary_v6.csv`)
- **Explicit Scope**: Measures **only the 420-day online additive streaming operator** (excludes window post-processing, `UnifiedPrivacyLedger` admission, and `RollingDistinctState`/`CalendarJob`). Not full SDG-MH end-to-end performance.
- **Regime (A) — Including Per-Day Noise Sampling Inside Timer**:
  - `SDG-MH`: `159.60 ms` (`380.01 us/day`, includes `sample_discrete_gaussian` rejection sampling)
  - `Banded-MF (b=8)`: `134.65 ms` (`320.61 us/day`, includes `rng.normal`)
  - `Banded-MF (b=30)`: `135.66 ms` (`323.00 us/day`, includes `rng.normal`)
- **Regime (B) — Kernel-Only (Pre-Generated Noise Excluded from Both Timers)**:
  - `SDG-MH`: `8.63 ms` (`20.54 us/day`)
  - `Banded-MF (b=8)`: `33.11 ms` (`78.84 us/day`)
  - `Banded-MF (b=30)`: `32.61 ms` (`77.65 us/day`)
- **4-Way Memory Taxonomy**:
  1. Theoretical ring buffer: `16 B` (`SDG-MH`) vs `224 B` (`b=8`) / `928 B` (`b=30`)
  2. Online Python `tracemalloc` peak: `14.24 KB` (`SDG-MH`) vs `12.20 KB` / `13.57 KB` (`Banded-MF`)
  3. 90d history cache: separate `RollingDistinctState` subsystem for SDG-MH distinct queries (`N/A` for Banded-MF)
  4. Offline dense matrices: `0.0 KB` (`SDG-MH`) vs `1378.1 KB` (`C_inv`) and `6897.2 KB` (calibration matrices)

## 3. P0 Calendar Origin & Phase Mapping
- Added `experiment_phase_to_runtime_phase(phi_exp, step)` (`(phi_exp + 1) % step`) and `runtime_phase_to_experiment_phase(phi_runtime, step)` (`(phi_runtime - 1) % step`) in `src/private_llm_analytics/sdg_mh/calendar_accounting.py`.
- Verified over all 420 days (`2023-11-20` Monday to `2025-01-12`) for 7d/30d/90d releases and UTC Sundays with zero `+-1` day offset (`tests/sdg_mh/test_v6_audit_regressions.py`).

## 4. P1 Real-Data Statistics, Top-k Differential Check & Manuscript Provenance
- Cell-by-cell differential test (`test_v5_topk_multiuser_multiwindow_ties_differential_against_rolling_state`) confirms exact agreement between v5 `precompute_cohort_suff_stats` and production `RollingDistinctState` across multiple users, 7d/30d/90d windows, frequency ties, and `>3` categories.
- Manuscript (`docs/pvlbd_2027_paper/main.tex` / `main.pdf`) compiled cleanly under PVLDB official template: **12 pages, 0 LaTeX errors, 0 Overfull boxes**.
- Structured provenance gate: **499/499 checks pass**.

## 5. Residual Risks
- **Critical**: None.
- **Major**: WildChat $K=8$ evaluation is conditional on a single 420-day trace (`K4/K8/K16/HashK8` support directional consistency only; 30d significance is grouping-sensitive), top-level `timestamp` marks last conversation turn (not HTTP request time), and WildChat has no HTTP error logs.
- **Minor**: Windows process RSS delta has coarse OS paging granularity; memory comparisons rely on `tracemalloc` peak + analytical byte counts.
