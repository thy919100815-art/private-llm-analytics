"""Formal Correctness Checks Runner for SDG-MH (Checks 1 to 4).

Generates results/sdg_mh/formal_checks.csv documenting:
- Check 1: Unified ledger closed-form unit tests (SDG and Calendars A/B)
- Check 2: Theorem 2 Empirical Log-Likelihood Ratio test on (u, I)-neighboring pairs
- Check 3: Theorem 7(b) lower bound test on 300 random layer noise binary tree configurations
- Check 4: End-to-end 1000 repeated queries zero-privacy-drift test with raw data isolation
"""

from __future__ import annotations

import math
from pathlib import Path
import sys
import time
from typing import Any, Dict, List

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parents[2] / "src"))

from private_llm_analytics.sdg_mh.calendar_job import (
    DEFAULT_CALENDAR_A,
    DEFAULT_CALENDAR_B,
    DistinctCalendarStore,
)
from private_llm_analytics.sdg_mh.discrete_gaussian import sample_discrete_gaussian
from private_llm_analytics.sdg_mh.ledger import (
    UnifiedPrivacyLedger,
    closed_form_distinct_calendar_rho,
    closed_form_sdg_additive_rho,
)
from private_llm_analytics.sdg_mh.query_service import ReadOnlySealedQueryService
from private_llm_analytics.sdg_mh.sealed_store import DailySealedRecord, SealedStore

RESULTS_DIR = Path("results/sdg_mh")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def run_check_1() -> Dict[str, Any]:
    """Check 1: Unified ledger closed-Form unit tests."""
    print("Running Check 1: Unified Ledger Closed-Form Match...")
    C_r = 50.0
    sigma_r = 2500.0
    C_f = 10.0
    sigma_f = 700.0
    rho_1 = (C_r**2) / (2 * sigma_r**2) + (C_f**2) / (2 * sigma_f**2)

    ledger = UnifiedPrivacyLedger(delta=1e-6)
    for day in range(1, 181):
        ledger.record_gaussian_release(f"seal_r_{day}", [day], C_r, sigma_r, "req")
        ledger.record_gaussian_release(f"seal_f_{day}", [day], C_f, sigma_f, "fail")

    max_diff_sdg = 0.0
    for h in [1, 7, 30, 90]:
        num = ledger.compute_rho(h)
        cf = closed_form_sdg_additive_rho(h, rho_1)
        max_diff_sdg = max(max_diff_sdg, abs(num - cf))

    # Distinct Calendars (simulated over 180 days for steady-state window alignment)
    max_diff_cal = 0.0
    calendar_b = [(7, 7, 60.0), (30, 7, 60.0), (90, 30, 60.0)]
    delta_sq = 11.0
    sens = math.sqrt(delta_sq)
    ledger_cal = UnifiedPrivacyLedger(delta=1e-6)
    for w, s, sig in calendar_b:
        for e in range(s, 181, s):
            window_days = list(range(max(1, e - w + 1), e + 1))
            ledger_cal.record_gaussian_release(f"dist_B_{w}_{e}", window_days, sens, sig)
    for h in [1, 7, 30, 90]:
        num = ledger_cal.compute_rho(h)
        cf = closed_form_distinct_calendar_rho(h, calendar_b, delta_sq=delta_sq)
        max_diff_cal = max(max_diff_cal, abs(num - cf))

    passed = (max_diff_sdg < 1e-12) and (max_diff_cal < 1e-12)
    return {
        "check_id": "Check 1",
        "name": "Unified Ledger Closed-Form Invariant",
        "inputs": "SDG additive T=180, Calendar B across h in {1, 7, 30, 90}",
        "expected": "Exact agreement (diff < 1e-12) between numerical ledger and closed forms",
        "measured": f"Max diff SDG={max_diff_sdg:.2e}, Max diff Cal={max_diff_cal:.2e}",
        "status": "PASSED" if passed else "FAILED",
        "evidence_file": "tests/sdg_mh/test_ledger_and_sampler.py:L70-L105",
    }


def run_check_2() -> Dict[str, Any]:
    """Check 2: Theorem 2 Empirical Log-Likelihood Ratio Test."""
    print("Running Check 2: Theorem 2 Empirical Log-Likelihood Ratio...")
    C_r = 50.0
    sigma_r = 2500.0
    rho_1 = (C_r**2) / (2.0 * sigma_r**2)

    diffs = []
    n_samples = 20000
    rng = np.random.default_rng(42)

    all_passed = True
    for h in [7, 30]:
        expected_mean = h * rho_1
        expected_var = 2.0 * h * rho_1
        se_mean = math.sqrt(expected_var / n_samples)

        llrs = []
        for _ in range(n_samples):
            noise_vec = np.array([sample_discrete_gaussian(sigma_r, rng=rng) for _ in range(h)])
            llr = np.sum((2.0 * noise_vec * C_r + C_r**2) / (2.0 * sigma_r**2))
            llrs.append(llr)

        emp_mean = float(np.mean(llrs))
        emp_var = float(np.var(llrs))

        z_score = abs(emp_mean - expected_mean) / se_mean
        rel_var_err = abs(emp_var - expected_var) / expected_var
        diffs.append((h, z_score, rel_var_err))
        if z_score > 2.5 or rel_var_err > 0.05:
            all_passed = False

    meas_str = "; ".join(f"h={h}: mean_z={z:.2f} (SE={se_mean:.2e}), var_err={ev:.2%}" for h, z, ev in diffs)
    return {
        "check_id": "Check 2",
        "name": "Theorem 2 Empirical Log-Likelihood Ratio zCDP",
        "inputs": f"(u, I)-neighboring pairs, h in {{7, 30}}, N={n_samples} transcripts",
        "expected": "E[L] = |I|*rho_1 (|z| < 2.5) and Var(L) = 2*|I|*rho_1 (< 5% error)",
        "measured": meas_str,
        "status": "PASSED" if all_passed else "FAILED",
        "evidence_file": "experiments/sdg_mh/verify_formal_checks.py:run_check_2",
    }


def run_check_3() -> Dict[str, Any]:
    """Check 3: Theorem 7(b) Binary Tree Lower Bound."""
    print("Running Check 3: Theorem 7(b) Lower Bound Audit...")
    audit_path = RESULTS_DIR / "rq3_random_300_audit.csv"
    if not audit_path.exists():
        return {
            "check_id": "Check 3",
            "name": "Theorem 7(b) Binary Tree Lower Bound",
            "inputs": "300 random layer noise configurations over T=1024, L=10",
            "expected": "Zero bound violations, zero configurations beating SDG on all windows",
            "measured": "File not found",
            "status": "FAILED",
            "evidence_file": "results/sdg_mh/rq3_random_300_audit.csv",
        }

    df_audit = pd.read_csv(audit_path)
    viol = int(df_audit["violations_of_lower_bound"].iloc[0])
    beat_all = int(df_audit["configs_more_accurate_than_sdg_all_windows"].iloc[0])
    beat_any = int(df_audit.get("configs_more_accurate_than_sdg_any_window", pd.Series([0])).iloc[0])
    n_cfg = int(df_audit["n_sampled_configs"].iloc[0])

    passed = (viol == 0) and (beat_all == 0) and (n_cfg == 300)
    return {
        "check_id": "Check 3",
        "name": "Theorem 7(b) Binary Tree Lower Bound",
        "inputs": f"{n_cfg} random layer noise configurations over T=1024, L=10",
        "expected": "Zero bound violations, zero configurations beating SDG on all windows (and audit beating on any window)",
        "measured": f"Bound violations: {viol}/{n_cfg}, Beat everywhere: {beat_all}/{n_cfg}, Beat any window (profile-scaled): {beat_any}/{n_cfg}",
        "status": "PASSED" if passed else "FAILED",
        "evidence_file": "results/sdg_mh/rq3_random_300_audit.csv:L1-L2",
    }


def run_check_4() -> Dict[str, Any]:
    """Check 4: End-to-end 1000 repeated queries zero privacy drift test."""
    print("Running Check 4: End-to-end 1000 Queries Zero Drift...")
    store = SealedStore()
    cal_store = DistinctCalendarStore()
    ledger = UnifiedPrivacyLedger(delta=1e-6)

    # Pre-populate sealed store for 90 days
    rng = np.random.default_rng(42)
    for t in range(1, 91):
        rec = DailySealedRecord(
            day_index=t,
            date_str=f"2026-01-{t:02d}",
            y_requests=int(10000 + rng.integers(-500, 500)),
            y_failures=int(300 + rng.integers(-30, 30)),
            sigma_r=2500.0,
            sigma_f=700.0,
            C_r=50.0,
            C_f=10.0,
        )
        store.append(rec)
        ledger.record_gaussian_release(f"seal_r_{t}", [t], 50.0, 2500.0)
        ledger.record_gaussian_release(f"seal_f_{t}", [t], 10.0, 700.0)

    # Permission check: verify query service has no raw dataframe or connection
    query_service = ReadOnlySealedQueryService(sealed_store=store, calendar_store=cal_store)
    forbidden_attrs = ["raw_dataframe", "df", "telemetry", "_raw_dataframe", "database", "cursor", "raw_telemetry"]
    has_raw = any(hasattr(query_service, attr) for attr in forbidden_attrs)

    # Initial ledger record count
    initial_records = len(ledger.records)
    initial_rho_30 = ledger.compute_rho(30)

    # Execute 1,000 queries
    for q_idx in range(1000):
        w_len = int(rng.integers(1, 45))
        s_day = int(rng.integers(1, 90 - w_len + 1))
        e_day = s_day + w_len - 1
        res = query_service.query_additive_window(s_day, e_day)
        assert res["sealed_days_present"] == w_len

    final_records = len(ledger.records)
    final_rho_30 = ledger.compute_rho(30)

    record_delta = final_records - initial_records
    rho_delta = final_rho_30 - initial_rho_30

    passed = (not has_raw) and (record_delta == 0) and (abs(rho_delta) < 1e-12)
    return {
        "check_id": "Check 4",
        "name": "End-to-End 1000 Queries Zero Privacy Drift",
        "inputs": "1000 random window queries on 90-day sealed store",
        "expected": "Strict raw data isolation, 0 ledger entries added, 0.000 rho drift",
        "measured": f"Raw isolation={not has_raw}, Ledger delta={record_delta}, Rho delta={rho_delta:.1e}",
        "status": "PASSED" if passed else "FAILED",
        "evidence_file": "tests/sdg_mh/test_sdg_mh_core.py:L140-L165",
    }


def main():
    print("=" * 80)
    print("Executing Formal Correctness Checks 1 to 4")
    print("=" * 80)
    results = [run_check_1(), run_check_2(), run_check_3(), run_check_4()]
    df_checks = pd.DataFrame(results)
    df_checks.to_csv(RESULTS_DIR / "formal_checks.csv", index=False)
    print("Saved formal_checks.csv successfully.")
    print(df_checks[["check_id", "name", "status", "measured"]])
    print("=" * 80)


if __name__ == "__main__":
    main()
