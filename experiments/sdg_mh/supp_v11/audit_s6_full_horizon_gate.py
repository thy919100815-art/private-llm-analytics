"""S6: Full-Horizon Privacy Profile Gate Audit.

Audits:
1. Verification that rho(h) is strictly constant for h >= T_span.
2. Discrepancy between 5-point grid checking and full-horizon checking for 10-year deployment (T_span = 3650):
   - Computes piecewise linear upper bound rho_bar(h) between grid points {1, 7, 30, 90, 365}.
   - Finds max_h rho(h) / rho_bar(h) and corresponding horizon h.
3. Historical Revisit Scenario:
   - 180-day backfill with candidate daily budget rho_new.
   - Identifies exact interval (rho_new_min, rho_new_max] where grid check passes but full gate rejects.
4. Latency benchmark for full-horizon check at T_span = 3650 and N ~ 10-year release volume.
"""

from __future__ import annotations

import math
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Tuple
import numpy as np
import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).parents[3] / "src"))

from private_llm_analytics.sdg_mh.calibration import compute_sdg_target_profile
from private_llm_analytics.sdg_mh.ledger import (
    ReleaseRecord,
    UnifiedPrivacyLedger,
    closed_form_distinct_calendar_rho,
)

OUTPUT_DIR = Path("results/sdg_mh/supp_v11")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def build_10year_deployment_ledger(t_span: int = 3650) -> UnifiedPrivacyLedger:
    """Build UnifiedPrivacyLedger for 10-year deployment with Additive + Calendar B."""
    ledger = UnifiedPrivacyLedger()
    C_r = 104.0
    sigma_r = 5200.0
    C_f = 7.0
    sigma_f = 489.9

    # Additive channel daily releases
    for d in range(1, t_span + 1):
        ledger.record_gaussian_release(
            release_id=f"sdg_additive_req_{d}",
            days_accessed=[d],
            sensitivity=C_r,
            sigma=sigma_r,
            tag="sdg_requests",
        )
        ledger.record_gaussian_release(
            release_id=f"sdg_additive_fail_{d}",
            days_accessed=[d],
            sensitivity=C_f,
            sigma=sigma_f,
            tag="sdg_failures",
        )

    # Distinct Calendar B: (7, 7, 60), (30, 7, 60), (90, 30, 60), delta_sq = 11.0
    delta_sq = 11.0
    cal_b = [(7, 7, 60.0), (30, 7, 60.0), (90, 30, 60.0)]
    for w, s, sig in cal_b:
        for end_d in range(w, t_span + 1, s):
            start_d = max(1, end_d - w + 1)
            ledger.record_gaussian_release(
                release_id=f"cal_{w}_{s}_{end_d}",
                days_accessed=list(range(start_d, end_d + 1)),
                sensitivity=math.sqrt(delta_sq),
                sigma=sig,
                tag=f"cal_{w}_{s}",
            )

    return ledger


def run_s6_audit():
    print("[S6] Running Full-Horizon Privacy Profile Gate Audit...")
    t_span = 3650
    ledger = build_10year_deployment_ledger(t_span=t_span)
    n_releases = len(ledger.records)
    print(f"  Built 10-year ledger: N={n_releases:,} releases across T_span={t_span} days.")

    # 1. Benchmark full-horizon profile computation
    print("  Benchmarking compute_full_horizon_profile across all 3,650 horizons...")
    t0 = time.perf_counter()
    full_rhos = ledger.compute_full_horizon_profile(t_span=t_span)
    elapsed_sec = time.perf_counter() - t0
    print(f"  Full-horizon profile computed in {elapsed_sec:.4f} seconds ({elapsed_sec*1000.0:.2f} ms).")

    # Verify invariance for h >= T_span
    total_budget_sum = sum(r.rho_cost for r in ledger.records)
    rho_at_tspan = float(full_rhos[t_span])
    print(f"  Total release budget sum: {total_budget_sum:.6f}")
    print(f"  rho(T_span = 3650): {rho_at_tspan:.6f}")

    # Check beyond T_span: e.g. h = 3651, 3700, 4000
    rho_3700 = ledger.compute_rho(3700)
    rho_4000 = ledger.compute_rho(4000)
    print(f"  rho(3700) = {rho_3700:.6f}, rho(4000) = {rho_4000:.6f}")
    assert abs(rho_at_tspan - total_budget_sum) < 1e-9
    assert abs(rho_3700 - total_budget_sum) < 1e-9
    assert abs(rho_4000 - total_budget_sum) < 1e-9
    print("  Invariant verified: rho(h) is strictly constant and equals total budget for all h >= T_span!")

    # 2. Compare 5-Point Grid Interpolation vs Full Profile
    grid_points = [1, 7, 30, 90, 365]
    grid_rhos = {h: float(full_rhos[h]) for h in grid_points}

    interp_rhos = np.zeros(366)
    for h in range(1, 366):
        for i in range(len(grid_points) - 1):
            h0, h1 = grid_points[i], grid_points[i + 1]
            if h0 <= h <= h1:
                r0, r1 = grid_rhos[h0], grid_rhos[h1]
                interp_rhos[h] = r0 + (h - h0) * (r1 - r0) / (h1 - h0)
                break

    ratios = []
    exceeded_count = 0
    max_ratio = 1.0
    worst_h = -1

    for h in range(1, 366):
        true_val = float(full_rhos[h])
        interp_val = float(interp_rhos[h])
        ratio = true_val / interp_val if interp_val > 0 else 1.0
        if ratio > 1.00000001:
            exceeded_count += 1
        if ratio > max_ratio:
            max_ratio = ratio
            worst_h = h
        ratios.append({
            "horizon_h": h,
            "true_rho": round(true_val, 6),
            "interp_rho": round(interp_val, 6),
            "ratio": round(ratio, 6),
            "is_grid_point": bool(h in grid_points),
            "exceeds_interpolated": bool(ratio > 1.00000001),
        })

    df_ratios = pd.DataFrame(ratios)
    df_ratios.to_csv(OUTPUT_DIR / "s6_grid_vs_full_ratio.csv", index=False)
    print(f"\n  Grid vs Full-Horizon Audit (h in 1..365):")
    print(f"  Total non-grid horizons exceeding linear bound: {exceeded_count} / 365")
    print(f"  Maximum exceed ratio: {max_ratio:.6f} (+{(max_ratio - 1.0)*100.0:.2f}%) at h={worst_h}!")

    # 3. Revisit Scenario: 180-day backfill
    print("\n  Evaluating 180-day backfill revisit scenario...")
    # Bound defined with 20% margin above grid interpolation: E_bar(h) = 1.20 * interp_rho(h)
    margin = 1.20
    bound_arr = interp_rhos * margin

    # Find maximum rho_new tolerated by grid gate vs full gate
    # For backfill over days 1..180: interval of length h intersects backfill on min(h, 180) days
    grid_limits = {
        h: (bound_arr[h] - float(full_rhos[h])) / min(h, 180)
        for h in grid_points
    }
    all_limits = {
        h: (bound_arr[h] - float(full_rhos[h])) / min(h, 180)
        for h in range(1, 366)
    }

    grid_bottleneck_h = min(grid_limits, key=grid_limits.get)
    max_grid_rho_new = grid_limits[grid_bottleneck_h]

    full_bottleneck_h = min(all_limits, key=all_limits.get)
    max_full_rho_new = all_limits[full_bottleneck_h]

    print(f"  Grid Gate Threshold: rho_new <= {max_grid_rho_new:.8f} (bottleneck at h={grid_bottleneck_h})")
    print(f"  Full Gate Threshold: rho_new <= {max_full_rho_new:.8f} (bottleneck at h={full_bottleneck_h})")
    print(f"  Target Gap Interval: ({max_full_rho_new:.8f}, {max_grid_rho_new:.8f}]")

    # Concrete check with rho_new in this interval
    test_rho_new = 0.5 * (max_full_rho_new + max_grid_rho_new)
    print(f"  Testing candidate backfill with rho_new = {test_rho_new:.8f}...")

    candidate_records = [
        ReleaseRecord(
            release_id=f"backfill_{d}",
            days_accessed=frozenset([d]),
            mechanism_type="zcdp",
            params={"rho": test_rho_new},
            sensitivity=1.0,
            rho_cost=test_rho_new,
            tag="backfill",
        )
        for d in range(1, 181)
    ]

    # Check 5-point grid gate
    grid_bound_dict = {h: bound_arr[h] for h in grid_points}
    grid_allowed, grid_msg, _ = ledger.check_revisit_gate(
        candidate_records=candidate_records,
        max_profile_bound=grid_bound_dict,
    )

    # Check full horizon gate
    full_allowed, full_msg, full_diag = ledger.check_full_horizon_gate(
        candidate_records=candidate_records,
        max_profile_bound_fn_or_array=bound_arr,
        t_span=365,
    )

    print(f"  Grid Gate Result: Allowed={grid_allowed} -> {grid_msg}")
    print(f"  Full Gate Result: Allowed={full_allowed} -> {full_msg}")
    assert grid_allowed is True
    assert full_allowed is False

    summary_rows = [{
        "t_span_days": t_span,
        "n_releases": n_releases,
        "full_check_duration_sec": round(elapsed_sec, 4),
        "full_check_duration_ms": round(elapsed_sec * 1000.0, 2),
        "rho_at_tspan": round(rho_at_tspan, 6),
        "total_budget_sum": round(total_budget_sum, 6),
        "asymptotic_constant_verified": bool(abs(rho_at_tspan - total_budget_sum) < 1e-9),
        "exceeded_grid_points_count": exceeded_count,
        "max_exceed_ratio": round(max_ratio, 6),
        "worst_horizon_h": worst_h,
        "margin_headroom_pct": 20.0,
        "grid_max_rho_new": round(max_grid_rho_new, 8),
        "full_max_rho_new": round(max_full_rho_new, 8),
        "gap_interval": f"({max_full_rho_new:.8f}, {max_grid_rho_new:.8f}]",
        "test_rho_new": round(test_rho_new, 8),
        "grid_gate_passed": grid_allowed,
        "full_gate_passed": full_allowed,
    }]

    df_sum = pd.DataFrame(summary_rows)
    df_sum.to_csv(OUTPUT_DIR / "s6_full_horizon_gate_audit.csv", index=False)
    print(f"  Saved S6 audit summary to {OUTPUT_DIR / 's6_full_horizon_gate_audit.csv'}")
    print(df_sum.to_string(index=False))


if __name__ == "__main__":
    run_s6_audit()
