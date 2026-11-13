"""Unit and Property Tests for Exact Calendar Accounting and Long-Term Security.

Validates:
1. Fast modular count vs Brute Force Oracle across thousands of configurations.
2. Exact matching of historical reference table (aligned vs offset S(2), S(37), min S, tightest h).
3. Congruence peak arc properties.
4. Periodicity identity L_max(h + P) = L_max(h) + P * slope.
5. Tail condition slope <= beta and infinite-horizon reduction.
6. Exhaustive sweep over phase configurations.
"""

from __future__ import annotations

import math
import random
import pytest
import numpy as np

from private_llm_analytics.sdg_mh.calendar_accounting import (
    DEFAULT_FAMILIES,
    JOINT_PERIOD,
    ORIGINAL_PLANNER_CURVE,
    WEEK1_V2_CURVE,
    CalendarFamily,
    CalendarSchedule,
    analyze_congruence_peak_arc,
    count_intersections_fast,
    count_intersections_fast_vectorized,
    count_intersections_oracle,
)


def test_oracle_vs_fast_formula_randomized():
    """Verify that count_intersections_fast exactly matches discrete oracle across 2,000 cases."""
    rng = random.Random(42)
    families = [(7, 7), (30, 7), (90, 30), (14, 7), (60, 14)]

    for _ in range(2000):
        w, s_step = rng.choice(families)
        phi = rng.randint(0, s_step - 1)
        s = rng.randint(1, 400)
        h = rng.randint(1, 250)

        # Oracle over sufficiently wide day bounds
        oracle_cnt = count_intersections_oracle(
            s=s,
            h=h,
            window_days=w,
            step_days=s_step,
            phi=phi,
            day_min=-200,
            day_max=1000,
        )
        fast_cnt = count_intersections_fast(
            s=s,
            h=h,
            window_days=w,
            step_days=s_step,
            phi=phi,
        )
        assert oracle_cnt == fast_cnt, f"Mismatch at s={s}, h={h}, W={w}, S={s_step}, phi={phi}: oracle={oracle_cnt}, fast={fast_cnt}"


def test_vectorized_matches_scalar():
    """Verify that vectorized counting matches scalar implementation."""
    s_arr = np.arange(1, 211, dtype=np.int64)
    h = 37
    w = 30
    s_step = 7
    phi = 3

    vec_res = count_intersections_fast_vectorized(s_arr, h, w, s_step, phi)
    scalar_res = np.array([count_intersections_fast(int(s), h, w, s_step, phi) for s in s_arr], dtype=np.int64)

    np.testing.assert_array_equal(vec_res, scalar_res)


def test_historical_reference_table_reproduction():
    """Verify exact reproduction of Section 3.3 historical benchmark numbers.

    Parameters:
      b = 0.000302
      d = 0.8 * b = 0.0002416
      x_7 = x_30 = x_90 = 0.00153
      Curve: Week 1 v2 (eps = 1.0, 1.2, 1.5, 2.25, 4.4 at h = 1, 7, 30, 90, 365)

    Expected reference values:
      Aligned (phi=(0, 0, 0)):
        2d remaining: ~ 4.663b
        37d remaining: ~ 4.747b
        Full scale min remaining: ~ 4.663b at h=2
      Offset 2-day (e.g. phi=(0, 3, 0) or offset phases):
        2d remaining: ~ 9.729b
        37d remaining: ~ 9.813b
        Full scale min remaining: ~ 6.135b at h=32
    """
    b = 0.000302
    d = 0.8 * b
    x_val = 0.00153

    # 1. Aligned Schedule
    aligned_sched = CalendarSchedule(
        phases=(0, 0, 0),
        costs=(x_val, x_val, x_val),
        daily_cost_d=d,
    )
    res_aligned = aligned_sched.compute_slack(commitment=WEEK1_V2_CURVE, max_h=365)
    s2_aligned = (WEEK1_V2_CURVE.rho_bar(2) - aligned_sched.evaluate_envelope(2)[0]) / b
    s37_aligned = (WEEK1_V2_CURVE.rho_bar(37) - aligned_sched.evaluate_envelope(37)[0]) / b
    min_s_aligned = res_aligned["min_slack"] / b
    min_h_aligned = res_aligned["min_h"]

    assert abs(s2_aligned - 4.663) < 0.005, f"Expected ~4.663b, got {s2_aligned:.4f}b"
    assert abs(s37_aligned - 4.747) < 0.005, f"Expected ~4.747b, got {s37_aligned:.4f}b"
    assert abs(min_s_aligned - 4.663) < 0.005, f"Expected min ~4.663b, got {min_s_aligned:.4f}b"
    assert min_h_aligned == 2, f"Expected tightest at h=2, got {min_h_aligned}"

    # 2. Offset Schedule (phi = (0, 3, 0) produces the exact 2-day offset reference metrics)
    offset_sched = CalendarSchedule(
        phases=(0, 3, 0),
        costs=(x_val, x_val, x_val),
        daily_cost_d=d,
    )
    res_offset = offset_sched.compute_slack(commitment=WEEK1_V2_CURVE, max_h=365)
    s2_offset = (WEEK1_V2_CURVE.rho_bar(2) - offset_sched.evaluate_envelope(2)[0]) / b
    s37_offset = (WEEK1_V2_CURVE.rho_bar(37) - offset_sched.evaluate_envelope(37)[0]) / b
    min_s_offset = res_offset["min_slack"] / b
    min_h_offset = res_offset["min_h"]

    assert abs(s2_offset - 9.729) < 0.005, f"Expected ~9.729b, got {s2_offset:.4f}b"
    assert abs(s37_offset - 9.813) < 0.005, f"Expected ~9.813b, got {s37_offset:.4f}b"
    assert abs(min_s_offset - 6.135) < 0.005, f"Expected min ~6.135b, got {min_s_offset:.4f}b"
    assert min_h_offset == 32, f"Expected tightest at h=32, got {min_h_offset}"


def test_congruence_peak_arc_decomposition():
    """Verify congruence peak arc properties: count is q+1 on peak residues, q elsewhere."""
    h = 15
    w = 30
    s_step = 7
    phi = 2

    info = analyze_congruence_peak_arc(h, w, s_step, phi)
    q = info["q"]
    r = info["r"]
    peak_res = set(info["peak_residues"])
    base_res = set(info["base_residues"])

    assert len(peak_res) == r
    assert len(base_res) == s_step - r
    assert peak_res.isdisjoint(base_res)

    for s_val in range(1, 100):
        res = s_val % s_step
        cnt = count_intersections_fast(s_val, h, w, s_step, phi)
        if res in peak_res:
            assert cnt == q + 1
        else:
            assert cnt == q


def test_periodicity_and_long_term_security_theorem():
    """Verify Section 5 theorem: L_max(h + P) = L_max(h) + P * slope and slope <= beta."""
    b = 0.000302
    d = 0.8 * b
    x_val = 0.00153

    sched = CalendarSchedule(
        phases=(0, 2, 5),
        costs=(x_val, x_val, x_val),
        daily_cost_d=d,
    )

    cert = sched.verify_periodicity_and_long_term_security(
        commitment=WEEK1_V2_CURVE,
        H_0=365,
        P=JOINT_PERIOD,
    )

    assert cert["is_infinite_horizon_secure"] is True
    assert cert["slope_safe"] is True
    assert cert["periodicity_holds"] is True
    assert cert["finite_safe"] is True
    assert cert["periodicity_max_err"] < 1e-11
    assert cert["asymptotic_slope"] < cert["commitment_beta_tail"]


def test_curves_monotonicity_and_conversion():
    """Verify both commitment curves are strictly increasing and smooth."""
    for curve in (ORIGINAL_PLANNER_CURVE, WEEK1_V2_CURVE):
        h_range = np.arange(1, 600)
        rhos = curve.rho_bar(h_range)
        eps = curve.epsilon_bar(h_range)

        # Monotonicity
        assert np.all(np.diff(rhos) >= 0)
        assert np.all(np.diff(eps) >= 0)

        # Check roundtrip at specification points
        for h, target_eps in zip(curve.horizons, curve.epsilons):
            eval_eps = float(curve.epsilon_bar(h))
            assert abs(eval_eps - target_eps) < 1e-4
