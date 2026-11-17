"""Unit and Property Tests for Joint Calendar and Budget Planner.

Validates:
1. Complete 4-Group execution (A, B, C, D) under Week 1 v2 commitment.
2. Complete 4-Group execution under Original Planner commitment (including Calibrated Baseline fallback).
3. Business constraint enforcement: fixed weekday/phase pinning.
4. Fail-closed robustness standard: NO_FEASIBLE_POLICY on impossible constraints.
5. Transactional binding to UnifiedPrivacyLedger.
"""

from __future__ import annotations

import math
import pytest
import numpy as np

from private_llm_analytics.sdg_mh.calendar_accounting import (
    ORIGINAL_PLANNER_CURVE,
    WEEK1_V2_CURVE,
)
from private_llm_analytics.sdg_mh.joint_planner import (
    CANONICAL_7_ORBIT_REPRESENTATIVES,
    OPT_TIER_FEASIBLE_CERTIFIED,
    OPT_TIER_GLOBAL_PROVED,
    OPT_TIER_NEAR_GLOBAL_CERTIFIED,
    OPT_TIER_SOLVER_CONVERGED,
    JointCalendarPlanner,
    PlannerConfig,
    PlanningResult,
    build_admissible_phase_space,
    canonicalize_phase_orbit,
    certify_lagrangian_dual_bound_exact,
    enumerate_canonical_phase_orbits,
)
from private_llm_analytics.sdg_mh.ledger import (
    PrivacyAdmissionError,
    ReleaseRecord,
    ReleaseStatus,
    UnifiedPrivacyLedger,
)


def test_four_groups_week1_v2():
    """Verify Groups A, B, C, D under Week 1 v2 commitment curve."""
    config = PlannerConfig(
        commitment=WEEK1_V2_CURVE,
        workload_weights=(0.40, 0.25, 0.25, 0.10),
    )
    planner = JointCalendarPlanner(config)
    results = planner.plan_all_groups()

    # Group A: Baseline
    res_A = results["A_Baseline"]
    assert res_A.status == "FEASIBLE"
    assert res_A.phases == (0, 0, 0)
    assert res_A.min_slack > 0
    assert res_A.is_infinite_horizon_secure is True

    # Group B: Budget Only
    res_B = results["B_BudgetOnly"]
    assert res_B.status == "OPTIMAL"
    assert res_B.phases == (0, 0, 0)
    assert res_B.weighted_workload_error < res_A.weighted_workload_error
    assert res_B.is_infinite_horizon_secure is True

    # Group C: Phase Only
    res_C = results["C_PhaseOnly"]
    assert res_C.status == "OPTIMAL"
    assert res_C.phases != (0, 0, 0)
    assert res_C.min_slack > res_A.min_slack  # +31.6% slack headroom
    assert res_C.is_infinite_horizon_secure is True

    # Group D: Joint Planning
    res_D = results["D_JointPlanning"]
    assert res_D.status == "OPTIMAL"
    assert res_D.weighted_workload_error <= res_B.weighted_workload_error + 1e-6
    assert res_D.is_infinite_horizon_secure is True


def test_original_planner_calibrated_baseline_fallback():
    """Verify that Original Planner curve triggers CALIBRATED_BASELINE fallback for Group A."""
    config = PlannerConfig(
        commitment=ORIGINAL_PLANNER_CURVE,
        workload_weights=(0.40, 0.25, 0.25, 0.10),
    )
    planner = JointCalendarPlanner(config)
    res_A = planner.plan_group_a()

    assert res_A.status == "CALIBRATED_BASELINE"
    assert res_A.diagnostics["calibration_needed"] is True
    assert res_A.diagnostics["unscaled_min_slack"] < 0  # Original breached bound
    assert res_A.diagnostics["unscaled_tightest_h"] == 2
    assert abs(res_A.diagnostics["scale_factor_gamma"] - 0.8790) < 0.01
    assert res_A.is_infinite_horizon_secure is True


def test_business_constraint_pinned_phase():
    """Verify that fixed_phi_7 constraint is strictly honored."""
    pinned_phase = 4
    config = PlannerConfig(
        commitment=WEEK1_V2_CURVE,
        fixed_phi_7=pinned_phase,
    )
    planner = JointCalendarPlanner(config)

    res_C = planner.plan_group_c()
    assert res_C.phases[0] == pinned_phase

    res_D = planner.plan_group_d()
    assert res_D.phases[0] == pinned_phase


def test_fail_closed_impossible_budget():
    """Verify that solver returns NO_FEASIBLE_POLICY when budget cannot meet constraints."""
    config = PlannerConfig(
        commitment=WEEK1_V2_CURVE,
        min_allocations=(0.01, 0.05, 0.05, 0.05),  # Way beyond total allowable privacy budget
    )
    planner = JointCalendarPlanner(config)

    res_B = planner.plan_group_b()
    assert res_B.status == "NO_FEASIBLE_POLICY"
    assert res_B.weighted_workload_error == float("inf")

    res_D = planner.plan_group_d()
    assert res_D.status == "NO_FEASIBLE_POLICY"
    assert res_D.weighted_workload_error == float("inf")


def test_commit_plan_to_ledger_transactional():
    """Verify atomic integration with UnifiedPrivacyLedger."""
    config = PlannerConfig(commitment=WEEK1_V2_CURVE)
    planner = JointCalendarPlanner(config)
    plan_D = planner.plan_group_d()

    ledger = UnifiedPrivacyLedger(delta=1e-6)
    passed, msg, diag = planner.commit_plan_to_ledger(plan_D, ledger, start_day=1, simulated_days=90)

    assert passed is True
    assert len(ledger.records) > 90
    assert diag["exceeded_count"] == 0


def test_distinct_heavy_shrinkage_certification_and_three_tier_flags():
    """Verify DistinctHeavy JointPlanning (which previously breached by -3e-12) strictly passes after shrinkage."""
    for curve in (WEEK1_V2_CURVE, ORIGINAL_PLANNER_CURVE):
        cfg = PlannerConfig(
            commitment=curve,
            workload_weights=(0.10, 0.40, 0.40, 0.10),
        )
        planner = JointCalendarPlanner(cfg)
        plans = planner.plan_all_groups()

        for grp_name in ("B_BudgetOnly", "D_JointPlanning"):
            res = plans[grp_name]
            assert res.status == "OPTIMAL"
            assert res.min_slack >= 0.0
            assert res.asymptotic_slope <= curve.beta_tail
            assert res.is_solver_approx_optimal is True
            # Numerical convex optimization + shrinkage is NEAR_GLOBAL_OPTIMAL_CERTIFIED, never symbolic GLOBAL_OPTIMAL_PROVED
            assert res.optimality_certification_tier == OPT_TIER_NEAR_GLOBAL_CERTIFIED
            assert res.optimality_certification_tier != OPT_TIER_GLOBAL_PROVED
            assert res.is_math_global_optimal is False
            assert res.is_certified_publishable is True
            assert res.is_infinite_horizon_secure is True
            assert res.certification_status == "CERTIFIED_PUBLISHABLE"
            assert res.primal_objective_certified == pytest.approx(res.weighted_workload_error, rel=1e-12)
            assert res.dual_lower_bound > 0.0
            assert res.primal_objective_certified >= res.dual_lower_bound
            assert res.optimality_rel_gap <= 1e-6

            ledger = UnifiedPrivacyLedger(delta=1e-6)
            passed, msg, diag = planner.commit_plan_to_ledger(res, ledger, start_day=1, simulated_days=365)
            assert passed is True, f"{curve.name} {grp_name} commit failed: {msg}"

        assert plans["D_JointPlanning"].optimality_scope == "JOINT_EXHAUSTIVE_MIXED_CONVEX"
        assert plans["D_JointPlanning"].phase_certification_summary["total_phases"] == 7
        assert plans["D_JointPlanning"].phase_certification_summary["canonical_orbits_total"] == 7
        assert plans["D_JointPlanning"].phase_certification_summary["expanded_210_slice_phases"] == 210
        assert plans["D_JointPlanning"].phase_certification_summary["full_1470_phase_space"] == 1470
        assert plans["D_JointPlanning"].phase_certification_summary["all_phases_accounted_for"] is True
        assert plans["B_BudgetOnly"].optimality_scope == "FIXED_PHASE_CONVEX"


def test_inconsistent_unshrunk_solver_result_rejected_everywhere():
    """Failure test: disabling conservative shrinkage exposes ~-3e-12 violation and rejects everywhere."""
    cfg_raw = PlannerConfig(
        commitment=WEEK1_V2_CURVE,
        workload_weights=(0.10, 0.40, 0.40, 0.10),
        apply_conservative_shrinkage=False,
    )
    planner_raw = JointCalendarPlanner(cfg_raw)
    res_raw = planner_raw.plan_group_d()

    # Solver converged approx-optimally, yet failed strict certification -> MUST NOT report OPTIMAL or GLOBAL_PROVED
    assert res_raw.min_slack < -1e-12  # ~ -2.986e-12
    assert res_raw.is_solver_approx_optimal is True
    assert res_raw.is_math_global_optimal is False
    assert res_raw.optimality_certification_tier == OPT_TIER_SOLVER_CONVERGED
    assert res_raw.is_certified_publishable is False
    assert res_raw.is_infinite_horizon_secure is False
    assert res_raw.status == "SOLVER_OPT_UNCERTIFIED"
    assert res_raw.certification_status == "UNCERTIFIED_VIOLATION"

    # Certificate verification must fail closed
    cert = res_raw.schedule.verify_periodicity_and_long_term_security(WEEK1_V2_CURVE)
    assert cert["is_infinite_horizon_secure"] is False
    assert cert["finite_safe"] is False

    # commit_plan_to_ledger must fail closed
    ledger = UnifiedPrivacyLedger(delta=1e-6)
    passed, msg, _ = planner_raw.commit_plan_to_ledger(res_raw, ledger)
    assert passed is False
    assert "uncertified" in msg.lower()

    # UnifiedPrivacyLedger direct admission must ALSO reject because -3e-12 breaches PRIVACY_LEDGER_TOL (1e-12)
    records = []
    eta = 25.0 / 26.0
    rho_r = eta * res_raw.daily_d
    rho_f = (1.0 - eta) * res_raw.daily_d
    sigma_iso = math.sqrt(2600.0 / (2.0 * res_raw.daily_d))
    sched = res_raw.schedule
    for day in range(1, 366):
        records.append(ReleaseRecord(f"r_{day}", frozenset([day]), "gaussian", {"sigma": sigma_iso}, 50.0, rho_r, status=ReleaseStatus.RESERVED.value))
        records.append(ReleaseRecord(f"f_{day}", frozenset([day]), "gaussian", {"sigma": sigma_iso}, 10.0, rho_f, status=ReleaseStatus.RESERVED.value))
        for fam, phi, cost in zip(sched.families, sched.phases, sched.costs):
            if day % fam.step_days == phi % fam.step_days:
                w_days = frozenset(range(max(1, day - fam.window_days + 1), day + 1))
                records.append(ReleaseRecord(f"c_{fam.window_days}_{day}", w_days, "gaussian", {"sigma": math.sqrt(11.0 / (2.0 * cost))}, math.sqrt(11.0), cost, status=ReleaseStatus.RESERVED.value))

    with pytest.raises(PrivacyAdmissionError):
        ledger.admit_and_commit_batch(records, commitment_bound=WEEK1_V2_CURVE, as_status=ReleaseStatus.RESERVED.value)


def test_forged_optimal_status_with_negative_slack_rejected():
    """Failure test: forged status='OPTIMAL' & is_certified_publishable=True with -3e-12 slack is rejected at commit."""
    cfg_raw = PlannerConfig(
        commitment=WEEK1_V2_CURVE,
        workload_weights=(0.10, 0.40, 0.40, 0.10),
        apply_conservative_shrinkage=False,
    )
    planner = JointCalendarPlanner(cfg_raw)
    res_raw = planner.plan_group_d()

    forged = PlanningResult(
        group=res_raw.group,
        status="OPTIMAL",
        phases=res_raw.phases,
        daily_d=res_raw.daily_d,
        distinct_costs=res_raw.distinct_costs,
        weighted_workload_error=res_raw.weighted_workload_error,
        min_slack=res_raw.min_slack,
        tightest_horizon_h=res_raw.tightest_horizon_h,
        asymptotic_slope=res_raw.asymptotic_slope,
        is_infinite_horizon_secure=True,
        runtime_seconds=res_raw.runtime_seconds,
        is_solver_approx_optimal=True,
        is_math_global_optimal=True,
        is_certified_publishable=True,
    )

    ledger = UnifiedPrivacyLedger(delta=1e-6)
    passed, msg, cert = planner.commit_plan_to_ledger(forged, ledger)
    assert passed is False
    assert "failed infinite-horizon privacy verification" in msg.lower()
    assert cert["is_infinite_horizon_secure"] is False


def test_shrinkage_violating_min_allocations_fails_closed():
    """Failure test: when conservative shrinkage pushes allocations below min_allocations, certification fails."""
    cfg = PlannerConfig(
        commitment=WEEK1_V2_CURVE,
        workload_weights=(0.10, 0.40, 0.40, 0.10),
    )
    planner = JointCalendarPlanner(cfg)
    # Construct unshrunk budget at exact tight boundary, then require min_allocations equal to unshrunk vector
    cfg_raw = PlannerConfig(
        commitment=WEEK1_V2_CURVE,
        workload_weights=(0.10, 0.40, 0.40, 0.10),
        apply_conservative_shrinkage=False,
    )
    res_raw = JointCalendarPlanner(cfg_raw).plan_group_d()
    v_raw = np.array([res_raw.daily_d, *res_raw.distinct_costs], dtype=np.float64)

    # Pin min_allocations to v_raw itself: shrinking below v_raw must trigger bounds violation
    cfg_tight = PlannerConfig(
        commitment=WEEK1_V2_CURVE,
        workload_weights=(0.10, 0.40, 0.40, 0.10),
        min_allocations=(float(v_raw[0]), float(v_raw[1]), float(v_raw[2]), float(v_raw[3])),
    )
    planner_tight = JointCalendarPlanner(cfg_tight)
    cert_ok, _, diag = planner_tight._certify_and_shrink_budget(res_raw.phases, v_raw)
    assert diag["bounds_satisfied_after_shrink"] is False
    assert cert_ok is False


def test_primal_dual_bounds_across_all_workloads_and_zero_tol_commit():
    """Verify post-shrinkage primal UB >= dual LB across all 7 canonical phase orbits, rel_gap <= 1e-6, and zero-tol ledger commit."""
    for curve in (WEEK1_V2_CURVE, ORIGINAL_PLANNER_CURVE):
        for wl in ((0.40, 0.25, 0.25, 0.10), (0.10, 0.40, 0.40, 0.10)):
            planner = JointCalendarPlanner(PlannerConfig(commitment=curve, workload_weights=wl))
            res_d = planner.plan_group_d()
            assert res_d.optimality_certification_tier == OPT_TIER_NEAR_GLOBAL_CERTIFIED
            assert res_d.is_math_global_optimal is False
            assert res_d.phase_certification_summary["total_phases"] == 7
            assert res_d.phase_certification_summary["canonical_orbits_total"] == 7
            assert res_d.phase_certification_summary["expanded_210_slice_phases"] == 210
            assert res_d.phase_certification_summary["uncertified_phases"] == 0
            assert res_d.primal_objective_certified >= res_d.dual_lower_bound > 0.0
            assert res_d.optimality_abs_gap >= 0.0
            assert res_d.optimality_rel_gap <= 1e-6

            # Commit with strict_zero_tol=True (executed inside commit_plan_to_ledger)
            ledger = UnifiedPrivacyLedger(delta=1e-6)
            ok, msg, _ = planner.commit_plan_to_ledger(res_d, ledger, start_day=1, simulated_days=365)
            assert ok is True, f"{curve.name} {wl} failed zero-tol commit: {msg}"


def test_uncertified_phase_downgrades_tier_to_feasible_certified(monkeypatch):
    """If any phase stalls without infeasibility proof, global LB is unknown -> must downgrade to FEASIBLE_CERTIFIED."""
    planner = JointCalendarPlanner(PlannerConfig(commitment=WEEK1_V2_CURVE, workload_weights=(0.10, 0.40, 0.40, 0.10)))
    orig_solve = planner.solve_budget_for_phase

    def flaky_solve(phi, initial_v=None):
        if phi in ((0, 2, 0), (0, 2, 10)):
            return False, None, float("inf"), {"status": "SIMULATED_STALL"}
        return orig_solve(phi, initial_v=initial_v)

    monkeypatch.setattr(planner, "solve_budget_for_phase", flaky_solve)
    res_d = planner.plan_group_d()

    assert res_d.status == "FEASIBLE"
    assert res_d.status != "OPTIMAL"
    assert res_d.solver_status == "PARTIAL_CONVERGED"
    assert res_d.is_certified_publishable is True
    assert res_d.is_math_global_optimal is False
    assert res_d.optimality_certification_tier == OPT_TIER_FEASIBLE_CERTIFIED
    assert res_d.phase_certification_summary["uncertified_phases"] == 1
    assert res_d.phase_certification_summary["all_phases_accounted_for"] is False
    assert res_d.dual_lower_bound == float("-inf")
    assert res_d.optimality_rel_gap == float("inf")


def test_ledger_tolerance_conservative_drift_and_strict_zero_tol():
    """Verify UnifiedPrivacyLedger tolerance conservative epsilon bound and strict_zero_tol gate."""
    # 1. Verify conservative epsilon drift for Original_Planner min_rho=0.0141986, delta=1e-6, tol=1e-12
    drift = UnifiedPrivacyLedger.conservative_epsilon_drift_bound(
        min_rho=ORIGINAL_PLANNER_CURVE.rhos[0], delta=1e-6, rho_tol=1e-12
    )
    assert 0.0 < drift < 5e-11  # ~3.22e-11

    # 2. Verify a +5e-13 excess passes default PRIVACY_LEDGER_TOL (1e-12) but is blocked by strict_zero_tol=True
    bound_h1 = {1: 0.01}
    rec_exceed_5e13 = ReleaseRecord(
        release_id="edge_5e13",
        days_accessed=frozenset([1]),
        mechanism_type="gaussian",
        params={"sigma": 1.0},
        sensitivity=1.0,
        rho_cost=0.01 + 5e-13,
        status=ReleaseStatus.RESERVED.value,
    )
    ledger_strict = UnifiedPrivacyLedger(delta=1e-6)
    with pytest.raises(PrivacyAdmissionError):
        ledger_strict.admit_and_commit_batch(
            [rec_exceed_5e13], commitment_bound=bound_h1, strict_zero_tol=True
        )

    ledger_default = UnifiedPrivacyLedger(delta=1e-6)
    committed = ledger_default.admit_and_commit_batch(
        [rec_exceed_5e13], commitment_bound=bound_h1, strict_zero_tol=False
    )
    assert len(committed) == 1


def test_naive_float64_dual_deduction_fails_cancellation_counterexample():
    """Counterexample proving float64 q*(1-1e-14) violates weak duality while Decimal directed rounding holds."""
    A = np.array([[1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0]], dtype=np.float64)
    b = np.array([4.3, 4.3], dtype=np.float64)
    c = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float64)
    l = np.array([1.075, 1.075, 1.075, 1.075], dtype=np.float64)
    u = np.array([2.0, 2.0, 2.0, 2.0], dtype=np.float64)
    lam = np.array([1e15 / 7.0, 1e14 / 3.0], dtype=np.float64)

    # Naive IEEE-754 float64 evaluation with heuristic (1 - 1e-14)
    g = A.T @ lam
    naive_q = -float(np.dot(lam, b))
    for j in range(4):
        vj = float(np.clip(math.sqrt(c[j] / g[j]), l[j], u[j]))
        naive_q += float(c[j] / vj + g[j] * vj)
    naive_safe = naive_q * (1.0 - 1e-14)
    primal_feasible_obj = float(np.sum(c / l))  # l is strictly feasible (4 * 1.075 == 4.3)

    # Counterexample demonstrates naive (1 - 1e-14) overestimates primal optimum!
    assert naive_safe > primal_feasible_obj + 0.1

    # Directed-rounding 100-digit Decimal certificate rigorously respects weak duality
    cert = certify_lagrangian_dual_bound_exact(A, b, c, l, u, lam, tail_row_index=None, prec=100)
    assert cert["cert_method"] == "DECIMAL100_DIRECTED_ROUNDING"
    assert cert["dual_lb"] <= primal_feasible_obj

    # Verify compute_dual_lower_bound inside JointCalendarPlanner uses DECIMAL100_DIRECTED_ROUNDING
    planner = JointCalendarPlanner(PlannerConfig(commitment=WEEK1_V2_CURVE))
    res_b = planner.plan_group_b()
    assert res_b.diagnostics.get("dual_cert_method") == "DECIMAL100_DIRECTED_ROUNDING"
    assert res_b.dual_lower_bound <= res_b.primal_objective_certified


def test_p0a_p0b_commitment_curve_exact_params_and_drift_bounds():
    """Verify exact source-code commitment parameters (P0-A, P0-B) and unified epsilon drift bounds."""
    # P0-B: Verify exact source-code knots for ORIGINAL_PLANNER_CURVE and WEEK1_V2_CURVE
    assert ORIGINAL_PLANNER_CURVE.horizons == (1, 7, 30, 90, 365)
    assert ORIGINAL_PLANNER_CURVE.epsilons == (0.90, 1.08, 1.49, 2.24, 4.39)
    assert WEEK1_V2_CURVE.horizons == (1, 7, 30, 90, 365)
    assert WEEK1_V2_CURVE.epsilons == (1.0, 1.2, 1.5, 2.25, 4.4)

    # Check exact rhos, beta_tail, and a_tail
    assert ORIGINAL_PLANNER_CURVE.rhos[0] == pytest.approx(0.01419861, rel=1e-6)
    assert ORIGINAL_PLANNER_CURVE.rhos[3] == pytest.approx(0.08410616, rel=1e-6)
    assert ORIGINAL_PLANNER_CURVE.rhos[4] == pytest.approx(0.30235637, rel=1e-6)
    assert ORIGINAL_PLANNER_CURVE.beta_tail == pytest.approx(0.00079364, rel=1e-5)
    assert ORIGINAL_PLANNER_CURVE.a_tail == pytest.approx(0.01267882, rel=1e-5)

    assert WEEK1_V2_CURVE.rhos[0] == pytest.approx(0.01746890, rel=1e-6)
    assert WEEK1_V2_CURVE.rhos[3] == pytest.approx(0.08483138, rel=1e-6)
    assert WEEK1_V2_CURVE.rhos[4] == pytest.approx(0.30364628, rel=1e-6)
    assert WEEK1_V2_CURVE.beta_tail == pytest.approx(0.00079569, rel=1e-5)
    assert WEEK1_V2_CURVE.a_tail == pytest.approx(0.01321923, rel=1e-5)

    # P0-A: Verify commitment_curve_drift_summary across all curves and horizons
    summary = UnifiedPrivacyLedger.commitment_curve_drift_summary()
    assert summary["global_min_curve"] == "Original_Planner"
    assert summary["global_min_horizon_h"] == 1
    assert summary["global_min_rho"] == pytest.approx(0.01419861, rel=1e-6)
    assert summary["global_max_epsilon_drift"] == pytest.approx(3.219325e-11, rel=1e-5)
    assert summary["per_curve"]["Week1_v2"]["min_rho"] == pytest.approx(0.01746890, rel=1e-6)
    assert summary["per_curve"]["Week1_v2"]["max_epsilon_drift"] == pytest.approx(2.912229e-11, rel=1e-5)
    assert summary["additive_subchannel_max_epsilon_drift"] == pytest.approx(2.147065e-10, rel=1e-5)
    assert summary["solver_box_floor_max_epsilon_drift"] == pytest.approx(3.718365e-9, rel=1e-5)
    assert summary["universal_nonsingular_root_bound"] == pytest.approx(7.433845e-6, rel=1e-5)


def test_p0c_crt_7_canonical_orbits_partition_1470_phases_and_match_210_slice():
    """Verify CRT bijection partitions 1,470 phases into 7 orbits of size 210 and matches 210-tuple search."""
    orbits = enumerate_canonical_phase_orbits()
    assert len(orbits) == 7
    assert orbits == list(CANONICAL_7_ORBIT_REPRESENTATIVES)

    # 1. Partition all 1,470 phases in Z_7 x Z_7 x Z_30 and all 210 phases in {0} x Z_7 x Z_30
    orbit_counts_1470 = {rep: 0 for rep in orbits}
    orbit_counts_210 = {rep: 0 for rep in orbits}
    for p7 in range(7):
        for p30 in range(7):
            for p90 in range(30):
                rep, delta = canonicalize_phase_orbit((p7, p30, p90))
                assert 0 <= delta < 210
                assert (p7 - delta) % 7 == rep[0] == 0
                assert (p30 - delta) % 7 == rep[1] == (p30 - p7) % 7
                assert (p90 - delta) % 30 == rep[2] == 0
                orbit_counts_1470[rep] += 1
                if p7 == 0:
                    orbit_counts_210[rep] += 1

    assert all(cnt == 210 for cnt in orbit_counts_1470.values())
    assert all(cnt == 30 for cnt in orbit_counts_210.values())

    # 2. Verify 7-orbit search and 210-slice search yield identical solutions across all 4 (curve, workload) pairs
    for curve in (WEEK1_V2_CURVE, ORIGINAL_PLANNER_CURVE):
        for wl in ((0.40, 0.25, 0.25, 0.10), (0.10, 0.40, 0.40, 0.10)):
            p_7 = JointCalendarPlanner(
                PlannerConfig(commitment=curve, workload_weights=wl, use_canonical_7_orbits=True)
            )
            p_210 = JointCalendarPlanner(
                PlannerConfig(commitment=curve, workload_weights=wl, use_canonical_7_orbits=False)
            )
            res_c7 = p_7.plan_group_c()
            res_c210 = p_210.plan_group_c()
            assert res_c7.min_slack == pytest.approx(res_c210.min_slack, abs=1e-15)
            assert canonicalize_phase_orbit(res_c7.phases)[0] == canonicalize_phase_orbit(res_c210.phases)[0]

            res_d7 = p_7.plan_group_d()
            res_d210 = p_210.plan_group_d()
            assert res_d7.phase_certification_summary["total_phases"] == 7
            assert res_d210.phase_certification_summary["total_phases"] == 210
            assert res_d7.phases == res_d210.phases
            assert res_d7.daily_d == pytest.approx(res_d210.daily_d, abs=1e-15)
            assert res_d7.distinct_costs == pytest.approx(res_d210.distinct_costs, abs=1e-15)
            assert res_d7.primal_objective_certified == pytest.approx(res_d210.primal_objective_certified, abs=1e-12)
            assert res_d7.dual_lower_bound == pytest.approx(res_d210.dual_lower_bound, abs=1e-12)


def test_p0c_pinned_phase_breaks_z210_orbit_symmetry():
    """Verify all 8 pinned-phase patterns, the (3,0,0) counterexample, and incomplete-search downgrade."""
    wl_dh = (0.10, 0.40, 0.40, 0.10)

    # 1. Counterexample: fixed_phi_30=0, fixed_phi_90=0, fixed_phi_7=None on Week1_v2 / DistinctHeavy.
    # Previously, setting p7_candidates=[0] when fixed_phi_7=None only tested (0,0,0) (obj ~932.6324)
    # and missed (3,0,0) and (2,0,0), which lie in the same Z_210 orbits as (0,4,0) and (0,5,0)
    # and achieve obj ~922.9791!
    cfg_ce = PlannerConfig(
        commitment=WEEK1_V2_CURVE,
        workload_weights=wl_dh,
        fixed_phi_7=None,
        fixed_phi_30=0,
        fixed_phi_90=0,
    )
    planner_ce = JointCalendarPlanner(cfg_ce)
    res_ce = planner_ce.plan_group_d()
    assert res_ce.phases[1] == 0 and res_ce.phases[2] == 0
    assert res_ce.phases in ((2, 0, 0), (3, 0, 0))
    assert canonicalize_phase_orbit(res_ce.phases)[0] in ((0, 4, 0), (0, 5, 0))
    assert res_ce.primal_objective_certified == pytest.approx(922.97911225, rel=1e-6)
    assert res_ce.primal_objective_certified < 932.63236399 - 9.0
    assert res_ce.phase_certification_summary["total_phases"] == 7
    assert res_ce.phase_certification_summary["full_admissible_phases"] == 7
    assert res_ce.phase_certification_summary["residual_symmetry_group_order"] == 1
    assert res_ce.phase_certification_summary["quotient_orbits_complete"] is True
    assert res_ce.optimality_scope == "CONSTRAINED_PHASE_CONVEX"
    assert res_ce.optimality_certification_tier == OPT_TIER_NEAR_GLOBAL_CERTIFIED

    # 2. Verify all 2^3 = 8 pinning patterns: quotient search (use_canonical_7_orbits=True)
    # must match exhaustive admissible enumeration (use_canonical_7_orbits=False) identically.
    patterns = [
        (None, None, None, 1470, 210, 7, "JOINT_EXHAUSTIVE_MIXED_CONVEX"),
        (4, None, None, 210, 30, 7, "CONSTRAINED_PHASE_CONVEX"),
        (None, 0, None, 210, 30, 7, "CONSTRAINED_PHASE_CONVEX"),
        (None, None, 17, 49, 7, 7, "CONSTRAINED_PHASE_CONVEX"),
        (4, 0, None, 30, 30, 1, "CONSTRAINED_PHASE_CONVEX"),
        (4, None, 17, 7, 1, 7, "CONSTRAINED_PHASE_CONVEX"),
        (None, 0, 17, 7, 1, 7, "CONSTRAINED_PHASE_CONVEX"),
        (4, 0, 17, 1, 1, 1, "CONSTRAINED_PHASE_CONVEX"),
    ]
    for p7_pin, p30_pin, p90_pin, exp_full, exp_gres, exp_quot, exp_scope in patterns:
        sp = build_admissible_phase_space(p7_pin, p30_pin, p90_pin, use_quotient_orbits=True)
        assert sp["full_admissible_count"] == exp_full
        assert sp["residual_group_order"] == exp_gres
        assert sp["quotient_orbits_count"] == exp_quot
        assert len(sp["candidates"]) == exp_quot

        p_quot = JointCalendarPlanner(
            PlannerConfig(
                commitment=WEEK1_V2_CURVE,
                workload_weights=wl_dh,
                fixed_phi_7=p7_pin,
                fixed_phi_30=p30_pin,
                fixed_phi_90=p90_pin,
                use_canonical_7_orbits=True,
            )
        )
        p_full = JointCalendarPlanner(
            PlannerConfig(
                commitment=WEEK1_V2_CURVE,
                workload_weights=wl_dh,
                fixed_phi_7=p7_pin,
                fixed_phi_30=p30_pin,
                fixed_phi_90=p90_pin,
                use_canonical_7_orbits=False,
            )
        )
        r_quot = p_quot.plan_group_d()
        r_full = p_full.plan_group_d()
        if p7_pin is not None:
            assert r_quot.phases[0] == p7_pin and r_full.phases[0] == p7_pin
        if p30_pin is not None:
            assert r_quot.phases[1] == p30_pin and r_full.phases[1] == p30_pin
        if p90_pin is not None:
            assert r_quot.phases[2] == p90_pin and r_full.phases[2] == p90_pin
        assert r_quot.optimality_scope == exp_scope
        assert r_quot.phase_certification_summary["quotient_orbits_complete"] is True
        assert r_quot.primal_objective_certified == pytest.approx(r_full.primal_objective_certified, abs=1e-12)
        assert r_quot.dual_lower_bound == pytest.approx(r_full.dual_lower_bound, abs=1e-12)

    # 3. Verify that if only a partial subset of admissible phases is searched, the result
    # is NEVER labeled NEAR_GLOBAL_OPTIMAL_CERTIFIED or CONSTRAINED_PHASE_CONVEX.
    import private_llm_analytics.sdg_mh.joint_planner as jp_mod
    orig_builder = jp_mod.build_admissible_phase_space

    def truncated_builder(*args, **kwargs):
        d = orig_builder(*args, **kwargs)
        # Drop candidates so only 1 of the 7 quotient orbits is searched
        d["candidates"] = d["candidates"][:1]
        return d

    planner_sub = JointCalendarPlanner(cfg_ce)
    jp_mod.build_admissible_phase_space = truncated_builder
    try:
        res_sub = planner_sub.plan_group_d()
    finally:
        jp_mod.build_admissible_phase_space = orig_builder

    assert res_sub.status == "FEASIBLE"
    assert res_sub.status != "OPTIMAL"
    assert res_sub.solver_status == "PARTIAL_CONVERGED"
    assert res_sub.is_certified_publishable is True
    assert res_sub.phase_certification_summary["quotient_orbits_complete"] is False
    assert res_sub.phase_certification_summary["all_phases_accounted_for"] is False
    assert res_sub.optimality_scope == "INCOMPLETE_PHASE_SUBSET"
    assert res_sub.optimality_certification_tier == OPT_TIER_FEASIBLE_CERTIFIED
    assert res_sub.optimality_certification_tier != OPT_TIER_NEAR_GLOBAL_CERTIFIED


def test_status_separation_solver_convergence_privacy_feasibility_and_optimality(monkeypatch):
    """Verify strict separation of solver convergence, privacy feasibility, and optimality status."""
    planner = JointCalendarPlanner(
        PlannerConfig(commitment=WEEK1_V2_CURVE, workload_weights=(0.10, 0.40, 0.40, 0.10))
    )

    # Case 1: Loose dual lower bound -> solver converged & privacy feasible, but gap not certified
    def loose_dual_lb(phi, v_opt):
        return {
            "dual_lb": 100.0,  # Loose LB -> rel_gap > NEAR_GLOBAL_REL_GAP_TOL
            "dual_certified": True,
            "dual_cert_method": "DECIMAL100_DIRECTED_ROUNDING",
        }

    monkeypatch.setattr(planner, "compute_dual_lower_bound", loose_dual_lb)
    res_b_loose = planner.plan_group_b()
    assert res_b_loose.is_solver_approx_optimal is True
    assert res_b_loose.solver_status == "CONVERGED"
    assert res_b_loose.is_certified_publishable is True
    assert res_b_loose.is_infinite_horizon_secure is True
    assert res_b_loose.optimality_certification_tier == OPT_TIER_FEASIBLE_CERTIFIED
    assert res_b_loose.status == "FEASIBLE"
    assert res_b_loose.status != "OPTIMAL"

    res_d_loose = planner.plan_group_d()
    assert res_d_loose.is_solver_approx_optimal is True
    assert res_d_loose.solver_status == "CONVERGED"
    assert res_d_loose.is_certified_publishable is True
    assert res_d_loose.is_infinite_horizon_secure is True
    assert res_d_loose.optimality_certification_tier == OPT_TIER_FEASIBLE_CERTIFIED
    assert res_d_loose.status == "FEASIBLE"
    assert res_d_loose.status != "OPTIMAL"


def test_p0_manuscript_to_csv_provenance_verifier():
    """Verify structured manuscript-to-CSV provenance check passes and catches all 4 required mutations."""
    from pathlib import Path
    from scripts.verify_manuscript_csv_provenance import verify_manuscript_provenance

    report = verify_manuscript_provenance()
    assert report["all_passed"] is True
    assert report["failed_checks"] == 0
    assert report["passed_checks"] >= 260

    orig_tex = (Path(__file__).resolve().parents[2] / "docs" / "pvlbd_2027_paper" / "main.tex").read_text(encoding="utf-8")

    # Mutation 1: Modify a MAPE difference in Table 4 (-0.0271% -> -0.0371%)
    # Even if we append the original string "-0.0271\%" in a comment, structured table parsing must fail!
    mut_mape = orig_tex.replace(r"\mathbf{-0.0271\%}", r"\mathbf{-0.0371\%}", 1) + "\n% -0.0271\\%"
    with pytest.raises(AssertionError, match="tab4_.*rel_7d_mape_pct.*_diff"):
        verify_manuscript_provenance(tex_text=mut_mape, write_report=False)

    # Mutation 2: Modify a p_holm value in Table 4 (1.0000 -> 0.0412 on rel_30d_mape_pct)
    target_row = r"DistHeavy / Week1\_v2 & D $-$ B & \texttt{rel\_30d\_mape\_pct} (Own release) & $2.4907\%$ & $2.4666\%$ & $-0.0241\%$ ($\pm 0.0835\%$) & $0.5166$ & $1.0000$"
    replacement_row = r"DistHeavy / Week1\_v2 & D $-$ B & \texttt{rel\_30d\_mape\_pct} (Own release) & $2.4907\%$ & $2.4666\%$ & $-0.0241\%$ ($\pm 0.0835\%$) & $0.5166$ & $0.0412$"
    assert target_row in orig_tex
    mut_holm = orig_tex.replace(target_row, replacement_row, 1)
    with pytest.raises(AssertionError, match="tab4_.*rel_30d_mape_pct.*_holm_p"):
        verify_manuscript_provenance(tex_text=mut_holm, write_report=False)

    # Mutation 3: Modify a baseline method label in Table 5
    mut_method = orig_tex.replace(r"\textbf{Tree-2l (MH-RemEnv)}", r"\textbf{Tree-2l (Uncalibrated-Raw)}", 1)
    with pytest.raises(AssertionError, match="tab5_method_present_Tree-2l \\(MH-RemEnv\\)"):
        verify_manuscript_provenance(tex_text=mut_method, write_report=False)

    # Mutation 4: Modify service/task dimension in Section 2 (m = 5 -> m = 9 or |A| = 20 -> |A| = 12)
    mut_dim = orig_tex.replace(r"$|\mathcal{S}| = m = 5$", r"$|\mathcal{S}| = m = 9$", 1)
    with pytest.raises(AssertionError, match="sys_dim_services_m_eq_5"):
        verify_manuscript_provenance(tex_text=mut_dim, write_report=False)





