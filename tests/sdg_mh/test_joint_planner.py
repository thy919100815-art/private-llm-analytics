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
    OPT_TIER_FEASIBLE_CERTIFIED,
    OPT_TIER_GLOBAL_PROVED,
    OPT_TIER_NEAR_GLOBAL_CERTIFIED,
    OPT_TIER_SOLVER_CONVERGED,
    JointCalendarPlanner,
    PlannerConfig,
    PlanningResult,
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
        assert plans["D_JointPlanning"].phase_certification_summary["total_phases"] == 210
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
    """Verify post-shrinkage primal UB >= dual LB across all 210 phases, rel_gap <= 1e-6, and zero-tol ledger commit."""
    for curve in (WEEK1_V2_CURVE, ORIGINAL_PLANNER_CURVE):
        for wl in ((0.40, 0.25, 0.25, 0.10), (0.10, 0.40, 0.40, 0.10)):
            planner = JointCalendarPlanner(PlannerConfig(commitment=curve, workload_weights=wl))
            res_d = planner.plan_group_d()
            assert res_d.optimality_certification_tier == OPT_TIER_NEAR_GLOBAL_CERTIFIED
            assert res_d.is_math_global_optimal is False
            assert res_d.phase_certification_summary["total_phases"] == 210
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
        if phi == (0, 2, 10):
            return False, None, float("inf"), {"status": "SIMULATED_STALL"}
        return orig_solve(phi, initial_v=initial_v)

    monkeypatch.setattr(planner, "solve_budget_for_phase", flaky_solve)
    res_d = planner.plan_group_d()

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
