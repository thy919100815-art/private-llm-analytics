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
    JointCalendarPlanner,
    PlannerConfig,
    PlanningResult,
)
from private_llm_analytics.sdg_mh.ledger import UnifiedPrivacyLedger


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
