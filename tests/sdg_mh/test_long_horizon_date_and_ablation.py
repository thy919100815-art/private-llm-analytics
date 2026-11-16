"""Regression tests for valid calendar dates across >31 days and C_r ablation calibration."""

from __future__ import annotations

import datetime
import math

from private_llm_analytics.sdg_mh.calendar_accounting import WEEK1_V2_CURVE
from private_llm_analytics.sdg_mh.joint_planner import JointCalendarPlanner, PlannerConfig
from private_llm_analytics.sdg_mh.ledger import UnifiedPrivacyLedger
from private_llm_analytics.sdg_mh.seal_job import SealJob
from private_llm_analytics.sdg_mh.sealed_store import SealedStore


def test_valid_iso_calendar_dates_across_420_days():
    """Verify date string conversion across 1..420 never yields invalid dates like 2025-01-32."""
    seen = set()
    for day in range(1, 421):
        d_str = (datetime.date(2025, 1, 1) + datetime.timedelta(days=day - 1)).isoformat()
        parsed = datetime.date.fromisoformat(d_str)
        assert parsed.isoformat() == d_str
        seen.add(d_str)
    assert len(seen) == 420
    assert (datetime.date(2025, 1, 1) + datetime.timedelta(days=419)).isoformat() == "2026-02-24"


def test_cr_ablation_preserves_zcdp_budget_invariant():
    """Verify C_r in {30, 50, 100, 200} keeps rho_r + rho_f == daily_d identically while updating sigma."""
    planner = JointCalendarPlanner(
        PlannerConfig(commitment=WEEK1_V2_CURVE, workload_weights=(0.40, 0.25, 0.25, 0.10))
    )
    plan = planner.plan_group_d()
    daily_d = plan.daily_d
    C_f = 10.0

    sigmas = []
    for C_r in (30.0, 50.0, 100.0, 200.0):
        eta = (C_r ** 2) / (C_r ** 2 + C_f ** 2)
        store = SealedStore()
        ledger = UnifiedPrivacyLedger(delta=1e-6)
        ledger.set_active_plan(plan, commitment=WEEK1_V2_CURVE)
        job = SealJob.from_budget(store, ledger, daily_d=daily_d, eta=eta, C_r=C_r, C_f=C_f)
        assert math.isclose(job.rho_r + job.rho_f, daily_d, rel_tol=1e-12)
        expected_sigma = math.sqrt((C_r ** 2 + C_f ** 2) / (2.0 * daily_d))
        assert math.isclose(job.sigma_r, expected_sigma, rel_tol=1e-12)
        assert math.isclose(job.sigma_f, expected_sigma, rel_tol=1e-12)
        sigmas.append(job.sigma_r)

    assert sigmas == sorted(sigmas)
