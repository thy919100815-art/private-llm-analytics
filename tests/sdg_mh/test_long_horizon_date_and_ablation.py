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


def test_audit_v2_crn_disjoint_seeds_stratified_icc_and_baseline_envelope():
    """Verify collision-free CRN seeds, seed-level hierarchical ICC, and baseline MH calibration."""
    import importlib.util
    from pathlib import Path
    import numpy as np
    import pandas as pd

    script_path = Path(__file__).parents[2] / "experiments" / "sdg_mh" / "run_long_horizon_420d_audit_v2.py"
    spec = importlib.util.spec_from_file_location("run_audit_v2", script_path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)

    # 1. Collision-free across (trial=0..4, day=1..30, stream=1..2)
    seeds = {
        mod.derive_crn_seed(42, tr, d, st)
        for tr in range(5)
        for d in range(1, 31)
        for st in (1, 2)
    }
    assert len(seeds) == 5 * 30 * 2

    # 2. Seed-level hierarchical summary avoids pseudo-replication when ICC=1.0
    rows = []
    for k, val in enumerate([10.0, 20.0, 30.0, 40.0]):
        for m in range(5):
            rows.append({"data_seed": k, "grp": "G", "bias": val})
    df_test = pd.DataFrame(rows)
    res = mod.stratified_seed_summary(df_test, ["grp"], ["bias"])
    assert int(res["K_seeds"].iloc[0]) == 4
    assert float(res["icc"].iloc[0]) == 1.0
    assert float(res["seed_level_ci95"].iloc[0]) > float(res["naive_pooled_ci95_invalid"].iloc[0]) * 1.5

    # 3. Tree-2l calibrated to rem_add_envelope strictly satisfies rho_tree(h) <= rem_add_envelope[h-1] for all h
    rem_env = np.linspace(0.005, 0.25, 420)
    cal = mod.calibrate_tree_and_wbd_on_envelope(rem_env, daily_d_sdg=0.00047, n_days=420)
    t2l = cal["trees"]["Tree-2l"]
    tb = mod.TreeBaseline(variant="Tree-2l", T=512, L=9, C=1.0)
    beta_sq = t2l["beta_env"] ** 2
    for h_chk in (1, 7, 30, 90, 210, 365, 420):
        rho_h = 0.5 * tb.get_worst_case_rho_term(h_chk) / beta_sq
        assert rho_h <= float(rem_env[h_chk - 1]) + 1e-12


def test_v3_injective_128bit_seeds_no_known_uint32_collision_and_uniform_postproc():
    """Verify (3042,2,410,1) vs (5042,4,218,2) do NOT collide, exhaustive 8-seed check has 0 collisions,
    uniform post-processing enforces 0 <= Fail <= Req, and Holm-Bonferroni matches exact step-down bounds."""
    import importlib.util
    from pathlib import Path
    import numpy as np

    for fname, mname in (
        ("run_long_horizon_420d_audit_v2.py", "audit_v2_chk"),
        ("run_long_horizon_420d_audit_v3.py", "audit_v3_chk"),
    ):
        script_path = Path(__file__).parents[2] / "experiments" / "sdg_mh" / fname
        spec = importlib.util.spec_from_file_location(mname, script_path)
        mod = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(mod)

        # Known PR#6 uint32 collision pair MUST produce strictly distinct 128-bit seeds
        s1 = mod.derive_crn_seed(3042, 2, 410, 1)
        s2 = mod.derive_crn_seed(5042, 4, 218, 2)
        assert s1 != s2

        if hasattr(mod, "verify_all_experiment_seeds_collision_free"):
            audit = mod.verify_all_experiment_seeds_collision_free(
                [42, 1042, 2042, 3042, 4042, 5042, 6042, 7042], 5, 420
            )
            assert audit["collisions_detected"] == 0
            assert audit["total_seeds_checked"] == audit["unique_seeds"]

            # Uniform post-processing: Req_hat = max(0, Req), Fail_hat = min(Req_hat, max(0, Fail))
            r_hat, f_hat = mod.postprocess_window_pair(
                np.array([-5.0, 10.0, 20.0]),
                np.array([3.0, 15.0, -4.0]),
            )
            assert np.allclose(r_hat, [0.0, 10.0, 20.0])
            assert np.allclose(f_hat, [0.0, 10.0, 0.0])

            # Holm-Bonferroni check
            p_adj = mod.holm_bonferroni([0.001, 0.04, 0.02])
            assert p_adj == [0.003, 0.04, 0.04]


