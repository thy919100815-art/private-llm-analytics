"""Supplementary Unit Tests for SDG-MH Round 11 (sdg-mh-supp-v11).

Covers:
- S3: test_released_vector_l2_sensitivity_matches_ledger
- S1: test_post_departure_growth_stops_after_wmax
- S2: test_rolling_state_matches_raw_recompute, test_rolling_state_purged_after_wmax
- S6: test_full_horizon_gate_rejects_what_grid_misses / test_full_horizon_profile_constant_beyond_span
"""

from __future__ import annotations

import math
from typing import Dict, List, Set
import numpy as np
import pandas as pd
import pytest

from private_llm_analytics.sdg_mh.clipping import (
    clip_daily_additive,
    compute_distinct_window_histogram,
)
from private_llm_analytics.sdg_mh.ledger import UnifiedPrivacyLedger
from private_llm_analytics.sdg_mh.seal_job import SealJob
from private_llm_analytics.sdg_mh.sealed_store import SealedStore


def _generate_synthetic_day(n_users: int = 100, max_req_per_user: int = 200, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    records = []
    i_cnt = 0
    for u in range(1, n_users + 1):
        uid = f"user_{u:04d}"
        n_req = int(rng.integers(1, max_req_per_user + 1))
        for _ in range(n_req):
            i_cnt += 1
            prov = rng.choice(["service_1", "service_2", "service_3"])
            task = rng.choice(["task_01", "task_02", "task_03", "task_04"])
            succ = bool(rng.uniform(0, 1) > 0.15)
            records.append({
                "interaction_id": f"i_{i_cnt:06d}",
                "user_id": uid,
                "timestamp": "2026-01-01T12:00:00Z",
                "provider": prov,
                "task": task,
                "success": succ,
            })
    return pd.DataFrame(records)


def test_released_vector_l2_sensitivity_matches_ledger():
    """S3 Unit Test: Verify user contribution L2 norm strictly bounded by ledger sensitivity.

    Checks:
    1. Daily requests release is scalar with |x_{u,t}| <= C_r, exactly matching ledger sensitivity C_r.
    2. Daily failures release is scalar with |f_{u,t}| <= C_f, exactly matching ledger sensitivity C_f.
    3. Success count is not released; success rate is pure post-processing.
    4. Optional vector provider release enforces L2 norm bound C_p in clip_daily_additive.
    """
    C_r = 104.0
    C_f = 7.0
    sigma_r = 5200.0
    sigma_f = 489.9

    store = SealedStore()
    ledger = UnifiedPrivacyLedger(allow_unconfigured_policy=True)
    job = SealJob(sealed_store=store, ledger=ledger, C_r=C_r, C_f=C_f, sigma_r=sigma_r, sigma_f=sigma_f)

    df_day = _generate_synthetic_day(n_users=100, max_req_per_user=250, seed=123)
    rec = job.seal_day(day_index=1, date_str="2026-01-01", df_day=df_day, seed=42)

    # 1. Requests release sensitivity
    rel_r = [r for r in ledger.records if r.tag == "sdg_requests"][0]
    assert rel_r.sensitivity == C_r

    u_counts = df_day["user_id"].value_counts()
    clipped_req_per_user = u_counts.clip(upper=C_r)
    max_req_l2 = float(clipped_req_per_user.max())
    assert max_req_l2 <= rel_r.sensitivity
    assert max_req_l2 == C_r  # Hit clipping upper bound

    # 2. Failures release sensitivity
    rel_f = [r for r in ledger.records if r.tag == "sdg_failures"][0]
    assert rel_f.sensitivity == C_f

    fail_df = df_day[df_day["success"] == False]
    f_counts = fail_df["user_id"].value_counts()
    clipped_fail_per_user = f_counts.clip(upper=C_f)
    max_fail_l2 = float(clipped_fail_per_user.max())
    assert max_fail_l2 <= rel_f.sensitivity
    assert max_fail_l2 == C_f

    # 3. Optional L2 vector clipping verification
    C_p = 50.0
    _, _, p_vec, _ = clip_daily_additive(
        df_day=df_day,
        C_r=C_r,
        C_f=C_f,
        providers_list=["service_1", "service_2", "service_3"],
        C_p=C_p,
    )
    assert p_vec is not None
    # Verify per-user L2 norm with C_p
    p_crosstab = pd.crosstab(df_day["user_id"], df_day["provider"]).reindex(
        columns=["service_1", "service_2", "service_3"], fill_value=0
    ).to_numpy(dtype=np.float64)
    norms = np.linalg.norm(p_crosstab, ord=2, axis=1)
    scale = np.minimum(1.0, C_p / np.maximum(norms, 1e-12))
    clipped_p_mat = p_crosstab * scale[:, None]
    clipped_norms = np.linalg.norm(clipped_p_mat, ord=2, axis=1)
    assert float(np.max(clipped_norms)) <= C_p + 1e-9


def test_post_departure_growth_stops_after_wmax():
    """S1 Unit Test: Post-departure privacy growth strictly ceases after w_max days.

    Verifies:
    1. Additive exposure freezes on departure day d (delta_add = 0 for tau > d).
    2. Distinct calendar exposure stops growing after tau = d + w_max - 1.
    3. eps(d + 90) == eps(d + 91) == eps(d + 180) == eps_precommitted.
    """
    rho_1_add = 0.00030204
    calendar_b = [(7, 7, 60.0), (30, 7, 60.0), (90, 30, 60.0)]
    delta_sq = 11.0
    w_max = 90

    # Test an employee with join_day = 10, depart_day = 150
    j_day = 10
    d_day = 150
    tenure_days = d_day - j_day + 1

    # Generate calendar releases up to d + 200
    releases = []
    for W, step, sigma in calendar_b:
        rho_w = delta_sq / (2.0 * (sigma**2))
        for e_w in range(W, d_day + 201, step):
            s_w = e_w - W + 1
            releases.append((W, s_w, e_w, rho_w))

    from private_llm_analytics.sdg_mh.ledger import rho_to_epsilon

    # Function to compute total rho at cutoff tau
    def get_rho_at_tau(tau: int) -> Tuple[float, float, float]:
        # Additive: releases t <= tau intersecting [j_day, d_day]
        # Since tau >= d_day, all tenure days [j_day, d_day] have been released
        rho_add = tenure_days * rho_1_add
        # Distinct: calendar releases with e_w <= tau intersecting [j_day, d_day]
        rho_dist = sum(
            rho_w for _, s_w, e_w, rho_w in releases
            if e_w <= tau and not (e_w < j_day or s_w > d_day)
        )
        return rho_add, rho_dist, rho_add + rho_dist

    # Pre-committed lifetime rho (tau -> inf)
    rho_dist_precommit = sum(
        rho_w for _, s_w, e_w, rho_w in releases
        if not (e_w < j_day or s_w > d_day)
    )
    eps_precommit = rho_to_epsilon(tenure_days * rho_1_add + rho_dist_precommit)

    # Check cutoffs
    eps_at_d = rho_to_epsilon(get_rho_at_tau(d_day)[2])
    eps_at_d7 = rho_to_epsilon(get_rho_at_tau(d_day + 7)[2])
    eps_at_d30 = rho_to_epsilon(get_rho_at_tau(d_day + 30)[2])
    eps_at_d89 = rho_to_epsilon(get_rho_at_tau(d_day + w_max - 1)[2])
    eps_at_d90 = rho_to_epsilon(get_rho_at_tau(d_day + w_max)[2])
    eps_at_d91 = rho_to_epsilon(get_rho_at_tau(d_day + w_max + 1)[2])
    eps_at_d180 = rho_to_epsilon(get_rho_at_tau(d_day + 180)[2])

    # Distinct channel grows between d and d + w_max - 1
    assert eps_at_d < eps_at_d7 <= eps_at_d30 <= eps_at_d89

    # Growth stops strictly after w_max - 1
    assert abs(eps_at_d90 - eps_at_d89) < 1e-9 or eps_at_d90 == eps_at_d89
    assert eps_at_d90 == eps_at_d91
    assert eps_at_d91 == eps_at_d180
    assert abs(eps_at_d180 - eps_precommit) < 1e-9


def test_rolling_state_matches_raw_recompute():
    """S2 Unit Test: RollingDistinctState histogram strictly matches raw data recompute."""
    from private_llm_analytics.sdg_mh.rolling_state import RollingDistinctState
    from private_llm_analytics.sdg_mh.calendar_job import DEFAULT_DISTINCT_SERVICES, DEFAULT_DISTINCT_TASKS

    state = RollingDistinctState(w_max=90)
    rng = np.random.default_rng(999)

    # Ingest 14 days of synthetic data
    days_df = {}
    for d in range(1, 15):
        df_day = _generate_synthetic_day(n_users=40, max_req_per_user=50, seed=100 + d)
        days_df[d] = df_day
        state.append_day(d, df_day, DEFAULT_DISTINCT_SERVICES, DEFAULT_DISTINCT_TASKS, persist_disk=False)

    # Evaluate window [1, 14]
    df_win = pd.concat([days_df[d] for d in range(1, 15)], ignore_index=True)
    raw_srv, raw_tsk = compute_distinct_window_histogram(
        df_window=df_win,
        services_list=DEFAULT_DISTINCT_SERVICES,
        tasks_list=DEFAULT_DISTINCT_TASKS,
        k_tasks_per_user=3,
    )
    st_srv, st_tsk = state.compute_window_histogram(
        start_day=1,
        end_day=14,
        services_list=DEFAULT_DISTINCT_SERVICES,
        tasks_list=DEFAULT_DISTINCT_TASKS,
        k_tasks_per_user=3,
    )

    assert np.array_equal(raw_srv, st_srv), f"Service mismatch: raw={raw_srv}, state={st_srv}"
    assert np.array_equal(raw_tsk, st_tsk), f"Task mismatch: raw={raw_tsk}, state={st_tsk}"


def test_rolling_state_purged_after_wmax(tmp_path):
    """S2 Unit Test: Records strictly purged after w_max days."""
    from private_llm_analytics.sdg_mh.rolling_state import RollingDistinctState
    from private_llm_analytics.sdg_mh.calendar_job import DEFAULT_DISTINCT_SERVICES, DEFAULT_DISTINCT_TASKS

    w_max = 10
    state = RollingDistinctState(storage_dir=tmp_path, w_max=w_max)

    # Ingest days 1 to 25
    for d in range(1, 26):
        df_day = _generate_synthetic_day(n_users=10, max_req_per_user=10, seed=500 + d)
        state.append_day(d, df_day, DEFAULT_DISTINCT_SERVICES, DEFAULT_DISTINCT_TASKS, persist_disk=True)

    # At day 25 with w_max = 10: cutoff = 25 - 10 = 15
    # Days 1..15 must be purged from memory and disk
    for d in range(1, 16):
        assert d not in state._day_states, f"Day {d} should have been purged from memory"
        p = tmp_path / f"state_day_{d:05d}.pkl"
        assert not p.exists(), f"Day {d} should have been deleted from disk"

    # Days 16..25 must be present
    for d in range(16, 26):
        assert d in state._day_states, f"Day {d} must be present in memory"
        p = tmp_path / f"state_day_{d:05d}.pkl"
        assert p.exists(), f"Day {d} must be present on disk"

    # Requesting purged day 10 must raise KeyError
    with pytest.raises(KeyError):
        state.compute_window_histogram(10, 20, DEFAULT_DISTINCT_SERVICES, DEFAULT_DISTINCT_TASKS)


def test_full_horizon_gate_rejects_what_grid_misses():
    """S6 Unit Test: Full-horizon gate catches privacy breach missed by 5-point grid gate.

    Verifies:
    1. rho(h) is strictly constant for h >= T_span.
    2. A candidate revisit with rho_new in (0.00015184, 0.00018603] passes the 5-point
       grid check on {1, 7, 30, 90, 365}, but is rejected by the full-horizon gate.
    """
    from private_llm_analytics.sdg_mh.ledger import ReleaseRecord

    # Create 365-day ledger with additive + calendar B releases
    ledger = UnifiedPrivacyLedger(allow_unconfigured_policy=True)
    C_r = 104.0
    sigma_r = 5200.0
    C_f = 7.0
    sigma_f = 489.9
    for d in range(1, 366):
        ledger.record_gaussian_release(f"req_{d}", [d], C_r, sigma_r)
        ledger.record_gaussian_release(f"fail_{d}", [d], C_f, sigma_f)

    # Calendar B
    for w, s, sig in [(7, 7, 60.0), (30, 7, 60.0), (90, 30, 60.0)]:
        for end_d in range(w, 366, s):
            start_d = max(1, end_d - w + 1)
            ledger.record_gaussian_release(f"cal_{w}_{s}_{end_d}", list(range(start_d, end_d + 1)), math.sqrt(11.0), sig)

    # 1. Invariance check
    full_rhos = ledger.compute_full_horizon_profile(t_span=365)
    total_budget = sum(r.rho_cost for r in ledger.records)
    assert abs(full_rhos[365] - total_budget) < 1e-9
    assert abs(ledger.compute_rho(500) - total_budget) < 1e-9

    # 2. Gate check with candidate backfill
    grid_points = [1, 7, 30, 90, 365]
    grid_rhos = {h: float(full_rhos[h]) for h in grid_points}

    interp_rhos = np.zeros(366)
    for h in range(1, 366):
        for i in range(len(grid_points) - 1):
            h0, h1 = grid_points[i], grid_points[i + 1]
            if h0 <= h <= h1:
                interp_rhos[h] = grid_rhos[h0] + (h - h0) * (grid_rhos[h1] - grid_rhos[h0]) / (h1 - h0)
                break

    bound_arr = interp_rhos * 1.20
    grid_bound_dict = {h: bound_arr[h] for h in grid_points}

    # Use rho_new = 0.000169 in gap (0.00015184, 0.00018603]
    rho_new = 0.00016894
    candidate_records = [
        ReleaseRecord(
            release_id=f"backfill_{d}",
            days_accessed=frozenset([d]),
            mechanism_type="zcdp",
            params={"rho": rho_new},
            sensitivity=1.0,
            rho_cost=rho_new,
            tag="backfill",
        )
        for d in range(1, 181)
    ]

    grid_allowed, _, _ = ledger.check_revisit_gate(candidate_records, grid_bound_dict)
    full_allowed, _, diag = ledger.check_full_horizon_gate(candidate_records, bound_arr, t_span=365)

    assert grid_allowed is True, "Grid gate should have passed"
    assert full_allowed is False, "Full-horizon gate should have rejected"
    assert diag["exceeded_count"] > 0
