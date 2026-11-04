"""Core system integration and invariant tests for SDG-MH."""

import math
import numpy as np
import pandas as pd
import pytest

from private_llm_analytics.sdg_mh.calendar_job import (
    CalendarJob,
    DEFAULT_CALENDAR_B,
    DistinctCalendarStore,
)
from private_llm_analytics.sdg_mh.clipping import (
    clip_daily_additive,
    clip_window_l2_additive,
    compute_distinct_window_histogram,
)
from private_llm_analytics.sdg_mh.display_rules import DisplayRulesEngine
from private_llm_analytics.sdg_mh.ledger import UnifiedPrivacyLedger
from private_llm_analytics.sdg_mh.query_service import ReadOnlySealedQueryService
from private_llm_analytics.sdg_mh.seal_job import SealJob
from private_llm_analytics.sdg_mh.sealed_store import DailySealedRecord, SealedStore


def _create_synthetic_day_df(n_users=50, req_per_user=10, fail_ratio=0.1, seed=42):
    rng = np.random.default_rng(seed)
    records = []
    i_cnt = 0
    for u in range(1, n_users + 1):
        uid = f"user_{u:04d}"
        for _ in range(req_per_user):
            i_cnt += 1
            prov = rng.choice(["service_1", "service_2", "service_3"])
            task = rng.choice(["task_01", "task_02", "task_03", "task_04"])
            succ = rng.uniform(0, 1) > fail_ratio
            records.append({
                "interaction_id": f"i_{i_cnt:06d}",
                "user_id": uid,
                "timestamp": "2026-01-01T10:00:00Z",
                "provider": prov,
                "task": task,
                "success": succ,
            })
    return pd.DataFrame(records)


def test_clipping_bounds():
    """Verify daily clipping enforces upper bounds C_r and C_f per user."""
    df = _create_synthetic_day_df(n_users=20, req_per_user=30, fail_ratio=0.5)
    # Total raw requests: 20 * 30 = 600
    # With C_r = 15, clipped sum should be 20 * 15 = 300
    c_r, c_f, _, diags = clip_daily_additive(df, C_r=15.0, C_f=5.0)
    assert c_r == 300.0
    assert diags["requests"].clipping_rate == 1.0
    assert diags["requests"].clipping_bias == 300.0
    assert c_f == 100.0  # 20 * 5


def test_sealed_store_append_only_security():
    """Verify sealed store strictly prevents overwriting existing days."""
    store = SealedStore()
    rec1 = DailySealedRecord(day_index=1, date_str="2026-01-01", y_requests=1000, y_failures=30)
    store.append(rec1)
    assert store.total_sealed_days() == 1

    # Attempting to re-seal day 1 must raise ValueError
    rec1_duplicate = DailySealedRecord(day_index=1, date_str="2026-01-01", y_requests=9999, y_failures=99)
    with pytest.raises(ValueError, match="Append-only violation"):
        store.append(rec1_duplicate)


def test_query_service_raw_data_isolation():
    """Section 2.5 requirement: Verify query service cannot read raw data at permission level."""
    store = SealedStore()
    cal_store = DistinctCalendarStore()
    query_service = ReadOnlySealedQueryService(sealed_store=store, calendar_store=cal_store)

    # Invariant: query_service must not have attributes referencing raw data
    forbidden_attrs = ["raw_dataframe", "df", "telemetry", "_raw_dataframe", "database", "cursor"]
    for attr in forbidden_attrs:
        assert not hasattr(query_service, attr), f"Security violation: query service holds raw data attribute '{attr}'"

    # Sealed store itself also does not hold raw data
    for attr in forbidden_attrs:
        assert not hasattr(store, attr)


def test_end_to_end_1000_repeated_queries_and_zero_ledger_drift():
    """Section 5 correctness check 4:

    Verify same window queried 1,000 times returns 100% identical outputs
    and ledger record count remains completely unchanged.
    """
    ledger = UnifiedPrivacyLedger(delta=1e-6)
    store = SealedStore()
    cal_store = DistinctCalendarStore()
    seal_job = SealJob(sealed_store=store, ledger=ledger, sigma_r=100.0, sigma_f=50.0)

    # Seal 10 days
    for d in range(1, 11):
        df_d = _create_synthetic_day_df(n_users=10, req_per_user=5, seed=d)
        seal_job.seal_day(day_index=d, date_str=f"2026-01-{d:02d}", df_day=df_d, seed=d * 100)

    initial_ledger_count = len(ledger.records)
    query_service = ReadOnlySealedQueryService(sealed_store=store, calendar_store=cal_store)

    # First query
    first_res = query_service.query_additive_window(start_day=1, end_day=7)

    # Repeat 1,000 times
    for _ in range(1000):
        repeat_res = query_service.query_additive_window(start_day=1, end_day=7)
        assert repeat_res["requests_estimate"] == first_res["requests_estimate"]
        assert repeat_res["failures_estimate"] == first_res["failures_estimate"]
        assert repeat_res["success_rate"] == first_res["success_rate"]

    # Ledger must not have changed at all (zero privacy cost)
    assert len(ledger.records) == initial_ledger_count


def test_distinct_calendar_and_topk_ties():
    """Verify Distinct calendar execution and Top-K tie detection."""
    ledger = UnifiedPrivacyLedger(delta=1e-6)
    cal_store = DistinctCalendarStore()
    cal_job = CalendarJob(
        calendar_store=cal_store,
        ledger=ledger,
        calendar_schedule=[(7, 7, 20.0)],
        k_tasks_per_user=3,
    )

    df_window = _create_synthetic_day_df(n_users=30, req_per_user=10, seed=77)
    executed = cal_job.execute_day(
        current_day=7,
        df_history_fn=lambda s, e: df_window,
        seed=42,
    )
    assert len(executed) == 1
    rec = executed[0]
    assert rec.window_length == 7
    assert rec.end_day == 7

    query_service = ReadOnlySealedQueryService(
        sealed_store=SealedStore(),
        calendar_store=cal_store,
        public_population_size=100,
    )

    topk_res = query_service.query_top_k_tasks(window_length=7, end_day=7, k=5)
    assert topk_res["status"] == "OK"
    assert len(topk_res["top_k"]) == 5
    assert "is_tied" in topk_res["top_k"][0]


def test_display_rules_suppression():
    """Verify confidence interval display rules suppress overly wide intervals."""
    engine = DisplayRulesEngine(sigma_r=2500.0, sigma_f=700.0)

    # Small single day window: high relative error in volume and rate
    day1_eval = engine.evaluate_additive_window(
        window_days=1,
        prior_daily_requests=500.0,  # Very small workforce
        prior_failure_rate=0.03,
    )
    assert not day1_eval["success_rate_displayable"]
    assert day1_eval["suppression_reason"] is not None

    # Large 90-day window with 2000 users (22500 requests/day): should be displayable
    day90_eval = engine.evaluate_additive_window(
        window_days=90,
        prior_daily_requests=22500.0,
        prior_failure_rate=0.03,
    )
    assert day90_eval["requests_displayable"]
    assert day90_eval["success_rate_displayable"]
    assert day90_eval["suppression_reason"] is None
