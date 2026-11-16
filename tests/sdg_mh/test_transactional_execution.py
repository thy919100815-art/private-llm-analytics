"""Tests for Transactional Execution, Privacy Admission, and Daily Calibration.

Verifies:
1. Daily Gaussian privacy budget split: d = rho_r + rho_f (+ rho_p), sigma_r = sigma_f = sqrt(2600/(2d)).
2. CalendarJob phase execution integration matches calendar_accounting.py exactly for (0,0,0), (0,3,0), (0,4,0).
3. Idempotent execution in SealJob and CalendarJob (no double-charging).
4. Fail-closed privacy admission gating on budget breach.
5. Atomic rollback on store persistence failure.
6. Multithreaded concurrency safety in UnifiedPrivacyLedger.
7. Plan updates preserve historical committed records.
"""

import math
import threading
from typing import Dict, List, Optional
import numpy as np
import pandas as pd
import pytest

from private_llm_analytics.sdg_mh.ledger import (
    UnifiedPrivacyLedger,
    ReleaseRecord,
    ReleaseStatus,
    PrivacyAdmissionError,
)
from private_llm_analytics.sdg_mh.seal_job import SealJob
from private_llm_analytics.sdg_mh.calendar_job import (
    CalendarJob,
    DistinctCalendarStore,
    DEFAULT_DISTINCT_SERVICES,
    DEFAULT_DISTINCT_TASKS,
)
from private_llm_analytics.sdg_mh.sealed_store import SealedStore
from private_llm_analytics.sdg_mh.calendar_accounting import (
    CalendarFamily,
    iterate_family_releases,
)
from private_llm_analytics.sdg_mh.joint_planner import (
    JointCalendarPlanner,
    PlannerConfig,
    WEEK1_V2_CURVE,
)
from private_llm_analytics.sdg_mh.rolling_state import RollingDistinctState


# ==============================================================================
# 1. Daily Gaussian Privacy Budget Split Calibration
# ==============================================================================

def test_daily_gaussian_budget_split_isotropic():
    """Verify d = rho_r + rho_f and sigma_r == sigma_f == sqrt(2600/(2d)) with eta=25/26."""
    daily_d = 0.0001
    C_r = 50.0
    C_f = 10.0
    eta = 25.0 / 26.0

    store = SealedStore()
    ledger = UnifiedPrivacyLedger(delta=1e-6)
    job = SealJob.from_budget(
        sealed_store=store,
        ledger=ledger,
        daily_d=daily_d,
        eta=eta,
        C_r=C_r,
        C_f=C_f,
    )

    # 1. Budget split identity
    assert math.isclose(job.rho_r + job.rho_f, daily_d, rel_tol=1e-12)
    assert math.isclose(job.rho_r, (25.0 / 26.0) * daily_d, rel_tol=1e-12)
    assert math.isclose(job.rho_f, (1.0 / 26.0) * daily_d, rel_tol=1e-12)

    # 2. Isotropic noise standard deviations: sqrt(2600 / (2d))
    expected_sigma = math.sqrt(2600.0 / (2.0 * daily_d))
    assert math.isclose(job.sigma_r, expected_sigma, rel_tol=1e-12)
    assert math.isclose(job.sigma_f, expected_sigma, rel_tol=1e-12)

    # 3. Sum of actual mechanism costs must equal daily_d
    mech_rho_r = (C_r ** 2) / (2.0 * (job.sigma_r ** 2))
    mech_rho_f = (C_f ** 2) / (2.0 * (job.sigma_f ** 2))
    assert math.isclose(mech_rho_r + mech_rho_f, daily_d, rel_tol=1e-12)


def test_daily_gaussian_budget_split_custom_eta():
    """Verify custom eta correctly splits budget and satisfies rho_r + rho_f == daily_d."""
    daily_d = 0.00025
    C_r = 50.0
    C_f = 10.0
    eta = 0.80

    store = SealedStore()
    ledger = UnifiedPrivacyLedger(delta=1e-6)
    job = SealJob.from_budget(
        sealed_store=store,
        ledger=ledger,
        daily_d=daily_d,
        eta=eta,
        C_r=C_r,
        C_f=C_f,
    )

    assert math.isclose(job.rho_r, 0.80 * daily_d, rel_tol=1e-12)
    assert math.isclose(job.rho_f, 0.20 * daily_d, rel_tol=1e-12)
    assert math.isclose(job.rho_r + job.rho_f, daily_d, rel_tol=1e-12)

    expected_sigma_r = C_r / math.sqrt(2.0 * 0.80 * daily_d)
    expected_sigma_f = C_f / math.sqrt(2.0 * 0.20 * daily_d)
    assert math.isclose(job.sigma_r, expected_sigma_r, rel_tol=1e-12)
    assert math.isclose(job.sigma_f, expected_sigma_f, rel_tol=1e-12)


def test_daily_gaussian_budget_split_with_provider_channel():
    """Verify provider channel accounted in total daily budget: rho_r + rho_f + rho_p == daily_d."""
    daily_d = 0.0003
    eta = 0.70
    eta_p = 0.10
    C_r = 50.0
    C_f = 10.0
    C_p = 5.0

    store = SealedStore()
    ledger = UnifiedPrivacyLedger(delta=1e-6)
    job = SealJob.from_budget(
        sealed_store=store,
        ledger=ledger,
        daily_d=daily_d,
        eta=eta,
        eta_p=eta_p,
        C_r=C_r,
        C_f=C_f,
        C_p=C_p,
        providers_list=["p1", "p2"],
    )

    assert math.isclose(job.rho_r, 0.70 * daily_d, rel_tol=1e-12)
    assert math.isclose(job.rho_p, 0.10 * daily_d, rel_tol=1e-12)
    assert math.isclose(job.rho_f, 0.20 * daily_d, rel_tol=1e-12)
    assert math.isclose(job.rho_r + job.rho_f + job.rho_p, daily_d, rel_tol=1e-12)


# ==============================================================================
# 2. Calendar Phase Execution Integration
# ==============================================================================

@pytest.mark.parametrize("phases", [
    (0, 0, 0),
    (0, 3, 0),
    (0, 4, 0),
])
def test_calendar_job_phase_execution_matches_analytical(phases):
    """Verify CalendarJob execution days strictly match calendar_accounting.py."""
    phi_7, phi_30, phi_90 = phases
    T = 90
    sigma = 30.0

    families = [
        CalendarFamily("f7", window_days=7, step_days=7),
        CalendarFamily("f30", window_days=30, step_days=7),
        CalendarFamily("f90", window_days=90, step_days=30),
    ]

    expected_releases_per_day: Dict[int, List[str]] = {t: [] for t in range(1, T + 1)}
    for fam, phi in zip(families, phases):
        rel_days = iterate_family_releases(fam, phi, T)
        for d in rel_days:
            expected_releases_per_day[d].append(fam.name)

    cal_store = DistinctCalendarStore()
    ledger = UnifiedPrivacyLedger(delta=1e-6, allow_unconfigured_policy=True)
    schedule = [
        (7, 7, sigma, phi_7),
        (30, 7, sigma, phi_30),
        (90, 30, sigma, phi_90),
    ]
    job = CalendarJob(
        calendar_store=cal_store,
        ledger=ledger,
        calendar_schedule=schedule,
        services_list=DEFAULT_DISTINCT_SERVICES,
        tasks_list=DEFAULT_DISTINCT_TASKS,
    )

    dummy_state = RollingDistinctState(storage_dir=None)
    empty_df = pd.DataFrame(columns=["user_id", "provider", "task", "timestamp", "success"])
    for d in range(1, T + 1):
        dummy_state.append_day(
            day_index=d,
            df_day=empty_df,
            services_list=DEFAULT_DISTINCT_SERVICES,
            tasks_list=DEFAULT_DISTINCT_TASKS,
            persist_disk=False,
        )

    for day in range(1, T + 1):
        recs = job.execute_day(current_day=day, rolling_state=dummy_state)
        executed_fams = []
        for r in recs:
            if r.window_length == 7 and r.step == 7:
                executed_fams.append("f7")
            elif r.window_length == 30 and r.step == 7:
                executed_fams.append("f30")
            elif r.window_length == 90 and r.step == 30:
                executed_fams.append("f90")

        expected = expected_releases_per_day[day]
        assert sorted(executed_fams) == sorted(expected), (
            f"Day {day} phase {phases}: executed {executed_fams} != expected {expected}"
        )


# ==============================================================================
# 3. Idempotency & Duplicate Prevention
# ==============================================================================

def test_seal_job_idempotency():
    """Verify executing seal_day twice returns the existing record without duplicate spending."""
    store = SealedStore()
    ledger = UnifiedPrivacyLedger(delta=1e-6)
    ledger.set_active_plan(plan=None, commitment=WEEK1_V2_CURVE)
    job = SealJob.from_budget(sealed_store=store, ledger=ledger, daily_d=0.0001)

    df_day = pd.DataFrame([
        {"user_id": "u1", "provider": "prov1", "task": "t1", "timestamp": "2025-01-01 10:00:00", "success": True},
    ])

    rec1 = job.seal_day(day_index=1, date_str="2025-01-01", df_day=df_day, seed=42)
    assert len(ledger.records) == 2  # requests + failures

    # Second call for the same day
    rec2 = job.seal_day(day_index=1, date_str="2025-01-01", df_day=df_day, seed=999)

    assert rec1 == rec2
    assert len(ledger.records) == 2  # Unchanged
    assert len(store) == 1


def test_calendar_job_idempotency():
    """Verify executing execute_day twice does not charge ledger or store twice."""
    cal_store = DistinctCalendarStore()
    ledger = UnifiedPrivacyLedger(delta=1e-6, allow_unconfigured_policy=True)
    schedule = [(7, 7, 30.0, 0)]
    job = CalendarJob(
        calendar_store=cal_store,
        ledger=ledger,
        calendar_schedule=schedule,
    )
    state = RollingDistinctState(storage_dir=None)
    empty_df = pd.DataFrame(columns=["user_id", "provider", "task", "timestamp", "success"])
    for d in range(1, 8):
        state.append_day(
            day_index=d,
            df_day=empty_df,
            services_list=DEFAULT_DISTINCT_SERVICES,
            tasks_list=DEFAULT_DISTINCT_TASKS,
            persist_disk=False,
        )

    # Day 7 triggers window
    recs1 = job.execute_day(current_day=7, rolling_state=state, seed=42)
    assert len(recs1) == 1
    assert len(ledger.records) == 1

    # Second call
    recs2 = job.execute_day(current_day=7, rolling_state=state, seed=999)
    assert len(recs2) == 1
    assert recs1[0] == recs2[0]
    assert len(ledger.records) == 1  # Not double charged
    assert len(cal_store.all_releases()) == 1


# ==============================================================================
# 4. Fail-Closed Privacy Admission
# ==============================================================================

def test_fail_closed_privacy_admission():
    """Verify release exceeding commitment is rejected before storage or release."""
    store = SealedStore()
    ledger = UnifiedPrivacyLedger(delta=1e-6)

    # Set very tight commitment
    ledger.set_active_plan(plan=None, commitment=WEEK1_V2_CURVE)

    # Candidate exceeding bound at h=1 (e.g. daily budget huge)
    huge_daily_d = 100.0  # Far exceeding rho_bar(1) ≈ 0.0003
    job = SealJob.from_budget(sealed_store=store, ledger=ledger, daily_d=huge_daily_d)

    df_day = pd.DataFrame([
        {"user_id": "u1", "provider": "prov1", "task": "t1", "timestamp": "2025-01-01 10:00:00", "success": True},
    ])

    with pytest.raises(PrivacyAdmissionError) as exc_info:
        job.seal_day(day_index=1, date_str="2025-01-01", df_day=df_day)

    assert "Privacy admission rejected" in str(exc_info.value) or "SealJob rejected day 1" in str(exc_info.value)
    # Verify fail-closed: nothing committed to store or ledger
    assert len(ledger.records) == 0
    assert store.get_day(1) is None


# ==============================================================================
# 5. Atomic Rollback on Store Persistence Failure
# ==============================================================================

class BrokenSealedStore(SealedStore):
    """Store that simulates an IO or persistence failure."""
    def append(self, record):
        raise IOError("Disk write failed: simulated hardware fault")


def test_atomic_rollback_on_storage_failure():
    """Verify that if store persistence fails, unexposed reservation is safely cancelled."""
    broken_store = BrokenSealedStore()
    ledger = UnifiedPrivacyLedger(delta=1e-6)
    ledger.set_active_plan(plan=None, commitment=WEEK1_V2_CURVE)
    job = SealJob.from_budget(sealed_store=broken_store, ledger=ledger, daily_d=0.0001)

    df_day = pd.DataFrame([
        {"user_id": "u1", "provider": "prov1", "task": "t1", "timestamp": "2025-01-01 10:00:00", "success": True},
    ])

    with pytest.raises(IOError) as exc_info:
        job.seal_day(day_index=1, date_str="2025-01-01", df_day=df_day)

    assert "simulated hardware fault" in str(exc_info.value)
    # Output was never stored or exposed; reservation was safely cancelled
    assert all(r.status == ReleaseStatus.CANCELLED.value for r in ledger.records)
    # Cancelled records contribute exactly 0.0 to rho(h)
    assert ledger.compute_rho(1) == 0.0
    assert ledger.get_release("sdg_additive_req_1").status == ReleaseStatus.CANCELLED.value
    assert ledger.get_release("sdg_additive_fail_1").status == ReleaseStatus.CANCELLED.value


# ==============================================================================
# 6. Multithreaded Concurrency & Thread-Safety
# ==============================================================================

def test_multithreaded_concurrency():
    """Verify concurrent release registrations do not corrupt internal indices."""
    ledger = UnifiedPrivacyLedger(delta=1e-6, allow_unconfigured_policy=True)
    num_threads = 10
    releases_per_thread = 20

    def worker(tid: int):
        for i in range(releases_per_thread):
            rel_id = f"release_t{tid}_i{i}"
            day = (tid * releases_per_thread + i) % 50 + 1
            ledger.record_gaussian_release(
                release_id=rel_id,
                days_accessed=[day],
                sensitivity=1.0,
                sigma=10.0,
                tag=f"thread_{tid}",
            )

    threads = [threading.Thread(target=worker, args=(t,)) for t in range(num_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    total_expected = num_threads * releases_per_thread
    assert len(ledger.records) == total_expected
    assert len(ledger._release_ids) == total_expected
    # Verify index integrity
    day_indices_count = sum(len(indices) for indices in ledger._day_to_records.values())
    assert day_indices_count == total_expected


# ==============================================================================
# 7. Plan Updates Preserve Historical Spend
# ==============================================================================

def test_plan_update_preserves_historical_spend():
    """Verify changing active plan retains all historical release records and indices."""
    ledger = UnifiedPrivacyLedger(delta=1e-6, allow_unconfigured_policy=True)
    # Record 10 days of releases
    for d in range(1, 11):
        ledger.record_gaussian_release(
            release_id=f"hist_req_{d}",
            days_accessed=[d],
            sensitivity=50.0,
            sigma=3000.0,
            tag="historical",
        )

    assert len(ledger.records) == 10
    max_d_before = ledger._max_day
    assert max_d_before == 10

    # New plan arrives
    config = PlannerConfig(commitment=WEEK1_V2_CURVE)
    planner = JointCalendarPlanner(config)
    plan_D = planner.plan_group_d()

    ledger.set_active_plan(plan_D, commitment=planner.commitment)

    # Verify historical records remain fully intact
    assert len(ledger.records) == 10
    assert ledger._max_day == 10
    for d in range(1, 11):
        assert ledger.get_release(f"hist_req_{d}") is not None
        assert d in ledger._day_to_records


# ==============================================================================
# 8. P0 Minimal Counterexample: Group Atomic Rejection
# ==============================================================================

def test_minimal_counterexample_batch_rejection():
    """P0 Counterexample Test:
    rho_bar(1) = 0.0001.
    rho_r = 0.00008, rho_f = 0.00008.
    Each individual check succeeds (0.00008 <= 0.0001).
    Group admission must fail atomically (0.00016 > 0.0001).
    Fail-closed: exactly 0 records committed to ledger.
    """
    ledger = UnifiedPrivacyLedger(delta=1e-6)
    bound = {1: 0.0001}

    cand_r = ReleaseRecord(
        release_id="sdg_req_test",
        days_accessed=frozenset([1]),
        mechanism_type="zcdp",
        params={"rho": 0.00008},
        sensitivity=50.0,
        rho_cost=0.00008,
        tag="sdg_requests",
    )
    cand_f = ReleaseRecord(
        release_id="sdg_fail_test",
        days_accessed=frozenset([1]),
        mechanism_type="zcdp",
        params={"rho": 0.00008},
        sensitivity=10.0,
        rho_cost=0.00008,
        tag="sdg_failures",
    )

    # 1. Verify that naive individual checks would pass falsely
    admitted_r, _ = ledger.check_admission(cand_r, commitment_bound=bound)
    admitted_f, _ = ledger.check_admission(cand_f, commitment_bound=bound)
    assert admitted_r is True, "Individual check r must pass"
    assert admitted_f is True, "Individual check f must pass"

    # 2. Group atomic admission must reject the entire batch
    with pytest.raises(PrivacyAdmissionError) as exc_info:
        ledger.admit_and_commit_batch([cand_r, cand_f], commitment_bound=bound)

    assert "breached bound at h=1" in str(exc_info.value)
    # 3. Fail-closed guarantee: zero residual records in ledger
    assert len(ledger.records) == 0
    assert ledger.get_release("sdg_req_test") is None
    assert ledger.get_release("sdg_fail_test") is None


# ==============================================================================
# 9. P0 Fail-Closed Default on Unconfigured Commitment
# ==============================================================================

def test_fail_closed_unconfigured_commitment():
    """Verify that ledger rejects admissions fail-closed if no commitment is configured."""
    ledger = UnifiedPrivacyLedger(delta=1e-6)
    cand = ReleaseRecord(
        release_id="unconf_1",
        days_accessed=frozenset([1]),
        mechanism_type="zcdp",
        params={"rho": 0.0001},
        sensitivity=1.0,
        rho_cost=0.0001,
    )

    # Default: allow_unconfigured_policy=False -> must reject
    with pytest.raises(PrivacyAdmissionError) as exc_info:
        ledger.admit_and_commit_batch([cand])

    assert "No privacy commitment bound configured" in str(exc_info.value)
    assert len(ledger.records) == 0

    # Explicit escape hatch: allow_unconfigured_policy=True -> admitted
    committed = ledger.admit_and_commit_batch([cand], allow_unconfigured_policy=True)
    assert len(committed) == 1
    assert len(ledger.records) == 1


# ==============================================================================
# 10. P1 Reservation Conversion: Zero Double-Charging
# ==============================================================================

def test_reservation_conversion_zero_double_charging():
    """Verify Planner reservation transitions in-place to COMMITTED with zero budget double-counting."""
    config = PlannerConfig(commitment=WEEK1_V2_CURVE)
    planner = JointCalendarPlanner(config)
    plan_D = planner.plan_group_d()

    ledger = UnifiedPrivacyLedger(delta=1e-6)
    passed, msg, diag = planner.commit_plan_to_ledger(
        plan_D, ledger, start_day=1, simulated_days=30, populate_records=True
    )
    assert passed is True

    # Check Day 1 reservations
    req_res = ledger.get_release("sdg_additive_req_1")
    fail_res = ledger.get_release("sdg_additive_fail_1")
    assert req_res is not None and req_res.status == ReleaseStatus.RESERVED.value
    assert fail_res is not None and fail_res.status == ReleaseStatus.RESERVED.value

    rec_count_before = len(ledger.records)
    rho_1_before = ledger.compute_rho(1)

    # Execute SealJob for Day 1
    store = SealedStore()
    job = SealJob.from_budget(sealed_store=store, ledger=ledger, daily_d=plan_D.daily_d)
    df_day = pd.DataFrame([
        {"user_id": "u1", "provider": "prov1", "task": "t1", "timestamp": "2025-01-01 10:00:00", "success": True},
    ])
    sealed_rec = job.seal_day(day_index=1, date_str="2025-01-01", df_day=df_day, seed=42)
    assert sealed_rec is not None

    # Check that status transitioned in-place to COMMITTED
    req_after = ledger.get_release("sdg_additive_req_1")
    fail_after = ledger.get_release("sdg_additive_fail_1")
    assert req_after.status == ReleaseStatus.COMMITTED.value
    assert fail_after.status == ReleaseStatus.COMMITTED.value

    # Verify zero double-charging: record count and rho(1) remain identical
    assert len(ledger.records) == rec_count_before
    assert math.isclose(ledger.compute_rho(1), rho_1_before, rel_tol=1e-9)


# ==============================================================================
# 11. P1 Cancellation Frees Headroom
# ==============================================================================

def test_cancel_reservation_frees_headroom():
    """Verify cancelling an unreleased RESERVED record frees up budget headroom."""
    ledger = UnifiedPrivacyLedger(delta=1e-6)
    bound = {1: 0.0001}

    rec_res = ReleaseRecord(
        release_id="res_blocker",
        days_accessed=frozenset([1]),
        mechanism_type="zcdp",
        params={"rho": 0.0001},
        sensitivity=1.0,
        rho_cost=0.0001,
        status=ReleaseStatus.RESERVED.value,
    )
    ledger.admit_and_commit_batch([rec_res], commitment_bound=bound, as_status=ReleaseStatus.RESERVED.value)
    assert ledger.compute_rho(1) == 0.0001

    # Candidate release that would exceed bound if res_blocker is active
    rec_next = ReleaseRecord(
        release_id="next_rel",
        days_accessed=frozenset([1]),
        mechanism_type="zcdp",
        params={"rho": 0.00005},
        sensitivity=1.0,
        rho_cost=0.00005,
    )
    with pytest.raises(PrivacyAdmissionError):
        ledger.admit_and_commit_batch([rec_next], commitment_bound=bound)

    # Cancel unreleased reservation
    cancelled = ledger.cancel_reservation("res_blocker")
    assert cancelled is True
    assert ledger.get_release("res_blocker").status == ReleaseStatus.CANCELLED.value
    assert ledger.compute_rho(1) == 0.0  # Headroom freed!

    # Now candidate release passes admission
    committed = ledger.admit_and_commit_batch([rec_next], commitment_bound=bound)
    assert len(committed) == 1
    assert ledger.compute_rho(1) == 0.00005


# ==============================================================================
# 12. P1 Committed Spend Irrevocability
# ==============================================================================

def test_committed_spend_irrevocability():
    """Verify committed releases cannot be cancelled or rolled back without recovery flag."""
    ledger = UnifiedPrivacyLedger(delta=1e-6)
    bound = {1: 0.0005}

    rec = ReleaseRecord(
        release_id="committed_release",
        days_accessed=frozenset([1]),
        mechanism_type="zcdp",
        params={"rho": 0.0001},
        sensitivity=1.0,
        rho_cost=0.0001,
    )
    ledger.admit_and_commit_batch([rec], commitment_bound=bound, as_status=ReleaseStatus.COMMITTED.value)

    # cancel_reservation must refuse
    with pytest.raises(ValueError) as exc1:
        ledger.cancel_reservation("committed_release")
    assert "privacy spend is irrevocable" in str(exc1.value)

    # rollback_release without allow_committed=True must refuse
    with pytest.raises(ValueError) as exc2:
        ledger.rollback_release("committed_release", allow_committed=False)
    assert "committed privacy spend is irrevocable" in str(exc2.value)


# ==============================================================================
# 13. Batch Parameter Conflict Detection
# ==============================================================================

def test_conflicting_batch_parameters_rejected():
    """Verify duplicate release_ids with different parameters in same batch raise ValueError."""
    ledger = UnifiedPrivacyLedger(delta=1e-6)
    c1 = ReleaseRecord(
        release_id="dup_id",
        days_accessed=frozenset([1]),
        mechanism_type="zcdp",
        params={"rho": 0.0001},
        sensitivity=1.0,
        rho_cost=0.0001,
    )
    c2 = ReleaseRecord(
        release_id="dup_id",
        days_accessed=frozenset([1]),
        mechanism_type="zcdp",
        params={"rho": 0.0002},  # Conflicting cost
        sensitivity=1.0,
        rho_cost=0.0002,
    )

    with pytest.raises(ValueError) as exc_info:
        ledger.admit_and_commit_batch([c1, c2], allow_unconfigured_policy=True)

    assert "Duplicate release_id" in str(exc_info.value)
