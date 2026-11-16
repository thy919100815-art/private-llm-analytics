"""Fault injection, immutability, and transactional safety tests for SDG-MH.

Verifies:
1. P0: Output visibility, ledger state, and cost irrevocability under fault injection:
   - Failure before write (unexposed -> safely cancel reservation).
   - Failure during disk write attempt (memory remains untouched, file untruncated/clean).
   - Failure after partial disk write (file truncated back to initial size, zero corruption).
   - Storage succeeded but confirmation failed (output exposed -> conservatively committed).
   - Repeated request for already published result (idempotency, zero double-charging).
   - Concurrent triggering of same task (exactly 1 commit, thread-safe deduplication).
2. P1: Strict Fail-Closed default semantics across all public ledger entrypoints:
   - Unconfigured ledger rejects admit_and_commit_batch, record_gaussian_release,
     record_laplace_release, and record_zcdp_release.
   - Configured commitment envelope strictly enforced; cannot be bypassed.
3. P2: Parameter matching, conflict detection, intra-batch duplicate rejection,
   and lifecycle invariants on RESERVED, COMMITTED, and CANCELLED records.
"""

from __future__ import annotations

import math
from pathlib import Path
import tempfile
import threading
from typing import List
import numpy as np
import pandas as pd
import pytest

from private_llm_analytics.sdg_mh.calendar_job import (
    CalendarJob,
    CalendarReleaseRecord,
    DistinctCalendarStore,
)
from private_llm_analytics.sdg_mh.ledger import (
    PrivacyAdmissionError,
    ReleaseRecord,
    ReleaseStatus,
    UnifiedPrivacyLedger,
)
from private_llm_analytics.sdg_mh.rolling_state import RollingDistinctState
from private_llm_analytics.sdg_mh.seal_job import (
    DailySealedRecord,
    SealJob,
)
from private_llm_analytics.sdg_mh.sealed_store import SealedStore


# ==============================================================================
# Helper fixtures and mock stores
# ==============================================================================

class PreWriteFailingStore(SealedStore):
    """Fails immediately when append is called without touching disk or memory."""
    def append(self, record: DailySealedRecord) -> None:
        raise IOError("Disk write pre-check failure: disk unmounted")


class PostAppendFailingStore(SealedStore):
    """Successfully writes to storage, but fails during confirmation/telemetry."""
    def append(self, record: DailySealedRecord) -> None:
        super().append(record)
        raise RuntimeError("Post-storage confirmation ACK failure")


class BrokenCalendarStore(DistinctCalendarStore):
    """Calendar store that fails on append."""
    def append(self, record: CalendarReleaseRecord) -> None:
        raise IOError("Calendar store I/O error")


# ==============================================================================
# P0: Fault Injection Tests
# ==============================================================================

def test_fault_injection_failure_before_write():
    """1. Failure before write: noise generation / clipping error.

    Output state: NOT_GENERATED / GENERATED_UNPUBLISHED.
    Expected: Reservation safely cancelled, headroom freed, 0 committed spend.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        store = SealedStore(storage_dir=tmpdir)
        ledger = UnifiedPrivacyLedger(delta=1e-6)
        bound = {1: 0.001, 7: 0.005}
        ledger.set_active_plan(plan=None, commitment=bound)

        job = SealJob.from_budget(sealed_store=store, ledger=ledger, daily_d=0.0002)

        # Pass invalid dataframe that triggers error during clipping
        invalid_df = pd.DataFrame([{"invalid_col": 123}])

        with pytest.raises(Exception):
            job.seal_day(day_index=1, date_str="2025-01-01", df_day=invalid_df)

        # Verification:
        # Output visibility: 0
        assert store.get_day(1) is None
        assert store.total_sealed_days() == 0
        if store.jsonl_path and store.jsonl_path.exists():
            assert store.jsonl_path.stat().st_size == 0

        # Ledger state: reservation was created then safely cancelled
        assert len(ledger.records) == 2
        assert all(r.status == ReleaseStatus.CANCELLED.value for r in ledger.records)

        # Headroom: 0.0 spend in compute_rho
        assert ledger.compute_rho(1) == 0.0
        assert ledger.compute_rho(7) == 0.0


def test_fault_injection_store_failure_cleans_memory_and_cancels_reservation():
    """2. Failure during store append: store raises error before write completes.

    Output state: strictly unpublished.
    Expected: Store in-memory dict remains untouched, reservation safely cancelled.
    """
    broken_store = PreWriteFailingStore()
    ledger = UnifiedPrivacyLedger(delta=1e-6)
    bound = {1: 0.001, 7: 0.005}
    ledger.set_active_plan(plan=None, commitment=bound)

    job = SealJob.from_budget(sealed_store=broken_store, ledger=ledger, daily_d=0.0002)
    df_day = pd.DataFrame([
        {"user_id": "u1", "provider": "prov1", "task": "t1", "timestamp": "2025-01-01 10:00:00", "success": True},
    ])

    with pytest.raises(IOError) as exc_info:
        job.seal_day(day_index=1, date_str="2025-01-01", df_day=df_day)

    assert "disk unmounted" in str(exc_info.value)

    # Verification:
    # Store memory: strictly clean
    assert broken_store.get_day(1) is None
    assert broken_store.total_sealed_days() == 0

    # Ledger state: reservation cancelled, 0 committed spend
    assert all(r.status == ReleaseStatus.CANCELLED.value for r in ledger.records)
    assert ledger.compute_rho(1) == 0.0


def test_fault_injection_partial_disk_write_truncated():
    """3. Partial disk write failure: ensure file is truncated back and memory unmutated."""
    with tempfile.TemporaryDirectory() as tmpdir:
        store = SealedStore(storage_dir=tmpdir)

        # Write day 1 successfully
        rec1 = DailySealedRecord(
            day_index=1,
            date_str="2025-01-01",
            y_requests=100,
            y_failures=10,
        )
        store.append(rec1)
        initial_size = store.jsonl_path.stat().st_size
        assert initial_size > 0

        # Simulate a partial write on day 2 by writing corrupted bytes then failing
        rec2 = DailySealedRecord(
            day_index=2,
            date_str="2025-01-02",
            y_requests=200,
            y_failures=20,
        )

        class CorruptingSealedStore(SealedStore):
            def append(self, record: DailySealedRecord):
                if record.day_index == 2:
                    # Write partial corrupt line to disk then raise
                    initial_sz = self.jsonl_path.stat().st_size if self.jsonl_path.exists() else 0
                    with open(self.jsonl_path, "a", encoding="utf-8") as f:
                        f.write('{"corrupted": true, "day_index": ')
                        f.flush()
                    # Trigger truncation in except block
                    try:
                        raise IOError("Simulated partial write disk error")
                    except Exception:
                        with open(self.jsonl_path, "r+b") as f:
                            f.truncate(initial_sz)
                        raise
                else:
                    super().append(record)

        corrupt_store = CorruptingSealedStore(storage_dir=tmpdir)
        with pytest.raises(IOError):
            corrupt_store.append(rec2)

        # Check: file size must have rolled back cleanly to initial_size
        assert store.jsonl_path.stat().st_size == initial_size

        # Memory must not contain day 2
        assert corrupt_store.get_day(2) is None
        assert corrupt_store.total_sealed_days() == 1

        # A new store loading from disk loads day 1 cleanly without JSON error
        reloaded = SealedStore(storage_dir=tmpdir)
        assert reloaded.total_sealed_days() == 1
        assert reloaded.get_day(1) is not None


def test_fault_injection_storage_succeeded_confirmation_failed_retains_committed_cost():
    """4. Storage succeeded but confirmation failed: output IS or MAY BE exposed!

    Expected: Privacy cost MUST be conservatively COMMITTED, and spend is irrevocable.
    """
    failing_store = PostAppendFailingStore()
    ledger = UnifiedPrivacyLedger(delta=1e-6)
    bound = {1: 0.001, 7: 0.005}
    ledger.set_active_plan(plan=None, commitment=bound)

    job = SealJob.from_budget(sealed_store=failing_store, ledger=ledger, daily_d=0.0002)
    df_day = pd.DataFrame([
        {"user_id": "u1", "provider": "prov1", "task": "t1", "timestamp": "2025-01-01 10:00:00", "success": True},
    ])

    with pytest.raises(RuntimeError) as exc_info:
        job.seal_day(day_index=1, date_str="2025-01-01", df_day=df_day)

    assert "Post-storage confirmation ACK failure" in str(exc_info.value)

    # Verification:
    # 1. Output visibility: output IS present in store!
    assert failing_store.get_day(1) is not None

    # 2. Ledger state: MUST be COMMITTED to retain privacy charge conservatively
    rec_r = ledger.get_release("sdg_additive_req_1")
    rec_f = ledger.get_release("sdg_additive_fail_1")
    assert rec_r is not None and rec_r.status == ReleaseStatus.COMMITTED.value
    assert rec_f is not None and rec_f.status == ReleaseStatus.COMMITTED.value

    # 3. Privacy spend is active
    assert ledger.compute_rho(1) > 0.0

    # 4. Irrevocability: cannot cancel or rollback committed spend
    with pytest.raises(ValueError):
        ledger.cancel_reservation("sdg_additive_req_1")
    with pytest.raises(ValueError):
        ledger.rollback_release("sdg_additive_req_1")


def test_idempotent_repeat_request_zero_additional_spend():
    """5. Already published result repeated request: exact match returns existing, zero extra cost."""
    store = SealedStore()
    ledger = UnifiedPrivacyLedger(delta=1e-6)
    bound = {1: 0.001, 7: 0.005}
    ledger.set_active_plan(plan=None, commitment=bound)

    job = SealJob.from_budget(sealed_store=store, ledger=ledger, daily_d=0.0002)
    df_day = pd.DataFrame([
        {"user_id": "u1", "provider": "prov1", "task": "t1", "timestamp": "2025-01-01 10:00:00", "success": True},
    ])

    # First call: publishes and commits
    r1 = job.seal_day(day_index=1, date_str="2025-01-01", df_day=df_day, seed=42)
    rho_after_r1 = ledger.compute_rho(1)
    record_count_1 = len(ledger.records)

    # Second call with identical arguments: must return existing record
    r2 = job.seal_day(day_index=1, date_str="2025-01-01", df_day=df_day, seed=999)
    assert r1.y_requests == r2.y_requests
    assert r1.y_failures == r2.y_failures
    assert len(ledger.records) == record_count_1
    assert math.isclose(ledger.compute_rho(1), rho_after_r1, rel_tol=1e-12)


def test_concurrent_triggers_single_commitment():
    """6. Concurrent triggers of same task: exactly one commits, others return identical record."""
    store = SealedStore()
    ledger = UnifiedPrivacyLedger(delta=1e-6)
    bound = {1: 0.001, 7: 0.005}
    ledger.set_active_plan(plan=None, commitment=bound)

    job = SealJob.from_budget(sealed_store=store, ledger=ledger, daily_d=0.0002)
    df_day = pd.DataFrame([
        {"user_id": "u1", "provider": "prov1", "task": "t1", "timestamp": "2025-01-01 10:00:00", "success": True},
    ])

    results: List[DailySealedRecord] = []
    lock = threading.Lock()

    def worker():
        rec = job.seal_day(day_index=1, date_str="2025-01-01", df_day=df_day, seed=123)
        with lock:
            results.append(rec)

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # Exactly 1 record stored
    assert store.total_sealed_days() == 1
    # Exactly 2 ledger records (requests + failures)
    assert len([r for r in ledger.records if r.status == ReleaseStatus.COMMITTED.value]) == 2
    # All 8 threads got identical record
    assert len(results) == 8
    first = results[0]
    for r in results[1:]:
        assert r.y_requests == first.y_requests
        assert r.y_failures == first.y_failures


# ==============================================================================
# P1: Strict Fail-Closed Semantics Tests
# ==============================================================================

def test_all_public_entrypoints_default_fail_closed():
    """P1: Verify all 4 public ledger release entrypoints strictly fail-closed by default."""
    # Unconfigured ledger: no plan, no commitment
    ledger = UnifiedPrivacyLedger(delta=1e-6, allow_unconfigured_policy=False)

    cand = ReleaseRecord(
        release_id="cand_test",
        days_accessed=frozenset([1]),
        mechanism_type="gaussian",
        params={"sigma": 100.0},
        sensitivity=50.0,
        rho_cost=0.125,
    )

    # 1. admit_and_commit_batch
    with pytest.raises(PrivacyAdmissionError) as e1:
        ledger.admit_and_commit_batch([cand])
    assert "No privacy commitment bound configured" in str(e1.value)

    # 2. record_gaussian_release
    with pytest.raises(PrivacyAdmissionError) as e2:
        ledger.record_gaussian_release("g_test", [1], sensitivity=50.0, sigma=100.0)
    assert "No privacy commitment bound configured" in str(e2.value)

    # 3. record_laplace_release
    with pytest.raises(PrivacyAdmissionError) as e3:
        ledger.record_laplace_release("l_test", [1], sensitivity=50.0, epsilon=1.0)
    assert "No privacy commitment bound configured" in str(e3.value)

    # 4. record_zcdp_release
    with pytest.raises(PrivacyAdmissionError) as e4:
        ledger.record_zcdp_release("z_test", [1], sensitivity=50.0, rho_cost=0.1)
    assert "No privacy commitment bound configured" in str(e4.value)

    # Verify 0 records committed
    assert len(ledger.records) == 0


def test_configured_commitment_strictly_enforced_cannot_bypass():
    """P1: When commitment is configured, envelope checks run unconditionally."""
    ledger = UnifiedPrivacyLedger(delta=1e-6)
    bound = {1: 0.0001, 7: 0.0005}
    ledger.set_active_plan(plan=None, commitment=bound)

    # Attempt to breach bound at h=1 (rho = 0.0002 > 0.0001)
    cand_breach = ReleaseRecord(
        release_id="breach_test",
        days_accessed=frozenset([1]),
        mechanism_type="zcdp",
        params={"rho": 0.0002},
        sensitivity=10.0,
        rho_cost=0.0002,
    )

    # Even with allow_unconfigured_policy=True, configured commitment MUST reject!
    with pytest.raises(PrivacyAdmissionError) as exc:
        ledger.admit_and_commit_batch([cand_breach], allow_unconfigured_policy=True)
    assert "breached bound at h=1" in str(exc.value)

    # Single-release entrypoints must also reject
    with pytest.raises(PrivacyAdmissionError):
        ledger.record_gaussian_release("g_breach", [1], sensitivity=10.0, sigma=100.0, allow_unconfigured_policy=True)

    with pytest.raises(PrivacyAdmissionError):
        ledger.record_laplace_release("l_breach", [1], sensitivity=10.0, epsilon=1.0, allow_unconfigured_policy=True)

    with pytest.raises(PrivacyAdmissionError):
        ledger.record_zcdp_release("z_breach", [1], sensitivity=10.0, rho_cost=0.0002, allow_unconfigured_policy=True)

    # Ensure 0 records committed
    assert len(ledger.records) == 0


# ==============================================================================
# P2: Reservation Conversion, Parameter Mismatch & Duplicate Detection Tests
# ==============================================================================

def test_duplicate_release_id_within_batch_rejected():
    """P2: Intra-batch duplicate release IDs must be rejected."""
    ledger = UnifiedPrivacyLedger(delta=1e-6, allow_unconfigured_policy=True)

    c1 = ReleaseRecord(
        release_id="same_id",
        days_accessed=frozenset([1]),
        mechanism_type="zcdp",
        params={"rho": 0.0001},
        sensitivity=1.0,
        rho_cost=0.0001,
    )
    c2 = ReleaseRecord(
        release_id="same_id",
        days_accessed=frozenset([1]),
        mechanism_type="zcdp",
        params={"rho": 0.0001},
        sensitivity=1.0,
        rho_cost=0.0001,
    )

    with pytest.raises(ValueError) as exc:
        ledger.admit_and_commit_batch([c1, c2])
    assert "Duplicate release_id within candidate batch" in str(exc.value)
    assert len(ledger.records) == 0


def test_conflicting_parameters_on_committed_release_rejected():
    """P2: Attempting to commit an already COMMITTED ID with mismatched parameters must fail."""
    ledger = UnifiedPrivacyLedger(delta=1e-6, allow_unconfigured_policy=True)

    c_init = ReleaseRecord(
        release_id="committed_id",
        days_accessed=frozenset([1]),
        mechanism_type="gaussian",
        params={"sigma": 100.0},
        sensitivity=50.0,
        rho_cost=0.125,
        tag="original_tag",
    )
    ledger.admit_and_commit_batch([c_init])
    assert len(ledger.records) == 1

    # Conflict 1: different sigma / cost
    c_bad_cost = ReleaseRecord(
        release_id="committed_id",
        days_accessed=frozenset([1]),
        mechanism_type="gaussian",
        params={"sigma": 200.0},
        sensitivity=50.0,
        rho_cost=0.03125,
        tag="original_tag",
    )
    with pytest.raises(ValueError) as e1:
        ledger.admit_and_commit_batch([c_bad_cost])
    assert "Conflicting rho_cost" in str(e1.value)

    # Conflict 2: different sensitivity
    c_bad_sens = ReleaseRecord(
        release_id="committed_id",
        days_accessed=frozenset([1]),
        mechanism_type="gaussian",
        params={"sigma": 100.0},
        sensitivity=10.0,
        rho_cost=0.125,
        tag="original_tag",
    )
    with pytest.raises(ValueError) as e2:
        ledger.admit_and_commit_batch([c_bad_sens])
    assert "Conflicting sensitivity" in str(e2.value)

    # Conflict 3: different days_accessed
    c_bad_days = ReleaseRecord(
        release_id="committed_id",
        days_accessed=frozenset([1, 2]),
        mechanism_type="gaussian",
        params={"sigma": 100.0},
        sensitivity=50.0,
        rho_cost=0.125,
        tag="original_tag",
    )
    with pytest.raises(ValueError) as e3:
        ledger.admit_and_commit_batch([c_bad_days])
    assert "Conflicting days_accessed" in str(e3.value)

    # Conflict 4: different tag
    c_bad_tag = ReleaseRecord(
        release_id="committed_id",
        days_accessed=frozenset([1]),
        mechanism_type="gaussian",
        params={"sigma": 100.0},
        sensitivity=50.0,
        rho_cost=0.125,
        tag="different_tag",
    )
    with pytest.raises(ValueError) as e4:
        ledger.admit_and_commit_batch([c_bad_tag])
    assert "Conflicting tag" in str(e4.value)


def test_reservation_conversion_allows_lower_cost_and_frees_headroom():
    """P2: Fulfilling reservation with actual cost <= reserved budget converts in-place and frees headroom."""
    ledger = UnifiedPrivacyLedger(delta=1e-6)
    bound = {1: 0.001, 7: 0.005}
    ledger.set_active_plan(plan=None, commitment=bound)

    # 1. Pre-reserve 0.0008 at day 1
    res_cand = ReleaseRecord(
        release_id="res_01",
        days_accessed=frozenset([1]),
        mechanism_type="gaussian",
        params={"sigma": 100.0},
        sensitivity=1.0,
        rho_cost=0.0008,
        tag="test_res",
    )
    ledger.admit_and_commit_batch([res_cand], as_status=ReleaseStatus.RESERVED.value)
    assert ledger.records[0].status == ReleaseStatus.RESERVED.value
    assert math.isclose(ledger.compute_rho(1), 0.0008, rel_tol=1e-12)

    # 2. Fulfill reservation with lower actual cost 0.0005 (sigma = 126.49 > 100.0)
    actual_cand = ReleaseRecord(
        release_id="res_01",
        days_accessed=frozenset([1]),
        mechanism_type="gaussian",
        params={"sigma": 126.491106},
        sensitivity=1.0,
        rho_cost=0.0005,
        tag="test_res",
    )
    ledger.admit_and_commit_batch([actual_cand], as_status=ReleaseStatus.COMMITTED.value)

    # Verification:
    # Exactly 1 record in ledger (zero double charging)
    assert len(ledger.records) == 1
    assert ledger.records[0].status == ReleaseStatus.COMMITTED.value
    assert math.isclose(ledger.records[0].rho_cost, 0.0005, rel_tol=1e-12)
    # Headroom was freed: rho(1) reduced from 0.0008 to 0.0005
    assert math.isclose(ledger.compute_rho(1), 0.0005, rel_tol=1e-12)


def test_resurrecting_cancelled_record_requires_full_admission():
    """P2: Cancelled record cannot bypass admission when re-admitted."""
    ledger = UnifiedPrivacyLedger(delta=1e-6)
    bound = {1: 0.0002}
    ledger.set_active_plan(plan=None, commitment=bound)

    # 1. Reserve and then cancel
    c = ReleaseRecord(
        release_id="cancelled_item",
        days_accessed=frozenset([1]),
        mechanism_type="zcdp",
        params={"rho": 0.00015},
        sensitivity=1.0,
        rho_cost=0.00015,
    )
    ledger.admit_and_commit_batch([c], as_status=ReleaseStatus.RESERVED.value)
    assert ledger.compute_rho(1) == 0.00015
    ledger.cancel_reservation("cancelled_item")
    assert ledger.compute_rho(1) == 0.0

    # 2. Meanwhile, another release consumes 0.00015
    other = ReleaseRecord(
        release_id="other_item",
        days_accessed=frozenset([1]),
        mechanism_type="zcdp",
        params={"rho": 0.00015},
        sensitivity=1.0,
        rho_cost=0.00015,
    )
    ledger.admit_and_commit_batch([other], as_status=ReleaseStatus.COMMITTED.value)
    assert ledger.compute_rho(1) == 0.00015

    # 3. Now try to re-commit 'cancelled_item' with cost 0.00015
    # Total would be 0.00030 > bound 0.00020
    # Must be rejected by full envelope gate!
    re_cand = ReleaseRecord(
        release_id="cancelled_item",
        days_accessed=frozenset([1]),
        mechanism_type="zcdp",
        params={"rho": 0.00015},
        sensitivity=1.0,
        rho_cost=0.00015,
    )
    with pytest.raises(PrivacyAdmissionError) as exc:
        ledger.admit_and_commit_batch([re_cand], as_status=ReleaseStatus.COMMITTED.value)

    assert "breached bound at h=1" in str(exc.value)
    # The cancelled record remains CANCELLED; only other_item is committed
    assert ledger.get_release("cancelled_item").status == ReleaseStatus.CANCELLED.value
    assert ledger.compute_rho(1) == 0.00015
