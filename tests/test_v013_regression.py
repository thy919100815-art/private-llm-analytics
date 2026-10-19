"""Regression tests for Private LLM Analytics v0.1.3 Freeze Candidate.

Covers:
  - C10: Single source of truth (no legacy cache fallback, reusable payload implies committed reservation).
  - C11: Single-flight memoization, slow owner waiting, CAS generation fencing, and stale worker rejection.
  - C12: Commit irreversibility, post-commit fault survival, and abort fencing.
  - H9: Fresh database zero-budget hygiene, state cleanup.
  - H10: Deployment preconditions, recipient policies, and schema backward compatibility.
"""

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path
import sqlite3
import time
import numpy as np
import pytest

from private_llm_analytics.aggregation.user_matrix import (
    UserMatrix,
    build_cell_definitions,
)
from private_llm_analytics.config.loader import load_app_config
from private_llm_analytics.metrics.base import DPReleaseResult
from private_llm_analytics.planner.query import AnalyticsQuery
from private_llm_analytics.privacy.budget import PrivacyBudget
from private_llm_analytics.privacy.exceptions import (
    CommitIrreversibleError,
    PrivacyBudgetExceededError,
    RecipientSecurityViolation,
    ReleaseInProgressError,
    ReservationLostError,
)
from private_llm_analytics.privacy.state import PrivacyStateManager, ReservationTuple
from private_llm_analytics.release.cache import PersistentReleaseStore
from private_llm_analytics.release.release import ReleaseOrchestrator


@pytest.fixture
def v013_test_matrix() -> UserMatrix:
    provs = ["provider_a", "provider_b", "provider_c"]
    tsks = ["technical", "document", "analysis"]
    cell_names, cell_map = build_cell_definitions(provs, tsks)
    np_matrix = np.ones((10, 9), dtype=np.float64) * 20.0
    return UserMatrix(
        matrix=np_matrix,
        user_ids=[f"user_{i}" for i in range(10)],
        providers=provs,
        tasks=tsks,
        cell_names=cell_names,
        cell_map=cell_map,
        public_population_size=50000,
    )


# =========================================================================
# 1. Critical Fix 1: Single Source of Truth (Audit C10)
# =========================================================================

def test_v013_single_source_of_truth_no_legacy_cache_fallback(tmp_path, v013_test_matrix):
    """Test that legacy cache payload is strictly ignored if absent from PrivacyStateManager."""
    config = load_app_config("configs")
    state_db = tmp_path / "privacy_state.db"
    legacy_cache_db = tmp_path / "legacy_cache.db"

    state_mgr = PrivacyStateManager(state_db)
    legacy_cache = PersistentReleaseStore(legacy_cache_db)

    orch = ReleaseOrchestrator(app_config=config, state_manager=state_mgr, cache=legacy_cache)

    q = AnalyticsQuery(metric="user_normalized_share", recipient="internal", epsilon=1.0, delta=1e-6)
    plan = orch.planner.create_plan(q, public_population_size=v013_test_matrix.public_population_size)

    # Populate ONLY legacy cache with a dummy release under the canonical release key
    dummy_result = DPReleaseResult(
        metric_name="user_normalized_share",
        dp_vector=np.array([0.5] * 9),
        post_processed={"cell_shares": {c: 0.5 for c in v013_test_matrix.cell_names}},
        epsilon=1.0,
        delta=1e-6,
        mechanism="Gaussian",
        noise_scale=1.0,
        sensitivity=1.0,
        cell_names=v013_test_matrix.cell_names,
        n_users=10,
        public_population_size=50000,
    )
    legacy_cache.put(plan.release_key, dummy_result)

    # Assert that authoritative state reports zero spent and no committed release
    assert state_mgr.get_total_spent() == (0.0, 0.0)
    assert state_mgr.get_committed_release(plan.release_key) is None

    # Execute release: Must NOT return the un-accounted legacy cache entry as a cache hit
    res = orch.execute_release(q, v013_test_matrix)

    # Must be a fresh execution (cache_hit=False) that consumed privacy budget in state_mgr
    assert res["cache_hit"] is False
    assert state_mgr.get_total_spent()[0] == 1.0

    # The payload in state_mgr is now authoritative
    authoritative_release = state_mgr.get_committed_release(plan.release_key)
    assert authoritative_release is not None


def test_v013_reusable_payload_strictly_requires_committed_reservation(tmp_path):
    """Test get_committed_release enforces payload + committed reservation join."""
    state_mgr = PrivacyStateManager(tmp_path / "privacy_state.db")

    with state_mgr._connection() as conn:
        # Insert a release in release_store marked COMMITTED, but reservation is ABORTED
        conn.execute("""
            INSERT INTO privacy_reservations (reservation_id, release_key, owner_id, generation, epsilon, delta, status, created_at)
            VALUES ('res_orphan', 'key_orphan', 'worker1', 1, 1.0, 1e-6, 'ABORTED', '2026-01-01T00:00:00Z')
        """)
        conn.execute("""
            INSERT INTO release_store (
                release_key, reservation_id, owner_id, generation, release_id, created_at, status,
                canonical_semantics_json, metric, dataset_version, mechanism, epsilon, delta, sensitivity,
                noise_scale, cell_names_json, n_users, public_population_size, dp_vector_json, post_processed_json
            ) VALUES (
                'key_orphan', 'res_orphan', 'worker1', 1, 'rel_orphan', '2026-01-01T00:00:00Z', 'COMMITTED',
                '{}', 'metric', 'v1', 'Gaussian', 1.0, 1e-6, 1.0, 1.0, '[]', 10, 50000, '[1.0]', '{}'
            )
        """)
        conn.commit()

    # Even though release_store status is COMMITTED, reservation is ABORTED -> must return None
    assert state_mgr.get_committed_release("key_orphan") is None


# =========================================================================
# 2. Critical Fix 2: Single-Flight, Slow Owner, and CAS Fencing (Audit C11)
# =========================================================================

def test_v013_single_flight_slow_owner_thread_memoization(tmp_path, v013_test_matrix):
    """Test that a slower in-flight worker is waited on and memoized, not duplicated."""
    config = load_app_config("configs")
    state_mgr = PrivacyStateManager(tmp_path / "privacy_state.db")

    # Injected slow delay on Worker A
    hooks_a = {"after_budget_reserve": lambda: time.sleep(0.4)}
    orch_a = ReleaseOrchestrator(app_config=config, state_manager=state_mgr, test_hooks=hooks_a)
    orch_b = ReleaseOrchestrator(app_config=config, state_manager=state_mgr)

    query = AnalyticsQuery(metric="user_normalized_share", recipient="internal", epsilon=1.0, delta=1e-6)

    def run_worker_a():
        return orch_a.execute_release(query, v013_test_matrix)

    def run_worker_b():
        time.sleep(0.05)  # Start slightly after worker A
        return orch_b.execute_release(query, v013_test_matrix)

    with ThreadPoolExecutor(max_workers=2) as executor:
        fut_a = executor.submit(run_worker_a)
        fut_b = executor.submit(run_worker_b)

        res_a = fut_a.result()
        res_b = fut_b.result()

    assert res_a["cache_hit"] is False
    assert res_b["cache_hit"] is True

    # Bitwise identical
    assert res_a["results"]["cell_shares"] == res_b["results"]["cell_shares"]

    # Budget charged exactly once
    spent_eps, _ = state_mgr.get_total_spent()
    assert spent_eps == 1.0


def test_v013_cas_commit_generation_fencing_rejects_stale_worker(tmp_path):
    """Test compare-and-set fencing: worker with stale generation is rejected."""
    state_mgr = PrivacyStateManager(tmp_path / "privacy_state.db")
    global_budget = PrivacyBudget(epsilon=5.0, delta=1e-5)
    query_budget = PrivacyBudget(epsilon=1.0, delta=1e-6)
    release_key = "test_fencing_key"

    # Worker A reserves gen 1
    res_a = state_mgr.reserve_or_get_release(
        release_key=release_key,
        canonical_semantics={"k": "v"},
        global_budget=global_budget,
        budget_needed=query_budget,
        plan_metadata={},
        owner_id="worker_a",
        lease_timeout=0.2,
    )
    assert res_a.generation == 1
    res_id_a = res_a[2]
    rel_id_a = res_a[3]

    # Lease expires
    time.sleep(0.3)

    # Worker B takes over at gen 2
    res_b = state_mgr.reserve_or_get_release(
        release_key=release_key,
        canonical_semantics={"k": "v"},
        global_budget=global_budget,
        budget_needed=query_budget,
        plan_metadata={},
        owner_id="worker_b",
        lease_timeout=10.0,
    )
    assert res_b.generation == 2
    res_id_b = res_b[2]
    rel_id_b = res_b[3]

    # Worker B commits gen 2
    dummy_b = DPReleaseResult(
        metric_name="user_prevalence",
        dp_vector=np.array([1.0, 2.0]),
        post_processed={},
        epsilon=1.0,
        delta=1e-6,
        mechanism="Gaussian",
        noise_scale=1.0,
        sensitivity=1.0,
        cell_names=["c1", "c2"],
        n_users=10,
        public_population_size=50000,
    )
    state_mgr.commit_release(
        reservation_id=res_id_b,
        release_key=release_key,
        release_id=rel_id_b,
        release_result=dummy_b,
        recipient="internal",
        dataset_version="v1.0",
        generation=2,
    )

    # Worker A attempts to commit gen 1 -> MUST BE REJECTED
    dummy_a = DPReleaseResult(
        metric_name="user_prevalence",
        dp_vector=np.array([9.9, 9.9]),
        post_processed={},
        epsilon=1.0,
        delta=1e-6,
        mechanism="Gaussian",
        noise_scale=1.0,
        sensitivity=1.0,
        cell_names=["c1", "c2"],
        n_users=10,
        public_population_size=50000,
    )

    with pytest.raises(ReservationLostError, match="lost, expired, or already finalized"):
        state_mgr.commit_release(
            reservation_id=res_id_a,
            release_key=release_key,
            release_id=rel_id_a,
            release_result=dummy_a,
            recipient="internal",
            dataset_version="v1.0",
            generation=1,
        )

    # Committed payload is still Worker B's
    committed = state_mgr.get_committed_release(release_key)
    np.testing.assert_allclose(committed.dp_vector, np.array([1.0, 2.0]))
    assert state_mgr.get_total_spent()[0] == 1.0


def test_v013_abort_reservation_fenced_cannot_abort_superseded_generation(tmp_path):
    """Test that a stale worker's abort does not affect a newer owner's reservation."""
    state_mgr = PrivacyStateManager(tmp_path / "privacy_state.db")
    global_budget = PrivacyBudget(epsilon=5.0, delta=1e-5)
    query_budget = PrivacyBudget(epsilon=1.0, delta=1e-6)
    release_key = "test_abort_fencing"

    # Worker A reserves gen 1
    res_a = state_mgr.reserve_or_get_release(
        release_key=release_key,
        canonical_semantics={"k": "v"},
        global_budget=global_budget,
        budget_needed=query_budget,
        plan_metadata={},
        lease_timeout=0.1,
    )
    res_id_a = res_a[2]

    time.sleep(0.15)

    # Worker B takes over gen 2
    res_b = state_mgr.reserve_or_get_release(
        release_key=release_key,
        canonical_semantics={"k": "v"},
        global_budget=global_budget,
        budget_needed=query_budget,
        plan_metadata={},
        lease_timeout=10.0,
    )
    assert res_b.generation == 2
    res_id_b = res_b[2]

    # Worker A attempts to abort gen 1
    state_mgr.abort_reservation(res_id_a, release_key, generation=1)

    # Worker B's reservation must still be PENDING
    with state_mgr._connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT status FROM privacy_reservations WHERE reservation_id = ?", (res_id_b,))
        assert cursor.fetchone()[0] == "PENDING"
        cursor.execute("SELECT status, generation FROM release_store WHERE release_key = ?", (release_key,))
        row = cursor.fetchone()
        assert row[0] == "PENDING"
        assert row[1] == 2


def test_v013_release_in_progress_error_on_wait_timeout(tmp_path):
    """Test that ReleaseInProgressError is raised when single_flight_timeout is reached."""
    state_mgr = PrivacyStateManager(tmp_path / "privacy_state.db")
    global_budget = PrivacyBudget(epsilon=5.0, delta=1e-5)
    query_budget = PrivacyBudget(epsilon=1.0, delta=1e-6)
    release_key = "in_flight_timeout_key"

    # Worker A creates reservation with long lease (100s)
    state_mgr.reserve_or_get_release(
        release_key=release_key,
        canonical_semantics={"k": "v"},
        global_budget=global_budget,
        budget_needed=query_budget,
        plan_metadata={},
        lease_timeout=100.0,
    )

    # Worker B tries to reserve with very short wait timeout (0.15s)
    with pytest.raises(ReleaseInProgressError, match="actively being computed"):
        state_mgr.reserve_or_get_release(
            release_key=release_key,
            canonical_semantics={"k": "v"},
            global_budget=global_budget,
            budget_needed=query_budget,
            plan_metadata={},
            single_flight_timeout=0.15,
            lease_timeout=100.0,
        )


# =========================================================================
# 3. Critical Fix 3: Commit Irreversibility & Post-Commit Faults (Audit C12)
# =========================================================================

def test_v013_abort_reservation_on_committed_raises_commit_irreversible(tmp_path):
    """Test that abort_reservation raises CommitIrreversibleError on committed reservation."""
    state_mgr = PrivacyStateManager(tmp_path / "privacy_state.db")
    global_budget = PrivacyBudget(epsilon=5.0, delta=1e-5)
    query_budget = PrivacyBudget(epsilon=1.0, delta=1e-6)

    res = state_mgr.reserve_or_get_release(
        release_key="key_commit_irrev",
        canonical_semantics={},
        global_budget=global_budget,
        budget_needed=query_budget,
        plan_metadata={},
    )
    res_id = res[2]
    rel_id = res[3]

    dummy = DPReleaseResult(
        metric_name="user_prevalence",
        dp_vector=np.array([1.0]),
        post_processed={},
        epsilon=1.0,
        delta=1e-6,
        mechanism="Gaussian",
        noise_scale=1.0,
        sensitivity=1.0,
        cell_names=["c1"],
        n_users=10,
        public_population_size=50000,
    )
    state_mgr.commit_release(
        reservation_id=res_id,
        release_key="key_commit_irrev",
        release_id=rel_id,
        release_result=dummy,
        recipient="internal",
        dataset_version="v1.0",
    )

    # Attempting to abort after commit MUST raise CommitIrreversibleError
    with pytest.raises(CommitIrreversibleError, match="already been COMMITTED"):
        state_mgr.abort_reservation(res_id, "key_commit_irrev")

    # Status remains COMMITTED
    with state_mgr._connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT status FROM privacy_reservations WHERE reservation_id = ?", (res_id,))
        assert cursor.fetchone()[0] == "COMMITTED"


def test_v013_post_commit_exception_preserves_committed_release(tmp_path, v013_test_matrix):
    """Test that ReleaseOrchestrator preserves release when exception occurs post-commit."""
    config = load_app_config("configs")
    state_mgr = PrivacyStateManager(tmp_path / "privacy_state.db")
    hooks = {"after_release_commit": RuntimeError("Simulated crash after DB commit before HTTP response")}
    orch = ReleaseOrchestrator(app_config=config, state_manager=state_mgr, test_hooks=hooks)

    q = AnalyticsQuery(metric="user_normalized_share", recipient="internal", epsilon=1.0, delta=1e-6)

    # Execution crashes after commit
    with pytest.raises(RuntimeError, match="Simulated crash after DB commit"):
        orch.execute_release(q, v013_test_matrix)

    # Crucial check: Authoritative status remains COMMITTED and spend is 1.0
    spent_eps, _ = state_mgr.get_total_spent()
    assert spent_eps == 1.0

    with state_mgr._connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT status FROM privacy_reservations")
        assert cursor.fetchone()[0] == "COMMITTED"
        cursor.execute("SELECT status FROM release_store")
        assert cursor.fetchone()[0] == "COMMITTED"

    # Subsequent restart query hits cache without error or duplicate spend
    orch_fresh = ReleaseOrchestrator(app_config=config, state_manager=state_mgr)
    res_hit = orch_fresh.execute_release(q, v013_test_matrix)
    assert res_hit["cache_hit"] is True
    assert state_mgr.get_total_spent()[0] == 1.0


# =========================================================================
# 4. Stale Recovery & Lease Semantics (Audit C11 / H9)
# =========================================================================

def test_v013_stale_recovery_age_bounded_preserves_active_workers(tmp_path):
    """Test recover_stale_reservations only aborts reservations older than max_age_seconds."""
    state_mgr = PrivacyStateManager(tmp_path / "privacy_state.db")
    now = datetime.now(timezone.utc)
    old_time = (now - timedelta(seconds=600)).isoformat()
    fresh_time = (now - timedelta(seconds=10)).isoformat()

    with state_mgr._connection() as conn:
        conn.execute("""
            INSERT INTO privacy_reservations (reservation_id, release_key, owner_id, generation, epsilon, delta, status, created_at)
            VALUES ('res_old', 'key_old', 'w1', 1, 1.0, 1e-6, 'PENDING', ?)
        """, (old_time,))
        conn.execute("""
            INSERT INTO release_store (release_key, reservation_id, owner_id, generation, release_id, created_at, status,
                canonical_semantics_json, metric, dataset_version, mechanism, epsilon, delta, sensitivity, noise_scale, cell_names_json, n_users)
            VALUES ('key_old', 'res_old', 'w1', 1, 'rel_old', ?, 'PENDING', '{}', 'metric', 'v1', 'Gaussian', 1.0, 1e-6, 1.0, 1.0, '[]', 10)
        """, (old_time,))

        conn.execute("""
            INSERT INTO privacy_reservations (reservation_id, release_key, owner_id, generation, epsilon, delta, status, created_at)
            VALUES ('res_fresh', 'key_fresh', 'w2', 1, 1.0, 1e-6, 'PENDING', ?)
        """, (fresh_time,))
        conn.execute("""
            INSERT INTO release_store (release_key, reservation_id, owner_id, generation, release_id, created_at, status,
                canonical_semantics_json, metric, dataset_version, mechanism, epsilon, delta, sensitivity, noise_scale, cell_names_json, n_users)
            VALUES ('key_fresh', 'res_fresh', 'w2', 1, 'rel_fresh', ?, 'PENDING', '{}', 'metric', 'v1', 'Gaussian', 1.0, 1e-6, 1.0, 1.0, '[]', 10)
        """, (fresh_time,))
        conn.commit()

    # Run recovery with max_age_seconds=300
    swept = state_mgr.recover_stale_reservations(max_age_seconds=300.0)
    assert swept == 1

    with state_mgr._connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT status FROM privacy_reservations WHERE reservation_id = 'res_old'")
        assert cursor.fetchone()[0] == "ABORTED"
        # Fresh reservation must remain PENDING
        cursor.execute("SELECT status FROM privacy_reservations WHERE reservation_id = 'res_fresh'")
        assert cursor.fetchone()[0] == "PENDING"


def test_v013_fresh_database_has_zero_spend(tmp_path):
    """Test that a newly initialized database starts with zero spent budget."""
    state_mgr = PrivacyStateManager(tmp_path / "fresh_privacy_state.db")
    spent_eps, spent_del = state_mgr.get_total_spent()
    assert spent_eps == 0.0
    assert spent_del == 0.0
    assert state_mgr.get_total_reserved_or_spent() == (0.0, 0.0)


def test_v013_schema_migration_backward_compatible(tmp_path):
    """Test opening a v0.1.2 schema without generation/owner_id automatically upgrades."""
    old_db = tmp_path / "v012_legacy.db"
    conn = sqlite3.connect(old_db)
    # Create old schema
    conn.execute("""
        CREATE TABLE privacy_reservations (
            reservation_id TEXT PRIMARY KEY,
            release_key TEXT NOT NULL,
            epsilon REAL NOT NULL,
            delta REAL NOT NULL,
            status TEXT NOT NULL,
            created_at TEXT NOT NULL,
            committed_at TEXT
        )
    """)
    conn.execute("""
        CREATE TABLE release_store (
            release_key TEXT PRIMARY KEY,
            reservation_id TEXT NOT NULL,
            release_id TEXT NOT NULL,
            created_at TEXT NOT NULL,
            status TEXT NOT NULL,
            canonical_semantics_json TEXT NOT NULL,
            metric TEXT NOT NULL,
            window_start TEXT,
            window_end TEXT,
            dataset_version TEXT NOT NULL,
            mechanism TEXT NOT NULL,
            epsilon REAL NOT NULL,
            delta REAL NOT NULL,
            sensitivity REAL NOT NULL,
            noise_scale REAL NOT NULL,
            cell_names_json TEXT NOT NULL,
            n_users INTEGER NOT NULL,
            public_population_size INTEGER,
            dp_vector_json TEXT,
            post_processed_json TEXT,
            committed_at TEXT
        )
    """)
    conn.close()

    # Initialize v0.1.3 PrivacyStateManager on old db
    state_mgr = PrivacyStateManager(old_db)

    with state_mgr._connection() as conn:
        cursor = conn.cursor()
        cursor.execute("PRAGMA table_info(privacy_reservations)")
        res_cols = {row[1] for row in cursor.fetchall()}
        assert "generation" in res_cols
        assert "owner_id" in res_cols

        cursor.execute("PRAGMA table_info(release_store)")
        rel_cols = {row[1] for row in cursor.fetchall()}
        assert "generation" in rel_cols
        assert "owner_id" in rel_cols


def test_v013_recipient_security_external_rng_restriction_retained(tmp_path, v013_test_matrix):
    """Verify external callers are forbidden from supplying RNG."""
    config = load_app_config("configs")
    state_mgr = PrivacyStateManager(tmp_path / "privacy_state.db")
    orch = ReleaseOrchestrator(app_config=config, state_manager=state_mgr)

    q = AnalyticsQuery(metric="user_prevalence", recipient="provider_a")
    bad_rng = np.random.default_rng(12345)

    with pytest.raises(RecipientSecurityViolation, match="strictly forbidden from supplying or controlling RNG"):
        orch.execute_release(q, v013_test_matrix, rng=bad_rng)


def test_v013_cas_commit_requires_matching_release_key(tmp_path):
    """Test commit_release requires release_key matching reservation row."""
    state_mgr = PrivacyStateManager(tmp_path / "privacy_state.db")
    global_budget = PrivacyBudget(epsilon=5.0, delta=1e-5)
    query_budget = PrivacyBudget(epsilon=1.0, delta=1e-6)

    res = state_mgr.reserve_or_get_release(
        release_key="correct_key",
        canonical_semantics={},
        global_budget=global_budget,
        budget_needed=query_budget,
        plan_metadata={},
    )
    res_id = res[2]
    rel_id = res[3]

    dummy = DPReleaseResult(
        metric_name="user_prevalence",
        dp_vector=np.array([1.0]),
        post_processed={},
        epsilon=1.0,
        delta=1e-6,
        mechanism="Gaussian",
        noise_scale=1.0,
        sensitivity=1.0,
        cell_names=["c1"],
        n_users=10,
        public_population_size=50000,
    )

    # Calling commit_release with mismatched key MUST be rejected
    with pytest.raises(ReservationLostError, match="lost, expired, or already finalized"):
        state_mgr.commit_release(
            reservation_id=res_id,
            release_key="wrong_mismatched_key",
            release_id=rel_id,
            release_result=dummy,
            recipient="internal",
            dataset_version="v1.0",
        )


def test_v013_multiworker_same_key_single_flight_thread_concurrency(tmp_path, v013_test_matrix):
    """Test multi-threaded concurrent same-key queries result in 1 charge and bitwise identical outputs."""
    config = load_app_config("configs")
    state_mgr = PrivacyStateManager(tmp_path / "privacy_state.db")
    query = AnalyticsQuery(metric="user_normalized_share", recipient="internal", epsilon=1.0, delta=1e-6)

    def thread_worker(idx: int):
        orch = ReleaseOrchestrator(app_config=config, state_manager=state_mgr)
        return orch.execute_release(query, v013_test_matrix)

    with ThreadPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(thread_worker, range(4)))

    assert len(results) == 4
    first_shares = results[0]["results"]["cell_shares"]
    for r in results[1:]:
        assert r["results"]["cell_shares"] == first_shares

    # Total spend must be exactly 1.0 (NOT 4.0)
    spent_eps, _ = state_mgr.get_total_spent()
    assert spent_eps == 1.0


def test_v013_fenced_abort_cannot_abort_different_reservation_id(tmp_path):
    """Test that abort_reservation with a different reservation_id does not touch active reservations."""
    state_mgr = PrivacyStateManager(tmp_path / "privacy_state.db")
    global_budget = PrivacyBudget(epsilon=5.0, delta=1e-5)
    query_budget = PrivacyBudget(epsilon=1.0, delta=1e-6)

    res = state_mgr.reserve_or_get_release(
        release_key="active_key",
        canonical_semantics={},
        global_budget=global_budget,
        budget_needed=query_budget,
        plan_metadata={},
    )
    active_res_id = res[2]

    # Attempt to abort with unrelated random reservation_id
    state_mgr.abort_reservation("random_other_res_id", "active_key")

    with state_mgr._connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT status FROM privacy_reservations WHERE reservation_id = ?", (active_res_id,))
        assert cursor.fetchone()[0] == "PENDING"


def test_v013_total_reserved_or_spent_tracks_pending_and_committed(tmp_path):
    """Test get_total_reserved_or_spent counts COMMITTED and PENDING but ignores ABORTED."""
    state_mgr = PrivacyStateManager(tmp_path / "privacy_state.db")
    now_iso = datetime.now(timezone.utc).isoformat()

    with state_mgr._connection() as conn:
        conn.execute("""
            INSERT INTO privacy_reservations (reservation_id, release_key, epsilon, delta, status, created_at)
            VALUES ('res1', 'k1', 1.0, 1e-6, 'COMMITTED', ?)
        """, (now_iso,))
        conn.execute("""
            INSERT INTO privacy_reservations (reservation_id, release_key, epsilon, delta, status, created_at)
            VALUES ('res2', 'k2', 0.5, 1e-6, 'PENDING', ?)
        """, (now_iso,))
        conn.execute("""
            INSERT INTO privacy_reservations (reservation_id, release_key, epsilon, delta, status, created_at)
            VALUES ('res3', 'k3', 2.0, 1e-6, 'ABORTED', ?)
        """, (now_iso,))
        conn.commit()

    # COMMITTED (1.0) + PENDING (0.5) = 1.5. ABORTED is ignored.
    res_or_spent_eps, _ = state_mgr.get_total_reserved_or_spent()
    assert abs(res_or_spent_eps - 1.5) < 1e-9

    spent_eps, _ = state_mgr.get_total_spent()
    assert abs(spent_eps - 1.0) < 1e-9

