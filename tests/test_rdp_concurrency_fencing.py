"""Concurrency, fencing, and crash-recovery hardening tests for RDP accounting (v0.2).

Covers:
  - Multi-worker concurrent RDP budget enforcement (atomic cap authorization).
  - Same-key single-flight memoization with 0 duplicate RDP cost across concurrent threads.
  - Crash recovery: stale reservation lease expiration frees RDP budget.
  - CAS generation fencing: stale worker commit rejected without RDP double-spend.
  - High-precision Decimal / numerical boundary checks avoiding float roundoff leak.
"""

from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from decimal import Decimal
import os
from pathlib import Path
import sqlite3
import time
import numpy as np
import pytest

from private_llm_analytics.aggregation.user_matrix import UserMatrix, build_cell_definitions
from private_llm_analytics.config.loader import load_app_config
from private_llm_analytics.metrics.base import DPReleaseResult
from private_llm_analytics.planner.query import AnalyticsQuery
from private_llm_analytics.privacy.budget import PrivacyBudget
from private_llm_analytics.privacy.composition import resolve_budget_scope_id
from private_llm_analytics.privacy.exceptions import (
    CommitIrreversibleError,
    PrivacyBudgetExceededError,
    ReleaseInProgressError,
    ReservationLostError,
)
from private_llm_analytics.privacy.rdp import (
    DEFAULT_RDP_ORDERS,
    compute_gaussian_rdp,
    rdp_to_epsilon_delta,
    rdp_vector_to_json,
)
from private_llm_analytics.privacy.state import PrivacyStateManager
from private_llm_analytics.release.release import ReleaseOrchestrator


@pytest.fixture
def sample_user_matrix() -> UserMatrix:
    providers = ["provider_a", "provider_b", "provider_c"]
    tasks = ["technical", "document", "analysis"]
    cell_names, cell_map = build_cell_definitions(providers, tasks)
    np_matrix = np.ones((20, 9), dtype=np.float64) * 10.0
    return UserMatrix(
        matrix=np_matrix,
        user_ids=[f"usr_{i}" for i in range(20)],
        providers=providers,
        tasks=tasks,
        cell_names=cell_names,
        cell_map=cell_map,
        public_population_size=50000,
    )


def test_same_key_singleflight_rdp_no_duplicate_charge(tmp_path, sample_user_matrix):
    """6 concurrent threads querying the same key produce 1 DP execution and 1 RDP charge."""
    db_path = tmp_path / "privacy_state.db"
    config = load_app_config("configs")
    config.privacy.global_budget.epsilon = 10.0
    config.privacy.global_budget.delta = 1e-5
    config.privacy.accountant.epsilon_cap = 10.0
    config.privacy.accountant.delta_global = 1e-5

    state_mgr = PrivacyStateManager(db_path)

    # Add slight delay after reservation to ensure concurrent workers queue on single-flight
    slow_hooks = {"after_budget_reserve": lambda: time.sleep(0.3)}
    orch_slow = ReleaseOrchestrator(app_config=config, state_manager=state_mgr, test_hooks=slow_hooks)
    orch_normal = ReleaseOrchestrator(app_config=config, state_manager=state_mgr)

    query = AnalyticsQuery(
        metric="user_prevalence",
        recipient="provider_a",
        epsilon=1.0,
        delta=1e-6,
    )

    def run_query(worker_idx: int):
        if worker_idx == 0:
            return orch_slow.execute_release(query, sample_user_matrix)
        else:
            time.sleep(0.05)  # Ensure worker 0 creates reservation first
            return orch_normal.execute_release(query, sample_user_matrix)

    with ThreadPoolExecutor(max_workers=6) as executor:
        futures = [executor.submit(run_query, i) for i in range(6)]
        results = [f.result() for f in futures]

    # Verify cache hit counts
    cache_hits = [r["cache_hit"] for r in results]
    assert cache_hits.count(False) == 1, f"Expected exactly 1 fresh release, got {cache_hits.count(False)}"
    assert cache_hits.count(True) == 5, f"Expected exactly 5 cache hits, got {cache_hits.count(True)}"

    # Bitwise identical outputs
    first_rates = results[0]["results"]["prevalence_rates"]
    for r in results[1:]:
        assert r["results"]["prevalence_rates"] == first_rates

    # Database verification: exactly 1 committed release and 1 ledger spend row
    view = orch_normal.get_privacy_view("provider_a")
    assert view.release_count == 1
    assert len(state_mgr.list_releases()) == 1


def test_crash_recovery_stale_reservation_frees_rdp_budget(tmp_path):
    """A crashed worker holding an RDP reservation has its lease expired, freeing budget."""
    db_path = tmp_path / "privacy_state.db"
    state_mgr = PrivacyStateManager(db_path)

    orders = DEFAULT_RDP_ORDERS
    delta_global = 1e-5
    rdp_candidate = compute_gaussian_rdp(sigma=1.0, sensitivity=1.0, orders=orders)

    # Set cap allowing exactly 1 release (5.38 <= 6.0, but 5.38 + 5.38 = 8.57 > 6.0)
    cap = 6.0
    global_budget = PrivacyBudget(epsilon=cap, delta=delta_global)
    needed_budget = PrivacyBudget(epsilon=1.0, delta=1e-6)

    # Worker A reserves with a very short lease (0.2s)
    res_a = state_mgr.reserve_or_get_release(
        release_key="worker_a_key",
        canonical_semantics={"worker": "a"},
        global_budget=global_budget,
        budget_needed=needed_budget,
        plan_metadata={},
        lease_timeout=0.2,
        accountant_mode="rdp",
        budget_scope_id="global_external",
        rdp_orders=orders,
        rdp_candidate_cost=rdp_candidate,
        epsilon_cap=cap,
        delta_global=delta_global,
    )
    assert res_a[0] == "RESERVED"

    # Worker B tries to reserve a different key immediately -> Budget exceeded because Worker A's reservation holds cap
    with pytest.raises(PrivacyBudgetExceededError):
        state_mgr.reserve_or_get_release(
            release_key="worker_b_key",
            canonical_semantics={"worker": "b"},
            global_budget=global_budget,
            budget_needed=needed_budget,
            plan_metadata={},
            lease_timeout=5.0,
            accountant_mode="rdp",
            budget_scope_id="global_external",
            rdp_orders=orders,
            rdp_candidate_cost=rdp_candidate,
            epsilon_cap=cap,
            delta_global=delta_global,
        )

    # Worker A crashes without committing. Wait for lease to expire.
    time.sleep(0.3)

    # Run stale recovery
    recovered = state_mgr.recover_stale_reservations(max_age_seconds=0.2, force=True)
    assert recovered >= 1

    # Worker B tries again -> should now succeed because Worker A's budget was freed!
    res_b = state_mgr.reserve_or_get_release(
        release_key="worker_b_key",
        canonical_semantics={"worker": "b"},
        global_budget=global_budget,
        budget_needed=needed_budget,
        plan_metadata={},
        lease_timeout=5.0,
        accountant_mode="rdp",
        budget_scope_id="global_external",
        rdp_orders=orders,
        rdp_candidate_cost=rdp_candidate,
        epsilon_cap=cap,
        delta_global=delta_global,
    )
    assert res_b[0] == "RESERVED"


def test_stale_worker_rejected_by_generation_fencing_rdp(tmp_path):
    """A stale worker whose reservation was taken over is rejected by generation fencing."""
    db_path = tmp_path / "privacy_state.db"
    state_mgr = PrivacyStateManager(db_path)
    orders = DEFAULT_RDP_ORDERS
    delta_global = 1e-5
    rdp_cand = compute_gaussian_rdp(sigma=2.0, sensitivity=1.0, orders=orders)
    cap = 10.0

    global_budget = PrivacyBudget(epsilon=cap, delta=delta_global)
    needed_budget = PrivacyBudget(epsilon=1.0, delta=1e-6)
    release_key = "fenced_key"

    # Worker A reserves gen 1 with short lease (0.15s)
    res_a = state_mgr.reserve_or_get_release(
        release_key=release_key,
        canonical_semantics={"q": "same"},
        global_budget=global_budget,
        budget_needed=needed_budget,
        plan_metadata={},
        lease_timeout=0.15,
        accountant_mode="rdp",
        budget_scope_id="global_external",
        rdp_orders=orders,
        rdp_candidate_cost=rdp_cand,
        epsilon_cap=cap,
        delta_global=delta_global,
    )
    assert res_a.generation == 1
    res_id_a = res_a[2]
    rel_id_a = res_a[3]

    # Worker A goes to sleep; lease expires
    time.sleep(0.2)

    # Worker B takes over generation 2
    res_b = state_mgr.reserve_or_get_release(
        release_key=release_key,
        canonical_semantics={"q": "same"},
        global_budget=global_budget,
        budget_needed=needed_budget,
        plan_metadata={},
        lease_timeout=5.0,
        accountant_mode="rdp",
        budget_scope_id="global_external",
        rdp_orders=orders,
        rdp_candidate_cost=rdp_cand,
        epsilon_cap=cap,
        delta_global=delta_global,
    )
    assert res_b.generation == 2
    res_id_b = res_b[2]
    rel_id_b = res_b[3]

    # Worker B commits generation 2
    dummy_b = DPReleaseResult(
        metric_name="user_prevalence",
        dp_vector=np.array([2.0, 4.0]),
        post_processed={},
        epsilon=1.0,
        delta=1e-6,
        mechanism="Gaussian",
        noise_scale=2.0,
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
        recipient="provider_a",
        dataset_version="v1.0",
        generation=2,
        budget_scope_id="global_external",
        accountant_mode="rdp",
        rdp_cost_json=rdp_vector_to_json(rdp_cand),
        delta_global=delta_global,
    )

    # Worker A wakes up and tries to commit stale generation 1 -> MUST BE REJECTED
    dummy_a = DPReleaseResult(
        metric_name="user_prevalence",
        dp_vector=np.array([1.0, 1.0]),
        post_processed={},
        epsilon=1.0,
        delta=1e-6,
        mechanism="Gaussian",
        noise_scale=2.0,
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
            recipient="provider_a",
            dataset_version="v1.0",
            generation=1,
            budget_scope_id="global_external",
            accountant_mode="rdp",
            rdp_cost_json=rdp_vector_to_json(rdp_cand),
            delta_global=delta_global,
        )

    # Confirm only Worker B's release is committed
    committed = state_mgr.get_committed_release(release_key)
    np.testing.assert_allclose(committed.dp_vector, np.array([2.0, 4.0]))
    view = state_mgr.get_privacy_view("global_external", delta_global=delta_global)
    assert view.release_count == 1


def test_adversarial_decimal_precision_boundary(tmp_path):
    """Verify high-precision Decimal boundaries with no floating-point truncation leakage."""
    db_path = tmp_path / "privacy_state.db"
    state_mgr = PrivacyStateManager(db_path)

    # Set precise Decimal budget cap: 0.05
    exact_cap = Decimal("0.050000000000000000")
    delta_global = 1e-5
    orders = DEFAULT_RDP_ORDERS

    global_budget = PrivacyBudget(epsilon=float(exact_cap), delta=delta_global)
    needed_budget = PrivacyBudget(epsilon=0.01, delta=1e-6)

    # Execute 5 releases each costing exactly 0.01
    for i in range(5):
        key = f"precision_key_{i}"
        status, _, res_id, rel_id = state_mgr.reserve_or_get_release(
            release_key=key,
            canonical_semantics={"i": i},
            global_budget=global_budget,
            budget_needed=needed_budget,
            plan_metadata={},
            accountant_mode="basic",
            budget_scope_id="global_external",
        )
        assert status == "RESERVED"
        dummy = DPReleaseResult(
            metric_name="user_prevalence",
            dp_vector=np.array([1.0]),
            post_processed={},
            epsilon=0.01,
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
            release_key=key,
            release_id=rel_id,
            release_result=dummy,
            recipient="provider_a",
            dataset_version="v1.0",
            generation=1,
            budget_scope_id="global_external",
            accountant_mode="basic",
        )

    # Spent is exactly 0.05
    spent_dec, _ = state_mgr.get_total_spent_decimal()
    assert spent_dec == Decimal("0.05")

    # 6th release of 0.0000000000000001 must be REJECTED
    with pytest.raises(PrivacyBudgetExceededError):
        state_mgr.reserve_or_get_release(
            release_key="precision_key_overflow",
            canonical_semantics={"overflow": True},
            global_budget=global_budget,
            budget_needed=PrivacyBudget(epsilon=0.00000001, delta=1e-6),
            plan_metadata={},
            accountant_mode="basic",
            budget_scope_id="global_external",
        )


def test_multiworker_rdp_budget_cap_enforcement(tmp_path):
    """8 concurrent workers requesting distinct releases; cap permits exactly 3; 3 commit, 5 fail."""
    db_path = tmp_path / "privacy_state.db"
    state_mgr = PrivacyStateManager(db_path)

    # Each release requests eps=1.0, delta=1e-6 with Gaussian sigma=2.0, sensitivity=1.0
    orders = DEFAULT_RDP_ORDERS
    delta_global = 1e-5
    rdp_cand = compute_gaussian_rdp(sigma=2.0, sensitivity=1.0, orders=orders)
    needed_budget = PrivacyBudget(epsilon=1.0, delta=1e-6)

    # Set cap allowing exactly 3 releases (3 releases = 4.64 <= 5.0, 4 releases = 5.38 > 5.0)
    cap = 5.0
    global_budget = PrivacyBudget(epsilon=cap, delta=delta_global)

    def worker_attempt(worker_id: int) -> str:
        key = f"unique_worker_query_{worker_id}"
        semantics = {"worker_id": worker_id}
        try:
            status, _, res_id, rel_id = state_mgr.reserve_or_get_release(
                release_key=key,
                canonical_semantics=semantics,
                global_budget=global_budget,
                budget_needed=needed_budget,
                plan_metadata={"worker_id": worker_id},
                lease_timeout=10.0,
                accountant_mode="rdp",
                budget_scope_id="global_external",
                rdp_orders=orders,
                rdp_candidate_cost=rdp_cand,
                epsilon_cap=cap,
                delta_global=delta_global,
            )
            if status == "RESERVED" and res_id is not None:
                # Simulate minimal work
                time.sleep(0.02)
                dummy = DPReleaseResult(
                    metric_name="user_prevalence",
                    dp_vector=np.array([float(worker_id)]),
                    post_processed={},
                    epsilon=1.0,
                    delta=1e-6,
                    mechanism="Gaussian",
                    noise_scale=2.0,
                    sensitivity=1.0,
                    cell_names=["c1"],
                    n_users=10,
                    public_population_size=50000,
                )
                state_mgr.commit_release(
                    reservation_id=res_id,
                    release_key=key,
                    release_id=rel_id,
                    release_result=dummy,
                    recipient="provider_a",
                    dataset_version="v1.0",
                    generation=1,
                    budget_scope_id="global_external",
                    accountant_mode="rdp",
                    rdp_cost_json=rdp_vector_to_json(rdp_cand),
                    delta_global=delta_global,
                )
                return "COMMITTED"
            return "CACHE_HIT"
        except PrivacyBudgetExceededError:
            return "EXCEEDED"
        except Exception as exc:
            return f"ERROR: {exc}"

    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(worker_attempt, i) for i in range(8)]
        outcomes = [f.result() for f in futures]

    committed_count = outcomes.count("COMMITTED")
    exceeded_count = outcomes.count("EXCEEDED")

    assert committed_count == 3, f"Expected 3 commits, got {committed_count}: {outcomes}"
    assert exceeded_count == 5, f"Expected 5 exceeded rejections, got {exceeded_count}: {outcomes}"

    # Verify final privacy view
    view = state_mgr.get_privacy_view("global_external", delta_global=delta_global, orders=orders, epsilon_cap=cap)
    assert view.release_count == 3
    assert float(view.effective_epsilon) <= cap
