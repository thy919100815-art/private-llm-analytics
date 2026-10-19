"""Comprehensive regression test suite for Private LLM Analytics v0.1.2.

Covers all second-round independent audit items:
  - C6 / Critical Fix A: Strict RNG restrictions (RNG-1..4)
  - C7 / Critical Fix B: Atomic budget accounting & multi-worker safety
  - C8 / Critical Fix C: Crash consistency & atomic release persistence
  - C9 / Critical Fix 1: Mandatory public population semantics
  - H5 / H6: 15-point pre-validation before any budget spend
  - H7: Canonical semantic fingerprinting (release key uniqueness)
  - H8 / E4: Service synthetic data & population consistency
  - E3: Full streaming SHA-256 dataset versioning
"""

from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from private_llm_analytics.aggregation.user_matrix import (
    UserMatrix,
    build_cell_definitions,
    build_user_matrix,
)
from private_llm_analytics.config.loader import load_app_config
from private_llm_analytics.config.models import AppConfig
from private_llm_analytics.config.validator import ExecutionMode
from private_llm_analytics.metrics.prevalence import UserPrevalenceMetric
from private_llm_analytics.planner.planner import AnalyticsPlanner
from private_llm_analytics.planner.query import AnalyticsQuery
from private_llm_analytics.privacy.budget import PrivacyBudget
from private_llm_analytics.privacy.exceptions import (
    DatasetConfigurationMismatchError,
    MetricDisabledError,
    PrivacyBudgetExceededError,
    PrivacyConfigurationError,
    RecipientSecurityViolation,
    UnsupportedMechanismError,
)
from private_llm_analytics.privacy.ledger import PrivacyLedger
from private_llm_analytics.privacy.state import PrivacyStateManager
from private_llm_analytics.release.cache import PersistentReleaseStore
from private_llm_analytics.release.release import ReleaseOrchestrator
from private_llm_analytics.service.analytics import AnalyticsService


@pytest.fixture
def v012_test_matrix() -> UserMatrix:
    provs = ["provider_a", "provider_b", "provider_c"]
    tsks = ["technical", "document", "analysis"]
    cell_names, cell_map = build_cell_definitions(provs, tsks)
    np_matrix = np.ones((10, 9), dtype=np.float64) * 25.0
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
# 1. Critical Fix A: Caller-Controlled RNG Security (Audit C6 / Tests RNG-1..4)
# =========================================================================

def test_v012_rng1_external_caller_provided_rng_rejected(tmp_path, v012_test_matrix):
    """Test RNG-1: External recipient calling with caller-supplied RNG must be rejected."""
    config = load_app_config("configs")
    state_mgr = PrivacyStateManager(tmp_path / "privacy_state.db")
    orch = ReleaseOrchestrator(app_config=config, state_manager=state_mgr)

    query = AnalyticsQuery(metric="user_prevalence", recipient="provider_a")
    bad_rng = np.random.default_rng(12345)

    with pytest.raises(RecipientSecurityViolation, match="strictly forbidden from supplying or controlling RNG"):
        orch.execute_release(query, v012_test_matrix, rng=bad_rng)

    # In service layer as well
    service = AnalyticsService(config)
    service.orchestrator = orch
    service.set_data(
        pd.DataFrame({
            "interaction_id": ["i1"],
            "user_id": ["u1"],
            "timestamp": ["2026-01-01T00:00:00Z"],
            "provider": ["provider_a"],
            "task": ["technical"],
        }),
        dataset_version="v1.0",
    )
    with pytest.raises(RecipientSecurityViolation, match="cannot provide or control RNG"):
        service.compute(metric="user_prevalence", recipient="provider_a", rng=bad_rng)


def test_v012_rng2_external_caller_cannot_control_noise(tmp_path, v012_test_matrix):
    """Test RNG-2: External caller cannot reproduce noise; internal RNG generates independent noise."""
    config = load_app_config("configs")
    state_mgr = PrivacyStateManager(tmp_path / "privacy_state.db")
    orch = ReleaseOrchestrator(app_config=config, state_manager=state_mgr, execution_mode=ExecutionMode.PRODUCTION)

    query1 = AnalyticsQuery(metric="user_prevalence", recipient="provider_a", window_start="2026-01-01T00:00:00Z", epsilon=1.0, delta=1e-6)
    res1 = orch.execute_release(query1, v012_test_matrix)

    query2 = AnalyticsQuery(metric="user_prevalence", recipient="provider_a", window_start="2026-01-02T00:00:00Z", epsilon=1.0, delta=1e-6)
    res2 = orch.execute_release(query2, v012_test_matrix)

    # Internal secure RNG must produce distinct independent noise vectors
    rates1 = res1["results"]["prevalence_rates"]
    rates2 = res2["results"]["prevalence_rates"]
    assert rates1 != rates2


def test_v012_rng3_experiment_mode_deterministic_reproducibility(tmp_path):
    """Test RNG-3: Experiment mode allows fixed seed for trusted recipient reproducibility."""
    config = load_app_config("configs")
    config.privacy.release.state_db_path = str(tmp_path / "privacy_state.db")
    config.privacy.global_budget.epsilon = 10.0
    config.privacy.global_budget.delta = 1e-4
    config.privacy.release.reuse_existing_release = False
    service = AnalyticsService(config)
    service.generate_synthetic_data(n_users=100, seed=42)

    res1 = service.compute_experiment(
        metric="user_normalized_share",
        recipient="internal",
        epsilon=1.0,
        delta=1e-6,
        noise_seed=999,
    )
    res2 = service.compute_experiment(
        metric="user_normalized_share",
        recipient="internal",
        epsilon=1.0,
        delta=1e-6,
        noise_seed=999,
    )
    assert res1["results"]["cell_shares"] == res2["results"]["cell_shares"]


def test_v012_rng4_experiment_mode_blocks_external_recipients(tmp_path):
    """Test RNG-4: Experiment mode with seeded noise cannot target external recipients."""
    config = load_app_config("configs")
    config.privacy.release.state_db_path = str(tmp_path / "privacy_state.db")
    service = AnalyticsService(config)
    service.generate_synthetic_data(n_users=50, seed=42)

    with pytest.raises(RecipientSecurityViolation, match="cannot target external commercial recipients"):
        service.compute_experiment(
            metric="user_prevalence",
            recipient="provider_a",
            noise_seed=123,
        )


# =========================================================================
# 2. Critical Fix B: Atomic Budget Compare-and-Reserve (Audit C7)
# =========================================================================

def test_v012_atomic_budget_overspend_blocked(tmp_path, v012_test_matrix):
    """Test persistent budget blocks query when cumulative budget is exceeded."""
    config = load_app_config("configs")
    config.privacy.global_budget.epsilon = 2.0
    config.privacy.global_budget.delta = 1e-5
    config.privacy.release.allow_budget_overrun = False
    config.privacy.release.reuse_existing_release = False

    state_mgr = PrivacyStateManager(tmp_path / "privacy_state.db")
    orch = ReleaseOrchestrator(app_config=config, state_manager=state_mgr)

    # First query consumes 1.5 epsilon (remaining: 0.5)
    q1 = AnalyticsQuery(metric="user_prevalence", recipient="internal", epsilon=1.5, delta=1e-6)
    orch.execute_release(q1, v012_test_matrix)

    assert orch.accountant.spent_epsilon == 1.5

    # Second query asks for 1.0 epsilon > remaining 0.5 -> MUST RAISE
    q2 = AnalyticsQuery(metric="user_prevalence", recipient="internal", epsilon=1.0, delta=1e-6)
    with pytest.raises(PrivacyBudgetExceededError):
        orch.execute_release(q2, v012_test_matrix)

    # Spent budget remains exactly 1.5 (no partial or failed consumption)
    assert orch.accountant.spent_epsilon == 1.5


def test_v012_atomic_budget_thread_concurrency(tmp_path):
    """Test concurrent thread reservations strictly respect the global budget cap."""
    test_db = tmp_path / "privacy_state.db"
    state_mgr = PrivacyStateManager(test_db)

    global_budget = PrivacyBudget(epsilon=2.0, delta=1e-5)
    budget_needed = PrivacyBudget(epsilon=1.0, delta=1e-6)

    def attempt_reserve(idx: int):
        try:
            status, _, res_id, rel_id = state_mgr.reserve_or_get_release(
                release_key=f"thread_query_{idx}",
                canonical_semantics={"idx": idx},
                global_budget=global_budget,
                budget_needed=budget_needed,
                plan_metadata={},
                allow_overrun=False,
            )
            return status
        except PrivacyBudgetExceededError:
            return "EXCEEDED"

    with ThreadPoolExecutor(max_workers=5) as executor:
        results = list(executor.map(attempt_reserve, range(5)))

    reserved_count = sum(1 for r in results if r == "RESERVED")
    exceeded_count = sum(1 for r in results if r == "EXCEEDED")

    assert reserved_count == 2
    assert exceeded_count == 3


def test_v012_atomic_budget_aborted_reservation_restores_budget(tmp_path, v012_test_matrix):
    """Test that a failed computation aborts its reservation and leaves budget intact."""
    config = load_app_config("configs")
    config.privacy.global_budget.epsilon = 1.0
    config.privacy.global_budget.delta = 1e-5
    config.privacy.release.allow_budget_overrun = False

    state_mgr = PrivacyStateManager(tmp_path / "privacy_state.db")
    hooks = {"after_budget_reserve": RuntimeError("Simulated worker failure")}
    orch = ReleaseOrchestrator(app_config=config, state_manager=state_mgr, test_hooks=hooks)

    q = AnalyticsQuery(metric="user_normalized_share", recipient="internal", epsilon=1.0, delta=1e-6)

    with pytest.raises(RuntimeError, match="Simulated worker failure"):
        orch.execute_release(q, v012_test_matrix)

    # Spent budget must be 0.0
    spent_eps, _ = state_mgr.get_total_spent()
    assert spent_eps == 0.0

    # Retry without fault injection: should succeed now that budget is available
    orch.test_hooks = {}
    res = orch.execute_release(q, v012_test_matrix)
    assert res is not None
    assert orch.accountant.spent_epsilon == 1.0


# =========================================================================
# 3. Critical Fix C: Crash Consistency & Atomic Persistence (Audit C8)
# =========================================================================

def test_v012_crash_before_commit_no_payload_in_cache(tmp_path, v012_test_matrix):
    """Test that a crash before commit leaves no reusable cached payload."""
    config = load_app_config("configs")
    state_mgr = PrivacyStateManager(tmp_path / "privacy_state.db")
    hooks = {"before_release_commit": RuntimeError("Disk full / crash before commit")}
    orch = ReleaseOrchestrator(app_config=config, state_manager=state_mgr, test_hooks=hooks)

    q = AnalyticsQuery(metric="user_prevalence", recipient="internal", epsilon=1.0, delta=1e-6)

    with pytest.raises(RuntimeError):
        orch.execute_release(q, v012_test_matrix)

    # Release store must have 0 committed releases
    with state_mgr._connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM release_store WHERE status = 'COMMITTED'")
        assert cursor.fetchone()[0] == 0


def test_v012_cache_hit_zero_privacy_budget(tmp_path, v012_test_matrix):
    """Test that retrieving an existing release from cache consumes 0 privacy budget."""
    config = load_app_config("configs")
    config.privacy.global_budget.epsilon = 1.0
    config.privacy.release.allow_budget_overrun = False
    state_mgr = PrivacyStateManager(tmp_path / "privacy_state.db")
    orch = ReleaseOrchestrator(app_config=config, state_manager=state_mgr)

    q = AnalyticsQuery(metric="user_prevalence", recipient="internal", epsilon=1.0, delta=1e-6)

    # First execution consumes entire 1.0 budget
    res1 = orch.execute_release(q, v012_test_matrix)
    assert res1["cache_hit"] is False
    assert orch.accountant.spent_epsilon == 1.0

    # Second execution must hit cache and NOT raise PrivacyBudgetExceededError
    res2 = orch.execute_release(q, v012_test_matrix)
    assert res2["cache_hit"] is True
    assert orch.accountant.spent_epsilon == 1.0
    assert res1["results"]["prevalence_rates"] == res2["results"]["prevalence_rates"]


def test_v012_restart_recovery_sweeps_pending_reservations(tmp_path):
    """Test recover_stale_reservations sweeps uncommitted PENDING reservations."""
    state_mgr = PrivacyStateManager(tmp_path / "privacy_state.db")
    with state_mgr._connection() as conn:
        conn.execute("""
            INSERT INTO privacy_reservations (reservation_id, release_key, epsilon, delta, status, created_at)
            VALUES ('res_stale', 'key_stale', 1.0, 1e-6, 'PENDING', '2026-01-01T00:00:00Z')
        """)
        conn.execute("""
            INSERT INTO release_store (
                release_key, reservation_id, release_id, created_at, status,
                canonical_semantics_json, metric, dataset_version, mechanism, epsilon, delta, sensitivity,
                noise_scale, cell_names_json, n_users
            ) VALUES ('key_stale', 'res_stale', 'rel_stale', '2026-01-01T00:00:00Z', 'PENDING',
                      '{}', 'metric', 'v1', 'Gaussian', 1.0, 1e-6, 1.0, 1.0, '[]', 100)
        """)
        conn.commit()

    swept = state_mgr.recover_stale_reservations()
    assert swept > 0

    with state_mgr._connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT status FROM privacy_reservations WHERE reservation_id = 'res_stale'")
        assert cursor.fetchone()[0] == "ABORTED"


# =========================================================================
# 4. Mandatory Public Population Denominator (Audit C9 / H8 / E4)
# =========================================================================

def test_v012_prevalence_requires_public_population(v012_test_matrix):
    """Test that UserPrevalenceMetric strictly requires public_population_size."""
    metric = UserPrevalenceMetric()
    bad_matrix = UserMatrix(
        matrix=v012_test_matrix.matrix,
        user_ids=v012_test_matrix.user_ids,
        providers=v012_test_matrix.providers,
        tasks=v012_test_matrix.tasks,
        cell_names=v012_test_matrix.cell_names,
        cell_map=v012_test_matrix.cell_map,
        public_population_size=None,
    )

    with pytest.raises(PrivacyConfigurationError, match="strictly requires public_population_size"):
        metric.post_process(np.zeros(9), bad_matrix)

    with pytest.raises(PrivacyConfigurationError):
        UserMatrix(
            matrix=v012_test_matrix.matrix,
            user_ids=v012_test_matrix.user_ids,
            providers=v012_test_matrix.providers,
            tasks=v012_test_matrix.tasks,
            cell_names=v012_test_matrix.cell_names,
            cell_map=v012_test_matrix.cell_map,
            public_population_size=0,
        )


def test_v012_prevalence_rate_uses_public_denominator(v012_test_matrix):
    """Test that prevalence rate divides by public population size, not sample user count."""
    metric = UserPrevalenceMetric()
    # 10 active users in sample, but public roster is 50,000
    noisy_counts = np.array([100.0] * 9)
    pp = metric.post_process(noisy_counts, v012_test_matrix)

    # Each prevalence rate should be 100.0 / 50000 = 0.002, NOT 100.0 / 10 = 10.0
    for cell, rate in pp["prevalence_rates"].items():
        assert abs(rate - (100.0 / 50000.0)) < 1e-7


def test_v012_active_users_exceeding_public_population_raises():
    """Test that active users exceeding public population size raises DatasetConfigurationMismatchError."""
    df = pd.DataFrame({
        "interaction_id": [f"i_{i}" for i in range(10)],
        "user_id": [f"user_{i}" for i in range(10)],
        "timestamp": ["2026-01-01T00:00:00Z"] * 10,
        "provider": ["provider_a"] * 10,
        "task": ["technical"] * 10,
    })

    with pytest.raises(DatasetConfigurationMismatchError, match="cannot be less than active user count"):
        build_user_matrix(
            df,
            providers=["provider_a", "provider_b", "provider_c"],
            tasks=["technical", "document", "analysis"],
            public_population_size=5,  # Only 5 public users, but 10 active users in data!
        )


def test_v012_synthetic_override_synchronizes_public_population(tmp_path):
    """Test Audit H8: generate_synthetic_data(n_users=500) updates public population to 500."""
    config = load_app_config("configs")
    config.privacy.release.state_db_path = str(tmp_path / "privacy_state.db")
    service = AnalyticsService(config)

    # Request 500 users
    service.generate_synthetic_data(n_users=500, seed=123)

    assert service.config.scenario.population.n_users == 500
    u_mat = service.get_user_matrix()
    assert u_mat.public_population_size == 500


# =========================================================================
# 5. Full 15-Point Pre-Validation (Audit H5 & H6)
# =========================================================================

def test_v012_prevalidation_unsupported_mechanism_zero_budget(tmp_path, v012_test_matrix):
    """Test Audit H5: Unsupported mechanism fails before reserving any budget."""
    config = load_app_config("configs")
    state_mgr = PrivacyStateManager(tmp_path / "privacy_state.db")
    orch = ReleaseOrchestrator(app_config=config, state_manager=state_mgr)

    q = AnalyticsQuery(metric="user_prevalence", recipient="internal", mechanism="quantum_mechanism")

    with pytest.raises(UnsupportedMechanismError):
        orch.execute_release(q, v012_test_matrix)

    spent_eps, _ = state_mgr.get_total_spent()
    assert spent_eps == 0.0


def test_v012_prevalidation_invalid_delta_zero_budget(tmp_path, v012_test_matrix):
    """Test Audit H5: Gaussian mechanism with invalid delta (delta=0) fails with 0 budget spent."""
    config = load_app_config("configs")
    state_mgr = PrivacyStateManager(tmp_path / "privacy_state.db")
    orch = ReleaseOrchestrator(app_config=config, state_manager=state_mgr)

    q = AnalyticsQuery(metric="user_prevalence", recipient="internal", mechanism="gaussian", delta=0.0)

    with pytest.raises(PrivacyConfigurationError, match="0 < delta < 1"):
        orch.execute_release(q, v012_test_matrix)

    spent_eps, _ = state_mgr.get_total_spent()
    assert spent_eps == 0.0


def test_v012_prevalidation_negative_epsilon_zero_budget(tmp_path, v012_test_matrix):
    """Test negative epsilon fails pre-validation with 0 budget spent."""
    config = load_app_config("configs")
    state_mgr = PrivacyStateManager(tmp_path / "privacy_state.db")
    orch = ReleaseOrchestrator(app_config=config, state_manager=state_mgr)

    q = AnalyticsQuery(metric="user_prevalence", recipient="internal", epsilon=-1.0)

    with pytest.raises(PrivacyConfigurationError, match="strictly positive"):
        orch.execute_release(q, v012_test_matrix)

    spent_eps, _ = state_mgr.get_total_spent()
    assert spent_eps == 0.0


def test_v012_prevalidation_disabled_metric_zero_budget(tmp_path, v012_test_matrix):
    """Test Audit H6: Querying disabled metric fails before budget reservation."""
    config = load_app_config("configs")
    config.metrics.metrics["user_prevalence"].enabled = False

    state_mgr = PrivacyStateManager(tmp_path / "privacy_state.db")
    orch = ReleaseOrchestrator(app_config=config, state_manager=state_mgr)

    q = AnalyticsQuery(metric="user_prevalence", recipient="internal")

    with pytest.raises(MetricDisabledError, match="is disabled in configuration"):
        orch.execute_release(q, v012_test_matrix)

    spent_eps, _ = state_mgr.get_total_spent()
    assert spent_eps == 0.0


def test_v012_prevalidation_unknown_metric_zero_budget(tmp_path, v012_test_matrix):
    """Test querying non-existent metric fails before budget reservation."""
    config = load_app_config("configs")
    state_mgr = PrivacyStateManager(tmp_path / "privacy_state.db")
    orch = ReleaseOrchestrator(app_config=config, state_manager=state_mgr)

    q = AnalyticsQuery(metric="non_existent_metric", recipient="internal")

    with pytest.raises(ValueError, match="Unknown metric"):
        orch.execute_release(q, v012_test_matrix)

    spent_eps, _ = state_mgr.get_total_spent()
    assert spent_eps == 0.0


def test_v012_prevalidation_unauthorized_recipient_zero_budget(tmp_path, v012_test_matrix):
    """Test unauthorized recipient requesting exact results fails before budget reservation."""
    config = load_app_config("configs")
    state_mgr = PrivacyStateManager(tmp_path / "privacy_state.db")
    orch = ReleaseOrchestrator(app_config=config, state_manager=state_mgr)

    q = AnalyticsQuery(metric="user_prevalence", recipient="provider_a")

    with pytest.raises(RecipientSecurityViolation, match="forbidden from receiving exact results"):
        orch.execute_release(q, v012_test_matrix, include_exact=True)

    spent_eps, _ = state_mgr.get_total_spent()
    assert spent_eps == 0.0


# =========================================================================
# 6. Canonical Semantic Fingerprinting (Audit H7)
# =========================================================================

def test_v012_release_key_public_population_differs():
    """Test Audit H7: Different public population sizes generate distinct release keys."""
    config = load_app_config("configs")
    planner = AnalyticsPlanner(config)

    q = AnalyticsQuery(metric="user_prevalence", recipient="internal")
    p1 = planner.create_plan(q, dataset_version="v1.0", public_population_size=10000)
    p2 = planner.create_plan(q, dataset_version="v1.0", public_population_size=50000)

    assert p1.release_key != p2.release_key


def test_v012_release_key_clip_norm_differs():
    """Test Audit H7: Different clip_norm generates distinct release keys."""
    config1 = load_app_config("configs")
    config1.metrics.metrics["interaction_volume"].contribution.clip_norm = 10.0
    planner1 = AnalyticsPlanner(config1)

    config2 = load_app_config("configs")
    config2.metrics.metrics["interaction_volume"].contribution.clip_norm = 20.0
    planner2 = AnalyticsPlanner(config2)

    q = AnalyticsQuery(metric="interaction_volume", recipient="internal")
    p1 = planner1.create_plan(q, dataset_version="v1.0")
    p2 = planner2.create_plan(q, dataset_version="v1.0")

    assert p1.release_key != p2.release_key


def test_v012_release_key_dataset_version_differs():
    """Test different dataset versions generate distinct release keys."""
    config = load_app_config("configs")
    planner = AnalyticsPlanner(config)

    q = AnalyticsQuery(metric="user_prevalence", recipient="internal")
    p1 = planner.create_plan(q, dataset_version="telemetry_batch_A")
    p2 = planner.create_plan(q, dataset_version="telemetry_batch_B")

    assert p1.release_key != p2.release_key


def test_v012_release_key_metric_version_differs():
    """Test incrementing metric version changes release key."""
    config1 = load_app_config("configs")
    config1.metrics.metrics["user_prevalence"].version = 1
    planner1 = AnalyticsPlanner(config1)

    config2 = load_app_config("configs")
    config2.metrics.metrics["user_prevalence"].version = 2
    planner2 = AnalyticsPlanner(config2)

    q = AnalyticsQuery(metric="user_prevalence", recipient="internal")
    p1 = planner1.create_plan(q, dataset_version="v1.0")
    p2 = planner2.create_plan(q, dataset_version="v1.0")

    assert p1.release_key != p2.release_key


def test_v012_release_key_canonical_ordering_invariant():
    """Test release key is SHA-256 hash invariant to dictionary insertion order."""
    config = load_app_config("configs")
    planner = AnalyticsPlanner(config)

    q = AnalyticsQuery(metric="user_prevalence", recipient="internal")
    p = planner.create_plan(q, dataset_version="v1.0")

    # Canonical semantics JSON sorted vs unsorted keys must produce identical hash
    raw_dict = dict(p.canonical_semantics)
    reversed_dict = dict(reversed(list(raw_dict.items())))

    hash1 = hashlib.sha256(json.dumps(raw_dict, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
    hash2 = hashlib.sha256(json.dumps(reversed_dict, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()

    assert hash1 == hash2
    assert p.release_key == hash1


# =========================================================================
# 7. Dataset Fingerprinting & Windows (Audit E3)
# =========================================================================

def test_v012_load_data_full_streaming_sha256(tmp_path):
    """Test Audit E3: Modifications in later chunks alter the dataset hash version."""
    config = load_app_config("configs")
    config.privacy.release.state_db_path = str(tmp_path / "privacy_state.db")
    service = AnalyticsService(config)

    # Create CSV > 100 KiB
    records = []
    for i in range(2000):
        records.append({
            "interaction_id": f"int_{i:06d}",
            "user_id": f"user_{i % 50:04d}",
            "timestamp": "2026-01-01T12:00:00Z",
            "provider": "provider_a",
            "task": "technical",
        })
    df1 = pd.DataFrame(records)
    csv1 = tmp_path / "telemetry1.csv"
    df1.to_csv(csv1, index=False)

    service.load_data(csv1)
    v1 = service._dataset_version

    # Modify only the last record at the very end of the file
    df2 = df1.copy()
    df2.loc[1999, "task"] = "analysis"
    csv2 = tmp_path / "telemetry2.csv"
    df2.to_csv(csv2, index=False)

    service.load_data(csv2)
    v2 = service._dataset_version

    assert v1 != v2, f"Full streaming hash failed to detect modification at end of file! {v1} == {v2}"


def test_v012_time_window_filtering_in_canonical_key():
    """Test that specifying different time windows creates distinct canonical keys."""
    config = load_app_config("configs")
    planner = AnalyticsPlanner(config)

    q1 = AnalyticsQuery(metric="user_prevalence", recipient="internal", window_start="2026-01-01", window_end="2026-01-07")
    q2 = AnalyticsQuery(metric="user_prevalence", recipient="internal", window_start="2026-01-08", window_end="2026-01-14")

    p1 = planner.create_plan(q1, dataset_version="v1.0")
    p2 = planner.create_plan(q2, dataset_version="v1.0")

    assert p1.release_key != p2.release_key


def test_v012_exact_results_internal_policy_allowed(tmp_path, v012_test_matrix):
    """Test that trusted/internal recipient can receive exact results when requested."""
    config = load_app_config("configs")
    state_mgr = PrivacyStateManager(tmp_path / "privacy_state.db")
    orch = ReleaseOrchestrator(app_config=config, state_manager=state_mgr)

    q = AnalyticsQuery(metric="user_normalized_share", recipient="internal")
    res = orch.execute_release(q, v012_test_matrix, include_exact=True)

    assert "exact_ground_truth" in res
    assert "clean_vector" in res["exact_ground_truth"]
    assert len(res["exact_ground_truth"]["clean_vector"]) == 9


def test_v012_same_key_single_flight_threads(tmp_path, v012_test_matrix):
    """Test that concurrent threads with identical release key hit single flight and consume budget once."""
    config = load_app_config("configs")
    state_mgr = PrivacyStateManager(tmp_path / "privacy_state.db")
    orch = ReleaseOrchestrator(app_config=config, state_manager=state_mgr)
    q = AnalyticsQuery(metric="user_prevalence", recipient="internal", epsilon=1.0, delta=1e-6)

    def run_query(idx: int):
        return orch.execute_release(q, v012_test_matrix)

    with ThreadPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(run_query, range(4)))

    first_rates = results[0]["results"]["prevalence_rates"]
    for r in results[1:]:
        assert r["results"]["prevalence_rates"] == first_rates

    spent_eps, _ = state_mgr.get_total_spent()
    assert abs(spent_eps - 1.0) < 1e-9


def test_v012_multiprocess_budget_atomic_execution(tmp_path):
    """Test multiprocessing atomic budget reservation with ProcessPoolExecutor."""
    from concurrent.futures import ProcessPoolExecutor
    from experiments.verify_multiworker_budget import worker_spend_task

    db_path = str(tmp_path / "verify_mp.db")
    state_mgr = PrivacyStateManager(db_path)

    global_eps = 2.0
    global_delta = 1e-5
    request_eps = 1.0
    request_delta = 1e-6

    with ProcessPoolExecutor(max_workers=4) as executor:
        futures = [
            executor.submit(worker_spend_task, db_path, i, request_eps, request_delta, global_eps, global_delta)
            for i in range(4)
        ]
        results = [f.result() for f in futures]

    committed = sum(1 for r in results if r["status"] == "COMMITTED")
    exceeded = sum(1 for r in results if r["status"] == "EXCEEDED")

    assert committed == 2
    assert exceeded == 2
    spent_eps, _ = state_mgr.get_total_spent()
    assert abs(spent_eps - 2.0) < 1e-9
