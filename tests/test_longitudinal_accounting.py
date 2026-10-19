"""Tests for longitudinal multi-recipient privacy accounting in v0.2.

Covers:
  - Historical composition over multiple time windows
  - Same-query replay yields cache_hit=True and 0 incremental privacy cost
  - Recipient isolation modes: PER_RECIPIENT, COLLUSION_GROUP, GLOBAL_EXTERNAL
  - Switching provider cannot bypass GLOBAL_EXTERNAL cap
  - PrivacyView retrieval and reconstruction
  - Database schema migration for v0.1 legacy records
"""

from decimal import Decimal
import numpy as np
import pytest
from pathlib import Path
import sqlite3
import uuid

from private_llm_analytics.aggregation.user_matrix import UserMatrix, build_cell_definitions
from private_llm_analytics.config.loader import load_app_config
from private_llm_analytics.config.models import AppConfig
from private_llm_analytics.config.validator import ProductionSafetyConfigurationError, validate_production_config
from private_llm_analytics.planner.query import AnalyticsQuery
from private_llm_analytics.privacy.composition import resolve_budget_scope_id
from private_llm_analytics.privacy.exceptions import PrivacyBudgetExceededError, RecipientSecurityViolation
from private_llm_analytics.privacy.rdp import DEFAULT_RDP_ORDERS
from private_llm_analytics.privacy.state import PrivacyStateManager
from private_llm_analytics.release.release import ReleaseOrchestrator


@pytest.fixture
def test_matrix():
    providers = ["provider_a", "provider_b", "provider_c"]
    tasks = ["code", "document", "analysis"]
    cell_names, cell_map = build_cell_definitions(providers, tasks)
    users = [f"user_{i}" for i in range(100)]
    matrix = np.ones((100, 9), dtype=np.float64)
    return UserMatrix(
        matrix=matrix,
        providers=providers,
        tasks=tasks,
        cell_names=cell_names,
        cell_map=cell_map,
        user_ids=users,
        public_population_size=50000,
    )


def test_same_release_replay_zero_incremental_cost(tmp_path, test_matrix):
    """Same semantic query must return cache_hit=True and 0 incremental privacy cost."""
    config = load_app_config("configs")
    config.privacy.global_budget.epsilon = 4.0
    config.privacy.global_budget.delta = 1e-5
    state_mgr = PrivacyStateManager(tmp_path / "privacy_state.db")
    orch = ReleaseOrchestrator(app_config=config, state_manager=state_mgr)

    q = AnalyticsQuery(metric="user_prevalence", recipient="provider_a", epsilon=1.0, delta=1e-6)

    # First execution: new release
    res1 = orch.execute_release(q, test_matrix)
    assert res1["cache_hit"] is False
    assert res1["incremental_privacy_cost"] == 1.0

    # Total ledger rows = 1
    with state_mgr._connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM privacy_ledger WHERE cache_hit = 0")
        assert cursor.fetchone()[0] == 1

    # Second execution: memoized release
    res2 = orch.execute_release(q, test_matrix)
    assert res2["cache_hit"] is True
    assert res2["incremental_privacy_cost"] == 0.0
    assert res2["incremental_privacy_cost_decimal"] == "0.0"
    assert res1["results"]["prevalence_rates"] == res2["results"]["prevalence_rates"]

    # No additional non-cache-hit ledger rows created
    with state_mgr._connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM privacy_ledger WHERE cache_hit = 0")
        assert cursor.fetchone()[0] == 1
        cursor.execute("SELECT COUNT(*) FROM privacy_ledger WHERE cache_hit = 1")
        assert cursor.fetchone()[0] == 1


def test_recipient_isolation_per_recipient(tmp_path, test_matrix):
    """Under PER_RECIPIENT mode, provider_a and provider_b have independent privacy views."""
    config = load_app_config("configs")
    config.privacy.recipient_composition.mode = "per_recipient"
    config.privacy.global_budget.epsilon = 1.5
    config.privacy.global_budget.delta = 1e-5
    state_mgr = PrivacyStateManager(tmp_path / "privacy_state.db")
    orch = ReleaseOrchestrator(app_config=config, state_manager=state_mgr)

    # Provider A consumes 1.0 epsilon
    qa = AnalyticsQuery(metric="user_prevalence", recipient="provider_a", epsilon=1.0, delta=1e-6)
    res_a = orch.execute_release(qa, test_matrix)
    assert res_a["cache_hit"] is False
    assert res_a["privacy_accounting"]["budget_scope_id"] == "recipient:provider_a"

    # Provider B consumes 1.0 epsilon in its own scope (not blocked by provider_a's spend)
    qb = AnalyticsQuery(metric="user_normalized_share", recipient="provider_b", epsilon=1.0, delta=1e-6)
    res_b = orch.execute_release(qb, test_matrix)
    assert res_b["cache_hit"] is False
    assert res_b["privacy_accounting"]["budget_scope_id"] == "recipient:provider_b"

    # Query PrivacyView for each provider
    view_a = orch.get_privacy_view("provider_a")
    view_b = orch.get_privacy_view("provider_b")
    assert view_a.budget_scope_id == "recipient:provider_a"
    assert view_a.release_count == 1
    assert view_b.budget_scope_id == "recipient:provider_b"
    assert view_b.release_count == 1


def test_recipient_collusion_group(tmp_path, test_matrix):
    """Under COLLUSION_GROUP mode, colluding recipients share a common budget scope."""
    config = load_app_config("configs")
    config.privacy.recipient_composition.mode = "collusion_group"
    config.privacy.recipient_composition.groups = {
        "group_ab": ["provider_a", "provider_b"],
        "group_c": ["provider_c"],
    }
    # Cap allows about 1.5 epsilon
    config.privacy.global_budget.epsilon = 1.5
    config.privacy.global_budget.delta = 1e-5
    state_mgr = PrivacyStateManager(tmp_path / "privacy_state.db")
    orch = ReleaseOrchestrator(app_config=config, state_manager=state_mgr)

    # Provider A makes query 1 (1.0 epsilon)
    qa1 = AnalyticsQuery(metric="user_prevalence", recipient="provider_a", epsilon=1.0, delta=1e-6)
    res_a1 = orch.execute_release(qa1, test_matrix)
    assert res_a1["privacy_accounting"]["budget_scope_id"] == "collusion:group_ab"

    # Provider B in group_ab tries to make query 2 with another 1.0 epsilon
    # Since they share collusion:group_ab, 1.0 + 1.0 = 2.0 > 1.5 -> MUST RAISE
    qb1 = AnalyticsQuery(metric="user_normalized_share", recipient="provider_b", epsilon=1.0, delta=1e-6)
    with pytest.raises(PrivacyBudgetExceededError):
        orch.execute_release(qb1, test_matrix)

    # Meanwhile, Provider C (group_c) is unaffected and can execute 1.0 epsilon query
    qc1 = AnalyticsQuery(metric="interaction_volume", recipient="provider_c", epsilon=1.0, delta=1e-6)
    res_c1 = orch.execute_release(qc1, test_matrix)
    assert res_c1["cache_hit"] is False
    assert res_c1["privacy_accounting"]["budget_scope_id"] == "collusion:group_c"


def test_global_external_blocks_provider_switching_bypass(tmp_path, test_matrix):
    """Under GLOBAL_EXTERNAL, switching provider cannot bypass the privacy cap."""
    config = load_app_config("configs")
    config.privacy.recipient_composition.mode = "global_external"
    config.privacy.global_budget.epsilon = 1.5
    config.privacy.global_budget.delta = 1e-5
    state_mgr = PrivacyStateManager(tmp_path / "privacy_state.db")
    orch = ReleaseOrchestrator(app_config=config, state_manager=state_mgr)

    # Provider A consumes 1.0 epsilon
    qa = AnalyticsQuery(metric="user_prevalence", recipient="provider_a", epsilon=1.0, delta=1e-6)
    orch.execute_release(qa, test_matrix)

    # Switching to Provider B for a different metric cannot bypass global cap (1.0 + 1.0 > 1.5)
    qb = AnalyticsQuery(metric="user_normalized_share", recipient="provider_b", epsilon=1.0, delta=1e-6)
    with pytest.raises(PrivacyBudgetExceededError):
        orch.execute_release(qb, test_matrix)

    # Switching to Provider C also blocked
    qc = AnalyticsQuery(metric="interaction_volume", recipient="provider_c", epsilon=1.0, delta=1e-6)
    with pytest.raises(PrivacyBudgetExceededError):
        orch.execute_release(qc, test_matrix)


def test_historical_composition_rdp_benefit(tmp_path, test_matrix):
    """Verify that multiple releases accumulate RDP vectors and provide composition savings."""
    config = load_app_config("configs")
    config.privacy.global_budget.epsilon = 4.0
    config.privacy.global_budget.delta = 1e-5
    state_mgr = PrivacyStateManager(tmp_path / "privacy_state.db")
    orch = ReleaseOrchestrator(app_config=config, state_manager=state_mgr)

    # Run 3 queries with different metrics or windows to avoid cache hits
    q1 = AnalyticsQuery(metric="user_prevalence", recipient="provider_a", epsilon=0.5, delta=1e-6)
    res1 = orch.execute_release(q1, test_matrix)
    assert res1["cache_hit"] is False

    q2 = AnalyticsQuery(metric="user_normalized_share", recipient="provider_a", epsilon=0.5, delta=1e-6)
    res2 = orch.execute_release(q2, test_matrix)
    assert res2["cache_hit"] is False

    q3 = AnalyticsQuery(metric="interaction_volume", recipient="provider_a", epsilon=0.5, delta=1e-6)
    res3 = orch.execute_release(q3, test_matrix)
    assert res3["cache_hit"] is False

    # Check PrivacyView
    view = orch.get_privacy_view("provider_a")
    assert view.release_count == 3
    assert view.total_releases == 3
    assert view.budget_scope_id == "global_external"
    assert view.effective_epsilon > Decimal("0.0")
    # Basic sum would be 1.5; effective epsilon is bounded
    assert view.effective_epsilon <= Decimal("1.5")
    assert view.remaining_epsilon == Decimal("4.0") - view.effective_epsilon


def test_production_validator_rejects_rolling_reset():
    """Production validator must fail-closed if rolling reset is configured."""
    config = load_app_config("configs")
    config.privacy.history.mode = "rolling_reset"
    with pytest.raises(ProductionSafetyConfigurationError, match=r"Rolling privacy reset|lifetime"):
        validate_production_config(config)


def test_legacy_v01_database_migration(tmp_path):
    """Verify migration of legacy v0.1 SQLite database adds new columns and migrates Gaussian records."""
    db_path = tmp_path / "legacy_v01.db"

    # Create legacy schema without v0.2 columns
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE release_store (
            release_key TEXT PRIMARY KEY,
            reservation_id TEXT NOT NULL,
            owner_id TEXT NOT NULL,
            generation INTEGER NOT NULL,
            release_id TEXT NOT NULL,
            created_at TEXT NOT NULL,
            status TEXT NOT NULL,
            canonical_semantics_json TEXT NOT NULL,
            metric TEXT NOT NULL,
            window_start TEXT,
            window_end TEXT,
            dataset_version TEXT NOT NULL,
            mechanism TEXT NOT NULL,
            epsilon TEXT NOT NULL,
            delta TEXT NOT NULL,
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
    cursor.execute("""
        CREATE TABLE privacy_reservations (
            reservation_id TEXT PRIMARY KEY,
            release_key TEXT NOT NULL,
            owner_id TEXT NOT NULL,
            generation INTEGER NOT NULL,
            epsilon TEXT NOT NULL,
            delta TEXT NOT NULL,
            status TEXT NOT NULL,
            created_at TEXT NOT NULL,
            committed_at TEXT
        )
    """)
    cursor.execute("""
        CREATE TABLE privacy_ledger (
            entry_id INTEGER PRIMARY KEY AUTOINCREMENT,
            release_id TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            metric TEXT NOT NULL,
            recipient TEXT NOT NULL,
            window_start TEXT,
            window_end TEXT,
            dataset_version TEXT NOT NULL,
            mechanism TEXT NOT NULL,
            epsilon TEXT NOT NULL,
            delta TEXT NOT NULL,
            sensitivity REAL NOT NULL,
            noise_scale REAL NOT NULL,
            release_key TEXT NOT NULL,
            cache_hit INTEGER NOT NULL
        )
    """)

    # Insert a committed legacy Gaussian release
    res_id = str(uuid.uuid4())
    cursor.execute("""
        INSERT INTO privacy_reservations (
            reservation_id, release_key, owner_id, generation, epsilon, delta, status, created_at, committed_at
        ) VALUES (?, 'legacy_rel_key_1', 'legacy_owner', 1, '1.0', '1e-6', 'COMMITTED', '2026-01-01T00:00:00Z', '2026-01-01T00:00:01Z')
    """, (res_id,))
    cursor.execute("""
        INSERT INTO release_store (
            release_key, reservation_id, owner_id, generation, release_id, created_at, status,
            canonical_semantics_json, metric, dataset_version, mechanism, epsilon, delta,
            sensitivity, noise_scale, cell_names_json, n_users, public_population_size,
            dp_vector_json, post_processed_json, committed_at
        ) VALUES (
            'legacy_rel_key_1', ?, 'legacy_owner', 1, 'rel_1', '2026-01-01T00:00:00Z', 'COMMITTED',
            '{}', 'user_prevalence', 'v1.0', 'gaussian', '1.0', '1e-6',
            1.0, 4.22, '[]', 100, 50000,
            '[1.0, 2.0]', '{}', '2026-01-01T00:00:01Z'
        )
    """, (res_id,))
    conn.commit()
    conn.close()

    # Instantiate PrivacyStateManager on legacy DB -> triggers auto-migration
    mgr = PrivacyStateManager(db_path)

    # Check that columns were added and legacy record has RDP vector computed
    with mgr._connection() as c:
        cur = c.cursor()
        cur.execute("SELECT rdp_cost_json, budget_scope_id FROM release_store WHERE release_key = 'legacy_rel_key_1'")
        row = cur.fetchone()
        assert row is not None
        assert row[0] is not None, "Legacy release must have RDP cost computed during migration"
        assert row[1] == "global_external"
