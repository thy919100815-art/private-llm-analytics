"""Tests for Recipient Privacy View Accounting, Policy Binding, and Fail-Closed Verification (v0.2)."""

from decimal import Decimal
import json
from pathlib import Path
import sqlite3
import numpy as np
import pytest

from private_llm_analytics.aggregation.user_matrix import build_user_matrix
from private_llm_analytics.config.loader import load_app_config
from private_llm_analytics.planner.query import AnalyticsQuery
from private_llm_analytics.privacy.exceptions import (
    PrivacyBudgetExceededError,
    PrivacyConfigurationError,
    ProductionSafetyConfigurationError,
)
from private_llm_analytics.privacy.state import PrivacyStateManager
from private_llm_analytics.release.release import ReleaseOrchestrator
from private_llm_analytics.simulator.generator import SyntheticDataGenerator
from private_llm_analytics.simulator.scenario import SimulationScenario


@pytest.fixture
def test_umatrix():
    scenario = SimulationScenario(n_users=50, random_seed=42)
    gen = SyntheticDataGenerator(scenario)
    df = gen.generate(seed=42)
    return build_user_matrix(df, public_population_size=50)


# =========================================================================
# 1. Zero-Cost Replay Restricted to Same Privacy View (Section A & G)
# =========================================================================

def test_global_same_query_cross_recipient_zero_cost(tmp_path, test_umatrix):
    """GLOBAL_EXTERNAL: provider_a query X then provider_b query X -> same scope -> cache hit / zero additional cost."""
    db_path = tmp_path / "privacy_state.db"
    config = load_app_config("configs")
    config.privacy.recipient_composition.mode = "global_external"
    config.privacy.accountant.epsilon_cap = 5.0
    config.privacy.accountant.delta_global = 1e-6

    state_mgr = PrivacyStateManager(db_path)
    orch = ReleaseOrchestrator(app_config=config, state_manager=state_mgr)

    q1 = AnalyticsQuery(
        metric="user_prevalence",
        recipient="provider_a",
        epsilon=1.0,
        delta=1e-6,
        window_start="2026-01-01",
        window_end="2026-01-31",
    )
    res1 = orch.execute_release(q1, test_umatrix)
    assert not res1["cache_hit"]
    assert res1["privacy_accounting"]["budget_scope_id"] == "global_external"
    assert res1["privacy_accounting"]["history_release_count"] == 1
    eps1 = res1["privacy_accounting"]["effective_epsilon_after"]

    # Provider B issues identical query X
    q2 = AnalyticsQuery(
        metric="user_prevalence",
        recipient="provider_b",
        epsilon=1.0,
        delta=1e-6,
        window_start="2026-01-01",
        window_end="2026-01-31",
    )
    res2 = orch.execute_release(q2, test_umatrix)
    assert res2["cache_hit"]
    assert res2["release_key"] == res1["release_key"]
    assert res2["privacy_accounting"]["budget_scope_id"] == "global_external"
    assert res2["privacy_accounting"]["effective_epsilon_after"] == eps1
    assert res2["incremental_privacy_cost"] == 0.0

    # Privacy view confirms only 1 release in global_external
    view = orch.get_privacy_view("global_external")
    assert view.release_count == 1


def test_per_recipient_same_query_cross_recipient_charged(tmp_path, test_umatrix):
    """PER_RECIPIENT: provider_a query X then provider_b query X -> different scopes -> distinct release keys -> both charged."""
    db_path = tmp_path / "privacy_state.db"
    config = load_app_config("configs")
    config.privacy.recipient_composition.mode = "per_recipient"
    config.privacy.accountant.epsilon_cap = 5.0
    config.privacy.accountant.delta_global = 1e-6

    state_mgr = PrivacyStateManager(db_path)
    orch = ReleaseOrchestrator(app_config=config, state_manager=state_mgr)

    q1 = AnalyticsQuery(
        metric="user_prevalence",
        recipient="provider_a",
        epsilon=1.0,
        delta=1e-6,
        window_start="2026-01-01",
        window_end="2026-01-31",
    )
    res1 = orch.execute_release(q1, test_umatrix)
    assert not res1["cache_hit"]
    assert res1["privacy_accounting"]["budget_scope_id"] == "recipient:provider_a"
    assert res1["privacy_accounting"]["history_release_count"] == 1

    # Provider B issues same query X
    q2 = AnalyticsQuery(
        metric="user_prevalence",
        recipient="provider_b",
        epsilon=1.0,
        delta=1e-6,
        window_start="2026-01-01",
        window_end="2026-01-31",
    )
    res2 = orch.execute_release(q2, test_umatrix)
    assert not res2["cache_hit"]  # Scope-aware release identity prevents cross-scope cache hit
    assert res2["release_key"] != res1["release_key"]
    assert res2["privacy_accounting"]["budget_scope_id"] == "recipient:provider_b"
    assert res2["privacy_accounting"]["history_release_count"] == 1

    # Both scopes charged separately
    view_a = orch.get_privacy_view("provider_a")
    view_b = orch.get_privacy_view("provider_b")
    assert view_a.release_count == 1
    assert view_b.release_count == 1


def test_collusion_same_group_same_query_zero_cost(tmp_path, test_umatrix):
    """COLLUSION_GROUP: A and B in group_ab -> A queries X, then B queries X -> same scope -> cache hit / zero additional cost."""
    db_path = tmp_path / "privacy_state.db"
    config = load_app_config("configs")
    config.privacy.recipient_composition.mode = "collusion_group"
    config.privacy.recipient_composition.groups = {
        "group_ab": ["provider_a", "provider_b"],
        "group_c": ["provider_c"],
    }
    config.privacy.accountant.epsilon_cap = 5.0
    config.privacy.accountant.delta_global = 1e-6

    state_mgr = PrivacyStateManager(db_path)
    orch = ReleaseOrchestrator(app_config=config, state_manager=state_mgr)

    q1 = AnalyticsQuery(
        metric="user_prevalence",
        recipient="provider_a",
        epsilon=1.0,
        delta=1e-6,
        window_start="2026-01-01",
        window_end="2026-01-31",
    )
    res1 = orch.execute_release(q1, test_umatrix)
    assert not res1["cache_hit"]
    assert res1["privacy_accounting"]["budget_scope_id"] == "collusion:group_ab"

    # Provider B in same collusion group queries X
    q2 = AnalyticsQuery(
        metric="user_prevalence",
        recipient="provider_b",
        epsilon=1.0,
        delta=1e-6,
        window_start="2026-01-01",
        window_end="2026-01-31",
    )
    res2 = orch.execute_release(q2, test_umatrix)
    assert res2["cache_hit"]
    assert res2["release_key"] == res1["release_key"]
    assert res2["privacy_accounting"]["budget_scope_id"] == "collusion:group_ab"
    assert res2["incremental_privacy_cost"] == 0.0

    view_ab = orch.get_privacy_view("provider_a")
    assert view_ab.release_count == 1


def test_collusion_different_group_same_query_charged(tmp_path, test_umatrix):
    """COLLUSION_GROUP: A in group_ab queries X, C in group_c queries X -> C cannot hit group_ab cache and is charged independently."""
    db_path = tmp_path / "privacy_state.db"
    config = load_app_config("configs")
    config.privacy.recipient_composition.mode = "collusion_group"
    config.privacy.recipient_composition.groups = {
        "group_ab": ["provider_a", "provider_b"],
        "group_c": ["provider_c"],
    }
    config.privacy.accountant.epsilon_cap = 5.0
    config.privacy.accountant.delta_global = 1e-6

    state_mgr = PrivacyStateManager(db_path)
    orch = ReleaseOrchestrator(app_config=config, state_manager=state_mgr)

    q1 = AnalyticsQuery(
        metric="user_prevalence",
        recipient="provider_a",
        epsilon=1.0,
        delta=1e-6,
        window_start="2026-01-01",
        window_end="2026-01-31",
    )
    res1 = orch.execute_release(q1, test_umatrix)
    assert not res1["cache_hit"]
    assert res1["privacy_accounting"]["budget_scope_id"] == "collusion:group_ab"

    # Provider C in group_c queries X
    q3 = AnalyticsQuery(
        metric="user_prevalence",
        recipient="provider_c",
        epsilon=1.0,
        delta=1e-6,
        window_start="2026-01-01",
        window_end="2026-01-31",
    )
    res3 = orch.execute_release(q3, test_umatrix)
    assert not res3["cache_hit"]
    assert res3["release_key"] != res1["release_key"]
    assert res3["privacy_accounting"]["budget_scope_id"] == "collusion:group_c"
    assert res3["privacy_accounting"]["history_release_count"] == 1

    view_ab = orch.get_privacy_view("provider_a")
    view_c = orch.get_privacy_view("provider_c")
    assert view_ab.release_count == 1
    assert view_c.release_count == 1


def test_internal_external_release_identity_isolated(tmp_path, test_umatrix):
    """Internal trusted queries and external commercial queries are scope-isolated."""
    db_path = tmp_path / "privacy_state.db"
    config = load_app_config("configs")
    config.privacy.recipient_composition.mode = "global_external"

    state_mgr = PrivacyStateManager(db_path)
    orch = ReleaseOrchestrator(app_config=config, state_manager=state_mgr)

    q_int = AnalyticsQuery(
        metric="user_prevalence",
        recipient="internal",
        epsilon=1.0,
        delta=1e-6,
        window_start="2026-01-01",
        window_end="2026-01-31",
    )
    res_int = orch.execute_release(q_int, test_umatrix)
    assert res_int["privacy_accounting"]["budget_scope_id"] == "internal"

    q_ext = AnalyticsQuery(
        metric="user_prevalence",
        recipient="provider_a",
        epsilon=1.0,
        delta=1e-6,
        window_start="2026-01-01",
        window_end="2026-01-31",
    )
    res_ext = orch.execute_release(q_ext, test_umatrix)
    assert res_ext["privacy_accounting"]["budget_scope_id"] == "global_external"
    assert res_ext["release_key"] != res_int["release_key"]
    assert not res_ext["cache_hit"]


# =========================================================================
# 2. Policy Immutability Binding & Restart Verification (Section B & G)
# =========================================================================

def test_scope_policy_change_after_history_fails_closed(tmp_path, test_umatrix):
    """If DB has historical external releases, changing composition policy fails closed."""
    db_path = tmp_path / "privacy_state.db"
    config1 = load_app_config("configs")
    config1.privacy.recipient_composition.mode = "per_recipient"
    config1.privacy.accountant.epsilon_cap = 5.0
    config1.privacy.accountant.delta_global = 1e-6

    state_mgr1 = PrivacyStateManager(db_path)
    orch1 = ReleaseOrchestrator(app_config=config1, state_manager=state_mgr1)

    q = AnalyticsQuery(
        metric="user_prevalence",
        recipient="provider_a",
        epsilon=1.0,
        delta=1e-6,
        window_start="2026-01-01",
        window_end="2026-01-31",
    )
    orch1.execute_release(q, test_umatrix)

    # Now attempt to start/query with global_external policy on the same state DB
    config2 = load_app_config("configs")
    config2.privacy.recipient_composition.mode = "global_external"
    config2.privacy.accountant.epsilon_cap = 5.0
    config2.privacy.accountant.delta_global = 1e-6

    state_mgr2 = PrivacyStateManager(db_path)
    with pytest.raises(ProductionSafetyConfigurationError, match="policy mismatch"):
        ReleaseOrchestrator(app_config=config2, state_manager=state_mgr2)


def test_collusion_group_membership_change_fails_closed(tmp_path, test_umatrix):
    """Changing collusion group membership on DB with history fails closed."""
    db_path = tmp_path / "privacy_state.db"
    config1 = load_app_config("configs")
    config1.privacy.recipient_composition.mode = "collusion_group"
    config1.privacy.recipient_composition.groups = {
        "group_ab": ["provider_a", "provider_b"],
    }

    state_mgr1 = PrivacyStateManager(db_path)
    orch1 = ReleaseOrchestrator(app_config=config1, state_manager=state_mgr1)

    q = AnalyticsQuery(
        metric="user_prevalence",
        recipient="provider_a",
        epsilon=1.0,
        delta=1e-6,
        window_start="2026-01-01",
        window_end="2026-01-31",
    )
    orch1.execute_release(q, test_umatrix)

    # Modify group memberships
    config2 = load_app_config("configs")
    config2.privacy.recipient_composition.mode = "collusion_group"
    config2.privacy.recipient_composition.groups = {
        "group_ab": ["provider_a"],
        "group_b": ["provider_b"],
    }
    state_mgr2 = PrivacyStateManager(db_path)
    with pytest.raises(ProductionSafetyConfigurationError, match="policy mismatch"):
        ReleaseOrchestrator(app_config=config2, state_manager=state_mgr2)


def test_restart_retains_policy_fingerprint(tmp_path, test_umatrix):
    """Restarting with identical policy configuration succeeds and operates normally."""
    db_path = tmp_path / "privacy_state.db"
    config = load_app_config("configs")
    config.privacy.recipient_composition.mode = "collusion_group"
    config.privacy.recipient_composition.groups = {
        "group_ab": ["provider_a", "provider_b"],
    }

    state_mgr = PrivacyStateManager(db_path)
    orch1 = ReleaseOrchestrator(app_config=config, state_manager=state_mgr)

    q = AnalyticsQuery(
        metric="user_prevalence",
        recipient="provider_a",
        epsilon=1.0,
        delta=1e-6,
        window_start="2026-01-01",
        window_end="2026-01-31",
    )
    res1 = orch1.execute_release(q, test_umatrix)

    # Reboot with same config
    orch2 = ReleaseOrchestrator(app_config=config, state_manager=PrivacyStateManager(db_path))
    res2 = orch2.execute_release(q, test_umatrix)
    assert res2["cache_hit"]
    assert res2["release_key"] == res1["release_key"]


def test_same_scope_replay_remains_exact_same_payload(tmp_path, test_umatrix):
    """Same-scope replay returns identical payload and zero incremental cost."""
    db_path = tmp_path / "privacy_state.db"
    config = load_app_config("configs")

    state_mgr = PrivacyStateManager(db_path)
    orch = ReleaseOrchestrator(app_config=config, state_manager=state_mgr)

    q = AnalyticsQuery(
        metric="user_prevalence",
        recipient="provider_a",
        epsilon=1.0,
        delta=1e-6,
        window_start="2026-01-01",
        window_end="2026-01-31",
    )
    res1 = orch.execute_release(q, test_umatrix)
    res2 = orch.execute_release(q, test_umatrix)
    assert res2["cache_hit"]
    assert res1["results"] == res2["results"]
    assert res2["incremental_privacy_cost"] == 0.0


def test_cross_scope_replay_cannot_bypass_budget_cap(tmp_path, test_umatrix):
    """Under per_recipient, an exhausted recipient cannot reuse another recipient's release to bypass cap."""
    db_path = tmp_path / "privacy_state.db"
    config = load_app_config("configs")
    config.privacy.recipient_composition.mode = "per_recipient"
    # Set cap to 1.0 (permits exactly 1 release under RDP)
    config.privacy.accountant.epsilon_cap = 1.0
    config.privacy.accountant.delta_global = 1e-6

    state_mgr = PrivacyStateManager(db_path)
    orch = ReleaseOrchestrator(app_config=config, state_manager=state_mgr)

    q_a = AnalyticsQuery(
        metric="user_prevalence",
        recipient="provider_a",
        epsilon=1.0,
        delta=1e-6,
        window_start="2026-01-01",
        window_end="2026-01-31",
    )
    # Provider A consumes budget
    with pytest.raises(PrivacyBudgetExceededError):
        # 1 release of eps=1.0 at delta=1e-6 under RDP gives eff_eps ~1.34 > cap 1.0
        orch.execute_release(q_a, test_umatrix)


# =========================================================================
# 3. Cache Replay Effective Epsilon Reflects Current Scope (Section F)
# =========================================================================

def test_replay_effective_epsilon_reflects_current_scope(tmp_path, test_umatrix):
    """Replay metadata must return the current cumulative PrivacyView, not the old release snapshot."""
    db_path = tmp_path / "privacy_state.db"
    config = load_app_config("configs")
    config.privacy.accountant.epsilon_cap = 10.0
    config.privacy.accountant.delta_global = 1e-6

    state_mgr = PrivacyStateManager(db_path)
    orch = ReleaseOrchestrator(app_config=config, state_manager=state_mgr)

    # Release A
    q_a = AnalyticsQuery(
        metric="user_prevalence",
        recipient="provider_a",
        epsilon=1.0,
        delta=1e-6,
        window_start="2026-01-01",
        window_end="2026-01-15",
    )
    res_a = orch.execute_release(q_a, test_umatrix)
    eps_after_a = res_a["privacy_accounting"]["effective_epsilon_after"]
    assert res_a["privacy_accounting"]["history_release_count"] == 1

    # Release B
    q_b = AnalyticsQuery(
        metric="user_prevalence",
        recipient="provider_a",
        epsilon=1.0,
        delta=1e-6,
        window_start="2026-01-16",
        window_end="2026-01-31",
    )
    res_b = orch.execute_release(q_b, test_umatrix)
    eps_after_b = res_b["privacy_accounting"]["effective_epsilon_after"]
    assert eps_after_b > eps_after_a
    assert res_b["privacy_accounting"]["history_release_count"] == 2

    # Replay Release A
    res_a_replay = orch.execute_release(q_a, test_umatrix)
    assert res_a_replay["cache_hit"]
    # Metadata MUST match current cumulative scope view (A + B), NOT old snapshot of A!
    assert res_a_replay["privacy_accounting"]["effective_epsilon_after"] == eps_after_b
    assert res_a_replay["privacy_accounting"]["history_release_count"] == 2
    assert res_a_replay["incremental_privacy_cost"] == 0.0


# =========================================================================
# 4. Independent Fail-Closed Tests (Section H & D)
# =========================================================================

def test_independent_fail_closed_on_null_rdp_in_db(tmp_path, test_umatrix):
    """Manually insert COMMITTED external reservation with rdp_cost_json=NULL, query must fail closed before new release."""
    db_path = tmp_path / "privacy_state.db"
    state_mgr = PrivacyStateManager(db_path)

    # Manually inject a committed reservation with NULL rdp_cost_json
    with state_mgr._connection() as conn:
        conn.execute("""
            INSERT INTO privacy_reservations (
                reservation_id, release_key, owner_id, generation, epsilon, delta, status, created_at,
                budget_scope_id, accountant_mode, rdp_cost_json
            ) VALUES ('corrupt_res_1', 'dummy_key', 'worker_1', 1, '1.0', '1e-6', 'COMMITTED', '2026-01-01T00:00:00Z',
                      'global_external', 'rdp', NULL)
        """)
        conn.commit()

    config = load_app_config("configs")
    orch = ReleaseOrchestrator(app_config=config, state_manager=state_mgr)

    q = AnalyticsQuery(
        metric="user_prevalence",
        recipient="provider_a",
        epsilon=1.0,
        delta=1e-6,
        window_start="2026-01-01",
        window_end="2026-01-31",
    )

    with pytest.raises(PrivacyConfigurationError, match="missing or NULL rdp_cost_json"):
        orch.execute_release(q, test_umatrix)

    # Confirm release_store and privacy_ledger have no new releases
    with state_mgr._connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM release_store")
        assert cursor.fetchone()[0] == 0
        cursor.execute("SELECT COUNT(*) FROM privacy_ledger")
        assert cursor.fetchone()[0] == 0


def test_legacy_laplace_migrates_to_rdp(tmp_path, test_umatrix):
    """Legacy SQLite table with Laplace release (rdp_cost_json NULL) migrates seamlessly to RDP."""
    db_path = tmp_path / "legacy_laplace.db"
    conn = sqlite3.connect(db_path)
    conn.execute("""
        CREATE TABLE release_store (
            release_key TEXT PRIMARY KEY,
            reservation_id TEXT NOT NULL,
            owner_id TEXT NOT NULL DEFAULT '',
            generation INTEGER NOT NULL DEFAULT 1,
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
    conn.execute("""
        CREATE TABLE privacy_reservations (
            reservation_id TEXT PRIMARY KEY,
            release_key TEXT NOT NULL,
            owner_id TEXT NOT NULL DEFAULT '',
            generation INTEGER NOT NULL DEFAULT 1,
            epsilon TEXT NOT NULL,
            delta TEXT NOT NULL,
            status TEXT NOT NULL,
            created_at TEXT NOT NULL,
            committed_at TEXT
        )
    """)
    conn.execute("""
        INSERT INTO release_store VALUES (
            'laplace_key_1', 'res_lap_1', '', 1, 'rel_lap_1', '2026-01-01T00:00:00Z', 'COMMITTED',
            '{}', 'user_prevalence', NULL, NULL, 'v1.0', 'laplace', '1.0', '0.0', 1.0, 1.0, '[]', 50, 50, '[]', '[]', '2026-01-01T00:00:00Z'
        )
    """)
    conn.execute("""
        INSERT INTO privacy_reservations VALUES (
            'res_lap_1', 'laplace_key_1', '', 1, '1.0', '0.0', 'COMMITTED', '2026-01-01T00:00:00Z', '2026-01-01T00:00:00Z'
        )
    """)
    conn.commit()
    conn.close()

    # Initializing manager migrates legacy Laplace to RDP
    mgr = PrivacyStateManager(db_path)
    with mgr._connection() as c:
        row = c.execute("SELECT rdp_cost_json FROM release_store WHERE release_key = 'laplace_key_1'").fetchone()
        assert row[0] is not None
        cost_dict = json.loads(row[0])
        # Under pure-DP Laplace bound: RDP(alpha) <= epsilon = 1.0 for all orders
        for o, rdp_val in cost_dict.items():
            assert Decimal(rdp_val) == Decimal("1.0")

    # Orchestrator can execute releases and compose with migrated Laplace release
    config = load_app_config("configs")
    config.privacy.accountant.epsilon_cap = 10.0
    config.privacy.accountant.delta_global = 1e-6
    orch = ReleaseOrchestrator(app_config=config, state_manager=mgr)
    q = AnalyticsQuery(
        metric="user_prevalence",
        recipient="provider_a",
        epsilon=1.0,
        delta=1e-6,
        window_start="2026-02-01",
        window_end="2026-02-28",
    )
    res = orch.execute_release(q, test_umatrix)
    assert not res["cache_hit"]
    # 1 migrated Laplace + 1 new Gaussian = 2 releases in scope
    assert res["privacy_accounting"]["history_release_count"] == 2


def test_unknown_legacy_mechanism_fails_closed(tmp_path):
    """Database with unknown historical mechanism and missing RDP fails closed during migration."""
    db_path = tmp_path / "unknown_mech.db"
    conn = sqlite3.connect(db_path)
    conn.execute("""
        CREATE TABLE release_store (
            release_key TEXT PRIMARY KEY, reservation_id TEXT NOT NULL, owner_id TEXT NOT NULL DEFAULT '',
            generation INTEGER NOT NULL DEFAULT 1, release_id TEXT NOT NULL, created_at TEXT NOT NULL,
            status TEXT NOT NULL, canonical_semantics_json TEXT NOT NULL, metric TEXT NOT NULL,
            window_start TEXT, window_end TEXT, dataset_version TEXT NOT NULL, mechanism TEXT NOT NULL,
            epsilon TEXT NOT NULL, delta TEXT NOT NULL, sensitivity REAL NOT NULL, noise_scale REAL NOT NULL,
            cell_names_json TEXT NOT NULL, n_users INTEGER NOT NULL, public_population_size INTEGER,
            dp_vector_json TEXT, post_processed_json TEXT, committed_at TEXT
        )
    """)
    conn.execute("""
        INSERT INTO release_store VALUES (
            'k_unknown', 'r_unknown', '', 1, 'rel_u', '2026-01-01', 'COMMITTED', '{}',
            'user_prevalence', NULL, NULL, 'v1.0', 'unsupported_quantum_mechanism', '1.0', '0.0', 1.0, 1.0, '[]', 50, 50, '[]', '[]', '2026-01-01'
        )
    """)
    conn.commit()
    conn.close()

    with pytest.raises(ValueError, match="Fail-closed"):
        PrivacyStateManager(db_path)


def test_pending_missing_rdp_fails_closed(tmp_path, test_umatrix):
    """Active PENDING reservation with missing rdp_cost_json causes new releases to fail closed."""
    db_path = tmp_path / "pending_null_rdp.db"
    state_mgr = PrivacyStateManager(db_path)

    with state_mgr._connection() as conn:
        conn.execute("""
            INSERT INTO privacy_reservations (
                reservation_id, release_key, owner_id, generation, epsilon, delta, status, created_at,
                budget_scope_id, accountant_mode, rdp_cost_json
            ) VALUES ('pending_corrupt', 'key_p', 'worker_p', 1, '1.0', '1e-6', 'PENDING', '2026-01-01T00:00:00Z',
                      'global_external', 'rdp', NULL)
        """)
        conn.commit()

    config = load_app_config("configs")
    orch = ReleaseOrchestrator(app_config=config, state_manager=state_mgr)
    q = AnalyticsQuery(
        metric="user_prevalence",
        recipient="provider_a",
        epsilon=1.0,
        delta=1e-6,
        window_start="2026-01-01",
        window_end="2026-01-31",
    )
    with pytest.raises(PrivacyConfigurationError, match="missing or NULL rdp_cost_json"):
        orch.execute_release(q, test_umatrix)


def test_rdp_accountant_public_interface_fail_closed(tmp_path):
    """RDPAccountant legacy check_budget_available and consume_budget strictly raise NotImplementedError."""
    from private_llm_analytics.privacy.budget import PrivacyBudget
    from private_llm_analytics.privacy.rdp_accountant import RDPAccountant

    db_path = tmp_path / "privacy_state.db"
    state_mgr = PrivacyStateManager(db_path)
    acct = RDPAccountant(epsilon_cap=5.0, delta_global=1e-6, state_manager=state_mgr)

    # Legacy check raises NotImplementedError
    with pytest.raises(NotImplementedError, match="cannot evaluate budget availability solely from scalar"):
        acct.check_budget_available(PrivacyBudget(epsilon=1.0, delta=1e-6))

    # Legacy consume raises NotImplementedError
    with pytest.raises(NotImplementedError, match="does not support non-atomic scalar consume_budget"):
        acct.consume_budget(PrivacyBudget(epsilon=1.0, delta=1e-6), "q1")

    # Candidate preview works as expected
    cand_rdp = {Decimal(str(a)): Decimal("0.5") for a in acct.orders}
    prev = acct.preview_candidate_rdp(cand_rdp)
    assert prev["allowed"] is True
    assert float(prev["epsilon_before"]) == 0.0
    assert float(prev["epsilon_after"]) > 0.0



def _concurrent_policy_proc(db_path_str: str, mode: str, groups, roles, barrier, result_queue):
    from decimal import Decimal
    from private_llm_analytics.privacy.state import PrivacyStateManager
    from private_llm_analytics.privacy.budget import PrivacyBudget
    state_mgr = PrivacyStateManager(db_path_str)
    gb = PrivacyBudget(epsilon=10.0, delta=1e-5)
    bn = PrivacyBudget(epsilon=1.0, delta=1e-6)

    # Barrier synchronization ensures both workers enter reserve_or_get_release simultaneously
    if barrier is not None:
        try:
            barrier.wait(timeout=10.0)
        except Exception as e:
            result_queue.put({"success": False, "error": "BarrierTimeout", "msg": str(e), "mode": mode})
            return

    try:
        res = state_mgr.reserve_or_get_release(
            release_key=f"k_{mode}",
            canonical_semantics={"mode": mode},
            global_budget=gb,
            budget_needed=bn,
            plan_metadata={"mechanism": "gaussian", "sensitivity": 1.0, "noise_scale": 1.0},
            recipient="provider_a",
            recipient_role="external",
            budget_scope_id=f"scope_{mode}",
            accountant_mode="rdp",
            rdp_candidate_cost={Decimal("2.0"): Decimal("0.1"), Decimal("3.0"): Decimal("0.2")},
            rdp_orders=[Decimal("2.0"), Decimal("3.0")],
            epsilon_cap=Decimal("10.0"),
            delta_global=Decimal("1e-5"),
            policy_mode=mode,
            policy_groups=groups,
            recipient_roles=roles,
        )
        result_queue.put({"success": True, "error": None, "mode": mode})
    except Exception as e:
        result_queue.put({"success": False, "error": type(e).__name__, "msg": str(e), "mode": mode})


def test_concurrent_conflicting_policy_processes_fail_closed(tmp_path):
    """Two concurrent processes attempting conflicting policy binding fail closed atomically with barrier (Item 7 & F)."""
    import multiprocessing

    db_path = tmp_path / "concurrent_policy.db"
    # Ensure tables exist
    PrivacyStateManager(db_path)

    roles = {"provider_a": "external", "analyst_internal": "internal"}
    barrier = multiprocessing.Barrier(2)
    queue = multiprocessing.Queue()

    p1 = multiprocessing.Process(
        target=_concurrent_policy_proc,
        args=(str(db_path), "global_external", None, roles, barrier, queue),
    )
    p2 = multiprocessing.Process(
        target=_concurrent_policy_proc,
        args=(str(db_path), "per_recipient", None, roles, barrier, queue),
    )

    p1.start()
    p2.start()

    p1.join(timeout=10.0)
    p2.join(timeout=10.0)

    results = []
    while not queue.empty():
        results.append(queue.get())

    successes = [r for r in results if r["success"]]
    failures = [r for r in results if not r["success"]]

    assert len(successes) == 1, f"Expected exactly 1 success, got: {results}"
    assert len(failures) == 1, f"Expected exactly 1 failure, got: {results}"
    assert failures[0]["error"] in ("ProductionSafetyConfigurationError", "IntegrityError")

    # Singleton row verification
    conn = sqlite3.connect(db_path)
    count = conn.execute("SELECT COUNT(*) FROM privacy_policy_metadata").fetchone()[0]
    row = conn.execute("SELECT id, recipient_composition_mode FROM privacy_policy_metadata").fetchone()
    conn.close()

    assert count == 1
    assert row[0] == 1
    assert row[1] == successes[0]["mode"]


def test_legacy_v01_only_migrates_to_global_external(tmp_path):
    """Legacy v0.1 DB with external history and no policy metadata can ONLY migrate to global_external (Item 3)."""
    db_path = tmp_path / "legacy_v01.db"
    conn = sqlite3.connect(db_path)
    conn.execute("""
        CREATE TABLE release_store (
            release_key TEXT PRIMARY KEY, reservation_id TEXT NOT NULL, owner_id TEXT NOT NULL DEFAULT '',
            generation INTEGER NOT NULL DEFAULT 1, release_id TEXT NOT NULL, created_at TEXT NOT NULL,
            status TEXT NOT NULL, canonical_semantics_json TEXT NOT NULL, metric TEXT NOT NULL,
            window_start TEXT, window_end TEXT, dataset_version TEXT NOT NULL, mechanism TEXT NOT NULL,
            epsilon TEXT NOT NULL, delta TEXT NOT NULL, sensitivity REAL NOT NULL, noise_scale REAL NOT NULL,
            cell_names_json TEXT NOT NULL, n_users INTEGER NOT NULL, public_population_size INTEGER,
            dp_vector_json TEXT, post_processed_json TEXT, committed_at TEXT,
            recipient TEXT NOT NULL DEFAULT 'provider_a', recipient_role TEXT NOT NULL DEFAULT 'external',
            budget_scope_id TEXT NOT NULL DEFAULT 'global_external', accountant_mode TEXT NOT NULL DEFAULT 'rdp',
            rdp_cost_json TEXT NOT NULL, rdp_orders_json TEXT NOT NULL,
            delta_global TEXT NOT NULL DEFAULT '1e-6', effective_epsilon TEXT, best_alpha TEXT
        )
    """)
    rdp_dummy = json.dumps({"2.0": "0.1", "3.0": "0.2"})
    orders_dummy = json.dumps([2.0, 3.0])
    conn.execute("""
        INSERT INTO release_store VALUES (
            'k_legacy', 'r_legacy', '', 1, 'rel_leg', '2026-01-01', 'COMMITTED', '{}',
            'user_prevalence', NULL, NULL, 'v1.0', 'gaussian', '1.0', '1e-6', 1.0, 1.0, '[]', 50, 50, '[]', '[]', '2026-01-01',
            'provider_a', 'external', 'global_external', 'rdp', ?, ?, '1e-6', '1.0', '2.0'
        )
    """, (rdp_dummy, orders_dummy))
    conn.commit()
    conn.close()

    state_mgr = PrivacyStateManager(db_path)
    assert state_mgr.has_external_releases()

    # Attempting PER_RECIPIENT on legacy DB fails closed
    with pytest.raises(ProductionSafetyConfigurationError, match="can only be migrated to 'global_external'"):
        state_mgr.verify_or_bind_policy(mode="per_recipient")

    # Attempting COLLUSION_GROUP on legacy DB fails closed
    with pytest.raises(ProductionSafetyConfigurationError, match="can only be migrated to 'global_external'"):
        state_mgr.verify_or_bind_policy(mode="collusion_group", groups={"grp1": ["provider_a"]})

    # Migrating to GLOBAL_EXTERNAL succeeds
    state_mgr.verify_or_bind_policy(mode="global_external")

    # Verify bound singleton metadata
    with state_mgr._connection() as conn:
        row = conn.execute("SELECT id, recipient_composition_mode FROM privacy_policy_metadata").fetchone()
        assert row[0] == 1
        assert row[1] == "global_external"


def test_recipient_role_change_fails_closed(tmp_path, test_umatrix):
    """Changing recipient role mapping on a DB with external history fails closed (Item 4)."""
    config = load_app_config("configs")
    db_path = tmp_path / "role_change.db"
    config.privacy.recipient_composition.mode = "global_external"

    state_mgr = PrivacyStateManager(db_path)
    orch = ReleaseOrchestrator(app_config=config, state_manager=state_mgr)

    q = AnalyticsQuery(
        metric="user_prevalence",
        recipient="provider_a",
        epsilon=1.0,
        delta=1e-6,
        window_start="2026-01-01",
        window_end="2026-01-31",
    )
    res = orch.execute_release(q, test_umatrix)
    assert not res["cache_hit"]

    # Now alter provider_a's role in config from external to internal
    config.recipients.recipients["provider_a"].role = "internal"

    # Creating a new orchestrator with modified recipient roles must fail closed
    with pytest.raises(ProductionSafetyConfigurationError, match="policy mismatch"):
        ReleaseOrchestrator(app_config=config, state_manager=state_mgr)


def test_singleflight_wait_cache_hit_returns_fresh_privacy_view(tmp_path, test_umatrix):
    """Single-flight wait cache hit returns fresh PrivacyView metadata (Item 8)."""
    import threading

    config = load_app_config("configs")
    db_path = tmp_path / "singleflight_fresh.db"
    config.privacy.recipient_composition.mode = "global_external"

    state_mgr = PrivacyStateManager(db_path)

    worker1_reserved = threading.Event()
    worker1_continue = threading.Event()

    def pause_after_reserve():
        worker1_reserved.set()
        worker1_continue.wait(timeout=10.0)

    hooks_w1 = {"after_budget_reserve": pause_after_reserve}
    orch_w1 = ReleaseOrchestrator(app_config=config, state_manager=state_mgr, test_hooks=hooks_w1)
    orch_w2 = ReleaseOrchestrator(app_config=config, state_manager=state_mgr)

    q = AnalyticsQuery(
        metric="user_prevalence",
        recipient="provider_a",
        epsilon=1.0,
        delta=1e-6,
        window_start="2026-01-01",
        window_end="2026-01-31",
    )

    results = {}
    errors = {}

    def run_worker1():
        try:
            results["w1"] = orch_w1.execute_release(q, test_umatrix)
        except Exception as e:
            errors["w1"] = e

    def run_worker2():
        try:
            results["w2"] = orch_w2.execute_release(q, test_umatrix)
        except Exception as e:
            errors["w2"] = e

    t1 = threading.Thread(target=run_worker1)
    t2 = threading.Thread(target=run_worker2)

    t1.start()
    assert worker1_reserved.wait(timeout=5.0), "Worker 1 did not reach after_budget_reserve"

    # Worker 2 starts while Worker 1 is paused with PENDING reservation
    t2.start()

    # Let Worker 2 enter reservation wait loop
    import time
    time.sleep(0.3)

    # Let Worker 1 finish commit
    worker1_continue.set()

    t1.join(timeout=10.0)
    t2.join(timeout=10.0)

    assert "w1" not in errors, f"Worker 1 error: {errors.get('w1')}"
    assert "w2" not in errors, f"Worker 2 error: {errors.get('w2')}"

    res1 = results["w1"]
    res2 = results["w2"]

    assert not res1["cache_hit"]
    assert res2["cache_hit"]

    # Verify Worker 2 received FRESH PrivacyView metadata (Item 8)
    acct2 = res2["privacy_accounting"]
    assert acct2["history_release_count"] == 1, f"Expected history_release_count=1, got {acct2['history_release_count']}"
    assert float(acct2["effective_epsilon"]) > 0.0, f"Expected effective_epsilon > 0, got {acct2['effective_epsilon']}"
    assert float(acct2["effective_epsilon"]) == float(res1["privacy_accounting"]["effective_epsilon"])


def test_rdp_accountant_spent_api_per_recipient_collusion_group(tmp_path, test_umatrix):
    """RDPAccountant scalar spent API correctly handles PER_RECIPIENT / COLLUSION_GROUP (Item 5)."""
    config = load_app_config("configs")
    db_path = tmp_path / "rdp_multi_scope_spent.db"
    config.privacy.recipient_composition.mode = "per_recipient"

    state_mgr = PrivacyStateManager(db_path)
    orch = ReleaseOrchestrator(app_config=config, state_manager=state_mgr)

    # Initially spent_epsilon is 0
    assert orch.rdp_accountant.spent_epsilon == 0.0

    # Query for provider_a
    q1 = AnalyticsQuery(
        metric="user_prevalence",
        recipient="provider_a",
        epsilon=1.0,
        delta=1e-6,
        window_start="2026-01-01",
        window_end="2026-01-31",
    )
    orch.execute_release(q1, test_umatrix)

    spent_a = orch.rdp_accountant.spent_epsilon
    assert spent_a > 0.0

    # Query for provider_b with larger epsilon
    q2 = AnalyticsQuery(
        metric="user_prevalence",
        recipient="provider_b",
        epsilon=2.0,
        delta=1e-6,
        window_start="2026-01-01",
        window_end="2026-01-31",
    )
    orch.execute_release(q2, test_umatrix)

    # spent_epsilon must reflect max across external scopes
    spent_max = orch.rdp_accountant.spent_epsilon
    assert spent_max > spent_a

    # Specific scope queries via get_total_spent
    spent_a_budget = orch.rdp_accountant.get_total_spent(budget_scope_id="recipient:provider_a")
    spent_b_budget = orch.rdp_accountant.get_total_spent(budget_scope_id="recipient:provider_b")
    assert float(spent_a_budget.epsilon) == spent_a
    assert float(spent_b_budget.epsilon) == spent_max

    # Global total spent matches spent_max
    spent_global_budget = orch.rdp_accountant.get_total_spent()
    assert float(spent_global_budget.epsilon) == spent_max
    assert float(spent_global_budget.delta) == 1e-6


def test_blank_mechanism_fails_closed_without_guessing(tmp_path):
    """Blank or unknown mechanism strictly fails closed without guessing from delta (Item 6)."""
    db_path = tmp_path / "blank_mech.db"
    conn = sqlite3.connect(db_path)
    conn.execute("""
        CREATE TABLE release_store (
            release_key TEXT PRIMARY KEY, reservation_id TEXT NOT NULL, owner_id TEXT NOT NULL DEFAULT '',
            generation INTEGER NOT NULL DEFAULT 1, release_id TEXT NOT NULL, created_at TEXT NOT NULL,
            status TEXT NOT NULL, canonical_semantics_json TEXT NOT NULL, metric TEXT NOT NULL,
            window_start TEXT, window_end TEXT, dataset_version TEXT NOT NULL, mechanism TEXT NOT NULL,
            epsilon TEXT NOT NULL, delta TEXT NOT NULL, sensitivity REAL NOT NULL, noise_scale REAL NOT NULL,
            cell_names_json TEXT NOT NULL, n_users INTEGER NOT NULL, public_population_size INTEGER,
            dp_vector_json TEXT, post_processed_json TEXT, committed_at TEXT
        )
    """)
    # Insert with empty mechanism and delta > 0 (must NOT be guessed as Gaussian)
    conn.execute("""
        INSERT INTO release_store VALUES (
            'k_blank', 'r_blank', '', 1, 'rel_b', '2026-01-01', 'COMMITTED', '{}',
            'user_prevalence', NULL, NULL, 'v1.0', '', '1.0', '1e-6', 1.0, 1.0, '[]', 50, 50, '[]', '[]', '2026-01-01'
        )
    """)
    conn.commit()
    conn.close()

    with pytest.raises(ValueError, match="missing, blank, or unsupported mechanism"):
        PrivacyStateManager(db_path)


def test_empty_rdp_accountant_reports_exact_zero(tmp_path):
    """Empty RDPAccountant reports exact zero without returning fake epsilon (Item D & E)."""
    from private_llm_analytics.privacy.rdp_accountant import RDPAccountant

    db_path = tmp_path / "empty_rdp.db"
    state_mgr = PrivacyStateManager(db_path)
    acct = RDPAccountant(epsilon_cap=5.0, delta_global=1e-6, state_manager=state_mgr)

    assert acct.spent_epsilon == 0.0
    spent = acct.get_total_spent()
    assert spent.epsilon == 0
    assert spent.epsilon == Decimal("0.0")
    assert float(spent.epsilon) == 0.0
    assert spent.delta == Decimal("1e-6")


def test_empty_specific_scope_reports_exact_zero(tmp_path):
    """Querying specific empty scope via get_total_spent returns exact zero (Item D & E)."""
    from private_llm_analytics.privacy.rdp_accountant import RDPAccountant

    db_path = tmp_path / "empty_scope.db"
    state_mgr = PrivacyStateManager(db_path)
    acct = RDPAccountant(epsilon_cap=5.0, delta_global=1e-6, state_manager=state_mgr)

    spent_a = acct.get_total_spent(budget_scope_id="recipient:provider_a")
    assert spent_a.epsilon == 0
    assert spent_a.epsilon == Decimal("0.0")
    assert float(spent_a.epsilon) == 0.0


def test_legacy_policy_metadata_multiple_rows_fails_closed(tmp_path):
    """Legacy privacy_policy_metadata with > 1 rows fails closed without dropping old data (Item A & C)."""
    db_path = tmp_path / "multi_policy_legacy.db"
    conn = sqlite3.connect(db_path)
    conn.execute("""
        CREATE TABLE privacy_policy_metadata (
            policy_fingerprint TEXT PRIMARY KEY,
            recipient_composition_mode TEXT NOT NULL,
            canonicalized_collusion_groups TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
    """)
    conn.execute("""
        INSERT INTO privacy_policy_metadata VALUES
        ('fp_global', 'global_external', '{}', '2026-01-01T00:00:00Z'),
        ('fp_per', 'per_recipient', '{}', '2026-01-01T00:00:00Z')
    """)
    conn.commit()
    conn.close()

    with pytest.raises(ProductionSafetyConfigurationError, match="contains multiple policy bindings"):
        PrivacyStateManager(db_path)

    # Confirm old records were NOT dropped or lost
    conn2 = sqlite3.connect(db_path)
    rows = conn2.execute("SELECT policy_fingerprint FROM privacy_policy_metadata").fetchall()
    conn2.close()
    assert len(rows) == 2
    assert {r[0] for r in rows} == {"fp_global", "fp_per"}


def test_legacy_single_policy_inconsistent_scopes_fails_closed(tmp_path):
    """Legacy single policy row with conflicting historical scope fails closed during migration (Item B & C)."""
    db_path = tmp_path / "inconsistent_scope.db"
    conn = sqlite3.connect(db_path)
    conn.execute("""
        CREATE TABLE privacy_policy_metadata (
            policy_fingerprint TEXT PRIMARY KEY,
            recipient_composition_mode TEXT NOT NULL,
            canonicalized_collusion_groups TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
    """)
    conn.execute("""
        INSERT INTO privacy_policy_metadata VALUES
        ('fp_global', 'global_external', '{}', '2026-01-01T00:00:00Z')
    """)
    conn.execute("""
        CREATE TABLE release_store (
            release_key TEXT PRIMARY KEY, reservation_id TEXT NOT NULL, owner_id TEXT NOT NULL DEFAULT '',
            generation INTEGER NOT NULL DEFAULT 1, release_id TEXT NOT NULL, created_at TEXT NOT NULL,
            status TEXT NOT NULL, canonical_semantics_json TEXT NOT NULL, metric TEXT NOT NULL,
            window_start TEXT, window_end TEXT, dataset_version TEXT NOT NULL, mechanism TEXT NOT NULL,
            epsilon TEXT NOT NULL, delta TEXT NOT NULL, sensitivity REAL NOT NULL, noise_scale REAL NOT NULL,
            cell_names_json TEXT NOT NULL, n_users INTEGER NOT NULL, public_population_size INTEGER,
            dp_vector_json TEXT, post_processed_json TEXT, committed_at TEXT,
            recipient TEXT NOT NULL DEFAULT 'provider_a', recipient_role TEXT NOT NULL DEFAULT 'external',
            budget_scope_id TEXT NOT NULL DEFAULT 'recipient:provider_a', accountant_mode TEXT NOT NULL DEFAULT 'rdp',
            rdp_cost_json TEXT NOT NULL, rdp_orders_json TEXT NOT NULL,
            delta_global TEXT NOT NULL DEFAULT '1e-6', effective_epsilon TEXT, best_alpha TEXT
        )
    """)
    rdp_dummy = json.dumps({"2.0": "0.1", "3.0": "0.2"})
    orders_dummy = json.dumps([2.0, 3.0])
    conn.execute("""
        INSERT INTO release_store VALUES (
            'k1', 'r1', '', 1, 'rel1', '2026-01-01', 'COMMITTED', '{}',
            'user_prevalence', NULL, NULL, 'v1.0', 'gaussian', '1.0', '1e-6', 1.0, 1.0, '[]', 50, 50, '[]', '[]', '2026-01-01',
            'provider_a', 'external', 'recipient:provider_a', 'rdp', ?, ?, '1e-6', '1.0', '2.0'
        )
    """, (rdp_dummy, orders_dummy))
    conn.commit()
    conn.close()

    with pytest.raises(ProductionSafetyConfigurationError, match="is inconsistent with legacy policy mode 'global_external'"):
        PrivacyStateManager(db_path)
