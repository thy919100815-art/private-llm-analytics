"""Hardening and regression test suite for RDP policy configuration binding and composition semantics (v0.2).

Covers:
  - Requirement A: Authoritative RDP policy binding (accountant.epsilon_cap, accountant.delta_global, accountant.orders)
  - Requirement B: Deletion of invalid min(basic_sum, rdp_eff) across reserve, commit, and view
  - Requirement D: RDPAccountant and PrivacyStateManager fail-closed on corrupt/malformed state (no fail-open epsilon=0)
  - Requirement E: Legacy migration fail-closed when historical sensitivity is missing or <= 0 (no heuristic guessing)
  - Requirement F: Rejection of overlapping collusion groups in config and resolution
  - Requirement G: High-precision Decimal logarithm and adversarial boundary verification near epsilon_cap
"""

from decimal import Decimal
import json
from pathlib import Path
import sqlite3
import numpy as np
import pytest

from private_llm_analytics.aggregation.user_matrix import UserMatrix, build_cell_definitions
from private_llm_analytics.config.loader import load_app_config
from private_llm_analytics.config.models import RecipientCompositionConfig
from private_llm_analytics.planner.query import AnalyticsQuery
from private_llm_analytics.privacy.composition import resolve_budget_scope_id
from private_llm_analytics.privacy.exceptions import PrivacyBudgetExceededError
from private_llm_analytics.privacy.rdp import (
    DEFAULT_RDP_ORDERS,
    compute_gaussian_rdp,
    compose_rdp_vectors,
    rdp_to_epsilon_delta,
    rdp_vector_to_json,
    to_decimal,
)
from private_llm_analytics.privacy.rdp_accountant import RDPAccountant
from private_llm_analytics.privacy.state import PrivacyStateManager
from private_llm_analytics.release.release import ReleaseOrchestrator


@pytest.fixture
def sample_matrix():
    providers = ["provider_a", "provider_b", "provider_c"]
    tasks = ["technical", "document", "analysis"]
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


# =============================================================================
# Requirement A: Authoritative RDP Policy Binding
# =============================================================================

def test_rdp_policy_config_is_authoritative(tmp_path, sample_matrix):
    """Confirm that in RDP mode, accountant policy overrides global_budget."""
    config = load_app_config("configs")
    # Set huge budget in legacy global_budget, but tight cap in accountant
    config.privacy.global_budget.epsilon = 100.0
    config.privacy.global_budget.delta = 0.5
    config.privacy.accountant.epsilon_cap = 2.3
    config.privacy.accountant.delta_global = 1e-6
    config.privacy.accountant.orders = [2.0, 4.0, 8.0]

    state_mgr = PrivacyStateManager(tmp_path / "test_auth.db")
    orch = ReleaseOrchestrator(app_config=config, state_manager=state_mgr)

    # Verify RDPAccountant construction matches accountant policy
    assert orch.rdp_accountant.epsilon_cap == Decimal("2.3")
    assert orch.rdp_accountant.delta_global == Decimal("1e-6")
    assert orch.rdp_accountant.orders == [Decimal("2.0"), Decimal("4.0"), Decimal("8.0")]

    # Query 1: requested eps=1.0 consumes ~2.20 RDP epsilon with orders [2, 4, 8] -> succeeds (2.20 <= 2.3)
    q1 = AnalyticsQuery(metric="user_prevalence", recipient="provider_a", epsilon=1.0, delta=1e-6)
    r1 = orch.execute_release(q1, sample_matrix)
    assert not r1["cache_hit"]

    # Query 2: second query pushes RDP effective eps to 2.42 > 2.3 cap -> MUST RAISE
    # (If it erroneously used global_budget.epsilon=100.0, this would have passed!)
    q2 = AnalyticsQuery(metric="user_normalized_share", recipient="provider_a", epsilon=1.0, delta=1e-6)
    with pytest.raises(PrivacyBudgetExceededError):
        orch.execute_release(q2, sample_matrix)


def test_custom_rdp_orders_are_used(tmp_path, sample_matrix):
    """Confirm accountant.orders = [2, 4] is strictly respected, not DEFAULT_RDP_ORDERS."""
    config = load_app_config("configs")
    config.privacy.accountant.orders = [2.0, 4.0]
    config.privacy.accountant.epsilon_cap = 10.0
    config.privacy.accountant.delta_global = 1e-6

    state_mgr = PrivacyStateManager(tmp_path / "test_orders.db")
    orch = ReleaseOrchestrator(app_config=config, state_manager=state_mgr)

    q = AnalyticsQuery(metric="user_prevalence", recipient="provider_a", epsilon=1.0, delta=1e-6)
    orch.execute_release(q, sample_matrix)

    view = orch.get_privacy_view("provider_a")
    assert view.orders == [Decimal("2.0"), Decimal("4.0")]
    assert len(view.orders) == 2
    assert set(view.total_rdp_vector.keys()) == {Decimal("2.0"), Decimal("4.0")}
    assert view.best_alpha in (Decimal("2.0"), Decimal("4.0"))


def test_rdp_uses_accountant_delta_global(tmp_path, sample_matrix):
    """Specific regression test: global_budget.delta = 4e-6, accountant.delta_global = 1e-6.

    Confirm Production RDP:
      - authorization uses 1e-6
      - PrivacyView uses 1e-6
      - response metadata displays 1e-6
    """
    config = load_app_config("configs")
    config.privacy.global_budget.delta = 4e-6
    config.privacy.accountant.delta_global = 1e-6
    config.privacy.accountant.epsilon_cap = 5.0

    state_mgr = PrivacyStateManager(tmp_path / "test_delta.db")
    orch = ReleaseOrchestrator(app_config=config, state_manager=state_mgr)

    q = AnalyticsQuery(metric="user_prevalence", recipient="provider_a", epsilon=1.0, delta=1e-6)
    res = orch.execute_release(q, sample_matrix)

    # 1. Response metadata displays accountant.delta_global (1e-6), NOT global_budget.delta (4e-6)
    assert res["privacy_accounting"]["delta_global"] == 1e-6

    # 2. PrivacyView reflects 1e-6
    view = orch.get_privacy_view("provider_a")
    assert view.delta_global == Decimal("1e-6")

    # 3. Database stored delta_global is 1e-6
    with sqlite3.connect(tmp_path / "test_delta.db") as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT delta_global FROM release_store WHERE release_key = ?", (res["release_key"],))
        stored_dg = cursor.fetchone()[0]
        assert stored_dg == "0.000001" or float(stored_dg) == 1e-6


# =============================================================================
# Requirement B: No Invalid min(basic_sum, rdp_eff)
# =============================================================================

def test_no_invalid_basic_rdp_min(tmp_path, sample_matrix):
    """Confirm that RDP mode never takes min(basic_sum, rdp_eff).

    At T=1 for user_prevalence (Delta=3.0, eps=1.0, delta=1e-6),
    analytic Gaussian RDP conversion at delta_global=1e-6 evaluates to ~1.3421.
    If min(basic_sum, rdp_eff) were present, it would be illegally clamped to 1.0.
    """
    config = load_app_config("configs")
    config.privacy.accountant.epsilon_cap = 5.0
    config.privacy.accountant.delta_global = 1e-6

    state_mgr = PrivacyStateManager(tmp_path / "test_no_min.db")
    orch = ReleaseOrchestrator(app_config=config, state_manager=state_mgr)

    q = AnalyticsQuery(metric="user_prevalence", recipient="provider_a", epsilon=1.0, delta=1e-6)
    res = orch.execute_release(q, sample_matrix)

    view = orch.get_privacy_view("provider_a")
    # Must NOT be clamped to requested 1.0
    assert view.effective_epsilon > Decimal("1.0")
    # Response metadata effective_epsilon must match exact RDP conversion (~1.34)
    assert res["privacy_accounting"]["effective_epsilon"] > 1.0
    assert abs(view.effective_epsilon - Decimal(str(res["privacy_accounting"]["effective_epsilon"]))) < Decimal("1e-9")


# =============================================================================
# Requirement D: RDPAccountant and StateManager Fail Closed
# =============================================================================

def test_rdp_accountant_fail_closed(tmp_path):
    """Malformed or corrupt RDP state must raise, NEVER return epsilon=0."""
    db_path = tmp_path / "corrupt.db"
    state_mgr = PrivacyStateManager(db_path)

    # Insert corrupted release
    with sqlite3.connect(db_path) as conn:
        conn.execute("""
            INSERT INTO release_store (
                release_key, reservation_id, owner_id, generation, release_id,
                created_at, status, canonical_semantics_json,
                metric, dataset_version, recipient, epsilon, delta, mechanism,
                sensitivity, noise_scale, cell_names_json, n_users,
                budget_scope_id, accountant_mode, rdp_cost_json
            ) VALUES (
                'key_bad', 'res_bad', 'owner_1', 1, 'rel_bad',
                '2026-01-01T00:00:00Z', 'COMMITTED', '{}',
                'user_prevalence', 'v1.0', 'provider_a', '1.0', '1e-6', 'gaussian',
                3.0, 10.0, '[]', 10,
                'global_external', 'rdp', '{MALFORMED_JSON'
            )
        """)
        conn.commit()

    accountant = RDPAccountant(state_manager=state_mgr)

    # get_total_spent MUST raise, not return epsilon=0
    with pytest.raises(Exception):
        accountant.get_total_spent()

    # get_privacy_view MUST raise
    with pytest.raises(Exception):
        state_mgr.get_privacy_view(budget_scope_id="global_external", delta_global=1e-6)


# =============================================================================
# Requirement E: Legacy Missing Sensitivity Fails Closed
# =============================================================================

def test_legacy_missing_sensitivity_fails_closed(tmp_path):
    """Historical Gaussian row with sensitivity = 0 must fail migration."""
    db_path = tmp_path / "legacy_invalid_sens.db"

    with sqlite3.connect(db_path) as conn:
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
                recipient TEXT NOT NULL,
                epsilon REAL NOT NULL,
                delta REAL NOT NULL,
                sensitivity REAL NOT NULL,
                mechanism TEXT NOT NULL,
                noise_scale REAL NOT NULL,
                result_json TEXT NOT NULL
            )
        """)
        conn.execute("""
            CREATE TABLE privacy_reservations (
                reservation_id TEXT PRIMARY KEY,
                release_key TEXT NOT NULL,
                owner_id TEXT NOT NULL DEFAULT '',
                generation INTEGER NOT NULL DEFAULT 1,
                epsilon REAL NOT NULL,
                delta REAL NOT NULL,
                status TEXT NOT NULL,
                created_at TEXT NOT NULL,
                committed_at TEXT
            )
        """)
        conn.execute("""
            CREATE TABLE privacy_ledger (
                entry_id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                event_type TEXT NOT NULL,
                reservation_id TEXT,
                release_key TEXT NOT NULL,
                recipient TEXT NOT NULL,
                dataset_version TEXT NOT NULL,
                metric TEXT NOT NULL,
                mechanism TEXT NOT NULL,
                epsilon REAL NOT NULL,
                delta REAL NOT NULL,
                sensitivity REAL NOT NULL,
                noise_scale REAL NOT NULL,
                cache_hit INTEGER NOT NULL,
                window_start TEXT,
                window_end TEXT,
                caller_info TEXT
            )
        """)
        # Insert with 0.0 sensitivity (invalid for Gaussian RDP)
        conn.execute("""
            INSERT INTO release_store VALUES (
                'key_1', 'res_1', 'owner_1', 1, 'rel_1', '2026-01-01T00:00:00Z',
                'COMMITTED', '{}', 'user_prevalence', 'provider_a', 0.1, 0.00001,
                0.0, 'gaussian', 1.0, '{}'
            )
        """)
        conn.commit()

    # Initialization triggers migration -> MUST fail closed because sensitivity is <= 0
    with pytest.raises(ValueError, match=r"missing valid sensitivity/sigma"):
        PrivacyStateManager(db_path=db_path)


# =============================================================================
# Requirement F: Collusion Group Overlap Rejected
# =============================================================================

def test_collusion_group_overlap_rejected():
    """Overlapping collusion groups must be rejected by Pydantic model and scope resolver."""
    overlapping_groups = {
        "group_1": ["provider_a", "provider_b"],
        "group_2": ["provider_a", "provider_c"],
    }

    # 1. Pydantic configuration validation rejects
    with pytest.raises(ValueError, match=r"Collusion group overlap detected"):
        RecipientCompositionConfig(mode="collusion_group", groups=overlapping_groups)

    # 2. Direct scope resolution rejects without picking first dict item
    with pytest.raises(ValueError, match=r"Collusion group overlap detected"):
        resolve_budget_scope_id("provider_a", mode="collusion_group", groups=overlapping_groups)


# =============================================================================
# Requirement G: Decimal RDP Boundary Precision
# =============================================================================

def test_decimal_rdp_boundary():
    """Adversarial boundary test near epsilon_cap verifying Decimal precision >= 50 digits."""
    # Test ln(1/delta) with extreme delta=1e-15
    delta = Decimal("1e-15")
    # Verify exact Decimal conversion without binary float intermediate
    rdp_vec = {Decimal("2.0"): Decimal("1.5000000000000000000000000000000000000000000000000000")}
    best_a, eff_eps = rdp_to_epsilon_delta(rdp_vec, delta_global=delta)

    assert best_a == Decimal("2.0")
    # ln(1e15) = 15 * ln(10) ~= 34.538776394910684...
    # epsilon(2) = 1.5 + 34.538776394910684 / (2-1) = 36.038776394910684...
    assert eff_eps > Decimal("36.0")

    # Boundary check: cap is within 10^-50 of value
    exact_cap = eff_eps
    under_cap = eff_eps - Decimal("1e-45")
    over_cap = eff_eps + Decimal("1e-45")

    assert eff_eps <= exact_cap
    assert not (eff_eps <= under_cap)
    assert eff_eps <= over_cap
