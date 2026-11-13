"""Verification script: Multi-Recipient Privacy Accounting & Collusion Policies (v0.2).

Verifies that:
  1. GLOBAL_EXTERNAL: All external providers (provider_a, provider_b, provider_c)
     share a unified budget scope ('global_external'). Switching providers cannot
     evade the global privacy cap.
  2. COLLUSION_GROUP: Specified groups (e.g., group_ab = [provider_a, provider_b])
     share a collective privacy scope ('collusion:group_ab'), while independent groups
     (group_c = [provider_c]) maintain separate accounting.
  3. PER_RECIPIENT: When strictly non-colluding, each recipient has an isolated
     accounting scope ('recipient:<name>').
  4. Privacy views can be accurately queried and reconstructed per recipient.
"""

from decimal import Decimal
import os
from pathlib import Path
import sys
import tempfile
import numpy as np

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from private_llm_analytics.aggregation.user_matrix import UserMatrix, build_cell_definitions
from private_llm_analytics.config.loader import load_app_config
from private_llm_analytics.planner.query import AnalyticsQuery
from private_llm_analytics.privacy.exceptions import PrivacyBudgetExceededError
from private_llm_analytics.privacy.state import PrivacyStateManager
from private_llm_analytics.release.release import ReleaseOrchestrator


def build_test_matrix() -> UserMatrix:
    providers = ["provider_a", "provider_b", "provider_c"]
    tasks = ["technical", "document", "analysis"]
    cell_names, cell_map = build_cell_definitions(providers, tasks)
    np_matrix = np.ones((50, 9), dtype=np.float64) * 5.0
    return UserMatrix(
        matrix=np_matrix,
        user_ids=[f"user_{i}" for i in range(50)],
        providers=providers,
        tasks=tasks,
        cell_names=cell_names,
        cell_map=cell_map,
        public_population_size=50000,
    )


def test_global_external():
    print("[1] Verifying GLOBAL_EXTERNAL collusion policy...")
    with tempfile.TemporaryDirectory() as tmp_dir:
        db_path = Path(tmp_dir) / "privacy_state.db"
        config = load_app_config("configs")
        config.privacy.recipient_composition.mode = "global_external"
        # Set a cap that allows roughly one 1.0 epsilon release, but not two
        config.privacy.global_budget.epsilon = 1.5
        config.privacy.global_budget.delta = 1e-5
        config.privacy.accountant.epsilon_cap = 1.5
        config.privacy.accountant.delta_global = 1e-5

        state_mgr = PrivacyStateManager(db_path)
        orch = ReleaseOrchestrator(app_config=config, state_manager=state_mgr)
        matrix = build_test_matrix()

        # Query 1 from provider_a consumes 1.0 epsilon
        q1 = AnalyticsQuery(metric="user_prevalence", recipient="provider_a", epsilon=1.0, delta=1e-6)
        r1 = orch.execute_release(q1, matrix)
        assert r1["privacy_accounting"]["budget_scope_id"] == "global_external"
        assert not r1["cache_hit"]

        # Query 2 from provider_b (different metric) attempts 1.0 epsilon
        # Under global_external, total spend would be 2.0 > 1.5, so it MUST be rejected
        q2 = AnalyticsQuery(metric="user_normalized_share", recipient="provider_b", epsilon=1.0, delta=1e-6)
        rejected = False
        try:
            orch.execute_release(q2, matrix)
        except PrivacyBudgetExceededError:
            rejected = True
        assert rejected, "GLOBAL_EXTERNAL failed to reject provider_b after provider_a consumed budget!"

        view = orch.get_privacy_view("provider_b")
        assert view.budget_scope_id == "global_external"
        assert view.release_count == 1
        print("    [PASS] GLOBAL_EXTERNAL correctly blocked provider switching bypass.")


def test_collusion_group():
    print("[2] Verifying COLLUSION_GROUP policy...")
    with tempfile.TemporaryDirectory() as tmp_dir:
        db_path = Path(tmp_dir) / "privacy_state.db"
        config = load_app_config("configs")
        config.privacy.recipient_composition.mode = "collusion_group"
        config.privacy.recipient_composition.groups = {
            "group_ab": ["provider_a", "provider_b"],
            "group_c": ["provider_c"],
        }
        config.privacy.global_budget.epsilon = 1.5
        config.privacy.global_budget.delta = 1e-5
        config.privacy.accountant.epsilon_cap = 1.5
        config.privacy.accountant.delta_global = 1e-5

        state_mgr = PrivacyStateManager(db_path)
        orch = ReleaseOrchestrator(app_config=config, state_manager=state_mgr)
        matrix = build_test_matrix()

        # Provider A (group_ab) executes release 1 (1.0 eps)
        q1 = AnalyticsQuery(metric="user_prevalence", recipient="provider_a", epsilon=1.0, delta=1e-6)
        r1 = orch.execute_release(q1, matrix)
        assert r1["privacy_accounting"]["budget_scope_id"] == "collusion:group_ab"

        # Provider B (group_ab) attempts release 2 (1.0 eps) -> Must be rejected (shared cap 1.5)
        q2 = AnalyticsQuery(metric="user_normalized_share", recipient="provider_b", epsilon=1.0, delta=1e-6)
        rejected = False
        try:
            orch.execute_release(q2, matrix)
        except PrivacyBudgetExceededError:
            rejected = True
        assert rejected, "COLLUSION_GROUP failed to reject provider_b in group_ab!"

        # Provider C (group_c) is independent and can execute release 3 (1.0 eps)
        q3 = AnalyticsQuery(metric="interaction_volume", recipient="provider_c", epsilon=1.0, delta=1e-6)
        r3 = orch.execute_release(q3, matrix)
        assert r3["privacy_accounting"]["budget_scope_id"] == "collusion:group_c"
        assert not r3["cache_hit"]

        view_ab = orch.get_privacy_view("provider_a")
        view_c = orch.get_privacy_view("provider_c")
        assert view_ab.budget_scope_id == "collusion:group_ab"
        assert view_ab.release_count == 1
        assert view_c.budget_scope_id == "collusion:group_c"
        assert view_c.release_count == 1
        print("    [PASS] COLLUSION_GROUP properly grouped A+B and isolated C.")


def test_per_recipient():
    print("[3] Verifying PER_RECIPIENT policy...")
    with tempfile.TemporaryDirectory() as tmp_dir:
        db_path = Path(tmp_dir) / "privacy_state.db"
        config = load_app_config("configs")
        config.privacy.recipient_composition.mode = "per_recipient"
        config.privacy.global_budget.epsilon = 1.5
        config.privacy.global_budget.delta = 1e-5
        config.privacy.accountant.epsilon_cap = 1.5
        config.privacy.accountant.delta_global = 1e-5

        state_mgr = PrivacyStateManager(db_path)
        orch = ReleaseOrchestrator(app_config=config, state_manager=state_mgr)
        matrix = build_test_matrix()

        # Provider A executes release (1.0 eps)
        qa = AnalyticsQuery(metric="user_prevalence", recipient="provider_a", epsilon=1.0, delta=1e-6)
        ra = orch.execute_release(qa, matrix)
        assert ra["privacy_accounting"]["budget_scope_id"] == "recipient:provider_a"

        # Provider B executes release (1.0 eps) - isolated scope, succeeds
        qb = AnalyticsQuery(metric="user_normalized_share", recipient="provider_b", epsilon=1.0, delta=1e-6)
        rb = orch.execute_release(qb, matrix)
        assert rb["privacy_accounting"]["budget_scope_id"] == "recipient:provider_b"
        assert not rb["cache_hit"]

        view_a = orch.get_privacy_view("provider_a")
        view_b = orch.get_privacy_view("provider_b")
        assert view_a.budget_scope_id == "recipient:provider_a"
        assert view_a.release_count == 1
        assert view_b.budget_scope_id == "recipient:provider_b"
        assert view_b.release_count == 1
        print("    [PASS] PER_RECIPIENT properly maintains independent views.")


def main():
    print("=" * 70)
    print("VERIFICATION: Multi-Recipient Longitudinal Privacy Accounting")
    print("=" * 70)
    test_global_external()
    test_collusion_group()
    test_per_recipient()
    print("\n[VERIFICATION PASSED] All multi-recipient privacy accounting checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
