"""Verification script: Same Semantic Query Zero Cost & Memoization (v0.2).

Verifies that:
  1. The initial query execution generates noise, commits the release,
     and consumes privacy budget.
  2. Exact replay of the same semantic query returns cache_hit=True,
     exact identical outputs, and strictly 0 incremental privacy cost.
  3. No redundant historical ledger entries or noise generation occur.
  4. Query with modified semantics (e.g., metric, time window) is recognized
     as a distinct release and incurs fresh privacy accounting cost.
"""

from decimal import Decimal
from pathlib import Path
import sys
import tempfile
import numpy as np

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from private_llm_analytics.aggregation.user_matrix import UserMatrix, build_cell_definitions
from private_llm_analytics.config.loader import load_app_config
from private_llm_analytics.planner.query import AnalyticsQuery
from private_llm_analytics.privacy.state import PrivacyStateManager
from private_llm_analytics.release.release import ReleaseOrchestrator


def build_test_matrix() -> UserMatrix:
    providers = ["provider_a", "provider_b", "provider_c"]
    tasks = ["technical", "document", "analysis"]
    cell_names, cell_map = build_cell_definitions(providers, tasks)
    np_matrix = np.ones((50, 9), dtype=np.float64) * 3.0
    return UserMatrix(
        matrix=np_matrix,
        user_ids=[f"user_{i}" for i in range(50)],
        providers=providers,
        tasks=tasks,
        cell_names=cell_names,
        cell_map=cell_map,
        public_population_size=50000,
    )


def main():
    print("=" * 70)
    print("VERIFICATION: Same Semantic Query Replay Zero Incremental Cost")
    print("=" * 70)

    with tempfile.TemporaryDirectory() as tmp_dir:
        db_path = Path(tmp_dir) / "privacy_state.db"
        config = load_app_config("configs")
        config.privacy.global_budget.epsilon = 4.0
        config.privacy.global_budget.delta = 1e-5
        config.privacy.accountant.epsilon_cap = 4.0
        config.privacy.accountant.delta_global = 1e-5

        state_mgr = PrivacyStateManager(db_path)
        orch = ReleaseOrchestrator(app_config=config, state_manager=state_mgr)
        matrix = build_test_matrix()

        # Step 1: Initial Query
        print("[1] Executing initial DP release...")
        q1 = AnalyticsQuery(
            metric="user_prevalence",
            recipient="provider_a",
            epsilon=1.0,
            delta=1e-6,
            window_start="2026-01-01",
            window_end="2026-01-31",
        )
        res1 = orch.execute_release(q1, matrix)
        assert res1["cache_hit"] is False, "Initial query should not be a cache hit"
        assert res1["incremental_privacy_cost"] > 0, "Initial query must incur positive epsilon cost"

        initial_spent_eps, initial_spent_delta = state_mgr.get_total_spent()
        initial_releases_count = len(state_mgr.list_releases())
        print(f"    Initial spend: eps={initial_spent_eps}, delta={initial_spent_delta}")
        print(f"    Committed releases: {initial_releases_count}")

        # Step 2: Replay identical query
        print("[2] Replaying identical semantic query...")
        res2 = orch.execute_release(q1, matrix)
        assert res2["cache_hit"] is True, "Replayed query must return cache_hit=True"
        assert res2["incremental_privacy_cost"] == 0.0, "Replayed query must have 0 incremental epsilon"
        assert res2["incremental_privacy_cost_decimal"] == "0.0", "Replayed query must have '0.0' incremental Decimal"

        # Verify bitwise identical outputs
        assert res1["results"]["prevalence_rates"] == res2["results"]["prevalence_rates"], (
            "Replayed release outputs must be identical"
        )

        after_replay_eps, after_replay_delta = state_mgr.get_total_spent()
        after_replay_count = len(state_mgr.list_releases())
        assert after_replay_eps == initial_spent_eps, "Committed epsilon budget must not increase on replay"
        assert after_replay_delta == initial_spent_delta, "Committed delta budget must not increase on replay"
        assert after_replay_count == initial_releases_count, "No new release store entry should be created on replay"
        print("    [PASS] Replay incurred strictly 0 incremental privacy budget.")

        # Step 3: Distinct query with changed time window
        print("[3] Executing query with modified time window...")
        q3 = AnalyticsQuery(
            metric="user_prevalence",
            recipient="provider_a",
            epsilon=1.0,
            delta=1e-6,
            window_start="2026-02-01",
            window_end="2026-02-28",
        )
        res3 = orch.execute_release(q3, matrix)
        assert res3["cache_hit"] is False, "Distinct window query must not hit cache"
        assert res3["incremental_privacy_cost"] > 0, "Distinct window query must incur positive epsilon"

        after_q3_eps, after_q3_delta = state_mgr.get_total_spent()
        after_q3_count = len(state_mgr.list_releases())
        assert after_q3_eps > initial_spent_eps, "Committed spend must increase for distinct release"
        assert after_q3_count == initial_releases_count + 1, "Distinct release must create 1 new committed release"
        print(f"    Spend after distinct release: eps={after_q3_eps:.4f}")
        print("    [PASS] Semantically distinct query properly incurred new privacy cost.")

    print("\n[VERIFICATION PASSED] Same semantic query zero-cost guarantee is fully verified.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
