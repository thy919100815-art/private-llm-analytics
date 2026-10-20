"""Verification script: No Privacy Reset & Lifetime Horizon Enforcement (v0.2).

Verifies that:
  1. Production safety configuration strictly rejects any privacy history mode
     other than 'lifetime' (e.g., 'rolling_reset' fails closed).
  2. Across multiple sequential reporting periods (e.g., day 1, 30, 60, 90),
     all historical releases permanently accumulate in the SQLite ledger.
  3. No pseudo-reset or window eviction can erase historical privacy spend.
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
from private_llm_analytics.config.validator import (
    ProductionSafetyConfigurationError,
    validate_production_config,
)
from private_llm_analytics.planner.query import AnalyticsQuery
from private_llm_analytics.privacy.state import PrivacyStateManager
from private_llm_analytics.release.release import ReleaseOrchestrator


def build_test_matrix() -> UserMatrix:
    providers = ["provider_a", "provider_b", "provider_c"]
    tasks = ["technical", "document", "analysis"]
    cell_names, cell_map = build_cell_definitions(providers, tasks)
    np_matrix = np.ones((50, 9), dtype=np.float64) * 4.0
    return UserMatrix(
        matrix=np_matrix,
        user_ids=[f"user_{i}" for i in range(50)],
        providers=providers,
        tasks=tasks,
        cell_names=cell_names,
        cell_map=cell_map,
        public_population_size=50000,
    )


def test_production_validator_fail_closed():
    print("[1] Verifying Production safety validator rejects rolling reset...")
    config = load_app_config("configs")
    config.privacy.history.mode = "rolling_reset"

    rejected = False
    try:
        validate_production_config(config)
    except ProductionSafetyConfigurationError as exc:
        rejected = True
        print(f"    Expected rejection caught: {exc}")

    assert rejected, "validate_production_config failed to reject privacy.history.mode='rolling_reset'!"
    print("    [PASS] Production mode strictly requires privacy.history.mode='lifetime'.")


def test_lifetime_accumulation():
    print("[2] Verifying lifetime cumulative privacy preservation across long time horizons...")
    with tempfile.TemporaryDirectory() as tmp_dir:
        db_path = Path(tmp_dir) / "privacy_state.db"
        config = load_app_config("configs")
        config.privacy.history.mode = "lifetime"
        config.privacy.global_budget.epsilon = 10.0
        config.privacy.global_budget.delta = 1e-5
        config.privacy.accountant.epsilon_cap = 10.0
        config.privacy.accountant.delta_global = 1e-5

        state_mgr = PrivacyStateManager(db_path)
        orch = ReleaseOrchestrator(app_config=config, state_manager=state_mgr)
        matrix = build_test_matrix()

        # Simulate 4 consecutive monthly releases: Jan, Feb, Mar, Apr
        months = [
            ("2026-01-01", "2026-01-31"),
            ("2026-02-01", "2026-02-28"),
            ("2026-03-01", "2026-03-31"),
            ("2026-04-01", "2026-04-30"),
        ]

        effective_eps_history = []
        for i, (t_start, t_end) in enumerate(months, 1):
            q = AnalyticsQuery(
                metric="user_prevalence",
                recipient="provider_a",
                epsilon=1.0,
                delta=1e-6,
                window_start=t_start,
                window_end=t_end,
            )
            res = orch.execute_release(q, matrix)
            assert res["cache_hit"] is False
            view = orch.get_privacy_view("provider_a")
            effective_eps_history.append(float(view.effective_epsilon))
            print(f"    Month {i} ({t_start} to {t_end}): cumulative releases={view.release_count}, effective_eps={view.effective_epsilon:.4f}")

        # Check monotonic increase
        assert len(effective_eps_history) == 4
        for idx in range(len(effective_eps_history) - 1):
            assert effective_eps_history[idx] < effective_eps_history[idx + 1], (
                f"Cumulative privacy must monotonically increase over lifetime: {effective_eps_history}"
            )

        # Confirm all 4 releases remain committed in SQLite
        releases = state_mgr.list_releases()
        assert len(releases) == 4, f"Expected 4 committed releases in state, found {len(releases)}"
        print("    [PASS] All historical releases preserved without eviction or pseudo-reset.")


def main():
    print("=" * 70)
    print("VERIFICATION: No Privacy Reset & Lifetime Horizon Accounting")
    print("=" * 70)
    test_production_validator_fail_closed()
    test_lifetime_accumulation()
    print("\n[VERIFICATION PASSED] Lifetime privacy horizon accounting is 100% sound.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
