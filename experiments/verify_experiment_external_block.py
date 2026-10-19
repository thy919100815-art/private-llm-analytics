"""Verification script: Low-Level ReleaseOrchestrator Experiment Mode External Recipient Block.

Verifies:
  1. ReleaseOrchestrator(ExecutionMode.EXPERIMENT) strictly blocks external recipients
     (provider_a, provider_b, provider_c, and any role='external') with RecipientSecurityViolation.
  2. Block occurs immediately before plan creation, data access, reservation, noise, or commit.
  3. Zero accounting side-effects: 10 repeated blocked external attempts leave spent budget at 0,
     reservation count at 0, release count at 0, and ledger count at 0.
  4. Internal experiment releases (recipient='internal') succeed under EXPERIMENT mode.
  5. Production releases to external recipients (e.g. provider_a) succeed under PRODUCTION mode.
"""

from decimal import Decimal
import gc
from pathlib import Path
import sys
import pandas as pd

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from private_llm_analytics.aggregation.user_matrix import build_user_matrix
from private_llm_analytics.config.loader import load_app_config
from private_llm_analytics.config.validator import ExecutionMode, ExperimentAccountingMode
from private_llm_analytics.privacy.exceptions import RecipientSecurityViolation
from private_llm_analytics.privacy.state import PrivacyStateManager
from private_llm_analytics.planner.query import AnalyticsQuery
from private_llm_analytics.release.release import ReleaseOrchestrator


def cleanup_db(db_path: Path):
    gc.collect()
    for ext in ["", "-wal", "-shm", ".jsonl"]:
        f = Path(str(db_path) + ext)
        try:
            if f.exists():
                f.unlink(missing_ok=True)
        except PermissionError:
            pass


def main():
    print("=" * 70)
    print("VERIFICATION: Low-Level Experiment External Recipient Block (Gate Fix A)")
    print("=" * 70)

    exp_db = Path("results/verify_ext_block_exp.db")
    prod_db = Path("results/verify_ext_block_prod.db")

    cleanup_db(exp_db)
    cleanup_db(prod_db)

    # 1. Setup config
    config = load_app_config("configs")
    config.privacy.release.state_db_path = str(prod_db)
    config.privacy.global_budget.epsilon = 10.0
    config.privacy.global_budget.delta = 1e-4

    # Sample dataset
    df = pd.DataFrame({
        "interaction_id": [f"ix_{i}" for i in range(20)],
        "user_id": [f"user_{i % 5}" for i in range(20)],
        "timestamp": ["2026-01-01T00:00:00Z"] * 20,
        "provider": ["provider_a"] * 10 + ["provider_b"] * 10,
        "task": ["technical"] * 10 + ["analysis"] * 10,
    })
    umatrix = build_user_matrix(df, public_population_size=100)

    # 2. Directly construct low-level ReleaseOrchestrator in EXPERIMENT mode
    exp_state = PrivacyStateManager(db_path=exp_db)
    prod_state = PrivacyStateManager(db_path=prod_db)

    orchestrator = ReleaseOrchestrator(
        app_config=config,
        execution_mode=ExecutionMode.EXPERIMENT,
        state_manager=exp_state,
        accounting_mode=ExperimentAccountingMode.EVALUATION,
    )

    # 3. Test provider_a, provider_b, provider_c
    external_providers = ["provider_a", "provider_b", "provider_c"]
    for rec in external_providers:
        query = AnalyticsQuery(
            metric="user_prevalence",
            recipient=rec,
            epsilon=1.0,
            delta=1e-5,
        )
        blocked = False
        try:
            orchestrator.execute_release(
                query=query,
                user_matrix=umatrix,
                dataset_version="test_v1",
            )
        except RecipientSecurityViolation:
            blocked = True

        assert blocked, f"Expected RecipientSecurityViolation for {rec} in EXPERIMENT mode!"
        print(f"{rec}: BLOCKED")

    # 4. Attempt 10 repeated releases for provider_a
    for i in range(10):
        query = AnalyticsQuery(
            metric="user_prevalence",
            recipient="provider_a",
            epsilon=1.0,
            delta=1e-5,
        )
        blocked = False
        try:
            orchestrator.execute_release(
                query=query,
                user_matrix=umatrix,
                dataset_version="test_v1",
            )
        except RecipientSecurityViolation:
            blocked = True
        assert blocked, f"Attempt {i+1} was not blocked!"

    # 5. Verify zero accounting side effects
    exp_spent_eps, exp_spent_del = exp_state.get_total_spent_decimal()
    prod_spent_eps, prod_spent_del = prod_state.get_total_spent_decimal()

    assert exp_spent_eps == Decimal("0"), f"Expected 0 experiment eps, got {exp_spent_eps}"
    assert prod_spent_eps == Decimal("0"), f"Expected 0 production eps, got {prod_spent_eps}"

    import sqlite3
    with sqlite3.connect(exp_db) as conn:
        res_count = conn.execute("SELECT COUNT(*) FROM privacy_reservations").fetchone()[0]
        rel_count = conn.execute("SELECT COUNT(*) FROM release_store").fetchone()[0]
        led_count = conn.execute("SELECT COUNT(*) FROM privacy_ledger").fetchone()[0]
    assert res_count == 0, f"Expected 0 reservations, found {res_count}"
    assert rel_count == 0, f"Expected 0 releases, found {rel_count}"
    assert led_count == 0, f"Expected 0 ledger rows, found {led_count}"

    print(f"experiment budget after blocked attempts: {exp_spent_eps}")
    print(f"production budget after blocked attempts: {prod_spent_eps}")

    # 6. Verify internal experiment release succeeds
    internal_query = AnalyticsQuery(
        metric="user_prevalence",
        recipient="internal",
        epsilon=1.0,
        delta=1e-5,
    )
    res_int = orchestrator.execute_release(
        query=internal_query,
        user_matrix=umatrix,
        dataset_version="test_v1",
        noise_seed=12345,
    )
    assert res_int is not None
    assert "results" in res_int
    print("internal experiment: PASS")

    # 7. Verify production external release succeeds
    prod_orchestrator = ReleaseOrchestrator(
        app_config=config,
        execution_mode=ExecutionMode.PRODUCTION,
        state_manager=prod_state,
    )
    prod_query = AnalyticsQuery(
        metric="user_prevalence",
        recipient="provider_a",
        epsilon=1.0,
        delta=1e-5,
    )
    res_prod = prod_orchestrator.execute_release(
        query=prod_query,
        user_matrix=umatrix,
        dataset_version="test_v1",
    )
    assert res_prod is not None
    assert "results" in res_prod
    print("production provider_a: PASS")

    # Clean up test DBs
    cleanup_db(exp_db)
    cleanup_db(prod_db)

    print("PASS")


if __name__ == "__main__":
    main()
