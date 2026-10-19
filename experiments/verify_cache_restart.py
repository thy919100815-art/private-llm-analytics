"""Verification script demonstrating durable repeated-query cache and accountant spend restoration across process restarts."""

from pathlib import Path
import sqlite3
import pandas as pd
import numpy as np

from private_llm_analytics.aggregation.user_matrix import build_user_matrix
from private_llm_analytics.config.loader import load_app_config
from private_llm_analytics.planner.query import AnalyticsQuery
from private_llm_analytics.privacy.accountant import BasicAccountant
from private_llm_analytics.privacy.budget import PrivacyBudget
from private_llm_analytics.privacy.ledger import PrivacyLedger
from private_llm_analytics.release.cache import PersistentReleaseStore
from private_llm_analytics.release.release import ReleaseOrchestrator


def run_cache_restart_verification():
    print("=================================================================")
    print("Starting Persistent Release Store & Accountant Restart Verification")
    print("=================================================================")

    # Temporary isolated test databases for demonstration
    test_cache_db = Path("results/test_restart_cache.db")
    test_ledger_db = Path("results/test_restart_ledger.db")

    if test_cache_db.exists():
        test_cache_db.unlink()
    if test_ledger_db.exists():
        test_ledger_db.unlink()

    config = load_app_config("configs")

    # Load 500-user sample dataset for instant speed
    df = pd.read_parquet("data/interactions_n500.parquet")
    umatrix = build_user_matrix(df, public_population_size=50000)

    print(f"\n[PHASE 1] Initial Process (Process A):")
    cache_a = PersistentReleaseStore(test_cache_db)
    ledger_a = PrivacyLedger(test_ledger_db)
    accountant_a = BasicAccountant(
        global_budget=PrivacyBudget(epsilon=4.0, delta=1e-5),
        ledger=ledger_a,
    )
    orch_a = ReleaseOrchestrator(
        app_config=config,
        cache=cache_a,
        ledger=ledger_a,
        accountant=accountant_a,
    )

    query = AnalyticsQuery(
        metric="user_normalized_share",
        recipient="internal",
        epsilon=1.0,
        delta=1e-6,
    )

    print("  Executing Query 1 on Process A...")
    res_a = orch_a.execute_release(query, umatrix, dataset_version="v1.0")
    shares_a = res_a["results"]["cell_shares"]
    print(f"  Query 1 executed. Cache hit: {res_a['cache_hit']}. Spent eps: {orch_a.accountant.spent_epsilon:.4f}")
    print(f"  Released shares (first 3 cells): {[round(v, 5) for v in list(shares_a.values())[:3]]}")

    # Terminate / simulate process exit
    del orch_a
    del accountant_a
    del ledger_a
    del cache_a
    print("\n[PHASE 2] Simulating process termination and restart...")

    # Process B (Fresh restart with fresh memory objects pointing to persistent DBs)
    print("\n[PHASE 3] New Process (Process B) Startup:")
    cache_b = PersistentReleaseStore(test_cache_db)
    ledger_b = PrivacyLedger(test_ledger_db)
    accountant_b = BasicAccountant(
        global_budget=PrivacyBudget(epsilon=4.0, delta=1e-5),
        ledger=ledger_b,
    )
    orch_b = ReleaseOrchestrator(
        app_config=config,
        cache=cache_b,
        ledger=ledger_b,
        accountant=accountant_b,
    )

    print(f"  Process B accountant restored spent eps: {orch_b.accountant.spent_epsilon:.4f}")
    assert abs(orch_b.accountant.spent_epsilon - 1.0) < 1e-9, "Accountant failed to restore spent epsilon!"

    print("  Executing identical Query 1 on Process B...")
    res_b = orch_b.execute_release(query, umatrix, dataset_version="v1.0")
    shares_b = res_b["results"]["cell_shares"]
    print(f"  Query 1 executed. Cache hit: {res_b['cache_hit']}. Spent eps: {orch_b.accountant.spent_epsilon:.4f}")
    print(f"  Released shares (first 3 cells): {[round(v, 5) for v in list(shares_b.values())[:3]]}")

    assert res_b["cache_hit"] is True, "Expected cache hit across process restart!"
    assert shares_a == shares_b, "DP payload across process restart must be identical!"
    assert abs(orch_b.accountant.spent_epsilon - 1.0) < 1e-9, "Cache hit must not consume additional budget!"

    print("\n[SUCCESS] Persistent release store successfully prevented noise re-sampling across restarts!")
    print("[SUCCESS] Accountant successfully restored historical budget spend from persistent ledger!")

    # Cleanup test files
    if test_cache_db.exists():
        test_cache_db.unlink()
    if test_ledger_db.exists():
        test_ledger_db.unlink()


if __name__ == "__main__":
    run_cache_restart_verification()
