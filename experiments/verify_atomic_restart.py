"""Verification script: Atomic crash consistency and restart recovery (Audit C8).

Verifies that:
  1. Fault injected before/during noise generation aborts pending reservations.
  2. Uncommitted crashes leave 0 leaked privacy budget.
  3. No partial/unauthorized releases remain in cache.
  4. Process restarts properly recover state and sweep stale reservations.
  5. Committed releases survive restart and are successfully memoized with 0 spend.
"""

from pathlib import Path
import sys
import numpy as np

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from private_llm_analytics.aggregation.user_matrix import (
    UserMatrix,
    build_cell_definitions,
)
from private_llm_analytics.config.loader import load_app_config
from private_llm_analytics.planner.query import AnalyticsQuery
from private_llm_analytics.privacy.budget import PrivacyBudget
from private_llm_analytics.privacy.state import PrivacyStateManager
from private_llm_analytics.release.release import ReleaseOrchestrator


def make_test_user_matrix() -> UserMatrix:
    provs = ["provider_a", "provider_b", "provider_c"]
    tsks = ["technical", "document", "analysis"]
    cell_names, cell_map = build_cell_definitions(provs, tsks)
    np_matrix = np.ones((5, 9), dtype=np.float64) * 15.0
    return UserMatrix(
        matrix=np_matrix,
        user_ids=[f"U{i}" for i in range(5)],
        providers=provs,
        tasks=tsks,
        cell_names=cell_names,
        cell_map=cell_map,
        public_population_size=50000,
    )


def test_crash_after_reserve():
    print("\n--- Test 1: Crash after budget reservation, before noise generation ---")
    test_db = Path("results/verify_restart_test1.db")
    if test_db.exists():
        test_db.unlink(missing_ok=True)

    config = load_app_config("configs")
    state_mgr = PrivacyStateManager(test_db)
    u_mat = make_test_user_matrix()

    # Inject fault after reservation
    hooks = {"after_budget_reserve": RuntimeError("Simulated worker process crash during computation")}
    orch = ReleaseOrchestrator(app_config=config, state_manager=state_mgr, test_hooks=hooks)

    query = AnalyticsQuery(metric="user_normalized_share", recipient="internal", epsilon=1.0, delta=1e-6)

    try:
        orch.execute_release(query, u_mat)
        assert False, "Should have crashed due to injected fault!"
    except RuntimeError as e:
        print(f"  Caught expected simulated crash: {e}")

    # Verify no committed spend
    spent_eps, spent_del = state_mgr.get_total_spent()
    print(f"  Spent budget after crash: eps={spent_eps}, delta={spent_del} (Expected: 0.0, 0.0)")
    assert spent_eps == 0.0
    assert spent_del == 0.0

    # Verify reservation status is ABORTED
    with state_mgr._connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT status FROM privacy_reservations")
        statuses = [row[0] for row in cursor.fetchall()]
    print(f"  Reservation statuses in DB: {statuses} (Expected: ['ABORTED'])")
    assert statuses == ["ABORTED"]
    print("  [PASS] Crash after reserve cleanly aborted; 0 budget leaked.")


def test_crash_before_commit():
    print("\n--- Test 2: Crash after noise generation, before release commit ---")
    test_db = Path("results/verify_restart_test2.db")
    if test_db.exists():
        test_db.unlink(missing_ok=True)

    config = load_app_config("configs")
    state_mgr = PrivacyStateManager(test_db)
    u_mat = make_test_user_matrix()

    hooks = {"before_release_commit": RuntimeError("Simulated DB connection lost before commit")}
    orch = ReleaseOrchestrator(app_config=config, state_manager=state_mgr, test_hooks=hooks)

    query = AnalyticsQuery(metric="user_normalized_share", recipient="internal", epsilon=1.0, delta=1e-6)

    try:
        orch.execute_release(query, u_mat)
        assert False, "Should have crashed before commit!"
    except RuntimeError as e:
        print(f"  Caught expected simulated crash: {e}")

    spent_eps, spent_del = state_mgr.get_total_spent()
    print(f"  Spent budget after crash: eps={spent_eps}, delta={spent_del} (Expected: 0.0, 0.0)")
    assert spent_eps == 0.0
    assert spent_del == 0.0

    # Ensure no committed release in store
    with state_mgr._connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM release_store WHERE status = 'COMMITTED'")
        committed_releases = cursor.fetchone()[0]
    print(f"  Committed releases in DB: {committed_releases} (Expected: 0)")
    assert committed_releases == 0
    print("  [PASS] Crash before commit left 0 committed releases and 0 budget spent.")


def test_restart_recovery_and_cache():
    print("\n--- Test 3: Restart recovery and committed cache survival ---")
    test_db = Path("results/verify_restart_test3.db")
    for f in [test_db, Path("results/verify_restart_test3_cache.db"), Path("results/verify_restart_test3_ledger.db")]:
        if f.exists():
            f.unlink(missing_ok=True)

    config = load_app_config("configs")
    u_mat = make_test_user_matrix()

    # Step 1: Normal commit in Process 1
    state_mgr1 = PrivacyStateManager(test_db)
    orch1 = ReleaseOrchestrator(app_config=config, state_manager=state_mgr1)
    query = AnalyticsQuery(metric="user_prevalence", recipient="internal", epsilon=1.5, delta=1e-6)

    res1 = orch1.execute_release(query, u_mat)
    assert res1["cache_hit"] is False
    spent1_eps, spent1_del = state_mgr1.get_total_spent()
    print(f"  Process 1 committed: eps={spent1_eps}, delta={spent1_del}")
    assert spent1_eps == 1.5

    # Step 2: Fresh restart (Process 2) opens same DB
    state_mgr2 = PrivacyStateManager(test_db)
    recovered_count = state_mgr2.recover_stale_reservations()
    print(f"  Process 2 recovery swept {recovered_count} stale reservations.")

    spent2_eps, spent2_del = state_mgr2.get_total_spent()
    print(f"  Process 2 restored spent budget: eps={spent2_eps}, delta={spent2_del}")
    assert spent2_eps == 1.5
    assert spent2_del == 1e-6

    orch2 = ReleaseOrchestrator(app_config=config, state_manager=state_mgr2)
    res2 = orch2.execute_release(query, u_mat)
    print(f"  Process 2 re-execution cache_hit: {res2['cache_hit']}")
    assert res2["cache_hit"] is True

    # Check identical values
    assert res1["results"]["prevalence_rates"] == res2["results"]["prevalence_rates"]

    # Budget spent must NOT increase
    spent3_eps, spent3_del = state_mgr2.get_total_spent()
    print(f"  Spent budget after cache hit: eps={spent3_eps}, delta={spent3_del} (Unchanged: 1.5)")
    assert spent3_eps == 1.5

    print("  [PASS] Restart recovery and cache persistence verified.")


def test_crash_after_commit():
    print("\n--- Test 4: Exception after release commit, before response (Audit C12) ---")
    test_db = Path("results/verify_restart_test4.db")
    for f in [test_db, Path("results/verify_restart_test4_cache.db"), Path("results/verify_restart_test4_ledger.db")]:
        if f.exists():
            f.unlink(missing_ok=True)

    config = load_app_config("configs")
    state_mgr = PrivacyStateManager(test_db)
    u_mat = make_test_user_matrix()

    # Injected fault AFTER commit
    hooks = {"after_release_commit": RuntimeError("Simulated network crash after DB commit before HTTP response")}
    orch = ReleaseOrchestrator(app_config=config, state_manager=state_mgr, test_hooks=hooks)

    query = AnalyticsQuery(metric="user_normalized_share", recipient="internal", epsilon=1.0, delta=1e-6)

    try:
        orch.execute_release(query, u_mat)
        assert False, "Should have crashed after commit!"
    except RuntimeError as e:
        print(f"  Caught expected simulated crash: {e}")

    # Crucial check: Release MUST remain COMMITTED, NOT ABORTED!
    spent_eps, spent_del = state_mgr.get_total_spent()
    print(f"  Spent budget after crash: eps={spent_eps}, delta={spent_del} (Expected: 1.0, 1e-06)")
    assert spent_eps == 1.0
    assert spent_del == 1e-6

    with state_mgr._connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT status FROM privacy_reservations")
        statuses = [row[0] for row in cursor.fetchall()]
        cursor.execute("SELECT status FROM release_store")
        rel_statuses = [row[0] for row in cursor.fetchall()]
    print(f"  Reservation status: {statuses} (Expected: ['COMMITTED'])")
    print(f"  Release status:     {rel_statuses} (Expected: ['COMMITTED'])")
    assert statuses == ["COMMITTED"]
    assert rel_statuses == ["COMMITTED"]

    # Re-query after crash without fault hook: should hit cache with 0 additional spend!
    orch_recovered = ReleaseOrchestrator(app_config=config, state_manager=state_mgr)
    res_recovered = orch_recovered.execute_release(query, u_mat)
    assert res_recovered["cache_hit"] is True
    spent_eps_after, _ = state_mgr.get_total_spent()
    assert spent_eps_after == 1.0
    print("  [PASS] Commit is irreversible; release survived crash and was served as cache hit.")


def main():
    print("=" * 70)
    print("VERIFICATION: Atomic Restart & Crash Consistency (Audit C8 & C12)")
    print("=" * 70)

    test_crash_after_reserve()
    test_crash_before_commit()
    test_restart_recovery_and_cache()
    test_crash_after_commit()

    print("\n" + "=" * 70)
    print("[ALL RECOVERY VERIFICATIONS PASSED SUCCESSFULLY]")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
