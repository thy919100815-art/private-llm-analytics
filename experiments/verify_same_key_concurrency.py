"""Verification script: Multi-worker same-key single-flight memoization.

Verifies that concurrent queries with the identical canonical semantic key:
  1. Do NOT generate multiple distinct noise vectors.
  2. Single-flight locks or memoizes to return the identical DP release payload.
  3. Consumes budget exactly once (budget spend = 1 * epsilon, NOT N * epsilon).
  4. Returns bitwise identical results across all concurrent workers.
"""

from concurrent.futures import ProcessPoolExecutor, as_completed
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
from private_llm_analytics.privacy.state import PrivacyStateManager
from private_llm_analytics.release.release import ReleaseOrchestrator


def same_key_worker_task(db_path_str: str, worker_id: int):
    """Worker task executing an identical query against the same persistent database."""
    config = load_app_config("configs")
    state_mgr = PrivacyStateManager(db_path_str)
    orchestrator = ReleaseOrchestrator(app_config=config, state_manager=state_mgr)

    # Build standard dummy user matrix
    provs = ["provider_a", "provider_b", "provider_c"]
    tsks = ["technical", "document", "analysis"]
    cell_names, cell_map = build_cell_definitions(provs, tsks)
    np_matrix = np.ones((10, 9), dtype=np.float64) * 20.0
    u_matrix = UserMatrix(
        matrix=np_matrix,
        user_ids=[f"U{i}" for i in range(10)],
        providers=provs,
        tasks=tsks,
        cell_names=cell_names,
        cell_map=cell_map,
        public_population_size=50000,
    )

    query = AnalyticsQuery(
        metric="user_normalized_share",
        recipient="internal",
        epsilon=1.0,
        delta=1e-6,
    )

    try:
        res = orchestrator.execute_release(
            query=query,
            user_matrix=u_matrix,
            dataset_version="bench_dataset_v1",
        )
        shares = res["results"]["cell_shares"]
        cache_hit = res["cache_hit"]
        rel_key = res["release_key"]
        return {
            "worker_id": worker_id,
            "status": "SUCCESS",
            "cache_hit": cache_hit,
            "release_key": rel_key,
            "cell_shares": shares,
            "error": None,
        }
    except Exception as e:
        return {
            "worker_id": worker_id,
            "status": "ERROR",
            "error": f"{type(e).__name__}: {str(e)}",
        }


def main():
    print("=" * 70)
    print("VERIFICATION: Multi-worker Same-Key Single-Flight Memoization (Audit C8/H7)")
    print("=" * 70)

    test_db = Path("results/verify_same_key_concurrency.db")
    if test_db.exists():
        test_db.unlink(missing_ok=True)
    for ext in ["-wal", "-shm"]:
        p = Path(str(test_db) + ext)
        if p.exists():
            p.unlink(missing_ok=True)

    state_mgr = PrivacyStateManager(test_db)
    state_mgr.clear()

    n_workers = 6
    print(f"Spawning {n_workers} concurrent workers with identical semantic query...")

    results = []
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = [
            executor.submit(same_key_worker_task, str(test_db), i)
            for i in range(n_workers)
        ]
        for f in as_completed(futures):
            results.append(f.result())

    successes = [r for r in results if r["status"] == "SUCCESS"]
    errors = [r for r in results if r["status"] == "ERROR"]

    print(f"\nExecution Summary:")
    print(f"  Success count: {len(successes)} / {n_workers}")
    print(f"  Error count:   {len(errors)}")

    if errors:
        for err in errors:
            print(f"    Worker {err.get('worker_id')}: {err.get('error')}")
        raise RuntimeError(f"Encountered {len(errors)} worker errors during same-key concurrency.")

    # Verify identical results
    first_shares = successes[0]["cell_shares"]
    first_key = successes[0]["release_key"]
    for idx, s in enumerate(successes[1:], start=1):
        assert s["release_key"] == first_key, f"Release key mismatch: {s['release_key']} != {first_key}"
        for k in first_shares:
            diff = abs(first_shares[k] - s["cell_shares"][k])
            assert diff == 0.0, f"Worker {idx} generated different noise for same semantic query! diff={diff}"

    print(f"  All {n_workers} workers returned BITWISE IDENTICAL results.")

    # Verify database state
    spent_eps, spent_del = state_mgr.get_total_spent()
    print(f"\nPersistent State Audit:")
    print(f"  Total Committed Spent Epsilon: {spent_eps:.4f} (Expected: exactly 1.0000)")
    print(f"  Total Committed Spent Delta:   {spent_del:.2e} (Expected: exactly 1.00e-06)")

    assert abs(spent_eps - 1.0) < 1e-9, f"Budget overspent across identical queries! {spent_eps} != 1.0"
    assert abs(spent_del - 1e-6) < 1e-12, f"Delta overspent across identical queries! {spent_del} != 1e-6"

    # Verify exactly 1 reservation in DB
    with state_mgr._connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM privacy_reservations WHERE status = 'COMMITTED'")
        res_count = cursor.fetchone()[0]
        cursor.execute("SELECT COUNT(*) FROM release_store WHERE status = 'COMMITTED'")
        store_count = cursor.fetchone()[0]

    print(f"  Committed Reservations in DB: {res_count} (Expected: 1)")
    print(f"  Committed Releases in DB:     {store_count} (Expected: 1)")
    assert res_count == 1, f"Expected 1 committed reservation, found {res_count}"
    assert store_count == 1, f"Expected 1 committed release payload, found {store_count}"

    print("\n[VERIFICATION PASSED] Single-flight memoization guarantees identical output with 0 budget duplication.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
