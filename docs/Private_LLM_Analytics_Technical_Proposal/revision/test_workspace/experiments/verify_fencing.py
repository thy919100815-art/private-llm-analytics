"""Verification script: Fencing token and compare-and-set commit rejection (Audit C11).

Verifies that:
  1. When a reservation's lease is superseded (e.g. taken over by a new worker at generation 2),
     the original worker (generation 1) is transactionally prevented from committing.
  2. The original worker's commit attempt fails with ReservationLostError.
  3. The active generation 2 release remains intact and uncorrupted.
  4. Total committed privacy budget is charged exactly once (epsilon=1.0, NOT 2.0).
  5. Aborting a superseded reservation does not affect the new owner's active release.
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
from private_llm_analytics.metrics.base import DPReleaseResult
from private_llm_analytics.planner.query import AnalyticsQuery
from private_llm_analytics.privacy.budget import PrivacyBudget
from private_llm_analytics.privacy.exceptions import ReservationLostError
from private_llm_analytics.privacy.state import PrivacyStateManager
from private_llm_analytics.release.release import ReleaseOrchestrator


def make_test_user_matrix() -> UserMatrix:
    provs = ["provider_a", "provider_b", "provider_c"]
    tsks = ["technical", "document", "analysis"]
    cell_names, cell_map = build_cell_definitions(provs, tsks)
    np_matrix = np.ones((5, 9), dtype=np.float64) * 10.0
    return UserMatrix(
        matrix=np_matrix,
        user_ids=[f"U{i}" for i in range(5)],
        providers=provs,
        tasks=tsks,
        cell_names=cell_names,
        cell_map=cell_map,
        public_population_size=50000,
    )


def main():
    print("=" * 70)
    print("VERIFICATION: Lease Fencing and CAS Commit Rejection (Audit C11)")
    print("=" * 70)

    test_db = Path("results/verify_fencing.db")
    for f in [test_db, Path("results/verify_fencing_cache.db"), Path("results/verify_fencing_ledger.db")]:
        if f.exists():
            f.unlink(missing_ok=True)
    for ext in ["-wal", "-shm", ".jsonl"]:
        p = Path(str(test_db) + ext)
        if p.exists():
            p.unlink(missing_ok=True)

    state_mgr = PrivacyStateManager(test_db)
    state_mgr.clear()

    global_budget = PrivacyBudget(epsilon=5.0, delta=1e-5)
    query_budget = PrivacyBudget(epsilon=1.0, delta=1e-6)
    release_key = "fencing_canonical_key_123"
    canonical_semantics = {"query": "fencing_test", "key": release_key}

    print("\n--- Step 1: Worker A reserves release_key at generation 1 ---")
    res_a = state_mgr.reserve_or_get_release(
        release_key=release_key,
        canonical_semantics=canonical_semantics,
        global_budget=global_budget,
        budget_needed=query_budget,
        plan_metadata={"metric": "user_normalized_share", "sensitivity": 1.0, "cell_names": ["c1", "c2"], "n_users": 10},
        lease_timeout=1.0,  # 1.0 second lease
        owner_id="worker_a",
    )
    status_a, _, res_id_a, rel_id_a = res_a
    gen_a = res_a.generation
    print(f"  Worker A reserved: status={status_a}, reservation_id={res_id_a}, generation={gen_a}")
    assert status_a == "RESERVED"
    assert gen_a == 1

    print("\n--- Step 2: Simulate lease expiry and takeover by Worker B at generation 2 ---")
    # Worker B comes in after Worker A's lease expires
    import time
    time.sleep(1.2)  # Wait for 1.2s > 1.0s lease_timeout

    res_b = state_mgr.reserve_or_get_release(
        release_key=release_key,
        canonical_semantics=canonical_semantics,
        global_budget=global_budget,
        budget_needed=query_budget,
        plan_metadata={"metric": "user_normalized_share", "sensitivity": 1.0, "cell_names": ["c1", "c2"], "n_users": 10},
        lease_timeout=1.0,
        owner_id="worker_b",
    )
    status_b, _, res_id_b, rel_id_b = res_b
    gen_b = res_b.generation
    print(f"  Worker B took over: status={status_b}, reservation_id={res_id_b}, generation={gen_b}")
    assert status_b == "RESERVED"
    assert gen_b == 2
    assert res_id_b != res_id_a

    print("\n--- Step 3: Worker B completes computation and successfully commits generation 2 ---")
    dummy_b_result = DPReleaseResult(
        metric_name="user_normalized_share",
        dp_vector=np.array([0.4, 0.6]),
        post_processed={"cell_shares": {"c1": 0.4, "c2": 0.6}},
        epsilon=1.0,
        delta=1e-6,
        mechanism="Gaussian",
        noise_scale=1.5,
        sensitivity=1.0,
        cell_names=["c1", "c2"],
        n_users=10,
        public_population_size=50000,
    )
    state_mgr.commit_release(
        reservation_id=res_id_b,
        release_key=release_key,
        release_id=rel_id_b,
        release_result=dummy_b_result,
        recipient="internal",
        dataset_version="v1.0",
        generation=gen_b,
    )
    print("  Worker B successfully committed generation 2.")

    print("\n--- Step 4: Worker A attempts to commit generation 1 (MUST BE REJECTED) ---")
    dummy_a_result = DPReleaseResult(
        metric_name="user_normalized_share",
        dp_vector=np.array([0.9, 0.1]),  # Stale, duplicate noise
        post_processed={"cell_shares": {"c1": 0.9, "c2": 0.1}},
        epsilon=1.0,
        delta=1e-6,
        mechanism="Gaussian",
        noise_scale=1.5,
        sensitivity=1.0,
        cell_names=["c1", "c2"],
        n_users=10,
        public_population_size=50000,
    )

    try:
        state_mgr.commit_release(
            reservation_id=res_id_a,
            release_key=release_key,
            release_id=rel_id_a,
            release_result=dummy_a_result,
            recipient="internal",
            dataset_version="v1.0",
            generation=gen_a,
        )
        assert False, "Worker A commit should have been rejected with ReservationLostError!"
    except ReservationLostError as exc:
        print(f"  Worker A commit correctly rejected: {exc}")

    print("\n--- Step 5: Worker A exception handler attempts abort_reservation ---")
    # Worker A calls abort_reservation for its own stale generation
    state_mgr.abort_reservation(res_id_a, release_key, generation=gen_a)
    print("  Worker A abort executed cleanly without touching Worker B's release.")

    print("\n--- Step 6: Verify Authoritative Persistent State ---")
    committed_release = state_mgr.get_committed_release(release_key)
    assert committed_release is not None
    # Verify the payload is Worker B's payload (0.4, 0.6), NOT Worker A's stale payload (0.9, 0.1)
    np.testing.assert_allclose(committed_release.dp_vector, np.array([0.4, 0.6]))
    print(f"  Committed DP Vector: {committed_release.dp_vector.tolist()} (Confirmed Worker B's payload)")

    spent_eps, spent_del = state_mgr.get_total_spent()
    print(f"  Total Committed Spent Epsilon: {spent_eps:.4f} (Expected: exactly 1.0000)")
    print(f"  Total Committed Spent Delta:   {spent_del:.2e} (Expected: exactly 1.00e-06)")
    assert abs(spent_eps - 1.0) < 1e-9
    assert abs(spent_del - 1e-6) < 1e-12

    with state_mgr._connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT reservation_id, generation, status FROM privacy_reservations")
        reservations = cursor.fetchall()
        cursor.execute("SELECT release_key, generation, status FROM release_store")
        releases = cursor.fetchall()

    print(f"  Reservations in DB: {reservations}")
    print(f"  Releases in DB:     {releases}")

    # Check reservation statuses: Worker A is ABORTED, Worker B is COMMITTED
    comm_res = [r for r in reservations if r[2] == "COMMITTED"]
    assert len(comm_res) == 1
    assert comm_res[0][0] == res_id_b
    assert comm_res[0][1] == 2

    # Check release store: exactly 1 release, generation 2, COMMITTED
    assert len(releases) == 1
    assert releases[0][1] == 2
    assert releases[0][2] == "COMMITTED"

    print("\n[VERIFICATION PASSED] CAS generation fencing prevents stale commits and ensures release integrity.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
