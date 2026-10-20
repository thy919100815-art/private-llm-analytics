"""Verification script: Multi-worker atomic privacy budget enforcement across independent processes.

Audit Item C7 verification.
Spawns multiple concurrent worker processes attempting to spend budget simultaneously.
Verifies that:
  1. No overspend occurs beyond global budget.
  2. Race conditions are handled atomically via SQLite BEGIN IMMEDIATE.
  3. Total committed budget <= global budget.
"""

from concurrent.futures import ProcessPoolExecutor, as_completed
import os
from pathlib import Path
import sys
import uuid

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from private_llm_analytics.privacy.budget import PrivacyBudget
from private_llm_analytics.privacy.exceptions import PrivacyBudgetExceededError
from private_llm_analytics.privacy.state import PrivacyStateManager


def worker_spend_task(db_path_str: str, worker_id: int, request_eps: float, request_delta: float, global_eps: float, global_delta: float):
    """Worker process task attempting atomic budget reservation and commit."""
    state_mgr = PrivacyStateManager(db_path_str)
    global_budget = PrivacyBudget(epsilon=global_eps, delta=global_delta)
    budget_needed = PrivacyBudget(epsilon=request_eps, delta=request_delta)
    query_key = f"worker_query_{worker_id}_{uuid.uuid4().hex[:8]}"
    semantics = {"worker_id": worker_id, "query_key": query_key}

    try:
        status, _, reservation_id, release_id = state_mgr.reserve_or_get_release(
            release_key=query_key,
            canonical_semantics=semantics,
            global_budget=global_budget,
            budget_needed=budget_needed,
            plan_metadata={
                "worker_id": worker_id,
                "mechanism": "gaussian",
                "sensitivity": 1.0,
                "noise_scale": 1.0,
            },
            allow_overrun=False,
        )

        if status == "RESERVED" and reservation_id is not None:
            # Simulate minimal work then commit
            from private_llm_analytics.metrics.base import DPReleaseResult
            import numpy as np

            dummy_result = DPReleaseResult(
                metric_name="test_worker_metric",
                dp_vector=np.array([1.0, 2.0]),
                post_processed={},
                epsilon=request_eps,
                delta=request_delta,
                mechanism="Gaussian",
                noise_scale=1.0,
                sensitivity=1.0,
                cell_names=["c1", "c2"],
                n_users=100,
                public_population_size=100,
            )
            state_mgr.commit_release(
                reservation_id=reservation_id,
                release_key=query_key,
                release_id=release_id,
                release_result=dummy_result,
                recipient="internal",
                dataset_version="v1.0",
            )
            return {"worker_id": worker_id, "status": "COMMITTED", "error": None}
        return {"worker_id": worker_id, "status": status, "error": None}

    except PrivacyBudgetExceededError as e:
        return {"worker_id": worker_id, "status": "EXCEEDED", "error": str(e)}
    except Exception as e:
        return {"worker_id": worker_id, "status": "ERROR", "error": f"{type(e).__name__}: {str(e)}"}


def main():
    print("=" * 70)
    print("VERIFICATION: Multi-worker Atomic Privacy Budget Enforcement (Audit C7)")
    print("=" * 70)

    test_db = Path("results/verify_multiworker_budget.db")
    if test_db.exists():
        test_db.unlink(missing_ok=True)
    for ext in ["-wal", "-shm"]:
        p = Path(str(test_db) + ext)
        if p.exists():
            p.unlink(missing_ok=True)

    # Initialize DB schema
    state_mgr = PrivacyStateManager(test_db)
    state_mgr.clear()

    global_eps = 3.0
    global_delta = 1e-5
    request_eps = 1.0
    request_delta = 2e-6
    n_workers = 8

    print(f"Global Budget: epsilon={global_eps}, delta={global_delta}")
    print(f"Spawning {n_workers} worker processes, each requesting eps={request_eps}, delta={request_delta}...")
    print(f"Expected: exactly 3 workers succeed (total eps=3.0), remaining {n_workers - 3} fail with PrivacyBudgetExceededError.")

    results = []
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = [
            executor.submit(
                worker_spend_task,
                str(test_db),
                i,
                request_eps,
                request_delta,
                global_eps,
                global_delta,
            )
            for i in range(n_workers)
        ]
        for f in as_completed(futures):
            results.append(f.result())

    committed_count = sum(1 for r in results if r["status"] == "COMMITTED")
    exceeded_count = sum(1 for r in results if r["status"] == "EXCEEDED")
    error_count = sum(1 for r in results if r["status"] == "ERROR")

    print(f"\nWorker Execution Results:")
    print(f"  COMMITTED: {committed_count}")
    print(f"  EXCEEDED:  {exceeded_count}")
    print(f"  ERROR:     {error_count}")

    total_spent_eps, total_spent_delta = state_mgr.get_total_spent()
    print(f"\nPersistent Database Audit:")
    print(f"  Final Spent Epsilon: {total_spent_eps:.4f} (Cap: {global_eps:.4f})")
    print(f"  Final Spent Delta:   {total_spent_delta:.2e} (Cap: {global_delta:.2e})")

    assert error_count == 0, f"Unexpected worker errors: {[r for r in results if r['status'] == 'ERROR']}"
    assert committed_count == 3, f"Expected exactly 3 commits, got {committed_count}"
    assert exceeded_count == 5, f"Expected 5 exceeded rejections, got {exceeded_count}"
    assert total_spent_eps <= global_eps + 1e-9, f"Overspend! {total_spent_eps} > {global_eps}"
    assert total_spent_delta <= global_delta + 1e-12, f"Overspend! {total_spent_delta} > {global_delta}"

    print("\n[VERIFICATION PASSED] Atomic budget compare-and-reserve across multi-workers is 100% sound.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
