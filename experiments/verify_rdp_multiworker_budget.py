"""Verification script: Multi-Worker Concurrent RDP Budget Enforcement (v0.2).

Verifies that:
  1. Multiple concurrent workers attempting to spend RDP privacy budget
     simultaneously are handled atomically via SQLite BEGIN IMMEDIATE.
  2. With budget cap permitting exactly 3 distinct releases out of 8 requests:
     - Exactly 3 workers succeed (COMMITTED)
     - Exactly 5 workers are rejected (EXCEEDED / PrivacyBudgetExceededError)
     - 0 unexpected errors or race condition corruptions occur
  3. Total committed effective epsilon <= global epsilon cap.
"""

from concurrent.futures import ProcessPoolExecutor, as_completed
from decimal import Decimal
import os
from pathlib import Path
import sys
import tempfile
import time
import uuid
import numpy as np

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from private_llm_analytics.metrics.base import DPReleaseResult
from private_llm_analytics.privacy.budget import PrivacyBudget
from private_llm_analytics.privacy.exceptions import PrivacyBudgetExceededError
from private_llm_analytics.privacy.rdp import (
    DEFAULT_RDP_ORDERS,
    compute_gaussian_rdp,
    rdp_vector_to_json,
)
from private_llm_analytics.privacy.state import PrivacyStateManager


def worker_task(db_path_str: str, worker_id: int, cap: float, delta_global: float) -> dict:
    """Worker task executed concurrently."""
    state_mgr = PrivacyStateManager(db_path_str)
    orders = DEFAULT_RDP_ORDERS
    rdp_cand = compute_gaussian_rdp(sigma=2.0, sensitivity=1.0, orders=orders)
    needed_budget = PrivacyBudget(epsilon=1.0, delta=1e-6)
    global_budget = PrivacyBudget(epsilon=cap, delta=delta_global)
    query_key = f"worker_query_{worker_id}_{uuid.uuid4().hex[:8]}"
    semantics = {"worker_id": worker_id, "query_key": query_key}

    try:
        status, _, res_id, rel_id = state_mgr.reserve_or_get_release(
            release_key=query_key,
            canonical_semantics=semantics,
            global_budget=global_budget,
            budget_needed=needed_budget,
            plan_metadata={"worker_id": worker_id},
            lease_timeout=15.0,
            accountant_mode="rdp",
            budget_scope_id="global_external",
            rdp_orders=orders,
            rdp_candidate_cost=rdp_cand,
            epsilon_cap=cap,
            delta_global=delta_global,
        )

        if status == "RESERVED" and res_id is not None:
            # Simulate slight workload
            time.sleep(0.03)
            dummy = DPReleaseResult(
                metric_name="user_prevalence",
                dp_vector=np.array([float(worker_id)]),
                post_processed={},
                epsilon=1.0,
                delta=1e-6,
                mechanism="Gaussian",
                noise_scale=2.0,
                sensitivity=1.0,
                cell_names=["c1"],
                n_users=10,
                public_population_size=50000,
            )
            state_mgr.commit_release(
                reservation_id=res_id,
                release_key=query_key,
                release_id=rel_id,
                release_result=dummy,
                recipient="provider_a",
                dataset_version="v1.0",
                generation=1,
                budget_scope_id="global_external",
                accountant_mode="rdp",
                rdp_cost_json=rdp_vector_to_json(rdp_cand),
                delta_global=delta_global,
            )
            return {"worker_id": worker_id, "status": "COMMITTED"}
        return {"worker_id": worker_id, "status": "CACHE_HIT"}
    except PrivacyBudgetExceededError:
        return {"worker_id": worker_id, "status": "EXCEEDED"}
    except Exception as exc:
        return {"worker_id": worker_id, "status": "ERROR", "error": str(exc)}


def main():
    print("=" * 70)
    print("VERIFICATION: Multi-Worker Concurrent RDP Budget Enforcement")
    print("=" * 70)

    with tempfile.TemporaryDirectory() as tmp_dir:
        db_path = Path(tmp_dir) / "privacy_state.db"
        state_mgr = PrivacyStateManager(db_path)

        cap = 3.5
        delta_global = 1e-5
        n_workers = 8

        print(f"Spawning {n_workers} concurrent worker processes (cap: {cap:.2f}, delta: {delta_global:.1e})...")

        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = [
                executor.submit(worker_task, str(db_path), i, cap, delta_global)
                for i in range(n_workers)
            ]
            results = [f.result() for f in as_completed(futures)]

        committed_count = sum(1 for r in results if r["status"] == "COMMITTED")
        exceeded_count = sum(1 for r in results if r["status"] == "EXCEEDED")
        error_count = sum(1 for r in results if r["status"] == "ERROR")

        print(f"\nExecution Outcomes across {n_workers} workers:")
        print(f"  COMMITTED: {committed_count}")
        print(f"  EXCEEDED:  {exceeded_count}")
        print(f"  ERROR:     {error_count}")

        if error_count > 0:
            for r in results:
                if r["status"] == "ERROR":
                    print(f"    Error: {r}")

        assert error_count == 0, f"Encountered {error_count} unexpected worker errors"
        assert committed_count == 3, f"Expected exactly 3 commits, got {committed_count}"
        assert exceeded_count == 5, f"Expected exactly 5 rejections, got {exceeded_count}"

        # Authoritative database audit
        view = state_mgr.get_privacy_view(
            "global_external",
            delta_global=delta_global,
            orders=DEFAULT_RDP_ORDERS,
            epsilon_cap=cap,
        )
        print(f"\nDatabase Audit:")
        print(f"  Committed Releases: {view.release_count}")
        print(f"  Effective Epsilon:  {float(view.effective_epsilon):.4f} (Cap: {cap:.4f})")
        print(f"  Best Alpha:         {float(view.best_alpha)}")

        assert view.release_count == 3
        assert float(view.effective_epsilon) <= cap, f"Overspend! {view.effective_epsilon} > {cap}"

    print("\n[VERIFICATION PASSED] Atomic multi-worker RDP budget enforcement is sound.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
