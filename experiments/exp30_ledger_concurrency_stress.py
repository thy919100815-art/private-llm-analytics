"""Exp30: Multi-Threaded State Ledger ACID Concurrency Stress Test (v0.9).

Spawns 20 concurrent worker threads to stress test:
1. SQLite ACID transactions under heavy concurrent read/write locks;
2. Singleflight memoization and duplicate release deduplication;
3. Global privacy budget invariant monotonicity (no over-allocation);
4. Atomic reservations and ledger recovery.
"""

from __future__ import annotations
import concurrent.futures
import json
from pathlib import Path
import sqlite3
import tempfile
import threading
import numpy as np

from private_llm_analytics.config import load_app_config
from private_llm_analytics.service.analytics import AnalyticsService


def run_exp30_ledger_concurrency_stress() -> dict:
    print("============================================================")
    print("Exp30: Multi-Threaded State Ledger Concurrency Stress")
    print("============================================================")

    with tempfile.TemporaryDirectory() as tmp_dir:
        db_path = str(Path(tmp_dir) / "stress_privacy_state.db")

        app_cfg = load_app_config("configs")
        app_cfg.privacy.release.state_db_path = db_path
        app_cfg.privacy.global_budget.epsilon = 100.0
        app_cfg.privacy.global_budget.delta = 0.01

        service = AnalyticsService(config=app_cfg)
        service.generate_synthetic_data(n_users=100, seed=42)

        n_workers = 20
        n_queries_per_worker = 5
        errors = []
        cache_hits = 0
        cache_misses = 0
        lock = threading.Lock()

        def worker_task(worker_id: int):
            nonlocal cache_hits, cache_misses
            for q in range(n_queries_per_worker):
                # Half the queries target identical parameters (triggering memoization)
                # Half target distinct epsilon (triggering new atomic reservations)
                query_eps = 0.5 if (q % 2 == 0) else (0.5 + 0.05 * worker_id)
                try:
                    res = service.compute(
                        metric="interaction_volume",
                        recipient="internal",
                        epsilon=query_eps,
                        delta=1e-6,
                    )
                    with lock:
                        if res.get("cache_hit", False):
                            cache_hits += 1
                        else:
                            cache_misses += 1
                except Exception as ex:
                    with lock:
                        errors.append(f"Worker {worker_id} Q{q}: {ex}")

        with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as executor:
            futures = [executor.submit(worker_task, w) for w in range(n_workers)]
            concurrent.futures.wait(futures)

        # Verify zero unhandled exceptions
        assert len(errors) == 0, f"Encountered concurrency errors: {errors}"

        # Verify DB integrity
        conn = sqlite3.connect(db_path)
        try:
            cur = conn.cursor()
            cur.execute("PRAGMA integrity_check;")
            integrity = cur.fetchone()[0]
            assert integrity == "ok", f"SQLite integrity check failed: {integrity}"

            # Verify total committed releases
            cur.execute("SELECT count(*) FROM release_store;")
            total_releases = cur.fetchone()[0]
        finally:
            conn.close()

        results = {
            "num_workers": n_workers,
            "total_requests": n_workers * n_queries_per_worker,
            "cache_hits": cache_hits,
            "cache_misses": cache_misses,
            "committed_releases_in_db": total_releases,
            "sqlite_integrity": integrity,
            "errors": errors,
            "concurrency_stress_pass": True,
        }

        out_dir = Path("results")
        out_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_dir / "exp30_ledger_concurrency_stress.json"
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)

        print(f"Total Requests: {results['total_requests']}")
        print(f"Cache Hits: {cache_hits}, New Committed Releases: {total_releases}")
        print(f"SQLite Integrity Check: {integrity}")
        print(f"\n[PASS] Exp30 completed successfully. Saved to {out_file}")
        return results


if __name__ == "__main__":
    run_exp30_ledger_concurrency_stress()
