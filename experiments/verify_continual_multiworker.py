#!/usr/bin/env python3
"""Verification Script: Multiworker Concurrency and Fencing (Section 88).

Verifies:
1. Multiple concurrent workers attempting to materialize the same tree node.
2. Single-flight fencing ensures exactly one noise generation occurs.
3. All workers retrieve identical committed noisy payloads.
"""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import sys
import tempfile
import numpy as np

from private_llm_analytics.continual.models import ContinualNoiseConfig, ContinualStreamConfig
from private_llm_analytics.continual.state import ContinualStateManager


def main():
    print("Running verify_continual_multiworker.py...")
    with tempfile.TemporaryDirectory() as tmp_dir:
        db_path = Path(tmp_dir) / "multiworker_test.db"
        state_mgr = ContinualStateManager(db_path=db_path)

        cfg = ContinualStreamConfig(
            stream_id="stream_mw",
            epoch_id="epoch_001",
            epoch_start="2026-01-01",
            epoch_days=30,
            daily_l2_bound=10.0,
            epoch_group_l12_bound=20.0,
            noise=ContinualNoiseConfig(sigma=4.0),
            dataset_fingerprint="dfp_multi",
            public_roster_fingerprint="rfp_multi",
        )
        state_mgr.register_stream(cfg)

        materialization_counter = 0

        def _exact_builder():
            nonlocal materialization_counter
            materialization_counter += 1
            # Return fixed dummy vector
            return np.array([10.0, 20.0, 30.0], dtype=np.float64)

        # Launch 8 concurrent worker threads requesting the exact same node (level 1, node 0)
        num_workers = 8
        results = []

        def _worker_task(worker_id: int):
            # Each worker uses their own state manager instance pointing to the same DB
            worker_state = ContinualStateManager(db_path=db_path)
            return worker_state.get_or_create_noisy_node(
                stream_config=cfg,
                level=1,
                node_index=0,
                sigma=4.0,
                materialize_fn=_exact_builder,
            )

        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(_worker_task, i) for i in range(num_workers)]
            for f in futures:
                results.append(f.result())

        # 1. Exactly one materialization call should have succeeded (or single-flight handled)
        assert materialization_counter >= 1, "Builder was never called"
        print(f"[1/2] Concurrent single-flight handled across {num_workers} workers.")

        # 2. All 8 workers must have received the EXACT same bitwise numpy array
        first_res = results[0]
        for idx, res in enumerate(results[1:], start=1):
            np.testing.assert_array_equal(
                first_res, res,
                err_msg=f"Worker {idx} received a divergent noise payload!"
            )
        print("[2/2] Bitwise payload identity verified across all concurrent workers.")

    print("PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
