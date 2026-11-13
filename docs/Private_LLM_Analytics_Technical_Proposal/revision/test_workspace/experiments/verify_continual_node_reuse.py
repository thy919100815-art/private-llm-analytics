#!/usr/bin/env python3
"""Verification Script: Continual Node Reuse and Noise Consistency (Section 88).

Verifies:
1. Canonical noisy node payloads are shared across recipients (prevents noise averaging attacks).
2. Committed nodes persist across process restarts and return bitwise identical payloads.
3. Node identity is strictly decoupled from recipient and query scopes.
"""

from pathlib import Path
import sys
import tempfile
import numpy as np
import pandas as pd

from private_llm_analytics.continual.models import (
    ContinualNoiseConfig,
    ContinualStreamConfig,
)
from private_llm_analytics.continual.query import ContinualQueryEngine
from private_llm_analytics.continual.state import ContinualStateManager


def main():
    print("Running verify_continual_node_reuse.py...")
    with tempfile.TemporaryDirectory() as tmp_dir:
        db_path = Path(tmp_dir) / "reuse_test.db"

        state_mgr = ContinualStateManager(db_path=db_path)
        engine = ContinualQueryEngine(state_manager=state_mgr, composition_mode="per_recipient")

        cfg = ContinualStreamConfig(
            stream_id="stream_reuse",
            epoch_id="epoch_001",
            epoch_start="2026-01-01",
            epoch_days=30,
            daily_l2_bound=10.0,
            epoch_group_l12_bound=20.0,
            noise=ContinualNoiseConfig(target_epoch_epsilon=2.0),
        )

        roster = [f"u_{i}" for i in range(5)]
        records = [{
            "user_id": "u_0",
            "timestamp": "2026-01-01 12:00:00",
            "provider": "provider_a",
            "task": "technical",
        }]
        df = pd.DataFrame(records)

        # 1. Recipient A queries interval [1, 4]
        res_a = engine.query_interval(cfg, 1, 4, recipient="ProviderA", interactions=df, public_user_roster=roster, finalized_up_to_day=cfg.epoch_days)

        # 2. Recipient B queries the exact same interval [1, 4]
        res_b = engine.query_interval(cfg, 1, 4, recipient="ProviderB", interactions=df, public_user_roster=roster, finalized_up_to_day=cfg.epoch_days)

        # Must have bitwise identical vectors
        np.testing.assert_array_equal(
            res_a.vector, res_b.vector,
            err_msg="Security violation: Different recipients received distinct noise payloads for the same node!"
        )
        print("[1/2] Noise averaging attack prevention verified: bitwise identical payload across recipients.")

        # 3. Simulate process restart by closing state manager and reopening fresh connection
        del engine
        del state_mgr

        fresh_state_mgr = ContinualStateManager(db_path=db_path)
        fresh_engine = ContinualQueryEngine(state_manager=fresh_state_mgr, composition_mode="per_recipient")

        # Query with interactions=None to guarantee no re-computation
        res_restart = fresh_engine.query_interval(
            cfg, 1, 4, recipient="ProviderA", interactions=None, public_user_roster=None
        )

        np.testing.assert_array_equal(
            res_a.vector, res_restart.vector,
            err_msg="Restart consistency violation: Payload changed after reconnection!"
        )
        print("[2/2] Restart consistency verified: identical payload retrieved without accessing raw data.")

    print("PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
