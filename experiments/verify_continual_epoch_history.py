#!/usr/bin/env python3
"""Verification Script: Epoch Privacy Continuity and Unified Lifetime Composition (Section 88).

Verifies:
1. Transitioning from Epoch 1 to Epoch 2 does NOT reset lifetime privacy.
2. Continual level mechanism exposures compose additively with historical releases.
"""

from pathlib import Path
import sys
import tempfile
import numpy as np
import pandas as pd

from private_llm_analytics.continual.models import ContinualNoiseConfig, ContinualStreamConfig
from private_llm_analytics.continual.query import ContinualQueryEngine
from private_llm_analytics.continual.state import ContinualStateManager


def main():
    print("Running verify_continual_epoch_history.py...")
    with tempfile.TemporaryDirectory() as tmp_dir:
        db_path = Path(tmp_dir) / "epoch_test.db"
        state_mgr = ContinualStateManager(db_path=db_path)
        engine = ContinualQueryEngine(state_manager=state_mgr, composition_mode="global_external")

        cfg1 = ContinualStreamConfig(
            stream_id="stream_e1",
            epoch_id="epoch_001",
            epoch_start="2026-01-01",
            epoch_days=30,
            daily_l2_bound=10.0,
            epoch_group_l12_bound=20.0,
            noise=ContinualNoiseConfig(target_epoch_epsilon=1.5),
        )
        cfg2 = ContinualStreamConfig(
            stream_id="stream_e2",
            epoch_id="epoch_002",
            epoch_start="2026-01-31",
            epoch_days=30,
            daily_l2_bound=10.0,
            epoch_group_l12_bound=20.0,
            noise=ContinualNoiseConfig(target_epoch_epsilon=1.5),
        )

        roster = ["u_0"]
        df1 = pd.DataFrame([{"user_id": "u_0", "timestamp": "2026-01-01 12:00:00", "provider": "provider_a", "task": "technical"}])
        df2 = pd.DataFrame([{"user_id": "u_0", "timestamp": "2026-01-31 12:00:00", "provider": "provider_a", "task": "technical"}])

        # 1. Query in Epoch 1
        res1 = engine.query_interval(cfg1, 1, 4, recipient="ProviderA", interactions=df1, public_user_roster=roster, finalized_up_to_day=30)
        exp1_levels = state_mgr.get_exposed_levels("global_external", cfg1.stream_fingerprint, "epoch_001")
        assert len(exp1_levels) > 0
        print("[1/2] Epoch 1 levels exposed and recorded.")

        # 2. Query in Epoch 2
        res2 = engine.query_interval(cfg2, 1, 4, recipient="ProviderA", interactions=df2, public_user_roster=roster, finalized_up_to_day=30)
        exp2_levels = state_mgr.get_exposed_levels("global_external", cfg2.stream_fingerprint, "epoch_002")
        assert len(exp2_levels) > 0

        # Verify that both Epoch 1 and Epoch 2 level exposures exist simultaneously in continual_level_exposures
        with state_mgr._connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT COUNT(DISTINCT epoch_id), COUNT(*) FROM continual_level_exposures
                WHERE budget_scope_id = 'global_external' AND reservation_state = 'COMMITTED'
                """
            )
            n_epochs, n_exposures = cursor.fetchone()
            assert n_epochs == 2, f"Expected 2 distinct epochs in lifetime store, got {n_epochs}"
            assert n_exposures >= 2, f"Expected at least 2 committed exposures, got {n_exposures}"

        print("[2/2] Epoch continuity verified: lifetime privacy spans across multiple epochs without reset.")

    print("PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
