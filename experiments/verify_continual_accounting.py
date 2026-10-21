#!/usr/bin/env python3
"""Verification Script: Continual Multi-Recipient Accounting and Atomic Debit (Section 88).

Verifies:
1. GLOBAL_EXTERNAL mode charges levels once globally; subsequent queries are free.
2. PER_RECIPIENT mode charges levels once per recipient scope.
3. COLLUSION_GROUP mode charges levels once per consortium.
4. Multi-level atomic budget failure rejects queries all-or-nothing with zero partial debit.
"""

from pathlib import Path
import sys
import tempfile
import pandas as pd

from private_llm_analytics.continual.models import (
    ContinualNoiseConfig,
    ContinualStreamConfig,
)
from private_llm_analytics.continual.query import ContinualQueryEngine
from private_llm_analytics.continual.state import ContinualStateManager
from private_llm_analytics.privacy.exceptions import PrivacyBudgetExceededError


def main():
    print("Running verify_continual_accounting.py...")
    with tempfile.TemporaryDirectory() as tmp_dir:
        db_path = Path(tmp_dir) / "accounting_test.db"

        state_mgr = ContinualStateManager(db_path=db_path)
        cfg = ContinualStreamConfig(
            stream_id="stream_acc",
            epoch_id="epoch_001",
            epoch_start="2026-01-01",
            epoch_days=30,
            daily_l2_bound=10.0,
            epoch_group_l12_bound=20.0,
            noise=ContinualNoiseConfig(target_epoch_epsilon=2.0),
        )

        roster = ["u_0"]
        df = pd.DataFrame([{
            "user_id": "u_0",
            "timestamp": "2026-01-01 12:00:00",
            "provider": "provider_a",
            "task": "technical",
        }])

        # 1. Verify GLOBAL_EXTERNAL
        state_mgr_global = ContinualStateManager(db_path=Path(tmp_dir) / "global.db")
        eng_global = ContinualQueryEngine(state_manager=state_mgr_global, composition_mode="global_external")
        res_g1 = eng_global.query_interval(cfg, 1, 4, recipient="ExtA", interactions=df, public_user_roster=roster)
        assert res_g1.newly_exposed_levels == [2]
        res_g2 = eng_global.query_interval(cfg, 1, 4, recipient="ExtB", interactions=df, public_user_roster=roster)
        assert res_g2.newly_exposed_levels == []
        print("[1/4] GLOBAL_EXTERNAL level charge once verified.")

        # 2. Verify PER_RECIPIENT
        state_mgr_per = ContinualStateManager(db_path=Path(tmp_dir) / "per.db")
        eng_per = ContinualQueryEngine(state_manager=state_mgr_per, composition_mode="per_recipient")
        res_p1 = eng_per.query_interval(cfg, 1, 2, recipient="RecA", interactions=df, public_user_roster=roster)
        assert res_p1.newly_exposed_levels == [1]
        res_p2 = eng_per.query_interval(cfg, 1, 2, recipient="RecB", interactions=df, public_user_roster=roster)
        assert res_p2.newly_exposed_levels == [1]
        print("[2/4] PER_RECIPIENT isolated scope accounting verified.")

        # 3. Verify COLLUSION_GROUP
        state_mgr_col = ContinualStateManager(db_path=Path(tmp_dir) / "col.db")
        groups = {"g_team": ["PartnerX", "PartnerY"]}
        eng_col = ContinualQueryEngine(state_manager=state_mgr_col, composition_mode="collusion_group", collusion_groups=groups)
        res_c1 = eng_col.query_interval(cfg, 1, 2, recipient="PartnerX", interactions=df, public_user_roster=roster)
        assert res_c1.newly_exposed_levels == [1]
        res_c2 = eng_col.query_interval(cfg, 1, 2, recipient="PartnerY", interactions=df, public_user_roster=roster)
        assert res_c2.newly_exposed_levels == []
        print("[3/4] COLLUSION_GROUP shared scope accounting verified.")

        # 4. Verify Multi-Level Atomic Debit Failure
        state_mgr_cap = ContinualStateManager(db_path=Path(tmp_dir) / "cap.db")
        eng_capped = ContinualQueryEngine(state_manager=state_mgr_cap, composition_mode="per_recipient", epsilon_cap=0.01)
        try:
            eng_capped.query_interval(cfg, 1, 7, recipient="PoorRec", interactions=df, public_user_roster=roster)
            assert False, "Expected PrivacyBudgetExceededError"
        except PrivacyBudgetExceededError:
            pass

        # Zero levels recorded
        exposed = state_mgr_cap.get_exposed_levels("recipient:PoorRec", cfg.stream_fingerprint, cfg.epoch_id)
        assert len(exposed) == 0
        print("[4/4] Multi-level atomic budget reservation failure verified (all-or-nothing).")

    print("PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
