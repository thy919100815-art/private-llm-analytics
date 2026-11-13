"""Audit of Canonical Replay, Recipient Scopes, and Continual Observation."""

import tempfile
from pathlib import Path
import numpy as np
import pandas as pd

from private_llm_analytics.config import load_app_config
from private_llm_analytics.config.validator import ExecutionMode
from private_llm_analytics.continual.models import (
    ContinualNoiseConfig,
    ContinualStreamConfig,
    ContinualTreeConfig,
)
from private_llm_analytics.continual.query import ContinualQueryEngine
from private_llm_analytics.continual.state import ContinualStateManager
from private_llm_analytics.continual.tree import BinaryTimeTree
from private_llm_analytics.privacy.state import PrivacyStateManager
from private_llm_analytics.service.analytics import AnalyticsService


def run_replay_audit():
    print("=== 6. CANONICAL REPLAY & RECIPIENT COMPOSITION AUDIT ===")
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test_privacy_state.db"
        cfg = load_app_config("configs")
        # Ensure production execution mode with custom db
        cfg.privacy.release.state_db_path = str(db_path)
        service = AnalyticsService(cfg)
        service.state_manager = PrivacyStateManager(db_path)
        service.orchestrator.state_manager = service.state_manager

        # Generate telemetry with 200 users
        service.generate_synthetic_data(n_users=200, seed=42)

        # Release 1: provider_a queries user_prevalence
        pv_before = service.get_privacy_view("provider_a")
        eps_before = float(pv_before.effective_epsilon)
        res1 = service.compute("user_prevalence", recipient="provider_a", epsilon=1.0, delta=1e-6)
        pv_after = service.get_privacy_view("provider_a")
        eps_after = float(pv_after.effective_epsilon)

        print(f"Release 1 (provider_a): incremental eps = {eps_after - eps_before:.4f}")
        assert eps_after > eps_before, "Release 1 should have consumed budget"

        # Release 2: EXACT REPLAY by provider_a
        res2 = service.compute("user_prevalence", recipient="provider_a", epsilon=1.0, delta=1e-6)
        pv_replay = service.get_privacy_view("provider_a")
        eps_replay = float(pv_replay.effective_epsilon)

        print(f"Release 2 (Replay): incremental eps = {eps_replay - eps_after:.4f}, cache_hit = {res2.get('cache_hit')}")
        assert eps_replay == eps_after, "Replay must consume ZERO incremental budget"
        assert res2["release_key"] == res1["release_key"], "Replay must return same release_key"
        assert res2["results"] == res1["results"], "Replay must return identical results"

        # Release 3: External provider_b queries under global_external scope
        pv_b_before = service.get_privacy_view("provider_b")
        print(f"Recipient provider_b sees global_external effective epsilon: {float(pv_b_before.effective_epsilon):.4f}")
        assert float(pv_b_before.effective_epsilon) == eps_after, "provider_b must share global_external budget scope with provider_a"

        # Release 4: Internal recipient has separate scope
        pv_internal = service.get_privacy_view("internal")
        print(f"Recipient internal effective epsilon: {float(pv_internal.effective_epsilon):.4f}")
        assert float(pv_internal.effective_epsilon) == 0.0, "internal recipient must have separate budget scope"
        print("Canonical Replay and Recipient Composition: PASS")


def run_continual_observation_audit():
    print("\n=== 7. CONTINUAL OBSERVATION BINARY TREE AUDIT ===")
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "reuse_test.db"

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
        np.testing.assert_array_equal(res_a.vector, res_b.vector)
        print("  - Bitwise identical noisy node payload across different recipients: PASS")

        # 3. Simulate process restart by closing state manager and reopening fresh connection
        del engine
        del state_mgr

        fresh_state_mgr = ContinualStateManager(db_path=db_path)
        fresh_engine = ContinualQueryEngine(state_manager=fresh_state_mgr, composition_mode="per_recipient")

        # Query with interactions=None to guarantee no re-computation
        res_restart = fresh_engine.query_interval(
            cfg, 1, 4, recipient="ProviderA", interactions=None, public_user_roster=None
        )
        np.testing.assert_array_equal(res_a.vector, res_restart.vector)
        print("  - Bitwise identical payload across process restart without raw data: PASS")
        print("Continual Observation Binary Tree: PASS")


if __name__ == "__main__":
    run_replay_audit()
    run_continual_observation_audit()
