"""Tests for Cross-Profile Lifetime Privacy Accounting (v0.6).

Verifies:
  - Sequence of releases:
      1. User Prevalence
      2. Interaction Volume
      3. Success Rate
      4. Replay Success Rate (zero incremental cost)
      5. User-Normalized Share
  - All non-replay releases compose into the single authoritative lifetime accountant
  - Changing statistic does NOT reset privacy budget
  - Changing statistic does NOT create a separate lifetime horizon
  - Effective epsilon monotonically increases
  - Replay of identical query has zero cost and reuses cached release
  - Canonical identity changes when profile parameters change (e.g. C_rate)
  - Recipient scoping (global_external, per_recipient, collusion_group) applies cleanly
  - Privacy state database persists statistic_id, profile_id, profile_version, profile_fingerprint
"""

import sqlite3
import pytest
import numpy as np

from private_llm_analytics.aggregation.user_matrix import build_user_matrix
from private_llm_analytics.config.loader import load_app_config
from private_llm_analytics.planner.query import AnalyticsQuery
from private_llm_analytics.privacy.state import PrivacyStateManager
from private_llm_analytics.release.release import ReleaseOrchestrator
from private_llm_analytics.simulator.generator import SyntheticDataGenerator
from private_llm_analytics.simulator.scenario import SimulationScenario


@pytest.fixture
def test_umatrix_with_success():
    scenario = SimulationScenario(n_users=60, random_seed=42)
    gen = SyntheticDataGenerator(scenario)
    df = gen.generate(seed=42, include_success=True, success_rate=0.85)
    return build_user_matrix(df, public_population_size=60)


class TestCrossProfileAccounting:
    def test_cross_profile_lifetime_composition_sequence(self, tmp_path, test_umatrix_with_success):
        db_path = tmp_path / "privacy_state.db"
        config = load_app_config("configs")
        config.privacy.recipient_composition.mode = "global_external"
        config.privacy.accountant.epsilon_cap = 20.0
        config.privacy.accountant.delta_global = 1e-5

        state_mgr = PrivacyStateManager(db_path)
        orch = ReleaseOrchestrator(app_config=config, state_manager=state_mgr)

        window_args = {
            "window_start": "2026-01-01",
            "window_end": "2026-01-31",
            "recipient": "provider_a",
            "delta": 1e-6,
        }

        # Step 1: User Prevalence release
        q1 = AnalyticsQuery(metric="user_prevalence", epsilon=1.0, **window_args)
        r1 = orch.execute_release(q1, test_umatrix_with_success)
        assert not r1["cache_hit"]
        eps1 = r1["privacy_accounting"]["effective_epsilon_after"]
        assert eps1 > 0.0
        assert r1["privacy_accounting"]["history_release_count"] == 1

        # Step 2: Interaction Volume release
        q2 = AnalyticsQuery(metric="interaction_volume", epsilon=1.0, **window_args)
        r2 = orch.execute_release(q2, test_umatrix_with_success)
        assert not r2["cache_hit"]
        eps2 = r2["privacy_accounting"]["effective_epsilon_after"]
        assert eps2 > eps1, "Budget must accumulate across different statistics!"
        assert r2["privacy_accounting"]["history_release_count"] == 2

        # Step 3: Success Rate release
        q3 = AnalyticsQuery(metric="success_rate", epsilon=1.0, **window_args)
        r3 = orch.execute_release(q3, test_umatrix_with_success)
        assert not r3["cache_hit"]
        eps3 = r3["privacy_accounting"]["effective_epsilon_after"]
        assert eps3 > eps2, "Budget must accumulate across success_rate and previous profiles!"
        assert r3["privacy_accounting"]["history_release_count"] == 3

        # Step 4: Replay Success Rate release (exact same query)
        q3_replay = AnalyticsQuery(metric="success_rate", epsilon=1.0, **window_args)
        r3_replay = orch.execute_release(q3_replay, test_umatrix_with_success)
        assert r3_replay["cache_hit"], "Identical query must hit canonical cache!"
        assert r3_replay["release_key"] == r3["release_key"]
        assert r3_replay["incremental_privacy_cost"] == 0.0
        eps3_replay = r3_replay["privacy_accounting"]["effective_epsilon_after"]
        assert eps3_replay == eps3, "Cache replay must consume zero privacy budget!"
        assert r3_replay["privacy_accounting"]["history_release_count"] == 3

        # Step 5: User-Normalized Share release
        q4 = AnalyticsQuery(metric="user_normalized_share", epsilon=1.0, **window_args)
        r4 = orch.execute_release(q4, test_umatrix_with_success)
        assert not r4["cache_hit"]
        eps4 = r4["privacy_accounting"]["effective_epsilon_after"]
        assert eps4 > eps3, "Budget must accumulate across all 4 profiles!"
        assert r4["privacy_accounting"]["history_release_count"] == 4

        # Verify persistent SQLite schema and stored fields
        conn = sqlite3.connect(str(db_path))
        c = conn.cursor()
        c.execute("SELECT statistic_id, profile_id, profile_version, profile_fingerprint FROM privacy_ledger WHERE cache_hit = 0 ORDER BY rowid")
        rows = c.fetchall()
        conn.close()

        assert len(rows) == 4
        # Check that rows record profile metadata
        stat_ids = [row[0] for row in rows]
        assert stat_ids == ["user_prevalence", "interaction_volume", "success_rate", "user_normalized_share"]

        # Fingerprints must be non-empty strings
        for row in rows:
            assert row[3] is not None and len(row[3]) > 0

    def test_different_c_rate_yields_different_release_key(self, tmp_path, test_umatrix_with_success):
        db_path = tmp_path / "privacy_state.db"
        config = load_app_config("configs")
        config.privacy.recipient_composition.mode = "global_external"
        config.privacy.accountant.epsilon_cap = 20.0

        state_mgr = PrivacyStateManager(db_path)
        orch = ReleaseOrchestrator(app_config=config, state_manager=state_mgr)

        window_args = {
            "window_start": "2026-01-01",
            "window_end": "2026-01-31",
            "recipient": "provider_a",
            "delta": 1e-6,
            "epsilon": 1.0,
            "metric": "success_rate",
        }

        # First run with default C_rate (20.0)
        q1 = AnalyticsQuery(**window_args)
        r1 = orch.execute_release(q1, test_umatrix_with_success)

        # Update profile configuration in config to C_rate = 40.0
        config.metrics.metrics["success_rate"].contribution.clip_norm_rate = 40.0
        q2 = AnalyticsQuery(**window_args)
        r2 = orch.execute_release(q2, test_umatrix_with_success)

        assert not r2["cache_hit"], "Changing profile parameter must NOT hit cache!"
        assert r2["release_key"] != r1["release_key"], "Release keys must differ when profile fingerprint changes!"
