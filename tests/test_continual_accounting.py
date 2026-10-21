"""Unit tests for persistent noisy nodes and level exposure accounting (Commit 3)."""

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import numpy as np
import pandas as pd
import pytest

from private_llm_analytics.continual.mechanism import (
    calibrate_tree_gaussian_sigma,
    compute_level_rdp_vector,
    evaluate_rdp_epsilon,
)
from private_llm_analytics.continual.models import (
    ContinualNoiseConfig,
    ContinualStreamConfig,
)
from private_llm_analytics.continual.query import (
    ContinualQueryEngine,
    ContinualQueryResult,
)
from private_llm_analytics.continual.state import ContinualStateManager
from private_llm_analytics.continual.tree import BinaryTimeTree
from private_llm_analytics.privacy.exceptions import (
    PrivacyBudgetExceededError,
)


def _build_test_stream_config(tmp_path: Path, epoch_days: int = 30) -> Tuple[ContinualStreamConfig, ContinualStateManager]:
    db_file = tmp_path / "test_state.db"
    state_mgr = ContinualStateManager(db_path=db_file)
    cfg = ContinualStreamConfig(
        stream_id="test_stream",
        epoch_id="epoch_001",
        epoch_start="2026-01-01",
        epoch_days=epoch_days,
        daily_l2_bound=10.0,
        epoch_group_l12_bound=30.0,
        noise=ContinualNoiseConfig(
            target_epoch_epsilon=2.0,
            calibration_delta=1e-6,
            sigma=None,  # Automatically calibrate to target_epoch_epsilon
        ),
    )
    return cfg, state_mgr


def _create_sample_interactions(epoch_days: int = 30) -> Tuple[pd.DataFrame, list]:
    roster = [f"user_{i}" for i in range(10)]
    records = []
    base_date = pd.to_datetime("2026-01-01")
    rng = np.random.default_rng(123)

    for uid in roster[:5]:
        for day in range(epoch_days):
            records.append({
                "user_id": uid,
                "timestamp": base_date + pd.Timedelta(days=day, hours=10),
                "provider": "provider_a",
                "task": "technical",
            })
    return pd.DataFrame(records), roster


def test_tree_sigma_monotonic_with_epoch_bound():
    """Verify sigma does not decrease when epoch bound C_epoch increases (Section 42)."""
    sig1 = calibrate_tree_gaussian_sigma(epoch_days=30, epoch_l12_bound=20.0, target_tree_epsilon=2.0)
    sig2 = calibrate_tree_gaussian_sigma(epoch_days=30, epoch_l12_bound=40.0, target_tree_epsilon=2.0)
    assert sig2 >= sig1


def test_tree_sigma_monotonic_with_epsilon():
    """Verify sigma does not increase when target epsilon increases (Section 42)."""
    sig1 = calibrate_tree_gaussian_sigma(epoch_days=30, epoch_l12_bound=30.0, target_tree_epsilon=1.0)
    sig2 = calibrate_tree_gaussian_sigma(epoch_days=30, epoch_l12_bound=30.0, target_tree_epsilon=3.0)
    assert sig2 <= sig1


def test_tree_composed_rdp_matches_target():
    """Verify calibrated sigma achieves effective epsilon <= target_tree_epsilon (Section 42)."""
    target_eps = 2.0
    delta = 1e-6
    epoch_days = 90
    bound = 40.0

    sigma = calibrate_tree_gaussian_sigma(
        epoch_days=epoch_days,
        epoch_l12_bound=bound,
        target_tree_epsilon=target_eps,
        calibration_delta=delta,
    )

    tree = BinaryTimeTree(epoch_days=epoch_days)
    level_rdp = compute_level_rdp_vector(epoch_l12_bound=bound, sigma=sigma)
    total_tree_rdp = {k: v * tree.num_levels for k, v in level_rdp.items()}

    effective_eps = evaluate_rdp_epsilon(total_tree_rdp, delta=delta)
    assert effective_eps <= target_eps + 1e-6
    assert abs(effective_eps - target_eps) < 0.05


def test_same_node_same_noise_across_recipients(tmp_path: Path):
    """Security invariant (Section 23, 78):
    All recipients querying the same node receive the exact bitwise identical noisy payload.
    Prevents noise averaging attacks across colluding recipients.
    """
    cfg, state_mgr = _build_test_stream_config(tmp_path)
    engine = ContinualQueryEngine(state_manager=state_mgr, composition_mode="per_recipient")
    df, roster = _create_sample_interactions()

    # Recipient A queries day [1, 2]
    res_a = engine.query_interval(cfg, 1, 2, recipient="ProviderA", interactions=df, public_user_roster=roster)

    # Recipient B queries day [1, 2]
    res_b = engine.query_interval(cfg, 1, 2, recipient="ProviderB", interactions=df, public_user_roster=roster)

    # Both must use the exact same node (level 1, node 0)
    assert res_a.nodes_used == res_b.nodes_used
    # Vectors must be bitwise identical!
    np.testing.assert_array_equal(res_a.vector, res_b.vector)


def test_global_external_level_charge_once(tmp_path: Path):
    """Under GLOBAL_EXTERNAL: first query charges levels; subsequent external queries reuse with zero cost."""
    cfg, state_mgr = _build_test_stream_config(tmp_path)
    engine = ContinualQueryEngine(state_manager=state_mgr, composition_mode="global_external")
    df, roster = _create_sample_interactions()

    # Query 1: Provider A queries days [1, 4] -> uses level 2, node 0
    res1 = engine.query_interval(cfg, 1, 4, recipient="ProviderA", interactions=df, public_user_roster=roster)
    assert res1.newly_exposed_levels == [2]
    assert res1.reused_levels == []

    # Query 2: Provider B queries days [1, 4] -> reuses level 2 with zero new level cost!
    res2 = engine.query_interval(cfg, 1, 4, recipient="ProviderB", interactions=df, public_user_roster=roster)
    assert res2.newly_exposed_levels == []
    assert res2.reused_levels == [2]

    # Query 3: Provider B queries days [5, 6] -> uses level 1, node 2 (level 1 newly exposed)
    res3 = engine.query_interval(cfg, 5, 6, recipient="ProviderB", interactions=df, public_user_roster=roster)
    assert res3.newly_exposed_levels == [1]
    assert res3.reused_levels == []


def test_per_recipient_level_charge_once_each(tmp_path: Path):
    """Under PER_RECIPIENT: each recipient pays for levels once, but receives identical noise."""
    cfg, state_mgr = _build_test_stream_config(tmp_path)
    engine = ContinualQueryEngine(state_manager=state_mgr, composition_mode="per_recipient")
    df, roster = _create_sample_interactions()

    # Recipient A queries days [1, 2] -> level 1 newly exposed for A
    res_a1 = engine.query_interval(cfg, 1, 2, recipient="UserA", interactions=df, public_user_roster=roster)
    assert res_a1.budget_scope_id == "recipient:UserA"
    assert res_a1.newly_exposed_levels == [1]

    # Recipient A re-queries days [1, 2] -> zero newly exposed
    res_a2 = engine.query_interval(cfg, 1, 2, recipient="UserA", interactions=df, public_user_roster=roster)
    assert res_a2.newly_exposed_levels == []
    assert res_a2.reused_levels == [1]

    # Recipient B queries days [1, 2] -> level 1 newly exposed for B
    res_b = engine.query_interval(cfg, 1, 2, recipient="UserB", interactions=df, public_user_roster=roster)
    assert res_b.budget_scope_id == "recipient:UserB"
    assert res_b.newly_exposed_levels == [1]

    # Identical noise returned
    np.testing.assert_array_equal(res_a1.vector, res_b.vector)


def test_collusion_group_level_charge_once(tmp_path: Path):
    """Under COLLUSION_GROUP: members in the same group share level exposures."""
    cfg, state_mgr = _build_test_stream_config(tmp_path)
    groups = {"alpha_consortium": ["Member1", "Member2"]}
    engine = ContinualQueryEngine(
        state_manager=state_mgr,
        composition_mode="collusion_group",
        collusion_groups=groups,
    )
    df, roster = _create_sample_interactions()

    # Member 1 queries days [1, 4]
    res1 = engine.query_interval(cfg, 1, 4, recipient="Member1", interactions=df, public_user_roster=roster)
    assert res1.budget_scope_id == "collusion:alpha_consortium"
    assert res1.newly_exposed_levels == [2]

    # Member 2 queries days [1, 4] -> already exposed to the group!
    res2 = engine.query_interval(cfg, 1, 4, recipient="Member2", interactions=df, public_user_roster=roster)
    assert res2.budget_scope_id == "collusion:alpha_consortium"
    assert res2.newly_exposed_levels == []
    assert res2.reused_levels == [2]


def test_multi_level_atomic_budget_reservation(tmp_path: Path):
    """If projected epsilon exceeds epsilon_cap, query fails without partial level debit (Section 31, 32)."""
    cfg, state_mgr = _build_test_stream_config(tmp_path)
    # Set tight epsilon_cap so multiple levels exceed it
    engine = ContinualQueryEngine(
        state_manager=state_mgr,
        composition_mode="per_recipient",
        epsilon_cap=0.1,  # Very tight cap
    )
    df, roster = _create_sample_interactions()

    # Query days [1, 7] requires multiple levels (e.g. 0, 1, 2)
    with pytest.raises(PrivacyBudgetExceededError):
        engine.query_interval(cfg, 1, 7, recipient="PoorUser", interactions=df, public_user_roster=roster)

    # Verify atomic all-or-nothing: NO levels were committed for PoorUser
    exposed = state_mgr.get_exposed_levels("recipient:PoorUser", cfg.stream_fingerprint, cfg.epoch_id)
    assert len(exposed) == 0


def test_query_has_no_rng_and_no_raw_scan_when_materialized(tmp_path: Path):
    """Verify that once nodes are materialized, query engine uses NO RNG and does NOT access raw data (Section 45, 99)."""
    cfg, state_mgr = _build_test_stream_config(tmp_path)
    engine = ContinualQueryEngine(state_manager=state_mgr, composition_mode="global_external")
    df, roster = _create_sample_interactions()

    # 1. First query materializes nodes
    res1 = engine.query_interval(cfg, 1, 7, recipient="ProviderA", interactions=df, public_user_roster=roster)

    # 2. Repeated query with interactions=None and public_user_roster=None
    # Must succeed without accessing raw data!
    res2 = engine.query_interval(cfg, 1, 7, recipient="ProviderA", interactions=None, public_user_roster=None)

    # Results must be strictly identical
    np.testing.assert_array_equal(res1.vector, res2.vector)
    assert res2.newly_exposed_levels == []


def test_cross_epoch_query(tmp_path: Path):
    """Verify cross-epoch query decomposes correctly and combines noisy vectors (Section 103)."""
    db_file = tmp_path / "test_cross_state.db"
    state_mgr = ContinualStateManager(db_path=db_file)

    cfg1 = ContinualStreamConfig(
        stream_id="stream_ep1",
        epoch_id="epoch_001",
        epoch_start="2026-01-01",
        epoch_days=30,
        epoch_group_l12_bound=10.0,
        noise=ContinualNoiseConfig(target_epoch_epsilon=1.0, sigma=None),
    )
    cfg2 = ContinualStreamConfig(
        stream_id="stream_ep2",
        epoch_id="epoch_002",
        epoch_start="2026-01-31",
        epoch_days=30,
        epoch_group_l12_bound=10.0,
        noise=ContinualNoiseConfig(target_epoch_epsilon=1.0, sigma=None),
    )

    engine = ContinualQueryEngine(state_manager=state_mgr, composition_mode="global_external")
    df1, roster = _create_sample_interactions(30)
    df2, _ = _create_sample_interactions(30)

    # Query from Day 25 of Epoch 1 to Day 10 of Epoch 2
    res = engine.query_cross_epoch(
        epoch_1_config=cfg1,
        epoch_1_start_day=25,
        epoch_2_config=cfg2,
        epoch_2_end_day=10,
        recipient="ProviderA",
        interactions_epoch_1=df1,
        interactions_epoch_2=df2,
        public_user_roster=roster,
    )

    assert res.query_type == "CROSS_EPOCH"
    assert len(res.nodes_used) > 0
    assert res.vector.shape == (9,)
