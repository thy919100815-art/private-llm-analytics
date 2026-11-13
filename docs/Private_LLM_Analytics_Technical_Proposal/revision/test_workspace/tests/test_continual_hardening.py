"""Unit tests for continual observation security hardening and fail-closed guards (Section 89)."""

from pathlib import Path
import sqlite3
import numpy as np
import pandas as pd
import pytest

from private_llm_analytics.continual.models import (
    ContinualNoiseConfig,
    ContinualStreamConfig,
    ContinualTreeConfig,
)
from private_llm_analytics.continual.query import ContinualQueryEngine
from private_llm_analytics.continual.state import ContinualStateManager
from private_llm_analytics.privacy.exceptions import (
    PrivacyConfigurationError,
    ProductionSafetyConfigurationError,
)


def test_continual_store_contains_no_exact_node_column(tmp_path: Path):
    """Security Invariant (Section 29): continual_nodes must strictly contain NO exact_node_value column."""
    db_file = tmp_path / "schema_check.db"
    state_mgr = ContinualStateManager(db_path=db_file)

    with sqlite3.connect(str(db_file)) as conn:
        cursor = conn.cursor()
        cursor.execute("PRAGMA table_info(continual_nodes)")
        columns = [row[1].lower() for row in cursor.fetchall()]

    for forbidden in ["exact", "raw", "exact_node", "exact_node_value", "ground_truth"]:
        assert forbidden not in columns, f"Forbidden exact column '{forbidden}' found in continual_nodes table!"


def test_late_data_cannot_mutate_released_node(tmp_path: Path):
    """Security Invariant (Section 36, 81): once a node is materialized/released, changing raw data cannot alter it."""
    db_file = tmp_path / "immutability.db"
    state_mgr = ContinualStateManager(db_path=db_file)
    engine = ContinualQueryEngine(state_manager=state_mgr, composition_mode="global_external")

    cfg = ContinualStreamConfig(
        stream_id="stream_mut",
        epoch_id="epoch_001",
        epoch_start="2026-01-01",
        epoch_days=10,
        noise=ContinualNoiseConfig(target_epoch_epsilon=2.0),
    )

    roster = ["u_0"]
    # Day 1: 1 interaction
    df1 = pd.DataFrame([{
        "user_id": "u_0",
        "timestamp": "2026-01-01 12:00:00",
        "provider": "provider_a",
        "task": "technical",
    }])

    # Query day [1, 2]
    res1 = engine.query_interval(cfg, 1, 2, recipient="Auditor", interactions=df1, public_user_roster=roster, finalized_up_to_day=10, current_wall_clock_day=20)

    # Later: attacker injects 10,000 interactions into Day 1 retroactively
    records_late = []
    for _ in range(100):
        records_late.append({
            "user_id": "u_0",
            "timestamp": "2026-01-01 12:00:00",
            "provider": "provider_a",
            "task": "technical",
        })
    df_tampered = pd.DataFrame(records_late)

    # Attempting to query with tampered interactions on sealed epoch must fail closed
    with pytest.raises(ProductionSafetyConfigurationError, match=r"(DATASET_MANIFEST_MISMATCH|already finalized partition|DATASET_PARTITION_MISMATCH)"):
        engine.query_interval(cfg, 1, 2, recipient="Auditor", interactions=df_tampered, public_user_roster=roster, finalized_up_to_day=10, current_wall_clock_day=20)

    # Query same day [1, 2] again using cached nodes
    res2 = engine.query_interval(cfg, 1, 2, recipient="Auditor", interactions=None, public_user_roster=None)

    # Output must remain 100% bitwise immutable!
    np.testing.assert_array_equal(res1.vector, res2.vector)


def test_tree_config_change_invalidates_stream():
    """Security Invariant (Section 62, 82): Any change to bounds, timezone, or topology alters stream fingerprint."""
    cfg_base = ContinualStreamConfig(
        stream_id="s1",
        epoch_id="ep1",
        daily_l2_bound=10.0,
        epoch_group_l12_bound=30.0,
        timezone="UTC",
    )
    fp_base = cfg_base.stream_fingerprint

    # Change epoch bound
    cfg_bound = ContinualStreamConfig(
        stream_id="s1",
        epoch_id="ep1",
        daily_l2_bound=10.0,
        epoch_group_l12_bound=35.0,
        timezone="UTC",
    )
    assert cfg_bound.stream_fingerprint != fp_base

    # Change daily bound
    cfg_daily = ContinualStreamConfig(
        stream_id="s1",
        epoch_id="ep1",
        daily_l2_bound=12.0,
        epoch_group_l12_bound=30.0,
        timezone="UTC",
    )
    assert cfg_daily.stream_fingerprint != fp_base

    # Change timezone
    cfg_tz = ContinualStreamConfig(
        stream_id="s1",
        epoch_id="ep1",
        daily_l2_bound=10.0,
        epoch_group_l12_bound=30.0,
        timezone="US/Eastern",
    )
    assert cfg_tz.stream_fingerprint != fp_base


def test_tree_requires_gaussian():
    """Security Invariant (Section 57): v0.4 continual observation requires Gaussian mechanism."""
    with pytest.raises(ProductionSafetyConfigurationError):
        ContinualNoiseConfig(mechanism="laplace")


def test_tree_topology_validation():
    """Section 60, 61: ContinualTreeConfig rejects unsupported topology."""
    with pytest.raises(ProductionSafetyConfigurationError):
        ContinualTreeConfig(topology="wavelet")

    with pytest.raises(ProductionSafetyConfigurationError):
        ContinualTreeConfig(padding="zero")


def test_corrupt_continual_rdp_fail_closed(tmp_path: Path):
    """Section 83: Corrupt or empty RDP JSON in continual exposures fails closed."""
    db_file = tmp_path / "corrupt_rdp.db"
    state_mgr = ContinualStateManager(db_path=db_file)

    cfg = ContinualStreamConfig(
        stream_id="stream_corrupt",
        epoch_id="epoch_001",
        epoch_days=10,
    )
    state_mgr.register_stream(cfg)

    # Insert a corrupt committed exposure
    with state_mgr._connection() as conn:
        conn.execute(
            """
            INSERT INTO continual_level_exposures (
                exposure_id, budget_scope_id, stream_fingerprint, epoch_id, level,
                rdp_cost_json, reservation_state, fencing_token, created_at, committed_at
            ) VALUES ('exp_bad', 'scope_bad', ?, 'epoch_001', 0, '', 'COMMITTED', 'fence', 'now', 'now')
            """,
            (cfg.stream_fingerprint,),
        )

    # Attempting to reserve further levels for this scope must fail closed
    with pytest.raises(ProductionSafetyConfigurationError):
        state_mgr.reserve_continual_level_exposures(
            budget_scope_id="scope_bad",
            stream_config=cfg,
            missing_levels=[1],
            sigma=5.0,
        )


def test_finalized_data_boundary_guard(tmp_path: Path):
    """Section 36, 101: querying non-finalized days raises ValueError."""
    db_file = tmp_path / "finalized_guard.db"
    state_mgr = ContinualStateManager(db_path=db_file)
    engine = ContinualQueryEngine(state_manager=state_mgr)

    cfg = ContinualStreamConfig(
        stream_id="stream_fin",
        epoch_id="epoch_001",
        epoch_days=30,
    )

    with pytest.raises(ValueError, match="DATA_NOT_FINALIZED"):
        engine.query_interval(
            stream_config=cfg,
            start_day=1,
            end_day=15,
            recipient="Auditor",
            finalized_up_to_day=10,  # Day 15 is not finalized yet
        )
