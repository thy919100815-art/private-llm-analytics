"""Unit tests for user-level temporal contribution bounding (Section 14, 15, 16)."""

import math
from typing import List
import numpy as np
import pandas as pd
import pytest

from private_llm_analytics.aggregation.grouping import (
    DEFAULT_PROVIDERS,
    DEFAULT_TASKS,
)
from private_llm_analytics.continual.contributions import (
    build_temporal_user_contributions,
)
from private_llm_analytics.continual.models import (
    ContinualNoiseConfig,
    ContinualStreamConfig,
    ContinualTreeConfig,
)
from private_llm_analytics.privacy.exceptions import (
    PrivacyConfigurationError,
    ProductionSafetyConfigurationError,
)


def _generate_synthetic_interactions(
    user_ids: List[str],
    epoch_start: str,
    epoch_days: int,
    pattern: str = "uniform",
    seed: int = 42,
) -> pd.DataFrame:
    """Helper to generate interaction DataFrame with controllable temporal pattern."""
    rng = np.random.default_rng(seed)
    records = []
    base_date = pd.to_datetime(epoch_start)

    for uid in user_ids:
        if pattern == "zero":
            continue
        elif pattern == "same_day":
            day = 0
            for _ in range(rng.integers(10, 50)):
                records.append({
                    "user_id": uid,
                    "timestamp": base_date + pd.Timedelta(days=day, hours=int(rng.integers(0, 23))),
                    "provider": rng.choice(DEFAULT_PROVIDERS),
                    "task": rng.choice(DEFAULT_TASKS),
                })
        elif pattern == "concentrated":
            for day in range(min(3, epoch_days)):
                for _ in range(rng.integers(20, 100)):
                    records.append({
                        "user_id": uid,
                        "timestamp": base_date + pd.Timedelta(days=day, hours=int(rng.integers(0, 23))),
                        "provider": rng.choice(DEFAULT_PROVIDERS),
                        "task": rng.choice(DEFAULT_TASKS),
                    })
        elif pattern == "every_day":
            for day in range(epoch_days):
                for _ in range(rng.integers(1, 10)):
                    records.append({
                        "user_id": uid,
                        "timestamp": base_date + pd.Timedelta(days=day, hours=int(rng.integers(0, 23))),
                        "provider": rng.choice(DEFAULT_PROVIDERS),
                        "task": rng.choice(DEFAULT_TASKS),
                    })
        elif pattern == "uniform":
            for day in range(epoch_days):
                for _ in range(rng.integers(2, 6)):
                    records.append({
                        "user_id": uid,
                        "timestamp": base_date + pd.Timedelta(days=day, hours=int(rng.integers(0, 23))),
                        "provider": rng.choice(DEFAULT_PROVIDERS),
                        "task": rng.choice(DEFAULT_TASKS),
                    })
        elif pattern == "bursty":
            active_days = rng.choice(epoch_days, size=max(1, epoch_days // 4), replace=False)
            for day in active_days:
                for _ in range(rng.integers(15, 80)):
                    records.append({
                        "user_id": uid,
                        "timestamp": base_date + pd.Timedelta(days=int(day), hours=int(rng.integers(0, 23))),
                        "provider": rng.choice(DEFAULT_PROVIDERS),
                        "task": rng.choice(DEFAULT_TASKS),
                    })

    return pd.DataFrame(records)


def test_daily_l2_bound():
    """Verify ||z_{u, t}||_2 <= C_day for all users and days."""
    roster = [f"user_{i}" for i in range(20)]
    df = _generate_synthetic_interactions(roster, "2026-01-01", 30, pattern="bursty")

    c_day = 8.0
    c_epoch = 50.0
    contribs, diag, _ = build_temporal_user_contributions(
        interactions=df,
        public_user_roster=roster,
        epoch_start="2026-01-01",
        epoch_days=30,
        daily_l2_bound=c_day,
        epoch_l12_bound=c_epoch,
    )

    daily_l2_norms = np.linalg.norm(contribs, axis=2)  # (n_users, 30)
    assert np.all(daily_l2_norms <= c_day + 1e-10)
    assert diag.raw_total >= diag.daily_clipped_total >= diag.epoch_clipped_total


def test_epoch_group_l12_bound():
    """Verify sum_t ||z_{u, t}||_2 <= C_epoch for all users."""
    roster = [f"user_{i}" for i in range(25)]
    df = _generate_synthetic_interactions(roster, "2026-01-01", 60, pattern="every_day")

    c_day = 15.0
    c_epoch = 35.0
    contribs, diag, _ = build_temporal_user_contributions(
        interactions=df,
        public_user_roster=roster,
        epoch_start="2026-01-01",
        epoch_days=60,
        daily_l2_bound=c_day,
        epoch_l12_bound=c_epoch,
    )

    daily_l2_norms = np.linalg.norm(contribs, axis=2)  # (n_users, 60)
    epoch_l12_sums = np.sum(daily_l2_norms, axis=1)    # (n_users,)

    assert np.all(epoch_l12_sums <= c_epoch + 1e-10)
    assert diag.epoch_clipped_total <= diag.daily_clipped_total


def test_zero_user_stays_zero():
    """Users in roster with zero activity must have exactly zero vectors."""
    roster = ["active_user", "zero_user_1", "zero_user_2"]
    df = pd.DataFrame([
        {
            "user_id": "active_user",
            "timestamp": "2026-01-01 10:00:00",
            "provider": "provider_a",
            "task": "technical",
        }
    ])

    contribs, _, _ = build_temporal_user_contributions(
        interactions=df,
        public_user_roster=roster,
        epoch_start="2026-01-01",
        epoch_days=7,
        daily_l2_bound=10.0,
        epoch_l12_bound=30.0,
    )

    assert contribs.shape == (3, 7, 9)
    # active user has non-zero
    assert np.sum(contribs[0]) > 0
    # zero users must be strictly 0.0
    assert np.all(contribs[1] == 0.0)
    assert np.all(contribs[2] == 0.0)


def test_target_granularity_before_temporal_clipping():
    """Verify target granularity projection occurs before clipping."""
    roster = ["user_1"]
    df = pd.DataFrame([
        {
            "user_id": "user_1",
            "timestamp": "2026-01-01 10:00:00",
            "provider": "provider_a",
            "task": "technical",
        },
        {
            "user_id": "user_1",
            "timestamp": "2026-01-01 11:00:00",
            "provider": "provider_b",
            "task": "analysis",
        },
    ])

    # Granularity: overall_1_cell -> dimension 1
    contribs_1, _, names_1 = build_temporal_user_contributions(
        interactions=df,
        public_user_roster=roster,
        epoch_start="2026-01-01",
        epoch_days=5,
        granularity="overall_1_cell",
        daily_l2_bound=10.0,
        epoch_l12_bound=20.0,
    )
    assert contribs_1.shape == (1, 5, 1)
    assert names_1 == ["overall_total"]
    assert contribs_1[0, 0, 0] == 2.0

    # Granularity: provider_3_cell -> dimension 3
    contribs_3, _, names_3 = build_temporal_user_contributions(
        interactions=df,
        public_user_roster=roster,
        epoch_start="2026-01-01",
        epoch_days=5,
        granularity="provider_3_cell",
        daily_l2_bound=10.0,
        epoch_l12_bound=20.0,
    )
    assert contribs_3.shape == (1, 5, 3)
    assert len(names_3) == 3


def test_padding_days_zero():
    """Days beyond epoch_days or without events have zero contributions."""
    roster = ["u1"]
    df = pd.DataFrame([
        {
            "user_id": "u1",
            "timestamp": "2026-01-01 10:00:00",
            "provider": "provider_a",
            "task": "technical",
        }
    ])

    contribs, _, _ = build_temporal_user_contributions(
        interactions=df,
        public_user_roster=roster,
        epoch_start="2026-01-01",
        epoch_days=10,
        daily_l2_bound=10.0,
        epoch_l12_bound=30.0,
    )
    assert contribs.shape == (1, 10, 9)
    # Day 0 has activity, days 1..9 are strictly zero
    assert np.sum(contribs[0, 0]) > 0
    assert np.all(contribs[0, 1:] == 0.0)


def test_unsupported_metric_fail_closed():
    """Verify unsupported metrics fail closed with ProductionSafetyConfigurationError."""
    roster = ["user_1"]
    df = pd.DataFrame()

    with pytest.raises(ProductionSafetyConfigurationError):
        build_temporal_user_contributions(
            interactions=df,
            public_user_roster=roster,
            epoch_start="2026-01-01",
            epoch_days=10,
            metric="user_prevalence",
        )

    with pytest.raises(ProductionSafetyConfigurationError):
        build_temporal_user_contributions(
            interactions=df,
            public_user_roster=roster,
            epoch_start="2026-01-01",
            epoch_days=10,
            metric="user_normalized_share",
        )


@pytest.mark.parametrize("T", [7, 30, 90, 128])
@pytest.mark.parametrize("pattern", ["same_day", "every_day", "concentrated", "uniform", "bursty"])
def test_per_level_user_l2_sensitivity_bound(T: int, pattern: str):
    """Rigorous mathematical proof verification (Section 7, 16):

    For any binary tree with tree_size = 2^k >= T, on ANY level l,
    the temporal node intervals I_{l, j} are pairwise disjoint.
    User u's contribution to node j is v_{u, l, j} = sum_{t in I_{l, j}} z_{u, t}.
    Concatenating all nodes on level l: V_{u, l} = [v_{u, l, 0}, v_{u, l, 1}, ...].
    Theorem: ||V_{u, l}||_2 <= sum_j ||v_{u, l, j}||_2 <= sum_t ||z_{u, t}||_2 <= C_epoch.
    """
    roster = [f"user_{i}" for i in range(5)]
    df = _generate_synthetic_interactions(roster, "2026-01-01", T, pattern=pattern, seed=T + len(pattern))

    c_day = 12.0
    c_epoch = 30.0

    contribs, _, _ = build_temporal_user_contributions(
        interactions=df,
        public_user_roster=roster,
        epoch_start="2026-01-01",
        epoch_days=T,
        daily_l2_bound=c_day,
        epoch_l12_bound=c_epoch,
    )

    # Next power of two for binary tree
    tree_size = 1 if T == 1 else 2 ** int(math.ceil(math.log2(T)))
    num_levels = int(math.log2(tree_size)) + 1

    # Pad contributions up to tree_size with zeros: (n_users, tree_size, dim)
    n_users, _, dim = contribs.shape
    padded = np.zeros((n_users, tree_size, dim), dtype=np.float64)
    padded[:, :T, :] = contribs

    # Test each level l in 0 .. num_levels - 1
    for level in range(num_levels):
        block_size = 2 ** level
        num_nodes = tree_size // block_size

        for u in range(n_users):
            node_vectors = []
            for j in range(num_nodes):
                start_idx = j * block_size
                end_idx = (j + 1) * block_size
                # Sum of z_{u, t} over interval I_{l, j}
                node_v = np.sum(padded[u, start_idx:end_idx, :], axis=0)  # (dim,)
                node_vectors.append(node_v)

            # Concatenated joint level vector V_{u, l}
            concat_v = np.concatenate(node_vectors)  # (num_nodes * dim,)
            level_l2_norm = float(np.linalg.norm(concat_v))

            # Must strictly satisfy ||V_{u, l}||_2 <= C_epoch
            assert level_l2_norm <= c_epoch + 1e-9, (
                f"Violation on level {level} (block_size={block_size}) for user {u}: "
                f"||V_{{u, {level}}}||_2 = {level_l2_norm} > C_epoch = {c_epoch}"
            )
