"""Unit and integration tests for Differentially Private Top-K Tasks Profile (v0.8)."""

import hashlib
import numpy as np
import pandas as pd
import pytest

from private_llm_analytics.aggregation.grouping import build_cell_definitions
from private_llm_analytics.aggregation.user_matrix import UserMatrix, build_user_matrix
from private_llm_analytics.metrics.catalog import MetricCatalog
from private_llm_analytics.metrics.top_k_tasks import TopKTasksMetric
from private_llm_analytics.privacy.exceptions import InvalidStatisticProfileConfiguration
from private_llm_analytics.statistics.registry import StatisticProfileRegistry
from private_llm_analytics.statistics.top_k_tasks import TopKTasksProfile


def make_test_user_matrix(n_tasks: int = 15, n_users: int = 50) -> UserMatrix:
    providers = ["prov_a", "prov_b"]
    tasks = [f"task_{i:02d}" for i in range(n_tasks)]
    cell_names, cell_map = build_cell_definitions(providers, tasks)

    rng = np.random.default_rng(20261027)
    # Generate Zipf-like task distribution: lower indexed tasks have higher counts
    raw_counts = np.zeros((n_users, len(cell_names)), dtype=np.float64)
    for u in range(n_users):
        # Sample number of tasks user is active in
        active_task_indices = rng.choice(n_tasks, size=rng.integers(1, min(12, n_tasks)), replace=False)
        for t_idx in active_task_indices:
            p_idx = rng.choice(2)
            c_idx = p_idx * n_tasks + t_idx
            raw_counts[u, c_idx] = rng.integers(1, 10)

    return UserMatrix(
        matrix=raw_counts,
        user_ids=[f"user_{u}" for u in range(n_users)],
        providers=providers,
        tasks=tasks,
        cell_names=cell_names,
        cell_map=cell_map,
        public_population_size=1000,
    )


class TestTopKTasksMathematics:
    def test_sensitivity_derivation(self):
        p1 = TopKTasksProfile(k=5, max_tasks_per_user=5)
        assert np.isclose(p1.sensitivity(), np.sqrt(5.0))

        p2 = TopKTasksProfile(k=10, max_tasks_per_user=8)
        assert np.isclose(p2.sensitivity(), np.sqrt(8.0))

    def test_invalid_parameters_fail_closed(self):
        with pytest.raises(InvalidStatisticProfileConfiguration, match="k must be strictly positive"):
            TopKTasksProfile(k=0, max_tasks_per_user=5)

        with pytest.raises(InvalidStatisticProfileConfiguration, match="max_tasks_per_user must be strictly positive"):
            TopKTasksProfile(k=5, max_tasks_per_user=0)

        with pytest.raises(InvalidStatisticProfileConfiguration, match="threshold cannot be negative"):
            TopKTasksProfile(k=5, max_tasks_per_user=5, threshold=-1.0)

    def test_user_contribution_bounding(self):
        umatrix = make_test_user_matrix(n_tasks=20, n_users=40)
        profile = TopKTasksProfile(k=5, max_tasks_per_user=4)

        contributions = profile.build_user_contributions(umatrix)
        assert contributions.shape == (40, 20)

        # Every user row must have sum <= max_tasks_per_user and L2 norm <= sqrt(max_tasks_per_user)
        row_sums = np.sum(contributions, axis=1)
        row_l2_norms = np.linalg.norm(contributions, ord=2, axis=1)

        assert np.all(row_sums <= 4.0 + 1e-12)
        assert np.all(row_l2_norms <= np.sqrt(4.0) + 1e-12)
        # Verify values are binary
        assert np.all(np.isin(np.unique(contributions), [0.0, 1.0]))

    def test_mechanism_plan_and_rdp_cost(self):
        profile = TopKTasksProfile(k=5, max_tasks_per_user=5)
        plan = profile.build_mechanism_plan(epsilon=1.0, delta=1e-6, dimension=15)

        assert plan.mechanism_name == "gaussian"
        assert np.isclose(plan.sensitivity, np.sqrt(5.0))
        assert plan.noise_scale > 0

        cost = profile.privacy_cost_descriptor(plan)
        assert cost.accounting_family == "rdp"
        assert len(cost.orders) > 0
        assert len(cost.costs) == len(cost.orders)
        assert all(c > 0 for c in cost.costs)


class TestTopKTasksPostProcessing:
    def test_top_k_ranking_and_selection(self):
        tasks = [f"t_{i}" for i in range(6)]
        cell_names, cell_map = build_cell_definitions(["p1"], tasks)
        umatrix = UserMatrix(
            matrix=np.ones((5, 6)),
            user_ids=[f"u_{i}" for i in range(5)],
            providers=["p1"],
            tasks=tasks,
            cell_names=cell_names,
            cell_map=cell_map,
            public_population_size=100,
        )

        profile = TopKTasksProfile(k=3, max_tasks_per_user=5)
        # Mock noisy vector with known values
        dp_vec = np.array([12.5, 55.0, -3.0, 32.0, 8.0, 40.0])
        # Sorted order: t_1 (55.0), t_5 (40.0), t_3 (32.0), t_0 (12.5), t_4 (8.0), t_2 (0.0 clamped)

        res = profile.post_process(dp_vec, umatrix)

        assert len(res["top_k"]) == 3
        assert res["selected_tasks"] == ["t_1", "t_5", "t_3"]
        assert res["top_k"][0]["task"] == "t_1"
        assert res["top_k"][0]["noisy_count"] == 55.0
        assert np.isclose(res["top_k"][0]["estimated_prevalence"], 0.55)
        assert res["top_k"][1]["task"] == "t_5"
        assert res["top_k"][2]["task"] == "t_3"

    def test_threshold_filtering(self):
        tasks = [f"t_{i}" for i in range(4)]
        cell_names, cell_map = build_cell_definitions(["p1"], tasks)
        umatrix = UserMatrix(
            matrix=np.ones((2, 4)),
            user_ids=["u1", "u2"],
            providers=["p1"],
            tasks=tasks,
            cell_names=cell_names,
            cell_map=cell_map,
            public_population_size=50,
        )

        profile = TopKTasksProfile(k=4, max_tasks_per_user=3, threshold=20.0)
        dp_vec = np.array([50.0, 15.0, 5.0, 25.0])

        res = profile.post_process(dp_vec, umatrix)
        # Only t_0 (50.0) and t_3 (25.0) exceed threshold 20.0
        assert len(res["top_k"]) == 2
        assert res["selected_tasks"] == ["t_0", "t_3"]


class TestTopKTasksIntegration:
    def test_registry_resolution(self):
        profile = StatisticProfileRegistry.get("top_k_tasks")
        assert isinstance(profile, TopKTasksProfile)
        assert "top_k_tasks" in StatisticProfileRegistry.list_statistics()

    def test_metric_catalog_resolution(self):
        catalog = MetricCatalog()
        metric = catalog.get("top_k_tasks")
        assert isinstance(metric, TopKTasksMetric)

    def test_execute_release_end_to_end(self):
        umatrix = make_test_user_matrix(n_tasks=10, n_users=30)
        profile = TopKTasksProfile(k=5, max_tasks_per_user=3)

        result = profile.execute_release(umatrix, epsilon=1.0, delta=1e-6)
        assert result.metric_name == "top_k_tasks"
        assert len(result.dp_vector) == 10
        assert "top_k" in result.post_processed
        assert len(result.post_processed["top_k"]) <= 5
        assert len(result.post_processed["selected_tasks"]) <= 5

    def test_canonical_fingerprint_invariance(self):
        p1 = TopKTasksProfile(k=5, max_tasks_per_user=3)
        p2 = TopKTasksProfile(k=5, max_tasks_per_user=3)
        p3 = TopKTasksProfile(k=10, max_tasks_per_user=3)

        assert p1.canonical_profile_fingerprint() == p2.canonical_profile_fingerprint()
        assert p1.canonical_profile_fingerprint() != p3.canonical_profile_fingerprint()
