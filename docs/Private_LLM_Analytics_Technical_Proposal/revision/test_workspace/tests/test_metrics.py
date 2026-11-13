"""Unit tests for metric computations and error decomposition."""

import numpy as np
from private_llm_analytics.aggregation.user_matrix import UserMatrix, build_cell_definitions
from private_llm_analytics.mechanisms.gaussian import GaussianMechanism
from private_llm_analytics.metrics.interaction_volume import InteractionVolumeMetric
from private_llm_analytics.metrics.normalized_share import UserNormalizedShareMetric
from private_llm_analytics.metrics.prevalence import UserPrevalenceMetric


def get_test_matrix() -> UserMatrix:
    provs = ["provider_a", "provider_b", "provider_c"]
    tsks = ["technical", "document", "analysis"]
    cell_names, cell_map = build_cell_definitions(provs, tsks)
    # 4 users
    matrix = np.array([
        [10.0, 5.0, 0.0, 20.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 15.0, 15.0, 0.0, 0.0, 0.0],
        [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 50.0],
    ], dtype=np.float64)
    return UserMatrix(
        matrix=matrix,
        user_ids=["U1", "U2", "U3", "U4"],
        providers=provs,
        tasks=tsks,
        cell_names=cell_names,
        cell_map=cell_map,
    )


def test_metric_prevalence():
    umatrix = get_test_matrix()
    metric = UserPrevalenceMetric()
    exact_counts = metric.compute_exact(umatrix)
    assert exact_counts.shape == (9,)
    # Cell 0 ("provider_a:technical") is active for U1 and U3 -> count=2
    assert exact_counts[0] == 2.0


def test_metric_normalized_share():
    umatrix = get_test_matrix()
    metric = UserNormalizedShareMetric()
    exact_mean = metric.compute_exact_mean(umatrix)
    assert exact_mean.shape == (9,)
    assert abs(np.sum(exact_mean) - 1.0) <= 1e-10


def test_metric_interaction_volume_decomposition():
    umatrix = get_test_matrix()
    metric = InteractionVolumeMetric(clip_norm=10.0)

    raw_exact = metric.compute_raw_exact(umatrix)
    clipped_exact = metric.compute_clipped_exact(umatrix)
    assert raw_exact.shape == (9,)
    assert clipped_exact.shape == (9,)

    # U4 has count 50.0 in cell 8, L2 = 50.0 > 10.0 -> clipped to 10.0
    assert raw_exact[8] == 51.0 # U3(1) + U4(50)
    assert clipped_exact[8] < raw_exact[8]

    # Dummy DP vector
    dp_vec = clipped_exact + 2.0
    decomp = metric.decompose_error(dp_vec, umatrix)

    # Check total error = clipping bias + dp noise error
    expected_total = decomp["clipping_bias"] + decomp["dp_noise_error"]
    assert np.allclose(decomp["total_error"], expected_total)


def test_metric_normalized_share_zero_sum_dynamic_fallback():
    """Verify legacy UserNormalizedShareMetric zero-sum exact mean and post-processing on non-9-cell domains."""
    from private_llm_analytics.aggregation.grouping import build_cell_definitions

    # Test configuration 1: 2 providers x 4 tasks = 8 cells
    p_8 = ["provider_1", "provider_2"]
    t_8 = ["task_1", "task_2", "task_3", "task_4"]
    names_8, map_8 = build_cell_definitions(p_8, t_8)
    umatrix_8 = UserMatrix(
        matrix=np.zeros((3, len(names_8)), dtype=np.float64),
        user_ids=["u1", "u2", "u3"],
        providers=p_8,
        tasks=t_8,
        cell_names=names_8,
        cell_map=map_8,
        public_population_size=50,
    )

    metric = UserNormalizedShareMetric()
    exact_mean_8 = metric.compute_exact_mean(umatrix_8)
    assert exact_mean_8.shape == (8,)
    assert np.allclose(exact_mean_8, 1.0 / 8.0)
    assert np.isclose(np.sum(exact_mean_8), 1.0)

    post_8 = metric.post_process(np.zeros(8, dtype=np.float64), umatrix_8)
    assert post_8["share_distribution_array"].shape == (8,)
    assert np.allclose(post_8["share_distribution_array"], 1.0 / 8.0)
    assert post_8["grid_3x3"].shape == (2, 4)

    # Test configuration 2: 5 providers x 2 tasks = 10 cells
    p_10 = [f"p_{i}" for i in range(5)]
    t_10 = [f"t_{j}" for j in range(2)]
    names_10, map_10 = build_cell_definitions(p_10, t_10)
    umatrix_10 = UserMatrix(
        matrix=np.zeros((2, len(names_10)), dtype=np.float64),
        user_ids=["u1", "u2"],
        providers=p_10,
        tasks=t_10,
        cell_names=names_10,
        cell_map=map_10,
        public_population_size=50,
    )

    exact_mean_10 = metric.compute_exact_mean(umatrix_10)
    assert exact_mean_10.shape == (10,)
    assert np.allclose(exact_mean_10, 1.0 / 10.0)
    assert np.isclose(np.sum(exact_mean_10), 1.0)

    post_10 = metric.post_process(np.zeros(10, dtype=np.float64), umatrix_10)
    assert post_10["share_distribution_array"].shape == (10,)
    assert np.allclose(post_10["share_distribution_array"], 1.0 / 10.0)
    assert post_10["grid_3x3"].shape == (5, 2)

