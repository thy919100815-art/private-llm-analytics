"""Tests for Utility Estimation Framework, Monotonicity, and Tradeoffs (v0.3)."""

import math
import numpy as np
import pytest
from private_llm_analytics.planning.candidates import (
    ReleasePolicyCandidate,
    generate_candidate_set,
)
from private_llm_analytics.planning.utility import (
    ClippingEstimator,
    GranularityEstimator,
    LogNormalPrior,
    NoiseEstimator,
    StalenessEstimator,
    TruncatedNormalPrior,
    UtilityEstimator,
    UtilityWeights,
)
from private_llm_analytics.planning.diagnostics import (
    compute_calibration_error,
    compute_regret,
)
from private_llm_analytics.planning.policy import (
    aggregate_to_granularity,
    reconstruct_to_9_cell,
    compute_workload_reconstruction_error,
)
from private_llm_analytics.config.loader import load_app_config
from private_llm_analytics.config.models import (
    AppConfig,
    PlannerConfig,
    PublicPriorConfig,
    UtilityWeightsConfig,
)
from private_llm_analytics.config.validator import validate_production_config
from private_llm_analytics.privacy.exceptions import ProductionSafetyConfigurationError


def test_clipping_error_monotonic_decreases_with_C():
    """In fixed public prior, expected clipping error monotonically decreases as C increases (Item 58)."""
    prior = TruncatedNormalPrior(mean=60.0, std=20.0, min_val=5.0, max_val=150.0)
    n_users = 1000
    c_values = [5.0, 10.0, 20.0, 40.0, 80.0, 120.0, 150.0]

    errors = [
        ClippingEstimator.estimate_clipping_bias(
            contribution_bound=c,
            population_size=n_users,
            prior=prior,
        )
        for c in c_values
    ]

    for i in range(len(errors) - 1):
        assert errors[i] >= errors[i + 1], f"Clipping error should decrease: C={c_values[i]} had {errors[i]}, C={c_values[i+1]} had {errors[i+1]}"

    # At upper bound, clipping error must be 0
    assert errors[-1] == 0.0


def test_volume_noise_error_increases_with_C():
    """For interaction volume, noise error monotonically increases as C increases (Item 59)."""
    c_values = [5.0, 10.0, 20.0, 40.0, 80.0]
    n_users = 1000

    noise_errors = []
    for c in c_values:
        cand = ReleasePolicyCandidate(
            contribution_bound=c,
            granularity="full_9_cell",
            mechanism="gaussian",
            epsilon_per_release=1.0,
            delta_per_release=1e-6,
        )
        err = NoiseEstimator.estimate_noise_error("interaction_volume", cand, n_users)
        noise_errors.append(err)

    for i in range(len(noise_errors) - 1):
        assert noise_errors[i] < noise_errors[i + 1], f"Noise error should increase: C={c_values[i]} had {noise_errors[i]}, C={c_values[i+1]} had {noise_errors[i+1]}"


def test_tradeoff_has_interior_optimum():
    """Volume total error balances clipping bias and noise, yielding an interior optimum (Item 60)."""
    prior = TruncatedNormalPrior(mean=25.0, std=10.0, min_val=2.0, max_val=80.0)
    estimator = UtilityEstimator(
        weights=UtilityWeights(clipping=1.0, noise=1.0, aggregation=0.0, staleness=0.0),
        public_prior=prior,
    )
    n_users = 100
    c_candidates = [5.0, 10.0, 20.0, 40.0, 80.0]

    evaluations = []
    for c in c_candidates:
        cand = ReleasePolicyCandidate(contribution_bound=c, granularity="full_9_cell")
        res = estimator.evaluate_candidate("interaction_volume", cand, n_users)
        evaluations.append((c, res.clipping_error, res.noise_error, res.total_error))

    totals = [e[3] for e in evaluations]
    best_c_idx = totals.index(min(totals))

    # Optimal C is an interior point (C=20 or C=40, not the boundary extremes 5 or 80)
    assert 0 < best_c_idx < len(c_candidates) - 1, f"Expected interior optimum, but got index {best_c_idx} (C={c_candidates[best_c_idx]}): {evaluations}"


def test_candidate_validation_and_enumeration():
    """Candidates are strictly validated and finite set enumeration is deterministic (Item 57 & 64)."""
    # Invalid contribution bound
    with pytest.raises(ValueError, match="contribution_bound must be > 0"):
        ReleasePolicyCandidate(contribution_bound=-1.0)

    # Invalid granularity
    with pytest.raises(ValueError, match="Invalid granularity"):
        ReleasePolicyCandidate(granularity="invalid_dimension_cell")

    # Invalid frequency
    with pytest.raises(ValueError, match="frequency_days must be > 0"):
        ReleasePolicyCandidate(frequency_days=0)

    # Invalid mechanism
    with pytest.raises(ValueError, match="Invalid mechanism"):
        ReleasePolicyCandidate(mechanism="quantum_random")

    # Enumeration determinism
    c1 = generate_candidate_set(
        contribution_bounds=[10.0, 20.0],
        granularities=["full_9_cell", "overall_1_cell"],
        window_days=[30],
        frequency_days=[7, 14],
    )
    c2 = generate_candidate_set(
        contribution_bounds=[10.0, 20.0],
        granularities=["full_9_cell", "overall_1_cell"],
        window_days=[30],
        frequency_days=[7, 14],
    )

    assert len(c1) == 2 * 2 * 1 * 2 * 1 == 8
    assert [c.candidate_id for c in c1] == [c.candidate_id for c in c2]


def test_granularity_and_staleness_estimators():
    """Granularity and staleness penalties behave as expected."""
    # Granularity penalty is 0 for full 9 cell and positive for collapsed cells
    g_full = GranularityEstimator.estimate_granularity_error("interaction_volume", "full_9_cell", 1000)
    g_prov = GranularityEstimator.estimate_granularity_error("interaction_volume", "provider_3_cell", 1000)
    g_over = GranularityEstimator.estimate_granularity_error("interaction_volume", "overall_1_cell", 1000)

    assert g_full == 0.0
    assert g_prov > 0.0
    assert g_over > g_prov

    # Staleness increases with interval F
    s_daily = StalenessEstimator.estimate_staleness_error("interaction_volume", frequency_days=1, lambda_stale=2.0)
    s_monthly = StalenessEstimator.estimate_staleness_error("interaction_volume", frequency_days=30, lambda_stale=2.0)
    assert s_daily < s_monthly
    assert s_daily == 1.0  # 2.0 * (1 / 2)
    assert s_monthly == 30.0  # 2.0 * (30 / 2)


def test_diagnostics_regret_and_calibration():
    """Regret and calibration metrics calculate exact values."""
    abs_r, rel_r = compute_regret(true_error_selected=150.0, true_error_oracle=100.0)
    assert abs_r == 50.0
    assert rel_r == 0.5

    cal = compute_calibration_error(
        predicted_errors=[10.0, 20.0, 30.0],
        observed_errors=[12.0, 22.0, 32.0],
    )
    assert cal["bias"] == -2.0
    assert cal["mae"] == 2.0
    assert cal["spearman_corr"] == 1.0


def test_granularity_aggregation_and_reconstruction():
    """Granularity transformations aggregate correctly and compute reconstruction error."""
    cell_names = [
        "provider_a:technical", "provider_a:document", "provider_a:analysis",
        "provider_b:technical", "provider_b:document", "provider_b:analysis",
        "provider_c:technical", "provider_c:document", "provider_c:analysis",
    ]
    # Synthetic ground truth
    true_9_cell = np.array([10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0])

    # Full 9 cell aggregation is identity
    agg_full, names_full = aggregate_to_granularity(true_9_cell, cell_names, "full_9_cell")
    np.testing.assert_array_equal(agg_full, true_9_cell)
    assert names_full == cell_names

    # Provider 3 cell
    agg_prov, names_prov = aggregate_to_granularity(true_9_cell, cell_names, "provider_3_cell")
    assert len(agg_prov) == 3
    assert agg_prov[0] == 60.0  # 10 + 20 + 30
    assert agg_prov[1] == 150.0 # 40 + 50 + 60
    assert agg_prov[2] == 240.0 # 70 + 80 + 90

    # Task 3 cell
    agg_task, names_task = aggregate_to_granularity(true_9_cell, cell_names, "task_3_cell")
    assert len(agg_task) == 3
    # Technical: 10 + 40 + 70 = 120
    # Document: 20 + 50 + 80 = 150
    # Analysis: 30 + 60 + 90 = 180
    assert 120.0 in agg_task and 150.0 in agg_task and 180.0 in agg_task

    # Overall 1 cell
    agg_over, names_over = aggregate_to_granularity(true_9_cell, cell_names, "overall_1_cell")
    assert len(agg_over) == 1
    assert agg_over[0] == 450.0

    # Reconstruction error
    err_full = compute_workload_reconstruction_error(true_9_cell, cell_names, "full_9_cell")
    err_prov = compute_workload_reconstruction_error(true_9_cell, cell_names, "provider_3_cell")
    err_over = compute_workload_reconstruction_error(true_9_cell, cell_names, "overall_1_cell")

    assert err_full == 0.0
    assert err_prov > 0.0
    assert err_over >= err_prov


def test_planner_config_validation_and_production_isolation():
    """Planner config correctly enforces bounds and forbids oracle_eval_only in production."""
    # Invalid bounds on prior
    with pytest.raises(ValueError, match="lower .* must be strictly less than upper"):
        PublicPriorConfig(lower=100.0, upper=50.0)

    # Invalid weights (all zeros)
    with pytest.raises(ValueError, match="Sum of utility weights must be strictly positive"):
        UtilityWeightsConfig(clipping=0.0, noise=0.0, aggregation=0.0, staleness=0.0, window=0.0)

    # Load default app config
    app_config = load_app_config("configs")
    assert app_config.privacy.planner.enabled is True
    assert app_config.privacy.planner.mode == "static_public"

    # Production safety check passes with static_public
    validate_production_config(app_config)

    # Production safety check FAILS if planner is oracle_eval_only
    app_config.privacy.planner.mode = "oracle_eval_only"
    with pytest.raises(ProductionSafetyConfigurationError, match="strictly forbids privacy.planner.mode='oracle_eval_only'"):
        validate_production_config(app_config)
