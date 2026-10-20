"""Tests for Utility Estimation Framework, Monotonicity, and Tradeoffs (v0.3)."""

import math
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
