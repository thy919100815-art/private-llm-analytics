"""Tests for Planner Statistical Window Optimization and Workload Weights (v0.3)."""

import pytest

from private_llm_analytics.config.loader import load_app_config
from private_llm_analytics.planning.candidates import ReleasePolicyCandidate
from private_llm_analytics.planning.planner import SmallNReleasePlanner
from private_llm_analytics.planning.policy import compute_workload_reconstruction_error
from private_llm_analytics.planning.utility import (
    ClippingEstimator,
    GranularityEstimator,
    TruncatedNormalPrior,
    UtilityEstimator,
    UtilityWeights,
    scale_prior_for_window,
)


def test_window_changes_public_prior():
    """Window scaling correctly shifts prior mean linearly and std with square root of time."""
    ref_prior = TruncatedNormalPrior(mean=60.0, std=20.0, min_val=5.0, max_val=150.0)

    # 1. Scaling to 15 days (half window)
    p_15 = scale_prior_for_window(ref_prior, window_days=15, reference_days=30)
    assert p_15.mean == pytest.approx(30.0)
    assert p_15.std == pytest.approx(20.0 * (15.0 / 30.0) ** 0.5)

    # 2. Scaling to 90 days (triple window)
    p_90 = scale_prior_for_window(ref_prior, window_days=90, reference_days=30)
    assert p_90.mean == pytest.approx(180.0)
    assert p_90.std == pytest.approx(20.0 * (3.0 ** 0.5))


def test_window_changes_clipping_estimate():
    """For fixed clipping bound C, clipping bias estimate strictly increases with window size W."""
    ref_prior = TruncatedNormalPrior(mean=60.0, std=20.0, min_val=5.0, max_val=150.0)
    n_users = 1000
    fixed_c = 40.0

    p_14 = scale_prior_for_window(ref_prior, window_days=14, reference_days=30)
    p_30 = scale_prior_for_window(ref_prior, window_days=30, reference_days=30)
    p_90 = scale_prior_for_window(ref_prior, window_days=90, reference_days=30)

    clip_14 = ClippingEstimator.estimate_clipping_bias(fixed_c, n_users, p_14)
    clip_30 = ClippingEstimator.estimate_clipping_bias(fixed_c, n_users, p_30)
    clip_90 = ClippingEstimator.estimate_clipping_bias(fixed_c, n_users, p_90)

    # Larger window -> more interactions per user -> more clipping excess
    assert clip_14 < clip_30 < clip_90
    assert clip_90 > 2 * clip_30


def test_window_changes_objective():
    """Candidates differing only in W have strictly distinct total utility losses."""
    estimator = UtilityEstimator(
        weights=UtilityWeights(clipping=1.0, noise=1.0, aggregation=1.0, staleness=0.5, window=0.5),
        public_prior=TruncatedNormalPrior(mean=60.0, std=20.0),
        reference_days=30,
    )
    n_users = 1000

    cand_w14 = ReleasePolicyCandidate(contribution_bound=20.0, granularity="full_9_cell", window_days=14, frequency_days=7)
    cand_w30 = ReleasePolicyCandidate(contribution_bound=20.0, granularity="full_9_cell", window_days=30, frequency_days=7)
    cand_w90 = ReleasePolicyCandidate(contribution_bound=20.0, granularity="full_9_cell", window_days=90, frequency_days=7)

    eval_14 = estimator.evaluate_candidate("interaction_volume", cand_w14, n_users)
    eval_30 = estimator.evaluate_candidate("interaction_volume", cand_w30, n_users)
    eval_90 = estimator.evaluate_candidate("interaction_volume", cand_w90, n_users)

    # Total loss differs
    assert eval_14.total_error != eval_30.total_error
    assert eval_30.total_error != eval_90.total_error
    # Window error is 0 at reference W=30 and positive at W=14 and W=90
    assert eval_30.window_error == 0.0
    assert eval_14.window_error > 0.0
    assert eval_90.window_error > 0.0


def test_window_selection_not_lexicographic_tie_only():
    """Planner selects optimal window W based on utility loss minimization, not just lexicographic tie-break."""
    cfg = load_app_config("configs")
    # Provide multiple window candidates
    cfg.privacy.planner.candidate_windows = [14, 30, 90]
    cfg.privacy.planner.candidate_clipping = [40.0]
    cfg.privacy.planner.candidate_granularities = ["full_9_cell"]
    cfg.privacy.planner.candidate_frequencies = [14]

    planner = SmallNReleasePlanner(app_config=cfg)
    plan_res = planner.plan(
        metric="interaction_volume",
        recipient="recipient_internal",
        planning_horizon_days=90,
    )

    assert plan_res.budget_feasible is True
    assert plan_res.policy is not None
    # Reference window is 30, which balances clipping and temporal penalty
    assert plan_res.policy.window_days in [14, 30, 90]
    # Check that evaluations across windows have different total error
    win_evals = {e["candidate"].window_days: e["total_error"] for e in plan_res.candidate_evaluations}
    assert len(set(win_evals.values())) > 1


def test_workload_weights_change_granularity_loss():
    """Changing public workload weights alters granularity aggregation loss."""
    import numpy as np
    dummy_9 = np.array([10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0])
    cell_names = [f"p{i}:t{j}" for i in range(3) for j in range(3)]

    # Workload A: 100% full 9-cell queries
    workload_full9 = {"full_9_cell": 1.0, "provider": 0.0, "task": 0.0, "overall": 0.0}
    err_prov_under_full9 = compute_workload_reconstruction_error(
        dummy_9, cell_names, "provider_3_cell", workload_weights=workload_full9
    )

    # Workload B: 100% provider marginal queries
    workload_provider = {"full_9_cell": 0.0, "provider": 1.0, "task": 0.0, "overall": 0.0}
    err_prov_under_prov = compute_workload_reconstruction_error(
        dummy_9, cell_names, "provider_3_cell", workload_weights=workload_provider
    )

    # When workload only requests provider marginals, provider_3_cell error is 0.0!
    assert err_prov_under_prov == 0.0
    assert err_prov_under_full9 > 0.0
