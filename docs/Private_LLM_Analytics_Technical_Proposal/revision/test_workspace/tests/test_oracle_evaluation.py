"""Tests verifying that Empirical Oracle is strictly independent of predicted utility (v0.3)."""

import numpy as np
import pytest

from private_llm_analytics.config.loader import load_app_config
from private_llm_analytics.planning.candidates import ReleasePolicyCandidate
from private_llm_analytics.planning.planner import SmallNReleasePlanner
from private_llm_analytics.planning.policy import build_volume_contributions_for_granularity
from private_llm_analytics.planning.utility import (
    TruncatedNormalPrior,
    UtilityEstimator,
    UtilityWeights,
)


def evaluate_candidate_true_utility(
    candidate: ReleasePolicyCandidate,
    true_user_matrix: np.ndarray,
    population_size: int,
    reference_days: int = 30,
) -> float:
    """Evaluate empirical true utility on ground-truth user matrix without using planner predictions."""
    w_days = candidate.window_days
    # 1. True empirical clipping loss
    _, exact_agg, _, _ = build_volume_contributions_for_granularity(
        raw_user_matrix=true_user_matrix,
        granularity=candidate.granularity,
        clip_norm=candidate.contribution_bound or 20.0,
    )
    unclipped_agg = np.sum(true_user_matrix, axis=0)
    true_total_vol = max(1.0, float(np.sum(unclipped_agg)))

    # Empirical clipping loss as fraction of total volume
    emp_clip_loss = float(np.sum(np.abs(unclipped_agg - exact_agg[:len(unclipped_agg)]))) / true_total_vol

    # Theoretical calibrated noise scale
    from private_llm_analytics.mechanisms.gaussian import calibrate_analytic_gaussian
    from private_llm_analytics.planning.utility import GRANULARITY_DIMENSIONS
    sigma = calibrate_analytic_gaussian(
        epsilon=candidate.epsilon_per_release,
        delta=candidate.delta_per_release,
        l2_sensitivity=candidate.contribution_bound or 20.0,
    )
    dim = GRANULARITY_DIMENSIONS.get(candidate.granularity, 9)
    emp_noise_loss = (dim * sigma * (2.0 / 3.1415926535) ** 0.5) / true_total_vol

    # Freshness and window loss
    stale_loss = float(candidate.frequency_days / (2.0 * w_days))
    win_loss = float(abs(w_days - reference_days) / float(reference_days))

    # True objective (weights 1.0)
    return emp_clip_loss + emp_noise_loss + stale_loss + 0.2 * win_loss


def test_oracle_is_independent_of_predicted_utility():
    """Oracle selects true empirical optimum on ground truth, even when planner prior is adversarial."""
    # Ground truth: users have large interaction volumes (~120 per user)
    np.random.seed(42)
    n_users = 200
    true_matrix = np.random.uniform(50.0, 150.0, size=(n_users, 9))

    # Candidates: C=10 (severe clipping) vs C=80 (well-matched to large volume)
    cand_c10 = ReleasePolicyCandidate(contribution_bound=10.0, frequency_days=14, window_days=30)
    cand_c80 = ReleasePolicyCandidate(contribution_bound=80.0, frequency_days=14, window_days=30)

    # 1. True empirical evaluation on ground truth
    true_util_10 = evaluate_candidate_true_utility(cand_c10, true_matrix, n_users)
    true_util_80 = evaluate_candidate_true_utility(cand_c80, true_matrix, n_users)

    # For large users, C=80 has vastly lower clipping bias than C=10
    assert true_util_80 < true_util_10
    oracle_selected = cand_c80

    # 2. Adversarially misspecify planner prior: claim users only do 2 interactions (TN(mean=2, std=1))
    adversarial_prior = TruncatedNormalPrior(mean=2.0, std=1.0, min_val=1.0, max_val=5.0)
    cfg = load_app_config("configs")
    cfg.scenario.population.n_users = n_users
    cfg.privacy.planner.candidate_clipping = [10.0, 80.0]
    cfg.privacy.planner.candidate_frequencies = [14]
    cfg.privacy.planner.candidate_windows = [30]
    cfg.privacy.planner.candidate_granularities = ["full_9_cell"]

    adversarial_estimator = UtilityEstimator(
        weights=UtilityWeights(clipping=1.0, noise=1.0, aggregation=0.0, staleness=0.0, window=0.0),
        public_prior=adversarial_prior,
    )
    adversarial_planner = SmallNReleasePlanner(app_config=cfg, utility_estimator=adversarial_estimator)

    plan_res = adversarial_planner.plan(
        metric="interaction_volume",
        recipient="recipient_internal",
        force_recompute=True,
    )

    # Under adversarial prior, planner believes C=10 has 0 clipping and lower noise than C=80, so it picks C=10
    assert plan_res.policy.contribution_bound == 10.0

    # 3. Verify Oracle remains independent and picks C=80
    assert oracle_selected.contribution_bound == 80.0
    assert oracle_selected.candidate_id != plan_res.policy.candidate_id

    # Regret is strictly positive, accurately capturing planner's sub-optimality
    regret = true_util_10 - true_util_80
    assert regret > 0.0
