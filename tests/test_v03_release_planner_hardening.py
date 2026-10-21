"""Regression and hardening tests for Private LLM Analytics v0.3 Release Planner."""

from decimal import Decimal
import math
import numpy as np
import pandas as pd
import pytest

from experiments.exp13_joint_planner import (
    compute_true_candidate_utility,
    generate_rate_consistent_temporal_truth,
)
from private_llm_analytics.aggregation.user_matrix import build_user_matrix
from private_llm_analytics.config.loader import load_app_config
from private_llm_analytics.config.models import UtilityWeightsConfig
from private_llm_analytics.mechanisms.gaussian import calibrate_analytic_gaussian
from private_llm_analytics.planning.candidates import ReleasePolicyCandidate
from private_llm_analytics.planning.planner import (
    SmallNReleasePlanner,
    compute_planner_config_fingerprint,
)
from private_llm_analytics.planning.utility import (
    PlannerConfigurationError,
    PublicContributionPrior,
    UtilityEstimator,
    UtilityWeights,
    scale_prior_for_window,
)


def test_rate_consistent_30_day_truth():
    """generate_rate_consistent_temporal_truth with 30-day window yields volume consistent with reference_mean ~ 60."""
    df = generate_rate_consistent_temporal_truth(
        n_users=500,
        horizon_days=30,
        reference_days=30,
        reference_mean=60.0,
        reference_std=20.0,
        seed=42,
    )
    mean_vol = len(df) / 500.0
    assert 50.0 <= mean_vol <= 70.0


def test_rate_consistent_90_day_truth():
    """generate_rate_consistent_temporal_truth over 90 days yields scaled mean 180 and 30-day slice ~ 60."""
    df = generate_rate_consistent_temporal_truth(
        n_users=500,
        horizon_days=90,
        reference_days=30,
        reference_mean=60.0,
        reference_std=20.0,
        seed=42,
    )
    mean_vol_90 = len(df) / 500.0
    assert 160.0 <= mean_vol_90 <= 200.0

    t_end = df["timestamp"].max()
    t_30 = t_end - pd.Timedelta(days=30)
    df_30 = df[(df["timestamp"] > t_30) & (df["timestamp"] <= t_end)]
    mean_vol_30 = len(df_30) / 500.0
    assert 50.0 <= mean_vol_30 <= 70.0


def test_public_prior_window_std_scaling():
    """with_window(90) scales std by sqrt(3) under sqrt_time scaling."""
    prior30 = PublicContributionPrior(
        mean=60.0,
        std=20.0,
        reference_days=30,
        window_days=30,
        std_scaling="sqrt_time",
    )
    prior90 = prior30.with_window(90)
    std30 = np.std(prior30.ref_totals)
    std90 = np.std(prior90.ref_totals)
    ratio = std90 / std30
    assert 1.4 <= ratio <= 2.0  # sqrt(3) ~ 1.732


def test_public_prior_window_mean_scaling():
    """with_window(90) scales mean linearly by 3 under linear scaling."""
    prior30 = PublicContributionPrior(
        mean=60.0,
        std=20.0,
        reference_days=30,
        window_days=30,
        mean_scaling="linear",
    )
    prior90 = prior30.with_window(90)
    ratio = prior90.expected_user_volume / prior30.expected_user_volume
    assert 2.8 <= ratio <= 3.2


def test_window_30_matches_reference_distribution():
    """W=30 reference sample matches public reference parameter bounds and mean."""
    prior = PublicContributionPrior(
        mean=60.0,
        std=20.0,
        min_val=5.0,
        max_val=150.0,
        reference_days=30,
        window_days=30,
    )
    assert 55.0 <= prior.expected_user_volume <= 65.0
    assert np.all(prior.ref_totals >= 5.0)
    assert np.all(prior.ref_totals <= 150.0)


def test_window_90_not_simple_matrix_multiplication():
    """W=90 reference sample is directly sampled, not a simple scalar multiplication."""
    prior30 = PublicContributionPrior(reference_days=30, window_days=30)
    prior90 = prior30.with_window(90)
    assert not np.allclose(prior90.ref_matrix, prior30.ref_matrix * 3.0)
    assert np.all(prior90.ref_matrix == np.rint(prior90.ref_matrix))


def test_exp10_clipping_only_candidate_space():
    """Exp10 isolates candidate space to full_9_cell, window_days=30, frequency_days=30."""
    cfg = load_app_config("configs")
    cfg.scenario.population.n_users = 500
    cfg.privacy.planner.candidate_clipping = [10.0, 20.0, 40.0]
    cfg.privacy.planner.candidate_granularities = ["full_9_cell"]
    cfg.privacy.planner.candidate_windows = [30]
    cfg.privacy.planner.candidate_frequencies = [30]
    cfg.privacy.planner.planning_horizon_days = 30
    planner = SmallNReleasePlanner(app_config=cfg)
    res = planner.plan(metric="interaction_volume", recipient="recipient_internal")
    assert res.policy is not None
    assert res.policy.granularity == "full_9_cell"
    assert res.policy.window_days == 30
    assert res.policy.frequency_days == 30


def test_exp14_clipping_only_candidate_space():
    """Exp14 isolates candidate space to full_9_cell, window_days=30, frequency_days=30."""
    cfg = load_app_config("configs")
    cfg.privacy.planner.candidate_clipping = [10.0, 20.0, 40.0]
    cfg.privacy.planner.candidate_granularities = ["full_9_cell"]
    cfg.privacy.planner.candidate_windows = [30]
    cfg.privacy.planner.candidate_frequencies = [30]
    cfg.privacy.planner.planning_horizon_days = 30
    planner = SmallNReleasePlanner(app_config=cfg)
    res = planner.plan(metric="interaction_volume", recipient="recipient_internal")
    assert res.policy is not None
    assert res.policy.granularity == "full_9_cell"
    assert res.policy.window_days == 30
    assert res.policy.frequency_days == 30


def test_public_seed_changes_fingerprint():
    """Changing public_seed in prior configuration alters planner config fingerprint."""
    cfg1 = load_app_config("configs")
    cfg1.privacy.planner.prior.public_seed = 900001
    planner1 = SmallNReleasePlanner(app_config=cfg1)
    fp1 = planner1.config_fingerprint

    cfg2 = load_app_config("configs")
    cfg2.privacy.planner.prior.public_seed = 900002
    planner2 = SmallNReleasePlanner(app_config=cfg2)
    fp2 = planner2.config_fingerprint

    assert fp1 != fp2


def test_reference_population_changes_fingerprint():
    """Changing num_reference_users in prior configuration alters planner config fingerprint."""
    cfg1 = load_app_config("configs")
    cfg1.privacy.planner.prior.num_reference_users = 2000
    planner1 = SmallNReleasePlanner(app_config=cfg1)
    fp1 = planner1.config_fingerprint

    cfg2 = load_app_config("configs")
    cfg2.privacy.planner.prior.num_reference_users = 3000
    planner2 = SmallNReleasePlanner(app_config=cfg2)
    fp2 = planner2.config_fingerprint

    assert fp1 != fp2


def test_public_seed_changes_reference_sample_deterministically():
    """Different public_seed yields different reference matrix; same seed yields identical matrix."""
    prior1a = PublicContributionPrior(public_seed=900001)
    prior1b = PublicContributionPrior(public_seed=900001)
    prior2 = PublicContributionPrior(public_seed=900002)

    assert np.array_equal(prior1a.ref_matrix, prior1b.ref_matrix)
    assert not np.array_equal(prior1a.ref_matrix, prior2.ref_matrix)


def test_lognormal_uses_contribution_model():
    """Configuring distribution='lognormal' initializes PublicContributionPrior with lognormal mode."""
    cfg = load_app_config("configs")
    cfg.privacy.planner.prior.distribution = "lognormal"
    planner = SmallNReleasePlanner(app_config=cfg)
    assert isinstance(planner.utility_estimator.public_prior, PublicContributionPrior)
    assert planner.utility_estimator.public_prior.distribution == "lognormal"


def test_changing_contribution_shape_changes_induced_rg_distribution():
    """Modulating provider_shares or task_shares changes the induced target-space norm distribution."""
    prior1 = PublicContributionPrior(
        provider_shares={"provider_a": 0.8, "provider_b": 0.1, "provider_c": 0.1},
        public_seed=900001,
    )
    prior2 = PublicContributionPrior(
        provider_shares={"provider_a": 0.1, "provider_b": 0.1, "provider_c": 0.8},
        public_seed=900001,
    )
    assert not np.allclose(prior1._norms["full_9_cell"], prior2._norms["full_9_cell"])
    assert not np.allclose(prior1._norms["provider_3_cell"], prior2._norms["provider_3_cell"])


def test_expected_folded_normal_matches_monte_carlo():
    """Analytical folded normal expectation matches Monte Carlo empirical trials within 1%."""
    prior = PublicContributionPrior(public_seed=900001, num_reference_users=2000)
    cand = ReleasePolicyCandidate(
        contribution_bound=40.0,
        granularity="full_9_cell",
        mechanism="gaussian",
        epsilon_per_release=1.0,
        delta_per_release=1e-6,
    )
    analytical_err, _ = prior.expected_folded_normal_release_l1(
        metric="interaction_volume",
        candidate=cand,
        population_size=1000,
        workload_weights={"full_9_cell": 1.0, "provider": 0.0, "task": 0.0, "overall": 0.0},
    )

    norms = prior._norms["full_9_cell"]
    clip_scale = np.minimum(1.0, 40.0 / np.maximum(norms, 1e-9))
    recon_mat = prior.ref_matrix * clip_scale[:, np.newaxis]
    b = 1000.0 * (np.mean(recon_mat, axis=0) - prior.expected_raw_vector)
    sigma = calibrate_analytic_gaussian(1.0, 1e-6, 40.0)

    rng = np.random.default_rng(12345)
    mc_trials = 50000
    noise_batch = rng.normal(0, sigma, size=(mc_trials, 9))
    mc_l1 = np.sum(np.abs(b[np.newaxis, :] + noise_batch), axis=1)
    mc_mean = float(np.mean(mc_l1))

    rel_diff = abs(analytical_err - mc_mean) / mc_mean
    assert rel_diff < 0.01, f"Expected < 1% difference, got {rel_diff * 100:.2f}%"


def test_expected_release_l1_selects_known_small_n_optimum():
    """SmallNReleasePlanner under EXPECTED_RELEASE_L1 selects optimal C=40.0 for N=500 on full-9-cell query."""
    cfg = load_app_config("configs")
    cfg.scenario.population.n_users = 500
    cfg.privacy.planner.candidate_clipping = [5.0, 10.0, 20.0, 40.0, 80.0]
    cfg.privacy.planner.candidate_granularities = ["full_9_cell"]
    cfg.privacy.planner.candidate_windows = [30]
    cfg.privacy.planner.candidate_frequencies = [30]
    cfg.privacy.planner.planning_horizon_days = 30
    cfg.privacy.planner.workloads.full_9_cell_weight = 1.0
    cfg.privacy.planner.workloads.provider_weight = 0.0
    cfg.privacy.planner.workloads.task_weight = 0.0
    cfg.privacy.planner.workloads.overall_weight = 0.0

    planner = SmallNReleasePlanner(app_config=cfg)
    res = planner.plan(metric="interaction_volume", recipient="recipient_internal", force_recompute=True)
    assert res.policy is not None
    assert res.policy.contribution_bound == 40.0


def test_exp13_oracle_uses_actual_reconstructed_release_error():
    """Verify that Oracle evaluates true end-to-end reconstructed release error E|B + Z| rather than |B| + E|Z|."""
    df = generate_rate_consistent_temporal_truth(
        n_users=200,
        horizon_days=30,
        reference_days=30,
        reference_mean=60.0,
        reference_std=20.0,
        seed=42,
    )
    umat = build_user_matrix(df, public_population_size=200)
    cand = ReleasePolicyCandidate(
        contribution_bound=10.0,  # Tight bound creates substantial clipping bias
        granularity="full_9_cell",
        window_days=30,
        frequency_days=30,
        mechanism="gaussian",
        epsilon_per_release=0.1,  # Low epsilon creates large noise
        delta_per_release=1e-5,
    )
    # Compute true utility via Oracle
    mean_util, stderr, ci = compute_true_candidate_utility(
        candidate=cand,
        umatrix=umat,
        cell_names=umat.cell_names,
        reference_days=30,
        n_trials=200,
        seed=42,
        weights_data=1.0,
        weights_stale=0.0,
        weights_win=0.0,
    )

    # Compute additive surrogate: |B| + E|Z|
    raw_mat = umat.matrix
    unclipped_9 = np.sum(raw_mat, axis=0)
    true_total_vol = max(1.0, float(np.sum(unclipped_9)))
    user_norms = np.linalg.norm(raw_mat, axis=1)
    scales = np.minimum(1.0, 10.0 / np.maximum(1e-12, user_norms))
    clipped_mat = raw_mat * scales[:, np.newaxis]
    clipped_9 = np.sum(clipped_mat, axis=0)
    bias_l1 = np.sum(np.abs(unclipped_9 - clipped_9))
    emp_clip_loss = bias_l1 / true_total_vol

    sigma = calibrate_analytic_gaussian(0.1, 1e-5, 10.0)
    expected_noise_l1 = 9.0 * sigma * math.sqrt(2.0 / math.pi)
    additive_surrogate = emp_clip_loss + (expected_noise_l1 / true_total_vol)

    # Because noise partially cancels bias on each cell, E|B + Z| < |B| + E|Z|
    assert mean_util < additive_surrogate
    assert (additive_surrogate - mean_util) > 3.0 * stderr


def test_expected_release_l1_uses_workload_weights():
    """Verify that expected_folded_normal_release_l1 responds to workload query weights."""
    prior = PublicContributionPrior(reference_days=30, window_days=30)
    cand = ReleasePolicyCandidate(
        contribution_bound=80.0,
        granularity="provider_3_cell",
        window_days=30,
        frequency_days=7,
        epsilon_per_release=0.5,
        delta_per_release=1e-5,
    )
    # Under full_9_cell heavy workload
    err_full9, _ = prior.expected_folded_normal_release_l1(
        metric="interaction_volume",
        candidate=cand,
        population_size=1000,
        workload_weights={"full_9_cell": 1.0, "provider": 0.0, "task": 0.0, "overall": 0.0},
    )
    # Under provider heavy workload
    err_prov, _ = prior.expected_folded_normal_release_l1(
        metric="interaction_volume",
        candidate=cand,
        population_size=1000,
        workload_weights={"full_9_cell": 0.0, "provider": 1.0, "task": 0.0, "overall": 0.0},
    )
    # For provider_3_cell, provider query has lower error than full 9-cell query (no aggregation bias on provider query)
    assert err_prov < err_full9


def test_provider_heavy_workload_can_prefer_provider_granularity():
    """Under provider-heavy workload, provider_3_cell achieves lower utility loss than full_9_cell."""
    prior = PublicContributionPrior(reference_days=30, window_days=30)
    weights = UtilityWeights(objective_mode="EXPECTED_RELEASE_L1", data=1.0, staleness=0.0, window=0.0)
    estimator = UtilityEstimator(
        weights=weights,
        public_prior=prior,
        workload_weights={"full_9_cell": 0.0, "provider": 1.0, "task": 0.0, "overall": 0.0},
    )
    cand_prov = ReleasePolicyCandidate(
        contribution_bound=80.0,
        granularity="provider_3_cell",
        window_days=30,
        frequency_days=7,
        epsilon_per_release=0.5,
        delta_per_release=1e-5,
    )
    cand_full = ReleasePolicyCandidate(
        contribution_bound=80.0,
        granularity="full_9_cell",
        window_days=30,
        frequency_days=7,
        epsilon_per_release=0.5,
        delta_per_release=1e-5,
    )
    eval_prov = estimator.evaluate_candidate("interaction_volume", cand_prov, 500)
    eval_full = estimator.evaluate_candidate("interaction_volume", cand_full, 500)
    assert eval_prov.total_error < eval_full.total_error


def test_full9_heavy_workload_prefers_full9_when_expected():
    """Under full-9-cell heavy workload, full_9_cell achieves lower loss than provider_3_cell."""
    prior = PublicContributionPrior(reference_days=30, window_days=30)
    weights = UtilityWeights(objective_mode="EXPECTED_RELEASE_L1", data=1.0, staleness=0.0, window=0.0)
    estimator = UtilityEstimator(
        weights=weights,
        public_prior=prior,
        workload_weights={"full_9_cell": 1.0, "provider": 0.0, "task": 0.0, "overall": 0.0},
    )
    cand_prov = ReleasePolicyCandidate(
        contribution_bound=40.0,
        granularity="provider_3_cell",
        window_days=30,
        frequency_days=7,
        epsilon_per_release=2.0,  # high epsilon -> low noise, reconstruction bias dominates
        delta_per_release=1e-5,
    )
    cand_full = ReleasePolicyCandidate(
        contribution_bound=40.0,
        granularity="full_9_cell",
        window_days=30,
        frequency_days=7,
        epsilon_per_release=2.0,
        delta_per_release=1e-5,
    )
    eval_prov = estimator.evaluate_candidate("interaction_volume", cand_prov, 2000)
    eval_full = estimator.evaluate_candidate("interaction_volume", cand_full, 2000)
    assert eval_full.total_error < eval_prov.total_error


def test_expected_release_l1_rejects_laplace():
    """EXPECTED_RELEASE_L1 must fail closed with PlannerConfigurationError when mechanism is laplace."""
    prior = PublicContributionPrior(reference_days=30, window_days=30)
    weights = UtilityWeights(objective_mode="EXPECTED_RELEASE_L1", data=1.0)
    estimator = UtilityEstimator(weights=weights, public_prior=prior)
    cand = ReleasePolicyCandidate(
        contribution_bound=20.0,
        granularity="full_9_cell",
        window_days=30,
        frequency_days=7,
        mechanism="laplace",
        epsilon_per_release=0.5,
    )
    with pytest.raises(PlannerConfigurationError, match="Gaussian only"):
        estimator.evaluate_candidate("interaction_volume", cand, 1000)


def test_expected_release_l1_rejects_prevalence():
    """EXPECTED_RELEASE_L1 must fail closed with PlannerConfigurationError for metric prevalence."""
    prior = PublicContributionPrior(reference_days=30, window_days=30)
    weights = UtilityWeights(objective_mode="EXPECTED_RELEASE_L1", data=1.0)
    estimator = UtilityEstimator(weights=weights, public_prior=prior)
    cand = ReleasePolicyCandidate(
        contribution_bound=20.0,
        granularity="full_9_cell",
        window_days=30,
        frequency_days=7,
        mechanism="gaussian",
        epsilon_per_release=0.5,
        delta_per_release=1e-5,
    )
    with pytest.raises(PlannerConfigurationError, match="Unsupported metric 'prevalence'"):
        estimator.evaluate_candidate("prevalence", cand, 1000)


def test_expected_release_l1_rejects_share():
    """EXPECTED_RELEASE_L1 must fail closed with PlannerConfigurationError for normalized_share."""
    prior = PublicContributionPrior(reference_days=30, window_days=30)
    weights = UtilityWeights(objective_mode="EXPECTED_RELEASE_L1", data=1.0)
    estimator = UtilityEstimator(weights=weights, public_prior=prior)
    cand = ReleasePolicyCandidate(
        contribution_bound=20.0,
        granularity="full_9_cell",
        window_days=30,
        frequency_days=7,
        mechanism="gaussian",
        epsilon_per_release=0.5,
        delta_per_release=1e-5,
    )
    with pytest.raises(PlannerConfigurationError, match="Unsupported metric 'normalized_share'"):
        estimator.evaluate_candidate("normalized_share", cand, 1000)


def test_utility_weights_validator_expected_release_l1():
    """UtilityWeights and UtilityWeightsConfig validate active objective mode correctly."""
    # Under EXPECTED_RELEASE_L1: setting surrogate weights to 0 while data > 0 must pass
    uw = UtilityWeights(
        objective_mode="EXPECTED_RELEASE_L1",
        data=1.0,
        clipping=0.0,
        noise=0.0,
        aggregation=0.0,
        staleness=0.0,
        window=0.0,
    )
    assert uw.data == 1.0

    # Pydantic model validator
    cfg = UtilityWeightsConfig(
        objective_mode="EXPECTED_RELEASE_L1",
        data=1.0,
        clipping=0.0,
        noise=0.0,
        aggregation=0.0,
        staleness=0.0,
        window=0.0,
    )
    assert cfg.data == 1.0

    # Total <= 0 under EXPECTED_RELEASE_L1 must raise ValueError
    with pytest.raises(ValueError, match="strictly positive"):
        UtilityWeights(
            objective_mode="EXPECTED_RELEASE_L1",
            data=0.0,
            staleness=0.0,
            window=0.0,
        )

    with pytest.raises(ValueError, match="strictly positive"):
        UtilityWeightsConfig(
            objective_mode="EXPECTED_RELEASE_L1",
            data=0.0,
            staleness=0.0,
            window=0.0,
        )
