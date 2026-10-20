"""Tests for Planner Privacy Provenance, Production Isolation, and Fail-Closed Enforcements (v0.3)."""

from decimal import Decimal
import pytest

from private_llm_analytics.config.loader import load_app_config
from private_llm_analytics.config.models import (
    AppConfig,
    PlannerConfig,
    PublicPriorConfig,
    UtilityWeightsConfig,
)
from private_llm_analytics.config.validator import validate_production_config
from private_llm_analytics.planning.candidates import (
    ReleasePolicyCandidate,
    generate_candidate_set,
)
from private_llm_analytics.planning.planner import (
    ReleasePlanResult,
    SmallNReleasePlanner,
)
from private_llm_analytics.planning.utility import (
    NoiseEstimator,
    TruncatedNormalPrior,
    UtilityEstimator,
    UtilityWeights,
)
from private_llm_analytics.privacy.composition import PrivacyView, resolve_budget_scope_id
from private_llm_analytics.privacy.exceptions import ProductionSafetyConfigurationError
from private_llm_analytics.privacy.rdp import (
    compute_gaussian_rdp,
    compose_rdp_vectors,
    rdp_to_epsilon_delta,
)


class PoisonedUserMatrix:
    """Poisoned object simulating raw private data that fails closed if touched."""

    def __getattr__(self, name):
        raise AssertionError(f"PROVENANCE BREACH: Private raw user data attribute '{name}' was accessed!")

    def __getitem__(self, item):
        raise AssertionError(f"PROVENANCE BREACH: Private raw user data item '{item}' was accessed!")

    def __iter__(self):
        raise AssertionError("PROVENANCE BREACH: Private raw user data was iterated!")

    def __len__(self):
        raise AssertionError("PROVENANCE BREACH: Private raw user data length was queried!")


def test_static_public_planner_does_not_access_user_matrix():
    """STATIC_PUBLIC planner must never access private raw user data (Item 54 & 62)."""
    cfg = load_app_config("configs")
    cfg.privacy.planner.mode = "static_public"
    planner = SmallNReleasePlanner(app_config=cfg)

    poison = PoisonedUserMatrix()
    res = planner.plan(
        metric="interaction_volume",
        recipient="recipient_internal",
        planning_horizon_days=90,
        user_matrix=poison,
    )
    assert res.budget_feasible is True
    assert res.policy is not None


def test_oracle_planner_rejected_in_production():
    """ORACLE_EVAL_ONLY mode must be strictly rejected in production configurations (Item 56)."""
    cfg = load_app_config("configs")
    cfg.privacy.planner.mode = "oracle_eval_only"

    # 1. Config validation fails
    with pytest.raises(ProductionSafetyConfigurationError, match="strictly forbids privacy.planner.mode='oracle_eval_only'"):
        validate_production_config(cfg)

    # 2. Planner execution fails for external recipients
    planner = SmallNReleasePlanner(app_config=cfg)
    with pytest.raises(ProductionSafetyConfigurationError, match="Oracle evaluation planning is strictly forbidden for external recipients"):
        planner.plan(
            metric="interaction_volume",
            recipient="recipient_external_a",
            planning_horizon_days=90,
        )


def test_projected_rdp_matches_manual_composition():
    """Projected RDP for candidate across K future releases must match manual order-wise composition (Item 61)."""
    cfg = load_app_config("configs")
    c_bound = 20.0
    cand = ReleasePolicyCandidate(
        contribution_bound=c_bound,
        granularity="full_9_cell",
        frequency_days=14,  # ceil(90 / 14) = 7 releases
        mechanism="gaussian",
        epsilon_per_release=1.0,
        delta_per_release=1e-6,
    )
    cfg.privacy.planner.candidate_clipping = [c_bound]
    cfg.privacy.planner.candidate_granularities = ["full_9_cell"]
    cfg.privacy.planner.candidate_windows = [30]
    cfg.privacy.planner.candidate_frequencies = [14]

    planner = SmallNReleasePlanner(app_config=cfg)
    res = planner.plan(
        metric="interaction_volume",
        recipient="recipient_internal",
        planning_horizon_days=90,
    )

    k_releases = 7
    eff_orders = [Decimal(str(a)) for a in cfg.privacy.accountant.orders]
    sigma, sens = NoiseEstimator.estimate_noise_sigma("interaction_volume", cand)
    single_rdp = compute_gaussian_rdp(sigma=sigma, sensitivity=sens, orders=eff_orders)
    k_rdp = {o: single_rdp[o] * Decimal(str(k_releases)) for o in eff_orders}
    _, expected_eps = rdp_to_epsilon_delta(k_rdp, delta_global=cfg.privacy.accountant.delta_global)

    assert res.planned_releases_count == k_releases
    assert abs(res.projected_epsilon - float(expected_eps)) < 1e-9


def test_planning_horizon_does_not_reset_history():
    """Planning horizon does NOT reset past privacy budget consumption (Item 55)."""
    cfg = load_app_config("configs")
    eff_orders = [Decimal(str(a)) for a in cfg.privacy.accountant.orders]
    cand = ReleasePolicyCandidate(contribution_bound=20.0, frequency_days=30)
    sigma, sens = NoiseEstimator.estimate_noise_sigma("interaction_volume", cand)
    single_rdp = compute_gaussian_rdp(sigma=sigma, sensitivity=sens, orders=eff_orders)

    class MockHistoricalState:
        def get_privacy_view(self, budget_scope_id, delta_global, orders, epsilon_cap, accountant_mode):
            past_rdp = {o: single_rdp[o] * Decimal("4") for o in orders}  # 4 past releases
            _, eff_eps = rdp_to_epsilon_delta(past_rdp, delta_global=delta_global)
            return PrivacyView(
                budget_scope_id=budget_scope_id,
                release_count=4,
                total_rdp_vector=past_rdp,
                effective_epsilon=eff_eps,
                best_alpha=orders[0],
                delta_global=Decimal(str(delta_global)),
                epsilon_cap=Decimal(str(epsilon_cap)),
                remaining_epsilon=max(Decimal("0"), Decimal(str(epsilon_cap)) - eff_eps),
                accountant_mode=accountant_mode,
                orders=orders,
            )

    cfg.privacy.planner.candidate_clipping = [20.0]
    cfg.privacy.planner.candidate_granularities = ["full_9_cell"]
    cfg.privacy.planner.candidate_windows = [30]
    cfg.privacy.planner.candidate_frequencies = [30]  # 3 planned releases

    planner = SmallNReleasePlanner(app_config=cfg, state_manager=MockHistoricalState())
    res = planner.plan(
        metric="interaction_volume",
        recipient="recipient_internal",
        planning_horizon_days=90,
    )
    # Total releases = 4 past + 3 planned = 7 releases
    combined_rdp = {o: single_rdp[o] * Decimal("7") for o in eff_orders}
    _, expected_eps = rdp_to_epsilon_delta(combined_rdp, delta_global=cfg.privacy.accountant.delta_global)

    assert abs(res.projected_epsilon - float(expected_eps)) < 1e-9
    assert res.historical_epsilon > 0.0


def test_candidate_count_deterministic():
    """Candidate enumeration is deterministic and strictly finite (Item 57)."""
    c_set = generate_candidate_set(
        contribution_bounds=[5.0, 10.0, 20.0, 40.0, 80.0],
        granularities=["full_9_cell", "provider_3_cell", "task_3_cell", "overall_1_cell"],
        window_days=[30, 60, 90],
        frequency_days=[1, 7, 14, 30],
        mechanisms=["gaussian"],
    )
    # 5 * 4 * 3 * 4 * 1 = 240 candidates
    assert len(c_set) == 240
    # Re-running produces identical order and identifiers
    c_set2 = generate_candidate_set(
        contribution_bounds=[5.0, 10.0, 20.0, 40.0, 80.0],
        granularities=["full_9_cell", "provider_3_cell", "task_3_cell", "overall_1_cell"],
        window_days=[30, 60, 90],
        frequency_days=[1, 7, 14, 30],
        mechanisms=["gaussian"],
    )
    assert [c.candidate_id for c in c_set] == [c.candidate_id for c in c_set2]


def test_frequency_privacy_monotonicity():
    """Higher frequency releases consume more privacy budget under longitudinal RDP."""
    cfg = load_app_config("configs")
    cfg.privacy.planner.candidate_clipping = [20.0]
    cfg.privacy.planner.candidate_granularities = ["full_9_cell"]
    cfg.privacy.planner.candidate_windows = [30]
    cfg.privacy.planner.candidate_frequencies = [1, 7, 14, 30]

    planner = SmallNReleasePlanner(app_config=cfg)
    res = planner.plan(
        metric="interaction_volume",
        recipient="recipient_internal",
        planning_horizon_days=90,
    )

    evals = {e["candidate"].frequency_days: e["projected_epsilon"] for e in res.candidate_evaluations}
    # F=1 (90 releases) > F=7 (13 releases) > F=14 (7 releases) > F=30 (3 releases)
    assert evals[1] > evals[7] > evals[14] > evals[30]


def test_no_feasible_policy_fail_closed():
    """When budget is exhausted, planner fails closed returning NO_FEASIBLE_POLICY."""
    cfg = load_app_config("configs")
    cfg.privacy.accountant.epsilon_cap = 0.5  # Impossible cap for Gaussian eps_single=1.0

    planner = SmallNReleasePlanner(app_config=cfg)
    res = planner.plan(
        metric="interaction_volume",
        recipient="recipient_internal",
        planning_horizon_days=90,
    )
    assert res.budget_feasible is False
    assert res.policy is None
    assert res.decision_reason == "NO_FEASIBLE_POLICY"


def test_utility_weights_validation():
    """Utility weights must have positive sum and non-negative coefficients."""
    with pytest.raises(ValueError, match="Sum of utility weights must be strictly positive"):
        UtilityWeightsConfig(clipping=0.0, noise=0.0, aggregation=0.0, staleness=0.0)

    # Valid weights
    w = UtilityWeightsConfig(clipping=1.0, noise=2.0, aggregation=0.5, staleness=0.1)
    assert w.clipping == 1.0


def test_recipient_composition_scope_mapping_in_planner():
    """Planner resolves budget scopes according to recipient composition policy."""
    cfg = load_app_config("configs")
    cfg.privacy.recipient_composition.mode = "per_recipient"
    planner = SmallNReleasePlanner(app_config=cfg)

    res_a = planner.plan(
        metric="interaction_volume",
        recipient="recipient_external_a",
        planning_horizon_days=90,
    )
    assert res_a.budget_scope_id == "recipient:recipient_external_a"
