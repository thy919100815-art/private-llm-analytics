"""Verification Script: Small-n Utility-Aware Release Planner Architecture (v0.3).

Verifies core architectural and security invariants:
  1. Static public planner NEVER accesses private raw data (provenance guarantee).
  2. Oracle evaluation mode is strictly forbidden in production configurations.
  3. Longitudinal RDP composition projects future releases onto historical view.
  4. Planning horizon does NOT reset privacy budget (permanent accounting).
  5. Frequency-privacy tradeoff: high frequency consumes more privacy budget.
  6. Fail-closed behavior: budget exhaustion produces NO_FEASIBLE_POLICY.
  7. Deterministic memoization and state caching.
"""

from decimal import Decimal
import math
from pathlib import Path
import sys

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from private_llm_analytics.config.loader import load_app_config
from private_llm_analytics.config.models import AppConfig
from private_llm_analytics.planning.candidates import (
    ReleasePolicyCandidate,
    generate_candidate_set,
)
from private_llm_analytics.planning.planner import (
    ReleasePlanResult,
    SmallNReleasePlanner,
)
from private_llm_analytics.planning.utility import (
    TruncatedNormalPrior,
    UtilityEstimator,
    UtilityWeights,
)
from private_llm_analytics.privacy.exceptions import ProductionSafetyConfigurationError
from private_llm_analytics.privacy.rdp import (
    compute_gaussian_rdp,
    compose_rdp_vectors,
    rdp_to_epsilon_delta,
)


class PoisonedDataSentinel:
    """Poisoned data object that immediately raises if accessed in any way."""

    def __getattr__(self, name):
        raise AssertionError(f"SECURITY INVARIANT VIOLATION: Private raw data accessed via attribute '{name}'!")

    def __getitem__(self, item):
        raise AssertionError(f"SECURITY INVARIANT VIOLATION: Private raw data accessed via item '{item}'!")

    def __len__(self):
        raise AssertionError("SECURITY INVARIANT VIOLATION: Private raw data length inspected!")


def main():
    print("=" * 75)
    print("VERIFICATION: Small-n Utility-Aware Release Planner Architecture (v0.3)")
    print("=" * 75)

    base_config = load_app_config("configs")

    # 1. Static Public mode never touches raw user records
    print("[1] Verifying Static Public planner data isolation...")
    planner = SmallNReleasePlanner(app_config=base_config)
    poisoned_raw_matrix = PoisonedDataSentinel()

    # Plan should succeed without accessing poisoned_raw_matrix
    plan_res = planner.plan(
        metric="interaction_volume",
        recipient="recipient_internal",
        planning_horizon_days=90,
        user_matrix=poisoned_raw_matrix,
    )
    assert plan_res is not None
    assert plan_res.budget_feasible is True
    assert plan_res.policy is not None
    print(f"    [PASS] Static public planner selected policy: {plan_res.policy.candidate_id}")
    print("           Zero private raw data attributes or items accessed.")

    # 2. Oracle evaluation mode rejected in production
    print("[2] Verifying Oracle mode rejection in production...")
    oracle_config = load_app_config("configs")
    oracle_config.privacy.planner.mode = "oracle_eval_only"
    oracle_planner = SmallNReleasePlanner(app_config=oracle_config)

    try:
        oracle_planner.plan(
            metric="interaction_volume",
            recipient="recipient_external_a",
            planning_horizon_days=90,
        )
        raise AssertionError("Expected ProductionSafetyConfigurationError for oracle mode with external recipient!")
    except ProductionSafetyConfigurationError as e:
        print(f"    [PASS] Oracle mode correctly blocked: {e}")

    # 3. RDP Composition projection matches manual calculation
    print("[3] Verifying RDP composition projection across planning horizon...")
    test_cand = ReleasePolicyCandidate(
        contribution_bound=20.0,
        granularity="full_9_cell",
        frequency_days=30,  # 90 / 30 = 3 releases
        mechanism="gaussian",
        epsilon_per_release=1.0,
        delta_per_release=1e-6,
    )

    custom_cfg = load_app_config("configs")
    custom_cfg.privacy.planner.candidate_clipping = [20.0]
    custom_cfg.privacy.planner.candidate_granularities = ["full_9_cell"]
    custom_cfg.privacy.planner.candidate_windows = [30]
    custom_cfg.privacy.planner.candidate_frequencies = [30]
    custom_planner = SmallNReleasePlanner(app_config=custom_cfg)

    plan_res3 = custom_planner.plan(
        metric="interaction_volume",
        recipient="recipient_internal",
        planning_horizon_days=90,
    )
    assert plan_res3.planned_releases_count == 3

    # Manual calculation of 3 releases under RDP
    eff_orders = [Decimal(str(a)) for a in custom_cfg.privacy.accountant.orders]
    from private_llm_analytics.planning.utility import NoiseEstimator
    sigma, sens = NoiseEstimator.estimate_noise_sigma("interaction_volume", test_cand)
    single_rdp = compute_gaussian_rdp(sigma=sigma, sensitivity=sens, orders=eff_orders)
    triple_rdp = {o: single_rdp[o] * Decimal("3") for o in eff_orders}
    _, expected_eps = rdp_to_epsilon_delta(triple_rdp, delta_global=custom_cfg.privacy.accountant.delta_global)

    assert abs(plan_res3.projected_epsilon - float(expected_eps)) < 1e-9, (
        f"Projected eps {plan_res3.projected_epsilon} != expected {float(expected_eps)}"
    )
    print(f"    [PASS] Projected RDP (3 releases) matches exact math: eps={plan_res3.projected_epsilon:.4f}")

    # 4. Planning horizon does NOT reset privacy budget
    print("[4] Verifying planning horizon does not perform privacy reset...")
    # Mock state manager with prior spent budget
    class MockStateManager:
        def __init__(self, prior_rdp_multiplier=1.0):
            self.prior_mult = prior_rdp_multiplier

        def get_privacy_view(self, budget_scope_id, delta_global, orders, epsilon_cap, accountant_mode):
            from private_llm_analytics.privacy.composition import PrivacyView
            # Create pre-existing RDP history
            hist_rdp = {o: single_rdp[o] * Decimal(str(self.prior_mult)) for o in orders}
            _, eff_eps = rdp_to_epsilon_delta(hist_rdp, delta_global=delta_global)
            return PrivacyView(
                budget_scope_id=budget_scope_id,
                release_count=int(self.prior_mult),
                total_rdp_vector=hist_rdp,
                effective_epsilon=eff_eps,
                best_alpha=orders[0],
                delta_global=Decimal(str(delta_global)),
                epsilon_cap=Decimal(str(epsilon_cap)),
                remaining_epsilon=max(Decimal("0"), Decimal(str(epsilon_cap)) - eff_eps),
                accountant_mode=accountant_mode,
                orders=orders,
            )

    mock_sm = MockStateManager(prior_rdp_multiplier=2.0)
    history_planner = SmallNReleasePlanner(app_config=custom_cfg, state_manager=mock_sm)
    plan_with_history = history_planner.plan(
        metric="interaction_volume",
        recipient="recipient_internal",
        planning_horizon_days=90,
    )
    # Total releases: 2 historical + 3 planned = 5 releases equivalent
    five_rdp = {o: single_rdp[o] * Decimal("5") for o in eff_orders}
    _, expected_five_eps = rdp_to_epsilon_delta(five_rdp, delta_global=custom_cfg.privacy.accountant.delta_global)
    assert abs(plan_with_history.projected_epsilon - float(expected_five_eps)) < 1e-9
    print(f"    [PASS] Historical RDP preserved: 2 past + 3 planned = {plan_with_history.projected_epsilon:.4f} eps")

    # 5. Frequency-privacy tradeoff
    print("[5] Verifying frequency-privacy monotonicity...")
    freq_cfg = load_app_config("configs")
    freq_cfg.privacy.accountant.epsilon_cap = 3.0  # Constrained budget: fits F=30 (3 releases), rejects F=1 and F=7
    freq_cfg.privacy.planner.candidate_frequencies = [1, 7, 30]
    freq_cfg.privacy.planner.candidate_clipping = [20.0]
    freq_cfg.privacy.planner.candidate_granularities = ["full_9_cell"]
    freq_planner = SmallNReleasePlanner(app_config=freq_cfg)

    plan_freq = freq_planner.plan(
        metric="interaction_volume",
        recipient="recipient_internal",
        planning_horizon_days=90,
    )
    # Daily (90 releases) and Weekly (13 releases) exceed eps_cap=3.0, while F=30 (3 releases, eps~2.27) fits
    assert plan_freq.budget_feasible is True
    assert plan_freq.policy.frequency_days == 30, (
        f"Expected monthly frequency F=30 under budget cap 3.0, got F={plan_freq.policy.frequency_days}"
    )
    print(f"    [PASS] Budget cap 3.0 selected frequency F={plan_freq.policy.frequency_days} days (daily & weekly were rejected).")

    # 6. Fail-closed behavior when budget exhausted
    print("[6] Verifying fail-closed behavior on budget exhaustion...")
    exhausted_sm = MockStateManager(prior_rdp_multiplier=50.0) # Huge past consumption
    exhausted_planner = SmallNReleasePlanner(app_config=custom_cfg, state_manager=exhausted_sm)
    plan_exhausted = exhausted_planner.plan(
        metric="interaction_volume",
        recipient="recipient_internal",
        planning_horizon_days=90,
    )
    assert plan_exhausted.budget_feasible is False
    assert plan_exhausted.policy is None
    assert plan_exhausted.decision_reason == "NO_FEASIBLE_POLICY"
    print("    [PASS] Budget exhaustion cleanly yields NO_FEASIBLE_POLICY.")

    # 7. Memoization determinism
    print("[7] Verifying memoization determinism...")
    plan_first = planner.plan(
        metric="interaction_volume",
        recipient="recipient_internal",
        planning_horizon_days=90,
    )
    plan_second = planner.plan(
        metric="interaction_volume",
        recipient="recipient_internal",
        planning_horizon_days=90,
    )
    assert plan_first is plan_second, "Expected cached reference on identical state fingerprint"
    print("    [PASS] Memoization returned identical instance from state fingerprint cache.")

    print("=" * 75)
    print("ALL PLANNER ARCHITECTURE VERIFICATIONS PASSED")
    print("=" * 75)


if __name__ == "__main__":
    main()
