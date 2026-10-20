"""Small-n Utility-Aware Differential Privacy Release Planner (v0.3).

Automatically plans release policies (C, G, W, F) under:
- user-level central DP
- multi-recipient privacy isolation (per-recipient, collusion-group, global-external)
- longitudinal RDP composition without privacy resets
- strictly isolated execution modes (STATIC_PUBLIC vs ORACLE_EVAL_ONLY)
"""

from dataclasses import dataclass, field
from decimal import Decimal
import hashlib
import json
import math
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from private_llm_analytics.config.models import AppConfig, PlannerConfig
from private_llm_analytics.planning.candidates import (
    ReleasePolicyCandidate,
    generate_candidate_set,
)
from private_llm_analytics.planning.utility import (
    LogNormalPrior,
    NoiseEstimator,
    PublicPriorDistribution,
    TruncatedNormalPrior,
    UtilityComponents,
    UtilityEstimator,
    UtilityWeights,
)
from private_llm_analytics.privacy.composition import resolve_budget_scope_id
from private_llm_analytics.privacy.exceptions import ProductionSafetyConfigurationError
from private_llm_analytics.privacy.rdp import (
    DEFAULT_RDP_ORDERS,
    compute_gaussian_rdp,
    compute_laplace_rdp_bound,
    compose_rdp_vectors,
    rdp_to_epsilon_delta,
    to_decimal,
)


@dataclass(frozen=True)
class ReleasePlanResult:
    """Outcome of utility-aware release planning."""

    policy: Optional[ReleasePolicyCandidate]
    budget_feasible: bool
    projected_epsilon: float
    historical_epsilon: float
    projected_rdp_vector: Dict[Decimal, Decimal]
    planned_releases_count: int
    estimated_utility: Optional[UtilityComponents]
    budget_scope_id: str
    decision_reason: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    candidate_evaluations: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "policy": self.policy.to_dict() if self.policy else None,
            "budget_feasible": self.budget_feasible,
            "projected_epsilon": self.projected_epsilon,
            "historical_epsilon": self.historical_epsilon,
            "projected_rdp_vector": {str(k): str(v) for k, v in self.projected_rdp_vector.items()},
            "planned_releases_count": self.planned_releases_count,
            "estimated_utility": self.estimated_utility.to_dict() if self.estimated_utility else None,
            "budget_scope_id": self.budget_scope_id,
            "decision_reason": self.decision_reason,
            "metadata": self.metadata,
        }


class SmallNReleasePlanner:
    """Production small-n differential privacy release planner.

    Guarantees:
    1. Static public mode never touches private raw user records.
    2. Oracle mode is strictly blocked in production configurations.
    3. Future releases compose with historical RDP; planning horizon != privacy reset.
    4. Deterministic policy ranking with fail-closed behavior when unfeasible.
    """

    def __init__(
        self,
        app_config: AppConfig,
        state_manager: Optional[Any] = None,
        utility_estimator: Optional[UtilityEstimator] = None,
    ):
        self.app_config = app_config
        self.state_manager = state_manager
        self.planner_config: PlannerConfig = app_config.privacy.planner

        # Setup utility estimator
        if utility_estimator is not None:
            self.utility_estimator = utility_estimator
        else:
            prior_cfg = self.planner_config.prior
            if prior_cfg.distribution == "truncated_normal":
                prior: PublicPriorDistribution = TruncatedNormalPrior(
                    mean=prior_cfg.mean,
                    std=prior_cfg.std,
                    min_val=prior_cfg.lower,
                    max_val=prior_cfg.upper,
                )
            elif prior_cfg.distribution == "lognormal":
                prior = LogNormalPrior(
                    mean=prior_cfg.mean,
                    std=prior_cfg.std,
                    max_cutoff=prior_cfg.upper,
                )
            else:
                raise ValueError(f"Unsupported prior distribution '{prior_cfg.distribution}'")

            w_cfg = self.planner_config.weights
            weights = UtilityWeights(
                clipping=w_cfg.clipping,
                noise=w_cfg.noise,
                aggregation=w_cfg.aggregation,
                staleness=w_cfg.staleness,
            )
            self.utility_estimator = UtilityEstimator(
                weights=weights,
                public_prior=prior,
                lambda_stale=self.planner_config.workloads.default_arrival_rate,
            )

        self._plan_cache: Dict[str, ReleasePlanResult] = {}

    def plan(
        self,
        metric: str,
        recipient: str,
        planning_horizon_days: Optional[int] = None,
        public_context: Optional[Dict[str, Any]] = None,
        user_matrix: Optional[Any] = None,
        force_recompute: bool = False,
    ) -> ReleasePlanResult:
        """Evaluate candidate policies and return optimal budget-feasible plan."""
        mode = self.planner_config.mode

        # 1. Mode safety enforcement
        recip_policy = self.app_config.recipients.recipients.get(recipient)
        recip_role = recip_policy.role if recip_policy else "external"

        if mode == "oracle_eval_only":
            # Forbid in production or when recipient is external
            if recip_role == "external":
                raise ProductionSafetyConfigurationError(
                    "Security violation: Oracle evaluation planning is strictly forbidden for external recipients."
                )
            if self.app_config.privacy.planner.mode == "oracle_eval_only":
                # Check production invariants
                if (
                    self.app_config.privacy.release.reuse_existing_release
                    and not self.app_config.privacy.release.allow_budget_overrun
                    and self.app_config.privacy.history.mode == "lifetime"
                ):
                    # In production configuration, oracle_eval_only is blocked
                    raise ProductionSafetyConfigurationError(
                        "Production configuration strictly forbids privacy.planner.mode='oracle_eval_only'."
                    )

        # In static_public mode, strictly do not touch user_matrix
        if mode == "static_public":
            # user_matrix is intentionally ignored to protect private data provenance
            pass

        # 2. Parameters & Budget scope resolution
        horizon = int(planning_horizon_days or self.planner_config.planning_horizon_days)
        if horizon <= 0:
            raise ValueError(f"planning_horizon_days must be > 0, got {horizon}")

        pop_size = int(self.app_config.scenario.population.n_users)
        comp_mode = self.app_config.privacy.recipient_composition.mode
        groups = self.app_config.privacy.recipient_composition.groups
        budget_scope_id = resolve_budget_scope_id(
            recipient=recipient,
            recipient_role=recip_role,
            mode=comp_mode,
            groups=groups,
        )

        eff_orders = [to_decimal(a) for a in self.app_config.privacy.accountant.orders]
        accountant_mode = self.app_config.privacy.accountant.mode
        delta_global = to_decimal(self.app_config.privacy.accountant.delta_global)
        epsilon_cap = to_decimal(self.app_config.privacy.accountant.epsilon_cap)

        # 3. Retrieve historical privacy state
        if self.state_manager is not None:
            view = self.state_manager.get_privacy_view(
                budget_scope_id=budget_scope_id,
                delta_global=delta_global,
                orders=eff_orders,
                epsilon_cap=epsilon_cap,
                accountant_mode=accountant_mode,
            )
            historical_rdp = view.total_rdp_vector
            historical_eps = float(view.effective_epsilon)
            historical_releases = view.release_count
        else:
            historical_rdp = {o: Decimal("0.0") for o in eff_orders}
            historical_eps = 0.0
            historical_releases = 0

        # 4. Fingerprint & Memoization
        rdp_sig = tuple((str(k), str(v)) for k, v in sorted(historical_rdp.items()))
        cache_key = (
            f"{metric}|{recipient}|{budget_scope_id}|{historical_releases}|"
            f"{rdp_sig}|{horizon}|{str(epsilon_cap)}|{mode}"
        )
        if not force_recompute and cache_key in self._plan_cache:
            return self._plan_cache[cache_key]

        # 5. Candidate enumeration
        candidates = generate_candidate_set(
            contribution_bounds=self.planner_config.candidate_clipping,
            granularities=self.planner_config.candidate_granularities,
            window_days=self.planner_config.candidate_windows,
            frequency_days=self.planner_config.candidate_frequencies,
            mechanisms=[self.app_config.privacy.default_mechanism.type],
            epsilon_per_release=self.app_config.privacy.default_query_budget.epsilon,
            delta_per_release=self.app_config.privacy.default_query_budget.delta,
        )

        # 6. Evaluate each candidate under composed RDP and utility
        evaluated_candidates: List[Dict[str, Any]] = []
        feasible_candidates: List[Dict[str, Any]] = []

        for cand in candidates:
            # Future release count K = ceil(H / F)
            k_releases = int(math.ceil(horizon / cand.frequency_days))

            # Compute candidate RDP contribution per release
            if accountant_mode == "rdp":
                sigma, sens = NoiseEstimator.estimate_noise_sigma(metric, cand)
                if cand.mechanism == "gaussian":
                    cand_single_rdp = compute_gaussian_rdp(
                        sigma=sigma,
                        sensitivity=sens,
                        orders=eff_orders,
                    )
                elif cand.mechanism == "laplace":
                    cand_single_rdp = compute_laplace_rdp_bound(
                        epsilon=cand.epsilon_per_release,
                        orders=eff_orders,
                    )
                else:
                    raise ValueError(f"Unsupported mechanism '{cand.mechanism}'")

                future_rdp = {
                    o: to_decimal(cand_single_rdp[o]) * Decimal(str(k_releases))
                    for o in eff_orders
                }
                projected_rdp = compose_rdp_vectors([historical_rdp, future_rdp], orders=eff_orders)
                best_alpha, dec_eps_proj = rdp_to_epsilon_delta(projected_rdp, delta_global=delta_global)
                eps_projected = float(dec_eps_proj)
            else:
                # Basic accountant: linear addition of epsilon
                eps_projected = historical_eps + (k_releases * cand.epsilon_per_release)
                projected_rdp = {}

            # Budget feasibility check against authoritative epsilon_cap
            is_feasible = to_decimal(eps_projected) <= epsilon_cap

            # Estimate utility
            util_comp = self.utility_estimator.evaluate_candidate(
                metric=metric,
                candidate=cand,
                population_size=pop_size,
            )

            record = {
                "candidate": cand,
                "feasible": is_feasible,
                "projected_epsilon": eps_projected,
                "projected_rdp": projected_rdp,
                "k_releases": k_releases,
                "utility": util_comp,
                "total_error": util_comp.total_error,
            }
            evaluated_candidates.append(record)
            if is_feasible:
                feasible_candidates.append(record)

        # 7. Select optimal policy or fail closed
        if not feasible_candidates:
            # No candidate can satisfy the remaining budget over planning horizon
            min_proj_eps = min(r["projected_epsilon"] for r in evaluated_candidates) if evaluated_candidates else float("inf")
            result = ReleasePlanResult(
                policy=None,
                budget_feasible=False,
                projected_epsilon=min_proj_eps,
                historical_epsilon=historical_eps,
                projected_rdp_vector=historical_rdp,
                planned_releases_count=0,
                estimated_utility=None,
                budget_scope_id=budget_scope_id,
                decision_reason="NO_FEASIBLE_POLICY",
                metadata={
                    "total_candidates": len(candidates),
                    "feasible_candidates": 0,
                    "planning_horizon_days": horizon,
                    "epsilon_cap": float(epsilon_cap),
                    "historical_spent_epsilon": historical_eps,
                },
                candidate_evaluations=evaluated_candidates,
            )
        else:
            # Tie-break priority:
            # 1. Total utility error (ascending)
            # 2. Projected epsilon (ascending: lower budget consumption)
            # 3. Frequency days (ascending: 1 > 7 > 14 > 30, fresher releases)
            # 4. Granularity ranking (full_9_cell > provider_3_cell / task_3_cell > overall_1_cell)
            # 5. Deterministic candidate_id
            g_rank = {"full_9_cell": 0, "provider_3_cell": 1, "task_3_cell": 1, "overall_1_cell": 2}

            def ranking_key(item: Dict[str, Any]) -> Tuple[float, float, int, int, str]:
                cand = item["candidate"]
                return (
                    round(item["total_error"], 6),
                    round(item["projected_epsilon"], 6),
                    cand.frequency_days,
                    g_rank.get(cand.granularity, 3),
                    cand.candidate_id,
                )

            best = min(feasible_candidates, key=ranking_key)
            result = ReleasePlanResult(
                policy=best["candidate"],
                budget_feasible=True,
                projected_epsilon=best["projected_epsilon"],
                historical_epsilon=historical_eps,
                projected_rdp_vector=best["projected_rdp"],
                planned_releases_count=best["k_releases"],
                estimated_utility=best["utility"],
                budget_scope_id=budget_scope_id,
                decision_reason="OPTIMAL_POLICY_SELECTED",
                metadata={
                    "total_candidates": len(candidates),
                    "feasible_candidates": len(feasible_candidates),
                    "planning_horizon_days": horizon,
                    "epsilon_cap": float(epsilon_cap),
                    "historical_spent_epsilon": historical_eps,
                },
                candidate_evaluations=evaluated_candidates,
            )

        self._plan_cache[cache_key] = result
        return result
