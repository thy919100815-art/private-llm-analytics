"""Small-n Utility-Aware Differential Privacy Release Planner (v0.3).

Automatically plans release policies (C, G, W, F) under:
- user-level central DP
- multi-recipient privacy isolation (per-recipient, collusion-group, global-external)
- longitudinal RDP composition without privacy resets
- strictly isolated execution modes (STATIC_PUBLIC, DP_FEEDBACK, ORACLE_EVAL_ONLY)
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal
import hashlib
import json
import math
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import uuid

from private_llm_analytics.config.models import AppConfig, PlannerConfig
from private_llm_analytics.planning.candidates import (
    ReleasePolicyCandidate,
    generate_candidate_set,
)
from private_llm_analytics.planning.utility import (
    LogNormalPrior,
    NoiseEstimator,
    PublicContributionPrior,
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


def compute_planner_config_fingerprint(config: Union[AppConfig, PlannerConfig, Any]) -> str:
    """Compute deterministic SHA-256 fingerprint over complete planner configuration."""
    if hasattr(config, "privacy") and hasattr(config.privacy, "planner"):
        planner = config.privacy.planner
        priv = config.privacy
    elif hasattr(config, "candidate_clipping"):
        planner = config
        priv = getattr(config, "privacy", None)
    else:
        raise ValueError(f"Expected AppConfig or PlannerConfig, got {type(config)}")

    workload = planner.workloads
    prior = planner.prior
    weights = planner.weights
    win_model = planner.window_model

    mech_type = priv.default_mechanism.type if priv and hasattr(priv, "default_mechanism") else "gaussian"
    q_eps = float(priv.default_query_budget.epsilon) if priv and hasattr(priv, "default_query_budget") else 1.0
    q_delta = float(priv.default_query_budget.delta) if priv and hasattr(priv, "default_query_budget") else 1e-6
    acc_mode = priv.accountant.mode if priv and hasattr(priv, "accountant") else "rdp"
    eps_cap = float(priv.accountant.epsilon_cap) if priv and hasattr(priv, "accountant") else 4.0
    delta_global = float(priv.accountant.delta_global) if priv and hasattr(priv, "accountant") else 1e-6
    orders = (
        [float(o) for o in priv.accountant.orders]
        if priv and hasattr(priv, "accountant")
        else [1.25, 1.5, 2.0, 3.0, 4.0, 5.0, 8.0, 10.0, 16.0, 32.0, 64.0, 128.0]
    )

    canonical: Dict[str, Any] = {
        "planner_mode": planner.mode,
        "candidate_clipping": sorted([float(c) for c in planner.candidate_clipping]),
        "candidate_granularities": sorted(list(planner.candidate_granularities)),
        "candidate_windows": sorted([int(w) for w in planner.candidate_windows]),
        "candidate_frequencies": sorted([int(f) for f in planner.candidate_frequencies]),
        "planning_horizon_days": int(planner.planning_horizon_days),
        "prior": {
            "distribution": prior.distribution,
            "mean": float(prior.mean),
            "std": float(prior.std),
            "lower": float(prior.lower),
            "upper": float(prior.upper),
        },
        "weights": {
            "clipping": float(weights.clipping),
            "noise": float(weights.noise),
            "aggregation": float(weights.aggregation),
            "staleness": float(weights.staleness),
            "window": float(weights.window),
        },
        "workloads": {
            "default_arrival_rate": float(workload.default_arrival_rate),
            "full_9_cell_weight": float(workload.full_9_cell_weight),
            "provider_weight": float(workload.provider_weight),
            "task_weight": float(workload.task_weight),
            "overall_weight": float(workload.overall_weight),
            "provider_shares": {k: float(v) for k, v in sorted(workload.provider_shares.items())},
            "task_shares": {k: float(v) for k, v in sorted(workload.task_shares.items())},
        },
        "window_model": {
            "reference_days": int(win_model.reference_days),
            "mean_scaling": win_model.mean_scaling,
            "std_scaling": win_model.std_scaling,
            "temporal_averaging_penalty": float(win_model.temporal_averaging_penalty),
        },
        "default_mechanism": mech_type,
        "query_epsilon": q_eps,
        "query_delta": q_delta,
        "accountant_mode": acc_mode,
        "epsilon_cap": eps_cap,
        "delta_global": delta_global,
        "orders": orders,
    }

    raw = json.dumps(canonical, sort_keys=True)
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()
    return f"pcfg_{digest[:24]}"


def compute_history_fingerprint(
    scope_id: str,
    release_count: int,
    total_rdp_vector: Dict[Decimal, Decimal],
) -> str:
    """Compute deterministic SHA-256 fingerprint of historical privacy state."""
    rdp_sorted = sorted([(str(k), str(v)) for k, v in total_rdp_vector.items()])
    raw = f"{scope_id}|{release_count}|{rdp_sorted}"
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()
    return f"hist_{digest[:24]}"


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
    input_provenance: List[str] = field(default_factory=list)
    planner_config_fingerprint: str = ""
    history_fingerprint: str = ""

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
            "input_provenance": self.input_provenance,
            "planner_config_fingerprint": self.planner_config_fingerprint,
            "history_fingerprint": self.history_fingerprint,
        }


class SmallNReleasePlanner:
    """Production small-n differential privacy release planner.

    Guarantees:
    1. Static public mode never touches private raw user records.
    2. Oracle mode is strictly blocked in production configurations.
    3. Future releases compose with historical RDP; planning horizon != privacy reset.
    4. Deterministic policy ranking with fail-closed behavior when unfeasible.
    5. Cache key strictly binds full planner configuration and historical state.
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
            if prior_cfg.distribution in ("contribution_model", "truncated_normal"):
                prior: PublicPriorDistribution = PublicContributionPrior(
                    mean=prior_cfg.mean,
                    std=prior_cfg.std,
                    min_val=prior_cfg.lower,
                    max_val=prior_cfg.upper,
                    reference_days=self.planner_config.window_model.reference_days,
                    window_days=self.planner_config.window_model.reference_days,
                    provider_shares=self.planner_config.workloads.provider_shares,
                    task_shares=self.planner_config.workloads.task_shares,
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
                window=w_cfg.window,
            )
            workload_cfg = self.planner_config.workloads
            workload_weights = {
                "full_9_cell": workload_cfg.full_9_cell_weight,
                "provider": workload_cfg.provider_weight,
                "task": workload_cfg.task_weight,
                "overall": workload_cfg.overall_weight,
            }
            self.utility_estimator = UtilityEstimator(
                weights=weights,
                public_prior=prior,
                reference_days=self.planner_config.window_model.reference_days,
                workload_weights=workload_weights,
                provider_shares=workload_cfg.provider_shares,
                task_shares=workload_cfg.task_shares,
                lambda_stale=workload_cfg.default_arrival_rate,
                lambda_window=self.planner_config.window_model.temporal_averaging_penalty,
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

        is_production_config = (
            self.app_config.privacy.release.reuse_existing_release
            and not self.app_config.privacy.release.allow_budget_overrun
            and self.app_config.privacy.history.mode == "lifetime"
        )

        if mode == "oracle_eval_only":
            if recip_role == "external":
                raise ProductionSafetyConfigurationError(
                    "Security violation: Oracle evaluation planning is strictly forbidden for external recipients."
                )
            if is_production_config:
                raise ProductionSafetyConfigurationError(
                    "Production configuration strictly forbids privacy.planner.mode='oracle_eval_only'."
                )

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

        # 4. Fingerprint computation & Input Provenance
        config_fp = compute_planner_config_fingerprint(self.app_config)
        history_fp = compute_history_fingerprint(budget_scope_id, historical_releases, historical_rdp)

        input_provenance: List[str] = []
        feedback_used = False
        decision_reason_prefix = ""
        metadata_extra: Dict[str, Any] = {}

        if mode == "oracle_eval_only":
            input_provenance.append("ORACLE_EXACT_EVAL")
            # In production, presence of ORACLE_EXACT_EVAL is an immediate safety violation
            if is_production_config:
                raise ProductionSafetyConfigurationError(
                    "Production safety violation: ORACLE_EXACT_EVAL input provenance detected."
                )
        elif mode == "dp_feedback":
            # Strict validation of feedback context
            if public_context is not None:
                # Reject raw exact data or non-DP aggregates
                forbidden_keys = {"raw_data", "exact_results", "unclipped_data", "raw_user_matrix"}
                if any(k in public_context for k in forbidden_keys):
                    raise ValueError(
                        "Security violation: dp_feedback mode strictly rejects non-DP raw data in feedback context."
                    )

                release_ids = public_context.get("feedback_release_ids", [])
                valid_ids = []
                if self.state_manager is not None and release_ids:
                    # Validate each release in state store
                    with self.state_manager._connection() as conn:
                        cursor = conn.cursor()
                        for rid in release_ids:
                            cursor.execute(
                                "SELECT status, budget_scope_id, release_key FROM release_store WHERE release_id = ?",
                                (rid,),
                            )
                            row = cursor.fetchone()
                            if row and row[0] == "COMMITTED" and row[1] == budget_scope_id:
                                valid_ids.append(rid)

                if valid_ids:
                    feedback_used = True
                    input_provenance.extend(["PUBLIC_CONFIG", "PUBLIC_PRIOR", "PUBLIC_POPULATION"])
                    for vid in valid_ids:
                        input_provenance.append(f"DP_RELEASE:{vid}")
                    metadata_extra["feedback_release_ids"] = valid_ids
                    metadata_extra["feedback_used"] = True
                else:
                    # Explicit fallback
                    feedback_used = False
                    input_provenance.extend(["PUBLIC_CONFIG", "PUBLIC_PRIOR", "PUBLIC_POPULATION"])
                    decision_reason_prefix = "DP_FEEDBACK_SAFE_FALLBACK_TO_STATIC_PUBLIC"
                    metadata_extra["planner_mode"] = "dp_feedback"
                    metadata_extra["feedback_used"] = False
                    metadata_extra["fallback"] = "static_public"
            else:
                # No context provided -> explicit fallback
                feedback_used = False
                input_provenance.extend(["PUBLIC_CONFIG", "PUBLIC_PRIOR", "PUBLIC_POPULATION"])
                decision_reason_prefix = "DP_FEEDBACK_SAFE_FALLBACK_TO_STATIC_PUBLIC"
                metadata_extra["planner_mode"] = "dp_feedback"
                metadata_extra["feedback_used"] = False
                metadata_extra["fallback"] = "static_public"
        else:
            # STATIC_PUBLIC
            input_provenance.extend(["PUBLIC_CONFIG", "PUBLIC_PRIOR", "PUBLIC_POPULATION"])

        # Context fingerprint
        ctx_raw = json.dumps(public_context or {}, sort_keys=True)
        ctx_fp = hashlib.sha256(ctx_raw.encode("utf-8")).hexdigest()[:12]

        cache_key = f"{metric}|{recipient}|{budget_scope_id}|{history_fp}|{config_fp}|{horizon}|{ctx_fp}"
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
                eps_projected = historical_eps + (k_releases * cand.epsilon_per_release)
                projected_rdp = {}

            # Budget feasibility check against authoritative epsilon_cap
            is_feasible = to_decimal(eps_projected) <= epsilon_cap

            # Estimate utility loss (normalized dimensionless loss)
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
            min_proj_eps = min(r["projected_epsilon"] for r in evaluated_candidates) if evaluated_candidates else float("inf")
            reason = decision_reason_prefix or "NO_FEASIBLE_POLICY"
            if decision_reason_prefix and decision_reason_prefix != "NO_FEASIBLE_POLICY":
                reason = f"{decision_reason_prefix}_NO_FEASIBLE_POLICY"

            meta = {
                "total_candidates": len(candidates),
                "feasible_candidates": 0,
                "planning_horizon_days": horizon,
                "epsilon_cap": float(epsilon_cap),
                "historical_spent_epsilon": historical_eps,
                **metadata_extra,
            }

            decision_record = {
                "planner_id": f"plan_{uuid.uuid4().hex[:12]}",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "metric": metric,
                "recipient": recipient,
                "scope": budget_scope_id,
                "selected_policy": None,
                "candidate_count": len(candidates),
                "predicted_utility": None,
                "privacy_projection": {
                    "projected_epsilon": min_proj_eps,
                    "epsilon_cap": float(epsilon_cap),
                    "budget_feasible": False,
                },
                "config_fingerprint": config_fp,
                "history_fingerprint": history_fp,
                "input_provenance": input_provenance,
            }
            meta["decision_record"] = decision_record

            result = ReleasePlanResult(
                policy=None,
                budget_feasible=False,
                projected_epsilon=min_proj_eps,
                historical_epsilon=historical_eps,
                projected_rdp_vector=historical_rdp,
                planned_releases_count=0,
                estimated_utility=None,
                budget_scope_id=budget_scope_id,
                decision_reason=reason,
                metadata=meta,
                candidate_evaluations=evaluated_candidates,
                input_provenance=input_provenance,
                planner_config_fingerprint=config_fp,
                history_fingerprint=history_fp,
            )
        else:
            # Multi-objective lexicographic ranking:
            # 1. Total normalized loss (ascending)
            # 2. Projected epsilon (ascending)
            # 3. Frequency days (ascending: fresher releases)
            # 4. Granularity rank (richer cells first)
            # 5. Window days (closer to reference days first)
            # 6. Deterministic candidate ID
            g_rank = {"full_9_cell": 0, "provider_3_cell": 1, "task_3_cell": 1, "overall_1_cell": 2}

            def ranking_key(item: Dict[str, Any]) -> Tuple[float, float, int, int, int, str]:
                cand = item["candidate"]
                w_dist = abs(cand.window_days - self.planner_config.window_model.reference_days)
                return (
                    round(item["total_error"], 7),
                    round(item["projected_epsilon"], 6),
                    cand.frequency_days,
                    g_rank.get(cand.granularity, 3),
                    w_dist,
                    cand.candidate_id,
                )

            best = min(feasible_candidates, key=ranking_key)
            reason = decision_reason_prefix or "OPTIMAL_POLICY_SELECTED"

            meta = {
                "total_candidates": len(candidates),
                "feasible_candidates": len(feasible_candidates),
                "planning_horizon_days": horizon,
                "epsilon_cap": float(epsilon_cap),
                "historical_spent_epsilon": historical_eps,
                **metadata_extra,
            }

            decision_record = {
                "planner_id": f"plan_{uuid.uuid4().hex[:12]}",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "metric": metric,
                "recipient": recipient,
                "scope": budget_scope_id,
                "selected_policy": best["candidate"].candidate_id,
                "candidate_count": len(candidates),
                "predicted_utility": best["utility"].to_dict(),
                "privacy_projection": {
                    "projected_epsilon": best["projected_epsilon"],
                    "epsilon_cap": float(epsilon_cap),
                    "budget_feasible": True,
                },
                "config_fingerprint": config_fp,
                "history_fingerprint": history_fp,
                "input_provenance": input_provenance,
            }
            meta["decision_record"] = decision_record

            result = ReleasePlanResult(
                policy=best["candidate"],
                budget_feasible=True,
                projected_epsilon=best["projected_epsilon"],
                historical_epsilon=historical_eps,
                projected_rdp_vector=best["projected_rdp"],
                planned_releases_count=best["k_releases"],
                estimated_utility=best["utility"],
                budget_scope_id=budget_scope_id,
                decision_reason=reason,
                metadata=meta,
                candidate_evaluations=evaluated_candidates,
                input_provenance=input_provenance,
                planner_config_fingerprint=config_fp,
                history_fingerprint=history_fp,
            )

        self._plan_cache[cache_key] = result
        return result
