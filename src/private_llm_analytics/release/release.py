"""Release orchestrator coordinating planning, accounting, caching, and filtering in v0.1.4."""

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Union
import uuid
import numpy as np

from private_llm_analytics.aggregation.user_matrix import UserMatrix
from private_llm_analytics.config.models import AppConfig
from private_llm_analytics.config.validator import (
    ExecutionMode,
    ExperimentAccountingMode,
    validate_production_config,
)
from private_llm_analytics.mechanisms.gaussian import GaussianMechanism
from private_llm_analytics.mechanisms.laplace import LaplaceMechanism
from private_llm_analytics.metrics.base import DPReleaseResult
from private_llm_analytics.metrics.catalog import MetricCatalog
from private_llm_analytics.planner.plan import ExecutionPlan
from private_llm_analytics.planner.planner import AnalyticsPlanner
from private_llm_analytics.planner.query import AnalyticsQuery
from private_llm_analytics.privacy.accountant import BasicAccountant, PrivacyAccountant
from private_llm_analytics.privacy.budget import PrivacyBudget
from private_llm_analytics.privacy.exceptions import (
    ExperimentProductionStateCollisionError,
    PrivacyBudgetExceededError,
    PrivacyConfigurationError,
    ProductionRNGControlError,
    ProductionSafetyConfigurationError,
    RecipientSecurityViolation,
    UnsupportedMechanismError,
)
from private_llm_analytics.privacy.composition import PrivacyView, resolve_budget_scope_id
from private_llm_analytics.privacy.rdp import (
    DEFAULT_RDP_ORDERS,
    compute_gaussian_rdp,
    compute_laplace_rdp_bound,
    rdp_vector_to_json,
    to_decimal,
)
from private_llm_analytics.privacy.ledger import PrivacyLedger
from private_llm_analytics.privacy.rdp_accountant import RDPAccountant
from private_llm_analytics.privacy.state import PrivacyStateManager, _to_decimal
from private_llm_analytics.release.cache import PersistentReleaseStore
from private_llm_analytics.release.recipients import filter_release_for_recipient



class ReleaseOrchestrator:
    """Manages the lifecycle of DP releases with strict pre-authorization, atomic budget reservation, and single-flight memoization."""

    def __init__(
        self,
        app_config: AppConfig,
        cache: Optional[PersistentReleaseStore] = None,
        ledger: Optional[PrivacyLedger] = None,
        accountant: Optional[PrivacyAccountant] = None,
        catalog: Optional[MetricCatalog] = None,
        state_manager: Optional[PrivacyStateManager] = None,
        test_hooks: Optional[Dict[str, Any]] = None,
        execution_mode: Optional[Union[str, ExecutionMode]] = None,
        accounting_mode: Optional[Union[str, ExperimentAccountingMode]] = None,
    ):
        self.config = app_config
        self.planner = AnalyticsPlanner(app_config)
        self.catalog = catalog or MetricCatalog(app_config.metrics)
        self.test_hooks = test_hooks or {}

        # Resolve execution mode
        if execution_mode is not None:
            self.execution_mode = (
                ExecutionMode(execution_mode) if isinstance(execution_mode, str) else execution_mode
            )
        else:
            if (not app_config.privacy.release.reuse_existing_release) or app_config.privacy.release.allow_budget_overrun:
                self.execution_mode = ExecutionMode.EXPERIMENT
            else:
                self.execution_mode = ExecutionMode.PRODUCTION

        if self.execution_mode == ExecutionMode.PRODUCTION:
            validate_production_config(app_config)

        # Resolve experiment accounting mode
        if self.execution_mode == ExecutionMode.EXPERIMENT:
            if accounting_mode is not None:
                self.accounting_mode = (
                    ExperimentAccountingMode(accounting_mode)
                    if isinstance(accounting_mode, str)
                    else accounting_mode
                )
            elif getattr(app_config.privacy.release, "allow_budget_overrun", False):
                self.accounting_mode = ExperimentAccountingMode.EVALUATION
            else:
                self.accounting_mode = ExperimentAccountingMode.BOUNDED
        else:
            self.accounting_mode = None

        # Determine state manager with isolated scoping when test provides temp ledger/cache
        if state_manager is not None:
            self.state_manager = state_manager
        elif ledger is not None:
            self.state_manager = PrivacyStateManager(
                ledger.db_path.parent / ("experiment_state.db" if self.execution_mode == ExecutionMode.EXPERIMENT else "privacy_state.db")
            )
        elif cache is not None:
            self.state_manager = PrivacyStateManager(
                cache.db_path.parent / ("experiment_state.db" if self.execution_mode == ExecutionMode.EXPERIMENT else "privacy_state.db")
            )
        elif self.execution_mode == ExecutionMode.EXPERIMENT:
            self.state_manager = PrivacyStateManager("results/experiment_state.db")
        elif hasattr(app_config.privacy.release, "state_db_path") and app_config.privacy.release.state_db_path:
            self.state_manager = PrivacyStateManager(app_config.privacy.release.state_db_path)
        else:
            self.state_manager = PrivacyStateManager("results/privacy_state.db")

        # Fail closed if in EXPERIMENT mode and state_manager points to ANY production privacy state database (Fix B)
        if self.execution_mode == ExecutionMode.EXPERIMENT:
            prod_default = (Path("results") / "privacy_state.db").resolve()
            configured_prod = (
                Path(app_config.privacy.release.state_db_path).resolve()
                if hasattr(app_config.privacy.release, "state_db_path") and app_config.privacy.release.state_db_path
                else prod_default
            )
            canonical_prod_paths = {prod_default, configured_prod}
            if self.state_manager.db_path.resolve() in canonical_prod_paths:
                raise ExperimentProductionStateCollisionError(
                    f"Experiment mode cannot use production privacy state database ('{self.state_manager.db_path}'). "
                    "Use an isolated state database (e.g. 'results/experiment_state.db')."
                )

        # Persistent release store & ledger for legacy backward compatibility
        is_default_prod_db = (
            self.state_manager.db_path.resolve() == (Path("results") / "privacy_state.db").resolve()
        )
        if not is_default_prod_db:
            default_cache_p = self.state_manager.db_path.parent / f"{self.state_manager.db_path.stem}_cache.db"
            default_ledger_p = self.state_manager.db_path.parent / f"{self.state_manager.db_path.stem}_ledger.db"
        else:
            default_cache_p = getattr(app_config.privacy.release, "cache_db_path", "results/release_cache.db")
            default_ledger_p = getattr(app_config.privacy.release, "ledger_db_path", "results/privacy_ledger.db")

        self.cache = cache or PersistentReleaseStore(default_cache_p)
        self.ledger = ledger or PrivacyLedger(default_ledger_p)

        self.accountant = accountant or BasicAccountant(
            global_budget=PrivacyBudget(
                epsilon=app_config.privacy.global_budget.epsilon,
                delta=app_config.privacy.global_budget.delta,
            ),
            allow_budget_overrun=app_config.privacy.release.allow_budget_overrun,
            ledger=self.ledger,
            state_manager=self.state_manager,
        )

        acct_cfg = getattr(app_config.privacy, "accountant", None)
        self.accountant_mode = getattr(acct_cfg, "mode", "rdp") if acct_cfg else "rdp"
        if self.accountant_mode == "rdp" and acct_cfg:
            self.epsilon_cap = getattr(acct_cfg, "epsilon_cap", 4.0)
            self.delta_global = getattr(acct_cfg, "delta_global", 1e-6)
            self.orders = getattr(acct_cfg, "orders", DEFAULT_RDP_ORDERS)
        else:
            self.epsilon_cap = app_config.privacy.global_budget.epsilon
            self.delta_global = app_config.privacy.global_budget.delta
            self.orders = DEFAULT_RDP_ORDERS

        self.rdp_accountant = RDPAccountant(
            epsilon_cap=self.epsilon_cap,
            delta_global=self.delta_global,
            orders=self.orders,
            state_manager=self.state_manager,
        )

    def _get_rdp_policy(self):
        acct_cfg = getattr(self.config.privacy, "accountant", None)
        if acct_cfg and getattr(acct_cfg, "mode", "rdp") == "rdp":
            eps_cap = _to_decimal(getattr(acct_cfg, "epsilon_cap", 4.0))
            delta_g = _to_decimal(getattr(acct_cfg, "delta_global", 1e-6))
            orders = getattr(acct_cfg, "orders", DEFAULT_RDP_ORDERS)
        else:
            eps_cap = _to_decimal(self.config.privacy.global_budget.epsilon)
            delta_g = _to_decimal(self.config.privacy.global_budget.delta)
            orders = DEFAULT_RDP_ORDERS
        return eps_cap, delta_g, orders


    def _trigger_hook(self, hook_name: str) -> None:
        """Helper to invoke test fault injection hooks."""
        if not self.test_hooks or hook_name not in self.test_hooks:
            return
        action = self.test_hooks[hook_name]
        if callable(action):
            action()
        elif isinstance(action, BaseException):
            raise action
        elif isinstance(action, type) and issubclass(action, BaseException):
            raise action(f"Fault injected at {hook_name}")
        else:
            raise RuntimeError(f"Fault injected at {hook_name}")

    def execute_release(
        self,
        query: AnalyticsQuery,
        user_matrix: UserMatrix,
        dataset_version: str = "v1.0",
        rng: Optional[np.random.Generator] = None,
        include_exact: bool = False,
        noise_seed: Optional[int] = None,
        trial_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Execute or retrieve a DP release with strict pre-authorization and persistence.

        Execution pipeline:
          1. Recipient existence & permissions check
          2. External caller RNG restriction check (Audit C6)
          3. Exact-result policy check
          4. Full 15-point pre-validation & canonical fingerprint generation via planner (Audit H5, H6, H7)
          5. Fault injection hook: before_budget_reserve
          6. Atomic compare-and-reserve transaction via state_manager (Audit C7)
          7. Cache-hit check: if hit, memoized payload returned (0 spend)
          8. DP noise generation using secure internal RNG (or test RNG for trusted caller)
          9. Atomic release commit to persistent SQLite (Audit C8)
          10. Recipient filtering
        """
        # 1. Pre-authorization & Policy Checks
        rec_name = query.recipient
        if rec_name not in self.config.recipients.recipients:
            raise ValueError(f"Unknown recipient '{rec_name}'")
        policy = self.config.recipients.recipients[rec_name]

        # Experiment Mode external recipient restriction (v0.1.7 Gate Fix A)
        if self.execution_mode == ExecutionMode.EXPERIMENT and policy.role == "external":
            raise RecipientSecurityViolation(
                f"Experiment mode must never target external recipients (attempted for '{rec_name}')."
            )

        # 2. Strict RNG Restriction: Forbidden in production for all recipients, and forbidden for external callers (Audit F1, C6, v0.1.5 Finding B)
        caller_has_rng_control = (rng is not None) or (noise_seed is not None)
        if policy.role == "external" and caller_has_rng_control:
            raise ProductionRNGControlError(
                f"External recipient '{rec_name}' is strictly forbidden from supplying or controlling RNG / seed."
            )
        elif self.execution_mode == ExecutionMode.PRODUCTION and caller_has_rng_control:
            raise ProductionRNGControlError(
                f"Production release strictly forbids caller-supplied RNG / seed for all recipients (attempted for '{rec_name}')."
            )

        # 3. Exact results authorization check
        if include_exact:
            if policy.role == "external" or not policy.exact_results:
                raise RecipientSecurityViolation(
                    f"Recipient '{rec_name}' is strictly forbidden from receiving exact results."
                )

        # 4. DP results authorization check
        if not policy.dp_results:
            raise RecipientSecurityViolation(
                f"Recipient '{rec_name}' is not authorized to receive DP results (dp_results=False)."
            )

        # 5. Metric authorization check
        if "*" not in policy.metrics and query.metric not in policy.metrics:
            raise PermissionError(
                f"Recipient '{rec_name}' is not authorized to query metric '{query.metric}'"
            )

        # Populate public population size from config if mode is fixed_public_population
        if user_matrix.public_population_size is None and self.config.scenario.population.mode == "fixed_public_population":
            user_matrix.public_population_size = self.config.scenario.population.n_users

        # 6. Pre-validation and Plan generation (validates metric, mechanism, population size, etc.)
        plan: ExecutionPlan = self.planner.create_plan(
            query,
            dataset_version=dataset_version,
            public_population_size=user_matrix.public_population_size,
            execution_mode=self.execution_mode,
            noise_seed=noise_seed,
            trial_id=trial_id,
        )

        # 7. Check if cache reuse is enabled (fail-closed in production)
        reuse_cache = (
            True
            if self.execution_mode == ExecutionMode.PRODUCTION
            else self.config.privacy.release.reuse_existing_release
        )

        recipient_role = policy.role
        rc_config = getattr(self.config.privacy, "recipient_composition", None)
        rc_mode = rc_config.mode if rc_config else "global_external"
        rc_groups = rc_config.groups if rc_config else None
        budget_scope_id = resolve_budget_scope_id(
            recipient=query.recipient,
            recipient_role=recipient_role,
            mode=rc_mode,
            groups=rc_groups,
        )
        effective_accountant_mode = "basic" if budget_scope_id == "internal" else self.accountant_mode
        if effective_accountant_mode == "rdp":
            effective_epsilon_cap, effective_delta_global, effective_orders = self._get_rdp_policy()
        else:
            effective_epsilon_cap = _to_decimal(self.config.privacy.global_budget.epsilon)
            effective_delta_global = _to_decimal(self.config.privacy.global_budget.delta)
            effective_orders = DEFAULT_RDP_ORDERS

        # 8. Check persistent state (single source of truth: no legacy cache fallback)
        if reuse_cache:
            committed = self.state_manager.get_committed_release(plan.release_key)
            if committed is not None:
                # Cache hit: 0 budget consumed
                self._record_cache_hit(
                    plan, committed, query.recipient, dataset_version,
                    budget_scope_id=budget_scope_id,
                    accountant_mode=effective_accountant_mode,
                )
                return self._finalize_output(
                    release_result=committed,
                    query=query,
                    policy=policy,
                    user_matrix=user_matrix,
                    include_exact=include_exact,
                    cache_hit=True,
                    release_key=plan.release_key,
                    budget_scope_id=budget_scope_id,
                    accountant_mode=effective_accountant_mode,
                    effective_epsilon=getattr(committed, "effective_epsilon", None),
                    best_alpha=getattr(committed, "best_alpha", None),
                )

        # 9. Fault hook: before_budget_reserve
        self._trigger_hook("before_budget_reserve")

        # 10. Atomic compare-and-reserve transaction
        budget_needed = PrivacyBudget(epsilon=plan.epsilon, delta=plan.delta)
        global_budget = PrivacyBudget(
            epsilon=self.config.privacy.global_budget.epsilon,
            delta=self.config.privacy.global_budget.delta,
        )

        # Calibrate mechanism scale and compute candidate RDP vector
        if "gaussian" in plan.mechanism:
            mech_for_scale = GaussianMechanism(
                epsilon=plan.epsilon,
                delta=plan.delta,
                sensitivity=plan.sensitivity,
                method="analytic",
            )
            candidate_sigma = mech_for_scale.noise_scale
            candidate_rdp = compute_gaussian_rdp(
                sigma=candidate_sigma,
                sensitivity=plan.sensitivity,
                orders=effective_orders,
            )
        elif "laplace" in plan.mechanism:
            mech_for_scale = LaplaceMechanism(
                epsilon=plan.epsilon,
                delta=plan.delta,
                sensitivity=plan.sensitivity,
            )
            candidate_sigma = mech_for_scale.noise_scale
            candidate_rdp = compute_laplace_rdp_bound(
                epsilon=plan.epsilon,
                orders=effective_orders,
            )
        else:
            raise UnsupportedMechanismError(f"Unsupported mechanism: {plan.mechanism}")

        plan_metadata = {
            "metric": plan.metric,
            "window_start": str(plan.time_window[0]) if plan.time_window[0] else None,
            "window_end": str(plan.time_window[1]) if plan.time_window[1] else None,
            "dataset_version": dataset_version,
            "mechanism": plan.mechanism,
            "sensitivity": plan.sensitivity,
            "noise_scale": candidate_sigma,
            "cell_names": user_matrix.cell_names,
            "n_users": user_matrix.n_users,
            "public_population_size": user_matrix.public_population_size,
        }

        single_flight_timeout = getattr(self.config.privacy.release, "single_flight_timeout", 30.0)
        lease_timeout = getattr(self.config.privacy.release, "lease_timeout", 300.0)
        if self.execution_mode == ExecutionMode.PRODUCTION:
            allow_overrun = False
        elif self.accounting_mode == ExperimentAccountingMode.EVALUATION:
            allow_overrun = True
        else:
            allow_overrun = self.config.privacy.release.allow_budget_overrun

        res_tuple = self.state_manager.reserve_or_get_release(
            release_key=plan.release_key,
            canonical_semantics=plan.canonical_semantics,
            global_budget=global_budget,
            budget_needed=budget_needed,
            plan_metadata=plan_metadata,
            allow_overrun=allow_overrun,
            single_flight_timeout=single_flight_timeout,
            lease_timeout=lease_timeout,
            reuse_cache=reuse_cache,
            accountant_mode=effective_accountant_mode,
            budget_scope_id=budget_scope_id,
            recipient=query.recipient,
            recipient_role=recipient_role,
            rdp_orders=effective_orders,
            rdp_candidate_cost=candidate_rdp,
            epsilon_cap=effective_epsilon_cap,
            delta_global=effective_delta_global,
        )

        status, cached_res, reservation_id, release_id = res_tuple
        generation = getattr(res_tuple, "generation", 1)

        if status == "CACHE_HIT" and cached_res is not None:
            self._record_cache_hit(
                plan, cached_res, query.recipient, dataset_version,
                budget_scope_id=budget_scope_id,
                accountant_mode=effective_accountant_mode,
            )
            return self._finalize_output(
                release_result=cached_res,
                query=query,
                policy=policy,
                user_matrix=user_matrix,
                include_exact=include_exact,
                cache_hit=True,
                release_key=plan.release_key,
                budget_scope_id=budget_scope_id,
                accountant_mode=effective_accountant_mode,
                effective_epsilon=getattr(res_tuple, "effective_epsilon", None),
                best_alpha=getattr(res_tuple, "best_alpha", None),
            )

        # 11. Execute DP Generation with reservation active
        committed = False
        eff_eps_committed = None
        best_a_committed = None
        try:
            self._trigger_hook("after_budget_reserve")

            # Determine RNG: secure OS entropy for production/external callers
            if rng is not None:
                execution_rng = rng
            elif noise_seed is not None and self.execution_mode == ExecutionMode.EXPERIMENT:
                execution_rng = np.random.default_rng(noise_seed)
            else:
                execution_rng = np.random.default_rng()

            if "gaussian" in plan.mechanism:
                mechanism = GaussianMechanism(
                    epsilon=plan.epsilon,
                    delta=plan.delta,
                    sensitivity=plan.sensitivity,
                    method="analytic",
                )
            elif "laplace" in plan.mechanism:
                mechanism = LaplaceMechanism(
                    epsilon=plan.epsilon,
                    delta=plan.delta,
                    sensitivity=plan.sensitivity,
                )
            else:
                raise UnsupportedMechanismError(f"Unsupported mechanism: {plan.mechanism}")

            metric = self.catalog.get(plan.metric)
            release_result = metric.release_dp(user_matrix, mechanism, rng=execution_rng)

            self._trigger_hook("after_noise_generation")
            self._trigger_hook("before_release_commit")

            # Commit release atomically with generation fencing (Audit C11)
            eff_eps_committed, best_a_committed = self.state_manager.commit_release(
                reservation_id=reservation_id,
                release_key=plan.release_key,
                release_id=release_id,
                release_result=release_result,
                recipient=query.recipient,
                dataset_version=dataset_version,
                window_start=str(plan.time_window[0]) if plan.time_window[0] else None,
                window_end=str(plan.time_window[1]) if plan.time_window[1] else None,
                generation=generation,
                budget_scope_id=budget_scope_id,
                accountant_mode=effective_accountant_mode,
                rdp_cost_json=rdp_vector_to_json(candidate_rdp) if candidate_rdp else None,
                delta_global=effective_delta_global,
                orders=effective_orders,
            )
            committed = True

            try:
                self.ledger.record_release(
                    metric=plan.metric,
                    recipient=query.recipient,
                    dataset_version=dataset_version,
                    mechanism=plan.mechanism,
                    epsilon=plan.epsilon,
                    delta=plan.delta,
                    sensitivity=plan.sensitivity,
                    noise_scale=release_result.noise_scale,
                    release_key=plan.release_key,
                    cache_hit=False,
                    window_start=str(plan.time_window[0]) if plan.time_window[0] else None,
                    window_end=str(plan.time_window[1]) if plan.time_window[1] else None,
                )
            except Exception:
                pass

            self._trigger_hook("after_release_commit")

        except Exception as exc:
            # Commit irreversibility: Never abort an already committed reservation (Audit C12)
            if not committed and reservation_id is not None:
                try:
                    self.state_manager.abort_reservation(
                        reservation_id, plan.release_key, generation=generation
                    )
                except Exception:
                    pass
            raise exc

        return self._finalize_output(
            release_result=release_result,
            query=query,
            policy=policy,
            user_matrix=user_matrix,
            include_exact=include_exact,
            cache_hit=False,
            release_key=plan.release_key,
            budget_scope_id=budget_scope_id,
            accountant_mode=effective_accountant_mode,
            effective_epsilon=eff_eps_committed,
            best_alpha=best_a_committed,
        )

    def _record_cache_hit(
        self,
        plan: ExecutionPlan,
        release_result: DPReleaseResult,
        recipient: str,
        dataset_version: str,
        budget_scope_id: str = "global_external",
        accountant_mode: str = "basic",
    ) -> None:
        try:
            self.state_manager.record_cache_hit_ledger(
                release_id=str(uuid.uuid4()),
                release_key=plan.release_key,
                release_result=release_result,
                recipient=recipient,
                dataset_version=dataset_version,
                window_start=str(plan.time_window[0]) if plan.time_window[0] else None,
                window_end=str(plan.time_window[1]) if plan.time_window[1] else None,
                budget_scope_id=budget_scope_id,
                accountant_mode=accountant_mode,
            )
        except Exception:
            pass
        try:
            self.ledger.record_release(
                metric=plan.metric,
                recipient=recipient,
                dataset_version=dataset_version,
                mechanism=plan.mechanism,
                epsilon=plan.epsilon,
                delta=plan.delta,
                sensitivity=plan.sensitivity,
                noise_scale=release_result.noise_scale,
                release_key=plan.release_key,
                cache_hit=True,
                window_start=str(plan.time_window[0]) if plan.time_window[0] else None,
                window_end=str(plan.time_window[1]) if plan.time_window[1] else None,
            )
        except Exception:
            pass

    def _finalize_output(
        self,
        release_result: DPReleaseResult,
        query: AnalyticsQuery,
        policy: Any,
        user_matrix: UserMatrix,
        include_exact: bool,
        cache_hit: bool,
        release_key: str,
        budget_scope_id: str = "global_external",
        accountant_mode: str = "basic",
        effective_epsilon: Optional[Any] = None,
        best_alpha: Optional[Any] = None,
    ) -> Dict[str, Any]:
        exact_data = None
        if policy.role == "trusted" and include_exact:
            metric = self.catalog.get(query.metric)
            exact_clean = metric.compute_exact(user_matrix)
            exact_data = {
                "clean_vector": exact_clean.tolist(),
                "cell_names": user_matrix.cell_names,
            }

        output = filter_release_for_recipient(
            release_result=release_result,
            recipient_name=query.recipient,
            policy=policy,
            include_exact=include_exact,
            exact_data=exact_data,
        )
        output["cache_hit"] = cache_hit
        output["release_key"] = release_key
        output["incremental_privacy_cost"] = 0.0 if cache_hit else float(release_result.epsilon)
        output["incremental_privacy_cost_decimal"] = "0.0" if cache_hit else str(_to_decimal(release_result.epsilon))

        # Attach privacy accounting metadata
        eff_eps_float = float(effective_epsilon) if effective_epsilon is not None else None
        eff_eps_str = str(effective_epsilon) if effective_epsilon is not None else None
        best_alpha_float = float(best_alpha) if best_alpha is not None else None

        if accountant_mode == "rdp":
            eff_cap, eff_delta_g, _ = self._get_rdp_policy()
        else:
            eff_cap = _to_decimal(self.config.privacy.global_budget.epsilon)
            eff_delta_g = _to_decimal(self.config.privacy.global_budget.delta)

        output["privacy_accounting"] = {
            "accountant_mode": accountant_mode,
            "budget_scope_id": budget_scope_id,
            "effective_epsilon": eff_eps_float,
            "effective_epsilon_decimal": eff_eps_str,
            "best_alpha": best_alpha_float,
            "epsilon_cap": float(eff_cap),
            "delta_global": float(eff_delta_g),
        }
        return output

    def get_privacy_view(self, recipient: str = "global_external") -> PrivacyView:
        """Get authoritative PrivacyView for a given recipient or budget scope."""
        policy = self.config.recipients.recipients.get(recipient)
        rec_role = policy.role if policy else ("trusted" if recipient == "internal" else "external")
        rc_config = getattr(self.config.privacy, "recipient_composition", None)
        rc_mode = rc_config.mode if rc_config else "global_external"
        rc_groups = rc_config.groups if rc_config else None

        scope_id = resolve_budget_scope_id(
            recipient=recipient,
            recipient_role=rec_role,
            mode=rc_mode,
            groups=rc_groups,
        )

        eff_mode = "basic" if scope_id == "internal" else self.accountant_mode
        if eff_mode == "rdp":
            eff_cap, eff_delta_g, eff_orders = self._get_rdp_policy()
        else:
            eff_cap = _to_decimal(self.config.privacy.global_budget.epsilon)
            eff_delta_g = _to_decimal(self.config.privacy.global_budget.delta)
            eff_orders = DEFAULT_RDP_ORDERS

        return self.state_manager.get_privacy_view(
            budget_scope_id=scope_id,
            delta_global=eff_delta_g,
            orders=eff_orders,
            epsilon_cap=eff_cap,
            accountant_mode=eff_mode,
        )


