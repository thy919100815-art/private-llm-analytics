"""Analytics Planner generating execution plans from configs and queries in v0.1.4."""

from datetime import datetime
import hashlib
import json
from typing import Any, Dict, Optional, Tuple, Union
import pandas as pd

from private_llm_analytics.config.models import AppConfig
from private_llm_analytics.config.validator import ExecutionMode
from private_llm_analytics.contribution.sensitivity import get_metric_sensitivities
from private_llm_analytics.planner.plan import ExecutionPlan
from private_llm_analytics.planner.query import AnalyticsQuery
from private_llm_analytics.privacy.composition import resolve_budget_scope_id
from private_llm_analytics.privacy.exceptions import (
    MetricDisabledError,
    PrivacyConfigurationError,
    RecipientSecurityViolation,
    UnsupportedMechanismError,
)


class AnalyticsPlanner:
    """Plans query execution, determines sensitivities, budgets, and stable release keys."""

    def __init__(self, app_config: AppConfig):
        self.config = app_config

    def create_plan(
        self,
        query: AnalyticsQuery,
        dataset_version: str = "v1.0",
        public_population_size: Optional[int] = None,
        execution_mode: Union[str, ExecutionMode] = ExecutionMode.PRODUCTION,
        noise_seed: Optional[int] = None,
        trial_id: Optional[str] = None,
        privacy_scope_id: Optional[str] = None,
    ) -> ExecutionPlan:
        """Create an ExecutionPlan for an analytics query with full pre-validation."""
        # 1. Metric existence and enablement check
        metric_name = query.metric
        if metric_name not in self.config.metrics.metrics:
            raise ValueError(f"Unknown metric '{metric_name}'")
        m_def = self.config.metrics.metrics[metric_name]
        if not m_def.enabled:
            raise MetricDisabledError(f"Metric '{metric_name}' is disabled in configuration.")

        # 2. Recipient policy check
        rec_name = query.recipient
        if rec_name not in self.config.recipients.recipients:
            raise ValueError(f"Unknown recipient '{rec_name}'")
        rec_policy = self.config.recipients.recipients[rec_name]

        # 2.1 Check DP authorization
        if not rec_policy.dp_results:
            raise RecipientSecurityViolation(f"Recipient '{rec_name}' is not authorized for DP results (dp_results=False)")

        # 2.2 Check metric authorization
        if "*" not in rec_policy.metrics and metric_name not in rec_policy.metrics:
            raise PermissionError(
                f"Recipient '{rec_name}' is not authorized to query metric '{metric_name}'"
            )

        # 3. Time window
        t_start = pd.to_datetime(query.window_start) if query.window_start else None
        t_end = pd.to_datetime(query.window_end) if query.window_end else None

        # 4. Mechanism and budget determination with strict parameter validation
        raw_mech = query.mechanism or m_def.mechanism.type or self.config.privacy.default_mechanism.type
        mech_type = raw_mech.lower()
        if mech_type not in ("gaussian", "laplace"):
            raise UnsupportedMechanismError(
                f"Unsupported mechanism '{raw_mech}'. v0.1.2 strictly supports only 'gaussian' or 'laplace'."
            )

        default_query_b = getattr(self.config.privacy, "default_query_budget", self.config.privacy.global_budget)
        eps = query.epsilon if query.epsilon is not None else default_query_b.epsilon
        if eps <= 0.0:
            raise PrivacyConfigurationError(f"Privacy budget epsilon must be strictly positive, got {eps}")

        if "gaussian" in mech_type:
            delta = query.delta if query.delta is not None else default_query_b.delta
            if delta <= 0.0 or delta >= 1.0:
                raise PrivacyConfigurationError(f"Gaussian mechanism requires 0 < delta < 1, got delta={delta}")
        elif "laplace" in mech_type:
            if query.delta is not None and query.delta != 0.0:
                raise PrivacyConfigurationError(f"Laplace mechanism is pure-DP and requires delta=0, got delta={query.delta}")
            delta = 0.0
        else:
            delta = 0.0

        # 5. Public population size resolution and validation
        resolved_pop_size = public_population_size
        if resolved_pop_size is None:
            pop_cfg = self.config.scenario.population
            if pop_cfg.mode == "fixed_public_population":
                resolved_pop_size = pop_cfg.n_users

        if metric_name == "user_prevalence":
            if resolved_pop_size is None or resolved_pop_size <= 0:
                raise PrivacyConfigurationError(
                    "User Prevalence strictly requires positive public_population_size in fixed_public_population mode."
                )

        # 6. Sensitivity determination: strictly from metric definition (no query overrides!)
        clip_c = float(m_def.contribution.clip_norm or 20.0)
        max_k = int(m_def.contribution.max_cells_per_user or 9)
        norm_b = float(m_def.contribution.norm_bound or 1.0)

        l2_s, l1_s = get_metric_sensitivities(
            metric_type=m_def.type,
            max_cells_per_user=max_k,
            norm_bound=norm_b,
            clip_norm=clip_c,
        )

        sensitivity = l2_s if "gaussian" in mech_type else l1_s

        # 7. Build stable, canonical semantic release key fingerprint
        exec_mode_str = (
            execution_mode.value if isinstance(execution_mode, ExecutionMode) else str(execution_mode)
        ).lower()

        if exec_mode_str == "experiment":
            if noise_seed is not None:
                noise_identity = f"seed_{noise_seed}"
            elif trial_id is not None:
                noise_identity = f"trial_{trial_id}"
            else:
                import uuid
                noise_identity = f"trial_{uuid.uuid4().hex[:12]}"
        else:
            noise_identity = "production_unseeded"

        if privacy_scope_id is None:
            rec_role = rec_policy.role if rec_policy else "external"
            rc_config = getattr(self.config.privacy, "recipient_composition", None)
            rc_mode = rc_config.mode if rc_config else "global_external"
            rc_groups = rc_config.groups if rc_config else None
            resolved_scope_id = resolve_budget_scope_id(
                recipient=query.recipient,
                recipient_role=rec_role,
                mode=rc_mode,
                groups=rc_groups,
            )
        else:
            resolved_scope_id = str(privacy_scope_id)

        canonical_semantics = {
            "calibration_version": "analytic_gaussian_v1" if "gaussian" in mech_type else "laplace_v1",
            "clip_norm": float(clip_c),
            "contribution_mode": str(m_def.contribution.mode),
            "dataset_version": str(dataset_version),
            "delta": float(delta),
            "dimensions": sorted(list(m_def.dimensions)),
            "epsilon": float(eps),
            "execution_mode": exec_mode_str,
            "max_cells_per_user": int(max_k),
            "mechanism": str(mech_type),
            "metric_name": str(metric_name),
            "metric_version": int(getattr(m_def, "version", 1)),
            "noise_seed_identity": noise_identity,
            "norm_bound": float(norm_b),
            "population_mode": str(self.config.scenario.population.mode),
            "privacy_scope_id": str(resolved_scope_id),
            "providers": sorted(list(self.config.scenario.providers)),
            "public_population_size": int(resolved_pop_size) if resolved_pop_size is not None else -1,
            "tasks": sorted(list(self.config.scenario.tasks)),
            "window_end": str(t_end) if t_end else "",
            "window_start": str(t_start) if t_start else "",
        }
        raw_key_bytes = json.dumps(canonical_semantics, sort_keys=True, separators=(",", ":")).encode("utf-8")
        release_key = hashlib.sha256(raw_key_bytes).hexdigest()

        return ExecutionPlan(
            metric=metric_name,
            time_window=(t_start, t_end),
            privacy_unit="user",
            contribution_method=m_def.contribution.mode,
            sensitivity=sensitivity,
            mechanism=mech_type,
            epsilon=eps,
            delta=delta,
            action="execute_dp_release",
            release_key=release_key,
            canonical_semantics=canonical_semantics,
            public_population_size=resolved_pop_size,
            execution_mode=exec_mode_str,
            privacy_scope_id=str(resolved_scope_id),
        )

