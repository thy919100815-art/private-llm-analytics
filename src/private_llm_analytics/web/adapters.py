"""Adapters connecting the local Web Console to authoritative backend services in v0.5."""

from datetime import datetime, timezone
import io
from pathlib import Path
from typing import Any, Dict, List, Optional
import pandas as pd

from private_llm_analytics.config import load_app_config
from private_llm_analytics.config.models import AppConfig
from private_llm_analytics.continual.models import (
    ContinualNoiseConfig,
    ContinualStreamConfig,
    ContinualTreeConfig,
)
from private_llm_analytics.continual.query import ContinualQueryEngine
from private_llm_analytics.planning.planner import SmallNReleasePlanner
from private_llm_analytics.service.analytics import AnalyticsService
from private_llm_analytics.web.schemas import (
    ContinualQueryRequest,
    ReleaseRequest,
)


class WebAnalyticsAdapter:
    """Production adapter mediating Web Console requests to existing AnalyticsService."""

    def __init__(self, service_or_config_dir: Any = "configs"):
        if isinstance(service_or_config_dir, AnalyticsService):
            self.service = service_or_config_dir
        elif isinstance(service_or_config_dir, AppConfig):
            self.service = AnalyticsService(config=service_or_config_dir)
        else:
            self.service = AnalyticsService.from_config_dir(service_or_config_dir)

        self._dataset_source: str = "none"

    def get_config(self) -> Dict[str, Any]:
        """Expose safe, non-sensitive configuration to the browser."""
        cfg = self.service.config
        providers = list(cfg.scenario.providers)
        tasks = list(cfg.scenario.tasks)
        metrics = [
            {
                "id": "user_prevalence",
                "name": "User Prevalence",
                "description": "Proportion of workforce using model capability",
            },
            {
                "id": "user_normalized_share",
                "name": "User-Normalized Share",
                "description": "User workflow distribution across capabilities",
            },
            {
                "id": "interaction_volume",
                "name": "Interaction Volume",
                "description": "Total volume of model interactions",
            },
        ]
        default_query_budget = getattr(cfg.privacy, "default_query_budget", None)
        default_eps = float(default_query_budget.epsilon if default_query_budget else 1.0)
        default_delta = float(default_query_budget.delta if default_query_budget else 1e-6)

        planner_cfg = getattr(cfg.privacy, "planner", None)
        candidate_windows = (
            list(planner_cfg.candidate_windows)
            if planner_cfg and hasattr(planner_cfg, "candidate_windows")
            else [7, 14, 30, 60, 90]
        )

        recipients = list(cfg.recipients.recipients.keys())
        continual_cfg = getattr(cfg.privacy, "continual", None)
        continual_enabled = bool(continual_cfg.enabled if continual_cfg else False)

        return {
            "providers": providers,
            "tasks": tasks,
            "metrics": metrics,
            "default_epsilon": default_eps,
            "default_delta": default_delta,
            "candidate_windows": candidate_windows,
            "recipients": recipients,
            "continual_enabled": continual_enabled,
            "granularities": ["full_9_cell", "provider_3_cell", "task_3_cell", "overall_1_cell"],
        }

    def get_dataset_status(self) -> Dict[str, Any]:
        """Return status of in-memory telemetry dataset."""
        df = self.service._raw_dataframe
        if df is None:
            return {
                "loaded": False,
                "rows": 0,
                "active_users": 0,
                "dataset_version": None,
                "source": "none",
            }
        active_u = int(df["user_id"].nunique()) if "user_id" in df.columns else len(df)
        return {
            "loaded": True,
            "rows": len(df),
            "active_users": active_u,
            "dataset_version": self.service._dataset_version,
            "source": self._dataset_source,
        }

    def load_demo_data(self) -> Dict[str, Any]:
        """Generate demo synthetic telemetry using existing scenario generator."""
        self.service.generate_synthetic_data()
        self._dataset_source = "demo"
        return self.get_dataset_status()

    def load_uploaded_data(self, filename: str, content: bytes) -> Dict[str, Any]:
        """Validate and ingest uploaded CSV or Parquet telemetry file."""
        lower_name = filename.lower()
        if lower_name.endswith(".csv"):
            try:
                df = pd.read_csv(io.BytesIO(content))
            except Exception as e:
                raise ValueError(f"Failed to parse uploaded CSV: {e}")
        elif lower_name.endswith(".parquet"):
            try:
                df = pd.read_parquet(io.BytesIO(content))
            except Exception as e:
                raise ValueError(f"Failed to parse uploaded Parquet: {e}")
        else:
            raise ValueError("Unsupported format. Allowed formats: .csv, .parquet")

        required_cols = {"user_id", "provider", "task"}
        missing = required_cols - set(df.columns)
        if missing:
            raise ValueError(f"Missing required columns in dataset: {sorted(list(missing))}")

        allowed_providers = set(self.service.config.scenario.providers)
        allowed_tasks = set(self.service.config.scenario.tasks)

        dataset_providers = set(df["provider"].dropna().unique())
        unknown_providers = dataset_providers - allowed_providers
        if unknown_providers:
            raise ValueError(
                f"Unknown provider(s) in uploaded dataset: {sorted(list(unknown_providers))}. "
                f"Configured providers: {sorted(list(allowed_providers))}"
            )

        dataset_tasks = set(df["task"].dropna().unique())
        unknown_tasks = dataset_tasks - allowed_tasks
        if unknown_tasks:
            raise ValueError(
                f"Unknown task(s) in uploaded dataset: {sorted(list(unknown_tasks))}. "
                f"Configured tasks: {sorted(list(allowed_tasks))}"
            )

        stem = Path(filename).stem
        self.service.set_data(df, dataset_version=f"upload_{stem}_{len(df)}_rows")
        self._dataset_source = "uploaded"
        return self.get_dataset_status()

    def get_privacy_status(self, recipient: str = "provider_a") -> Dict[str, Any]:
        """Read authoritative privacy-accounting state from state manager."""
        pv = self.service.orchestrator.get_privacy_view(recipient)
        eff_eps = float(pv.effective_epsilon)
        eps_cap = float(pv.epsilon_cap)
        rem_headroom = float(pv.remaining_epsilon)
        spend_pct = (
            min(100.0, max(0.0, (eff_eps / eps_cap * 100.0))) if eps_cap > 0 else 0.0
        )

        return {
            "budget_scope_id": pv.budget_scope_id,
            "accountant_mode": "RDP -> (ε, δ)" if pv.accountant_mode == "rdp" else "Basic Additive",
            "history_mode": getattr(self.service.config.privacy.history, "mode", "lifetime"),
            "recipient_composition_mode": getattr(
                self.service.config.privacy.recipient_composition, "mode", "global_external"
            ),
            "effective_epsilon": round(eff_eps, 4),
            "epsilon_cap": round(eps_cap, 4),
            "remaining_headroom": round(rem_headroom, 4),
            "delta_global": float(pv.delta_global),
            "release_count": pv.release_count,
            "best_alpha": float(pv.best_alpha) if pv.best_alpha is not None else None,
            "spend_percentage": round(spend_pct, 1),
        }

    def get_planner_recommendation(
        self,
        metric: str,
        recipient: str,
        planning_horizon_days: int = 90,
    ) -> Dict[str, Any]:
        """Invoke existing SmallNReleasePlanner to provide utility recommendations."""
        planner = SmallNReleasePlanner(
            app_config=self.service.config,
            state_manager=self.service.orchestrator.state_manager,
        )
        plan = planner.plan(
            metric=metric,
            recipient=recipient,
            planning_horizon_days=planning_horizon_days,
        )

        policy = plan.policy
        util = plan.estimated_utility

        return {
            "recommended_clipping": float(policy.contribution_bound) if policy and policy.contribution_bound is not None else None,
            "recommended_granularity": policy.granularity if policy else "full_9_cell",
            "recommended_window_days": int(policy.window_days) if policy else 30,
            "recommended_frequency_days": int(policy.frequency_days) if policy else 7,
            "estimated_clipping_loss": round(float(util.clipping_error), 4) if util else None,
            "estimated_dp_noise": round(float(util.noise_error), 4) if util else None,
            "estimated_utility": round(float(util.total_error), 4) if util else None,
            "budget_feasible": bool(plan.budget_feasible),
            "projected_epsilon": round(float(plan.projected_epsilon), 4),
            "decision_reason": plan.decision_reason,
            "is_recommendation_only": True,
        }

    def execute_release(self, request: ReleaseRequest) -> Dict[str, Any]:
        """Execute differential privacy release through authoritative AnalyticsService."""
        if self.service._raw_dataframe is None:
            raise ValueError(
                "No telemetry data loaded. Please load demo data or upload a dataset first."
            )

        raw_result = self.service.compute(
            metric=request.metric,
            recipient=request.recipient,
            start=request.window_start,
            end=request.window_end,
            epsilon=request.epsilon,
            delta=request.delta,
            include_exact=False,
        )

        results = raw_result.get("results", {})
        pv = self.service.orchestrator.get_privacy_view(request.recipient)

        return {
            "release_id": str(raw_result.get("release_id", raw_result.get("release_key", ""))),
            "release_key": raw_result.get("release_key", ""),
            "metric": raw_result.get("metric", request.metric),
            "recipient": raw_result.get("recipient", request.recipient),
            "epsilon": float(raw_result.get("epsilon", request.epsilon)),
            "delta": float(raw_result.get("delta", request.delta)),
            "noise_scale": round(float(raw_result.get("noise_scale", 0.0)), 4),
            "mechanism": raw_result.get("mechanism", "gaussian"),
            "cache_hit": bool(raw_result.get("cache_hit", False)),
            "incremental_privacy_cost": float(raw_result.get("incremental_privacy_cost", 0.0)),
            "results": results,
            "privacy_summary": {
                "effective_epsilon_after": round(float(pv.effective_epsilon), 4),
                "epsilon_cap": round(float(pv.epsilon_cap), 4),
                "remaining_headroom": round(float(pv.remaining_epsilon), 4),
                "release_count": int(pv.release_count),
            },
        }

    def list_releases(self) -> List[Dict[str, Any]]:
        """Return sanitized historical releases from persistent ledger."""
        entries = self.service.orchestrator.ledger.get_entries()
        items = []
        for e in reversed(entries):
            items.append({
                "release_id": e.release_id,
                "timestamp": e.timestamp,
                "metric": e.metric,
                "recipient": e.recipient,
                "budget_scope_id": getattr(e, "budget_scope_id", "global_external"),
                "window_start": e.window_start,
                "window_end": e.window_end,
                "mechanism": e.mechanism,
                "epsilon": float(e.epsilon),
                "delta": float(e.delta),
                "cache_hit": bool(e.cache_hit),
                "status": "COMMITTED",
            })
        return items

    def get_continual_status(self) -> Dict[str, Any]:
        """Return continual observation configuration status."""
        continual_cfg = getattr(self.service.config.privacy, "continual", None)
        enabled = bool(continual_cfg.enabled if continual_cfg else False)
        epoch_days = int(continual_cfg.epoch_days if continual_cfg else 90)
        mode = str(continual_cfg.mode if continual_cfg else "binary_tree")

        msg = (
            "Continual observation is enabled."
            if enabled
            else "Continual observation is disabled in the current configuration. "
            "Enable continual observation explicitly in trusted local configuration."
        )

        return {
            "enabled": enabled,
            "mode": mode,
            "epoch_days": epoch_days,
            "supported_metric": "interaction_volume",
            "message": msg,
        }

    def query_continual_interval(self, request: ContinualQueryRequest) -> Dict[str, Any]:
        """Execute dyadic continual interval query when enabled."""
        status = self.get_continual_status()
        if not status["enabled"]:
            raise ValueError(
                "Continual observation is disabled in the current configuration. "
                "Enable continual observation explicitly in trusted local configuration."
            )
        if request.metric != "interaction_volume":
            raise ValueError(
                "Continual observation is currently supported for Interaction Volume only."
            )
        if self.service._raw_dataframe is None:
            raise ValueError(
                "No telemetry data loaded. Please load demo data or upload a dataset first."
            )

        continual_cfg = self.service.config.privacy.continual
        engine = ContinualQueryEngine.from_app_config(self.service.config)

        tree_cfg = ContinualTreeConfig(
            topology=getattr(continual_cfg.tree, "topology", "binary"),
            padding=getattr(continual_cfg.tree, "padding", "next_power_of_two"),
        )
        noise_cfg = ContinualNoiseConfig(
            mechanism=getattr(continual_cfg.noise, "mechanism", "gaussian"),
            target_epoch_epsilon=getattr(continual_cfg.noise, "target_epoch_epsilon", 2.0),
            calibration_delta=getattr(continual_cfg.noise, "calibration_delta", 1e-6),
            equal_sigma_per_level=getattr(continual_cfg.noise, "equal_sigma_per_level", True),
        )

        epoch_days = getattr(continual_cfg, "epoch_days", 90)
        daily_l2 = getattr(continual_cfg, "daily_l2_bound", 10.0)
        epoch_group_l12 = getattr(
            continual_cfg,
            "epoch_group_l12_bound",
            getattr(continual_cfg, "epoch_l12_bound", 40.0),
        )

        stream_cfg = ContinualStreamConfig(
            stream_id="web_stream_1",
            metric=request.metric,
            epoch_id="web_epoch_1",
            epoch_start=datetime.now(timezone.utc).strftime("%Y-%m-%d"),
            epoch_days=epoch_days,
            daily_l2_bound=daily_l2,
            epoch_group_l12_bound=epoch_group_l12,
            tree=tree_cfg,
            noise=noise_cfg,
        )

        res = engine.query_interval(
            stream_config=stream_cfg,
            start_day=request.start_day,
            end_day=request.end_day,
            recipient=request.recipient,
            interactions=self.service._raw_dataframe,
        )

        return res.to_dict()
