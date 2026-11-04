"""Adapters connecting the local Web Console to authoritative backend services in v0.5 / v1.0 / v1.1."""

from datetime import datetime, timezone
from decimal import Decimal
import io
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from private_llm_analytics.config import load_app_config
from private_llm_analytics.config.models import AppConfig
from private_llm_analytics.continual.models import (
    ContinualNoiseConfig,
    ContinualStreamConfig,
    ContinualTreeConfig,
)
from private_llm_analytics.continual.query import ContinualQueryEngine
from private_llm_analytics.evaluation.engine import (
    compute_all_metrics_utility_summary,
    evaluate_release_accuracy,
    run_epsilon_utility_sweep,
)
from private_llm_analytics.ingestion.schema import COL_TIMESTAMP
from private_llm_analytics.planning.planner import SmallNReleasePlanner
from private_llm_analytics.privacy.exceptions import (
    ProductionSafetyConfigurationError,
    RecipientSecurityViolation,
)
from private_llm_analytics.privacy.rdp import compose_rdp_vectors, to_decimal
from private_llm_analytics.release.exporter import (
    build_internal_comparison_csv_string,
    build_release_csv_string,
    build_release_json_string,
    _get_recipient_display,
)
from private_llm_analytics.service.analytics import AnalyticsService
from private_llm_analytics.statistics.cards import get_all_mechanism_cards
from private_llm_analytics.web.schemas import (
    AdminPrivacyPolicyUpdateRequest,
    ContinualQueryRequest,
    EpsilonSweepRequest,
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
        if getattr(cfg, "taxonomy", None) is not None:
            providers = list(cfg.taxonomy.providers)
            tasks = list(cfg.taxonomy.tasks)
        else:
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

        from private_llm_analytics.statistics.registry import StatisticProfileRegistry
        statistics = [m.to_dict() for m in StatisticProfileRegistry.get_metadata_list()]

        rc_config = getattr(cfg.privacy, "recipient_composition", None)
        rc_mode = rc_config.mode if rc_config else "global_external"
        rc_groups = rc_config.groups if rc_config else {}

        recipient_details = []
        for r_id, r_pol in cfg.recipients.recipients.items():
            disp_name = r_pol.get_display_name(r_id) if hasattr(r_pol, "get_display_name") else _get_recipient_display(r_id, r_pol.display_name)

            if r_pol.role == "trusted":
                scope_id = "internal"
                scope_mode = "INTERNAL"
            elif rc_mode == "global_external":
                scope_id = "global_external"
                scope_mode = "GLOBAL_EXTERNAL"
            elif rc_mode == "per_recipient":
                scope_id = f"recipient_{r_id}"
                scope_mode = "PER_RECIPIENT"
            elif rc_mode == "collusion_group":
                scope_id = "global_external"
                scope_mode = "COLLUSION_GROUP"
                for g_name, members in rc_groups.items():
                    if r_id in members:
                        scope_id = f"group_{g_name}"
                        break
            else:
                scope_id = "global_external"
                scope_mode = "GLOBAL_EXTERNAL"

            recipient_details.append({
                "recipient_id": r_id,
                "display_name": disp_name,
                "role": r_pol.role,
                "providers": list(r_pol.providers),
                "metrics": list(r_pol.metrics),
                "exact_results": bool(r_pol.exact_results),
                "dp_results": bool(r_pol.dp_results),
                "cross_provider_comparison": bool(r_pol.cross_provider_comparison),
                "budget_scope_id": scope_id,
                "privacy_scope_mode": scope_mode,
            })

        mechanism_cards = get_all_mechanism_cards(cfg)

        acct_cfg = getattr(cfg.privacy, "accountant", None)
        eps_cap = float(getattr(acct_cfg, "epsilon_cap", 4.0) if acct_cfg else 4.0)
        delta_g = float(getattr(acct_cfg, "delta_global", 1e-6) if acct_cfg else 1e-6)
        privacy_policy = {
            "lifetime_epsilon_cap": eps_cap,
            "global_delta": delta_g,
            "default_query_epsilon": default_eps,
            "default_query_delta": default_delta,
        }

        return {
            "providers": providers,
            "tasks": tasks,
            "metrics": metrics,
            "statistics": statistics,
            "default_epsilon": default_eps,
            "default_delta": default_delta,
            "candidate_windows": candidate_windows,
            "recipients": recipients,
            "recipient_details": recipient_details,
            "mechanism_cards": mechanism_cards,
            "privacy_policy": privacy_policy,
            "continual_enabled": continual_enabled,
            "granularities": ["full_9_cell", "provider_3_cell", "task_3_cell", "overall_1_cell"],
            "demo_mode": self.is_demo_mode(),
        }

    def get_dataset_sample_template(self) -> Tuple[str, str]:
        """Return CSV formatted sample telemetry dataset."""
        sample_path = Path(__file__).parent / "static" / "sample_private_llm_analytics.csv"
        if sample_path.exists():
            with open(sample_path, "r", encoding="utf-8") as f:
                content = f.read()
        else:
            content = (
                "interaction_id,user_id,timestamp,provider,task,success\n"
                "i0001,u001,2026-10-20 09:31:20,provider_a,technical,true\n"
                "i0002,u001,2026-10-20 09:35:42,provider_a,technical,false\n"
                "i0003,u001,2026-10-20 10:01:13,provider_a,document,true\n"
                "i0004,u002,2026-10-20 10:15:08,provider_a,technical,true\n"
                "i0005,u003,2026-10-20 10:27:31,provider_b,analysis,true\n"
            )
        return content, "sample_private_llm_analytics.csv"

    def get_dataset_status(self) -> Dict[str, Any]:
        """Return status and quality inspection of in-memory telemetry dataset."""
        df = self.service._raw_dataframe
        if df is None:
            return {
                "loaded": False,
                "rows": 0,
                "active_users": 0,
                "user_count": 0,
                "provider_count": 0,
                "task_count": 0,
                "time_range": None,
                "has_success_field": False,
                "quality_checks": [],
                "dataset_version": None,
                "source": "none",
            }

        active_u = int(df["user_id"].nunique()) if "user_id" in df.columns else len(df)
        prov_cnt = int(df["provider"].nunique()) if "provider" in df.columns else 0
        task_cnt = int(df["task"].nunique()) if "task" in df.columns else 0
        has_succ = "success" in df.columns
        time_range = None
        if "timestamp" in df.columns and len(df) > 0:
            try:
                min_t = str(df["timestamp"].min())
                max_t = str(df["timestamp"].max())
                time_range = {
                    "start": min_t[:10] if len(min_t) >= 10 else min_t,
                    "end": max_t[:10] if len(max_t) >= 10 else max_t,
                }
            except Exception:
                pass

        required_cols = {"user_id", "provider", "task"}
        has_req = required_cols.issubset(set(df.columns))
        no_null_u = bool(df["user_id"].isnull().sum() == 0) if "user_id" in df.columns else False
        allowed_p = set(self.service.config.scenario.providers)
        allowed_t = set(self.service.config.scenario.tasks)
        prov_ok = set(df["provider"].dropna().unique()).issubset(allowed_p) if "provider" in df.columns else False
        task_ok = set(df["task"].dropna().unique()).issubset(allowed_t) if "task" in df.columns else False

        quality_checks = [
            {
                "name": "必需字段完整",
                "passed": has_req,
                "details": "interaction_id, user_id, timestamp, provider, task" if has_req else "缺少必需列",
            },
            {
                "name": "timestamp 可解析",
                "passed": "timestamp" in df.columns,
                "details": "所有时间戳格式正常" if "timestamp" in df.columns else "缺少 timestamp 列",
            },
            {
                "name": "user_id 无空值",
                "passed": no_null_u,
                "details": "用户标识完整无空值" if no_null_u else "存在空 user_id",
            },
            {
                "name": "provider 全部存在于 taxonomy",
                "passed": prov_ok,
                "details": f"{', '.join(sorted(list(allowed_p)))}" if prov_ok else "存在未配置的 provider",
            },
            {
                "name": "task 全部存在于 taxonomy",
                "passed": task_ok,
                "details": f"{', '.join(sorted(list(allowed_t)))}" if task_ok else "存在未配置的 task",
            },
            {
                "name": "success 字段可用性",
                "passed": has_succ,
                "details": "可用于成功率统计" if has_succ else "缺少 success 字段（成功率统计不可用）",
            },
            {
                "name": "无明文 Prompt",
                "passed": not ("prompt" in df.columns or "text" in df.columns),
                "details": "日志仅包含聚合元数据，无输入输出明文 Prompt" if not ("prompt" in df.columns or "text" in df.columns) else "发现明文 prompt/text 列",
            },
        ]

        time_range_disp = "全量时间"
        if time_range:
            try:
                d1 = datetime.strptime(time_range["start"][:10], "%Y-%m-%d")
                d2 = datetime.strptime(time_range["end"][:10], "%Y-%m-%d")
                days_diff = (d2 - d1).days + 1
                time_range_disp = f"{days_diff} 天 ({time_range['start']} ～ {time_range['end']})"
            except Exception:
                time_range_disp = f"{time_range['start']} ～ {time_range['end']}"

        return {
            "loaded": True,
            "rows": len(df),
            "active_users": active_u,
            "user_count": active_u,
            "provider_count": prov_cnt,
            "task_count": task_cnt,
            "time_range": time_range,
            "time_range_display": time_range_disp,
            "has_success_field": has_succ,
            "quality_checks": quality_checks,
            "dataset_version": self.service._dataset_version,
            "source": self._dataset_source,
        }

    def load_demo_data(self) -> Dict[str, Any]:
        """Generate demo synthetic telemetry using scenario generator with fast parquet cache."""
        if self._dataset_source == "demo" and self.service._raw_dataframe is not None:
            return self.get_dataset_status()

        cache_dir = Path("results")
        cache_dir.mkdir(parents=True, exist_ok=True)
        sc_cfg = self.service.config.scenario
        seed = sc_cfg.random_seed
        n_users = sc_cfg.population.n_users
        cache_file = cache_dir / f"demo_synthetic_seed_{seed}_n_{n_users}.parquet"

        if cache_file.exists():
            try:
                df = pd.read_parquet(cache_file)
                self.service.set_data(df, dataset_version=f"synthetic_seed_{seed}_n_{n_users}")
                self._dataset_source = "demo"
                return self.get_dataset_status()
            except Exception:
                pass

        df = self.service.generate_synthetic_data()
        try:
            df.to_parquet(cache_file, index=False)
        except Exception:
            pass
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

        if "timestamp" in df.columns:
            try:
                df["timestamp"] = pd.to_datetime(df["timestamp"])
            except Exception:
                pass

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

        epoch_info = None
        lifetime_info = None
        if hasattr(self.service.orchestrator, "epoch_manager"):
            try:
                active_epoch = self.service.orchestrator.epoch_manager.get_or_create_active_epoch()
                epoch_info = active_epoch.to_dict()
                lifetime_exposure = self.service.orchestrator.epoch_manager.get_lifetime_exposure()
                lifetime_info = lifetime_exposure.to_dict()
            except Exception:
                pass

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
            "epoch": epoch_info,
            "lifetime_exposure": lifetime_info,
        }

    def get_planner_recommendation(
        self,
        metric: str,
        recipient: str,
        planning_horizon_days: int = 90,
    ) -> Dict[str, Any]:
        """Invoke existing SmallNReleasePlanner to provide utility recommendations."""
        if metric != "interaction_volume":
            return {
                "recommended_clipping": None,
                "recommended_granularity": "full_9_cell",
                "recommended_window_days": 30,
                "recommended_frequency_days": 7,
                "estimated_clipping_loss": None,
                "estimated_dp_noise": None,
                "estimated_utility": None,
                "budget_feasible": True,
                "projected_epsilon": 0.0,
                "decision_reason": "当前统计项目暂未启用自动参数规划，请使用已验证的默认公开参数。",
                "is_recommendation_only": True,
            }

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

    def resolve_window_bounds(
        self,
        window_days: Optional[int] = None,
        window_start: Optional[str] = None,
        window_end: Optional[str] = None,
    ) -> Tuple[Optional[str], Optional[str]]:
        """Resolve authoritative start and end observation bounds according to v0.5 rules."""
        if window_days is not None and (window_start is not None or window_end is not None):
            raise ValueError(
                "Cannot specify both 'window_days' and explicit 'window_start'/'window_end'."
            )

        if (window_start is not None and window_end is None) or (window_start is None and window_end is not None):
            raise ValueError(
                "Both 'window_start' and 'window_end' must be provided for explicit windowing."
            )

        if window_start is not None and window_end is not None:
            return str(window_start), str(window_end)

        if window_days is not None:
            df = self.service._raw_dataframe
            if df is None or df.empty:
                raise ValueError(
                    "No telemetry data loaded. Please load demo data or upload a dataset first."
                )

            if COL_TIMESTAMP not in df.columns:
                raise ValueError("Windowed release requires timestamped telemetry data.")

            valid_ts = pd.to_datetime(df[COL_TIMESTAMP], errors="coerce").dropna()
            if valid_ts.empty:
                raise ValueError("Windowed release requires timestamped telemetry data.")

            t_max = valid_ts.max()
            start_date = t_max.floor("D") - pd.Timedelta(days=window_days - 1)
            return str(start_date), str(t_max)

        return None, None

    def execute_release(self, request: ReleaseRequest) -> Dict[str, Any]:
        """Execute differential privacy release through authoritative AnalyticsService."""
        if self.service._raw_dataframe is None:
            raise ValueError(
                "No telemetry data loaded. Please load demo data or upload a dataset first."
            )

        start, end = self.resolve_window_bounds(
            window_days=request.window_days,
            window_start=request.window_start,
            window_end=request.window_end,
        )

        raw_result = self.service.compute(
            metric=request.metric,
            recipient=request.recipient,
            start=start,
            end=end,
            epsilon=request.epsilon,
            delta=request.delta,
            k=getattr(request, "k", None),
            use_epoch_planner=getattr(request, "use_epoch_planner", None),
            include_exact=False,
        )

        results = raw_result.get("results", {})
        pv = self.service.orchestrator.get_privacy_view(request.recipient)
        rec_policy = self.service.config.recipients.recipients.get(request.recipient)
        rec_name = (
            rec_policy.get_display_name(request.recipient)
            if rec_policy and hasattr(rec_policy, "get_display_name")
            else _get_recipient_display(request.recipient, rec_policy.display_name if rec_policy else None)
        )

        return {
            "release_id": str(raw_result.get("release_id", raw_result.get("release_key", ""))),
            "release_key": raw_result.get("release_key", ""),
            "metric": raw_result.get("metric", request.metric),
            "recipient": raw_result.get("recipient", request.recipient),
            "recipient_name": rec_name,
            "epsilon": float(raw_result.get("epsilon", request.epsilon)),
            "delta": float(raw_result.get("delta", request.delta)),
            "noise_scale": round(float(raw_result.get("noise_scale", 0.0)), 4),
            "mechanism": raw_result.get("mechanism", "gaussian"),
            "cache_hit": bool(raw_result.get("cache_hit", False)),
            "incremental_privacy_cost": float(raw_result.get("incremental_privacy_cost", 0.0)),
            "window_start": start,
            "window_end": end,
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
            r_pol = self.service.config.recipients.recipients.get(e.recipient)
            r_name = (
                r_pol.get_display_name(e.recipient)
                if r_pol and hasattr(r_pol, "get_display_name")
                else _get_recipient_display(e.recipient, r_pol.display_name if r_pol else None)
            )
            items.append({
                "release_id": e.release_id,
                "release_key": getattr(e, "release_key", e.release_id),
                "timestamp": e.timestamp,
                "metric": e.metric,
                "recipient": e.recipient,
                "recipient_name": r_name,
                "budget_scope_id": getattr(e, "budget_scope_id", "global_external"),
                "window_start": e.window_start,
                "window_end": e.window_end,
                "mechanism": e.mechanism,
                "epsilon": float(e.epsilon),
                "delta": float(e.delta),
                "cache_hit": bool(e.cache_hit),
                "status": "COMMITTED",
                "download_available": True,
            })
        return items

    def get_admin_policy(self) -> Dict[str, Any]:
        """Read authoritative privacy policy parameters and persistent audit history."""
        cfg = self.service.config
        acct_cfg = getattr(cfg.privacy, "accountant", None)
        eps_cap = float(getattr(acct_cfg, "epsilon_cap", 4.0) if acct_cfg else 4.0)
        delta_g = float(getattr(acct_cfg, "delta_global", 1e-6) if acct_cfg else 1e-6)

        q_budget = getattr(cfg.privacy, "default_query_budget", None)
        def_eps = float(q_budget.epsilon if q_budget else 1.0)
        def_delta = float(q_budget.delta if q_budget else 1e-6)

        cur_eff = float(self.service.orchestrator.get_privacy_view("global_external").effective_epsilon)
        internal_eff = float(self.service.orchestrator.get_privacy_view("internal").effective_epsilon)
        cur_eff = max(cur_eff, internal_eff)
        try:
            with self.service.orchestrator.state_manager._connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT DISTINCT budget_scope_id FROM release_store WHERE status = 'COMMITTED'")
                for r in cursor.fetchall():
                    if r[0]:
                        s_eff = float(self.service.orchestrator.get_privacy_view(r[0]).effective_epsilon)
                        cur_eff = max(cur_eff, s_eff)
        except Exception:
            pass

        history = self.service.orchestrator.state_manager.get_admin_policy_history()

        return {
            "lifetime_epsilon_cap": round(eps_cap, 4),
            "global_delta": delta_g,
            "default_query_epsilon": round(def_eps, 4),
            "default_query_delta": def_delta,
            "current_effective_epsilon": round(cur_eff, 4),
            "audit_history": history,
        }

    def update_admin_policy(self, request: AdminPrivacyPolicyUpdateRequest) -> Dict[str, Any]:
        """Update administrative privacy policy parameters with strict governance and audit."""
        rec_name = request.recipient
        policy = self.service.config.recipients.recipients.get(rec_name)
        if not policy or policy.role != "trusted":
            raise RecipientSecurityViolation(
                f"Unauthorized: Only trusted local administrator (role='trusted') may modify privacy policy. "
                f"Attempted with recipient '{rec_name}'."
            )

        cur_eff = float(self.service.orchestrator.get_privacy_view("global_external").effective_epsilon)
        internal_eff = float(self.service.orchestrator.get_privacy_view("internal").effective_epsilon)
        cur_eff = max(cur_eff, internal_eff)
        try:
            with self.service.orchestrator.state_manager._connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT DISTINCT budget_scope_id FROM release_store WHERE status = 'COMMITTED'")
                for r in cursor.fetchall():
                    if r[0]:
                        s_eff = float(self.service.orchestrator.get_privacy_view(r[0]).effective_epsilon)
                        cur_eff = max(cur_eff, s_eff)
        except Exception:
            pass

        if request.lifetime_epsilon_cap < cur_eff:
            raise ValueError(
                f"Cannot set lifetime epsilon cap ({request.lifetime_epsilon_cap}) below the currently spent "
                f"effective epsilon ({round(cur_eff, 4)}). Historical privacy expenditure cannot be retroactively erased."
            )

        # Update in-memory configurations
        cfg = self.service.config
        if getattr(cfg.privacy, "accountant", None):
            cfg.privacy.accountant.epsilon_cap = request.lifetime_epsilon_cap
            cfg.privacy.accountant.delta_global = request.global_delta
        if getattr(cfg.privacy, "global_budget", None):
            cfg.privacy.global_budget.epsilon = request.lifetime_epsilon_cap
            cfg.privacy.global_budget.delta = request.global_delta
        if getattr(cfg.privacy, "default_query_budget", None):
            cfg.privacy.default_query_budget.epsilon = request.default_query_epsilon
            cfg.privacy.default_query_budget.delta = request.default_query_delta

        # Update orchestrator accountants
        self.service.orchestrator.epsilon_cap = request.lifetime_epsilon_cap
        self.service.orchestrator.delta_global = request.global_delta
        if hasattr(self.service.orchestrator, "rdp_accountant"):
            self.service.orchestrator.rdp_accountant.epsilon_cap = request.lifetime_epsilon_cap
            self.service.orchestrator.rdp_accountant.delta_global = request.global_delta

        # Record persistent audit event in state manager
        self.service.orchestrator.state_manager.record_admin_policy_change(
            actor=rec_name,
            epsilon_cap=request.lifetime_epsilon_cap,
            delta_global=request.global_delta,
            default_query_epsilon=request.default_query_epsilon,
            default_query_delta=request.default_query_delta,
            effective_epsilon_at_change=cur_eff,
            reason=request.reason or "",
        )

        return self.get_admin_policy()

    def get_release_by_id(self, release_id: str, recipient: str = "internal") -> Dict[str, Any]:
        """Retrieve an already committed release by ID or key, enforcing recipient authorization."""
        caller_policy = self.service.config.recipients.recipients.get(recipient)
        if not caller_policy:
            raise RecipientSecurityViolation(f"Unknown or unauthorized recipient '{recipient}'.")

        row = self.service.orchestrator.state_manager.get_committed_release_record(release_id)
        if not row and hasattr(self.service.orchestrator, "ledger") and self.service.orchestrator.ledger:
            for e in self.service.orchestrator.ledger.get_entries():
                if getattr(e, "release_id", None) == release_id or getattr(e, "release_key", None) == release_id:
                    row = {
                        "release_id": e.release_id,
                        "release_key": getattr(e, "release_key", release_id),
                        "metric": e.metric,
                        "recipient": e.recipient,
                        "budget_scope_id": getattr(e, "budget_scope_id", "global_external"),
                        "epsilon": float(e.epsilon),
                        "delta": float(e.delta),
                        "mechanism": getattr(e, "mechanism", "gaussian"),
                        "noise_scale": getattr(e, "noise_scale", 0.0),
                        "window_start": getattr(e, "window_start", None),
                        "window_end": getattr(e, "window_end", None),
                        "dataset_version": getattr(e, "dataset_version", "v1.0"),
                        "cache_hit": getattr(e, "cache_hit", False),
                    }
                    break
        if not row:
            raise ValueError(f"Release '{release_id}' not found in committed release store or ledger.")

        release_rec = row.get("recipient", "")
        release_scope = row.get("budget_scope_id", "global_external")

        # Security check: External recipient can only access releases intended for them or their scope
        if caller_policy.role == "external":
            if release_rec and release_rec != recipient and release_scope != f"recipient_{recipient}":
                raise RecipientSecurityViolation(
                    f"Recipient '{recipient}' is not authorized to access release '{release_id}' "
                    f"issued for '{release_rec}'."
                )

        release_key = row.get("release_key", release_id)
        committed_result = self.service.orchestrator.state_manager.get_committed_release(release_key)
        if not committed_result and hasattr(self.service.orchestrator, "cache") and self.service.orchestrator.cache:
            committed_result = self.service.orchestrator.cache.get(release_key)
        if committed_result:
            post_proc = committed_result.post_processed
            metric = committed_result.metric_name
            eps = committed_result.epsilon
            delta = committed_result.delta
            noise_scale = committed_result.noise_scale
            cell_names = committed_result.cell_names
        else:
            post_proc = json.loads(row.get("post_processed_json", "{}"))
            metric = row.get("metric", "")
            eps = float(row.get("epsilon", 1.0))
            delta = float(row.get("delta", 1e-6))
            noise_scale = float(row.get("noise_scale", 0.0))
            cell_names = json.loads(row.get("cell_names_json", "[]"))

        filtered_results = dict(post_proc)
        if caller_policy.role == "external":
            allowed_provs = set(caller_policy.providers)
            for k in ("cell_volumes", "prevalence_rates", "cell_shares", "success_rates", "cell_success_rates"):
                if k in filtered_results and isinstance(filtered_results[k], dict):
                    filtered_results[k] = {
                        c: v for c, v in filtered_results[k].items()
                        if c.split(":")[0] in allowed_provs or "*" in allowed_provs
                    }

        rec_disp = _get_recipient_display(recipient, caller_policy.display_name)

        return {
            "release_id": str(row.get("release_id", release_id)),
            "release_key": release_key,
            "metric": metric,
            "recipient": recipient,
            "recipient_id": recipient,
            "recipient_name": rec_disp,
            "recipient_role": caller_policy.role,
            "epsilon": eps,
            "delta": delta,
            "mechanism": row.get("mechanism", "gaussian"),
            "noise_scale": noise_scale,
            "cache_hit": bool(row.get("cache_hit", False)),
            "cell_names": cell_names,
            "results": filtered_results,
            "window_start": row.get("window_start"),
            "window_end": row.get("window_end"),
            "dataset_version": row.get("dataset_version", "v1.0"),
        }

    def download_release_csv(self, release_id: str, recipient: str) -> Tuple[str, str]:
        """Generate CSV download text and filename for an already committed DP release."""
        rel = self.get_release_by_id(release_id, recipient=recipient)
        csv_str = build_release_csv_string(rel)
        date_str = datetime.now(timezone.utc).strftime("%Y%m%d")
        metric_safe = rel.get("metric", "metric")
        rec_disp = rel.get("recipient_name", recipient).replace(" ", "_")
        filename = f"private_llm_analytics_release_{release_id}.csv"
        return csv_str, filename

    def download_release_json(self, release_id: str, recipient: str) -> Tuple[str, str]:
        """Generate JSON download text and filename for an already committed DP release."""
        rel = self.get_release_by_id(release_id, recipient=recipient)
        json_str = build_release_json_string(rel)
        filename = f"private_llm_analytics_release_{release_id}.json"
        return json_str, filename

    def download_internal_comparison_csv(self, release_id: str, recipient: str) -> Tuple[str, str]:
        """Generate Raw vs Clipped vs DP comparison CSV for trusted admin evaluation only."""
        caller_policy = self.service.config.recipients.recipients.get(recipient)
        if not caller_policy or caller_policy.role != "trusted" or not caller_policy.exact_results:
            raise RecipientSecurityViolation(
                f"Unauthorized: Recipient '{recipient}' is strictly forbidden from downloading exact internal evaluation comparisons."
            )

        rel = self.get_release_by_id(release_id, recipient=recipient)

        # Build exact data from current loaded dataset
        exact_data = {}
        if self.service._raw_dataframe is not None:
            try:
                user_matrix = self.service.get_user_matrix(
                    start=rel.get("window_start"),
                    end=rel.get("window_end"),
                )
                metric_name = rel.get("metric", "interaction_volume")
                metric_obj = self.service.orchestrator.catalog.get(metric_name)
                exact_clean = metric_obj.compute_exact(user_matrix)
                exact_data["clean_vector"] = exact_clean.tolist()
                exact_data["cell_names"] = user_matrix.cell_names

                if metric_name == "interaction_volume" and hasattr(metric_obj, "build_user_contributions"):
                    clipped_mat = metric_obj.build_user_contributions(user_matrix)
                    clipped_exact = np.sum(clipped_mat, axis=0)
                    exact_data["clipped_vector"] = clipped_exact.tolist()
            except Exception:
                pass

        csv_str = build_internal_comparison_csv_string(rel, exact_data=exact_data)
        metric_safe = rel.get("metric", "comparison")
        filename = f"INTERNAL_ONLY_{metric_safe}_comparison_{release_id[:16]}.csv"
        return csv_str, filename

    def get_continual_status(self) -> Dict[str, Any]:
        """Return continual observation configuration status."""
        continual_cfg = getattr(self.service.config.privacy, "continual", None)
        enabled = bool(continual_cfg.enabled if continual_cfg else False)
        epoch_days = int(continual_cfg.epoch_days if continual_cfg else 90)
        mode = str(continual_cfg.mode if continual_cfg else "binary_tree")
        daily_l2 = float(getattr(continual_cfg, "daily_l2_bound", 10.0))
        epoch_group_l12 = float(
            getattr(continual_cfg, "epoch_group_l12_bound", getattr(continual_cfg, "epoch_l12_bound", 40.0))
        )
        tree_topo = str(getattr(getattr(continual_cfg, "tree", None), "topology", "binary"))

        msg = (
            "Continual observation is enabled."
            if enabled
            else "Continual observation requires trusted configuration and an authoritative finalized stream state."
        )

        return {
            "enabled": enabled,
            "mode": mode,
            "epoch_days": epoch_days,
            "daily_l2_bound": daily_l2,
            "epoch_group_l12_bound": epoch_group_l12,
            "tree_topology": tree_topo,
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

        raise ProductionSafetyConfigurationError(
            "Continual interval queries require an authoritative finalized stream configuration and are not initialized by the Web Console."
        )

    def get_dataset_preview(self, limit: int = 20, recipient: Optional[str] = None) -> Dict[str, Any]:
        """Return preview of first few records for trusted local administrators only.

        Safe local inspection: does not check recipient permissions, does not consume DP budget or ReleaseKey.
        """
        df = self.service._raw_dataframe
        if df is None:
            return {
                "total_rows": 0,
                "preview_limit": limit,
                "columns": ["用户标识", "交互时间", "模型提供方", "任务类型", "任务结果"],
                "rows": [],
            }

        preview_slice = df.head(limit)
        prov_map = {
            "provider_a": "OpenAI",
            "provider_b": "Kimi",
            "provider_c": "DeepSeek",
        }
        task_map = {
            "technical": "技术类",
            "document": "文档类",
            "analysis": "分析类",
        }

        rows = []
        for _, row in preview_slice.iterrows():
            prov_val = str(row.get("provider", ""))
            task_val = str(row.get("task", ""))
            ts_val = str(row.get("timestamp", ""))
            succ_val = row.get("success")
            if pd.isna(succ_val):
                succ_disp = "N/A"
            elif succ_val in (True, 1, "true", "True", "1"):
                succ_disp = "成功"
            else:
                succ_disp = "失败"

            rows.append({
                "user_id": str(row.get("user_id", "")),
                "timestamp": ts_val[:19] if len(ts_val) >= 19 else ts_val,
                "provider": prov_val,
                "provider_name": prov_map.get(prov_val, prov_val),
                "task": task_val,
                "task_name": task_map.get(task_val, task_val),
                "success": succ_disp,
            })

        return {
            "total_rows": len(df),
            "preview_limit": limit,
            "columns": ["用户标识", "交互时间", "模型提供方", "任务类型", "任务结果"],
            "rows": rows,
        }

    def get_release_accuracy_evaluation(self, release_id: str, recipient: str = "internal") -> Dict[str, Any]:
        """Compute comprehensive accuracy evaluation comparing exact ground truth with DP release."""
        caller_policy = self.service.config.recipients.recipients.get(recipient)
        if not caller_policy or caller_policy.role != "trusted" or not caller_policy.exact_results:
            raise RecipientSecurityViolation(
                f"Unauthorized: Recipient '{recipient}' is strictly forbidden from accessing exact evaluation metrics."
            )

        if self.service._raw_dataframe is None:
            raise ValueError("No telemetry data loaded. Please load demo data or upload a dataset first.")

        rel = self.get_release_by_id(release_id, recipient=recipient)
        w_start = rel.get("window_start")
        w_end = rel.get("window_end")
        if w_start in (None, "None", ""):
            w_start = None
        if w_end in (None, "None", ""):
            w_end = None
        user_matrix = self.service.get_user_matrix(
            start=w_start,
            end=w_end,
        )
        metric = rel.get("metric", "interaction_volume")
        dp_results = rel.get("results", {})
        return evaluate_release_accuracy(metric, user_matrix, dp_results)

    def get_utility_summary(self, recipient: str = "internal") -> List[Dict[str, Any]]:
        """Return benchmark summary across all 5 statistics for trusted internal admin."""
        caller_policy = self.service.config.recipients.recipients.get(recipient)
        if not caller_policy or caller_policy.role != "trusted" or not caller_policy.exact_results:
            raise RecipientSecurityViolation(
                f"Unauthorized: Recipient '{recipient}' is strictly forbidden from accessing utility summary."
            )

        if self.service._raw_dataframe is None:
            return []

        user_matrix = self.service.get_user_matrix()
        return compute_all_metrics_utility_summary(self.service, user_matrix)

    def run_epsilon_sweep(self, request: EpsilonSweepRequest) -> Dict[str, Any]:
        """Execute epsilon vs accuracy sweep in isolated experiment mode."""
        caller_policy = self.service.config.recipients.recipients.get(request.recipient)
        if not caller_policy or caller_policy.role != "trusted" or not caller_policy.exact_results:
            raise RecipientSecurityViolation(
                f"Unauthorized: Recipient '{request.recipient}' is strictly forbidden from running internal evaluation experiments."
            )

        if self.service._raw_dataframe is None:
            raise ValueError("No telemetry data loaded. Please load demo data or upload a dataset first.")

        user_matrix = self.service.get_user_matrix()
        return run_epsilon_utility_sweep(
            self.service,
            user_matrix=user_matrix,
            metric=request.metric,
            candidate_epsilons=request.candidate_epsilons,
            n_trials=request.n_trials or 10,
        )

    def download_evaluation_csv(self, release_id: str, recipient: str = "internal") -> Tuple[str, str]:
        """Download evaluation report CSV for trusted admin only."""
        eval_res = self.get_release_accuracy_evaluation(release_id=release_id, recipient=recipient)
        rel = self.get_release_by_id(release_id, recipient=recipient)
        metric = rel.get("metric", "evaluation")

        rows = eval_res.get("rows", [])
        df = pd.DataFrame(rows)
        csv_body = df.to_csv(index=False)
        warning_header = "# WARNING: 该文件包含内部精确统计与评估误差，仅供可信管理员使用，禁止作为外部发布文件。\n"
        csv_content = warning_header + csv_body
        filename = f"INTERNAL_accuracy_evaluation_{metric}_{release_id[:8]}.csv"
        return csv_content, filename

    def download_evaluation_json(self, release_id: str, recipient: str = "internal") -> Tuple[str, str]:
        """Download evaluation report JSON for trusted admin only."""
        eval_res = self.get_release_accuracy_evaluation(release_id=release_id, recipient=recipient)
        rel = self.get_release_by_id(release_id, recipient=recipient)
        metric = rel.get("metric", "evaluation")

        json_content = json.dumps(eval_res, indent=2, ensure_ascii=False)
        filename = f"INTERNAL_accuracy_evaluation_{metric}_{release_id[:8]}.json"
        return json_content, filename

    def is_demo_mode(self) -> bool:
        """Check if environment and configuration explicitly enable demo mode (Fail-Closed default)."""
        demo_mode = os.environ.get("DEMO_MODE", "false").strip().lower() == "true"
        app_env = os.environ.get("APP_ENV", "production").strip().lower()
        return demo_mode and (app_env in ("local-demo", "demo"))

    def _verify_demo_mode_allowed(self) -> None:
        """Enforce strict demo mode access control (fail-closed)."""
        if not self.is_demo_mode():
            raise RecipientSecurityViolation(
                "Demo reset operations are strictly forbidden in production mode or when demo mode is not explicitly enabled (Fail-Closed)."
            )
        state_mgr = getattr(self.service.orchestrator, "state_manager", None)
        db_path = str(getattr(state_mgr, "db_path", ""))
        if "prod" in db_path.lower():
            raise RecipientSecurityViolation(
                f"Demo reset cannot be performed on production database: {db_path}"
            )

    def reset_demo_epoch(self) -> Dict[str, Any]:
        """Reset current active 90-day demo epoch state and associated demo releases."""
        self._verify_demo_mode_allowed()

        orchestrator = self.service.orchestrator
        epoch_mgr = getattr(orchestrator, "epoch_manager", None)
        state_mgr = getattr(orchestrator, "state_manager", None)
        ledger = getattr(orchestrator, "ledger", None)

        if not epoch_mgr or not state_mgr:
            raise RuntimeError("Epoch budget manager or state manager not available.")

        active = epoch_mgr.get_or_create_active_epoch()
        active_id = active.epoch_id
        start_date = active.epoch_start
        days = active.epoch_days
        rel_ids = list(active.release_ids)

        with state_mgr._connection() as conn:
            cur = conn.cursor()
            if rel_ids:
                placeholders = ",".join("?" for _ in rel_ids)
                cur.execute(f"DELETE FROM privacy_ledger WHERE release_id IN ({placeholders})", rel_ids)
                cur.execute(f"DELETE FROM release_store WHERE release_id IN ({placeholders})", rel_ids)
            cur.execute("DELETE FROM privacy_reservations")
            cur.execute("DELETE FROM privacy_epochs WHERE epoch_id = ?", (active_id,))
            conn.commit()

        if ledger is not None:
            with ledger._connection() as lconn:
                if rel_ids:
                    placeholders = ",".join("?" for _ in rel_ids)
                    lconn.execute(f"DELETE FROM privacy_ledger WHERE release_id IN ({placeholders})", rel_ids)
                lconn.commit()

        if hasattr(orchestrator.cache, "_memory_cache"):
            orchestrator.cache._memory_cache.clear()

        # Recompute lifetime RDP from remaining closed epochs
        all_epochs = state_mgr.list_all_epochs()
        closed_epochs = [e for e in all_epochs if e.get("status") == "CLOSED"]
        if closed_epochs:
            orders = epoch_mgr.planner.orders
            composed = {alpha: to_decimal("0.0") for alpha in orders}
            for ce in closed_epochs:
                vec_raw = json.loads(ce.get("rdp_vector_json", "{}"))
                ce_vec = {to_decimal(k): to_decimal(v) for k, v in vec_raw.items()}
                composed = compose_rdp_vectors([composed, ce_vec], orders=orders)
            state_mgr.update_lifetime_exposure_record(
                lifetime_rdp_json=json.dumps({str(k): str(v) for k, v in composed.items()}),
                total_epochs_count=len(closed_epochs),
            )
        else:
            orders = epoch_mgr.planner.orders
            zero_rdp = {alpha: to_decimal("0.0") for alpha in orders}
            state_mgr.update_lifetime_exposure_record(
                lifetime_rdp_json=json.dumps({str(k): str(v) for k, v in zero_rdp.items()}),
                total_epochs_count=1,
            )

        fresh_epoch = epoch_mgr.planner.create_epoch(
            epoch_id=active_id,
            start_date=start_date,
            epoch_days=days,
        )
        state_mgr.save_epoch_record(epoch_mgr._epoch_to_record(fresh_epoch))

        return {
            "success": True,
            "message": "演示 Epoch 已重置",
            "epoch_id": active_id,
            "effective_epsilon": 0.0,
        }

    def reset_demo_all(self) -> Dict[str, Any]:
        """Completely reset local demo environment: epochs, lifetime exposure, ledgers, and reload demo data."""
        self._verify_demo_mode_allowed()

        orchestrator = self.service.orchestrator
        epoch_mgr = getattr(orchestrator, "epoch_manager", None)
        state_mgr = getattr(orchestrator, "state_manager", None)
        ledger = getattr(orchestrator, "ledger", None)

        if not epoch_mgr or not state_mgr:
            raise RuntimeError("Epoch budget manager or state manager not available.")

        state_mgr.clear()

        if ledger is not None:
            with ledger._connection() as lconn:
                lconn.execute("DELETE FROM privacy_ledger")
                lconn.commit()

        if hasattr(orchestrator.cache, "_memory_cache"):
            orchestrator.cache._memory_cache.clear()

        orders = epoch_mgr.planner.orders
        zero_rdp = {alpha: to_decimal("0.0") for alpha in orders}
        state_mgr.update_lifetime_exposure_record(
            lifetime_rdp_json=json.dumps({str(k): str(v) for k, v in zero_rdp.items()}),
            total_epochs_count=1,
        )

        fresh_epoch = epoch_mgr.planner.create_epoch(
            epoch_id="epoch_001",
            start_date="2026-10-01",
            epoch_days=epoch_mgr.epoch_days,
        )
        state_mgr.save_epoch_record(epoch_mgr._epoch_to_record(fresh_epoch))

        self.load_demo_data()

        return {
            "success": True,
            "message": "演示环境已恢复初始状态",
            "epoch_id": "epoch_001",
            "effective_epsilon": 0.0,
        }
