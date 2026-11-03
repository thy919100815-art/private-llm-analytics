"""Exporter for publishing differential privacy release outputs to JSON or CSV (v1.0.x / v1.1).

Supports:
1. Standard cell statistics (user_prevalence, user_normalized_share, interaction_volume)
2. Success Rate (safe DP rates, counts, suppression states)
3. Top-K Tasks (rank, task name, noisy count score, prevalence)
4. Sanitized JSON export preserving rich metadata without leaking internal RNG/raw data
5. Trusted internal evaluation comparison export (raw_exact vs clipped_exact vs dp_released)
"""

from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import numpy as np
import pandas as pd


WARNING_INTERNAL_ONLY = (
    "# WARNING: 该文件包含内部精确统计，仅供可信管理员评估，禁止作为外部发布文件。\n"
    "# CONFIDENTIAL: INTERNAL EVALUATION ONLY - DO NOT DISTRIBUTE EXTERNALLY.\n"
)


def _get_provider_display(prov_id: str, configured_name: Optional[str] = None) -> str:
    if configured_name and configured_name != prov_id:
        return configured_name
    defaults = {
        "provider_a": "OpenAI",
        "provider_b": "Kimi",
        "provider_c": "DeepSeek",
        "internal": "内部 AI 治理团队",
    }
    return defaults.get(prov_id, prov_id)


def _get_metric_friendly_name(metric_id: str) -> str:
    mapping = {
        "user_prevalence": "用户使用率 (user_prevalence)",
        "user_normalized_share": "用户归一化占比 (user_normalized_share)",
        "interaction_volume": "交互量 (interaction_volume)",
        "success_rate": "成功率 (success_rate)",
        "top_k_tasks": "热门任务 Top-K (top_k_tasks)",
    }
    return mapping.get(metric_id, metric_id)


def _get_metric_unit(metric_id: str) -> str:
    mapping = {
        "user_prevalence": "使用率 (%)",
        "user_normalized_share": "工作流占比 (%)",
        "interaction_volume": "次交互 (interactions)",
        "success_rate": "成功率 (%)",
        "top_k_tasks": "排名与加噪频次 (rank & score)",
    }
    return mapping.get(metric_id, "数值")


def _get_recipient_display(rec_id: str, rec_name: Optional[str] = None) -> str:
    if rec_name:
        return rec_name
    defaults = {
        "internal": "内部 AI 治理团队",
        "provider_a": "OpenAI",
        "provider_b": "Kimi",
        "provider_c": "DeepSeek",
    }
    return defaults.get(rec_id, rec_id)


def build_release_dataframe(release_output: Dict[str, Any]) -> pd.DataFrame:
    """Transform authoritative DP release output into a clean, human-readable DataFrame."""
    results = release_output.get("results", {})
    metric = release_output.get("metric", "")
    rec_id = release_output.get("recipient", release_output.get("recipient_id", ""))
    rec_name = _get_recipient_display(rec_id, release_output.get("recipient_name"))
    eps = release_output.get("epsilon")
    delta = release_output.get("delta")
    mech = release_output.get("mechanism", "gaussian")
    sigma = release_output.get("noise_scale")
    rel_id = release_output.get("release_id", release_output.get("release_key", ""))
    ts = release_output.get("timestamp") or release_output.get("created_at") or ""
    window = release_output.get("reporting_window") or release_output.get("window_end") or "30d"
    metric_friendly = _get_metric_friendly_name(metric)
    stat_unit = _get_metric_unit(metric)
    disclaimer = "以下结果已加入差分隐私噪声，仅用于隐私保护后的统计共享"

    rows: List[Dict[str, Any]] = []

    # 1. Success Rate
    if metric == "success_rate" or "success_rates" in results or "cell_success_rates" in results:
        rates = results.get("success_rates", results.get("cell_success_rates", {}))
        dp_totals = results.get("dp_total_counts", {})
        dp_successes = results.get("dp_success_counts", {})
        suppressed = set(results.get("suppressed_cells", []))

        for cell_name, rate in rates.items():
            parts = cell_name.split(":")
            prov = parts[0] if len(parts) > 1 else ""
            tsk = parts[1] if len(parts) > 1 else parts[0]
            prov_disp = _get_provider_display(prov)
            cell_disp = f"{prov_disp} × {tsk}" if prov else tsk
            is_sup = (cell_name in suppressed) or (rate is None)
            rate_val = round(rate, 4) if rate is not None else None
            dp_est = f"{rate_val * 100:.2f}%" if rate_val is not None else "SUPPRESSED (受保护抑制)"
            rows.append({
                "provider": prov,
                "provider_display_name": prov_disp,
                "task": tsk,
                "cell_display": cell_disp,
                "metric": "success_rate",
                "metric_name": metric_friendly,
                "dp_value": rate_val,
                "dp_estimate": dp_est,
                "statistic_unit": stat_unit,
                "status": "SUPPRESSED" if is_sup else "NORMAL",
                "dp_success_rate": rate_val,
                "dp_success_count": round(dp_successes.get(cell_name, 0.0), 2) if cell_name in dp_successes else None,
                "dp_total_count": round(dp_totals.get(cell_name, 0.0), 2) if cell_name in dp_totals else None,
                "suppressed": is_sup,
                "epsilon": eps,
                "delta": delta,
                "mechanism": mech,
                "noise_scale": sigma,
                "sigma": sigma,
                "reporting_window": window,
                "recipient_id": rec_id,
                "recipient_internal_id": rec_id,
                "recipient_name": rec_name,
                "recipient_display_name": rec_name,
                "release_id": rel_id,
                "timestamp": ts,
                "disclaimer": disclaimer,
            })
        return pd.DataFrame(rows)

    # 2. Top-K Tasks
    if metric == "top_k_tasks" or "top_k" in results:
        top_k_items = results.get("top_k", [])
        denom_val = results.get("public_population_size", release_output.get("public_population_size", 50000))
        for item in top_k_items:
            noisy_score = round(item.get("noisy_count", 0.0), 2)
            est_prev = round(item.get("estimated_prevalence", 0.0), 4) if item.get("estimated_prevalence") is not None else None
            rows.append({
                "rank": item.get("rank"),
                "task": item.get("task"),
                "noisy_count": noisy_score,
                "estimated_prevalence": est_prev,
                "prevalence_denominator_type": "public_enterprise_population",
                "prevalence_denominator_value": denom_val,
                "cell_display": str(item.get("task")),
                "metric": "top_k_tasks",
                "metric_name": metric_friendly,
                "dp_value": noisy_score,
                "dp_estimate": f"Rank #{item.get('rank')} (score: {noisy_score})",
                "statistic_unit": stat_unit,
                "status": "NORMAL",
                "dp_noisy_score": noisy_score,
                "epsilon": eps,
                "delta": delta,
                "mechanism": mech,
                "noise_scale": sigma,
                "sigma": sigma,
                "reporting_window": window,
                "recipient_id": rec_id,
                "recipient_internal_id": rec_id,
                "recipient_name": rec_name,
                "recipient_display_name": rec_name,
                "release_id": rel_id,
                "timestamp": ts,
                "disclaimer": disclaimer,
            })
        return pd.DataFrame(rows)

    # 3. Standard Cell Statistics (volume, prevalence, normalized share)
    cell_data = None
    metric_label = metric
    for key in ("cell_volumes", "prevalence_rates", "cell_shares", "dp_counts"):
        if key in results and isinstance(results[key], dict):
            cell_data = results[key]
            if not metric_label:
                metric_label = key
            break

    if cell_data is not None:
        for cell_name, val in cell_data.items():
            parts = cell_name.split(":")
            prov = parts[0] if len(parts) > 1 else ""
            tsk = parts[1] if len(parts) > 1 else parts[0]
            prov_disp = _get_provider_display(prov)
            cell_disp = f"{prov_disp} × {tsk}" if prov else tsk
            num_val = round(float(val), 4) if val is not None else None
            dp_est = f"{num_val:.4f}" if num_val is not None else "-"
            rows.append({
                "provider": prov,
                "provider_display_name": prov_disp,
                "task": tsk,
                "cell_display": cell_disp,
                "metric": metric_label,
                "metric_name": metric_friendly,
                "dp_value": num_val,
                "dp_estimate": dp_est,
                "statistic_unit": stat_unit,
                "status": "NORMAL",
                "epsilon": eps,
                "delta": delta,
                "mechanism": mech,
                "noise_scale": sigma,
                "sigma": sigma,
                "reporting_window": window,
                "recipient_id": rec_id,
                "recipient_internal_id": rec_id,
                "recipient_name": rec_name,
                "recipient_display_name": rec_name,
                "release_id": rel_id,
                "timestamp": ts,
                "disclaimer": disclaimer,
            })
        return pd.DataFrame(rows)

    # Fallback to key-value rows
    for k, v in results.items():
        if isinstance(v, (int, float, str, bool)):
            rows.append({
                "metric_field": k,
                "value": v,
                "metric": metric_label,
                "metric_name": metric_friendly,
                "dp_value": v if isinstance(v, (int, float)) else None,
                "dp_estimate": str(v),
                "statistic_unit": stat_unit,
                "status": "NORMAL",
                "epsilon": eps,
                "delta": delta,
                "noise_scale": sigma,
                "sigma": sigma,
                "reporting_window": window,
                "recipient_id": rec_id,
                "recipient_internal_id": rec_id,
                "recipient_name": rec_name,
                "recipient_display_name": rec_name,
                "release_id": rel_id,
                "timestamp": ts,
                "disclaimer": disclaimer,
            })
    return pd.DataFrame(rows)


def build_release_csv_string(release_output: Dict[str, Any]) -> str:
    """Serialize DP release output to CSV formatted string with UTF-8 BOM for Windows Excel compatibility."""
    df = build_release_dataframe(release_output)
    return "\ufeff" + df.to_csv(index=False)


def export_release_csv(release_output: Dict[str, Any], file_path: Union[str, Path]) -> None:
    """Export DP release output as CSV file."""
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df = build_release_dataframe(release_output)
    df.to_csv(path, index=False)


def sanitize_release_for_export(release_output: Dict[str, Any]) -> Dict[str, Any]:
    """Strip any internal raw data or RNG state while preserving rich release metadata."""
    forbidden_keys = {
        "raw_dataframe",
        "rng",
        "seed",
        "noise_seed",
        "rng_state",
        "random_state",
        "user_ids",
        "raw_users",
        "exact_ground_truth",
        "exact_data",
        "clean_vector",
        "clipped_vector",
        "raw_prompt",
        "prompt",
    }
    sanitized: Dict[str, Any] = {}
    for k, v in release_output.items():
        if k in forbidden_keys:
            continue
        if k == "exact_results":
            # If recipient is external, never export exact results
            rec_role = release_output.get("recipient_role", "")
            if rec_role == "external":
                continue
        sanitized[k] = v

    # Add rich metadata and display names
    rec_id = sanitized.get("recipient", sanitized.get("recipient_id", ""))
    rec_name = _get_recipient_display(rec_id, sanitized.get("recipient_name"))
    metric = sanitized.get("metric", "")

    sanitized["recipient_id"] = rec_id
    sanitized["recipient_internal_id"] = rec_id
    sanitized["recipient_name"] = rec_name
    sanitized["recipient_display_name"] = rec_name
    sanitized["metric_name"] = _get_metric_friendly_name(metric)
    sanitized["statistic_unit"] = _get_metric_unit(metric)
    sanitized["sigma"] = sanitized.get("noise_scale")
    sanitized["disclaimer"] = "以下结果已加入差分隐私噪声，仅用于隐私保护后的统计共享"
    if "timestamp" not in sanitized:
        from datetime import datetime, timezone
        sanitized["timestamp"] = datetime.now(timezone.utc).isoformat()
    if "reporting_window" not in sanitized:
        sanitized["reporting_window"] = sanitized.get("window_end") or "30d"

    if metric == "top_k_tasks":
        res_dict = sanitized.get("results", {}) if isinstance(sanitized.get("results"), dict) else {}
        denom_val = res_dict.get("public_population_size", sanitized.get("public_population_size", 50000))
        sanitized["prevalence_denominator_type"] = "public_enterprise_population"
        sanitized["prevalence_denominator_value"] = denom_val

    return sanitized


def build_release_json_string(release_output: Dict[str, Any]) -> str:
    """Serialize sanitized DP release output to JSON string."""
    clean = sanitize_release_for_export(release_output)

    def _default(obj: Any) -> Any:
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.floating, float)):
            return float(obj)
        if isinstance(obj, (np.integer, int)):
            return int(obj)
        return str(obj)

    return json.dumps(clean, indent=2, default=_default)


def export_release_json(release_output: Dict[str, Any], file_path: Union[str, Path]) -> None:
    """Export sanitized DP release output as JSON file."""
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = build_release_json_string(release_output)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def build_internal_comparison_dataframe(
    release_output: Dict[str, Any],
    exact_data: Optional[Dict[str, Any]] = None,
) -> pd.DataFrame:
    """Build detailed Raw vs Clipped vs DP comparison DataFrame for trusted admin evaluation."""
    results = release_output.get("results", {})
    metric = release_output.get("metric", "")
    exact = exact_data or release_output.get("exact_results", {})

    rows: List[Dict[str, Any]] = []

    # 1. Interaction Volume
    if metric == "interaction_volume" or "cell_volumes" in results:
        dp_volumes = results.get("cell_volumes", {})
        raw_exact = exact.get("clean_vector", exact.get("raw_exact", {}))
        clipped_exact = exact.get("clipped_vector", exact.get("clipped_exact", {}))

        # Check if vectors are lists or dicts
        cell_names = release_output.get("cell_names", list(dp_volumes.keys()))

        for idx, cell in enumerate(cell_names):
            parts = cell.split(":")
            prov = parts[0] if len(parts) > 1 else ""
            tsk = parts[1] if len(parts) > 1 else parts[0]

            dp_val = float(dp_volumes.get(cell, 0.0))
            if isinstance(raw_exact, list) and idx < len(raw_exact):
                r_val = float(raw_exact[idx])
            elif isinstance(raw_exact, dict):
                r_val = float(raw_exact.get(cell, 0.0))
            else:
                r_val = dp_val

            if isinstance(clipped_exact, list) and idx < len(clipped_exact):
                c_val = float(clipped_exact[idx])
            elif isinstance(clipped_exact, dict):
                c_val = float(clipped_exact.get(cell, 0.0))
            else:
                c_val = r_val

            clipping_bias = c_val - r_val
            noise_error = dp_val - c_val
            total_error = dp_val - r_val
            rel_error = (abs(total_error) / r_val) if r_val > 0 else 0.0

            rows.append({
                "cell": cell,
                "provider": prov,
                "task": tsk,
                "raw_exact": round(r_val, 2),
                "clipped_exact": round(c_val, 2),
                "dp_released": round(dp_val, 2),
                "clipping_bias": round(clipping_bias, 2),
                "dp_noise_error": round(noise_error, 2),
                "total_error": round(total_error, 2),
                "relative_error": round(rel_error, 4),
            })
        return pd.DataFrame(rows)

    # 2. Other metrics (general comparison)
    # Extract DP values
    dp_dict = {}
    for k in ("prevalence_rates", "cell_shares", "success_rates", "cell_success_rates"):
        if k in results and isinstance(results[k], dict):
            dp_dict = results[k]
            break

    raw_exact = exact.get("clean_vector", exact.get("raw_exact", {}))
    cell_names = release_output.get("cell_names", list(dp_dict.keys()))

    for idx, cell in enumerate(cell_names):
        parts = cell.split(":")
        prov = parts[0] if len(parts) > 1 else ""
        tsk = parts[1] if len(parts) > 1 else parts[0]

        dp_val = dp_dict.get(cell)
        if isinstance(raw_exact, list) and idx < len(raw_exact):
            r_val = float(raw_exact[idx])
        elif isinstance(raw_exact, dict):
            r_val = float(raw_exact.get(cell, 0.0))
        else:
            r_val = None

        if dp_val is not None and r_val is not None:
            tot_err = float(dp_val) - float(r_val)
            rel_err = (abs(tot_err) / r_val) if r_val > 0 else 0.0
        else:
            tot_err = None
            rel_err = None

        rows.append({
            "cell": cell,
            "provider": prov,
            "task": tsk,
            "raw_exact": round(r_val, 4) if r_val is not None else None,
            "dp_released": round(dp_val, 4) if dp_val is not None else None,
            "total_error": round(tot_err, 4) if tot_err is not None else None,
            "relative_error": round(rel_err, 4) if rel_err is not None else None,
        })

    return pd.DataFrame(rows)


def build_internal_comparison_csv_string(
    release_output: Dict[str, Any],
    exact_data: Optional[Dict[str, Any]] = None,
) -> str:
    """Return CSV text with prominent confidential warning banner for trusted admin evaluation."""
    df = build_internal_comparison_dataframe(release_output, exact_data=exact_data)
    csv_body = df.to_csv(index=False)
    return WARNING_INTERNAL_ONLY + csv_body


def export_internal_comparison_csv(
    release_output: Dict[str, Any],
    file_path: Union[str, Path],
    exact_data: Optional[Dict[str, Any]] = None,
) -> None:
    """Export internal evaluation comparison CSV with prominent warning header."""
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = build_internal_comparison_csv_string(release_output, exact_data=exact_data)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
