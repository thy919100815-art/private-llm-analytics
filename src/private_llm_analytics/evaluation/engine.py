"""Differential Privacy Utility and Accuracy Evaluation Engine (v1.1).

Computes metric-specific accuracy, decomposition (Raw / Clipped / DP),
and multi-trial epsilon-utility sweep experiments in isolated experiment mode.
Strictly zero privacy spend on production ledger.
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd

from private_llm_analytics.aggregation.user_matrix import UserMatrix
from private_llm_analytics.evaluation.accuracy import (
    compute_l1_distance,
    compute_mae,
    compute_relative_error,
    compute_rmse,
)
from private_llm_analytics.service.analytics import AnalyticsService
from private_llm_analytics.statistics.registry import StatisticProfileRegistry


def evaluate_interaction_volume_accuracy(
    user_matrix: UserMatrix,
    dp_results: Dict[str, Any],
) -> Dict[str, Any]:
    """Compute detailed decomposition and error metrics for Interaction Volume.

    Raw Exact: M_raw
    Clipped Exact: M_clip
    DP Released: M_DP
    Clipping Bias: E_clip = M_clip - M_raw
    DP Noise Error: E_noise = M_DP - M_clip
    Total Error: E_total = M_DP - M_raw
    Relative Error: RE = |M_DP - M_raw| / max(|M_raw|, 1.0)
    """
    profile = StatisticProfileRegistry.get("interaction_volume")
    cell_names = list(user_matrix.cell_names)

    # Extract DP vector
    dp_volumes_dict = dp_results.get("cell_volumes", dp_results)
    dp_vector = np.array([float(dp_volumes_dict.get(c, 0.0)) for c in cell_names], dtype=np.float64)

    decomp = profile.decompose_error(dp_vector, user_matrix)
    raw_exact = decomp["raw_exact"]
    clipped_exact = decomp["clipped_exact"]
    clipping_bias = decomp["clipping_bias"]
    dp_noise_error = decomp["dp_noise_error"]
    total_error = decomp["total_error"]

    denom = np.maximum(np.abs(raw_exact), 1.0)
    rel_errors = np.abs(total_error) / denom

    rows: List[Dict[str, Any]] = []
    for i, c_name in enumerate(cell_names):
        parts = c_name.split(":")
        prov = parts[0] if len(parts) > 1 else ""
        tsk = parts[1] if len(parts) > 1 else parts[0]
        rows.append({
            "cell": c_name,
            "provider": prov,
            "task": tsk,
            "raw_exact": round(float(raw_exact[i]), 1),
            "clipped_exact": round(float(clipped_exact[i]), 1),
            "dp_value": round(float(dp_vector[i]), 1),
            "clipping_bias": round(float(clipping_bias[i]), 1),
            "dp_noise_error": round(float(dp_noise_error[i]), 1),
            "total_error": round(float(total_error[i]), 1),
            "relative_error_pct": round(float(rel_errors[i] * 100.0), 2),
        })

    mae = compute_mae(raw_exact, dp_vector)
    rmse = compute_rmse(raw_exact, dp_vector)
    mre = float(np.mean(rel_errors))
    max_re = float(np.max(rel_errors))

    return {
        "metric": "interaction_volume",
        "primary_metric_name": "平均相对误差",
        "primary_metric_display": f"{round(mre * 100.0, 1)}%",
        "primary_metric_value": round(mre * 100.0, 4),
        "mae": round(mae, 1),
        "rmse": round(rmse, 1),
        "mean_relative_error_pct": round(mre * 100.0, 2),
        "max_relative_error_pct": round(max_re * 100.0, 2),
        "rows": rows,
    }


def evaluate_prevalence_accuracy(
    user_matrix: UserMatrix,
    dp_results: Dict[str, Any],
) -> Dict[str, Any]:
    """Compute percentage-point error for User Prevalence."""
    profile = StatisticProfileRegistry.get("user_prevalence")
    cell_names = list(user_matrix.cell_names)

    # Compute exact prevalence rates
    exact_vec = profile.compute_exact(user_matrix)
    exact_post = profile.post_process(exact_vec, user_matrix)
    exact_rates = exact_post["prevalence_rates"]

    dp_rates = dp_results.get("prevalence_rates", dp_results)

    rows: List[Dict[str, Any]] = []
    pp_errors: List[float] = []

    for c_name in cell_names:
        parts = c_name.split(":")
        prov = parts[0] if len(parts) > 1 else ""
        tsk = parts[1] if len(parts) > 1 else parts[0]

        e_rate = float(exact_rates.get(c_name, 0.0))
        d_rate = float(dp_rates.get(c_name, 0.0)) if dp_rates.get(c_name) is not None else 0.0
        pp_err = abs(d_rate - e_rate) * 100.0
        pp_errors.append(pp_err)

        rows.append({
            "cell": c_name,
            "provider": prov,
            "task": tsk,
            "exact_rate_pct": round(e_rate * 100.0, 2),
            "dp_rate_pct": round(d_rate * 100.0, 2),
            "pp_error": round(pp_err, 2),
        })

    mean_pp = float(np.mean(pp_errors)) if pp_errors else 0.0
    max_pp = float(np.max(pp_errors)) if pp_errors else 0.0

    return {
        "metric": "user_prevalence",
        "primary_metric_name": "平均百分点误差",
        "primary_metric_display": f"{round(mean_pp, 2)} pp",
        "primary_metric_value": round(mean_pp, 4),
        "mean_pp_error": round(mean_pp, 2),
        "max_pp_error": round(max_pp, 2),
        "rows": rows,
    }


def evaluate_normalized_share_accuracy(
    user_matrix: UserMatrix,
    dp_results: Dict[str, Any],
) -> Dict[str, Any]:
    """Compute percentage-point error and L1 distance for User-Normalized Share."""
    profile = StatisticProfileRegistry.get("user_normalized_share")
    cell_names = list(user_matrix.cell_names)

    exact_vec = profile.compute_exact(user_matrix)
    exact_post = profile.post_process(exact_vec, user_matrix)
    exact_shares = exact_post["cell_shares"]

    dp_shares = dp_results.get("cell_shares", dp_results)

    rows: List[Dict[str, Any]] = []
    pp_errors: List[float] = []
    exact_arr = []
    dp_arr = []

    for c_name in cell_names:
        parts = c_name.split(":")
        prov = parts[0] if len(parts) > 1 else ""
        tsk = parts[1] if len(parts) > 1 else parts[0]

        e_share = float(exact_shares.get(c_name, 0.0))
        d_share = float(dp_shares.get(c_name, 0.0)) if dp_shares.get(c_name) is not None else 0.0
        pp_err = abs(d_share - e_share) * 100.0
        pp_errors.append(pp_err)
        exact_arr.append(e_share)
        dp_arr.append(d_share)

        rows.append({
            "cell": c_name,
            "provider": prov,
            "task": tsk,
            "exact_share_pct": round(e_share * 100.0, 2),
            "dp_share_pct": round(d_share * 100.0, 2),
            "pp_error": round(pp_err, 2),
        })

    mean_pp = float(np.mean(pp_errors)) if pp_errors else 0.0
    max_pp = float(np.max(pp_errors)) if pp_errors else 0.0
    l1_dist = compute_l1_distance(np.array(exact_arr), np.array(dp_arr))

    return {
        "metric": "user_normalized_share",
        "primary_metric_name": "平均百分点误差",
        "primary_metric_display": f"{round(mean_pp, 2)} pp",
        "primary_metric_value": round(mean_pp, 4),
        "mean_pp_error": round(mean_pp, 2),
        "max_pp_error": round(max_pp, 2),
        "l1_distance": round(l1_dist, 4),
        "rows": rows,
    }


def evaluate_success_rate_accuracy(
    user_matrix: UserMatrix,
    dp_results: Dict[str, Any],
) -> Dict[str, Any]:
    """Compute percentage-point error for Success Rate."""
    profile = StatisticProfileRegistry.get("success_rate")
    cell_names = list(user_matrix.cell_names)

    s_raw, n_raw, r_raw = profile.compute_raw_exact(user_matrix)
    dp_rates = dp_results.get("success_rates", {})

    rows: List[Dict[str, Any]] = []
    pp_errors: List[float] = []
    suppressed_count = 0

    for i, c_name in enumerate(cell_names):
        parts = c_name.split(":")
        prov = parts[0] if len(parts) > 1 else ""
        tsk = parts[1] if len(parts) > 1 else parts[0]

        e_rate = float(r_raw[i]) if not np.isnan(r_raw[i]) else 0.0
        d_val = dp_rates.get(c_name)

        if d_val is None:
            suppressed = True
            suppressed_count += 1
            d_rate_display = None
            pp_err = None
        else:
            suppressed = False
            d_rate = float(d_val)
            d_rate_display = round(d_rate * 100.0, 2)
            pp_err = round(abs(d_rate - e_rate) * 100.0, 2)
            pp_errors.append(pp_err)

        rows.append({
            "cell": c_name,
            "provider": prov,
            "task": tsk,
            "raw_success_count": round(float(s_raw[i]), 1),
            "raw_total_count": round(float(n_raw[i]), 1),
            "exact_rate_pct": round(e_rate * 100.0, 2),
            "dp_rate_pct": d_rate_display,
            "pp_error": pp_err,
            "suppressed": suppressed,
        })

    mean_pp = float(np.mean(pp_errors)) if pp_errors else 0.0
    max_pp = float(np.max(pp_errors)) if pp_errors else 0.0

    return {
        "metric": "success_rate",
        "primary_metric_name": "平均百分点误差",
        "primary_metric_display": f"{round(mean_pp, 2)} pp",
        "primary_metric_value": round(mean_pp, 4),
        "mean_pp_error": round(mean_pp, 2),
        "max_pp_error": round(max_pp, 2),
        "suppressed_count": suppressed_count,
        "rows": rows,
    }


def evaluate_top_k_accuracy(
    user_matrix: UserMatrix,
    dp_results: Dict[str, Any],
    k: int = 5,
) -> Dict[str, Any]:
    """Compute Precision@K and Jaccard similarity for Top-K Tasks discovery."""
    profile = StatisticProfileRegistry.get("top_k_tasks")
    tasks = list(user_matrix.tasks)

    # Compute exact tasks counts
    cont = profile.build_user_contributions(user_matrix)
    exact_counts = np.sum(cont, axis=0)
    sorted_idx = sorted(range(len(tasks)), key=lambda i: (-exact_counts[i], tasks[i]))
    effective_k = min(k, len(tasks))

    exact_top_k = [tasks[i] for i in sorted_idx[:effective_k]]

    dp_items = dp_results.get("top_k", [])
    dp_top_k = [item["task"] for item in dp_items][:effective_k]

    set_exact = set(exact_top_k)
    set_dp = set(dp_top_k)
    intersection = set_exact.intersection(set_dp)
    union = set_exact.union(set_dp)

    overlap_count = len(intersection)
    precision_at_k = (overlap_count / effective_k * 100.0) if effective_k > 0 else 100.0
    jaccard = (overlap_count / len(union) * 100.0) if union else 100.0

    rows: List[Dict[str, Any]] = []
    for rank in range(effective_k):
        e_t = exact_top_k[rank] if rank < len(exact_top_k) else ""
        d_t = dp_top_k[rank] if rank < len(dp_top_k) else ""
        hit = d_t in set_exact if d_t else False
        rows.append({
            "rank": rank + 1,
            "exact_task": e_t,
            "dp_task": d_t,
            "hit": hit,
        })

    explanation = f"真实热门任务前{effective_k}名中，DP结果正确保留了其中{overlap_count}个。"

    return {
        "metric": "top_k_tasks",
        "primary_metric_name": f"Top-{effective_k} 命中率",
        "primary_metric_display": f"{round(precision_at_k, 1)}%",
        "primary_metric_value": round(precision_at_k, 2),
        "precision_at_k": round(precision_at_k, 2),
        "jaccard_similarity": round(jaccard, 2),
        "k": effective_k,
        "overlap_count": overlap_count,
        "explanation": explanation,
        "rows": rows,
    }


def evaluate_release_accuracy(
    metric: str,
    user_matrix: UserMatrix,
    dp_results: Dict[str, Any],
) -> Dict[str, Any]:
    """Unified dispatcher for evaluating accuracy across any of the 5 statistics."""
    if metric == "interaction_volume":
        return evaluate_interaction_volume_accuracy(user_matrix, dp_results)
    elif metric == "user_prevalence":
        return evaluate_prevalence_accuracy(user_matrix, dp_results)
    elif metric == "user_normalized_share":
        return evaluate_normalized_share_accuracy(user_matrix, dp_results)
    elif metric == "success_rate":
        return evaluate_success_rate_accuracy(user_matrix, dp_results)
    elif metric == "top_k_tasks":
        return evaluate_top_k_accuracy(user_matrix, dp_results)
    else:
        raise ValueError(f"Unsupported metric for accuracy evaluation: '{metric}'")


def compute_all_metrics_utility_summary(
    service: AnalyticsService,
    user_matrix: UserMatrix,
    epsilon: float = 1.0,
    delta: float = 1e-6,
) -> List[Dict[str, Any]]:
    """Compute benchmark utility overview across all 5 metrics in isolated experiment mode."""
    metrics_info = [
        ("user_prevalence", "用户使用率"),
        ("user_normalized_share", "用户使用结构"),
        ("interaction_volume", "交互量"),
        ("success_rate", "任务成功率"),
        ("top_k_tasks", "热门任务 Top-K"),
    ]

    summary_list: List[Dict[str, Any]] = []

    for m_id, name_zh in metrics_info:
        try:
            exp_res = service.compute_experiment(
                metric=m_id,
                recipient="internal",
                epsilon=epsilon,
                delta=delta,
                include_exact=True,
                noise_seed=42,
            )
            dp_res = exp_res.get("results", {})
            eval_res = evaluate_release_accuracy(m_id, user_matrix, dp_res)
            summary_list.append({
                "metric_id": m_id,
                "metric_name_zh": name_zh,
                "primary_metric": eval_res.get("primary_metric_name"),
                "value_display": eval_res.get("primary_metric_display"),
                "numeric_value": eval_res.get("primary_metric_value"),
            })
        except Exception as e:
            summary_list.append({
                "metric_id": m_id,
                "metric_name_zh": name_zh,
                "primary_metric": "评估异常",
                "value_display": "N/A",
                "numeric_value": None,
                "error": str(e),
            })

    return summary_list


def run_epsilon_utility_sweep(
    service: AnalyticsService,
    user_matrix: UserMatrix,
    metric: str,
    candidate_epsilons: Optional[List[float]] = None,
    n_trials: int = 10,
    delta: float = 1e-6,
) -> Dict[str, Any]:
    """Execute rigorous multi-trial epsilon vs utility sweep in isolated experiment mode.

    Guarantees:
      - 0 production privacy ledger entries
      - 0 production lifetime epsilon consumed
      - Mean, std, median, p90 error over n_trials distinct RNG seeds for each epsilon
    """
    eps_list = candidate_epsilons or [0.1, 0.25, 0.5, 1.0, 2.0, 4.0]
    results_by_eps: List[Dict[str, Any]] = []

    y_axis_label = "平均相对误差 (%)"
    if metric in ("user_prevalence", "user_normalized_share", "success_rate"):
        y_axis_label = "平均百分点误差 (pp)"
    elif metric == "top_k_tasks":
        y_axis_label = "Top-K 命中率 (%)"

    for eps in eps_list:
        trial_errors: List[float] = []
        for trial_idx in range(n_trials):
            seed = 1000 + int(eps * 100) + trial_idx
            exp_res = service.compute_experiment(
                metric=metric,
                recipient="internal",
                epsilon=eps,
                delta=delta,
                include_exact=True,
                noise_seed=seed,
            )
            dp_res = exp_res.get("results", {})
            eval_res = evaluate_release_accuracy(metric, user_matrix, dp_res)
            val = eval_res.get("primary_metric_value")
            if val is not None:
                trial_errors.append(float(val))

        if trial_errors:
            mean_val = float(np.mean(trial_errors))
            std_val = float(np.std(trial_errors))
            median_val = float(np.median(trial_errors))
            p90_val = float(np.percentile(trial_errors, 90))
        else:
            mean_val = std_val = median_val = p90_val = 0.0

        results_by_eps.append({
            "epsilon": float(eps),
            "mean_error": round(mean_val, 2),
            "std_error": round(std_val, 2),
            "median_error": round(median_val, 2),
            "p90_error": round(p90_val, 2),
            "trials_count": len(trial_errors),
        })

    explanation = (
        "ε 越小通常意味着更强隐私保护，但统计噪声通常更大。\n"
        "ε 越大通常能获得更高统计精度，但会消耗更多隐私预算。"
    )

    return {
        "metric": metric,
        "y_axis_label": y_axis_label,
        "candidate_epsilons": eps_list,
        "n_trials": n_trials,
        "points": results_by_eps,
        "explanation": explanation,
    }
