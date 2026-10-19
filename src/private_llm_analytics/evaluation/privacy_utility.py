"""Multi-trial experimental evaluation runner for privacy-utility trade-offs in v0.1.1."""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from private_llm_analytics.aggregation.user_matrix import UserMatrix
from private_llm_analytics.evaluation.accuracy import (
    compute_mae,
    compute_mae_percentage_points,
    compute_ranking_accuracy,
    compute_relative_error,
    compute_rmse,
    compute_total_variation_distance,
)
from private_llm_analytics.mechanisms.gaussian import GaussianMechanism
from private_llm_analytics.mechanisms.laplace import LaplaceMechanism
from private_llm_analytics.metrics.base import BaseMetric
from private_llm_analytics.metrics.interaction_volume import InteractionVolumeMetric
from private_llm_analytics.metrics.normalized_share import UserNormalizedShareMetric
from private_llm_analytics.metrics.prevalence import UserPrevalenceMetric


def run_repeated_noise_trials(
    user_matrix: UserMatrix,
    metric: BaseMetric,
    mechanism_type: str = "gaussian",
    epsilon: float = 1.0,
    delta: float = 1e-6,
    n_trials: int = 500,
    seed: int = 1001,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Run repeated noise trials on a fixed synthetic dataset ground truth.

    Returns:
        trials_df: DataFrame with error metrics for each trial.
        summary: Dictionary of aggregated statistics (mean, median, std, p95).
    """
    rng = np.random.default_rng(seed)
    sensitivity = metric.get_sensitivity(mechanism_type)

    # 1. Instantiate mechanism
    if "gaussian" in mechanism_type.lower():
        mechanism = GaussianMechanism(
            epsilon=epsilon,
            delta=delta,
            sensitivity=sensitivity,
            method="analytic",
        )
    elif "laplace" in mechanism_type.lower():
        mechanism = LaplaceMechanism(
            epsilon=epsilon,
            sensitivity=sensitivity,
            delta=delta,
        )
    else:
        raise ValueError(f"Unknown mechanism: {mechanism_type}")

    # 2. Compute exact ground truths
    contributions = metric.build_user_contributions(user_matrix)
    bounded_aggregate = np.sum(contributions, axis=0)

    # Ground truth reference vectors
    is_simplex_distribution = False
    is_prevalence = False

    if isinstance(metric, UserPrevalenceMetric):
        # Target: prevalence rate R^{prev} in [0, 1] using public population denominator
        denom = user_matrix.public_population_size if user_matrix.public_population_size is not None else max(user_matrix.n_users, 1)
        gt_target = bounded_aggregate / denom
        is_prevalence = True
    elif isinstance(metric, UserNormalizedShareMetric):
        # Target: normalized share vector summing to 1.0 on simplex
        tot = float(np.sum(bounded_aggregate))
        gt_target = bounded_aggregate / tot if tot > 0 else np.ones(9) / 9.0
        is_simplex_distribution = True
    elif isinstance(metric, InteractionVolumeMetric):
        # Target for Volume: raw exact and clipped exact
        raw_exact = metric.compute_raw_exact(user_matrix)
        clipped_exact = bounded_aggregate
        gt_target = raw_exact
    else:
        gt_target = bounded_aggregate

    records: List[Dict[str, float]] = []

    for trial_idx in range(n_trials):
        # Add independent noise
        noisy_vec = mechanism.add_noise(bounded_aggregate, rng=rng)

        # Post-process
        pp = metric.post_process(noisy_vec, user_matrix)

        if isinstance(metric, UserPrevalenceMetric):
            pred_vec = pp["prevalence_rates_array"]
        elif isinstance(metric, UserNormalizedShareMetric):
            pred_vec = pp["share_distribution_array"]
        elif isinstance(metric, InteractionVolumeMetric):
            pred_vec = pp["clamped_volume_array"]
        else:
            pred_vec = noisy_vec

        mae = compute_mae(gt_target, pred_vec)
        rmse = compute_rmse(gt_target, pred_vec)
        rel_err = compute_relative_error(gt_target, pred_vec)

        row: Dict[str, float] = {
            "trial": trial_idx,
            "mae": mae,
            "rmse": rmse,
            "relative_error": rel_err,
        }

        if is_prevalence:
            row["mae_pp"] = compute_mae_percentage_points(gt_target, pred_vec)
            # NOTE M1: TV is strictly N/A for prevalence because rates are overlapping and do not sum to 1.0!
        elif is_simplex_distribution:
            row["mae_pp"] = compute_mae_percentage_points(gt_target, pred_vec)
            row["total_variation"] = compute_total_variation_distance(gt_target, pred_vec)

        ranking = compute_ranking_accuracy(gt_target, pred_vec)
        row["top1_match"] = ranking["top1_match"]
        row["spearman"] = ranking["spearman_correlation"]

        # Decompose Interaction Volume errors
        if isinstance(metric, InteractionVolumeMetric):
            clip_bias = compute_mae(raw_exact, clipped_exact)
            dp_err = compute_mae(clipped_exact, pred_vec)
            tot_err = compute_mae(raw_exact, pred_vec)

            row["clipping_bias_mae"] = clip_bias
            row["dp_noise_mae"] = dp_err
            row["total_volume_mae"] = tot_err

        records.append(row)

    trials_df = pd.DataFrame(records)

    # 3. Compute aggregate summary
    summary: Dict[str, Any] = {
        "metric": metric.name,
        "mechanism": mechanism.mechanism_name,
        "epsilon": epsilon,
        "delta": delta,
        "sensitivity": sensitivity,
        "noise_scale": mechanism.noise_scale,
        "n_users": user_matrix.n_users,
        "public_population_size": user_matrix.public_population_size,
        "n_trials": n_trials,
        "mae_mean": float(trials_df["mae"].mean()),
        "mae_median": float(trials_df["mae"].median()),
        "mae_std": float(trials_df["mae"].std()),
        "mae_p95": float(trials_df["mae"].quantile(0.95)),
        "rmse_mean": float(trials_df["rmse"].mean()),
        "relative_error_mean": float(trials_df["relative_error"].mean()),
        "top1_accuracy": float(trials_df["top1_match"].mean()),
        "spearman_mean": float(trials_df["spearman"].mean()),
    }

    if is_prevalence:
        summary["mae_pp_mean"] = float(trials_df["mae_pp"].mean())
        summary["mae_pp_median"] = float(trials_df["mae_pp"].median())
        summary["mae_pp_p95"] = float(trials_df["mae_pp"].quantile(0.95))
        # Explicitly omit or mark TV distance as N/A (Methodology M1)

    elif is_simplex_distribution:
        summary["mae_pp_mean"] = float(trials_df["mae_pp"].mean())
        summary["mae_pp_median"] = float(trials_df["mae_pp"].median())
        summary["mae_pp_p95"] = float(trials_df["mae_pp"].quantile(0.95))
        summary["tv_mean"] = float(trials_df["total_variation"].mean())
        summary["tv_median"] = float(trials_df["total_variation"].median())
        summary["tv_p95"] = float(trials_df["total_variation"].quantile(0.95))

    if isinstance(metric, InteractionVolumeMetric):
        summary["clipping_bias_mae"] = float(trials_df["clipping_bias_mae"].mean())
        summary["dp_noise_mae_mean"] = float(trials_df["dp_noise_mae"].mean())
        summary["total_volume_mae_mean"] = float(trials_df["total_volume_mae"].mean())

    return trials_df, summary
