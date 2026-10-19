"""Evaluation module."""

from private_llm_analytics.evaluation.accuracy import (
    compute_l1_distance,
    compute_mae,
    compute_mae_percentage_points,
    compute_max_abs_error,
    compute_ranking_accuracy,
    compute_relative_error,
    compute_rmse,
    compute_total_variation_distance,
    evaluate_vector_accuracy,
)
from private_llm_analytics.evaluation.ground_truth import (
    compute_ground_truth_interaction_volume,
    compute_ground_truth_normalized_share,
    compute_ground_truth_prevalence,
)
from private_llm_analytics.evaluation.privacy_utility import run_repeated_noise_trials
from private_llm_analytics.evaluation.report import (
    plot_single_line_chart,
    save_summary_table,
    set_matplotlib_style,
)

__all__ = [
    "compute_ground_truth_interaction_volume",
    "compute_ground_truth_normalized_share",
    "compute_ground_truth_prevalence",
    "compute_l1_distance",
    "compute_mae",
    "compute_mae_percentage_points",
    "compute_max_abs_error",
    "compute_ranking_accuracy",
    "compute_relative_error",
    "compute_rmse",
    "compute_total_variation_distance",
    "evaluate_vector_accuracy",
    "plot_single_line_chart",
    "run_repeated_noise_trials",
    "save_summary_table",
    "set_matplotlib_style",
]
