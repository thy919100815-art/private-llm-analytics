"""Accuracy and utility evaluation metrics."""

from typing import Dict
import numpy as np
from scipy.stats import spearmanr


def compute_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Error."""
    return float(np.mean(np.abs(y_pred - y_true)))


def compute_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Squared Error."""
    return float(np.sqrt(np.mean((y_pred - y_true) ** 2)))


def compute_max_abs_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Maximum Absolute Error across all coordinates."""
    return float(np.max(np.abs(y_pred - y_true)))


def compute_relative_error(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-6) -> float:
    """Mean Relative Error with denominator smoothing."""
    denom = np.maximum(np.abs(y_true), eps)
    return float(np.mean(np.abs(y_pred - y_true) / denom))


def compute_mae_percentage_points(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Error in percentage points (for probabilities/shares)."""
    return float(np.mean(np.abs(y_pred - y_true)) * 100.0)


def compute_l1_distance(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """L1 distance between two vectors."""
    return float(np.sum(np.abs(y_pred - y_true)))


def compute_total_variation_distance(p: np.ndarray, q: np.ndarray) -> float:
    """Total Variation Distance between probability distributions:

    TV(P, Q) = 0.5 * sum_i |P_i - Q_i|
    """
    return float(0.5 * np.sum(np.abs(p - q)))


def compute_ranking_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Evaluate ranking preservation."""
    # Top-1 match
    top1_true = int(np.argmax(y_true))
    top1_pred = int(np.argmax(y_pred))
    top1_match = 1.0 if top1_true == top1_pred else 0.0

    # Top-3 set overlap (Jaccard)
    top3_true = set(np.argsort(y_true)[-3:])
    top3_pred = set(np.argsort(y_pred)[-3:])
    top3_overlap = float(len(top3_true.intersection(top3_pred)) / 3.0)

    # Spearman rank correlation
    corr, _ = spearmanr(y_true, y_pred)
    spearman = float(corr) if not np.isnan(corr) else 0.0

    return {
        "top1_match": top1_match,
        "top3_overlap": top3_overlap,
        "spearman_correlation": spearman,
    }


def evaluate_vector_accuracy(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    is_probability_distribution: bool = False,
) -> Dict[str, float]:
    """Compute comprehensive accuracy report for a vector release."""
    metrics: Dict[str, float] = {
        "mae": compute_mae(y_true, y_pred),
        "rmse": compute_rmse(y_true, y_pred),
        "max_abs_error": compute_max_abs_error(y_true, y_pred),
        "relative_error": compute_relative_error(y_true, y_pred),
        "l1_distance": compute_l1_distance(y_true, y_pred),
    }

    if is_probability_distribution:
        metrics["mae_pp"] = compute_mae_percentage_points(y_true, y_pred)
        metrics["total_variation"] = compute_total_variation_distance(y_true, y_pred)

    ranking = compute_ranking_accuracy(y_true, y_pred)
    metrics.update(ranking)

    return metrics
