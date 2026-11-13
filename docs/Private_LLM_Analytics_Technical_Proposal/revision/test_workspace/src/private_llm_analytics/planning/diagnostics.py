"""Diagnostics and evaluation metrics for DP release planner performance (v0.3)."""

import math
from typing import Dict, Sequence, Tuple
import numpy as np
from scipy.stats import spearmanr


def compute_regret(
    true_error_selected: float,
    true_error_oracle: float,
) -> Tuple[float, float]:
    """Compute absolute and relative regret against the oracle baseline.

    regret = true_error_selected - true_error_oracle
    relative_regret = regret / max(1e-12, true_error_oracle)
    """
    abs_regret = max(0.0, float(true_error_selected) - float(true_error_oracle))
    base = max(1e-12, float(true_error_oracle))
    rel_regret = abs_regret / base
    return abs_regret, rel_regret


def compute_calibration_error(
    predicted_errors: Sequence[float],
    observed_errors: Sequence[float],
) -> Dict[str, float]:
    """Evaluate calibration fidelity of the model-based utility estimator against empirical truth."""
    if len(predicted_errors) != len(observed_errors):
        raise ValueError("predicted_errors and observed_errors must have identical length")
    if len(predicted_errors) == 0:
        return {"bias": 0.0, "mae": 0.0, "rmse": 0.0, "spearman_corr": 1.0}

    preds = np.array(predicted_errors, dtype=float)
    actuals = np.array(observed_errors, dtype=float)

    diff = preds - actuals
    bias = float(np.mean(diff))
    mae = float(np.mean(np.abs(diff)))
    rmse = float(math.sqrt(float(np.mean(diff**2))))

    if len(preds) > 1 and np.std(preds) > 1e-12 and np.std(actuals) > 1e-12:
        corr, _ = spearmanr(preds, actuals)
        spearman_corr = float(corr) if not math.isnan(corr) else 0.0
    else:
        spearman_corr = 1.0

    return {
        "bias": bias,
        "mae": mae,
        "rmse": rmse,
        "spearman_corr": spearman_corr,
    }
