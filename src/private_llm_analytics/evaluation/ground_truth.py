"""Ground truth calculation module (for internal evaluation only)."""

from typing import Dict
import numpy as np

from private_llm_analytics.aggregation.user_matrix import UserMatrix
from private_llm_analytics.contribution.binary import transform_user_prevalence
from private_llm_analytics.contribution.clipping import transform_interaction_volume_l2_clip
from private_llm_analytics.contribution.normalization import transform_user_normalized_share


def compute_ground_truth_prevalence(
    user_matrix: UserMatrix,
    max_cells_per_user: int = 9,
) -> Dict[str, np.ndarray]:
    """Compute exact prevalence counts and rates."""
    binary_contributions = transform_user_prevalence(
        user_matrix.matrix,
        max_cells_per_user=max_cells_per_user,
    )
    exact_counts = np.sum(binary_contributions, axis=0)
    exact_rates = exact_counts / max(user_matrix.n_users, 1)
    return {
        "exact_counts": exact_counts,
        "exact_rates": exact_rates,
    }


def compute_ground_truth_normalized_share(
    user_matrix: UserMatrix,
) -> Dict[str, np.ndarray]:
    """Compute exact population normalized share vector summing to 1.0."""
    normalized_contributions = transform_user_normalized_share(user_matrix.matrix)
    exact_sum = np.sum(normalized_contributions, axis=0)
    total = float(np.sum(exact_sum))
    exact_share = exact_sum / total if total > 0 else np.ones(9) / 9.0
    return {
        "exact_sum": exact_sum,
        "exact_share": exact_share,
    }


def compute_ground_truth_interaction_volume(
    user_matrix: UserMatrix,
    clip_norm: float = 20.0,
) -> Dict[str, np.ndarray]:
    """Compute exact raw volume, clipped exact volume, and clipping bias."""
    raw_exact = np.sum(user_matrix.matrix, axis=0)
    clipped_contributions = transform_interaction_volume_l2_clip(
        user_matrix.matrix,
        clip_norm=clip_norm,
    )
    clipped_exact = np.sum(clipped_contributions, axis=0)
    clipping_bias = clipped_exact - raw_exact
    return {
        "raw_exact": raw_exact,
        "clipped_exact": clipped_exact,
        "clipping_bias": clipping_bias,
    }
