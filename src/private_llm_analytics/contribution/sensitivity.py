"""Global sensitivity calculation under add/remove user adjacency."""

import math
from typing import Literal, Tuple


def get_prevalence_sensitivity(max_cells_per_user: int = 9) -> Tuple[float, float]:
    """Return (l2_sensitivity, l1_sensitivity) for User Prevalence."""
    l2_sens = math.sqrt(max_cells_per_user)
    l1_sens = float(max_cells_per_user)
    return l2_sens, l1_sens


def get_normalized_share_sensitivity(norm_bound: float = 1.0) -> Tuple[float, float]:
    """Return (l2_sensitivity, l1_sensitivity) for User-Normalized Share."""
    l2_sens = float(norm_bound)
    l1_sens = float(norm_bound)
    return l2_sens, l1_sens


def get_interaction_volume_sensitivity(clip_norm: float = 20.0, d_cells: int = 9) -> Tuple[float, float]:
    """Return (l2_sensitivity, l1_sensitivity) for Interaction Volume under L2 clipping.

    Under add/remove user adjacency:
      - L2 sensitivity = C
      - L1 sensitivity = sqrt(d) * C (by Cauchy-Schwarz on R^d, sharp when coordinates are equal)
    """
    l2_sens = float(clip_norm)
    l1_sens = float(math.sqrt(d_cells) * clip_norm)
    return l2_sens, l1_sens


def get_metric_sensitivities(
    metric_type: Literal["prevalence", "normalized_share", "volume"],
    **kwargs,
) -> Tuple[float, float]:
    """Dispatch sensitivity calculation for given metric type.

    Returns:
        (l2_sensitivity, l1_sensitivity)
    """
    if metric_type == "prevalence":
        k = kwargs.get("max_cells_per_user", 9)
        return get_prevalence_sensitivity(max_cells_per_user=k)
    elif metric_type == "normalized_share":
        bound = kwargs.get("norm_bound", 1.0)
        return get_normalized_share_sensitivity(norm_bound=bound)
    elif metric_type == "volume":
        c = kwargs.get("clip_norm", 20.0)
        d = kwargs.get("d_cells", 9)
        return get_interaction_volume_sensitivity(clip_norm=c, d_cells=d)
    else:
        raise ValueError(f"Unknown metric type: {metric_type}")
