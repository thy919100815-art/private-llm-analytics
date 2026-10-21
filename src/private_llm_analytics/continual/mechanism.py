"""Tree Gaussian Noise Calibration and RDP Level Vector Computation (v0.4).

Implements:
- calibrate_tree_gaussian_sigma: Binary search calibration for equal-sigma tree levels.
- compute_level_rdp_vector: Exact per-level RDP vector for user-level Delta_2 = C_epoch.
- evaluate_rdp_epsilon: Conversion from RDP vector to (epsilon, delta) upper bound.
"""

from decimal import Decimal
import math
from typing import Dict, List, Optional, Sequence, Tuple
import numpy as np

from private_llm_analytics.continual.models import DEFAULT_CONTINUAL_RDP_ORDERS
from private_llm_analytics.continual.tree import BinaryTimeTree
from private_llm_analytics.privacy.exceptions import (
    PrivacyConfigurationError,
    ProductionSafetyConfigurationError,
)
from private_llm_analytics.privacy.rdp import (
    rdp_to_epsilon_delta,
    to_decimal,
)


def evaluate_rdp_epsilon(
    rdp_vector: Dict[float, float],
    delta: float,
) -> float:
    """Compute differential privacy epsilon at given delta using Renyi DP upper bound."""
    if not (0 < delta < 1):
        raise PrivacyConfigurationError(f"delta must be in (0, 1), got {delta}")

    min_eps = float("inf")
    log_inv_delta = math.log(1.0 / delta)

    for alpha_val, rdp_val in rdp_vector.items():
        alpha = float(alpha_val)
        rdp = float(rdp_val)
        if alpha <= 1.0:
            continue
        eps = rdp + log_inv_delta / (alpha - 1.0)
        if eps < min_eps:
            min_eps = eps

    return min_eps


def compute_level_rdp_vector(
    epoch_l12_bound: float,
    sigma: float,
    rdp_orders: Optional[Sequence[float]] = None,
) -> Dict[float, float]:
    """Compute exact RDP vector for a single tree level with L2 sensitivity C_epoch."""
    if epoch_l12_bound <= 0:
        raise PrivacyConfigurationError(f"epoch_l12_bound must be > 0, got {epoch_l12_bound}")
    if sigma <= 0:
        raise PrivacyConfigurationError(f"sigma must be > 0, got {sigma}")

    orders = rdp_orders or DEFAULT_CONTINUAL_RDP_ORDERS
    sens_sq = float(epoch_l12_bound) ** 2
    two_sig_sq = 2.0 * (float(sigma) ** 2)

    rdp_vec: Dict[float, float] = {}
    for alpha in orders:
        a = float(alpha)
        if a <= 1.0:
            continue
        rdp_vec[a] = (a * sens_sq) / two_sig_sq

    return rdp_vec


def calibrate_tree_gaussian_sigma(
    epoch_days: int,
    epoch_l12_bound: float,
    target_tree_epsilon: float,
    calibration_delta: float = 1e-6,
    rdp_orders: Optional[Sequence[float]] = None,
    tolerance: float = 1e-6,
    max_iterations: int = 100,
) -> float:
    """Calibrate tree Gaussian sigma across all tree levels via binary search (Section 40).

    In v0.4, all L levels share equal sigma:
      RDP_tree(alpha) = L * (alpha * C_epoch^2) / (2 * sigma^2)
    Binary search finds sigma such that the effective composed epsilon under
    calibration_delta satisfies:
      effective_epsilon <= target_tree_epsilon
    and |effective_epsilon - target_tree_epsilon| <= tolerance.
    """
    if epoch_days <= 0:
        raise PrivacyConfigurationError(f"epoch_days must be > 0, got {epoch_days}")
    if epoch_l12_bound <= 0:
        raise PrivacyConfigurationError(f"epoch_l12_bound must be > 0, got {epoch_l12_bound}")
    if target_tree_epsilon <= 0:
        raise PrivacyConfigurationError(f"target_tree_epsilon must be > 0, got {target_tree_epsilon}")
    if not (0 < calibration_delta < 1):
        raise PrivacyConfigurationError(f"calibration_delta must be in (0, 1), got {calibration_delta}")

    tree = BinaryTimeTree(epoch_days=epoch_days)
    num_levels = tree.num_levels
    orders = tuple(rdp_orders or DEFAULT_CONTINUAL_RDP_ORDERS)

    def _eval_sigma(sig: float) -> float:
        sens_sq = float(epoch_l12_bound) ** 2
        two_sig_sq = 2.0 * (sig ** 2)
        log_inv_delta = math.log(1.0 / calibration_delta)

        best_eps = float("inf")
        for alpha in orders:
            if alpha <= 1.0:
                continue
            # Total RDP across num_levels
            total_rdp = num_levels * (alpha * sens_sq) / two_sig_sq
            eps = total_rdp + log_inv_delta / (alpha - 1.0)
            if eps < best_eps:
                best_eps = eps
        return best_eps

    # Initial brackets
    # Epsilon is strictly decreasing with sigma
    low_sigma = 1e-4
    high_sigma = 10.0

    # Expand high_sigma until eps(high_sigma) < target_tree_epsilon
    while _eval_sigma(high_sigma) > target_tree_epsilon:
        high_sigma *= 2.0
        if high_sigma > 1e9:
            raise RuntimeError(f"Failed to find upper bracket for sigma in tree calibration")

    # Expand low_sigma until eps(low_sigma) > target_tree_epsilon
    while _eval_sigma(low_sigma) < target_tree_epsilon:
        low_sigma /= 2.0
        if low_sigma < 1e-12:
            break

    # Binary search for sigma
    for _ in range(max_iterations):
        mid_sigma = (low_sigma + high_sigma) / 2.0
        cur_eps = _eval_sigma(mid_sigma)

        if abs(cur_eps - target_tree_epsilon) <= tolerance:
            # Conservative: if slightly exceeds target, nudge high
            if cur_eps > target_tree_epsilon:
                low_sigma = mid_sigma
            else:
                return mid_sigma
        elif cur_eps > target_tree_epsilon:
            # Need larger sigma to reduce epsilon
            low_sigma = mid_sigma
        else:
            # Smaller epsilon than target -> can use smaller sigma
            high_sigma = mid_sigma

    # Return high_sigma to ensure effective_epsilon <= target_tree_epsilon (conservative fail-safe)
    return high_sigma
