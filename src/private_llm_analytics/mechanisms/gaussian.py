"""Gaussian Differential Privacy Mechanism with Analytic Calibration."""

import math
from typing import Optional
import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm

from private_llm_analytics.mechanisms.base import DPMechanism


def gaussian_privacy_loss_tail(sigma: float, epsilon: float, sensitivity: float) -> float:
    """Evaluate the exact Gaussian privacy loss tail delta(sigma; epsilon, Delta)

    From Balle & Wang (ICML 2018), Theorem 1:
      delta(sigma) = Phi( s/2 - epsilon/s ) - exp(epsilon) * Phi( -s/2 - epsilon/s )
    where s = Delta / sigma, and Phi is the standard normal CDF.
    """
    if sigma <= 0:
        return 1.0
    s = sensitivity / sigma
    v1 = s / 2.0 - epsilon / s
    v2 = -s / 2.0 - epsilon / s
    term1 = float(norm.cdf(v1))
    term2 = math.exp(epsilon) * float(norm.cdf(v2))
    return term1 - term2


def calibrate_analytic_gaussian(
    epsilon: float,
    delta: float,
    l2_sensitivity: float,
) -> float:
    """Compute the exact optimal sigma for (epsilon, delta)-DP under L2 sensitivity.

    Uses the Analytic Gaussian Mechanism (Balle & Wang, ICML 2018) via exact 1D root finding
    on the privacy loss distribution CDF.
    """
    if epsilon <= 0:
        raise ValueError(f"epsilon must be > 0, got {epsilon}")
    if delta <= 0 or delta >= 1.0:
        raise ValueError(f"delta must be in (0, 1), got {delta}")
    if l2_sensitivity <= 0:
        raise ValueError(f"l2_sensitivity must be > 0, got {l2_sensitivity}")

    # Bracket search bounds
    # Lower bound: sigma where privacy loss is high
    low = 1e-5 * l2_sensitivity
    # Upper bound: classic bound * 2
    classic_upper = (l2_sensitivity * math.sqrt(2.0 * math.log(1.25 / delta))) / min(epsilon, 1.0)
    high = max(classic_upper * 2.5, 100.0 * l2_sensitivity)

    def objective(sig: float) -> float:
        return gaussian_privacy_loss_tail(sig, epsilon, l2_sensitivity) - delta

    sigma = brentq(objective, low, high, xtol=1e-12, rtol=1e-12)
    return float(sigma)


def calibrate_classical_gaussian(
    epsilon: float,
    delta: float,
    l2_sensitivity: float,
) -> float:
    """Compute Gaussian sigma using the classical approximation:

    sigma = Delta_2 * sqrt(2 * ln(1.25 / delta)) / epsilon
    Note: formally valid only for epsilon in (0, 1].
    """
    if epsilon <= 0:
        raise ValueError(f"epsilon must be > 0, got {epsilon}")
    if delta <= 0 or delta >= 1.0:
        raise ValueError(f"delta must be in (0, 1), got {delta}")
    if l2_sensitivity <= 0:
        raise ValueError(f"l2_sensitivity must be > 0, got {l2_sensitivity}")

    return float(l2_sensitivity * math.sqrt(2.0 * math.log(1.25 / delta)) / epsilon)


class GaussianMechanism(DPMechanism):
    """Gaussian DP Mechanism for L2-bounded vector queries."""

    def __init__(
        self,
        epsilon: float,
        delta: float,
        sensitivity: float,
        method: str = "analytic",
    ):
        if method == "analytic":
            noise_scale = calibrate_analytic_gaussian(epsilon, delta, sensitivity)
        elif method == "classical":
            noise_scale = calibrate_classical_gaussian(epsilon, delta, sensitivity)
        else:
            raise ValueError(f"Unknown calibration method: '{method}'")

        super().__init__(
            epsilon=epsilon,
            delta=delta,
            sensitivity=sensitivity,
            noise_scale=noise_scale,
            mechanism_name="Gaussian",
        )
        self.calibration_method = method

    def add_noise(
        self,
        clean_vector: np.ndarray,
        rng: Optional[np.random.Generator] = None,
    ) -> np.ndarray:
        generator = rng or np.random.default_rng()
        noise = generator.normal(loc=0.0, scale=self.noise_scale, size=clean_vector.shape)
        return clean_vector + noise
