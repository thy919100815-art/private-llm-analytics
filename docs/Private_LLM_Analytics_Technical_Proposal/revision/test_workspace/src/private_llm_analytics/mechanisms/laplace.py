"""Laplace Differential Privacy Mechanism for L1 sensitivity vector queries."""

from typing import Optional
import numpy as np

from private_llm_analytics.mechanisms.base import DPMechanism


def calibrate_laplace(epsilon: float, l1_sensitivity: float) -> float:
    """Compute scale b = Delta_1 / epsilon for pure epsilon-DP."""
    if epsilon <= 0:
        raise ValueError(f"epsilon must be > 0, got {epsilon}")
    if l1_sensitivity <= 0:
        raise ValueError(f"l1_sensitivity must be > 0, got {l1_sensitivity}")
    return float(l1_sensitivity / epsilon)


class LaplaceMechanism(DPMechanism):
    """Laplace DP Mechanism satisfying pure epsilon-DP."""

    def __init__(
        self,
        epsilon: float,
        sensitivity: float,
        delta: float = 0.0,
    ):
        noise_scale = calibrate_laplace(epsilon, sensitivity)
        super().__init__(
            epsilon=epsilon,
            delta=delta,
            sensitivity=sensitivity,
            noise_scale=noise_scale,
            mechanism_name="Laplace",
        )

    def add_noise(
        self,
        clean_vector: np.ndarray,
        rng: Optional[np.random.Generator] = None,
    ) -> np.ndarray:
        generator = rng or np.random.default_rng()
        noise = generator.laplace(loc=0.0, scale=self.noise_scale, size=clean_vector.shape)
        return clean_vector + noise
