"""Base classes and interfaces for DP mechanisms."""

from abc import ABC, abstractmethod
from typing import Optional
import numpy as np


class DPMechanism(ABC):
    """Abstract Differential Privacy Mechanism for vector release."""

    def __init__(
        self,
        epsilon: float,
        delta: float,
        sensitivity: float,
        noise_scale: float,
        mechanism_name: str,
    ):
        self.epsilon = epsilon
        self.delta = delta
        self.sensitivity = sensitivity
        self.noise_scale = noise_scale
        self.mechanism_name = mechanism_name

    @abstractmethod
    def add_noise(
        self,
        clean_vector: np.ndarray,
        rng: Optional[np.random.Generator] = None,
    ) -> np.ndarray:
        """Add calibrated noise to the clean summary vector."""
        pass

    def __repr__(self) -> str:
        return (
            f"{self.mechanism_name}Mechanism("
            f"eps={self.epsilon}, delta={self.delta}, "
            f"sens={self.sensitivity}, scale={self.noise_scale:.6f})"
        )
