"""Base class and result containers for metrics in v0.1.2."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List, Optional
import numpy as np

if TYPE_CHECKING:
    from private_llm_analytics.aggregation.user_matrix import UserMatrix

from private_llm_analytics.mechanisms.base import DPMechanism


@dataclass
class DPReleaseResult:
    """Encapsulates the output of a DP release."""
    metric_name: str
    dp_vector: np.ndarray             # 9-dim raw noisy aggregate
    post_processed: Dict[str, Any]    # derived public statistics
    epsilon: float
    delta: float
    mechanism: str
    noise_scale: float
    sensitivity: float
    cell_names: List[str]
    n_users: int                      # Active user count in sample (internal/audit only)
    public_population_size: Optional[int] = None  # Fixed public eligible population denominator


class BaseMetric(ABC):
    """Abstract base class for analytics metrics."""

    def __init__(self, name: str, metric_type: str):
        self.name = name
        self.metric_type = metric_type

    @abstractmethod
    def compute_exact(self, user_matrix: UserMatrix) -> np.ndarray:
        """Compute the clean, unnoised aggregate statistic (internal/eval only)."""
        pass

    @abstractmethod
    def build_user_contributions(self, user_matrix: UserMatrix) -> np.ndarray:
        """Compute user-level bounded contribution vectors (shape: N x 9)."""
        pass

    @abstractmethod
    def get_sensitivity(self, mechanism_type: str) -> float:
        """Return the appropriate sensitivity (L2 for Gaussian, L1 for Laplace)."""
        pass

    @abstractmethod
    def post_process(
        self,
        dp_vector: np.ndarray,
        user_matrix: UserMatrix,
    ) -> Dict[str, Any]:
        """Post-process noisy vector into interpretable statistics without private data access."""
        pass

    def release_dp(
        self,
        user_matrix: UserMatrix,
        mechanism: DPMechanism,
        rng: Optional[np.random.Generator] = None,
    ) -> DPReleaseResult:
        """Execute user aggregation -> contribution bounding -> DP mechanism -> post-processing."""
        # 1. Transform user matrix into bounded user contributions
        contributions = self.build_user_contributions(user_matrix)

        # 2. Sum user contributions to form the true bounded aggregate
        bounded_aggregate = np.sum(contributions, axis=0)

        # 3. Add calibrated noise via DP mechanism
        noisy_aggregate = mechanism.add_noise(bounded_aggregate, rng=rng)

        # 4. Post-process public statistics
        post_processed = self.post_process(noisy_aggregate, user_matrix)

        return DPReleaseResult(
            metric_name=self.name,
            dp_vector=noisy_aggregate,
            post_processed=post_processed,
            epsilon=mechanism.epsilon,
            delta=mechanism.delta,
            mechanism=mechanism.mechanism_name,
            noise_scale=mechanism.noise_scale,
            sensitivity=mechanism.sensitivity,
            cell_names=user_matrix.cell_names,
            n_users=user_matrix.n_users,
            public_population_size=user_matrix.public_population_size,
        )
