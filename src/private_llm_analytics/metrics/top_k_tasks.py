"""Top-K Tasks metric wrapper in v0.8."""

from __future__ import annotations
from typing import Any, Dict, List, Optional
import numpy as np

from private_llm_analytics.aggregation.user_matrix import UserMatrix
from private_llm_analytics.mechanisms.base import DPMechanism
from private_llm_analytics.metrics.base import BaseMetric, DPReleaseResult
from private_llm_analytics.statistics.top_k_tasks import TopKTasksProfile


class TopKTasksMetric(BaseMetric):
    """Metric wrapper over the authoritative TopKTasksProfile."""

    def __init__(
        self,
        name: str = "top_k_tasks",
        k: int = 5,
        max_tasks_per_user: int = 5,
        threshold: Optional[float] = None,
    ):
        super().__init__(name=name, metric_type="top_k_tasks")
        self.k = int(k)
        self.max_tasks_per_user = int(max_tasks_per_user)
        self.threshold = float(threshold) if threshold is not None else None
        self.profile = TopKTasksProfile(
            k=self.k,
            max_tasks_per_user=self.max_tasks_per_user,
            threshold=self.threshold,
        )

    def compute_exact(self, user_matrix: UserMatrix) -> np.ndarray:
        return self.profile.compute_exact(user_matrix)

    def build_user_contributions(self, user_matrix: UserMatrix) -> np.ndarray:
        return self.profile.build_user_contributions(user_matrix)

    def get_sensitivity(self, mechanism_type: str = "gaussian") -> float:
        return self.profile.sensitivity()

    def post_process(
        self,
        dp_vector: np.ndarray,
        user_matrix: UserMatrix,
    ) -> Dict[str, Any]:
        return self.profile.post_process(dp_vector, user_matrix)

    def release_dp(
        self,
        user_matrix: UserMatrix,
        mechanism: DPMechanism,
        rng: Optional[np.random.Generator] = None,
    ) -> DPReleaseResult:
        contributions = self.build_user_contributions(user_matrix)
        bounded_sum = np.sum(contributions, axis=0)
        noisy_vec = mechanism.add_noise(bounded_sum, rng=rng)
        post_processed = self.post_process(noisy_vec, user_matrix)

        return DPReleaseResult(
            metric_name=self.name,
            dp_vector=noisy_vec,
            post_processed=post_processed,
            epsilon=mechanism.epsilon,
            delta=mechanism.delta,
            mechanism=mechanism.mechanism_name,
            noise_scale=mechanism.noise_scale,
            sensitivity=mechanism.sensitivity,
            cell_names=user_matrix.tasks,
            n_users=user_matrix.n_users,
            public_population_size=user_matrix.public_population_size,
        )
