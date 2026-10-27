"""Success Rate metric in v0.6."""

from __future__ import annotations
from typing import Any, Dict, List, Optional
import numpy as np

from private_llm_analytics.aggregation.user_matrix import UserMatrix
from private_llm_analytics.mechanisms.base import DPMechanism
from private_llm_analytics.metrics.base import BaseMetric, DPReleaseResult
from private_llm_analytics.statistics.success_rate import SuccessRateProfile


class SuccessRateMetric(BaseMetric):
    """Metric wrapper over the authoritative SuccessRateProfile."""

    def __init__(
        self,
        name: str = "success_rate",
        clip_norm_rate: float = 10.0,
        min_noisy_denominator: float = 5.0,
    ):
        super().__init__(name=name, metric_type="success_rate")
        self.clip_norm_rate = float(clip_norm_rate)
        self.min_noisy_denominator = float(min_noisy_denominator)
        self.profile = SuccessRateProfile(
            clip_norm_rate=self.clip_norm_rate,
            min_noisy_denominator=self.min_noisy_denominator,
        )

    def compute_exact(self, user_matrix: UserMatrix) -> np.ndarray:
        return self.profile.compute_exact(user_matrix)

    def build_user_contributions(self, user_matrix: UserMatrix) -> np.ndarray:
        return self.profile.build_user_contributions(user_matrix)

    def get_sensitivity(self, mechanism_type: str = "gaussian") -> float:
        return self.profile.get_sensitivity(mechanism_type)

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
        # User joint contributions
        joint_contributions = self.build_user_contributions(user_matrix)
        joint_bounded_sum = np.sum(joint_contributions, axis=0) # shape (18,)
        noisy_joint = mechanism.add_noise(joint_bounded_sum, rng=rng)
        post_processed = self.post_process(noisy_joint, user_matrix)

        return DPReleaseResult(
            metric_name=self.name,
            dp_vector=noisy_joint,
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
