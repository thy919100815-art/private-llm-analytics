"""User-Normalized Share Metric implementation."""

from typing import Any, Dict
import numpy as np

from private_llm_analytics.aggregation.user_matrix import UserMatrix
from private_llm_analytics.contribution.normalization import transform_user_normalized_share
from private_llm_analytics.contribution.sensitivity import get_normalized_share_sensitivity
from private_llm_analytics.metrics.base import BaseMetric


class UserNormalizedShareMetric(BaseMetric):
    """Answers: How does a typical employee distribute AI usage across Provider x Task?"""

    def __init__(
        self,
        name: str = "user_normalized_share",
        norm_bound: float = 1.0,
    ):
        if norm_bound is None or abs(norm_bound - 1.0) > 1e-12:
            raise ValueError(
                f"User-Normalized Share strictly requires norm_bound == 1.0 in v0.1.2, got {norm_bound}."
            )

        super().__init__(name=name, metric_type="normalized_share")
        self.norm_bound = float(norm_bound)

    def build_user_contributions(self, user_matrix: UserMatrix) -> np.ndarray:
        return transform_user_normalized_share(user_matrix.matrix)

    def compute_exact(self, user_matrix: UserMatrix) -> np.ndarray:
        """Compute exact aggregate sum S = sum_u x_u."""
        contributions = self.build_user_contributions(user_matrix)
        return np.sum(contributions, axis=0)

    def compute_exact_mean(self, user_matrix: UserMatrix) -> np.ndarray:
        """Compute average user-normalized share vector (summing to 1.0 across population)."""
        s_exact = self.compute_exact(user_matrix)
        total_s = float(np.sum(s_exact))
        if total_s > 0:
            return s_exact / total_s
        return np.ones(9, dtype=np.float64) / 9.0

    def get_sensitivity(self, mechanism_type: str) -> float:
        l2_sens, l1_sens = get_normalized_share_sensitivity(self.norm_bound)
        m_lower = mechanism_type.lower()
        if "gaussian" in m_lower:
            return l2_sens
        elif "laplace" in m_lower:
            return l1_sens
        else:
            raise ValueError(f"Unsupported mechanism type: {mechanism_type}")

    def post_process(
        self,
        dp_vector: np.ndarray,
        user_matrix: UserMatrix,
    ) -> Dict[str, Any]:
        """Post-process noisy aggregate into simplex-normalized distributions.

        Derives:
          - overall Provider x Task share
          - provider-conditional task distributions
          - overall provider distribution
          - overall task distribution
        """
        # 1. Project to non-negative and normalize to simplex
        clamped = np.maximum(0.0, dp_vector)
        total_sum = float(np.sum(clamped))
        if total_sum > 0:
            share_9d = clamped / total_sum
        else:
            share_9d = np.ones(9, dtype=np.float64) / 9.0

        # Reshape to (n_providers, n_tasks) = (3, 3)
        grid_3x3 = share_9d.reshape((len(user_matrix.providers), len(user_matrix.tasks)))

        # Provider marginals
        provider_dist = {
            p: float(grid_3x3[i, :].sum())
            for i, p in enumerate(user_matrix.providers)
        }

        # Task marginals
        task_dist = {
            t: float(grid_3x3[:, j].sum())
            for j, t in enumerate(user_matrix.tasks)
        }

        # Provider-conditional task distributions: P(task | provider)
        provider_task_dist: Dict[str, Dict[str, float]] = {}
        for i, p in enumerate(user_matrix.providers):
            p_total = float(grid_3x3[i, :].sum())
            if p_total > 0:
                p_cond = (grid_3x3[i, :] / p_total).tolist()
            else:
                p_cond = [1.0 / len(user_matrix.tasks)] * len(user_matrix.tasks)
            provider_task_dist[p] = {
                t: float(p_cond[j]) for j, t in enumerate(user_matrix.tasks)
            }

        cell_shares = {name: float(share_9d[k]) for k, name in enumerate(user_matrix.cell_names)}

        return {
            "cell_shares": cell_shares,
            "share_distribution_array": share_9d,
            "provider_distribution": provider_dist,
            "task_distribution": task_dist,
            "provider_conditional_task_distribution": provider_task_dist,
            "grid_3x3": grid_3x3,
        }
