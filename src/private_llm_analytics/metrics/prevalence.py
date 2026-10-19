"""User Prevalence Metric implementation in v0.1.1."""

from typing import Any, Dict, Optional
import numpy as np

from private_llm_analytics.aggregation.user_matrix import UserMatrix
from private_llm_analytics.contribution.binary import transform_user_prevalence
from private_llm_analytics.contribution.sensitivity import get_prevalence_sensitivity
from private_llm_analytics.metrics.base import BaseMetric


from private_llm_analytics.privacy.exceptions import PrivacyConfigurationError


class UserPrevalenceMetric(BaseMetric):
    """Answers: How many employees used a given Provider for a given Task in the window?"""

    def __init__(
        self,
        name: str = "user_prevalence",
        max_cells_per_user: int = 9,
    ):
        if max_cells_per_user != 9:
            raise ValueError(
                f"User Prevalence strictly requires max_cells_per_user == 9 in v0.1.2, got {max_cells_per_user}."
            )
        super().__init__(name=name, metric_type="prevalence")
        self.max_cells_per_user = max_cells_per_user

    def build_user_contributions(self, user_matrix: UserMatrix) -> np.ndarray:
        return transform_user_prevalence(
            user_matrix.matrix,
            max_cells_per_user=self.max_cells_per_user,
        )

    def compute_exact(self, user_matrix: UserMatrix) -> np.ndarray:
        """Compute exact prevalence count histogram."""
        contributions = self.build_user_contributions(user_matrix)
        return np.sum(contributions, axis=0)

    def compute_exact_rate(self, user_matrix: UserMatrix) -> np.ndarray:
        """Compute exact prevalence rate R^{prev} = H^{prev} / N_public."""
        if user_matrix.public_population_size is None or user_matrix.public_population_size <= 0:
            raise PrivacyConfigurationError(
                "User Prevalence strictly requires public_population_size in fixed_public_population mode; "
                "fallback to active users is prohibited."
            )
        h_exact = self.compute_exact(user_matrix)
        return h_exact / user_matrix.public_population_size

    def get_sensitivity(self, mechanism_type: str) -> float:
        l2_sens, l1_sens = get_prevalence_sensitivity(self.max_cells_per_user)
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
        """Post-process DP counts into clamped counts and prevalence rates using public denominator."""
        if user_matrix.public_population_size is None or user_matrix.public_population_size <= 0:
            raise PrivacyConfigurationError(
                "User Prevalence strictly requires public_population_size in fixed_public_population mode; "
                "fallback to active users is prohibited."
            )
        denominator = user_matrix.public_population_size

        clamped_counts = np.maximum(0.0, dp_vector)
        prevalence_rates = np.clip(clamped_counts / denominator, 0.0, 1.0)

        cell_rates = {name: float(prevalence_rates[i]) for i, name in enumerate(user_matrix.cell_names)}
        cell_counts = {name: float(clamped_counts[i]) for i, name in enumerate(user_matrix.cell_names)}

        return {
            "dp_counts": cell_counts,
            "prevalence_rates": cell_rates,
            "clamped_counts_array": clamped_counts,
            "prevalence_rates_array": prevalence_rates,
            "denominator_used": denominator,
        }

