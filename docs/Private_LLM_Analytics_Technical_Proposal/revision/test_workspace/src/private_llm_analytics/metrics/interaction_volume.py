"""Interaction Volume Metric implementation with error decomposition."""

from typing import Any, Dict, Tuple
import numpy as np

from private_llm_analytics.aggregation.user_matrix import UserMatrix
from private_llm_analytics.contribution.clipping import transform_interaction_volume_l2_clip
from private_llm_analytics.contribution.sensitivity import get_interaction_volume_sensitivity
from private_llm_analytics.metrics.base import BaseMetric


class InteractionVolumeMetric(BaseMetric):
    """Answers: Where is enterprise total interaction volume distributed?"""

    def __init__(
        self,
        name: str = "interaction_volume",
        clip_norm: float = 20.0,
    ):
        super().__init__(name=name, metric_type="volume")
        self.clip_norm = clip_norm

    def build_user_contributions(self, user_matrix: UserMatrix) -> np.ndarray:
        return transform_interaction_volume_l2_clip(
            user_matrix.matrix,
            clip_norm=self.clip_norm,
        )

    def compute_raw_exact(self, user_matrix: UserMatrix) -> np.ndarray:
        """Compute unclipped raw interaction volume sum_u m_u."""
        return np.sum(user_matrix.matrix, axis=0)

    def compute_clipped_exact(self, user_matrix: UserMatrix) -> np.ndarray:
        """Compute clipped exact population aggregate sum_u \bar{m}_u."""
        contributions = self.build_user_contributions(user_matrix)
        return np.sum(contributions, axis=0)

    def compute_exact(self, user_matrix: UserMatrix) -> np.ndarray:
        """Default exact computation returns clipped exact for sensitivity matching."""
        return self.compute_clipped_exact(user_matrix)

    def get_sensitivity(self, mechanism_type: str) -> float:
        l2_sens, l1_sens = get_interaction_volume_sensitivity(
            clip_norm=self.clip_norm,
            d_cells=9,
        )
        m_lower = mechanism_type.lower()
        if "gaussian" in m_lower:
            return l2_sens
        elif "laplace" in m_lower:
            return l1_sens
        else:
            raise ValueError(f"Unsupported mechanism type: {mechanism_type}")

    def decompose_error(
        self,
        dp_vector: np.ndarray,
        user_matrix: UserMatrix,
    ) -> Dict[str, np.ndarray]:
        """Decompose total error into clipping bias and DP noise error."""
        v_raw = self.compute_raw_exact(user_matrix)
        v_clip = self.compute_clipped_exact(user_matrix)
        v_dp = dp_vector

        clipping_bias = v_clip - v_raw
        dp_noise_error = v_dp - v_clip
        total_error = v_dp - v_raw

        return {
            "raw_exact": v_raw,
            "clipped_exact": v_clip,
            "dp_volume": v_dp,
            "clipping_bias": clipping_bias,
            "dp_noise_error": dp_noise_error,
            "total_error": total_error,
        }

    def post_process(
        self,
        dp_vector: np.ndarray,
        user_matrix: UserMatrix,
    ) -> Dict[str, Any]:
        """Post-process noisy volume vector (clamp negative noise to zero)."""
        clamped_volume = np.maximum(0.0, dp_vector)
        cell_volumes = {name: float(clamped_volume[i]) for i, name in enumerate(user_matrix.cell_names)}
        total_vol = float(np.sum(clamped_volume))

        return {
            "cell_volumes": cell_volumes,
            "clamped_volume_array": clamped_volume,
            "total_estimated_volume": total_vol,
        }
