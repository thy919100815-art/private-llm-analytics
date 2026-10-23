"""Interaction Volume Statistic Profile (v0.6)."""

from typing import Any, Dict, List, Optional
import numpy as np

from private_llm_analytics.aggregation.user_matrix import UserMatrix
from private_llm_analytics.contribution.clipping import transform_interaction_volume_l2_clip
from private_llm_analytics.ingestion.schema import REQUIRED_COLUMNS
from private_llm_analytics.mechanisms.gaussian import GaussianMechanism
from private_llm_analytics.privacy.exceptions import InvalidStatisticProfileConfiguration
from private_llm_analytics.privacy.rdp import (
    DEFAULT_RDP_ORDERS,
    compute_gaussian_rdp,
)
from private_llm_analytics.statistics.base import (
    DPStatisticProfile,
    MechanismPlan,
    PrivacyCostDescriptor,
    ProfileMetadata,
)


class InteractionVolumeProfile(DPStatisticProfile):
    """Approved DP Release Profile for Interaction Volume.

    Target:
      How is total enterprise LLM interaction volume distributed across Provider x Task cells?

    Mathematical Specification:
      - Unit: Complete user trajectory within observation window
      - Adjacency: Add/remove single user
      - Contribution: Per-user interaction vector m_u with user-level L2 clipping at bound C
        mbar_u = m_u * min(1, C / ||m_u||_2)
      - Bound: ||mbar_u||_2 <= C
      - Sensitivity: Delta_2 = C
      - Mechanism: Analytic Gaussian Vector Mechanism
      - Post-processing: Nonnegative count clamping (max(0, noisy_vector))
      - Continual Observation: Supported via binary canonical tree temporal profile
      - Planning: Supported via utility-aware small-n adaptive planner
    """

    statistic_id = "interaction_volume"
    profile_id = "clipped_volume_gaussian"
    profile_version = "1"

    def __init__(self, clip_norm: float = 20.0):
        if clip_norm is None or float(clip_norm) <= 0:
            raise InvalidStatisticProfileConfiguration(
                f"Interaction Volume profile requires clip_norm > 0, got {clip_norm}."
            )
        self.clip_norm = float(clip_norm)

    def metadata(self) -> ProfileMetadata:
        return ProfileMetadata(
            statistic_id=self.statistic_id,
            profile_id=self.profile_id,
            profile_version=self.profile_version,
            display_name="Interaction Volume",
            description="Total count of interactions with user-level L2 clipping and Gaussian noise",
            display_name_zh="交互量",
            description_zh="交互总量（按用户日/窗口进行 L2 贡献约束）",
            profile_name_zh="用户级 L2 裁剪 + 高斯向量机制",
            privacy_unit="user",
            adjacency="add_remove_user",
            required_columns=list(REQUIRED_COLUMNS),
            contribution_definition="mbar_u = m_u * min(1, C / ||m_u||_2)",
            contribution_bound_description=f"clip_norm (C) = {self.clip_norm}",
            sensitivity_type="L2",
            sensitivity_description=f"Delta_2 = C = {self.clip_norm}",
            mechanism_family="gaussian_vector",
            accounting_family="rdp",
            post_processing_description="Clamp negative noisy counts to zero",
            supports_continual=True,
            supports_planner=True,
            public_parameters={"clip_norm": self.clip_norm},
            configurable_parameters=["clip_norm"],
        )

    def required_columns(self) -> List[str]:
        return list(REQUIRED_COLUMNS)

    def contribution_bound(self) -> float:
        return self.clip_norm

    def sensitivity(self) -> float:
        return self.clip_norm

    def build_user_contributions(self, user_matrix: UserMatrix) -> np.ndarray:
        return transform_interaction_volume_l2_clip(
            user_matrix.matrix,
            clip_norm=self.clip_norm,
        )

    def build_mechanism_plan(self, epsilon: float, delta: float) -> MechanismPlan:
        sens = self.sensitivity()
        mech = GaussianMechanism(
            epsilon=epsilon,
            delta=delta,
            sensitivity=sens,
            method="analytic",
        )
        return MechanismPlan(
            mechanism_name="gaussian",
            sensitivity=sens,
            noise_scale=float(mech.noise_scale),
            epsilon=float(epsilon),
            delta=float(delta),
            dimension=9,
        )

    def privacy_cost_descriptor(
        self,
        plan: MechanismPlan,
        orders: Optional[List[float]] = None,
    ) -> PrivacyCostDescriptor:
        eff_orders = orders or DEFAULT_RDP_ORDERS
        costs = compute_gaussian_rdp(
            sigma=plan.noise_scale,
            sensitivity=plan.sensitivity,
            orders=eff_orders,
        )
        return PrivacyCostDescriptor(
            accounting_family="rdp",
            orders=list(eff_orders),
            costs=[float(c) for c in costs],
            mechanism_profile_id=self.profile_id,
            sensitivity=plan.sensitivity,
            noise_scale=plan.noise_scale,
            epsilon=plan.epsilon,
            delta=plan.delta,
        )

    def execute_mechanism(
        self,
        bounded_aggregate: np.ndarray,
        plan: MechanismPlan,
        rng: Optional[np.random.Generator] = None,
    ) -> np.ndarray:
        mech = GaussianMechanism(
            epsilon=plan.epsilon,
            delta=plan.delta,
            sensitivity=plan.sensitivity,
            method="analytic",
        )
        return mech.add_noise(bounded_aggregate, rng=rng)

    def post_process(
        self,
        dp_vector: np.ndarray,
        user_matrix: UserMatrix,
    ) -> Dict[str, Any]:
        clamped_volume = np.maximum(0.0, dp_vector)
        cell_volumes = {name: float(clamped_volume[i]) for i, name in enumerate(user_matrix.cell_names)}
        total_vol = float(np.sum(clamped_volume))

        return {
            "cell_volumes": cell_volumes,
            "clamped_volume_array": clamped_volume,
            "total_estimated_volume": total_vol,
        }

    # Error decomposition for experiment / evaluation code only
    def compute_raw_exact(self, user_matrix: UserMatrix) -> np.ndarray:
        return np.sum(user_matrix.matrix, axis=0)

    def compute_clipped_exact(self, user_matrix: UserMatrix) -> np.ndarray:
        contributions = self.build_user_contributions(user_matrix)
        return np.sum(contributions, axis=0)

    def decompose_error(
        self,
        dp_vector: np.ndarray,
        user_matrix: UserMatrix,
    ) -> Dict[str, np.ndarray]:
        v_raw = self.compute_raw_exact(user_matrix)
        v_clip = self.compute_clipped_exact(user_matrix)
        v_dp = dp_vector

        return {
            "raw_exact": v_raw,
            "clipped_exact": v_clip,
            "dp_volume": v_dp,
            "clipping_bias": v_clip - v_raw,
            "dp_noise_error": v_dp - v_clip,
            "total_error": v_dp - v_raw,
        }
