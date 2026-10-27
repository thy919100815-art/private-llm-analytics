"""User-Normalized Share Statistic Profile (v0.6)."""

from typing import Any, Dict, List, Optional
import numpy as np

from private_llm_analytics.aggregation.user_matrix import UserMatrix
from private_llm_analytics.contribution.normalization import transform_user_normalized_share
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


class UserNormalizedShareProfile(DPStatisticProfile):
    """Approved DP Release Profile for User-Normalized Share.

    Target:
      How does a typical employee's LLM usage distribute across Provider x Task cells?

    Mathematical Specification:
      - Unit: Complete user trajectory within observation window
      - Adjacency: Add/remove single user
      - Contribution: Per-user L1-normalized share vector x_u = m_u / ||m_u||_1 on simplex
      - Norm bound: ||x_u||_1 <= 1.0, ||x_u||_2 <= 1.0
      - Sensitivity: Delta_2 <= 1.0
      - Mechanism: Analytic Gaussian Vector Mechanism
      - Post-processing: Nonnegative simplex renormalization (clamp to >=0, renormalize to sum 1.0)
    """

    statistic_id = "user_normalized_share"
    profile_id = "simplex_share_gaussian"
    profile_version = "1"

    def __init__(self, norm_bound: float = 1.0):
        if norm_bound is None or abs(float(norm_bound) - 1.0) > 1e-12:
            raise InvalidStatisticProfileConfiguration(
                f"User-Normalized Share profile strictly requires norm_bound == 1.0, got {norm_bound}."
            )
        self.norm_bound = 1.0

    def metadata(self) -> ProfileMetadata:
        return ProfileMetadata(
            statistic_id=self.statistic_id,
            profile_id=self.profile_id,
            profile_version=self.profile_version,
            display_name="User-Normalized Share",
            description="Typical user workflow distribution on simplex with nonnegative renormalization",
            display_name_zh="用户归一化占比",
            description_zh="用户工作流在不同类别中的归一化分布（单纯形分布）",
            profile_name_zh="用户归一化贡献 + 高斯向量机制 + 单纯形归一化",
            privacy_unit="user",
            adjacency="add_remove_user",
            required_columns=list(REQUIRED_COLUMNS),
            contribution_definition="x_u = m_u / ||m_u||_1 for nonzero users, ||x_u||_1 <= 1",
            contribution_bound_description="norm_bound = 1.0",
            sensitivity_type="L2",
            sensitivity_description="Delta_2 = 1.0",
            mechanism_family="gaussian_vector",
            accounting_family="rdp",
            post_processing_description="Clamp coordinates to >= 0, renormalize vector to sum to 1.0",
            supports_continual=False,
            supports_planner=False,
            public_parameters={"norm_bound": self.norm_bound},
            configurable_parameters=[],
        )

    def required_columns(self) -> List[str]:
        return list(REQUIRED_COLUMNS)

    def contribution_bound(self) -> float:
        return self.norm_bound

    def sensitivity(self) -> float:
        return 1.0

    def build_user_contributions(self, user_matrix: UserMatrix) -> np.ndarray:
        return transform_user_normalized_share(user_matrix.matrix)

    def build_mechanism_plan(
        self,
        epsilon: float,
        delta: float,
        dimension: Optional[int] = None,
    ) -> MechanismPlan:
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
            dimension=dimension or 9,
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
        clamped = np.maximum(0.0, dp_vector)
        total_sum = float(np.sum(clamped))
        d_cells = len(user_matrix.cell_names)
        if total_sum > 0:
            share_vec = clamped / total_sum
        else:
            share_vec = np.ones(d_cells, dtype=np.float64) / float(d_cells)

        grid = share_vec.reshape((len(user_matrix.providers), len(user_matrix.tasks)))

        provider_dist = {
            p: float(grid[i, :].sum())
            for i, p in enumerate(user_matrix.providers)
        }
        task_dist = {
            t: float(grid[:, j].sum())
            for j, t in enumerate(user_matrix.tasks)
        }

        provider_task_dist: Dict[str, Dict[str, float]] = {}
        for i, p in enumerate(user_matrix.providers):
            p_total = float(grid[i, :].sum())
            if p_total > 0:
                p_cond = (grid[i, :] / p_total).tolist()
            else:
                p_cond = [1.0 / len(user_matrix.tasks)] * len(user_matrix.tasks)
            provider_task_dist[p] = {
                t: float(p_cond[j]) for j, t in enumerate(user_matrix.tasks)
            }

        cell_shares = {name: float(share_vec[i]) for i, name in enumerate(user_matrix.cell_names)}

        return {
            "cell_shares": cell_shares,
            "share_distribution_array": share_vec,
            "provider_distribution": provider_dist,
            "task_distribution": task_dist,
            "provider_conditional_task_distribution": provider_task_dist,
            "grid_3x3": grid,
            "grid": grid,
            "provider_task_distribution": provider_task_dist,
            "share_9d_array": share_vec,
            "raw_noisy_sum": total_sum,
        }
