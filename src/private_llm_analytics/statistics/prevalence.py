"""User Prevalence Statistic Profile (v0.6)."""

from typing import Any, Dict, List, Optional
import numpy as np

from private_llm_analytics.aggregation.user_matrix import UserMatrix
from private_llm_analytics.contribution.binary import transform_user_prevalence
from private_llm_analytics.ingestion.schema import REQUIRED_COLUMNS
from private_llm_analytics.mechanisms.gaussian import GaussianMechanism
from private_llm_analytics.privacy.exceptions import (
    InvalidStatisticProfileConfiguration,
    PrivacyConfigurationError,
)
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


class UserPrevalenceProfile(DPStatisticProfile):
    """Approved DP Release Profile for User Prevalence.

    Target:
      What fraction of eligible enterprise users adopted each Provider x Task cell?

    Mathematical Specification:
      - Unit: Complete user trajectory within observation window
      - Adjacency: Add/remove single user
      - Contribution: Binary participation b_{u,j} = 1[m_{u,j} > 0] in {0, 1}^9
      - Sensitivity: Delta_2 <= sqrt(k) = 3.0 (for k=9 cells)
      - Mechanism: Analytic Gaussian Vector Mechanism
      - Post-processing: Clamp counts >= 0, divide strictly by public population size N_public,
        and clamp rates to [0, 1].
    """

    statistic_id = "user_prevalence"
    profile_id = "binary_prevalence_gaussian"
    profile_version = "1"

    def __init__(self, max_cells_per_user: int = 9):
        if max_cells_per_user != 9:
            raise InvalidStatisticProfileConfiguration(
                f"User Prevalence profile strictly requires max_cells_per_user == 9, got {max_cells_per_user}."
            )
        self.max_cells_per_user = int(max_cells_per_user)

    def metadata(self) -> ProfileMetadata:
        return ProfileMetadata(
            statistic_id=self.statistic_id,
            profile_id=self.profile_id,
            profile_version=self.profile_version,
            display_name="User Prevalence",
            description="Workforce capability adoption proportion per cell using public population denominator",
            display_name_zh="用户使用率",
            description_zh="使用该模型能力的用户占比（基于公开基准人数分母）",
            profile_name_zh="用户参与二值化 + 高斯直方图机制",
            privacy_unit="user",
            adjacency="add_remove_user",
            required_columns=list(REQUIRED_COLUMNS),
            contribution_definition="b_{u,j} = 1[m_{u,j} > 0] in {0, 1}^9",
            contribution_bound_description="max_cells_per_user = 9",
            sensitivity_type="L2",
            sensitivity_description="Delta_2 = sqrt(9) = 3.0",
            mechanism_family="gaussian_vector",
            accounting_family="rdp",
            post_processing_description="Clamp noisy counts to >= 0, divide by public population size, clamp rates to [0, 1]",
            supports_continual=False,
            supports_planner=False,
            public_parameters={"max_cells_per_user": self.max_cells_per_user},
            configurable_parameters=[],
        )

    def required_columns(self) -> List[str]:
        return list(REQUIRED_COLUMNS)

    def contribution_bound(self) -> float:
        return float(self.max_cells_per_user)

    def sensitivity(self) -> float:
        return float(np.sqrt(self.max_cells_per_user))

    def build_user_contributions(self, user_matrix: UserMatrix) -> np.ndarray:
        return transform_user_prevalence(
            user_matrix.matrix,
            max_cells_per_user=self.max_cells_per_user,
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
