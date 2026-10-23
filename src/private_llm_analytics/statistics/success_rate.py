"""Bounded Success Rate Joint Gaussian Statistic Profile (v0.6)."""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from private_llm_analytics.aggregation.user_matrix import UserMatrix
from private_llm_analytics.ingestion.schema import COL_SUCCESS, REQUIRED_COLUMNS
from private_llm_analytics.mechanisms.gaussian import GaussianMechanism
from private_llm_analytics.privacy.exceptions import (
    InvalidStatisticProfileConfiguration,
    MissingRequiredStatisticColumn,
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


class SuccessRateProfile(DPStatisticProfile):
    """Approved DP Release Profile for Interaction Success Rate.

    Target:
      For each Provider x Task cell, what fraction of bounded interactions succeeded?

    Mathematical Specification:
      - Unit: Complete user trajectory within observation window
      - Adjacency: Add/remove single user
      - Inputs per user u and cell j:
          n_{u,j}: total interaction count
          s_{u,j}: successful interaction count, with 0 <= s_{u,j} <= n_{u,j}
      - Shared user-level clipping:
          a_u = min(1, C_rate / ||n_u||_2)  (where a_u = 1.0 if ||n_u||_2 == 0)
          nbar_u = a_u * n_u
          sbar_u = a_u * s_u
          Preserves 0 <= sbar_{u,j} <= nbar_{u,j} and ||nbar_u||_2 <= C_rate
      - Joint contribution vector:
          w_u = [sbar_u ; nbar_u] in R^18
          ||w_u||_2 = sqrt(||sbar_u||_2^2 + ||nbar_u||_2^2) <= sqrt(2) * C_rate
      - Sensitivity:
          Delta_2 = sqrt(2) * C_rate
      - Mechanism:
          Single 18-dimensional Joint Gaussian Vector Mechanism
      - Accounting:
          ONE joint RDP release charged with sensitivity Delta_2 = sqrt(2) * C_rate
      - Post-processing:
          total_dp = max(noisy_total, 0)
          success_dp = clip(noisy_success, 0, total_dp)
          If total_dp < min_noisy_denominator: rate is suppressed (None)
          Else: rate = clip(success_dp / total_dp, 0, 1)
    """

    statistic_id = "success_rate"
    profile_id = "bounded_success_rate_joint_gaussian"
    profile_version = "1"

    def __init__(
        self,
        clip_norm: Optional[float] = None,
        clip_norm_rate: Optional[float] = None,
        min_noisy_denominator: float = 1.0,
    ):
        eff_clip = clip_norm if clip_norm is not None else (clip_norm_rate if clip_norm_rate is not None else 20.0)
        if eff_clip is None or float(eff_clip) <= 0:
            raise InvalidStatisticProfileConfiguration(
                f"Success Rate profile requires clip_norm (C_rate) > 0, got {eff_clip}."
            )
        if min_noisy_denominator is None or float(min_noisy_denominator) < 0:
            raise InvalidStatisticProfileConfiguration(
                f"Success Rate profile requires min_noisy_denominator >= 0, got {min_noisy_denominator}."
            )
        self.clip_norm = float(eff_clip)
        self.clip_norm_rate = float(eff_clip)
        self.min_noisy_denominator = float(min_noisy_denominator)

    def metadata(self) -> ProfileMetadata:
        return ProfileMetadata(
            statistic_id=self.statistic_id,
            profile_id=self.profile_id,
            profile_version=self.profile_version,
            display_name="Success Rate",
            description="Interaction success rate via joint success/total Gaussian release with DP-safe ratio post-processing",
            display_name_zh="成功率",
            description_zh="模型交互任务执行成功率（成功数与总数联合加噪，比率后处理）",
            profile_name_zh="共享用户级裁剪 + 成功数/总数联合高斯机制 + 比率后处理",
            privacy_unit="user",
            adjacency="add_remove_user",
            required_columns=list(REQUIRED_COLUMNS) + [COL_SUCCESS],
            contribution_definition="Shared user-level clipping: a_u = min(1, C_rate / ||n_u||_2), w_u = [a_u*s_u; a_u*n_u]",
            contribution_bound_description=f"C_rate = {self.clip_norm}, ||w_u||_2 <= sqrt(2)*C_rate",
            sensitivity_type="L2",
            sensitivity_description=f"Delta_2 = sqrt(2) * C_rate = {np.sqrt(2.0) * self.clip_norm:.4f}",
            mechanism_family="joint_gaussian_vector",
            accounting_family="rdp",
            post_processing_description="Clamp total >= 0, clip success to [0, total_dp], suppress if total_dp < threshold, else divide",
            supports_continual=False,
            supports_planner=False,
            public_parameters={
                "clip_norm": self.clip_norm,
                "min_noisy_denominator": self.min_noisy_denominator,
            },
            configurable_parameters=["clip_norm", "min_noisy_denominator"],
        )

    def required_columns(self) -> List[str]:
        return list(REQUIRED_COLUMNS) + [COL_SUCCESS]

    def contribution_bound(self) -> float:
        return self.clip_norm

    def sensitivity(self) -> float:
        # ||[sbar_u ; nbar_u]||_2 <= sqrt(||sbar_u||_2^2 + ||nbar_u||_2^2) <= sqrt(2) * C_rate
        return float(np.sqrt(2.0) * self.clip_norm)

    def build_user_contributions(self, user_matrix: UserMatrix) -> np.ndarray:
        """Compute 18-dimensional joint contribution vectors: [sbar_u ; nbar_u]."""
        if getattr(user_matrix, "success_matrix", None) is None:
            raise MissingRequiredStatisticColumn(
                f"Statistic '{self.statistic_id}' requires interaction column '{COL_SUCCESS}' to be present."
            )

        n_matrix = user_matrix.matrix         # shape: (n_users, 9)
        s_matrix = user_matrix.success_matrix # shape: (n_users, 9)

        # Enforce mathematical prerequisite: 0 <= s_{u,j} <= n_{u,j}
        s_matrix_bounded = np.clip(s_matrix, 0.0, n_matrix)

        # Compute user L2 norm of total interaction vector ||n_u||_2
        n_norms = np.linalg.norm(n_matrix, ord=2, axis=1) # shape: (n_users,)

        # Compute shared user-level clipping factor a_u = min(1, C_rate / ||n_u||_2)
        # Avoid division by zero for inactive users
        scale_factors = np.ones(len(n_norms), dtype=np.float64)
        nonzero_mask = n_norms > 0.0
        scale_factors[nonzero_mask] = np.minimum(1.0, self.clip_norm / n_norms[nonzero_mask])

        # Apply identical shared factor to both success and total vectors
        # shape: (n_users, 9)
        sbar = s_matrix_bounded * scale_factors[:, np.newaxis]
        nbar = n_matrix * scale_factors[:, np.newaxis]

        # Concatenate horizontally to form 18-dimensional joint vectors: shape (n_users, 18)
        # Columns 0..8: sbar (success)
        # Columns 9..17: nbar (total)
        joint_contributions = np.hstack([sbar, nbar])
        return joint_contributions

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
            dimension=18,
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
        assert len(bounded_aggregate) == 18
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
        """Post-process 18-dim noisy joint vector into DP success rates."""
        assert len(dp_vector) == 18
        noisy_success = dp_vector[:9]
        noisy_total = dp_vector[9:18]

        # 1. Total counts must be >= 0
        total_dp = np.maximum(0.0, noisy_total)

        # 2. Success counts must be clamped to [0, total_dp]
        success_dp = np.clip(noisy_success, 0.0, total_dp)

        success_rates: Dict[str, Optional[float]] = {}
        cell_status: Dict[str, str] = {}
        dp_totals: Dict[str, float] = {}
        dp_successes: Dict[str, float] = {}
        suppressed_cells: List[str] = []

        for i, cell in enumerate(user_matrix.cell_names):
            t_val = float(total_dp[i])
            s_val = float(success_dp[i])
            dp_totals[cell] = t_val
            dp_successes[cell] = s_val

            if t_val < self.min_noisy_denominator:
                success_rates[cell] = None
                cell_status[cell] = "noisy denominator below reporting threshold"
                suppressed_cells.append(cell)
            else:
                rate = float(np.clip(s_val / t_val, 0.0, 1.0))
                success_rates[cell] = rate
                cell_status[cell] = "released"

        return {
            "success_rates": success_rates,
            "dp_total_counts": dp_totals,
            "dp_success_counts": dp_successes,
            "cell_status": cell_status,
            "suppressed_cells": suppressed_cells,
            "min_noisy_denominator": self.min_noisy_denominator,
            "total_dp_array": total_dp,
            "success_dp_array": success_dp,
        }

    # Internal / evaluation helpers (do not expose to production web)
    def compute_raw_exact(self, user_matrix: UserMatrix) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        n_matrix = user_matrix.matrix
        s_matrix = user_matrix.success_matrix if getattr(user_matrix, "success_matrix", None) is not None else np.zeros_like(n_matrix)
        s_raw = np.sum(s_matrix, axis=0)
        n_raw = np.sum(n_matrix, axis=0)
        rates_raw = np.zeros(9, dtype=np.float64)
        for i in range(9):
            rates_raw[i] = s_raw[i] / n_raw[i] if n_raw[i] > 0 else 0.0
        return s_raw, n_raw, rates_raw

    def compute_clipped_exact(self, user_matrix: UserMatrix) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        joint = self.build_user_contributions(user_matrix)
        joint_sum = np.sum(joint, axis=0)
        s_clip = joint_sum[:9]
        n_clip = joint_sum[9:18]
        rates_clip = np.zeros(9, dtype=np.float64)
        for i in range(9):
            rates_clip[i] = s_clip[i] / n_clip[i] if n_clip[i] > 0 else 0.0
        return s_clip, n_clip, rates_clip

    def evaluate_utility(
        self,
        dp_vector: np.ndarray,
        user_matrix: UserMatrix,
    ) -> Dict[str, Any]:
        """Compute evaluation metrics comparing raw exact, clipped exact, and DP release."""
        s_raw, n_raw, r_raw = self.compute_raw_exact(user_matrix)
        s_clip, n_clip, r_clip = self.compute_clipped_exact(user_matrix)
        post = self.post_process(dp_vector, user_matrix)

        r_dp = np.array([post["success_rates"][c] if post["success_rates"][c] is not None else np.nan for c in user_matrix.cell_names])
        valid_mask = ~np.isnan(r_dp)
        num_valid = int(valid_mask.sum())
        suppression_rate = float((len(r_dp) - num_valid) / len(r_dp))

        if num_valid > 0:
            mae_vs_raw = float(np.mean(np.abs(r_dp[valid_mask] - r_raw[valid_mask])))
            rmse_vs_raw = float(np.sqrt(np.mean((r_dp[valid_mask] - r_raw[valid_mask]) ** 2)))
            mae_vs_clip = float(np.mean(np.abs(r_dp[valid_mask] - r_clip[valid_mask])))
            rmse_vs_clip = float(np.sqrt(np.mean((r_dp[valid_mask] - r_clip[valid_mask]) ** 2)))
            clipping_bias = float(np.mean(np.abs(r_clip[valid_mask] - r_raw[valid_mask])))
        else:
            mae_vs_raw = rmse_vs_raw = mae_vs_clip = rmse_vs_clip = clipping_bias = float("nan")

        return {
            "suppression_rate": suppression_rate,
            "mae_vs_raw": mae_vs_raw,
            "rmse_vs_raw": rmse_vs_raw,
            "mae_vs_clip": mae_vs_clip,
            "rmse_vs_clip": rmse_vs_clip,
            "clipping_bias": clipping_bias,
            "num_valid_cells": num_valid,
        }
