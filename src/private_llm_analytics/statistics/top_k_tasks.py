"""Differentially Private Top-K Tasks Profile (v0.8)."""

from __future__ import annotations
import hashlib
from typing import Any, Dict, List, Optional
import numpy as np

from private_llm_analytics.aggregation.user_matrix import UserMatrix
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


class TopKTasksProfile(DPStatisticProfile):
    """Approved DP Release Profile for Long-Tail Top-K Tasks Discovery.

    Target:
      Discover the top-K task categories and their workforce participation counts
      under user-level differential privacy.

    Mathematical Specification:
      - Unit: Complete user trajectory within observation window.
      - Adjacency: Add/remove single user.
      - Contribution Bounding:
          User participates in at most max_tasks_per_user distinct tasks.
          If active in > max_tasks_per_user tasks, subsample max_tasks_per_user tasks.
          Binary vector v_u in {0, 1}^|Tasks| with ||v_u||_0 <= max_tasks_per_user.
      - Sensitivity:
          Delta_2 <= sqrt(max_tasks_per_user)
      - Mechanism:
          Analytic Gaussian Vector Mechanism on task participation counts.
      - Post-processing:
          Clamp noisy counts >= 0, sort in descending order, filter by threshold if set,
          and select top-K items.
    """

    statistic_id = "top_k_tasks"
    profile_id = "bounded_top_k_gaussian"
    profile_version = "1"

    def __init__(
        self,
        k: int = 5,
        max_tasks_per_user: int = 5,
        threshold: Optional[float] = None,
    ):
        if k <= 0:
            raise InvalidStatisticProfileConfiguration(f"k must be strictly positive, got {k}")
        if max_tasks_per_user <= 0:
            raise InvalidStatisticProfileConfiguration(
                f"max_tasks_per_user must be strictly positive, got {max_tasks_per_user}"
            )
        if threshold is not None and threshold < 0:
            raise InvalidStatisticProfileConfiguration(f"threshold cannot be negative, got {threshold}")

        self.k = int(k)
        self.max_tasks_per_user = int(max_tasks_per_user)
        self.threshold = float(threshold) if threshold is not None else None

    def metadata(self) -> ProfileMetadata:
        sens = self.sensitivity()
        return ProfileMetadata(
            statistic_id=self.statistic_id,
            profile_id=self.profile_id,
            profile_version=self.profile_version,
            display_name="Top-K Tasks",
            description="Differentially private discovery of top-K task categories with user contribution bounding",
            display_name_zh="热门任务 Top-K",
            description_zh="在差分隐私保障下发现前 K 个最常使用的任务类型（高斯向量机制 + 排序截断）",
            profile_name_zh="任务参与约束 + 高斯向量机制 + 差分隐私 Top-K 选择",
            privacy_unit="user",
            adjacency="add_remove_user",
            required_columns=list(REQUIRED_COLUMNS),
            contribution_definition=f"User participates in at most {self.max_tasks_per_user} tasks: ||v_u||_0 <= {self.max_tasks_per_user}",
            contribution_bound_description=f"max_tasks_per_user = {self.max_tasks_per_user}",
            sensitivity_type="L2",
            sensitivity_description=f"Delta_2 = sqrt({self.max_tasks_per_user}) = {sens:.4f}",
            mechanism_family="gaussian_vector",
            accounting_family="rdp",
            post_processing_description="Clamp counts >= 0, sort descending, filter by threshold, take top-K",
            supports_continual=False,
            supports_planner=False,
            public_parameters={
                "k": self.k,
                "max_tasks_per_user": self.max_tasks_per_user,
                "threshold": self.threshold,
            },
            configurable_parameters=["k", "max_tasks_per_user", "threshold"],
        )

    def required_columns(self) -> List[str]:
        return list(REQUIRED_COLUMNS)

    def contribution_bound(self) -> float:
        return float(self.max_tasks_per_user)

    def sensitivity(self) -> float:
        return float(np.sqrt(self.max_tasks_per_user))

    def build_user_contributions(self, user_matrix: UserMatrix) -> np.ndarray:
        n_users = user_matrix.n_users
        tasks = user_matrix.tasks
        n_tasks = len(tasks)

        task_counts = np.zeros((n_users, n_tasks), dtype=np.float64)
        for (p, t), cell_idx in user_matrix.cell_map.items():
            if t in tasks:
                t_idx = tasks.index(t)
                task_counts[:, t_idx] += user_matrix.matrix[:, cell_idx]

        binary_tasks = (task_counts > 0).astype(np.float64)
        k_bound = self.max_tasks_per_user
        bounded_contributions = np.zeros_like(binary_tasks)

        for u in range(n_users):
            active_indices = np.where(binary_tasks[u] > 0)[0]
            if len(active_indices) <= k_bound:
                bounded_contributions[u, active_indices] = 1.0
            else:
                # Deterministic selection based on user_id hash to prevent unnecessary variance
                uid = user_matrix.user_ids[u]
                seed = int(hashlib.sha256(uid.encode("utf-8")).hexdigest()[:8], 16)
                user_rng = np.random.default_rng(seed)
                chosen = user_rng.choice(active_indices, size=k_bound, replace=False)
                bounded_contributions[u, chosen] = 1.0

        return bounded_contributions

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
            dimension=dimension or 1,
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
        tasks = user_matrix.tasks
        assert len(dp_vector) == len(tasks)

        clamped = np.maximum(0.0, dp_vector)
        all_noisy_counts = {t: float(clamped[i]) for i, t in enumerate(tasks)}

        # Sort tasks descending by noisy count, with lexicographic tie-breaking
        sorted_indices = sorted(
            range(len(tasks)),
            key=lambda i: (-clamped[i], tasks[i]),
        )

        top_k_list = []
        selected_tasks = []
        denom = user_matrix.public_population_size

        for rank, idx in enumerate(sorted_indices):
            t_name = tasks[idx]
            count = float(clamped[idx])

            # Apply threshold if configured
            if self.threshold is not None and count < self.threshold:
                continue

            prev = float(np.clip(count / denom, 0.0, 1.0)) if denom is not None and denom > 0 else None
            top_k_list.append({
                "rank": len(top_k_list) + 1,
                "task": t_name,
                "noisy_count": count,
                "estimated_prevalence": prev,
            })
            selected_tasks.append(t_name)

            if len(top_k_list) >= self.k:
                break

        return {
            "top_k": top_k_list,
            "selected_tasks": selected_tasks,
            "all_noisy_counts": all_noisy_counts,
            "k": self.k,
            "max_tasks_per_user": self.max_tasks_per_user,
            "threshold": self.threshold,
            "public_population_size": denom,
        }
