"""Base abstractions for Statistic-Specific Differential Privacy Release Profiles (v0.6)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import hashlib
import json
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union
import numpy as np

from private_llm_analytics.aggregation.user_matrix import UserMatrix
from private_llm_analytics.mechanisms.base import DPMechanism
from private_llm_analytics.mechanisms.gaussian import GaussianMechanism
if TYPE_CHECKING:
    from private_llm_analytics.metrics.base import DPReleaseResult
from private_llm_analytics.privacy.exceptions import (
    MissingRequiredStatisticColumn,
    PrivacyConfigurationError,
    UnsupportedProductionMechanism,
)
from private_llm_analytics.privacy.rdp import (
    DEFAULT_RDP_ORDERS,
    compute_gaussian_rdp,
)


@dataclass(frozen=True)
class ProfileMetadata:
    """Public, non-sensitive metadata describing an approved statistic-specific DP profile."""
    statistic_id: str
    profile_id: str
    profile_version: str
    display_name: str
    description: str
    display_name_zh: str
    description_zh: str
    profile_name_zh: str
    privacy_unit: str = "user"
    adjacency: str = "add_remove_user"
    required_columns: List[str] = field(default_factory=list)
    contribution_definition: str = ""
    contribution_bound_description: str = ""
    sensitivity_type: str = "L2"
    sensitivity_description: str = ""
    mechanism_family: str = "gaussian_vector"
    accounting_family: str = "rdp"
    post_processing_description: str = ""
    supports_continual: bool = False
    supports_planner: bool = False
    public_parameters: Dict[str, Any] = field(default_factory=dict)
    configurable_parameters: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.statistic_id,
            "statistic_id": self.statistic_id,
            "profile_id": self.profile_id,
            "profile_version": self.profile_version,
            "display_name": self.display_name,
            "description": self.description,
            "display_name_zh": self.display_name_zh,
            "description_zh": self.description_zh,
            "profile_name_zh": self.profile_name_zh,
            "privacy_unit": self.privacy_unit,
            "adjacency": self.adjacency,
            "required_columns": list(self.required_columns),
            "contribution_definition": self.contribution_definition,
            "contribution_bound_description": self.contribution_bound_description,
            "sensitivity_type": self.sensitivity_type,
            "sensitivity_description": self.sensitivity_description,
            "mechanism_family": self.mechanism_family,
            "accounting_family": self.accounting_family,
            "post_processing_description": self.post_processing_description,
            "supports_continual": self.supports_continual,
            "supports_planner": self.supports_planner,
            "public_parameters": dict(self.public_parameters),
            "configurable_parameters": list(self.configurable_parameters),
        }

    dict = to_dict


@dataclass(frozen=True)
class MechanismPlan:
    """Concrete mechanism execution parameters calibrated by the profile."""
    mechanism_name: str
    sensitivity: float
    noise_scale: float
    epsilon: float
    delta: float
    dimension: int


@dataclass(frozen=True)
class PrivacyCostDescriptor:
    """Contract between mechanism and the unified lifetime privacy accountant."""
    accounting_family: str
    orders: List[float]
    costs: List[float]
    mechanism_profile_id: str
    sensitivity: float
    noise_scale: float
    epsilon: float
    delta: float


class DPStatisticProfile(ABC):
    """Abstract base class for all statistic-specific DP profiles in v0.6.

    The profile owns:
      1. User-level contribution semantics and bounding
      2. Sensitivity derivation
      3. Approved mechanism selection and noise calibration
      4. Privacy cost representation for unified lifetime accounting
      5. Post-processing into safe public statistics
      6. Profile fingerprinting for canonical release identity
    """

    statistic_id: str
    profile_id: str
    profile_version: str = "1"

    @property
    def version(self) -> str:
        return self.profile_version

    @abstractmethod
    def metadata(self) -> ProfileMetadata:
        """Return machine-readable public metadata for this profile."""
        pass

    @abstractmethod
    def required_columns(self) -> List[str]:
        """List of interaction telemetry columns required by this profile."""
        pass

    @abstractmethod
    def build_user_contributions(self, user_matrix: UserMatrix) -> np.ndarray:
        """Compute user-level bounded contribution vectors."""
        pass

    @abstractmethod
    def contribution_bound(self) -> float:
        """Return the scalar contribution bound (e.g. C, norm bound, or k)."""
        pass

    @abstractmethod
    def sensitivity(self) -> float:
        """Return the exact mathematical sensitivity under add/remove user adjacency."""
        pass

    @abstractmethod
    def build_mechanism_plan(self, epsilon: float, delta: float) -> MechanismPlan:
        """Calibrate the approved mechanism to requested (epsilon, delta)."""
        pass

    @abstractmethod
    def privacy_cost_descriptor(
        self,
        plan: MechanismPlan,
        orders: Optional[List[float]] = None,
    ) -> PrivacyCostDescriptor:
        """Construct the exact RDP cost descriptor for unified lifetime accounting."""
        pass

    @abstractmethod
    def execute_mechanism(
        self,
        bounded_aggregate: np.ndarray,
        plan: MechanismPlan,
        rng: Optional[np.random.Generator] = None,
    ) -> np.ndarray:
        """Apply the calibrated noise to the true bounded aggregate vector."""
        pass

    @abstractmethod
    def post_process(
        self,
        dp_vector: np.ndarray,
        user_matrix: UserMatrix,
    ) -> Dict[str, Any]:
        """Post-process noisy vector into interpretable public statistics."""
        pass

    def compute_exact(self, user_matrix: UserMatrix) -> np.ndarray:
        """Compute unnoised bounded aggregate (internal/audit/experiment only)."""
        contributions = self.build_user_contributions(user_matrix)
        return np.sum(contributions, axis=0)

    def validate_configuration(self) -> None:
        """Verify profile configuration parameters; fail closed if invalid."""
        pass

    def canonical_profile_fingerprint(
        self,
        providers: Optional[List[str]] = None,
        tasks: Optional[List[str]] = None,
    ) -> str:
        """Generate deterministic SHA-256 fingerprint of semantic profile inputs.

        Includes:
          - statistic_id
          - profile_id
          - profile_version
          - adjacency
          - contribution bound parameters
          - mechanism family
          - post-processing semantic parameters
          - taxonomy fingerprint
        Excludes:
          - RNG state
          - raw private counts
          - active user count
        """
        meta = self.metadata()
        provs = sorted(providers) if providers else ["provider_a", "provider_b", "provider_c"]
        tsks = sorted(tasks) if tasks else ["analysis", "document", "technical"]

        canonical_dict = {
            "statistic_id": self.statistic_id,
            "profile_id": self.profile_id,
            "profile_version": self.profile_version,
            "adjacency": meta.adjacency,
            "mechanism_family": meta.mechanism_family,
            "accounting_family": meta.accounting_family,
            "sensitivity": float(self.sensitivity()),
            "contribution_bound": float(self.contribution_bound()),
            "public_parameters": {k: meta.public_parameters[k] for k in sorted(meta.public_parameters)},
            "taxonomy": {
                "providers": provs,
                "tasks": tsks,
            },
        }
        encoded = json.dumps(canonical_dict, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def execute_release(
        self,
        user_matrix: UserMatrix,
        epsilon: float,
        delta: float,
        rng: Optional[np.random.Generator] = None,
    ) -> DPReleaseResult:
        """Execute the end-to-end statistic release pipeline."""
        self.validate_configuration()

        # 1. Enforce required columns / prerequisites
        req_cols = self.required_columns()
        if "success" in req_cols and getattr(user_matrix, "success_matrix", None) is None:
            raise MissingRequiredStatisticColumn(
                f"Statistic '{self.statistic_id}' requires telemetry column 'success'."
            )

        # 2. Build bounded user contributions
        contributions = self.build_user_contributions(user_matrix)

        # 3. Aggregate bounded contributions
        bounded_aggregate = np.sum(contributions, axis=0)

        # 4. Calibrate mechanism plan
        plan = self.build_mechanism_plan(epsilon=epsilon, delta=delta)

        # 5. Add calibrated noise
        noisy_aggregate = self.execute_mechanism(bounded_aggregate, plan=plan, rng=rng)

        # 6. Post-process public statistics
        post_processed = self.post_process(noisy_aggregate, user_matrix)

        from private_llm_analytics.metrics.base import DPReleaseResult

        return DPReleaseResult(
            metric_name=self.statistic_id,
            dp_vector=noisy_aggregate,
            post_processed=post_processed,
            epsilon=plan.epsilon,
            delta=plan.delta,
            mechanism=plan.mechanism_name,
            noise_scale=plan.noise_scale,
            sensitivity=plan.sensitivity,
            cell_names=user_matrix.cell_names,
            n_users=user_matrix.n_users,
            public_population_size=user_matrix.public_population_size,
        )
