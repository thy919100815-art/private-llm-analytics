"""Data models, configuration classes, and diagnostics for Continual Observation (v0.4)."""

from dataclasses import asdict, dataclass, field
from datetime import datetime
import hashlib
import json
from typing import Any, Dict, List, Optional, Sequence, Tuple
import zoneinfo

from private_llm_analytics.planning.candidates import VALID_GRANULARITIES
from private_llm_analytics.privacy.exceptions import (
    PrivacyConfigurationError,
    ProductionSafetyConfigurationError,
)

from private_llm_analytics.privacy.rdp import DEFAULT_RDP_ORDERS

SUPPORTED_CONTINUAL_METRICS = ("interaction_volume",)
SUPPORTED_CONTINUAL_MECHANISMS = ("gaussian",)
SUPPORTED_CONTINUAL_TOPOLOGIES = ("binary",)

DEFAULT_CONTINUAL_RDP_ORDERS: Tuple[float, ...] = tuple(float(x) for x in DEFAULT_RDP_ORDERS)


@dataclass(frozen=True)
class ContinualTreeConfig:
    """Configuration for binary continual observation tree topology."""

    topology: str = "binary"
    padding: str = "next_power_of_two"

    def __post_init__(self):
        if self.topology not in SUPPORTED_CONTINUAL_TOPOLOGIES:
            raise ProductionSafetyConfigurationError(
                f"Unsupported continual topology '{self.topology}'. Only {SUPPORTED_CONTINUAL_TOPOLOGIES} is supported."
            )
        if self.padding != "next_power_of_two":
            raise ProductionSafetyConfigurationError(
                f"Unsupported tree padding '{self.padding}'. Only 'next_power_of_two' is supported."
            )


@dataclass(frozen=True)
class ContinualNoiseConfig:
    """Noise parameters and calibration targets for binary tree levels."""

    mechanism: str = "gaussian"
    target_epoch_epsilon: float = 2.0
    calibration_delta: float = 1e-6
    equal_sigma_per_level: bool = True
    sigma: Optional[float] = None
    rdp_orders: Tuple[float, ...] = DEFAULT_CONTINUAL_RDP_ORDERS

    def __post_init__(self):
        mech = str(self.mechanism).lower().strip()
        if mech != "gaussian":
            raise ProductionSafetyConfigurationError(
                f"Continual observation in v0.4 requires Gaussian mechanism, got '{self.mechanism}'."
            )
        if self.target_epoch_epsilon <= 0:
            raise PrivacyConfigurationError(f"target_epoch_epsilon must be > 0, got {self.target_epoch_epsilon}")
        if not (0 < self.calibration_delta < 1):
            raise PrivacyConfigurationError(f"calibration_delta must be in (0, 1), got {self.calibration_delta}")
        if self.sigma is not None and self.sigma <= 0:
            raise PrivacyConfigurationError(f"sigma must be > 0 when specified, got {self.sigma}")
        if not self.equal_sigma_per_level:
            raise ProductionSafetyConfigurationError("v0.4 requires equal_sigma_per_level=True.")


@dataclass(frozen=True)
class ContinualStreamConfig:
    """Authoritative stream identity and continual observation bounding configuration."""

    stream_id: str = "stream_default"
    metric: str = "interaction_volume"
    granularity: str = "full_9_cell"
    epoch_id: str = "epoch_001"
    epoch_start: str = "2026-01-01"
    epoch_days: int = 90
    timezone: str = "UTC"
    daily_l2_bound: float = 10.0
    epoch_group_l12_bound: float = 40.0
    lateness_grace_days: int = 1
    tree: ContinualTreeConfig = field(default_factory=ContinualTreeConfig)
    noise: ContinualNoiseConfig = field(default_factory=ContinualNoiseConfig)
    dataset_fingerprint: Optional[str] = None
    public_roster_fingerprint: Optional[str] = None
    taxonomy_fingerprint: Optional[str] = None

    def __post_init__(self):
        norm_metric = str(self.metric).lower().strip()
        if norm_metric not in SUPPORTED_CONTINUAL_METRICS:
            raise ProductionSafetyConfigurationError(
                f"Continual observation path only supports metrics {SUPPORTED_CONTINUAL_METRICS}, got '{self.metric}'."
            )
        if self.granularity not in VALID_GRANULARITIES:
            raise ProductionSafetyConfigurationError(
                f"Unsupported granularity '{self.granularity}'. Must be one of {VALID_GRANULARITIES}."
            )
        if self.epoch_days <= 0:
            raise PrivacyConfigurationError(f"epoch_days must be > 0, got {self.epoch_days}")
        if self.daily_l2_bound <= 0:
            raise PrivacyConfigurationError(f"daily_l2_bound must be > 0, got {self.daily_l2_bound}")
        if self.epoch_group_l12_bound <= 0:
            raise PrivacyConfigurationError(f"epoch_group_l12_bound must be > 0, got {self.epoch_group_l12_bound}")
        if self.lateness_grace_days < 0:
            raise PrivacyConfigurationError(f"lateness_grace_days must be >= 0, got {self.lateness_grace_days}")

        # Validate timezone
        try:
            zoneinfo.ZoneInfo(self.timezone)
        except Exception as e:
            raise PrivacyConfigurationError(f"Invalid timezone '{self.timezone}': {e}")

    def compute_stream_fingerprint(self) -> str:
        """Compute canonical SHA-256 fingerprint uniquely identifying this stream configuration."""
        canonical_dict = {
            "stream_id": self.stream_id,
            "metric": self.metric,
            "granularity": self.granularity,
            "epoch_id": self.epoch_id,
            "epoch_start": str(self.epoch_start),
            "epoch_days": int(self.epoch_days),
            "timezone": self.timezone,
            "daily_l2_bound": float(self.daily_l2_bound),
            "epoch_group_l12_bound": float(self.epoch_group_l12_bound),
            "lateness_grace_days": int(self.lateness_grace_days),
            "tree_topology": self.tree.topology,
            "tree_padding": self.tree.padding,
            "noise_mechanism": self.noise.mechanism,
            "target_epoch_epsilon": float(self.noise.target_epoch_epsilon),
            "calibration_delta": float(self.noise.calibration_delta),
            "equal_sigma_per_level": bool(self.noise.equal_sigma_per_level),
            "sigma": float(self.noise.sigma) if self.noise.sigma is not None else None,
            "rdp_orders": list(self.noise.rdp_orders),
            "dataset_fingerprint": self.dataset_fingerprint,
            "public_roster_fingerprint": self.public_roster_fingerprint,
            "taxonomy_fingerprint": self.taxonomy_fingerprint,
        }
        encoded = json.dumps(canonical_dict, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    @property
    def stream_fingerprint(self) -> str:
        return self.compute_stream_fingerprint()


@dataclass(frozen=True)
class TemporalContributionDiagnostics:
    """Empirical diagnostics for temporal clipping (Section 14 & 73)."""

    raw_total: float
    daily_clipped_total: float
    epoch_clipped_total: float
    daily_clipping_bias: float
    temporal_clipping_bias: float
    fraction_daily_clipped: float
    fraction_epoch_clipped: float
    n_users: int
    epoch_days: int
    dimension: int
