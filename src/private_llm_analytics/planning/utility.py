"""Utility estimation framework for Small-N Differential Privacy Release Planning (v0.3).

Provides model-based, data-independent estimation of:
  - Clipping bias under public prior distributions
  - DP noise error using exact Analytic Gaussian calibration
  - Dimensionality / Granularity information loss
  - Release staleness penalty under longitudinal frequency
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
import math
from typing import Any, Dict, Optional, Tuple, Union
import numpy as np
from scipy import integrate
from scipy.stats import norm, lognorm

from private_llm_analytics.mechanisms.gaussian import calibrate_analytic_gaussian
from private_llm_analytics.planning.candidates import ReleasePolicyCandidate


GRANULARITY_DIMENSIONS: Dict[str, int] = {
    "full_9_cell": 9,
    "provider_3_cell": 3,
    "task_3_cell": 3,
    "overall_1_cell": 1,
}

DEFAULT_GRANULARITY_PENALTIES: Dict[str, float] = {
    "full_9_cell": 0.0,
    "provider_3_cell": 0.25,
    "task_3_cell": 0.25,
    "overall_1_cell": 0.50,
}


@dataclass(frozen=True)
class UtilityWeights:
    """Non-negative weights for multi-objective utility loss formulation."""

    clipping: float = 1.0
    noise: float = 1.0
    aggregation: float = 1.0
    staleness: float = 1.0

    def __post_init__(self):
        for name, val in [
            ("clipping", self.clipping),
            ("noise", self.noise),
            ("aggregation", self.aggregation),
            ("staleness", self.staleness),
        ]:
            if val < 0:
                raise ValueError(f"Utility weight '{name}' must be >= 0, got {val}")
        if self.clipping == 0 and self.noise == 0 and self.aggregation == 0 and self.staleness == 0:
            raise ValueError("At least one utility weight must be strictly positive.")


@dataclass(frozen=True)
class UtilityComponents:
    """Individual decomposed error estimates for a candidate release policy."""

    clipping_error: float
    noise_error: float
    aggregation_error: float
    staleness_error: float
    total_error: float

    def to_dict(self) -> Dict[str, float]:
        return {
            "clipping": self.clipping_error,
            "noise": self.noise_error,
            "aggregation": self.aggregation_error,
            "staleness": self.staleness_error,
            "total": self.total_error,
        }


class PublicPriorDistribution(ABC):
    """Abstract base class for public prior distributions over user interaction norms."""

    @abstractmethod
    def expected_excess(self, threshold: float) -> float:
        """Compute E[(R - C)_+] under the prior distribution."""
        pass

    @abstractmethod
    def to_dict(self) -> Dict[str, Any]:
        pass


class TruncatedNormalPrior(PublicPriorDistribution):
    """Truncated Normal prior distribution: TN(mean, std, min_val, max_val)."""

    def __init__(
        self,
        mean: float = 60.0,
        std: float = 20.0,
        min_val: float = 5.0,
        max_val: float = 150.0,
    ):
        if std <= 0:
            raise ValueError(f"std must be > 0, got {std}")
        if min_val >= max_val:
            raise ValueError(f"min_val ({min_val}) must be < max_val ({max_val})")
        self.mean = float(mean)
        self.std = float(std)
        self.min_val = float(min_val)
        self.max_val = float(max_val)

        # Normalization factor for truncation
        self._alpha = (self.min_val - self.mean) / self.std
        self._beta = (self.max_val - self.mean) / self.std
        self._denom = float(norm.cdf(self._beta) - norm.cdf(self._alpha))
        if self._denom <= 0:
            raise ValueError("Truncation bounds yield zero probability mass.")

    def pdf(self, x: float) -> float:
        if x < self.min_val or x > self.max_val:
            return 0.0
        z = (x - self.mean) / self.std
        return float(norm.pdf(z) / (self.std * self._denom))

    def expected_excess(self, threshold: float) -> float:
        """Compute E[max(0, R - threshold)]. Monotonically non-increasing in threshold."""
        c = float(threshold)
        if c >= self.max_val:
            return 0.0
        low = max(c, self.min_val)
        val, _ = integrate.quad(lambda r: (r - c) * self.pdf(r), low, self.max_val)
        return max(0.0, float(val))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "truncated_normal",
            "mean": self.mean,
            "std": self.std,
            "min_val": self.min_val,
            "max_val": self.max_val,
        }


class LogNormalPrior(PublicPriorDistribution):
    """LogNormal prior distribution: LN(mean, std) parametrized by arithmetic mean and std."""

    def __init__(self, mean: float = 60.0, std: float = 25.0, max_cutoff: float = 300.0):
        if mean <= 0:
            raise ValueError(f"mean must be > 0, got {mean}")
        if std <= 0:
            raise ValueError(f"std must be > 0, got {std}")
        self.mean = float(mean)
        self.std = float(std)
        self.max_cutoff = float(max_cutoff)

        # Convert arithmetic mean and variance to underlying normal parameters (mu, sigma)
        var = std**2
        self.sigma = math.sqrt(math.log(1.0 + var / (mean**2)))
        self.mu = math.log(mean) - 0.5 * (self.sigma**2)

    def pdf(self, x: float) -> float:
        if x <= 0 or x > self.max_cutoff:
            return 0.0
        return float(lognorm.pdf(x, s=self.sigma, scale=math.exp(self.mu)))

    def expected_excess(self, threshold: float) -> float:
        c = float(threshold)
        if c >= self.max_cutoff:
            return 0.0
        val, _ = integrate.quad(lambda r: (r - c) * self.pdf(r), c, self.max_cutoff)
        return max(0.0, float(val))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "lognormal",
            "mean": self.mean,
            "std": self.std,
            "max_cutoff": self.max_cutoff,
        }


class ClippingEstimator:
    """Estimates expected clipping bias under public prior distributions."""

    @staticmethod
    def estimate_clipping_bias(
        contribution_bound: Optional[float],
        population_size: int,
        prior: PublicPriorDistribution,
    ) -> float:
        if contribution_bound is None:
            return 0.0
        excess = prior.expected_excess(contribution_bound)
        return float(population_size * excess)


class NoiseEstimator:
    """Estimates expected noise error using exact DP calibration."""

    @staticmethod
    def estimate_noise_sigma(
        metric: str,
        candidate: ReleasePolicyCandidate,
    ) -> Tuple[float, float]:
        """Calibrate noise scale sigma and return (sigma, sensitivity)."""
        sens = NoiseEstimator.get_sensitivity(metric, candidate)
        eps = float(candidate.epsilon_per_release)
        delta = float(candidate.delta_per_release)

        if candidate.mechanism == "gaussian":
            sigma = calibrate_analytic_gaussian(
                epsilon=eps,
                delta=delta,
                l2_sensitivity=sens,
            )
        elif candidate.mechanism == "laplace":
            sigma = sens / eps
        else:
            raise ValueError(f"Unsupported mechanism '{candidate.mechanism}'")

        return float(sigma), float(sens)

    @staticmethod
    def get_sensitivity(metric: str, candidate: ReleasePolicyCandidate) -> float:
        metric_norm = metric.lower().strip()
        dim = GRANULARITY_DIMENSIONS.get(candidate.granularity, 9)

        if "volume" in metric_norm:
            c = candidate.contribution_bound if candidate.contribution_bound is not None else 20.0
            return float(c)
        elif "prevalence" in metric_norm:
            # Binary participation vector sensitivity sqrt(min(d, 9))
            return float(math.sqrt(min(dim, 9)))
        elif "share" in metric_norm:
            # L2 sensitivity <= 1 for user-normalized vectors
            return 1.0
        else:
            c = candidate.contribution_bound if candidate.contribution_bound is not None else 1.0
            return float(c)

    @staticmethod
    def estimate_noise_error(
        metric: str,
        candidate: ReleasePolicyCandidate,
        population_size: int,
    ) -> float:
        """Estimate expected absolute noise error according to metric semantics."""
        sigma, _ = NoiseEstimator.estimate_noise_sigma(metric, candidate)
        dim = GRANULARITY_DIMENSIONS.get(candidate.granularity, 9)
        metric_norm = metric.lower().strip()

        # Expected absolute value of N(0, sigma^2) is sigma * sqrt(2/pi)
        e_abs_1d = sigma * math.sqrt(2.0 / math.pi)

        if "volume" in metric_norm:
            # Total volume noise error across all d active cells
            return float(dim * e_abs_1d)
        elif "prevalence" in metric_norm:
            # Prevalence rate error in percentage points: 100 * (E|Z| / N) per cell
            n = max(1, population_size)
            return float(100.0 * (dim * e_abs_1d) / n)
        elif "share" in metric_norm:
            # Normalized share error: sum of noise over user count N
            n = max(1, population_size)
            return float((dim * e_abs_1d) / n)
        else:
            return float(dim * e_abs_1d)


class GranularityEstimator:
    """Estimates information loss caused by dimension reduction / coarse granularity."""

    @staticmethod
    def estimate_granularity_error(
        metric: str,
        granularity: str,
        population_size: int,
        base_scale: float = 1.0,
        penalties: Optional[Dict[str, float]] = None,
    ) -> float:
        pen_map = penalties or DEFAULT_GRANULARITY_PENALTIES
        penalty_factor = pen_map.get(granularity, 0.5)
        metric_norm = metric.lower().strip()

        if "volume" in metric_norm:
            # Scale by typical interaction scale (e.g. population_size * expected_interactions)
            return float(penalty_factor * population_size * base_scale)
        elif "prevalence" in metric_norm:
            # Percentage points loss (e.g. up to 10-25 pp when collapsing task detail)
            return float(penalty_factor * 25.0)
        elif "share" in metric_norm:
            # Proportion loss (e.g. TV distance up to 0.5)
            return float(penalty_factor * 0.5)
        else:
            return float(penalty_factor * base_scale)


class StalenessEstimator:
    """Estimates staleness penalty caused by release frequency interval F."""

    @staticmethod
    def estimate_staleness_error(
        metric: str,
        frequency_days: int,
        lambda_stale: float = 1.0,
    ) -> float:
        # Expected age of latest released data is roughly F / 2
        expected_age_days = float(frequency_days) / 2.0
        return float(lambda_stale * expected_age_days)


class UtilityEstimator:
    """Composite utility estimator evaluating candidates under objective E_total(pi)."""

    def __init__(
        self,
        weights: Optional[UtilityWeights] = None,
        public_prior: Optional[PublicPriorDistribution] = None,
        lambda_stale: float = 1.0,
        granularity_penalties: Optional[Dict[str, float]] = None,
    ):
        self.weights = weights or UtilityWeights()
        self.public_prior = public_prior or TruncatedNormalPrior()
        self.lambda_stale = float(lambda_stale)
        self.granularity_penalties = granularity_penalties or DEFAULT_GRANULARITY_PENALTIES

    def evaluate_candidate(
        self,
        metric: str,
        candidate: ReleasePolicyCandidate,
        population_size: int,
        base_interaction_scale: float = 60.0,
    ) -> UtilityComponents:
        """Compute full decomposed error estimates for a policy candidate."""
        # 1. Clipping bias
        if "volume" in metric.lower():
            e_clip = ClippingEstimator.estimate_clipping_bias(
                contribution_bound=candidate.contribution_bound,
                population_size=population_size,
                prior=self.public_prior,
            )
        else:
            e_clip = 0.0

        # 2. Noise error
        e_noise = NoiseEstimator.estimate_noise_error(
            metric=metric,
            candidate=candidate,
            population_size=population_size,
        )

        # 3. Granularity error
        e_agg = GranularityEstimator.estimate_granularity_error(
            metric=metric,
            granularity=candidate.granularity,
            population_size=population_size,
            base_scale=base_interaction_scale,
            penalties=self.granularity_penalties,
        )

        # 4. Staleness error
        e_stale = StalenessEstimator.estimate_staleness_error(
            metric=metric,
            frequency_days=candidate.frequency_days,
            lambda_stale=self.lambda_stale,
        )

        # Total weighted objective
        total = (
            self.weights.clipping * e_clip
            + self.weights.noise * e_noise
            + self.weights.aggregation * e_agg
            + self.weights.staleness * e_stale
        )

        return UtilityComponents(
            clipping_error=float(e_clip),
            noise_error=float(e_noise),
            aggregation_error=float(e_agg),
            staleness_error=float(e_stale),
            total_error=float(total),
        )
