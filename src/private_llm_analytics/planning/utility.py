"""Utility estimation framework for Small-N Differential Privacy Release Planning (v0.3).

Provides model-based, data-independent estimation of:
  - Clipping bias under window-scaled public prior distributions
  - DP noise error using exact Analytic Gaussian calibration and target-space sensitivities
  - Workload-weighted aggregation loss across granularities
  - Release staleness penalty under longitudinal frequency
  - Normalized dimensionless total loss formulation
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
import math
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
from scipy import integrate
from scipy.stats import norm, lognorm

from private_llm_analytics.mechanisms.gaussian import calibrate_analytic_gaussian
from private_llm_analytics.planning.candidates import ReleasePolicyCandidate
from private_llm_analytics.planning.policy import (
    compute_workload_reconstruction_error,
    get_direct_granularity_sensitivity,
)


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
    """Non-negative weights for multi-objective normalized utility loss formulation."""

    clipping: float = 1.0
    noise: float = 1.0
    aggregation: float = 1.0
    staleness: float = 0.5
    window: float = 0.2

    def __post_init__(self):
        for name, val in [
            ("clipping", self.clipping),
            ("noise", self.noise),
            ("aggregation", self.aggregation),
            ("staleness", self.staleness),
            ("window", self.window),
        ]:
            if val < 0:
                raise ValueError(f"Utility weight '{name}' must be >= 0, got {val}")
        if (
            self.clipping == 0
            and self.noise == 0
            and self.aggregation == 0
            and self.staleness == 0
            and self.window == 0
        ):
            raise ValueError("At least one utility weight must be strictly positive.")


@dataclass(frozen=True)
class UtilityComponents:
    """Individual decomposed error estimates for a candidate release policy.

    Provides both:
      - Raw metric-scale errors (for logging and inspection)
      - Normalized dimensionless losses (for objective summation)
    """

    clipping_error: float
    noise_error: float
    aggregation_error: float
    staleness_error: float
    window_error: float
    total_error: float

    # Normalized dimensionless losses in [0, inf)
    norm_clipping_loss: float = 0.0
    norm_noise_loss: float = 0.0
    norm_aggregation_loss: float = 0.0
    norm_staleness_loss: float = 0.0
    norm_window_loss: float = 0.0

    def to_dict(self) -> Dict[str, float]:
        return {
            "clipping": self.clipping_error,
            "noise": self.noise_error,
            "aggregation": self.aggregation_error,
            "staleness": self.staleness_error,
            "window": self.window_error,
            "total": self.total_error,
            "norm_clipping": self.norm_clipping_loss,
            "norm_noise": self.norm_noise_loss,
            "norm_aggregation": self.norm_aggregation_loss,
            "norm_staleness": self.norm_staleness_loss,
            "norm_window": self.norm_window_loss,
        }


class PublicPriorDistribution(ABC):
    """Abstract base class for public prior distributions over user interaction norms."""

    mean: float
    std: float

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
    """LogNormal prior distribution parametrized by arithmetic mean and std."""

    def __init__(self, mean: float = 60.0, std: float = 25.0, max_cutoff: float = 300.0):
        if mean <= 0:
            raise ValueError(f"mean must be > 0, got {mean}")
        if std <= 0:
            raise ValueError(f"std must be > 0, got {std}")
        self.mean = float(mean)
        self.std = float(std)
        self.max_cutoff = float(max_cutoff)

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


def scale_prior_for_window(
    prior: PublicPriorDistribution,
    window_days: int,
    reference_days: int = 30,
) -> PublicPriorDistribution:
    """Scale reference public prior distribution to candidate statistical window.

    Scaling laws:
      mean_W = mean_ref * (W / reference_days)
      std_W = std_ref * sqrt(W / reference_days)
    """
    if window_days <= 0:
        raise ValueError(f"window_days must be > 0, got {window_days}")
    if reference_days <= 0:
        raise ValueError(f"reference_days must be > 0, got {reference_days}")

    rho = float(window_days) / float(reference_days)

    if isinstance(prior, TruncatedNormalPrior):
        new_mean = prior.mean * rho
        new_std = prior.std * math.sqrt(rho)
        new_min = max(1.0, prior.min_val * rho)
        new_max = max(new_min + 1.0, prior.max_val * rho)
        return TruncatedNormalPrior(
            mean=new_mean,
            std=new_std,
            min_val=new_min,
            max_val=new_max,
        )
    elif isinstance(prior, LogNormalPrior):
        new_mean = prior.mean * rho
        new_std = prior.std * math.sqrt(rho)
        new_max = max(new_mean + 10.0, prior.max_cutoff * rho)
        return LogNormalPrior(
            mean=new_mean,
            std=new_std,
            max_cutoff=new_max,
        )
    else:
        # Default fallback: return prior as is
        return prior


class ClippingEstimator:
    """Estimates population clipping bias under public prior."""

    @staticmethod
    def estimate_clipping_bias(
        contribution_bound: Optional[float],
        population_size: int,
        prior: PublicPriorDistribution,
    ) -> float:
        """Estimate E_clip(C) = N * E[(R - C)_+]."""
        if contribution_bound is None:
            return 0.0
        c = float(contribution_bound)
        if c <= 0:
            raise ValueError(f"contribution_bound must be > 0, got {c}")
        excess = prior.expected_excess(c)
        return float(population_size * excess)


class NoiseEstimator:
    """Estimates DP noise standard deviation and expected absolute error."""

    @staticmethod
    def estimate_noise_sigma(
        metric: str,
        candidate: ReleasePolicyCandidate,
    ) -> Tuple[float, float]:
        """Calibrate noise scale sigma and return (sigma, sensitivity)."""
        sens = get_direct_granularity_sensitivity(
            metric=metric,
            granularity=candidate.granularity,
            clip_norm=candidate.contribution_bound,
        )
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
    def estimate_noise_error(
        metric: str,
        candidate: ReleasePolicyCandidate,
        population_size: int,
    ) -> float:
        """Estimate expected absolute noise error according to direct metric semantics."""
        sigma, _ = NoiseEstimator.estimate_noise_sigma(metric, candidate)
        dim = GRANULARITY_DIMENSIONS.get(candidate.granularity, 9)
        metric_norm = metric.lower().strip()

        # Expected 1D absolute error for N(0, sigma^2) is sigma * sqrt(2 / pi)
        e_abs_1d = sigma * math.sqrt(2.0 / math.pi)

        if "volume" in metric_norm:
            return float(dim * e_abs_1d)
        elif "prevalence" in metric_norm:
            # Absolute count error across active target cells
            return float(dim * e_abs_1d)
        elif "share" in metric_norm:
            return float(dim * e_abs_1d)
        else:
            return float(dim * e_abs_1d)


class GranularityEstimator:
    """Estimates aggregation / reconstruction loss under public workload weights."""

    @staticmethod
    def estimate_granularity_error(
        metric: str,
        granularity: str,
        population_size: int,
        base_scale: float = 60.0,
        workload_weights: Optional[Dict[str, float]] = None,
        provider_shares: Optional[Dict[str, float]] = None,
        task_shares: Optional[Dict[str, float]] = None,
    ) -> float:
        """Estimate normalized workload reconstruction error in [0, 1]."""
        if granularity == "full_9_cell":
            return 0.0

        p_shares = provider_shares or {"provider_a": 0.5, "provider_b": 0.3, "provider_c": 0.2}
        t_shares = task_shares or {"technical": 0.4, "document": 0.35, "analysis": 0.25}

        # Build expected 9-cell distribution
        provs = sorted(p_shares.keys())
        tasks = sorted(t_shares.keys())
        expected_9 = np.zeros(9, dtype=float)
        cell_names = []
        idx = 0
        for p in provs:
            for t in tasks:
                expected_9[idx] = p_shares[p] * t_shares[t] * population_size * base_scale
                cell_names.append(f"{p}:{t}")
                idx += 1

        return compute_workload_reconstruction_error(
            true_9_cell=expected_9,
            cell_names=cell_names,
            granularity=granularity,
            workload_weights=workload_weights,
        )


class StalenessEstimator:
    """Estimates reporting staleness / lag relative to the operational window."""

    @staticmethod
    def estimate_staleness_error(
        frequency_or_metric: Union[str, int] = 7,
        frequency_days: Optional[int] = None,
        window_days: Optional[int] = None,
        lambda_stale: float = 1.0,
    ) -> float:
        """Dimensionless reporting freshness loss."""
        if isinstance(frequency_or_metric, str):
            f = frequency_days if frequency_days is not None else 7
        else:
            f = frequency_or_metric if frequency_days is None else frequency_days
        denom = (2.0 * float(window_days)) if window_days is not None else 2.0
        return float(lambda_stale * (float(f) / denom))


class WindowEstimator:
    """Estimates business relevance loss caused by window misalignment from reference."""

    @staticmethod
    def estimate_window_error(
        window_days: int,
        reference_days: int = 30,
        lambda_window: float = 0.2,
    ) -> float:
        """Dimensionless temporal relevance loss: lambda * |W - W_ref| / W_ref."""
        w_ref = max(1, reference_days)
        return float(lambda_window * (abs(window_days - w_ref) / float(w_ref)))


class UtilityEstimator:
    """Composite utility estimator evaluating candidates under unified normalized loss."""

    def __init__(
        self,
        weights: Optional[UtilityWeights] = None,
        public_prior: Optional[PublicPriorDistribution] = None,
        reference_days: int = 30,
        workload_weights: Optional[Dict[str, float]] = None,
        provider_shares: Optional[Dict[str, float]] = None,
        task_shares: Optional[Dict[str, float]] = None,
        lambda_stale: float = 1.0,
        lambda_window: float = 0.2,
    ):
        self.weights = weights or UtilityWeights()
        self.public_prior = public_prior or TruncatedNormalPrior()
        self.reference_days = int(reference_days)
        self.workload_weights = workload_weights or {
            "full_9_cell": 0.4,
            "provider": 0.3,
            "task": 0.2,
            "overall": 0.1,
        }
        self.provider_shares = provider_shares or {"provider_a": 0.5, "provider_b": 0.3, "provider_c": 0.2}
        self.task_shares = task_shares or {"technical": 0.4, "document": 0.35, "analysis": 0.25}
        self.lambda_stale = float(lambda_stale)
        self.lambda_window = float(lambda_window)

    def evaluate_candidate(
        self,
        metric: str,
        candidate: ReleasePolicyCandidate,
        population_size: int,
    ) -> UtilityComponents:
        """Compute full normalized dimensionless utility loss for a policy candidate."""
        w_days = candidate.window_days or self.reference_days
        window_prior = scale_prior_for_window(
            prior=self.public_prior,
            window_days=w_days,
            reference_days=self.reference_days,
        )

        n_pop = max(1, population_size)
        expected_user_volume = window_prior.mean
        expected_pop_volume = max(1.0, float(n_pop * expected_user_volume))

        # 1. Clipping bias
        if "volume" in metric.lower():
            raw_clip = ClippingEstimator.estimate_clipping_bias(
                contribution_bound=candidate.contribution_bound,
                population_size=n_pop,
                prior=window_prior,
            )
            norm_clip = raw_clip / expected_pop_volume
        else:
            raw_clip = 0.0
            norm_clip = 0.0

        # 2. DP Noise error
        raw_noise = NoiseEstimator.estimate_noise_error(
            metric=metric,
            candidate=candidate,
            population_size=n_pop,
        )
        if "volume" in metric.lower():
            norm_noise = raw_noise / expected_pop_volume
        elif "prevalence" in metric.lower():
            norm_noise = raw_noise / float(n_pop)  # rate error scale in [0, 1]
        elif "share" in metric.lower():
            norm_noise = raw_noise / float(n_pop)  # simplex share scale in [0, 1]
        else:
            norm_noise = raw_noise / expected_pop_volume

        # 3. Workload aggregation error
        norm_agg = GranularityEstimator.estimate_granularity_error(
            metric=metric,
            granularity=candidate.granularity,
            population_size=n_pop,
            base_scale=expected_user_volume,
            workload_weights=self.workload_weights,
            provider_shares=self.provider_shares,
            task_shares=self.task_shares,
        )
        raw_agg = norm_agg * expected_pop_volume

        # 4. Freshness / Staleness loss
        norm_stale = StalenessEstimator.estimate_staleness_error(
            frequency_days=candidate.frequency_days,
            window_days=w_days,
            lambda_stale=self.lambda_stale,
        )
        raw_stale = float(candidate.frequency_days)

        # 5. Window misalignment loss
        norm_win = WindowEstimator.estimate_window_error(
            window_days=w_days,
            reference_days=self.reference_days,
            lambda_window=self.lambda_window,
        )
        raw_win = float(abs(w_days - self.reference_days))

        # Unified Dimensionless Objective
        total_loss = (
            self.weights.clipping * norm_clip
            + self.weights.noise * norm_noise
            + self.weights.aggregation * norm_agg
            + self.weights.staleness * norm_stale
            + self.weights.window * norm_win
        )

        return UtilityComponents(
            clipping_error=raw_clip,
            noise_error=raw_noise,
            aggregation_error=raw_agg,
            staleness_error=raw_stale,
            window_error=raw_win,
            total_error=total_loss,
            norm_clipping_loss=norm_clip,
            norm_noise_loss=norm_noise,
            norm_aggregation_loss=norm_agg,
            norm_staleness_loss=norm_stale,
            norm_window_loss=norm_win,
        )
