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
import hashlib
import math
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
from scipy import integrate, special
from scipy.stats import norm, lognorm

from private_llm_analytics.mechanisms.gaussian import calibrate_analytic_gaussian
from private_llm_analytics.planning.candidates import ReleasePolicyCandidate
from private_llm_analytics.planning.policy import (
    build_aggregation_matrix,
    build_reconstruction_matrix,
    compute_workload_reconstruction_error,
    get_direct_granularity_sensitivity,
    get_direct_granularity_l1_sensitivity,
)
from private_llm_analytics.privacy.exceptions import ProductionSafetyConfigurationError


class PlannerConfigurationError(ProductionSafetyConfigurationError):
    """Raised when release planner configuration or candidate evaluation violates safety or scope guards."""
    pass


SUPPORTED_EXPECTED_RELEASE_L1_METRICS = {"interaction_volume"}


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

    objective_mode: str = "EXPECTED_RELEASE_L1"
    data: Optional[float] = None
    staleness: float = 0.5
    window: float = 0.2
    clipping: float = 1.0
    noise: float = 1.0
    aggregation: float = 1.0

    def __post_init__(self):
        resolved_data = self.data
        if resolved_data is None:
            if (
                self.clipping == 0
                and self.noise == 0
                and self.aggregation == 0
                and self.staleness == 0
                and self.window == 0
            ):
                resolved_data = 0.0
            else:
                resolved_data = 1.0
        object.__setattr__(self, "data", resolved_data)

        for name, val in [
            ("data", self.data),
            ("staleness", self.staleness),
            ("window", self.window),
            ("clipping", self.clipping),
            ("noise", self.noise),
            ("aggregation", self.aggregation),
        ]:
            if val is not None and val < 0:
                raise ValueError(f"Utility weight '{name}' must be >= 0, got {val}")
        if self.objective_mode == "EXPECTED_RELEASE_L1":
            total = (self.data or 0.0) + self.staleness + self.window
        else:
            total = self.clipping + self.noise + self.aggregation + self.staleness + self.window
        if total <= 0:
            raise ValueError("At least one active utility weight must be strictly positive.")


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
    data_error: float = 0.0

    # Normalized dimensionless losses in [0, inf)
    norm_clipping_loss: float = 0.0
    norm_noise_loss: float = 0.0
    norm_aggregation_loss: float = 0.0
    norm_staleness_loss: float = 0.0
    norm_window_loss: float = 0.0
    norm_data_loss: float = 0.0

    def to_dict(self) -> Dict[str, float]:
        return {
            "clipping": self.clipping_error,
            "noise": self.noise_error,
            "aggregation": self.aggregation_error,
            "staleness": self.staleness_error,
            "window": self.window_error,
            "total": self.total_error,
            "data": self.data_error,
            "norm_clipping": self.norm_clipping_loss,
            "norm_noise": self.norm_noise_loss,
            "norm_aggregation": self.norm_aggregation_loss,
            "norm_staleness": self.norm_staleness_loss,
            "norm_window": self.norm_window_loss,
            "norm_data": self.norm_data_loss,
        }


class PublicPriorDistribution(ABC):
    """Abstract base class for public prior distributions over user interaction norms."""

    mean: float
    std: float

    @abstractmethod
    def expected_excess(self, threshold: float) -> float:
        """Compute E[(R - C)_+] under the prior distribution."""
        pass

    def expected_folded_normal_release_l1(
        self,
        metric: str,
        candidate: ReleasePolicyCandidate,
        population_size: int,
        granularity: Optional[str] = None,
        window_days: Optional[int] = None,
        workload_weights: Optional[Dict[str, float]] = None,
    ) -> Tuple[float, float]:
        return 0.0, 0.0

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


class PublicContributionPrior(PublicPriorDistribution):
    """Public generative contribution prior modeling target-space user norm distributions.

    Models the joint 9-cell interaction vector m_u under public reference parameters:
      - Total interactions T ~ TruncatedNormal(mean, std, min_val, max_val) (or LogNormal)
      - Category distribution: provider preferences conditional on tasks (via sample_user_profiles)
        with cell shape control modulated by provider_shares and task_shares
      - Target-space linear aggregation A_G:
          R_G = ||A_G m_u||_2
      - User total count T = ||A_G m_u||_1
      - Expected target-space clipping excess:
          E[ T * max(0, 1 - C / R_G) ]
      - Expected end-to-end L1 release error via exact folded-normal closed form on 9-cell workload

    Evaluated via deterministic public Monte Carlo with a fixed public seed (default 900001).
    Strictly isolated from private runtime data.
    """

    def __init__(
        self,
        mean: float = 60.0,
        std: float = 20.0,
        min_val: float = 5.0,
        max_val: float = 150.0,
        distribution: str = "truncated_normal",
        mean_scaling: str = "linear",
        std_scaling: str = "sqrt_time",
        reference_days: int = 30,
        window_days: int = 30,
        providers: Optional[List[str]] = None,
        tasks: Optional[List[str]] = None,
        provider_shares: Optional[Dict[str, float]] = None,
        task_shares: Optional[Dict[str, float]] = None,
        public_seed: int = 900001,
        num_reference_users: int = 2000,
    ):
        if std <= 0:
            raise ValueError(f"std must be > 0, got {std}")
        if min_val >= max_val:
            raise ValueError(f"min_val ({min_val}) must be < max_val ({max_val})")
        if reference_days <= 0 or window_days <= 0:
            raise ValueError("reference_days and window_days must be > 0")

        self.mean = float(mean)
        self.std = float(std)
        self.min_val = float(min_val)
        self.max_val = float(max_val)
        self.distribution = str(distribution).lower().strip()
        self.mean_scaling = str(mean_scaling).lower().strip()
        self.std_scaling = str(std_scaling).lower().strip()
        self.reference_days = int(reference_days)
        self.window_days = int(window_days)
        self.providers = list(providers or ["provider_a", "provider_b", "provider_c"])
        self.tasks = list(tasks or ["technical", "document", "analysis"])
        self.provider_shares = dict(provider_shares) if provider_shares is not None else {p: 1.0 / len(self.providers) for p in self.providers}
        self.task_shares = dict(task_shares) if task_shares is not None else {t: 1.0 / len(self.tasks) for t in self.tasks}
        self.public_seed = int(public_seed)
        self.num_reference_users = int(num_reference_users)

        self._generate_reference_sample()

    def _generate_reference_sample(self):
        # 1. Compute window scaling factors
        rho = float(self.window_days) / float(self.reference_days)
        if self.mean_scaling == "sublinear":
            mu_scale = rho ** 0.75
        else:
            mu_scale = rho

        if self.std_scaling == "linear":
            std_scale = rho
        else:
            std_scale = math.sqrt(rho)

        scaled_mean = self.mean * mu_scale
        scaled_std = self.std * std_scale
        scaled_min = max(1.0, self.min_val * mu_scale)
        scaled_max = max(scaled_min + 1.0, self.max_val * mu_scale)

        # 2. Derive deterministic window-specific public seed
        shape_repr = f"{sorted(self.provider_shares.items())}_{sorted(self.task_shares.items())}"
        seed_raw = f"{self.public_seed}_{self.window_days}_{self.mean}_{self.std}_{self.min_val}_{self.max_val}_{self.distribution}_{self.mean_scaling}_{self.std_scaling}_{shape_repr}"
        seed_int = int(hashlib.sha256(seed_raw.encode("utf-8")).hexdigest()[:8], 16)
        rng = np.random.default_rng(seed_int)

        # 3. Sample window-specific user interaction counts
        if self.distribution == "lognormal":
            var = scaled_std ** 2
            sigma_ln = math.sqrt(math.log(1.0 + var / (scaled_mean ** 2)))
            mu_ln = math.log(scaled_mean) - 0.5 * (sigma_ln ** 2)
            raw = rng.lognormal(mean=mu_ln, sigma=sigma_ln, size=self.num_reference_users)
        else:
            a = (scaled_min - scaled_mean) / scaled_std
            b = (scaled_max - scaled_mean) / scaled_std
            from scipy.stats import truncnorm
            raw = truncnorm.rvs(a, b, loc=scaled_mean, scale=scaled_std, size=self.num_reference_users, random_state=rng)

        counts = np.clip(np.rint(raw).astype(np.int64), int(scaled_min), int(scaled_max))

        # 4. Sample profiles and apply provider_shares / task_shares shape control (Section I)
        from private_llm_analytics.simulator.users import sample_user_profiles
        _, joint_probs = sample_user_profiles(self.num_reference_users, counts, self.providers, self.tasks, rng)

        cell_weights = np.array([
            self.provider_shares.get(p, 1.0 / len(self.providers)) * self.task_shares.get(t, 1.0 / len(self.tasks))
            for p in self.providers
            for t in self.tasks
        ], dtype=np.float64)
        cell_weights = cell_weights / np.sum(cell_weights)
        joint_probs = joint_probs * cell_weights
        joint_probs = joint_probs / np.sum(joint_probs, axis=1, keepdims=True)

        # 5. Direct window counts matrix (NO simple scalar matrix multiplication!)
        base_mat = np.zeros((self.num_reference_users, 9), dtype=np.float64)
        for u in range(self.num_reference_users):
            base_mat[u] = rng.multinomial(counts[u], joint_probs[u])

        self.ref_matrix = base_mat
        self.ref_totals = np.sum(self.ref_matrix, axis=1)
        self.expected_user_volume = float(np.mean(self.ref_totals))
        self.expected_raw_vector = np.mean(self.ref_matrix, axis=0)
        self.cell_names = [f"{p}:{t}" for p in self.providers for t in self.tasks]

        # Precompute target-space norms for each granularity via single-source aggregation operator
        self._norms: Dict[str, np.ndarray] = {}
        # 1. full_9_cell
        self._norms["full_9_cell"] = np.sqrt(np.sum(self.ref_matrix**2, axis=1))

        # 2. provider_3_cell
        A_prov, _ = build_aggregation_matrix("provider_3_cell", self.cell_names)
        self._norms["provider_3_cell"] = np.sqrt(np.sum((self.ref_matrix @ A_prov.T)**2, axis=1))

        # 3. task_3_cell
        A_task, _ = build_aggregation_matrix("task_3_cell", self.cell_names)
        self._norms["task_3_cell"] = np.sqrt(np.sum((self.ref_matrix @ A_task.T)**2, axis=1))

        # 4. overall_1_cell
        self._norms["overall_1_cell"] = self.ref_totals.copy()

        # Cache for expected excess and folded normal
        self._excess_cache: Dict[Tuple[float, str], float] = {}
        self._folded_normal_cache: Dict[Tuple[float, str, int, float, float], Tuple[float, float]] = {}

    def expected_excess(
        self,
        threshold: float,
        granularity: str = "full_9_cell",
        window_days: Optional[int] = None,
    ) -> float:
        """Compute E[ T_u * max(0, 1 - C / R_{G,u}) ]. Monotonically non-increasing in threshold."""
        if window_days is not None and window_days != self.window_days:
            scaled = self.with_window(window_days)
            return scaled.expected_excess(threshold, granularity=granularity)

        c = float(threshold)
        if c <= 0:
            raise ValueError(f"threshold must be > 0, got {c}")

        cache_key = (c, granularity)
        if cache_key in self._excess_cache:
            return self._excess_cache[cache_key]

        norms = self._norms.get(granularity, self._norms["full_9_cell"])
        excess_factor = np.maximum(0.0, 1.0 - c / np.maximum(norms, 1e-9))
        excess_l1 = self.ref_totals * excess_factor
        val = float(np.mean(excess_l1))
        self._excess_cache[cache_key] = val
        return val

    def expected_norm_clipping_loss(
        self,
        threshold: float,
        granularity: str = "full_9_cell",
        window_days: Optional[int] = None,
    ) -> float:
        excess = self.expected_excess(threshold, granularity=granularity, window_days=window_days)
        denom = max(1.0, self.expected_user_volume)
        return float(excess / denom)

    def expected_folded_normal_release_l1(
        self,
        metric: str,
        candidate: ReleasePolicyCandidate,
        population_size: int,
        granularity: Optional[str] = None,
        window_days: Optional[int] = None,
        workload_weights: Optional[Dict[str, float]] = None,
    ) -> Tuple[float, float]:
        """Compute public-model expected end-to-end L1 release error via folded-normal closed form."""
        if window_days is not None and window_days != self.window_days:
            scaled = self.with_window(window_days)
            return scaled.expected_folded_normal_release_l1(
                metric=metric,
                candidate=candidate,
                population_size=population_size,
                granularity=granularity,
                workload_weights=workload_weights,
            )

        # Safety & Scope guards (Tasks E and F)
        if candidate.mechanism == "laplace":
            raise PlannerConfigurationError(
                "v0.3 EXPECTED_RELEASE_L1 currently supports Gaussian only."
            )
        norm_metric = metric.lower().strip()
        if norm_metric not in SUPPORTED_EXPECTED_RELEASE_L1_METRICS:
            raise PlannerConfigurationError(
                f"Unsupported metric '{metric}' under EXPECTED_RELEASE_L1. "
                f"v0.3 utility-aware parameter optimization is formally evaluated and production-enabled for "
                f"Interaction Volume only ({SUPPORTED_EXPECTED_RELEASE_L1_METRICS})."
            )

        n_pop = max(1, population_size)
        c = float(candidate.contribution_bound) if candidate.contribution_bound is not None else 20.0
        g = granularity or (candidate.granularity if candidate else None) or "full_9_cell"
        eps = float(candidate.epsilon_per_release)
        delta = float(candidate.delta_per_release)

        w_dict = workload_weights or {
            "full_9_cell": 0.4,
            "provider": 0.3,
            "task": 0.2,
            "overall": 0.1,
        }
        w9 = float(w_dict.get("full_9_cell", 0.0))
        wp = float(w_dict.get("provider", 0.0))
        wt = float(w_dict.get("task", 0.0))
        wo = float(w_dict.get("overall", 0.0))
        tot_w = w9 + wp + wt + wo
        if tot_w <= 0:
            w9 = 1.0
            tot_w = 1.0

        cache_key = (c, g, n_pop, eps, delta, w9, wp, wt, wo)
        if cache_key in self._folded_normal_cache:
            return self._folded_normal_cache[cache_key]

        # 1. Target sensitivity and noise std
        sens = get_direct_granularity_sensitivity(metric, g, c)
        if candidate.mechanism == "gaussian":
            sigma = calibrate_analytic_gaussian(eps, delta, sens)
        else:
            raise PlannerConfigurationError("v0.3 EXPECTED_RELEASE_L1 currently supports Gaussian only.")

        # 2. Reconstructed clipped vectors via single-source R_G and A_G
        norms = self._norms.get(g, self._norms["full_9_cell"])
        clip_scale = np.minimum(1.0, c / np.maximum(norms, 1e-9))

        R_G = build_reconstruction_matrix(
            source_granularity=g,
            target_9_cell_names=self.cell_names,
            provider_shares=self.provider_shares,
            task_shares=self.task_shares,
        )
        A_G, _ = build_aggregation_matrix(g, self.cell_names)
        user_target_mat = self.ref_matrix @ A_G.T
        clipped_target = user_target_mat * clip_scale[:, np.newaxis]
        recon_mat = clipped_target @ R_G.T

        # 3. Population bias vector in 9-cell space
        b9 = float(n_pop) * (np.mean(recon_mat, axis=0) - self.expected_raw_vector)

        # 4. Workload queries: Q_9 (9x9), Q_p (3x9), Q_t (3x9), Q_o (1x9)
        Q9, _ = build_aggregation_matrix("full_9_cell", self.cell_names)
        Qp, _ = build_aggregation_matrix("provider_3_cell", self.cell_names)
        Qt, _ = build_aggregation_matrix("task_3_cell", self.cell_names)
        Qo, _ = build_aggregation_matrix("overall_1_cell", self.cell_names)

        query_specs = [
            ("full_9_cell", w9, Q9),
            ("provider", wp, Qp),
            ("task", wt, Qt),
            ("overall", wo, Qo),
        ]

        total_weighted_error = 0.0
        for q_name, w_q, Q in query_specs:
            if w_q <= 0:
                continue
            b_Q = Q @ b9
            M_Q = Q @ R_G
            d_Q = len(Q)
            q_l1 = 0.0
            for k in range(d_Q):
                sig_k = float(sigma * np.sqrt(np.sum(M_Q[k] ** 2)))
                beta_k = float(abs(b_Q[k]))
                if sig_k <= 1e-12:
                    e_abs = beta_k
                else:
                    z = beta_k / (math.sqrt(2.0) * sig_k)
                    e_abs = sig_k * math.sqrt(2.0 / math.pi) * math.exp(-(beta_k ** 2) / (2.0 * sig_k ** 2)) + beta_k * float(special.erf(z))
                q_l1 += e_abs
            total_weighted_error += w_q * q_l1

        total_data_error = total_weighted_error / tot_w

        # 5. Normalized loss
        expected_pop_vol = max(1.0, float(n_pop * self.expected_user_volume))
        norm_data_loss = float(total_data_error / expected_pop_vol)

        res = (float(total_data_error), float(norm_data_loss))
        self._folded_normal_cache[cache_key] = res
        return res

    def with_window(self, window_days: int, reference_days: Optional[int] = None) -> "PublicContributionPrior":
        ref_days = reference_days if reference_days is not None else self.reference_days
        if window_days == self.window_days and ref_days == self.reference_days:
            return self
        return PublicContributionPrior(
            mean=self.mean,
            std=self.std,
            min_val=self.min_val,
            max_val=self.max_val,
            distribution=self.distribution,
            mean_scaling=self.mean_scaling,
            std_scaling=self.std_scaling,
            reference_days=ref_days,
            window_days=window_days,
            providers=self.providers,
            tasks=self.tasks,
            provider_shares=self.provider_shares,
            task_shares=self.task_shares,
            public_seed=self.public_seed,
            num_reference_users=self.num_reference_users,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "contribution_model",
            "distribution": self.distribution,
            "mean": self.mean,
            "std": self.std,
            "min_val": self.min_val,
            "max_val": self.max_val,
            "mean_scaling": self.mean_scaling,
            "std_scaling": self.std_scaling,
            "reference_days": self.reference_days,
            "window_days": self.window_days,
            "public_seed": self.public_seed,
            "num_reference_users": self.num_reference_users,
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

    if isinstance(prior, PublicContributionPrior):
        return prior.with_window(window_days, reference_days)

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
        granularity: str = "full_9_cell",
        window_days: int = 30,
    ) -> float:
        """Estimate E_clip(C) = N * E[excess]."""
        if contribution_bound is None:
            return 0.0
        c = float(contribution_bound)
        if c <= 0:
            raise ValueError(f"contribution_bound must be > 0, got {c}")
        if isinstance(prior, PublicContributionPrior):
            excess = prior.expected_excess(c, granularity=granularity, window_days=window_days)
        else:
            try:
                excess = prior.expected_excess(c, granularity=granularity, window_days=window_days)
            except TypeError:
                excess = prior.expected_excess(c)
        return float(population_size * excess)


class NoiseEstimator:
    """Estimates DP noise standard deviation and expected absolute error."""

    @staticmethod
    def estimate_noise_sigma(
        metric: str,
        candidate: ReleasePolicyCandidate,
    ) -> Tuple[float, float]:
        """Calibrate noise scale (sigma for Gaussian, b for Laplace) and sensitivity."""
        eps = float(candidate.epsilon_per_release)
        delta = float(candidate.delta_per_release)

        if candidate.mechanism == "gaussian":
            sens = get_direct_granularity_sensitivity(
                metric=metric,
                granularity=candidate.granularity,
                clip_norm=candidate.contribution_bound,
            )
            sigma = calibrate_analytic_gaussian(
                epsilon=eps,
                delta=delta,
                l2_sensitivity=sens,
            )
            return float(sigma), float(sens)
        elif candidate.mechanism == "laplace":
            sens_l1 = get_direct_granularity_l1_sensitivity(
                metric=metric,
                granularity=candidate.granularity,
                clip_norm=candidate.contribution_bound,
            )
            b = sens_l1 / eps
            return float(b), float(sens_l1)
        else:
            raise ValueError(f"Unsupported mechanism '{candidate.mechanism}'")

    @staticmethod
    def estimate_noise_error(
        metric: str,
        candidate: ReleasePolicyCandidate,
        population_size: int,
    ) -> float:
        """Estimate expected absolute noise error according to direct metric semantics."""
        scale, _ = NoiseEstimator.estimate_noise_sigma(metric, candidate)
        dim = GRANULARITY_DIMENSIONS.get(candidate.granularity, 9)
        metric_norm = metric.lower().strip()

        # Expected 1D absolute error:
        # Gaussian N(0, sigma^2) => E|Z| = sigma * sqrt(2 / pi)
        # Laplace(0, b) with scale b => E|Z| = b
        if candidate.mechanism == "laplace":
            e_abs_1d = float(scale)
        else:
            e_abs_1d = float(scale * math.sqrt(2.0 / math.pi))

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


def compute_temporal_utility_components(
    frequency_days: int,
    window_days: Optional[int] = None,
    reference_days: int = 30,
    lambda_stale: float = 1.0,
    lambda_window: float = 1.0,
) -> Tuple[float, float]:
    """Single source of truth for normalized dimensionless temporal utility components.

    Returns:
        (norm_staleness_loss, norm_window_loss)
    where:
        norm_staleness_loss = lambda_stale * (frequency_days / (2.0 * window_days))
        norm_window_loss = lambda_window * (abs(window_days - reference_days) / float(reference_days))
    By default lambda_stale=1.0 and lambda_window=1.0, so dimensionless penalties are
    F/(2W) and |W - Wref|/Wref, and all business weighting is governed strictly
    by weights.staleness and weights.window.
    """
    denom = (2.0 * float(window_days)) if (window_days is not None and window_days > 0) else 2.0
    norm_stale = float(lambda_stale * (float(frequency_days) / denom))
    w_ref = max(1, reference_days)
    w_val = window_days if window_days is not None else reference_days
    norm_win = float(lambda_window * (abs(w_val - w_ref) / float(w_ref)))
    return norm_stale, norm_win


def compute_total_expected_release_utility(
    data_loss: float,
    staleness_loss: float,
    window_loss: float,
    weights: UtilityWeights,
) -> float:
    """Single source of truth for total expected release utility under active objective mode."""
    if weights.objective_mode == "EXPECTED_RELEASE_L1":
        return float(
            weights.data * data_loss
            + weights.staleness * staleness_loss
            + weights.window * window_loss
        )
    else:
        return float(
            weights.clipping * data_loss
            + weights.staleness * staleness_loss
            + weights.window * window_loss
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
        """Dimensionless reporting freshness loss: lambda_stale * F / (2 * W)."""
        if isinstance(frequency_or_metric, str):
            f = frequency_days if frequency_days is not None else 7
        else:
            f = frequency_or_metric if frequency_days is None else frequency_days
        norm_stale, _ = compute_temporal_utility_components(
            frequency_days=f,
            window_days=window_days,
            reference_days=30,
            lambda_stale=lambda_stale,
            lambda_window=1.0,
        )
        return norm_stale


class WindowEstimator:
    """Estimates business relevance loss caused by window misalignment from reference."""

    @staticmethod
    def estimate_window_error(
        window_days: int,
        reference_days: int = 30,
        lambda_window: float = 1.0,
    ) -> float:
        """Dimensionless temporal relevance loss: lambda_window * |W - W_ref| / W_ref."""
        _, norm_win = compute_temporal_utility_components(
            frequency_days=7,
            window_days=window_days,
            reference_days=reference_days,
            lambda_stale=1.0,
            lambda_window=lambda_window,
        )
        return norm_win


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
        lambda_window: float = 1.0,
    ):
        self.weights = weights or UtilityWeights()
        self.public_prior = public_prior or PublicContributionPrior(
            reference_days=int(reference_days),
            window_days=int(reference_days),
        )
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
        if self.weights.objective_mode == "EXPECTED_RELEASE_L1":
            if candidate.mechanism == "laplace":
                raise PlannerConfigurationError(
                    "v0.3 EXPECTED_RELEASE_L1 currently supports Gaussian only."
                )
            norm_metric = metric.lower().strip()
            if norm_metric not in SUPPORTED_EXPECTED_RELEASE_L1_METRICS:
                raise PlannerConfigurationError(
                    f"Unsupported metric '{metric}' under EXPECTED_RELEASE_L1. "
                    f"v0.3 utility-aware parameter optimization is formally evaluated and production-enabled for "
                    f"Interaction Volume only ({SUPPORTED_EXPECTED_RELEASE_L1_METRICS})."
                )

        w_days = candidate.window_days or self.reference_days
        window_prior = scale_prior_for_window(
            prior=self.public_prior,
            window_days=w_days,
            reference_days=self.reference_days,
        )

        n_pop = max(1, population_size)
        expected_user_volume = (
            window_prior.expected_user_volume
            if hasattr(window_prior, "expected_user_volume")
            else window_prior.mean
        )
        expected_pop_volume = max(1.0, float(n_pop * expected_user_volume))

        # 1. Clipping bias
        if "volume" in metric.lower():
            raw_clip = ClippingEstimator.estimate_clipping_bias(
                contribution_bound=candidate.contribution_bound,
                population_size=n_pop,
                prior=window_prior,
                granularity=candidate.granularity,
                window_days=w_days,
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

        # 4 & 5. Temporal losses via shared single source of truth
        norm_stale, norm_win = compute_temporal_utility_components(
            frequency_days=candidate.frequency_days,
            window_days=w_days,
            reference_days=self.reference_days,
            lambda_stale=self.lambda_stale,
            lambda_window=self.lambda_window,
        )
        raw_stale = float(candidate.frequency_days)
        raw_win = float(abs(w_days - self.reference_days))

        # 6. End-to-end data error via folded-normal expectation
        data_error = 0.0
        norm_data_loss = 0.0
        if hasattr(window_prior, "expected_folded_normal_release_l1"):
            data_error, norm_data_loss = window_prior.expected_folded_normal_release_l1(
                metric=metric,
                candidate=candidate,
                population_size=n_pop,
                granularity=candidate.granularity,
                window_days=w_days,
                workload_weights=self.workload_weights,
            )

        # Unified Dimensionless Objective via shared helper
        if self.weights.objective_mode == "EXPECTED_RELEASE_L1" and (norm_data_loss > 0.0 or data_error > 0.0 or (norm_clip == 0.0 and norm_noise == 0.0)):
            total_loss = compute_total_expected_release_utility(
                data_loss=norm_data_loss,
                staleness_loss=norm_stale,
                window_loss=norm_win,
                weights=self.weights,
            )
        else:
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
            data_error=data_error,
            norm_clipping_loss=norm_clip,
            norm_noise_loss=norm_noise,
            norm_aggregation_loss=norm_agg,
            norm_staleness_loss=norm_stale,
            norm_window_loss=norm_win,
            norm_data_loss=norm_data_loss,
        )
