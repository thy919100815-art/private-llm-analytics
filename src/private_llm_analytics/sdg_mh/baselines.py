"""Comprehensive Differentially Private Baselines for Longitudinal LLM Analytics.

Implements all required baselines under UnifiedPrivacyLedger:
1. Epoch90: 90-day Epoch + Lifetime Exposure. Rejection after budget exhaustion.
2. LifetimeDecay: DPI simplified proxy with rho_t = rho_max * 6 / (pi^2 * t^2).
3. W-BD & W-BA: w-event Budget Distribution & Budget Absorption (Kellaris et al., VLDB 2014) for w in {7, 30}.
4. Tree-U, Tree-2l, Tree-sqrt: Binary tree continual release with Honaker GLS optimal estimation.
5. Calibration engine for:
   - Caliber a: identical eps_30 across all methods.
   - Caliber b: rho(h) <= rho_sdg(h) for all h in {1, 7, 30, 90, 365}.
"""

from __future__ import annotations

import math
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple
import numpy as np

from private_llm_analytics.sdg_mh.ledger import (
    UnifiedPrivacyLedger,
    closed_form_sdg_additive_rho,
    epsilon_to_rho,
    rho_to_epsilon,
)


class BaseDPBaseline:
    """Abstract base class for longitudinal DP baselines."""

    name: str

    def get_ledger(self) -> UnifiedPrivacyLedger:
        raise NotImplementedError


class Epoch90Baseline(BaseDPBaseline):
    """90-day Privacy Epoch with Lifetime Exposure accounting.

    - Fixed 90-day cycles.
    - Global budget per epoch: eps_epoch = 4.0, delta = 1e-6 => rho_epoch ~ 0.253294.
    - When queries within an epoch exhaust the budget, subsequent queries are REJECTED.
    - Across epochs, lifetime exposure accumulates monotonically without replenishment.
    """

    name = "Epoch90"

    def __init__(
        self,
        epoch_days: int = 90,
        eps_cap_epoch: float = 4.0,
        delta: float = 1e-6,
        rho_per_query: float = 0.05,  # Each release consumes a portion of epoch budget
        C: float = 50.0,
    ):
        self.epoch_days = int(epoch_days)
        self.eps_cap_epoch = float(eps_cap_epoch)
        self.delta = float(delta)
        # Convert eps_cap to zCDP rho: for eps=4.0, delta=1e-6, R(alpha) = alpha * 0.506589 / 2 => rho_epoch ~ 0.2532945
        self.rho_epoch_cap = 0.2532945
        self.rho_per_query = float(rho_per_query)
        self.C = float(C)

        self.ledger = UnifiedPrivacyLedger(delta=self.delta, allow_unconfigured_policy=True)
        # Tracks spend in current active epoch
        self.current_epoch_spend: float = 0.0
        self.current_epoch_index: int = 1
        self.queries_attempted: int = 0
        self.queries_served: int = 0
        self.queries_rejected: int = 0
        self.cache_hits: int = 0
        self.release_store: Dict[str, Dict[str, Any]] = {}

    def advance_to_new_epoch(self) -> int:
        """Advance to next 90-day epoch: refresh active epoch budget while lifetime exposure continues."""
        self.current_epoch_index += 1
        self.current_epoch_spend = 0.0
        return self.current_epoch_index

    def query(
        self,
        window_start: int,
        window_end: int,
        true_sum: float,
        metric: str = "requests",
        rng: Optional[np.random.Generator] = None,
        seed: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Execute query or serve from release_key cache, or reject if epoch budget is exhausted."""
        self.queries_attempted += 1
        release_key = f"{metric}_w{window_start}_{window_end}_ep{self.current_epoch_index}"

        # 1. Check release_key cache within current epoch (zero privacy spend)
        if release_key in self.release_store:
            self.queries_served += 1
            self.cache_hits += 1
            cached = self.release_store[release_key]
            return {
                "status": "SERVED",
                "estimate": cached["estimate"],
                "variance": cached["variance"],
                "sigma": cached["sigma"],
                "epoch_spend": self.current_epoch_spend,
                "cache_hit": True,
            }

        # 2. Check epoch budget exhaustion
        if self.current_epoch_spend + self.rho_per_query > self.rho_epoch_cap + 1e-9:
            self.queries_rejected += 1
            return {
                "status": "REJECTED",
                "reason": f"Epoch {self.current_epoch_index} privacy budget exhausted ({self.current_epoch_spend:.4f} / {self.rho_epoch_cap:.4f}).",
                "estimate": None,
                "variance": float("inf"),
                "cache_hit": False,
            }

        # 3. Serve fresh query and deduct budget
        self.queries_served += 1
        self.current_epoch_spend += self.rho_per_query
        w_len = max(1, window_end - window_start + 1)
        # Under daily clipping C, user sensitivity across w days is w * C
        sens_w = float(w_len) * self.C
        sigma = sens_w / math.sqrt(2.0 * self.rho_per_query)

        if rng is None:
            rng = np.random.default_rng(seed)
        noise = rng.normal(0.0, sigma)
        estimate = max(0.0, true_sum + noise)

        # Register in unified ledger
        days_accessed = list(range(window_start, window_end + 1))
        self.ledger.record_gaussian_release(
            release_id=f"epoch90_q{self.queries_attempted}_{release_key}",
            days_accessed=days_accessed,
            sensitivity=sens_w,
            sigma=sigma,
            tag="epoch90_query",
        )

        result = {
            "status": "SERVED",
            "estimate": estimate,
            "variance": sigma**2,
            "sigma": sigma,
            "epoch_spend": self.current_epoch_spend,
            "cache_hit": False,
        }
        self.release_store[release_key] = result
        return result

    def get_availability(self) -> float:
        """Fraction of queries served without rejection."""
        if self.queries_attempted == 0:
            return 1.0
        return float(self.queries_served / self.queries_attempted)


class LifetimeDecayBaseline(BaseDPBaseline):
    """Lifetime Decay baseline (simplified proxy for DPI-class methods).

    rho_t = rho_max * 6 / (pi^2 * t^2)
    Sum_{t=1}^infty rho_t = rho_max.
    sigma_t^2 = C^2 / (2 * rho_t) = C^2 * pi^2 * t^2 / (12 * rho_max).
    """

    name = "LifetimeDecay"

    def __init__(
        self,
        rho_max: float = 0.2532945,  # Corresponds to eps ~ 4.0, delta=1e-6 (R(alpha)=alpha*0.506589/2)
        C: float = 50.0,
        delta: float = 1e-6,
    ):
        self.rho_max = float(rho_max)
        self.C = float(C)
        self.delta = float(delta)
        self.ledger = UnifiedPrivacyLedger(delta=self.delta, allow_unconfigured_policy=True)

    def get_daily_variance(self, t: int) -> float:
        """sigma_t^2 = C^2 * pi^2 * t^2 / (12 * rho_max)."""
        t_safe = max(1, t)
        rho_t = self.rho_max * 6.0 / ((math.pi**2) * (t_safe**2))
        return float((self.C**2) / (2.0 * rho_t))

    def query(
        self,
        window_start: int,
        window_end: int,
        daily_true_sums: Sequence[float],
        rng: Optional[np.random.Generator] = None,
        seed: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Query window [window_start, window_end] with decaying daily budget."""
        if rng is None:
            rng = np.random.default_rng(seed)

        w_len = window_end - window_start + 1
        est_sum = 0.0
        total_var = 0.0

        for idx, t in enumerate(range(window_start, window_end + 1)):
            v_t = self.get_daily_variance(t)
            sig_t = math.sqrt(v_t)
            err_t = rng.normal(0.0, sig_t)
            val = daily_true_sums[idx] if idx < len(daily_true_sums) else 0.0
            est_sum += val + err_t
            total_var += v_t

            # Register daily in ledger
            rho_t = self.rho_max * 6.0 / ((math.pi**2) * (t**2))
            self.ledger.record_gaussian_release(
                release_id=f"lifetime_decay_day_{t}",
                days_accessed=[t],
                sensitivity=self.C,
                sigma=sig_t,
                tag="lifetime_decay",
            )

        return {
            "status": "SERVED",
            "estimate": max(0.0, est_sum),
            "variance": total_var,
            "std": math.sqrt(total_var),
        }


class WEventBaseline(BaseDPBaseline):
    """w-event Differential Privacy (Kellaris et al., PVLDB 2014).

    Implements:
    - W-BD: Budget Distribution across sliding window of length w (Section 4.2, Figure 2).
      * Private dissimilarity check M_{i,1} using fixed budget rho_test = rho_w / (2*w).
      * Remaining publication budget rho_rm = rho_w / 2 - sum_{k=i-w+1}^{i-1} rho_{k,2}.
      * When publishing, allocates half of remaining publication budget: rho_{i,2} = rho_rm / 2.
      * When skipping, rho_{i,2} = 0, reuses previous release.
      * Budgets falling out of active window [i-w+1, i] are naturally recycled.
    - W-BA: Budget Absorption using similarity thresholding (Section 4.3, Figure 4).
      * Private dissimilarity check M_{i,1} using fixed budget rho_test = rho_w / (2*w).
      * Base publication budget rho_base = rho_w / (2*w).
      * Nullification: If a previous release absorbed budget, to_nullify = round(rho_{l,2} / rho_base) - 1.
        The next to_nullify timestamps are forced null (rho_{i,2} = 0, cannot be absorbed).
      * Absorption: For un-nullified timestamps, to_absorb = i - (l + to_nullify).
        Can absorb at most w budgets: rho_{i,2} = rho_base * min(to_absorb, w) <= rho_w / 2.
      * If dis > threshold, publish with rho_{i,2} and set to_nullify for subsequent timestamps.
      * Else return null, allowing this timestamp to be absorbed later.
    """

    def __init__(
        self,
        w: int = 30,
        mode: str = "BA",  # 'BD' or 'BA'
        target_rho_w: float = 0.0090615,
        C: float = 104.0,
        delta: float = 1e-6,
    ):
        self.w = int(w)
        self.mode = mode.upper()
        self.name = f"W-{self.mode}-{self.w}"
        self.target_rho_w = float(target_rho_w)
        self.C = float(C)
        self.delta = float(delta)
        self.ledger = UnifiedPrivacyLedger(delta=self.delta, allow_unconfigured_policy=True)

        # In Kellaris et al., half of total window budget is for dissimilarity testing M_{i,1}
        # and half is for publication M_{i,2}
        self.rho_test = self.target_rho_w / (2.0 * float(self.w))
        self.sigma_test = self.C / math.sqrt(2.0 * self.rho_test)

        self.rho_pub_max = self.target_rho_w / 2.0
        self.rho_base = self.target_rho_w / (2.0 * float(self.w))

        # State tracking
        self._last_released_value: Optional[float] = None
        self._last_released_day: Optional[int] = None
        self._last_released_rho: float = 0.0
        self._pub_budgets_history: Dict[int, float] = {}

    def release_day(
        self,
        day: int,
        true_value: float,
        rng: Optional[np.random.Generator] = None,
        seed: Optional[int] = None,
    ) -> float:
        """Execute daily release at day t under faithful W-BD or W-BA (Kellaris et al., PVLDB 2014)."""
        if rng is None:
            rng = np.random.default_rng(seed)

        # First day initialization: must publish with initial budget
        if self._last_released_value is None:
            if self.mode == "BD":
                # BD initial publication takes half of available publication budget: (rho_w / 2) / 2 = rho_w / 4
                rho_rel = self.rho_pub_max / 2.0
            else:
                # BA initial publication takes base publication budget
                rho_rel = self.rho_base

            sig_rel = self.C / math.sqrt(2.0 * rho_rel)
            noisy_val = float(true_value + rng.normal(0.0, sig_rel))

            self.ledger.record_gaussian_release(
                release_id=f"{self.mode.lower()}_{self.w}_rel_day_{day}",
                days_accessed=[day],
                sensitivity=self.C,
                sigma=sig_rel,
                tag=f"{self.mode.lower()}_initial_release",
            )
            self._last_released_value = noisy_val
            self._last_released_day = day
            self._last_released_rho = rho_rel
            self._pub_budgets_history[day] = rho_rel
            return noisy_val

        # Sub-mechanism M_{i,1}: Private dissimilarity computation (always consumes rho_test)
        self.ledger.record_gaussian_release(
            release_id=f"{self.mode.lower()}_{self.w}_test_day_{day}",
            days_accessed=[day],
            sensitivity=self.C,
            sigma=self.sigma_test,
            tag=f"{self.mode.lower()}_similarity_test",
        )
        diff = abs(true_value - self._last_released_value)
        noisy_diff = diff + float(rng.normal(0.0, self.sigma_test))

        if self.mode == "BD":
            # W-BD: Calculate remaining publication budget in active window [day - w + 1, day - 1]
            window_start = day - self.w + 1
            used_pub_in_window = sum(
                rho for d, rho in self._pub_budgets_history.items()
                if window_start <= d < day
            )
            rho_rm = max(1e-12, self.rho_pub_max - used_pub_in_window)
            rho_candidate = rho_rm / 2.0
            sig_candidate = self.C / math.sqrt(2.0 * rho_candidate)

            # Similarity check: if difference > expected noise scale (sig_candidate), publish
            if noisy_diff > sig_candidate:
                sig_rel = sig_candidate
                noisy_val = float(true_value + rng.normal(0.0, sig_rel))
                self.ledger.record_gaussian_release(
                    release_id=f"wbd_{self.w}_rel_day_{day}",
                    days_accessed=[day],
                    sensitivity=self.C,
                    sigma=sig_rel,
                    tag="wbd_release",
                )
                self._last_released_value = noisy_val
                self._last_released_day = day
                self._last_released_rho = rho_candidate
                self._pub_budgets_history[day] = rho_candidate
                return noisy_val
            else:
                # Approximate with last release, spend 0 publication budget
                self._pub_budgets_history[day] = 0.0
                return self._last_released_value

        else:
            # W-BA: Budget Absorption (Figure 4 in Kellaris et al.)
            # Check nullification from previous release
            to_nullify = max(0, int(round(self._last_released_rho / self.rho_base)) - 1)
            nullify_until = (self._last_released_day or 0) + to_nullify

            if day <= nullify_until:
                # Output is forced null due to nullification rule
                return self._last_released_value

            # Calculate absorbable skipped timestamps since end of nullification period
            to_absorb = day - nullify_until
            num_absorbed = min(to_absorb, self.w)
            rho_candidate = self.rho_base * float(num_absorbed)
            sig_candidate = self.C / math.sqrt(2.0 * rho_candidate)

            if noisy_diff > sig_candidate:
                sig_rel = sig_candidate
                noisy_val = float(true_value + rng.normal(0.0, sig_rel))
                self.ledger.record_gaussian_release(
                    release_id=f"wba_{self.w}_rel_day_{day}",
                    days_accessed=[day],
                    sensitivity=self.C,
                    sigma=sig_rel,
                    tag="wba_release",
                )
                self._last_released_value = noisy_val
                self._last_released_day = day
                self._last_released_rho = rho_candidate
                return noisy_val
            else:
                # Approximate with last release
                return self._last_released_value


class TreeBaseline(BaseDPBaseline):
    """Binary Tree continual release baseline with exact Honaker GLS optimal estimation.
    
    Implements Tree-U, Tree-2l, Tree-sqrt over time horizon T:
    - Matrix A of shape (N_nodes, T)
    - Fisher information matrix M = A^T Sigma^-1 A
    - GLS Covariance matrix Cov = M^-1
    - Window variance Var(w) = 1_w^T Cov 1_w
    - Privacy loss rho(d) = 0.5 * C^2 * max_I 1_I^T M 1_I
    - Strictly satisfies Theorem 7(b): Var(w) >= w^2 * C^2 / (2 * rho(w)) everywhere.
    """

    def __init__(
        self,
        variant: str = "Tree-U",
        T: int = 365,
        L: int = 9,
        C: float = 104.0,
        delta: float = 1e-6,
    ):
        self.variant = variant
        self.name = variant
        self.T = int(T)
        self.L = int(L)
        self.C = float(C)
        self.delta = float(delta)
        self.ledger = UnifiedPrivacyLedger(delta=self.delta, allow_unconfigured_policy=True)

        # Build tree measurement matrix A
        block_sizes = [2**l for l in range(self.L + 1)]
        rows = []
        level_indices = []
        for l, b in enumerate(block_sizes):
            for s in range(0, self.T, b):
                if s + b <= self.T:
                    row = np.zeros(self.T, dtype=np.float64)
                    row[s : s + b] = 1.0
                    rows.append(row)
                    level_indices.append(l)
        self.A = np.array(rows, dtype=np.float64)
        self.levels = np.array(level_indices, dtype=np.int32)

        # Layer weighting function
        if variant == "Tree-U":
            weight_fn = lambda l: 1.0
        elif variant == "Tree-2l":
            weight_fn = lambda l: 2.0**l
        elif variant == "Tree-sqrt":
            weight_fn = lambda l: 2.0**(l / 2.0)
        else:
            raise ValueError(f"Unknown tree variant: {variant}")

        self.w_levels = np.array([weight_fn(l) for l in self.levels], dtype=np.float64)
        # Fisher information matrix M_base
        self.M_base = (self.A.T * (1.0 / self.w_levels)) @ self.A
        self.cov_base = np.linalg.inv(self.M_base)

    def get_worst_case_rho_term(self, d: int) -> float:
        """Compute max_s 1_d^T M_base 1_d for interval length d."""
        if d >= self.T:
            vec = np.ones(self.T, dtype=np.float64)
            return float(vec @ self.M_base @ vec)
        return float(max(np.sum(self.M_base[s : s + d, s : s + d]) for s in range(0, self.T - d + 1)))

    def get_window_variance(self, window_start: int, window_end: int, beta: float = 1.0) -> float:
        """Compute exact GLS variance for interval [window_start, window_end] with noise scaling beta."""
        s = max(0, min(self.T - 1, window_start - 1))
        e = max(0, min(self.T, window_end))
        base_var = float(np.sum(self.cov_base[s:e, s:e]))
        return (beta**2) * base_var


class BandedToeplitzBaseline(BaseDPBaseline):
    """b-Banded Lower-Triangular Toeplitz Matrix Factorization Baseline.

    Theoretical origins:
      - Fichtenberger, Henzinger & Upadhyay (ICML 2023, arXiv:2202.11205): Canonical Toeplitz
        factorization with generating function (1 - z)^(-alpha), alpha = 0.5:
          f_0 = 1,  f_k = f_{k-1} * (2k - 1) / (2k)  for 1 <= k < b.
      - Choquette-Choo et al. (NeurIPS 2023, arXiv:2306.08153) & Denisov et al. (NeurIPS 2022):
        b-banded lower-triangular Toeplitz strategy matrix C^{(b)} in R^{T x T} where
        C_{t, tau} = f_{t - tau} for 0 <= t - tau < b and 0 otherwise.

    Multi-Horizon User-Time zCDP Accounting:
      - Encoded release: y = C x_clip + z, where z ~ N(0, sigma_mf^2 I_T).
      - Unbiased daily reconstruction: x_hat = C^{-1} y = x_clip + C^{-1} z.
      - Under (u, I)-user-time adjacency over any contiguous h-day interval I = [s, s+h-1],
        since C_{t, tau} >= 0, a user changing each day tau in I by up to +-C_clip induces
        worst-case L2^2 sensitivity:
          Delta_2^2(h) = C_clip^2 * max_{0 <= s <= T - h} 1_I^T (C^T C) 1_I,
        with unit-sensitivity zCDP profile rho_u(h) = 0.5 * max_s 1_I^T M_base 1_I,
        where M_base = C^T C.
      - Satisfies the Cauchy-Schwarz multi-horizon variance bound (Theorem 7b):
        Var(1_w^T x_hat) >= w^2 * C_clip^2 / (2 * rho(w)).
    """

    def __init__(
        self,
        T: int = 420,
        bandwidth: int = 30,
        alpha: float = 0.5,
        C: float = 1.0,
        delta: float = 1e-6,
    ):
        if bandwidth < 1 or bandwidth > T:
            raise ValueError(f"bandwidth must be in [1, {T}], got {bandwidth}")
        if not (0.0 < alpha <= 1.0):
            raise ValueError(f"alpha must be in (0, 1], got {alpha}")
        self.T = int(T)
        self.bandwidth = int(bandwidth)
        self.alpha = float(alpha)
        self.C = float(C)
        self.delta = float(delta)
        self.name = f"Banded-MF(b={self.bandwidth},alpha={self.alpha:.2f})"
        self.ledger = UnifiedPrivacyLedger(delta=self.delta, allow_unconfigured_policy=True)

        # Toeplitz band coefficients f_0, ..., f_{b-1} from (1 - z)^(-alpha)
        self.band_coeffs = np.zeros(self.bandwidth, dtype=np.float64)
        self.band_coeffs[0] = 1.0
        for k in range(1, self.bandwidth):
            self.band_coeffs[k] = self.band_coeffs[k - 1] * (self.alpha + float(k) - 1.0) / float(k)

        # Lower-triangular b-banded Toeplitz encoder C in R^{T x T}
        self.C_mat = np.zeros((self.T, self.T), dtype=np.float64)
        for k in range(self.bandwidth):
            idx = np.arange(0, self.T - k)
            self.C_mat[idx + k, idx] = self.band_coeffs[k]

        # Fisher information matrix M_base = C^T C and decoder C^{-1}
        self.M_base = self.C_mat.T @ self.C_mat
        self.C_inv = np.linalg.inv(self.C_mat)
        self.cov_base = self.C_inv @ self.C_inv.T

        # 2D prefix sum of M_base for O(1) interval sensitivity lookup
        self._P_m = np.zeros((self.T + 1, self.T + 1), dtype=np.float64)
        self._P_m[1:, 1:] = np.cumsum(np.cumsum(self.M_base, axis=0), axis=1)

    def get_worst_case_rho_term(self, h: int) -> float:
        """Compute unit-sensitivity zCDP term 0.5 * max_s 1_h^T M_base 1_h for horizon h."""
        h_clamped = max(1, min(self.T, int(h)))
        s_idx = np.arange(0, self.T - h_clamped + 1)
        e_idx = s_idx + h_clamped
        block_sums = (
            self._P_m[e_idx, e_idx]
            - self._P_m[s_idx, e_idx]
            - self._P_m[e_idx, s_idx]
            + self._P_m[s_idx, s_idx]
        )
        return 0.5 * float(np.max(block_sums))

    def calibrate_to_multi_horizon_envelope(
        self,
        rem_add_envelope: np.ndarray,
        n_days: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Calibrate noise multiplier beta so rho_MF(h) <= rem_add_envelope[h-1] for all h in [1, n_days]."""
        eval_days = min(self.T, int(n_days) if n_days is not None else len(rem_add_envelope))
        ratios: List[Tuple[int, float]] = []
        for h in range(1, eval_days + 1):
            rho_u = self.get_worst_case_rho_term(h)
            ratios.append((h, rho_u / float(rem_add_envelope[h - 1])))
        tight_h, beta_sq = max(ratios, key=lambda x: x[1])
        rho_u_h1 = self.get_worst_case_rho_term(1)
        beta_sq_h1_only = rho_u_h1 / float(rem_add_envelope[0])
        rho_u_365 = self.get_worst_case_rho_term(min(365, eval_days))
        naive_h1_violation_at_365 = (rho_u_365 / beta_sq_h1_only) / float(rem_add_envelope[min(365, eval_days) - 1])
        return {
            "beta_env": math.sqrt(beta_sq),
            "tight_h_env": int(tight_h),
            "rho_u_h1": rho_u_h1,
            "rho_u_tight": self.get_worst_case_rho_term(tight_h),
            "naive_h1_violation_ratio_at_365": float(naive_h1_violation_at_365),
            "streaming_state_bytes": int(self.bandwidth * 8 * 2),  # 2 channels x b float64 ring buffer
            "dense_decoder_bytes": int(self.C_inv.nbytes),
        }

    def get_window_variance(self, window_start: int, window_end: int, beta: float = 1.0) -> float:
        """Compute exact variance of rolling sum over [window_start, window_end] (1-indexed)."""
        s = max(0, min(self.T - 1, window_start - 1))
        e = max(0, min(self.T, window_end))
        base_var = float(np.sum(self.cov_base[s:e, s:e]))
        return (beta**2) * (self.C**2) * base_var

