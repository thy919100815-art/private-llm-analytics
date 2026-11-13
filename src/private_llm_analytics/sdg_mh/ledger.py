"""Unified Time-Indexed Privacy Ledger and Historical Revisit Gate.

Implements:
1. Unified zCDP accounting under (u, I)-adjacency for all release families:
   - Each release record: (release_id, days_accessed B_n, mechanism_type, params, sensitivity Delta_n).
   - Single Gaussian release cost: Delta_n^2 / (2 * sigma_n^2).
   - Pure eps-DP (e.g. Laplace) converted to zCDP via rho = eps^2 / 2.
2. Exact privacy profile computation:
   rho(h) = max_{|I| <= h} sum_{n: B_n cap I != emptyset} rho_n(I).
   Includes analytical closed-form fast-paths and exact discrete interval enumeration.
3. Canonical epsilon conversion:
   eps_h = rho(h) + 2 * sqrt(rho(h) * ln(1/delta)), default delta = 1e-6.
   Horizons: h in {1, 7, 30, 90, 365}.
4. Historical Revisit Gate:
   Evaluates candidate releases against bound E_bar(h); rejects if breached.
   Combines previous and new estimates via inverse-variance weighting.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple
import numpy as np


@dataclass(frozen=True)
class ReleaseRecord:
    """Record of a committed privacy release in the unified ledger."""

    release_id: str
    days_accessed: frozenset[int]  # 1-based day indices B_n
    mechanism_type: str            # 'gaussian' or 'laplace'
    params: Dict[str, float]       # {'sigma': ...} or {'epsilon': ...}
    sensitivity: float             # Delta_n
    rho_cost: float                # Single-release zCDP rho
    tag: str = ""                  # 'sdg_additive', 'distinct_calendar', 'baseline', etc.


@dataclass(frozen=True)
class PrivacyProfile:
    """Privacy profile rho(h) and canonical epsilon_h across horizons."""

    delta: float
    rho_map: Dict[int, float]      # h -> rho(h)
    eps_map: Dict[int, float]      # h -> eps_h

    def __getitem__(self, h: int) -> float:
        return self.rho_map[h]

    def eps(self, h: int) -> float:
        return self.eps_map[h]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "delta": self.delta,
            "horizons": {
                h: {
                    "rho": round(self.rho_map[h], 6),
                    "eps": round(self.eps_map[h], 4),
                }
                for h in sorted(self.rho_map.keys())
            },
        }


def rho_to_epsilon(rho: float, delta: float = 1e-6) -> float:
    """Convert zCDP rho to canonical (eps, delta)-DP bound:

    eps = rho + 2 * sqrt(rho * ln(1 / delta))
    """
    if rho <= 0:
        return 0.0
    if not (0 < delta < 1):
        raise ValueError(f"delta must be in (0, 1), got {delta}")
    return float(rho + 2.0 * math.sqrt(rho * math.log(1.0 / delta)))


def epsilon_to_rho(eps: float) -> float:
    """Convert pure eps-DP to canonical zCDP bound rho = eps^2 / 2."""
    if eps < 0:
        raise ValueError(f"eps must be non-negative, got {eps}")
    return float((eps * eps) / 2.0)


def closed_form_sdg_additive_rho(h: int, daily_rho: float) -> float:
    """Closed-form profile for SDG additive channel: rho^add(h) = h * rho_1^add."""
    if h <= 0:
        return 0.0
    return float(h * daily_rho)


def closed_form_distinct_calendar_rho(
    h: int,
    calendar_entries: Sequence[Tuple[int, int, float]],
    delta_sq: float = 11.0,
) -> float:
    """Closed-form profile for Distinct-user calendar:

    rho^dist(h) = sum_{(w, s)} ceil((h + w - 1) / s) * delta^2 / (2 * sigma_w^2)
    Each entry is (w, s, sigma_w).
    """
    if h <= 0:
        return 0.0
    total_rho = 0.0
    for w, s, sigma_w in calendar_entries:
        if s <= 0 or sigma_w <= 0:
            continue
        num_windows = int(math.ceil((h + w - 1) / float(s)))
        rho_single = delta_sq / (2.0 * (sigma_w ** 2))
        total_rho += num_windows * rho_single
    return float(total_rho)


class UnifiedPrivacyLedger:
    """Time-indexed unified privacy ledger maintaining zCDP accountability."""

    DEFAULT_HORIZONS: Tuple[int, ...] = (1, 7, 30, 90, 365)

    def __init__(self, delta: float = 1e-6):
        if not (0 < delta < 1):
            raise ValueError(f"delta must be in (0, 1), got {delta}")
        self.delta: float = float(delta)
        self.records: List[ReleaseRecord] = []
        self._day_to_records: Dict[int, List[int]] = {}  # day -> list of record indices
        self._max_day: int = 0

    def record_gaussian_release(
        self,
        release_id: str,
        days_accessed: Sequence[int],
        sensitivity: float,
        sigma: float,
        tag: str = "",
    ) -> ReleaseRecord:
        """Record an analytic Gaussian release: rho = Delta^2 / (2 * sigma^2)."""
        if sigma <= 0:
            raise ValueError(f"sigma must be > 0, got {sigma}")
        if sensitivity < 0:
            raise ValueError(f"sensitivity must be >= 0, got {sensitivity}")

        days_set = frozenset(int(d) for d in days_accessed)
        rho_cost = float((sensitivity * sensitivity) / (2.0 * sigma * sigma))
        record = ReleaseRecord(
            release_id=release_id,
            days_accessed=days_set,
            mechanism_type="gaussian",
            params={"sigma": float(sigma)},
            sensitivity=float(sensitivity),
            rho_cost=rho_cost,
            tag=tag,
        )
        self._add_record(record)
        return record

    def record_laplace_release(
        self,
        release_id: str,
        days_accessed: Sequence[int],
        sensitivity: float,
        epsilon: float,
        tag: str = "",
    ) -> ReleaseRecord:
        """Record a pure Laplace release: rho = eps^2 / 2."""
        if epsilon <= 0:
            raise ValueError(f"epsilon must be > 0, got {epsilon}")
        days_set = frozenset(int(d) for d in days_accessed)
        rho_cost = epsilon_to_rho(epsilon)
        record = ReleaseRecord(
            release_id=release_id,
            days_accessed=days_set,
            mechanism_type="laplace",
            params={"epsilon": float(epsilon)},
            sensitivity=float(sensitivity),
            rho_cost=rho_cost,
            tag=tag,
        )
        self._add_record(record)
        return record

    def record_zcdp_release(
        self,
        release_id: str,
        days_accessed: Sequence[int],
        sensitivity: float,
        rho_cost: float,
        tag: str = "",
    ) -> ReleaseRecord:
        """Directly record a release with known zCDP cost rho."""
        days_set = frozenset(int(d) for d in days_accessed)
        record = ReleaseRecord(
            release_id=release_id,
            days_accessed=days_set,
            mechanism_type="zcdp",
            params={"rho": float(rho_cost)},
            sensitivity=float(sensitivity),
            rho_cost=float(rho_cost),
            tag=tag,
        )
        self._add_record(record)
        return record

    def _add_record(self, record: ReleaseRecord) -> None:
        idx = len(self.records)
        self.records.append(record)
        for d in record.days_accessed:
            if d not in self._day_to_records:
                self._day_to_records[d] = []
            self._day_to_records[d].append(idx)
            if d > self._max_day:
                self._max_day = d

    def compute_rho(self, h: int) -> float:
        """Compute exact worst-case rho(h) over all intervals I of length h:

        rho(h) = max_{s} sum_{n: B_n cap [s, s + h - 1] != emptyset} rho_n
        """
        if h <= 0 or not self.records:
            return 0.0

        # Fast-path: pure single-day additive releases
        is_all_single_day = all(len(r.days_accessed) == 1 for r in self.records)
        if is_all_single_day:
            day_costs: Dict[int, float] = {}
            for r in self.records:
                d = next(iter(r.days_accessed))
                day_costs[d] = day_costs.get(d, 0.0) + r.rho_cost
            if not day_costs:
                return 0.0
            # Rolling window max over day costs
            min_d = min(day_costs.keys())
            max_d = max(day_costs.keys())
            if max_d - min_d + 1 <= h:
                return float(sum(day_costs.values()))
            
            # Sliding window sum of length h
            max_sum = 0.0
            curr_sum = sum(day_costs.get(d, 0.0) for d in range(min_d, min_d + h))
            max_sum = curr_sum
            for s in range(min_d + 1, max_d - h + 2):
                curr_sum += day_costs.get(s + h - 1, 0.0) - day_costs.get(s - 1, 0.0)
                if curr_sum > max_sum:
                    max_sum = curr_sum
            return float(max_sum)

        # General discrete interval enumeration
        # Active candidate interval start points: all unique boundary days of releases
        start_candidates: Set[int] = set()
        for r in self.records:
            for d in r.days_accessed:
                # Potential start point s such that d is inside [s, s + h - 1]
                start_candidates.add(d)
                start_candidates.add(max(1, d - h + 1))

        if not start_candidates:
            return 0.0

        max_rho = 0.0
        for s in start_candidates:
            end = s + h - 1
            cost_for_interval = 0.0
            for r in self.records:
                # Check intersection: any d in B_n with s <= d <= end
                # Fast check using set intersection if small
                intersects = any(s <= d <= end for d in r.days_accessed)
                if intersects:
                    cost_for_interval += r.rho_cost
            if cost_for_interval > max_rho:
                max_rho = cost_for_interval

        return float(max_rho)

    def get_profile(
        self,
        horizons: Optional[Sequence[int]] = None,
        delta: Optional[float] = None,
    ) -> PrivacyProfile:
        """Compute the complete privacy profile for specified horizons."""
        eff_delta = self.delta if delta is None else float(delta)
        hzs = self.DEFAULT_HORIZONS if horizons is None else tuple(horizons)

        rho_map: Dict[int, float] = {}
        eps_map: Dict[int, float] = {}

        for h in hzs:
            r = self.compute_rho(h)
            rho_map[h] = r
            eps_map[h] = rho_to_epsilon(r, eff_delta)

        return PrivacyProfile(delta=eff_delta, rho_map=rho_map, eps_map=eps_map)

    def check_revisit_gate(
        self,
        candidate_records: Sequence[ReleaseRecord],
        max_profile_bound: Dict[int, float],
    ) -> Tuple[bool, str, Dict[int, float]]:
        """Historical Revisit Gate:

        Checks if adding candidate_records preserves rho(h) <= max_profile_bound[h]
        for all specified horizons.

        Returns:
            (is_allowed, reason_msg, projected_rho_map)
        """
        # Create trial ledger
        trial_ledger = UnifiedPrivacyLedger(delta=self.delta)
        trial_ledger.records = list(self.records) + list(candidate_records)
        trial_ledger._max_day = max(
            [self._max_day] + [max(r.days_accessed) for r in candidate_records if r.days_accessed]
        )

        projected: Dict[int, float] = {}
        for h, bound in max_profile_bound.items():
            proj_r = trial_ledger.compute_rho(h)
            projected[h] = proj_r
            if proj_r > bound + 1e-9:
                return (
                    False,
                    f"Revisit gate rejected: at horizon h={h}, projected rho={proj_r:.6f} exceeds limit E_bar={bound:.6f}.",
                    projected,
                )

        return True, "Revisit gate passed: candidate release is within historical privacy bounds.", projected

    def compute_full_horizon_profile(self, t_span: Optional[int] = None) -> np.ndarray:
        """Compute exact worst-case rho(h) for all horizons h = 1..T_span.

        Complexity: O(T_span * (N + T_span)) via difference array.
        For all h >= T_span, rho(h) is constant and equals the total sum of all releases.
        """
        if not self.records:
            t = 1 if t_span is None else max(1, int(t_span))
            return np.zeros(t + 1, dtype=np.float64)

        max_day = max(max(r.days_accessed) for r in self.records if r.days_accessed)
        T = max_day if t_span is None else max(int(t_span), max_day)

        starts = np.array([min(r.days_accessed) for r in self.records], dtype=np.int64)
        ends = np.array([max(r.days_accessed) for r in self.records], dtype=np.int64)
        costs = np.array([r.rho_cost for r in self.records], dtype=np.float64)

        rhos = np.zeros(T + 1, dtype=np.float64)
        for h in range(1, T + 1):
            max_s = T - h + 1
            diff = np.zeros(max_s + 2, dtype=np.float64)
            l = np.maximum(1, starts - h + 1)
            r = np.minimum(max_s, ends)
            valid = l <= r
            np.add.at(diff, l[valid], costs[valid])
            np.add.at(diff, r[valid] + 1, -costs[valid])
            rhos[h] = float(np.max(np.cumsum(diff)[1 : max_s + 1]))

        return rhos

    def check_full_horizon_gate(
        self,
        candidate_records: Sequence[ReleaseRecord],
        max_profile_bound_fn_or_array: Any,
        t_span: Optional[int] = None,
    ) -> Tuple[bool, str, Dict[str, Any]]:
        """Evaluate candidate releases against upper bound across ALL horizons h = 1..T_span.

        Returns:
            (is_allowed, reason_msg, diagnostics_dict)
        """
        trial_ledger = UnifiedPrivacyLedger(delta=self.delta)
        trial_ledger.records = list(self.records) + list(candidate_records)
        trial_ledger._max_day = max(
            [self._max_day] + [max(r.days_accessed) for r in candidate_records if r.days_accessed]
        )

        T = int(t_span) if t_span is not None else trial_ledger._max_day
        if not callable(max_profile_bound_fn_or_array):
            T = min(T, len(max_profile_bound_fn_or_array) - 1)
        full_rhos = trial_ledger.compute_full_horizon_profile(t_span=trial_ledger._max_day)

        # Check against bound
        exceeded_horizons = []
        max_excess_ratio = 1.0
        worst_h = -1

        for h in range(1, T + 1):
            if callable(max_profile_bound_fn_or_array):
                bound_h = float(max_profile_bound_fn_or_array(h))
            else:
                bound_h = float(max_profile_bound_fn_or_array[h])

            proj_r = float(full_rhos[h])
            if proj_r > bound_h + 1e-9:
                exceeded_horizons.append(h)
                ratio = proj_r / bound_h if bound_h > 0 else float("inf")
                if ratio > max_excess_ratio:
                    max_excess_ratio = ratio
                    worst_h = h

        diag = {
            "total_horizons_checked": T,
            "exceeded_count": len(exceeded_horizons),
            "worst_horizon_h": worst_h,
            "max_excess_ratio": round(max_excess_ratio, 6),
            "projected_rhos": full_rhos,
        }

        if exceeded_horizons:
            return (
                False,
                f"Full horizon gate rejected: {len(exceeded_horizons)}/{T} horizons exceeded limit. "
                f"Worst breach at h={worst_h}: projected={full_rhos[worst_h]:.6f} exceeds limit={bound_h:.6f} (ratio={max_excess_ratio:.4f}).",
                diag,
            )

        return True, f"Full horizon gate passed: all {T} horizons within bounds.", diag


def combine_inverse_variance(
    estimate_old: float,
    variance_old: float,
    estimate_new: float,
    variance_new: float,
) -> Tuple[float, float]:
    """Combine old and new unbiased estimates using optimal inverse-variance weighting:

    theta_merged = (theta_old / var_old + theta_new / var_new) / (1 / var_old + 1 / var_new)
    var_merged = 1 / (1 / var_old + 1 / var_new)
    """
    if variance_old <= 0 or variance_new <= 0:
        raise ValueError("Variances must be strictly positive.")
    w_old = 1.0 / variance_old
    w_new = 1.0 / variance_new
    w_sum = w_old + w_new
    theta_merged = (estimate_old * w_old + estimate_new * w_new) / w_sum
    var_merged = 1.0 / w_sum
    return float(theta_merged), float(var_merged)
