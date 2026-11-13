"""Exact Calendar Accounting and Long-Term Privacy Analysis for SDG-MH.

Implements:
1. Exact Calendar Family representation:
   - Fixed families: (W_f, S_f) in {(7, 7), (30, 7), (90, 30)}.
   - Phase phi_f in {0, ..., S_f - 1}.
   - Clear semantic definition: release on day t >= 1 iff t % S_f == phi_f % S_f.
   - Release at day t covers data window [t - W_f + 1, t].
   - Target protection interval I = [s, s + h - 1].
   - Exact intersection condition: B_{f, k} cap I != emptyset <=> t_k in [s, s + h + W_f - 2].
2. Discrete Event Enumeration Oracle (for ground-truth verification).
3. Fast Modular Arithmetic Formula with Congruence Peak Arc characterization:
   - L = W_f + h - 1
   - delta = (phi_f - s) % S_f
   - n_f(s, h; phi_f) = 1 + (L - 1 - delta) // S_f if delta <= L - 1 else 0
   - Joint period P = lcm(7, 7, 30) = 210.
4. Worst-case envelope L_max(h) = max_{s in 0..P-1} L(s, h) and slack S(h) = rho_bar(h) - L_max(h).
5. Formal Periodicity and Long-Term Privacy Verification:
   - Identity: L_max(h + P) = L_max(h) + P * (d + sum_f x_f / S_f).
   - Asymptotic tail safety condition: d + sum_f x_f / S_f <= beta.
   - Finite reduction bound verification on h in [1, 575].
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import numpy as np


@dataclass(frozen=True)
class CalendarFamily:
    """Represents a periodic release family (W_f, S_f)."""

    name: str
    window_days: int  # W_f: statistical data window size
    step_days: int    # S_f: release cadence / step

    def __post_init__(self) -> None:
        if self.window_days <= 0:
            raise ValueError(f"window_days must be positive, got {self.window_days}")
        if self.step_days <= 0:
            raise ValueError(f"step_days must be positive, got {self.step_days}")


# Standard three families for long-running analytics
DEFAULT_FAMILIES = (
    CalendarFamily(name="distinct_7d", window_days=7, step_days=7),
    CalendarFamily(name="distinct_30d", window_days=30, step_days=7),
    CalendarFamily(name="distinct_90d", window_days=90, step_days=30),
)

JOINT_PERIOD: int = 210  # lcm(7, 7, 30)


@dataclass(frozen=True)
class PrivacyCommitmentCurve:
    """Multi-horizon user-time privacy commitment curve rho_bar(h)."""

    name: str
    horizons: Tuple[int, ...]
    epsilons: Tuple[float, ...]
    delta: float = 1e-6
    rhos: Tuple[float, ...] = field(init=False)
    beta_tail: float = field(init=False)
    a_tail: float = field(init=False)

    def __post_init__(self) -> None:
        if len(self.horizons) != len(self.epsilons):
            raise ValueError("horizons and epsilons must have identical length")
        if len(self.horizons) < 2:
            raise ValueError("Must provide at least two horizon points")

        # Invert canonical epsilon to zCDP rho:
        # eps = rho + 2 * sqrt(rho * ln(1/delta))
        # Let y = sqrt(rho). Then y^2 + 2*sqrt(c)*y - eps = 0 => y = sqrt(c + eps) - sqrt(c)
        c = math.log(1.0 / self.delta)
        rho_vals = []
        for eps in self.epsilons:
            if eps <= 0:
                rho_vals.append(0.0)
            else:
                y = math.sqrt(c + eps) - math.sqrt(c)
                rho_vals.append(float(y * y))
        object.__setattr__(self, "rhos", tuple(rho_vals))

        # Tail linear extrapolation after H_0 = horizons[-1]:
        # rho_bar(h) = a_tail + beta_tail * h
        # beta_tail = (rho[-1] - rho[-2]) / (h[-1] - h[-2])
        h_last = self.horizons[-1]
        h_prev = self.horizons[-2]
        r_last = self.rhos[-1]
        r_prev = self.rhos[-2]
        beta = (r_last - r_prev) / float(h_last - h_prev)
        a = r_last - beta * float(h_last)
        object.__setattr__(self, "beta_tail", float(beta))
        object.__setattr__(self, "a_tail", float(a))

    def rho_bar(self, h: Union[int, float, np.ndarray]) -> Union[float, np.ndarray]:
        """Evaluate rho_bar(h) with linear interpolation on [h[0], h[-1]] and linear tail extrapolation."""
        h_arr = np.asarray(h, dtype=np.float64)
        is_scalar = h_arr.ndim == 0
        h_flat = h_arr.ravel()

        h_max = float(self.horizons[-1])
        h_min = float(self.horizons[0])

        # Piecewise linear interpolation within [h_min, h_max]
        out = np.interp(h_flat, self.horizons, self.rhos)

        # Tail extrapolation for h > h_max
        tail_mask = h_flat > h_max
        if np.any(tail_mask):
            out[tail_mask] = self.a_tail + self.beta_tail * h_flat[tail_mask]

        if is_scalar:
            return float(out[0])
        return out.reshape(h_arr.shape)

    def epsilon_bar(self, h: Union[int, float, np.ndarray]) -> Union[float, np.ndarray]:
        """Evaluate canonical epsilon(h) from rho_bar(h)."""
        r = self.rho_bar(h)
        if isinstance(r, (int, float)):
            if r <= 0:
                return 0.0
            return float(r + 2.0 * math.sqrt(r * math.log(1.0 / self.delta)))
        r_arr = np.asarray(r, dtype=np.float64)
        out = np.zeros_like(r_arr)
        pos = r_arr > 0
        out[pos] = r_arr[pos] + 2.0 * np.sqrt(r_arr[pos] * math.log(1.0 / self.delta))
        return out


# Canonical Historical Privacy Commitment Curves
ORIGINAL_PLANNER_CURVE = PrivacyCommitmentCurve(
    name="Original_Planner",
    horizons=(1, 7, 30, 90, 365),
    epsilons=(0.90, 1.08, 1.49, 2.24, 4.39),
    delta=1e-6,
)

WEEK1_V2_CURVE = PrivacyCommitmentCurve(
    name="Week1_v2",
    horizons=(1, 7, 30, 90, 365),
    epsilons=(1.0, 1.2, 1.5, 2.25, 4.4),
    delta=1e-6,
)


def iterate_family_releases(
    family: CalendarFamily,
    phi: int,
    total_days: int,
    start_day: int = 1,
) -> List[int]:
    """Iterate all release days t in [start_day, total_days] where t % S_f == phi % S_f."""
    phi_norm = phi % family.step_days
    return [t for t in range(start_day, total_days + 1) if t % family.step_days == phi_norm]


def count_intersections_oracle(
    s: int,
    h: int,
    window_days: int,
    step_days: int,
    phi: int,
    day_min: int = -500,
    day_max: int = 2000,
) -> int:
    """Discrete Event Enumeration Oracle for ground-truth intersection counting.

    Iterates over all release dates t in [day_min, day_max] such that t % step_days == phi % step_days.
    For each release, data window is B = [t - window_days + 1, t].
    Target interval is I = [s, s + h - 1].
    Checks if max(start_B, start_I) <= min(end_B, end_I).
    """
    if h <= 0:
        return 0
    start_I = s
    end_I = s + h - 1
    phi_norm = phi % step_days

    count = 0
    for t in range(day_min, day_max + 1):
        if t % step_days == phi_norm:
            start_B = t - window_days + 1
            end_B = t
            if max(start_B, start_I) <= min(end_B, end_I):
                count += 1
    return count


def count_intersections_fast(
    s: int,
    h: int,
    window_days: int,
    step_days: int,
    phi: int,
) -> int:
    """Fast closed-form modular arithmetic intersection count n_f(s, h; phi_f).

    Mathematical equivalence:
    Release on day t intersects [s, s + h - 1] iff t in [s, s + h + W_f - 2].
    Length of integer range is L = W_f + h - 1.
    First integer t >= s with t % S_f == phi % S_f occurs at t = s + delta,
    where delta = (phi - s) % S_f in {0, ..., S_f - 1}.
    Number of such releases is:
      1 + (L - 1 - delta) // S_f  if delta <= L - 1 else 0.
    """
    if h <= 0:
        return 0
    L = window_days + h - 1
    delta = (phi - s) % step_days
    if delta > L - 1:
        return 0
    return int(1 + (L - 1 - delta) // step_days)


def count_intersections_fast_vectorized(
    s_arr: np.ndarray,
    h: int,
    window_days: int,
    step_days: int,
    phi: int,
) -> np.ndarray:
    """Vectorized version of count_intersections_fast across an array of start days s."""
    if h <= 0:
        return np.zeros_like(s_arr, dtype=np.int64)
    L = window_days + h - 1
    delta = (phi - s_arr) % step_days
    out = np.zeros_like(s_arr, dtype=np.int64)
    valid = delta <= L - 1
    out[valid] = 1 + (L - 1 - delta[valid]) // step_days
    return out


def analyze_congruence_peak_arc(
    h: int,
    window_days: int,
    step_days: int,
    phi: int,
) -> Dict[str, Any]:
    """Analytical decomposition into quotient, remainder, and congruence peak arc.

    q_f(h) = (W_f + h - 1) // S_f
    r_f(h) = (W_f + h - 1) % S_f
    Peak arc is the set of starting residues s mod S_f where count is q_f + 1:
      {s mod S_f : (phi - s) % S_f < r_f(h)}.
    For all other residues, count is q_f.
    """
    L = window_days + h - 1
    q = L // step_days
    r = L % step_days
    peak_residues = []
    base_residues = []
    for s_res in range(step_days):
        if (phi - s_res) % step_days < r:
            peak_residues.append(s_res)
        else:
            base_residues.append(s_res)
    return {
        "h": h,
        "L": L,
        "q": q,
        "r": r,
        "peak_count": q + 1 if r > 0 else q,
        "base_count": q,
        "peak_residues": peak_residues,
        "base_residues": base_residues,
        "arc_fraction": float(r) / float(step_days),
    }


@dataclass(frozen=True)
class CalendarSchedule:
    """Complete specification of joint publication schedule."""

    families: Tuple[CalendarFamily, ...] = DEFAULT_FAMILIES
    phases: Tuple[int, ...] = (0, 0, 0)
    costs: Tuple[float, ...] = (0.00153, 0.00153, 0.00153)
    daily_cost_d: float = 0.8 * 0.000302

    def __post_init__(self) -> None:
        if len(self.families) != len(self.phases) or len(self.families) != len(self.costs):
            raise ValueError("families, phases, and costs must have matching lengths")
        for fam, phi in zip(self.families, self.phases):
            if not (0 <= phi < fam.step_days):
                raise ValueError(f"Phase {phi} out of range [0, {fam.step_days - 1}] for {fam.name}")

    @property
    def asymptotic_slope(self) -> float:
        """Asymptotic rate of privacy accumulation: d + sum_f (x_f / S_f)."""
        return float(self.daily_cost_d + sum(x / fam.step_days for x, fam in zip(self.costs, self.families)))

    def cost_at(self, s: int, h: int) -> float:
        """Exact privacy cost L(s, h) for protection interval [s, s + h - 1]."""
        cost = self.daily_cost_d * h
        for fam, phi, x in zip(self.families, self.phases, self.costs):
            cnt = count_intersections_fast(s, h, fam.window_days, fam.step_days, phi)
            cost += x * cnt
        return float(cost)

    def evaluate_envelope(self, h: int, P: int = JOINT_PERIOD) -> Tuple[float, int, List[int]]:
        """Compute worst-case privacy envelope L_max(h) = max_{s in 0..P-1} L(s, h).

        Returns:
            (L_max, worst_s, counts_at_worst_s)
        """
        s_arr = np.arange(P, dtype=np.int64)
        total_costs = np.full(P, self.daily_cost_d * float(h), dtype=np.float64)
        all_counts = []
        for fam, phi, x in zip(self.families, self.phases, self.costs):
            cnts = count_intersections_fast_vectorized(s_arr, h, fam.window_days, fam.step_days, phi)
            all_counts.append(cnts)
            total_costs += float(x) * cnts.astype(np.float64)

        worst_idx = int(np.argmax(total_costs))
        l_max = float(total_costs[worst_idx])
        worst_counts = [int(cnts[worst_idx]) for cnts in all_counts]
        return l_max, worst_idx, worst_counts

    def evaluate_full_envelope(self, max_h: int = 575, P: int = JOINT_PERIOD) -> Tuple[np.ndarray, np.ndarray]:
        """Compute L_max(h) for all h in 1..max_h.

        Returns:
            (l_max_array, worst_s_array) where index h corresponds to horizon h (1-indexed, len max_h + 1).
        """
        s_arr = np.arange(P, dtype=np.int64)
        l_max = np.zeros(max_h + 1, dtype=np.float64)
        worst_s = np.zeros(max_h + 1, dtype=np.int64)

        for h in range(1, max_h + 1):
            costs = np.full(P, self.daily_cost_d * float(h), dtype=np.float64)
            for fam, phi, x in zip(self.families, self.phases, self.costs):
                cnts = count_intersections_fast_vectorized(s_arr, h, fam.window_days, fam.step_days, phi)
                costs += float(x) * cnts.astype(np.float64)
            idx = int(np.argmax(costs))
            l_max[h] = costs[idx]
            worst_s[h] = idx

        return l_max, worst_s

    def compute_slack(
        self,
        commitment: PrivacyCommitmentCurve,
        max_h: int = 575,
        P: int = JOINT_PERIOD,
    ) -> Dict[str, Any]:
        """Compute slack S(h) = rho_bar(h) - L_max(h) across all h in 1..max_h."""
        l_max, worst_s = self.evaluate_full_envelope(max_h=max_h, P=P)
        h_vals = np.arange(1, max_h + 1)
        rho_bars = commitment.rho_bar(h_vals)

        slacks = rho_bars - l_max[1:]
        min_idx = int(np.argmin(slacks))
        min_slack = float(slacks[min_idx])
        min_h = int(h_vals[min_idx])

        return {
            "max_h": max_h,
            "min_slack": min_slack,
            "min_h": min_h,
            "l_max": l_max,
            "slacks": slacks,
            "worst_s": worst_s,
            "is_feasible": bool(min_slack >= -1e-12),
        }

    def verify_periodicity_and_long_term_security(
        self,
        commitment: PrivacyCommitmentCurve,
        H_0: int = 365,
        P: int = JOINT_PERIOD,
    ) -> Dict[str, Any]:
        """Rigorous mathematical proof verification of infinite-horizon privacy security.

        Checks:
        1. Exact periodicity identity: L_max(h + P) == L_max(h) + P * slope for all h in 1..H_0.
        2. Tail asymptotic growth condition: slope <= beta_tail.
        3. Finite reduction verification on [1, H_0 + P = 575]: min_{h <= 575} S(h) >= 0.
        """
        slope = self.asymptotic_slope
        beta = commitment.beta_tail

        # 1. Check slope condition
        slope_safe = bool(slope <= beta + 1e-12)

        # 2. Check periodicity identity on h in 1..H_0
        l_max_575, _ = self.evaluate_full_envelope(max_h=H_0 + P, P=P)
        periodicity_errors = []
        for h in range(1, H_0 + 1):
            lhs = l_max_575[h + P]
            rhs = l_max_575[h] + P * slope
            err = abs(lhs - rhs)
            if err > 1e-9:
                periodicity_errors.append((h, lhs, rhs, err))

        periodicity_holds = len(periodicity_errors) == 0

        # 3. Check finite slack on [1, H_0 + P]
        h_vals = np.arange(1, H_0 + P + 1)
        rho_bars = commitment.rho_bar(h_vals)
        slacks = rho_bars - l_max_575[1:]
        min_idx = int(np.argmin(slacks))
        min_slack = float(slacks[min_idx])
        min_h = int(h_vals[min_idx])
        finite_safe = bool(min_slack >= -1e-12)

        is_guaranteed = slope_safe and periodicity_holds and finite_safe

        return {
            "is_infinite_horizon_secure": is_guaranteed,
            "slope_safe": slope_safe,
            "periodicity_holds": periodicity_holds,
            "finite_safe": finite_safe,
            "asymptotic_slope": slope,
            "commitment_beta_tail": beta,
            "slope_headroom": float(beta - slope),
            "periodicity_max_err": max([e[3] for e in periodicity_errors]) if periodicity_errors else 0.0,
            "min_slack_up_to_575": min_slack,
            "min_slack_horizon_h": min_h,
            "H_0": H_0,
            "P": P,
            "check_horizon": H_0 + P,
        }
