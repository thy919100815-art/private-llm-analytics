"""Precomputed Confidence Interval Display Rules for SDG-MH.

Enforces:
- Using publicly declared noise parameters (sigma_r, sigma_f, sigma_w, C_r, C_f) and window length |W|,
  precomputes analytical confidence interval width for each metric.
- Evaluates whether relative margin of error or absolute CI width exceeds acceptable reporting thresholds.
- Suppresses unreliable combinations (e.g. single-day Success Rate when sample is small or variance is wide).
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional, Tuple


class DisplayRulesEngine:
    """Precomputed display suppression rules based on analytical confidence intervals."""

    def __init__(
        self,
        sigma_r: float = 2500.0,
        sigma_f: float = 700.0,
        sigma_w: float = 60.0,
        max_allowed_ci_width_rate: float = 0.05,        # Success rate CI width > 0.05 is suppressed
        max_relative_half_width_volume: float = 0.20,   # Volume relative 95% CI half-width > 20% suppressed
    ):
        self.sigma_r = float(sigma_r)
        self.sigma_f = float(sigma_f)
        self.sigma_w = float(sigma_w)
        self.max_allowed_ci_width_rate = float(max_allowed_ci_width_rate)
        self.max_relative_half_width_volume = float(max_relative_half_width_volume)
        # Retain backward compatible attribute alias
        self.max_relative_ci_width_volume = float(max_relative_half_width_volume)

    def evaluate_additive_window(
        self,
        window_days: int,
        prior_daily_requests: float = 22500.0,
        prior_failure_rate: float = 0.03,
    ) -> Dict[str, Any]:
        """Precompute confidence interval widths for Requests, Failures, and Success Rate."""
        if window_days <= 0:
            raise ValueError(f"window_days must be positive, got {window_days}")

        sqrt_w = math.sqrt(float(window_days))
        half_width_r = 1.96 * sqrt_w * self.sigma_r
        half_width_f = 1.96 * sqrt_w * self.sigma_f

        total_ci_width_r = 2.0 * half_width_r
        total_ci_width_f = 2.0 * half_width_f

        expected_req = window_days * prior_daily_requests
        expected_fail = expected_req * prior_failure_rate

        # Relative 95% CI half-width (half-width / expected_req)
        rel_half_width_r = half_width_r / max(1.0, expected_req)
        volume_displayable = rel_half_width_r <= self.max_relative_half_width_volume

        # Success rate CI approximation via delta method:
        # Var(1 - F/R) ~ (1 / R^2) * Var(F) + (F^2 / R^4) * Var(R)
        if expected_req > 0:
            var_r = window_days * (self.sigma_r ** 2)
            var_f = window_days * (self.sigma_f ** 2)
            var_rate = (1.0 / (expected_req ** 2)) * var_f + ((expected_fail ** 2) / (expected_req ** 4)) * var_r
            std_rate = math.sqrt(var_rate)
            rate_ci_width = 2.0 * 1.96 * std_rate
        else:
            rate_ci_width = 1.0

        rate_displayable = rate_ci_width <= self.max_allowed_ci_width_rate

        reasons = []
        if not volume_displayable:
            reasons.append(f"Volume relative half-width ({rel_half_width_r * 100:.1f}%) > 20%")
        if not rate_displayable:
            reasons.append(f"Success rate CI width ({rate_ci_width:.4f}) > 0.05")

        return {
            "window_days": window_days,
            "requests_half_width": round(half_width_r, 1),
            "requests_ci_width": round(total_ci_width_r, 1),
            "requests_relative_half_width": round(rel_half_width_r, 4),
            "requests_relative_width": round(rel_half_width_r, 4),
            "requests_displayable": volume_displayable,
            "success_rate_ci_width": round(rate_ci_width, 4),
            "success_rate_displayable": rate_displayable,
            "suppression_reason": "; ".join(reasons) if reasons else None,
        }
