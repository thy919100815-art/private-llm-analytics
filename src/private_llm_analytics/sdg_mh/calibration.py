"""Calibration Engine for Differentially Private Baselines under Unified Ledger.

Implements channel-specific calibration (Item 1 & Item 2):
1. Additive Counting Channel:
   - Evaluated strictly against SDG-MH Additive profile: rho_1^add = 0.00030204 / day.
   - Caliber a: Matched to target rho^add(30) = 0.009061 (eps_30^add = 0.7167).
   - Caliber b: Matched such that rho_scaled(h) <= rho^add_sdg(h) for all h in {1, 7, 30, 90, 365}:
     beta_b^2 = max_{h} (rho_unscaled(h) / rho^add_sdg(h)).
2. Distinct Calendar Channel:
   - Evaluated strictly against Calendar B profile: Delta^2 = 11, sigma_w = 60.
   - Caliber a: Matched to target rho^dist(30) = 0.029028 (eps_30^dist = 1.2942).
   - Caliber b: Matched such that rho_scaled(h) <= rho^dist_sdg(h) for all h.
3. Monotonicity Guarantee:
   - All baseline profiles rho(h) and eps(h) are strictly non-decreasing in h.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple
import numpy as np

from private_llm_analytics.sdg_mh.ledger import (
    closed_form_distinct_calendar_rho,
    closed_form_sdg_additive_rho,
    rho_to_epsilon,
)


HORIZONS = (1, 7, 30, 90, 365)


def compute_sdg_target_profile(
    C_r: float = 50.0,
    sigma_r: float = 2500.0,
    C_f: float = 10.0,
    sigma_f: float = 700.0,
    calendar: Sequence[Tuple[int, int, float]] = ((7, 7, 60.0), (30, 7, 60.0), (90, 30, 60.0)),
    delta_sq: float = 11.0,
    delta: float = 1e-6,
) -> Dict[str, Any]:
    """Compute exact target privacy profile rho(h) and eps_h separated by channel."""
    rho_1_add = (C_r**2) / (2.0 * (sigma_r**2)) + (C_f**2) / (2.0 * (sigma_f**2))
    
    rho_add = {}
    eps_add = {}
    rho_dist = {}
    eps_dist = {}
    rho_tot = {}
    eps_tot = {}

    for h in HORIZONS:
        r_a = closed_form_sdg_additive_rho(h, rho_1_add)
        r_d = closed_form_distinct_calendar_rho(h, calendar, delta_sq=delta_sq)
        r_t = r_a + r_d

        rho_add[h] = r_a
        eps_add[h] = rho_to_epsilon(r_a, delta)
        rho_dist[h] = r_d
        eps_dist[h] = rho_to_epsilon(r_d, delta)
        rho_tot[h] = r_t
        eps_tot[h] = rho_to_epsilon(r_t, delta)

    return {
        "rho_1_add": rho_1_add,
        # Additive Channel
        "rho_add": rho_add,
        "eps_add": eps_add,
        "target_rho_30_add": rho_add[30],
        "target_eps_30_add": eps_add[30],
        # Distinct Channel
        "rho_dist": rho_dist,
        "eps_dist": eps_dist,
        "target_rho_30_dist": rho_dist[30],
        "target_eps_30_dist": eps_dist[30],
        # Total Channel
        "rho": rho_tot,
        "eps": eps_tot,
        "target_eps_30": eps_tot[30],
        "target_rho_30": rho_tot[30],
    }


def calibrate_noise_multiplier(
    unscaled_rho_profile: Dict[int, float],
    target_rhos: Dict[int, float],
    target_rho_30: float,
    delta: float = 1e-6,
) -> Dict[str, Any]:
    """Calculate noise multipliers beta_a and beta_b for a specific channel.

    Under noise scaling by beta:
      rho_scaled(h) = rho_unscaled(h) / (beta^2)
    - Caliber a: rho_scaled(30) = target_rho_30 => beta_a = sqrt(rho_unscaled(30) / target_rho_30)
    - Caliber b: rho_scaled(h) <= target_rhos[h] for all h => beta_b^2 = max_h (rho_unscaled(h) / target_rhos[h])
    """
    unscaled_30 = unscaled_rho_profile.get(30, target_rho_30)
    beta_a_sq = max(1e-12, unscaled_30 / target_rho_30)
    beta_a = math.sqrt(beta_a_sq)

    ratios_b = [
        unscaled_rho_profile[h] / max(1e-12, target_rhos[h])
        for h in HORIZONS
        if h in unscaled_rho_profile
    ]
    beta_b_sq = max(ratios_b) if ratios_b else 1.0
    beta_b = math.sqrt(beta_b_sq)

    # Scaled profiles
    profile_a = {h: unscaled_rho_profile[h] / beta_a_sq for h in HORIZONS}
    profile_b = {h: unscaled_rho_profile[h] / beta_b_sq for h in HORIZONS}

    # Verify monotonicity
    for p_name, prof in [("Caliber a", profile_a), ("Caliber b", profile_b)]:
        for i in range(len(HORIZONS) - 1):
            h_curr = HORIZONS[i]
            h_next = HORIZONS[i + 1]
            if prof[h_next] < prof[h_curr] - 1e-9:
                raise ValueError(
                    f"Profile monotonicity violation in {p_name}: h={h_curr} ({prof[h_curr]}) > h={h_next} ({prof[h_next]})"
                )

    eps_a = {h: rho_to_epsilon(profile_a[h], delta) for h in HORIZONS}
    eps_b = {h: rho_to_epsilon(profile_b[h], delta) for h in HORIZONS}

    return {
        "beta_a": round(beta_a, 4),
        "beta_b": round(beta_b, 4),
        "profile_a": profile_a,
        "profile_b": profile_b,
        "eps_a": eps_a,
        "eps_b": eps_b,
    }
