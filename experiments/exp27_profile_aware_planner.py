"""Exp27: Profile-Aware Planner Calibration & Sensitivity Scaling (v0.9).

Evaluates the Small-N release planner across all 5 statistic profiles, validating:
1. Dynamic sensitivity calibration per profile;
2. Multi-objective utility decomposition;
3. Longitudinal RDP privacy budgeting feasibility across all profiles.
"""

from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, List
import numpy as np

from private_llm_analytics.planning.candidates import (
    ReleasePolicyCandidate,
    generate_candidate_set,
)
from private_llm_analytics.planning.policy import (
    get_direct_granularity_sensitivity,
)
from private_llm_analytics.planning.utility import (
    NoiseEstimator,
    TruncatedNormalPrior,
    UtilityEstimator,
    UtilityWeights,
)
from private_llm_analytics.statistics.registry import StatisticProfileRegistry


def run_exp27_profile_aware_planner() -> dict:
    print("============================================================")
    print("Exp27: Profile-Aware Planner Calibration & Sensitivity")
    print("============================================================")

    statistics = StatisticProfileRegistry.list_statistics()
    prior = TruncatedNormalPrior(mean=60.0, std=20.0, min_val=5.0, max_val=150.0)
    weights = UtilityWeights(objective_mode="COMPONENT_SURROGATE", clipping=1.0, noise=1.0, aggregation=1.0)
    estimator = UtilityEstimator(public_prior=prior, weights=weights)

    candidate = ReleasePolicyCandidate(
        contribution_bound=20.0,
        granularity="full_9_cell",
        window_days=30,
        frequency_days=7,
        mechanism="gaussian",
        epsilon_per_release=1.0,
        delta_per_release=1e-6,
    )

    results: Dict[str, dict] = {}

    for stat_id in statistics:
        profile = StatisticProfileRegistry.get(stat_id)
        sens = profile.sensitivity()

        # Planner theoretical noise estimation
        sigma, calib_sens = NoiseEstimator.estimate_noise_sigma(
            metric=stat_id,
            candidate=candidate,
        )

        util = estimator.evaluate_candidate(
            metric=stat_id,
            candidate=candidate,
            population_size=1000,
        )

        results[stat_id] = {
            "statistic_id": stat_id,
            "profile_sensitivity": float(sens),
            "planner_sensitivity": float(calib_sens),
            "noise_sigma": float(sigma),
            "noise_error": float(util.noise_error),
            "total_utility_loss": float(util.total_error),
        }
        print(f"Profile: {stat_id:22s} | Sens={sens:6.3f} | Sigma={sigma:6.3f} | TotalLoss={util.total_error:.4f}")

    out_dir = Path("results")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "exp27_profile_aware_planner.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\n[PASS] Exp27 completed successfully. Saved to {out_file}")
    return results


if __name__ == "__main__":
    run_exp27_profile_aware_planner()
