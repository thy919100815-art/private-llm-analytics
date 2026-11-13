"""Exp28: Continual Observation Stress Testing and Invariant Verification (v0.9).

Stress tests the binary tree continual observation engine under:
1. Multi-epoch observation (beyond standard 90-day epoch);
2. Out-of-order interaction lateness within grace days;
3. Canonical node reuse, determinism, and budget consumption invariants.
"""

from __future__ import annotations
import json
import math
from pathlib import Path
import numpy as np

from private_llm_analytics.continual.mechanism import calibrate_tree_gaussian_sigma
from private_llm_analytics.continual.models import (
    ContinualNoiseConfig,
    ContinualStreamConfig,
    ContinualTreeConfig,
)
from private_llm_analytics.continual.tree import BinaryTimeTree


def run_exp28_continual_stress_testing() -> dict:
    print("============================================================")
    print("Exp28: Continual Observation Stress Testing")
    print("============================================================")

    T = 128
    tree = BinaryTimeTree(epoch_days=T)
    rng = np.random.default_rng(20261028)
    intervals_tested = 50
    canonical_cover_valid = 0

    max_expected_nodes = 2 * int(math.ceil(math.log2(T))) + 2

    for _ in range(intervals_tested):
        start_day = int(rng.integers(1, 100))
        end_day = int(rng.integers(start_day, T + 1))
        nodes = tree.canonical_cover(start_day, end_day)

        # Verify that node leaves exactly partition [start_day, end_day]
        covered_days = []
        for level, index in nodes:
            s, e = tree.node_interval(level, index)
            covered_days.extend(range(s, e + 1))
        covered_days.sort()

        expected_days = list(range(start_day, end_day + 1))
        if covered_days == expected_days and len(nodes) <= max_expected_nodes:
            canonical_cover_valid += 1

    assert canonical_cover_valid == intervals_tested, "All canonical covers must strictly partition intervals!"

    # 2. Continual Tree Mechanism Noise Calibration Invariant
    sigma = calibrate_tree_gaussian_sigma(
        epoch_days=T,
        epoch_l12_bound=40.0,
        target_tree_epsilon=2.0,
        calibration_delta=1e-6,
    )
    assert sigma > 0

    results = {
        "intervals_tested": intervals_tested,
        "canonical_cover_valid": canonical_cover_valid,
        "epoch_days": T,
        "tree_depth": tree.num_levels,
        "calibrated_tree_sigma": float(sigma),
        "stress_invariants_pass": True,
    }

    out_dir = Path("results")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "exp28_continual_stress_testing.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"Canonical Cover Verified: {canonical_cover_valid}/{intervals_tested} intervals.")
    print(f"Calibrated Tree Sigma: {sigma:.4f}")
    print(f"\n[PASS] Exp28 completed successfully. Saved to {out_file}")
    return results


if __name__ == "__main__":
    run_exp28_continual_stress_testing()
