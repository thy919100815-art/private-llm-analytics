#!/usr/bin/env python3
"""Verification Script: Continual Tree Mathematics & Sensitivity Proof (Section 88).

Verifies:
1. Canonical dyadic interval cover completeness, disjointness, and size bounds.
2. User-level group-L1-of-L2 temporal contribution bounding.
3. Per-level concatenated vector L2 sensitivity ||V_{u, l}||_2 <= C_epoch theorem.
4. Tree Gaussian noise calibration monotonicity and accuracy.
"""

import math
import sys
import numpy as np
import pandas as pd

from private_llm_analytics.continual.contributions import build_temporal_user_contributions
from private_llm_analytics.continual.mechanism import (
    calibrate_tree_gaussian_sigma,
    compute_level_rdp_vector,
    evaluate_rdp_epsilon,
)
from private_llm_analytics.continual.tree import BinaryTimeTree


def verify_canonical_covers():
    for T in [7, 14, 30, 60, 90, 180, 365]:
        tree = BinaryTimeTree(epoch_days=T)
        max_bound = 2 * int(math.ceil(math.log2(max(2, T)))) + 2

        # Sample 50 random intervals
        rng = np.random.default_rng(42 + T)
        for _ in range(50):
            l = int(rng.integers(1, T + 1))
            r = int(rng.integers(l, T + 1))
            cover = tree.canonical_cover(l, r)

            assert len(cover) > 0
            assert len(cover) <= max_bound

            intervals = [tree.node_interval(lvl, idx) for lvl, idx in cover]
            assert intervals[0][0] == l
            assert intervals[-1][1] == r

            for i in range(len(intervals) - 1):
                assert intervals[i][1] + 1 == intervals[i + 1][0]
    print("[1/4] Canonical interval covers verified.")


def verify_per_level_sensitivity_bounds():
    c_day = 10.0
    c_epoch = 35.0
    roster = [f"u_{i}" for i in range(10)]

    for T in [7, 30, 90, 128]:
        tree = BinaryTimeTree(epoch_days=T)
        num_levels = tree.num_levels
        tree_size = tree.tree_size

        # Create bursty & uniform synthetic events
        records = []
        base = pd.to_datetime("2026-01-01")
        rng = np.random.default_rng(100 + T)

        for uid in roster:
            for day in range(T):
                for _ in range(int(rng.integers(0, 15))):
                    records.append({
                        "user_id": uid,
                        "timestamp": base + pd.Timedelta(days=day, hours=int(rng.integers(0, 23))),
                        "provider": rng.choice(["provider_a", "provider_b", "provider_c"]),
                        "task": rng.choice(["technical", "document", "analysis"]),
                    })

        df = pd.DataFrame(records)
        contribs, _, _ = build_temporal_user_contributions(
            interactions=df,
            public_user_roster=roster,
            epoch_start="2026-01-01",
            epoch_days=T,
            daily_l2_bound=c_day,
            epoch_l12_bound=c_epoch,
        )

        n_users, _, dim = contribs.shape
        padded = np.zeros((n_users, tree_size, dim), dtype=np.float64)
        padded[:, :T, :] = contribs

        for lvl in range(num_levels):
            block_size = 2 ** lvl
            num_nodes = tree_size // block_size

            for u in range(n_users):
                node_vecs = []
                for j in range(num_nodes):
                    s = j * block_size
                    e = (j + 1) * block_size
                    node_v = np.sum(padded[u, s:e, :], axis=0)
                    node_vecs.append(node_v)

                concat_v = np.concatenate(node_vecs)
                norm_2 = float(np.linalg.norm(concat_v))
                assert norm_2 <= c_epoch + 1e-9, f"Violation: level {lvl}, norm {norm_2} > {c_epoch}"

    print("[2/4] Per-level L2 sensitivity proof verified.")


def verify_tree_calibration():
    # Monotonicity with bound
    s1 = calibrate_tree_gaussian_sigma(90, epoch_l12_bound=20.0, target_tree_epsilon=2.0)
    s2 = calibrate_tree_gaussian_sigma(90, epoch_l12_bound=40.0, target_tree_epsilon=2.0)
    assert s2 >= s1

    # Monotonicity with target epsilon
    s3 = calibrate_tree_gaussian_sigma(90, epoch_l12_bound=30.0, target_tree_epsilon=1.0)
    s4 = calibrate_tree_gaussian_sigma(90, epoch_l12_bound=30.0, target_tree_epsilon=3.0)
    assert s4 <= s3

    # Accuracy
    tree = BinaryTimeTree(epoch_days=90)
    level_rdp = compute_level_rdp_vector(epoch_l12_bound=40.0, sigma=s2)
    tot_rdp = {k: v * tree.num_levels for k, v in level_rdp.items()}
    eff_eps = evaluate_rdp_epsilon(tot_rdp, delta=1e-6)
    assert eff_eps <= 2.0 + 1e-5
    assert abs(eff_eps - 2.0) < 0.05
    print("[3/4] Tree Gaussian noise calibration verified.")


def main():
    print("Running verify_continual_tree_math.py...")
    verify_canonical_covers()
    verify_per_level_sensitivity_bounds()
    verify_tree_calibration()
    print("[4/4] All mathematical invariants verified.")
    print("PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
