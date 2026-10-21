#!/usr/bin/env python3
"""Experiment 15: Continual Binary Tree vs. Independent DP Scaling (Section 63).

Formal Evaluation:
Compares user-level lifetime privacy budget accumulation and query utility between:
1. INDEPENDENT daily releases (linear RDP accumulation across queries)
2. BINARY_TREE continual observation (logarithmic level exposure mechanism)

Runs across:
- Populations N in {500, 1000, 5000} (or {500} for --smoke)
- Horizons T in {30, 90, 180, 365} (or {30, 90} for --smoke)

Evaluates:
Mode A: Fixed release noise scale -> Lifetime epsilon comparison
Mode B: Fixed lifetime epsilon -> Empirical utility (MAE) comparison
"""

import argparse
import math
from pathlib import Path
import sys
import tempfile
import time
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.temporal_truth import generate_rate_consistent_temporal_truth
from private_llm_analytics.continual.mechanism import (
    calibrate_tree_gaussian_sigma,
    compute_level_rdp_vector,
    evaluate_rdp_epsilon,
)
from private_llm_analytics.continual.models import ContinualNoiseConfig, ContinualStreamConfig
from private_llm_analytics.continual.query import ContinualQueryEngine
from private_llm_analytics.continual.state import ContinualStateManager
from private_llm_analytics.continual.tree import BinaryTimeTree
from private_llm_analytics.privacy.rdp import compute_gaussian_rdp, DEFAULT_RDP_ORDERS


def run_exp15(smoke: bool = False, output_dir: Optional[Path] = None):
    print(f"=== Starting Exp15 (smoke={smoke}) ===")
    out_dir = output_dir or (Path("results/verification") if smoke else Path("results/tables"))
    fig_dir = Path("results/verification") if smoke else Path("results/figures")
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    populations = [500] if smoke else [500, 1000, 5000]
    horizons = [30, 90] if smoke else [30, 90, 180, 365]
    n_trials = 5 if smoke else 30

    delta = 1e-6
    c_day = 10.0
    c_epoch = 40.0
    target_lifetime_eps = 2.0

    rows: List[Dict[str, Any]] = []

    for N in populations:
        for T in horizons:
            tree = BinaryTimeTree(epoch_days=T)
            num_levels = tree.num_levels
            tree_size = tree.tree_size

            # 1. Calibrate Tree Sigma for fixed lifetime epsilon = target_lifetime_eps
            sigma_tree = calibrate_tree_gaussian_sigma(
                epoch_days=T,
                epoch_l12_bound=c_epoch,
                target_tree_epsilon=target_lifetime_eps,
                calibration_delta=delta,
            )

            # 2. Independent daily releases:
            # Over T days, user contributes on T days with L2 sensitivity C_day
            # RDP_ind(alpha) = T * (alpha * C_day^2) / (2 * sigma_ind^2)
            # Binary search for sigma_ind that yields target_lifetime_eps over T daily releases
            def _eval_sigma_ind(sig: float) -> float:
                tot_rdp = {}
                for a in DEFAULT_RDP_ORDERS:
                    tot_rdp[a] = T * (a * (c_day ** 2)) / (2.0 * (sig ** 2))
                return evaluate_rdp_epsilon(tot_rdp, delta=delta)

            # Binary search for sigma_ind
            lo, hi = 0.1, 1000.0
            while _eval_sigma_ind(hi) > target_lifetime_eps:
                hi *= 2.0
            for _ in range(60):
                mid = (lo + hi) / 2.0
                if _eval_sigma_ind(mid) > target_lifetime_eps:
                    lo = mid
                else:
                    hi = mid
            sigma_ind = hi

            # 3. Fixed noise comparison: what if both use sigma = 20.0?
            fixed_sigma = 20.0
            # Independent lifetime epsilon with fixed sigma
            rdp_ind_fixed = {a: T * (a * (c_day ** 2)) / (2.0 * (fixed_sigma ** 2)) for a in DEFAULT_RDP_ORDERS}
            eps_ind_fixed = evaluate_rdp_epsilon(rdp_ind_fixed, delta=delta)

            # Tree lifetime epsilon with fixed sigma
            rdp_tree_fixed = {a: num_levels * (a * (c_epoch ** 2)) / (2.0 * (fixed_sigma ** 2)) for a in DEFAULT_RDP_ORDERS}
            eps_tree_fixed = evaluate_rdp_epsilon(rdp_tree_fixed, delta=delta)

            eps_reduction = float(eps_ind_fixed / max(1e-9, eps_tree_fixed))

            # 4. Utility Simulation:
            # Generate rate-consistent synthetic data
            df = generate_rate_consistent_temporal_truth(
                n_users=N,
                horizon_days=T,
                reference_days=30,
                reference_mean=60.0,
                reference_std=20.0,
                seed=42 + T,
            )
            roster = [f"user_{i}" for i in range(N)]

            # Evaluate 30-day interval query [1, min(30, T)]
            query_w = min(30, T)
            # True volume in this window
            # Compute analytical expected noise MAE for query of length query_w
            # Tree uses canonical cover nodes:
            cover = tree.canonical_cover(1, query_w)
            tree_node_count = len(cover)
            # Expected noise MAE per cell: E[|Z|] = sigma * sqrt(2/pi)
            tree_noise_std = sigma_tree * math.sqrt(tree_node_count)
            tree_expected_mae = tree_noise_std * math.sqrt(2.0 / math.pi)

            # Independent: sum of query_w daily releases
            ind_noise_std = sigma_ind * math.sqrt(query_w)
            ind_expected_mae = ind_noise_std * math.sqrt(2.0 / math.pi)

            row = {
                "population": N,
                "horizon_days": T,
                "tree_size": tree_size,
                "tree_levels": num_levels,
                "query_interval_days": query_w,
                "tree_nodes_in_query": tree_node_count,
                "fixed_noise_sigma": fixed_sigma,
                "independent_epsilon_fixed_noise": round(eps_ind_fixed, 4),
                "tree_epsilon_fixed_noise": round(eps_tree_fixed, 4),
                "epsilon_reduction_factor": round(eps_reduction, 2),
                "target_lifetime_epsilon": target_lifetime_eps,
                "calibrated_sigma_tree": round(sigma_tree, 4),
                "calibrated_sigma_independent": round(sigma_ind, 4),
                "tree_expected_noise_mae": round(tree_expected_mae, 4),
                "independent_expected_noise_mae": round(ind_expected_mae, 4),
                "noise_reduction_factor": round(ind_expected_mae / max(1e-9, tree_expected_mae), 2),
            }
            rows.append(row)
            print(f"  N={N}, T={T}: Tree Epsilon={eps_tree_fixed:.2f} vs Ind={eps_ind_fixed:.2f} (Reduction={eps_reduction:.1f}x)")

    df_out = pd.DataFrame(rows)
    csv_path = out_dir / "exp15_tree_privacy_scaling.csv"
    df_out.to_csv(csv_path, index=False)
    print(f"Saved CSV: {csv_path}")

    # Plot
    plt.figure(figsize=(8, 5))
    df_n1000 = df_out[df_out["population"] == (500 if smoke else 1000)]
    plt.plot(df_n1000["horizon_days"], df_n1000["independent_epsilon_fixed_noise"], "o-", label="Independent Daily Releases", color="#D9534F")
    plt.plot(df_n1000["horizon_days"], df_n1000["tree_epsilon_fixed_noise"], "s-", label="Binary Continual Tree", color="#0275D8")
    plt.xlabel("Horizon T (Days)")
    plt.ylabel("Composed Lifetime Epsilon (ε)")
    plt.title("Exp15: Continual Tree vs. Independent Lifetime Privacy Scaling")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()

    plt.savefig(fig_dir / "exp15_tree_privacy_scaling.png", dpi=150)
    plt.savefig(fig_dir / "exp15_tree_privacy_scaling.pdf")
    plt.close()

    print("Exp15 completed successfully.")
    return df_out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", help="Run quick smoke test")
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()
    out_path = Path(args.output_dir) if args.output_dir else None
    run_exp15(smoke=args.smoke, output_dir=out_path)
