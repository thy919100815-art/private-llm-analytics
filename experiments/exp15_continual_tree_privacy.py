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
from typing import Any, Dict, List, Optional, Tuple
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.temporal_truth import generate_rate_consistent_temporal_truth
from private_llm_analytics.continual.contributions import build_temporal_user_contributions
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

            # 4. Fair End-to-End Empirical Utility Simulation (Section H):
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

            # Build contributions:
            # - Raw (unbounded) tensor (N, T, 9)
            # - Independent (daily L2 bounded only) tensor (N, T, 9)
            # - Tree (daily L2 + epoch group-L12 bounded) tensor (N, T, 9)
            raw_contrib, _, _ = build_temporal_user_contributions(
                interactions=df, public_user_roster=roster, epoch_start="2026-10-01",
                epoch_days=T, daily_l2_bound=1e9, epoch_l12_bound=1e9,
            )
            ind_contrib, _, _ = build_temporal_user_contributions(
                interactions=df, public_user_roster=roster, epoch_start="2026-10-01",
                epoch_days=T, daily_l2_bound=c_day, epoch_l12_bound=1e9,
            )
            tree_contrib, _, _ = build_temporal_user_contributions(
                interactions=df, public_user_roster=roster, epoch_start="2026-10-01",
                epoch_days=T, daily_l2_bound=c_day, epoch_l12_bound=c_epoch,
            )

            # Evaluate 30-day interval query [1, min(30, T)]
            query_w = min(30, T)
            cover = tree.canonical_cover(1, query_w)
            tree_node_count = len(cover)

            # 9-cell ground truth vectors for the interval [1, query_w]
            V_raw = np.sum(raw_contrib[:, :query_w, :], axis=(0, 1))
            V_ind_bounded = np.sum(ind_contrib[:, :query_w, :], axis=(0, 1))
            V_tree_bounded = np.sum(tree_contrib[:, :query_w, :], axis=(0, 1))

            tree_clip_l1 = float(np.sum(np.abs(V_tree_bounded - V_raw)))
            ind_clip_l1 = float(np.sum(np.abs(V_ind_bounded - V_raw)))

            # Analytical expected noise MAE per cell: E[|Z|] = sigma * sqrt(2/pi)
            tree_noise_std = sigma_tree * math.sqrt(tree_node_count)
            tree_expected_mae = tree_noise_std * math.sqrt(2.0 / math.pi)

            ind_noise_std = sigma_ind * math.sqrt(query_w)
            ind_expected_mae = ind_noise_std * math.sqrt(2.0 / math.pi)

            # Empirical noise trials under FIXED LIFETIME EPSILON (Section H)
            rng = np.random.default_rng(seed=1000 + N + T)
            tree_tot_l1_trials, ind_tot_l1_trials = [], []
            tree_noise_l1_trials, ind_noise_l1_trials = [], []
            tree_mae_trials, ind_mae_trials = [], []
            tree_rmse_trials, ind_rmse_trials = [], []
            tree_rel_mae_trials, ind_rel_mae_trials = [], []
            tree_dp_tot_trials, ind_dp_tot_trials = [], []

            raw_mean_cell = max(1e-9, float(np.mean(V_raw)))

            for _ in range(n_trials):
                # Tree DP noise: sum of independent noises on canonical cover nodes
                tree_noise = rng.normal(0.0, sigma_tree * math.sqrt(tree_node_count), size=9)
                V_tree_dp = V_tree_bounded + tree_noise
                t_noise_l1 = float(np.sum(np.abs(tree_noise)))
                t_tot_l1 = float(np.sum(np.abs(V_tree_dp - V_raw)))
                t_mae = t_tot_l1 / 9.0
                t_rmse = float(math.sqrt(np.mean((V_tree_dp - V_raw) ** 2)))
                t_rel_mae = t_mae / raw_mean_cell

                tree_noise_l1_trials.append(t_noise_l1)
                tree_tot_l1_trials.append(t_tot_l1)
                tree_mae_trials.append(t_mae)
                tree_rmse_trials.append(t_rmse)
                tree_rel_mae_trials.append(t_rel_mae)
                tree_dp_tot_trials.append(float(np.sum(V_tree_dp)))

                # Independent DP noise: sum of independent noises on daily releases
                ind_noise = rng.normal(0.0, sigma_ind * math.sqrt(query_w), size=9)
                V_ind_dp = V_ind_bounded + ind_noise
                i_noise_l1 = float(np.sum(np.abs(ind_noise)))
                i_tot_l1 = float(np.sum(np.abs(V_ind_dp - V_raw)))
                i_mae = i_tot_l1 / 9.0
                i_rmse = float(math.sqrt(np.mean((V_ind_dp - V_raw) ** 2)))
                i_rel_mae = i_mae / raw_mean_cell

                ind_noise_l1_trials.append(i_noise_l1)
                ind_tot_l1_trials.append(i_tot_l1)
                ind_mae_trials.append(i_mae)
                ind_rmse_trials.append(i_rmse)
                ind_rel_mae_trials.append(i_rel_mae)
                ind_dp_tot_trials.append(float(np.sum(V_ind_dp)))

            tree_mae_mean = float(np.mean(tree_mae_trials))
            tree_mae_std = float(np.std(tree_mae_trials, ddof=1)) if n_trials > 1 else 0.0
            ind_mae_mean = float(np.mean(ind_mae_trials))
            ind_mae_std = float(np.std(ind_mae_trials, ddof=1)) if n_trials > 1 else 0.0

            ci_half_tree = 1.96 * tree_mae_std / math.sqrt(n_trials)
            ci_half_ind = 1.96 * ind_mae_std / math.sqrt(n_trials)

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
                # Section H: Fair end-to-end empirical comparison under fixed lifetime epsilon
                "raw_truth_total": round(float(np.sum(V_raw)), 2),
                "tree_bounded_truth_total": round(float(np.sum(V_tree_bounded)), 2),
                "ind_bounded_truth_total": round(float(np.sum(V_ind_bounded)), 2),
                "tree_dp_output_total_mean": round(float(np.mean(tree_dp_tot_trials)), 2),
                "ind_dp_output_total_mean": round(float(np.mean(ind_dp_tot_trials)), 2),
                "tree_clipping_error_l1": round(tree_clip_l1, 4),
                "ind_clipping_error_l1": round(ind_clip_l1, 4),
                "tree_noise_error_l1_mean": round(float(np.mean(tree_noise_l1_trials)), 4),
                "ind_noise_error_l1_mean": round(float(np.mean(ind_noise_l1_trials)), 4),
                "tree_total_l1_mean": round(float(np.mean(tree_tot_l1_trials)), 4),
                "ind_total_l1_mean": round(float(np.mean(ind_tot_l1_trials)), 4),
                "tree_mae_mean": round(tree_mae_mean, 4),
                "ind_mae_mean": round(ind_mae_mean, 4),
                "tree_rmse_mean": round(float(np.mean(tree_rmse_trials)), 4),
                "ind_rmse_mean": round(float(np.mean(ind_rmse_trials)), 4),
                "tree_relative_mae_mean": round(float(np.mean(tree_rel_mae_trials)), 4),
                "ind_relative_mae_mean": round(float(np.mean(ind_rel_mae_trials)), 4),
                "tree_mae_ci95_low": round(tree_mae_mean - ci_half_tree, 4),
                "tree_mae_ci95_high": round(tree_mae_mean + ci_half_tree, 4),
                "ind_mae_ci95_low": round(ind_mae_mean - ci_half_ind, 4),
                "ind_mae_ci95_high": round(ind_mae_mean + ci_half_ind, 4),
                "empirical_mae_reduction_factor": round(ind_mae_mean / max(1e-9, tree_mae_mean), 2),
            }
            rows.append(row)
            print(f"  N={N}, T={T}: Tree MAE={tree_mae_mean:.2f} vs Ind MAE={ind_mae_mean:.2f} (Empirical Reduction={row['empirical_mae_reduction_factor']:.2f}x)")

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
