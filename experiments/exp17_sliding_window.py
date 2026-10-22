#!/usr/bin/env python3
"""Experiment 17: Sliding-Window Differential Privacy (Section 65).

Formal Evaluation:
Compares daily repeated sliding-window releases (linear RDP accumulation) vs.
binary tree sliding-window queries (bounded level exposure with zero additional cost for reused levels).

Evaluates:
- Sliding window durations W in {7, 30, 90}
- Horizons H in {90, 180, 365}
"""

import argparse
import math
from pathlib import Path
import sys
import tempfile
from typing import Any, Dict, List, Optional
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
    evaluate_rdp_epsilon,
)
from private_llm_analytics.continual.models import ContinualNoiseConfig, ContinualStreamConfig
from private_llm_analytics.continual.query import ContinualQueryEngine
from private_llm_analytics.continual.state import ContinualStateManager
from private_llm_analytics.continual.tree import BinaryTimeTree
from private_llm_analytics.privacy.rdp import DEFAULT_RDP_ORDERS


def run_exp17(smoke: bool = False, output_dir: Optional[Path] = None):
    print(f"=== Starting Exp17 (smoke={smoke}) ===")
    out_dir = output_dir or (Path("results/verification") if smoke else Path("results/tables"))
    fig_dir = Path("results/verification") if smoke else Path("results/figures")
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    windows = [7, 30] if smoke else [7, 30, 90]
    horizons = [90] if smoke else [90, 180, 365]

    c_day = 10.0
    c_epoch = 40.0
    fixed_sigma = 20.0
    delta = 1e-6

    rows: List[Dict[str, Any]] = []

    for H in horizons:
        tree = BinaryTimeTree(epoch_days=H)
        num_levels = tree.num_levels

        for W in windows:
            if W > H:
                continue
            # Number of daily sliding window releases over horizon H
            # Day W, W+1, ..., H
            num_queries = H - W + 1
            target_eps = 2.0

            # 1. Independent overlapping releases:
            # Each daily sliding window release of window W has user-level L2 sensitivity bounded by:
            # Under user-level DP, a user's entire history across W days can contribute up to W * C_day or group bound.
            # In direct window release with L2 clipping C_window = min(c_epoch, W * c_day):
            c_window = min(c_epoch, float(W) * c_day)
            # Each of the num_queries overlapping releases adds fresh noise and fresh RDP:
            # RDP_ind_total = num_queries * (alpha * C_window^2) / (2 * sigma^2)
            rdp_ind_tot = {
                a: num_queries * (a * (c_window ** 2)) / (2.0 * (fixed_sigma ** 2))
                for a in DEFAULT_RDP_ORDERS
            }
            eps_ind = evaluate_rdp_epsilon(rdp_ind_tot, delta=delta)

            # Find budget exhaustion day for independent baseline under fixed_sigma
            ind_exhaustion_day = None
            for k in range(1, num_queries + 1):
                rdp_k = {a: k * (a * (c_window ** 2)) / (2.0 * (fixed_sigma ** 2)) for a in DEFAULT_RDP_ORDERS}
                if evaluate_rdp_epsilon(rdp_k, delta=delta) > target_eps:
                    ind_exhaustion_day = W + k - 1
                    break

            # 2. Binary Tree Continual Observation:
            # All num_queries sliding-window queries decompose into canonical tree nodes.
            # Tree level exposures are charged ONCE per level.
            # Total RDP is strictly bounded by num_levels level mechanisms:
            rdp_tree_tot = {
                a: num_levels * (a * (c_epoch ** 2)) / (2.0 * (fixed_sigma ** 2))
                for a in DEFAULT_RDP_ORDERS
            }
            eps_tree = evaluate_rdp_epsilon(rdp_tree_tot, delta=delta)

            eps_savings_ratio = float(eps_ind / max(1e-9, eps_tree))

            # Canonical nodes used per sliding-window query
            nodes_used_list = [len(tree.sliding_window_cover(end_d, W)) for end_d in range(W, H + 1)]
            avg_nodes_used = float(np.mean(nodes_used_list))
            p95_nodes_used = float(np.percentile(nodes_used_list, 95))
            max_nodes_used = int(np.max(nodes_used_list))

            # 3. Fixed Lifetime Epsilon Utility Comparison (Section H):
            sigma_tree_cal = calibrate_tree_gaussian_sigma(
                epoch_days=H,
                epoch_l12_bound=c_epoch,
                target_tree_epsilon=target_eps,
                calibration_delta=delta,
            )

            # Calibrate independent baseline to avoid exhausting target_eps over num_queries
            def _eval_sigma_ind_sw(sig: float) -> float:
                tot_rdp = {a: num_queries * (a * (c_window ** 2)) / (2.0 * (sig ** 2)) for a in DEFAULT_RDP_ORDERS}
                return evaluate_rdp_epsilon(tot_rdp, delta=delta)

            lo, hi = 0.1, 5000.0
            while _eval_sigma_ind_sw(hi) > target_eps:
                hi *= 2.0
            for _ in range(60):
                mid = (lo + hi) / 2.0
                if _eval_sigma_ind_sw(mid) > target_eps:
                    lo = mid
                else:
                    hi = mid
            sigma_ind_cal = hi

            # Synthetic data evaluation across sliding windows
            N_eval = 500 if smoke else 1000
            df = generate_rate_consistent_temporal_truth(
                n_users=N_eval, horizon_days=H, reference_days=30, reference_mean=60.0, reference_std=20.0, seed=77 + H + W
            )
            roster = sorted(df["user_id"].unique().tolist())
            raw_contrib, _, _ = build_temporal_user_contributions(
                df, roster, "2026-10-01", H, daily_l2_bound=1e9, epoch_l12_bound=1e9
            )
            ind_contrib, _, _ = build_temporal_user_contributions(
                df, roster, "2026-10-01", H, daily_l2_bound=c_day, epoch_l12_bound=1e9
            )
            tree_contrib, _, _ = build_temporal_user_contributions(
                df, roster, "2026-10-01", H, daily_l2_bound=c_day, epoch_l12_bound=c_epoch
            )

            rng = np.random.default_rng(seed=3000 + H + W)
            # Sample up to 15 sliding windows across the horizon
            step_stride = max(1, num_queries // (5 if smoke else 15))
            eval_ends = list(range(W, H + 1, step_stride))

            t_maes, i_maes = [], []
            t_rmses, i_rmses = [], []
            t_rel_maes, i_rel_maes = [], []
            t_tot_l1s, i_tot_l1s = [], []

            for end_d in eval_ends:
                start_d = end_d - W + 1
                V_raw = np.sum(raw_contrib[:, start_d-1:end_d, :], axis=(0, 1))
                V_ind_bounded = np.sum(ind_contrib[:, start_d-1:end_d, :], axis=(0, 1))
                V_tree_bounded = np.sum(tree_contrib[:, start_d-1:end_d, :], axis=(0, 1))

                raw_mean = max(1e-9, float(np.mean(V_raw)))
                cover_nodes = tree.sliding_window_cover(end_d, W)

                # Tree DP noise
                noise_tree = rng.normal(0.0, sigma_tree_cal * math.sqrt(len(cover_nodes)), size=9)
                V_tree_dp = V_tree_bounded + noise_tree
                t_l1 = float(np.sum(np.abs(V_tree_dp - V_raw)))
                t_mae = t_l1 / 9.0
                t_rmse = float(math.sqrt(np.mean((V_tree_dp - V_raw) ** 2)))
                t_rel = t_mae / raw_mean

                # Ind DP noise
                noise_ind = rng.normal(0.0, sigma_ind_cal, size=9)
                V_ind_dp = V_ind_bounded + noise_ind
                i_l1 = float(np.sum(np.abs(V_ind_dp - V_raw)))
                i_mae = i_l1 / 9.0
                i_rmse = float(math.sqrt(np.mean((V_ind_dp - V_raw) ** 2)))
                i_rel = i_mae / raw_mean

                t_tot_l1s.append(t_l1)
                i_tot_l1s.append(i_l1)
                t_maes.append(t_mae)
                i_maes.append(i_mae)
                t_rmses.append(t_rmse)
                i_rmses.append(i_rmse)
                t_rel_maes.append(t_rel)
                i_rel_maes.append(i_rel)

            mean_t_mae = float(np.mean(t_maes))
            mean_i_mae = float(np.mean(i_maes))
            mean_t_rmse = float(np.mean(t_rmses))
            mean_i_rmse = float(np.mean(i_rmses))
            mean_t_rel = float(np.mean(t_rel_maes))
            mean_i_rel = float(np.mean(i_rel_maes))
            mean_t_l1 = float(np.mean(t_tot_l1s))
            mean_i_l1 = float(np.mean(i_tot_l1s))

            row = {
                "horizon_H": H,
                "window_W": W,
                "num_daily_queries": num_queries,
                "tree_levels": num_levels,
                "avg_nodes_per_window": round(avg_nodes_used, 2),
                "p95_nodes_per_window": round(p95_nodes_used, 2),
                "max_nodes_per_window": max_nodes_used,
                "independent_lifetime_epsilon": round(eps_ind, 4),
                "tree_lifetime_epsilon": round(eps_tree, 4),
                "epsilon_savings_ratio": round(eps_savings_ratio, 2),
                "independent_budget_exhaustion_day": ind_exhaustion_day if ind_exhaustion_day is not None else -1,
                "target_lifetime_epsilon": target_eps,
                "calibrated_sigma_tree": round(sigma_tree_cal, 4),
                "calibrated_sigma_independent": round(sigma_ind_cal, 4),
                "tree_mae": round(mean_t_mae, 4),
                "independent_mae": round(mean_i_mae, 4),
                "tree_rmse": round(mean_t_rmse, 4),
                "independent_rmse": round(mean_i_rmse, 4),
                "tree_relative_mae": round(mean_t_rel, 4),
                "independent_relative_mae": round(mean_i_rel, 4),
                "tree_total_l1": round(mean_t_l1, 4),
                "independent_total_l1": round(mean_i_l1, 4),
                "fixed_budget_utility_ratio": round(mean_i_mae / max(1e-9, mean_t_mae), 2),
            }
            rows.append(row)
            print(f"  H={H}, W={W}: Queries={num_queries} | Tree MAE={mean_t_mae:.2f} vs Ind MAE={mean_i_mae:.2f} (Savings={eps_savings_ratio:.1f}x, Util Ratio={row['fixed_budget_utility_ratio']:.2f}x)")

    df_out = pd.DataFrame(rows)
    csv_path = out_dir / "exp17_sliding_window.csv"
    df_out.to_csv(csv_path, index=False)
    print(f"Saved CSV: {csv_path}")

    # Plot for H=365 or max H
    max_h = max(horizons)
    df_sub = df_out[df_out["horizon_H"] == max_h]

    plt.figure(figsize=(8, 5))
    x_indices = np.arange(len(df_sub))
    width = 0.35

    plt.bar(x_indices - width/2, df_sub["independent_lifetime_epsilon"], width, label="Independent Daily Releases", color="#D9534F")
    plt.bar(x_indices + width/2, df_sub["tree_lifetime_epsilon"], width, label="Binary Continual Tree", color="#0275D8")

    plt.xticks(x_indices, [f"W={w}d" for w in df_sub["window_W"]])
    plt.xlabel(f"Sliding Window Size (Horizon H={max_h} days)")
    plt.ylabel("Composed Lifetime Epsilon (ε)")
    plt.title(f"Exp17: Sliding-Window DP Lifetime Privacy (H={max_h} Days)")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5, axis="y")
    plt.tight_layout()

    plt.savefig(fig_dir / "exp17_sliding_window.png", dpi=150)
    plt.savefig(fig_dir / "exp17_sliding_window.pdf")
    plt.close()

    print("Exp17 completed successfully.")
    return df_out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", help="Run quick smoke test")
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()
    out_path = Path(args.output_dir) if args.output_dir else None
    run_exp17(smoke=args.smoke, output_dir=out_path)
