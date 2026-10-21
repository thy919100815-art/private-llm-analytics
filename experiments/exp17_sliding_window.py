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
from typing import Dict, List, Optional
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

            # Average canonical nodes used per sliding-window query
            nodes_used_list = [len(tree.sliding_window_cover(end_d, W)) for end_d in range(W, H + 1)]
            avg_nodes_used = float(np.mean(nodes_used_list))
            max_nodes_used = int(np.max(nodes_used_list))

            row = {
                "horizon_H": H,
                "window_W": W,
                "num_daily_queries": num_queries,
                "tree_levels": num_levels,
                "avg_nodes_per_window": round(avg_nodes_used, 2),
                "max_nodes_per_window": max_nodes_used,
                "independent_lifetime_epsilon": round(eps_ind, 4),
                "tree_lifetime_epsilon": round(eps_tree, 4),
                "epsilon_savings_ratio": round(eps_savings_ratio, 2),
            }
            rows.append(row)
            print(f"  H={H}, W={W}: Queries={num_queries} | Tree eps={eps_tree:.2f} vs Ind eps={eps_ind:.2f} (Savings={eps_savings_ratio:.1f}x)")

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
