#!/usr/bin/env python3
"""Experiment 16: Prefix Tracking (Quarter-to-Date / Horizon Prefix) (Section 64).

Formal Evaluation:
Analyzes cumulative error growth for prefix queries [1, t] as t advances:
1. Binary Tree prefix queries (noise grows as O(log t))
2. Independent daily release accumulation (noise variance grows as O(t))
Across t in {7, 14, 30, 60, 90} for horizon T=90, N=1000.
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
from private_llm_analytics.continual.mechanism import calibrate_tree_gaussian_sigma, evaluate_rdp_epsilon
from private_llm_analytics.continual.models import ContinualNoiseConfig, ContinualStreamConfig
from private_llm_analytics.continual.query import ContinualQueryEngine
from private_llm_analytics.continual.state import ContinualStateManager
from private_llm_analytics.continual.tree import BinaryTimeTree
from private_llm_analytics.privacy.rdp import DEFAULT_RDP_ORDERS


def run_exp16(smoke: bool = False, output_dir: Optional[Path] = None):
    print(f"=== Starting Exp16 (smoke={smoke}) ===")
    out_dir = output_dir or (Path("results/verification") if smoke else Path("results/tables"))
    fig_dir = Path("results/verification") if smoke else Path("results/figures")
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    T = 90
    N = 500 if smoke else 1000
    prefix_days = [7, 14, 30] if smoke else [7, 14, 30, 45, 60, 75, 90]
    n_trials = 5 if smoke else 50

    c_day = 10.0
    c_epoch = 40.0
    target_eps = 2.0
    delta = 1e-6

    tree = BinaryTimeTree(epoch_days=T)
    sigma_tree = calibrate_tree_gaussian_sigma(
        epoch_days=T,
        epoch_l12_bound=c_epoch,
        target_tree_epsilon=target_eps,
        calibration_delta=delta,
    )

    # Independent daily release calibration for lifetime target_eps
    def _eval_sigma_ind(sig: float) -> float:
        tot_rdp = {a: T * (a * (c_day ** 2)) / (2.0 * (sig ** 2)) for a in DEFAULT_RDP_ORDERS}
        return evaluate_rdp_epsilon(tot_rdp, delta=delta)

    lo, hi = 0.1, 1000.0
    while _eval_sigma_ind(hi) > target_eps:
        hi *= 2.0
    for _ in range(60):
        mid = (lo + hi) / 2.0
        if _eval_sigma_ind(mid) > target_eps:
            lo = mid
        else:
            hi = mid
    sigma_ind = hi

    df = generate_rate_consistent_temporal_truth(
        n_users=N,
        horizon_days=T,
        reference_days=30,
        reference_mean=60.0,
        reference_std=20.0,
        seed=123,
    )
    roster = sorted(df["user_id"].unique().tolist())

    # Build reference tensors for Fair End-to-End Comparison (Section H):
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

    rows: List[Dict[str, Any]] = []
    rng = np.random.default_rng(seed=2026)

    with tempfile.TemporaryDirectory() as tmp_dir:
        db_path = Path(tmp_dir) / "exp16.db"
        state_mgr = ContinualStateManager(db_path=db_path)
        cfg = ContinualStreamConfig(
            stream_id="exp16_stream",
            epoch_id="epoch_001",
            epoch_start="2026-10-01",
            epoch_days=T,
            daily_l2_bound=c_day,
            epoch_group_l12_bound=c_epoch,
            noise=ContinualNoiseConfig(target_epoch_epsilon=target_eps, sigma=sigma_tree),
        )
        engine = ContinualQueryEngine(state_manager=state_mgr, composition_mode="global_external")

        for t in prefix_days:
            # Tree prefix query [1, t]
            prefix_res = engine.query_prefix(
                cfg,
                end_day=t,
                recipient="Analyst",
                interactions=df,
                public_user_roster=roster,
                finalized_up_to_day=T,
                current_wall_clock_day=T + 1,
            )
            nodes_count = len(prefix_res.nodes_used)

            # Expected noise std
            tree_noise_std = float(sigma_tree * math.sqrt(nodes_count))
            tree_mae = tree_noise_std * math.sqrt(2.0 / math.pi)

            # Independent: sum of t independent daily releases
            ind_noise_std = float(sigma_ind * math.sqrt(t))
            ind_mae = ind_noise_std * math.sqrt(2.0 / math.pi)

            noise_reduction = float(ind_mae / max(1e-9, tree_mae))

            # Ground truth vectors for prefix [1, t]
            V_raw = np.sum(raw_contrib[:, :t, :], axis=(0, 1))
            V_ind_bounded = np.sum(ind_contrib[:, :t, :], axis=(0, 1))
            V_tree_bounded = np.sum(tree_contrib[:, :t, :], axis=(0, 1))

            tree_clip_l1 = float(np.sum(np.abs(V_tree_bounded - V_raw)))
            ind_clip_l1 = float(np.sum(np.abs(V_ind_bounded - V_raw)))
            raw_mean_cell = max(1e-9, float(np.mean(V_raw)))

            # Empirical noise trials for prefix query (Section H)
            tree_tot_l1_trials, ind_tot_l1_trials = [], []
            tree_noise_l1_trials, ind_noise_l1_trials = [], []
            tree_mae_trials, ind_mae_trials = [], []
            tree_rmse_trials, ind_rmse_trials = [], []
            tree_rel_mae_trials, ind_rel_mae_trials = [], []
            tree_dp_tot_trials, ind_dp_tot_trials = [], []

            for _ in range(n_trials):
                tree_noise = rng.normal(0.0, tree_noise_std, size=9)
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

                ind_noise = rng.normal(0.0, ind_noise_std, size=9)
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
                "prefix_day_t": t,
                "tree_nodes_used": nodes_count,
                "tree_noise_std": round(tree_noise_std, 4),
                "tree_noise_mae": round(tree_mae, 4),
                "independent_days_summed": t,
                "independent_noise_std": round(ind_noise_std, 4),
                "independent_noise_mae": round(ind_mae, 4),
                "noise_reduction_factor": round(noise_reduction, 2),
                # Section H: End-to-end empirical comparison under fixed lifetime epsilon
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
                "empirical_error_reduction_factor": round(ind_mae_mean / max(1e-9, tree_mae_mean), 2),
            }
            rows.append(row)
            print(f"  Prefix t={t}: Tree MAE={tree_mae_mean:.2f} (nodes={nodes_count}) vs Ind MAE={ind_mae_mean:.2f} (days={t})")

    df_out = pd.DataFrame(rows)
    csv_path = out_dir / "exp16_prefix_queries.csv"
    df_out.to_csv(csv_path, index=False)
    print(f"Saved CSV: {csv_path}")

    # Plot
    plt.figure(figsize=(8, 5))
    plt.plot(df_out["prefix_day_t"], df_out["independent_noise_mae"], "o-", label="Independent Daily Sum (O(t) variance)", color="#D9534F")
    plt.plot(df_out["prefix_day_t"], df_out["tree_noise_mae"], "s-", label="Binary Continual Tree (O(log t) variance)", color="#0275D8")
    plt.xlabel("Prefix Day t (Interval [1, t])")
    plt.ylabel("Expected Noise MAE per Cell")
    plt.title("Exp16: Prefix Tracking Noise Comparison")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()

    plt.savefig(fig_dir / "exp16_prefix_queries.png", dpi=150)
    plt.savefig(fig_dir / "exp16_prefix_queries.pdf")
    plt.close()

    print("Exp16 completed successfully.")
    return df_out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", help="Run quick smoke test")
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()
    out_path = Path(args.output_dir) if args.output_dir else None
    run_exp16(smoke=args.smoke, output_dir=out_path)
