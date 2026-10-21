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

    rows: List[Dict[str, Any]] = []

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
            prefix_res = engine.query_prefix(cfg, end_day=t, recipient="Analyst", interactions=df, public_user_roster=roster)
            nodes_count = len(prefix_res.nodes_used)

            # Expected noise std
            tree_noise_std = float(sigma_tree * math.sqrt(nodes_count))
            tree_mae = tree_noise_std * math.sqrt(2.0 / math.pi)

            # Independent: sum of t independent daily releases
            ind_noise_std = float(sigma_ind * math.sqrt(t))
            ind_mae = ind_noise_std * math.sqrt(2.0 / math.pi)

            noise_reduction = float(ind_mae / max(1e-9, tree_mae))

            row = {
                "prefix_day_t": t,
                "tree_nodes_used": nodes_count,
                "tree_noise_std": round(tree_noise_std, 4),
                "tree_noise_mae": round(tree_mae, 4),
                "independent_days_summed": t,
                "independent_noise_std": round(ind_noise_std, 4),
                "independent_noise_mae": round(ind_mae, 4),
                "noise_reduction_factor": round(noise_reduction, 2),
            }
            rows.append(row)
            print(f"  Prefix t={t}: Tree MAE={tree_mae:.2f} (nodes={nodes_count}) vs Ind MAE={ind_mae:.2f} (days={t})")

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
