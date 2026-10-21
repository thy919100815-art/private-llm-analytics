#!/usr/bin/env python3
"""Experiment 18: Temporal Clipping Tradeoff (C_day vs C_epoch) (Section 66).

Formal Evaluation:
Measures the fundamental bias-variance tradeoff introduced by temporal contribution bounding:
- Daily L2 clipping (C_day)
- Epoch group-L1-of-L2 clipping (C_epoch)
Shows how C_epoch controls the tradeoff between:
1. Temporal clipping bias: ||x - z||_1
2. Calibrated tree DP noise scale: sigma_tree(C_epoch)
"""

import argparse
import math
from pathlib import Path
import sys
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
from private_llm_analytics.continual.contributions import build_temporal_user_contributions
from private_llm_analytics.continual.mechanism import calibrate_tree_gaussian_sigma


def run_exp18(smoke: bool = False, output_dir: Optional[Path] = None):
    print(f"=== Starting Exp18 (smoke={smoke}) ===")
    out_dir = output_dir or (Path("results/verification") if smoke else Path("results/tables"))
    fig_dir = Path("results/verification") if smoke else Path("results/figures")
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    T = 90
    N = 500 if smoke else 1000
    target_eps = 2.0
    delta = 1e-6

    c_day_grid = [10.0, 20.0] if smoke else [5.0, 10.0, 20.0, 40.0]
    c_epoch_grid = [20.0, 40.0, 80.0] if smoke else [10.0, 20.0, 40.0, 80.0, 160.0]

    df = generate_rate_consistent_temporal_truth(
        n_users=N,
        horizon_days=T,
        reference_days=30,
        reference_mean=60.0,
        reference_std=20.0,
        seed=42,
    )
    roster = sorted(df["user_id"].unique().tolist())

    rows: List[Dict[str, Any]] = []

    for c_day in c_day_grid:
        for c_epoch in c_epoch_grid:
            contribs, diag, _ = build_temporal_user_contributions(
                interactions=df,
                public_user_roster=roster,
                epoch_start="2026-10-01",
                epoch_days=T,
                daily_l2_bound=c_day,
                epoch_l12_bound=c_epoch,
            )

            # Calibrate sigma for tree at target_eps
            sigma = calibrate_tree_gaussian_sigma(
                epoch_days=T,
                epoch_l12_bound=c_epoch,
                target_tree_epsilon=target_eps,
                calibration_delta=delta,
            )

            # Relative biases normalized by raw total
            raw_tot = max(1e-9, diag.raw_total)
            rel_daily_bias = float(diag.daily_clipping_bias / raw_tot)
            rel_temporal_bias = float(diag.temporal_clipping_bias / raw_tot)
            total_clipping_bias = diag.daily_clipping_bias + diag.temporal_clipping_bias
            rel_total_clipping_bias = float(total_clipping_bias / raw_tot)

            # Expected 30-day query noise error (uses approx 4 tree nodes)
            nodes_in_30d = 4
            expected_noise_l1 = 9 * sigma * math.sqrt(nodes_in_30d) * math.sqrt(2.0 / math.pi)
            rel_noise_l1 = float(expected_noise_l1 / raw_tot)

            rel_total_error = rel_total_clipping_bias + rel_noise_l1

            row = {
                "horizon_T": T,
                "population_N": N,
                "C_day": c_day,
                "C_epoch": c_epoch,
                "raw_total": round(diag.raw_total, 2),
                "daily_clipping_bias": round(diag.daily_clipping_bias, 2),
                "temporal_clipping_bias": round(diag.temporal_clipping_bias, 2),
                "total_clipping_bias": round(total_clipping_bias, 2),
                "fraction_daily_clipped": round(diag.fraction_daily_clipped, 4),
                "fraction_epoch_clipped": round(diag.fraction_epoch_clipped, 4),
                "calibrated_sigma": round(sigma, 4),
                "rel_clipping_bias": round(rel_total_clipping_bias, 6),
                "rel_noise_l1": round(rel_noise_l1, 6),
                "rel_total_error": round(rel_total_error, 6),
            }
            rows.append(row)
            print(f"  C_day={c_day}, C_epoch={c_epoch}: ClipBias={rel_total_clipping_bias:.4f}, Noise={rel_noise_l1:.4f}, Total={rel_total_error:.4f}")

    df_out = pd.DataFrame(rows)
    csv_path = out_dir / "exp18_temporal_clipping.csv"
    df_out.to_csv(csv_path, index=False)
    print(f"Saved CSV: {csv_path}")

    # Plot tradeoff for fixed C_day=10.0
    df_sub = df_out[df_out["C_day"] == 10.0]
    plt.figure(figsize=(8, 5))
    plt.plot(df_sub["C_epoch"], df_sub["rel_clipping_bias"], "o-", label="Relative Clipping Bias", color="#D9534F")
    plt.plot(df_sub["C_epoch"], df_sub["rel_noise_l1"], "s-", label="Relative DP Noise L1", color="#0275D8")
    plt.plot(df_sub["C_epoch"], df_sub["rel_total_error"], "^--", label="Total Relative Error", color="#5CB85C", linewidth=2)

    plt.xlabel("Epoch Group Norm Bound (C_epoch)")
    plt.ylabel("Relative Error (Dimensionless)")
    plt.title("Exp18: Temporal Contribution Bounding Tradeoff (C_day=10.0, T=90)")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()

    plt.savefig(fig_dir / "exp18_temporal_clipping.png", dpi=150)
    plt.savefig(fig_dir / "exp18_temporal_clipping.pdf")
    plt.close()

    print("Exp18 completed successfully.")
    return df_out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", help="Run quick smoke test")
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()
    out_path = Path(args.output_dir) if args.output_dir else None
    run_exp18(smoke=args.smoke, output_dir=out_path)
