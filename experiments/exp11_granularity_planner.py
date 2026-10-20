"""Experiment 11: Granularity Release Planner (Direct Coarse Release vs Post-Processing).

Compares:
  - DIRECT_GRANULARITY_RELEASE:
      Per-user linear aggregation A_G -> target-space L2 clip to C -> noise calibration with sensitivity C.
  - POST_PROCESS_9_CELL:
      Full 9-cell DP release -> post-hoc linear aggregation to target granularity (0 incremental privacy budget).
Across sample sizes N in {500, 1000, 5000} and privacy budgets epsilon in {0.5, 1.0, 2.0, 4.0}.
"""

import argparse
from pathlib import Path
import time
from typing import Dict, List, Optional, Sequence
import numpy as np
import pandas as pd

from private_llm_analytics.aggregation.user_matrix import build_user_matrix
from private_llm_analytics.evaluation.report import plot_single_line_chart, save_summary_table
from private_llm_analytics.mechanisms.gaussian import calibrate_analytic_gaussian
from private_llm_analytics.planning.candidates import GRANULARITY_DIMENSIONS
from private_llm_analytics.planning.policy import (
    aggregate_to_granularity,
    build_volume_contributions_for_granularity,
    reconstruct_to_9_cell,
)
from private_llm_analytics.simulator.generator import SyntheticDataGenerator
from private_llm_analytics.simulator.scenario import SimulationScenario


def run_granularity_experiment(
    populations: Sequence[int] = (500, 1000, 5000),
    epsilons: Sequence[float] = (0.5, 1.0, 2.0, 4.0),
    delta: float = 1e-6,
    clip_norm: float = 20.0,
    n_trials: int = 200,
    seed: int = 42,
    output_dir: str = "results",
) -> pd.DataFrame:
    print("\n=======================================================")
    print("Running Exp 11: Granularity Release Planner (Direct vs Post-Processing)")
    print(f"Populations: {populations}")
    print(f"Epsilons: {epsilons}")
    print(f"Clip norm: {clip_norm}, Trials: {n_trials}")
    print("=======================================================")

    out_base = Path(output_dir)
    tables_dir = out_base / "tables"
    figures_dir = out_base / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    granularities = ["full_9_cell", "provider_3_cell", "task_3_cell", "overall_1_cell"]
    records: List[Dict] = []

    for n in populations:
        print(f"\n--- Population N = {n} ---")
        p = Path(f"data/interactions_n{n}.parquet")
        if p.is_file():
            df = pd.read_parquet(p)
        else:
            scenario = SimulationScenario(n_users=n, random_seed=seed)
            gen = SyntheticDataGenerator(scenario)
            df = gen.generate(seed=seed)
        umatrix = build_user_matrix(df, public_population_size=n)
        cell_names = umatrix.cell_names

        # Unclipped 9-cell ground truth
        unclipped_9 = np.sum(umatrix.matrix, axis=0)

        for eps in epsilons:
            # Calibrate Gaussian noise scale sigma for C=20
            sigma = calibrate_analytic_gaussian(epsilon=eps, delta=delta, l2_sensitivity=clip_norm)

            # Precompute 9-cell DP release for post-processing baseline
            # Clipped 9-cell exact
            clipped_9_users, exact_9, names_9, _ = build_volume_contributions_for_granularity(
                raw_user_matrix=umatrix,
                granularity="full_9_cell",
                clip_norm=clip_norm,
            )

            for g in granularities:
                # 1. DIRECT_GRANULARITY_RELEASE
                # Clip in target space to C -> sensitivity is strictly C
                clipped_mat, exact_target, target_names, sens = build_volume_contributions_for_granularity(
                    raw_user_matrix=umatrix,
                    granularity=g,
                    clip_norm=clip_norm,
                )
                assert sens == clip_norm

                direct_trial_errors_9cell: List[float] = []
                direct_trial_errors_target: List[float] = []
                post_proc_trial_errors_target: List[float] = []

                for t in range(n_trials):
                    rng = np.random.default_rng(seed + t * 100)

                    # A. Direct release in target G-space
                    noise_direct = rng.normal(loc=0.0, scale=sigma, size=len(exact_target))
                    dp_direct = exact_target + noise_direct

                    # Error on target granularity
                    true_target_unclipped, _ = aggregate_to_granularity(unclipped_9, cell_names, g)
                    err_target = float(np.mean(np.abs(dp_direct - true_target_unclipped)))
                    direct_trial_errors_target.append(err_target)

                    # Reconstruct to 9-cell
                    reconstructed_9 = reconstruct_to_9_cell(dp_direct, g, target_names, cell_names)
                    err_9cell = float(np.mean(np.abs(reconstructed_9 - unclipped_9)))
                    direct_trial_errors_9cell.append(err_9cell)

                    # B. POST_PROCESS_9_CELL (9-cell DP -> linear aggregation to target G)
                    noise_9 = rng.normal(loc=0.0, scale=sigma, size=9)
                    dp_9 = exact_9 + noise_9
                    post_proc_target, _ = aggregate_to_granularity(dp_9, cell_names, g)
                    err_post_proc = float(np.mean(np.abs(post_proc_target - true_target_unclipped)))
                    post_proc_trial_errors_target.append(err_post_proc)

                mean_direct_target = float(np.mean(direct_trial_errors_target))
                mean_direct_9cell = float(np.mean(direct_trial_errors_9cell))
                mean_post_proc_target = float(np.mean(post_proc_trial_errors_target))

                rec = {
                    "population_n": n,
                    "epsilon": eps,
                    "granularity": g,
                    "dimension": len(exact_target),
                    "direct_target_mae": round(mean_direct_target, 2),
                    "post_process_target_mae": round(mean_post_proc_target, 2),
                    "direct_reconstructed_9cell_mae": round(mean_direct_9cell, 2),
                    "direct_variance_reduction_pct": round(
                        max(0.0, (mean_post_proc_target - mean_direct_target) / max(1e-9, mean_post_proc_target) * 100), 2
                    ),
                }
                records.append(rec)
                print(
                    f"  N={n:4d}, eps={eps:3.1f}, G={g:15s} | Direct Target MAE={mean_direct_target:6.2f} | "
                    f"Post-Proc MAE={mean_post_proc_target:6.2f} | 9Cell Reconstructed={mean_direct_9cell:6.2f}"
                )

    results_df = pd.DataFrame(records)
    save_summary_table(results_df, str(tables_dir / "exp11_granularity_planner.csv"))

    # Plot Direct vs Post-Processing for small N=500
    sub_500 = results_df[results_df["population_n"] == populations[0]]
    plot_series = {}
    for g in ["provider_3_cell", "task_3_cell", "overall_1_cell"]:
        g_data = sub_500[sub_500["granularity"] == g]
        plot_series[f"{g} (Direct)"] = g_data["direct_target_mae"].tolist()
        plot_series[f"{g} (Post-Processed 9-cell)"] = g_data["post_process_target_mae"].tolist()

    plot_single_line_chart(
        x=list(epsilons),
        y_series=plot_series,
        x_label="Privacy Budget (epsilon)",
        y_label="Target Marginal MAE",
        title=f"Direct Granularity vs Post-Processing (N={populations[0]})",
        output_path=str(figures_dir / "exp11_granularity_planner"),
        x_log=True,
    )

    print(f"\n[SUCCESS] Exp 11 completed. Saved to {tables_dir}/ and {figures_dir}/")
    return results_df


def main():
    parser = argparse.ArgumentParser(description="Run Experiment 11: Granularity Release Planner")
    parser.add_argument("--smoke", action="store_true", help="Run rapid smoke test")
    parser.add_argument("--output-dir", type=str, default="results", help="Output directory")
    args = parser.parse_args()

    out_root = "results/verification" if args.smoke else args.output_dir
    if args.smoke:
        run_granularity_experiment(
            populations=(500, 1000),
            epsilons=(0.5, 2.0),
            n_trials=5,
            output_dir=out_root,
        )
    else:
        run_granularity_experiment(output_dir=out_root)


if __name__ == "__main__":
    main()
