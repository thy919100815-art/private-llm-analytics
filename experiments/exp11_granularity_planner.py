"""Experiment 11: Granularity Release Planner (Aggregation vs Noise Trade-off).

Evaluates granularities:
  - full_9_cell
  - provider_3_cell
  - task_3_cell
  - overall_1_cell
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
from private_llm_analytics.metrics.interaction_volume import InteractionVolumeMetric
from private_llm_analytics.mechanisms.gaussian import calibrate_analytic_gaussian
from private_llm_analytics.planning.candidates import ReleasePolicyCandidate
from private_llm_analytics.planning.utility import GRANULARITY_DIMENSIONS
from private_llm_analytics.planning.policy import (
    aggregate_to_granularity,
    reconstruct_to_9_cell,
)
from private_llm_analytics.simulator.generator import SyntheticDataGenerator
from private_llm_analytics.simulator.scenario import SimulationScenario


def run_granularity_experiment(
    populations: Sequence[int] = (500, 1000, 5000),
    epsilons: Sequence[float] = (0.5, 1.0, 2.0, 4.0),
    delta: float = 1e-6,
    clip_norm: float = 20.0,
    n_trials: int = 100,
    seed: int = 42,
    output_dir: str = "results",
) -> pd.DataFrame:
    print("\n=======================================================")
    print("Running Exp 11: Granularity Release Planner")
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

        # Ground truth unclipped & clipped 9-cell
        metric = InteractionVolumeMetric(clip_norm=clip_norm)
        true_9_cell = metric.compute_exact(umatrix)
        cell_names = umatrix.cell_names

        for eps in epsilons:
            # Calibrate Gaussian noise scale sigma for C=20
            sigma = calibrate_analytic_gaussian(epsilon=eps, delta=delta, l2_sensitivity=clip_norm)

            for g in granularities:
                # 1. Deterministic marginal aggregation
                agg_vec, agg_names = aggregate_to_granularity(true_9_cell, cell_names, g)

                # 2. Repeated noisy release and reconstruction to 9-cell
                trial_errors: List[float] = []
                for t in range(n_trials):
                    rng = np.random.default_rng(seed + t * 100)
                    noise = rng.normal(loc=0.0, scale=sigma, size=len(agg_vec))
                    noisy_agg = agg_vec + noise

                    # Reconstruct to 9-cell
                    reconstructed_9 = reconstruct_to_9_cell(noisy_agg, g, agg_names, cell_names)

                    # Compute L1 MAE per cell against true 9-cell
                    mae = float(np.mean(np.abs(reconstructed_9 - true_9_cell)))
                    trial_errors.append(mae)

                mean_mae = float(np.mean(trial_errors))
                std_mae = float(np.std(trial_errors))

                rec = {
                    "population_n": n,
                    "epsilon": eps,
                    "granularity": g,
                    "dimension": len(agg_vec),
                    "mean_mae_9cell": round(mean_mae, 2),
                    "std_mae_9cell": round(std_mae, 2),
                }
                records.append(rec)
                print(f"  N={n:4d}, eps={eps:3.1f}, G={g:15s} | Reconstructed MAE = {mean_mae:6.2f}")

    results_df = pd.DataFrame(records)
    save_summary_table(results_df, str(tables_dir / "exp11_granularity_planner.csv"))

    # Plot for small N=500 across epsilons
    sub_500 = results_df[results_df["population_n"] == populations[0]]
    plot_series = {}
    for g in granularities:
        plot_series[g] = sub_500[sub_500["granularity"] == g]["mean_mae_9cell"].tolist()

    plot_single_line_chart(
        x=list(epsilons),
        y_series=plot_series,
        x_label="Privacy Budget (epsilon)",
        y_label="Reconstructed 9-Cell MAE",
        title=f"Granularity Error vs Privacy Budget (N={populations[0]})",
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
            n_trials=10,
            output_dir=out_root,
        )
    else:
        run_granularity_experiment(output_dir=out_root)


if __name__ == "__main__":
    main()
