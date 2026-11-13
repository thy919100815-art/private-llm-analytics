"""Experiment 1: Baseline Evaluation (N=50,000, epsilon=1.0, delta=1e-6)."""

import argparse
from pathlib import Path
import time
import numpy as np
import pandas as pd

from private_llm_analytics.aggregation.user_matrix import build_user_matrix
from private_llm_analytics.evaluation.privacy_utility import run_repeated_noise_trials
from private_llm_analytics.evaluation.report import plot_single_line_chart, save_summary_table
from private_llm_analytics.metrics.interaction_volume import InteractionVolumeMetric
from private_llm_analytics.metrics.normalized_share import UserNormalizedShareMetric
from private_llm_analytics.metrics.prevalence import UserPrevalenceMetric
from private_llm_analytics.simulator.generator import SyntheticDataGenerator
from private_llm_analytics.simulator.scenario import SimulationScenario


def run_baseline_experiment(
    n_users: int = 50000,
    epsilon: float = 1.0,
    delta: float = 1e-6,
    n_trials: int = 500,
    seed: int = 42,
    data_path: str = "data/interactions_n50000.parquet",
    output_dir: str = "results",
) -> pd.DataFrame:
    print(f"\n=======================================================")
    print(f"Running Exp 01: Baseline (N={n_users}, eps={epsilon}, delta={delta}, trials={n_trials})")
    print(f"=======================================================")

    # 1. Generate or load synthetic telemetry
    p = Path(data_path)
    if p.is_file():
        print(f"Loading existing dataset from {p}...")
        df = pd.read_parquet(p)
    else:
        print(f"Generating synthetic dataset with N={n_users} users...")
        t0 = time.time()
        scenario = SimulationScenario(n_users=n_users, random_seed=seed)
        gen = SyntheticDataGenerator(scenario)
        df = gen.generate_and_save(p, file_format="parquet", seed=seed)
        print(f"Generated {len(df):,} interactions in {time.time() - t0:.2f}s, saved to {p}")

    # 2. Build user-level matrix
    t0 = time.time()
    umatrix = build_user_matrix(df, public_population_size=n_users)
    print(f"Aggregated user matrix shape: {umatrix.matrix.shape} (public N={n_users}) in {time.time() - t0:.2f}s")


    # 3. Evaluate the 3 metrics
    metrics = [
        UserPrevalenceMetric(max_cells_per_user=9),
        UserNormalizedShareMetric(norm_bound=1.0),
        InteractionVolumeMetric(clip_norm=20.0),
    ]

    all_summaries = []
    all_trials_records = []

    for metric in metrics:
        print(f"\n--- Running {n_trials} trials for Metric: {metric.name} ---")
        t0 = time.time()
        trials_df, summary = run_repeated_noise_trials(
            user_matrix=umatrix,
            metric=metric,
            mechanism_type="gaussian",
            epsilon=epsilon,
            delta=delta,
            n_trials=n_trials,
            seed=seed + 100,
        )
        trials_df["metric"] = metric.name
        trials_df["population"] = n_users
        trials_df["epsilon"] = epsilon
        all_trials_records.append(trials_df)

        print(f"Completed in {time.time() - t0:.2f}s")
        print(f"  MAE (mean): {summary['mae_mean']:.6f} | Median: {summary['mae_median']:.6f} | P95: {summary['mae_p95']:.6f}")
        if "mae_pp_mean" in summary:
            tv_info = f" | TV (mean): {summary['tv_mean']:.5f}" if "tv_mean" in summary else ""
            print(f"  MAE (percentage points): {summary['mae_pp_mean']:.4f}%{tv_info}")

        if "total_volume_mae_mean" in summary:
            print(f"  Volume Clipping Bias: {summary['clipping_bias_mae']:.2f} | DP Noise MAE: {summary['dp_noise_mae_mean']:.2f} | Total Error: {summary['total_volume_mae_mean']:.2f}")
        print(f"  Ranking Top-1 Match: {summary['top1_accuracy']*100:.1f}% | Spearman: {summary['spearman_mean']:.4f}")

        all_summaries.append(summary)

    # 4. Save results to designated output_dir
    base_dir = Path(output_dir)
    tables_dir = base_dir / "tables"
    raw_dir = base_dir / "raw"
    figures_dir = base_dir / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    summary_df = pd.DataFrame(all_summaries)
    save_summary_table(summary_df, str(tables_dir / "exp01_baseline_summary.csv"))

    combined_trials = pd.concat(all_trials_records, ignore_index=True)
    combined_trials.to_csv(str(raw_dir / "exp01_baseline_trials.csv"), index=False)

    # 5. Plot figure
    plot_single_line_chart(
        x=[m["metric"] for m in all_summaries],
        y_series={
            "Mean Relative Error": [m["relative_error_mean"] for m in all_summaries],
            "Ranking Accuracy": [m["top1_accuracy"] for m in all_summaries],
        },
        x_label="Metric",
        y_label="Value",
        title=f"Baseline Error & Ranking Accuracy (N={n_users}, eps={epsilon})",
        output_path=str(figures_dir / "exp01_baseline_error"),
    )

    print(f"\n[SUCCESS] Exp 01 Baseline complete. Results saved to {tables_dir}/ and {figures_dir}/")
    return summary_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_users", type=int, default=50000)
    parser.add_argument("--trials", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="results")
    args = parser.parse_args()

    run_baseline_experiment(n_users=args.n_users, n_trials=args.trials, seed=args.seed, output_dir=args.output_dir)
