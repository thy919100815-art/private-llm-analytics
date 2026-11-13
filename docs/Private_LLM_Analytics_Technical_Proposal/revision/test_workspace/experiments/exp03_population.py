"""Experiment 3: Population Sweep (N in {500, 1000, 5000, 10000, 50000, 100000}) & Small-N Analysis."""

import argparse
from pathlib import Path
import time
import pandas as pd

from private_llm_analytics.aggregation.user_matrix import build_user_matrix
from private_llm_analytics.evaluation.privacy_utility import run_repeated_noise_trials
from private_llm_analytics.evaluation.report import plot_single_line_chart, save_summary_table
from private_llm_analytics.metrics.interaction_volume import InteractionVolumeMetric
from private_llm_analytics.metrics.normalized_share import UserNormalizedShareMetric
from private_llm_analytics.metrics.prevalence import UserPrevalenceMetric
from private_llm_analytics.simulator.generator import SyntheticDataGenerator
from private_llm_analytics.simulator.scenario import SimulationScenario


def run_population_sweep(
    populations = (500, 1000, 5000, 10000, 50000, 100000),
    epsilon: float = 1.0,
    delta: float = 1e-6,
    default_trials: int = 500,
    large_n_trials: int = 100,
    seed: int = 42,
    output_dir: str = "results",
) -> pd.DataFrame:
    print(f"\n=======================================================")
    print(f"Running Exp 03: Population Sweep (N in {populations}, eps={epsilon})")
    print(f"=======================================================")

    all_summaries = []
    all_trials_records = []
    scalability_benchmarks = []

    for n in populations:
        print(f"\n====================== Population N = {n:,} ======================")
        data_file = Path(f"data/interactions_n{n}.parquet")
        t_gen_start = time.time()

        if data_file.is_file():
            print(f"Loading existing dataset from {data_file}...")
            df = pd.read_parquet(data_file)
            gen_time = 0.0
        else:
            print(f"Generating synthetic telemetry for N={n:,}...")
            scenario = SimulationScenario(n_users=n, random_seed=seed)
            gen = SyntheticDataGenerator(scenario)
            df = gen.generate_and_save(data_file, file_format="parquet", seed=seed)
            gen_time = time.time() - t_gen_start
            print(f"Generated & saved {len(df):,} interactions in {gen_time:.2f}s")

        t_agg_start = time.time()
        umatrix = build_user_matrix(df, public_population_size=n)
        agg_time = time.time() - t_agg_start
        print(f"User matrix aggregated in {agg_time:.2f}s (Shape: {umatrix.matrix.shape}, public N={n})")


        file_size_mb = data_file.stat().st_size / (1024 * 1024)
        scalability_benchmarks.append({
            "n_users": n,
            "n_interactions": len(df),
            "parquet_size_mb": file_size_mb,
            "gen_time_sec": gen_time,
            "agg_time_sec": agg_time,
        })

        # Set trials: 100 for N=100k for responsiveness, 500 for N <= 50k
        n_trials = large_n_trials if n >= 100000 else default_trials

        metrics = [
            UserPrevalenceMetric(max_cells_per_user=9),
            UserNormalizedShareMetric(norm_bound=1.0),
            InteractionVolumeMetric(clip_norm=20.0),
        ]

        for metric in metrics:
            t0 = time.time()
            trials_df, summary = run_repeated_noise_trials(
                user_matrix=umatrix,
                metric=metric,
                mechanism_type="gaussian",
                epsilon=epsilon,
                delta=delta,
                n_trials=n_trials,
                seed=seed + int(n % 10000),
            )
            trials_df["metric"] = metric.name
            trials_df["population"] = n
            trials_df["epsilon"] = epsilon
            all_trials_records.append(trials_df)

            print(f"  Metric: {metric.name:<22} | MAE: {summary['mae_mean']:.5f} | "
                  f"Top-1: {summary['top1_accuracy']*100:.1f}% | Time: {time.time()-t0:.2f}s")
            all_summaries.append(summary)

    base_dir = Path(output_dir)
    tables_dir = base_dir / "tables"
    raw_dir = base_dir / "raw"
    figures_dir = base_dir / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    summary_df = pd.DataFrame(all_summaries)
    save_summary_table(summary_df, str(tables_dir / "exp03_population_summary.csv"))

    bench_df = pd.DataFrame(scalability_benchmarks)
    save_summary_table(bench_df, str(tables_dir / "scalability_benchmark.csv"))

    combined_trials = pd.concat(all_trials_records, ignore_index=True)
    combined_trials.to_csv(str(raw_dir / "exp03_population_trials.csv"), index=False)

    # Figures
    share_sum = summary_df[summary_df["metric"] == "user_normalized_share"]
    prev_sum = summary_df[summary_df["metric"] == "user_prevalence"]

    # 1. MAE (pp) vs Population N
    plot_single_line_chart(
        x=[float(n) for n in populations],
        y_series={
            "User-Normalized Share MAE (pp)": share_sum["mae_pp_mean"].tolist(),
            "User Prevalence Rate MAE (pp)": prev_sum["mae_pp_mean"].tolist(),
        },
        x_label="Population Size (N)",
        y_label="MAE (percentage points)",
        title="Privacy Utility Scaling with Population N (eps=1.0)",
        output_path=str(figures_dir / "exp03_mae_vs_population"),
        x_log=True,
    )

    # 2. TV vs Population N (Valid for User-Normalized Share probability simplex)
    plot_single_line_chart(
        x=[float(n) for n in populations],
        y_series={
            "User-Normalized Share TV": share_sum["tv_mean"].tolist(),
        },
        x_label="Population Size (N)",
        y_label="Total Variation Distance",
        title="Normalized Share TV Distance vs Population N (eps=1.0)",
        output_path=str(figures_dir / "exp03_tv_vs_population"),
        x_log=True,
    )

    # 3. Small-N Breakdown Figure: Comparison of Relative Error and Top1 Accuracy across N
    vol_sum = summary_df[summary_df["metric"] == "interaction_volume"]
    plot_single_line_chart(
        x=[float(n) for n in populations],
        y_series={
            "Prevalence Top-1": prev_sum["top1_accuracy"].tolist(),
            "Share Top-1": share_sum["top1_accuracy"].tolist(),
            "Volume Top-1": vol_sum["top1_accuracy"].tolist(),
        },
        x_label="Population Size (N)",
        y_label="Top-1 Ranking Accuracy",
        title="Ranking Preservation vs Population Size (eps=1.0)",
        output_path=str(figures_dir / "exp03_small_n_breakdown"),
        x_log=True,
    )

    print(f"\n[SUCCESS] Exp 03 Population sweep complete. Results saved to {tables_dir}/ and {figures_dir}/")
    return summary_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epsilon", type=float, default=1.0)
    parser.add_argument("--trials", type=int, default=500)
    parser.add_argument("--large_n_trials", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="results")
    args = parser.parse_args()

    run_population_sweep(
        epsilon=args.epsilon,
        default_trials=args.trials,
        large_n_trials=args.large_n_trials,
        seed=args.seed,
        output_dir=args.output_dir,
    )
