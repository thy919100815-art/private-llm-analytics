"""Experiment 2: Epsilon Sweep (epsilon in {0.25, 0.5, 1, 2, 4, 8}) & Gaussian vs Laplace."""

import argparse
from pathlib import Path
import time
import pandas as pd

from typing import Optional
from private_llm_analytics.privacy.exceptions import DatasetConfigurationMismatchError
from private_llm_analytics.aggregation.user_matrix import build_user_matrix
from private_llm_analytics.evaluation.privacy_utility import run_repeated_noise_trials
from private_llm_analytics.evaluation.report import plot_single_line_chart, save_summary_table
from private_llm_analytics.metrics.interaction_volume import InteractionVolumeMetric
from private_llm_analytics.metrics.normalized_share import UserNormalizedShareMetric
from private_llm_analytics.metrics.prevalence import UserPrevalenceMetric
from private_llm_analytics.simulator.generator import SyntheticDataGenerator
from private_llm_analytics.simulator.scenario import SimulationScenario


def run_epsilon_sweep(
    n_users: int = 50000,
    epsilons = (0.25, 0.5, 1.0, 2.0, 4.0, 8.0),
    delta: float = 1e-6,
    n_trials: int = 500,
    seed: int = 42,
    data_path: str = "data/interactions_n50000.parquet",
    output_dir: str = "results",
    save_artifacts: bool = True,
) -> pd.DataFrame:
    print(f"\n=======================================================")
    print(f"Running Exp 02: Epsilon Sweep (N={n_users}, epsilons={epsilons}, trials={n_trials})")
    print(f"=======================================================")

    p = Path(data_path)
    if p.is_file():
        print(f"Loading dataset from {p}...")
        df = pd.read_parquet(p)
    else:
        print(f"Generating synthetic dataset with N={n_users} users...")
        scenario = SimulationScenario(n_users=n_users, random_seed=seed)
        gen = SyntheticDataGenerator(scenario)
        df = gen.generate_and_save(p, file_format="parquet", seed=seed)

    active_users = int(df["user_id"].nunique())
    print(f"Dataset: {p.name}")
    print(f"Active users: {active_users}")
    print(f"Public population: {n_users}")

    if active_users > n_users:
        raise DatasetConfigurationMismatchError(
            f"Active user count ({active_users}) exceeds declared public population size ({n_users})."
        )

    umatrix = build_user_matrix(df, public_population_size=n_users)


    metrics = [
        UserPrevalenceMetric(max_cells_per_user=9),
        UserNormalizedShareMetric(norm_bound=1.0),
        InteractionVolumeMetric(clip_norm=20.0),
    ]

    all_summaries = []
    all_trials_records = []

    # 1. Sweep Gaussian
    for metric in metrics:
        print(f"\n--- Metric: {metric.name} (Gaussian) ---")
        for eps in epsilons:
            t0 = time.time()
            trials_df, summary = run_repeated_noise_trials(
                user_matrix=umatrix,
                metric=metric,
                mechanism_type="gaussian",
                epsilon=eps,
                delta=delta,
                n_trials=n_trials,
                seed=seed + int(eps * 100),
            )
            trials_df["metric"] = metric.name
            trials_df["mechanism"] = "gaussian"
            trials_df["epsilon"] = eps
            all_trials_records.append(trials_df)

            print(f"  eps={eps:4.2f} (Gaussian): MAE={summary['mae_mean']:.5f}, "
                  f"RelErr={summary['relative_error_mean']:.4f}, "
                  f"Top1={summary['top1_accuracy']*100:.1f}%, time={time.time()-t0:.2f}s")
            all_summaries.append(summary)

    # 2. Sweep Laplace for comparison on User-Normalized Share and User Prevalence
    for metric in [UserNormalizedShareMetric(norm_bound=1.0), UserPrevalenceMetric(max_cells_per_user=9)]:
        print(f"\n--- Metric: {metric.name} (Laplace Baseline) ---")
        for eps in epsilons:
            t0 = time.time()
            trials_df, summary = run_repeated_noise_trials(
                user_matrix=umatrix,
                metric=metric,
                mechanism_type="laplace",
                epsilon=eps,
                delta=0.0,
                n_trials=n_trials,
                seed=seed + 500 + int(eps * 100),
            )
            trials_df["metric"] = metric.name
            trials_df["mechanism"] = "laplace"
            trials_df["epsilon"] = eps
            all_trials_records.append(trials_df)

            print(f"  eps={eps:4.2f} (Laplace):  MAE={summary['mae_mean']:.5f}, "
                  f"RelErr={summary['relative_error_mean']:.4f}, "
                  f"Top1={summary['top1_accuracy']*100:.1f}%, time={time.time()-t0:.2f}s")
            all_summaries.append(summary)

    summary_df = pd.DataFrame(all_summaries)

    if save_artifacts:
        base_dir = Path(output_dir) if output_dir else Path("results")
        tables_dir = base_dir / "tables"
        raw_dir = base_dir / "raw"
        figures_dir = base_dir / "figures"
        tables_dir.mkdir(parents=True, exist_ok=True)
        raw_dir.mkdir(parents=True, exist_ok=True)
        figures_dir.mkdir(parents=True, exist_ok=True)

        save_summary_table(summary_df, str(tables_dir / "exp02_epsilon_summary.csv"))

        combined_trials = pd.concat(all_trials_records, ignore_index=True)
        combined_trials.to_csv(str(raw_dir / "exp02_epsilon_trials.csv"), index=False)

        # 3. Generate Figures
        # A. MAE vs epsilon for Gaussian (Share and Prevalence)
        share_g = summary_df[(summary_df["metric"] == "user_normalized_share") & (summary_df["mechanism"].str.lower() == "gaussian")]
        share_l = summary_df[(summary_df["metric"] == "user_normalized_share") & (summary_df["mechanism"].str.lower() == "laplace")]
        prev_g = summary_df[(summary_df["metric"] == "user_prevalence") & (summary_df["mechanism"].str.lower() == "gaussian")]
        prev_l = summary_df[(summary_df["metric"] == "user_prevalence") & (summary_df["mechanism"].str.lower() == "laplace")]

        plot_single_line_chart(
            x=list(epsilons),
            y_series={
                "User-Normalized Share MAE (pp)": share_g["mae_pp_mean"].tolist(),
                "User Prevalence Rate MAE (pp)": prev_g["mae_pp_mean"].tolist(),
            },
            x_label="Privacy Budget (epsilon)",
            y_label="MAE (percentage points)",
            title=f"Accuracy Error vs Epsilon (Gaussian Mechanism, N={n_users:,})",
            output_path=str(figures_dir / "exp02_mae_vs_epsilon"),
            x_log=True,
        )

        # B. TV vs epsilon (Only valid for User-Normalized Share probability simplex)
        plot_single_line_chart(
            x=list(epsilons),
            y_series={
                "Normalized Share TV (Gaussian)": share_g["tv_mean"].tolist(),
                "Normalized Share TV (Laplace)": share_l["tv_mean"].tolist(),
            },
            x_label="Privacy Budget (epsilon)",
            y_label="Total Variation Distance",
            title=f"Distribution TV Distance vs Epsilon (Normalized Share, N={n_users:,})",
            output_path=str(figures_dir / "exp02_tv_vs_epsilon"),
            x_log=True,
        )

        # C. Relative Error vs epsilon
        vol_g = summary_df[(summary_df["metric"] == "interaction_volume") & (summary_df["mechanism"] == "Gaussian")]
        plot_single_line_chart(
            x=list(epsilons),
            y_series={
                "User Prevalence": prev_g["relative_error_mean"].tolist(),
                "User-Normalized Share": share_g["relative_error_mean"].tolist(),
                "Interaction Volume": vol_g["relative_error_mean"].tolist(),
            },
            x_label="Privacy Budget (epsilon)",
            y_label="Mean Relative Error",
            title=f"Relative Error vs Epsilon across Metrics (N={n_users:,})",
            output_path=str(figures_dir / "exp02_relerr_vs_epsilon"),
            x_log=True,
            y_log=True,
        )

        # D. Gaussian vs Laplace Comparison on Normalized Share
        share_l = summary_df[(summary_df["metric"] == "user_normalized_share") & (summary_df["mechanism"] == "Laplace")]
        plot_single_line_chart(
            x=list(epsilons),
            y_series={
                "Gaussian (eps, 1e-6)-DP": share_g["mae_pp_mean"].tolist(),
                "Laplace eps-DP": share_l["mae_pp_mean"].tolist(),
            },
            x_label="Privacy Budget (epsilon)",
            y_label="MAE (percentage points)",
            title="Gaussian vs Laplace Trade-off: User-Normalized Share",
            output_path=str(figures_dir / "exp02_gaussian_vs_laplace"),
            x_log=True,
        )

        print("\n[SUCCESS] Exp 02 Epsilon sweep complete. Tables & figures saved.")
    else:
        print("\n[SUCCESS] Exp 02 Epsilon sweep complete (artifacts not saved).")
    return summary_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_users", type=int, default=50000)
    parser.add_argument("--trials", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="results")
    args = parser.parse_args()
    run_epsilon_sweep(n_users=args.n_users, n_trials=args.trials, seed=args.seed, output_dir=args.output_dir)
