"""Experiment 4: Interaction Volume Clipping Sweep (C in {5, 10, 20, 40, 80})."""

import argparse
from pathlib import Path
import time
import pandas as pd

from private_llm_analytics.aggregation.user_matrix import build_user_matrix
from private_llm_analytics.evaluation.privacy_utility import run_repeated_noise_trials
from private_llm_analytics.evaluation.report import plot_single_line_chart, save_summary_table
from private_llm_analytics.metrics.interaction_volume import InteractionVolumeMetric
from private_llm_analytics.simulator.generator import SyntheticDataGenerator
from private_llm_analytics.simulator.scenario import SimulationScenario


def run_clipping_sweep(
    n_users: int = 10000,
    clipping_norms = (5.0, 10.0, 20.0, 40.0, 80.0),
    epsilon: float = 1.0,
    delta: float = 1e-6,
    n_trials: int = 500,
    seed: int = 42,
    data_path: str = "data/interactions_n10000.parquet",
    output_dir: str = "results",
) -> pd.DataFrame:
    print(f"\n=======================================================")
    print(f"Running Exp 04: Clipping Sweep (C in {clipping_norms}, N={n_users}, eps={epsilon})")
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

    umatrix = build_user_matrix(df, public_population_size=n_users)


    all_summaries = []
    all_trials_records = []

    for c in clipping_norms:
        print(f"\n--- Testing Clipping Norm C = {c:.1f} ---")
        metric = InteractionVolumeMetric(clip_norm=c)
        t0 = time.time()
        trials_df, summary = run_repeated_noise_trials(
            user_matrix=umatrix,
            metric=metric,
            mechanism_type="gaussian",
            epsilon=epsilon,
            delta=delta,
            n_trials=n_trials,
            seed=seed + int(c * 10),
        )
        trials_df["clip_norm"] = c
        trials_df["epsilon"] = epsilon
        all_trials_records.append(trials_df)

        print(f"  C={c:4.1f}: Clipping Bias MAE={summary['clipping_bias_mae']:.2f} | "
              f"DP Noise MAE={summary['dp_noise_mae_mean']:.2f} | "
              f"Total Error MAE={summary['total_volume_mae_mean']:.2f} | Time={time.time()-t0:.2f}s")
        summary["clip_norm"] = c
        all_summaries.append(summary)

    base_dir = Path(output_dir)
    tables_dir = base_dir / "tables"
    raw_dir = base_dir / "raw"
    figures_dir = base_dir / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    summary_df = pd.DataFrame(all_summaries)
    save_summary_table(summary_df, str(tables_dir / "exp04_clipping_summary.csv"))

    combined_trials = pd.concat(all_trials_records, ignore_index=True)
    combined_trials.to_csv(str(raw_dir / "exp04_clipping_trials.csv"), index=False)

    # Plot figure: Error Decomposition Trade-off vs Clipping Norm C
    plot_single_line_chart(
        x=list(clipping_norms),
        y_series={
            "Clipping Bias": summary_df["clipping_bias_mae"].tolist(),
            "DP Noise Error": summary_df["dp_noise_mae_mean"].tolist(),
            "Total Volume Error": summary_df["total_volume_mae_mean"].tolist(),
        },
        x_label="User L2 Clipping Bound (C)",
        y_label="Error (MAE on cell volumes)",
        title=f"Interaction Volume Error Decomposition (N={n_users}, eps={epsilon})",
        output_path=str(figures_dir / "exp04_clipping_tradeoff"),
        x_log=True,
    )

    print(f"\n[SUCCESS] Exp 04 Clipping sweep complete. Results saved to {tables_dir}/ and {figures_dir}/")
    return summary_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_users", type=int, default=10000)
    parser.add_argument("--trials", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="results")
    args = parser.parse_args()

    run_clipping_sweep(
        n_users=args.n_users,
        n_trials=args.trials,
        seed=args.seed,
        output_dir=args.output_dir,
    )
