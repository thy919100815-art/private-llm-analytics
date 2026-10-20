"""Top-level script to execute all research experiments sequentially."""

import argparse
import sys
import time
from typing import Optional

from experiments.exp01_baseline import run_baseline_experiment
from experiments.exp02_epsilon import run_epsilon_sweep
from experiments.exp03_population import run_population_sweep
from experiments.exp04_clipping import run_clipping_sweep
from experiments.exp05_composition import run_composition_experiment
from experiments.exp06_release_frequency import run_release_frequency_experiment
from experiments.exp07_recipient_composition import run_recipient_composition_experiment
from experiments.exp08_longitudinal_budget import run_longitudinal_budget_experiment
from experiments.exp09_small_n_error_decomposition import run_error_decomposition_experiment
from experiments.exp10_clipping_planner import run_clipping_planner_experiment
from experiments.exp11_granularity_planner import run_granularity_experiment
from experiments.exp12_frequency_planner import run_frequency_planner_experiment
from experiments.exp13_joint_planner import run_joint_planner_experiment
from experiments.exp14_prior_misspecification import run_prior_misspecification_experiment


def run_all_experiments(
    smoke: bool = False,
    output_dir: Optional[str] = None,
    include_longitudinal: bool = False,
    include_planning: bool = False,
):
    out_root = output_dir or ("results/verification" if smoke else "results")
    t_start = time.time()

    if smoke:
        print("============================================================")
        print("SMOKE MODE")
        print("Official research artifacts will NOT be modified.")
        print(f"Output root: {out_root}")
        print("============================================================")

        # Exp 01 Baseline smoke (n_trials=2)
        run_baseline_experiment(
            n_users=50000,
            n_trials=2,
            data_path="data/interactions_n50000.parquet",
            output_dir=out_root,
        )

        # Exp 02 Epsilon Sweep smoke (canonical N=50,000 public population, n_trials=2)
        run_epsilon_sweep(
            data_path="data/interactions_n50000.parquet",
            n_users=50000,
            epsilons=(0.5, 1.0),
            n_trials=2,
            output_dir=out_root,
        )

        # Exp 03 Population Sweep smoke (small populations, default_trials=2, large_n_trials=2)
        run_population_sweep(
            populations=(500, 1000),
            default_trials=2,
            large_n_trials=2,
            output_dir=out_root,
        )

        # Exp 04 Clipping Sweep smoke (clipping_norms=(10.0, 20.0), n_trials=2)
        run_clipping_sweep(
            n_users=10000,
            clipping_norms=(10.0, 20.0),
            n_trials=2,
            data_path="data/interactions_n10000.parquet",
            output_dir=out_root,
        )

        if include_longitudinal:
            print("\n--- Running Longitudinal Accounting Experiments (Smoke) ---")
            run_composition_experiment(t_values=[1, 2, 5], n_trials=2, output_dir=out_root)
            run_release_frequency_experiment(cadences={"weekly": 6, "monthly": 3}, n_trials=2, output_dir=out_root)
            run_recipient_composition_experiment(n_weeks=4, output_dir=out_root)
            run_longitudinal_budget_experiment(max_steps=6, output_dir=out_root)

        if include_planning:
            print("\n--- Running Small-n Release Planning Experiments (Smoke) ---")
            run_error_decomposition_experiment(
                populations=(500, 1000),
                clipping_norms=(10.0, 20.0, 40.0),
                n_trials=5,
                output_dir=out_root,
            )
            run_clipping_planner_experiment(
                populations=(500, 1000),
                c_candidates=(10.0, 20.0, 40.0),
                n_trials=5,
                output_dir=out_root,
            )
            run_granularity_experiment(
                populations=(500, 1000),
                epsilons=(0.5, 2.0),
                n_trials=5,
                output_dir=out_root,
            )
            run_frequency_planner_experiment(
                spent_ratios=(0.0, 0.50),
                output_dir=out_root,
            )
            run_joint_planner_experiment(
                populations=(500, 1000),
                epsilon_caps=(2.0, 4.0),
                n_trials=5,
                output_dir=out_root,
            )
            run_prior_misspecification_experiment(
                n_users=500,
                n_trials=5,
                output_dir=out_root,
                regimes=["default"],
            )

            from experiments.generate_key_results_summary import generate_key_results_json
            generate_key_results_json(
                tables_dir=f"{out_root}/tables",
                output_file=f"{out_root}/tables/v03_key_results.json",
            )
    else:
        print("============================================================")
        print("FULL EXPERIMENT MODE")
        print(f"Output root: {out_root}")
        print("============================================================")

        # 1. Baseline
        run_baseline_experiment(
            n_users=50000,
            n_trials=500,
            data_path="data/interactions_n50000.parquet",
            output_dir=out_root,
        )

        # 2. Epsilon Sweep (Gaussian + Laplace) - canonical N=50,000 public population
        run_epsilon_sweep(
            data_path="data/interactions_n50000.parquet",
            n_users=50000,
            n_trials=500,
            output_dir=out_root,
        )

        # 3. Population Sweep (500 to 100,000)
        run_population_sweep(
            populations=(500, 1000, 5000, 10000, 50000, 100000),
            output_dir=out_root,
        )

        # 4. Clipping Sweep (C in {5, 10, 20, 40, 80})
        run_clipping_sweep(
            n_users=10000,
            n_trials=500,
            data_path="data/interactions_n10000.parquet",
            output_dir=out_root,
        )

        if include_longitudinal:
            print("\n--- Running Longitudinal Accounting Experiments (Full) ---")
            run_composition_experiment(output_dir=out_root)
            run_release_frequency_experiment(output_dir=out_root)
            run_recipient_composition_experiment(output_dir=out_root)
            run_longitudinal_budget_experiment(output_dir=out_root)

        if include_planning:
            print("\n--- Running Small-n Release Planning Experiments (Full) ---")
            run_error_decomposition_experiment(output_dir=out_root)
            run_clipping_planner_experiment(output_dir=out_root)
            run_granularity_experiment(output_dir=out_root)
            run_frequency_planner_experiment(output_dir=out_root)
            run_joint_planner_experiment(output_dir=out_root)
            run_prior_misspecification_experiment(output_dir=out_root)

            from experiments.generate_key_results_summary import generate_key_results_json
            generate_key_results_json(
                tables_dir=f"{out_root}/tables",
                output_file=f"{out_root}/tables/v03_key_results.json",
            )

    total_time = time.time() - t_start
    print("======================================================================")
    print(f"All experiments successfully completed in {total_time/60:.2f} minutes.")
    print(f"Tables saved to: {out_root}/tables/")
    print(f"Figures saved to: {out_root}/figures/")
    print(f"Raw trials saved to: {out_root}/raw/")
    print("======================================================================")


def main():
    parser = argparse.ArgumentParser(description="Run Private LLM Analytics experiments")
    parser.add_argument("--smoke", action="store_true", help="Run rapid smoke test with minimal trials into results/verification")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory for results (defaults to results/verification for smoke, results for full)")
    parser.add_argument("--include-longitudinal", "--v02", action="store_true", help="Also execute v0.2 longitudinal experiments (Exp 05-08)")
    parser.add_argument("--include-planning", "--v03", action="store_true", help="Also execute v0.3 release planning experiments (Exp 09-14)")
    args = parser.parse_args()
    run_all_experiments(
        smoke=args.smoke,
        output_dir=args.output_dir,
        include_longitudinal=args.include_longitudinal,
        include_planning=args.include_planning,
    )


if __name__ == "__main__":
    main()
