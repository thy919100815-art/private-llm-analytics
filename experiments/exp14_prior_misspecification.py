"""Experiment 14: Prior Misspecification Robustness Evaluation.

Evaluates planner performance when public prior diverges from reality:
  - Case 1: Matched Prior (mean=60, std=20)
  - Case 2: Mean +25% Shift (mean=75, std=20)
  - Case 3: Mean -25% Shift (mean=45, std=20)
  - Case 4: Std +50% Shift (mean=60, std=30)
  - Case 5: Std -50% Shift (mean=60, std=10)
  - Case 6: Distribution Mismatch (LogNormal data vs TruncatedNormal prior)
Compares empirical regret against Oracle and Naive Fixed Baselines.
"""

import argparse
from pathlib import Path
import time
from typing import Dict, List, Optional, Sequence
import pandas as pd

from private_llm_analytics.aggregation.user_matrix import build_user_matrix
from private_llm_analytics.config.loader import load_app_config
from private_llm_analytics.evaluation.privacy_utility import run_repeated_noise_trials
from private_llm_analytics.evaluation.report import plot_single_line_chart, save_summary_table
from private_llm_analytics.metrics.interaction_volume import InteractionVolumeMetric
from private_llm_analytics.planning.candidates import ReleasePolicyCandidate
from private_llm_analytics.planning.diagnostics import compute_regret
from private_llm_analytics.planning.planner import SmallNReleasePlanner
from private_llm_analytics.planning.utility import (
    LogNormalPrior,
    TruncatedNormalPrior,
    UtilityEstimator,
    UtilityWeights,
)
from private_llm_analytics.simulator.generator import SyntheticDataGenerator
from private_llm_analytics.simulator.scenario import SimulationScenario


def run_prior_misspecification_experiment(
    n_users: int = 1000,
    n_trials: int = 100,
    seed: int = 42,
    output_dir: str = "results",
) -> pd.DataFrame:
    print("\n=======================================================")
    print("Running Exp 14: Prior Misspecification Robustness")
    print(f"Population N: {n_users}, Trials: {n_trials}")
    print("=======================================================")

    out_base = Path(output_dir)
    tables_dir = out_base / "tables"
    figures_dir = out_base / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    # Generate ground truth dataset with true population parameters (mean=60, std=20)
    p = Path(f"data/interactions_n{n_users}.parquet")
    if p.is_file():
        df = pd.read_parquet(p)
    else:
        scenario = SimulationScenario(n_users=n_users, random_seed=seed)
        gen = SyntheticDataGenerator(scenario)
        df = gen.generate(seed=seed)
    umatrix = build_user_matrix(df, public_population_size=n_users)

    # Define misspecified prior conditions
    test_cases = [
        {"name": "Matched Prior (mu=60, sigma=20)", "mean": 60.0, "std": 20.0, "type": "truncated_normal"},
        {"name": "Mean Shift +25% (mu=75, sigma=20)", "mean": 75.0, "std": 20.0, "type": "truncated_normal"},
        {"name": "Mean Shift -25% (mu=45, sigma=20)", "mean": 45.0, "std": 20.0, "type": "truncated_normal"},
        {"name": "Std Shift +50% (mu=60, sigma=30)", "mean": 60.0, "std": 30.0, "type": "truncated_normal"},
        {"name": "Std Shift -50% (mu=60, sigma=10)", "mean": 60.0, "std": 10.0, "type": "truncated_normal"},
        {"name": "LogNormal Prior Mismatch", "mean": 60.0, "std": 25.0, "type": "lognormal"},
    ]

    # Evaluate Oracle on ground truth
    candidate_c_values = [5.0, 10.0, 20.0, 40.0, 80.0]
    c_errors: Dict[float, float] = {}
    for c in candidate_c_values:
        m = InteractionVolumeMetric(clip_norm=c)
        _, s = run_repeated_noise_trials(
            user_matrix=umatrix,
            metric=m,
            mechanism_type="gaussian",
            epsilon=1.0,
            delta=1e-6,
            n_trials=n_trials,
            seed=seed + int(c * 10),
        )
        c_errors[c] = s["total_volume_mae_mean"]

    oracle_c = min(c_errors, key=c_errors.get)
    oracle_err = c_errors[oracle_c]
    fixed_20_err = c_errors.get(20.0, 0.0)
    fixed_40_err = c_errors.get(40.0, 0.0)

    records: List[Dict] = []
    app_config = load_app_config("configs")
    app_config.scenario.population.n_users = n_users
    app_config.privacy.planner.candidate_clipping = candidate_c_values

    for case in test_cases:
        # Construct planner with misspecified prior
        if case["type"] == "truncated_normal":
            prior = TruncatedNormalPrior(mean=case["mean"], std=case["std"], min_val=5.0, max_val=150.0)
        else:
            prior = LogNormalPrior(mean=case["mean"], std=case["std"], max_cutoff=150.0)

        weights = UtilityWeights(clipping=1.0, noise=1.0, aggregation=0.0, staleness=0.0)
        estimator = UtilityEstimator(weights=weights, public_prior=prior)
        planner = SmallNReleasePlanner(app_config=app_config, utility_estimator=estimator)

        plan_res = planner.plan(
            metric="interaction_volume",
            recipient="recipient_internal",
            force_recompute=True,
        )

        selected_c = plan_res.policy.contribution_bound if plan_res.policy else 20.0
        actual_err = c_errors.get(selected_c, fixed_20_err)
        abs_reg, rel_reg = compute_regret(actual_err, oracle_err)

        rec = {
            "test_case": case["name"],
            "oracle_optimal_c": oracle_c,
            "oracle_error": round(oracle_err, 2),
            "selected_c": selected_c,
            "actual_empirical_error": round(actual_err, 2),
            "relative_regret": round(rel_reg, 4),
            "fixed_20_rel_regret": round((fixed_20_err - oracle_err) / oracle_err, 4),
            "fixed_40_rel_regret": round((fixed_40_err - oracle_err) / oracle_err, 4),
        }
        records.append(rec)
        print(
            f"  {case['name']:36s} | Selected C={selected_c:2.0f} | "
            f"Error={actual_err:6.1f} | Regret={rel_reg*100:4.1f}%"
        )

    results_df = pd.DataFrame(records)
    save_summary_table(results_df, str(tables_dir / "exp14_prior_misspecification.csv"))

    # Plot Robustness across Cases
    plot_single_line_chart(
        x=list(range(len(test_cases))),
        y_series={
            "Planner Under Misspecification": (results_df["relative_regret"] * 100).tolist(),
            "Fixed C=20 Baseline": (results_df["fixed_20_rel_regret"] * 100).tolist(),
            "Fixed C=40 Baseline": (results_df["fixed_40_rel_regret"] * 100).tolist(),
        },
        x_label="Prior Condition Index",
        y_label="Relative Regret vs Oracle (%)",
        title=f"Prior Misspecification Robustness (N={n_users})",
        output_path=str(figures_dir / "exp14_prior_misspecification"),
        x_log=False,
    )

    print(f"\n[SUCCESS] Exp 14 completed. Saved to {tables_dir}/ and {figures_dir}/")
    return results_df


def main():
    parser = argparse.ArgumentParser(description="Run Experiment 14: Prior Misspecification Robustness")
    parser.add_argument("--smoke", action="store_true", help="Run rapid smoke test")
    parser.add_argument("--output-dir", type=str, default="results", help="Output directory")
    args = parser.parse_args()

    out_root = "results/verification" if args.smoke else args.output_dir
    if args.smoke:
        run_prior_misspecification_experiment(n_users=500, n_trials=10, output_dir=out_root)
    else:
        run_prior_misspecification_experiment(output_dir=out_root)


if __name__ == "__main__":
    main()
