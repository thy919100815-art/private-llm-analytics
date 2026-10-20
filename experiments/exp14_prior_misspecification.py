"""Experiment 14: Prior Misspecification Robustness Evaluation Across Ground-Truth Regimes.

Evaluates planner performance when public prior diverges from reality across 3 distinct ground-truth regimes:
  - 'light': True user interaction ~ (mu=25, sigma=10)
  - 'default': True user interaction ~ (mu=60, sigma=20)
  - 'heavy_tail': True user interaction ~ (mu=100, sigma=40)

For each truth regime, tests 6 prior conditions:
  1. Matched Prior (exact truth parameters)
  2. Mean +25% Shift
  3. Mean -25% Shift
  4. Std +50% Shift
  5. Std -50% Shift
  6. Distribution Mismatch (LogNormal prior against TruncatedNormal truth)

Compares empirical regret of SmallNReleasePlanner against empirical Oracle and fixed baselines.
"""

import argparse
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Sequence
import numpy as np
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


TRUTH_REGIMES = {
    "light": {
        "freq_mean": 25.0,
        "freq_std": 10.0,
        "freq_min": 2,
        "freq_max": 80,
        "candidate_c": [5.0, 10.0, 20.0, 30.0, 50.0],
    },
    "default": {
        "freq_mean": 60.0,
        "freq_std": 20.0,
        "freq_min": 5,
        "freq_max": 150,
        "candidate_c": [10.0, 20.0, 40.0, 60.0, 80.0],
    },
    "heavy_tail": {
        "freq_mean": 100.0,
        "freq_std": 40.0,
        "freq_min": 10,
        "freq_max": 300,
        "candidate_c": [20.0, 40.0, 70.0, 100.0, 150.0],
    },
}


def build_test_cases_for_regime(regime_name: str, true_mean: float, true_std: float) -> List[Dict[str, Any]]:
    return [
        {
            "name": f"Matched ({true_mean:.0f}, {true_std:.0f})",
            "mean": true_mean,
            "std": true_std,
            "type": "truncated_normal",
        },
        {
            "name": f"Mean +25% ({true_mean * 1.25:.1f}, {true_std:.0f})",
            "mean": true_mean * 1.25,
            "std": true_std,
            "type": "truncated_normal",
        },
        {
            "name": f"Mean -25% ({true_mean * 0.75:.1f}, {true_std:.0f})",
            "mean": true_mean * 0.75,
            "std": true_std,
            "type": "truncated_normal",
        },
        {
            "name": f"Std +50% ({true_mean:.0f}, {true_std * 1.5:.0f})",
            "mean": true_mean,
            "std": true_std * 1.5,
            "type": "truncated_normal",
        },
        {
            "name": f"Std -50% ({true_mean:.0f}, {true_std * 0.5:.0f})",
            "mean": true_mean,
            "std": true_std * 0.5,
            "type": "truncated_normal",
        },
        {
            "name": f"LogNormal Mismatch ({true_mean:.0f}, {true_std * 1.2:.0f})",
            "mean": true_mean,
            "std": true_std * 1.2,
            "type": "lognormal",
        },
    ]


def run_prior_misspecification_experiment(
    n_users: int = 1000,
    n_trials: int = 100,
    seed: int = 42,
    output_dir: str = "results",
    regimes: Optional[Sequence[str]] = None,
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

    target_regimes = regimes or list(TRUTH_REGIMES.keys())
    all_records: List[Dict[str, Any]] = []

    app_config = load_app_config("configs")
    app_config.scenario.population.n_users = n_users

    for regime_idx, regime_name in enumerate(target_regimes):
        regime_spec = TRUTH_REGIMES[regime_name]
        t_mean = regime_spec["freq_mean"]
        t_std = regime_spec["freq_std"]
        candidate_c_values = regime_spec["candidate_c"]
        app_config.privacy.planner.candidate_clipping = candidate_c_values

        print(f"\n--- Ground Truth Regime: {regime_name.upper()} (mu={t_mean}, sigma={t_std}) ---")

        # Generate ground truth dataset for this regime
        scenario = SimulationScenario(
            n_users=n_users,
            freq_mean=t_mean,
            freq_std=t_std,
            freq_min=regime_spec["freq_min"],
            freq_max=regime_spec["freq_max"],
            random_seed=seed + regime_idx * 100,
        )
        gen = SyntheticDataGenerator(scenario)
        df_truth = gen.generate(seed=seed + regime_idx * 100)
        umatrix = build_user_matrix(df_truth, public_population_size=n_users)

        # 1. Compute empirical Oracle MAE for each candidate C on this ground truth
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
                seed=seed + int(c * 10) + regime_idx * 1000,
            )
            c_errors[c] = float(s["total_volume_mae_mean"])

        oracle_c = min(c_errors, key=c_errors.get)
        oracle_err = c_errors[oracle_c]
        mid_c = candidate_c_values[len(candidate_c_values) // 2]
        fixed_mid_err = c_errors[mid_c]

        print(f"  Oracle Optimal C={oracle_c:.0f} (MAE={oracle_err:.1f}), Fixed Mid C={mid_c:.0f} (MAE={fixed_mid_err:.1f})")

        # 2. Evaluate planner across misspecified prior conditions
        test_cases = build_test_cases_for_regime(regime_name, t_mean, t_std)
        for case in test_cases:
            if case["type"] == "truncated_normal":
                prior = TruncatedNormalPrior(
                    mean=case["mean"],
                    std=case["std"],
                    min_val=max(1.0, case["mean"] - 3.0 * case["std"]),
                    max_val=case["mean"] + 4.0 * case["std"],
                )
            else:
                prior = LogNormalPrior(
                    mean=case["mean"],
                    std=case["std"],
                    max_cutoff=case["mean"] + 4.0 * case["std"],
                )

            weights = UtilityWeights(clipping=1.0, noise=1.0, aggregation=0.0, staleness=0.0)
            estimator = UtilityEstimator(weights=weights, public_prior=prior)
            planner = SmallNReleasePlanner(app_config=app_config, utility_estimator=estimator)

            plan_res = planner.plan(
                metric="interaction_volume",
                recipient="recipient_internal",
                force_recompute=True,
            )

            selected_c = plan_res.policy.contribution_bound if plan_res.policy else mid_c
            actual_err = c_errors.get(selected_c, fixed_mid_err)
            abs_reg, rel_reg = compute_regret(actual_err, oracle_err)

            rec = {
                "truth_regime": regime_name,
                "truth_mean": t_mean,
                "truth_std": t_std,
                "prior_case": case["name"],
                "prior_mean": case["mean"],
                "prior_std": case["std"],
                "prior_type": case["type"],
                "oracle_optimal_c": oracle_c,
                "oracle_error": round(oracle_err, 2),
                "selected_c": selected_c,
                "actual_empirical_error": round(actual_err, 2),
                "relative_regret": round(rel_reg, 4),
                "fixed_mid_c": mid_c,
                "fixed_mid_rel_regret": round((fixed_mid_err - oracle_err) / max(1e-9, oracle_err), 4),
            }
            all_records.append(rec)
            print(
                f"    {case['name']:40s} | Selected C={selected_c:3.0f} | "
                f"Error={actual_err:6.1f} | Regret={rel_reg*100:4.1f}%"
            )

    results_df = pd.DataFrame(all_records)
    save_summary_table(results_df, str(tables_dir / "exp14_prior_misspecification.csv"))

    # Plot Robustness across regimes
    for regime_name in target_regimes:
        sub_df = results_df[results_df["truth_regime"] == regime_name].reset_index(drop=True)
        if not sub_df.empty:
            plot_single_line_chart(
                x=list(range(len(sub_df))),
                y_series={
                    "Planner Misspecification": (sub_df["relative_regret"] * 100).tolist(),
                    f"Fixed C={sub_df['fixed_mid_c'].iloc[0]} Baseline": (sub_df["fixed_mid_rel_regret"] * 100).tolist(),
                },
                x_label="Prior Condition Index",
                y_label="Relative Regret vs Oracle (%)",
                title=f"Prior Misspecification Robustness - Regime {regime_name.upper()} (N={n_users})",
                output_path=str(figures_dir / f"exp14_prior_misspecification_{regime_name}"),
                x_log=False,
            )

    print(f"\n[SUCCESS] Exp 14 completed for all {len(target_regimes)} regimes. Saved to {tables_dir}/ and {figures_dir}/")
    return results_df


def main():
    parser = argparse.ArgumentParser(description="Run Experiment 14: Prior Misspecification Robustness")
    parser.add_argument("--smoke", action="store_true", help="Run rapid smoke test")
    parser.add_argument("--output-dir", type=str, default="results", help="Output directory")
    args = parser.parse_args()

    out_root = "results/verification" if args.smoke else args.output_dir
    if args.smoke:
        run_prior_misspecification_experiment(
            n_users=500,
            n_trials=10,
            output_dir=out_root,
            regimes=["default"],
        )
    else:
        run_prior_misspecification_experiment(
            n_users=1000,
            n_trials=100,
            output_dir=out_root,
        )


if __name__ == "__main__":
    main()
