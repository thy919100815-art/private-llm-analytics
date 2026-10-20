"""Experiment 9: Small-n Error Decomposition (Clipping vs Noise across N and C).

Evaluates empirical clipping bias and analytic DP noise across:
- Sample sizes N in {500, 1000, 2500, 5000, 10000, 50000}
- Clipping bounds C in {5, 10, 20, 40, 80}
- Validates theoretical utility estimator calibration against empirical ground truth.
"""

import argparse
from pathlib import Path
import time
from typing import Dict, List, Optional, Sequence
import numpy as np
import pandas as pd

from private_llm_analytics.aggregation.user_matrix import build_user_matrix
from private_llm_analytics.evaluation.privacy_utility import run_repeated_noise_trials
from private_llm_analytics.evaluation.report import plot_single_line_chart, save_summary_table
from private_llm_analytics.metrics.interaction_volume import InteractionVolumeMetric
from private_llm_analytics.planning.candidates import ReleasePolicyCandidate
from private_llm_analytics.planning.diagnostics import compute_calibration_error
from private_llm_analytics.planning.utility import (
    ClippingEstimator,
    NoiseEstimator,
    TruncatedNormalPrior,
    UtilityEstimator,
    UtilityWeights,
)
from private_llm_analytics.simulator.generator import SyntheticDataGenerator
from private_llm_analytics.simulator.scenario import SimulationScenario


def run_error_decomposition_experiment(
    populations: Sequence[int] = (500, 1000, 2500, 5000, 10000, 50000),
    clipping_norms: Sequence[float] = (5.0, 10.0, 20.0, 40.0, 80.0),
    epsilon: float = 1.0,
    delta: float = 1e-6,
    n_trials: int = 500,
    seed: int = 42,
    output_dir: str = "results",
) -> pd.DataFrame:
    print("\n=======================================================")
    print("Running Exp 09: Small-n Error Decomposition")
    print(f"Populations: {populations}")
    print(f"Clipping Norms: {clipping_norms}")
    print(f"Trials: {n_trials}, Epsilon: {epsilon}, Delta: {delta}")
    print("=======================================================")

    out_base = Path(output_dir)
    tables_dir = out_base / "tables"
    figures_dir = out_base / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    prior = TruncatedNormalPrior(mean=60.0, std=20.0, min_val=5.0, max_val=150.0)
    estimator = UtilityEstimator(
        weights=UtilityWeights(clipping=1.0, noise=1.0, aggregation=0.0, staleness=0.0),
        public_prior=prior,
    )

    records: List[Dict] = []
    pred_errors: List[float] = []
    obs_errors: List[float] = []

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

        for c in clipping_norms:
            metric = InteractionVolumeMetric(clip_norm=c)
            cand = ReleasePolicyCandidate(
                contribution_bound=c,
                granularity="full_9_cell",
                mechanism="gaussian",
                epsilon_per_release=epsilon,
                delta_per_release=delta,
            )

            # Empirical simulation
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
            sim_time = time.time() - t0

            # Theoretical prediction from utility estimator
            theory_eval = estimator.evaluate_candidate("interaction_volume", cand, population_size=n)
            pred_total = theory_eval.clipping_error + theory_eval.noise_error
            obs_total = summary["total_volume_mae_mean"]

            pred_errors.append(pred_total)
            obs_errors.append(obs_total)

            rec = {
                "population_n": n,
                "clip_norm_c": c,
                "empirical_clipping_bias": summary["clipping_bias_mae"],
                "empirical_noise_mae": summary["dp_noise_mae_mean"],
                "empirical_total_mae": obs_total,
                "predicted_clipping_bias": theory_eval.clipping_error,
                "predicted_noise_mae": theory_eval.noise_error,
                "predicted_total_mae": pred_total,
                "prediction_error_mae": abs(pred_total - obs_total),
                "sim_time_sec": round(sim_time, 2),
            }
            records.append(rec)
            print(
                f"  N={n:5d}, C={c:4.1f} | Obs Total={obs_total:7.2f} | "
                f"Pred Total={pred_total:7.2f} | Diff={abs(pred_total - obs_total):6.2f}"
            )

    results_df = pd.DataFrame(records)
    save_summary_table(results_df, str(tables_dir / "exp09_small_n_error_decomposition.csv"))

    # Compute overall calibration error
    calib = compute_calibration_error(predicted_errors=pred_errors, observed_errors=obs_errors)
    print(f"\n[CALIBRATION] Bias: {calib['bias']:.2f}, MAE: {calib['mae']:.2f}, Spearman: {calib['spearman_corr']:.4f}")

    # Plot tradeoff for N=1000
    sub_1000 = results_df[results_df["population_n"] == (1000 if 1000 in populations else populations[0])]
    plot_single_line_chart(
        x=sub_1000["clip_norm_c"].tolist(),
        y_series={
            "Observed Total": sub_1000["empirical_total_mae"].tolist(),
            "Predicted Total": sub_1000["predicted_total_mae"].tolist(),
            "Clipping Bias": sub_1000["empirical_clipping_bias"].tolist(),
            "Noise Error": sub_1000["empirical_noise_mae"].tolist(),
        },
        x_label="User L2 Clipping Bound (C)",
        y_label="Error (MAE on cell volume)",
        title=f"Small-n Error Decomposition vs C (N={sub_1000['population_n'].iloc[0]})",
        output_path=str(figures_dir / "exp09_small_n_error_decomposition"),
        x_log=True,
    )

    print(f"\n[SUCCESS] Exp 09 completed. Saved to {tables_dir}/ and {figures_dir}/")
    return results_df


def main():
    parser = argparse.ArgumentParser(description="Run Experiment 9: Small-n Error Decomposition")
    parser.add_argument("--smoke", action="store_true", help="Run rapid smoke test")
    parser.add_argument("--output-dir", type=str, default="results", help="Output directory")
    args = parser.parse_args()

    out_root = "results/verification" if args.smoke else args.output_dir
    if args.smoke:
        run_error_decomposition_experiment(
            populations=(500, 1000),
            clipping_norms=(10.0, 20.0, 40.0),
            n_trials=10,
            output_dir=out_root,
        )
    else:
        run_error_decomposition_experiment(output_dir=out_root)


if __name__ == "__main__":
    main()
