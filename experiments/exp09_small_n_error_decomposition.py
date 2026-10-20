"""Experiment 9: Small-n Error Decomposition (Clipping vs Noise across N and C).

Evaluates empirical clipping bias and analytic DP noise across:
- Sample sizes N in {500, 1000, 2500, 5000, 10000, 50000}
- Clipping bounds C in {5, 10, 20, 40, 80}
- Validates theoretical utility estimator calibration against empirical ground truth under unified normalized total L1 loss.
"""

import argparse
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Sequence
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from private_llm_analytics.aggregation.user_matrix import build_user_matrix
from private_llm_analytics.evaluation.report import plot_single_line_chart, save_summary_table
from private_llm_analytics.mechanisms.gaussian import calibrate_analytic_gaussian
from private_llm_analytics.planning.candidates import ReleasePolicyCandidate
from private_llm_analytics.planning.policy import build_volume_contributions_for_granularity
from private_llm_analytics.planning.utility import (
    PublicContributionPrior,
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
    print("Running Exp 09: Small-n Error Decomposition (Unified Normalized L1 Loss)")
    print(f"Populations: {populations}")
    print(f"Clipping Norms: {clipping_norms}")
    print(f"Trials: {n_trials}, Epsilon: {epsilon}, Delta: {delta}")
    print("=======================================================")

    out_base = Path(output_dir)
    tables_dir = out_base / "tables"
    figures_dir = out_base / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    prior = PublicContributionPrior(
        mean=60.0,
        std=20.0,
        min_val=5.0,
        max_val=150.0,
        reference_days=30,
        public_seed=900001,
        num_reference_users=2000,
    )
    estimator = UtilityEstimator(
        weights=UtilityWeights(clipping=1.0, noise=1.0, aggregation=0.0, staleness=0.0, window=0.0),
        public_prior=prior,
    )

    records: List[Dict[str, Any]] = []
    pred_totals: List[float] = []
    obs_totals: List[float] = []

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
        raw_mat = umatrix.matrix
        v_raw = np.sum(raw_mat, axis=0)
        pop_total_vol = max(1.0, float(np.sum(v_raw)))

        for c in clipping_norms:
            t0 = time.time()
            cand = ReleasePolicyCandidate(
                contribution_bound=c,
                granularity="full_9_cell",
                mechanism="gaussian",
                epsilon_per_release=epsilon,
                delta_per_release=delta,
            )

            # 1. Exact direct target-space clipping
            _, exact_target, _, sens = build_volume_contributions_for_granularity(
                raw_user_matrix=umatrix,
                granularity="full_9_cell",
                clip_norm=c,
            )
            v_clip = exact_target

            # Empirical clipping loss
            emp_norm_clip = float(np.sum(np.abs(v_raw - v_clip))) / pop_total_vol

            # 2. Calibrate noise
            sigma = calibrate_analytic_gaussian(
                epsilon=epsilon,
                delta=delta,
                l2_sensitivity=sens,
            )

            # 3. Empirical noise trials
            noise_l1_trials: List[float] = []
            total_l1_trials: List[float] = []
            for t in range(n_trials):
                rng = np.random.default_rng(seed + int(c * 100) + t)
                noise = rng.normal(loc=0.0, scale=sigma, size=len(v_clip))
                dp_release = v_clip + noise
                noise_l1_trials.append(float(np.sum(np.abs(noise))))
                total_l1_trials.append(float(np.sum(np.abs(dp_release - v_raw))))

            emp_norm_noise = float(np.mean(noise_l1_trials)) / pop_total_vol
            emp_norm_total = float(np.mean(total_l1_trials)) / pop_total_vol
            per_cell_mae = float(np.mean(total_l1_trials)) / float(len(v_clip))

            # Theoretical predictions from utility estimator
            theory_eval = estimator.evaluate_candidate("interaction_volume", cand, population_size=n)
            pred_norm_clip = float(theory_eval.norm_clipping_loss)
            pred_norm_noise = float(theory_eval.norm_noise_loss)
            pred_norm_total = float(pred_norm_clip + pred_norm_noise)

            pred_totals.append(pred_norm_total)
            obs_totals.append(emp_norm_total)

            rec = {
                "population_n": n,
                "clip_norm_c": c,
                "empirical_norm_clip": round(emp_norm_clip, 4),
                "predicted_norm_clip": round(pred_norm_clip, 4),
                "empirical_norm_noise": round(emp_norm_noise, 4),
                "predicted_norm_noise": round(pred_norm_noise, 4),
                "empirical_norm_total": round(emp_norm_total, 4),
                "predicted_norm_total": round(pred_norm_total, 4),
                "per_cell_mae": round(per_cell_mae, 2),
                "calibration_bias": round(pred_norm_total - emp_norm_total, 4),
                "calibration_MAE": round(abs(pred_norm_total - emp_norm_total), 4),
                "sim_time_sec": round(time.time() - t0, 2),
            }
            records.append(rec)
            print(
                f"  N={n:5d}, C={c:4.1f} | Emp Total={emp_norm_total:6.4f} | "
                f"Pred Total={pred_norm_total:6.4f} | Diff={abs(pred_norm_total - emp_norm_total):6.4f} | "
                f"PerCell MAE={per_cell_mae:6.1f}"
            )

    results_df = pd.DataFrame(records)

    # Compute overall calibration error across all conditions
    bias = float(np.mean(np.array(pred_totals) - np.array(obs_totals)))
    mae = float(np.mean(np.abs(np.array(pred_totals) - np.array(obs_totals))))
    rel_calib = float(np.mean(np.abs(np.array(pred_totals) - np.array(obs_totals)) / np.maximum(1e-4, np.array(obs_totals))))
    spearman_corr, _ = spearmanr(pred_totals, obs_totals)

    results_df["Spearman"] = round(float(spearman_corr), 4)
    save_summary_table(results_df, str(tables_dir / "exp09_small_n_error_decomposition.csv"))

    print("\n=======================================================")
    print(f"[CALIBRATION SUMMARY]")
    print(f"  Absolute Calibration MAE:    {mae:.4f}")
    print(f"  Relative Calibration Error:  {rel_calib * 100:.2f}%")
    print(f"  Calibration Bias:            {bias:+.4f}")
    print(f"  Spearman Rank Correlation:   {spearman_corr:.4f}")
    print("=======================================================")

    # Plot tradeoff for N=1000
    sub_1000 = results_df[results_df["population_n"] == (1000 if 1000 in populations else populations[0])]
    plot_single_line_chart(
        x=sub_1000["clip_norm_c"].tolist(),
        y_series={
            "Observed Total Normalized L1": sub_1000["empirical_norm_total"].tolist(),
            "Predicted Total Normalized L1": sub_1000["predicted_norm_total"].tolist(),
            "Empirical Clipping Loss": sub_1000["empirical_norm_clip"].tolist(),
            "Empirical Noise Loss": sub_1000["empirical_norm_noise"].tolist(),
        },
        x_label="User L2 Clipping Bound (C)",
        y_label="Normalized L1 Error",
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
