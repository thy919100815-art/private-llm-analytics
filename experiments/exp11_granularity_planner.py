"""Experiment 11: Granularity Release Planner (Direct Coarse Release vs Post-Processing).

Decomposes and compares:
  - 1. Target clipping loss: ||V_G - bar_V_G||_1 / ||V_raw||_1
  - 2. DP noise loss: E ||Z_G||_1 / ||V_raw||_1
  - 3. Reconstruction / workload loss: ||A_G^dagger bar_V_G - V_raw||_1 / ||V_raw||_1
  - 4. Total normalized utility: E ||A_G^dagger (bar_V_G + Z_G) - V_raw||_1 / ||V_raw||_1
  - 5. Noise-only gain vs End-to-end total gain
Across sample sizes N in {500, 1000, 5000} and privacy budgets epsilon in {0.5, 1.0, 2.0, 4.0}.
"""

import argparse
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Sequence
import numpy as np
import pandas as pd

from private_llm_analytics.aggregation.user_matrix import build_user_matrix
from private_llm_analytics.evaluation.report import plot_single_line_chart, save_summary_table
from private_llm_analytics.mechanisms.gaussian import calibrate_analytic_gaussian
from private_llm_analytics.planning.candidates import GRANULARITY_DIMENSIONS
from private_llm_analytics.planning.policy import (
    aggregate_to_granularity,
    build_volume_contributions_for_granularity,
    reconstruct_to_9_cell,
)
from private_llm_analytics.simulator.generator import SyntheticDataGenerator
from private_llm_analytics.simulator.scenario import SimulationScenario


def run_granularity_experiment(
    populations: Sequence[int] = (500, 1000, 5000),
    epsilons: Sequence[float] = (0.5, 1.0, 2.0, 4.0),
    delta: float = 1e-6,
    clip_norm: float = 20.0,
    n_trials: int = 200,
    seed: int = 42,
    output_dir: str = "results",
) -> pd.DataFrame:
    print("\n=======================================================")
    print("Running Exp 11: Granularity Release Planner (Decomposed Utility)")
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
    records: List[Dict[str, Any]] = []

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
        cell_names = umatrix.cell_names

        # Unclipped 9-cell ground truth
        unclipped_9 = np.sum(umatrix.matrix, axis=0)
        pop_total_vol = max(1.0, float(np.sum(unclipped_9)))

        for eps in epsilons:
            # Calibrate Gaussian noise scale sigma for C=20
            sigma = calibrate_analytic_gaussian(epsilon=eps, delta=delta, l2_sensitivity=clip_norm)

            # Precompute 9-cell baseline for gain normalization
            _, exact_9, names_9, _ = build_volume_contributions_for_granularity(
                raw_user_matrix=umatrix,
                granularity="full_9_cell",
                clip_norm=clip_norm,
            )
            base_noise_9_trials = []
            base_total_9_trials = []
            for t in range(n_trials):
                rng = np.random.default_rng(seed + 9999 + t)
                noise_9 = rng.normal(loc=0.0, scale=sigma, size=9)
                base_noise_9_trials.append(float(np.sum(np.abs(noise_9))))
                base_total_9_trials.append(float(np.sum(np.abs(exact_9 + noise_9 - unclipped_9))))
            base_norm_noise_9 = float(np.mean(base_noise_9_trials)) / pop_total_vol
            base_norm_total_9 = float(np.mean(base_total_9_trials)) / pop_total_vol

            for g in granularities:
                # Direct target space release
                clipped_mat, exact_target, target_names, sens = build_volume_contributions_for_granularity(
                    raw_user_matrix=umatrix,
                    granularity=g,
                    clip_norm=clip_norm,
                )
                assert sens == clip_norm
                dim = len(exact_target)

                # 1. Target clipping loss
                true_target_unclipped, _ = aggregate_to_granularity(unclipped_9, cell_names, g)
                clipping_loss = float(np.sum(np.abs(true_target_unclipped - exact_target))) / pop_total_vol

                # 2. Reconstruction bias on exact clipped
                reconstructed_exact_9 = reconstruct_to_9_cell(exact_target, g, target_names, cell_names)
                reconstruction_loss = float(np.sum(np.abs(reconstructed_exact_9 - unclipped_9))) / pop_total_vol

                # 3. DP Noise trials
                noise_l1_trials: List[float] = []
                end_to_end_trials: List[float] = []
                target_dp_trials: List[float] = []

                for t in range(n_trials):
                    rng = np.random.default_rng(seed + t * 100)
                    noise = rng.normal(loc=0.0, scale=sigma, size=dim)
                    dp_release = exact_target + noise

                    noise_l1_trials.append(float(np.sum(np.abs(noise))))
                    target_dp_trials.append(float(np.mean(np.abs(dp_release - true_target_unclipped))))

                    reconstructed_dp_9 = reconstruct_to_9_cell(dp_release, g, target_names, cell_names)
                    end_to_end_trials.append(float(np.sum(np.abs(reconstructed_dp_9 - unclipped_9))))

                dp_noise_loss = float(np.mean(noise_l1_trials)) / pop_total_vol
                total_normalized_utility = float(np.mean(end_to_end_trials)) / pop_total_vol
                direct_target_mae = float(np.mean(target_dp_trials))

                # Comparative Gains vs 9-cell
                noise_only_gain = float((base_norm_noise_9 - dp_noise_loss) / max(1e-9, base_norm_noise_9))
                end_to_end_total_gain = float((base_norm_total_9 - total_normalized_utility) / max(1e-9, base_norm_total_9))

                rec = {
                    "population_n": n,
                    "epsilon": eps,
                    "granularity": g,
                    "dimension": dim,
                    "clipping_loss": round(clipping_loss, 4),
                    "dp_noise_loss": round(dp_noise_loss, 4),
                    "reconstruction_loss": round(reconstruction_loss, 4),
                    "total_normalized_utility": round(total_normalized_utility, 4),
                    "noise_only_gain": round(noise_only_gain, 4),
                    "end_to_end_total_gain": round(end_to_end_total_gain, 4),
                    "direct_target_mae": round(direct_target_mae, 2),
                }
                records.append(rec)
                print(
                    f"  N={n:4d}, eps={eps:3.1f}, G={g:15s} | Dim={dim} | Clip={clipping_loss:6.4f} | "
                    f"Noise={dp_noise_loss:6.4f} | TotalUtil={total_normalized_utility:6.4f} | "
                    f"NoiseGain={noise_only_gain*100:+5.1f}% | TotalGain={end_to_end_total_gain*100:+5.1f}%"
                )

    results_df = pd.DataFrame(records)
    save_summary_table(results_df, str(tables_dir / "exp11_granularity_planner.csv"))

    # Plot Decomposed Errors for N=500, eps=1.0
    sub_500 = results_df[(results_df["population_n"] == populations[0]) & (results_df["epsilon"] == 1.0)]
    plot_single_line_chart(
        x=list(range(len(sub_500))),
        y_series={
            "DP Noise Loss": sub_500["dp_noise_loss"].tolist(),
            "Clipping Loss": sub_500["clipping_loss"].tolist(),
            "Reconstruction Loss": sub_500["reconstruction_loss"].tolist(),
            "Total Normalized Utility Loss": sub_500["total_normalized_utility"].tolist(),
        },
        x_label="Granularity Index (0:full9, 1:prov3, 2:task3, 3:overall1)",
        y_label="Normalized Loss",
        title=f"Granularity Error Decomposition (N={populations[0]}, eps=1.0)",
        output_path=str(figures_dir / "exp11_granularity_planner"),
        x_log=False,
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
