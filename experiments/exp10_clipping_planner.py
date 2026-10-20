"""Experiment 10: Contribution Bound (Clipping) Release Planner.

Compares:
  - Baseline 1: Fixed C = 20
  - Baseline 2: Fixed C = 40
  - Static Public Planner (v0.3): automatically plans C* using public prior
  - Oracle Planner: selects empirical optimal C on ground truth
Across sample sizes N in {500, 1000, 2500, 5000}.
"""

import argparse
from pathlib import Path
import time
from typing import Dict, List, Optional, Sequence
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
from private_llm_analytics.simulator.generator import SyntheticDataGenerator
from private_llm_analytics.simulator.scenario import SimulationScenario


def run_clipping_planner_experiment(
    populations: Sequence[int] = (500, 1000, 2500, 5000),
    c_candidates: Sequence[float] = (5.0, 10.0, 20.0, 40.0, 80.0),
    epsilon: float = 1.0,
    delta: float = 1e-6,
    n_trials: int = 200,
    seed: int = 42,
    output_dir: str = "results",
) -> pd.DataFrame:
    print("\n=======================================================")
    print("Running Exp 10: Contribution Bound (Clipping) Planner")
    print(f"Populations: {populations}")
    print(f"Candidate C values: {c_candidates}")
    print(f"Trials: {n_trials}, Epsilon: {epsilon}")
    print("=======================================================")

    out_base = Path(output_dir)
    tables_dir = out_base / "tables"
    figures_dir = out_base / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    app_config = load_app_config("configs")
    app_config.privacy.planner.candidate_clipping = list(c_candidates)
    planner = SmallNReleasePlanner(app_config=app_config)

    records: List[Dict] = []

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

        # 1. Evaluate each C empirically to find Oracle ground truth
        empirical_errors: Dict[float, float] = {}
        for c in c_candidates:
            metric = InteractionVolumeMetric(clip_norm=c)
            _, summ = run_repeated_noise_trials(
                user_matrix=umatrix,
                metric=metric,
                mechanism_type="gaussian",
                epsilon=epsilon,
                delta=delta,
                n_trials=n_trials,
                seed=seed + int(c * 10),
            )
            empirical_errors[c] = summ["total_volume_mae_mean"]

        oracle_c = min(empirical_errors, key=empirical_errors.get)
        oracle_err = empirical_errors[oracle_c]

        # 2. Static Public Planner recommendation
        app_config.scenario.population.n_users = n
        plan_res = planner.plan(
            metric="interaction_volume",
            recipient="recipient_internal",
            force_recompute=True,
        )
        selected_c = plan_res.policy.contribution_bound if plan_res.policy else 20.0
        # If selected_c not directly in empirical_errors, run evaluation
        if selected_c in empirical_errors:
            planner_err = empirical_errors[selected_c]
        else:
            metric = InteractionVolumeMetric(clip_norm=selected_c)
            _, s_plan = run_repeated_noise_trials(
                user_matrix=umatrix,
                metric=metric,
                mechanism_type="gaussian",
                epsilon=epsilon,
                delta=delta,
                n_trials=n_trials,
                seed=seed + 999,
            )
            planner_err = s_plan["total_volume_mae_mean"]

        # 3. Baselines: Fixed C=20 and Fixed C=40
        fixed_20_err = empirical_errors.get(20.0, 0.0)
        fixed_40_err = empirical_errors.get(40.0, 0.0)

        # 4. Regrets
        abs_reg_plan, rel_reg_plan = compute_regret(planner_err, oracle_err)
        abs_reg_20, rel_reg_20 = compute_regret(fixed_20_err, oracle_err)
        abs_reg_40, rel_reg_40 = compute_regret(fixed_40_err, oracle_err)

        rec = {
            "population_n": n,
            "oracle_optimal_c": oracle_c,
            "oracle_error": round(oracle_err, 2),
            "planner_selected_c": selected_c,
            "planner_error": round(planner_err, 2),
            "planner_abs_regret": round(abs_reg_plan, 2),
            "planner_rel_regret": round(rel_reg_plan, 4),
            "fixed_20_error": round(fixed_20_err, 2),
            "fixed_20_abs_regret": round(abs_reg_20, 2),
            "fixed_40_error": round(fixed_40_err, 2),
            "fixed_40_abs_regret": round(abs_reg_40, 2),
        }
        records.append(rec)
        print(
            f"  N={n:5d} | Oracle C={oracle_c:2.0f} (Err={oracle_err:6.1f}) | "
            f"Planner C={selected_c:2.0f} (Err={planner_err:6.1f}, Regret={rel_reg_plan*100:4.1f}%) | "
            f"Fixed20 (Regret={rel_reg_20*100:4.1f}%) | Fixed40 (Regret={rel_reg_40*100:4.1f}%)"
        )

    results_df = pd.DataFrame(records)
    save_summary_table(results_df, str(tables_dir / "exp10_clipping_planner.csv"))

    # Plot Relative Regret Comparison
    plot_single_line_chart(
        x=results_df["population_n"].tolist(),
        y_series={
            "Static Public Planner Regret": (results_df["planner_rel_regret"] * 100).tolist(),
            "Fixed C=20 Regret": (results_df["fixed_20_abs_regret"] / results_df["oracle_error"] * 100).tolist(),
            "Fixed C=40 Regret": (results_df["fixed_40_abs_regret"] / results_df["oracle_error"] * 100).tolist(),
        },
        x_label="Enterprise Population (N)",
        y_label="Relative Regret vs Oracle (%)",
        title="Clipping Bound Planner Regret vs Fixed Baselines",
        output_path=str(figures_dir / "exp10_clipping_planner"),
        x_log=True,
    )

    print(f"\n[SUCCESS] Exp 10 completed. Saved to {tables_dir}/ and {figures_dir}/")
    return results_df


def main():
    parser = argparse.ArgumentParser(description="Run Experiment 10: Contribution Bound Planner")
    parser.add_argument("--smoke", action="store_true", help="Run rapid smoke test")
    parser.add_argument("--output-dir", type=str, default="results", help="Output directory")
    args = parser.parse_args()

    out_root = "results/verification" if args.smoke else args.output_dir
    if args.smoke:
        run_clipping_planner_experiment(
            populations=(500, 1000),
            c_candidates=(10.0, 20.0, 40.0),
            n_trials=10,
            output_dir=out_root,
        )
    else:
        run_clipping_planner_experiment(output_dir=out_root)


if __name__ == "__main__":
    main()
