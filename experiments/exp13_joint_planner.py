"""Experiment 13: Joint Policy Release Planner (C, G, W, F) with True Empirical Oracle.

Evaluates Joint Policy Optimization:
  - Baseline A: Fixed (C=20, G=full_9_cell, W=30, F=7)
  - Baseline B: Fixed (C=40, G=provider_3_cell, W=30, F=14)
  - Static Public Joint Planner (v0.3): automatic joint selection under public prior
  - True Empirical Oracle: argmin true_utility evaluated directly on ground truth data
Across populations N in {500, 1000, 5000} and budget caps epsilon_cap in {2.0, 4.0}.
"""

import argparse
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Sequence
import numpy as np
import pandas as pd

from private_llm_analytics.aggregation.user_matrix import build_user_matrix
from private_llm_analytics.config.loader import load_app_config
from private_llm_analytics.evaluation.report import plot_single_line_chart, save_summary_table
from private_llm_analytics.mechanisms.gaussian import calibrate_analytic_gaussian
from private_llm_analytics.planning.candidates import GRANULARITY_DIMENSIONS, ReleasePolicyCandidate
from private_llm_analytics.planning.diagnostics import compute_regret
from private_llm_analytics.planning.planner import SmallNReleasePlanner
from private_llm_analytics.planning.policy import (
    aggregate_to_granularity,
    build_volume_contributions_for_granularity,
    compute_workload_reconstruction_error,
    reconstruct_to_9_cell,
)
from private_llm_analytics.simulator.generator import SyntheticDataGenerator
from private_llm_analytics.simulator.scenario import SimulationScenario


def compute_true_candidate_utility(
    candidate: ReleasePolicyCandidate,
    umatrix: Any,
    cell_names: List[str],
    reference_days: int = 30,
    n_trials: int = 10,
    seed: int = 42,
) -> float:
    """Compute true empirical utility on ground truth user data without planner predictions."""
    raw_mat = umatrix.matrix
    n_users = raw_mat.shape[0]
    w_days = candidate.window_days

    # 1. True empirical clipping loss
    clipped_users, exact_agg, target_names, _ = build_volume_contributions_for_granularity(
        raw_user_matrix=umatrix,
        granularity=candidate.granularity,
        clip_norm=candidate.contribution_bound or 20.0,
    )
    unclipped_9 = np.sum(raw_mat, axis=0)
    true_total_vol = max(1.0, float(np.sum(unclipped_9)))

    # Compute empirical clipping loss against true target unclipped
    true_target_unclipped, _ = aggregate_to_granularity(unclipped_9, cell_names, candidate.granularity)
    emp_clip_loss = float(np.sum(np.abs(true_target_unclipped - exact_agg))) / true_total_vol

    # 2. True empirical DP noise error
    sigma = calibrate_analytic_gaussian(
        epsilon=candidate.epsilon_per_release,
        delta=candidate.delta_per_release,
        l2_sensitivity=candidate.contribution_bound or 20.0,
    )
    noise_errors = []
    for t in range(n_trials):
        rng = np.random.default_rng(seed + t * 100)
        noise = rng.normal(loc=0.0, scale=sigma, size=len(exact_agg))
        noise_errors.append(float(np.sum(np.abs(noise))))
    emp_noise_loss = float(np.mean(noise_errors)) / true_total_vol

    # 3. True empirical workload reconstruction loss
    reconstructed_9 = reconstruct_to_9_cell(exact_agg, candidate.granularity, target_names, cell_names)
    emp_agg_loss = compute_workload_reconstruction_error(
        true_9_cell=unclipped_9,
        cell_names=cell_names,
        granularity=candidate.granularity,
    )

    # 4. Freshness and window loss
    emp_stale_loss = float(candidate.frequency_days / (2.0 * w_days))
    emp_win_loss = float(abs(w_days - reference_days) / float(reference_days))

    # Total True Utility (dimensionless weights: 1.0, 1.0, 1.0, 0.5, 0.2)
    total_true = (
        1.0 * emp_clip_loss
        + 1.0 * emp_noise_loss
        + 1.0 * emp_agg_loss
        + 0.5 * emp_stale_loss
        + 0.2 * emp_win_loss
    )
    return float(total_true)


def run_joint_planner_experiment(
    populations: Sequence[int] = (500, 1000, 5000),
    epsilon_caps: Sequence[float] = (2.0, 4.0),
    horizon_days: int = 90,
    seed: int = 42,
    output_dir: str = "results",
) -> pd.DataFrame:
    print("\n=======================================================")
    print("Running Exp 13: Joint Policy Release Planner (C, G, W, F)")
    print(f"Populations: {populations}")
    print(f"Budget Caps: {epsilon_caps}")
    print(f"Planning Horizon: {horizon_days} days")
    print("=======================================================")

    out_base = Path(output_dir)
    tables_dir = out_base / "tables"
    figures_dir = out_base / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    app_config = load_app_config("configs")
    app_config.privacy.planner.planning_horizon_days = horizon_days

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
        cell_names = umatrix.cell_names
        app_config.scenario.population.n_users = n

        for cap in epsilon_caps:
            app_config.privacy.accountant.epsilon_cap = cap
            planner = SmallNReleasePlanner(app_config=app_config)

            # 1. Run Static Public Planner
            plan_res = planner.plan(
                metric="interaction_volume",
                recipient="recipient_internal",
                planning_horizon_days=horizon_days,
                force_recompute=True,
            )

            # 2. Evaluate all feasible candidates under True Empirical Oracle
            feasible_candidates = [
                rec["candidate"] for rec in plan_res.candidate_evaluations if rec["feasible"]
            ]
            if not feasible_candidates:
                continue

            # Compute true empirical utility for all feasible policies
            oracle_scores = {}
            for cand in feasible_candidates:
                score = compute_true_candidate_utility(
                    candidate=cand,
                    umatrix=umatrix,
                    cell_names=cell_names,
                    reference_days=app_config.privacy.planner.window_model.reference_days,
                    n_trials=10,
                    seed=seed,
                )
                oracle_scores[cand] = score

            # True Oracle is argmin of true empirical utility
            oracle_cand = min(oracle_scores, key=oracle_scores.get)
            oracle_true_util = oracle_scores[oracle_cand]

            planner_cand = plan_res.policy
            planner_true_util = (
                compute_true_candidate_utility(
                    candidate=planner_cand,
                    umatrix=umatrix,
                    cell_names=cell_names,
                    reference_days=app_config.privacy.planner.window_model.reference_days,
                    n_trials=10,
                    seed=seed,
                )
                if planner_cand
                else oracle_true_util
            )

            # 3. Baseline A: C=20, G=full_9_cell, W=30, F=7
            base_a_rec = next(
                (c for c in plan_res.candidate_evaluations
                 if c["candidate"].contribution_bound == 20.0
                 and c["candidate"].granularity == "full_9_cell"
                 and c["candidate"].frequency_days == 7),
                None
            )
            base_a_true = (
                compute_true_candidate_utility(base_a_rec["candidate"], umatrix, cell_names, seed=seed)
                if base_a_rec and base_a_rec["feasible"]
                else None
            )

            # 4. Baseline B: C=40, G=provider_3_cell, W=30, F=14
            base_b_rec = next(
                (c for c in plan_res.candidate_evaluations
                 if c["candidate"].contribution_bound == 40.0
                 and c["candidate"].granularity == "provider_3_cell"
                 and c["candidate"].frequency_days == 14),
                None
            )
            base_b_true = (
                compute_true_candidate_utility(base_b_rec["candidate"], umatrix, cell_names, seed=seed)
                if base_b_rec and base_b_rec["feasible"]
                else None
            )

            abs_reg_plan, rel_reg_plan = compute_regret(planner_true_util, oracle_true_util)

            rec = {
                "population_n": n,
                "epsilon_cap": cap,
                "oracle_candidate": oracle_cand.candidate_id,
                "oracle_true_utility": round(oracle_true_util, 4),
                "planner_candidate": planner_cand.candidate_id if planner_cand else None,
                "planner_true_utility": round(planner_true_util, 4),
                "planner_predicted_loss": round(plan_res.estimated_utility.total_error, 4) if plan_res.estimated_utility else None,
                "planner_abs_regret": round(abs_reg_plan, 4),
                "planner_rel_regret": round(rel_reg_plan, 4),
                "baseline_a_feasible": base_a_rec["feasible"] if base_a_rec else False,
                "baseline_a_true_utility": round(base_a_true, 4) if base_a_true is not None else None,
                "baseline_b_feasible": base_b_rec["feasible"] if base_b_rec else False,
                "baseline_b_true_utility": round(base_b_true, 4) if base_b_true is not None else None,
            }
            records.append(rec)
            print(
                f"  N={n:4d}, Cap={cap:3.1f} | Oracle: {oracle_cand.candidate_id} (TrueUtil={oracle_true_util:.4f}) | "
                f"Planner: {planner_cand.candidate_id if planner_cand else 'NONE'} "
                f"(TrueUtil={planner_true_util:.4f}, Regret={rel_reg_plan*100:4.1f}%) | "
                f"BaseA Feas={rec['baseline_a_feasible']} | BaseB Feas={rec['baseline_b_feasible']}"
            )

    results_df = pd.DataFrame(records)
    save_summary_table(results_df, str(tables_dir / "exp13_joint_planner.csv"))

    # Plot Relative Regret vs N
    sub_cap4 = results_df[results_df["epsilon_cap"] == 4.0]
    plot_single_line_chart(
        x=sub_cap4["population_n"].tolist(),
        y_series={
            "Static Public Joint Planner": (sub_cap4["planner_rel_regret"] * 100).tolist(),
        },
        x_label="Enterprise Population (N)",
        y_label="Relative Regret vs True Oracle (%)",
        title="Joint Release Planner Regret vs True Empirical Oracle (Cap=4.0)",
        output_path=str(figures_dir / "exp13_joint_planner"),
        x_log=True,
    )

    print(f"\n[SUCCESS] Exp 13 completed. Saved to {tables_dir}/ and {figures_dir}/")
    return results_df


def main():
    parser = argparse.ArgumentParser(description="Run Experiment 13: Joint Policy Release Planner")
    parser.add_argument("--smoke", action="store_true", help="Run rapid smoke test")
    parser.add_argument("--output-dir", type=str, default="results", help="Output directory")
    args = parser.parse_args()

    out_root = "results/verification" if args.smoke else args.output_dir
    if args.smoke:
        run_joint_planner_experiment(
            populations=(500, 1000),
            epsilon_caps=(2.0, 4.0),
            output_dir=out_root,
        )
    else:
        run_joint_planner_experiment(output_dir=out_root)


if __name__ == "__main__":
    main()
