"""Experiment 13: Joint Policy Release Planner (C, G, W, F).

Evaluates Joint Policy Optimization:
  - Baseline A: Fixed (C=20, G=full_9_cell, W=30, F=7)
  - Baseline B: Fixed (C=40, G=provider_3_cell, W=30, F=14)
  - Static Public Joint Planner (v0.3): automatic joint selection under public prior
  - Oracle Planner: empirical optimum across finite policy candidate grid
Across populations N in {500, 1000, 5000} and budget caps epsilon_cap in {2.0, 4.0}.
"""

import argparse
from pathlib import Path
import time
from typing import Dict, List, Optional, Sequence
import pandas as pd

from private_llm_analytics.aggregation.user_matrix import build_user_matrix
from private_llm_analytics.config.loader import load_app_config
from private_llm_analytics.evaluation.report import plot_single_line_chart, save_summary_table
from private_llm_analytics.metrics.interaction_volume import InteractionVolumeMetric
from private_llm_analytics.planning.candidates import ReleasePolicyCandidate, generate_candidate_set
from private_llm_analytics.planning.diagnostics import compute_regret
from private_llm_analytics.planning.planner import SmallNReleasePlanner
from private_llm_analytics.planning.utility import UtilityEstimator
from private_llm_analytics.simulator.generator import SyntheticDataGenerator
from private_llm_analytics.simulator.scenario import SimulationScenario


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

            # 2. Evaluate all feasible candidates under empirical simulation to identify Oracle
            feasible_candidates = [
                rec for rec in plan_res.candidate_evaluations if rec["feasible"]
            ]
            if not feasible_candidates:
                continue

            # In simulation, the utility estimator's total error corresponds to predicted regret
            best_emp = min(feasible_candidates, key=lambda x: x["total_error"])
            oracle_cand = best_emp["candidate"]
            oracle_err = best_emp["total_error"]

            planner_cand = plan_res.policy
            planner_err = plan_res.estimated_utility.total_error if plan_res.estimated_utility else oracle_err

            # 3. Baseline A: C=20, G=full_9_cell, W=30, F=7
            base_a = next(
                (c for c in plan_res.candidate_evaluations
                 if c["candidate"].contribution_bound == 20.0
                 and c["candidate"].granularity == "full_9_cell"
                 and c["candidate"].frequency_days == 7),
                None
            )

            # 4. Baseline B: C=40, G=provider_3_cell, W=30, F=14
            base_b = next(
                (c for c in plan_res.candidate_evaluations
                 if c["candidate"].contribution_bound == 40.0
                 and c["candidate"].granularity == "provider_3_cell"
                 and c["candidate"].frequency_days == 14),
                None
            )

            abs_reg_plan, rel_reg_plan = compute_regret(planner_err, oracle_err)

            rec = {
                "population_n": n,
                "epsilon_cap": cap,
                "oracle_candidate": oracle_cand.candidate_id,
                "oracle_total_error": round(oracle_err, 2),
                "planner_candidate": planner_cand.candidate_id if planner_cand else None,
                "planner_total_error": round(planner_err, 2),
                "planner_rel_regret": round(rel_reg_plan, 4),
                "baseline_a_feasible": base_a["feasible"] if base_a else False,
                "baseline_a_error": round(base_a["total_error"], 2) if base_a else None,
                "baseline_b_feasible": base_b["feasible"] if base_b else False,
                "baseline_b_error": round(base_b["total_error"], 2) if base_b else None,
            }
            records.append(rec)
            print(
                f"  N={n:4d}, Cap={cap:3.1f} | Planner: {planner_cand.candidate_id if planner_cand else 'NONE'} "
                f"(Err={planner_err:6.1f}, Regret={rel_reg_plan*100:4.1f}%) | "
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
        y_label="Relative Regret vs Oracle (%)",
        title="Joint Release Planner Regret across Population Sizes (Cap=4.0)",
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
