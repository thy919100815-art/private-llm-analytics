"""Experiment 8: Longitudinal Authorization Trajectory & Cap Exhaustion (v0.2).

Simulates Production longitudinal authorization under a real SQLite state database:
  - Sequential queries submitted one by one
  - Each step executes atomic reserve -> compose candidate with persistent history -> authorize or reject -> commit
  - Continues until the global privacy budget cap is exhausted
  - Records step-by-step trajectory (epsilon before, candidate incremental cost, epsilon after, best alpha, decision, remaining budget)
"""

import argparse
from decimal import Decimal
from pathlib import Path
import tempfile
import time
from typing import Dict, List, Optional
import numpy as np
import pandas as pd

from private_llm_analytics.aggregation.user_matrix import build_user_matrix
from private_llm_analytics.config.loader import load_app_config
from private_llm_analytics.evaluation.report import plot_single_line_chart, save_summary_table
from private_llm_analytics.planner.query import AnalyticsQuery
from private_llm_analytics.privacy.exceptions import PrivacyBudgetExceededError
from private_llm_analytics.privacy.state import PrivacyStateManager
from private_llm_analytics.release.release import ReleaseOrchestrator
from private_llm_analytics.simulator.generator import SyntheticDataGenerator
from private_llm_analytics.simulator.scenario import SimulationScenario


def run_longitudinal_budget_experiment(
    n_users: int = 50000,
    epsilon_per_release: float = 1.0,
    delta_calibration: float = 1e-6,
    delta_global: float = 1e-6,
    epsilon_cap: float = 5.0,
    max_steps: int = 15,
    seed: int = 42,
    data_path: str = "data/interactions_n50000.parquet",
    output_dir: str = "results",
) -> pd.DataFrame:
    print(f"\n=======================================================")
    print(f"Running Exp 08: Longitudinal Authorization Trajectory (Cap={epsilon_cap})")
    print(f"=======================================================")

    p = Path(data_path)
    if p.is_file():
        df = pd.read_parquet(p)
    else:
        scenario = SimulationScenario(n_users=n_users, random_seed=seed)
        gen = SyntheticDataGenerator(scenario)
        df = gen.generate_and_save(p, file_format="parquet", seed=seed)

    umatrix = build_user_matrix(df, public_population_size=n_users)

    # Use a temporary database for the experiment run
    with tempfile.TemporaryDirectory() as tmp_dir:
        db_path = Path(tmp_dir) / "privacy_state.db"
        config = load_app_config("configs")
        config.privacy.global_budget.epsilon = epsilon_cap
        config.privacy.global_budget.delta = delta_calibration
        config.privacy.accountant.epsilon_cap = epsilon_cap
        config.privacy.accountant.delta_global = delta_global

        state_mgr = PrivacyStateManager(db_path)
        orch = ReleaseOrchestrator(app_config=config, state_manager=state_mgr)

        trajectory_records = []
        metrics_cycle = ["user_prevalence", "user_normalized_share", "interaction_volume"]

        for step in range(1, max_steps + 1):
            view_before = orch.get_privacy_view("provider_a")
            eps_before = float(view_before.effective_epsilon)

            # Construct semantically distinct query each step
            metric_choice = metrics_cycle[(step - 1) % len(metrics_cycle)]
            q = AnalyticsQuery(
                metric=metric_choice,
                recipient="provider_a",
                epsilon=epsilon_per_release,
                delta=delta_calibration,
                window_start=f"2026-01-{(step % 28) + 1:02d}",
                window_end="2026-01-31",
            )

            decision = "ALLOWED"
            eps_after = eps_before
            best_alpha = float(view_before.best_alpha) if view_before.best_alpha else 1.0

            try:
                res = orch.execute_release(q, umatrix)
                assert not res["cache_hit"]
                view_after = orch.get_privacy_view("provider_a")
                eps_after = float(view_after.effective_epsilon)
                best_alpha = float(view_after.best_alpha) if view_after.best_alpha else 1.0
            except PrivacyBudgetExceededError:
                decision = "REJECTED"

            remaining_eps = max(0.0, epsilon_cap - eps_after)

            record = {
                "step": step,
                "metric": metric_choice,
                "query_epsilon": epsilon_per_release,
                "epsilon_before": round(eps_before, 4),
                "epsilon_after": round(eps_after, 4),
                "best_alpha": best_alpha,
                "decision": decision,
                "remaining_epsilon": round(remaining_eps, 4),
                "budget_cap": epsilon_cap,
            }
            trajectory_records.append(record)
            print(f"  Step {step:2d} | Metric: {metric_choice:22s} | Before: {eps_before:6.2f} | After: {eps_after:6.2f} | Decision: {decision:8s} | Remaining: {remaining_eps:6.2f}")

            # Stop a couple steps after first rejection to avoid unnecessary loops
            rejection_count = sum(1 for r in trajectory_records if r["decision"] == "REJECTED")
            if rejection_count >= 3:
                break

    trajectory_df = pd.DataFrame(trajectory_records)

    base_dir = Path(output_dir)
    tables_dir = base_dir / "tables"
    figures_dir = base_dir / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    csv_path = tables_dir / "exp08_longitudinal_budget_trajectory.csv"
    save_summary_table(trajectory_df, str(csv_path))

    # Plot
    plot_single_line_chart(
        x=[str(r["step"]) for r in trajectory_records],
        y_series={
            "Effective Epsilon": [r["epsilon_after"] for r in trajectory_records],
            "Remaining Budget": [r["remaining_epsilon"] for r in trajectory_records],
        },
        x_label="Sequential Query Step",
        y_label="Epsilon Budget",
        title=f"Longitudinal Authorization Trajectory until Cap ({epsilon_cap})",
        output_path=str(figures_dir / "exp08_longitudinal_budget"),
    )

    print(f"\n[SUCCESS] Exp 08 Complete. Results saved to {csv_path}")
    return trajectory_df


def main():
    parser = argparse.ArgumentParser(description="Exp 08: Longitudinal Authorization Trajectory")
    parser.add_argument("--smoke", action="store_true", help="Run quick smoke test")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory")
    args = parser.parse_args()

    out_dir = args.output_dir or ("results/verification" if args.smoke else "results")
    max_steps = 6 if args.smoke else 15

    run_longitudinal_budget_experiment(
        max_steps=max_steps,
        output_dir=out_dir,
    )


if __name__ == "__main__":
    main()
