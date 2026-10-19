"""Experiment 6: Release Frequency Planning over Longitudinal Horizons (v0.2).

Simulates a 90-day reporting period under varying enterprise release cadences:
  - Daily: 90 releases (every 1 day)
  - Weekly: 13 releases (every 7 days)
  - Biweekly: 6 releases (every 14 days)
  - Monthly: 3 releases (every 30 days)

Evaluates:
  - Lifetime RDP cumulative privacy loss
  - Budget cap exhaustion and rejection behavior under lifetime accounting
  - Trade-off between analytics frequency and cumulative privacy loss
"""

import argparse
from decimal import Decimal
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
import pandas as pd

from private_llm_analytics.aggregation.user_matrix import build_user_matrix
from private_llm_analytics.evaluation.privacy_utility import run_repeated_noise_trials
from private_llm_analytics.evaluation.report import plot_single_line_chart, save_summary_table
from private_llm_analytics.mechanisms.gaussian import GaussianMechanism
from private_llm_analytics.metrics.normalized_share import UserNormalizedShareMetric
from private_llm_analytics.privacy.rdp import (
    DEFAULT_RDP_ORDERS,
    compute_gaussian_rdp,
    compose_rdp_vectors,
    rdp_to_epsilon_delta,
)
from private_llm_analytics.simulator.generator import SyntheticDataGenerator
from private_llm_analytics.simulator.scenario import SimulationScenario


def run_release_frequency_experiment(
    n_users: int = 50000,
    epsilon_per_release: float = 1.0,
    delta_calibration: float = 1e-6,
    delta_global: float = 1e-6,
    epsilon_cap: float = 10.0,
    horizon_days: int = 90,
    cadences: Optional[Dict[str, int]] = None,
    n_trials: int = 30,
    seed: int = 42,
    data_path: str = "data/interactions_n50000.parquet",
    output_dir: str = "results",
) -> pd.DataFrame:
    if cadences is None:
        cadences = {
            "daily": 90,
            "weekly": 13,
            "biweekly": 6,
            "monthly": 3,
        }

    print(f"\n=======================================================")
    print(f"Running Exp 06: Release Frequency Planning ({horizon_days} Days, Cap={epsilon_cap})")
    print(f"=======================================================")

    p = Path(data_path)
    if p.is_file():
        df = pd.read_parquet(p)
    else:
        scenario = SimulationScenario(n_users=n_users, random_seed=seed)
        gen = SyntheticDataGenerator(scenario)
        df = gen.generate_and_save(p, file_format="parquet", seed=seed)

    umatrix = build_user_matrix(df, public_population_size=n_users)
    metric = UserNormalizedShareMetric(norm_bound=1.0)
    sensitivity = metric.get_sensitivity("gaussian")

    mech = GaussianMechanism(
        epsilon=epsilon_per_release,
        delta=delta_calibration,
        sensitivity=sensitivity,
        method="analytic",
    )
    sigma = mech.noise_scale
    single_rdp = compute_gaussian_rdp(sigma=sigma, sensitivity=sensitivity, orders=DEFAULT_RDP_ORDERS)

    # Evaluate baseline single-release utility
    _, u_summary = run_repeated_noise_trials(
        user_matrix=umatrix,
        metric=metric,
        mechanism_type="gaussian",
        epsilon=epsilon_per_release,
        delta=delta_calibration,
        n_trials=n_trials,
        seed=seed + 600,
    )
    mae_mean = u_summary["mae_mean"]
    top1_acc = u_summary.get("top1_accuracy", 1.0)

    records = []
    for cadence_name, total_attempts in cadences.items():
        # Sequentially authorize releases under lifetime RDP until cap is reached
        accepted = 0
        rejected = 0
        current_rdp_vectors = []
        final_eff_eps = 0.0
        final_best_alpha = 0.0

        for release_idx in range(1, total_attempts + 1):
            cand_vectors = current_rdp_vectors + [single_rdp]
            composed = compose_rdp_vectors(cand_vectors, orders=DEFAULT_RDP_ORDERS)
            best_alpha, rdp_eff = rdp_to_epsilon_delta(composed, delta_global=delta_global)
            basic_sum = release_idx * epsilon_per_release
            eff_eps = min(basic_sum, float(rdp_eff))

            if eff_eps <= epsilon_cap + 1e-9:
                accepted += 1
                current_rdp_vectors = cand_vectors
                final_eff_eps = eff_eps
                final_best_alpha = float(best_alpha)
            else:
                rejected += 1

        utilization = (final_eff_eps / epsilon_cap) * 100.0 if epsilon_cap > 0 else 0.0

        records.append({
            "cadence": cadence_name,
            "horizon_days": horizon_days,
            "releases_attempted": total_attempts,
            "releases_accepted": accepted,
            "releases_rejected": rejected,
            "effective_epsilon": round(final_eff_eps, 4),
            "best_alpha": final_best_alpha,
            "budget_cap": epsilon_cap,
            "budget_utilization_pct": round(utilization, 2),
            "utility_mae_mean": round(mae_mean, 6),
            "utility_top1_accuracy": round(top1_acc, 4),
        })

        print(f"  Cadence: {cadence_name:8s} | Attempted: {total_attempts:2d} | Accepted: {accepted:2d} | Rejected: {rejected:2d} | Eff Eps: {final_eff_eps:6.2f} (Cap {epsilon_cap}) | Utilization: {utilization:5.1f}%")

    summary_df = pd.DataFrame(records)

    base_dir = Path(output_dir)
    tables_dir = base_dir / "tables"
    figures_dir = base_dir / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    csv_path = tables_dir / "exp06_release_frequency_summary.csv"
    save_summary_table(summary_df, str(csv_path))

    # Plot
    plot_single_line_chart(
        x=[r["cadence"] for r in records],
        y_series={
            "Effective Epsilon": [r["effective_epsilon"] for r in records],
            "Accepted Releases": [r["releases_accepted"] for r in records],
        },
        x_label="Release Cadence",
        y_label="Value",
        title=f"90-Day Longitudinal Privacy Spend by Cadence (Cap={epsilon_cap})",
        output_path=str(figures_dir / "exp06_release_frequency"),
    )

    print(f"\n[SUCCESS] Exp 06 Complete. Results saved to {csv_path}")
    return summary_df


def main():
    parser = argparse.ArgumentParser(description="Exp 06: Release Frequency Planning")
    parser.add_argument("--smoke", action="store_true", help="Run quick smoke test")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory")
    args = parser.parse_args()

    out_dir = args.output_dir or ("results/verification" if args.smoke else "results")
    cadences = {"weekly": 6, "monthly": 3} if args.smoke else None
    trials = 2 if args.smoke else 30

    run_release_frequency_experiment(
        cadences=cadences,
        n_trials=trials,
        output_dir=out_dir,
    )


if __name__ == "__main__":
    main()
