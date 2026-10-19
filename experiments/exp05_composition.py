"""Experiment 5: Basic vs RDP Longitudinal Composition (v0.2).

Compares:
  - Basic composition (sum of epsilons)
  - RDP composition over T in [1, 2, 5, 10, 20, 50, 100] releases
  - Canonical public population N = 50,000, metric: user_normalized_share
  - Evaluates both Privacy metrics (effective epsilon, best alpha, relative reduction)
    and Utility metrics (MAE, RMSE, TV, ranking accuracy).
"""

import argparse
from decimal import Decimal
from pathlib import Path
import time
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
    to_decimal,
)
from private_llm_analytics.simulator.generator import SyntheticDataGenerator
from private_llm_analytics.simulator.scenario import SimulationScenario


def run_composition_experiment(
    n_users: int = 50000,
    epsilon_release: float = 1.0,
    delta_calibration: float = 1e-6,
    delta_global: float = 1e-6,
    t_values: Optional[List[int]] = None,
    n_trials: int = 50,
    seed: int = 42,
    data_path: str = "data/interactions_n50000.parquet",
    output_dir: str = "results",
) -> pd.DataFrame:
    if t_values is None:
        t_values = [1, 2, 5, 10, 20, 50, 100]

    print(f"\n=======================================================")
    print(f"Running Exp 05: Basic vs RDP Composition (N={n_users}, T={t_values})")
    print(f"=======================================================")

    # 1. Load or generate data
    p = Path(data_path)
    if p.is_file():
        print(f"Loading existing dataset from {p}...")
        df = pd.read_parquet(p)
    else:
        print(f"Generating synthetic dataset with N={n_users} users...")
        scenario = SimulationScenario(n_users=n_users, random_seed=seed)
        gen = SyntheticDataGenerator(scenario)
        df = gen.generate_and_save(p, file_format="parquet", seed=seed)

    umatrix = build_user_matrix(df, public_population_size=n_users)
    metric = UserNormalizedShareMetric(norm_bound=1.0)
    sensitivity = metric.get_sensitivity("gaussian")

    # Calibrate Gaussian mechanism
    mech = GaussianMechanism(
        epsilon=epsilon_release,
        delta=delta_calibration,
        sensitivity=sensitivity,
        method="analytic",
    )
    sigma = mech.noise_scale

    # Base RDP vector per single release
    single_rdp = compute_gaussian_rdp(
        sigma=sigma,
        sensitivity=sensitivity,
        orders=DEFAULT_RDP_ORDERS,
    )

    # Measure baseline utility for a single release
    print(f"Evaluating empirical utility over {n_trials} noise trials...")
    _, utility_summary = run_repeated_noise_trials(
        user_matrix=umatrix,
        metric=metric,
        mechanism_type="gaussian",
        epsilon=epsilon_release,
        delta=delta_calibration,
        n_trials=n_trials,
        seed=seed + 500,
    )
    mae_mean = utility_summary["mae_mean"]
    rmse_mean = utility_summary["rmse_mean"]
    tv_mean = utility_summary.get("tv_mean", 0.0)
    top1_acc = utility_summary.get("top1_accuracy", 1.0)

    records = []
    for t in t_values:
        basic_eps = float(epsilon_release * t)

        # RDP composition: scale vector by t
        composed_rdp = compose_rdp_vectors([single_rdp] * t, orders=DEFAULT_RDP_ORDERS)
        best_alpha, rdp_eff = rdp_to_epsilon_delta(composed_rdp, delta_global=delta_global)
        rdp_eps = float(rdp_eff)
        effective_eps = min(basic_eps, rdp_eps)

        reduction = max(0.0, (basic_eps - effective_eps) / basic_eps) if basic_eps > 0 else 0.0

        records.append({
            "T": t,
            "epsilon_per_release": epsilon_release,
            "delta_calibration": delta_calibration,
            "delta_global": delta_global,
            "basic_epsilon": round(basic_eps, 4),
            "rdp_epsilon": round(rdp_eps, 4),
            "effective_epsilon": round(effective_eps, 4),
            "best_alpha": float(best_alpha),
            "relative_reduction": round(reduction, 4),
            "reduction_percentage": f"{reduction * 100:.1f}%",
            "utility_mae_mean": round(mae_mean, 6),
            "utility_rmse_mean": round(rmse_mean, 6),
            "utility_tv_mean": round(tv_mean, 6),
            "utility_top1_accuracy": round(top1_acc, 4),
        })
        print(f"  T={t:3d} | Basic: {basic_eps:7.2f} | RDP: {rdp_eps:7.2f} | Effective: {effective_eps:7.2f} | Best alpha: {float(best_alpha):4.1f} | Reduction: {reduction*100:5.1f}%")

    summary_df = pd.DataFrame(records)

    # Save outputs
    base_dir = Path(output_dir)
    tables_dir = base_dir / "tables"
    figures_dir = base_dir / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    csv_path = tables_dir / "exp05_composition_summary.csv"
    save_summary_table(summary_df, str(csv_path))

    # Plot
    plot_single_line_chart(
        x=[str(r["T"]) for r in records],
        y_series={
            "Basic Composition": [r["basic_epsilon"] for r in records],
            "RDP Composition": [r["effective_epsilon"] for r in records],
        },
        x_label="Number of Releases (T)",
        y_label="Cumulative Epsilon",
        title="Privacy Loss: Basic vs RDP Composition (N=50,000, delta=1e-6)",
        output_path=str(figures_dir / "exp05_composition"),
    )

    print(f"\n[SUCCESS] Exp 05 Complete. Results saved to {csv_path}")
    return summary_df


def main():
    parser = argparse.ArgumentParser(description="Exp 05: Basic vs RDP Composition")
    parser.add_argument("--smoke", action="store_true", help="Run quick smoke test with reduced trials/T")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory")
    args = parser.parse_args()

    out_dir = args.output_dir or ("results/verification" if args.smoke else "results")
    t_vals = [1, 2, 5] if args.smoke else [1, 2, 5, 10, 20, 50, 100]
    trials = 2 if args.smoke else 50

    run_composition_experiment(
        t_values=t_vals,
        n_trials=trials,
        output_dir=out_dir,
    )


if __name__ == "__main__":
    main()
