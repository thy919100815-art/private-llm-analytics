"""Experiment 7: Multi-Recipient Collusion & Composition Policies (v0.2).

Simulates 3 external recipients (Provider A, Provider B, Provider C) over a 12-week horizon:
  - Provider A: 1 release per week (12 releases)
  - Provider B: 1 release per week (12 releases)
  - Provider C: 1 release every 2 weeks (6 releases)
  Total attempted: 30 releases.

Evaluates under 3 Recipient Composition Policies:
  1. GLOBAL_EXTERNAL: All external recipients share a single scope (global_external, cap=6.0).
  2. COLLUSION_GROUP: Provider A and B share collusion:group_ab (cap=6.0); Provider C in collusion:group_c (cap=6.0).
  3. PER_RECIPIENT: Provider A, B, and C each possess independent privacy scopes (cap=6.0 each).
"""

import argparse
from decimal import Decimal
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
import pandas as pd

from private_llm_analytics.aggregation.user_matrix import build_user_matrix
from private_llm_analytics.evaluation.report import plot_single_line_chart, save_summary_table
from private_llm_analytics.mechanisms.gaussian import GaussianMechanism
from private_llm_analytics.metrics.normalized_share import UserNormalizedShareMetric
from private_llm_analytics.privacy.composition import resolve_budget_scope_id
from private_llm_analytics.privacy.rdp import (
    DEFAULT_RDP_ORDERS,
    compute_gaussian_rdp,
    compose_rdp_vectors,
    rdp_to_epsilon_delta,
)
from private_llm_analytics.simulator.generator import SyntheticDataGenerator
from private_llm_analytics.simulator.scenario import SimulationScenario


def run_recipient_composition_experiment(
    n_users: int = 50000,
    epsilon_per_release: float = 1.0,
    delta_calibration: float = 1e-6,
    delta_global: float = 1e-6,
    epsilon_cap: float = 6.0,
    n_weeks: int = 12,
    seed: int = 42,
    data_path: str = "data/interactions_n50000.parquet",
    output_dir: str = "results",
) -> pd.DataFrame:
    print(f"\n=======================================================")
    print(f"Running Exp 07: Multi-Recipient Collusion Policies ({n_weeks} Weeks, Cap={epsilon_cap})")
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

    # Build sequential query schedule across weeks
    # Each item: (week, recipient)
    schedule = []
    for w in range(1, n_weeks + 1):
        schedule.append((w, "provider_a"))
        schedule.append((w, "provider_b"))
        if w % 2 == 1:
            schedule.append((w, "provider_c"))

    policies = [
        ("GLOBAL_EXTERNAL", "global_external", None),
        (
            "COLLUSION_GROUP",
            "collusion_group",
            {"group_ab": ["provider_a", "provider_b"], "group_c": ["provider_c"]},
        ),
        ("PER_RECIPIENT", "per_recipient", None),
    ]

    records = []

    for policy_label, mode, groups in policies:
        print(f"\n--- Evaluating Policy: {policy_label} ---")
        # Track history per budget_scope_id
        # scope_id -> list of RDP vectors
        scopes_rdp: Dict[str, List] = {}
        scopes_stats: Dict[str, Dict] = {}

        for week, recipient in schedule:
            scope_id = resolve_budget_scope_id(
                recipient=recipient,
                recipient_role="external",
                mode=mode,
                groups=groups,
            )
            if scope_id not in scopes_rdp:
                scopes_rdp[scope_id] = []
                scopes_stats[scope_id] = {
                    "attempted": 0,
                    "accepted": 0,
                    "rejected": 0,
                    "exhaustion_week": None,
                    "final_eff_eps": 0.0,
                    "best_alpha": 0.0,
                }

            scopes_stats[scope_id]["attempted"] += 1
            cand_vectors = scopes_rdp[scope_id] + [single_rdp]
            composed = compose_rdp_vectors(cand_vectors, orders=DEFAULT_RDP_ORDERS)
            best_alpha, rdp_eff = rdp_to_epsilon_delta(composed, delta_global=delta_global)
            eff_eps = float(rdp_eff)

            if eff_eps <= epsilon_cap + 1e-9:
                scopes_rdp[scope_id] = cand_vectors
                scopes_stats[scope_id]["accepted"] += 1
                scopes_stats[scope_id]["final_eff_eps"] = eff_eps
                scopes_stats[scope_id]["best_alpha"] = float(best_alpha)
            else:
                scopes_stats[scope_id]["rejected"] += 1
                if scopes_stats[scope_id]["exhaustion_week"] is None:
                    scopes_stats[scope_id]["exhaustion_week"] = week

        for scope_id, stats in scopes_stats.items():
            records.append({
                "policy": policy_label,
                "scope_id": scope_id,
                "releases_attempted": stats["attempted"],
                "releases_accepted": stats["accepted"],
                "releases_rejected": stats["rejected"],
                "effective_epsilon": round(stats["final_eff_eps"], 4),
                "best_alpha": stats["best_alpha"],
                "exhaustion_week": stats["exhaustion_week"] if stats["exhaustion_week"] is not None else "None",
                "budget_cap": epsilon_cap,
            })
            print(f"  Scope: {scope_id:20s} | Attempted: {stats['attempted']:2d} | Accepted: {stats['accepted']:2d} | Rejected: {stats['rejected']:2d} | Eff Eps: {stats['final_eff_eps']:6.2f} | Exhausted Week: {stats['exhaustion_week']}")

    summary_df = pd.DataFrame(records)

    base_dir = Path(output_dir)
    tables_dir = base_dir / "tables"
    figures_dir = base_dir / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    csv_path = tables_dir / "exp07_recipient_composition_summary.csv"
    save_summary_table(summary_df, str(csv_path))

    # Plot
    labels = [f"{r['policy']}\n({r['scope_id'].split(':')[-1]})" for r in records]
    plot_single_line_chart(
        x=labels,
        y_series={
            "Releases Accepted": [r["releases_accepted"] for r in records],
            "Effective Epsilon": [r["effective_epsilon"] for r in records],
        },
        x_label="Policy & Scope",
        y_label="Value",
        title=f"Multi-Recipient Composition & Collusion Impact (Cap={epsilon_cap})",
        output_path=str(figures_dir / "exp07_recipient_composition"),
    )

    print(f"\n[SUCCESS] Exp 07 Complete. Results saved to {csv_path}")
    return summary_df


def main():
    parser = argparse.ArgumentParser(description="Exp 07: Multi-Recipient Collusion Policies")
    parser.add_argument("--smoke", action="store_true", help="Run quick smoke test")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory")
    args = parser.parse_args()

    out_dir = args.output_dir or ("results/verification" if args.smoke else "results")
    weeks = 4 if args.smoke else 12

    run_recipient_composition_experiment(
        n_weeks=weeks,
        output_dir=out_dir,
    )


if __name__ == "__main__":
    main()
