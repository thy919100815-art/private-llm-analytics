"""Experiment 12: Longitudinal Frequency Release Planner.

Evaluates frequency adaptation across initial budget exhaustion levels:
  - Spent budget ratios: {0.0, 0.25, 0.50, 0.75} of epsilon_cap = 4.0
  - Candidate cadences F in {1 (daily), 7 (weekly), 14 (biweekly), 30 (monthly)}
  - Over a 90-day planning horizon.
"""

import argparse
from decimal import Decimal
from pathlib import Path
import time
from typing import Dict, List, Optional, Sequence
import pandas as pd

from private_llm_analytics.config.loader import load_app_config
from private_llm_analytics.evaluation.report import plot_single_line_chart, save_summary_table
from private_llm_analytics.planning.candidates import ReleasePolicyCandidate
from private_llm_analytics.planning.planner import SmallNReleasePlanner
from private_llm_analytics.planning.utility import NoiseEstimator
from private_llm_analytics.privacy.composition import PrivacyView
from private_llm_analytics.privacy.rdp import (
    compute_gaussian_rdp,
    rdp_to_epsilon_delta,
)


class MockPriorHistoryState:
    """Mock state manager holding pre-existing historical RDP consumption."""

    def __init__(self, spent_epsilon: float, delta_global: float, orders: Sequence[Decimal]):
        self.spent_epsilon = float(spent_epsilon)
        self.delta_global = Decimal(str(delta_global))
        self.orders = list(orders)

        if spent_epsilon > 0.0:
            # Construct a base RDP vector that converts to spent_epsilon
            cand = ReleasePolicyCandidate(contribution_bound=20.0, mechanism="gaussian")
            sigma, sens = NoiseEstimator.estimate_noise_sigma("interaction_volume", cand)
            base_rdp = compute_gaussian_rdp(sigma=sigma, sensitivity=sens, orders=self.orders)
            _, single_eps = rdp_to_epsilon_delta(base_rdp, delta_global=self.delta_global)
            multiplier = Decimal(str(spent_epsilon)) / single_eps
            self.hist_rdp = {o: base_rdp[o] * multiplier for o in self.orders}
            self.release_count = int(float(multiplier))
        else:
            self.hist_rdp = {o: Decimal("0.0") for o in self.orders}
            self.release_count = 0

    def get_privacy_view(self, budget_scope_id, delta_global, orders, epsilon_cap, accountant_mode):
        _, eff_eps = rdp_to_epsilon_delta(self.hist_rdp, delta_global=self.delta_global) if self.spent_epsilon > 0 else (None, Decimal("0.0"))
        rem_eps = max(Decimal("0.0"), Decimal(str(epsilon_cap)) - eff_eps)
        return PrivacyView(
            budget_scope_id=budget_scope_id,
            release_count=self.release_count,
            total_rdp_vector=self.hist_rdp,
            effective_epsilon=eff_eps,
            best_alpha=self.orders[0] if self.orders else None,
            delta_global=self.delta_global,
            epsilon_cap=Decimal(str(epsilon_cap)),
            remaining_epsilon=rem_eps,
            accountant_mode=accountant_mode,
            orders=self.orders,
        )


def run_frequency_planner_experiment(
    spent_ratios: Sequence[float] = (0.0, 0.25, 0.50, 0.75),
    epsilon_cap: float = 4.0,
    horizon_days: int = 90,
    candidate_frequencies: Sequence[int] = (1, 7, 14, 30),
    output_dir: str = "results",
) -> pd.DataFrame:
    print("\n=======================================================")
    print("Running Exp 12: Longitudinal Frequency Release Planner")
    print(f"Spent Ratios: {spent_ratios} of Cap = {epsilon_cap}")
    print(f"Planning Horizon: {horizon_days} days")
    print(f"Candidate Frequencies: {candidate_frequencies}")
    print("=======================================================")

    out_base = Path(output_dir)
    tables_dir = out_base / "tables"
    figures_dir = out_base / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    app_config = load_app_config("configs")
    app_config.privacy.accountant.epsilon_cap = epsilon_cap
    app_config.privacy.planner.candidate_frequencies = list(candidate_frequencies)
    app_config.privacy.planner.planning_horizon_days = horizon_days
    eff_orders = [Decimal(str(a)) for a in app_config.privacy.accountant.orders]

    records: List[Dict] = []

    for ratio in spent_ratios:
        spent_eps = ratio * epsilon_cap
        state_mgr = MockPriorHistoryState(
            spent_epsilon=spent_eps,
            delta_global=app_config.privacy.accountant.delta_global,
            orders=eff_orders,
        )

        planner = SmallNReleasePlanner(app_config=app_config, state_manager=state_mgr)
        plan_res = planner.plan(
            metric="interaction_volume",
            recipient="recipient_internal",
            planning_horizon_days=horizon_days,
            force_recompute=True,
        )

        rec = {
            "spent_budget_ratio": ratio,
            "initial_spent_epsilon": round(spent_eps, 2),
            "remaining_epsilon_budget": round(epsilon_cap - spent_eps, 2),
            "selected_frequency_days": plan_res.policy.frequency_days if plan_res.policy else None,
            "planned_releases_count": plan_res.planned_releases_count,
            "projected_total_epsilon": round(plan_res.projected_epsilon, 4),
            "budget_feasible": plan_res.budget_feasible,
            "decision_reason": plan_res.decision_reason,
            "selected_granularity": plan_res.policy.granularity if plan_res.policy else None,
            "selected_clipping": plan_res.policy.contribution_bound if plan_res.policy else None,
        }
        records.append(rec)
        print(
            f"  Spent={spent_eps:4.1f}/{epsilon_cap:4.1f} ({ratio*100:2.0f}%) | "
            f"Selected F={rec['selected_frequency_days']} days (K={rec['planned_releases_count']} releases) | "
            f"Projected eps={rec['projected_total_epsilon']:.2f} | Feasible={rec['budget_feasible']}"
        )

    results_df = pd.DataFrame(records)
    save_summary_table(results_df, str(tables_dir / "exp12_frequency_planner.csv"))

    # Plot Selected Frequency vs Initial Spent Budget
    valid_df = results_df[results_df["budget_feasible"] == True]
    plot_single_line_chart(
        x=(valid_df["spent_budget_ratio"] * 100).tolist(),
        y_series={
            "Release Cadence Interval (Days)": valid_df["selected_frequency_days"].tolist(),
            "Planned Releases (Count)": valid_df["planned_releases_count"].tolist(),
        },
        x_label="Pre-existing Budget Exhaustion (%)",
        y_label="Cadence / Release Count",
        title="Adaptive Release Frequency Downshifting vs Privacy Consumption",
        output_path=str(figures_dir / "exp12_frequency_planner"),
        x_log=False,
    )

    print(f"\n[SUCCESS] Exp 12 completed. Saved to {tables_dir}/ and {figures_dir}/")
    return results_df


def main():
    parser = argparse.ArgumentParser(description="Run Experiment 12: Frequency Release Planner")
    parser.add_argument("--smoke", action="store_true", help="Run rapid smoke test")
    parser.add_argument("--output-dir", type=str, default="results", help="Output directory")
    args = parser.parse_args()

    out_root = "results/verification" if args.smoke else args.output_dir
    if args.smoke:
        run_frequency_planner_experiment(
            spent_ratios=(0.0, 0.50),
            output_dir=out_root,
        )
    else:
        run_frequency_planner_experiment(output_dir=out_root)


if __name__ == "__main__":
    main()
