"""Experiment 12: Longitudinal Frequency Release Planner.

Evaluates cadence adaptation across discrete integer historical releases:
  - Finds integer past release counts k closest to {0%, 25%, 50%, 75%} budget exhaustion of epsilon_cap = 4.0
  - Composes exact Gaussian RDP across longitudinal horizons
  - Records actual_initial_epsilon, actual_initial_ratio, and selected cadences F in {1, 7, 14, 30}.
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
    compose_rdp_vectors,
    rdp_to_epsilon_delta,
)


class DiscreteHistoricalState:
    """Mock state manager holding exact discrete historical release RDP vectors."""

    def __init__(self, historical_release_count: int, single_rdp: Dict[Decimal, Decimal], orders: Sequence[Decimal], delta_global: Decimal, epsilon_cap: Decimal):
        self.release_count = int(historical_release_count)
        self.orders = list(orders)
        self.delta_global = delta_global
        self.epsilon_cap = epsilon_cap

        if self.release_count > 0:
            self.total_rdp = {o: single_rdp[o] * Decimal(str(self.release_count)) for o in self.orders}
            best_alpha, eff_eps = rdp_to_epsilon_delta(self.total_rdp, delta_global=self.delta_global)
            self.effective_epsilon = eff_eps
            self.best_alpha = best_alpha
        else:
            self.total_rdp = {o: Decimal("0.0") for o in self.orders}
            self.effective_epsilon = Decimal("0.0")
            self.best_alpha = None

        self.remaining_epsilon = max(Decimal("0.0"), self.epsilon_cap - self.effective_epsilon)

    def get_privacy_view(self, budget_scope_id, delta_global, orders, epsilon_cap, accountant_mode):
        return PrivacyView(
            budget_scope_id=budget_scope_id,
            release_count=self.release_count,
            total_rdp_vector=self.total_rdp,
            effective_epsilon=self.effective_epsilon,
            best_alpha=self.best_alpha,
            delta_global=self.delta_global,
            epsilon_cap=self.epsilon_cap,
            remaining_epsilon=self.remaining_epsilon,
            accountant_mode=accountant_mode,
            orders=self.orders,
        )


def run_frequency_planner_experiment(
    target_ratios: Optional[Sequence[float]] = None,
    spent_ratios: Optional[Sequence[float]] = None,
    epsilon_cap: float = 4.0,
    horizon_days: int = 90,
    candidate_frequencies: Sequence[int] = (1, 7, 14, 30),
    output_dir: str = "results",
) -> pd.DataFrame:
    ratios = target_ratios if target_ratios is not None else (spent_ratios if spent_ratios is not None else (0.0, 0.25, 0.50, 0.75))
    print("\n=======================================================")
    print("Running Exp 12: Longitudinal Frequency Release Planner")
    print(f"Target Spent Ratios: {ratios} of Cap = {epsilon_cap}")
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
    delta_global = Decimal(str(app_config.privacy.accountant.delta_global))
    dec_cap = Decimal(str(epsilon_cap))

    # Single release RDP cost
    cand_single = ReleasePolicyCandidate(contribution_bound=20.0, mechanism="gaussian")
    sigma, sens = NoiseEstimator.estimate_noise_sigma("interaction_volume", cand_single)
    single_rdp = compute_gaussian_rdp(sigma=sigma, sensitivity=sens, orders=eff_orders)

    # 1. Precompute discrete release trajectory (k = 0..20)
    discrete_table: List[Tuple[int, float]] = []
    for k in range(0, 25):
        if k == 0:
            discrete_table.append((0, 0.0))
        else:
            k_rdp = {o: single_rdp[o] * Decimal(str(k)) for o in eff_orders}
            _, eps_k = rdp_to_epsilon_delta(k_rdp, delta_global=delta_global)
            discrete_table.append((k, float(eps_k)))

    records: List[Dict] = []

    for target_ratio in ratios:
        target_eps = target_ratio * epsilon_cap
        # Find closest integer release count k
        best_k, best_eps = min(discrete_table, key=lambda x: abs(x[1] - target_eps))
        actual_ratio = best_eps / epsilon_cap

        state_mgr = DiscreteHistoricalState(
            historical_release_count=best_k,
            single_rdp=single_rdp,
            orders=eff_orders,
            delta_global=delta_global,
            epsilon_cap=dec_cap,
        )

        planner = SmallNReleasePlanner(app_config=app_config, state_manager=state_mgr)
        plan_res = planner.plan(
            metric="interaction_volume",
            recipient="recipient_internal",
            planning_horizon_days=horizon_days,
            force_recompute=True,
        )

        rec = {
            "target_ratio": target_ratio,
            "historical_release_count": best_k,
            "actual_initial_epsilon": round(best_eps, 4),
            "actual_initial_ratio": round(actual_ratio, 4),
            "remaining_epsilon_budget": round(epsilon_cap - best_eps, 4),
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
            f"  Target={target_ratio*100:2.0f}% | Discrete Past Releases: k={best_k:2d} (eps={best_eps:4.2f}, ratio={actual_ratio*100:4.1f}%) | "
            f"Selected F={rec['selected_frequency_days']} days (K={rec['planned_releases_count']} planned) | "
            f"Projected Total eps={rec['projected_total_epsilon']:4.2f} | Feasible={rec['budget_feasible']}"
        )

    results_df = pd.DataFrame(records)
    save_summary_table(results_df, str(tables_dir / "exp12_frequency_planner.csv"))

    # Plot Selected Frequency vs Actual Initial Budget Exhaustion
    valid_df = results_df[results_df["budget_feasible"] == True]
    plot_single_line_chart(
        x=(valid_df["actual_initial_ratio"] * 100).tolist(),
        y_series={
            "Release Cadence Interval (Days)": valid_df["selected_frequency_days"].tolist(),
            "Planned Releases (Count)": valid_df["planned_releases_count"].tolist(),
        },
        x_label="Actual Historical Budget Exhaustion (%)",
        y_label="Cadence / Release Count",
        title="Adaptive Release Frequency Downshifting vs Discrete Past Releases",
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
            target_ratios=(0.0, 0.50),
            output_dir=out_root,
        )
    else:
        run_frequency_planner_experiment(output_dir=out_root)


if __name__ == "__main__":
    main()
