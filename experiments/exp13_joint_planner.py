"""Experiment 13: Joint Policy Release Planner (C, G, W, F) with True Empirical Oracle.

Evaluates Joint Policy Optimization:
  - Baseline A: Fixed (C=20, G=full_9_cell, W=30, F=7)
  - Baseline B: Fixed (C=40, G=provider_3_cell, W=30, F=14)
  - Static Public Joint Planner (v0.3): automatic joint selection under PublicContributionPrior
  - True Empirical Oracle: argmin true_utility evaluated directly on window-specific ground truth data
Across populations N in {500, 1000, 5000} and budget caps epsilon_cap in {2.0, 4.0}.

Uses declared workload and temporal loss functions on actual window-filtered data (W in {7, 14, 30, 60, 90}).
Reports Oracle mean, standard error, 95% CI, and logs explicit records for infeasible conditions.
"""

import argparse
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from experiments.temporal_truth import generate_rate_consistent_temporal_truth
except ImportError:
    from temporal_truth import generate_rate_consistent_temporal_truth
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
from private_llm_analytics.planning.utility import (
    PublicContributionPrior,
    UtilityEstimator,
    UtilityWeights,
    compute_temporal_utility_components,
    compute_total_expected_release_utility,
)


def check_all_windows_temporal_match(
    diag_by_w: Dict[int, Dict[str, float]],
    mean_threshold: float = 0.05,
    std_threshold: float = 0.10,
) -> bool:
    """Return True if all candidate windows satisfy mean and std thresholds, False if any window mismatches."""
    return not any(
        d.get("mean_relative_model_error", 0.0) >= mean_threshold
        or d.get("std_relative_model_error", 0.0) >= std_threshold
        for d in diag_by_w.values()
    )


def compute_true_candidate_utility(
    candidate: ReleasePolicyCandidate,
    umatrix: Any,
    cell_names: List[str],
    reference_days: int = 30,
    n_trials: int = 200,
    seed: int = 42,
    workload_weights: Optional[Dict[str, float]] = None,
    weights_data: float = 1.0,
    weights_stale: float = 0.5,
    weights_win: float = 0.2,
) -> Tuple[float, float, Tuple[float, float]]:
    """Compute true empirical utility on ground truth user data without planner predictions.

    Uses actual reconstructed end-to-end release error under declared workload queries:
        V_9_hat = R_G(V_G_DP)
        L_data_trial = sum(w_Q * ||Q V_9_hat - Q V_raw||_1) / (sum(w_Q) * max(1, ||V_raw||_1))
        L_trial = w_data * L_data_trial + w_stale * L_stale + w_win * L_win
    """
    raw_mat = umatrix.matrix
    w_days = candidate.window_days
    V_raw = np.sum(raw_mat, axis=0)
    true_total_vol = max(1.0, float(np.sum(V_raw)))

    # 1. Direct target-space contribution + clipping: bar_V_G
    clipped_users, bar_V_G, target_names, _ = build_volume_contributions_for_granularity(
        raw_user_matrix=umatrix,
        granularity=candidate.granularity,
        clip_norm=candidate.contribution_bound or 20.0,
    )

    # 2. DP Gaussian noise scale sigma
    sigma = calibrate_analytic_gaussian(
        epsilon=candidate.epsilon_per_release,
        delta=candidate.delta_per_release,
        l2_sensitivity=candidate.contribution_bound or 20.0,
    )
    dim = len(bar_V_G)

    # 3. Workload queries
    w_dict = workload_weights or {
        "full_9_cell": 0.4,
        "provider": 0.3,
        "task": 0.2,
        "overall": 0.1,
    }
    w9 = float(w_dict.get("full_9_cell", 0.0))
    wp = float(w_dict.get("provider", 0.0))
    wt = float(w_dict.get("task", 0.0))
    wo = float(w_dict.get("overall", 0.0))
    tot_w = w9 + wp + wt + wo
    if tot_w <= 0:
        w9 = 1.0
        tot_w = 1.0

    raw_prov, _ = aggregate_to_granularity(V_raw, cell_names, "provider_3_cell")
    raw_task, _ = aggregate_to_granularity(V_raw, cell_names, "task_3_cell")
    raw_over, _ = aggregate_to_granularity(V_raw, cell_names, "overall_1_cell")

    # 4. Freshness and window relevance loss via shared single source of truth
    emp_stale_loss, emp_win_loss = compute_temporal_utility_components(
        frequency_days=candidate.frequency_days,
        window_days=w_days,
        reference_days=reference_days,
    )
    utility_weights_obj = UtilityWeights(
        objective_mode="EXPECTED_RELEASE_L1",
        data=weights_data,
        staleness=weights_stale,
        window=weights_win,
    )

    # Evaluate across trials using Common Random Numbers (CRN)
    trial_scores = []
    for t in range(n_trials):
        rng = np.random.default_rng(seed + t * 100)
        Z_G = rng.normal(loc=0.0, scale=sigma, size=dim)
        V_G_DP = bar_V_G + Z_G

        # Deterministic reconstruction to 9-cell
        V_9_hat = reconstruct_to_9_cell(
            marginal_vector=V_G_DP,
            source_granularity=candidate.granularity,
            marginal_names=target_names,
            target_9_cell_names=cell_names,
        )

        weighted_err = 0.0
        if w9 > 0:
            weighted_err += w9 * np.sum(np.abs(V_9_hat - V_raw))
        if wp > 0:
            rec_prov, _ = aggregate_to_granularity(V_9_hat, cell_names, "provider_3_cell")
            weighted_err += wp * np.sum(np.abs(rec_prov - raw_prov))
        if wt > 0:
            rec_task, _ = aggregate_to_granularity(V_9_hat, cell_names, "task_3_cell")
            weighted_err += wt * np.sum(np.abs(rec_task - raw_task))
        if wo > 0:
            rec_over, _ = aggregate_to_granularity(V_9_hat, cell_names, "overall_1_cell")
            weighted_err += wo * np.sum(np.abs(rec_over - raw_over))

        L_data_trial = float((weighted_err / tot_w) / true_total_vol)
        trial_score = compute_total_expected_release_utility(
            data_loss=L_data_trial,
            staleness_loss=emp_stale_loss,
            window_loss=emp_win_loss,
            weights=utility_weights_obj,
        )
        trial_scores.append(trial_score)

    mean_util = float(np.mean(trial_scores))
    std_err = float(np.std(trial_scores) / np.sqrt(max(1, len(trial_scores))))
    ci_low = float(mean_util - 1.96 * std_err)
    ci_high = float(mean_util + 1.96 * std_err)
    return mean_util, std_err, (ci_low, ci_high)


def run_joint_planner_experiment(
    populations: Sequence[int] = (500, 1000, 5000),
    epsilon_caps: Sequence[float] = (2.0, 4.0),
    horizon_days: int = 90,
    n_trials: int = 200,
    seed: int = 42,
    output_dir: str = "results",
) -> pd.DataFrame:
    print("\n=======================================================")
    print("Running Exp 13: Joint Policy Release Planner (C, G, W, F)")
    print(f"Populations: {populations}")
    print(f"Budget Caps: {epsilon_caps}")
    print(f"Planning Horizon: {horizon_days} days, Oracle Trials: {n_trials}")
    print("=======================================================")

    out_base = Path(output_dir)
    tables_dir = out_base / "tables"
    figures_dir = out_base / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    app_config = load_app_config("configs")
    app_config.privacy.planner.planning_horizon_days = horizon_days
    ref_days = app_config.privacy.planner.window_model.reference_days

    records: List[Dict[str, Any]] = []
    temporal_records: List[Dict[str, Any]] = []

    for n in populations:
        print(f"\n--- Population N = {n} ---")
        app_config.scenario.population.n_users = n

        # Generate rate-consistent 90-day timestamped synthetic dataset
        df_90 = generate_rate_consistent_temporal_truth(
            n_users=n,
            horizon_days=horizon_days,
            reference_days=ref_days,
            reference_mean=60.0,
            reference_std=20.0,
            seed=seed,
        )
        t_end = df_90["timestamp"].max()

        # Build window-specific user matrices for W in candidate_windows
        candidate_windows = app_config.privacy.planner.candidate_windows
        umatrices_by_window: Dict[int, Any] = {}
        for w in candidate_windows:
            t_start = t_end - pd.Timedelta(days=w)
            df_w = df_90[(df_90["timestamp"] > t_start) & (df_90["timestamp"] <= t_end)]
            umatrices_by_window[w] = build_user_matrix(df_w, public_population_size=n)

        cell_names = umatrices_by_window[ref_days].cell_names

        # Evaluate temporal distribution diagnostics for all W in {7, 14, 30, 60, 90}
        all_user_ids = df_90["user_id"].unique()
        diag_by_w: Dict[int, Dict[str, float]] = {}
        for w in [7, 14, 30, 60, 90]:
            t_start = t_end - pd.Timedelta(days=w)
            df_w = df_90[(df_90["timestamp"] > t_start) & (df_90["timestamp"] <= t_end)]
            counts_w = df_w.groupby("user_id").size().reindex(all_user_ids, fill_value=0)
            emp_mean_w = float(counts_w.mean())
            emp_std_w = float(counts_w.std())
            pub_mean_w = 60.0 * (float(w) / float(ref_days))
            pub_std_w = 20.0 * np.sqrt(float(w) / float(ref_days))
            mean_rel_err_w = float(abs(emp_mean_w - pub_mean_w) / max(1e-9, pub_mean_w))
            std_rel_err_w = float(abs(emp_std_w - pub_std_w) / max(1e-9, pub_std_w))
            diag_w = {
                "population_n": n,
                "window_days": w,
                "empirical_mean_user_volume": round(emp_mean_w, 4),
                "public_expected_mean_user_volume": round(pub_mean_w, 4),
                "mean_relative_model_error": round(mean_rel_err_w, 4),
                "empirical_std_user_volume": round(emp_std_w, 4),
                "public_expected_std_user_volume": round(pub_std_w, 4),
                "std_relative_model_error": round(std_rel_err_w, 4),
            }
            diag_by_w[w] = diag_w
            temporal_records.append(diag_w)

        diag_30 = diag_by_w.get(ref_days, diag_by_w[30])
        # Section F: Verify temporal match across ALL candidate windows {7, 14, 30, 60, 90}
        is_temporal_mismatch = not check_all_windows_temporal_match(
            diag_by_w,
            mean_threshold=0.05,
            std_threshold=0.10,
        )

        # Workload and utility weights
        workload_cfg = app_config.privacy.planner.workloads
        wl_weights = {
            "full_9_cell": workload_cfg.full_9_cell_weight,
            "provider": workload_cfg.provider_weight,
            "task": workload_cfg.task_weight,
            "overall": workload_cfg.overall_weight,
        }
        w_cfg = app_config.privacy.planner.weights
        w_data = float(w_cfg.data if w_cfg.data is not None else 1.0)
        w_stale = float(w_cfg.staleness)
        w_win = float(w_cfg.window)

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

            # 2. Feasible candidates under budget cap
            feasible_evals = [rec for rec in plan_res.candidate_evaluations if rec["feasible"]]
            feasible_candidates = [rec["candidate"] for rec in feasible_evals]

            # Infeasible condition check
            if not feasible_candidates:
                all_projected = [rec.get("projected_epsilon", 999.0) for rec in plan_res.candidate_evaluations]
                min_projected = float(min(all_projected)) if all_projected else 999.0
                rec = {
                    "population_n": n,
                    "epsilon_cap": cap,
                    "status": "NO_FEASIBLE_POLICY",
                    "planner_candidate": "NULL",
                    "oracle_candidate": "NULL",
                    "oracle_true_utility": None,
                    "planner_true_utility": None,
                    "planner_rel_regret": None,
                    "empirical_mean_user_volume": diag_30["empirical_mean_user_volume"],
                    "public_expected_mean_user_volume": diag_30["public_expected_mean_user_volume"],
                    "mean_relative_model_error": diag_30["mean_relative_model_error"],
                    "empirical_std_user_volume": diag_30["empirical_std_user_volume"],
                    "public_expected_std_user_volume": diag_30["public_expected_std_user_volume"],
                    "std_relative_model_error": diag_30["std_relative_model_error"],
                    "relative_volume_model_error": diag_30["mean_relative_model_error"],
                    "minimum_projected_epsilon": round(min_projected, 4),
                    "feasible_candidate_count": 0,
                    "base_a_feasible": False,
                    "base_b_feasible": False,
                }
                records.append(rec)
                print(f"  N={n:5d}, Cap={cap:3.1f} | STATUS: NO_FEASIBLE_POLICY | Min Projected Eps={min_projected:.2f} | Feasible Count=0")
                continue

            if is_temporal_mismatch:
                rec = {
                    "population_n": n,
                    "epsilon_cap": cap,
                    "status": "TEMPORAL_MODEL_MISMATCH",
                    "planner_candidate": "NULL",
                    "oracle_candidate": "NULL",
                    "oracle_true_utility": None,
                    "planner_true_utility": None,
                    "planner_rel_regret": None,
                    "empirical_mean_user_volume": diag_30["empirical_mean_user_volume"],
                    "public_expected_mean_user_volume": diag_30["public_expected_mean_user_volume"],
                    "mean_relative_model_error": diag_30["mean_relative_model_error"],
                    "empirical_std_user_volume": diag_30["empirical_std_user_volume"],
                    "public_expected_std_user_volume": diag_30["public_expected_std_user_volume"],
                    "std_relative_model_error": diag_30["std_relative_model_error"],
                    "relative_volume_model_error": diag_30["mean_relative_model_error"],
                    "minimum_projected_epsilon": 0.0,
                    "feasible_candidate_count": len(feasible_candidates),
                    "base_a_feasible": False,
                    "base_b_feasible": False,
                }
                records.append(rec)
                print(f"  N={n:5d}, Cap={cap:3.1f} | STATUS: TEMPORAL_MODEL_MISMATCH (mean err={diag_30['mean_relative_model_error']*100:.1f}%, std err={diag_30['std_relative_model_error']*100:.1f}%)")
                continue

            # Compute true empirical utility for all feasible policies
            oracle_scores: Dict[ReleasePolicyCandidate, float] = {}
            oracle_stderr: Dict[ReleasePolicyCandidate, float] = {}
            oracle_cis: Dict[ReleasePolicyCandidate, Tuple[float, float]] = {}

            for cand in feasible_candidates:
                umat_w = umatrices_by_window.get(cand.window_days, umatrices_by_window[ref_days])
                score, stderr, ci = compute_true_candidate_utility(
                    candidate=cand,
                    umatrix=umat_w,
                    cell_names=cell_names,
                    reference_days=ref_days,
                    n_trials=n_trials,
                    seed=seed,
                    workload_weights=wl_weights,
                    weights_data=w_data,
                    weights_stale=w_stale,
                    weights_win=w_win,
                )
                oracle_scores[cand] = score
                oracle_stderr[cand] = stderr
                oracle_cis[cand] = ci

            # True Oracle is argmin of true empirical utility
            oracle_cand = min(oracle_scores, key=oracle_scores.get)
            oracle_true_util = oracle_scores[oracle_cand]
            oracle_err = oracle_stderr[oracle_cand]
            oracle_ci = oracle_cis[oracle_cand]

            planner_cand = plan_res.policy
            umat_plan_w = umatrices_by_window.get(planner_cand.window_days, umatrices_by_window[ref_days])
            planner_true_util, planner_err, planner_ci = compute_true_candidate_utility(
                candidate=planner_cand,
                umatrix=umat_plan_w,
                cell_names=cell_names,
                reference_days=ref_days,
                n_trials=n_trials,
                seed=seed,
                workload_weights=wl_weights,
                weights_data=w_data,
                weights_stale=w_stale,
                weights_win=w_win,
            )

            abs_reg, rel_reg = compute_regret(planner_true_util, oracle_true_util)

            # Evaluate Fixed Baselines
            base_a = ReleasePolicyCandidate(contribution_bound=20.0, granularity="full_9_cell", window_days=30, frequency_days=7)
            base_b = ReleasePolicyCandidate(contribution_bound=40.0, granularity="provider_3_cell", window_days=30, frequency_days=14)

            eval_dict = {rec["candidate"]: rec for rec in plan_res.candidate_evaluations}
            base_a_rec = eval_dict.get(base_a, {})
            base_b_rec = eval_dict.get(base_b, {})
            base_a_feas = bool(base_a_rec.get("feasible", False))
            base_b_feas = bool(base_b_rec.get("feasible", False))

            rec = {
                "population_n": n,
                "epsilon_cap": cap,
                "status": "SUCCESS",
                "planner_candidate": f"C{planner_cand.contribution_bound:.0f}_G{planner_cand.granularity}_W{planner_cand.window_days}_F{planner_cand.frequency_days}",
                "oracle_candidate": f"C{oracle_cand.contribution_bound:.0f}_G{oracle_cand.granularity}_W{oracle_cand.window_days}_F{oracle_cand.frequency_days}",
                "oracle_true_utility": round(oracle_true_util, 4),
                "oracle_stderr": round(oracle_err, 4),
                "oracle_ci_95": f"[{oracle_ci[0]:.4f}, {oracle_ci[1]:.4f}]",
                "planner_true_utility": round(planner_true_util, 4),
                "planner_stderr": round(planner_err, 4),
                "planner_ci_95": f"[{planner_ci[0]:.4f}, {planner_ci[1]:.4f}]",
                "planner_abs_regret": round(abs_reg, 4),
                "planner_rel_regret": round(rel_reg, 4),
                "empirical_mean_user_volume": diag_30["empirical_mean_user_volume"],
                "public_expected_mean_user_volume": diag_30["public_expected_mean_user_volume"],
                "mean_relative_model_error": diag_30["mean_relative_model_error"],
                "empirical_std_user_volume": diag_30["empirical_std_user_volume"],
                "public_expected_std_user_volume": diag_30["public_expected_std_user_volume"],
                "std_relative_model_error": diag_30["std_relative_model_error"],
                "relative_volume_model_error": diag_30["mean_relative_model_error"],
                "feasible_candidate_count": len(feasible_candidates),
                "base_a_feasible": base_a_feas,
                "base_b_feasible": base_b_feas,
            }
            records.append(rec)
            print(
                f"  N={n:5d}, Cap={cap:3.1f} | Oracle: {rec['oracle_candidate']} (TrueUtil={oracle_true_util:.4f}) | "
                f"Planner: {rec['planner_candidate']} (TrueUtil={planner_true_util:.4f}, Regret={rel_reg*100:4.1f}%) | "
                f"Feasible Count={len(feasible_candidates)} | BaseA Feas={base_a_feas} | BaseB Feas={base_b_feas}"
            )

    results_df = pd.DataFrame(records)
    save_summary_table(results_df, str(tables_dir / "exp13_joint_planner.csv"))

    temporal_df = pd.DataFrame(temporal_records)
    save_summary_table(temporal_df, str(tables_dir / "exp13_temporal_diagnostics.csv"))

    # Plot True Utility comparison for Cap=4.0
    sub_df = results_df[(results_df["epsilon_cap"] == 4.0) & (results_df["status"] == "SUCCESS")]
    if not sub_df.empty:
        plot_single_line_chart(
            x=sub_df["population_n"].tolist(),
            y_series={
                "True Empirical Oracle Utility": sub_df["oracle_true_utility"].tolist(),
                "Static Public Planner Utility": sub_df["planner_true_utility"].tolist(),
            },
            x_label="Population (N)",
            y_label="True Normalized Empirical Loss",
            title="Joint Release Policy Planner: Public Model vs True Empirical Oracle (Cap=4.0)",
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
            n_trials=20,
            output_dir=out_root,
        )
    else:
        run_joint_planner_experiment(
            populations=(500, 1000, 5000),
            epsilon_caps=(2.0, 4.0),
            n_trials=200,
            output_dir=out_root,
        )


if __name__ == "__main__":
    main()
