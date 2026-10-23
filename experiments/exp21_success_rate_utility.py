"""Exp21: Success Rate DP Utility Evaluation (v0.6).

Evaluates the differential privacy utility of the approved Success Rate profile
(joint Gaussian release with shared user-level clipping and DP-safe ratio post-processing)
against a valid DP baseline (separately budget-split numerator and denominator releases).

Measures across varying:
  - clipping bound C_rate
  - privacy budget epsilon
  - population size N

Metrics:
  - MAE (vs raw exact)
  - RMSE (vs raw exact)
  - Clipping bias (clipped exact vs raw exact)
  - DP error (DP estimate vs clipped exact)
  - Suppression rate (fraction of cells below min_noisy_denominator)
"""

from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Dict, List, Sequence
import numpy as np
import pandas as pd

from private_llm_analytics.aggregation.user_matrix import build_user_matrix
from private_llm_analytics.evaluation.report import plot_single_line_chart
from private_llm_analytics.mechanisms.gaussian import GaussianMechanism
from private_llm_analytics.simulator.generator import SyntheticDataGenerator
from private_llm_analytics.simulator.scenario import SimulationScenario
from private_llm_analytics.statistics.success_rate import SuccessRateProfile


def run_budget_split_baseline(
    user_matrix,
    clip_norm: float,
    epsilon: float,
    delta: float,
    min_noisy_denominator: float = 1.0,
    rng: np.random.Generator = None,
) -> Dict[str, Any]:
    """Valid DP baseline: Split total budget (epsilon, delta) evenly into (eps/2, delta/2)

    for separate numerator and denominator Gaussian releases, each with L2 sensitivity C_rate.
    """
    profile = SuccessRateProfile(clip_norm=clip_norm, min_noisy_denominator=min_noisy_denominator)
    joint_contrib = profile.build_user_contributions(user_matrix)
    s_clip = np.sum(joint_contrib[:, :9], axis=0)
    n_clip = np.sum(joint_contrib[:, 9:18], axis=0)

    # Budget split: eps/2, delta/2
    eps_split = epsilon / 2.0
    delta_split = delta / 2.0
    sens_split = float(clip_norm)

    mech_num = GaussianMechanism(epsilon=eps_split, delta=delta_split, sensitivity=sens_split, method="analytic")
    mech_den = GaussianMechanism(epsilon=eps_split, delta=delta_split, sensitivity=sens_split, method="analytic")

    noisy_s = mech_num.add_noise(s_clip, rng=rng)
    noisy_n = mech_den.add_noise(n_clip, rng=rng)

    # DP-safe post-processing
    total_dp = np.maximum(0.0, noisy_n)
    success_dp = np.clip(noisy_s, 0.0, total_dp)

    rates = {}
    suppressed = 0
    for i, cell in enumerate(user_matrix.cell_names):
        t_val = float(total_dp[i])
        s_val = float(success_dp[i])
        if t_val < min_noisy_denominator:
            rates[cell] = None
            suppressed += 1
        else:
            rates[cell] = float(np.clip(s_val / t_val, 0.0, 1.0))

    return {
        "rates": rates,
        "suppression_rate": suppressed / len(user_matrix.cell_names),
        "total_dp": total_dp,
        "success_dp": success_dp,
    }


def run_exp21_utility_evaluation(
    n_users_list: Sequence[int] = (1000,),
    c_candidates: Sequence[float] = (10.0, 20.0, 40.0),
    epsilon_candidates: Sequence[float] = (0.5, 1.0, 2.0),
    delta: float = 1e-6,
    n_trials: int = 40,
    seed: int = 202610,
    output_dir: str = "results",
) -> pd.DataFrame:
    print("\n=======================================================")
    print("Running Exp 21: Success Rate DP Utility Evaluation")
    print(f"Populations: {n_users_list}")
    print(f"C_rate candidates: {c_candidates}")
    print(f"Epsilon candidates: {epsilon_candidates}")
    print(f"Trials per setting: {n_trials}, Delta: {delta}")
    print("=======================================================")

    records: List[Dict[str, Any]] = []

    for n_users in n_users_list:
        scenario = SimulationScenario(n_users=n_users, random_seed=seed)
        gen = SyntheticDataGenerator(scenario)
        df = gen.generate(seed=seed, include_success=True, success_rate=0.82)
        umatrix = build_user_matrix(df, public_population_size=n_users)

        for c_rate in c_candidates:
            profile = SuccessRateProfile(clip_norm=c_rate, min_noisy_denominator=1.0)
            s_raw, n_raw, r_raw = profile.compute_raw_exact(umatrix)
            s_clip, n_clip, r_clip = profile.compute_clipped_exact(umatrix)

            # Clipping bias is deterministic given data and C_rate
            clipping_bias = float(np.mean(np.abs(r_clip - r_raw)))

            for eps in epsilon_candidates:
                # 1. Evaluate Approved Joint Mechanism
                plan = profile.build_mechanism_plan(epsilon=eps, delta=delta)
                joint_contrib = profile.build_user_contributions(umatrix)
                joint_exact = np.sum(joint_contrib, axis=0)

                joint_maes = []
                joint_rmses = []
                joint_dp_errors = []
                joint_suppressions = []

                split_maes = []
                split_rmses = []
                split_dp_errors = []
                split_suppressions = []

                for trial in range(n_trials):
                    trial_seed = seed + trial * 1000 + int(eps * 100) + int(c_rate)
                    rng = np.random.default_rng(trial_seed)

                    # A. Joint release
                    noisy_joint = profile.execute_mechanism(joint_exact, plan, rng=rng)
                    post_joint = profile.post_process(noisy_joint, umatrix)

                    r_joint = np.array([
                        post_joint["success_rates"][cell] if post_joint["success_rates"][cell] is not None else np.nan
                        for cell in umatrix.cell_names
                    ])
                    valid_j = ~np.isnan(r_joint)
                    if valid_j.sum() > 0:
                        joint_maes.append(float(np.mean(np.abs(r_joint[valid_j] - r_raw[valid_j]))))
                        joint_rmses.append(float(np.sqrt(np.mean((r_joint[valid_j] - r_raw[valid_j]) ** 2))))
                        joint_dp_errors.append(float(np.mean(np.abs(r_joint[valid_j] - r_clip[valid_j]))))
                    joint_suppressions.append(len(post_joint["suppressed_cells"]) / len(umatrix.cell_names))

                    # B. Split release
                    rng_split = np.random.default_rng(trial_seed + 500)
                    post_split = run_budget_split_baseline(
                        umatrix,
                        clip_norm=c_rate,
                        epsilon=eps,
                        delta=delta,
                        min_noisy_denominator=1.0,
                        rng=rng_split,
                    )
                    r_split = np.array([
                        post_split["rates"][cell] if post_split["rates"][cell] is not None else np.nan
                        for cell in umatrix.cell_names
                    ])
                    valid_s = ~np.isnan(r_split)
                    if valid_s.sum() > 0:
                        split_maes.append(float(np.mean(np.abs(r_split[valid_s] - r_raw[valid_s]))))
                        split_rmses.append(float(np.sqrt(np.mean((r_split[valid_s] - r_raw[valid_s]) ** 2))))
                        split_dp_errors.append(float(np.mean(np.abs(r_split[valid_s] - r_clip[valid_s]))))
                    split_suppressions.append(post_split["suppression_rate"])

                records.append({
                    "n_users": n_users,
                    "c_rate": c_rate,
                    "epsilon": eps,
                    "mechanism": "approved_joint_gaussian",
                    "sensitivity": profile.sensitivity(),
                    "noise_scale": plan.noise_scale,
                    "clipping_bias": clipping_bias,
                    "mae": float(np.mean(joint_maes)) if joint_maes else float("nan"),
                    "rmse": float(np.mean(joint_rmses)) if joint_rmses else float("nan"),
                    "dp_error": float(np.mean(joint_dp_errors)) if joint_dp_errors else float("nan"),
                    "suppression_rate": float(np.mean(joint_suppressions)),
                })

                records.append({
                    "n_users": n_users,
                    "c_rate": c_rate,
                    "epsilon": eps,
                    "mechanism": "budget_split_baseline",
                    "sensitivity": float(c_rate),
                    "noise_scale": float("nan"),
                    "clipping_bias": clipping_bias,
                    "mae": float(np.mean(split_maes)) if split_maes else float("nan"),
                    "rmse": float(np.mean(split_rmses)) if split_rmses else float("nan"),
                    "dp_error": float(np.mean(split_dp_errors)) if split_dp_errors else float("nan"),
                    "suppression_rate": float(np.mean(split_suppressions)),
                })

    df_results = pd.DataFrame(records)

    out_base = Path(output_dir)
    out_base.mkdir(parents=True, exist_ok=True)
    csv_path = out_base / "exp21_success_rate_utility.csv"
    json_path = out_base / "exp21_success_rate_utility.json"

    df_results.to_csv(csv_path, index=False)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)

    # Generate comparative plot for C=20 across epsilons
    try:
        sub_c20 = df_results[(df_results["c_rate"] == 20.0) & (df_results["n_users"] == 1000)]
        eps_vals = sorted(sub_c20["epsilon"].unique())
        joint_maes = [
            float(sub_c20[(sub_c20["epsilon"] == e) & (sub_c20["mechanism"] == "approved_joint_gaussian")]["mae"].iloc[0])
            for e in eps_vals
        ]
        split_maes = [
            float(sub_c20[(sub_c20["epsilon"] == e) & (sub_c20["mechanism"] == "budget_split_baseline")]["mae"].iloc[0])
            for e in eps_vals
        ]

        fig_path = out_base / "figures" / "exp21_success_rate_utility.png"
        plot_single_line_chart(
            x=list(eps_vals),
            y_series={
                "Approved Joint Gaussian": joint_maes,
                "Budget-Split Baseline (eps/2, delta/2)": split_maes,
            },
            x_label="Privacy Budget (epsilon)",
            y_label="Success Rate MAE",
            title="Success Rate Error vs Epsilon (C_rate = 20, N = 1000)",
            output_path=fig_path,
        )
        print(f"Figure saved to: {fig_path}")
    except Exception as e:
        print(f"Note: plotting skipped or encountered: {e}")

    print(f"Exp21 PASS: Artifacts written to {csv_path} and {json_path}")
    return df_results


if __name__ == "__main__":
    run_exp21_utility_evaluation()
