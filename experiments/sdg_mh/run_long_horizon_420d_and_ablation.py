"""420-Day Cross-Period True E2E & Clipping Sensitivity (C_r) Ablation Benchmark.

Preserves frozen 90-day baseline (results/calendar_planning/true_e2e_benchmark_results.csv) untouched.
Writes all outputs cleanly into results/sdg_mh/long_horizon_420d/:
  - long_horizon_420d_summary.csv
  - long_horizon_420d_cross_period_ledger.csv
  - cr_clipping_ablation_420d.csv
  - figures/fig1_420d_cross_period_ledger_trajectory.png
  - figures/fig2_cr_clipping_pareto_ablation.png
  - figures/fig3_420d_distinct_mape_comparison.png
  - LONG_HORIZON_420D_REPORT.md

Supports:
  --smoke : Fast verification (200 users, 420 days, 1 seed, 2 MC trials)
  --full  : Official benchmark (2,000 users, 420 days, 3 data seeds [42, 1042, 2042], 10 MC trials each)
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import datetime
import math
from pathlib import Path
import sys
import tempfile
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parents[2] / "src"))

from private_llm_analytics.ingestion.schema import (
    COL_SUCCESS,
    COL_USER_ID,
)
from private_llm_analytics.sdg_mh.calendar_accounting import (
    JOINT_PERIOD,
    ORIGINAL_PLANNER_CURVE,
    WEEK1_V2_CURVE,
    PrivacyCommitmentCurve,
)
from private_llm_analytics.sdg_mh.calendar_job import CalendarJob, DistinctCalendarStore
from private_llm_analytics.sdg_mh.joint_planner import (
    JointCalendarPlanner,
    PlannerConfig,
    PlanningResult,
)
from private_llm_analytics.sdg_mh.ledger import UnifiedPrivacyLedger
from private_llm_analytics.sdg_mh.query_service import ReadOnlySealedQueryService
from private_llm_analytics.sdg_mh.rolling_state import RollingDistinctState
from private_llm_analytics.sdg_mh.seal_job import SealJob
from private_llm_analytics.sdg_mh.sealed_store import SealedStore
from private_llm_analytics.sdg_mh.synthetic_generator import SyntheticTelemetryGenerator


OUTPUT_DIR = Path("results/sdg_mh/long_horizon_420d")
FIGURES_DIR = OUTPUT_DIR / "figures"

CURVE_REGISTRY: Dict[str, PrivacyCommitmentCurve] = {
    "Week1_v2": WEEK1_V2_CURVE,
    "Original": ORIGINAL_PLANNER_CURVE,
}

# t-critical values for 95% two-sided CI for df = n - 1 (fallback 1.96 for large n)
T_CRIT_95 = {
    1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571,
    6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228,
    11: 2.201, 14: 2.145, 19: 2.093, 29: 2.045,
}


def compute_summary_stats(values: Sequence[float]) -> Tuple[float, float, float]:
    """Return (mean, sample_std, 95% CI half-width)."""
    arr = np.asarray(values, dtype=np.float64)
    n = len(arr)
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    if not np.all(np.isfinite(arr)):
        return float(arr[0]), 0.0, 0.0
    mean_v = float(np.mean(arr))
    if n == 1:
        return mean_v, 0.0, 0.0
    std_v = float(np.std(arr, ddof=1))
    t_crit = T_CRIT_95.get(n - 1, 1.96)
    ci95 = float(t_crit * std_v / math.sqrt(n))
    return mean_v, std_v, ci95


def make_valid_date_str(day: int, start_year: int = 2025) -> str:
    """Convert 1-indexed day into valid ISO-8601 date string."""
    d = datetime.date(start_year, 1, 1) + datetime.timedelta(days=day - 1)
    iso = d.isoformat()
    # Explicit roundtrip validation
    datetime.date.fromisoformat(iso)
    return iso


def _execute_single_seed(
    data_seed: int,
    n_users: int,
    n_days: int,
    n_mc_trials: int,
) -> Dict[str, Any]:
    """Generate 1 longitudinal telemetry seed and run both E2E 420d and Cr ablation."""
    t_start = time.time()
    gen = SyntheticTelemetryGenerator(
        n_initial_users=n_users,
        n_years=n_days / 365.0,
        seed=data_seed,
    )
    services_list = gen.services
    tasks_list = gen.tasks
    k_tasks_per_user = 3
    C_f_default = 10.0

    daily_dfs: Dict[int, pd.DataFrame] = {}
    daily_raw_req = np.zeros(n_days + 1, dtype=np.float64)
    daily_raw_fail = np.zeros(n_days + 1, dtype=np.float64)
    # Precompute daily user request counts array for fast Cr ablation clipping bias ground truth
    daily_user_req_arrays: Dict[int, np.ndarray] = {}
    daily_user_fail_arrays: Dict[int, np.ndarray] = {}

    with tempfile.TemporaryDirectory() as tmp_dir:
        # Keep w_max >= n_days + 10 so multi-trial replay can query exact [start_day, current_day]
        # across the entire 420-day stream without re-running telemetry generation per trial.
        rolling_state = RollingDistinctState(storage_dir=Path(tmp_dir), w_max=n_days + 10)
        for day, df_day in gen.stream_daily_telemetry(start_day=1, end_day=n_days):
            daily_dfs[day] = df_day
            if not df_day.empty:
                req_counts = df_day[COL_USER_ID].value_counts().to_numpy(dtype=np.float64)
                daily_user_req_arrays[day] = req_counts
                daily_raw_req[day] = float(np.sum(req_counts))

                fail_df = df_day[~df_day[COL_SUCCESS]]
                if not fail_df.empty:
                    fail_counts = fail_df[COL_USER_ID].value_counts().to_numpy(dtype=np.float64)
                    daily_user_fail_arrays[day] = fail_counts
                    daily_raw_fail[day] = float(np.sum(fail_counts))
                else:
                    daily_user_fail_arrays[day] = np.zeros(0, dtype=np.float64)
            else:
                daily_user_req_arrays[day] = np.zeros(0, dtype=np.float64)
                daily_user_fail_arrays[day] = np.zeros(0, dtype=np.float64)

            rolling_state.append_day(day, df_day, services_list, tasks_list, persist_disk=False)

        t_ingest = time.time() - t_start

        # Ground-truth distinct window cache for all windows reached across candidate phases
        gt_cache: Dict[Tuple[int, int], np.ndarray] = {}

        def get_gt_service_counts(w: int, end_d: int) -> np.ndarray:
            key = (end_d - w + 1, end_d)
            if key not in gt_cache:
                s_cnts, _ = rolling_state.compute_window_histogram(
                    start_day=key[0],
                    end_day=key[1],
                    services_list=services_list,
                    tasks_list=tasks_list,
                    k_tasks_per_user=k_tasks_per_user,
                )
                gt_cache[key] = s_cnts.astype(np.float64)
            return gt_cache[key]

        # Experiment 1 configurations: 420-day cross-period comparison
        e2e_configs = [
            ("Default", "Week1_v2", (0.40, 0.25, 0.25, 0.10), "A_Baseline"),
            ("Default", "Week1_v2", (0.40, 0.25, 0.25, 0.10), "D_JointPlanning"),
            ("DistinctHeavy", "Week1_v2", (0.10, 0.40, 0.40, 0.10), "B_BudgetOnly"),
            ("DistinctHeavy", "Week1_v2", (0.10, 0.40, 0.40, 0.10), "D_JointPlanning"),
            ("DistinctHeavy", "Original", (0.10, 0.40, 0.40, 0.10), "B_BudgetOnly"),
            ("DistinctHeavy", "Original", (0.10, 0.40, 0.40, 0.10), "D_JointPlanning"),
        ]

        e2e_trial_rows: List[Dict[str, Any]] = []
        ledger_traj_rows: List[Dict[str, Any]] = []

        for wl_name, curve_name, wl_weights, grp_key in e2e_configs:
            curve = CURVE_REGISTRY[curve_name]
            planner = JointCalendarPlanner(PlannerConfig(commitment=curve, workload_weights=wl_weights))
            plans = planner.plan_all_groups()
            plan: PlanningResult = plans[grp_key]

            assert plan.is_certified_publishable is True, f"Plan {wl_name}/{curve_name}/{grp_key} failed certification"
            assert plan.min_slack >= 0.0, f"Negative slack {plan.min_slack}"

            C_r = 50.0
            C_f = C_f_default
            eta = (C_r ** 2) / (C_r ** 2 + C_f ** 2)
            d_val = plan.daily_d
            phi7, phi30, phi90 = plan.phases

            daily_clipped_req = np.array(
                [0.0] + [float(np.minimum(daily_user_req_arrays[d], C_r).sum()) for d in range(1, n_days + 1)]
            )
            daily_clipped_fail = np.array(
                [0.0] + [float(np.minimum(daily_user_fail_arrays[d], C_f).sum()) for d in range(1, n_days + 1)]
            )
            req_clip_bias = float(np.mean(daily_clipped_req[1:] - daily_raw_req[1:]))
            fail_clip_bias = float(np.mean(daily_clipped_fail[1:] - daily_raw_fail[1:]))

            for trial in range(n_mc_trials):
                trial_seed = 100000 * data_seed + 1000 * (e2e_configs.index((wl_name, curve_name, wl_weights, grp_key)) + 1) + trial

                # Trial 0 executes full day-by-day UnifiedPrivacyLedger admission across all 420 days
                # and logs checkpoints at day 210, 365, and 420.
                # Trials 1..N-1 verify initial 420-day strict_zero_tol admission via commit_plan_to_ledger
                # and check final horizon gate while executing the real SealJob + CalendarJob + QueryService pipeline.
                full_day_by_day_gate = (trial == 0)
                trial_ledger = UnifiedPrivacyLedger(
                    delta=1e-6,
                    allow_unconfigured_policy=(not full_day_by_day_gate),
                )
                if full_day_by_day_gate:
                    trial_ledger.set_active_plan(plan=plan, commitment=curve)
                else:
                    # Also explicitly verify 420-day strict zero-tol ledger batch admission on trial 1
                    if trial == 1:
                        chk_ledger = UnifiedPrivacyLedger(delta=1e-6)
                        ok_comm, msg_comm, _ = planner.commit_plan_to_ledger(
                            plan, chk_ledger, start_day=1, simulated_days=n_days
                        )
                        assert ok_comm, f"420d commit_plan_to_ledger failed: {msg_comm}"

                sealed_store = SealedStore()
                cal_store = DistinctCalendarStore()
                seal_job = SealJob.from_budget(
                    sealed_store=sealed_store,
                    ledger=trial_ledger,
                    daily_d=d_val,
                    eta=eta,
                    C_r=C_r,
                    C_f=C_f,
                )
                cal_job = CalendarJob.from_planner_result(
                    calendar_store=cal_store,
                    ledger=trial_ledger,
                    plan=plan,
                    services_list=services_list,
                    tasks_list=tasks_list,
                    k_tasks_per_user=k_tasks_per_user,
                    rolling_state=rolling_state,
                )
                query_service = ReadOnlySealedQueryService(
                    sealed_store=sealed_store,
                    calendar_store=cal_store,
                    public_population_size=n_users,
                )

                checkpoint_rhos: Dict[int, np.ndarray] = {}
                for day in range(1, n_days + 1):
                    d_str = make_valid_date_str(day)
                    seal_job.seal_day(
                        day_index=day,
                        date_str=d_str,
                        df_day=daily_dfs[day],
                        seed=trial_seed + day,
                    )
                    cal_job.execute_day(
                        current_day=day,
                        rolling_state=rolling_state,
                        seed=trial_seed + 10000 + day,
                    )
                    if full_day_by_day_gate and day in (210, 365, 420):
                        checkpoint_rhos[day] = trial_ledger.compute_full_horizon_profile(t_span=day)

                # On trial 0, record full cross-period ledger checkpoints & verify against 575-day theorem
                if full_day_by_day_gate:
                    cert_575 = plan.schedule.verify_periodicity_and_long_term_security(curve, H_0=365, P=JOINT_PERIOD)
                    assert cert_575["is_infinite_horizon_secure"] is True

                    env_575, _ = plan.schedule.evaluate_full_envelope(max_h=575, P=JOINT_PERIOD)
                    for ck_day in (210, 365, 420):
                        prof = checkpoint_rhos[ck_day]
                        for h_chk in (1, 7, 30, 90, 210, 365, 420):
                            if h_chk <= ck_day:
                                obs_rho = float(prof[h_chk])
                                rho_limit = float(curve.rho_bar(h_chk))
                                theo_lmax = float(env_575[h_chk])
                                assert obs_rho <= theo_lmax + 1e-12, (
                                    f"Observed rho({h_chk})={obs_rho} > L_max({h_chk})={theo_lmax}"
                                )
                                assert obs_rho <= rho_limit + 1e-12, (
                                    f"Observed rho({h_chk})={obs_rho} > rho_bar({h_chk})={rho_limit}"
                                )
                                ledger_traj_rows.append({
                                    "data_seed": data_seed,
                                    "workload": wl_name,
                                    "commitment_curve": curve_name,
                                    "group": grp_key,
                                    "checkpoint_day": ck_day,
                                    "horizon_h": h_chk,
                                    "observed_ledger_max_rho": obs_rho,
                                    "theoretical_L_max_575": theo_lmax,
                                    "commitment_rho_bar": rho_limit,
                                    "ledger_headroom": rho_limit - obs_rho,
                                    "theorem_envelope_slack": rho_limit - theo_lmax,
                                    "cert_575_min_slack": float(cert_575["min_slack_up_to_575"]),
                                    "cert_575_secure": bool(cert_575["is_infinite_horizon_secure"]),
                                })

                # Reconstruct statistical accuracy across all 420 days
                req_raw_errs, req_post_errs, req_tot_errs = [], [], []
                fail_raw_errs, fail_post_errs, fail_tot_errs = [], [], []

                for day in range(1, n_days + 1):
                    s_rec = sealed_store.get_day(day)
                    add_q = query_service.query_additive_window(start_day=day, end_day=day)
                    y_r = float(s_rec.y_requests)
                    y_f = float(s_rec.y_failures)
                    r_hat = float(add_q["requests_estimate"])
                    f_hat = float(add_q["failures_estimate"])

                    req_raw_errs.append(y_r - daily_clipped_req[day])
                    req_post_errs.append(r_hat - daily_clipped_req[day])
                    req_tot_errs.append(r_hat - daily_raw_req[day])

                    fail_raw_errs.append(y_f - daily_clipped_fail[day])
                    fail_post_errs.append(f_hat - daily_clipped_fail[day])
                    fail_tot_errs.append(f_hat - daily_raw_fail[day])

                dist_stats: Dict[int, Dict[str, Any]] = {}
                for w, s_step, phi_w in ((7, 7, phi7), (30, 7, phi30), (90, 30, phi90)):
                    raw_e, tot_e, mapes = [], [], []
                    win_cnt = 0
                    for day in range(w, n_days + 1):
                        if day % s_step == phi_w % s_step:
                            win_cnt += 1
                            gt_s = get_gt_service_counts(w, day)
                            cal_rec = cal_store.get_release(window_length=w, end_day=day)
                            q_dist = query_service.query_distinct_prevalence(window_length=w, end_day=day)
                            for s_idx, s_name in enumerate(services_list):
                                y_val = float(cal_rec.noisy_services[s_idx])
                                true_v = float(gt_s[s_idx])
                                post_v = float(q_dist["services"].get(s_name, 0.0))
                                raw_e.append(y_val - true_v)
                                tot_e.append(post_v - true_v)
                                if true_v > 0:
                                    mapes.append(abs(post_v - true_v) / true_v)
                    dist_stats[w] = {
                        "windows": win_cnt,
                        "raw_rmse": float(np.sqrt(np.mean(np.square(raw_e)))),
                        "total_rmse": float(np.sqrt(np.mean(np.square(tot_e)))),
                        "mape_pct": float(100.0 * np.mean(mapes)),
                    }

                e2e_trial_rows.append({
                    "data_seed": data_seed,
                    "mc_trial": trial,
                    "workload": wl_name,
                    "commitment_curve": curve_name,
                    "group": grp_key,
                    "phases": str(plan.phases),
                    "daily_d": d_val,
                    "planner_abstract_objective": float(plan.weighted_workload_error),
                    "dual_lower_bound": float(plan.dual_lower_bound),
                    "optimality_rel_gap": float(plan.optimality_rel_gap),
                    "optimality_tier": plan.optimality_certification_tier,
                    "min_slack_575": float(plan.min_slack),
                    "requests_clipping_bias": req_clip_bias,
                    "requests_raw_noise_rmse": float(np.sqrt(np.mean(np.square(req_raw_errs)))),
                    "requests_postproc_rmse": float(np.sqrt(np.mean(np.square(req_post_errs)))),
                    "requests_total_rmse": float(np.sqrt(np.mean(np.square(req_tot_errs)))),
                    "failures_clipping_bias": fail_clip_bias,
                    "failures_raw_noise_rmse": float(np.sqrt(np.mean(np.square(fail_raw_errs)))),
                    "failures_total_rmse": float(np.sqrt(np.mean(np.square(fail_tot_errs)))),
                    "dist_7d_windows": dist_stats[7]["windows"],
                    "dist_7d_raw_rmse": dist_stats[7]["raw_rmse"],
                    "dist_7d_total_rmse": dist_stats[7]["total_rmse"],
                    "dist_7d_mape_pct": dist_stats[7]["mape_pct"],
                    "dist_30d_windows": dist_stats[30]["windows"],
                    "dist_30d_raw_rmse": dist_stats[30]["raw_rmse"],
                    "dist_30d_total_rmse": dist_stats[30]["total_rmse"],
                    "dist_30d_mape_pct": dist_stats[30]["mape_pct"],
                    "dist_90d_windows": dist_stats[90]["windows"],
                    "dist_90d_raw_rmse": dist_stats[90]["raw_rmse"],
                    "dist_90d_total_rmse": dist_stats[90]["total_rmse"],
                    "dist_90d_mape_pct": dist_stats[90]["mape_pct"],
                })

        # Experiment 2: C_r in {30, 50, 100, 200} Clipping Ablation under identical Week1_v2 commitment & d
        base_planner = JointCalendarPlanner(PlannerConfig(commitment=WEEK1_V2_CURVE, workload_weights=(0.40, 0.25, 0.25, 0.10)))
        base_plan = base_planner.plan_group_d()
        ablation_d = base_plan.daily_d
        C_f = C_f_default

        ablation_rows: List[Dict[str, Any]] = []
        for cr_val in (30.0, 50.0, 100.0, 200.0):
            eta_cr = (cr_val ** 2) / (cr_val ** 2 + C_f ** 2)
            sigma_cr = math.sqrt((cr_val ** 2 + C_f ** 2) / (2.0 * ablation_d))

            daily_clipped_r = np.zeros(n_days + 1, dtype=np.float64)
            daily_clipped_f = np.zeros(n_days + 1, dtype=np.float64)
            tot_user_days = 0
            tot_clipped_user_days = 0
            for day in range(1, n_days + 1):
                u_r = daily_user_req_arrays[day]
                u_f = daily_user_fail_arrays[day]
                tot_user_days += len(u_r)
                tot_clipped_user_days += int(np.sum(u_r > cr_val))
                daily_clipped_r[day] = float(np.minimum(u_r, cr_val).sum())
                daily_clipped_f[day] = float(np.minimum(u_f, C_f).sum())

            clip_rate_pct = 100.0 * tot_clipped_user_days / max(1, tot_user_days)
            clip_bias_r = float(np.mean(daily_clipped_r[1:] - daily_raw_req[1:]))

            for trial in range(n_mc_trials):
                trial_seed = 500000 * data_seed + 1000 * int(cr_val) + trial
                use_strict_daily_ledger = (trial == 0)
                trial_ledger = UnifiedPrivacyLedger(
                    delta=1e-6,
                    allow_unconfigured_policy=(not use_strict_daily_ledger),
                )
                if use_strict_daily_ledger:
                    trial_ledger.set_active_plan(plan=base_plan, commitment=WEEK1_V2_CURVE)

                sealed_store = SealedStore()
                cal_store = DistinctCalendarStore()
                seal_job = SealJob.from_budget(
                    sealed_store=sealed_store,
                    ledger=trial_ledger,
                    daily_d=ablation_d,
                    eta=eta_cr,
                    C_r=cr_val,
                    C_f=C_f,
                )
                # Confirm exact zCDP invariant rho_r + rho_f == ablation_d and sigma == sigma_cr
                assert math.isclose(seal_job.rho_r + seal_job.rho_f, ablation_d, rel_tol=1e-12)
                assert math.isclose(seal_job.sigma_r, sigma_cr, rel_tol=1e-12)

                qs = ReadOnlySealedQueryService(sealed_store, cal_store, n_users)

                req_raw_e, req_post_e, req_tot_e, fail_tot_e = [], [], [], []
                for day in range(1, n_days + 1):
                    d_str = make_valid_date_str(day)
                    seal_job.seal_day(day, d_str, daily_dfs[day], seed=trial_seed + day)
                    s_rec = sealed_store.get_day(day)
                    add_q = qs.query_additive_window(start_day=day, end_day=day)
                    y_r = float(s_rec.y_requests)
                    r_hat = float(add_q["requests_estimate"])
                    f_hat = float(add_q["failures_estimate"])

                    req_raw_e.append(y_r - daily_clipped_r[day])
                    req_post_e.append(r_hat - daily_clipped_r[day])
                    req_tot_e.append(r_hat - daily_raw_req[day])
                    fail_tot_e.append(f_hat - daily_raw_fail[day])

                ablation_rows.append({
                    "data_seed": data_seed,
                    "mc_trial": trial,
                    "C_r": cr_val,
                    "C_f": C_f,
                    "daily_d": ablation_d,
                    "eta": eta_cr,
                    "rho_r": seal_job.rho_r,
                    "rho_f": seal_job.rho_f,
                    "sigma_r_analytical": sigma_cr,
                    "clipping_rate_pct": clip_rate_pct,
                    "requests_clipping_bias": clip_bias_r,
                    "requests_raw_noise_rmse": float(np.sqrt(np.mean(np.square(req_raw_e)))),
                    "requests_postproc_rmse": float(np.sqrt(np.mean(np.square(req_post_e)))),
                    "requests_total_rmse": float(np.sqrt(np.mean(np.square(req_tot_e)))),
                    "failures_total_rmse": float(np.sqrt(np.mean(np.square(fail_tot_e)))),
                })

    return {
        "data_seed": data_seed,
        "ingest_seconds": t_ingest,
        "total_seconds": time.time() - t_start,
        "e2e_rows": e2e_trial_rows,
        "ledger_traj_rows": ledger_traj_rows,
        "ablation_rows": ablation_rows,
    }


def aggregate_with_ci(df: pd.DataFrame, group_cols: List[str], metric_cols: List[str]) -> pd.DataFrame:
    """Aggregate mean, std, and 95% CI half-width across seed x MC runs."""
    agg_rows: List[Dict[str, Any]] = []
    for keys, sub in df.groupby(group_cols, sort=False):
        row: Dict[str, Any] = {}
        if not isinstance(keys, tuple):
            keys = (keys,)
        for c_name, val in zip(group_cols, keys):
            row[c_name] = val
        row["n_runs"] = len(sub)
        row["n_data_seeds"] = int(sub["data_seed"].nunique())
        for m in metric_cols:
            vals = sub[m].to_numpy(dtype=np.float64)
            mean_v, std_v, ci95 = compute_summary_stats(vals)
            row[f"{m}_mean"] = round(mean_v, 4)
            row[f"{m}_std"] = round(std_v, 4)
            row[f"{m}_ci95"] = round(ci95, 4)
        agg_rows.append(row)
    return pd.DataFrame(agg_rows)


def generate_figures(
    e2e_sum: pd.DataFrame,
    traj_df: pd.DataFrame,
    abl_sum: pd.DataFrame,
) -> None:
    """Render publication-quality PNG figures into figures/."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 10, "axes.grid": True, "grid.alpha": 0.3})

    # Fig 1: Cross-Period Observed Ledger rho(h) vs Theoretical Envelope vs Commitment (Day 420 snapshot)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    sub_420 = traj_df[traj_df["checkpoint_day"] == 420]

    # Left: Week1_v2 DistinctHeavy B vs D vs commitment
    ax = axes[0]
    for grp, color, marker in [("B_BudgetOnly", "#1f77b4", "o"), ("D_JointPlanning", "#d62728", "s")]:
        g_sub = sub_420[
            (sub_420["workload"] == "DistinctHeavy")
            & (sub_420["commitment_curve"] == "Week1_v2")
            & (sub_420["group"] == grp)
        ].groupby("horizon_h", as_index=False).mean(numeric_only=True)
        if not g_sub.empty:
            ax.plot(g_sub["horizon_h"], g_sub["observed_ledger_max_rho"], marker=marker, label=f"{grp} Ledger max rho(h)", color=color)
            ax.plot(g_sub["horizon_h"], g_sub["theoretical_L_max_575"], linestyle="--", alpha=0.6, color=color, label=f"{grp} 575d Theo L_max(h)")
    comm_sub = sub_420[
        (sub_420["workload"] == "DistinctHeavy")
        & (sub_420["commitment_curve"] == "Week1_v2")
        & (sub_420["group"] == "D_JointPlanning")
    ].groupby("horizon_h", as_index=False).mean(numeric_only=True)
    if not comm_sub.empty:
        ax.plot(comm_sub["horizon_h"], comm_sub["commitment_rho_bar"], color="black", linewidth=2, linestyle=":", label="Commitment rho_bar(h)")
    for xv, lbl in [(210, "P=210"), (365, "H0=365"), (420, "2P=420")]:
        ax.axvline(xv, color="gray", linestyle="-.", alpha=0.5)
        ax.text(xv + 4, 0.02, lbl, rotation=90, fontsize=8, color="#444444")
    ax.set_title("Week1_v2 DistinctHeavy: Ledger max rho(h) Across 420d")
    ax.set_xlabel("Horizon h (Days)")
    ax.set_ylabel("zCDP Cumulative Bound rho(h)")
    ax.legend(fontsize=8)

    # Right: Positive headroom / slack across checkpoints
    ax = axes[1]
    for curve_name, grp, color in [
        ("Week1_v2", "B_BudgetOnly", "#1f77b4"),
        ("Week1_v2", "D_JointPlanning", "#d62728"),
        ("Original", "B_BudgetOnly", "#2ca02c"),
        ("Original", "D_JointPlanning", "#9467bd"),
    ]:
        g_sub = sub_420[
            (sub_420["workload"] == "DistinctHeavy")
            & (sub_420["commitment_curve"] == curve_name)
            & (sub_420["group"] == grp)
        ].groupby("horizon_h", as_index=False).mean(numeric_only=True)
        if not g_sub.empty:
            ax.plot(g_sub["horizon_h"], g_sub["ledger_headroom"], marker="o", markersize=4, label=f"{curve_name} {grp}", color=color)
    ax.axhline(0.0, color="black", linestyle=":", linewidth=1.2)
    ax.axvline(210, color="gray", linestyle="-.", alpha=0.5)
    ax.axvline(365, color="gray", linestyle="-.", alpha=0.5)
    ax.axvline(420, color="gray", linestyle="-.", alpha=0.5)
    ax.set_title("Ledger Privacy Headroom (rho_bar(h) - obs rho(h)) >= 0")
    ax.set_xlabel("Horizon h (Days)")
    ax.set_ylabel("zCDP Headroom")
    ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "fig1_420d_cross_period_ledger_trajectory.png", dpi=180)
    plt.close()

    # Fig 2: Cr Clipping Bias vs Mechanism Noise Pareto Ablation
    fig, ax = plt.subplots(figsize=(8, 5))
    crs = abl_sum["C_r"].to_numpy()
    ax.errorbar(
        crs, np.abs(abl_sum["requests_clipping_bias_mean"]),
        yerr=abl_sum["requests_clipping_bias_ci95"],
        marker="o", linewidth=2, capsize=4, label="|Clipping Bias| (Daily Requests)"
    )
    ax.errorbar(
        crs, abl_sum["requests_raw_noise_rmse_mean"],
        yerr=abl_sum["requests_raw_noise_rmse_ci95"],
        marker="s", linewidth=2, capsize=4, label="DP Mechanism Noise RMSE"
    )
    ax.errorbar(
        crs, abl_sum["requests_total_rmse_mean"],
        yerr=abl_sum["requests_total_rmse_ci95"],
        marker="^", linewidth=2.5, capsize=4, color="#d62728", label="End-to-End Total RMSE"
    )
    ax.set_xticks(crs)
    ax.set_title("Clipping Threshold C_r Ablation under Fixed Week1_v2 Budget (420d, 95% CI)")
    ax.set_xlabel("Per-User Daily Request Clipping Bound C_r")
    ax.set_ylabel("Daily Requests Error Magnitude")
    ax.legend()
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "fig2_cr_clipping_pareto_ablation.png", dpi=180)
    plt.close()

    # Fig 3: 420-Day Distinct MAPE Comparison (Group B vs D)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    horizons = ["dist_7d_mape_pct", "dist_30d_mape_pct", "dist_90d_mape_pct"]
    h_labels = ["7d (60 wins)", "30d (56 wins)", "90d (12 wins)"]
    x = np.arange(len(horizons))
    width = 0.34

    for ax, curve_name in zip(axes, ["Week1_v2", "Original"]):
        sub_b = e2e_sum[(e2e_sum["workload"] == "DistinctHeavy") & (e2e_sum["commitment_curve"] == curve_name) & (e2e_sum["group"] == "B_BudgetOnly")]
        sub_d = e2e_sum[(e2e_sum["workload"] == "DistinctHeavy") & (e2e_sum["commitment_curve"] == curve_name) & (e2e_sum["group"] == "D_JointPlanning")]
        if not sub_b.empty and not sub_d.empty:
            b_means = [float(sub_b[f"{h}_mean"].iloc[0]) for h in horizons]
            b_cis = [float(sub_b[f"{h}_ci95"].iloc[0]) for h in horizons]
            d_means = [float(sub_d[f"{h}_mean"].iloc[0]) for h in horizons]
            d_cis = [float(sub_d[f"{h}_ci95"].iloc[0]) for h in horizons]
            ax.bar(x - width / 2, b_means, width, yerr=b_cis, capsize=4, label="Group B (BudgetOnly)", color="#1f77b4", alpha=0.85)
            ax.bar(x + width / 2, d_means, width, yerr=d_cis, capsize=4, label="Group D (JointPlanning)", color="#d62728", alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels(h_labels)
        ax.set_ylabel("Distinct Service MAPE (%)")
        ax.set_title(f"{curve_name} DistinctHeavy (420d, 95% CI)")
        ax.legend()

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "fig3_420d_distinct_mape_comparison.png", dpi=180)
    plt.close()


def generate_markdown_report(
    mode: str,
    n_users: int,
    n_days: int,
    data_seeds: List[int],
    n_mc_trials: int,
    elapsed_sec: float,
    e2e_sum: pd.DataFrame,
    traj_df: pd.DataFrame,
    abl_sum: pd.DataFrame,
) -> None:
    """Write reproducible Markdown experiment report cleanly separating raw errors from abstract objectives."""
    lines: List[str] = [
        "# SDG-MH 420-Day Cross-Period True E2E & $C_r$ Ablation Report",
        "",
        f"- **Run Mode**: `{mode}`",
        f"- **Population / Horizon**: `{n_users}` initial users, `{n_days}` days (`1..{n_days}` valid ISO-8601 calendar dates `2025-01-01` .. `{make_valid_date_str(n_days)}`)",
        f"- **Statistical Replication**: Data seeds `{data_seeds}` $\\times$ `{n_mc_trials}` Monte Carlo noise draws = `{len(data_seeds) * n_mc_trials}` runs per configuration",
        f"- **Total Wall Time**: `{elapsed_sec:.2f}s`",
        "",
        "## 1. Reproducible Commands",
        "```powershell",
        "# Smoke Test:",
        ".\\.venv\\Scripts\\python.exe experiments/sdg_mh/run_long_horizon_420d_and_ablation.py --smoke",
        "# Official Full Experiment:",
        ".\\.venv\\Scripts\\python.exe experiments/sdg_mh/run_long_horizon_420d_and_ablation.py --full",
        "```",
        "",
        "## 2. 420-Day True E2E Results (Mean $\\pm$ 95% CI)",
        "",
        "> **Metric Separation Note**: Optimizer Abstract Objective ($\\sum c_j / v_j$), Primal/Dual Optimality Gap, and Certification Tier characterize convex calendar planning; Requests/Failures RMSE and Distinct 7d/30d/90d MAPE reflect actual end-to-end executed pipeline noise and clipping.",
        "",
        "| Workload | Curve | Group | Phases | Abstract Obj ($f^*$) | Dual LB / Rel Gap | Tier | Req Total RMSE | Fail Total RMSE | 7d MAPE (60w) | 30d MAPE (56w) | 90d MAPE (12w) |",
        "|---|---|---|---|---:|---:|:---:|---:|---:|---:|---:|---:|",
    ]
    for _, r in e2e_sum.iterrows():
        lines.append(
            f"| {r['workload']} | {r['commitment_curve']} | {r['group']} | `{r['phases']}` | "
            f"{r['planner_abstract_objective_mean']:.2f} | {r['dual_lower_bound_mean']:.2f} (`{r['optimality_rel_gap_mean']:.2e}`) | "
            f"`{r['optimality_tier']}` | "
            f"{r['requests_total_rmse_mean']:.2f} $\\pm$ {r['requests_total_rmse_ci95']:.2f} | "
            f"{r['failures_total_rmse_mean']:.2f} $\\pm$ {r['failures_total_rmse_ci95']:.2f} | "
            f"{r['dist_7d_mape_pct_mean']:.2f}% $\\pm$ {r['dist_7d_mape_pct_ci95']:.2f}% | "
            f"{r['dist_30d_mape_pct_mean']:.2f}% $\\pm$ {r['dist_30d_mape_pct_ci95']:.2f}% | "
            f"{r['dist_90d_mape_pct_mean']:.2f}% $\\pm$ {r['dist_90d_mape_pct_ci95']:.2f}% |"
        )

    lines.extend([
        "",
        "## 3. Cross-Period Ledger Trajectory Checkpoints ($t = 210, 365, 420$)",
        "",
        "| Curve | Group | Checkpoint Day | Horizon $h$ | Ledger $\\max \\rho(h)$ | Theo 575d $L_{\\max}(h)$ | Commitment $\\bar\\rho(h)$ | Headroom | 575d Cert Secure |",
        "|---|---|---:|---:|---:|---:|---:|---:|:---:|",
    ])
    sub_ck = traj_df[
        (traj_df["workload"] == "DistinctHeavy")
        & (traj_df["horizon_h"].isin([90, 210, 365, 420]))
    ].groupby(["commitment_curve", "group", "checkpoint_day", "horizon_h"], as_index=False).mean(numeric_only=True)
    for _, r in sub_ck.iterrows():
        lines.append(
            f"| {r['commitment_curve']} | {r['group']} | {int(r['checkpoint_day'])} | {int(r['horizon_h'])} | "
            f"{r['observed_ledger_max_rho']:.6f} | {r['theoretical_L_max_575']:.6f} | {r['commitment_rho_bar']:.6f} | "
            f"{r['ledger_headroom']:.6f} | {'PASS' if r['cert_575_secure'] >= 1.0 else 'FAIL'} |"
        )

    lines.extend([
        "",
        "## 4. $C_r \\in \\{30, 50, 100, 200\\}$ Clipping Sensitivity Ablation (Fixed Week1_v2 Budget $d$)",
        "",
        "| $C_r$ | Isotropic $\\eta$ | $\\sigma_r$ [Analytic] | Clipped User-Days (%) | Clipping Bias (Mean $\\pm$ 95% CI) | DP Mechanism Noise RMSE (Mean $\\pm$ 95% CI) | Postproc RMSE | Total End-to-End RMSE (Mean $\\pm$ 95% CI) |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for _, r in abl_sum.iterrows():
        lines.append(
            f"| {int(r['C_r'])} | {r['eta_mean']:.6f} | {r['sigma_r_analytical_mean']:.2f} | "
            f"{r['clipping_rate_pct_mean']:.2f}% | "
            f"{r['requests_clipping_bias_mean']:.2f} $\\pm$ {r['requests_clipping_bias_ci95']:.2f} | "
            f"{r['requests_raw_noise_rmse_mean']:.2f} $\\pm$ {r['requests_raw_noise_rmse_ci95']:.2f} | "
            f"{r['requests_postproc_rmse_mean']:.2f} | "
            f"**{r['requests_total_rmse_mean']:.2f} $\\pm$ {r['requests_total_rmse_ci95']:.2f}** |"
        )

    lines.extend([
        "",
        "## 5. Figures",
        "- [fig1_420d_cross_period_ledger_trajectory.png](figures/fig1_420d_cross_period_ledger_trajectory.png)",
        "- [fig2_cr_clipping_pareto_ablation.png](figures/fig2_cr_clipping_pareto_ablation.png)",
        "- [fig3_420d_distinct_mape_comparison.png](figures/fig3_420d_distinct_mape_comparison.png)",
    ])

    (OUTPUT_DIR / "LONG_HORIZON_420D_REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="420d Cross-Period E2E & Cr Ablation")
    parser.add_argument("--smoke", action="store_true", help="Run small smoke test")
    parser.add_argument("--full", action="store_true", help="Run full multi-seed benchmark (2000 users, 3 seeds x 10 MC)")
    args = parser.parse_args()

    if args.full:
        mode = "FULL"
        n_users = 2000
        n_days = 420
        data_seeds = [42, 1042, 2042]
        n_mc_trials = 10
    else:
        mode = "SMOKE"
        n_users = 200
        n_days = 420
        data_seeds = [42]
        n_mc_trials = 2

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    print(f"=== Running 420d Cross-Period & Cr Ablation [{mode}] ===")
    print(f"Users={n_users}, Days={n_days}, DataSeeds={data_seeds}, MC_Trials={n_mc_trials}")

    t_all = time.time()
    all_e2e_rows: List[Dict[str, Any]] = []
    all_traj_rows: List[Dict[str, Any]] = []
    all_abl_rows: List[Dict[str, Any]] = []

    if len(data_seeds) > 1:
        max_workers = min(len(data_seeds), 4)
        with ProcessPoolExecutor(max_workers=max_workers) as pool:
            futs = [
                pool.submit(_execute_single_seed, s, n_users, n_days, n_mc_trials)
                for s in data_seeds
            ]
            for fut in as_completed(futs):
                res = fut.result()
                print(f"  [Seed {res['data_seed']}] done in {res['total_seconds']:.2f}s (ingest {res['ingest_seconds']:.2f}s)")
                all_e2e_rows.extend(res["e2e_rows"])
                all_traj_rows.extend(res["ledger_traj_rows"])
                all_abl_rows.extend(res["ablation_rows"])
    else:
        res = _execute_single_seed(data_seeds[0], n_users, n_days, n_mc_trials)
        print(f"  [Seed {res['data_seed']}] done in {res['total_seconds']:.2f}s (ingest {res['ingest_seconds']:.2f}s)")
        all_e2e_rows.extend(res["e2e_rows"])
        all_traj_rows.extend(res["ledger_traj_rows"])
        all_abl_rows.extend(res["ablation_rows"])

    e2e_raw_df = pd.DataFrame(all_e2e_rows)
    traj_df = pd.DataFrame(all_traj_rows)
    abl_raw_df = pd.DataFrame(all_abl_rows)

    e2e_metric_cols = [
        "daily_d", "planner_abstract_objective", "dual_lower_bound", "optimality_rel_gap", "min_slack_575",
        "requests_clipping_bias", "requests_raw_noise_rmse", "requests_postproc_rmse", "requests_total_rmse",
        "failures_clipping_bias", "failures_raw_noise_rmse", "failures_total_rmse",
        "dist_7d_windows", "dist_7d_raw_rmse", "dist_7d_total_rmse", "dist_7d_mape_pct",
        "dist_30d_windows", "dist_30d_raw_rmse", "dist_30d_total_rmse", "dist_30d_mape_pct",
        "dist_90d_windows", "dist_90d_raw_rmse", "dist_90d_total_rmse", "dist_90d_mape_pct",
    ]
    e2e_sum = aggregate_with_ci(
        e2e_raw_df,
        ["workload", "commitment_curve", "group", "phases", "optimality_tier"],
        e2e_metric_cols,
    )

    abl_metric_cols = [
        "eta", "daily_d", "rho_r", "rho_f", "sigma_r_analytical", "clipping_rate_pct",
        "requests_clipping_bias", "requests_raw_noise_rmse", "requests_postproc_rmse",
        "requests_total_rmse", "failures_total_rmse",
    ]
    abl_sum = aggregate_with_ci(abl_raw_df, ["C_r", "C_f"], abl_metric_cols)

    e2e_raw_df.to_csv(OUTPUT_DIR / "long_horizon_420d_raw_runs.csv", index=False)
    e2e_sum.to_csv(OUTPUT_DIR / "long_horizon_420d_summary.csv", index=False)
    traj_df.to_csv(OUTPUT_DIR / "long_horizon_420d_cross_period_ledger.csv", index=False)
    abl_raw_df.to_csv(OUTPUT_DIR / "cr_clipping_ablation_raw_runs.csv", index=False)
    abl_sum.to_csv(OUTPUT_DIR / "cr_clipping_ablation_420d.csv", index=False)

    generate_figures(e2e_sum, traj_df, abl_sum)
    elapsed_all = time.time() - t_all
    generate_markdown_report(mode, n_users, n_days, data_seeds, n_mc_trials, elapsed_all, e2e_sum, traj_df, abl_sum)

    print(f"Completed [{mode}] in {elapsed_all:.2f}s. Reports written to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
