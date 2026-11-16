"""Final Paper-Grade 420-Day Statistical, Post-Processing, Same-Query-Date & Finite/Infinite Scope Audit (v3).

Preserves ALL frozen v1 and v2 baselines untouched:
  - Frozen v1 (90d):  results/calendar_planning/true_e2e_benchmark_results.csv
  - Frozen v1 (420d): results/sdg_mh/long_horizon_420d/
  - Frozen v2 (420d): results/sdg_mh/long_horizon_420d_v2_audit/
Writes all revised outputs cleanly into:
  - results/sdg_mh/long_horizon_420d_v3_final/

Key v3 Fixes over v2 (PR #6):
  1. Strictly Injective 128-bit CRN Seed Derivation & Exhaustive Collision Verification:
     Replaces PR#6's truncated uint32 SeedSequence (which collided on e.g. (3042,2,410,1)
     and (5042,4,218,2) -> 3129808095) with bit-packed injective 128-bit integers, verifying
     zero collisions across ALL primary and CalendarJob (+500003) task streams while preserving
     B_BudgetOnly / D_JointPlanning CRN pairing.
  2. Uniform Window-Level Post-Processing Across ALL Competitive Mechanisms:
     Applies identical window summation followed by single post-processing:
       Req_hat = max(0, sum_t y_req_t),  Fail_hat = min(Req_hat, max(0, sum_t y_fail_t))
     to SDG-MH, Tree-2l, Tree-U, and W-BD-30 (removing W-BD-30's per-day positive bias inflation
     and enforcing Fail <= Req uniformly across Tree-2l/Tree-U/W-BD-30), and outputs an explicit
     v2 vs v3 competitive baseline diff table.
  3. Same-Query-Date Comparison Separating Release-Day Accuracy, Latest-Available Accuracy & Staleness:
     Compares B_BudgetOnly and D_JointPlanning on IDENTICAL query request sets:
       - Regime A: Own Release Dates (staleness = 0; intrinsic DP noise, differs for phi_30=0 vs 4)
       - Regime B: Common Daily Dashboard Query Dates d in [d0, 420] (identical gt(w,d), decomposes
                   total served error, staleness days, and pure staleness lag drift RMSE)
       - Regime C: Common Sunday Weekly Report Dates d % 7 == 0 (exposes 0d vs 3d Sunday lag for phi_30=4)
  4. Explicit Finite vs Infinite Horizon Privacy Scope & Holm-Bonferroni / Wilcoxon Inference:
     Distinguishes SDG-MH's infinite-horizon certificate (h >= 1) from finite dyadic Tree (T <= 512)
     and bounded-horizon W-BD-30, restricting performance claims strictly to shared [1, 420] span;
     adds Holm-Bonferroni adjusted p-values and exact Wilcoxon signed-rank p-values alongside paired t-CIs.
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
from typing import Any, Dict, List, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).parents[2] / "src"))

from private_llm_analytics.ingestion.schema import COL_SUCCESS, COL_USER_ID
from private_llm_analytics.sdg_mh.baselines import TreeBaseline, WEventBaseline
from private_llm_analytics.sdg_mh.calendar_accounting import (
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


V2_SUMMARY_PATH = Path("results/sdg_mh/long_horizon_420d_v2_audit/audit_v2_competitive_baselines_summary.csv")
OUTPUT_DIR = Path("results/sdg_mh/long_horizon_420d_v3_final")
FIGURES_DIR = OUTPUT_DIR / "figures"

CURVE_REGISTRY: Dict[str, PrivacyCommitmentCurve] = {
    "Week1_v2": WEEK1_V2_CURVE,
    "Original": ORIGINAL_PLANNER_CURVE,
}

C_R_DEFAULT = 50.0
C_F_DEFAULT = 10.0
K_TASKS_PER_USER = 3


def derive_crn_seed(data_seed: int, mc_trial: int, day: int, stream_id: int) -> int:
    """Strictly injective 128-bit bit-packed seed for CRN pairing across B/D without uint32 collisions.

    Layout: (data_seed << 96) | (mc_trial << 64) | (day << 32) | stream_id.
    stream_id < 500000 ensures CalendarJob's `seed + 500003` offset stays inside the lowest 32-bit lane.
    """
    ds, mc, d, st = int(data_seed), int(mc_trial), int(day), int(stream_id)
    if not (0 <= ds < (1 << 32) and 0 <= mc < (1 << 32) and 0 <= d < (1 << 32) and 0 <= st < 500000):
        raise ValueError(f"Seed tuple out of injective 128-bit range: {(ds, mc, d, st)}")
    return (ds << 96) | (mc << 64) | (d << 32) | st


def verify_all_experiment_seeds_collision_free(
    data_seeds: Sequence[int],
    n_mc_trials: int,
    n_days: int,
) -> Dict[str, Any]:
    """Exhaustively generate every seed used across SDG-MH (including +500003 task offset) and baselines; assert 0 collisions."""
    seen: set[int] = set()
    total_expected = 0
    for ds in data_seeds:
        for mc in range(n_mc_trials):
            # Baseline stream seeds at day=0 (stream 31=Tree-2l, 32=Tree-U, 41=W-BD-30)
            for b_st in (31, 32, 41):
                s = derive_crn_seed(ds, mc, 0, b_st)
                assert s not in seen, f"Seed collision on baseline tuple {(ds, mc, 0, b_st)}"
                seen.add(s)
                total_expected += 1
            for day in range(1, n_days + 1):
                s_add = derive_crn_seed(ds, mc, day, 1)
                s_dist = derive_crn_seed(ds, mc, day, 2)
                s_tsk = s_dist + 500003  # CalendarJob internal task-histogram offset
                for s in (s_add, s_dist, s_tsk):
                    assert s not in seen, f"Seed collision on {(ds, mc, day)} seed={s}"
                    seen.add(s)
                    total_expected += 1
    assert len(seen) == total_expected
    return {
        "total_seeds_checked": total_expected,
        "unique_seeds": len(seen),
        "collisions_detected": 0,
    }


def postprocess_window_pair(
    raw_req_sum: np.ndarray,
    raw_fail_sum: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Uniform window-level non-negative clipping & Failures <= Requests post-processing across all mechanisms."""
    req_hat = np.maximum(0.0, np.asarray(raw_req_sum, dtype=np.float64))
    fail_hat = np.minimum(req_hat, np.maximum(0.0, np.asarray(raw_fail_sum, dtype=np.float64)))
    return req_hat, fail_hat


def holm_bonferroni(p_values: Sequence[float]) -> List[float]:
    """Step-down Holm-Bonferroni family-wise error rate adjusted p-values."""
    m = len(p_values)
    if m == 0:
        return []
    indexed = sorted(enumerate(p_values), key=lambda x: (1.0 if math.isnan(x[1]) else x[1]))
    adj = [float("nan")] * m
    prev_max = 0.0
    for rank, (orig_idx, p_val) in enumerate(indexed):
        if math.isnan(p_val):
            adj[orig_idx] = float("nan")
            continue
        scaled = min(1.0, p_val * (m - rank))
        prev_max = max(prev_max, scaled)
        adj[orig_idx] = float(prev_max)
    return adj


def make_valid_date_str(day: int, start_year: int = 2025) -> str:
    """Convert 1-indexed day into valid ISO-8601 calendar date string."""
    return (datetime.date(start_year, 1, 1) + datetime.timedelta(days=day - 1)).isoformat()


def precompute_plans() -> Tuple[Dict[Tuple[str, str, str], Dict[str, Any]], pd.DataFrame]:
    """Solve planner once globally and verify post-shrinkage Decimal100 dual lower bounds."""
    plan_specs = [
        ("Default", "Week1_v2", (0.40, 0.25, 0.25, 0.10), ["A_Baseline", "D_JointPlanning"]),
        ("DistinctHeavy", "Week1_v2", (0.10, 0.40, 0.40, 0.10), ["B_BudgetOnly", "D_JointPlanning"]),
        ("DistinctHeavy", "Original", (0.10, 0.40, 0.40, 0.10), ["B_BudgetOnly", "D_JointPlanning"]),
    ]
    solved_map: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
    cost_rows: List[Dict[str, Any]] = []

    for wl_name, curve_name, wl_weights, groups in plan_specs:
        curve = CURVE_REGISTRY[curve_name]
        planner = JointCalendarPlanner(PlannerConfig(commitment=curve, workload_weights=wl_weights))
        for grp_key in groups:
            t0 = time.perf_counter()
            if grp_key == "A_Baseline":
                plan = planner.plan_group_a()
            elif grp_key == "B_BudgetOnly":
                plan = planner.plan_group_b()
            else:
                plan = planner.plan_group_d()
            solve_sec = time.perf_counter() - t0
            assert plan.is_certified_publishable is True
            assert plan.min_slack >= 0.0

            n7 = planner.n7_tables[plan.phases[0]]
            n30 = planner.n30_tables[plan.phases[1]]
            n90 = planner.n90_tables[plan.phases[2]]
            l_dist = np.max(
                n7 * plan.distinct_costs[0]
                + n30 * plan.distinct_costs[1]
                + n90 * plan.distinct_costs[2],
                axis=1,
            )[:420]
            rem_add_env = np.maximum(1e-12, planner.rho_bars[:420] - l_dist)

            solved_map[(wl_name, curve_name, grp_key)] = {
                "curve": curve,
                "workload_weights": wl_weights,
                "plan": plan,
                "solve_seconds": solve_sec,
                "rem_add_envelope_420": rem_add_env,
            }
            sigs_dist = [round(math.sqrt(11.0 / (2.0 * x)), 4) for x in plan.distinct_costs]
            sig_add_eff = math.sqrt((C_R_DEFAULT**2 + C_F_DEFAULT**2) / (2.0 * plan.daily_d))
            cost_rows.append({
                "workload": wl_name,
                "commitment_curve": curve_name,
                "group": grp_key,
                "phases": str(plan.phases),
                "solve_seconds": round(solve_sec, 4),
                "daily_d": plan.daily_d,
                "x_7": plan.distinct_costs[0],
                "x_30": plan.distinct_costs[1],
                "x_90": plan.distinct_costs[2],
                "sigma_add_eff": round(sig_add_eff, 3),
                "sigma_7": sigs_dist[0],
                "sigma_30": sigs_dist[1],
                "sigma_90": sigs_dist[2],
                "primal_obj_certified": plan.primal_objective_certified,
                "dual_lower_bound": plan.dual_lower_bound,
                "optimality_rel_gap": plan.optimality_rel_gap,
                "optimality_tier": plan.optimality_certification_tier,
                "dual_cert_method": plan.diagnostics.get("dual_cert_method", "N/A"),
            })
    return solved_map, pd.DataFrame(cost_rows)


def calibrate_tree_and_wbd_on_envelope(
    rem_add_envelope: np.ndarray,
    daily_d_sdg: float,
    n_days: int = 420,
) -> Dict[str, Any]:
    """Calibrate finite dyadic Binary Trees (T<=512) and W-BD-30 to identical [1,420] Multi-Horizon curve."""
    t_pow2 = 512
    levels_pow2 = 9

    tree_configs: Dict[str, Any] = {}
    for vname in ("Tree-2l", "Tree-U"):
        t0 = time.perf_counter()
        tb = TreeBaseline(variant=vname, T=t_pow2, L=levels_pow2, C=1.0)
        ratios_env, ratios_d = [], []
        P_m = np.zeros((t_pow2 + 1, t_pow2 + 1), dtype=np.float64)
        P_m[1:, 1:] = np.cumsum(np.cumsum(tb.M_base, axis=0), axis=1)
        for h in range(1, n_days + 1):
            s_idx = np.arange(0, n_days - h + 1)
            e_idx = s_idx + h
            block_sums = P_m[e_idx, e_idx] - P_m[s_idx, e_idx] - P_m[e_idx, s_idx] + P_m[s_idx, s_idx]
            rho_u = 0.5 * float(np.max(block_sums))
            ratios_env.append((h, rho_u / float(rem_add_envelope[h - 1])))
            ratios_d.append((h, rho_u / float(h * daily_d_sdg)))

        tight_h_env, beta_sq_env = max(ratios_env, key=lambda x: x[1])
        tight_h_d, beta_sq_d = max(ratios_d, key=lambda x: x[1])
        recon_mat = tb.cov_base @ (tb.A.T * (1.0 / tb.w_levels)[None, :])
        tree_configs[vname] = {
            "A_mat": tb.A,
            "beta_env": math.sqrt(beta_sq_env),
            "beta_d": math.sqrt(beta_sq_d),
            "tight_h_env": tight_h_env,
            "tight_h_d": tight_h_d,
            "recon_mat": recon_mat,
            "node_std_base": np.sqrt(tb.w_levels),
            "setup_sec": time.perf_counter() - t0,
        }

    w_ev = 30
    wbd_factors = []
    for h in range(1, n_days + 1):
        fac = (1.0 / (2.0 * w_ev) + 0.25) if h == 1 else (h / (2.0 * w_ev) + 0.5 * math.ceil(h / w_ev))
        wbd_factors.append((h, float(rem_add_envelope[h - 1]) / fac))
    wbd_tight_h, rho_w_mh_safe = min(wbd_factors, key=lambda x: x[1])
    rho_w_single_h30 = float(rem_add_envelope[w_ev - 1])

    return {
        "trees": tree_configs,
        "wbd": {
            "w": w_ev,
            "rho_w_mh_safe": rho_w_mh_safe,
            "tight_h": wbd_tight_h,
            "rho_w_naive_h30": rho_w_single_h30,
            "naive_h1_ratio": (rho_w_single_h30 * (1.0 / (2.0 * w_ev) + 0.25)) / float(rem_add_envelope[0]),
            "naive_h365_ratio": (rho_w_single_h30 * (365.0 / (2.0 * w_ev) + 0.5 * math.ceil(365 / w_ev))) / float(rem_add_envelope[364]),
        },
    }


def _rolling_window_sums(arr: np.ndarray, w: int, start_day: int, end_day: int) -> np.ndarray:
    """Vector of rolling sums over [d-w+1, d] for d in [start_day, end_day] (arr is 1-indexed)."""
    cs = np.cumsum(arr)
    days = np.arange(start_day, end_day + 1)
    return cs[days] - cs[days - w]


def _precompute_all_daily_distinct_service_gt(
    rolling_state: RollingDistinctState,
    services_list: Sequence[str],
    tasks_list: Sequence[str],
    n_days: int,
) -> Dict[int, np.ndarray]:
    """O(N) sliding-window ground truth array gt[w][day, s_idx] for ALL days d in 1..n_days, cross-checked against RollingDistinctState."""
    m = len(services_list)
    gt_by_w: Dict[int, np.ndarray] = {}
    day_states = rolling_state._day_states

    for w in (7, 30, 90):
        arr = np.zeros((n_days + 1, m), dtype=np.float64)
        # user -> count of active days inside [d-w+1, d] per service index s
        srv_active_days: List[Dict[str, int]] = [{} for _ in range(m)]
        cur_counts = np.zeros(m, dtype=np.float64)

        for d in range(1, n_days + 1):
            d_add = day_states.get(d, {})
            for uid, (mask, _) in d_add.items():
                for s_i in range(m):
                    if (mask >> s_i) & 1:
                        prev = srv_active_days[s_i].get(uid, 0)
                        if prev == 0:
                            cur_counts[s_i] += 1.0
                        srv_active_days[s_i][uid] = prev + 1
            if d - w >= 1:
                d_rem = day_states.get(d - w, {})
                for uid, (mask, _) in d_rem.items():
                    for s_i in range(m):
                        if (mask >> s_i) & 1:
                            prev = srv_active_days[s_i][uid]
                            if prev == 1:
                                del srv_active_days[s_i][uid]
                                cur_counts[s_i] -= 1.0
                            else:
                                srv_active_days[s_i][uid] = prev - 1
            arr[d, :] = cur_counts
        gt_by_w[w] = arr

        # Exact mathematical equivalence check against RollingDistinctState.compute_window_histogram
        for chk_d in (w, min(n_days, 210), n_days):
            s_exact, _ = rolling_state.compute_window_histogram(
                max(1, chk_d - w + 1), chk_d, services_list, tasks_list, K_TASKS_PER_USER
            )
            assert np.array_equal(arr[chk_d], s_exact.astype(np.float64)), (
                f"Sliding GT mismatch w={w} day={chk_d}: {arr[chk_d]} vs {s_exact}"
            )

    return gt_by_w


def _execute_single_seed(
    data_seed: int,
    n_users: int,
    n_days: int,
    n_mc_trials: int,
    solved_plans: Dict[Tuple[str, str, str], Dict[str, Any]],
    baseline_cal: Dict[str, Any],
) -> Dict[str, Any]:
    """Run paired SDG-MH (release-day & common-query-date evaluation) + uniformly post-processed baselines."""
    t_start = time.perf_counter()
    gen = SyntheticTelemetryGenerator(
        n_initial_users=n_users,
        n_years=n_days / 365.0,
        seed=data_seed,
    )
    services_list = gen.services
    tasks_list = gen.tasks
    eta = (C_R_DEFAULT**2) / (C_R_DEFAULT**2 + C_F_DEFAULT**2)

    daily_dfs: Dict[int, pd.DataFrame] = {}
    daily_raw_req = np.zeros(n_days + 1, dtype=np.float64)
    daily_raw_fail = np.zeros(n_days + 1, dtype=np.float64)
    daily_clip_req = np.zeros(n_days + 1, dtype=np.float64)
    daily_clip_fail = np.zeros(n_days + 1, dtype=np.float64)

    with tempfile.TemporaryDirectory() as tmp_dir:
        rolling_state = RollingDistinctState(storage_dir=Path(tmp_dir), w_max=n_days + 10)
        for day, df_day in gen.stream_daily_telemetry(start_day=1, end_day=n_days):
            daily_dfs[day] = df_day
            if not df_day.empty:
                rc = df_day[COL_USER_ID].value_counts().to_numpy(dtype=np.float64)
                daily_raw_req[day] = float(np.sum(rc))
                daily_clip_req[day] = float(np.minimum(rc, C_R_DEFAULT).sum())

                fdf = df_day[~df_day[COL_SUCCESS]]
                if not fdf.empty:
                    fc = fdf[COL_USER_ID].value_counts().to_numpy(dtype=np.float64)
                    daily_raw_fail[day] = float(np.sum(fc))
                    daily_clip_fail[day] = float(np.minimum(fc, C_F_DEFAULT).sum())
            rolling_state.append_day(day, df_day, services_list, tasks_list, persist_disk=False)

        gt_by_w = _precompute_all_daily_distinct_service_gt(
            rolling_state, services_list, tasks_list, n_days
        )
        t_ingest = time.perf_counter() - t_start

        req_clip_bias = float(np.mean(daily_clip_req[1:] - daily_raw_req[1:]))
        fail_clip_bias = float(np.mean(daily_clip_fail[1:] - daily_raw_fail[1:]))

        raw_roll_req = {w: _rolling_window_sums(daily_raw_req, w, w, n_days) for w in (1, 7, 30, 90)}
        raw_roll_fail = {w: _rolling_window_sums(daily_raw_fail, w, w, n_days) for w in (1, 7, 30, 90)}

        sdg_rows: List[Dict[str, Any]] = []
        comp_rows: List[Dict[str, Any]] = []

        # Common query-request start days where BOTH B (phi=0) and D (phi_30=4) have at least 1 prior release:
        # w=7:  B & D both release on d=7  -> common dates d in [7, 420]  (414 dates)
        # w=30: D releases d=32, B releases d=35 -> common dates d in [35, 420] (386 dates)
        # w=90: B & D both release on d=90 -> common dates d in [90, 420] (331 dates)
        common_start_by_w = {7: 7, 30: 35, 90: 90}

        for (wl_name, curve_name, grp_key), info in solved_plans.items():
            plan: PlanningResult = info["plan"]
            curve: PrivacyCommitmentCurve = info["curve"]
            d_val = plan.daily_d
            phi7, phi30, phi90 = plan.phases

            for trial in range(n_mc_trials):
                use_strict_ledger = (trial == 0)
                trial_ledger = UnifiedPrivacyLedger(
                    delta=1e-6,
                    allow_unconfigured_policy=(not use_strict_ledger),
                )
                if use_strict_ledger:
                    trial_ledger.set_active_plan(plan=plan, commitment=curve)

                sealed_store = SealedStore()
                cal_store = DistinctCalendarStore()
                seal_job = SealJob.from_budget(
                    sealed_store=sealed_store,
                    ledger=trial_ledger,
                    daily_d=d_val,
                    eta=eta,
                    C_r=C_R_DEFAULT,
                    C_f=C_F_DEFAULT,
                )
                cal_job = CalendarJob.from_planner_result(
                    calendar_store=cal_store,
                    ledger=trial_ledger,
                    plan=plan,
                    services_list=services_list,
                    tasks_list=tasks_list,
                    k_tasks_per_user=K_TASKS_PER_USER,
                    rolling_state=rolling_state,
                )
                qs = ReadOnlySealedQueryService(
                    sealed_store=sealed_store,
                    calendar_store=cal_store,
                    public_population_size=n_users,
                )

                raw_daily_req_y = np.zeros(n_days + 1, dtype=np.float64)
                raw_daily_fail_y = np.zeros(n_days + 1, dtype=np.float64)

                for day in range(1, n_days + 1):
                    d_str = make_valid_date_str(day)
                    s_add = derive_crn_seed(data_seed, trial, day, stream_id=1)
                    s_dist = derive_crn_seed(data_seed, trial, day, stream_id=2)
                    rec_day = seal_job.seal_day(day, d_str, daily_dfs[day], seed=s_add)
                    cal_job.execute_day(day, rolling_state=rolling_state, seed=s_dist)
                    raw_daily_req_y[day] = float(rec_day.y_requests)
                    raw_daily_fail_y[day] = float(rec_day.y_failures)

                if use_strict_ledger:
                    for ck_day in (210, 365, 420):
                        prof = trial_ledger.compute_full_horizon_profile(t_span=ck_day)
                        for h_chk in range(1, ck_day + 1):
                            assert prof[h_chk] <= float(curve.rho_bar(h_chk)) + 1e-12

                # Uniform window-level additive post-processing (matches ReadOnlySealedQueryService)
                add_roll_metrics: Dict[int, Tuple[float, float]] = {}
                est_w_req_cache: Dict[int, np.ndarray] = {}
                for w_add in (1, 7, 30, 90):
                    hat_r_w, hat_f_w = postprocess_window_pair(
                        _rolling_window_sums(raw_daily_req_y, w_add, w_add, n_days),
                        _rolling_window_sums(raw_daily_fail_y, w_add, w_add, n_days),
                    )
                    est_w_req_cache[w_add] = hat_r_w
                    rmse_r_w = float(np.sqrt(np.mean(np.square(hat_r_w - raw_roll_req[w_add]))))
                    rmse_f_w = float(np.sqrt(np.mean(np.square(hat_f_w - raw_roll_fail[w_add]))))
                    add_roll_metrics[w_add] = (rmse_r_w, rmse_f_w)

                # Distinct Evaluation:
                #   A) Release-Day Intrinsic Accuracy (d % s == phi % s, staleness = 0)
                #   B) Common Daily Dashboard Query Set d in [common_start, n_days] (latest-available & staleness)
                #   C) Common Sunday Weekly Report Dates d in [35, n_days], d % 7 == 0
                dist_rel_stats: Dict[int, Dict[str, float]] = {}
                dist_daily_stats: Dict[int, Dict[str, float]] = {}
                dist_sun_stats: Dict[int, Dict[str, float]] = {}

                for w, s_step, phi_w in ((7, 7, phi7), (30, 7, phi30), (90, 30, phi90)):
                    gt_mat = gt_by_w[w]  # shape (n_days+1, 5)

                    # Regime A: Own release days
                    rel_raw_e, rel_tot_e, rel_mapes, rel_gts = [], [], [], []
                    for day in range(w, n_days + 1):
                        if day % s_step == phi_w % s_step:
                            cal_rec = cal_store.get_release(window_length=w, end_day=day)
                            q_dist = qs.query_distinct_prevalence(window_length=w, end_day=day)
                            assert q_dist["status"] == "EXACT_CALENDAR_HIT"
                            for s_idx, s_name in enumerate(services_list):
                                y_val = float(cal_rec.noisy_services[s_idx])
                                tv = float(gt_mat[day, s_idx])
                                pv = float(q_dist["services"][s_name])
                                rel_raw_e.append(y_val - tv)
                                rel_tot_e.append(pv - tv)
                                rel_gts.append(tv)
                                if tv > 0:
                                    rel_mapes.append(abs(pv - tv) / tv)
                    dist_rel_stats[w] = {
                        "raw_rmse": float(np.sqrt(np.mean(np.square(rel_raw_e)))),
                        "total_rmse": float(np.sqrt(np.mean(np.square(rel_tot_e)))),
                        "mape_pct": float(100.0 * np.mean(rel_mapes)),
                        "mean_gt": float(np.mean(rel_gts)),
                    }

                    # Regime B (Daily Common Dates) & Regime C (Sunday Common Dates d % 7 == 0)
                    c_start = common_start_by_w[w]
                    d_tot_e, d_mapes, d_lags, d_drift_e, d_gts = [], [], [], [], []
                    sun_tot_e, sun_mapes, sun_lags, sun_drift_e = [], [], [], []
                    cur_rel = None
                    for day in range(1, n_days + 1):
                        if day >= w and (day % s_step == phi_w % s_step):
                            cur_rel = cal_store.get_release(window_length=w, end_day=day)
                        if day >= c_start:
                            assert cur_rel is not None
                            t_latest = cur_rel.end_day
                            lag = float(day - t_latest)
                            # Matches ReadOnlySealedQueryService.query_distinct_prevalence non-negativity clamp
                            for s_idx in range(len(services_list)):
                                pv = float(max(0, cur_rel.noisy_services[s_idx]))
                                tv_req = float(gt_mat[day, s_idx])
                                tv_rel = float(gt_mat[t_latest, s_idx])
                                d_tot_e.append(pv - tv_req)
                                d_drift_e.append(tv_rel - tv_req)
                                d_lags.append(lag)
                                d_gts.append(tv_req)
                                if tv_req > 0:
                                    d_mapes.append(abs(pv - tv_req) / tv_req)
                                if day >= 35 and (day % 7 == 0):
                                    sun_tot_e.append(pv - tv_req)
                                    sun_drift_e.append(tv_rel - tv_req)
                                    sun_lags.append(lag)
                                    if tv_req > 0:
                                        sun_mapes.append(abs(pv - tv_req) / tv_req)

                    dist_daily_stats[w] = {
                        "served_total_rmse": float(np.sqrt(np.mean(np.square(d_tot_e)))),
                        "served_mape_pct": float(100.0 * np.mean(d_mapes)),
                        "mean_staleness_days": float(np.mean(d_lags)),
                        "staleness_drift_rmse": float(np.sqrt(np.mean(np.square(d_drift_e)))),
                        "mean_gt": float(np.mean(d_gts)),
                    }
                    dist_sun_stats[w] = {
                        "sunday_total_rmse": float(np.sqrt(np.mean(np.square(sun_tot_e)))),
                        "sunday_mape_pct": float(100.0 * np.mean(sun_mapes)),
                        "sunday_staleness_days": float(np.mean(sun_lags)),
                        "sunday_drift_rmse": float(np.sqrt(np.mean(np.square(sun_drift_e)))),
                    }

                sdg_rows.append({
                    "data_seed": data_seed,
                    "mc_trial": trial,
                    "workload": wl_name,
                    "commitment_curve": curve_name,
                    "group": grp_key,
                    "phases": str(plan.phases),
                    "daily_d": d_val,
                    "planner_abstract_obj": float(plan.weighted_workload_error),
                    "dual_lower_bound": float(plan.dual_lower_bound),
                    "optimality_rel_gap": float(plan.optimality_rel_gap),
                    "requests_clipping_bias": req_clip_bias,
                    "failures_clipping_bias": fail_clip_bias,
                    "requests_1d_total_rmse": add_roll_metrics[1][0],
                    "failures_1d_total_rmse": add_roll_metrics[1][1],
                    "requests_30d_total_rmse": add_roll_metrics[30][0],
                    "failures_30d_total_rmse": add_roll_metrics[30][1],
                    # Regime A (Own Release Dates)
                    "rel_7d_raw_rmse": dist_rel_stats[7]["raw_rmse"],
                    "rel_7d_mape_pct": dist_rel_stats[7]["mape_pct"],
                    "rel_7d_mean_gt": dist_rel_stats[7]["mean_gt"],
                    "rel_30d_raw_rmse": dist_rel_stats[30]["raw_rmse"],
                    "rel_30d_mape_pct": dist_rel_stats[30]["mape_pct"],
                    "rel_30d_mean_gt": dist_rel_stats[30]["mean_gt"],
                    "rel_90d_raw_rmse": dist_rel_stats[90]["raw_rmse"],
                    "rel_90d_mape_pct": dist_rel_stats[90]["mape_pct"],
                    "rel_90d_mean_gt": dist_rel_stats[90]["mean_gt"],
                    # Regime B (Common Daily Dashboard Query Set)
                    "common_daily_7d_served_rmse": dist_daily_stats[7]["served_total_rmse"],
                    "common_daily_7d_served_mape": dist_daily_stats[7]["served_mape_pct"],
                    "common_daily_7d_staleness_d": dist_daily_stats[7]["mean_staleness_days"],
                    "common_daily_30d_served_rmse": dist_daily_stats[30]["served_total_rmse"],
                    "common_daily_30d_served_mape": dist_daily_stats[30]["served_mape_pct"],
                    "common_daily_30d_staleness_d": dist_daily_stats[30]["mean_staleness_days"],
                    "common_daily_30d_drift_rmse": dist_daily_stats[30]["staleness_drift_rmse"],
                    "common_daily_30d_mean_gt": dist_daily_stats[30]["mean_gt"],
                    "common_daily_90d_served_rmse": dist_daily_stats[90]["served_total_rmse"],
                    "common_daily_90d_served_mape": dist_daily_stats[90]["served_mape_pct"],
                    "common_daily_90d_staleness_d": dist_daily_stats[90]["mean_staleness_days"],
                    # Regime C (Common Fixed Sunday Report Query Set d % 7 == 0)
                    "common_sunday_30d_served_rmse": dist_sun_stats[30]["sunday_total_rmse"],
                    "common_sunday_30d_served_mape": dist_sun_stats[30]["sunday_mape_pct"],
                    "common_sunday_30d_staleness_d": dist_sun_stats[30]["sunday_staleness_days"],
                    "common_sunday_30d_drift_rmse": dist_sun_stats[30]["sunday_drift_rmse"],
                })

                if (wl_name, curve_name, grp_key) == ("Default", "Week1_v2", "D_JointPlanning"):
                    comp_rows.append({
                        "data_seed": data_seed,
                        "mc_trial": trial,
                        "method": "SDG-MH (D_JointPlanning)",
                        "privacy_horizon_scope": "Infinite h >= 1 (evaluated [1, 420])",
                        "calibration_scope": "Exact Multi-Horizon rem_add_envelope (All h in 1..420)",
                        "requests_1d_rmse": add_roll_metrics[1][0],
                        "requests_7d_rmse": add_roll_metrics[7][0],
                        "requests_30d_rmse": add_roll_metrics[30][0],
                        "requests_90d_rmse": add_roll_metrics[90][0],
                        "failures_1d_rmse": add_roll_metrics[1][1],
                        "failures_7d_rmse": add_roll_metrics[7][1],
                        "failures_30d_rmse": add_roll_metrics[30][1],
                        "failures_90d_rmse": add_roll_metrics[90][1],
                    })

        # Competitive Additive Time-Series Baselines with IDENTICAL window-level postprocess_window_pair
        iso_sens_eff = math.sqrt(C_R_DEFAULT**2 + C_F_DEFAULT**2)
        x_pad_req = np.zeros(512, dtype=np.float64)
        x_pad_fail = np.zeros(512, dtype=np.float64)
        x_pad_req[:n_days] = daily_clip_req[1:]
        x_pad_fail[:n_days] = daily_clip_fail[1:]

        for vname, tcfg in baseline_cal["trees"].items():
            A_mat: np.ndarray = tcfg["A_mat"]
            recon_mat: np.ndarray = tcfg["recon_mat"]
            node_sig_unit: np.ndarray = tcfg["node_std_base"]
            mu_req = A_mat @ x_pad_req
            mu_fail = A_mat @ x_pad_fail

            for mode_lbl, beta_val in (
                (f"{vname} (MH-RemEnv)", tcfg["beta_env"]),
                (f"{vname} (Daily-d Parity)", tcfg["beta_d"]),
            ):
                if vname == "Tree-U" and mode_lbl.endswith("(Daily-d Parity)"):
                    continue
                sig_vec = node_sig_unit * beta_val * iso_sens_eff
                for trial in range(n_mc_trials):
                    rng_tr = np.random.default_rng(derive_crn_seed(data_seed, trial, 0, stream_id=31 if "2l" in vname else 32))
                    z_req = rng_tr.normal(0.0, sig_vec)
                    z_fail = rng_tr.normal(0.0, sig_vec)
                    hat_x_r = recon_mat @ (mu_req + z_req)
                    hat_x_f = recon_mat @ (mu_fail + z_fail)

                    est_r_arr = np.zeros(n_days + 1, dtype=np.float64)
                    est_f_arr = np.zeros(n_days + 1, dtype=np.float64)
                    est_r_arr[1:] = hat_x_r[:n_days]
                    est_f_arr[1:] = hat_x_f[:n_days]

                    m_r, m_f = {}, {}
                    for w_add in (1, 7, 30, 90):
                        win_r, win_f = postprocess_window_pair(
                            _rolling_window_sums(est_r_arr, w_add, w_add, n_days),
                            _rolling_window_sums(est_f_arr, w_add, w_add, n_days),
                        )
                        m_r[w_add] = float(np.sqrt(np.mean(np.square(win_r - raw_roll_req[w_add]))))
                        m_f[w_add] = float(np.sqrt(np.mean(np.square(win_f - raw_roll_fail[w_add]))))

                    comp_rows.append({
                        "data_seed": data_seed,
                        "mc_trial": trial,
                        "method": mode_lbl,
                        "privacy_horizon_scope": "Finite Dyadic T <= 512 (calibrated [1, 420])",
                        "calibration_scope": (
                            f"Exact Multi-Horizon rem_add_envelope (tight_h={tcfg['tight_h_env']})"
                            if "RemEnv" in mode_lbl
                            else f"Exact Multi-Horizon h*d_SDG (tight_h={tcfg['tight_h_d']})"
                        ),
                        "requests_1d_rmse": m_r[1],
                        "requests_7d_rmse": m_r[7],
                        "requests_30d_rmse": m_r[30],
                        "requests_90d_rmse": m_r[90],
                        "failures_1d_rmse": m_f[1],
                        "failures_7d_rmse": m_f[7],
                        "failures_30d_rmse": m_f[30],
                        "failures_90d_rmse": m_f[90],
                    })

        # W-BD-30 strictly shrunk to [1, 420] envelope, storing raw signed stream BEFORE uniform window postprocess_window_pair
        wbd_cfg = baseline_cal["wbd"]
        rho_w_safe = wbd_cfg["rho_w_mh_safe"]
        for trial in range(n_mc_trials):
            rng_w = np.random.default_rng(derive_crn_seed(data_seed, trial, 0, stream_id=41))
            wbd_r = WEventBaseline(w=30, mode="BD", target_rho_w=eta * rho_w_safe, C=C_R_DEFAULT)
            wbd_f = WEventBaseline(w=30, mode="BD", target_rho_w=(1.0 - eta) * rho_w_safe, C=C_F_DEFAULT)
            raw_r_arr = np.zeros(n_days + 1, dtype=np.float64)
            raw_f_arr = np.zeros(n_days + 1, dtype=np.float64)
            for day in range(1, n_days + 1):
                raw_r_arr[day] = float(wbd_r.release_day(day, daily_clip_req[day], rng=rng_w))
                raw_f_arr[day] = float(wbd_f.release_day(day, daily_clip_fail[day], rng=rng_w))

            m_r, m_f = {}, {}
            for w_add in (1, 7, 30, 90):
                win_r, win_f = postprocess_window_pair(
                    _rolling_window_sums(raw_r_arr, w_add, w_add, n_days),
                    _rolling_window_sums(raw_f_arr, w_add, w_add, n_days),
                )
                m_r[w_add] = float(np.sqrt(np.mean(np.square(win_r - raw_roll_req[w_add]))))
                m_f[w_add] = float(np.sqrt(np.mean(np.square(win_f - raw_roll_fail[w_add]))))

            comp_rows.append({
                "data_seed": data_seed,
                "mc_trial": trial,
                "method": "W-BD-30 (MH-Calibrated)",
                "privacy_horizon_scope": "Rolling Adaptive Finite [1, 420] envelope",
                "calibration_scope": f"Shrunk to all-h Multi-Horizon envelope (tight_h={wbd_cfg['tight_h']}, rho_w={rho_w_safe:.6f})",
                "requests_1d_rmse": m_r[1],
                "requests_7d_rmse": m_r[7],
                "requests_30d_rmse": m_r[30],
                "requests_90d_rmse": m_r[90],
                "failures_1d_rmse": m_f[1],
                "failures_7d_rmse": m_f[7],
                "failures_30d_rmse": m_f[30],
                "failures_90d_rmse": m_f[90],
            })

    return {
        "data_seed": data_seed,
        "ingest_seconds": t_ingest,
        "total_seconds": time.perf_counter() - t_start,
        "sdg_rows": sdg_rows,
        "comp_rows": comp_rows,
    }


def stratified_seed_summary(
    df: pd.DataFrame,
    group_cols: List[str],
    metric_cols: List[str],
) -> pd.DataFrame:
    """Two-level hierarchical variance decomposition treating data_seed as independent statistical unit."""
    rows: List[Dict[str, Any]] = []
    for keys, sub in df.groupby(group_cols, sort=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        seed_means_df = sub.groupby("data_seed", as_index=False)[metric_cols].mean(numeric_only=True)
        K = len(seed_means_df)
        M = int(sub.groupby("data_seed").size().mean())

        for m in metric_cols:
            s_vals = seed_means_df[m].to_numpy(dtype=np.float64)
            grand_mean = float(np.mean(s_vals))
            if K > 1:
                seed_sd = float(np.std(s_vals, ddof=1))
                t_crit = float(stats.t.ppf(0.975, df=K - 1))
                seed_ci95 = float(t_crit * seed_sd / math.sqrt(K))
            else:
                seed_sd = 0.0
                seed_ci95 = float("nan")

            if M > 1:
                mc_vars = sub.groupby("data_seed")[m].var(ddof=1).to_numpy(dtype=np.float64)
                within_mc_var = float(np.mean(mc_vars))
                within_mc_sd = float(math.sqrt(within_mc_var))
                pop_var_comp = max(0.0, (seed_sd**2) - (within_mc_var / M)) if K > 1 else 0.0
                denom_icc = pop_var_comp + within_mc_var
                icc = float(pop_var_comp / denom_icc) if denom_icc > 1e-18 else 0.0
            else:
                within_mc_sd = 0.0
                pop_var_comp = seed_sd**2
                icc = 1.0 if seed_sd > 0 else 0.0

            row: Dict[str, Any] = {c: v for c, v in zip(group_cols, keys)}
            row.update({
                "metric": m,
                "K_seeds": K,
                "M_mc_per_seed": M,
                "mean": round(grand_mean, 4),
                "seed_level_sd": round(seed_sd, 4),
                "seed_level_ci95": round(seed_ci95, 4) if math.isfinite(seed_ci95) else float("nan"),
                "within_mc_sd": round(within_mc_sd, 4),
                "between_pop_sd": round(math.sqrt(pop_var_comp), 4),
                "icc": round(icc, 4),
            })
            rows.append(row)
    return pd.DataFrame(rows)


def compute_paired_with_holm_and_wilcoxon(df: pd.DataFrame) -> pd.DataFrame:
    """Compute CRN seed-level paired t-CIs, exact Wilcoxon signed-rank p, and Holm-Bonferroni adjusted p."""
    pairs = [
        ("DistinctHeavy", "Week1_v2", "B_BudgetOnly", "D_JointPlanning"),
        ("DistinctHeavy", "Original", "B_BudgetOnly", "D_JointPlanning"),
        ("Default", "Week1_v2", "A_Baseline", "D_JointPlanning"),
    ]
    metrics = [
        "requests_1d_total_rmse",
        "failures_1d_total_rmse",
        "rel_7d_mape_pct",
        "rel_30d_mape_pct",
        "rel_90d_mape_pct",
        "common_daily_7d_served_mape",
        "common_daily_30d_served_mape",
        "common_daily_30d_served_rmse",
        "common_daily_30d_drift_rmse",
        "common_daily_30d_staleness_d",
        "common_daily_90d_served_mape",
        "common_sunday_30d_served_mape",
        "common_sunday_30d_staleness_d",
        "common_sunday_30d_drift_rmse",
    ]
    rows: List[Dict[str, Any]] = []
    for wl, curve, g_base, g_trt in pairs:
        sub = df[(df["workload"] == wl) & (df["commitment_curve"] == curve)]
        s_means = sub.groupby(["data_seed", "group"], as_index=False)[metrics].mean(numeric_only=True)
        for m in metrics:
            piv = s_means.pivot(index="data_seed", columns="group", values=m)
            base_arr = piv[g_base].to_numpy(dtype=np.float64)
            trt_arr = piv[g_trt].to_numpy(dtype=np.float64)
            diffs = trt_arr - base_arr
            K = len(diffs)
            mean_base = float(np.mean(base_arr))
            mean_trt = float(np.mean(trt_arr))
            mean_diff = float(np.mean(diffs))
            rel_pct = 100.0 * mean_diff / abs(mean_base) if abs(mean_base) > 1e-12 else 0.0

            if K > 1:
                sd_diff = float(np.std(diffs, ddof=1))
                se_diff = sd_diff / math.sqrt(K)
                t_crit = float(stats.t.ppf(0.975, df=K - 1))
                ci95_diff = float(t_crit * se_diff)
                if se_diff > 1e-15:
                    p_t = float(2.0 * stats.t.sf(abs(mean_diff / se_diff), df=K - 1))
                else:
                    p_t = 1.0
                if np.all(np.abs(diffs) < 1e-15):
                    p_wil = 1.0
                else:
                    _, p_wil = stats.wilcoxon(diffs, zero_method="wilcox", method="exact")
                    p_wil = float(p_wil)
            else:
                sd_diff, ci95_diff, p_t, p_wil = 0.0, float("nan"), float("nan"), float("nan")

            wins_trt = int(np.sum(diffs < 0))
            rows.append({
                "workload": wl,
                "commitment_curve": curve,
                "comparison": f"{g_trt} - {g_base}",
                "metric": m,
                "K_seeds": K,
                "base_mean": round(mean_base, 4),
                "trt_mean": round(mean_trt, 4),
                "paired_diff_mean": round(mean_diff, 4),
                "paired_rel_change_pct": round(rel_pct, 3),
                "paired_diff_ci95": round(ci95_diff, 4) if math.isfinite(ci95_diff) else float("nan"),
                "paired_t_p_value": round(p_t, 5) if math.isfinite(p_t) else float("nan"),
                "wilcoxon_p_value": round(p_wil, 5) if math.isfinite(p_wil) else float("nan"),
                "seeds_lower_error_count": f"{wins_trt}/{K}",
            })

    out_df = pd.DataFrame(rows)
    # Apply Holm-Bonferroni correction across primary error metrics within each curve contrast
    out_df["paired_t_p_holm"] = float("nan")
    primary_test_metrics = {
        "requests_1d_total_rmse", "failures_1d_total_rmse",
        "rel_7d_mape_pct", "rel_30d_mape_pct", "rel_90d_mape_pct",
        "common_daily_7d_served_mape", "common_daily_30d_served_mape", "common_daily_90d_served_mape",
        "common_sunday_30d_served_mape",
    }
    mask_family = out_df["metric"].isin(primary_test_metrics)
    p_raw = out_df.loc[mask_family, "paired_t_p_value"].tolist()
    out_df.loc[mask_family, "paired_t_p_holm"] = [round(v, 5) if math.isfinite(v) else float("nan") for v in holm_bonferroni(p_raw)]
    return out_df


def compute_v2_vs_v3_baseline_diff(comp_v3_sum: pd.DataFrame) -> pd.DataFrame:
    """Load frozen v2 competitive summary (read-only) and compute exact metric diffs after unified post-processing."""
    if not V2_SUMMARY_PATH.exists():
        return pd.DataFrame()
    v2_df = pd.read_csv(V2_SUMMARY_PATH)
    merged = pd.merge(
        v2_df[["method", "metric", "mean", "seed_level_ci95"]].rename(columns={"mean": "v2_mean", "seed_level_ci95": "v2_ci95"}),
        comp_v3_sum[["method", "metric", "mean", "seed_level_ci95"]].rename(columns={"mean": "v3_mean", "seed_level_ci95": "v3_ci95"}),
        on=["method", "metric"],
        how="inner",
    )
    merged["diff_v3_minus_v2"] = (merged["v3_mean"] - merged["v2_mean"]).round(4)
    merged["rel_change_pct"] = (100.0 * (merged["v3_mean"] - merged["v2_mean"]) / np.maximum(1e-12, merged["v2_mean"])).round(3)
    return merged


def generate_figures(
    paired_df: pd.DataFrame,
    comp_sum_df: pd.DataFrame,
) -> None:
    """Render clean v3 figures into results/sdg_mh/long_horizon_420d_v3_final/figures/."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 10, "axes.grid": True, "grid.alpha": 0.3})

    # Fig 1: Same-Query-Date (Common Daily vs Sunday vs Release-Day) 30d & 7d/90d Paired Diff D - B
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    m_list = ["rel_7d_mape_pct", "common_daily_30d_served_mape", "common_sunday_30d_served_mape", "rel_90d_mape_pct"]
    lbls = ["7d Release\n(=Daily Common)", "30d Common\nDaily Query", "30d Common\nSunday Query", "90d Release\n(=Daily Common)"]
    x = np.arange(len(m_list))

    for ax, curve_name in zip(axes, ["Week1_v2", "Original"]):
        sub = paired_df[(paired_df["workload"] == "DistinctHeavy") & (paired_df["commitment_curve"] == curve_name)]
        diffs = [float(sub[sub["metric"] == m]["paired_diff_mean"].iloc[0]) for m in m_list]
        cis = [float(sub[sub["metric"] == m]["paired_diff_ci95"].iloc[0]) for m in m_list]
        colors = ["#2ca02c" if d < 0 else "#d62728" for d in diffs]
        ax.bar(x, diffs, yerr=cis, capsize=5, color=colors, alpha=0.84, width=0.52)
        ax.axhline(0.0, color="black", linestyle="--", linewidth=1.1)
        ax.set_xticks(x)
        ax.set_xticklabels(lbls, fontsize=8.5)
        ax.set_ylabel("Paired MAPE Diff: D - B (% pts)\n(Negative = JointPlanning Lower Error)")
        ax.set_title(f"{curve_name} DistinctHeavy: Same-Date Served vs Staleness")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "fig1_v3_same_query_date_d_vs_b.png", dpi=180)
    plt.close()

    # Fig 2: Fair Multi-Horizon Competitive Baseline Comparison (Uniform Post-Processing, Week1_v2 [1,420])
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.0))
    for ax, prefix, title_str in [
        (axes[0], "requests", "Requests Rolling RMSE (Uniform Post-Proc, Log)"),
        (axes[1], "failures", "Failures Rolling RMSE (Uniform Post-Proc, Log)"),
    ]:
        w_cols = [f"{prefix}_1d_rmse", f"{prefix}_7d_rmse", f"{prefix}_30d_rmse", f"{prefix}_90d_rmse"]
        w_labels = ["w=1d", "w=7d", "w=30d", "w=90d"]
        methods = comp_sum_df["method"].unique().tolist()
        x_w = np.arange(len(w_cols))
        width = 0.8 / max(1, len(methods))
        for idx, m_name in enumerate(methods):
            m_sub = comp_sum_df[comp_sum_df["method"] == m_name]
            means = [float(m_sub[m_sub["metric"] == wc]["mean"].iloc[0]) for wc in w_cols]
            cis = [float(m_sub[m_sub["metric"] == wc]["seed_level_ci95"].iloc[0]) for wc in w_cols]
            offset = (idx - (len(methods) - 1) / 2.0) * width
            ax.bar(x_w + offset, means, width, yerr=cis, capsize=2.5, label=m_name, alpha=0.88)
        ax.set_yscale("log")
        ax.set_xticks(x_w)
        ax.set_xticklabels(w_labels)
        ax.set_ylabel("RMSE (95% Seed CI, df=7)")
        ax.set_title(title_str)
        ax.legend(fontsize=7.5)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "fig2_v3_uniform_postproc_baselines.png", dpi=180)
    plt.close()


def generate_markdown_report(
    mode: str,
    n_users: int,
    n_days: int,
    data_seeds: List[int],
    n_mc_trials: int,
    elapsed_all: float,
    mean_seed_sec: float,
    seed_audit: Dict[str, Any],
    cost_df: pd.DataFrame,
    paired_df: pd.DataFrame,
    comp_sum_df: pd.DataFrame,
    diff_v2_v3_df: pd.DataFrame,
    baseline_cal: Dict[str, Any],
) -> None:
    """Write final v3 audit report preserving v1/v2 frozen artifacts."""
    K = len(data_seeds)
    lines: List[str] = [
        "# SDG-MH Final Consistency Audit v3 Report (Post-PR #6)",
        "",
        f"- **Run Mode**: `{mode}`",
        f"- **Design**: `{n_users}` initial users, `{n_days}` days, **`K={K}` independent data seeds** (`{data_seeds}`) $\\times$ `{n_mc_trials}` CRN MC trials (`df={K-1}`)",
        f"- **128-bit Seed Collision Audit**: `{seed_audit['total_seeds_checked']}` seeds checked (including `CalendarJob +500003` task offset) -> **`{seed_audit['unique_seeds']}` unique, `{seed_audit['collisions_detected']}` collisions**",
        f"- **Total Wall Time**: `{elapsed_all:.2f}s` (`{mean_seed_sec:.2f}s`/seed avg across parallel workers)",
        "",
        "## 1. Optimization Cost & Directed-Rounding Dual Certificate (`DECIMAL100_DIRECTED_ROUNDING`)",
        "",
        "> **Assumption Clarification**: `DECIMAL100_DIRECTED_ROUNDING` bounds all dual-function rounding using `prec=100` `ROUND_FLOOR`/`ROUND_CEILING` relative to the operational `float64` RHS curve `rho_bars` (plus exact rational `1/7, 1/30` lower bounds), conditional on active constraint representation and $c_j \\ge 0, l_j > 0$. `CONSERVATIVE_BACKOFF_REL = 1e-11` ensures primal feasibility in both IEEE-754 and real arithmetic.",
        "",
        "| Workload | Curve | Group | Phases | Solve Time | Primal $f^*$ | Dual LB (`Rel Gap`) | Tier |",
        "|---|---|---|---|---:|---:|---:|:---:|",
    ]
    for _, r in cost_df.iterrows():
        lines.append(
            f"| {r['workload']} | {r['commitment_curve']} | {r['group']} | `{r['phases']}` | "
            f"{r['solve_seconds']:.3f}s | {r['primal_obj_certified']:.4f} | {r['dual_lower_bound']:.4f} (`{r['optimality_rel_gap']:.2e}`) | `{r['optimality_tier']}` |"
        )

    lines.extend([
        "",
        "## 2. Same-Query-Date Comparison (`B_BudgetOnly` vs `D_JointPlanning`, CRN + Holm-Bonferroni + Wilcoxon)",
        "",
        "- **Regime A (`rel_*`)**: Evaluated on each schedule's own release dates (0 staleness; for `30d`, B uses `day%7==0` Sunday while D uses `day%7==4` Thursday).",
        "- **Regime B (`common_daily_*`)**: Evaluated on **identical daily dashboard query dates** (`w=7`: `7..420`, `w=30`: `35..420`, `w=90`: `90..420`) against **identical ground truth `gt(w,d)`**, cleanly isolating latest-available served error vs mean staleness (`~3.0d` for 7d/30d).",
        "- **Regime C (`common_sunday_*`)**: Evaluated on **identical fixed Sunday report dates (`d%7==0, d in 35..420`)**, isolating phase-shift staleness lag (`B: 0.0d` vs `D: 3.0d` for `30d`).",
        "",
        "| Workload | Curve | Contrast | Metric | Base Mean | Trt Mean | Paired Diff (`95% CI`) | Raw $p_t$ | Holm $p_{\\text{holm}}$ | Exact Wilcoxon $p$ | Seeds `<0` |",
        "|---|---|---|---|---:|---:|---:|---:|---:|---:|:---:|",
    ])
    for _, r in paired_df.iterrows():
        ph = f"`{r['paired_t_p_holm']:.4f}`" if math.isfinite(r["paired_t_p_holm"]) else "diag"
        lines.append(
            f"| {r['workload']} | {r['commitment_curve']} | `{r['comparison']}` | `{r['metric']}` | "
            f"{r['base_mean']:.4f} | {r['trt_mean']:.4f} | **{r['paired_diff_mean']:+.4f}** ($\\pm${r['paired_diff_ci95']:.4f}) | "
            f"`{r['paired_t_p_value']:.4f}` | {ph} | `{r['wilcoxon_p_value']:.4f}` | `{r['seeds_lower_error_count']}` |"
        )

    lines.extend([
        "",
        "## 3. Uniformly Post-Processed Competitive Baselines (Identical `[1, 420]` Additive Scope, `C_r=50, C_f=10`)",
        "",
        "> **Finite vs Infinite Scope Note**: Performance comparison is strictly restricted to the **shared `[1, 420]` finite span**, **shared remaining additive Multi-Horizon curve $\\bar\\rho_{\\text{add}}(h)$ on $h \\in [1, 420]$**, and **shared additive rolling queries $w \\in \\{1,7,30,90\\}$**. `SDG-MH` simultaneously holds an **infinite-horizon ($h \\ge 1$) certificate**, whereas `Tree-2l / Tree-U (L=9)` covers only finite $T \\le 512$ and `W-BD-30` is calibrated over `[1, 420]`. Non-additive rolling Distinct queries remain `N/A` for baselines.",
        "",
        "| Method | Privacy Certificate Scope | Req 1d RMSE | Req 7d RMSE | Req 30d RMSE | Req 90d RMSE | Fail 1d RMSE | Fail 30d RMSE | Fail 90d RMSE |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for m_name in comp_sum_df["method"].unique():
        sub_m = comp_sum_df[comp_sum_df["method"] == m_name]
        hscope = str(sub_m["privacy_horizon_scope"].iloc[0])
        vals = {r["metric"]: (r["mean"], r["seed_level_ci95"]) for _, r in sub_m.iterrows()}
        lines.append(
            f"| **{m_name}** | {hscope} | "
            f"{vals['requests_1d_rmse'][0]:.1f}$\\pm${vals['requests_1d_rmse'][1]:.1f} | "
            f"{vals['requests_7d_rmse'][0]:.1f}$\\pm${vals['requests_7d_rmse'][1]:.1f} | "
            f"{vals['requests_30d_rmse'][0]:.1f}$\\pm${vals['requests_30d_rmse'][1]:.1f} | "
            f"{vals['requests_90d_rmse'][0]:.1f}$\\pm${vals['requests_90d_rmse'][1]:.1f} | "
            f"{vals['failures_1d_rmse'][0]:.1f}$\\pm${vals['failures_1d_rmse'][1]:.1f} | "
            f"{vals['failures_30d_rmse'][0]:.1f}$\\pm${vals['failures_30d_rmse'][1]:.1f} | "
            f"{vals['failures_90d_rmse'][0]:.1f}$\\pm${vals['failures_90d_rmse'][1]:.1f} |"
        )

    if not diff_v2_v3_df.empty:
        lines.extend([
            "",
            "## 4. Recomputed Competitive Baseline Diff: Frozen v2 (PR #6) vs Revised v3 (Unified Post-Processing + 128-bit Seeds)",
            "",
            "| Method | Metric | Frozen v2 Mean | Revised v3 Mean | Diff (`v3 - v2`) | Rel Change |",
            "|---|---|---:|---:|---:|---:|",
        ])
        for _, r in diff_v2_v3_df.iterrows():
            lines.append(
                f"| {r['method']} | `{r['metric']}` | {r['v2_mean']:.2f} | {r['v3_mean']:.2f} | **{r['diff_v3_minus_v2']:+.2f}** | {r['rel_change_pct']:+.2f}% |"
            )

    lines.extend([
        "",
        "## 5. Figures & Reproducible Commands",
        "```powershell",
        "# Smoke:",
        ".\\.venv\\Scripts\\python.exe experiments/sdg_mh/run_long_horizon_420d_audit_v3.py --smoke",
        "# Full (K=8 seeds x 5 CRN MC = 40 runs/config):",
        ".\\.venv\\Scripts\\python.exe experiments/sdg_mh/run_long_horizon_420d_audit_v3.py --full",
        "```",
        "- [fig1_v3_same_query_date_d_vs_b.png](figures/fig1_v3_same_query_date_d_vs_b.png)",
        "- [fig2_v3_uniform_postproc_baselines.png](figures/fig2_v3_uniform_postproc_baselines.png)",
    ])
    (OUTPUT_DIR / "AUDIT_V3_REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Final Consistency Audit v3 Benchmark")
    parser.add_argument("--smoke", action="store_true", help="Smoke test (200 users, 2 seeds x 2 MC)")
    parser.add_argument("--full", action="store_true", help="Full audit (2000 users, 8 seeds x 5 CRN MC)")
    args = parser.parse_args()

    if args.full:
        mode = "FULL_AUDIT_V3"
        n_users = 2000
        n_days = 420
        data_seeds = [42, 1042, 2042, 3042, 4042, 5042, 6042, 7042]
        n_mc_trials = 5
    else:
        mode = "SMOKE_AUDIT_V3"
        n_users = 200
        n_days = 420
        data_seeds = [42, 1042]
        n_mc_trials = 2

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    seed_audit = verify_all_experiment_seeds_collision_free(data_seeds, n_mc_trials, n_days)
    print(f"=== Running Audit v3 [{mode}] | Seeds verified collision-free: {seed_audit} ===")

    t0_all = time.perf_counter()
    solved_plans, cost_df = precompute_plans()
    ref_d_info = solved_plans[("Default", "Week1_v2", "D_JointPlanning")]
    baseline_cal = calibrate_tree_and_wbd_on_envelope(
        rem_add_envelope=ref_d_info["rem_add_envelope_420"],
        daily_d_sdg=float(ref_d_info["plan"].daily_d),
        n_days=n_days,
    )

    all_sdg_rows: List[Dict[str, Any]] = []
    all_comp_rows: List[Dict[str, Any]] = []
    seed_times: List[float] = []

    max_workers = min(len(data_seeds), 4)
    if max_workers > 1:
        with ProcessPoolExecutor(max_workers=max_workers) as pool:
            futs = [
                pool.submit(_execute_single_seed, s, n_users, n_days, n_mc_trials, solved_plans, baseline_cal)
                for s in data_seeds
            ]
            for fut in as_completed(futs):
                res = fut.result()
                seed_times.append(res["total_seconds"])
                print(f"  [Seed {res['data_seed']}] done in {res['total_seconds']:.2f}s (ingest {res['ingest_seconds']:.2f}s)")
                all_sdg_rows.extend(res["sdg_rows"])
                all_comp_rows.extend(res["comp_rows"])
    else:
        res = _execute_single_seed(data_seeds[0], n_users, n_days, n_mc_trials, solved_plans, baseline_cal)
        seed_times.append(res["total_seconds"])
        all_sdg_rows.extend(res["sdg_rows"])
        all_comp_rows.extend(res["comp_rows"])

    sdg_raw_df = pd.DataFrame(all_sdg_rows)
    comp_raw_df = pd.DataFrame(all_comp_rows)

    sdg_metric_cols = [
        "requests_clipping_bias", "failures_clipping_bias",
        "requests_1d_total_rmse", "failures_1d_total_rmse",
        "requests_30d_total_rmse", "failures_30d_total_rmse",
        "rel_7d_raw_rmse", "rel_7d_mape_pct", "rel_7d_mean_gt",
        "rel_30d_raw_rmse", "rel_30d_mape_pct", "rel_30d_mean_gt",
        "rel_90d_raw_rmse", "rel_90d_mape_pct", "rel_90d_mean_gt",
        "common_daily_7d_served_rmse", "common_daily_7d_served_mape", "common_daily_7d_staleness_d",
        "common_daily_30d_served_rmse", "common_daily_30d_served_mape", "common_daily_30d_staleness_d", "common_daily_30d_drift_rmse", "common_daily_30d_mean_gt",
        "common_daily_90d_served_rmse", "common_daily_90d_served_mape", "common_daily_90d_staleness_d",
        "common_sunday_30d_served_rmse", "common_sunday_30d_served_mape", "common_sunday_30d_staleness_d", "common_sunday_30d_drift_rmse",
    ]
    strat_df = stratified_seed_summary(
        sdg_raw_df, ["workload", "commitment_curve", "group", "phases"], sdg_metric_cols
    )
    paired_df = compute_paired_with_holm_and_wilcoxon(sdg_raw_df)

    comp_metric_cols = [
        "requests_1d_rmse", "requests_7d_rmse", "requests_30d_rmse", "requests_90d_rmse",
        "failures_1d_rmse", "failures_7d_rmse", "failures_30d_rmse", "failures_90d_rmse",
    ]
    comp_sum_df = stratified_seed_summary(
        comp_raw_df, ["method", "privacy_horizon_scope", "calibration_scope"], comp_metric_cols
    )
    diff_v2_v3_df = compute_v2_vs_v3_baseline_diff(comp_sum_df)

    elapsed_all = time.perf_counter() - t0_all
    mean_seed_sec = float(np.mean(seed_times))

    cost_df.to_csv(OUTPUT_DIR / "solver_runtime_and_dual_bounds.csv", index=False)
    sdg_raw_df.to_csv(OUTPUT_DIR / "audit_v3_sdg_raw_runs.csv", index=False)
    strat_df.to_csv(OUTPUT_DIR / "audit_v3_stratified_seed_summary.csv", index=False)
    paired_df.to_csv(OUTPUT_DIR / "audit_v3_paired_with_holm_wilcoxon.csv", index=False)
    comp_raw_df.to_csv(OUTPUT_DIR / "audit_v3_competitive_baselines_raw.csv", index=False)
    comp_sum_df.to_csv(OUTPUT_DIR / "audit_v3_competitive_baselines_summary.csv", index=False)
    if not diff_v2_v3_df.empty:
        diff_v2_v3_df.to_csv(OUTPUT_DIR / "v2_vs_v3_competitive_baseline_diff.csv", index=False)

    generate_figures(paired_df, comp_sum_df)
    generate_markdown_report(
        mode, n_users, n_days, data_seeds, n_mc_trials, elapsed_all, mean_seed_sec,
        seed_audit, cost_df, paired_df, comp_sum_df, diff_v2_v3_df, baseline_cal,
    )
    print(f"Completed [{mode}] in {elapsed_all:.2f}s -> {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
