"""Part 2: Real-Data Validation on AllenAI WildChat-4.8M (v4).

Evaluates SDG-MH (Multi-Horizon User-Time DP + Joint Calendar Planning + Hybrid
Distinct Accounting) and additive continual-release baselines (Tree-2l, Banded-MF)
on authentic multi-year public LLM telemetry from allenai/WildChat-4.8M
(stored locally in git object d5bb5a771282819cc1c7cbb8546f63c0d39b1fbe:data/wildchat_all_shards_4cols.parquet).

Strict Provenance & Zero-Fabrication Guarantees:
1. All timestamps, model strings, countries/regions, and pseudonymous user IDs
   (`hashed_ip`) come strictly from the authentic WildChat-4.8M parquet artifact.
   No synthetic timestamps, interpolated days, or fabricated users are generated.
2. Documented Real-Data Span & Limitations:
   - Full dataset: 3,199,860 conversations across 1,833,730 unique `hashed_ip`
     identifiers from 2023-04-09 to 2025-07-31 (845 calendar days, 809 active UTC
     dates, 36 server-outage missing dates).
   - Primary 420-day contiguous calendar window: 2023-11-20 to 2025-01-12
     (0-indexed calendar days 225..644 from 2023-04-09), covering 420 calendar days
     with 418 active UTC dates (1,721,032 conversations, 895,148 unique `hashed_ip`
     users) and 2 real Gradio server-outage zero-activity days (2024-01-18 and 2024-01-19).
   - Strictly gap-free sub-window: 2024-01-20 to 2025-01-12 (calendar indices
     286..644, 359 consecutive active UTC days with 0 missing dates).
   - Complete cohort partition: All 895,148 unique `hashed_ip` users active in the
     420-day window are partitioned without subsampling into K=8 disjoint real-user
     cohorts (~111,893.5 users per cohort) via deterministic stratified rank assignment
     (rank by 420d conversation count, cohort = rank mod 8).
"""

from __future__ import annotations

import hashlib
import io
import subprocess
import sys
import time
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from private_llm_analytics.sdg_mh.baselines import (
    BandedToeplitzBaseline,
    TreeBaseline,
)
from private_llm_analytics.sdg_mh.calendar_accounting import (
    ORIGINAL_PLANNER_CURVE,
    WEEK1_V2_CURVE,
)
from private_llm_analytics.sdg_mh.joint_planner import (
    JointCalendarPlanner,
    PlannerConfig,
)

GIT_BLOB_SPEC = "d5bb5a771282819cc1c7cbb8546f63c0d39b1fbe:data/wildchat_all_shards_4cols.parquet"

WORKLOAD_PRIORITIES = {
    "Default": (0.40, 0.25, 0.25, 0.10),
    "DistinctHeavy": (0.10, 0.40, 0.40, 0.10),
}
CURVE_MAP = {
    "Week1_v2": WEEK1_V2_CURVE,
    "Original": ORIGINAL_PLANNER_CURVE,
}

NUM_SERVICES = 5  # m = 5 OpenAI model families
NUM_CATEGORIES = 20  # |A| = 20 (5 model families x 4 UTC 6h quadrants)
MAX_CATEGORIES_PER_USER = 3  # k = 3 per-user window category cap
C_R = 50
C_F = 10
HORIZON_DAYS = 420
WIN_START_IDX = 225  # 2023-11-20 (0-indexed days since 2023-04-09)
WIN_END_IDX = 644    # 2025-01-12 (inclusive, 420 calendar days, 418 active)
GAPFREE_START_IDX = 286  # 2024-01-20 (359 consecutive active UTC days through 644)

K_COHORTS = 8
N_MC = 5


def derive_crn_seed_v4(cohort_id: int, mc_id: int, day: int, channel_code: int) -> int:
    """Collision-free 128-bit BLAKE2b seed derivation for paired CRN across B/A and D."""
    payload = f"wildchat_v4|{cohort_id}|{mc_id}|{day}|{channel_code}".encode("utf-8")
    digest = hashlib.blake2b(payload, digest_size=16).digest()
    return int.from_bytes(digest, "little")


def load_wildchat_from_git_blob() -> pd.DataFrame:
    print(f"[WildChat-v4] Loading parquet from local git blob {GIT_BLOB_SPEC} ...")
    t0 = time.perf_counter()
    raw_bytes = subprocess.check_output(["git", "cat-file", "blob", GIT_BLOB_SPEC], cwd=str(ROOT))
    df = pd.read_parquet(io.BytesIO(raw_bytes))
    elapsed = time.perf_counter() - t0
    print(f"[WildChat-v4] Loaded {len(df):,} rows in {elapsed:.2f}s. Columns: {list(df.columns)}")
    return df


def compute_exact_window_distinct_from_pairs(
    df_pairs: pd.DataFrame,
    entity_col: str,
    num_entities: int,
    windows: Tuple[int, ...] = (7, 30, 90),
) -> Tuple[Dict[int, np.ndarray], Dict[int, np.ndarray]]:
    """Compute exact sliding-window distinct user counts and sum of daily active users via interval union."""
    dedup = df_pairs[["u_id", entity_col, "rel_day"]].drop_duplicates().sort_values(
        ["u_id", entity_col, "rel_day"]
    )
    u_arr = dedup["u_id"].to_numpy(dtype=np.int64)
    e_arr = dedup[entity_col].to_numpy(dtype=np.int64)
    d_arr = dedup["rel_day"].to_numpy(dtype=np.int64)

    same_next = np.zeros(len(d_arr), dtype=bool)
    if len(d_arr) > 1:
        same_next[:-1] = (u_arr[:-1] == u_arr[1:]) & (e_arr[:-1] == e_arr[1:])
    next_d = np.full(len(d_arr), HORIZON_DAYS + 1000, dtype=np.int64)
    if len(d_arr) > 1:
        next_d[:-1] = np.where(same_next[:-1], d_arr[1:], HORIZON_DAYS + 1000)

    daily_active = np.zeros((HORIZON_DAYS, num_entities), dtype=np.float64)
    np.add.at(daily_active, (d_arr, e_arr), 1.0)

    win_distinct: Dict[int, np.ndarray] = {}
    win_daily_sum: Dict[int, np.ndarray] = {}

    for w in windows:
        diff = np.zeros((HORIZON_DAYS + 1, num_entities), dtype=np.float64)
        start_e = d_arr
        end_e = np.minimum(HORIZON_DAYS - 1, np.minimum(d_arr + w - 1, next_d - 1))
        valid = end_e >= start_e
        s_v = start_e[valid]
        en_v = end_e[valid] + 1
        ent_v = e_arr[valid]
        np.add.at(diff, (s_v, ent_v), 1.0)
        np.add.at(diff, (en_v, ent_v), -1.0)
        active_win = np.cumsum(diff[:-1], axis=0)
        active_win[: w - 1] = 0.0
        win_distinct[w] = active_win

        csum = np.vstack([np.zeros((1, num_entities)), np.cumsum(daily_active, axis=0)])
        rsum = np.zeros((HORIZON_DAYS, num_entities), dtype=np.float64)
        rsum[w - 1 :] = csum[w:] - csum[: HORIZON_DAYS - w + 1]
        win_daily_sum[w] = rsum

    return win_distinct, win_daily_sum


def precompute_cohort_suff_stats(df_cohort: pd.DataFrame) -> Dict[str, Any]:
    """Precompute scalar daily additive counts (C_R=50 per user-day) and exact window distinct counts."""
    u_codes, _ = pd.factorize(df_cohort["hashed_ip"], sort=False)
    df_c = df_cohort.assign(u_id=u_codes)

    # 1. Scalar daily additive requests per day (matching Theorem 1 & v3 SealJob: C_R = 50 per user-day)
    grp_req = (
        df_c.groupby(["rel_day", "u_id"], sort=False)
        .size()
        .reset_index(name="cnt")
    )
    daily_requests_raw = np.zeros(HORIZON_DAYS, dtype=np.float64)
    daily_requests_clipped = np.zeros(HORIZON_DAYS, dtype=np.float64)

    raw_s = grp_req.groupby("rel_day")["cnt"].sum()
    for d, val in raw_s.items():
        daily_requests_raw[int(d)] = float(val)

    grp_req["clip_cnt"] = np.minimum(grp_req["cnt"], C_R)
    clip_s = grp_req.groupby("rel_day")["clip_cnt"].sum()
    for d, val in clip_s.items():
        daily_requests_clipped[int(d)] = float(val)

    # 2. Exact service distinct counts across windows W in (7, 30, 90)
    svc_win_distinct, svc_win_daily_sum = compute_exact_window_distinct_from_pairs(
        df_c, "service_idx", NUM_SERVICES, (7, 30, 90)
    )

    # 3. Exact category distinct counts (unclipped and k=3 per-user window clipped)
    dedup_cat = df_c[["u_id", "cat_idx", "rel_day"]].drop_duplicates()
    u_cat_nunique = dedup_cat.groupby("u_id")["cat_idx"].nunique()
    multi_u_ids = set(u_cat_nunique[u_cat_nunique > MAX_CATEGORIES_PER_USER].index)

    cat_unclipped_all, _ = compute_exact_window_distinct_from_pairs(
        dedup_cat, "cat_idx", NUM_CATEGORIES, (7, 30, 90)
    )

    dedup_multi = dedup_cat[dedup_cat["u_id"].isin(multi_u_ids)]
    cat_unclipped_multi, _ = compute_exact_window_distinct_from_pairs(
        dedup_multi, "cat_idx", NUM_CATEGORIES, (7, 30, 90)
    )

    day_multi_user_cats: List[Dict[int, List[int]]] = [{} for _ in range(HORIZON_DAYS)]
    for row in dedup_multi.itertuples(index=False):
        d = int(row.rel_day)
        u = int(row.u_id)
        c = int(row.cat_idx)
        if u not in day_multi_user_cats[d]:
            day_multi_user_cats[d][u] = []
        day_multi_user_cats[d][u].append(c)

    distinct_windows: Dict[int, Dict[str, np.ndarray]] = {}
    for w in (7, 30, 90):
        clipped_multi = np.zeros((HORIZON_DAYS, NUM_CATEGORIES), dtype=np.float64)
        for d in range(w - 1, HORIZON_DAYS):
            u_cats_win: Dict[int, set[int]] = {}
            for tau in range(d - w + 1, d + 1):
                for u, clist in day_multi_user_cats[tau].items():
                    if u not in u_cats_win:
                        u_cats_win[u] = set(clist)
                    else:
                        u_cats_win[u].update(clist)
            row_counts = np.zeros(NUM_CATEGORIES, dtype=np.float64)
            for u, cset in u_cats_win.items():
                for c in sorted(cset)[:MAX_CATEGORIES_PER_USER]:
                    row_counts[c] += 1.0
            clipped_multi[d] = row_counts

        clipped_all = (cat_unclipped_all[w] - cat_unclipped_multi[w]) + clipped_multi
        distinct_windows[w] = {
            "true_svc": svc_win_distinct[w],
            "sum_daily_svc": svc_win_daily_sum[w],
            "true_cat": cat_unclipped_all[w],
            "clipped_cat": clipped_all,
        }

    return {
        "daily_requests_raw": daily_requests_raw,
        "daily_requests_clipped": daily_requests_clipped,
        "distinct_windows": distinct_windows,
        "total_convs": len(df_c),
        "active_users": int(df_c["u_id"].nunique()),
        "multi_cat_power_users": len(multi_u_ids),
    }


def extract_plan_parameters(plan_res: Any) -> Dict[str, Any]:
    """Extract phases and calibrated Gaussian sigmas from PlanningResult."""
    eta = 25.0 / 26.0
    delta_sq = float(NUM_SERVICES + 2 * MAX_CATEGORIES_PER_USER)  # 5 + 2*3 = 11.0
    phases = tuple(plan_res.phases)
    daily_d = float(plan_res.daily_d)
    dcosts = tuple(plan_res.distinct_costs)
    obj_val = float(plan_res.weighted_workload_error)

    rho_r = eta * daily_d
    sigma_req = float(C_R / np.sqrt(2.0 * max(1e-12, rho_r)))
    sigma_d7 = float(np.sqrt(delta_sq / (2.0 * max(1e-12, dcosts[0]))))
    sigma_d30 = float(np.sqrt(delta_sq / (2.0 * max(1e-12, dcosts[1]))))
    sigma_d90 = float(np.sqrt(delta_sq / (2.0 * max(1e-12, dcosts[2]))))
    return {
        "phases": phases,
        "daily_d": daily_d,
        "distinct_costs": dcosts,
        "objective_value": obj_val,
        "sigma_req": sigma_req,
        "sigma_d7": sigma_d7,
        "sigma_d30": sigma_d30,
        "sigma_d90": sigma_d90,
    }


def evaluate_schedule_on_cohort(
    suff: Dict[str, Any],
    plan_res: Any,
    cohort_id: int,
    mc_id: int,
    eval_mask: np.ndarray | None = None,
) -> Dict[str, float]:
    """Evaluate a planned SDG-MH schedule on one cohort & MC trial with paired CRN."""
    params = extract_plan_parameters(plan_res)
    sigma_req = params["sigma_req"]
    sigma_d7 = params["sigma_d7"]
    sigma_d30 = params["sigma_d30"]
    sigma_d90 = params["sigma_d90"]
    p7, p30, p90 = params["phases"]

    raw_req = suff["daily_requests_raw"]
    clip_req = suff["daily_requests_clipped"]

    raw_noise_req = np.zeros(HORIZON_DAYS, dtype=np.float64)
    for d in range(HORIZON_DAYS):
        rng = np.random.default_rng(derive_crn_seed_v4(cohort_id, mc_id, d, 1))
        raw_noise_req[d] = float(rng.normal(0.0, sigma_req))

    metrics: Dict[str, float] = {}
    for w in (1, 7, 30, 90):
        sq_errs = []
        for d in range(w - 1, HORIZON_DAYS):
            if eval_mask is not None and not eval_mask[d]:
                continue
            true_w = float(raw_req[d - w + 1 : d + 1].sum())
            est_w = max(0.0, float(clip_req[d - w + 1 : d + 1].sum() + raw_noise_req[d - w + 1 : d + 1].sum()))
            sq_errs.append((est_w - true_w) ** 2)
        metrics[f"req_{w}d_rmse"] = float(np.sqrt(np.mean(sq_errs)))

    for w, period, phase, sigma_d, ch_code in [
        (7, 7, p7, sigma_d7, 10),
        (30, 7, p30, sigma_d30, 20),
        (90, 30, p90, sigma_d90, 30),
    ]:
        dw = suff["distinct_windows"][w]
        true_svc = dw["true_svc"]
        true_cat = dw["true_cat"]
        clip_cat = dw["clipped_cat"]

        rel_active_mape_list = []
        rel_allcell_mape_list = []
        rel_rmse_list = []
        latest_active_mape_list = []
        sunday_staleness_list = []

        latest_est_svc = None
        latest_est_cat = None
        latest_rel_day = None

        for d in range(w - 1, HORIZON_DAYS):
            is_release = ((d - phase) % period == 0)
            if is_release:
                rng_s = np.random.default_rng(derive_crn_seed_v4(cohort_id, mc_id, d, ch_code))
                rng_c = np.random.default_rng(derive_crn_seed_v4(cohort_id, mc_id, d, ch_code + 1))
                est_svc = np.maximum(0.0, true_svc[d] + rng_s.normal(0.0, sigma_d, size=NUM_SERVICES))
                est_cat = np.maximum(0.0, clip_cat[d] + rng_c.normal(0.0, sigma_d, size=NUM_CATEGORIES))
                latest_est_svc = est_svc
                latest_est_cat = est_cat
                latest_rel_day = d

                if eval_mask is None or eval_mask[d]:
                    true_vec = np.concatenate([true_svc[d], true_cat[d]])
                    est_vec = np.concatenate([est_svc, est_cat])
                    denom = np.maximum(true_vec, 1.0)
                    rel_allcell_mape_list.append(float(np.mean(np.abs(est_vec - true_vec) / denom) * 100.0))
                    active_idx = true_vec >= 10.0
                    if np.any(active_idx):
                        rel_active_mape_list.append(
                            float(np.mean(np.abs(est_vec[active_idx] - true_vec[active_idx]) / true_vec[active_idx]) * 100.0)
                        )
                    rel_rmse_list.append(float(np.mean((est_vec - true_vec) ** 2)))

            if latest_est_svc is not None and (eval_mask is None or eval_mask[d]):
                true_vec = np.concatenate([true_svc[d], true_cat[d]])
                est_vec = np.concatenate([latest_est_svc, latest_est_cat])
                active_idx = true_vec >= 10.0
                if np.any(active_idx):
                    latest_active_mape_list.append(
                        float(np.mean(np.abs(est_vec[active_idx] - true_vec[active_idx]) / true_vec[active_idx]) * 100.0)
                    )
                if d % 7 == 3:
                    sunday_staleness_list.append(float(d - latest_rel_day))

        metrics[f"distinct_{w}d_active_mape"] = float(np.mean(rel_active_mape_list))
        metrics[f"distinct_{w}d_allcell_mape"] = float(np.mean(rel_allcell_mape_list))
        metrics[f"distinct_{w}d_release_rmse"] = float(np.sqrt(np.mean(rel_rmse_list)))
        metrics[f"distinct_{w}d_latest_active_mape"] = float(np.mean(latest_active_mape_list))
        metrics[f"distinct_{w}d_sunday_staleness"] = float(np.mean(sunday_staleness_list)) if sunday_staleness_list else 0.0

    return metrics


def main() -> None:
    out_dir = ROOT / "results" / "sdg_mh" / "wildchat_real_validation_v4"
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_wildchat_from_git_blob()
    ts = pd.to_datetime(df["timestamp"])
    df["date"] = ts.dt.date
    df["utc_quad"] = ts.dt.hour // 6
    min_date = date(2023, 4, 9)
    df["cal_idx"] = (df["date"] - min_date).apply(lambda td: td.days)

    total_convs = len(df)
    total_users = int(df["hashed_ip"].nunique())

    # Extract 420-day contiguous calendar window [225, 644] (2023-11-20 to 2025-01-12)
    df_win = df[(df["cal_idx"] >= WIN_START_IDX) & (df["cal_idx"] <= WIN_END_IDX)].copy()
    df_win["rel_day"] = df_win["cal_idx"] - WIN_START_IDX

    m_lower = df_win["model"].astype(str).str.lower()
    df_win["service_idx"] = np.select(
        [
            m_lower.str.contains("4o-mini", regex=False),
            m_lower.str.contains("4o", regex=False),
            m_lower.str.contains("3.5", regex=False),
            m_lower.str.contains("gpt-4", regex=False),
        ],
        [1, 0, 2, 3],
        default=4,
    )
    df_win["cat_idx"] = df_win["service_idx"] * 4 + df_win["utc_quad"]

    win_convs = len(df_win)
    win_users = int(df_win["hashed_ip"].nunique())
    win_active_days = int(df_win["rel_day"].nunique())
    missing_rel_days = sorted(set(range(HORIZON_DAYS)) - set(df_win["rel_day"].unique()))
    missing_dates = [str(min_date + timedelta(days=WIN_START_IDX + d)) for d in missing_rel_days]

    print(
        f"[WildChat-v4] 420d window (2023-11-20..2025-01-12): {win_convs:,} convs, "
        f"{win_users:,} unique hashed_ip, {win_active_days}/420 active days, "
        f"missing dates: {missing_dates}"
    )

    user_conv_counts = df_win.groupby("hashed_ip").size().sort_values(ascending=False)
    p50_convs = float(np.percentile(user_conv_counts, 50))
    p90_convs = float(np.percentile(user_conv_counts, 90))
    p99_convs = float(np.percentile(user_conv_counts, 99))
    max_convs = int(user_conv_counts.max())

    all_users_sorted = user_conv_counts.index.to_numpy()
    u_to_cohort = {u: (idx % K_COHORTS) for idx, u in enumerate(all_users_sorted)}
    df_win["cohort_id"] = df_win["hashed_ip"].map(u_to_cohort)

    print(f"[WildChat-v4] Precomputing exact sufficient statistics across all {win_users:,} users in K={K_COHORTS} disjoint cohorts...")
    t_pre = time.perf_counter()
    cohort_stats: List[Dict[str, Any]] = []
    for k in range(K_COHORTS):
        df_c = df_win[df_win["cohort_id"] == k]
        st = precompute_cohort_suff_stats(df_c)
        cohort_stats.append(st)
    print(f"[WildChat-v4] Precomputed K={K_COHORTS} cohorts in {time.perf_counter() - t_pre:.2f}s.")

    dedup_ratios_7d, dedup_ratios_30d, dedup_ratios_90d = [], [], []
    cat_clip_bias_30d, req_clip_bias_1d = [], []
    for st in cohort_stats:
        raw_r = st["daily_requests_raw"].sum()
        clip_r = st["daily_requests_clipped"].sum()
        req_clip_bias_1d.append((raw_r - clip_r) / raw_r * 100.0)
        for w, lst in [(7, dedup_ratios_7d), (30, dedup_ratios_30d), (90, dedup_ratios_90d)]:
            dw = st["distinct_windows"][w]
            true_s = dw["true_svc"][w - 1 :].sum()
            sum_d = dw["sum_daily_svc"][w - 1 :].sum()
            lst.append((1.0 - true_s / sum_d) * 100.0)
        dw30 = st["distinct_windows"][30]
        tc = dw30["true_cat"][29:].sum()
        cc = dw30["clipped_cat"][29:].sum()
        cat_clip_bias_30d.append((tc - cc) / tc * 100.0)

    dataset_audit_df = pd.DataFrame(
        [
            {
                "dataset_name": "allenai/WildChat-4.8M (wildchat_all_shards_4cols.parquet)",
                "git_blob": GIT_BLOB_SPEC,
                "license": "ODC-By 1.0 / AI2 ImpACT Low-Risk",
                "full_convs": total_convs,
                "full_unique_hashed_ip": total_users,
                "full_calendar_span": "2023-04-09 to 2025-07-31 (845d, 809 active)",
                "win420_calendar_span": "2023-11-20 to 2025-01-12 (420d, 418 active, 2 outage days)",
                "win420_convs": win_convs,
                "win420_unique_hashed_ip": win_users,
                "users_per_cohort_mean": float(win_users / K_COHORTS),
                "gapfree_subspan": "2024-01-20 to 2025-01-12 (359 consecutive active days)",
                "user_convs_p50": p50_convs,
                "user_convs_p90": p90_convs,
                "user_convs_p99": p99_convs,
                "user_convs_max": max_convs,
                "mean_dedup_reduction_7d_pct": float(np.mean(dedup_ratios_7d)),
                "mean_dedup_reduction_30d_pct": float(np.mean(dedup_ratios_30d)),
                "mean_dedup_reduction_90d_pct": float(np.mean(dedup_ratios_90d)),
                "mean_daily_req_clip_bias_pct": float(np.mean(req_clip_bias_1d)),
                "mean_cat30d_k3_clip_bias_pct": float(np.mean(cat_clip_bias_30d)),
            }
        ]
    )
    dataset_audit_df.to_csv(out_dir / "wildchat_dataset_audit_summary.csv", index=False)

    summary_rows = []
    paired_rows = []

    gapfree_mask = np.zeros(HORIZON_DAYS, dtype=bool)
    gapfree_mask[GAPFREE_START_IDX - WIN_START_IDX :] = True

    eval_configs = [
        ("DistinctHeavy", "Week1_v2", "B_BudgetOnly"),
        ("DistinctHeavy", "Original", "B_BudgetOnly"),
        ("Default", "Week1_v2", "A_Baseline"),
    ]

    for wl_name, curve_name, base_grp in eval_configs:
        curve = CURVE_MAP[curve_name]
        weights = WORKLOAD_PRIORITIES[wl_name]
        planner = JointCalendarPlanner(
            PlannerConfig(commitment=curve, workload_weights=weights)
        )
        plan_base = planner.plan_group_b() if base_grp == "B_BudgetOnly" else planner.plan_group_a()
        plan_d = planner.plan_group_d()
        planner_sun = JointCalendarPlanner(
            PlannerConfig(
                commitment=curve,
                workload_weights=weights,
                fixed_phi_7=3,
                fixed_phi_30=0,
                fixed_phi_90=0,
            )
        )
        plan_sun = planner_sun.plan_group_d()

        # Calibrate Tree-2l, Banded-MF (b=30), and Banded-MF (b=8) on remaining additive envelope
        n7 = planner.n7_tables[plan_d.phases[0]]
        n30 = planner.n30_tables[plan_d.phases[1]]
        n90 = planner.n90_tables[plan_d.phases[2]]
        l_dist = np.max(
            n7 * plan_d.distinct_costs[0]
            + n30 * plan_d.distinct_costs[1]
            + n90 * plan_d.distinct_costs[2],
            axis=1,
        )[:HORIZON_DAYS]
        rem_env = np.array(
            [float(curve.rho_bar(h)) - float(l_dist[h - 1]) for h in range(1, HORIZON_DAYS + 1)],
            dtype=np.float64,
        )
        eta = 25.0 / 26.0
        rem_env_req = eta * rem_env

        tree_2l = TreeBaseline(variant="Tree-2l", T=HORIZON_DAYS, L=8, C=1.0)
        tree_ratios = [
            0.5 * tree_2l.get_worst_case_rho_term(h) / float(rem_env_req[h - 1])
            for h in range(1, HORIZON_DAYS + 1)
        ]
        beta_tree = float(np.sqrt(max(tree_ratios)))
        tree_gl_proj = np.linalg.solve(
            tree_2l.M_base,
            tree_2l.A.T * (1.0 / tree_2l.w_levels),
        )
        tree_node_sigmas = beta_tree * C_R * np.sqrt(tree_2l.w_levels)

        banded_30 = BandedToeplitzBaseline(T=HORIZON_DAYS, bandwidth=30, alpha=0.50, C=1.0)
        b30_cal = banded_30.calibrate_to_multi_horizon_envelope(rem_env_req, n_days=HORIZON_DAYS)
        sigma_b30 = float(b30_cal["beta_env"]) * C_R

        banded_8 = BandedToeplitzBaseline(T=HORIZON_DAYS, bandwidth=8, alpha=0.25, C=1.0)
        b8_cal = banded_8.calibrate_to_multi_horizon_envelope(rem_env_req, n_days=HORIZON_DAYS)
        sigma_b8 = float(b8_cal["beta_env"]) * C_R

        cohort_base_metrics: List[Dict[str, float]] = []
        cohort_d_metrics: List[Dict[str, float]] = []
        cohort_sun_metrics: List[Dict[str, float]] = []
        cohort_d_gapfree_metrics: List[Dict[str, float]] = []
        cohort_tree_rmse30: List[float] = []
        cohort_banded30_rmse30: List[float] = []
        cohort_banded8_rmse30: List[float] = []

        for k in range(K_COHORTS):
            mc_b, mc_d, mc_sun, mc_d_gf = [], [], [], []
            mc_tree30, mc_band30, mc_band8 = [], [], []
            for m in range(N_MC):
                mb = evaluate_schedule_on_cohort(cohort_stats[k], plan_base, k, m, eval_mask=None)
                md = evaluate_schedule_on_cohort(cohort_stats[k], plan_d, k, m, eval_mask=None)
                msun = evaluate_schedule_on_cohort(cohort_stats[k], plan_sun, k, m, eval_mask=None)
                md_gf = evaluate_schedule_on_cohort(cohort_stats[k], plan_d, k, m, eval_mask=gapfree_mask)
                mc_b.append(mb)
                mc_d.append(md)
                mc_sun.append(msun)
                mc_d_gf.append(md_gf)

                rng_t = np.random.default_rng(derive_crn_seed_v4(k, m, 999, 50))
                rng_b30 = np.random.default_rng(derive_crn_seed_v4(k, m, 999, 60))
                rng_b8 = np.random.default_rng(derive_crn_seed_v4(k, m, 999, 70))
                raw_r = cohort_stats[k]["daily_requests_raw"]
                clip_r = cohort_stats[k]["daily_requests_clipped"]

                z_tree = rng_t.normal(0.0, 1.0, size=len(tree_2l.w_levels)) * tree_node_sigmas
                x_hat_tree = clip_r + (tree_gl_proj @ z_tree)

                z_b30 = rng_b30.normal(0.0, sigma_b30, size=HORIZON_DAYS)
                x_hat_b30 = clip_r + (banded_30.C_inv @ z_b30)

                z_b8 = rng_b8.normal(0.0, sigma_b8, size=HORIZON_DAYS)
                x_hat_b8 = clip_r + (banded_8.C_inv @ z_b8)

                t_sq, b30_sq, b8_sq = [], [], []
                for d in range(29, HORIZON_DAYS):
                    true_w30 = float(raw_r[d - 29 : d + 1].sum())
                    est_t30 = max(0.0, float(x_hat_tree[d - 29 : d + 1].sum()))
                    est_b30 = max(0.0, float(x_hat_b30[d - 29 : d + 1].sum()))
                    est_b8 = max(0.0, float(x_hat_b8[d - 29 : d + 1].sum()))
                    t_sq.append((est_t30 - true_w30) ** 2)
                    b30_sq.append((est_b30 - true_w30) ** 2)
                    b8_sq.append((est_b8 - true_w30) ** 2)
                mc_tree30.append(float(np.sqrt(np.mean(t_sq))))
                mc_band30.append(float(np.sqrt(np.mean(b30_sq))))
                mc_band8.append(float(np.sqrt(np.mean(b8_sq))))

            keys = list(mc_b[0].keys())
            cohort_base_metrics.append({key: float(np.mean([x[key] for x in mc_b])) for key in keys})
            cohort_d_metrics.append({key: float(np.mean([x[key] for x in mc_d])) for key in keys})
            cohort_sun_metrics.append({key: float(np.mean([x[key] for x in mc_sun])) for key in keys})
            cohort_d_gapfree_metrics.append({key: float(np.mean([x[key] for x in mc_d_gf])) for key in keys})
            cohort_tree_rmse30.append(float(np.mean(mc_tree30)))
            cohort_banded30_rmse30.append(float(np.mean(mc_band30)))
            cohort_banded8_rmse30.append(float(np.mean(mc_band8)))

        metric_specs = [
            ("distinct_7d_active_mape", "Distinct7d Active Release MAPE (%)"),
            ("distinct_30d_active_mape", "Distinct30d Active Release MAPE (%)"),
            ("distinct_90d_active_mape", "Distinct90d Active Release MAPE (%)"),
            ("distinct_7d_release_rmse", "Distinct7d Release RMSE (users)"),
            ("distinct_30d_release_rmse", "Distinct30d Release RMSE (users)"),
            ("distinct_90d_release_rmse", "Distinct90d Release RMSE (users)"),
            ("req_1d_rmse", "DailyRequests 1d RMSE"),
            ("req_30d_rmse", "Requests 30d RMSE"),
        ]
        raw_pvals = []
        temp_records = []
        contrast_str = f"D_JointPlanning vs {base_grp}"
        for m_key, m_label in metric_specs:
            b_vals = np.array([c[m_key] for c in cohort_base_metrics], dtype=np.float64)
            d_vals = np.array([c[m_key] for c in cohort_d_metrics], dtype=np.float64)
            diffs = d_vals - b_vals
            mean_b = float(np.mean(b_vals))
            mean_d = float(np.mean(d_vals))
            mean_diff = float(np.mean(diffs))
            se_diff = float(np.std(diffs, ddof=1) / np.sqrt(K_COHORTS))
            t_crit = float(stats.t.ppf(0.975, df=K_COHORTS - 1))
            ci_low = mean_diff - t_crit * se_diff
            ci_high = mean_diff + t_crit * se_diff
            _, p_t = stats.ttest_rel(d_vals, b_vals)
            try:
                _, p_w = stats.wilcoxon(diffs)
            except ValueError:
                p_w = 1.0
            neg_cnt = int(np.sum(diffs < 0))
            raw_pvals.append(float(p_t))
            temp_records.append(
                {
                    "workload": wl_name,
                    "curve": curve_name,
                    "contrast": contrast_str,
                    "metric_key": m_key,
                    "metric_label": m_label,
                    "K_cohorts": K_COHORTS,
                    "base_mean": mean_b,
                    "trt_mean": mean_d,
                    "paired_diff": mean_diff,
                    "ci95_low": ci_low,
                    "ci95_high": ci_high,
                    "p_t_raw": float(p_t),
                    "p_wilcoxon": float(p_w),
                    "cohorts_d_lt_base": neg_cnt,
                }
            )

        m_tests = len(raw_pvals)
        order = np.argsort(raw_pvals)
        holm_p = np.zeros(m_tests, dtype=np.float64)
        running_max = 0.0
        for rank_idx, orig_idx in enumerate(order):
            adj = min(1.0, raw_pvals[orig_idx] * (m_tests - rank_idx))
            running_max = max(running_max, adj)
            holm_p[orig_idx] = running_max
        for idx, rec in enumerate(temp_records):
            rec["p_holm"] = float(holm_p[idx])
            paired_rows.append(rec)

        summary_rows.append(
            {
                "workload": wl_name,
                "curve": curve_name,
                "contrast": contrast_str,
                "base_obj": float(plan_base.weighted_workload_error),
                "d_obj": float(plan_d.weighted_workload_error),
                "obj_reduction_pct": float(
                    (plan_base.weighted_workload_error - plan_d.weighted_workload_error)
                    / plan_base.weighted_workload_error
                    * 100.0
                ),
                "base_d7_active_mape": float(np.mean([c["distinct_7d_active_mape"] for c in cohort_base_metrics])),
                "d_d7_active_mape": float(np.mean([c["distinct_7d_active_mape"] for c in cohort_d_metrics])),
                "d_gapfree_d7_active_mape": float(np.mean([c["distinct_7d_active_mape"] for c in cohort_d_gapfree_metrics])),
                "base_d30_active_mape": float(np.mean([c["distinct_30d_active_mape"] for c in cohort_base_metrics])),
                "d_d30_active_mape": float(np.mean([c["distinct_30d_active_mape"] for c in cohort_d_metrics])),
                "d_gapfree_d30_active_mape": float(np.mean([c["distinct_30d_active_mape"] for c in cohort_d_gapfree_metrics])),
                "base_d90_active_mape": float(np.mean([c["distinct_90d_active_mape"] for c in cohort_base_metrics])),
                "d_d90_active_mape": float(np.mean([c["distinct_90d_active_mape"] for c in cohort_d_metrics])),
                "base_d7_rmse": float(np.mean([c["distinct_7d_release_rmse"] for c in cohort_base_metrics])),
                "d_d7_rmse": float(np.mean([c["distinct_7d_release_rmse"] for c in cohort_d_metrics])),
                "base_d30_rmse": float(np.mean([c["distinct_30d_release_rmse"] for c in cohort_base_metrics])),
                "d_d30_rmse": float(np.mean([c["distinct_30d_release_rmse"] for c in cohort_d_metrics])),
                "base_d90_rmse": float(np.mean([c["distinct_90d_release_rmse"] for c in cohort_base_metrics])),
                "d_d90_rmse": float(np.mean([c["distinct_90d_release_rmse"] for c in cohort_d_metrics])),
                "d_sun_staleness_30d": float(np.mean([c["distinct_30d_sunday_staleness"] for c in cohort_d_metrics])),
                "sun_pinned_staleness_30d": float(np.mean([c["distinct_30d_sunday_staleness"] for c in cohort_sun_metrics])),
                "sdg_req30d_rmse": float(np.mean([c["req_30d_rmse"] for c in cohort_d_metrics])),
                "tree2l_req30d_rmse": float(np.mean(cohort_tree_rmse30)),
                "banded30_req30d_rmse": float(np.mean(cohort_banded30_rmse30)),
                "banded8_req30d_rmse": float(np.mean(cohort_banded8_rmse30)),
            }
        )

    paired_df = pd.DataFrame(paired_rows)
    summary_df = pd.DataFrame(summary_rows)
    paired_df.to_csv(out_dir / "wildchat_k8_cohorts_paired_d_vs_b.csv", index=False)
    summary_df.to_csv(out_dir / "wildchat_k8_cohorts_summary.csv", index=False)

    print("\n[WildChat-v4] Dataset Audit Summary:")
    print(dataset_audit_df.to_string(index=False))
    print("\n[WildChat-v4] Paired Cohort Comparison (K=8 disjoint cohorts across all 895,148 users):")
    print(paired_df.to_string(index=False))
    print("\n[WildChat-v4] Summary Table (including 359d gap-free sub-span, Sunday-pinned, Tree-2l, Banded-MF):")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
