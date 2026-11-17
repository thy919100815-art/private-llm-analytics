"""Real-Data Validation on AllenAI WildChat-4.8M (v5 — Audit-Remediated).

Fixes all P0 WildChat audit items from PR #11 (01ae905):
1. Exact UTC Calendar Weekday Mapping:
   - 2023-11-20 is Monday (`date.weekday() == 0`). Fixed-Sunday queries use
     `calendar_dates[d].weekday() == 6` (`d % 7 == 6`), NOT `d % 7 == 3` (Thursday).
   - Explicitly notes that `phi = 0` in `[2023-11-20..]` is Monday, while Sunday
     is `phi = 6 (mod 7)` (`PlannerConfig(fixed_phi_30=6)`).
2. Exact Top-k=3 Category Truncation Matching `RollingDistinctState`:
   - Ranks each user's categories within `[d - W + 1, d]` by `(-in_window_count, cat_idx)`
     (frequency descending, category index ascending), NOT `sorted(cset)[:3]`.
3. Production Discrete Gaussian Mechanism (`sample_discrete_gaussian`):
   - Replaces continuous `rng.normal` with exact `sample_discrete_gaussian`
     matching production `SealJob` and `CalendarJob`.
4. Public Reproducible Input & Non-PII Outputs:
   - Loads via `prepare_wildchat_public_manifest.load_or_fetch_4cols_parquet`
     pinned to HF revision `c827c6df8fcf008219ffaffa4d1dd77491099367`
     (SHA-256 `581e68ce482d506a3985fcd4f8cc71c3ba349d5b021164629235d885a3476a2a`).
   - Explicitly treats the 20 categories (`5 model_families x 4 UTC_6h_quadrants`)
     as an interaction-category proxy, `hashed_ip` as an IP pseudonym, and
     1 conversation record as != 1 HTTP API request.
5. Explicit Conditional Inference Scope & Sensitivity Analysis:
   - Evaluates `K=8` disjoint cohorts (stratified `rank % 8`) x 5 CRN trials,
     explicitly framing `df=7` inference as conditional on the fixed 420d trace.
   - Runs a sensitivity check across `K in {4, 8, 16}` and deterministic
     `blake2b(hashed_ip) % 8` random user-hash partitioning.
6. Writes all outputs to `results/sdg_mh/wildchat_real_validation_v5/`.
"""

from __future__ import annotations

from datetime import date, timedelta
import hashlib
import io
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from prepare_wildchat_public_manifest import (  # noqa: E402
    EXPECTED_4COLS_SHA256,
    load_or_fetch_4cols_parquet,
)
from private_llm_analytics.sdg_mh.baselines import (  # noqa: E402
    BandedToeplitzBaseline,
    TreeBaseline,
)
from private_llm_analytics.sdg_mh.calendar_accounting import (  # noqa: E402
    ORIGINAL_PLANNER_CURVE,
    WEEK1_V2_CURVE,
)
from private_llm_analytics.sdg_mh.discrete_gaussian import sample_discrete_gaussian  # noqa: E402
from private_llm_analytics.sdg_mh.joint_planner import (  # noqa: E402
    JointCalendarPlanner,
    PlannerConfig,
)

OUT_DIR = ROOT / "results" / "sdg_mh" / "wildchat_real_validation_v5"
WORKLOAD_PRIORITIES = {
    "Default": (0.40, 0.25, 0.25, 0.10),
    "DistinctHeavy": (0.10, 0.40, 0.40, 0.10),
}
CURVE_MAP = {
    "Week1_v2": WEEK1_V2_CURVE,
    "Original": ORIGINAL_PLANNER_CURVE,
}

NUM_SERVICES = 5
NUM_CATEGORIES = 20  # interaction-category proxy: 5 model families x 4 UTC 6h quadrants
MAX_CATEGORIES_PER_USER = 3
C_R = 50
HORIZON_DAYS = 420
WIN_START_DATE = date(2023, 11, 20)  # Monday (weekday() == 0)
DATASET_ORIGIN = date(2023, 4, 9)
WIN_START_IDX = (WIN_START_DATE - DATASET_ORIGIN).days  # 225
WIN_END_IDX = WIN_START_IDX + HORIZON_DAYS - 1        # 644 (2025-01-12)
GAPFREE_START_DATE = date(2024, 1, 20)
GAPFREE_START_REL = (GAPFREE_START_DATE - WIN_START_DATE).days  # 61 (359 consecutive active days)

K_COHORTS = 8
N_MC = 5


def derive_crn_seed_v5(cohort_id: int, mc_id: int, day: int, channel_code: int) -> int:
    payload = f"wildchat_v5|{cohort_id}|{mc_id}|{day}|{channel_code}".encode("utf-8")
    return int.from_bytes(hashlib.blake2b(payload, digest_size=16).digest(), "little")


def compute_exact_window_distinct_from_pairs(
    df_pairs: pd.DataFrame,
    entity_col: str,
    num_entities: int,
    windows: Tuple[int, ...] = (7, 30, 90),
) -> Tuple[Dict[int, np.ndarray], Dict[int, np.ndarray]]:
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
        np.add.at(diff, (start_e[valid], e_arr[valid]), 1.0)
        np.add.at(diff, (end_e[valid] + 1, e_arr[valid]), -1.0)
        active_win = np.cumsum(diff[:-1], axis=0)
        active_win[: w - 1] = 0.0
        win_distinct[w] = active_win

        csum = np.vstack([np.zeros((1, num_entities)), np.cumsum(daily_active, axis=0)])
        rsum = np.zeros((HORIZON_DAYS, num_entities), dtype=np.float64)
        rsum[w - 1 :] = csum[w:] - csum[: HORIZON_DAYS - w + 1]
        win_daily_sum[w] = rsum

    return win_distinct, win_daily_sum


def precompute_cohort_suff_stats(df_cohort: pd.DataFrame) -> Dict[str, Any]:
    """Precompute scalar daily additive counts (C_R=50) and exact RollingDistinctState-aligned window distinct counts."""
    u_codes, _ = pd.factorize(df_cohort["hashed_ip"], sort=False)
    df_c = df_cohort.assign(u_id=u_codes)

    # 1. Daily additive conversation counts (clipped at C_R=50 per user-day)
    grp_req = df_c.groupby(["rel_day", "u_id"], sort=False).size().reset_index(name="cnt")
    daily_requests_raw = np.zeros(HORIZON_DAYS, dtype=np.float64)
    daily_requests_clipped = np.zeros(HORIZON_DAYS, dtype=np.float64)

    for d, val in grp_req.groupby("rel_day")["cnt"].sum().items():
        daily_requests_raw[int(d)] = float(val)
    grp_req["clip_cnt"] = np.minimum(grp_req["cnt"], C_R)
    for d, val in grp_req.groupby("rel_day")["clip_cnt"].sum().items():
        daily_requests_clipped[int(d)] = float(val)

    # 2. Exact service distinct counts
    svc_win_distinct, svc_win_daily_sum = compute_exact_window_distinct_from_pairs(
        df_c, "service_idx", NUM_SERVICES, (7, 30, 90)
    )

    # 3. Exact category distinct counts matching RollingDistinctState:
    #    - Users with <= 3 lifetime categories are never clipped in any window.
    #    - Users with > 3 lifetime categories are ranked by (-in_window_count, cat_idx)[:3].
    cat_day_counts = (
        df_c.groupby(["u_id", "rel_day", "cat_idx"], sort=False)
        .size()
        .reset_index(name="cnt")
    )
    u_cat_nunique = cat_day_counts.groupby("u_id")["cat_idx"].nunique()
    multi_u_ids = set(u_cat_nunique[u_cat_nunique > MAX_CATEGORIES_PER_USER].index)

    cat_unclipped_all, _ = compute_exact_window_distinct_from_pairs(
        cat_day_counts, "cat_idx", NUM_CATEGORIES, (7, 30, 90)
    )
    cat_multi_df = cat_day_counts[cat_day_counts["u_id"].isin(multi_u_ids)]
    cat_unclipped_multi, _ = compute_exact_window_distinct_from_pairs(
        cat_multi_df, "cat_idx", NUM_CATEGORIES, (7, 30, 90)
    )

    day_multi_user_cat_counts: List[Dict[int, List[Tuple[int, int]]]] = [{} for _ in range(HORIZON_DAYS)]
    for row in cat_multi_df.itertuples(index=False):
        day_multi_user_cat_counts[int(row.rel_day)].setdefault(int(row.u_id), []).append(
            (int(row.cat_idx), int(row.cnt))
        )

    distinct_windows: Dict[int, Dict[str, np.ndarray]] = {}
    for w in (7, 30, 90):
        clipped_multi = np.zeros((HORIZON_DAYS, NUM_CATEGORIES), dtype=np.float64)
        for d in range(w - 1, HORIZON_DAYS):
            u_cat_freq: Dict[int, Dict[int, int]] = {}
            for tau in range(d - w + 1, d + 1):
                for u, pairs in day_multi_user_cat_counts[tau].items():
                    cmap = u_cat_freq.setdefault(u, {})
                    for c_idx, cnt in pairs:
                        cmap[c_idx] = cmap.get(c_idx, 0) + cnt
            row_counts = np.zeros(NUM_CATEGORIES, dtype=np.float64)
            for cmap in u_cat_freq.values():
                # Exact RollingDistinctState rule: (-in_window_count, cat_idx)
                top_k = sorted(cmap.items(), key=lambda item: (-item[1], item[0]))[:MAX_CATEGORIES_PER_USER]
                for c_idx, _ in top_k:
                    row_counts[c_idx] += 1.0
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
    eta = 25.0 / 26.0
    delta_sq = float(NUM_SERVICES + 2 * MAX_CATEGORIES_PER_USER)
    daily_d = float(plan_res.daily_d)
    dcosts = tuple(plan_res.distinct_costs)
    return {
        "phases": tuple(plan_res.phases),
        "daily_d": daily_d,
        "distinct_costs": dcosts,
        "objective_value": float(plan_res.weighted_workload_error),
        "sigma_req": float(C_R / np.sqrt(2.0 * max(1e-12, eta * daily_d))),
        "sigma_d7": float(np.sqrt(delta_sq / (2.0 * max(1e-12, dcosts[0])))),
        "sigma_d30": float(np.sqrt(delta_sq / (2.0 * max(1e-12, dcosts[1])))),
        "sigma_d90": float(np.sqrt(delta_sq / (2.0 * max(1e-12, dcosts[2])))),
    }


def evaluate_schedule_on_cohort(
    suff: Dict[str, Any],
    plan_res: Any,
    cohort_id: int,
    mc_id: int,
    calendar_weekdays: List[int],
    eval_mask: np.ndarray | None = None,
) -> Dict[str, float]:
    """Evaluate SDG-MH schedule using production Discrete Gaussian noise and exact calendar Sundays."""
    params = extract_plan_parameters(plan_res)
    sigma_req = params["sigma_req"]
    p7, p30, p90 = params["phases"]

    raw_req = suff["daily_requests_raw"]
    clip_req = suff["daily_requests_clipped"]

    # Production Discrete Gaussian noise for daily additive requests
    raw_noise_req = np.zeros(HORIZON_DAYS, dtype=np.float64)
    for d in range(HORIZON_DAYS):
        rng = np.random.default_rng(derive_crn_seed_v5(cohort_id, mc_id, d, 1))
        raw_noise_req[d] = float(sample_discrete_gaussian(sigma=sigma_req, rng=rng))

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
        (7, 7, p7, params["sigma_d7"], 10),
        (30, 7, p30, params["sigma_d30"], 20),
        (90, 30, p90, params["sigma_d90"], 30),
    ]:
        dw = suff["distinct_windows"][w]
        true_svc = dw["true_svc"]
        true_cat = dw["true_cat"]
        clip_cat = dw["clipped_cat"]

        rel_active_mape_list = []
        rel_rmse_list = []
        latest_active_mape_list = []
        sunday_latest_active_mape_list = []
        sunday_staleness_list = []

        latest_est_svc = None
        latest_est_cat = None
        latest_rel_day = None

        for d in range(w - 1, HORIZON_DAYS):
            if (d - phase) % period == 0:
                rng_s = np.random.default_rng(derive_crn_seed_v5(cohort_id, mc_id, d, ch_code))
                rng_c = np.random.default_rng(derive_crn_seed_v5(cohort_id, mc_id, d, ch_code + 1))
                n_s = sample_discrete_gaussian(sigma=sigma_d, size=NUM_SERVICES, rng=rng_s)
                n_c = sample_discrete_gaussian(sigma=sigma_d, size=NUM_CATEGORIES, rng=rng_c)
                est_svc = np.maximum(0.0, true_svc[d] + n_s.astype(np.float64))
                est_cat = np.maximum(0.0, clip_cat[d] + n_c.astype(np.float64))
                latest_est_svc = est_svc
                latest_est_cat = est_cat
                latest_rel_day = d

                if eval_mask is None or eval_mask[d]:
                    true_vec = np.concatenate([true_svc[d], true_cat[d]])
                    est_vec = np.concatenate([est_svc, est_cat])
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
                    val_mape = float(
                        np.mean(np.abs(est_vec[active_idx] - true_vec[active_idx]) / true_vec[active_idx]) * 100.0
                    )
                    latest_active_mape_list.append(val_mape)
                    if calendar_weekdays[d] == 6:  # Authentic UTC Sunday!
                        sunday_latest_active_mape_list.append(val_mape)
                if calendar_weekdays[d] == 6:  # Authentic UTC Sunday!
                    sunday_staleness_list.append(float(d - latest_rel_day))

        metrics[f"distinct_{w}d_active_mape"] = float(np.mean(rel_active_mape_list))
        metrics[f"distinct_{w}d_release_rmse"] = float(np.sqrt(np.mean(rel_rmse_list)))
        metrics[f"distinct_{w}d_latest_active_mape"] = float(np.mean(latest_active_mape_list))
        metrics[f"distinct_{w}d_sunday_latest_active_mape"] = float(np.mean(sunday_latest_active_mape_list))
        metrics[f"distinct_{w}d_sunday_staleness"] = float(np.mean(sunday_staleness_list))

    return metrics


def holm_bonferroni(raw_pvals: List[float]) -> List[float]:
    m_tests = len(raw_pvals)
    order = np.argsort(raw_pvals)
    holm_p = np.zeros(m_tests, dtype=np.float64)
    running_max = 0.0
    for rank_idx, orig_idx in enumerate(order):
        adj = min(1.0, raw_pvals[orig_idx] * (m_tests - rank_idx))
        running_max = max(running_max, adj)
        holm_p[orig_idx] = running_max
    return [float(x) for x in holm_p]


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    calendar_dates = [WIN_START_DATE + timedelta(days=d) for d in range(HORIZON_DAYS)]
    calendar_weekdays = [dt.weekday() for dt in calendar_dates]
    assert calendar_weekdays[0] == 0  # 2023-11-20 is Monday
    assert calendar_weekdays[6] == 6  # 2023-11-26 is Sunday

    raw_bytes, source_mode = load_or_fetch_4cols_parquet(
        ROOT / ".cache" / "wildchat_all_shards_4cols.parquet",
        download_from_hf=False,
    )
    sha256_hex = hashlib.sha256(raw_bytes).hexdigest()
    assert sha256_hex == EXPECTED_4COLS_SHA256
    df = pd.read_parquet(io.BytesIO(raw_bytes))

    ts = pd.to_datetime(df["timestamp"])
    df["date"] = ts.dt.date
    df["utc_quad"] = ts.dt.hour // 6
    df["cal_idx"] = (df["date"] - DATASET_ORIGIN).apply(lambda td: td.days)

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
    user_conv_counts = df_win.groupby("hashed_ip").size().sort_values(ascending=False)
    all_users_sorted = user_conv_counts.index.to_numpy()

    # Primary K=8 stratified rank partition
    u_to_cohort = {u: (idx % K_COHORTS) for idx, u in enumerate(all_users_sorted)}
    df_win["cohort_id"] = df_win["hashed_ip"].map(u_to_cohort)

    print(f"[WildChat-v5] Precomputing exact RollingDistinctState-aligned stats (K={K_COHORTS}, {win_users:,} users)...")
    t_pre = time.perf_counter()
    cohort_stats = [precompute_cohort_suff_stats(df_win[df_win["cohort_id"] == k]) for k in range(K_COHORTS)]
    print(f"[WildChat-v5] Precomputed K={K_COHORTS} cohorts in {time.perf_counter() - t_pre:.2f}s.")

    dedup_7, dedup_30, dedup_90, req_clip_b, cat_clip_b = [], [], [], [], []
    for st in cohort_stats:
        raw_r = st["daily_requests_raw"].sum()
        clip_r = st["daily_requests_clipped"].sum()
        req_clip_b.append((raw_r - clip_r) / raw_r * 100.0)
        for w, lst in [(7, dedup_7), (30, dedup_30), (90, dedup_90)]:
            dw = st["distinct_windows"][w]
            lst.append((1.0 - dw["true_svc"][w - 1 :].sum() / dw["sum_daily_svc"][w - 1 :].sum()) * 100.0)
        dw30 = st["distinct_windows"][30]
        cat_clip_b.append((dw30["true_cat"][29:].sum() - dw30["clipped_cat"][29:].sum()) / dw30["true_cat"][29:].sum() * 100.0)

    audit_df = pd.DataFrame([{
        "source_mode": source_mode,
        "input_sha256": sha256_hex,
        "win420_start": str(WIN_START_DATE),
        "win420_start_weekday": "Monday",
        "win420_end": "2025-01-12",
        "win420_convs": win_convs,
        "win420_unique_hashed_ip": win_users,
        "mean_dedup_reduction_7d_pct": float(np.mean(dedup_7)),
        "mean_dedup_reduction_30d_pct": float(np.mean(dedup_30)),
        "mean_dedup_reduction_90d_pct": float(np.mean(dedup_90)),
        "mean_daily_req_clip_bias_pct": float(np.mean(req_clip_b)),
        "mean_cat30d_k3_clip_bias_pct": float(np.mean(cat_clip_b)),
    }])
    audit_df.to_csv(OUT_DIR / "wildchat_dataset_audit_summary_v5.csv", index=False)

    gapfree_mask = np.zeros(HORIZON_DAYS, dtype=bool)
    gapfree_mask[GAPFREE_START_REL:] = True

    eval_configs = [
        ("DistinctHeavy", "Week1_v2", "B_BudgetOnly"),
        ("DistinctHeavy", "Original", "B_BudgetOnly"),
        ("Default", "Week1_v2", "A_Baseline"),
    ]

    summary_rows, paired_rows = [], []
    for wl_name, curve_name, base_grp in eval_configs:
        curve = CURVE_MAP[curve_name]
        weights = WORKLOAD_PRIORITIES[wl_name]
        planner = JointCalendarPlanner(PlannerConfig(commitment=curve, workload_weights=weights))
        plan_base = planner.plan_group_b() if base_grp == "B_BudgetOnly" else planner.plan_group_a()
        plan_d = planner.plan_group_d()
        # Sunday-pinned 30d schedule: in 2023-11-20 (Monday) origin, Sunday is phi_30 = 6 (mod 7)!
        planner_sun = JointCalendarPlanner(
            PlannerConfig(commitment=curve, workload_weights=weights, fixed_phi_30=6, fixed_phi_90=0)
        )
        plan_sun = planner_sun.plan_group_d()

        cohort_b, cohort_d, cohort_sun, cohort_d_gf = [], [], [], []
        for k in range(K_COHORTS):
            mc_b, mc_d, mc_sun, mc_gf = [], [], [], []
            for m in range(N_MC):
                mc_b.append(evaluate_schedule_on_cohort(cohort_stats[k], plan_base, k, m, calendar_weekdays))
                mc_d.append(evaluate_schedule_on_cohort(cohort_stats[k], plan_d, k, m, calendar_weekdays))
                mc_sun.append(evaluate_schedule_on_cohort(cohort_stats[k], plan_sun, k, m, calendar_weekdays))
                mc_gf.append(evaluate_schedule_on_cohort(cohort_stats[k], plan_d, k, m, calendar_weekdays, eval_mask=gapfree_mask))
            keys = list(mc_b[0].keys())
            cohort_b.append({key: float(np.mean([x[key] for x in mc_b])) for key in keys})
            cohort_d.append({key: float(np.mean([x[key] for x in mc_d])) for key in keys})
            cohort_sun.append({key: float(np.mean([x[key] for x in mc_sun])) for key in keys})
            cohort_d_gf.append({key: float(np.mean([x[key] for x in mc_gf])) for key in keys})

        metric_specs = [
            ("distinct_7d_active_mape", "Distinct7d Active Release MAPE (%)"),
            ("distinct_30d_active_mape", "Distinct30d Active Release MAPE (%)"),
            ("distinct_90d_active_mape", "Distinct90d Active Release MAPE (%)"),
            ("distinct_7d_release_rmse", "Distinct7d Release RMSE (users)"),
            ("distinct_30d_release_rmse", "Distinct30d Release RMSE (users)"),
            ("distinct_90d_release_rmse", "Distinct90d Release RMSE (users)"),
            ("distinct_30d_latest_active_mape", "Distinct30d All-Day Latest Active MAPE (%)"),
            ("distinct_30d_sunday_latest_active_mape", "Distinct30d Sunday Latest Active MAPE (%)"),
            ("req_1d_rmse", "DailyRequests 1d RMSE"),
            ("req_30d_rmse", "Requests 30d RMSE"),
        ]
        contrast_str = f"D_JointPlanning vs {base_grp}"
        raw_pvals, tmp_rec = [], []
        t_crit = float(stats.t.ppf(0.975, df=K_COHORTS - 1))
        for m_key, m_label in metric_specs:
            b_vals = np.array([c[m_key] for c in cohort_b], dtype=np.float64)
            d_vals = np.array([c[m_key] for c in cohort_d], dtype=np.float64)
            diffs = d_vals - b_vals
            mean_diff = float(np.mean(diffs))
            se_diff = float(np.std(diffs, ddof=1) / np.sqrt(K_COHORTS))
            _, p_t = stats.ttest_rel(d_vals, b_vals)
            try:
                _, p_w = stats.wilcoxon(diffs)
            except ValueError:
                p_w = 1.0
            raw_pvals.append(float(p_t))
            tmp_rec.append({
                "workload": wl_name,
                "curve": curve_name,
                "contrast": contrast_str,
                "inference_scope": "Conditional on fixed 420d WildChat trace (K=8 disjoint cohorts, df=7)",
                "metric_key": m_key,
                "metric_label": m_label,
                "K_cohorts": K_COHORTS,
                "base_mean": float(np.mean(b_vals)),
                "trt_mean": float(np.mean(d_vals)),
                "paired_diff": mean_diff,
                "ci95_low": mean_diff - t_crit * se_diff,
                "ci95_high": mean_diff + t_crit * se_diff,
                "p_t_raw": float(p_t),
                "p_wilcoxon": float(p_w),
                "cohorts_d_lt_base": int(np.sum(diffs < 0)),
            })
        holm_p = holm_bonferroni(raw_pvals)
        for i, rec in enumerate(tmp_rec):
            rec["p_holm"] = holm_p[i]
            paired_rows.append(rec)

        summary_rows.append({
            "workload": wl_name,
            "curve": curve_name,
            "contrast": contrast_str,
            "d_phases": str(plan_d.phases),
            "sun_pinned_phases": str(plan_sun.phases),
            "base_sun_staleness_30d": float(np.mean([c["distinct_30d_sunday_staleness"] for c in cohort_b])),
            "d_sun_staleness_30d": float(np.mean([c["distinct_30d_sunday_staleness"] for c in cohort_d])),
            "sun_pinned_staleness_30d": float(np.mean([c["distinct_30d_sunday_staleness"] for c in cohort_sun])),
            "d_gapfree_d7_active_mape": float(np.mean([c["distinct_7d_active_mape"] for c in cohort_d_gf])),
            "d_gapfree_d30_active_mape": float(np.mean([c["distinct_30d_active_mape"] for c in cohort_d_gf])),
        })

    # Partition sensitivity analysis for DistinctHeavy / Week1_v2 (K=4, K=8, K=16 stratified & blake2b_hash % 8)
    sens_rows = []
    planner_dh = JointCalendarPlanner(PlannerConfig(commitment=WEEK1_V2_CURVE, workload_weights=WORKLOAD_PRIORITIES["DistinctHeavy"]))
    plan_b_dh = planner_dh.plan_group_b()
    plan_d_dh = planner_dh.plan_group_d()
    for part_name, k_parts, use_hash in [
        ("stratified_K4", 4, False),
        ("stratified_K8", 8, False),
        ("stratified_K16", 16, False),
        ("blake2b_hash_K8", 8, True),
    ]:
        if use_hash:
            u_map = {u: int.from_bytes(hashlib.blake2b(str(u).encode(), digest_size=4).digest(), "little") % k_parts for u in all_users_sorted}
        else:
            u_map = {u: (idx % k_parts) for idx, u in enumerate(all_users_sorted)}
        df_win["sens_cid"] = df_win["hashed_ip"].map(u_map)
        c_b, c_d = [], []
        for k in range(k_parts):
            st = precompute_cohort_suff_stats(df_win[df_win["sens_cid"] == k])
            mb = evaluate_schedule_on_cohort(st, plan_b_dh, k, 0, calendar_weekdays)
            md = evaluate_schedule_on_cohort(st, plan_d_dh, k, 0, calendar_weekdays)
            c_b.append(mb)
            c_d.append(md)
        for m_key in ("distinct_7d_active_mape", "distinct_30d_active_mape", "distinct_90d_active_mape"):
            diffs = np.array([d[m_key] - b[m_key] for d, b in zip(c_d, c_b)])
            _, p_t = stats.ttest_rel([d[m_key] for d in c_d], [b[m_key] for b in c_b])
            sens_rows.append({
                "partition_scheme": part_name,
                "K": k_parts,
                "df": k_parts - 1,
                "metric_key": m_key,
                "mean_diff": float(np.mean(diffs)),
                "p_t_raw": float(p_t),
                "cohorts_d_lt_b": int(np.sum(diffs < 0)),
            })

    pd.DataFrame(paired_rows).to_csv(OUT_DIR / "wildchat_k8_cohorts_paired_v5.csv", index=False)
    pd.DataFrame(summary_rows).to_csv(OUT_DIR / "wildchat_k8_cohorts_summary_v5.csv", index=False)
    pd.DataFrame(sens_rows).to_csv(OUT_DIR / "wildchat_partition_sensitivity_v5.csv", index=False)

    print("\n[WildChat-v5] Dataset Audit Summary:")
    print(audit_df.to_string(index=False))
    print("\n[WildChat-v5] Paired Results (Discrete Gaussian + Exact Top-k + True UTC Sundays):")
    print(pd.DataFrame(paired_rows)[["workload", "curve", "metric_key", "base_mean", "trt_mean", "paired_diff", "p_holm", "p_wilcoxon", "cohorts_d_lt_base"]].to_string(index=False))
    print("\n[WildChat-v5] Sunday Staleness & Gap-Free Summary:")
    print(pd.DataFrame(summary_rows).to_string(index=False))
    print("\n[WildChat-v5] Partition Sensitivity (K in {4,8,16} + blake2b hash):")
    print(pd.DataFrame(sens_rows).to_string(index=False))


if __name__ == "__main__":
    main()
