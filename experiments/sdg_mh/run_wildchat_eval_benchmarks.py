#!/usr/bin/env python3
"""Run WildChat evaluation period (2023-07-10 to 2025-07-31, 753 days) benchmarks.

Strictly implements:
1. Exact per-window GLS variance for Tree-2l, Tree-U, Tree-sqrt across all evaluation windows.
2. Both Caliber a and Caliber b.
3. Dual evaluation for C*=30.0 and C=19.0 on the exact same 753-day evaluation period.
4. Monte Carlo 1,000 trials for end-to-end MAPE on raw request volume.
"""

from __future__ import annotations

import math
from pathlib import Path
import time
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from private_llm_analytics.sdg_mh.baselines import TreeBaseline

WORKSPACE = Path(__file__).resolve().parent.parent.parent
RESULTS_DIR = WORKSPACE / "results" / "sdg_mh"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
PARQUET_PATH = WORKSPACE / "data" / "wildchat_all_shards_4cols.parquet"


def get_tree_window_var(tree_model: TreeBaseline, s: int, e: int) -> float:
    """Compute exact GLS variance for interval [s, e] across annual tree boundaries."""
    length = e - s + 1
    tree_s = (s - 1) % 365 + 1
    tree_e = tree_s + length - 1
    if tree_e <= 365:
        return float(tree_model.get_window_variance(tree_s, tree_e))
    else:
        # Crosses annual boundary: sum variances from two independent annual trees
        v1 = tree_model.get_window_variance(tree_s, 365)
        v2 = tree_model.get_window_variance(1, tree_e - 365)
        return float(v1 + v2)


def run_benchmark_for_c(
    c_val: float,
    cum_raw: np.ndarray,
    cum_clip: np.ndarray,
    eval_start_day: int = 93,
    eval_end_day: int = 845,
    seed: int = 20261106,
) -> pd.DataFrame:
    print(f"\n================ Running Benchmark for C = {c_val:.1f} ================")
    rho_1 = 0.000200
    sigma_1 = c_val / math.sqrt(2.0 * rho_1)
    total_eval_days = eval_end_day - eval_start_day + 1
    print(f"Daily sigma_1: {sigma_1:.1f}, Eval span: {eval_start_day} to {eval_end_day} ({total_eval_days} days)")

    # Tree models
    tree_models = {
        "Tree-U": TreeBaseline("Tree-U", T=365, L=9, C=c_val),
        "Tree-2l": TreeBaseline("Tree-2l", T=365, L=9, C=c_val),
        "Tree-sqrt": TreeBaseline("Tree-sqrt", T=365, L=9, C=c_val),
    }

    # Calibration scale factors (beta) from baseline_calibration_table.csv
    betas = {
        ("SDG-MH", "a"): 1.0,
        ("SDG-MH", "b"): 1.0,
        ("Tree-2l", "a"): 1.0,
        ("Tree-2l", "b"): 1.1125,
        ("Tree-U", "a"): 1.0,
        ("Tree-U", "b"): 1.6079,
        ("Tree-sqrt", "a"): 1.0,
        ("Tree-sqrt", "b"): 1.3230,
        ("LifetimeDecay", "a"): 5.2338,
        ("LifetimeDecay", "b"): 22.5775,
    }

    # Node scales for trees: node_scale_sq = sigma_1^2 * (rho_term(30) / 30.0)
    node_scale_sqs = {
        m: (sigma_1**2) * (tree_models[m].get_worst_case_rho_term(30) / 30.0)
        for m in tree_models
    }

    # Define evaluation windows
    workload_windows: List[Tuple[str, int, int, int, bool]] = []

    # 1. Representative windows for main table inspection
    rep_windows = [
        ("7d_Eval_Early", 93, 99, 7),
        ("7d_Eval_Mid", 366, 372, 7),
        ("14d_Eval_Mid", 366, 379, 14),
        ("30d_Eval_Early", 93, 122, 30),
        ("30d_Eval_Mid", 366, 395, 30),
        ("30d_Eval_Late", 700, 729, 30),
        ("90d_Eval_Q1", 93, 182, 90),
        ("90d_Eval_Mid", 366, 455, 90),
    ]
    for w_name, s, e, length in rep_windows:
        workload_windows.append((w_name, s, e, length, True))

    # 2. Rolling 7d
    for t in range(eval_start_day + 6, eval_end_day + 1):
        workload_windows.append(("rolling_7d", t - 6, t, 7, False))

    # 3. Rolling 30d
    for t in range(eval_start_day + 29, eval_end_day + 1):
        workload_windows.append(("rolling_30d", t - 29, t, 30, False))

    # 4. Rolling 90d
    for t in range(eval_start_day + 89, eval_end_day + 1):
        workload_windows.append(("rolling_90d", t - 89, t, 90, False))

    # 5. Calendar months (30-day blocks)
    n_months = total_eval_days // 30
    for m in range(n_months):
        s = eval_start_day + m * 30
        e = s + 29
        workload_windows.append(("calendar_month", s, e, 30, False))

    # 6. Calendar quarters (90-day blocks)
    n_quarters = total_eval_days // 90
    for q in range(n_quarters):
        s = eval_start_day + q * 90
        e = s + 89
        workload_windows.append(("calendar_quarter", s, e, 90, False))

    methods = ["SDG-MH", "Tree-2l", "Tree-U", "Tree-sqrt", "LifetimeDecay"]
    calibers = ["a", "b"]

    rows = []
    rng = np.random.default_rng(seed)

    t0 = time.time()
    for w_idx, (w_type, s, e, length, is_rep) in enumerate(workload_windows):
        raw_vol = float(cum_raw[e] - cum_raw[s - 1])
        clip_vol = float(cum_clip[e] - cum_clip[s - 1])
        if raw_vol <= 0 or clip_vol <= 0:
            continue

        trunc_bias_pct = (raw_vol - clip_vol) / raw_vol * 100.0

        for m in methods:
            for cal in calibers:
                beta = betas[(m, cal)]

                if m == "SDG-MH":
                    sig_w = beta * sigma_1 * math.sqrt(length)
                elif m == "LifetimeDecay":
                    # Lifetime decay over days s..e
                    t_indices = np.arange(s, e + 1, dtype=np.float64)
                    sum_t_sq = float(np.sum(t_indices**2))
                    var_factor = (c_val**2 * math.pi**2) / (12.0 * 0.2532945)
                    sig_w = beta * math.sqrt(var_factor * sum_t_sq)
                elif m in tree_models:
                    tree = tree_models[m]
                    raw_v = get_tree_window_var(tree, s, e)
                    sig_w = beta * math.sqrt(node_scale_sqs[m] * raw_v)

                # Analytical pure noise error
                pure_noise_mape = (sig_w * math.sqrt(2.0 / math.pi)) / clip_vol

                # Monte Carlo end-to-end trials (1,000 samples)
                noise_samples = rng.normal(0, sig_w, size=1000)
                estimates = np.maximum(0.0, clip_vol + noise_samples)
                e2e_mape = float(np.mean(np.abs(estimates - raw_vol) / raw_vol))
                e2e_std = float(np.std(np.abs(estimates - raw_vol) / raw_vol))

                rows.append({
                    "dataset": "WildChat-4.8M",
                    "window": w_type,
                    "start_day": s,
                    "end_day": e,
                    "length_days": length,
                    "is_representative": is_rep,
                    "raw_volume": int(raw_vol),
                    "clipped_volume": int(clip_vol),
                    "clipping_bound_C": c_val,
                    "clipping_bias_pct": round(trunc_bias_pct, 3),
                    "method": m,
                    "caliber": cal,
                    "noise_multiplier_beta": beta,
                    "noise_sigma": round(sig_w, 1),
                    "pure_noise_mape_analytical": pure_noise_mape,
                    "e2e_raw_mape_mean": e2e_mape,
                    "e2e_raw_mape_std": e2e_std,
                })

        if (w_idx + 1) % 500 == 0 or w_idx == len(workload_windows) - 1:
            print(f"  Processed {w_idx + 1}/{len(workload_windows)} windows ({time.time() - t0:.1f}s)...")

    df_res = pd.DataFrame(rows)
    print(f"Completed C={c_val:.1f}: {len(df_res)} rows generated.")
    return df_res


def main():
    print("Loading WildChat Parquet dataset...")
    df_raw = pd.read_parquet(PARQUET_PATH, columns=["hashed_ip", "timestamp"])
    df_raw["dt"] = pd.to_datetime(df_raw["timestamp"], utc=True)
    min_date = df_raw["dt"].min()
    df_raw["day"] = (df_raw["dt"] - min_date).dt.days + 1
    total_days = int(df_raw["day"].max())

    print("Computing daily user-level aggregations for C=30.0 and C=19.0...")
    user_day = df_raw.groupby(["day", "hashed_ip"]).size().reset_index(name="cnt")
    user_day["clip_30"] = np.minimum(user_day["cnt"], 30.0)
    user_day["clip_19"] = np.minimum(user_day["cnt"], 19.0)

    daily_agg = user_day.groupby("day").agg(
        raw_vol=("cnt", "sum"),
        clip_30=("clip_30", "sum"),
        clip_19=("clip_19", "sum"),
    ).reset_index()

    raw_arr = np.zeros(total_days + 1)
    clip_30_arr = np.zeros(total_days + 1)
    clip_19_arr = np.zeros(total_days + 1)

    for _, r in daily_agg.iterrows():
        d = int(r["day"])
        raw_arr[d] = r["raw_vol"]
        clip_30_arr[d] = r["clip_30"]
        clip_19_arr[d] = r["clip_19"]

    cum_raw = np.cumsum(raw_arr)
    cum_c30 = np.cumsum(clip_30_arr)
    cum_c19 = np.cumsum(clip_19_arr)

    # 1. Run for C*=30.0
    df_c30 = run_benchmark_for_c(30.0, cum_raw, cum_c30, seed=20261106)
    out_c30_path = RESULTS_DIR / "rq1_wildchat_benchmark_c30.csv"
    df_c30.to_csv(out_c30_path, index=False)
    print(f"Saved {out_c30_path} successfully ({len(df_c30)} rows).")

    # 2. Run for C=19.0 on the exact same evaluation period
    df_c19 = run_benchmark_for_c(19.0, cum_raw, cum_c19, seed=20261107)
    out_c19_path = RESULTS_DIR / "rq1_wildchat_benchmark_c19_eval.csv"
    df_c19.to_csv(out_c19_path, index=False)
    print(f"Saved {out_c19_path} successfully ({len(df_c19)} rows).")

    # Print summary comparison table for Dashboard Workloads
    print("\n" + "=" * 90)
    print("DASHBOARD WORKLOAD SUMMARY COMPARISON: C*=30.0 vs C=19.0 (EVAL PERIOD: 753 DAYS)")
    print("=" * 90)
    workload_types = ["rolling_7d", "rolling_30d", "rolling_90d", "calendar_month", "calendar_quarter"]
    
    for w_type in workload_types:
        print(f"\n--- Workload: {w_type} ---")
        for cal in ["a", "b"]:
            print(f"  [Caliber {cal}]")
            for m in ["SDG-MH", "Tree-2l", "Tree-U", "Tree-sqrt", "LifetimeDecay"]:
                sub30 = df_c30[(df_c30["window"] == w_type) & (df_c30["method"] == m) & (df_c30["caliber"] == cal)]
                sub19 = df_c19[(df_c19["window"] == w_type) & (df_c19["method"] == m) & (df_c19["caliber"] == cal)]
                
                sig30 = sub30["noise_sigma"].mean()
                pure30 = sub30["pure_noise_mape_analytical"].mean() * 100.0
                e2e30 = sub30["e2e_raw_mape_mean"].mean() * 100.0
                bias30 = sub30["clipping_bias_pct"].mean()

                sig19 = sub19["noise_sigma"].mean()
                pure19 = sub19["pure_noise_mape_analytical"].mean() * 100.0
                e2e19 = sub19["e2e_raw_mape_mean"].mean() * 100.0
                bias19 = sub19["clipping_bias_pct"].mean()

                print(
                    f"    {m:13s}: "
                    f"C*=30 [sig={sig30:8.1f}, pure={pure30:6.2f}%, bias={bias30:5.2f}%, e2e={e2e30:6.2f}%] | "
                    f"C=19 [sig={sig19:8.1f}, pure={pure19:6.2f}%, bias={bias19:5.2f}%, e2e={e2e19:6.2f}%]"
                )


if __name__ == "__main__":
    main()
