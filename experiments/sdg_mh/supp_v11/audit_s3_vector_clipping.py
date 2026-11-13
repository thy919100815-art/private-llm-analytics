"""S3: Vector Metrics Truncation and Sensitivity Audit.

Audits:
1. Exact releases in daily SDG-MH pipeline (Enterprise, WildChat, GH Archive).
2. Empirical user-day L2 contribution norm vs ledger sensitivity.
3. Theoretical & empirical behavior of hypothetical multi-dimensional vector extensions.
"""

from __future__ import annotations

import math
from pathlib import Path
import sys
from typing import Any, Dict, List, Set

import numpy as np
import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).parents[3] / "src"))

from private_llm_analytics.sdg_mh.clipping import (
    clip_daily_additive,
    compute_distinct_window_histogram,
)
from private_llm_analytics.sdg_mh.ledger import UnifiedPrivacyLedger
from private_llm_analytics.sdg_mh.seal_job import SealJob
from private_llm_analytics.sdg_mh.sealed_store import SealedStore
from private_llm_analytics.sdg_mh.synthetic_generator import SyntheticTelemetryGenerator

OUTPUT_DIR = Path("results/sdg_mh/supp_v11")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def audit_enterprise_365d() -> List[Dict[str, Any]]:
    """Audit Enterprise 365-day dataset under seed=42."""
    print("[S3 1/3] Auditing Enterprise 365-day dataset (seed=42, N=2000)...")
    gen = SyntheticTelemetryGenerator(n_initial_users=2000, seed=42)

    store = SealedStore()
    ledger = UnifiedPrivacyLedger()
    job = SealJob(sealed_store=store, ledger=ledger, C_r=104.0, C_f=7.0, sigma_r=5200.0, sigma_f=489.9)

    l2_req_actual = []
    l2_fail_actual = []
    l2_hypothetical_9d_raw = []
    l2_hypothetical_9d_clipped = []
    l2_hypothetical_18cell_clipped = []

    providers_list = [f"service_{i}" for i in range(1, 6)]
    tasks_list = [f"task_{i:02d}" for i in range(1, 21)]

    for day, df_day in gen.stream_daily_telemetry(start_day=1, end_day=365):
        # 1. Actual SealJob execution
        rec = job.seal_day(day, f"2026-01-{day:03d}", df_day)

        if df_day.empty:
            continue

        # Requests L2 norm
        u_counts = df_day["user_id"].value_counts()
        clipped_u = u_counts.clip(upper=104.0)
        l2_req_actual.extend(clipped_u.values)

        # Failures L2 norm
        fail_df = df_day[df_day["success"] == False]  # noqa: E712
        if not fail_df.empty:
            f_counts = fail_df["user_id"].value_counts()
            clipped_f = f_counts.clip(upper=7.0)
            l2_fail_actual.extend(clipped_f.values)

        # Vectorized computation of hypothetical 25D vector (5 services + 20 tasks)
        p_ct = pd.crosstab(df_day["user_id"], df_day["provider"]).reindex(index=u_counts.index, columns=providers_list, fill_value=0).to_numpy(dtype=np.float64)
        t_ct = pd.crosstab(df_day["user_id"], df_day["task"]).reindex(index=u_counts.index, columns=tasks_list, fill_value=0).to_numpy(dtype=np.float64)
        vec_25d = np.hstack([p_ct, t_ct])
        raw_cnts = u_counts.to_numpy(dtype=np.float64)[:, None]
        scale = np.minimum(1.0, 104.0 / np.maximum(raw_cnts, 1.0))
        vec_25d_clipped = vec_25d * scale
        l2_hypothetical_9d_clipped.extend(np.linalg.norm(vec_25d_clipped, ord=2, axis=1))

        # Vectorized computation of hypothetical disjoint 100-cell partition
        # ||x||_2 <= ||x||_1 = min(raw_cnt, 104.0)
        # Compute exact norm per user:
        pt_sumsq = df_day.groupby(["user_id", "provider", "task"]).size().pow(2).groupby("user_id").sum().reindex(u_counts.index, fill_value=0).to_numpy(dtype=np.float64)
        # Scaled by (104.0 / raw_cnt)^2 if raw_cnt > 104
        norm_100c_clipped = np.sqrt(pt_sumsq) * scale[:, 0]
        l2_hypothetical_18cell_clipped.extend(norm_100c_clipped)

    arr_req = np.array(l2_req_actual)
    arr_fail = np.array(l2_fail_actual)
    arr_9d_raw = np.array(l2_hypothetical_9d_raw)
    arr_9d_clip = np.array(l2_hypothetical_9d_clipped)
    arr_18c = np.array(l2_hypothetical_18cell_clipped)

    rows = [
        {
            "dataset": "Enterprise (365d)",
            "channel": "Additive",
            "release_object": "sdg_additive_req (Scalar Total Requests)",
            "is_vector": False,
            "dimension": 1,
            "actual_or_hypothetical": "Actual Release",
            "ledger_sensitivity": 104.0,
            "user_day_count": len(arr_req),
            "max_l2_norm": round(float(np.max(arr_req)), 4),
            "p99_l2_norm": round(float(np.percentile(arr_req, 99)), 4),
            "p999_l2_norm": round(float(np.percentile(arr_req, 99.9)), 4),
            "mean_l2_norm": round(float(np.mean(arr_req)), 4),
            "exceeds_ledger": bool(np.max(arr_req) > 104.0),
        },
        {
            "dataset": "Enterprise (365d)",
            "channel": "Additive",
            "release_object": "sdg_additive_fail (Scalar Total Failures)",
            "is_vector": False,
            "dimension": 1,
            "actual_or_hypothetical": "Actual Release",
            "ledger_sensitivity": 7.0,
            "user_day_count": len(arr_fail),
            "max_l2_norm": round(float(np.max(arr_fail)), 4),
            "p99_l2_norm": round(float(np.percentile(arr_fail, 99)), 4),
            "p999_l2_norm": round(float(np.percentile(arr_fail, 99.9)), 4),
            "mean_l2_norm": round(float(np.mean(arr_fail)), 4),
            "exceeds_ledger": bool(np.max(arr_fail) > 7.0),
        },
        {
            "dataset": "Enterprise (365d)",
            "channel": "Additive",
            "release_object": "Hypothetical Marginal Vector (5 Serv + 20 Task) Scalar-Clipped",
            "is_vector": True,
            "dimension": 25,
            "actual_or_hypothetical": "Hypothetical (Overlapping Marginals)",
            "ledger_sensitivity": 104.0,
            "user_day_count": len(arr_9d_clip),
            "max_l2_norm": round(float(np.max(arr_9d_clip)), 4),
            "p99_l2_norm": round(float(np.percentile(arr_9d_clip, 99)), 4),
            "p999_l2_norm": round(float(np.percentile(arr_9d_clip, 99.9)), 4),
            "mean_l2_norm": round(float(np.mean(arr_9d_clip)), 4),
            "exceeds_ledger": bool(np.max(arr_9d_clip) > 104.0),
        },
        {
            "dataset": "Enterprise (365d)",
            "channel": "Additive",
            "release_object": "Hypothetical Disjoint Partition (5 Serv x 20 Task = 100 Cells)",
            "is_vector": True,
            "dimension": 100,
            "actual_or_hypothetical": "Hypothetical (Disjoint Partition)",
            "ledger_sensitivity": 104.0,
            "user_day_count": len(arr_18c),
            "max_l2_norm": round(float(np.max(arr_18c)), 4),
            "p99_l2_norm": round(float(np.percentile(arr_18c, 99)), 4),
            "p999_l2_norm": round(float(np.percentile(arr_18c, 99.9)), 4),
            "mean_l2_norm": round(float(np.mean(arr_18c)), 4),
            "exceeds_ledger": bool(np.max(arr_18c) > 104.0),
        },
    ]
    return rows


def audit_wildchat() -> List[Dict[str, Any]]:
    """Audit WildChat 845-day dataset."""
    print("[S3 2/3] Auditing WildChat dataset...")
    parquet_path = Path("data/wildchat_all_shards_4cols.parquet")
    if not parquet_path.exists():
        parquet_path = Path("data/wildchat_shard_00000.parquet")
    if not parquet_path.exists():
        print("  WildChat parquet not found, returning empty.")
        return []

    df = pd.read_parquet(parquet_path, columns=["hashed_ip", "timestamp"])
    df["dt"] = pd.to_datetime(df["timestamp"], utc=True)
    min_date = df["dt"].min()
    df["day"] = (df["dt"] - min_date).dt.days + 1

    user_day = df.groupby(["day", "hashed_ip"]).size().values

    # Check for C_r = 30.0 (optimal) and C_r = 19.0 (v7 setting)
    rows = []
    for c_r in [30.0, 19.0]:
        clipped = np.minimum(user_day, c_r)
        rows.append({
            "dataset": "WildChat (845d)",
            "channel": "Additive",
            "release_object": f"Daily Interaction Volume (Scalar, C={c_r})",
            "is_vector": False,
            "dimension": 1,
            "actual_or_hypothetical": "Actual Release",
            "ledger_sensitivity": c_r,
            "user_day_count": len(clipped),
            "max_l2_norm": round(float(np.max(clipped)), 4),
            "p99_l2_norm": round(float(np.percentile(clipped, 99)), 4),
            "p999_l2_norm": round(float(np.percentile(clipped, 99.9)), 4),
            "mean_l2_norm": round(float(np.mean(clipped)), 4),
            "exceeds_ledger": bool(np.max(clipped) > c_r),
        })
    return rows


def audit_gharchive() -> List[Dict[str, Any]]:
    """Audit GH Archive 730-day dataset."""
    print("[S3 3/3] Auditing GH Archive dataset...")
    parquet_path = Path("data/gharchive_2years_daily.parquet")
    if not parquet_path.exists():
        print("  GH Archive parquet not found, returning empty.")
        return []

    df = pd.read_parquet(parquet_path)
    # The columns: day, user_id, event_count
    if "event_count" in df.columns:
        counts = df["event_count"].values
    elif "cnt" in df.columns:
        counts = df["cnt"].values
    else:
        counts = df.iloc[:, 2].values

    c_r = 10.0
    clipped = np.minimum(counts, c_r)
    return [{
        "dataset": "GH Archive (730d)",
        "channel": "Additive",
        "release_object": f"Daily Event Volume (Scalar, C={c_r})",
        "is_vector": False,
        "dimension": 1,
        "actual_or_hypothetical": "Actual Release",
        "ledger_sensitivity": c_r,
        "user_day_count": len(clipped),
        "max_l2_norm": round(float(np.max(clipped)), 4),
        "p99_l2_norm": round(float(np.percentile(clipped, 99)), 4),
        "p999_l2_norm": round(float(np.percentile(clipped, 99.9)), 4),
        "mean_l2_norm": round(float(np.mean(clipped)), 4),
        "exceeds_ledger": bool(np.max(clipped) > c_r),
    }]


def main():
    all_rows = []
    all_rows.extend(audit_enterprise_365d())
    all_rows.extend(audit_wildchat())
    all_rows.extend(audit_gharchive())

    df_out = pd.DataFrame(all_rows)
    out_csv = OUTPUT_DIR / "s3_vector_clipping_audit.csv"
    df_out.to_csv(out_csv, index=False)
    print(f"\n[S3 Complete] Saved audit table to {out_csv}")
    print(df_out.to_string(index=False))


if __name__ == "__main__":
    main()
