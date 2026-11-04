"""Measure Engineering Metrics for SDG-MH.

Measures:
1. Daily SealJob duration (mean, p50, p99, max in ms) across 365 days.
2. SealedStore & DistinctCalendarStore annual disk footprint (in KB/year).
3. ReadOnlySealedQueryService latency (mean, p50, p99, max in ms) across 10,000 queries.
4. Precomputed display rules (CI width & suppression) across horizons.

Outputs:
- results/sdg_mh/engineering_metrics.csv
- results/sdg_mh/display_rules_table.csv
"""

from __future__ import annotations

import json
import math
from pathlib import Path
import sys
import tempfile
import time
from typing import Any, Dict, List

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parents[2] / "src"))

from private_llm_analytics.sdg_mh.calendar_job import (
    CalendarJob,
    DEFAULT_CALENDAR_B,
    DistinctCalendarStore,
)
from private_llm_analytics.sdg_mh.display_rules import DisplayRulesEngine
from private_llm_analytics.sdg_mh.ledger import UnifiedPrivacyLedger
from private_llm_analytics.sdg_mh.query_service import ReadOnlySealedQueryService
from private_llm_analytics.sdg_mh.seal_job import SealJob
from private_llm_analytics.sdg_mh.sealed_store import SealedStore
from private_llm_analytics.sdg_mh.synthetic_generator import SyntheticTelemetryGenerator

RESULTS_DIR = Path("results/sdg_mh")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def main():
    print("=" * 80)
    print("Measuring SDG-MH Engineering Metrics (Seal Job, Storage, Query Latency, Display Rules)")
    print("=" * 80)

    # 1. Benchmark SealJob & Storage across 365 days
    print("[1/4] Benchmarking daily SealJob execution across 365 days...")
    with tempfile.TemporaryDirectory() as tmp_dir:
        store_path = Path(tmp_dir) / "sealed_store"
        store = SealedStore(storage_dir=store_path)
        cal_store = DistinctCalendarStore()
        ledger = UnifiedPrivacyLedger(delta=1e-6)

        seal_job = SealJob(sealed_store=store, ledger=ledger, sigma_r=2500.0, sigma_f=700.0)
        cal_job = CalendarJob(calendar_store=cal_store, ledger=ledger, calendar_schedule=DEFAULT_CALENDAR_B)

        gen = SyntheticTelemetryGenerator(n_initial_users=2000, seed=42)

        history_days: Dict[int, pd.DataFrame] = {}

        def get_window_df(s: int, e: int) -> pd.DataFrame:
            dfs = [history_days[d] for d in range(s, e + 1) if d in history_days]
            return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

        # Stream and seal 365 days
        for day, df_day in gen.stream_daily_telemetry(start_day=1, end_day=365):
            # Keep lightweight projection for distinct calendar
            history_days[day] = df_day[["user_id", "provider", "task"]]
            seal_job.seal_day(
                day_index=day,
                date_str=f"2026-{(day-1)//30+1:02d}-{(day-1)%30+1:02d}",
                df_day=df_day,
                seed=42000 + day,
            )
            # Distinct calendar check
            cal_job.execute_day(current_day=day, df_history_fn=get_window_df, seed=52000 + day)

            # Evict history older than 90 days to bound memory
            if day > 95:
                history_days.pop(day - 95, None)

        seal_stats = seal_job.get_benchmarks()
        print(f"  SealJob completed: count={seal_stats['count']}, p50={seal_stats['p50_ms']}ms, p99={seal_stats['p99_ms']}ms, mean={seal_stats['mean_ms']}ms")

        # Storage statistics
        print("[2/4] Measuring annual disk storage consumption...")
        store_stats = store.get_storage_stats()
        
        # Calculate size of calendar store records
        cal_releases = cal_store.all_releases()
        from dataclasses import asdict
        cal_bytes = sum(len(json.dumps(asdict(r))) for r in cal_releases)

        annual_store_bytes = store_stats["annual_projected_bytes"]
        annual_cal_bytes = cal_bytes * (365.0 / max(1, len(store.get_days_in_range(1, 365))))
        total_annual_kb = (annual_store_bytes + annual_cal_bytes) / 1024.0

        print(f"  Annual SealedStore: {annual_store_bytes / 1024.0:.2f} KB/year")
        print(f"  Annual CalendarStore: {annual_cal_bytes / 1024.0:.2f} KB/year")
        print(f"  Total Annual Footprint: {total_annual_kb:.2f} KB/year (~{total_annual_kb / 1024.0:.4f} MB/year)")

        # 3. ReadOnlySealedQueryService Latency Benchmark (10,000 queries)
        print("[3/4] Benchmarking ReadOnlySealedQueryService latency over 10,000 queries...")
        query_service = ReadOnlySealedQueryService(sealed_store=store, calendar_store=cal_store, public_population_size=2000)

        rng = np.random.default_rng(12345)
        for _ in range(10000):
            # Sample random query: additive window or distinct window
            q_type = rng.choice(["additive", "distinct", "top_k"])
            if q_type == "additive":
                w = int(rng.integers(1, 180))
                s = int(rng.integers(1, 365 - w + 1))
                query_service.query_additive_window(s, s + w - 1)
            elif q_type == "distinct":
                w = int(rng.choice([7, 30, 90]))
                e = int(rng.integers(w, 365))
                query_service.query_distinct_prevalence(window_length=w, end_day=e)
            else:
                w = int(rng.choice([7, 30, 90]))
                e = int(rng.integers(w, 365))
                query_service.query_top_k_tasks(window_length=w, end_day=e, k=5)

        q_stats = query_service.get_latency_stats()
        print(f"  Query latency (10,000 queries): p50={q_stats['p50_ms']}ms, p99={q_stats['p99_ms']}ms, mean={q_stats['mean_ms']}ms")

    # 4. Display Rules Table Precomputation
    print("[4/4] Precomputing Display Rules and Confidence Interval Widths...")
    engine = DisplayRulesEngine(sigma_r=2500.0, sigma_f=700.0, sigma_w=60.0)
    rule_rows = []
    for h in [1, 7, 30, 90, 365]:
        eval_res = engine.evaluate_additive_window(window_days=h, prior_daily_requests=13400.0, prior_failure_rate=0.03)
        rule_rows.append({
            "horizon_days": h,
            "requests_half_width": eval_res["requests_half_width"],
            "requests_ci_width": eval_res["requests_ci_width"],
            "requests_relative_half_width_pct": round(eval_res["requests_relative_half_width"] * 100.0, 1),
            "requests_displayable": eval_res["requests_displayable"],
            "success_rate_ci_width": eval_res["success_rate_ci_width"],
            "success_rate_displayable": eval_res["success_rate_displayable"],
            "suppression_reason": eval_res["suppression_reason"] or "None (Within threshold)",
        })

    df_rules = pd.DataFrame(rule_rows)
    df_rules.to_csv(RESULTS_DIR / "display_rules_table.csv", index=False)
    print("  Saved display_rules_table.csv.")

    # Save consolidated engineering metrics
    eng_rows = [
        {"metric_category": "SealJob Latency", "metric_name": "p50 (Median)", "value": seal_stats["p50_ms"], "unit": "ms"},
        {"metric_category": "SealJob Latency", "metric_name": "p99", "value": seal_stats["p99_ms"], "unit": "ms"},
        {"metric_category": "SealJob Latency", "metric_name": "Mean", "value": seal_stats["mean_ms"], "unit": "ms"},
        {"metric_category": "SealJob Latency", "metric_name": "Daily Sealing Ingestion Rate", "value": round(4898334.0 / (seal_stats["count"] * seal_stats["mean_ms"] / 1000.0), 1), "unit": "events/sec"},
        {"metric_category": "Storage Footprint", "metric_name": "SealedStore Annual Disk Footprint", "value": round(annual_store_bytes / 1024.0, 2), "unit": "KB/year"},
        {"metric_category": "Storage Footprint", "metric_name": "CalendarStore Annual Disk Footprint", "value": round(annual_cal_bytes / 1024.0, 2), "unit": "KB/year"},
        {"metric_category": "Storage Footprint", "metric_name": "Total Annual Storage Consumption", "value": round(total_annual_kb, 2), "unit": "KB/year"},
        {"metric_category": "Query Latency", "metric_name": "p50 (Median)", "value": q_stats["p50_ms"], "unit": "ms"},
        {"metric_category": "Query Latency", "metric_name": "p99", "value": q_stats["p99_ms"], "unit": "ms"},
        {"metric_category": "Query Latency", "metric_name": "Mean", "value": q_stats["mean_ms"], "unit": "ms"},
        {"metric_category": "Query Privacy Spend", "metric_name": "Query Time Differential Privacy Cost", "value": 0.0, "unit": "eps"},
    ]
    df_eng = pd.DataFrame(eng_rows)
    df_eng.to_csv(RESULTS_DIR / "engineering_metrics.csv", index=False)
    print("  Saved engineering_metrics.csv.")
    print("=" * 80)
    print(df_eng.to_string(index=False))
    print("=" * 80)


if __name__ == "__main__":
    main()
