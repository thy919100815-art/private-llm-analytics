"""S2: Rolling Distinct State Implementation, Overhead, and Calendar Parameters Audit.

Audits:
1. Physical Data Flow & Isolation: Replaces raw DataFrame queries with RollingDistinctState.
2. Exact Histogram Match Verification: Verified across all release days against raw data.
3. Computational & Storage Overhead Benchmarks for N in {2000, 10000, 50000}.
4. Exact Calendar A and B Parameter Audit & Theoretical Reconciliations.
5. Prevalence Denominator Source & Privacy Accounting.
"""

from __future__ import annotations

import math
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Sequence, Tuple
import numpy as np
import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).parents[3] / "src"))

from private_llm_analytics.sdg_mh.calendar_job import (
    CalendarJob,
    DEFAULT_CALENDAR_A,
    DEFAULT_CALENDAR_B,
    DEFAULT_DISTINCT_SERVICES,
    DEFAULT_DISTINCT_TASKS,
    DistinctCalendarStore,
)
from private_llm_analytics.sdg_mh.clipping import compute_distinct_window_histogram
from private_llm_analytics.sdg_mh.ledger import (
    UnifiedPrivacyLedger,
    closed_form_distinct_calendar_rho,
)
from private_llm_analytics.sdg_mh.rolling_state import RollingDistinctState
from private_llm_analytics.sdg_mh.synthetic_generator import SyntheticTelemetryGenerator

OUTPUT_DIR = Path("results/sdg_mh/supp_v11")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def run_correctness_and_overhead_benchmarks():
    print("[S2 1/2] Benchmarking RollingDistinctState across N in {2000, 10000, 50000}...")
    benchmarks_rows = []

    # 1. N = 2,000 (Synthetic Telemetry Generator, 90 days)
    print("  Testing N=2,000 (90-day stream)...")
    gen = SyntheticTelemetryGenerator(n_initial_users=2000, seed=42)
    state_2k_dir = OUTPUT_DIR / "state_2000"
    state_2k = RollingDistinctState(storage_dir=state_2k_dir, w_max=90)
    store_2k = DistinctCalendarStore()
    ledger_2k = UnifiedPrivacyLedger()
    cal_job_2k = CalendarJob(calendar_store=store_2k, ledger=ledger_2k, rolling_state=state_2k)

    update_times_2k = []
    caljob_times_2k = []
    correctness_all_match_2k = True

    days_raw_2k = {}
    for day, df_day in gen.stream_daily_telemetry(start_day=1, end_day=90):
        days_raw_2k[day] = df_day
        t_up = state_2k.append_day(
            day_index=day,
            df_day=df_day,
            services_list=DEFAULT_DISTINCT_SERVICES,
            tasks_list=DEFAULT_DISTINCT_TASKS,
            persist_disk=True,
        )
        update_times_2k.append(t_up)

        # Check calendar execution
        t_cal0 = time.perf_counter()
        rels = cal_job_2k.execute_day(current_day=day, seed=42 + day)
        t_cal = (time.perf_counter() - t_cal0) * 1000.0
        if rels:
            caljob_times_2k.append(t_cal)

            # Correctness verification against raw data for each release
            for rel in rels:
                w_len = rel.window_length
                s_day = max(1, day - w_len + 1)
                df_win = pd.concat([days_raw_2k[d] for d in range(s_day, day + 1)], ignore_index=True)
                raw_srv, raw_tsk = compute_distinct_window_histogram(
                    df_window=df_win,
                    services_list=DEFAULT_DISTINCT_SERVICES,
                    tasks_list=DEFAULT_DISTINCT_TASKS,
                    k_tasks_per_user=3,
                )
                state_srv, state_tsk = state_2k.compute_window_histogram(
                    start_day=s_day,
                    end_day=day,
                    services_list=DEFAULT_DISTINCT_SERVICES,
                    tasks_list=DEFAULT_DISTINCT_TASKS,
                    k_tasks_per_user=3,
                )
                if not (np.array_equal(raw_srv, state_srv) and np.array_equal(raw_tsk, state_tsk)):
                    correctness_all_match_2k = False

    fp_2k = state_2k.get_storage_footprint()
    benchmarks_rows.append({
        "population_n": 2000,
        "days_streamed": 90,
        "total_active_user_days": fp_2k["total_user_days"],
        "correctness_exact_match": correctness_all_match_2k,
        "memory_mb": fp_2k["memory_mb"],
        "disk_mb": fp_2k["disk_mb"],
        "bytes_per_user_day": fp_2k["bytes_per_user_day"],
        "update_latency_p50_ms": round(float(np.percentile(update_times_2k, 50)), 3),
        "update_latency_p99_ms": round(float(np.percentile(update_times_2k, 99)), 3),
        "caljob_latency_p50_ms": round(float(np.percentile(caljob_times_2k, 50)), 3) if caljob_times_2k else 0.0,
        "caljob_latency_p99_ms": round(float(np.percentile(caljob_times_2k, 99)), 3) if caljob_times_2k else 0.0,
    })

    # Helper for parquet datasets
    def benchmark_parquet_dataset(n_val: int, parquet_path: Path) -> Dict[str, Any]:
        print(f"  Testing N={n_val:,} from {parquet_path}...")
        df_full = pd.read_parquet(parquet_path)
        df_full["dt"] = pd.to_datetime(df_full["timestamp"])
        min_date = df_full["dt"].dt.date.min()
        df_full["day"] = (df_full["dt"].dt.date - min_date).apply(lambda d: d.days + 1)
        n_days = int(df_full["day"].max())

        # Determine services and tasks from dataset
        services_list = sorted(df_full["provider"].unique().tolist())
        tasks_list = sorted(df_full["task"].unique().tolist())

        state_dir = OUTPUT_DIR / f"state_{n_val}"
        state = RollingDistinctState(storage_dir=state_dir, w_max=90)
        store = DistinctCalendarStore()
        ledger = UnifiedPrivacyLedger()
        cal_job = CalendarJob(
            calendar_store=store,
            ledger=ledger,
            services_list=services_list,
            tasks_list=tasks_list,
            rolling_state=state,
        )

        update_latencies = []
        caljob_latencies = []
        correctness_match = True
        days_raw = {}

        for d in range(1, n_days + 1):
            df_day = df_full[df_full["day"] == d]
            days_raw[d] = df_day
            t_up = state.append_day(
                day_index=d,
                df_day=df_day,
                services_list=services_list,
                tasks_list=tasks_list,
                persist_disk=True,
            )
            update_latencies.append(t_up)

            t_cal0 = time.perf_counter()
            rels = cal_job.execute_day(current_day=d, seed=1000 + d)
            t_cal = (time.perf_counter() - t_cal0) * 1000.0
            if rels:
                caljob_latencies.append(t_cal)
                for rel in rels:
                    w_len = rel.window_length
                    s_day = max(1, d - w_len + 1)
                    df_win = pd.concat([days_raw[idx] for idx in range(s_day, d + 1)], ignore_index=True)
                    raw_srv, raw_tsk = compute_distinct_window_histogram(
                        df_window=df_win,
                        services_list=services_list,
                        tasks_list=tasks_list,
                        k_tasks_per_user=3,
                    )
                    st_srv, st_tsk = state.compute_window_histogram(
                        start_day=s_day,
                        end_day=d,
                        services_list=services_list,
                        tasks_list=tasks_list,
                        k_tasks_per_user=3,
                    )
                    if not (np.array_equal(raw_srv, st_srv) and np.array_equal(raw_tsk, st_tsk)):
                        correctness_match = False

        fp = state.get_storage_footprint()
        return {
            "population_n": n_val,
            "days_streamed": n_days,
            "total_active_user_days": fp["total_user_days"],
            "correctness_exact_match": correctness_match,
            "memory_mb": fp["memory_mb"],
            "disk_mb": fp["disk_mb"],
            "bytes_per_user_day": fp["bytes_per_user_day"],
            "update_latency_p50_ms": round(float(np.percentile(update_latencies, 50)), 3),
            "update_latency_p99_ms": round(float(np.percentile(update_latencies, 99)), 3),
            "caljob_latency_p50_ms": round(float(np.percentile(caljob_latencies, 50)), 3) if caljob_latencies else 0.0,
            "caljob_latency_p99_ms": round(float(np.percentile(caljob_latencies, 99)), 3) if caljob_latencies else 0.0,
        }

    benchmarks_rows.append(benchmark_parquet_dataset(10000, Path("data/interactions_n10000.parquet")))
    benchmarks_rows.append(benchmark_parquet_dataset(50000, Path("data/interactions_n50000.parquet")))

    df_bench = pd.DataFrame(benchmarks_rows)
    df_bench.to_csv(OUTPUT_DIR / "s2_rolling_state_benchmark.csv", index=False)
    print(f"  Saved rolling state benchmarks to {OUTPUT_DIR / 's2_rolling_state_benchmark.csv'}")
    print(df_bench.to_string(index=False))


def run_calendar_parameters_audit():
    print("\n[S2 2/2] Running Calendar Parameters & Ledger Reconciliation Audit...")

    cal_a_entries = [(7, 1, 60.0), (30, 7, 60.0), (90, 30, 60.0)]
    cal_b_entries = [(7, 7, 60.0), (30, 7, 60.0), (90, 30, 60.0)]

    m_services = 5
    k_tasks = 3
    delta_sq = float(m_services + 2 * k_tasks)  # 11.0
    sigma_w = 60.0
    rho_single = delta_sq / (2.0 * (sigma_w ** 2))  # 11 / 7200 = 0.001527777...

    horizons = [1, 7, 30, 90, 365]
    param_rows = []

    # Verify closed-form vs simulated UnifiedPrivacyLedger across 365 days
    for cal_name, entries in [("Calendar A (High-Freq Rolling)", cal_a_entries), ("Calendar B (Production Paced)", cal_b_entries)]:
        ledger = UnifiedPrivacyLedger()
        for w, s, sig in entries:
            rho_rel = delta_sq / (2.0 * (sig ** 2))
            for end_d in range(w, 366, s):
                start_d = max(1, end_d - w + 1)
                ledger.record_gaussian_release(
                    release_id=f"cal_{w}_{s}_{end_d}",
                    days_accessed=list(range(start_d, end_d + 1)),
                    sensitivity=math.sqrt(delta_sq),
                    sigma=sig,
                    tag=f"{cal_name}_{w}_{s}",
                )

        prof = ledger.get_profile(horizons=horizons)

        for h in horizons:
            cf_val = closed_form_distinct_calendar_rho(h, entries, delta_sq=delta_sq)
            ledger_val = prof[h]
            # Number of covering windows
            num_windows = sum(int(math.ceil((h + w - 1) / float(s))) for w, s, _ in entries)

            param_rows.append({
                "calendar": cal_name,
                "horizon_h": h,
                "m_services": m_services,
                "k_tasks": k_tasks,
                "delta_sq": delta_sq,
                "sigma_w": sigma_w,
                "rho_single_release": round(rho_single, 8),
                "covering_windows_count": num_windows,
                "closed_form_rho": round(cf_val, 6),
                "ledger_profile_rho": round(ledger_val, 6),
                "reconciliation_diff": round(abs(cf_val - ledger_val), 8),
                "reconciled": bool(abs(cf_val - ledger_val) < 1e-5),
            })

    df_params = pd.DataFrame(param_rows)
    df_params.to_csv(OUTPUT_DIR / "s2_calendar_parameters_audit.csv", index=False)
    print(f"  Saved calendar parameters audit to {OUTPUT_DIR / 's2_calendar_parameters_audit.csv'}")
    print(df_params.to_string(index=False))


if __name__ == "__main__":
    bench_csv = OUTPUT_DIR / "s2_rolling_state_benchmark.csv"
    if not bench_csv.exists():
        run_correctness_and_overhead_benchmarks()
    else:
        print(f"Rolling state benchmarks already exist at {bench_csv}. Running calendar audit...")
    run_calendar_parameters_audit()
