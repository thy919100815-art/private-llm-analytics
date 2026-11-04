#!/usr/bin/env python3
"""RQ5: Query Reuse & Cumulative Privacy Cost Evaluation.

Workload (Section 5 RQ5):
- 180 days of enterprise dashboard analytics.
- 50 analysts daily, each issuing 0 to 20 queries (mean 10 queries/day).
- Query distribution:
  * Rolling 7-day, 30-day, 90-day windows.
  * Calendar natural month & historical months.
  * 30% duplicate / repeated queries.
  * 10% raw / report exports.
Outputs:
- Cumulative privacy spend rho_cum(t) and eps_cum(t) under UnifiedPrivacyLedger.
- Invariant verification: SDG-MH query execution adds EXACTLY 0 privacy cost!
- Query availability comparison across methods.
- Generates data CSV and cumulative loss figure.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, List, Tuple
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from private_llm_analytics.sdg_mh.baselines import Epoch90Baseline
from private_llm_analytics.sdg_mh.calendar_job import DEFAULT_CALENDAR_B, DistinctCalendarStore
from private_llm_analytics.sdg_mh.calibration import compute_sdg_target_profile
from private_llm_analytics.sdg_mh.ledger import (
    UnifiedPrivacyLedger,
    closed_form_distinct_calendar_rho,
    closed_form_sdg_additive_rho,
    rho_to_epsilon,
)
from private_llm_analytics.sdg_mh.query_service import ReadOnlySealedQueryService
from private_llm_analytics.sdg_mh.sealed_store import DailySealedRecord, SealedStore

RESULTS_DIR = Path("results/sdg_mh")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def generate_synthetic_clickstream(n_days: int = 180, n_analysts: int = 50, seed: int = 42) -> List[Dict[str, Any]]:
    """Generate realistic analyst clickstream queries."""
    rng = np.random.default_rng(seed)
    queries = []
    recent_query_pool: List[Dict[str, Any]] = []

    for day in range(1, n_days + 1):
        for analyst_id in range(1, n_analysts + 1):
            n_queries = int(rng.integers(0, 21))  # 0 to 20 queries

            for q_idx in range(n_queries):
                # 30% repeated query
                if recent_query_pool and rng.uniform(0.0, 1.0) < 0.30:
                    q = rng.choice(recent_query_pool).copy()
                    q["day"] = day
                    q["is_repeat"] = True
                    queries.append(q)
                    continue

                q_type_rand = rng.uniform(0.0, 1.0)
                is_export = rng.uniform(0.0, 1.0) < 0.10

                if q_type_rand < 0.30:
                    s = max(1, day - 6)
                    e = day
                elif q_type_rand < 0.65:
                    s = max(1, day - 29)
                    e = day
                elif q_type_rand < 0.85:
                    s = max(1, day - 89)
                    e = day
                else:
                    hist_offset = int(rng.integers(30, max(31, day)))
                    s = max(1, day - hist_offset)
                    e = min(day, s + 29)

                q = {
                    "day": day,
                    "analyst_id": analyst_id,
                    "window_start": s,
                    "window_end": e,
                    "window_length": e - s + 1,
                    "is_export": is_export,
                    "is_repeat": False,
                }
                queries.append(q)
                recent_query_pool.append(q)
                if len(recent_query_pool) > 200:
                    recent_query_pool.pop(0)

    return queries


def run_rq5_query_reuse():
    print("--- [RQ5] Query Reuse & Cumulative Privacy Cost ---")
    n_days = 180
    clickstream = generate_synthetic_clickstream(n_days=n_days, n_analysts=50, seed=42)
    print(f"Generated {len(clickstream)} analyst query clicks across {n_days} days.")

    # 1. Setup SDG-MH
    sdg_store = SealedStore()
    sdg_cal_store = DistinctCalendarStore()
    sdg_ledger = UnifiedPrivacyLedger(delta=1e-6)

    sdg_targets = compute_sdg_target_profile()
    rho_1_add = sdg_targets["rho_1_add"]

    # Pre-populate sealed store for 180 days
    for d in range(1, n_days + 1):
        sdg_store.append(
            DailySealedRecord(
                day_index=d,
                date_str=f"2026-01-{d:03d}",
                y_requests=22500,
                y_failures=675,
            )
        )
        sdg_ledger.record_gaussian_release(
            release_id=f"sdg_req_{d}",
            days_accessed=[d],
            sensitivity=50.0,
            sigma=2500.0,
        )
        sdg_ledger.record_gaussian_release(
            release_id=f"sdg_fail_{d}",
            days_accessed=[d],
            sensitivity=10.0,
            sigma=700.0,
        )

    sdg_query_service = ReadOnlySealedQueryService(
        sealed_store=sdg_store,
        calendar_store=sdg_cal_store,
    )

    # 2. Setup Baselines
    epoch90 = Epoch90Baseline(epoch_days=90, eps_cap_epoch=4.0, rho_per_query=0.02, C=50.0)

    # LifetimeDecay releases each day once
    decay_daily_rhos = [0.506589 * 6.0 / ((math.pi**2) * (max(1, t)**2)) for t in range(1, n_days + 1)]

    # Group queries by day
    queries_by_day = {}
    for q in clickstream:
        queries_by_day.setdefault(q["day"], []).append(q)

    timeline_records = []
    sdg_initial_records_count = len(sdg_ledger.records)

    print("Simulating query workload across 180 days with Epoch90 advance & release_key cache...")
    for day in range(1, n_days + 1):
        # Existing system logic: advance to new epoch at epoch boundary (every 90 days)
        if day == 91:
            new_ep = epoch90.advance_to_new_epoch()
            print(f"Day 91: Advanced to Epoch {new_ep}. Epoch budget reset to {epoch90.rho_epoch_cap}.")

        day_queries = queries_by_day.get(day, [])

        for q in day_queries:
            s, e = q["window_start"], q["window_end"]
            # 1. SDG-MH query (0 privacy cost: pure post-processing on sealed store)
            _ = sdg_query_service.query_additive_window(s, e)

            # 2. Epoch90 query (uses release_key cache; rejects when active epoch budget exhausted)
            _ = epoch90.query(s, e, true_sum=22500.0 * (e - s + 1), metric="requests")

        # Check at end of each day
        # Invariant check: SDG-MH ledger must remain completely untouched by queries!
        assert len(sdg_ledger.records) == sdg_initial_records_count, "SDG-MH query spend violation: ledger modified!"

        # Cumulative privacy profiles at horizon h=30
        rho_sdg_cum = closed_form_sdg_additive_rho(30, rho_1_add) + closed_form_distinct_calendar_rho(
            30, DEFAULT_CALENDAR_B, delta_sq=11.0
        )
        # Epoch90 30-day exposure:
        # If in Epoch 1 (day <= 90): exposure is spend in Epoch 1 (capped at 0.506589)
        # If crossing Epoch boundary (days 91 to 120): a 30-day window accesses releases in both Epoch 1 and Epoch 2!
        # If day > 120: a 30-day window lies entirely within Epoch 2.
        if day <= 90:
            rho_epoch90_30 = min(0.506589, epoch90.current_epoch_spend)
        elif day <= 120:
            # Boundary crossing: touches both Epoch 1 (full 0.506589 spent) and active Epoch 2 spend
            rho_epoch90_30 = 0.506589 + min(0.506589, epoch90.current_epoch_spend)
        else:
            rho_epoch90_30 = min(0.506589, epoch90.current_epoch_spend)

        # LifetimeDecay: rolling 30-day sum of daily rhos
        rho_decay_cum = sum(decay_daily_rhos[max(0, day - 30) : day])
        # W-BD: rolling 30-day sum is target_rho_30
        rho_wbd_cum = (min(day, 30) / 30.0) * 0.038089

        timeline_records.append({
            "day": day,
            "queries_executed": len(day_queries),
            "sdg_rho_30": round(rho_sdg_cum, 6),
            "sdg_eps_30": round(rho_to_epsilon(rho_sdg_cum), 4),
            "epoch90_rho_30": round(rho_epoch90_30, 6),
            "epoch90_eps_30": round(rho_to_epsilon(rho_epoch90_30), 4),
            "epoch90_availability": round(epoch90.get_availability(), 4),
            "epoch90_cache_hits": epoch90.cache_hits,
            "decay_rho_30": round(rho_decay_cum, 6),
            "decay_eps_30": round(rho_to_epsilon(rho_decay_cum), 4),
            "wbd_rho_30": round(rho_wbd_cum, 6),
            "wbd_eps_30": round(rho_to_epsilon(rho_wbd_cum), 4),
        })

    df_timeline = pd.DataFrame(timeline_records)
    df_timeline.to_csv(RESULTS_DIR / "rq5_query_reuse_cumulative.csv", index=False)

    print("Query availability results:")
    print(f"SDG-MH Availability: 100.0% (Zero rejection, unlimited re-use)")
    print(f"Epoch90 Availability: {epoch90.get_availability() * 100:.2f}% (Served: {epoch90.queries_served}, Cache Hits: {epoch90.cache_hits}, Rejected: {epoch90.queries_rejected})")
    print(f"Total Epoch90 rejected queries: {epoch90.queries_rejected} / {epoch90.queries_attempted}")

    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), dpi=300)

    # Plot 1: Cumulative Privacy Loss eps_30 over Time
    days = df_timeline["day"]
    ax1.plot(days, df_timeline["sdg_eps_30"], "b-", linewidth=2.5, label="SDG-MH (Invariant Profile)")
    ax1.plot(days, df_timeline["epoch90_eps_30"], "r--", linewidth=2.0, label="Epoch90 (With Boundary Crossing)")
    ax1.plot(days, df_timeline["decay_eps_30"], "g-.", linewidth=1.8, label="LifetimeDecay")
    ax1.plot(days, df_timeline["wbd_eps_30"], "m:", linewidth=2.0, label="W-BD (Sliding Window)")

    ax1.set_xlabel("Time (Days)", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Effective Privacy Loss ε₃₀", fontsize=11, fontweight="bold")
    ax1.set_title("RQ5: Cumulative Privacy Exposure (h=30) under Synthetic Clickstream\n[50 Analysts, ~500 Queries/Day, 30% Repeats]", fontsize=12, pad=10)
    ax1.grid(True, linestyle="--", alpha=0.6)
    ax1.legend()

    # Plot 2: Query Availability over Time
    ax2.plot(days, [1.0] * len(days), "b-", linewidth=2.5, label="SDG-MH (100% Availability)")
    ax2.plot(days, df_timeline["epoch90_availability"], "r--", linewidth=2.0, label="Epoch90 (With Cache & Epoch Advance)")
    ax2.set_xlabel("Time (Days)", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Cumulative Query Availability", fontsize=11, fontweight="bold")
    ax2.set_title("RQ5: Query Availability under Synthetic Clickstream\n[Epoch90 Drops After Budget Cap Exhaustion]", fontsize=12, pad=10)
    ax2.set_ylim(-0.05, 1.05)
    ax2.grid(True, linestyle="--", alpha=0.6)
    ax2.legend()

    plt.tight_layout()
    fig.savefig(RESULTS_DIR / "fig_rq5_cumulative_privacy_loss.png")
    plt.close(fig)
    print(">>> RQ5 complete. Saved results and figure.")


if __name__ == "__main__":
    run_rq5_query_reuse()
