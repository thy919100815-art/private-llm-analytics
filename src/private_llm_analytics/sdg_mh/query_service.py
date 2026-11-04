"""Read-Only Sealed Query Service for SDG-MH.

Security Invariants:
1. Strict Permission-Level Raw Data Isolation:
   This service does NOT hold, receive, or access any raw interaction telemetry.
   It interacts strictly and exclusively with SealedStore and DistinctCalendarStore.
2. Zero Privacy Cost at Query Time:
   All queries are pure post-processing over sealed publications; 0 additional privacy spend.
3. Post-Processing:
   - Success Rate(W) = 1 - F_hat_W / R_hat_W (negative clipped to 0, rate clipped to [0, 1]).
   - Top-K tasks with tie-breaking: difference < 2 * sigma_w marked as tied.
   - Non-calendar windows: returns latest covering release with cutoff date, or sub-window bounds.
   - Analytical 95% confidence intervals: +/- 1.96 * sqrt(|W|) * sigma.
"""

from __future__ import annotations

import math
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple
import numpy as np

from private_llm_analytics.sdg_mh.calendar_job import DistinctCalendarStore
from private_llm_analytics.sdg_mh.sealed_store import SealedStore


class ReadOnlySealedQueryService:
    """Service providing differentially private analytics strictly from sealed stores."""

    def __init__(
        self,
        sealed_store: SealedStore,
        calendar_store: DistinctCalendarStore,
        public_population_size: Optional[int] = None,
    ):
        self.sealed_store = sealed_store
        self.calendar_store = calendar_store
        self.public_population_size = public_population_size
        self.query_latencies_ms: List[float] = []

    def query_additive_window(
        self,
        start_day: int,
        end_day: int,
    ) -> Dict[str, Any]:
        """Query additive metrics (Requests, Failures, Success Rate) for window [start_day, end_day]."""
        t0 = time.perf_counter()
        if start_day > end_day:
            raise ValueError(f"Invalid window: start_day {start_day} > end_day {end_day}")

        w_len = end_day - start_day + 1
        records = self.sealed_store.get_days_in_range(start_day, end_day)
        sealed_count = len(records)

        # Sum sealed noisy values
        sum_req = sum(r.y_requests for r in records)
        sum_fail = sum(r.y_failures for r in records)

        # Variance is |W| * sigma^2
        sigma_r = records[0].sigma_r if records else 2500.0
        sigma_f = records[0].sigma_f if records else 700.0

        var_r = float(w_len * (sigma_r ** 2))
        var_f = float(w_len * (sigma_f ** 2))
        std_r = math.sqrt(var_r)
        std_f = math.sqrt(var_f)

        # Post-processing: clamp negative values to 0
        clamped_req = max(0, sum_req)
        clamped_fail = min(clamped_req, max(0, sum_fail))

        # Success Rate = 1 - F / R
        if clamped_req > 0:
            raw_rate = 1.0 - (float(clamped_fail) / float(clamped_req))
            success_rate = max(0.0, min(1.0, raw_rate))
        else:
            success_rate = None

        # 95% confidence intervals
        ci_req = (max(0.0, sum_req - 1.96 * std_r), sum_req + 1.96 * std_r)
        ci_fail = (max(0.0, sum_fail - 1.96 * std_f), sum_fail + 1.96 * std_f)

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        self.query_latencies_ms.append(elapsed_ms)

        return {
            "window": [start_day, end_day],
            "window_length": w_len,
            "sealed_days_present": sealed_count,
            "requests_estimate": clamped_req,
            "failures_estimate": clamped_fail,
            "success_rate": round(success_rate, 4) if success_rate is not None else None,
            "requests_std": round(std_r, 2),
            "failures_std": round(std_f, 2),
            "requests_ci95": (round(ci_req[0], 1), round(ci_req[1], 1)),
            "failures_ci95": (round(ci_fail[0], 1), round(ci_fail[1], 1)),
        }

    def query_distinct_prevalence(
        self,
        window_length: int,
        end_day: int,
    ) -> Dict[str, Any]:
        """Query distinct-user prevalence for window length w."""
        t0 = time.perf_counter()
        exact_rel = self.calendar_store.get_release(window_length, end_day)

        if exact_rel is not None:
            # Exact release available
            srv_estimates = {
                name: max(0, count)
                for name, count in zip(exact_rel.service_names, exact_rel.noisy_services)
            }
            std = exact_rel.sigma_w
            ci = 1.96 * std
            srv_prevalence = {}
            if self.public_population_size and self.public_population_size > 0:
                srv_prevalence = {
                    name: min(1.0, count / self.public_population_size)
                    for name, count in srv_estimates.items()
                }

            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            self.query_latencies_ms.append(elapsed_ms)
            return {
                "status": "EXACT_CALENDAR_HIT",
                "window": [exact_rel.start_day, exact_rel.end_day],
                "services": srv_estimates,
                "prevalence_rates": srv_prevalence,
                "sigma_w": std,
                "ci95_margin": round(ci, 2),
                "is_stale": False,
            }

        # Fallback: latest covering release
        latest_rel = self.calendar_store.find_latest_covering(window_length, end_day)
        if latest_rel is not None:
            srv_estimates = {
                name: max(0, count)
                for name, count in zip(latest_rel.service_names, latest_rel.noisy_services)
            }
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            self.query_latencies_ms.append(elapsed_ms)
            return {
                "status": "LATEST_COVERING_FALLBACK",
                "requested_end_day": end_day,
                "actual_cutoff_day": latest_rel.end_day,
                "actual_window": [latest_rel.start_day, latest_rel.end_day],
                "services": srv_estimates,
                "sigma_w": latest_rel.sigma_w,
                "is_stale": True,
            }

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        self.query_latencies_ms.append(elapsed_ms)
        return {
            "status": "UNAVAILABLE",
            "message": f"No calendar release available covering window length {window_length} up to day {end_day}.",
        }

    def query_top_k_tasks(
        self,
        window_length: int,
        end_day: int,
        k: int = 5,
    ) -> Dict[str, Any]:
        """Query Top-K tasks with tie detection for differences < 2 * sigma_w."""
        t0 = time.perf_counter()
        rel = self.calendar_store.get_release(window_length, end_day)
        is_stale = False
        if rel is None:
            rel = self.calendar_store.find_latest_covering(window_length, end_day)
            is_stale = True

        if rel is None:
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            self.query_latencies_ms.append(elapsed_ms)
            return {"status": "UNAVAILABLE"}

        sigma_w = rel.sigma_w
        tie_threshold = 2.0 * sigma_w

        task_pairs = [
            (name, max(0, count))
            for name, count in zip(rel.task_names, rel.noisy_tasks)
        ]
        # Sort by count desc
        task_pairs.sort(key=lambda p: -p[1])

        # Detect ties
        ranked_items: List[Dict[str, Any]] = []
        current_rank = 1
        for i, (name, count) in enumerate(task_pairs[:max(k, 10)]):
            is_tied = False
            if i > 0 and abs(count - task_pairs[i - 1][1]) < tie_threshold:
                is_tied = True
            elif i < len(task_pairs) - 1 and abs(count - task_pairs[i + 1][1]) < tie_threshold:
                is_tied = True

            ranked_items.append({
                "rank": i + 1,
                "task": name,
                "noisy_distinct_users": count,
                "is_tied": is_tied,
            })

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        self.query_latencies_ms.append(elapsed_ms)

        return {
            "status": "OK",
            "window": [rel.start_day, rel.end_day],
            "is_stale": is_stale,
            "sigma_w": sigma_w,
            "top_k": ranked_items[:k],
            "all_ranked": ranked_items,
        }

    def get_latency_stats(self) -> Dict[str, float]:
        """Compute query latency percentiles (p50, p99, mean)."""
        if not self.query_latencies_ms:
            return {"count": 0, "p50_ms": 0.0, "p99_ms": 0.0, "mean_ms": 0.0}
        arr = np.array(self.query_latencies_ms)
        return {
            "count": len(arr),
            "p50_ms": round(float(np.percentile(arr, 50)), 3),
            "p99_ms": round(float(np.percentile(arr, 99)), 3),
            "mean_ms": round(float(np.mean(arr)), 3),
        }
