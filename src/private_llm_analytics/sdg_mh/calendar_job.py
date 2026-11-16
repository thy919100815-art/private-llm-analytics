"""Distinct-User Calendar Job for User Prevalence and Top-K Tasks.

Supports:
- Calendar A: (7, 1), (30, 7), (90, 30) with sigma_w = 60.
- Calendar B: (7, 7), (30, 7), (90, 30) with sigma_w = 60.
- Evaluates whether date e satisfies e mod s == 0.
- Computes distinct users for m services and top-k tasks (Delta^2 = m + 2k).
- Perturbs via Discrete Gaussian noise N_Z(0, sigma_w^2).
- Stores in DistinctCalendarStore and logs to UnifiedPrivacyLedger.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import math
from typing import Any, Dict, List, Optional, Sequence, Tuple
import numpy as np
import pandas as pd

from private_llm_analytics.sdg_mh.clipping import compute_distinct_window_histogram
from private_llm_analytics.sdg_mh.discrete_gaussian import sample_discrete_gaussian
from private_llm_analytics.sdg_mh.ledger import (
    PrivacyAdmissionError,
    ReleaseRecord,
    ReleaseStatus,
    UnifiedPrivacyLedger,
)


DEFAULT_CALENDAR_A = [(7, 1, 60.0), (30, 7, 60.0), (90, 30, 60.0)]
DEFAULT_CALENDAR_B = [(7, 7, 60.0), (30, 7, 60.0), (90, 30, 60.0)]

DEFAULT_DISTINCT_SERVICES = ["service_1", "service_2", "service_3", "service_4", "service_5"]
DEFAULT_DISTINCT_TASKS = [
    "task_01", "task_02", "task_03", "task_04", "task_05",
    "task_06", "task_07", "task_08", "task_09", "task_10",
    "task_11", "task_12", "task_13", "task_14", "task_15",
    "task_16", "task_17", "task_18", "task_19", "task_20",
]


@dataclass(frozen=True)
class CalendarReleaseRecord:
    """A single committed distinct-user calendar release."""

    window_length: int        # w
    step: int                 # s
    end_day: int              # e
    start_day: int            # e - w + 1
    sigma_w: float
    service_names: List[str]
    task_names: List[str]
    noisy_services: List[int]
    noisy_tasks: List[int]
    released_at: str


class DistinctCalendarStore:
    """Store of released distinct-user calendar histograms."""

    def __init__(self):
        # Key: (window_length, end_day) -> CalendarReleaseRecord
        self._releases: Dict[Tuple[int, int], CalendarReleaseRecord] = {}

    def append(self, record: CalendarReleaseRecord) -> None:
        key = (record.window_length, record.end_day)
        if key in self._releases:
            raise ValueError(
                f"Append-only violation: calendar release for window {record.window_length}, "
                f"end_day {record.end_day} already exists!"
            )
        self._releases[key] = record

    def get_release(self, window_length: int, end_day: int) -> Optional[CalendarReleaseRecord]:
        return self._releases.get((window_length, end_day))

    def find_latest_covering(self, w: int, requested_end_day: int) -> Optional[CalendarReleaseRecord]:
        """Find the latest available calendar release for window length w with end_day <= requested_end_day."""
        matching = [
            rec for (win, e), rec in self._releases.items()
            if win == w and e <= requested_end_day
        ]
        if not matching:
            return None
        return max(matching, key=lambda r: r.end_day)

    def all_releases(self) -> List[CalendarReleaseRecord]:
        return list(self._releases.values())


class CalendarJob:
    """Executes distinct-user calendar releases with exact phase and budget alignment."""

    def __init__(
        self,
        calendar_store: DistinctCalendarStore,
        ledger: UnifiedPrivacyLedger,
        calendar_schedule: Sequence[Union[Tuple[int, int, float], Tuple[int, int, float, int]]] = DEFAULT_CALENDAR_B,
        services_list: Sequence[str] = DEFAULT_DISTINCT_SERVICES,
        tasks_list: Sequence[str] = DEFAULT_DISTINCT_TASKS,
        k_tasks_per_user: int = 3,
        rolling_state: Optional[Any] = None,
    ):
        self.calendar_store = calendar_store
        self.ledger = ledger
        self.services_list = list(services_list)
        self.tasks_list = list(tasks_list)
        self.k_tasks_per_user = int(k_tasks_per_user)
        self.rolling_state = rolling_state
        self.delta_sq = float(len(self.services_list) + 2 * self.k_tasks_per_user)
        self.sensitivity = math.sqrt(self.delta_sq)

        # Normalize schedule to 4-tuples: (window_length, step, sigma_w, phase)
        norm_sched: List[Tuple[int, int, float, int]] = []
        for entry in calendar_schedule:
            if len(entry) == 3:
                w, s, sigma_w = entry
                norm_sched.append((int(w), int(s), float(sigma_w), 0))
            elif len(entry) == 4:
                w, s, sigma_w, phi = entry
                norm_sched.append((int(w), int(s), float(sigma_w), int(phi)))
            else:
                raise ValueError(f"Invalid calendar schedule entry format: {entry}")
        self.calendar_schedule = norm_sched

    @classmethod
    def from_planner_result(
        cls,
        calendar_store: DistinctCalendarStore,
        ledger: UnifiedPrivacyLedger,
        plan: Any,
        services_list: Sequence[str] = DEFAULT_DISTINCT_SERVICES,
        tasks_list: Sequence[str] = DEFAULT_DISTINCT_TASKS,
        k_tasks_per_user: int = 3,
        rolling_state: Optional[Any] = None,
    ) -> CalendarJob:
        """Create CalendarJob configured directly from PlanningResult (phases, budgets, noise)."""
        delta_sq = float(len(services_list) + 2 * int(k_tasks_per_user))
        sched: List[Tuple[int, int, float, int]] = []
        for fam, phi, cost in zip(plan.schedule.families, plan.schedule.phases, plan.distinct_costs):
            sigma = float(math.sqrt(delta_sq / (2.0 * max(1e-12, cost))))
            sched.append((fam.window_days, fam.step_days, sigma, int(phi)))
        return cls(
            calendar_store=calendar_store,
            ledger=ledger,
            calendar_schedule=sched,
            services_list=services_list,
            tasks_list=tasks_list,
            k_tasks_per_user=k_tasks_per_user,
            rolling_state=rolling_state,
        )

    def execute_day(
        self,
        current_day: int,
        df_history_fn: Optional[Any] = None,  # Legacy callable for backwards compatibility
        rolling_state: Optional[Any] = None,   # RollingDistinctState instance
        rng: Optional[np.random.Generator] = None,
        seed: Optional[int] = None,
    ) -> List[CalendarReleaseRecord]:
        """Check all entries in calendar schedule for current_day and release if current_day % s == phi % s."""
        executed: List[CalendarReleaseRecord] = []
        state = rolling_state if rolling_state is not None else self.rolling_state

        for w, s, sigma_w, phi in self.calendar_schedule:
            # Phase-aligned release condition: t % s == phi % s
            if current_day % s == phi % s:
                rel_id = f"calendar_{w}_{s}_day_{current_day}"

                # 1. Idempotency check: if release already exists in store, return existing
                existing = self.calendar_store.get_release(window_length=w, end_day=current_day)
                if existing is not None:
                    executed.append(existing)
                    continue

                start_day = max(1, current_day - w + 1)
                days_accessed = list(range(start_day, current_day + 1))
                rho_cost = float(self.delta_sq / (2.0 * sigma_w * sigma_w))

                # 2. Reserve in Unified Ledger (Fail-closed)
                cand = ReleaseRecord(
                    release_id=rel_id,
                    days_accessed=frozenset(days_accessed),
                    mechanism_type="gaussian",
                    params={"sigma": sigma_w},
                    sensitivity=self.sensitivity,
                    rho_cost=rho_cost,
                    tag=f"calendar_{w}_{s}",
                )
                try:
                    self.ledger.admit_and_commit_batch(
                        candidate_records=[cand],
                        as_status=ReleaseStatus.RESERVED.value,
                        allow_unconfigured_policy=getattr(self.ledger, "allow_unconfigured_policy", False),
                    )
                except PrivacyAdmissionError as e:
                    raise PrivacyAdmissionError(f"CalendarJob rejected release {rel_id}: {e}") from e

                # 3. Compute bounded distinct window histogram & add noise
                try:
                    if state is not None:
                        # Preferred: compute strictly from RollingDistinctState (no raw data access)
                        srv_counts, tsk_counts = state.compute_window_histogram(
                            start_day=start_day,
                            end_day=current_day,
                            services_list=self.services_list,
                            tasks_list=self.tasks_list,
                            k_tasks_per_user=self.k_tasks_per_user,
                        )
                    elif df_history_fn is not None:
                        # Legacy fallback
                        df_window = df_history_fn(start_day, current_day)
                        srv_counts, tsk_counts = compute_distinct_window_histogram(
                            df_window=df_window,
                            services_list=self.services_list,
                            tasks_list=self.tasks_list,
                            k_tasks_per_user=self.k_tasks_per_user,
                        )
                    else:
                        raise ValueError("Either rolling_state or df_history_fn must be provided to execute_day.")

                    # 4. Add discrete Gaussian noise
                    noise_srv = sample_discrete_gaussian(
                        sigma=sigma_w,
                        size=len(srv_counts),
                        rng=rng,
                        seed=seed,
                    )
                    noise_tsk = sample_discrete_gaussian(
                        sigma=sigma_w,
                        size=len(tsk_counts),
                        rng=rng,
                        seed=None if seed is None else seed + 500003,
                    )

                    noisy_srv = (srv_counts + noise_srv).astype(np.int64).tolist()
                    noisy_tsk = (tsk_counts + noise_tsk).astype(np.int64).tolist()

                    rec = CalendarReleaseRecord(
                        window_length=w,
                        step=s,
                        end_day=current_day,
                        start_day=start_day,
                        sigma_w=sigma_w,
                        service_names=self.services_list,
                        task_names=self.tasks_list,
                        noisy_services=noisy_srv,
                        noisy_tasks=noisy_tsk,
                        released_at=datetime.now(timezone.utc).isoformat(),
                    )
                except Exception:
                    # Output not generated or local only, safely cancel reservation
                    self.ledger.cancel_reservation(rel_id)
                    raise

                # 5. Append to store (with safe exposure check before cancellation)
                try:
                    self.calendar_store.append(rec)
                except ValueError as e:
                    if "already exists" in str(e):
                        # Concurrently sealed by another thread
                        existing = self.calendar_store.get_release(window_length=w, end_day=current_day)
                        if existing is not None:
                            executed.append(existing)
                            continue
                    raise
                except Exception:
                    # Check if output is stored/accessible
                    if self.calendar_store.get_release(window_length=w, end_day=current_day) is None:
                        # Output was not stored, cancel unexposed reservation
                        self.ledger.cancel_reservation(rel_id)
                    else:
                        # Stored or uncertain, conservatively commit privacy spend!
                        self.ledger.admit_and_commit_batch(
                            candidate_records=[cand],
                            as_status=ReleaseStatus.COMMITTED.value,
                            allow_unconfigured_policy=getattr(self.ledger, "allow_unconfigured_policy", False),
                        )
                    raise

                # 6. Commit in ledger atomically (irrevocable)
                try:
                    self.ledger.admit_and_commit_batch(
                        candidate_records=[cand],
                        as_status=ReleaseStatus.COMMITTED.value,
                        allow_unconfigured_policy=getattr(self.ledger, "allow_unconfigured_policy", False),
                    )
                except Exception:
                    # Output already stored in calendar_store, privacy spend is retained
                    raise

                executed.append(rec)

        return executed
