"""Rolling Distinct State for Long-Horizon Window De-duplication.

Enforces:
1. Physical raw data isolation:
   Raw interaction telemetry is ingested daily into this privacy-bounded state.
   CalendarJob reads exclusively from RollingDistinctState.
   Physical storage is kept isolated from SealedStore; QueryService has no access.
2. Bounded feature representation:
   - Services: compact bitmask (m bits <= 16 bits).
   - Tasks: sparse task interaction counts mapped to taxonomy indices.
3. Strict FIFO retention:
   Window horizon w_max = 90 days. Records older than w_max are immediately purged.
"""

from __future__ import annotations

import os
from pathlib import Path
import pickle
import sys
import time
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple
import numpy as np
import pandas as pd

from private_llm_analytics.ingestion.schema import (
    COL_PROVIDER,
    COL_TASK,
    COL_USER_ID,
)


class RollingDistinctState:
    """Manages rolling 90-day distinct state for multi-horizon calendar queries."""

    def __init__(
        self,
        storage_dir: Optional[Path] = None,
        w_max: int = 90,
    ):
        self.w_max = int(w_max)
        self.storage_dir = Path(storage_dir) if storage_dir is not None else Path("results/sdg_mh/rolling_state")
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        # In-memory daily state: day_index -> Dict[user_id, (service_mask, Dict[task_idx, count])]
        self._day_states: Dict[int, Dict[str, Tuple[int, Dict[int, int]]]] = {}
        self.update_latencies_ms: List[float] = []
        self.purge_latencies_ms: List[float] = []

    def append_day(
        self,
        day_index: int,
        df_day: pd.DataFrame,
        services_list: Sequence[str],
        tasks_list: Sequence[str],
        persist_disk: bool = True,
    ) -> float:
        """Ingest single day's interactions, extract distinct state, and purge expired days.

        Returns:
            Execution duration in milliseconds.
        """
        t0 = time.perf_counter()

        service_to_idx = {s: i for i, s in enumerate(services_list)}
        task_to_idx = {t: i for i, t in enumerate(tasks_list)}

        day_dict: Dict[str, Tuple[int, Dict[int, int]]] = {}

        if not df_day.empty:
            for uid, prov, task in zip(
                df_day[COL_USER_ID].tolist(),
                df_day[COL_PROVIDER].tolist(),
                df_day[COL_TASK].tolist(),
            ):
                u_key = str(uid)
                entry = day_dict.get(u_key)
                if entry is None:
                    s_mask = 0
                    t_counts: Dict[int, int] = {}
                else:
                    s_mask, t_counts = entry

                p_idx = service_to_idx.get(prov)
                if p_idx is not None:
                    s_mask |= (1 << p_idx)

                t_idx = task_to_idx.get(task)
                if t_idx is not None:
                    t_counts[t_idx] = t_counts.get(t_idx, 0) + 1

                day_dict[u_key] = (s_mask, t_counts)

        self._day_states[day_index] = day_dict

        # Persist to disk partition if requested
        if persist_disk:
            day_file = self.storage_dir / f"state_day_{day_index:05d}.pkl"
            with open(day_file, "wb") as f:
                pickle.dump(day_dict, f, protocol=pickle.HIGHEST_PROTOCOL)

        # Purge records older than w_max
        self.purge_expired(current_day=day_index, remove_disk=persist_disk)

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        self.update_latencies_ms.append(elapsed_ms)
        return elapsed_ms

    def purge_expired(self, current_day: int, remove_disk: bool = True) -> int:
        """Purge all records with day_index <= current_day - w_max."""
        t0 = time.perf_counter()
        cutoff_day = current_day - self.w_max
        purged_count = 0

        # Purge memory
        expired_days = [d for d in self._day_states if d <= cutoff_day]
        for d in expired_days:
            del self._day_states[d]
            purged_count += 1

        # Purge disk
        if remove_disk:
            for p in self.storage_dir.glob("state_day_*.pkl"):
                try:
                    # state_day_00001.pkl
                    stem = p.stem.replace("state_day_", "")
                    d_idx = int(stem)
                    if d_idx <= cutoff_day:
                        p.unlink(missing_ok=True)
                except ValueError:
                    pass

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        self.purge_latencies_ms.append(elapsed_ms)
        return purged_count

    def compute_window_histogram(
        self,
        start_day: int,
        end_day: int,
        services_list: Sequence[str],
        tasks_list: Sequence[str],
        k_tasks_per_user: int = 3,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute exact bounded distinct histograms for window [start_day, end_day]."""
        m = len(services_list)
        t_len = len(tasks_list)
        if start_day > end_day:
            raise ValueError(f"Invalid window: start_day {start_day} > end_day {end_day}")

        # Check retention
        missing_days = [d for d in range(start_day, end_day + 1) if d not in self._day_states]
        if missing_days:
            # Try reloading from disk if present
            for d in missing_days:
                day_file = self.storage_dir / f"state_day_{d:05d}.pkl"
                if day_file.exists():
                    with open(day_file, "rb") as f:
                        self._day_states[d] = pickle.load(f)
                else:
                    raise KeyError(f"Day {d} has been purged or is missing from RollingDistinctState.")

        # Aggregate user activities across window
        # user -> (combined_service_mask, combined_task_counts)
        user_srv_masks: Dict[str, int] = {}
        user_tsk_counts: Dict[str, Dict[int, int]] = {}

        for d in range(start_day, end_day + 1):
            day_dict = self._day_states[d]
            for uid, (s_mask, t_counts) in day_dict.items():
                user_srv_masks[uid] = user_srv_masks.get(uid, 0) | s_mask
                if t_counts:
                    u_t = user_tsk_counts.setdefault(uid, {})
                    for t_idx, cnt in t_counts.items():
                        u_t[t_idx] = u_t.get(t_idx, 0) + cnt

        # 1. Services distinct count
        service_counts = np.zeros(m, dtype=np.float64)
        for s_mask in user_srv_masks.values():
            for i in range(m):
                if (s_mask >> i) & 1:
                    service_counts[i] += 1.0

        # 2. Tasks distinct count (top-k per user with deterministic tie-breaking)
        task_counts = np.zeros(t_len, dtype=np.float64)
        for uid, t_map in user_tsk_counts.items():
            # Sort by count desc, then task index asc
            sorted_tasks = sorted(
                t_map.items(),
                key=lambda item: (-item[1], item[0]),
            )
            for t_idx, _ in sorted_tasks[:k_tasks_per_user]:
                if t_idx < t_len:
                    task_counts[t_idx] += 1.0

        return service_counts, task_counts

    def get_storage_footprint(self) -> Dict[str, Any]:
        """Compute current in-memory and on-disk footprint."""
        disk_bytes = sum(f.stat().st_size for f in self.storage_dir.glob("state_day_*.pkl") if f.is_file())
        total_user_days = sum(len(d) for d in self._day_states.values())
        mem_bytes = sys.getsizeof(self._day_states)
        for d, u_dict in self._day_states.items():
            mem_bytes += sys.getsizeof(u_dict)
            for uid, (mask, t_map) in u_dict.items():
                mem_bytes += sys.getsizeof(uid) + sys.getsizeof(mask) + sys.getsizeof(t_map)

        bytes_per_user_day = float(mem_bytes / total_user_days) if total_user_days > 0 else 0.0

        return {
            "active_days_in_state": len(self._day_states),
            "total_user_days": total_user_days,
            "memory_bytes": mem_bytes,
            "memory_mb": round(mem_bytes / (1024 * 1024), 3),
            "disk_bytes": disk_bytes,
            "disk_mb": round(disk_bytes / (1024 * 1024), 3),
            "bytes_per_user_day": round(bytes_per_user_day, 1),
        }
