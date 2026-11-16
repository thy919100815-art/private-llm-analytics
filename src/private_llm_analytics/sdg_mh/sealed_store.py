"""Append-only Sealed Store for SDG additive daily publications.

Security Invariants:
1. Append-only: Existing days cannot be overwritten, modified, or deleted.
2. Complete provenance: Records exact date, sealed timestamp, noisy estimates, noise parameters.
3. Storage accounting: Tracks bytes on disk and projects annual storage requirement.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union
import numpy as np


@dataclass(frozen=True)
class DailySealedRecord:
    """Immutable record for a sealed day."""

    day_index: int            # 1-based chronological day
    date_str: str             # 'YYYY-MM-DD'
    y_requests: int           # Bounded + discrete Gaussian noise
    y_failures: int           # Bounded + discrete Gaussian noise
    y_providers: Optional[List[float]] = None  # Optional noisy provider vector
    sigma_r: float = 2500.0
    sigma_f: float = 700.0
    sigma_p: Optional[float] = None
    C_r: float = 50.0
    C_f: float = 10.0
    sealed_at: str = ""       # ISO-8601 UTC timestamp


class SealedStore:
    """Append-only persistent store for sealed daily additive records."""

    def __init__(self, storage_dir: Optional[Union[str, Path]] = None):
        self.storage_dir = Path(storage_dir) if storage_dir else None
        if self.storage_dir:
            self.storage_dir.mkdir(parents=True, exist_ok=True)
            self.jsonl_path = self.storage_dir / "sealed_store.jsonl"
        else:
            self.jsonl_path = None

        self._records: Dict[int, DailySealedRecord] = {}
        if self.jsonl_path and self.jsonl_path.exists():
            self._load_from_disk()

    def _load_from_disk(self) -> None:
        assert self.jsonl_path is not None
        with open(self.jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                d = json.loads(line)
                rec = DailySealedRecord(**d)
                self._records[rec.day_index] = rec

    def append(self, record: DailySealedRecord) -> None:
        """Append a newly sealed day atomically. Raises ValueError if day_index already exists."""
        if record.day_index in self._records:
            raise ValueError(
                f"Append-only violation: day_index {record.day_index} ({record.date_str}) is already sealed!"
            )

        # Prepare serialization before disk I/O
        rec_dict = asdict(record)
        line = json.dumps(rec_dict) + "\n"

        if self.jsonl_path:
            initial_size = self.jsonl_path.stat().st_size if self.jsonl_path.exists() else 0
            try:
                with open(self.jsonl_path, "a", encoding="utf-8") as f:
                    f.write(line)
                    f.flush()
                    import os
                    os.fsync(f.fileno())
            except Exception:
                # If write failed or partially wrote, truncate file back to initial_size to avoid corruption
                if self.jsonl_path.exists():
                    try:
                        with open(self.jsonl_path, "r+b") as f:
                            f.truncate(initial_size)
                    except Exception:
                        pass
                # Memory state remains untouched on failure
                raise

        # Memory is updated ONLY after disk append has completely succeeded
        self._records[record.day_index] = record

    def get_day(self, day_index: int) -> Optional[DailySealedRecord]:
        return self._records.get(day_index)

    def get_days_in_range(self, start_day: int, end_day: int) -> List[DailySealedRecord]:
        """Retrieve all sealed days in [start_day, end_day] sorted by day_index."""
        return [
            self._records[d]
            for d in range(start_day, end_day + 1)
            if d in self._records
        ]

    def total_sealed_days(self) -> int:
        return len(self._records)

    def __len__(self) -> int:
        return len(self._records)

    def get_storage_stats(self) -> Dict[str, Any]:
        """Compute current file size in bytes and annual storage projection."""
        bytes_on_disk = 0
        if self.jsonl_path and self.jsonl_path.exists():
            bytes_on_disk = self.jsonl_path.stat().st_size
        else:
            # Approximate in-memory representation
            sample_bytes = len(json.dumps(asdict(next(iter(self._records.values()))))) if self._records else 180
            bytes_on_disk = sample_bytes * len(self._records)

        avg_bytes_per_day = bytes_on_disk / max(1, len(self._records)) if self._records else 180.0
        annual_bytes = avg_bytes_per_day * 365.25

        return {
            "total_sealed_days": len(self._records),
            "bytes_on_disk": bytes_on_disk,
            "avg_bytes_per_day": round(avg_bytes_per_day, 1),
            "annual_projected_bytes": round(annual_bytes, 1),
            "annual_projected_kb": round(annual_bytes / 1024.0, 2),
        }
