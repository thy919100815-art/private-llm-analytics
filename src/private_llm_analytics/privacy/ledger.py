from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3
from typing import List, Optional, Tuple, Union
import uuid


@dataclass
class LedgerEntry:
    release_id: str
    timestamp: str
    metric: str
    recipient: str
    window_start: Optional[str]
    window_end: Optional[str]
    dataset_version: str
    mechanism: str
    epsilon: float
    delta: float
    sensitivity: float
    noise_scale: float
    release_key: str
    cache_hit: bool


class PrivacyLedger:
    """Persistent audit ledger supporting SQLite and JSONL backends."""

    def __init__(self, db_path: Optional[Union[str, Path]] = None):
        self.db_path = Path(db_path) if db_path else Path("results/privacy_ledger.db")
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_sqlite()

    @contextmanager
    def _connection(self, timeout: float = 30.0):
        conn = sqlite3.connect(self.db_path, timeout=timeout)
        try:
            conn.execute(f"PRAGMA busy_timeout={int(timeout * 1000)};")
            conn.execute("PRAGMA synchronous=NORMAL;")
            with conn:
                yield conn
        finally:
            conn.close()

    def _init_sqlite(self):
        with self._connection() as conn:
            try:
                conn.execute("PRAGMA journal_mode=WAL;")
            except sqlite3.OperationalError:
                pass
            cursor = conn.cursor()

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS privacy_ledger (
                    release_id TEXT PRIMARY KEY,
                    timestamp TEXT NOT NULL,
                    metric TEXT NOT NULL,
                    recipient TEXT NOT NULL,
                    window_start TEXT,
                    window_end TEXT,
                    dataset_version TEXT NOT NULL,
                    mechanism TEXT NOT NULL,
                    epsilon REAL NOT NULL,
                    delta REAL NOT NULL,
                    sensitivity REAL NOT NULL,
                    noise_scale REAL NOT NULL,
                    release_key TEXT NOT NULL,
                    cache_hit INTEGER NOT NULL
                )
            """)
            conn.commit()

    def record_release(
        self,
        metric: str,
        recipient: str,
        dataset_version: str,
        mechanism: str,
        epsilon: float,
        delta: float,
        sensitivity: float,
        noise_scale: float,
        release_key: str,
        cache_hit: bool,
        window_start: Optional[str] = None,
        window_end: Optional[str] = None,
    ) -> LedgerEntry:
        """Record a DP release into the ledger."""
        entry = LedgerEntry(
            release_id=str(uuid.uuid4()),
            timestamp=datetime.now(timezone.utc).isoformat(),
            metric=metric,
            recipient=recipient,
            window_start=str(window_start) if window_start else None,
            window_end=str(window_end) if window_end else None,
            dataset_version=dataset_version,
            mechanism=mechanism,
            epsilon=float(epsilon),
            delta=float(delta),
            sensitivity=float(sensitivity),
            noise_scale=float(noise_scale),
            release_key=release_key,
            cache_hit=bool(cache_hit),
        )

        with self._connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO privacy_ledger (
                    release_id, timestamp, metric, recipient,
                    window_start, window_end, dataset_version,
                    mechanism, epsilon, delta, sensitivity,
                    noise_scale, release_key, cache_hit
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    entry.release_id,
                    entry.timestamp,
                    entry.metric,
                    entry.recipient,
                    entry.window_start,
                    entry.window_end,
                    entry.dataset_version,
                    entry.mechanism,
                    entry.epsilon,
                    entry.delta,
                    entry.sensitivity,
                    entry.noise_scale,
                    entry.release_key,
                    1 if entry.cache_hit else 0,
                ),
            )
            conn.commit()

        # Also append to a JSONL mirror for easy inspection
        jsonl_path = self.db_path.with_suffix(".jsonl")
        with open(jsonl_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(entry)) + "\n")

        return entry

    def get_total_spent(self) -> Tuple[float, float]:
        """Compute cumulative (spent_epsilon, spent_delta) from persistent non-cache-hit releases."""
        with self._connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT SUM(epsilon), SUM(delta) FROM privacy_ledger WHERE cache_hit = 0")
            row = cursor.fetchone()
            eps = float(row[0]) if row and row[0] is not None else 0.0
            delta = float(row[1]) if row and row[1] is not None else 0.0
            return eps, delta

    def get_entries(self, recipient: Optional[str] = None) -> List[LedgerEntry]:
        """Query entries from the ledger."""
        with self._connection() as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            if recipient:
                cursor.execute("SELECT * FROM privacy_ledger WHERE recipient = ?", (recipient,))
            else:
                cursor.execute("SELECT * FROM privacy_ledger")
            rows = cursor.fetchall()

        entries = []
        for r in rows:
            entries.append(
                LedgerEntry(
                    release_id=r["release_id"],
                    timestamp=r["timestamp"],
                    metric=r["metric"],
                    recipient=r["recipient"],
                    window_start=r["window_start"],
                    window_end=r["window_end"],
                    dataset_version=r["dataset_version"],
                    mechanism=r["mechanism"],
                    epsilon=r["epsilon"],
                    delta=r["delta"],
                    sensitivity=r["sensitivity"],
                    noise_scale=r["noise_scale"],
                    release_key=r["release_key"],
                    cache_hit=bool(r["cache_hit"]),
                )
            )
        return entries
