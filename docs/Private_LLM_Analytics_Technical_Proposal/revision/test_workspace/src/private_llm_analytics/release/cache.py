from contextlib import contextmanager
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3
from typing import Any, Dict, Optional, Union
import uuid
import numpy as np

from private_llm_analytics.metrics.base import DPReleaseResult


def _json_serializable(obj: Any) -> Any:
    """Helper to convert NumPy types to Python native types for JSON serialization."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)
    if isinstance(obj, dict):
        return {k: _json_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_serializable(item) for item in obj]
    return obj


class PersistentReleaseStore:
    """[DEPRECATED in v0.1.3] Legacy release store backed by SQLite.

    In v0.1.3+, PrivacyStateManager serves as the authoritative single source of truth
    for all release persistence, memoization, and privacy accounting. This class is
    retained exclusively for backward-compatibility with legacy test fixtures.
    """

    def __init__(self, db_path: Optional[Union[str, Path]] = None):
        self.db_path = Path(db_path) if db_path else Path("results/release_cache.db")
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._memory_cache: Dict[str, DPReleaseResult] = {}
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
                CREATE TABLE IF NOT EXISTS release_store (
                    release_key TEXT PRIMARY KEY,
                    release_id TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    metric TEXT NOT NULL,
                    window_start TEXT,
                    window_end TEXT,
                    dataset_version TEXT NOT NULL,
                    mechanism TEXT NOT NULL,
                    epsilon REAL NOT NULL,
                    delta REAL NOT NULL,
                    sensitivity REAL NOT NULL,
                    noise_scale REAL NOT NULL,
                    cell_names_json TEXT NOT NULL,
                    n_users INTEGER NOT NULL,
                    public_population_size INTEGER,
                    dp_vector_json TEXT NOT NULL,
                    post_processed_json TEXT NOT NULL
                )
            """)
            conn.commit()

    def get(self, release_key: str) -> Optional[DPReleaseResult]:
        """Lookup release by canonical key from memory or persistent SQLite."""
        if release_key in self._memory_cache:
            return self._memory_cache[release_key]

        with self._connection() as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM release_store WHERE release_key = ?", (release_key,))
            row = cursor.fetchone()

        if row is None:
            return None

        # Reconstruct DPReleaseResult
        dp_vec = np.array(json.loads(row["dp_vector_json"]), dtype=np.float64)
        pp_raw = json.loads(row["post_processed_json"])

        # Re-convert array fields to numpy arrays
        pp_reconstructed: Dict[str, Any] = {}
        for k, v in pp_raw.items():
            if k.endswith("_array") or k == "grid_3x3":
                pp_reconstructed[k] = np.array(v, dtype=np.float64)
            else:
                pp_reconstructed[k] = v

        result = DPReleaseResult(
            metric_name=row["metric"],
            dp_vector=dp_vec,
            post_processed=pp_reconstructed,
            epsilon=float(row["epsilon"]),
            delta=float(row["delta"]),
            mechanism=row["mechanism"],
            noise_scale=float(row["noise_scale"]),
            sensitivity=float(row["sensitivity"]),
            cell_names=json.loads(row["cell_names_json"]),
            n_users=int(row["n_users"]),
            public_population_size=row["public_population_size"],
        )

        self._memory_cache[release_key] = result
        return result

    def put(
        self,
        release_key: str,
        result: DPReleaseResult,
        window_start: Optional[str] = None,
        window_end: Optional[str] = None,
        dataset_version: str = "v1.0",
    ) -> None:
        """Store DP release both in memory and persistently in SQLite."""
        self._memory_cache[release_key] = result

        release_id = str(uuid.uuid4())
        created_at = datetime.now(timezone.utc).isoformat()
        dp_vector_json = json.dumps(_json_serializable(result.dp_vector))
        pp_json = json.dumps(_json_serializable(result.post_processed))
        cell_names_json = json.dumps(result.cell_names)

        with self._connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT OR REPLACE INTO release_store (
                    release_key, release_id, created_at, metric,
                    window_start, window_end, dataset_version,
                    mechanism, epsilon, delta, sensitivity, noise_scale,
                    cell_names_json, n_users, public_population_size,
                    dp_vector_json, post_processed_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    release_key,
                    release_id,
                    created_at,
                    result.metric_name,
                    str(window_start) if window_start else None,
                    str(window_end) if window_end else None,
                    dataset_version,
                    result.mechanism,
                    result.epsilon,
                    result.delta,
                    result.sensitivity,
                    result.noise_scale,
                    cell_names_json,
                    result.n_users,
                    result.public_population_size,
                    dp_vector_json,
                    pp_json,
                ),
            )
            conn.commit()

    def contains(self, release_key: str) -> bool:
        return self.get(release_key) is not None

    def clear(self) -> None:
        self._memory_cache.clear()
        with self._connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM release_store")
            conn.commit()


# Backward compatibility alias
ReleaseCache = PersistentReleaseStore
