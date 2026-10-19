"""Unified Persistent Privacy State Manager in v0.1.3.

Provides atomic budget compare-and-reserve, single-flight memoization,
fenced commit/abort, and crash-consistent release storage backed by SQLite (WAL mode).
Serves as the authoritative single source of truth for all release persistence
and privacy accounting.
"""

from contextlib import contextmanager
from datetime import datetime, timezone
from decimal import Decimal
import json
from pathlib import Path
import sqlite3
import time
from typing import Any, Dict, List, Optional, Tuple, Union
import uuid
import numpy as np

from private_llm_analytics.metrics.base import DPReleaseResult
from private_llm_analytics.privacy.budget import PrivacyBudget
from private_llm_analytics.privacy.exceptions import (
    CommitIrreversibleError,
    PrivacyBudgetExceededError,
    ReleaseInProgressError,
    ReservationLostError,
)


def _to_decimal(val: Any) -> Decimal:
    """Helper to convert float/int/str/Decimal to exact Decimal."""
    if isinstance(val, Decimal):
        return val
    return Decimal(str(val))


def _json_serializable(obj: Any) -> Any:
    """Helper to convert NumPy and collection types to JSON serializable objects."""
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


class ReservationTuple(tuple):
    """4-tuple for backward compatibility that exposes generation and owner_id."""

    def __new__(
        cls,
        status: str,
        result: Optional[DPReleaseResult],
        reservation_id: Optional[str],
        release_id: Optional[str],
        generation: int = 1,
        owner_id: str = "",
    ):
        return super().__new__(cls, (status, result, reservation_id, release_id))

    def __init__(
        self,
        status: str,
        result: Optional[DPReleaseResult],
        reservation_id: Optional[str],
        release_id: Optional[str],
        generation: int = 1,
        owner_id: str = "",
    ):
        self.status = status
        self.result = result
        self.reservation_id = reservation_id
        self.release_id = release_id
        self.generation = generation
        self.owner_id = owner_id


class PrivacyStateManager:
    """Authoritative single source of truth for privacy budget and release state backed by SQLite."""

    def __init__(self, db_path: Optional[Union[str, Path]] = None):
        self.db_path = Path(db_path) if db_path else Path("results/privacy_state.db")
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_sqlite()

    @contextmanager
    def _connection(self, timeout: float = 30.0):
        """Context manager yielding SQLite connection with WAL mode and robust busy timeout."""
        conn = sqlite3.connect(self.db_path, timeout=timeout)
        try:
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.execute("PRAGMA synchronous=NORMAL;")
            conn.execute("PRAGMA foreign_keys=ON;")
            conn.execute(f"PRAGMA busy_timeout={int(timeout * 1000)};")
            yield conn
        finally:
            conn.close()

    def _init_sqlite(self) -> None:
        """Initialize database schema if not present, and upgrade existing schemas seamlessly."""
        with self._connection() as conn:
            try:
                conn.execute("PRAGMA journal_mode=WAL;")
            except sqlite3.OperationalError:
                pass
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS privacy_reservations (
                    reservation_id TEXT PRIMARY KEY,
                    release_key TEXT NOT NULL,
                    owner_id TEXT NOT NULL DEFAULT '',
                    generation INTEGER NOT NULL DEFAULT 1,
                    epsilon TEXT NOT NULL,
                    delta TEXT NOT NULL,
                    status TEXT NOT NULL CHECK (status IN ('PENDING', 'COMMITTED', 'ABORTED')),
                    created_at TEXT NOT NULL,
                    committed_at TEXT
                )
            """)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS release_store (
                    release_key TEXT PRIMARY KEY,
                    reservation_id TEXT NOT NULL,
                    owner_id TEXT NOT NULL DEFAULT '',
                    generation INTEGER NOT NULL DEFAULT 1,
                    release_id TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    status TEXT NOT NULL CHECK (status IN ('PENDING', 'COMMITTED', 'ABORTED')),
                    canonical_semantics_json TEXT NOT NULL,
                    metric TEXT NOT NULL,
                    window_start TEXT,
                    window_end TEXT,
                    dataset_version TEXT NOT NULL,
                    mechanism TEXT NOT NULL,
                    epsilon TEXT NOT NULL,
                    delta TEXT NOT NULL,
                    sensitivity REAL NOT NULL,
                    noise_scale REAL NOT NULL,
                    cell_names_json TEXT NOT NULL,
                    n_users INTEGER NOT NULL,
                    public_population_size INTEGER,
                    dp_vector_json TEXT,
                    post_processed_json TEXT,
                    committed_at TEXT
                )
            """)
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
                    epsilon TEXT NOT NULL,
                    delta TEXT NOT NULL,
                    sensitivity REAL NOT NULL,
                    noise_scale REAL NOT NULL,
                    release_key TEXT NOT NULL,
                    cache_hit INTEGER NOT NULL
                )
            """)

            # Migration check: Ensure generation and owner_id columns exist on existing databases
            for table in ("privacy_reservations", "release_store"):
                cursor.execute(f"PRAGMA table_info({table})")
                cols = {row[1] for row in cursor.fetchall()}
                if "generation" not in cols:
                    cursor.execute(f"ALTER TABLE {table} ADD COLUMN generation INTEGER NOT NULL DEFAULT 1")
                if "owner_id" not in cols:
                    cursor.execute(f"ALTER TABLE {table} ADD COLUMN owner_id TEXT NOT NULL DEFAULT ''")

            # Migration check: Ensure decimal-safe TEXT schema for epsilon and delta (Fix C)
            for table in ("privacy_reservations", "release_store", "privacy_ledger"):
                cursor.execute(f"PRAGMA table_info({table})")
                col_types = {row[1]: (row[2] or "").upper() for row in cursor.fetchall()}
                if col_types.get("epsilon") == "REAL" or col_types.get("delta") == "REAL":
                    if table == "privacy_reservations":
                        cursor.execute("ALTER TABLE privacy_reservations RENAME TO _privacy_reservations_old")
                        cursor.execute("PRAGMA table_info(_privacy_reservations_old)")
                        old_cols = {row[1] for row in cursor.fetchall()}
                        cursor.execute("""
                            CREATE TABLE privacy_reservations (
                                reservation_id TEXT PRIMARY KEY,
                                release_key TEXT NOT NULL,
                                owner_id TEXT NOT NULL DEFAULT '',
                                generation INTEGER NOT NULL DEFAULT 1,
                                epsilon TEXT NOT NULL,
                                delta TEXT NOT NULL,
                                status TEXT NOT NULL CHECK (status IN ('PENDING', 'COMMITTED', 'ABORTED')),
                                created_at TEXT NOT NULL,
                                committed_at TEXT
                            )
                        """)
                        target_cols = [
                            "reservation_id", "release_key", "owner_id", "generation",
                            "epsilon", "delta", "status", "created_at", "committed_at"
                        ]
                        select_exprs = []
                        for col in target_cols:
                            if col in old_cols:
                                select_exprs.append(f"CAST({col} AS TEXT)" if col in ("epsilon", "delta") else col)
                            elif col == "owner_id":
                                select_exprs.append("''")
                            elif col == "generation":
                                select_exprs.append("1")
                            else:
                                select_exprs.append("NULL")
                        cursor.execute(f"""
                            INSERT INTO privacy_reservations ({', '.join(target_cols)})
                            SELECT {', '.join(select_exprs)}
                            FROM _privacy_reservations_old
                        """)
                        cursor.execute("DROP TABLE _privacy_reservations_old")
                    elif table == "release_store":
                        cursor.execute("ALTER TABLE release_store RENAME TO _release_store_old")
                        cursor.execute("PRAGMA table_info(_release_store_old)")
                        old_cols = {row[1] for row in cursor.fetchall()}
                        cursor.execute("""
                            CREATE TABLE release_store (
                                release_key TEXT PRIMARY KEY,
                                reservation_id TEXT NOT NULL,
                                owner_id TEXT NOT NULL DEFAULT '',
                                generation INTEGER NOT NULL DEFAULT 1,
                                release_id TEXT NOT NULL,
                                created_at TEXT NOT NULL,
                                status TEXT NOT NULL CHECK (status IN ('PENDING', 'COMMITTED', 'ABORTED')),
                                canonical_semantics_json TEXT NOT NULL,
                                metric TEXT NOT NULL,
                                window_start TEXT,
                                window_end TEXT,
                                dataset_version TEXT NOT NULL,
                                mechanism TEXT NOT NULL,
                                epsilon TEXT NOT NULL,
                                delta TEXT NOT NULL,
                                sensitivity REAL NOT NULL,
                                noise_scale REAL NOT NULL,
                                cell_names_json TEXT NOT NULL,
                                n_users INTEGER NOT NULL,
                                public_population_size INTEGER,
                                dp_vector_json TEXT,
                                post_processed_json TEXT,
                                committed_at TEXT
                            )
                        """)
                        target_cols = [
                            "release_key", "reservation_id", "owner_id", "generation", "release_id", "created_at", "status",
                            "canonical_semantics_json", "metric", "window_start", "window_end",
                            "dataset_version", "mechanism", "epsilon", "delta", "sensitivity", "noise_scale",
                            "cell_names_json", "n_users", "public_population_size", "dp_vector_json",
                            "post_processed_json", "committed_at"
                        ]
                        select_exprs = []
                        for col in target_cols:
                            if col in old_cols:
                                select_exprs.append(f"CAST({col} AS TEXT)" if col in ("epsilon", "delta") else col)
                            elif col == "owner_id":
                                select_exprs.append("''")
                            elif col == "generation":
                                select_exprs.append("1")
                            elif col == "status":
                                select_exprs.append("'COMMITTED'")
                            elif col == "canonical_semantics_json":
                                select_exprs.append("'{}'")
                            elif col == "cell_names_json":
                                select_exprs.append("'[]'")
                            elif col in ("dataset_version", "metric", "mechanism", "release_id", "created_at"):
                                select_exprs.append("''")
                            elif col in ("sensitivity", "noise_scale"):
                                select_exprs.append("0.0")
                            elif col == "n_users":
                                select_exprs.append("0")
                            else:
                                select_exprs.append("NULL")
                        cursor.execute(f"""
                            INSERT INTO release_store ({', '.join(target_cols)})
                            SELECT {', '.join(select_exprs)}
                            FROM _release_store_old
                        """)
                        cursor.execute("DROP TABLE _release_store_old")
                    elif table == "privacy_ledger":
                        cursor.execute("ALTER TABLE privacy_ledger RENAME TO _privacy_ledger_old")
                        cursor.execute("PRAGMA table_info(_privacy_ledger_old)")
                        old_cols = {row[1] for row in cursor.fetchall()}
                        cursor.execute("""
                            CREATE TABLE privacy_ledger (
                                release_id TEXT PRIMARY KEY,
                                timestamp TEXT NOT NULL,
                                metric TEXT NOT NULL,
                                recipient TEXT NOT NULL,
                                window_start TEXT,
                                window_end TEXT,
                                dataset_version TEXT NOT NULL,
                                mechanism TEXT NOT NULL,
                                epsilon TEXT NOT NULL,
                                delta TEXT NOT NULL,
                                sensitivity REAL NOT NULL,
                                noise_scale REAL NOT NULL,
                                release_key TEXT NOT NULL,
                                cache_hit INTEGER NOT NULL
                            )
                        """)
                        target_cols = [
                            "release_id", "timestamp", "metric", "recipient", "window_start", "window_end",
                            "dataset_version", "mechanism", "epsilon", "delta", "sensitivity", "noise_scale",
                            "release_key", "cache_hit"
                        ]
                        select_exprs = []
                        for col in target_cols:
                            if col in old_cols:
                                select_exprs.append(f"CAST({col} AS TEXT)" if col in ("epsilon", "delta") else col)
                            elif col == "release_id":
                                select_exprs.append("lower(hex(randomblob(16)))")
                            elif col in ("timestamp", "metric", "recipient", "dataset_version", "mechanism", "release_key"):
                                select_exprs.append("''")
                            elif col in ("sensitivity", "noise_scale"):
                                select_exprs.append("0.0")
                            elif col == "cache_hit":
                                select_exprs.append("0")
                            else:
                                select_exprs.append("NULL")
                        cursor.execute(f"""
                            INSERT INTO privacy_ledger ({', '.join(target_cols)})
                            SELECT {', '.join(select_exprs)}
                            FROM _privacy_ledger_old
                        """)
                        cursor.execute("DROP TABLE _privacy_ledger_old")

            conn.commit()

    def get_committed_release(self, release_key: str) -> Optional[DPReleaseResult]:
        """Lookup an already committed release by its canonical release key.

        Enforces the core invariant: Reusable DP payload => Authoritative committed accounting exists.
        """
        with self._connection() as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT r.* FROM release_store r
                JOIN privacy_reservations p ON r.reservation_id = p.reservation_id
                WHERE r.release_key = ? AND r.status = 'COMMITTED' AND p.status = 'COMMITTED'
                """,
                (release_key,),
            )
            row = cursor.fetchone()

        if row is None or not row["dp_vector_json"]:
            return None

        return self._row_to_release_result(row)

    def _row_to_release_result(self, row: sqlite3.Row) -> DPReleaseResult:
        dp_vec = np.array(json.loads(row["dp_vector_json"]), dtype=np.float64)
        pp_raw = json.loads(row["post_processed_json"])
        pp_reconstructed: Dict[str, Any] = {}
        for k, v in pp_raw.items():
            if k.endswith("_array") or k == "grid_3x3":
                pp_reconstructed[k] = np.array(v, dtype=np.float64)
            else:
                pp_reconstructed[k] = v

        return DPReleaseResult(
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

    def reserve_or_get_release(
        self,
        release_key: str,
        canonical_semantics: Dict[str, Any],
        global_budget: PrivacyBudget,
        budget_needed: PrivacyBudget,
        plan_metadata: Dict[str, Any],
        allow_overrun: bool = False,
        single_flight_timeout: float = 30.0,
        lease_timeout: float = 300.0,
        reuse_cache: bool = True,
        owner_id: Optional[str] = None,
    ) -> ReservationTuple:
        """Atomic compare-and-reserve transaction with single-flight memoization and lease fencing.

        Returns:
            ReservationTuple("CACHE_HIT", release_result, None, None, generation=gen, owner_id=owner)
            ReservationTuple("RESERVED", None, reservation_id, release_id, generation=gen, owner_id=owner)
        """
        start_wait = time.time()
        caller_owner = owner_id or str(uuid.uuid4())

        while True:
            with self._connection() as conn:
                conn.isolation_level = None  # Explicit transaction control
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                cursor.execute("BEGIN IMMEDIATE")
                try:
                    # 1. Check if release already exists in release_store
                    cursor.execute(
                        "SELECT * FROM release_store WHERE release_key = ?",
                        (release_key,),
                    )
                    row = cursor.fetchone()

                    if row is not None and reuse_cache:
                        status = row["status"]
                        if status == "COMMITTED" and row["dp_vector_json"]:
                            # Confirm corresponding committed reservation exists
                            cursor.execute(
                                "SELECT status FROM privacy_reservations WHERE reservation_id = ?",
                                (row["reservation_id"],),
                            )
                            p_row = cursor.fetchone()
                            if p_row and p_row["status"] == "COMMITTED":
                                cursor.execute("COMMIT")
                                return ReservationTuple(
                                    "CACHE_HIT",
                                    self._row_to_release_result(row),
                                    None,
                                    None,
                                    generation=row["generation"],
                                    owner_id=row["owner_id"],
                                )

                        elif status == "PENDING":
                            # Another worker is actively computing this release (Single-flight)
                            active_gen = row["generation"]
                            active_res_id = row["reservation_id"]
                            created_iso = row["created_at"]
                            cursor.execute("COMMIT")

                            now_utc = datetime.now(timezone.utc)
                            res_age = 0.0
                            if created_iso:
                                try:
                                    dt = datetime.fromisoformat(created_iso)
                                    if dt.tzinfo is None:
                                        dt = dt.replace(tzinfo=timezone.utc)
                                    res_age = (now_utc - dt).total_seconds()
                                except Exception:
                                    pass

                            # Only take over if the existing pending reservation has genuinely expired its lease
                            if res_age > lease_timeout:
                                cursor.execute("BEGIN IMMEDIATE")
                                cursor.execute(
                                    "SELECT status, generation, reservation_id FROM release_store WHERE release_key = ?",
                                    (release_key,),
                                )
                                check_row = cursor.fetchone()
                                if (
                                    check_row
                                    and check_row["status"] == "PENDING"
                                    and check_row["generation"] == active_gen
                                    and check_row["reservation_id"] == active_res_id
                                ):
                                    # Monotonically fence out the stale worker
                                    cursor.execute(
                                        "UPDATE privacy_reservations SET status = 'ABORTED' WHERE reservation_id = ? AND status = 'PENDING'",
                                        (active_res_id,),
                                    )
                                    new_gen = active_gen + 1
                                    new_res_id = str(uuid.uuid4())
                                    new_rel_id = str(uuid.uuid4())
                                    now_iso = now_utc.isoformat()

                                    cursor.execute(
                                        """
                                        INSERT INTO privacy_reservations (
                                            reservation_id, release_key, owner_id, generation,
                                            epsilon, delta, status, created_at
                                        ) VALUES (?, ?, ?, ?, ?, ?, 'PENDING', ?)
                                        """,
                                        (new_res_id, release_key, caller_owner, new_gen, str(_to_decimal(budget_needed.epsilon)), str(_to_decimal(budget_needed.delta)), now_iso),
                                    )
                                    cursor.execute(
                                        """
                                        UPDATE release_store
                                        SET reservation_id = ?, owner_id = ?, generation = ?, release_id = ?,
                                            created_at = ?, status = 'PENDING', epsilon = ?, delta = ?
                                        WHERE release_key = ? AND generation = ? AND status = 'PENDING'
                                        """,
                                        (new_res_id, caller_owner, new_gen, new_rel_id, now_iso, str(_to_decimal(budget_needed.epsilon)), str(_to_decimal(budget_needed.delta)), release_key, active_gen),
                                    )
                                    cursor.execute("COMMIT")
                                    return ReservationTuple(
                                        "RESERVED",
                                        None,
                                        new_res_id,
                                        new_rel_id,
                                        generation=new_gen,
                                        owner_id=caller_owner,
                                    )
                                else:
                                    cursor.execute("COMMIT")
                                    continue

                            # Not expired: wait for active worker
                            elapsed_wait = time.time() - start_wait
                            if elapsed_wait > single_flight_timeout:
                                raise ReleaseInProgressError(
                                    f"Release for key '{release_key}' is actively being computed by another worker "
                                    f"(single-flight wait timeout {single_flight_timeout}s exceeded)."
                                )

                            time.sleep(0.05)
                            continue

                    needed_eps = _to_decimal(budget_needed.epsilon)
                    needed_delta = _to_decimal(budget_needed.delta)
                    global_eps = _to_decimal(global_budget.epsilon)
                    global_delta = _to_decimal(global_budget.delta)

                    if not allow_overrun:
                        # Check privacy budget atomically across COMMITTED and PENDING using exact Decimal summation (Finding A)
                        cursor.execute("""
                            SELECT epsilon, delta
                            FROM privacy_reservations
                            WHERE status IN ('COMMITTED', 'PENDING')
                        """)
                        current_eps = Decimal("0.0")
                        current_delta = Decimal("0.0")
                        for r_eps, r_delta in cursor.fetchall():
                            current_eps += _to_decimal(r_eps)
                            current_delta += _to_decimal(r_delta)

                        eps_ok = (current_eps + needed_eps) <= global_eps
                        delta_ok = (current_delta + needed_delta) <= global_delta
                        if not (eps_ok and delta_ok):
                            raise PrivacyBudgetExceededError(
                                f"Requested budget (eps={budget_needed.epsilon}, delta={budget_needed.delta}) exceeds "
                                f"remaining persistent budget (global: eps={global_budget.epsilon}, delta={global_budget.delta}; "
                                f"reserved+spent: eps={float(current_eps):.4f}, delta={float(current_delta):.2e})"
                            )

                    # 3. Insert or update PENDING reservation
                    reservation_id = str(uuid.uuid4())
                    release_id = str(uuid.uuid4())
                    now_iso = datetime.now(timezone.utc).isoformat()
                    next_gen = (row["generation"] + 1) if (row is not None and "generation" in row.keys()) else 1

                    cursor.execute(
                        """
                        INSERT INTO privacy_reservations (
                            reservation_id, release_key, owner_id, generation,
                            epsilon, delta, status, created_at
                        ) VALUES (?, ?, ?, ?, ?, ?, 'PENDING', ?)
                        """,
                        (reservation_id, release_key, caller_owner, next_gen, str(needed_eps), str(needed_delta), now_iso),
                    )

                    if row is not None:
                        # Key exists in aborted state, update to PENDING with incremented generation
                        cursor.execute(
                            """
                            UPDATE release_store
                            SET reservation_id = ?, owner_id = ?, generation = ?, release_id = ?,
                                created_at = ?, status = 'PENDING', canonical_semantics_json = ?,
                                metric = ?, window_start = ?, window_end = ?, dataset_version = ?,
                                mechanism = ?, epsilon = ?, delta = ?, sensitivity = ?, noise_scale = ?,
                                cell_names_json = ?, n_users = ?, public_population_size = ?,
                                dp_vector_json = NULL, post_processed_json = NULL, committed_at = NULL
                            WHERE release_key = ?
                            """,
                            (
                                reservation_id,
                                caller_owner,
                                next_gen,
                                release_id,
                                now_iso,
                                json.dumps(canonical_semantics, sort_keys=True),
                                plan_metadata.get("metric", ""),
                                str(plan_metadata.get("window_start", "")) or None,
                                str(plan_metadata.get("window_end", "")) or None,
                                str(plan_metadata.get("dataset_version", "v1.0")),
                                plan_metadata.get("mechanism", ""),
                                str(needed_eps),
                                str(needed_delta),
                                plan_metadata.get("sensitivity", 1.0),
                                plan_metadata.get("noise_scale", 0.0),
                                json.dumps(plan_metadata.get("cell_names", [])),
                                plan_metadata.get("n_users", 0),
                                plan_metadata.get("public_population_size"),
                                release_key,
                            ),
                        )
                    else:
                        cursor.execute(
                            """
                            INSERT INTO release_store (
                                release_key, reservation_id, owner_id, generation, release_id, created_at, status,
                                canonical_semantics_json, metric, window_start, window_end,
                                dataset_version, mechanism, epsilon, delta, sensitivity, noise_scale,
                                cell_names_json, n_users, public_population_size, dp_vector_json,
                                post_processed_json
                            ) VALUES (?, ?, ?, ?, ?, ?, 'PENDING', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, NULL)
                            """,
                            (
                                release_key,
                                reservation_id,
                                caller_owner,
                                next_gen,
                                release_id,
                                now_iso,
                                json.dumps(canonical_semantics, sort_keys=True),
                                plan_metadata.get("metric", ""),
                                str(plan_metadata.get("window_start", "")) or None,
                                str(plan_metadata.get("window_end", "")) or None,
                                str(plan_metadata.get("dataset_version", "v1.0")),
                                plan_metadata.get("mechanism", ""),
                                str(needed_eps),
                                str(needed_delta),
                                plan_metadata.get("sensitivity", 1.0),
                                plan_metadata.get("noise_scale", 0.0),
                                json.dumps(plan_metadata.get("cell_names", [])),
                                plan_metadata.get("n_users", 0),
                                plan_metadata.get("public_population_size"),
                            ),
                        )

                    cursor.execute("COMMIT")
                    return ReservationTuple(
                        "RESERVED",
                        None,
                        reservation_id,
                        release_id,
                        generation=next_gen,
                        owner_id=caller_owner,
                    )

                except Exception:
                    try:
                        cursor.execute("ROLLBACK")
                    except Exception:
                        pass
                    raise

    def commit_release(
        self,
        reservation_id: str,
        release_key: str,
        release_id: str,
        release_result: DPReleaseResult,
        recipient: str,
        dataset_version: str,
        window_start: Optional[str] = None,
        window_end: Optional[str] = None,
        generation: Optional[int] = None,
        owner_id: Optional[str] = None,
    ) -> None:
        """Atomically mark reservation and release as COMMITTED, and record ledger audit event.

        Strictly fenced: Requires status='PENDING' and matching reservation_id (and generation).
        Raises ReservationLostError if rowcount != 1.
        """
        now_iso = datetime.now(timezone.utc).isoformat()
        dp_vec_json = json.dumps(_json_serializable(release_result.dp_vector))
        pp_json = json.dumps(_json_serializable(release_result.post_processed))
        cell_names_json = json.dumps(release_result.cell_names)

        with self._connection() as conn:
            conn.isolation_level = None
            cursor = conn.cursor()
            cursor.execute("BEGIN IMMEDIATE")
            try:
                # 1. Commit reservation with CAS fencing
                if generation is not None:
                    cursor.execute(
                        """
                        UPDATE privacy_reservations
                        SET status = 'COMMITTED', committed_at = ?
                        WHERE reservation_id = ? AND release_key = ? AND generation = ? AND status = 'PENDING'
                        """,
                        (now_iso, reservation_id, release_key, generation),
                    )
                else:
                    cursor.execute(
                        """
                        UPDATE privacy_reservations
                        SET status = 'COMMITTED', committed_at = ?
                        WHERE reservation_id = ? AND release_key = ? AND status = 'PENDING'
                        """,
                        (now_iso, reservation_id, release_key),
                    )

                if cursor.rowcount != 1:
                    raise ReservationLostError(
                        f"Commit rejected: Reservation '{reservation_id}' for release_key '{release_key}' "
                        f"was lost, expired, or already finalized (rowcount={cursor.rowcount})."
                    )

                # 2. Commit release payload with CAS fencing
                if generation is not None:
                    cursor.execute(
                        """
                        UPDATE release_store
                        SET status = 'COMMITTED',
                            dp_vector_json = ?,
                            post_processed_json = ?,
                            cell_names_json = ?,
                            noise_scale = ?,
                            committed_at = ?
                        WHERE release_key = ? AND reservation_id = ? AND generation = ? AND status = 'PENDING'
                        """,
                        (
                            dp_vec_json,
                            pp_json,
                            cell_names_json,
                            release_result.noise_scale,
                            now_iso,
                            release_key,
                            reservation_id,
                            generation,
                        ),
                    )
                else:
                    cursor.execute(
                        """
                        UPDATE release_store
                        SET status = 'COMMITTED',
                            dp_vector_json = ?,
                            post_processed_json = ?,
                            cell_names_json = ?,
                            noise_scale = ?,
                            committed_at = ?
                        WHERE release_key = ? AND reservation_id = ? AND status = 'PENDING'
                        """,
                        (
                            dp_vec_json,
                            pp_json,
                            cell_names_json,
                            release_result.noise_scale,
                            now_iso,
                            release_key,
                            reservation_id,
                        ),
                    )

                if cursor.rowcount != 1:
                    raise ReservationLostError(
                        f"Commit rejected: Release store record for key '{release_key}' "
                        f"was lost or superseded (rowcount={cursor.rowcount})."
                    )

                # 3. Insert audit ledger row
                cursor.execute(
                    """
                    INSERT INTO privacy_ledger (
                        release_id, timestamp, metric, recipient,
                        window_start, window_end, dataset_version,
                        mechanism, epsilon, delta, sensitivity,
                        noise_scale, release_key, cache_hit
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0)
                    """,
                    (
                        release_id,
                        now_iso,
                        release_result.metric_name,
                        recipient,
                        str(window_start) if window_start else None,
                        str(window_end) if window_end else None,
                        dataset_version,
                        release_result.mechanism,
                        str(_to_decimal(release_result.epsilon)),
                        str(_to_decimal(release_result.delta)),
                        release_result.sensitivity,
                        release_result.noise_scale,
                        release_key,
                    ),
                )

                cursor.execute("COMMIT")

            except Exception:
                try:
                    cursor.execute("ROLLBACK")
                except Exception:
                    pass
                raise

        # Append to jsonl mirror if ledger jsonl exists or requested
        jsonl_path = self.db_path.with_suffix(".jsonl")
        try:
            with open(jsonl_path, "a", encoding="utf-8") as f:
                event = {
                    "release_id": release_id,
                    "timestamp": now_iso,
                    "metric": release_result.metric_name,
                    "recipient": recipient,
                    "epsilon": release_result.epsilon,
                    "delta": release_result.delta,
                    "release_key": release_key,
                    "cache_hit": False,
                }
                f.write(json.dumps(event) + "\n")
        except Exception:
            pass

    def record_cache_hit_ledger(
        self,
        release_id: str,
        release_key: str,
        release_result: DPReleaseResult,
        recipient: str,
        dataset_version: str,
        window_start: Optional[str] = None,
        window_end: Optional[str] = None,
    ) -> None:
        """Record a cache hit in the audit ledger (0 budget consumed)."""
        now_iso = datetime.now(timezone.utc).isoformat()
        hit_id = str(uuid.uuid4())
        with self._connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO privacy_ledger (
                    release_id, timestamp, metric, recipient,
                    window_start, window_end, dataset_version,
                    mechanism, epsilon, delta, sensitivity,
                    noise_scale, release_key, cache_hit
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1)
                """,
                (
                    hit_id,
                    now_iso,
                    release_result.metric_name,
                    recipient,
                    str(window_start) if window_start else None,
                    str(window_end) if window_end else None,
                    dataset_version,
                    release_result.mechanism,
                    str(_to_decimal(release_result.epsilon)),
                    str(_to_decimal(release_result.delta)),
                    release_result.sensitivity,
                    release_result.noise_scale,
                    release_key,
                ),
            )
            conn.commit()

    def abort_reservation(
        self,
        reservation_id: str,
        release_key: str,
        generation: Optional[int] = None,
    ) -> None:
        """Mark reservation and release as ABORTED after a failed release computation.

        Must be conditional on status='PENDING' and matching reservation_id (and generation).
        Raises CommitIrreversibleError if the reservation is already COMMITTED.
        """
        with self._connection() as conn:
            conn.isolation_level = None
            cursor = conn.cursor()
            cursor.execute("BEGIN IMMEDIATE")
            try:
                cursor.execute(
                    "SELECT status FROM privacy_reservations WHERE reservation_id = ?",
                    (reservation_id,),
                )
                row = cursor.fetchone()
                if row is not None:
                    curr_status = row[0]
                    if curr_status == "COMMITTED":
                        cursor.execute("COMMIT")
                        raise CommitIrreversibleError(
                            f"Cannot abort reservation '{reservation_id}' because it has already been COMMITTED."
                        )
                    if curr_status == "ABORTED":
                        cursor.execute("COMMIT")
                        return

                if generation is not None:
                    cursor.execute(
                        """
                        UPDATE privacy_reservations
                        SET status = 'ABORTED'
                        WHERE reservation_id = ? AND release_key = ? AND generation = ? AND status = 'PENDING'
                        """,
                        (reservation_id, release_key, generation),
                    )
                    cursor.execute(
                        """
                        UPDATE release_store
                        SET status = 'ABORTED'
                        WHERE release_key = ? AND reservation_id = ? AND generation = ? AND status = 'PENDING'
                        """,
                        (release_key, reservation_id, generation),
                    )
                else:
                    cursor.execute(
                        """
                        UPDATE privacy_reservations
                        SET status = 'ABORTED'
                        WHERE reservation_id = ? AND release_key = ? AND status = 'PENDING'
                        """,
                        (reservation_id, release_key),
                    )
                    cursor.execute(
                        """
                        UPDATE release_store
                        SET status = 'ABORTED'
                        WHERE release_key = ? AND reservation_id = ? AND status = 'PENDING'
                        """,
                        (release_key, reservation_id),
                    )

                cursor.execute("COMMIT")
            except Exception:
                try:
                    cursor.execute("ROLLBACK")
                except Exception:
                    pass
                raise

    def recover_stale_reservations(self, max_age_seconds: float = 300.0, force: bool = False) -> int:
        """Abort lingering PENDING reservations older than max_age_seconds.

        Does NOT blindly abort active workers.
        """
        now_utc = datetime.now(timezone.utc)
        with self._connection() as conn:
            conn.isolation_level = None
            cursor = conn.cursor()
            cursor.execute("BEGIN IMMEDIATE")
            try:
                cursor.execute(
                    "SELECT reservation_id, release_key, created_at FROM privacy_reservations WHERE status = 'PENDING'"
                )
                pending_rows = cursor.fetchall()
                aborted_count = 0
                for res_id, r_key, created_iso in pending_rows:
                    is_stale = force
                    if not is_stale and created_iso:
                        try:
                            dt = datetime.fromisoformat(created_iso)
                            if dt.tzinfo is None:
                                dt = dt.replace(tzinfo=timezone.utc)
                            age = (now_utc - dt).total_seconds()
                            if age >= max_age_seconds:
                                is_stale = True
                        except Exception:
                            is_stale = True

                    if is_stale:
                        cursor.execute(
                            "UPDATE privacy_reservations SET status = 'ABORTED' WHERE reservation_id = ? AND status = 'PENDING'",
                            (res_id,),
                        )
                        cursor.execute(
                            "UPDATE release_store SET status = 'ABORTED' WHERE release_key = ? AND reservation_id = ? AND status = 'PENDING'",
                            (r_key, res_id),
                        )
                        aborted_count += 1
                cursor.execute("COMMIT")
                return aborted_count
            except Exception:
                try:
                    cursor.execute("ROLLBACK")
                except Exception:
                    pass
                raise

    def get_total_spent_decimal(self) -> Tuple[Decimal, Decimal]:
        """Return cumulative (spent_epsilon, spent_delta) committed in persistent storage using exact Decimal summation."""
        with self._connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT epsilon, delta
                FROM privacy_reservations
                WHERE status = 'COMMITTED'
            """)
            spent_eps = Decimal("0.0")
            spent_delta = Decimal("0.0")
            for r_eps, r_delta in cursor.fetchall():
                spent_eps += _to_decimal(r_eps)
                spent_delta += _to_decimal(r_delta)
            return spent_eps, spent_delta


    def list_releases(self) -> List[Dict[str, Any]]:
        """List all committed releases in release_store."""
        with self._connection() as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM release_store WHERE status = 'COMMITTED'")
            return [dict(row) for row in cursor.fetchall()]

    def get_total_spent(self) -> Tuple[float, float]:
        """Return cumulative (spent_epsilon, spent_delta) committed in persistent storage using exact Decimal summation."""
        eps, delta = self.get_total_spent_decimal()
        return float(eps), float(delta)

    def get_total_reserved_or_spent(self) -> Tuple[float, float]:
        """Return cumulative (epsilon, delta) either committed or actively reserved using exact Decimal summation."""
        with self._connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT epsilon, delta
                FROM privacy_reservations
                WHERE status IN ('COMMITTED', 'PENDING')
            """)
            eps = Decimal("0.0")
            delta = Decimal("0.0")
            for r_eps, r_delta in cursor.fetchall():
                eps += _to_decimal(r_eps)
                delta += _to_decimal(r_delta)
            return float(eps), float(delta)

    def clear(self) -> None:
        """Wipe all state (test-only helper)."""
        with self._connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM privacy_ledger")
            cursor.execute("DELETE FROM release_store")
            cursor.execute("DELETE FROM privacy_reservations")
            conn.commit()
