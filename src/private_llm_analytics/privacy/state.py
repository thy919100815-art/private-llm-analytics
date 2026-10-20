"""Unified Persistent Privacy State Manager in v0.1.3.

Provides atomic budget compare-and-reserve, single-flight memoization,
fenced commit/abort, and crash-consistent release storage backed by SQLite (WAL mode).
Serves as the authoritative single source of truth for all release persistence
and privacy accounting.
"""

from contextlib import contextmanager
from datetime import datetime, timezone
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import sqlite3
import time
from typing import Any, Dict, List, Optional, Tuple, Union
import uuid
import numpy as np

from private_llm_analytics.metrics.base import DPReleaseResult
from private_llm_analytics.privacy.budget import PrivacyBudget
from private_llm_analytics.privacy.composition import PrivacyView
from private_llm_analytics.privacy.exceptions import (
    CommitIrreversibleError,
    PrivacyBudgetExceededError,
    PrivacyConfigurationError,
    ProductionSafetyConfigurationError,
    ReleaseInProgressError,
    ReservationLostError,
)
from private_llm_analytics.privacy.rdp import (
    DEFAULT_RDP_ORDERS,
    compute_gaussian_rdp,
    compute_laplace_rdp_bound,
    compose_rdp_vectors,
    rdp_to_epsilon_delta,
    rdp_vector_from_json,
    rdp_vector_to_json,
    to_decimal,
    validate_delta,
    validate_orders,
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


def compute_recipient_policy_fingerprint(
    mode: str,
    groups: Optional[Dict[str, List[str]]] = None,
    recipient_roles: Optional[Dict[str, str]] = None,
) -> Tuple[str, str, str, str]:
    """Compute canonical policy fingerprint covering recipient composition mode, groups, and recipient roles."""
    mode_norm = str(mode).lower().strip()
    if groups and mode_norm == "collusion_group":
        canon_groups = {str(k): sorted([str(m) for m in v]) for k, v in sorted(groups.items())}
    else:
        canon_groups = {}
    canon_groups_str = json.dumps(canon_groups, sort_keys=True, separators=(",", ":"))

    if recipient_roles:
        canon_roles = {str(k): str(v).lower().strip() for k, v in sorted(recipient_roles.items())}
    else:
        canon_roles = {}
    canon_roles_str = json.dumps(canon_roles, sort_keys=True, separators=(",", ":"))

    payload = {
        "mode": mode_norm,
        "groups": json.loads(canon_groups_str),
        "roles": json.loads(canon_roles_str),
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    fp = hashlib.sha256(raw).hexdigest()
    return fp, mode_norm, canon_groups_str, canon_roles_str


class ReservationTuple(tuple):
    """4-tuple for backward compatibility that exposes generation, owner_id, and RDP metadata."""

    def __new__(
        cls,
        status: str,
        result: Optional[DPReleaseResult],
        reservation_id: Optional[str],
        release_id: Optional[str],
        generation: int = 1,
        owner_id: str = "",
        effective_epsilon: Optional[Any] = None,
        best_alpha: Optional[Any] = None,
        budget_scope_id: str = "global_external",
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
        effective_epsilon: Optional[Any] = None,
        best_alpha: Optional[Any] = None,
        budget_scope_id: str = "global_external",
    ):
        self.status = status
        self.result = result
        self.reservation_id = reservation_id
        self.release_id = release_id
        self.generation = generation
        self.owner_id = owner_id
        self.effective_epsilon = effective_epsilon
        self.best_alpha = best_alpha
        self.budget_scope_id = budget_scope_id


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
                    committed_at TEXT,
                    budget_scope_id TEXT NOT NULL DEFAULT 'global_external',
                    accountant_mode TEXT NOT NULL DEFAULT 'basic',
                    rdp_cost_json TEXT,
                    rdp_orders_json TEXT
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
                    committed_at TEXT,
                    recipient TEXT NOT NULL DEFAULT '',
                    recipient_role TEXT NOT NULL DEFAULT 'external',
                    budget_scope_id TEXT NOT NULL DEFAULT 'global_external',
                    accountant_mode TEXT NOT NULL DEFAULT 'basic',
                    rdp_cost_json TEXT,
                    rdp_orders_json TEXT,
                    delta_global TEXT,
                    effective_epsilon TEXT,
                    best_alpha TEXT
                )
            """)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS privacy_policy_metadata (
                    id INTEGER PRIMARY KEY CHECK (id = 1),
                    policy_fingerprint TEXT NOT NULL,
                    recipient_composition_mode TEXT NOT NULL,
                    canonicalized_collusion_groups TEXT NOT NULL,
                    canonicalized_recipient_roles TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
            """)
            # Upgrade privacy_policy_metadata schema if missing id or canonicalized_recipient_roles
            cursor.execute("PRAGMA table_info(privacy_policy_metadata)")
            ppm_cols = {row[1] for row in cursor.fetchall()}
            if ppm_cols and "id" not in ppm_cols:
                cursor.execute("ALTER TABLE privacy_policy_metadata RENAME TO _ppm_old")
                cursor.execute("""
                    CREATE TABLE privacy_policy_metadata (
                        id INTEGER PRIMARY KEY CHECK (id = 1),
                        policy_fingerprint TEXT NOT NULL,
                        recipient_composition_mode TEXT NOT NULL,
                        canonicalized_collusion_groups TEXT NOT NULL,
                        canonicalized_recipient_roles TEXT NOT NULL,
                        created_at TEXT NOT NULL
                    )
                """)
                cursor.execute("""
                    INSERT OR IGNORE INTO privacy_policy_metadata (id, policy_fingerprint, recipient_composition_mode, canonicalized_collusion_groups, canonicalized_recipient_roles, created_at)
                    SELECT 1, policy_fingerprint, recipient_composition_mode, canonicalized_collusion_groups, '{}', created_at
                    FROM _ppm_old LIMIT 1
                """)
                cursor.execute("DROP TABLE _ppm_old")
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
                    cache_hit INTEGER NOT NULL,
                    budget_scope_id TEXT NOT NULL DEFAULT 'global_external',
                    accountant_mode TEXT NOT NULL DEFAULT 'basic',
                    rdp_cost_json TEXT,
                    effective_epsilon TEXT,
                    best_alpha TEXT
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

            # v0.2 Longitudinal Schema Extensions
            reservations_new_cols = {
                "budget_scope_id": "TEXT NOT NULL DEFAULT 'global_external'",
                "accountant_mode": "TEXT NOT NULL DEFAULT 'basic'",
                "rdp_cost_json": "TEXT",
                "rdp_orders_json": "TEXT",
            }
            cursor.execute("PRAGMA table_info(privacy_reservations)")
            res_cols = {row[1] for row in cursor.fetchall()}
            for c_name, c_def in reservations_new_cols.items():
                if c_name not in res_cols:
                    cursor.execute(f"ALTER TABLE privacy_reservations ADD COLUMN {c_name} {c_def}")

            release_store_new_cols = {
                "recipient": "TEXT NOT NULL DEFAULT ''",
                "recipient_role": "TEXT NOT NULL DEFAULT 'external'",
                "budget_scope_id": "TEXT NOT NULL DEFAULT 'global_external'",
                "accountant_mode": "TEXT NOT NULL DEFAULT 'basic'",
                "rdp_cost_json": "TEXT",
                "rdp_orders_json": "TEXT",
                "delta_global": "TEXT",
                "effective_epsilon": "TEXT",
                "best_alpha": "TEXT",
            }
            cursor.execute("PRAGMA table_info(release_store)")
            rel_cols = {row[1] for row in cursor.fetchall()}
            for c_name, c_def in release_store_new_cols.items():
                if c_name not in rel_cols:
                    cursor.execute(f"ALTER TABLE release_store ADD COLUMN {c_name} {c_def}")

            ledger_new_cols = {
                "budget_scope_id": "TEXT NOT NULL DEFAULT 'global_external'",
                "accountant_mode": "TEXT NOT NULL DEFAULT 'basic'",
                "rdp_cost_json": "TEXT",
                "effective_epsilon": "TEXT",
                "best_alpha": "TEXT",
            }
            cursor.execute("PRAGMA table_info(privacy_ledger)")
            led_cols = {row[1] for row in cursor.fetchall()}
            for c_name, c_def in ledger_new_cols.items():
                if c_name not in led_cols:
                    cursor.execute(f"ALTER TABLE privacy_ledger ADD COLUMN {c_name} {c_def}")

            # Indices for performance
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_rel_scope_status ON release_store (budget_scope_id, status)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_res_scope_status ON privacy_reservations (budget_scope_id, status)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_led_scope ON privacy_ledger (budget_scope_id)")

            # Deterministic migration of legacy Gaussian & Laplace records to RDP vectors (v0.2 Section D)
            cursor.execute("""
                SELECT release_key, mechanism, sensitivity, noise_scale, reservation_id, metric, epsilon, delta, budget_scope_id
                FROM release_store
                WHERE rdp_cost_json IS NULL AND status = 'COMMITTED'
            """)
            legacy_releases = cursor.fetchall()
            for r_key, mech, sens, sig, r_res_id, r_metric, r_eps, r_del, r_scope in legacy_releases:
                mech_str = str(mech).lower() if mech else ""
                del_float = float(r_del) if r_del is not None else 0.0

                # Strict fail-closed mechanism validation: blank or unknown mechanisms rejected without guessing (Item 6)
                if "gaussian" in mech_str:
                    if sens is not None and sens > 0 and sig is not None and sig > 0:
                        rdp_v = compute_gaussian_rdp(sigma=sig, sensitivity=sens, orders=DEFAULT_RDP_ORDERS)
                        rdp_j = rdp_vector_to_json(rdp_v)
                        orders_j = json.dumps([float(o) for o in DEFAULT_RDP_ORDERS])
                        cursor.execute(
                            "UPDATE release_store SET rdp_cost_json = ?, rdp_orders_json = ?, sensitivity = ? WHERE release_key = ?",
                            (rdp_j, orders_j, sens, r_key),
                        )
                        cursor.execute(
                            "UPDATE privacy_reservations SET rdp_cost_json = ?, rdp_orders_json = ? WHERE release_key = ?",
                            (rdp_j, orders_j, r_key),
                        )
                    else:
                        raise ValueError(
                            f"Fail-closed: Legacy Gaussian release '{r_key}' missing valid sensitivity/sigma for RDP migration "
                            f"(sensitivity={sens}, noise_scale={sig})."
                        )
                elif "laplace" in mech_str:
                    eps_dec = _to_decimal(r_eps) if r_eps is not None else Decimal("0")
                    if eps_dec > Decimal("0"):
                        rdp_v = compute_laplace_rdp_bound(epsilon=float(eps_dec), orders=DEFAULT_RDP_ORDERS)
                        rdp_j = rdp_vector_to_json(rdp_v)
                        orders_j = json.dumps([float(o) for o in DEFAULT_RDP_ORDERS])
                        cursor.execute(
                            "UPDATE release_store SET rdp_cost_json = ?, rdp_orders_json = ? WHERE release_key = ?",
                            (rdp_j, orders_j, r_key),
                        )
                        cursor.execute(
                            "UPDATE privacy_reservations SET rdp_cost_json = ?, rdp_orders_json = ? WHERE release_key = ?",
                            (rdp_j, orders_j, r_key),
                        )
                    else:
                        raise ValueError(
                            f"Fail-closed: Legacy Laplace release '{r_key}' has invalid or non-positive epsilon ({r_eps})."
                        )
                else:
                    raise ValueError(
                        f"Fail-closed: Legacy release '{r_key}' has missing, blank, or unsupported mechanism '{mech}'."
                    )

            # Synchronize any remaining reservations from matching release_store entries
            cursor.execute("""
                UPDATE privacy_reservations
                SET rdp_cost_json = (
                    SELECT rdp_cost_json FROM release_store
                    WHERE release_store.release_key = privacy_reservations.release_key
                      AND release_store.rdp_cost_json IS NOT NULL
                    LIMIT 1
                ),
                rdp_orders_json = (
                    SELECT rdp_orders_json FROM release_store
                    WHERE release_store.release_key = privacy_reservations.release_key
                      AND release_store.rdp_orders_json IS NOT NULL
                    LIMIT 1
                )
                WHERE rdp_cost_json IS NULL AND EXISTS (
                    SELECT 1 FROM release_store
                    WHERE release_store.release_key = privacy_reservations.release_key
                      AND release_store.rdp_cost_json IS NOT NULL
                )
            """)

            # Forbid leaving COMMITTED + rdp_cost_json = NULL for external historical releases (Section D)
            cursor.execute("""
                SELECT COUNT(*) FROM release_store
                WHERE rdp_cost_json IS NULL AND status = 'COMMITTED' AND budget_scope_id != 'internal'
            """)
            null_count = cursor.fetchone()[0]
            if null_count > 0:
                raise ValueError(f"Fail-closed: Database contains {null_count} external COMMITTED releases with NULL rdp_cost_json.")

            conn.commit()

    def has_external_releases(self) -> bool:
        """Check if there are any committed or pending external releases or reservations in the database."""
        with self._connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT 1 FROM release_store
                WHERE (recipient_role = 'external' OR budget_scope_id != 'internal')
                  AND status = 'COMMITTED'
                LIMIT 1
            """)
            if cursor.fetchone() is not None:
                return True
            cursor.execute("""
                SELECT 1 FROM privacy_reservations
                WHERE budget_scope_id != 'internal'
                  AND status IN ('COMMITTED', 'PENDING')
                LIMIT 1
            """)
            return cursor.fetchone() is not None

    def has_policy_bound(self) -> bool:
        """Check if privacy policy metadata is already persisted."""
        with self._connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT 1 FROM privacy_policy_metadata WHERE id = 1")
            return cursor.fetchone() is not None

    def get_active_budget_scopes(self, exclude_internal: bool = True) -> List[str]:
        """Return distinct active budget scope IDs from release_store and privacy_reservations."""
        with self._connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT DISTINCT budget_scope_id FROM release_store WHERE status = 'COMMITTED'")
            scopes = {row[0] for row in cursor.fetchall()}
            cursor.execute("SELECT DISTINCT budget_scope_id FROM privacy_reservations WHERE status IN ('COMMITTED', 'PENDING')")
            scopes.update({row[0] for row in cursor.fetchall()})
        if exclude_internal:
            scopes.discard("internal")
        return sorted(list(scopes))

    def _verify_or_bind_policy_tx(
        self,
        cursor: sqlite3.Cursor,
        mode: str,
        groups: Optional[Dict[str, List[str]]] = None,
        recipient_roles: Optional[Dict[str, str]] = None,
    ) -> None:
        """Internal helper to verify or bind policy inside an active transaction.

        Enforces:
          1. Singleton row in privacy_policy_metadata (id = 1).
          2. Policy fingerprint covers mode, groups, and recipient roles.
          3. If DB has external releases and fingerprint mismatches -> ProductionSafetyConfigurationError.
          4. If legacy DB has external releases but no policy metadata -> only 'global_external' is permitted (Item 3).
        """
        cur_fp, cur_mode, cur_groups_str, cur_roles_str = compute_recipient_policy_fingerprint(
            mode, groups, recipient_roles
        )
        now_iso = datetime.now(timezone.utc).isoformat()
        cursor.execute(
            "SELECT policy_fingerprint, recipient_composition_mode FROM privacy_policy_metadata WHERE id = 1"
        )
        row = cursor.fetchone()
        if row is not None:
            persisted_fp, persisted_mode = row[0], row[1]
            if persisted_fp != cur_fp:
                cursor.execute("""
                    SELECT 1 FROM release_store
                    WHERE (recipient_role = 'external' OR budget_scope_id != 'internal')
                      AND status = 'COMMITTED'
                    LIMIT 1
                """)
                has_rel = cursor.fetchone() is not None
                if not has_rel:
                    cursor.execute("""
                        SELECT 1 FROM privacy_reservations
                        WHERE budget_scope_id != 'internal'
                          AND status IN ('COMMITTED', 'PENDING')
                        LIMIT 1
                    """)
                    has_rel = cursor.fetchone() is not None

                if has_rel:
                    raise ProductionSafetyConfigurationError(
                        f"Recipient composition policy mismatch with persisted historical releases! "
                        f"Persisted mode='{persisted_mode}', fingerprint='{persisted_fp}'; "
                        f"Current config mode='{cur_mode}', fingerprint='{cur_fp}'. "
                        "Modifying composition policy on a database with existing releases is strictly forbidden."
                    )
                else:
                    cursor.execute("DELETE FROM privacy_policy_metadata WHERE id = 1")
                    cursor.execute(
                        """
                        INSERT INTO privacy_policy_metadata (
                            id, policy_fingerprint, recipient_composition_mode,
                            canonicalized_collusion_groups, canonicalized_recipient_roles, created_at
                        ) VALUES (1, ?, ?, ?, ?, ?)
                        """,
                        (cur_fp, cur_mode, cur_groups_str, cur_roles_str, now_iso),
                    )
        else:
            # First policy binding: check if legacy DB with external releases exists
            cursor.execute("""
                SELECT 1 FROM release_store
                WHERE (recipient_role = 'external' OR budget_scope_id != 'internal')
                  AND status = 'COMMITTED'
                LIMIT 1
            """)
            has_rel = cursor.fetchone() is not None
            if not has_rel:
                cursor.execute("""
                    SELECT 1 FROM privacy_reservations
                    WHERE budget_scope_id != 'internal'
                      AND status IN ('COMMITTED', 'PENDING')
                    LIMIT 1
                """)
                has_rel = cursor.fetchone() is not None

            if has_rel:
                # Legacy DB with existing external releases: only GLOBAL_EXTERNAL permitted (Item 3)
                if cur_mode != "global_external":
                    raise ProductionSafetyConfigurationError(
                        f"Legacy database with existing external releases can only be migrated to 'global_external' composition policy; "
                        f"attempted mode '{cur_mode}'."
                    )

            # Insert singleton row (id = 1)
            cursor.execute(
                """
                INSERT INTO privacy_policy_metadata (
                    id, policy_fingerprint, recipient_composition_mode,
                    canonicalized_collusion_groups, canonicalized_recipient_roles, created_at
                ) VALUES (1, ?, ?, ?, ?, ?)
                """,
                (cur_fp, cur_mode, cur_groups_str, cur_roles_str, now_iso),
            )

    def verify_or_bind_policy(
        self,
        mode: str,
        groups: Optional[Dict[str, List[str]]] = None,
        recipient_roles: Optional[Dict[str, str]] = None,
    ) -> None:
        """Verify current recipient composition policy against persisted metadata, or bind if first release."""
        with self._connection() as conn:
            conn.isolation_level = None
            cursor = conn.cursor()
            cursor.execute("BEGIN IMMEDIATE")
            try:
                self._verify_or_bind_policy_tx(
                    cursor=cursor,
                    mode=mode,
                    groups=groups,
                    recipient_roles=recipient_roles,
                )
                cursor.execute("COMMIT")
            except Exception:
                try:
                    cursor.execute("ROLLBACK")
                except Exception:
                    pass
                raise

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

        res = self._row_to_release_result(row)
        if "effective_epsilon" in row.keys() and row["effective_epsilon"]:
            setattr(res, "effective_epsilon", row["effective_epsilon"])
        if "best_alpha" in row.keys() and row["best_alpha"]:
            setattr(res, "best_alpha", row["best_alpha"])
        return res

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
        accountant_mode: str = "basic",
        budget_scope_id: str = "global_external",
        recipient: str = "",
        recipient_role: str = "external",
        rdp_orders: Optional[List[Any]] = None,
        rdp_candidate_cost: Optional[Dict[Any, Any]] = None,
        epsilon_cap: Optional[Any] = None,
        delta_global: Optional[Any] = None,
        policy_mode: Optional[str] = None,
        policy_groups: Optional[Dict[str, List[str]]] = None,
        recipient_roles: Optional[Dict[str, str]] = None,
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
                    # 0. Atomic policy binding & verification inside reservation transaction (Item 1 & 2 & 3)
                    if policy_mode is not None and (recipient_role == "external" or budget_scope_id != "internal"):
                        self._verify_or_bind_policy_tx(
                            cursor=cursor,
                            mode=policy_mode,
                            groups=policy_groups,
                            recipient_roles=recipient_roles,
                        )

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
                                    effective_epsilon=row["effective_epsilon"] if "effective_epsilon" in row.keys() else None,
                                    best_alpha=row["best_alpha"] if "best_alpha" in row.keys() else None,
                                    budget_scope_id=row["budget_scope_id"] if "budget_scope_id" in row.keys() else budget_scope_id,
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
                                    rdp_j = rdp_vector_to_json(rdp_candidate_cost) if rdp_candidate_cost else None
                                    orders_j = json.dumps([float(o) for o in (rdp_orders or DEFAULT_RDP_ORDERS)]) if rdp_orders else None
                                    delta_g = str(_to_decimal(delta_global)) if delta_global is not None else str(_to_decimal(global_budget.delta))

                                    cursor.execute(
                                        """
                                        INSERT INTO privacy_reservations (
                                            reservation_id, release_key, owner_id, generation,
                                            epsilon, delta, status, created_at,
                                            budget_scope_id, accountant_mode, rdp_cost_json, rdp_orders_json
                                        ) VALUES (?, ?, ?, ?, ?, ?, 'PENDING', ?, ?, ?, ?, ?)
                                        """,
                                        (
                                            new_res_id,
                                            release_key,
                                            caller_owner,
                                            new_gen,
                                            str(_to_decimal(budget_needed.epsilon)),
                                            str(_to_decimal(budget_needed.delta)),
                                            now_iso,
                                            budget_scope_id,
                                            accountant_mode,
                                            rdp_j,
                                            orders_j,
                                        ),
                                    )
                                    cursor.execute(
                                        """
                                        UPDATE release_store
                                        SET reservation_id = ?, owner_id = ?, generation = ?, release_id = ?,
                                            created_at = ?, status = 'PENDING', epsilon = ?, delta = ?,
                                            recipient = ?, recipient_role = ?, budget_scope_id = ?,
                                            accountant_mode = ?, rdp_cost_json = ?, rdp_orders_json = ?, delta_global = ?
                                        WHERE release_key = ? AND generation = ? AND status = 'PENDING'
                                        """,
                                        (
                                            new_res_id,
                                            caller_owner,
                                            new_gen,
                                            new_rel_id,
                                            now_iso,
                                            str(_to_decimal(budget_needed.epsilon)),
                                            str(_to_decimal(budget_needed.delta)),
                                            recipient,
                                            recipient_role,
                                            budget_scope_id,
                                            accountant_mode,
                                            rdp_j,
                                            orders_j,
                                            delta_g,
                                            release_key,
                                            active_gen,
                                        ),
                                    )
                                    cursor.execute("COMMIT")
                                    return ReservationTuple(
                                        "RESERVED",
                                        None,
                                        new_res_id,
                                        new_rel_id,
                                        generation=new_gen,
                                        owner_id=caller_owner,
                                        budget_scope_id=budget_scope_id,
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
                    global_eps = _to_decimal(epsilon_cap) if epsilon_cap is not None else _to_decimal(global_budget.epsilon)
                    global_delta = _to_decimal(delta_global) if delta_global is not None else _to_decimal(global_budget.delta)

                    # 2. Check privacy budget atomically across COMMITTED and PENDING in scope
                    if not allow_overrun:
                        if accountant_mode == "rdp" and budget_scope_id != "internal":

                            eff_orders = [to_decimal(o) for o in (rdp_orders or DEFAULT_RDP_ORDERS)]
                            eff_delta = validate_delta(global_delta)

                            if not rdp_candidate_cost:
                                raise PrivacyConfigurationError(
                                    f"Fail-closed: RDP mode authorization in scope '{budget_scope_id}' requires valid non-empty rdp_candidate_cost."
                                )

                            cursor.execute(
                                """
                                SELECT reservation_id, epsilon, delta, rdp_cost_json, status
                                FROM privacy_reservations
                                WHERE budget_scope_id = ? AND status IN ('COMMITTED', 'PENDING')
                                """,
                                (budget_scope_id,),
                            )
                            active_rows = cursor.fetchall()
                            vecs = []
                            for r in active_rows:
                                rdp_json = r["rdp_cost_json"]
                                if not rdp_json:
                                    raise PrivacyConfigurationError(
                                        f"Fail-closed: Reservation '{r['reservation_id']}' (status={r['status']}) in scope '{budget_scope_id}' "
                                        "has missing or NULL rdp_cost_json. Authorization forbidden."
                                    )
                                try:
                                    vecs.append(rdp_vector_from_json(rdp_json))
                                except Exception as e:
                                    raise PrivacyConfigurationError(
                                        f"Fail-closed: Reservation '{r['reservation_id']}' in scope '{budget_scope_id}' "
                                        f"has corrupt rdp_cost_json: {e}. Authorization forbidden."
                                    )

                            try:
                                cand_vec = {to_decimal(k): to_decimal(v) for k, v in rdp_candidate_cost.items()}
                                for o in eff_orders:
                                    if o not in cand_vec:
                                        raise ValueError(f"Missing required order {o} in candidate RDP vector")
                                vecs.append(cand_vec)
                            except Exception as e:
                                raise PrivacyConfigurationError(f"Fail-closed: Invalid rdp_candidate_cost: {e}")

                            if vecs:
                                composed = compose_rdp_vectors(vecs, orders=eff_orders)
                                best_a, rdp_eff = rdp_to_epsilon_delta(composed, delta_global=eff_delta)
                                eff_eps = rdp_eff
                                if eff_eps > global_eps:
                                    raise PrivacyBudgetExceededError(
                                        f"Requested release in scope '{budget_scope_id}' exceeds RDP privacy budget cap: "
                                        f"resulting effective epsilon {float(eff_eps):.4f} > cap {float(global_eps):.4f} "
                                        f"(at delta={eff_delta}, best order alpha={best_a})."
                                    )

                        else:
                            cursor.execute(
                                """
                                SELECT epsilon, delta
                                FROM privacy_reservations
                                WHERE budget_scope_id = ? AND status IN ('COMMITTED', 'PENDING')
                                """,
                                (budget_scope_id,),
                            )
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
                                    f"remaining persistent budget for scope '{budget_scope_id}' "
                                    f"(cap: eps={global_eps}, delta={global_delta}; "
                                    f"reserved+spent: eps={float(current_eps):.4f}, delta={float(current_delta):.2e})"
                                )

                    # 3. Insert or update PENDING reservation
                    reservation_id = str(uuid.uuid4())
                    release_id = str(uuid.uuid4())
                    now_iso = datetime.now(timezone.utc).isoformat()
                    next_gen = (row["generation"] + 1) if (row is not None and "generation" in row.keys()) else 1
                    rdp_j = rdp_vector_to_json(rdp_candidate_cost) if rdp_candidate_cost else None
                    orders_j = json.dumps([float(o) for o in (rdp_orders or DEFAULT_RDP_ORDERS)]) if rdp_orders else None
                    delta_g = str(global_delta)

                    cursor.execute(
                        """
                        INSERT INTO privacy_reservations (
                            reservation_id, release_key, owner_id, generation,
                            epsilon, delta, status, created_at,
                            budget_scope_id, accountant_mode, rdp_cost_json, rdp_orders_json
                        ) VALUES (?, ?, ?, ?, ?, ?, 'PENDING', ?, ?, ?, ?, ?)
                        """,
                        (
                            reservation_id,
                            release_key,
                            caller_owner,
                            next_gen,
                            str(needed_eps),
                            str(needed_delta),
                            now_iso,
                            budget_scope_id,
                            accountant_mode,
                            rdp_j,
                            orders_j,
                        ),
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
                                dp_vector_json = NULL, post_processed_json = NULL, committed_at = NULL,
                                recipient = ?, recipient_role = ?, budget_scope_id = ?,
                                accountant_mode = ?, rdp_cost_json = ?, rdp_orders_json = ?, delta_global = ?
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
                                recipient,
                                recipient_role,
                                budget_scope_id,
                                accountant_mode,
                                rdp_j,
                                orders_j,
                                delta_g,
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
                                post_processed_json, recipient, recipient_role, budget_scope_id,
                                accountant_mode, rdp_cost_json, rdp_orders_json, delta_global
                            ) VALUES (?, ?, ?, ?, ?, ?, 'PENDING', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, NULL, ?, ?, ?, ?, ?, ?, ?)
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
                                recipient,
                                recipient_role,
                                budget_scope_id,
                                accountant_mode,
                                rdp_j,
                                orders_j,
                                delta_g,
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
                        budget_scope_id=budget_scope_id,
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
        budget_scope_id: str = "global_external",
        accountant_mode: str = "basic",
        rdp_cost_json: Optional[str] = None,
        effective_epsilon: Optional[Any] = None,
        best_alpha: Optional[Any] = None,
        delta_global: Optional[Any] = None,
        orders: Optional[Sequence[Any]] = None,
    ) -> Tuple[Optional[Decimal], Optional[Decimal]]:
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
            conn.row_factory = sqlite3.Row
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

                # Fetch reservation metadata if needed
                cursor.execute(
                    "SELECT budget_scope_id, accountant_mode, rdp_cost_json FROM privacy_reservations WHERE reservation_id = ?",
                    (reservation_id,),
                )
                res_row = cursor.fetchone()
                if res_row is not None:
                    if not budget_scope_id or budget_scope_id == "global_external":
                        budget_scope_id = res_row["budget_scope_id"] or budget_scope_id
                    if not accountant_mode or accountant_mode == "basic":
                        accountant_mode = res_row["accountant_mode"] or accountant_mode
                    if not rdp_cost_json:
                        rdp_cost_json = res_row["rdp_cost_json"]

                if not rdp_cost_json and release_result is not None:
                    eff_orders = [to_decimal(o) for o in (orders or DEFAULT_RDP_ORDERS)]
                    sig = float(getattr(release_result, "noise_scale", 0.0) or 0.0)
                    sens = float(getattr(release_result, "sensitivity", 0.0) or 0.0)
                    mech_str = str(getattr(release_result, "mechanism", "") or "").lower()
                    del_val = float(getattr(release_result, "delta", 0.0) or 0.0)
                    eps_val = float(getattr(release_result, "epsilon", 0.0) or 0.0)

                    if "gaussian" in mech_str:
                        if sig > 0 and sens > 0:
                            rdp_v = compute_gaussian_rdp(sigma=sig, sensitivity=sens, orders=eff_orders)
                            rdp_cost_json = rdp_vector_to_json(rdp_v)
                        else:
                            raise ValueError(
                                f"Fail-closed: Gaussian mechanism missing valid sensitivity/sigma (sensitivity={sens}, noise_scale={sig})."
                            )
                    elif "laplace" in mech_str:
                        if eps_val > 0:
                            rdp_v = compute_laplace_rdp_bound(epsilon=eps_val, orders=eff_orders)
                            rdp_cost_json = rdp_vector_to_json(rdp_v)
                        else:
                            raise ValueError(
                                f"Fail-closed: Laplace mechanism missing positive epsilon ({eps_val})."
                            )
                    elif mech_str:
                        raise ValueError(
                            f"Fail-closed: Unknown or unsupported mechanism '{mech_str}'."
                        )

                if rdp_cost_json:
                    cursor.execute(
                        "UPDATE privacy_reservations SET rdp_cost_json = COALESCE(rdp_cost_json, ?) WHERE reservation_id = ?",
                        (rdp_cost_json, reservation_id),
                    )

                eff_eps_dec = to_decimal(effective_epsilon) if effective_epsilon is not None else None
                best_alpha_dec = to_decimal(best_alpha) if best_alpha is not None else None

                # Compute cumulative effective epsilon and best alpha for this scope if not supplied
                if eff_eps_dec is None and accountant_mode == "rdp":
                    cursor.execute(
                        "SELECT delta_global FROM release_store WHERE release_key = ?",
                        (release_key,),
                    )
                    dg_row = cursor.fetchone()
                    raw_dg = dg_row["delta_global"] if dg_row and dg_row["delta_global"] else delta_global
                    eff_orders = [to_decimal(o) for o in (orders or DEFAULT_RDP_ORDERS)]
                    eff_delta = validate_delta(raw_dg) if raw_dg is not None else Decimal("1.0e-5")

                    cursor.execute(
                        "SELECT reservation_id, epsilon, rdp_cost_json FROM privacy_reservations WHERE budget_scope_id = ? AND status = 'COMMITTED'",
                        (budget_scope_id,),
                    )
                    committed_rdps = cursor.fetchall()
                    vecs = []
                    for r in committed_rdps:
                        if not r["rdp_cost_json"]:
                            raise ValueError(
                                f"Fail-closed: Committed reservation '{r['reservation_id']}' in scope '{budget_scope_id}' "
                                "has missing or NULL rdp_cost_json."
                            )
                        try:
                            vecs.append(rdp_vector_from_json(r["rdp_cost_json"]))
                        except Exception as e:
                            raise ValueError(
                                f"Fail-closed: Corrupt rdp_cost_json in committed reservation '{r['reservation_id']}': {e}"
                            )
                    if vecs:
                        composed = compose_rdp_vectors(vecs, orders=eff_orders)
                        best_alpha_dec, rdp_eff = rdp_to_epsilon_delta(composed, delta_global=eff_delta)
                        eff_eps_dec = rdp_eff

                eff_eps_str = str(eff_eps_dec) if eff_eps_dec is not None else None
                best_alpha_str = str(best_alpha_dec) if best_alpha_dec is not None else None


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
                            committed_at = ?,
                            effective_epsilon = ?,
                            best_alpha = ?,
                            rdp_cost_json = COALESCE(?, rdp_cost_json)
                        WHERE release_key = ? AND reservation_id = ? AND generation = ? AND status = 'PENDING'
                        """,
                        (
                            dp_vec_json,
                            pp_json,
                            cell_names_json,
                            release_result.noise_scale,
                            now_iso,
                            eff_eps_str,
                            best_alpha_str,
                            rdp_cost_json,
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
                            committed_at = ?,
                            effective_epsilon = ?,
                            best_alpha = ?,
                            rdp_cost_json = COALESCE(?, rdp_cost_json)
                        WHERE release_key = ? AND reservation_id = ? AND status = 'PENDING'
                        """,
                        (
                            dp_vec_json,
                            pp_json,
                            cell_names_json,
                            release_result.noise_scale,
                            now_iso,
                            eff_eps_str,
                            best_alpha_str,
                            rdp_cost_json,
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
                        noise_scale, release_key, cache_hit,
                        budget_scope_id, accountant_mode, rdp_cost_json,
                        effective_epsilon, best_alpha
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, ?, ?, ?, ?, ?)
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
                        budget_scope_id,
                        accountant_mode,
                        rdp_cost_json,
                        eff_eps_str,
                        best_alpha_str,
                    ),
                )

                cursor.execute("COMMIT")
                return (eff_eps_dec, best_alpha_dec)

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
        budget_scope_id: str = "global_external",
        accountant_mode: str = "basic",
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
                    noise_scale, release_key, cache_hit,
                    budget_scope_id, accountant_mode
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1, ?, ?)
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
                    budget_scope_id,
                    accountant_mode,
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

    def get_total_spent_decimal(self, budget_scope_id: Optional[str] = None) -> Tuple[Decimal, Decimal]:
        """Return cumulative (spent_epsilon, spent_delta) committed in persistent storage using exact Decimal summation."""
        with self._connection() as conn:
            cursor = conn.cursor()
            if budget_scope_id is not None:
                cursor.execute(
                    """
                    SELECT epsilon, delta
                    FROM privacy_reservations
                    WHERE status = 'COMMITTED' AND budget_scope_id = ?
                    """,
                    (budget_scope_id,),
                )
            else:
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

    def list_releases(self, budget_scope_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """List all committed releases in release_store, optionally filtered by budget_scope_id."""
        with self._connection() as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            if budget_scope_id is not None:
                cursor.execute("SELECT * FROM release_store WHERE status = 'COMMITTED' AND budget_scope_id = ?", (budget_scope_id,))
            else:
                cursor.execute("SELECT * FROM release_store WHERE status = 'COMMITTED'")
            return [dict(row) for row in cursor.fetchall()]

    def get_total_spent(self, budget_scope_id: Optional[str] = None) -> Tuple[float, float]:
        """Return cumulative (spent_epsilon, spent_delta) committed in persistent storage using exact Decimal summation."""
        eps, delta = self.get_total_spent_decimal(budget_scope_id=budget_scope_id)
        return float(eps), float(delta)

    def get_total_reserved_or_spent(self, budget_scope_id: Optional[str] = None) -> Tuple[float, float]:
        """Return cumulative (epsilon, delta) either committed or actively reserved using exact Decimal summation."""
        with self._connection() as conn:
            cursor = conn.cursor()
            if budget_scope_id is not None:
                cursor.execute(
                    """
                    SELECT epsilon, delta
                    FROM privacy_reservations
                    WHERE status IN ('COMMITTED', 'PENDING') AND budget_scope_id = ?
                    """,
                    (budget_scope_id,),
                )
            else:
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

    def get_privacy_view(
        self,
        budget_scope_id: str = "global_external",
        delta_global: Optional[Union[float, Decimal]] = None,
        orders: Optional[Sequence[Union[float, Decimal]]] = None,
        epsilon_cap: Optional[Union[float, Decimal]] = None,
        accountant_mode: Optional[str] = None,
    ) -> PrivacyView:
        """Reconstruct authoritative PrivacyView for a given budget scope."""
        eff_orders = [to_decimal(o) for o in (orders or DEFAULT_RDP_ORDERS)]
        eff_delta = validate_delta(delta_global) if delta_global is not None else Decimal("1.0e-5")
        eff_cap = to_decimal(epsilon_cap) if epsilon_cap is not None else Decimal("4.0")

        with self._connection() as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT accountant_mode, epsilon, delta, rdp_cost_json
                FROM release_store
                WHERE budget_scope_id = ? AND status = 'COMMITTED'
                """,
                (budget_scope_id,),
            )
            rows = cursor.fetchall()

        total_releases = len(rows)
        resolved_mode = accountant_mode or (rows[0]["accountant_mode"] if rows else "rdp")

        if resolved_mode == "rdp":
            if len(rows) > 0:
                vecs = []
                for r in rows:
                    raw_rdp = r["rdp_cost_json"]
                    if not raw_rdp:
                        raise ValueError(
                            f"Corrupted release record in scope '{budget_scope_id}': committed release missing RDP cost data."
                        )
                    vecs.append(rdp_vector_from_json(raw_rdp))
                cum_rdp = compose_rdp_vectors(vecs, orders=eff_orders)
                best_alpha, rdp_eff = rdp_to_epsilon_delta(cum_rdp, delta_global=eff_delta)
                eff_eps = rdp_eff
            else:
                cum_rdp = {o: Decimal("0.0") for o in eff_orders}
                best_alpha = None
                eff_eps = Decimal("0.0")

            remaining_eps = max(Decimal("0.0"), eff_cap - eff_eps)
            return PrivacyView(
                budget_scope_id=budget_scope_id,
                release_count=total_releases,
                total_rdp_vector=cum_rdp,
                effective_epsilon=eff_eps,
                best_alpha=best_alpha,
                delta_global=eff_delta,
                epsilon_cap=eff_cap,
                remaining_epsilon=remaining_eps,
                accountant_mode="rdp",
                orders=eff_orders,
            )
        else:
            spent_eps = sum(_to_decimal(r["epsilon"]) for r in rows)
            spent_delta = sum(_to_decimal(r["delta"]) for r in rows)
            remaining_eps = max(Decimal("0.0"), eff_cap - spent_eps)
            return PrivacyView(
                budget_scope_id=budget_scope_id,
                release_count=total_releases,
                total_rdp_vector={},
                effective_epsilon=spent_eps,
                best_alpha=None,
                delta_global=spent_delta,
                epsilon_cap=eff_cap,
                remaining_epsilon=remaining_eps,
                accountant_mode="basic",
                orders=eff_orders,
            )

    def clear(self) -> None:
        """Wipe all state (test-only helper)."""
        with self._connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM privacy_ledger")
            cursor.execute("DELETE FROM release_store")
            cursor.execute("DELETE FROM privacy_reservations")
            conn.commit()
