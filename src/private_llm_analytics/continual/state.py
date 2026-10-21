"""Persistent Continual Observation State Manager (v0.4).

Manages:
- continual_streams: stream definitions and cryptographic fingerprints.
- continual_nodes: global canonical noisy node payloads (strictly NO exact aggregate column).
- continual_level_exposures: scope-specific multi-level atomic privacy reservations and commitments.
- continual_query_log: audit log of served interval and sliding-window queries.
- Integration with PrivacyStateManager for lifetime RDP composition.
"""

from contextlib import contextmanager
from datetime import datetime, timezone
from decimal import Decimal
import json
from pathlib import Path
import sqlite3
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple, Union
import uuid
import numpy as np

from private_llm_analytics.continual.mechanism import compute_level_rdp_vector, evaluate_rdp_epsilon
from private_llm_analytics.continual.models import ContinualStreamConfig
from private_llm_analytics.continual.tree import BinaryTimeTree, compute_continual_node_id
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
    compose_rdp_vectors,
    rdp_vector_from_json,
    rdp_vector_to_json,
    to_decimal,
)
from private_llm_analytics.privacy.state import PrivacyStateManager


class ContinualStateManager:
    """Authoritative persistent state manager for Continual Observation."""

    def __init__(
        self,
        db_path: Union[str, Path] = "results/privacy_state.db",
        state_manager: Optional[PrivacyStateManager] = None,
        timeout: float = 30.0,
    ):
        self.db_path = Path(db_path)
        self.timeout = float(timeout)
        self.state_manager = state_manager or PrivacyStateManager(db_path=self.db_path)
        self._init_continual_tables()

    @contextmanager
    def _connection(self):
        """Context manager yielding SQLite connection with WAL mode and foreign keys enabled."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(self.db_path), timeout=self.timeout, isolation_level=None)
        try:
            conn.execute("PRAGMA journal_mode = WAL")
            conn.execute("PRAGMA synchronous = NORMAL")
            conn.execute("PRAGMA foreign_keys = ON")
            yield conn
        finally:
            conn.close()

    def _init_continual_tables(self):
        """Initialize continual observation SQLite tables (Section 30)."""
        with self._connection() as conn:
            cursor = conn.cursor()
            cursor.execute("BEGIN IMMEDIATE")
            try:
                # 1. continual_streams
                cursor.execute(
                    """
                    CREATE TABLE IF NOT EXISTS continual_streams (
                        stream_fingerprint TEXT PRIMARY KEY,
                        stream_id TEXT NOT NULL,
                        metric TEXT NOT NULL,
                        granularity TEXT NOT NULL,
                        epoch_id TEXT NOT NULL,
                        epoch_start TEXT NOT NULL,
                        epoch_days INTEGER NOT NULL,
                        timezone TEXT NOT NULL,
                        daily_l2_bound REAL NOT NULL,
                        epoch_group_l12_bound REAL NOT NULL,
                        lateness_grace_days INTEGER NOT NULL,
                        tree_topology TEXT NOT NULL,
                        tree_padding TEXT NOT NULL,
                        noise_config_json TEXT NOT NULL,
                        dataset_fingerprint TEXT,
                        public_roster_fingerprint TEXT,
                        created_at TEXT NOT NULL
                    )
                    """
                )

                # 2. continual_nodes (Notice: STRICTLY NO exact_node_value column!)
                cursor.execute(
                    """
                    CREATE TABLE IF NOT EXISTS continual_nodes (
                        node_id TEXT PRIMARY KEY,
                        stream_fingerprint TEXT NOT NULL,
                        epoch_id TEXT NOT NULL,
                        level INTEGER NOT NULL,
                        node_index INTEGER NOT NULL,
                        interval_start INTEGER NOT NULL,
                        interval_end INTEGER NOT NULL,
                        noisy_value_json TEXT NOT NULL,
                        sigma REAL NOT NULL,
                        dimension INTEGER NOT NULL,
                        status TEXT NOT NULL,
                        owner_id TEXT NOT NULL DEFAULT '',
                        fencing_token TEXT NOT NULL DEFAULT '',
                        lease_expires_at REAL NOT NULL DEFAULT 0.0,
                        created_at TEXT NOT NULL,
                        FOREIGN KEY (stream_fingerprint) REFERENCES continual_streams (stream_fingerprint),
                        UNIQUE (stream_fingerprint, epoch_id, level, node_index)
                    )
                    """
                )

                # 3. continual_level_exposures
                cursor.execute(
                    """
                    CREATE TABLE IF NOT EXISTS continual_level_exposures (
                        exposure_id TEXT PRIMARY KEY,
                        budget_scope_id TEXT NOT NULL,
                        stream_fingerprint TEXT NOT NULL,
                        epoch_id TEXT NOT NULL,
                        level INTEGER NOT NULL,
                        rdp_cost_json TEXT NOT NULL,
                        reservation_state TEXT NOT NULL,
                        fencing_token TEXT NOT NULL,
                        created_at TEXT NOT NULL,
                        committed_at TEXT,
                        FOREIGN KEY (stream_fingerprint) REFERENCES continual_streams (stream_fingerprint),
                        UNIQUE (budget_scope_id, stream_fingerprint, epoch_id, level)
                    )
                    """
                )

                # 4. continual_query_log
                cursor.execute(
                    """
                    CREATE TABLE IF NOT EXISTS continual_query_log (
                        query_id TEXT PRIMARY KEY,
                        budget_scope_id TEXT NOT NULL,
                        stream_fingerprint TEXT NOT NULL,
                        epoch_id TEXT NOT NULL,
                        query_type TEXT NOT NULL,
                        start_day INTEGER NOT NULL,
                        end_day INTEGER NOT NULL,
                        nodes_used_json TEXT NOT NULL,
                        levels_used_json TEXT NOT NULL,
                        newly_exposed_levels_json TEXT NOT NULL,
                        created_at TEXT NOT NULL
                    )
                    """
                )
                cursor.execute("COMMIT")
            except Exception:
                cursor.execute("ROLLBACK")
                raise

    def register_stream(self, stream_config: ContinualStreamConfig):
        """Register continual stream definition if not already present."""
        fp = stream_config.stream_fingerprint
        with self._connection() as conn:
            cursor = conn.cursor()
            cursor.execute("BEGIN IMMEDIATE")
            try:
                cursor.execute(
                    "SELECT stream_fingerprint FROM continual_streams WHERE stream_fingerprint = ?",
                    (fp,),
                )
                if cursor.fetchone() is None:
                    noise_dict = {
                        "mechanism": stream_config.noise.mechanism,
                        "target_epoch_epsilon": stream_config.noise.target_epoch_epsilon,
                        "calibration_delta": stream_config.noise.calibration_delta,
                        "equal_sigma_per_level": stream_config.noise.equal_sigma_per_level,
                        "sigma": stream_config.noise.sigma,
                        "rdp_orders": list(stream_config.noise.rdp_orders),
                    }
                    cursor.execute(
                        """
                        INSERT INTO continual_streams (
                            stream_fingerprint, stream_id, metric, granularity,
                            epoch_id, epoch_start, epoch_days, timezone,
                            daily_l2_bound, epoch_group_l12_bound, lateness_grace_days,
                            tree_topology, tree_padding, noise_config_json,
                            dataset_fingerprint, public_roster_fingerprint, created_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            fp,
                            stream_config.stream_id,
                            stream_config.metric,
                            stream_config.granularity,
                            stream_config.epoch_id,
                            str(stream_config.epoch_start),
                            int(stream_config.epoch_days),
                            stream_config.timezone,
                            float(stream_config.daily_l2_bound),
                            float(stream_config.epoch_group_l12_bound),
                            int(stream_config.lateness_grace_days),
                            stream_config.tree.topology,
                            stream_config.tree.padding,
                            json.dumps(noise_dict, sort_keys=True),
                            stream_config.dataset_fingerprint,
                            stream_config.public_roster_fingerprint,
                            datetime.now(timezone.utc).isoformat(),
                        ),
                    )
                cursor.execute("COMMIT")
            except Exception:
                cursor.execute("ROLLBACK")
                raise

    def get_exposed_levels(
        self,
        budget_scope_id: str,
        stream_fingerprint: str,
        epoch_id: str,
    ) -> Set[int]:
        """Return set of level indices already committed for this budget scope."""
        return self.state_manager.get_exposed_continual_levels(
            budget_scope_id=budget_scope_id,
            stream_fingerprint=stream_fingerprint,
            epoch_id=epoch_id,
        )

    def reserve_continual_level_exposures(
        self,
        budget_scope_id: str,
        stream_config: ContinualStreamConfig,
        missing_levels: Sequence[int],
        sigma: float,
        delta_global: float = 1e-5,
        epsilon_cap: float = 4.0,
        authoritative_orders: Optional[Sequence[Any]] = None,
        policy_mode: Optional[str] = None,
        policy_groups: Optional[Dict[str, List[str]]] = None,
        recipient_roles: Optional[Dict[str, str]] = None,
        owner_id: Optional[str] = None,
        lease_timeout: float = 300.0,
        single_flight_timeout: float = 30.0,
    ) -> List[Dict[str, Any]]:
        """Atomically authorize and reserve missing tree level exposures."""
        if not missing_levels:
            return []

        eff_orders = authoritative_orders or stream_config.noise.rdp_orders
        level_rdp = compute_level_rdp_vector(
            epoch_l12_bound=stream_config.epoch_group_l12_bound,
            sigma=sigma,
            rdp_orders=eff_orders,
        )

        return self.state_manager.reserve_continual_level_exposures(
            budget_scope_id=budget_scope_id,
            stream_fingerprint=stream_config.stream_fingerprint,
            epoch_id=stream_config.epoch_id,
            missing_levels=missing_levels,
            level_rdp_cost=level_rdp,
            epsilon_cap=epsilon_cap,
            delta_global=delta_global,
            authoritative_orders=eff_orders,
            policy_mode=policy_mode,
            policy_groups=policy_groups,
            recipient_roles=recipient_roles,
            owner_id=owner_id,
            lease_timeout=lease_timeout,
            single_flight_timeout=single_flight_timeout,
        )

    def commit_continual_level_exposures(self, reservations: Sequence[Dict[str, Any]]):
        """Commit reserved level exposures atomically with CAS fencing."""
        self.state_manager.commit_continual_level_exposures(reservations)

    def abort_continual_level_exposures(self, reservations: Sequence[Dict[str, Any]]):
        """Abort reserved level exposures atomically, releasing uncommitted budget."""
        self.state_manager.abort_continual_level_exposures(reservations)

    def recover_stale_reservations(self, max_age_seconds: float = 300.0) -> int:
        """Abort lingering RESERVED exposures older than max_age_seconds."""
        return self.state_manager.recover_stale_continual_reservations(max_age_seconds=max_age_seconds)

    def get_or_create_noisy_node(
        self,
        stream_config: ContinualStreamConfig,
        level: int,
        node_index: int,
        sigma: float,
        materialize_fn: Callable[[], np.ndarray],
        lease_seconds: float = 30.0,
        finalized_up_to_day: Optional[int] = None,
        manifest_fingerprint: Optional[str] = None,
        public_roster_fingerprint: Optional[str] = None,
        current_wall_clock_day: Optional[int] = None,
        partition_fingerprints: Optional[Dict[str, str]] = None,
    ) -> np.ndarray:
        """Retrieve canonical noisy node payload or materialize with single-flight fencing (Section 23, 29, 33).

        Guarantees:
        - Exactly one noisy payload per (stream_fingerprint, epoch_id, level, node_index).
        - Bitwise identical vector returned to all callers / recipients.
        - STRICTLY NO exact aggregate persisted.
        - Enforces immutable epoch manifest and finalized-day policy.
        """
        fp = stream_config.stream_fingerprint
        epoch_id = stream_config.epoch_id

        # 0. Ensure stream definition is registered
        self.register_stream(stream_config)

        # 1. Verify or seal immutable epoch dataset manifest
        mf = manifest_fingerprint or stream_config.dataset_fingerprint
        rf = public_roster_fingerprint or stream_config.public_roster_fingerprint
        fin_day = finalized_up_to_day if finalized_up_to_day is not None else stream_config.epoch_days

        manifest_record = self.state_manager.register_or_verify_epoch_manifest(
            stream_fingerprint=fp,
            epoch_id=epoch_id,
            manifest_fingerprint=mf or "",
            public_roster_fingerprint=rf or "",
            finalized_up_to_day=int(fin_day),
            partition_fingerprints=partition_fingerprints,
            lateness_grace_days=stream_config.lateness_grace_days,
            current_wall_clock_day=current_wall_clock_day,
        )

        node_id = compute_continual_node_id(fp, epoch_id, level, node_index)
        tree = BinaryTimeTree(epoch_days=stream_config.epoch_days)
        start_day, end_day = tree.node_interval(level, node_index)

        # 2. Finalized day guard
        if end_day > int(manifest_record["finalized_up_to_day"]):
            raise ProductionSafetyConfigurationError(
                f"Cannot materialize unfinalized node [{start_day}, {end_day}]: "
                f"epoch finalized_up_to_day={manifest_record['finalized_up_to_day']}."
            )

        worker_owner_id = f"worker_{uuid.uuid4().hex[:8]}"

        while True:
            with self._connection() as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                cursor.execute("BEGIN IMMEDIATE")
                try:
                    cursor.execute(
                        """
                        SELECT status, noisy_value_json, owner_id, lease_expires_at, fencing_token, manifest_fingerprint
                        FROM continual_nodes
                        WHERE node_id = ?
                        """,
                        (node_id,),
                    )
                    row = cursor.fetchone()
                    now_ts = time.time()

                    if row is not None:
                        status = row["status"]
                        noisy_json = row["noisy_value_json"]
                        lease_exp = float(row["lease_expires_at"] or 0.0)
                        fence = row["fencing_token"]
                        persisted_mf = row["manifest_fingerprint"]

                        if status == "COMMITTED":
                            cursor.execute("COMMIT")
                            if persisted_mf and persisted_mf != manifest_record["manifest_fingerprint"]:
                                raise ProductionSafetyConfigurationError(
                                    f"DATASET_MANIFEST_MISMATCH: node {node_id} was materialized with manifest "
                                    f"{persisted_mf}, but current request has sealed manifest {manifest_record['manifest_fingerprint']}."
                                )
                            val_list = json.loads(noisy_json)
                            return np.array(val_list, dtype=np.float64)

                        # If RESERVED by someone else, check lease
                        if now_ts < lease_exp:
                            cursor.execute("COMMIT")
                            # Another worker is materializing: wait briefly and retry
                            time.sleep(0.05)
                            continue
                        else:
                            # Lease expired: steal ownership
                            new_fence = f"fence_{uuid.uuid4().hex}"
                            cursor.execute(
                                """
                                UPDATE continual_nodes
                                SET owner_id = ?, lease_expires_at = ?, fencing_token = ?, manifest_fingerprint = ?
                                WHERE node_id = ? AND fencing_token = ?
                                """,
                                (worker_owner_id, now_ts + lease_seconds, new_fence, manifest_record["manifest_fingerprint"], node_id, fence),
                            )
                            cursor.execute("COMMIT")
                            fencing_token = new_fence
                            break
                    else:
                        # Insert reservation
                        new_fence = f"fence_{uuid.uuid4().hex}"
                        now_iso = datetime.now(timezone.utc).isoformat()
                        cursor.execute(
                            """
                            INSERT INTO continual_nodes (
                                node_id, stream_fingerprint, epoch_id, level, node_index,
                                interval_start, interval_end, noisy_value_json, sigma,
                                dimension, status, owner_id, fencing_token, lease_expires_at,
                                created_at, manifest_fingerprint
                            ) VALUES (?, ?, ?, ?, ?, ?, ?, '', ?, 0, 'RESERVED', ?, ?, ?, ?, ?)
                            """,
                            (
                                node_id, fp, epoch_id, level, node_index,
                                start_day, end_day, float(sigma),
                                worker_owner_id, new_fence, now_ts + lease_seconds, now_iso,
                                manifest_record["manifest_fingerprint"],
                            ),
                        )
                        cursor.execute("COMMIT")
                        fencing_token = new_fence
                        break
                except sqlite3.IntegrityError:
                    cursor.execute("ROLLBACK")
                    time.sleep(0.02)
                    continue
                except Exception:
                    cursor.execute("ROLLBACK")
                    raise

        # Now worker owns materialization
        # Compute exact aggregate (trusted curator step)
        exact_vector = materialize_fn()
        dim = len(exact_vector)

        # Add Gaussian noise
        noise = np.random.normal(loc=0.0, scale=sigma, size=exact_vector.shape)
        noisy_vector = exact_vector + noise

        # Store ONLY noisy_vector (Section 29)
        noisy_json = json.dumps([float(x) for x in noisy_vector])

        with self._connection() as conn:
            cursor = conn.cursor()
            cursor.execute("BEGIN IMMEDIATE")
            try:
                cursor.execute(
                    """
                    SELECT fencing_token, status FROM continual_nodes
                    WHERE node_id = ?
                    """,
                    (node_id,),
                )
                row = cursor.fetchone()
                if row is None or row[0] != fencing_token:
                    cursor.execute("ROLLBACK")
                    raise ReservationLostError(f"Node materialization ownership lost for node {node_id}")

                cursor.execute(
                    """
                    UPDATE continual_nodes
                    SET noisy_value_json = ?, dimension = ?, status = 'COMMITTED',
                        lease_expires_at = 0.0, manifest_fingerprint = ?
                    WHERE node_id = ?
                    """,
                    (noisy_json, dim, manifest_record["manifest_fingerprint"], node_id),
                )
                cursor.execute("COMMIT")
                return noisy_vector
            except Exception:
                cursor.execute("ROLLBACK")
                raise

    def log_query(
        self,
        budget_scope_id: str,
        stream_fingerprint: str,
        epoch_id: str,
        query_type: str,
        start_day: int,
        end_day: int,
        nodes_used: List[Tuple[int, int]],
        levels_used: List[int],
        newly_exposed_levels: List[int],
    ) -> str:
        """Write audit entry for served continual observation query."""
        query_id = f"qry_{uuid.uuid4().hex}"
        now_iso = datetime.now(timezone.utc).isoformat()
        with self._connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO continual_query_log (
                    query_id, budget_scope_id, stream_fingerprint, epoch_id,
                    query_type, start_day, end_day, nodes_used_json,
                    levels_used_json, newly_exposed_levels_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    query_id,
                    budget_scope_id,
                    stream_fingerprint,
                    epoch_id,
                    query_type,
                    start_day,
                    end_day,
                    json.dumps(nodes_used),
                    json.dumps(levels_used),
                    json.dumps(newly_exposed_levels),
                    now_iso,
                ),
            )
        return query_id
