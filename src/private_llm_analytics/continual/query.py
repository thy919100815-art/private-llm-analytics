"""Continual Observation Query Engine with Deterministic Post-Processing (v0.4).

Implements:
- query_prefix: Prefix [1, end_day] usage tracking.
- query_interval: Fixed interval [start_day, end_day] analysis.
- query_sliding_window: 7-day, 30-day, 90-day sliding-window queries.
- query_cross_epoch: Cross-epoch interval decomposition and linear combination.
- Strictly deterministic query post-processing with zero RNG at query time (Section 45).
- Replay and multi-recipient level exposure reuse (Section 22, 23, 49-52).
- Finalized data barrier enforcement (Section 36, 101).
- Direct cached query execution without scanning raw telemetry (Section 99).
"""

from dataclasses import dataclass, field
from decimal import Decimal
import hashlib
import json
import math
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple, Union
import numpy as np
import pandas as pd

from private_llm_analytics.continual.contributions import (
    build_temporal_user_contributions,
    compute_partition_fingerprints,
)
from private_llm_analytics.continual.mechanism import (
    calibrate_tree_gaussian_sigma,
    compute_level_rdp_vector,
    evaluate_rdp_epsilon,
)
from private_llm_analytics.continual.models import ContinualStreamConfig
from private_llm_analytics.continual.state import ContinualStateManager
from private_llm_analytics.continual.tree import BinaryTimeTree, compute_continual_node_id
from private_llm_analytics.planning.policy import build_aggregation_matrix
from private_llm_analytics.privacy.composition import PrivacyView, resolve_budget_scope_id
from private_llm_analytics.privacy.exceptions import (
    PrivacyBudgetExceededError,
    PrivacyConfigurationError,
    ProductionSafetyConfigurationError,
)
from private_llm_analytics.privacy.rdp import DEFAULT_RDP_ORDERS


@dataclass(frozen=True)
class ContinualQueryResult:
    """Audit-rich DP result for continual observation queries (Section 75)."""

    vector: np.ndarray
    cell_names: List[str]
    nodes_used: List[Tuple[int, int]]
    levels_used: List[int]
    newly_exposed_levels: List[int]
    reused_levels: List[int]
    budget_scope_id: str
    expected_query_noise_std: float
    query_id: str
    stream_fingerprint: str
    epoch_id: str
    start_day: int
    end_day: int
    query_type: str
    production_safe: bool = True

    @property
    def total_count(self) -> float:
        return float(np.sum(self.vector))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "vector": [float(x) for x in self.vector],
            "cell_names": list(self.cell_names),
            "nodes_used": list(self.nodes_used),
            "levels_used": list(self.levels_used),
            "newly_exposed_levels": list(self.newly_exposed_levels),
            "reused_levels": list(self.reused_levels),
            "budget_scope_id": self.budget_scope_id,
            "expected_query_noise_std": float(self.expected_query_noise_std),
            "query_id": self.query_id,
            "stream_fingerprint": self.stream_fingerprint,
            "epoch_id": self.epoch_id,
            "start_day": int(self.start_day),
            "end_day": int(self.end_day),
            "query_type": self.query_type,
            "production_safe": bool(self.production_safe),
        }


class ContinualQueryEngine:
    """Authoritative continual query engine executing post-processed tree releases."""

    def __init__(
        self,
        state_manager: ContinualStateManager,
        composition_mode: str = "global_external",
        collusion_groups: Optional[Dict[str, List[str]]] = None,
        recipient_roles: Optional[Dict[str, str]] = None,
        delta_global: float = 1e-6,
        epsilon_cap: float = 4.0,
        authoritative_orders: Optional[Sequence[float]] = None,
        app_config: Optional[Any] = None,
    ):
        self.state_manager = state_manager
        if app_config is not None:
            acc_mode = str(getattr(app_config.privacy.accountant, "mode", "")).lower()
            if acc_mode != "rdp":
                raise PrivacyConfigurationError(
                    f"Fail-closed: Continual observation strictly requires accountant mode 'rdp', got '{acc_mode}'."
                )
            self.enabled = bool(getattr(app_config.privacy.continual, "enabled", False))
            self.delta_global = float(app_config.privacy.accountant.delta_global)
            self.epsilon_cap = float(app_config.privacy.accountant.epsilon_cap)
            self.authoritative_orders = tuple(float(x) for x in app_config.privacy.accountant.orders)
            self.composition_mode = str(app_config.privacy.recipient_composition.mode)
            rc_cfg = app_config.privacy.recipient_composition
            self.collusion_groups = getattr(rc_cfg, "groups", None) or getattr(rc_cfg, "collusion_groups", {}) or {}
            roles_from_recipients: Dict[str, str] = {}
            if hasattr(app_config, "recipients") and hasattr(app_config.recipients, "recipients"):
                roles_from_recipients = {
                    name: getattr(r_cfg, "role", "external")
                    for name, r_cfg in app_config.recipients.recipients.items()
                }
            roles_from_rc = getattr(rc_cfg, "recipient_roles", {}) or {}
            self.recipient_roles = {**roles_from_rc, **roles_from_recipients}
        else:
            self.enabled = True
            self.composition_mode = composition_mode
            self.collusion_groups = collusion_groups or {}
            self.recipient_roles = recipient_roles or {}
            self.delta_global = float(delta_global)
            self.epsilon_cap = float(epsilon_cap)
            self.authoritative_orders = tuple(float(x) for x in (authoritative_orders or DEFAULT_RDP_ORDERS))

    @classmethod
    def from_app_config(
        cls,
        app_config: Any,
        state_manager: Optional[ContinualStateManager] = None,
    ) -> "ContinualQueryEngine":
        """Instantiate ContinualQueryEngine bound to authoritative AppConfig."""
        sm = state_manager or ContinualStateManager(db_path=app_config.privacy.release.state_db_path)
        return cls(state_manager=sm, app_config=app_config)

    def resolve_scope(self, recipient: str, recipient_role: Optional[str] = None) -> str:
        """Resolve budget scope ID according to active recipient composition policy."""
        configured_role = self.recipient_roles.get(recipient)
        if configured_role is not None:
            if recipient_role is not None and recipient_role != configured_role:
                raise ProductionSafetyConfigurationError(
                    "Recipient role override conflicts with authoritative configuration."
                )
            effective_role = configured_role
        else:
            effective_role = recipient_role or "external"

        return resolve_budget_scope_id(
            recipient=recipient,
            recipient_role=effective_role,
            mode=self.composition_mode,
            groups=self.collusion_groups,
        )

    def get_privacy_view(self, recipient: str = "global_external", recipient_role: Optional[str] = None) -> PrivacyView:
        """Authoritative reconstructed PrivacyView for recipient or budget scope."""
        scope = self.resolve_scope(recipient, recipient_role=recipient_role)
        return self.state_manager.get_privacy_view(
            budget_scope_id=scope,
            delta_global=self.delta_global,
            orders=self.authoritative_orders,
            epsilon_cap=self.epsilon_cap,
            accountant_mode="rdp",
        )

    def _get_calibrated_sigma(self, stream_config: ContinualStreamConfig) -> float:
        """Return configured or calibrated tree Gaussian noise sigma."""
        if stream_config.noise.sigma is not None:
            return float(stream_config.noise.sigma)
        return calibrate_tree_gaussian_sigma(
            epoch_days=stream_config.epoch_days,
            epoch_l12_bound=stream_config.epoch_group_l12_bound,
            target_tree_epsilon=stream_config.noise.target_epoch_epsilon,
            calibration_delta=stream_config.noise.calibration_delta,
            rdp_orders=self.authoritative_orders or stream_config.noise.rdp_orders,
        )

    def _execute_tree_query(
        self,
        stream_config: ContinualStreamConfig,
        start_day: int,
        end_day: int,
        recipient: str,
        query_type: str,
        interactions: Optional[Union[pd.DataFrame, Sequence[Dict[str, Any]]]] = None,
        public_user_roster: Optional[Sequence[str]] = None,
        recipient_role: Optional[str] = None,
        finalized_up_to_day: Optional[int] = None,
        precomputed_contributions: Optional[np.ndarray] = None,
        manifest_fingerprint: Optional[str] = None,
        public_roster_fingerprint: Optional[str] = None,
        current_wall_clock_day: Optional[int] = None,
        partition_fingerprints: Optional[Dict[str, str]] = None,
    ) -> ContinualQueryResult:
        """Internal execution of tree-based dyadic interval query."""
        if not self.enabled:
            raise PrivacyConfigurationError(
                "Continual observation is disabled in configuration (privacy.continual.enabled=false). "
                "Explicitly enable privacy.continual.enabled to authorize production continual queries."
            )

        # 1. Register stream configuration
        self.state_manager.register_stream(stream_config)

        # 2. Canonical cover
        tree = BinaryTimeTree(epoch_days=stream_config.epoch_days)
        nodes_used = tree.canonical_cover(start_day, end_day)
        levels_used = sorted(list({lvl for lvl, _ in nodes_used}))

        # 3. Finalized day guard (Section 36, 101)
        all_nodes_committed = True
        with self.state_manager._connection() as conn:
            cursor = conn.cursor()
            for lvl, n_idx in nodes_used:
                nid = compute_continual_node_id(stream_config.stream_fingerprint, stream_config.epoch_id, lvl, n_idx)
                cursor.execute("SELECT 1 FROM continual_nodes WHERE node_id = ? AND status = 'COMMITTED'", (nid,))
                if cursor.fetchone() is None:
                    all_nodes_committed = False
                    break

        if not all_nodes_committed:
            with self.state_manager._connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT finalized_up_to_day FROM continual_epoch_manifests WHERE stream_fingerprint = ? AND epoch_id = ?",
                    (stream_config.stream_fingerprint, stream_config.epoch_id),
                )
                m_row = cursor.fetchone()
                persisted_fin = m_row[0] if m_row is not None else None

            if finalized_up_to_day is None:
                if persisted_fin is not None:
                    effective_fin = persisted_fin
                else:
                    raise ProductionSafetyConfigurationError(
                        "Fail-closed: finalized_up_to_day REQUIRED in production: "
                        "no authoritative finalized boundary exists to materialize new continual tree nodes."
                    )
            else:
                effective_fin = finalized_up_to_day

            if end_day > effective_fin:
                raise ProductionSafetyConfigurationError(
                    f"DATA_NOT_FINALIZED: Requested interval [{start_day}, {end_day}] "
                    f"exceeds finalized day {effective_fin}."
                )
        else:
            effective_fin = finalized_up_to_day

        # 4. Validate interval
        if not (1 <= start_day <= end_day <= stream_config.epoch_days):
            raise ValueError(
                f"Requested interval [{start_day}, {end_day}] out of epoch bounds [1, {stream_config.epoch_days}]"
            )

        # 5. Determine sigma
        sigma = self._get_calibrated_sigma(stream_config)

        # 5b. Resolve dataset manifest and public roster fingerprints
        mf = manifest_fingerprint or stream_config.dataset_fingerprint
        rf = public_roster_fingerprint or stream_config.public_roster_fingerprint
        if not mf or not rf:
            with self.state_manager._connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT manifest_fingerprint, public_roster_fingerprint FROM continual_epoch_manifests WHERE stream_fingerprint = ? AND epoch_id = ?",
                    (stream_config.stream_fingerprint, stream_config.epoch_id),
                )
                m_row = cursor.fetchone()
                if m_row:
                    mf = mf or m_row[0]
                    rf = rf or m_row[1]

        if not mf:
            # Derive stable epoch manifest identity from immutable stream metadata
            mf = f"epoch_manifest:{stream_config.stream_fingerprint}:{stream_config.epoch_id}"

        if not rf and public_user_roster is not None:
            roster_bytes = json.dumps(sorted(list(public_user_roster))).encode("utf-8")
            rf = hashlib.sha256(roster_bytes).hexdigest()

        # 5c. Bind partition fingerprints from interactions if available
        if partition_fingerprints is None and interactions is not None:
            partition_fingerprints = compute_partition_fingerprints(
                interactions=interactions,
                epoch_start=stream_config.epoch_start,
                start_day=1,
                end_day=effective_fin,
                timezone_name=stream_config.timezone,
            )
        elif partition_fingerprints is not None:
            partition_fingerprints = {
                str(k): str(v)
                for k, v in partition_fingerprints.items()
                if int(k) <= effective_fin
            }

        # 5d. Verify dataset manifest and partition fingerprints against sealed manifest in DB
        with self.state_manager._connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT manifest_fingerprint, public_roster_fingerprint FROM continual_epoch_manifests WHERE stream_fingerprint = ? AND epoch_id = ?",
                (stream_config.stream_fingerprint, stream_config.epoch_id),
            )
            m_row = cursor.fetchone()
            if m_row is not None:
                persisted_mf, persisted_rf = m_row[0], m_row[1]
                if mf and persisted_mf and persisted_mf != mf:
                    raise ProductionSafetyConfigurationError(
                        f"DATASET_MANIFEST_MISMATCH: stream_fingerprint={stream_config.stream_fingerprint}, epoch_id={stream_config.epoch_id} "
                        f"has sealed manifest {persisted_mf}, but current request provided {mf}. Modifying epoch manifest is strictly forbidden."
                    )
                if rf and persisted_rf and persisted_rf != rf:
                    raise ProductionSafetyConfigurationError(
                        f"DATASET_MANIFEST_MISMATCH: public roster fingerprint mismatch for stream={stream_config.stream_fingerprint}, epoch={stream_config.epoch_id} "
                        f"(sealed={persisted_rf}, provided={rf})."
                    )
            if partition_fingerprints:
                for day_str, pfp in partition_fingerprints.items():
                    cursor.execute(
                        "SELECT partition_fingerprint FROM continual_epoch_partitions WHERE stream_fingerprint = ? AND epoch_id = ? AND day = ?",
                        (stream_config.stream_fingerprint, stream_config.epoch_id, int(day_str)),
                    )
                    p_row = cursor.fetchone()
                    if p_row is not None and p_row[0] != pfp:
                        raise ProductionSafetyConfigurationError(
                            f"DATASET_MANIFEST_MISMATCH: already finalized partition for day {day_str} cannot be mutated "
                            f"(sealed={p_row[0]}, provided={pfp})."
                        )

        # 6. Resolve scope
        budget_scope_id = self.resolve_scope(recipient, recipient_role=recipient_role)

        # 7. Check level exposures
        fp = stream_config.stream_fingerprint
        epoch_id = stream_config.epoch_id
        exposed_levels = self.state_manager.get_exposed_levels(budget_scope_id, fp, epoch_id)
        newly_exposed_levels = [lvl for lvl in levels_used if lvl not in exposed_levels]
        reused_levels = [lvl for lvl in levels_used if lvl in exposed_levels]

        # 8. Atomic multi-level authorization & reservation (Section 31, 32)
        reservations = []
        if newly_exposed_levels:
            reservations = self.state_manager.reserve_continual_level_exposures(
                budget_scope_id=budget_scope_id,
                stream_config=stream_config,
                missing_levels=newly_exposed_levels,
                sigma=sigma,
                delta_global=self.delta_global,
                epsilon_cap=self.epsilon_cap,
                authoritative_orders=self.authoritative_orders,
                policy_mode=self.composition_mode,
                policy_groups=self.collusion_groups,
                recipient_roles=self.recipient_roles,
            )

        # 9. Materialize / Retrieve Canonical Noisy Nodes with rollback protection
        contrib_tensor_holder: List[Optional[np.ndarray]] = [precomputed_contributions]

        def _get_contributions():
            if contrib_tensor_holder[0] is None:
                if interactions is None or public_user_roster is None:
                    raise ValueError(
                        "Cannot materialize un-cached tree nodes: raw interactions and public_user_roster required."
                    )
                tensor, _, _ = build_temporal_user_contributions(
                    interactions=interactions,
                    public_user_roster=public_user_roster,
                    epoch_start=stream_config.epoch_start,
                    epoch_days=stream_config.epoch_days,
                    granularity=stream_config.granularity,
                    daily_l2_bound=stream_config.daily_l2_bound,
                    epoch_l12_bound=stream_config.epoch_group_l12_bound,
                    metric=stream_config.metric,
                    timezone_name=stream_config.timezone,
                )
                contrib_tensor_holder[0] = tensor
            return contrib_tensor_holder[0]

        try:
            node_noisy_vectors: List[np.ndarray] = []
            for level, node_idx in nodes_used:
                u, v = tree.node_interval(level, node_idx)
                def _materialize_node_exact():
                    contribs = _get_contributions()
                    start_idx = u - 1
                    end_idx = min(v, stream_config.epoch_days)
                    node_slice = contribs[:, start_idx:end_idx, :]
                    exact_node = np.sum(node_slice, axis=(0, 1))
                    return exact_node

                noisy_node = self.state_manager.get_or_create_noisy_node(
                    stream_config=stream_config,
                    level=level,
                    node_index=node_idx,
                    sigma=sigma,
                    materialize_fn=_materialize_node_exact,
                    finalized_up_to_day=effective_fin,
                    manifest_fingerprint=mf,
                    public_roster_fingerprint=rf,
                    current_wall_clock_day=current_wall_clock_day,
                    partition_fingerprints=partition_fingerprints,
                )
                node_noisy_vectors.append(noisy_node)

            # 10. Commit level exposures
            if reservations:
                self.state_manager.commit_continual_level_exposures(reservations)
        except Exception:
            if reservations:
                self.state_manager.abort_continual_level_exposures(reservations)
            raise

        # 11. Deterministic post-processing summation (Section 22, 45 - NO RNG AT QUERY TIME)
        result_vector = np.sum(node_noisy_vectors, axis=0)
        expected_noise_std = float(sigma * math.sqrt(len(nodes_used)))

        # 12. Audit log
        query_id = self.state_manager.log_query(
            budget_scope_id=budget_scope_id,
            stream_fingerprint=fp,
            epoch_id=epoch_id,
            query_type=query_type,
            start_day=start_day,
            end_day=end_day,
            nodes_used=nodes_used,
            levels_used=levels_used,
            newly_exposed_levels=newly_exposed_levels,
        )

        _, cell_names = build_aggregation_matrix(stream_config.granularity)

        return ContinualQueryResult(
            vector=result_vector,
            cell_names=cell_names,
            nodes_used=nodes_used,
            levels_used=levels_used,
            newly_exposed_levels=newly_exposed_levels,
            reused_levels=reused_levels,
            budget_scope_id=budget_scope_id,
            expected_query_noise_std=expected_noise_std,
            query_id=query_id,
            stream_fingerprint=fp,
            epoch_id=epoch_id,
            start_day=start_day,
            end_day=end_day,
            query_type=query_type,
            production_safe=True,
        )

    def query_interval(
        self,
        stream_config: ContinualStreamConfig,
        start_day: int,
        end_day: int,
        recipient: str,
        interactions: Optional[Union[pd.DataFrame, Sequence[Dict[str, Any]]]] = None,
        public_user_roster: Optional[Sequence[str]] = None,
        recipient_role: Optional[str] = None,
        finalized_up_to_day: Optional[int] = None,
        precomputed_contributions: Optional[np.ndarray] = None,
        manifest_fingerprint: Optional[str] = None,
        public_roster_fingerprint: Optional[str] = None,
        current_wall_clock_day: Optional[int] = None,
        partition_fingerprints: Optional[Dict[str, str]] = None,
    ) -> ContinualQueryResult:
        """Fixed interval dyadic continual query [start_day, end_day] (Section 20)."""
        return self._execute_tree_query(
            stream_config=stream_config,
            start_day=start_day,
            end_day=end_day,
            recipient=recipient,
            query_type="INTERVAL",
            interactions=interactions,
            public_user_roster=public_user_roster,
            recipient_role=recipient_role,
            finalized_up_to_day=finalized_up_to_day,
            precomputed_contributions=precomputed_contributions,
            manifest_fingerprint=manifest_fingerprint,
            public_roster_fingerprint=public_roster_fingerprint,
            current_wall_clock_day=current_wall_clock_day,
            partition_fingerprints=partition_fingerprints,
        )

    def query_prefix(
        self,
        stream_config: ContinualStreamConfig,
        end_day: int,
        recipient: str,
        interactions: Optional[Union[pd.DataFrame, Sequence[Dict[str, Any]]]] = None,
        public_user_roster: Optional[Sequence[str]] = None,
        recipient_role: Optional[str] = None,
        finalized_up_to_day: Optional[int] = None,
        precomputed_contributions: Optional[np.ndarray] = None,
        manifest_fingerprint: Optional[str] = None,
        public_roster_fingerprint: Optional[str] = None,
        current_wall_clock_day: Optional[int] = None,
        partition_fingerprints: Optional[Dict[str, str]] = None,
    ) -> ContinualQueryResult:
        """Prefix continual query [1, end_day] (Section 19)."""
        return self._execute_tree_query(
            stream_config=stream_config,
            start_day=1,
            end_day=end_day,
            recipient=recipient,
            query_type="PREFIX",
            interactions=interactions,
            public_user_roster=public_user_roster,
            recipient_role=recipient_role,
            finalized_up_to_day=finalized_up_to_day,
            precomputed_contributions=precomputed_contributions,
            manifest_fingerprint=manifest_fingerprint,
            public_roster_fingerprint=public_roster_fingerprint,
            current_wall_clock_day=current_wall_clock_day,
            partition_fingerprints=partition_fingerprints,
        )

    def query_sliding_window(
        self,
        stream_config: ContinualStreamConfig,
        end_day: int,
        window_days: int,
        recipient: str,
        interactions: Optional[Union[pd.DataFrame, Sequence[Dict[str, Any]]]] = None,
        public_user_roster: Optional[Sequence[str]] = None,
        recipient_role: Optional[str] = None,
        finalized_up_to_day: Optional[int] = None,
        precomputed_contributions: Optional[np.ndarray] = None,
        manifest_fingerprint: Optional[str] = None,
        public_roster_fingerprint: Optional[str] = None,
        current_wall_clock_day: Optional[int] = None,
        partition_fingerprints: Optional[Dict[str, str]] = None,
    ) -> ContinualQueryResult:
        """Sliding-window continual query ending at end_day (Section 21)."""
        if window_days <= 0:
            raise ValueError(f"window_days must be positive, got {window_days}")
        start_day = max(1, end_day - window_days + 1)
        return self._execute_tree_query(
            stream_config=stream_config,
            start_day=start_day,
            end_day=end_day,
            recipient=recipient,
            query_type="SLIDING_WINDOW",
            interactions=interactions,
            public_user_roster=public_user_roster,
            recipient_role=recipient_role,
            finalized_up_to_day=finalized_up_to_day,
            precomputed_contributions=precomputed_contributions,
            manifest_fingerprint=manifest_fingerprint,
            public_roster_fingerprint=public_roster_fingerprint,
            current_wall_clock_day=current_wall_clock_day,
            partition_fingerprints=partition_fingerprints,
        )

    def query_cross_epoch(
        self,
        epoch_1_config: ContinualStreamConfig,
        epoch_1_start_day: int,
        epoch_2_config: ContinualStreamConfig,
        epoch_2_end_day: int,
        recipient: str,
        interactions_epoch_1: Optional[Union[pd.DataFrame, Sequence[Dict[str, Any]]]] = None,
        interactions_epoch_2: Optional[Union[pd.DataFrame, Sequence[Dict[str, Any]]]] = None,
        public_user_roster: Optional[Sequence[str]] = None,
        recipient_role: str = "external",
        finalized_up_to_day_epoch_1: Optional[int] = None,
        finalized_up_to_day_epoch_2: Optional[int] = None,
    ) -> ContinualQueryResult:
        """Cross-epoch interval query decomposing across epoch boundaries with atomic reservation (Section 103)."""
        if not self.enabled:
            raise PrivacyConfigurationError(
                "Continual observation is disabled in configuration (privacy.continual.enabled=false)."
            )

        self.state_manager.register_stream(epoch_1_config)
        self.state_manager.register_stream(epoch_2_config)

        # 1. Decompose both intervals
        tree_1 = BinaryTimeTree(epoch_days=epoch_1_config.epoch_days)
        nodes_used_1 = tree_1.canonical_cover(epoch_1_start_day, epoch_1_config.epoch_days)
        levels_used_1 = sorted(list({lvl for lvl, _ in nodes_used_1}))

        tree_2 = BinaryTimeTree(epoch_days=epoch_2_config.epoch_days)
        nodes_used_2 = tree_2.canonical_cover(1, epoch_2_end_day)
        levels_used_2 = sorted(list({lvl for lvl, _ in nodes_used_2}))

        budget_scope_id = self.resolve_scope(recipient, recipient_role=recipient_role)

        # 2. Check exposed levels
        exposed_1 = self.state_manager.get_exposed_levels(budget_scope_id, epoch_1_config.stream_fingerprint, epoch_1_config.epoch_id)
        newly_1 = [lvl for lvl in levels_used_1 if lvl not in exposed_1]
        reused_1 = [lvl for lvl in levels_used_1 if lvl in exposed_1]

        exposed_2 = self.state_manager.get_exposed_levels(budget_scope_id, epoch_2_config.stream_fingerprint, epoch_2_config.epoch_id)
        newly_2 = [lvl for lvl in levels_used_2 if lvl not in exposed_2]
        reused_2 = [lvl for lvl in levels_used_2 if lvl in exposed_2]

        sigma_1 = self._get_calibrated_sigma(epoch_1_config)
        sigma_2 = self._get_calibrated_sigma(epoch_2_config)

        rdp_1 = compute_level_rdp_vector(
            epoch_l12_bound=epoch_1_config.epoch_group_l12_bound,
            sigma=sigma_1,
            rdp_orders=self.authoritative_orders or epoch_1_config.noise.rdp_orders,
        )
        rdp_2 = compute_level_rdp_vector(
            epoch_l12_bound=epoch_2_config.epoch_group_l12_bound,
            sigma=sigma_2,
            rdp_orders=self.authoritative_orders or epoch_2_config.noise.rdp_orders,
        )

        epoch_requests = []
        if newly_1:
            epoch_requests.append((epoch_1_config.stream_fingerprint, epoch_1_config.epoch_id, newly_1, rdp_1))
        if newly_2:
            epoch_requests.append((epoch_2_config.stream_fingerprint, epoch_2_config.epoch_id, newly_2, rdp_2))

        # 3. ATOMIC CROSS-EPOCH RESERVATION: all-or-nothing
        all_reservations: List[Dict[str, Any]] = []
        if epoch_requests:
            all_reservations = self.state_manager.state_manager.reserve_cross_epoch_continual_level_exposures(
                budget_scope_id=budget_scope_id,
                epoch_requests=epoch_requests,
                epsilon_cap=self.epsilon_cap,
                delta_global=self.delta_global,
                authoritative_orders=self.authoritative_orders,
                policy_mode=self.composition_mode,
                policy_groups=self.collusion_groups,
                recipient_roles=self.recipient_roles,
            )

        mf_1 = epoch_1_config.dataset_fingerprint
        if not mf_1 and interactions_epoch_1 is not None:
            if isinstance(interactions_epoch_1, pd.DataFrame):
                data_bytes = str(interactions_epoch_1.to_dict(orient="records")).encode("utf-8")
            else:
                data_bytes = json.dumps(interactions_epoch_1, sort_keys=True).encode("utf-8")
            mf_1 = hashlib.sha256(data_bytes).hexdigest()

        rf_1 = epoch_1_config.public_roster_fingerprint
        if not rf_1 and public_user_roster is not None:
            roster_bytes = json.dumps(sorted(list(public_user_roster))).encode("utf-8")
            rf_1 = hashlib.sha256(roster_bytes).hexdigest()

        mf_2 = epoch_2_config.dataset_fingerprint
        if not mf_2 and interactions_epoch_2 is not None:
            if isinstance(interactions_epoch_2, pd.DataFrame):
                data_bytes = str(interactions_epoch_2.to_dict(orient="records")).encode("utf-8")
            else:
                data_bytes = json.dumps(interactions_epoch_2, sort_keys=True).encode("utf-8")
            mf_2 = hashlib.sha256(data_bytes).hexdigest()

        rf_2 = epoch_2_config.public_roster_fingerprint
        if not rf_2 and public_user_roster is not None:
            roster_bytes = json.dumps(sorted(list(public_user_roster))).encode("utf-8")
            rf_2 = hashlib.sha256(roster_bytes).hexdigest()

        # 4. Materialize nodes with rollback on error
        try:
            def _contribs_1():
                if interactions_epoch_1 is None or public_user_roster is None:
                    raise ValueError("Cannot materialize epoch 1 nodes: interactions and roster required.")
                tensor, _, _ = build_temporal_user_contributions(
                    interactions=interactions_epoch_1,
                    public_user_roster=public_user_roster,
                    epoch_start=epoch_1_config.epoch_start,
                    epoch_days=epoch_1_config.epoch_days,
                    granularity=epoch_1_config.granularity,
                    daily_l2_bound=epoch_1_config.daily_l2_bound,
                    epoch_l12_bound=epoch_1_config.epoch_group_l12_bound,
                    metric=epoch_1_config.metric,
                    timezone_name=epoch_1_config.timezone,
                )
                return tensor

            def _contribs_2():
                if interactions_epoch_2 is None or public_user_roster is None:
                    raise ValueError("Cannot materialize epoch 2 nodes: interactions and roster required.")
                tensor, _, _ = build_temporal_user_contributions(
                    interactions=interactions_epoch_2,
                    public_user_roster=public_user_roster,
                    epoch_start=epoch_2_config.epoch_start,
                    epoch_days=epoch_2_config.epoch_days,
                    granularity=epoch_2_config.granularity,
                    daily_l2_bound=epoch_2_config.daily_l2_bound,
                    epoch_l12_bound=epoch_2_config.epoch_group_l12_bound,
                    metric=epoch_2_config.metric,
                    timezone_name=epoch_2_config.timezone,
                )
                return tensor

            tensor_1_holder: List[Optional[np.ndarray]] = [None]
            tensor_2_holder: List[Optional[np.ndarray]] = [None]

            node_vectors_1 = []
            for lvl, n_idx in nodes_used_1:
                u, v = tree_1.node_interval(lvl, n_idx)
                def _mat_1():
                    if tensor_1_holder[0] is None:
                        tensor_1_holder[0] = _contribs_1()
                    s_idx = u - 1
                    e_idx = min(v, epoch_1_config.epoch_days)
                    return np.sum(tensor_1_holder[0][:, s_idx:e_idx, :], axis=(0, 1))

                noisy = self.state_manager.get_or_create_noisy_node(
                    stream_config=epoch_1_config,
                    level=lvl,
                    node_index=n_idx,
                    sigma=sigma_1,
                    materialize_fn=_mat_1,
                    finalized_up_to_day=finalized_up_to_day_epoch_1,
                    manifest_fingerprint=mf_1,
                    public_roster_fingerprint=rf_1,
                )
                node_vectors_1.append(noisy)

            node_vectors_2 = []
            for lvl, n_idx in nodes_used_2:
                u, v = tree_2.node_interval(lvl, n_idx)
                def _mat_2():
                    if tensor_2_holder[0] is None:
                        tensor_2_holder[0] = _contribs_2()
                    s_idx = u - 1
                    e_idx = min(v, epoch_2_config.epoch_days)
                    return np.sum(tensor_2_holder[0][:, s_idx:e_idx, :], axis=(0, 1))

                noisy = self.state_manager.get_or_create_noisy_node(
                    stream_config=epoch_2_config,
                    level=lvl,
                    node_index=n_idx,
                    sigma=sigma_2,
                    materialize_fn=_mat_2,
                    finalized_up_to_day=finalized_up_to_day_epoch_2,
                    manifest_fingerprint=mf_2,
                    public_roster_fingerprint=rf_2,
                )
                node_vectors_2.append(noisy)

            # 5. Commit all reservations across both epochs
            if all_reservations:
                self.state_manager.commit_continual_level_exposures(all_reservations)

        except Exception:
            if all_reservations:
                self.state_manager.abort_continual_level_exposures(all_reservations)
            raise

        vec_1 = np.sum(node_vectors_1, axis=0)
        vec_2 = np.sum(node_vectors_2, axis=0)
        combined_vector = vec_1 + vec_2

        std_1 = float(sigma_1 * math.sqrt(len(nodes_used_1)))
        std_2 = float(sigma_2 * math.sqrt(len(nodes_used_2)))
        combined_noise_std = math.sqrt(std_1**2 + std_2**2)

        combined_nodes = list(nodes_used_1) + list(nodes_used_2)
        combined_levels = sorted(list(set(levels_used_1 + levels_used_2)))
        newly_exposed = sorted(list(set(newly_1 + newly_2)))
        reused = sorted(list(set(reused_1 + reused_2)))

        q_id_1 = self.state_manager.log_query(
            budget_scope_id=budget_scope_id,
            stream_fingerprint=epoch_1_config.stream_fingerprint,
            epoch_id=epoch_1_config.epoch_id,
            query_type="CROSS_EPOCH_PART1",
            start_day=epoch_1_start_day,
            end_day=epoch_1_config.epoch_days,
            nodes_used=nodes_used_1,
            levels_used=levels_used_1,
            newly_exposed_levels=newly_1,
        )
        q_id_2 = self.state_manager.log_query(
            budget_scope_id=budget_scope_id,
            stream_fingerprint=epoch_2_config.stream_fingerprint,
            epoch_id=epoch_2_config.epoch_id,
            query_type="CROSS_EPOCH_PART2",
            start_day=1,
            end_day=epoch_2_end_day,
            nodes_used=nodes_used_2,
            levels_used=levels_used_2,
            newly_exposed_levels=newly_2,
        )

        _, cell_names = build_aggregation_matrix(epoch_1_config.granularity)

        return ContinualQueryResult(
            vector=combined_vector,
            cell_names=cell_names,
            nodes_used=combined_nodes,
            levels_used=combined_levels,
            newly_exposed_levels=newly_exposed,
            reused_levels=reused,
            budget_scope_id=budget_scope_id,
            expected_query_noise_std=combined_noise_std,
            query_id=f"cross_{q_id_1}_{q_id_2}",
            stream_fingerprint=f"{epoch_1_config.stream_fingerprint}+{epoch_2_config.stream_fingerprint}",
            epoch_id=f"{epoch_1_config.epoch_id}+{epoch_2_config.epoch_id}",
            start_day=epoch_1_start_day,
            end_day=epoch_2_end_day,
            query_type="CROSS_EPOCH",
            production_safe=True,
        )
