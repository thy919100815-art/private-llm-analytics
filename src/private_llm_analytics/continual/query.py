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
import json
import math
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple, Union
import numpy as np
import pandas as pd

from private_llm_analytics.continual.contributions import (
    build_temporal_user_contributions,
)
from private_llm_analytics.continual.mechanism import (
    calibrate_tree_gaussian_sigma,
    evaluate_rdp_epsilon,
)
from private_llm_analytics.continual.models import ContinualStreamConfig
from private_llm_analytics.continual.state import ContinualStateManager
from private_llm_analytics.continual.tree import BinaryTimeTree
from private_llm_analytics.planning.policy import build_aggregation_matrix
from private_llm_analytics.privacy.composition import resolve_budget_scope_id
from private_llm_analytics.privacy.exceptions import (
    PrivacyBudgetExceededError,
    PrivacyConfigurationError,
    ProductionSafetyConfigurationError,
)


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
        delta_global: float = 1e-5,
        epsilon_cap: float = 4.0,
    ):
        self.state_manager = state_manager
        self.composition_mode = composition_mode
        self.collusion_groups = collusion_groups or {}
        self.delta_global = float(delta_global)
        self.epsilon_cap = float(epsilon_cap)

    def resolve_scope(self, recipient: str, recipient_role: str = "external") -> str:
        """Resolve budget scope ID according to active recipient composition policy."""
        return resolve_budget_scope_id(
            recipient=recipient,
            recipient_role=recipient_role,
            mode=self.composition_mode,
            groups=self.collusion_groups,
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
            rdp_orders=stream_config.noise.rdp_orders,
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
        recipient_role: str = "external",
        finalized_up_to_day: Optional[int] = None,
        precomputed_contributions: Optional[np.ndarray] = None,
    ) -> ContinualQueryResult:
        """Internal execution of tree-based dyadic interval query."""
        # 1. Register stream configuration
        self.state_manager.register_stream(stream_config)

        # 2. Finalized day guard (Section 36, 101)
        if finalized_up_to_day is not None and end_day > finalized_up_to_day:
            raise ValueError(
                f"DATA_NOT_FINALIZED: Requested interval [{start_day}, {end_day}] "
                f"exceeds finalized day {finalized_up_to_day}."
            )

        # 3. Validate interval
        if not (1 <= start_day <= end_day <= stream_config.epoch_days):
            raise ValueError(
                f"Requested interval [{start_day}, {end_day}] out of epoch bounds [1, {stream_config.epoch_days}]"
            )

        # 4. Canonical cover
        tree = BinaryTimeTree(epoch_days=stream_config.epoch_days)
        nodes_used = tree.canonical_cover(start_day, end_day)
        levels_used = sorted(list({lvl for lvl, _ in nodes_used}))

        # 5. Determine sigma
        sigma = self._get_calibrated_sigma(stream_config)

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
            )

        # 9. Materialize / Retrieve Canonical Noisy Nodes
        # Helper to lazily prepare contributions tensor if materialization is needed
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

        node_noisy_vectors: List[np.ndarray] = []
        for level, node_idx in nodes_used:
            u, v = tree.node_interval(level, node_idx)
            # Define lazy materializer
            def _materialize_node_exact():
                contribs = _get_contributions()
                # 0-based slice [u - 1, min(v, epoch_days)]
                start_idx = u - 1
                end_idx = min(v, stream_config.epoch_days)
                # Sum over interval for all users: (n_users, dim) -> sum over users -> (dim,)
                node_slice = contribs[:, start_idx:end_idx, :]
                exact_node = np.sum(node_slice, axis=(0, 1))
                return exact_node

            noisy_node = self.state_manager.get_or_create_noisy_node(
                stream_config=stream_config,
                level=level,
                node_index=node_idx,
                sigma=sigma,
                materialize_fn=_materialize_node_exact,
            )
            node_noisy_vectors.append(noisy_node)

        # 10. Commit level exposures
        if reservations:
            self.state_manager.commit_continual_level_exposures(reservations)

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
        recipient_role: str = "external",
        finalized_up_to_day: Optional[int] = None,
        precomputed_contributions: Optional[np.ndarray] = None,
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
        )

    def query_prefix(
        self,
        stream_config: ContinualStreamConfig,
        end_day: int,
        recipient: str,
        interactions: Optional[Union[pd.DataFrame, Sequence[Dict[str, Any]]]] = None,
        public_user_roster: Optional[Sequence[str]] = None,
        recipient_role: str = "external",
        finalized_up_to_day: Optional[int] = None,
        precomputed_contributions: Optional[np.ndarray] = None,
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
        )

    def query_sliding_window(
        self,
        stream_config: ContinualStreamConfig,
        end_day: int,
        window_days: int,
        recipient: str,
        interactions: Optional[Union[pd.DataFrame, Sequence[Dict[str, Any]]]] = None,
        public_user_roster: Optional[Sequence[str]] = None,
        recipient_role: str = "external",
        finalized_up_to_day: Optional[int] = None,
        precomputed_contributions: Optional[np.ndarray] = None,
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
    ) -> ContinualQueryResult:
        """Cross-epoch interval query decomposing across epoch boundaries (Section 103)."""
        res_1 = self.query_interval(
            stream_config=epoch_1_config,
            start_day=epoch_1_start_day,
            end_day=epoch_1_config.epoch_days,
            recipient=recipient,
            interactions=interactions_epoch_1,
            public_user_roster=public_user_roster,
            recipient_role=recipient_role,
        )

        res_2 = self.query_interval(
            stream_config=epoch_2_config,
            start_day=1,
            end_day=epoch_2_end_day,
            recipient=recipient,
            interactions=interactions_epoch_2,
            public_user_roster=public_user_roster,
            recipient_role=recipient_role,
        )

        combined_vector = res_1.vector + res_2.vector
        combined_noise_std = math.sqrt(res_1.expected_query_noise_std**2 + res_2.expected_query_noise_std**2)
        combined_nodes = list(res_1.nodes_used) + list(res_2.nodes_used)
        combined_levels = sorted(list(set(res_1.levels_used + res_2.levels_used)))
        newly_exposed = sorted(list(set(res_1.newly_exposed_levels + res_2.newly_exposed_levels)))
        reused = sorted(list(set(res_1.reused_levels + res_2.reused_levels)))

        return ContinualQueryResult(
            vector=combined_vector,
            cell_names=res_1.cell_names,
            nodes_used=combined_nodes,
            levels_used=combined_levels,
            newly_exposed_levels=newly_exposed,
            reused_levels=reused,
            budget_scope_id=res_1.budget_scope_id,
            expected_query_noise_std=combined_noise_std,
            query_id=f"cross_{res_1.query_id}_{res_2.query_id}",
            stream_fingerprint=f"{epoch_1_config.stream_fingerprint}+{epoch_2_config.stream_fingerprint}",
            epoch_id=f"{epoch_1_config.epoch_id}+{epoch_2_config.epoch_id}",
            start_day=epoch_1_start_day,
            end_day=epoch_2_end_day,
            query_type="CROSS_EPOCH",
            production_safe=True,
        )
