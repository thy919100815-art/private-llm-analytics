"""Comprehensive verification and adversarial tests for Continual DP closure (v0.4).

Covers all required tests from Section A, B, C, D, E, F, G, and K:
- Section A: Unified Authoritative Lifetime RDP Accounting
- Section B: Authoritative RDP Order Grid
- Section C: Recipient Policy Binding
- Section D: Immutable Epoch Dataset Manifest
- Section E: Exposure Reservation Crash Recovery
- Section F: Production Config Integration
- Section G: Cross-Epoch Atomicity
- Section K: Adversarial Hardening
"""

from datetime import datetime, timezone
from decimal import Decimal
import json
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import numpy as np
import pandas as pd
import pytest

from private_llm_analytics.config.loader import load_app_config
from private_llm_analytics.config.models import AppConfig
from private_llm_analytics.continual.contributions import build_temporal_user_contributions
from private_llm_analytics.continual.mechanism import (
    calibrate_tree_gaussian_sigma,
    compute_level_rdp_vector,
)
from private_llm_analytics.continual.models import (
    ContinualNoiseConfig,
    ContinualStreamConfig,
)
from private_llm_analytics.continual.query import ContinualQueryEngine
from private_llm_analytics.continual.state import ContinualStateManager
from private_llm_analytics.continual.tree import BinaryTimeTree
from private_llm_analytics.metrics.base import DPReleaseResult
from private_llm_analytics.privacy.budget import PrivacyBudget
from private_llm_analytics.privacy.exceptions import (
    PrivacyBudgetExceededError,
    PrivacyConfigurationError,
    ProductionSafetyConfigurationError,
    ReleaseInProgressError,
    ReservationLostError,
)
from private_llm_analytics.privacy.rdp import (
    DEFAULT_RDP_ORDERS,
    compute_gaussian_rdp,
    compose_rdp_vectors,
    rdp_to_epsilon_delta,
    rdp_vector_to_json,
    to_decimal,
)
from private_llm_analytics.privacy.state import PrivacyStateManager


def _sample_data() -> Tuple[pd.DataFrame, List[str]]:
    roster = [f"u_{i}" for i in range(10)]
    records = []
    for d in range(1, 31):
        for u in roster[:5]:
            records.append({
                "user_id": u,
                "timestamp": f"2026-01-{d:02d} 10:00:00",
                "provider": "provider_a",
                "task": "technical",
            })
    return pd.DataFrame(records), roster


# =========================================================================
# SECTION A: UNIFIED AUTHORITATIVE LIFETIME RDP ACCOUNTING
# =========================================================================

def test_standard_release_after_continual_counts_tree_spend(tmp_path: Path):
    """Standard release authorization must see historical continual level exposures."""
    db_file = tmp_path / "unified_a1.db"
    state_mgr = PrivacyStateManager(db_path=db_file)
    cont_mgr = ContinualStateManager(db_path=db_file, state_manager=state_mgr)

    stream_cfg = ContinualStreamConfig(
        stream_id="s1",
        epoch_id="ep1",
        epoch_days=30,
        epoch_group_l12_bound=20.0,
        dataset_fingerprint="dfp1",
        public_roster_fingerprint="rfp1",
    )
    cont_mgr.register_stream(stream_cfg)

    # Continual reserves and commits 3 levels with target budget
    sigma = calibrate_tree_gaussian_sigma(
        epoch_days=30,
        epoch_l12_bound=20.0,
        target_tree_epsilon=3.0,
        calibration_delta=1e-6,
    )
    res = cont_mgr.reserve_continual_level_exposures(
        budget_scope_id="global_external",
        stream_config=stream_cfg,
        missing_levels=[0, 1, 2],
        sigma=sigma,
        epsilon_cap=4.0,
        delta_global=1e-6,
    )
    assert len(res) == 3
    cont_mgr.commit_continual_level_exposures(res)

    # Standard release attempt with large epsilon must exceed total cap=4.0
    # because tree already consumed ~2.5+ epsilon
    large_cand_rdp = compute_gaussian_rdp(sigma=5.0, sensitivity=20.0, orders=DEFAULT_RDP_ORDERS)
    with pytest.raises(PrivacyBudgetExceededError):
        state_mgr.reserve_or_get_release(
            release_key="std_rel_1",
            canonical_semantics={"metric": "vol"},
            global_budget=PrivacyBudget(epsilon=4.0, delta=1e-6),
            budget_needed=PrivacyBudget(epsilon=2.5, delta=1e-6),
            plan_metadata={},
            accountant_mode="rdp",
            rdp_orders=DEFAULT_RDP_ORDERS,
            rdp_candidate_cost=large_cand_rdp,
            epsilon_cap=4.0,
            delta_global=1e-6,
        )


def test_continual_after_standard_counts_standard_spend(tmp_path: Path):
    """Continual reservation must see historical committed standard releases."""
    db_file = tmp_path / "unified_a2.db"
    state_mgr = PrivacyStateManager(db_path=db_file)
    cont_mgr = ContinualStateManager(db_path=db_file, state_manager=state_mgr)

    # Commit a standard release consuming significant budget (~3.3 eps)
    cand_rdp = compute_gaussian_rdp(sigma=25.0, sensitivity=15.0, orders=DEFAULT_RDP_ORDERS)
    r = state_mgr.reserve_or_get_release(
        release_key="std_rel_commit",
        canonical_semantics={"metric": "vol"},
        global_budget=PrivacyBudget(epsilon=4.0, delta=1e-6),
        budget_needed=PrivacyBudget(epsilon=3.3, delta=1e-6),
        plan_metadata={},
        accountant_mode="rdp",
        rdp_orders=DEFAULT_RDP_ORDERS,
        rdp_candidate_cost=cand_rdp,
        epsilon_cap=4.0,
        delta_global=1e-6,
    )
    assert r.status == "RESERVED"
    state_mgr.commit_release(
        reservation_id=r.reservation_id,
        release_key="std_rel_commit",
        release_id=r.release_id,
        release_result=DPReleaseResult(
            metric_name="vol",
            dp_vector=np.array([10.0]),
            post_processed={},
            epsilon=3.3,
            delta=1e-6,
            mechanism="gaussian",
            noise_scale=25.0,
            sensitivity=15.0,
            cell_names=["c1"],
            n_users=10,
        ),
        recipient="all",
        dataset_version="v1",
        generation=r.generation,
        owner_id=r.owner_id,
    )

    # Continual attempting to reserve tree levels exceeding remaining ~0.7 eps must fail closed
    stream_cfg = ContinualStreamConfig(
        stream_id="s1",
        epoch_id="ep1",
        epoch_days=30,
        epoch_group_l12_bound=15.0,
        dataset_fingerprint="dfp1",
        public_roster_fingerprint="rfp1",
    )
    cont_mgr.register_stream(stream_cfg)
    sigma = 25.0  # level rdp cost (~3.3 eps) will push total over 4.0 cap
    with pytest.raises(PrivacyBudgetExceededError):
        cont_mgr.reserve_continual_level_exposures(
            budget_scope_id="global_external",
            stream_config=stream_cfg,
            missing_levels=[0, 1, 2, 3],
            sigma=sigma,
            epsilon_cap=4.0,
            delta_global=1e-6,
        )


def test_standard_and_continual_pending_reservations_compose(tmp_path: Path):
    """Standard release and continual exposure see each other's PENDING/RESERVED spend."""
    db_file = tmp_path / "unified_a3.db"
    state_mgr = PrivacyStateManager(db_path=db_file)
    cont_mgr = ContinualStateManager(db_path=db_file, state_manager=state_mgr)

    # 1. Standard reserves (status = PENDING)
    cand_rdp = compute_gaussian_rdp(sigma=25.0, sensitivity=15.0, orders=DEFAULT_RDP_ORDERS)
    r = state_mgr.reserve_or_get_release(
        release_key="std_pending_key",
        canonical_semantics={"metric": "vol"},
        global_budget=PrivacyBudget(epsilon=4.0, delta=1e-6),
        budget_needed=PrivacyBudget(epsilon=3.3, delta=1e-6),
        plan_metadata={},
        accountant_mode="rdp",
        rdp_orders=DEFAULT_RDP_ORDERS,
        rdp_candidate_cost=cand_rdp,
        epsilon_cap=4.0,
        delta_global=1e-6,
    )
    assert r.status == "RESERVED"

    # 2. Continual tries to reserve while standard is still PENDING
    stream_cfg = ContinualStreamConfig(
        stream_id="s1",
        epoch_id="ep1",
        epoch_days=30,
        epoch_group_l12_bound=15.0,
        dataset_fingerprint="dfp1",
        public_roster_fingerprint="rfp1",
    )
    cont_mgr.register_stream(stream_cfg)

    # Must see standard's PENDING reservation and reject when combined spend > cap
    sigma = 25.0
    with pytest.raises(PrivacyBudgetExceededError):
        cont_mgr.reserve_continual_level_exposures(
            budget_scope_id="global_external",
            stream_config=stream_cfg,
            missing_levels=[0, 1, 2],
            sigma=sigma,
            epsilon_cap=4.0,
            delta_global=1e-6,
        )


def test_two_concurrent_continual_queries_cannot_oversubscribe_cap(tmp_path: Path):
    """Two continual queries in same budget scope cannot concurrently oversubscribe cap."""
    db_file = tmp_path / "unified_a4.db"
    state_mgr = PrivacyStateManager(db_path=db_file)
    cont_mgr = ContinualStateManager(db_path=db_file, state_manager=state_mgr)

    stream_cfg = ContinualStreamConfig(
        stream_id="s1",
        epoch_id="ep1",
        epoch_days=30,
        epoch_group_l12_bound=20.0,
        dataset_fingerprint="dfp1",
        public_roster_fingerprint="rfp1",
    )
    cont_mgr.register_stream(stream_cfg)

    # Query 1 reserves levels 0, 1 consuming ~3.13 eps (held in RESERVED status)
    sigma = 50.0
    res1 = cont_mgr.reserve_continual_level_exposures(
        budget_scope_id="global_external",
        stream_config=stream_cfg,
        missing_levels=[0, 1],
        sigma=sigma,
        epsilon_cap=4.0,
        delta_global=1e-6,
    )
    assert len(res1) == 2

    # Query 2 concurrently tries to reserve levels 2, 3, 4 which together with res1 exceeds 4.0
    with pytest.raises(PrivacyBudgetExceededError):
        cont_mgr.reserve_continual_level_exposures(
            budget_scope_id="global_external",
            stream_config=stream_cfg,
            missing_levels=[2, 3, 4],
            sigma=sigma,
            epsilon_cap=4.0,
            delta_global=1e-6,
        )


def test_standard_and_continual_concurrent_reservation_cannot_oversubscribe(tmp_path: Path):
    """Standard PENDING + Continual RESERVED cannot exceed cap."""
    db_file = tmp_path / "unified_a5.db"
    state_mgr = PrivacyStateManager(db_path=db_file)
    cont_mgr = ContinualStateManager(db_path=db_file, state_manager=state_mgr)

    stream_cfg = ContinualStreamConfig(
        stream_id="s1",
        epoch_id="ep1",
        epoch_days=30,
        epoch_group_l12_bound=20.0,
        dataset_fingerprint="dfp1",
        public_roster_fingerprint="rfp1",
    )
    cont_mgr.register_stream(stream_cfg)

    # Continual reserves levels 0, 1 (~3.13 eps)
    res1 = cont_mgr.reserve_continual_level_exposures(
        budget_scope_id="global_external",
        stream_config=stream_cfg,
        missing_levels=[0, 1],
        sigma=50.0,
        epsilon_cap=4.0,
        delta_global=1e-6,
    )
    assert len(res1) == 2

    # Standard tries to reserve remaining budget with candidate cost exceeding available cap
    cand_rdp = compute_gaussian_rdp(sigma=25.0, sensitivity=15.0, orders=DEFAULT_RDP_ORDERS)
    with pytest.raises(PrivacyBudgetExceededError):
        state_mgr.reserve_or_get_release(
            release_key="std_concurrent_key",
            canonical_semantics={"metric": "vol"},
            global_budget=PrivacyBudget(epsilon=4.0, delta=1e-6),
            budget_needed=PrivacyBudget(epsilon=3.0, delta=1e-6),
            plan_metadata={},
            accountant_mode="rdp",
            rdp_orders=DEFAULT_RDP_ORDERS,
            rdp_candidate_cost=cand_rdp,
            epsilon_cap=4.0,
            delta_global=1e-6,
        )


def test_privacy_status_after_standard_continual_standard_matches(tmp_path: Path):
    """Standard -> Continual -> Standard releases report matching effective lifetime epsilon in get_privacy_view."""
    db_file = tmp_path / "unified_status.db"
    state_mgr = PrivacyStateManager(db_path=db_file)
    cont_mgr = ContinualStateManager(db_path=db_file, state_manager=state_mgr)
    engine = ContinualQueryEngine(state_manager=cont_mgr)
    cfg = ContinualStreamConfig(stream_id="s_stat", epoch_id="ep1", epoch_days=30)
    df = pd.DataFrame([
        {"timestamp": "2026-01-01T10:00:00Z", "user_id": "u1", "provider": "OpenAI", "task": "Chat"}
    ])

    # 1. Standard release 1
    orders = DEFAULT_RDP_ORDERS
    rdp_vec1 = compute_gaussian_rdp(sigma=10.0, sensitivity=1.0, orders=orders)
    r1 = state_mgr.reserve_or_get_release(
        release_key="rel_1",
        canonical_semantics={"metric": "vol"},
        global_budget=PrivacyBudget(epsilon=4.0, delta=1e-6),
        budget_needed=PrivacyBudget(epsilon=0.5, delta=1e-6),
        plan_metadata={},
        accountant_mode="rdp",
        rdp_orders=orders,
        rdp_candidate_cost=rdp_vec1,
        policy_mode="global_external",
        budget_scope_id="global_external",
        epsilon_cap=4.0,
        delta_global=1e-6,
    )
    state_mgr.commit_release(
        reservation_id=r1.reservation_id,
        release_key="rel_1",
        release_id=r1.release_id,
        release_result=DPReleaseResult("vol", np.array([10.0]), {}, 0.5, 1e-6, "gaussian", 10.0, 1.0, ["c1"], 10),
        recipient="external_a",
        dataset_version="v1",
        generation=r1.generation,
    )
    v1 = state_mgr.get_privacy_view("global_external")
    assert v1.effective_epsilon > Decimal("0.0")

    # 2. Continual query
    engine.query_interval(
        stream_config=cfg,
        start_day=1,
        end_day=2,
        recipient="external_a",
        interactions=df,
        public_user_roster=["u1"],
        finalized_up_to_day=10,
        current_wall_clock_day=20,
    )
    v2 = engine.get_privacy_view("external_a")
    assert v2.effective_epsilon > v1.effective_epsilon

    # 3. Standard release 2
    rdp_vec2 = compute_gaussian_rdp(sigma=12.0, sensitivity=1.0, orders=orders)
    r2 = state_mgr.reserve_or_get_release(
        release_key="rel_2",
        canonical_semantics={"metric": "vol_2"},
        global_budget=PrivacyBudget(epsilon=4.0, delta=1e-6),
        budget_needed=PrivacyBudget(epsilon=0.4, delta=1e-6),
        plan_metadata={},
        accountant_mode="rdp",
        rdp_orders=orders,
        rdp_candidate_cost=rdp_vec2,
        policy_mode="global_external",
        budget_scope_id="global_external",
        epsilon_cap=4.0,
        delta_global=1e-6,
    )
    state_mgr.commit_release(
        reservation_id=r2.reservation_id,
        release_key="rel_2",
        release_id=r2.release_id,
        release_result=DPReleaseResult("vol_2", np.array([10.0]), {}, 0.4, 1e-6, "gaussian", 12.0, 1.0, ["c1"], 10),
        recipient="external_a",
        dataset_version="v1",
        generation=r2.generation,
    )
    v3 = state_mgr.get_privacy_view("global_external")
    assert v3.effective_epsilon > v2.effective_epsilon


# =========================================================================
# SECTION B: AUTHORITATIVE RDP ORDER GRID
# =========================================================================

def test_continual_uses_authoritative_accountant_orders():
    """Continual default order set matches PrivacyAccountant authoritative order set."""
    app_cfg = load_app_config("configs")
    expected_orders = tuple(float(x) for x in app_cfg.privacy.accountant.orders)
    c_noise = ContinualNoiseConfig()
    assert c_noise.rdp_orders == expected_orders


def test_standard_history_plus_default_tree_orders_composes(tmp_path: Path):
    """Standard releases and continual exposures share bitwise matching RDP order grids."""
    db_file = tmp_path / "orders_b2.db"
    state_mgr = PrivacyStateManager(db_path=db_file)
    cont_mgr = ContinualStateManager(db_path=db_file, state_manager=state_mgr)

    cand_rdp = compute_gaussian_rdp(sigma=25.0, sensitivity=10.0, orders=DEFAULT_RDP_ORDERS)
    r = state_mgr.reserve_or_get_release(
        release_key="std_r1",
        canonical_semantics={"metric": "vol"},
        global_budget=PrivacyBudget(epsilon=4.0, delta=1e-6),
        budget_needed=PrivacyBudget(epsilon=0.5, delta=1e-6),
        plan_metadata={},
        accountant_mode="rdp",
        rdp_orders=DEFAULT_RDP_ORDERS,
        rdp_candidate_cost=cand_rdp,
        epsilon_cap=4.0,
        delta_global=1e-6,
    )
    assert r.status == "RESERVED"

    stream_cfg = ContinualStreamConfig(
        stream_id="s1",
        epoch_id="ep1",
        epoch_days=30,
        epoch_group_l12_bound=10.0,
        dataset_fingerprint="dfp1",
        public_roster_fingerprint="rfp1",
    )
    cont_mgr.register_stream(stream_cfg)

    # Continual reservation succeeds and composes without order mismatch error
    res = cont_mgr.reserve_continual_level_exposures(
        budget_scope_id="global_external",
        stream_config=stream_cfg,
        missing_levels=[0],
        sigma=30.0,
        epsilon_cap=4.0,
        delta_global=1e-6,
    )
    assert len(res) == 1


def test_rdp_order_mismatch_fails_closed(tmp_path: Path):
    """Attempting to compose with mismatched RDP order grid fails closed."""
    db_file = tmp_path / "orders_b3.db"
    state_mgr = PrivacyStateManager(db_path=db_file)

    # Insert a corrupt reservation with orders [2.0, 4.0] (mismatching standard 12 orders)
    with state_mgr._connection() as conn:
        conn.execute(
            """
            INSERT INTO privacy_reservations (
                reservation_id, release_key, status, created_at, budget_scope_id,
                epsilon, delta, accountant_mode, rdp_cost_json, rdp_orders_json
            ) VALUES (
                'res_mismatch', 'rel_mismatch', 'COMMITTED', '2026-01-01', 'global_external',
                '1.0', '1e-6', 'rdp', '{"2.0": "0.1", "4.0": "0.2"}', '[2.0, 4.0]'
            )
            """
        )
        conn.commit()

    # Subsequent authorization in same scope must fail closed
    cand_rdp = compute_gaussian_rdp(sigma=25.0, sensitivity=10.0, orders=DEFAULT_RDP_ORDERS)
    with pytest.raises(PrivacyConfigurationError, match="mismatch authoritative grid"):
        state_mgr.reserve_or_get_release(
            release_key="rel_next",
            canonical_semantics={"metric": "vol"},
            global_budget=PrivacyBudget(epsilon=4.0, delta=1e-6),
            budget_needed=PrivacyBudget(epsilon=0.5, delta=1e-6),
            plan_metadata={},
            accountant_mode="rdp",
            rdp_orders=DEFAULT_RDP_ORDERS,
            rdp_candidate_cost=cand_rdp,
            epsilon_cap=4.0,
            delta_global=1e-6,
        )


# =========================================================================
# SECTION C: RECIPIENT POLICY BINDING
# =========================================================================

def test_standard_global_then_continual_per_recipient_fails(tmp_path: Path):
    """If DB bound to global_external by standard release, continual per_recipient fails closed."""
    db_file = tmp_path / "policy_c1.db"
    state_mgr = PrivacyStateManager(db_path=db_file)
    cont_mgr = ContinualStateManager(db_path=db_file, state_manager=state_mgr)

    # Standard release binds global_external
    cand_rdp = compute_gaussian_rdp(sigma=30.0, sensitivity=10.0, orders=DEFAULT_RDP_ORDERS)
    r = state_mgr.reserve_or_get_release(
        release_key="std_global",
        canonical_semantics={"metric": "vol"},
        global_budget=PrivacyBudget(epsilon=4.0, delta=1e-6),
        budget_needed=PrivacyBudget(epsilon=0.5, delta=1e-6),
        plan_metadata={},
        accountant_mode="rdp",
        rdp_orders=DEFAULT_RDP_ORDERS,
        rdp_candidate_cost=cand_rdp,
        policy_mode="global_external",
        budget_scope_id="global_external",
        epsilon_cap=4.0,
        delta_global=1e-6,
    )
    assert r.status == "RESERVED"

    # Continual attempts per_recipient on existing DB
    stream_cfg = ContinualStreamConfig(
        stream_id="s1",
        epoch_id="ep1",
        epoch_days=30,
        dataset_fingerprint="dfp1",
        public_roster_fingerprint="rfp1",
    )
    cont_mgr.register_stream(stream_cfg)

    with pytest.raises(ProductionSafetyConfigurationError, match="Recipient composition policy mismatch"):
        cont_mgr.reserve_continual_level_exposures(
            budget_scope_id="recipient:provider_b",
            stream_config=stream_cfg,
            missing_levels=[0],
            sigma=30.0,
            policy_mode="per_recipient",
        )


def test_continual_global_then_standard_per_recipient_fails(tmp_path: Path):
    """If DB bound to global_external by continual query, standard per_recipient fails closed."""
    db_file = tmp_path / "policy_c2.db"
    state_mgr = PrivacyStateManager(db_path=db_file)
    cont_mgr = ContinualStateManager(db_path=db_file, state_manager=state_mgr)

    stream_cfg = ContinualStreamConfig(
        stream_id="s1",
        epoch_id="ep1",
        epoch_days=30,
        epoch_group_l12_bound=10.0,
        dataset_fingerprint="dfp1",
        public_roster_fingerprint="rfp1",
    )
    cont_mgr.register_stream(stream_cfg)

    res = cont_mgr.reserve_continual_level_exposures(
        budget_scope_id="global_external",
        stream_config=stream_cfg,
        missing_levels=[0],
        sigma=30.0,
        policy_mode="global_external",
    )
    assert len(res) == 1

    # Standard tries per_recipient
    cand_rdp = compute_gaussian_rdp(sigma=30.0, sensitivity=10.0, orders=DEFAULT_RDP_ORDERS)
    with pytest.raises(ProductionSafetyConfigurationError, match="Recipient composition policy mismatch"):
        state_mgr.reserve_or_get_release(
            release_key="std_per_rec",
            canonical_semantics={"metric": "vol"},
            global_budget=PrivacyBudget(epsilon=4.0, delta=1e-6),
            budget_needed=PrivacyBudget(epsilon=0.5, delta=1e-6),
            plan_metadata={},
            accountant_mode="rdp",
            rdp_orders=DEFAULT_RDP_ORDERS,
            rdp_candidate_cost=cand_rdp,
            policy_mode="per_recipient",
            budget_scope_id="recipient:auditor",
        )


def test_continual_policy_binding_is_atomic(tmp_path: Path):
    """Policy metadata is bound in same transaction as continual reservation."""
    db_file = tmp_path / "policy_c3.db"
    state_mgr = PrivacyStateManager(db_path=db_file)
    cont_mgr = ContinualStateManager(db_path=db_file, state_manager=state_mgr)

    assert not state_mgr.has_policy_bound()

    stream_cfg = ContinualStreamConfig(
        stream_id="s1",
        epoch_id="ep1",
        epoch_days=30,
        epoch_group_l12_bound=10.0,
        dataset_fingerprint="dfp1",
        public_roster_fingerprint="rfp1",
    )
    cont_mgr.register_stream(stream_cfg)

    res = cont_mgr.reserve_continual_level_exposures(
        budget_scope_id="global_external",
        stream_config=stream_cfg,
        missing_levels=[0],
        sigma=30.0,
        policy_mode="global_external",
    )
    assert len(res) == 1
    assert state_mgr.has_policy_bound()


def test_continual_recipient_role_change_fails_closed(tmp_path: Path):
    """Changing recipient role binding on active DB fails closed."""
    db_file = tmp_path / "policy_c4.db"
    state_mgr = PrivacyStateManager(db_path=db_file)
    cont_mgr = ContinualStateManager(db_path=db_file, state_manager=state_mgr)

    stream_cfg = ContinualStreamConfig(
        stream_id="s1",
        epoch_id="ep1",
        epoch_days=30,
        epoch_group_l12_bound=10.0,
        dataset_fingerprint="dfp1",
        public_roster_fingerprint="rfp1",
    )
    cont_mgr.register_stream(stream_cfg)

    res = cont_mgr.reserve_continual_level_exposures(
        budget_scope_id="global_external",
        stream_config=stream_cfg,
        missing_levels=[0],
        sigma=30.0,
        policy_mode="global_external",
        recipient_roles={"ProviderA": "external"},
    )
    assert len(res) == 1

    # Attempting to change role of ProviderA from external to internal fails closed
    with pytest.raises(ProductionSafetyConfigurationError, match="Recipient composition policy mismatch"):
        cont_mgr.reserve_continual_level_exposures(
            budget_scope_id="global_external",
            stream_config=stream_cfg,
            missing_levels=[1],
            sigma=30.0,
            policy_mode="global_external",
            recipient_roles={"ProviderA": "internal"},
        )


def test_from_app_config_uses_collusion_groups(tmp_path: Path):
    """ContinualQueryEngine.from_app_config properly binds collusion groups from app_config.privacy.recipient_composition.groups."""
    db_file = tmp_path / "app_config_groups.db"
    app_cfg = load_app_config("configs")
    app_cfg.privacy.release.state_db_path = str(db_file)
    app_cfg.privacy.accountant.mode = "rdp"
    app_cfg.privacy.continual.enabled = True
    app_cfg.privacy.recipient_composition.mode = "collusion_group"
    app_cfg.privacy.recipient_composition.groups = {
        "consortium": ["ProviderA", "ProviderB"]
    }

    engine = ContinualQueryEngine.from_app_config(app_cfg)
    assert engine.resolve_scope("ProviderA") == "collusion:consortium"
    assert engine.resolve_scope("ProviderB") == "collusion:consortium"


def test_app_config_collusion_group_members_share_budget(tmp_path: Path):
    """Members of a collusion group share continual level exposure spend and cap."""
    db_file = tmp_path / "collusion_share.db"
    app_cfg = load_app_config("configs")
    app_cfg.privacy.release.state_db_path = str(db_file)
    app_cfg.privacy.accountant.mode = "rdp"
    app_cfg.privacy.accountant.epsilon_cap = 4.0
    app_cfg.privacy.continual.enabled = True
    app_cfg.privacy.recipient_composition.mode = "collusion_group"
    app_cfg.privacy.recipient_composition.groups = {
        "consortium": ["ProviderA", "ProviderB"]
    }

    engine = ContinualQueryEngine.from_app_config(app_cfg)
    cfg = ContinualStreamConfig(stream_id="s_col", epoch_id="ep1", epoch_days=30)
    df = pd.DataFrame([
        {"timestamp": "2026-01-01T10:00:00Z", "user_id": "u1", "provider": "OpenAI", "task": "Chat"}
    ])

    res_a = engine.query_interval(
        stream_config=cfg,
        start_day=1,
        end_day=1,
        recipient="ProviderA",
        interactions=df,
        public_user_roster=["u1"],
        finalized_up_to_day=10,
        current_wall_clock_day=15,
    )
    assert res_a.budget_scope_id == "collusion:consortium"

    # Query by ProviderB reuses the exposed level and shares the same scope
    res_b = engine.query_interval(
        stream_config=cfg,
        start_day=1,
        end_day=1,
        recipient="ProviderB",
        interactions=df,
        public_user_roster=["u1"],
        finalized_up_to_day=10,
        current_wall_clock_day=15,
    )
    assert res_b.budget_scope_id == "collusion:consortium"
    assert len(res_b.reused_levels) > 0


# =========================================================================
# SECTION D: IMMUTABLE FINALIZED DATASET SNAPSHOT
# =========================================================================

def test_different_nodes_same_level_require_same_epoch_manifest(tmp_path: Path):
    """Nodes at the same level in an epoch must be materialized under the same sealed manifest."""
    db_file = tmp_path / "manifest_d1.db"
    cont_mgr = ContinualStateManager(db_path=db_file)

    cfg = ContinualStreamConfig(stream_id="s_man", epoch_id="ep1", epoch_days=30)
    cont_mgr.register_stream(cfg)

    # Node 0 at level 0 materialized with manifest 1
    n0 = cont_mgr.get_or_create_noisy_node(
        stream_config=cfg,
        level=0,
        node_index=0,
        sigma=5.0,
        materialize_fn=lambda: np.array([1.0]),
        manifest_fingerprint="manifest_hash_1",
        public_roster_fingerprint="roster_hash_1",
        finalized_up_to_day=30,
    )
    assert n0 is not None

    # Node 1 at level 0 attempted with manifest 2 -> must fail closed
    with pytest.raises(ProductionSafetyConfigurationError, match="DATASET_MANIFEST_MISMATCH"):
        cont_mgr.get_or_create_noisy_node(
            stream_config=cfg,
            level=0,
            node_index=1,
            sigma=5.0,
            materialize_fn=lambda: np.array([2.0]),
            manifest_fingerprint="manifest_hash_2",
            public_roster_fingerprint="roster_hash_1",
            finalized_up_to_day=30,
        )


def test_manifest_mutation_after_first_node_fails_closed(tmp_path: Path):
    """Mutating manifest after first node is sealed raises DATASET_MANIFEST_MISMATCH."""
    db_file = tmp_path / "manifest_d2.db"
    cont_mgr = ContinualStateManager(db_path=db_file)
    cfg = ContinualStreamConfig(stream_id="s_mut", epoch_id="ep1", epoch_days=30)

    cont_mgr.get_or_create_noisy_node(
        stream_config=cfg,
        level=0,
        node_index=0,
        sigma=5.0,
        materialize_fn=lambda: np.array([10.0]),
        manifest_fingerprint="m_initial",
        public_roster_fingerprint="r_initial",
        finalized_up_to_day=30,
    )

    with pytest.raises(ProductionSafetyConfigurationError, match="DATASET_MANIFEST_MISMATCH"):
        cont_mgr.get_or_create_noisy_node(
            stream_config=cfg,
            level=1,
            node_index=0,
            sigma=5.0,
            materialize_fn=lambda: np.array([10.0]),
            manifest_fingerprint="m_altered",
            public_roster_fingerprint="r_initial",
            finalized_up_to_day=30,
        )


def test_public_roster_change_fails_closed(tmp_path: Path):
    """Altering public user roster fingerprint for sealed epoch fails closed."""
    db_file = tmp_path / "manifest_d3.db"
    cont_mgr = ContinualStateManager(db_path=db_file)
    cfg = ContinualStreamConfig(stream_id="s_rost", epoch_id="ep1", epoch_days=30)

    cont_mgr.get_or_create_noisy_node(
        stream_config=cfg,
        level=0,
        node_index=0,
        sigma=5.0,
        materialize_fn=lambda: np.array([10.0]),
        manifest_fingerprint="m_fixed",
        public_roster_fingerprint="r_roster1",
        finalized_up_to_day=30,
    )

    with pytest.raises(ProductionSafetyConfigurationError, match="DATASET_MANIFEST_MISMATCH"):
        cont_mgr.get_or_create_noisy_node(
            stream_config=cfg,
            level=1,
            node_index=0,
            sigma=5.0,
            materialize_fn=lambda: np.array([10.0]),
            manifest_fingerprint="m_fixed",
            public_roster_fingerprint="r_roster2",
            finalized_up_to_day=30,
        )


def test_finalized_guard_cannot_be_omitted_in_production(tmp_path: Path):
    """Querying interval with finalized_up_to_day=None without prior manifest fails closed."""
    db_file = tmp_path / "manifest_d4.db"
    cont_mgr = ContinualStateManager(db_path=db_file)
    engine = ContinualQueryEngine(state_manager=cont_mgr)
    cfg = ContinualStreamConfig(stream_id="s_fin", epoch_id="ep1", epoch_days=30)
    df = pd.DataFrame([
        {"timestamp": "2026-01-01T10:00:00Z", "user_id": "u1", "provider": "OpenAI", "task": "Chat"}
    ])

    with pytest.raises(ProductionSafetyConfigurationError, match="finalized_up_to_day REQUIRED in production"):
        engine.query_interval(
            stream_config=cfg,
            start_day=1,
            end_day=10,
            recipient="external",
            interactions=df,
            public_user_roster=["u1"],
            finalized_up_to_day=None,
        )


def test_new_node_requires_authoritative_finalized_boundary(tmp_path: Path):
    """Direct node materialization without finalized boundary fails closed."""
    db_file = tmp_path / "new_node_bound.db"
    cont_mgr = ContinualStateManager(db_path=db_file)
    cfg = ContinualStreamConfig(stream_id="s_bound", epoch_id="ep1", epoch_days=30)

    with pytest.raises(ProductionSafetyConfigurationError, match="finalized_up_to_day REQUIRED in production"):
        cont_mgr.get_or_create_noisy_node(
            stream_config=cfg,
            level=0,
            node_index=0,
            sigma=5.0,
            materialize_fn=lambda: np.array([1.0]),
            manifest_fingerprint="m_test",
            public_roster_fingerprint="r_test",
            finalized_up_to_day=None,
        )


def test_lateness_grace_is_enforced(tmp_path: Path):
    """Setting finalized_up_to_day exceeding current_day - lateness_grace_days fails closed."""
    db_file = tmp_path / "manifest_d5.db"
    state_mgr = PrivacyStateManager(db_path=db_file)

    # current_day = 10, lateness_grace_days = 2 -> max allowed finalized = 8
    # Caller attempts finalized_up_to_day = 9 -> MUST fail closed
    with pytest.raises(ProductionSafetyConfigurationError, match="violates lateness_grace_days"):
        state_mgr.register_or_verify_epoch_manifest(
            stream_fingerprint="fp1",
            epoch_id="ep1",
            manifest_fingerprint="m1",
            public_roster_fingerprint="r1",
            finalized_up_to_day=9,
            lateness_grace_days=2,
            current_wall_clock_day=10,
        )


def test_lateness_grace_cannot_be_bypassed_by_omitting_wall_clock(tmp_path: Path):
    """Omitting current_wall_clock_day calculates it internally from epoch_start and enforces lateness grace."""
    db_file = tmp_path / "lateness_bypass.db"
    cont_mgr = ContinualStateManager(db_path=db_file)
    today_utc_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    cfg = ContinualStreamConfig(
        stream_id="s_late",
        epoch_id="ep1",
        epoch_start=today_utc_str,
        epoch_days=30,
        lateness_grace_days=1,
    )
    cont_mgr.register_stream(cfg)

    # Caller attempts to finalize through day 1 while omitting current_wall_clock_day
    with pytest.raises(ProductionSafetyConfigurationError, match="violates lateness_grace_days"):
        cont_mgr.state_manager.register_or_verify_epoch_manifest(
            stream_fingerprint=cfg.stream_fingerprint,
            epoch_id=cfg.epoch_id,
            manifest_fingerprint="m_test",
            public_roster_fingerprint="r_test",
            finalized_up_to_day=1,
            lateness_grace_days=cfg.lateness_grace_days,
            current_wall_clock_day=None,
        )


def test_finalized_frontier_advances_monotonically(tmp_path: Path):
    """Frontier advances monotonically from day 10 to day 11 in continual_epoch_manifests."""
    db_file = tmp_path / "manifest_mono.db"
    state_mgr = PrivacyStateManager(db_path=db_file)
    state_mgr.register_or_verify_epoch_manifest(
        stream_fingerprint="fp1",
        epoch_id="ep1",
        manifest_fingerprint="m1",
        public_roster_fingerprint="r1",
        finalized_up_to_day=10,
        current_wall_clock_day=20,
    )

    rec2 = state_mgr.register_or_verify_epoch_manifest(
        stream_fingerprint="fp1",
        epoch_id="ep1",
        manifest_fingerprint="m1",
        public_roster_fingerprint="r1",
        finalized_up_to_day=11,
        current_wall_clock_day=20,
    )
    assert rec2["finalized_up_to_day"] == 11


def test_finalized_frontier_cannot_move_backward(tmp_path: Path):
    """Attempting to roll back finalized frontier from day 11 to day 10 raises ProductionSafetyConfigurationError."""
    db_file = tmp_path / "manifest_noback.db"
    state_mgr = PrivacyStateManager(db_path=db_file)
    state_mgr.register_or_verify_epoch_manifest(
        stream_fingerprint="fp1",
        epoch_id="ep1",
        manifest_fingerprint="m1",
        public_roster_fingerprint="r1",
        finalized_up_to_day=11,
        current_wall_clock_day=20,
    )

    with pytest.raises(ProductionSafetyConfigurationError, match="cannot be rolled backward"):
        state_mgr.register_or_verify_epoch_manifest(
            stream_fingerprint="fp1",
            epoch_id="ep1",
            manifest_fingerprint="m1",
            public_roster_fingerprint="r1",
            finalized_up_to_day=10,
            current_wall_clock_day=20,
        )


def test_already_finalized_partition_cannot_mutate(tmp_path: Path):
    """Already finalized partition fingerprint in continual_epoch_partitions cannot be altered."""
    db_file = tmp_path / "partition_mut.db"
    state_mgr = PrivacyStateManager(db_path=db_file)
    state_mgr.register_or_verify_epoch_manifest(
        stream_fingerprint="fp1",
        epoch_id="ep1",
        manifest_fingerprint="m1",
        public_roster_fingerprint="r1",
        finalized_up_to_day=10,
        partition_fingerprints={"10": "hash_day10_v1"},
        current_wall_clock_day=20,
    )

    with pytest.raises(ProductionSafetyConfigurationError, match="already finalized partition for day 10 cannot be mutated"):
        state_mgr.register_or_verify_epoch_manifest(
            stream_fingerprint="fp1",
            epoch_id="ep1",
            manifest_fingerprint="m1",
            public_roster_fingerprint="r1",
            finalized_up_to_day=10,
            partition_fingerprints={"10": "hash_day10_tampered"},
            current_wall_clock_day=20,
        )


def test_future_partition_can_be_added_after_prior_days_sealed(tmp_path: Path):
    """Newly finalized day 11 partition can be sealed after days 1..10 are already sealed."""
    db_file = tmp_path / "future_part.db"
    state_mgr = PrivacyStateManager(db_path=db_file)
    state_mgr.register_or_verify_epoch_manifest(
        stream_fingerprint="fp1",
        epoch_id="ep1",
        manifest_fingerprint="m1",
        public_roster_fingerprint="r1",
        finalized_up_to_day=10,
        partition_fingerprints={"10": "hash_day10"},
        current_wall_clock_day=20,
    )

    rec = state_mgr.register_or_verify_epoch_manifest(
        stream_fingerprint="fp1",
        epoch_id="ep1",
        manifest_fingerprint="m1",
        public_roster_fingerprint="r1",
        finalized_up_to_day=11,
        partition_fingerprints={"11": "hash_day11"},
        current_wall_clock_day=20,
    )
    assert rec["finalized_up_to_day"] == 11
    parts = json.loads(rec["partition_fingerprints_json"])
    assert parts["10"] == "hash_day10"
    assert parts["11"] == "hash_day11"


def test_day11_query_succeeds_after_frontier_advances_10_to_11(tmp_path: Path):
    """Query reaching day 11 fails when frontier is 10, then succeeds when frontier advances to 11."""
    db_file = tmp_path / "day11_query.db"
    cont_mgr = ContinualStateManager(db_path=db_file)
    engine = ContinualQueryEngine(state_manager=cont_mgr)
    cfg = ContinualStreamConfig(stream_id="s_d11", epoch_id="ep1", epoch_days=30)
    df = pd.DataFrame([
        {"timestamp": f"2026-01-{d:02d}T10:00:00Z", "user_id": "u1", "provider": "OpenAI", "task": "Chat"}
        for d in range(1, 12)
    ])

    # First, register epoch through day 10
    cont_mgr.state_manager.register_or_verify_epoch_manifest(
        stream_fingerprint=cfg.stream_fingerprint,
        epoch_id=cfg.epoch_id,
        manifest_fingerprint="m1",
        public_roster_fingerprint="r1",
        finalized_up_to_day=10,
        current_wall_clock_day=25,
    )

    # Query reaching day 11 fails closed with DATA_NOT_FINALIZED
    with pytest.raises(ProductionSafetyConfigurationError, match="DATA_NOT_FINALIZED"):
        engine.query_interval(
            stream_config=cfg,
            start_day=1,
            end_day=11,
            recipient="external",
            interactions=df,
            public_user_roster=["u1"],
            manifest_fingerprint="m1",
            public_roster_fingerprint="r1",
            current_wall_clock_day=25,
        )

    # Advance frontier to 11
    res = engine.query_interval(
        stream_config=cfg,
        start_day=1,
        end_day=11,
        recipient="external",
        interactions=df,
        public_user_roster=["u1"],
        finalized_up_to_day=11,
        manifest_fingerprint="m1",
        public_roster_fingerprint="r1",
        current_wall_clock_day=25,
    )
    assert res is not None
    assert res.end_day == 11


def test_same_manifest_token_different_finalized_partition_data_fails_closed(tmp_path: Path):
    """Adversarial test: Supplying the same manifest token with altered underlying event telemetry fails closed."""
    db_file = tmp_path / "token_tamper.db"
    cont_mgr = ContinualStateManager(db_path=db_file)
    engine = ContinualQueryEngine(state_manager=cont_mgr)
    token = "manifest_token_abc"
    cfg = ContinualStreamConfig(
        stream_id="s_tamper",
        epoch_id="ep1",
        epoch_days=30,
        dataset_fingerprint=token,
        public_roster_fingerprint="roster_fixed",
    )

    df_original = pd.DataFrame([
        {"timestamp": f"2026-01-{d:02d}T10:00:00Z", "user_id": "u1", "provider": "OpenAI", "task": "Chat"}
        for d in range(1, 11)
    ])

    # Day 1..2 query seals partition fingerprints for days 1..10
    engine.query_interval(
        stream_config=cfg,
        start_day=1,
        end_day=2,
        recipient="external",
        interactions=df_original,
        public_user_roster=["u1"],
        finalized_up_to_day=10,
        current_wall_clock_day=20,
    )

    # Attacker alters events on Day 3 but claims the same manifest token
    df_tampered = pd.DataFrame([
        {"timestamp": f"2026-01-{d:02d}T10:00:00Z", "user_id": "u1", "provider": "OpenAI", "task": "Code" if d == 3 else "Chat"}
        for d in range(1, 11)
    ])

    with pytest.raises(ProductionSafetyConfigurationError, match=r"(DATASET_PARTITION_MISMATCH|already finalized partition)"):
        engine.query_interval(
            stream_config=cfg,
            start_day=3,
            end_day=4,
            recipient="external",
            interactions=df_tampered,
            public_user_roster=["u1"],
            finalized_up_to_day=10,
            current_wall_clock_day=20,
        )


def test_same_level_mixed_snapshot_attack_rejected(tmp_path: Path):
    """Security invariant: attacking level sensitivity by mixing different dataset snapshots is rejected."""
    db_file = tmp_path / "manifest_d6.db"
    cont_mgr = ContinualStateManager(db_path=db_file)
    cfg = ContinualStreamConfig(stream_id="s_atk", epoch_id="ep1", epoch_days=30)

    # Step 1: Materialize leaf node 1 from Dataset A
    cont_mgr.get_or_create_noisy_node(
        stream_config=cfg,
        level=0,
        node_index=0,
        sigma=5.0,
        materialize_fn=lambda: np.array([1.0]),
        manifest_fingerprint="dataset_A_hash",
        public_roster_fingerprint="roster_A_hash",
        finalized_up_to_day=30,
    )

    # Step 2: Attacker attempts to materialize leaf node 2 from Dataset B
    with pytest.raises(ProductionSafetyConfigurationError, match="DATASET_MANIFEST_MISMATCH"):
        cont_mgr.get_or_create_noisy_node(
            stream_config=cfg,
            level=0,
            node_index=1,
            sigma=5.0,
            materialize_fn=lambda: np.array([2.0]),
            manifest_fingerprint="dataset_B_hash",
            public_roster_fingerprint="roster_A_hash",
            finalized_up_to_day=30,
        )


# =========================================================================
# SECTION E: EXPOSURE RESERVATION CRASH RECOVERY
# =========================================================================

def test_exposure_crash_after_reserve_can_recover(tmp_path: Path):
    """If worker crashes after reserving levels, retry worker recovers reservation after lease expiry."""
    db_file = tmp_path / "crash_e1.db"
    state_mgr = PrivacyStateManager(db_path=db_file)
    cont_mgr = ContinualStateManager(db_path=db_file, state_manager=state_mgr)

    stream_cfg = ContinualStreamConfig(stream_id="s_cr", epoch_id="ep1", epoch_days=30)
    cont_mgr.register_stream(stream_cfg)

    # Worker 1 reserves level 0 with very short lease (0.01s)
    res1 = state_mgr.reserve_continual_level_exposures(
        budget_scope_id="global_external",
        stream_fingerprint=stream_cfg.stream_fingerprint,
        epoch_id="ep1",
        missing_levels=[0],
        level_rdp_cost=compute_level_rdp_vector(10.0, 30.0, DEFAULT_RDP_ORDERS),
        owner_id="worker_crashed",
        lease_timeout=0.01,
    )
    assert len(res1) == 1
    assert res1[0]["generation"] == 1

    # Worker 1 crashes (never commits or aborts)
    time.sleep(0.05)

    # Worker 2 retries the reservation -> takes over expired lease with generation 2
    res2 = state_mgr.reserve_continual_level_exposures(
        budget_scope_id="global_external",
        stream_fingerprint=stream_cfg.stream_fingerprint,
        epoch_id="ep1",
        missing_levels=[0],
        level_rdp_cost=compute_level_rdp_vector(10.0, 30.0, DEFAULT_RDP_ORDERS),
        owner_id="worker_recovered",
        lease_timeout=300.0,
    )
    assert len(res2) == 1
    assert res2[0]["generation"] == 2
    assert res2[0]["owner_id"] == "worker_recovered"

    # Worker 2 commits successfully
    state_mgr.commit_continual_level_exposures(res2)


def test_stale_exposure_reservation_takeover(tmp_path: Path):
    """Stale worker cannot commit exposure after takeover."""
    db_file = tmp_path / "crash_e2.db"
    state_mgr = PrivacyStateManager(db_path=db_file)

    stream_cfg = ContinualStreamConfig(stream_id="s_stale", epoch_id="ep1", epoch_days=30)

    # Worker 1 reserves with short lease
    res1 = state_mgr.reserve_continual_level_exposures(
        budget_scope_id="global_external",
        stream_fingerprint=stream_cfg.stream_fingerprint,
        epoch_id="ep1",
        missing_levels=[0],
        level_rdp_cost=compute_level_rdp_vector(10.0, 30.0, DEFAULT_RDP_ORDERS),
        owner_id="worker_1",
        lease_timeout=0.01,
    )
    time.sleep(0.05)

    # Worker 2 takes over
    res2 = state_mgr.reserve_continual_level_exposures(
        budget_scope_id="global_external",
        stream_fingerprint=stream_cfg.stream_fingerprint,
        epoch_id="ep1",
        missing_levels=[0],
        level_rdp_cost=compute_level_rdp_vector(10.0, 30.0, DEFAULT_RDP_ORDERS),
        owner_id="worker_2",
        lease_timeout=300.0,
    )
    assert res2[0]["generation"] == 2

    # Worker 1 wakes up and attempts to commit with stale generation 1 -> MUST be rejected!
    with pytest.raises(ReservationLostError, match="Stale worker cannot commit"):
        state_mgr.commit_continual_level_exposures(res1)


def test_stale_worker_cannot_commit_exposure(tmp_path: Path):
    """Stale worker token or generation mismatch raises ReservationLostError."""
    db_file = tmp_path / "crash_e3.db"
    state_mgr = PrivacyStateManager(db_path=db_file)

    stream_cfg = ContinualStreamConfig(stream_id="s_tok", epoch_id="ep1", epoch_days=30)
    res = state_mgr.reserve_continual_level_exposures(
        budget_scope_id="global_external",
        stream_fingerprint=stream_cfg.stream_fingerprint,
        epoch_id="ep1",
        missing_levels=[0],
        level_rdp_cost=compute_level_rdp_vector(10.0, 30.0, DEFAULT_RDP_ORDERS),
        owner_id="w1",
    )

    tampered_res = list(res)
    tampered_res[0] = dict(tampered_res[0])
    tampered_res[0]["fencing_token"] = "forged_fence_token"

    with pytest.raises(ReservationLostError):
        state_mgr.commit_continual_level_exposures(tampered_res)


def test_failed_node_materialization_aborts_or_recovers_exposure(tmp_path: Path):
    """If node materialization fails during query, reserved exposures are aborted."""
    db_file = tmp_path / "crash_e4.db"
    state_mgr = ContinualStateManager(db_path=db_file)
    engine = ContinualQueryEngine(state_manager=state_mgr)

    cfg = ContinualStreamConfig(stream_id="s_err", epoch_id="ep1", epoch_days=10, dataset_fingerprint="dfp", public_roster_fingerprint="rfp")

    # Force error during materialization by raising in materializer
    with pytest.raises(ValueError, match="Cannot materialize"):
        engine.query_interval(
            cfg, 1, 2, recipient="UserA", interactions=None, public_user_roster=None, finalized_up_to_day=10, current_wall_clock_day=15
        )

    # Exposed levels in DB must be empty (aborted)
    exposed = state_mgr.get_exposed_levels("global_external", cfg.stream_fingerprint, "ep1")
    assert len(exposed) == 0


# =========================================================================
# SECTION F: PRODUCTION CONFIG INTEGRATION
# =========================================================================

def test_continual_reads_global_delta_from_app_config():
    """ContinualQueryEngine initialized from AppConfig reads global_delta from accountant."""
    app_cfg = load_app_config("configs")
    engine = ContinualQueryEngine(state_manager=ContinualStateManager(), app_config=app_cfg)
    assert engine.delta_global == float(app_cfg.privacy.accountant.delta_global)


def test_continual_reads_epsilon_cap_from_app_config():
    """ContinualQueryEngine initialized from AppConfig reads epsilon_cap from accountant."""
    app_cfg = load_app_config("configs")
    engine = ContinualQueryEngine(state_manager=ContinualStateManager(), app_config=app_cfg)
    assert engine.epsilon_cap == float(app_cfg.privacy.accountant.epsilon_cap)


def test_continual_basic_accountant_fails_closed():
    """AppConfig with accountant.mode != 'rdp' fails closed on ContinualQueryEngine initialization."""
    app_cfg = load_app_config("configs")
    app_cfg.privacy.accountant.mode = "basic"
    with pytest.raises(PrivacyConfigurationError, match="strictly requires accountant mode 'rdp'"):
        ContinualQueryEngine(state_manager=ContinualStateManager(), app_config=app_cfg)


def test_continual_disabled_by_default(tmp_path: Path):
    """By default in configs/privacy.yaml, privacy.continual.enabled is False and blocks execution."""
    app_cfg = load_app_config("configs")
    assert app_cfg.privacy.continual.enabled is False

    engine = ContinualQueryEngine.from_app_config(app_cfg, state_manager=ContinualStateManager(db_path=tmp_path / "dis.db"))
    cfg = ContinualStreamConfig(stream_id="s_dis", epoch_id="ep1", epoch_days=10)

    with pytest.raises(PrivacyConfigurationError, match="disabled in configuration"):
        engine.query_interval(cfg, 1, 2, recipient="Auditor")


# =========================================================================
# SECTION G: CROSS-EPOCH ATOMICITY
# =========================================================================

def test_cross_epoch_budget_failure_has_no_partial_commit(tmp_path: Path):
    """If second epoch in a cross-epoch query exceeds budget, first epoch is NOT committed."""
    db_file = tmp_path / "cross_g1.db"
    state_mgr = PrivacyStateManager(db_path=db_file)
    cont_mgr = ContinualStateManager(db_path=db_file, state_manager=state_mgr)
    engine = ContinualQueryEngine(state_manager=cont_mgr, epsilon_cap=2.0)

    cfg1 = ContinualStreamConfig(stream_id="s1", epoch_id="ep1", epoch_days=30, epoch_group_l12_bound=20.0)
    cfg2 = ContinualStreamConfig(stream_id="s2", epoch_id="ep2", epoch_days=30, epoch_group_l12_bound=20.0)

    df, roster = _sample_data()

    # Pre-consume some budget so that epoch 1 alone could fit, but epoch 1 + epoch 2 exceeds cap
    cand_rdp = compute_gaussian_rdp(sigma=45.0, sensitivity=10.0, orders=DEFAULT_RDP_ORDERS)
    r = state_mgr.reserve_or_get_release(
        release_key="std_pre",
        canonical_semantics={"metric": "vol"},
        global_budget=PrivacyBudget(epsilon=2.0, delta=1e-6),
        budget_needed=PrivacyBudget(epsilon=1.23, delta=1e-6),
        plan_metadata={},
        accountant_mode="rdp",
        rdp_orders=DEFAULT_RDP_ORDERS,
        rdp_candidate_cost=cand_rdp,
        epsilon_cap=2.0,
        delta_global=1e-6,
    )
    assert r.status == "RESERVED"

    # Cross epoch query attempting both epoch 1 and epoch 2
    with pytest.raises(PrivacyBudgetExceededError):
        engine.query_cross_epoch(
            epoch_1_config=cfg1,
            epoch_1_start_day=20,
            epoch_2_config=cfg2,
            epoch_2_end_day=10,
            recipient="Auditor",
            interactions_epoch_1=df,
            interactions_epoch_2=df,
            public_user_roster=roster,
        )

    # Confirm ZERO exposures committed for epoch 1!
    exposed_1 = cont_mgr.get_exposed_levels("global_external", cfg1.stream_fingerprint, "ep1")
    exposed_2 = cont_mgr.get_exposed_levels("global_external", cfg2.stream_fingerprint, "ep2")
    assert len(exposed_1) == 0
    assert len(exposed_2) == 0


# =========================================================================
# SECTION K: ADVERSARIAL HARDENING
# =========================================================================

def test_adversarial_standard_tree_standard_three_phase_composition(tmp_path: Path):
    """Three-phase lifetime composition: Standard -> Continual Tree -> Standard."""
    db_file = tmp_path / "adv_k1.db"
    state_mgr = PrivacyStateManager(db_path=db_file)
    cont_mgr = ContinualStateManager(db_path=db_file, state_manager=state_mgr)

    # Phase 1: Standard release
    cand1 = compute_gaussian_rdp(sigma=25.0, sensitivity=10.0, orders=DEFAULT_RDP_ORDERS)
    r1 = state_mgr.reserve_or_get_release(
        release_key="p1",
        canonical_semantics={"m": "v"},
        global_budget=PrivacyBudget(epsilon=4.0, delta=1e-6),
        budget_needed=PrivacyBudget(epsilon=0.8, delta=1e-6),
        plan_metadata={},
        accountant_mode="rdp",
        rdp_orders=DEFAULT_RDP_ORDERS,
        rdp_candidate_cost=cand1,
        epsilon_cap=4.0,
        delta_global=1e-6,
    )
    state_mgr.commit_release(
        reservation_id=r1.reservation_id,
        release_key="p1",
        release_id=r1.release_id,
        release_result=DPReleaseResult("v", np.array([1.0]), {}, 0.8, 1e-6, "gaussian", 25.0, 10.0, ["c"], 10),
        recipient="all",
        dataset_version="v1",
        generation=r1.generation,
        owner_id=r1.owner_id,
    )

    # Phase 2: Continual Tree query
    stream_cfg = ContinualStreamConfig(stream_id="s_k1", epoch_id="ep1", epoch_days=30, epoch_group_l12_bound=10.0)
    cont_mgr.register_stream(stream_cfg)
    res_c = cont_mgr.reserve_continual_level_exposures(
        budget_scope_id="global_external",
        stream_config=stream_cfg,
        missing_levels=[0, 1],
        sigma=30.0,
        epsilon_cap=4.0,
        delta_global=1e-6,
    )
    assert len(res_c) == 2
    cont_mgr.commit_continual_level_exposures(res_c)

    # Phase 3: Standard release attempts large spend -> MUST count both Phase 1 and Phase 2
    large_cand = compute_gaussian_rdp(sigma=6.0, sensitivity=20.0, orders=DEFAULT_RDP_ORDERS)
    with pytest.raises(PrivacyBudgetExceededError):
        state_mgr.reserve_or_get_release(
            release_key="p3_over",
            canonical_semantics={"m": "v"},
            global_budget=PrivacyBudget(epsilon=4.0, delta=1e-6),
            budget_needed=PrivacyBudget(epsilon=3.0, delta=1e-6),
            plan_metadata={},
            accountant_mode="rdp",
            rdp_orders=DEFAULT_RDP_ORDERS,
            rdp_candidate_cost=large_cand,
            epsilon_cap=4.0,
            delta_global=1e-6,
        )


def test_adversarial_tree_pending_race_cap(tmp_path: Path):
    """Two concurrent tree queries with RESERVED exposures racing against epsilon_cap."""
    db_file = tmp_path / "adv_k2.db"
    state_mgr = PrivacyStateManager(db_path=db_file)
    cont_mgr = ContinualStateManager(db_path=db_file, state_manager=state_mgr)

    stream_cfg = ContinualStreamConfig(stream_id="s_k2", epoch_id="ep1", epoch_days=30, epoch_group_l12_bound=20.0)
    cont_mgr.register_stream(stream_cfg)

    # Query 1 reserves levels 0, 1 (~3.13 eps)
    r1 = cont_mgr.reserve_continual_level_exposures(
        budget_scope_id="global_external",
        stream_config=stream_cfg,
        missing_levels=[0, 1],
        sigma=50.0,
        epsilon_cap=3.5,
        delta_global=1e-6,
    )
    assert len(r1) == 2

    # Query 2 simultaneously reserves levels 2, 3 -> rejected because Query 1 is RESERVED
    with pytest.raises(PrivacyBudgetExceededError):
        cont_mgr.reserve_continual_level_exposures(
            budget_scope_id="global_external",
            stream_config=stream_cfg,
            missing_levels=[2, 3],
            sigma=50.0,
            epsilon_cap=3.5,
            delta_global=1e-6,
        )


def test_adversarial_standard_pending_tree_pending_race_cap(tmp_path: Path):
    """Standard PENDING + Tree RESERVED race: second applicant must fail closed."""
    db_file = tmp_path / "adv_k3.db"
    state_mgr = PrivacyStateManager(db_path=db_file)
    cont_mgr = ContinualStateManager(db_path=db_file, state_manager=state_mgr)

    # Standard takes PENDING reservation (~2.3 eps)
    cand = compute_gaussian_rdp(sigma=35.0, sensitivity=15.0, orders=DEFAULT_RDP_ORDERS)
    r = state_mgr.reserve_or_get_release(
        release_key="k3_std",
        canonical_semantics={"m": "v"},
        global_budget=PrivacyBudget(epsilon=3.0, delta=1e-6),
        budget_needed=PrivacyBudget(epsilon=2.3, delta=1e-6),
        plan_metadata={},
        accountant_mode="rdp",
        rdp_orders=DEFAULT_RDP_ORDERS,
        rdp_candidate_cost=cand,
        epsilon_cap=3.0,
        delta_global=1e-6,
    )
    assert r.status == "RESERVED"

    # Tree tries to reserve level with cost ~2.3 eps -> combined > 3.0 cap
    stream_cfg = ContinualStreamConfig(stream_id="s_k3", epoch_id="ep1", epoch_days=30, epoch_group_l12_bound=15.0)
    cont_mgr.register_stream(stream_cfg)

    with pytest.raises(PrivacyBudgetExceededError):
        cont_mgr.reserve_continual_level_exposures(
            budget_scope_id="global_external",
            stream_config=stream_cfg,
            missing_levels=[0],
            sigma=35.0,
            epsilon_cap=3.0,
            delta_global=1e-6,
        )


def test_adversarial_global_history_per_recipient_attack(tmp_path: Path):
    """Attacker attempts per_recipient tree attack after GLOBAL_EXTERNAL history."""
    db_file = tmp_path / "adv_k4.db"
    state_mgr = PrivacyStateManager(db_path=db_file)
    cont_mgr = ContinualStateManager(db_path=db_file, state_manager=state_mgr)

    # Bind global_external
    stream_cfg = ContinualStreamConfig(stream_id="s_k4", epoch_id="ep1", epoch_days=30, epoch_group_l12_bound=10.0)
    cont_mgr.register_stream(stream_cfg)
    r = cont_mgr.reserve_continual_level_exposures(
        budget_scope_id="global_external",
        stream_config=stream_cfg,
        missing_levels=[0],
        sigma=30.0,
        policy_mode="global_external",
    )
    cont_mgr.commit_continual_level_exposures(r)

    # Attempt to bypass global budget by claiming per_recipient mode
    with pytest.raises(ProductionSafetyConfigurationError, match="Recipient composition policy mismatch"):
        cont_mgr.reserve_continual_level_exposures(
            budget_scope_id="recipient:attacker",
            stream_config=stream_cfg,
            missing_levels=[1],
            sigma=30.0,
            policy_mode="per_recipient",
        )


def test_adversarial_same_level_mixed_dataset_attack(tmp_path: Path):
    """Reject attempts to materialize nodes at the same level from different dataset versions."""
    db_file = tmp_path / "adv_k5.db"
    cont_mgr = ContinualStateManager(db_path=db_file)
    cfg = ContinualStreamConfig(stream_id="s_k5", epoch_id="ep1", epoch_days=30)

    # Materialize node 0 with dataset hash A
    cont_mgr.get_or_create_noisy_node(
        stream_config=cfg,
        level=0,
        node_index=0,
        sigma=5.0,
        materialize_fn=lambda: np.array([1.0]),
        manifest_fingerprint="ds_A",
        public_roster_fingerprint="ros_A",
        finalized_up_to_day=30,
    )

    # Materialize node 1 with dataset hash B -> rejected!
    with pytest.raises(ProductionSafetyConfigurationError, match="DATASET_MANIFEST_MISMATCH"):
        cont_mgr.get_or_create_noisy_node(
            stream_config=cfg,
            level=0,
            node_index=1,
            sigma=5.0,
            materialize_fn=lambda: np.array([1.0]),
            manifest_fingerprint="ds_B",
            public_roster_fingerprint="ros_A",
            finalized_up_to_day=30,
        )


def test_adversarial_crash_after_exposure_reserve(tmp_path: Path):
    """Crash after reservation leaves DB consistent; budget is restored if aborted."""
    db_file = tmp_path / "adv_k6.db"
    state_mgr = PrivacyStateManager(db_path=db_file)
    cont_mgr = ContinualStateManager(db_path=db_file, state_manager=state_mgr)

    stream_cfg = ContinualStreamConfig(stream_id="s_k6", epoch_id="ep1", epoch_days=30, epoch_group_l12_bound=10.0)
    cont_mgr.register_stream(stream_cfg)

    # Reserve almost all budget
    res = cont_mgr.reserve_continual_level_exposures(
        budget_scope_id="global_external",
        stream_config=stream_cfg,
        missing_levels=[0, 1, 2],
        sigma=40.0,
        epsilon_cap=3.0,
        delta_global=1e-6,
    )
    assert len(res) == 3

    # Abort reservations (e.g. crash cleanup)
    cont_mgr.abort_continual_level_exposures(res)

    # Budget is now freed! New reservation succeeds
    res_retry = cont_mgr.reserve_continual_level_exposures(
        budget_scope_id="global_external",
        stream_config=stream_cfg,
        missing_levels=[0, 1, 2],
        sigma=40.0,
        epsilon_cap=3.0,
        delta_global=1e-6,
    )
    assert len(res_retry) == 3


def test_adversarial_cross_epoch_second_half_budget_failure(tmp_path: Path):
    """Ensure cross-epoch failure leaves no orphaned commits in first epoch."""
    db_file = tmp_path / "adv_k7.db"
    state_mgr = PrivacyStateManager(db_path=db_file)
    cont_mgr = ContinualStateManager(db_path=db_file, state_manager=state_mgr)
    engine = ContinualQueryEngine(state_manager=cont_mgr, epsilon_cap=1.5)

    cfg1 = ContinualStreamConfig(stream_id="s1", epoch_id="ep1", epoch_days=30, epoch_group_l12_bound=20.0)
    cfg2 = ContinualStreamConfig(stream_id="s2", epoch_id="ep2", epoch_days=30, epoch_group_l12_bound=20.0)

    df, roster = _sample_data()

    # Query requires multiple levels in both epochs exceeding 1.5 epsilon cap
    with pytest.raises(PrivacyBudgetExceededError):
        engine.query_cross_epoch(
            epoch_1_config=cfg1,
            epoch_1_start_day=1,
            epoch_2_config=cfg2,
            epoch_2_end_day=30,
            recipient="Auditor",
            interactions_epoch_1=df,
            interactions_epoch_2=df,
            public_user_roster=roster,
        )

    # Confirm zero level exposures committed in either epoch
    assert len(cont_mgr.get_exposed_levels("global_external", cfg1.stream_fingerprint, "ep1")) == 0
    assert len(cont_mgr.get_exposed_levels("global_external", cfg2.stream_fingerprint, "ep2")) == 0


def test_adversarial_different_authoritative_rdp_order_grids(tmp_path: Path):
    """Candidate with alien orders [1.5, 3.0] against authoritative 12 orders is rejected."""
    db_file = tmp_path / "adv_k8.db"
    state_mgr = PrivacyStateManager(db_path=db_file)

    with pytest.raises(PrivacyConfigurationError, match="mismatch authoritative order grid"):
        state_mgr.reserve_continual_level_exposures(
            budget_scope_id="global_external",
            stream_fingerprint="fp1",
            epoch_id="ep1",
            missing_levels=[0],
            level_rdp_cost={Decimal("1.5"): Decimal("0.1"), Decimal("3.0"): Decimal("0.2")},
            authoritative_orders=DEFAULT_RDP_ORDERS,
        )


# =========================================================================
# SECTION L: FINAL PROGRESSIVE FINALIZATION & AUTHORITATIVE ROLE CLOSURE
# =========================================================================

def _generate_day_records(users: Sequence[str], day: int, base_date: str = "2026-01-01", count_per_user: int = 2) -> List[Dict[str, Any]]:
    dt_base = pd.to_datetime(base_date) + pd.Timedelta(days=day - 1)
    date_str = dt_base.strftime("%Y-%m-%d")
    records = []
    for u in users:
        for c in range(count_per_user):
            records.append({
                "user_id": u,
                "timestamp": f"{date_str} 10:{c:02d}:00",
                "provider": "provider_a",
                "task": "technical",
            })
    return records


def test_day10_query_does_not_seal_days_11_to_90(tmp_path: Path):
    """Querying at finalized frontier 10 must NOT record partition fingerprints for days 11..90."""
    db_file = tmp_path / "day10_no_future.db"
    cont_mgr = ContinualStateManager(db_path=db_file)
    engine = ContinualQueryEngine(state_manager=cont_mgr)

    roster = [f"u_{i}" for i in range(5)]
    records = []
    for d in range(1, 11):
        records.extend(_generate_day_records(roster, d))
    df = pd.DataFrame(records)

    cfg = ContinualStreamConfig(
        stream_id="stream_d10",
        epoch_id="ep1",
        epoch_start="2026-01-01",
        epoch_days=90,
        epoch_group_l12_bound=20.0,
    )

    res = engine.query_interval(
        cfg,
        start_day=1,
        end_day=4,
        recipient="Analyst",
        interactions=df,
        public_user_roster=roster,
        finalized_up_to_day=10,
        current_wall_clock_day=12,
    )
    assert res is not None

    with cont_mgr._connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT day FROM continual_epoch_partitions WHERE epoch_id = 'ep1'")
        sealed_days = [row[0] for row in cursor.fetchall()]

    assert len(sealed_days) > 0
    assert all(d <= 10 for d in sealed_days)
    assert not any(d > 10 for d in sealed_days), f"Security violation: future unfinalized days sealed: {sealed_days}"


def test_real_progressive_engine_query_day10_then_day11(tmp_path: Path):
    """Engine query at Day 10 followed by Day 11 data growth and frontier advancement succeeds seamlessly."""
    db_file = tmp_path / "prog_10_11.db"
    cont_mgr = ContinualStateManager(db_path=db_file)
    engine = ContinualQueryEngine(state_manager=cont_mgr)

    roster = [f"u_{i}" for i in range(5)]
    cfg = ContinualStreamConfig(
        stream_id="stream_prog",
        epoch_id="ep1",
        epoch_start="2026-01-01",
        epoch_days=90,
        epoch_group_l12_bound=20.0,
    )

    # Step 1: Day 1..10
    records_10 = []
    for d in range(1, 11):
        records_10.extend(_generate_day_records(roster, d))
    df_10 = pd.DataFrame(records_10)

    res1 = engine.query_interval(
        cfg,
        start_day=1,
        end_day=4,
        recipient="Analyst",
        interactions=df_10,
        public_user_roster=roster,
        finalized_up_to_day=10,
        current_wall_clock_day=12,
    )
    assert res1 is not None

    # Step 2: Add real Day 11 interactions, advance frontier to 11
    records_11 = list(records_10)
    records_11.extend(_generate_day_records(roster, 11))
    df_11 = pd.DataFrame(records_11)

    res2 = engine.query_interval(
        cfg,
        start_day=1,
        end_day=11,
        recipient="Analyst",
        interactions=df_11,
        public_user_roster=roster,
        finalized_up_to_day=11,
        current_wall_clock_day=13,
    )
    assert res2 is not None


def test_future_unfinalized_partition_can_change_before_finalization(tmp_path: Path):
    """Day 11 telemetry can change while frontier is 10; once Day 11 is finalized, further changes fail closed."""
    db_file = tmp_path / "fut_change.db"
    cont_mgr = ContinualStateManager(db_path=db_file)
    engine = ContinualQueryEngine(state_manager=cont_mgr)

    roster = [f"u_{i}" for i in range(5)]
    cfg = ContinualStreamConfig(
        stream_id="stream_fut",
        epoch_id="ep1",
        epoch_start="2026-01-01",
        epoch_days=30,
        epoch_group_l12_bound=20.0,
    )

    # Day 1..10 + draft Day 11 v1
    records = []
    for d in range(1, 11):
        records.extend(_generate_day_records(roster, d))
    records_v1 = list(records) + _generate_day_records(roster, 11, count_per_user=1)
    df_v1 = pd.DataFrame(records_v1)

    # Query with frontier=10
    engine.query_interval(
        cfg, 1, 4, recipient="Analyst", interactions=df_v1,
        public_user_roster=roster, finalized_up_to_day=10, current_wall_clock_day=12,
    )

    # Change Day 11 draft data (v2) while frontier is still 10: MUST be allowed!
    records_v2 = list(records) + _generate_day_records(roster, 11, count_per_user=5)
    df_v2 = pd.DataFrame(records_v2)
    engine.query_interval(
        cfg, 1, 4, recipient="Analyst", interactions=df_v2,
        public_user_roster=roster, finalized_up_to_day=10, current_wall_clock_day=12,
    )

    # Now finalize Day 11 with v2: seals Day 11
    engine.query_interval(
        cfg, 1, 11, recipient="Analyst", interactions=df_v2,
        public_user_roster=roster, finalized_up_to_day=11, current_wall_clock_day=13,
    )

    # Mutate Day 11 after finalization (v3): MUST fail closed
    records_v3 = list(records) + _generate_day_records(roster, 11, count_per_user=9)
    df_v3 = pd.DataFrame(records_v3)
    with pytest.raises(ProductionSafetyConfigurationError, match="cannot be mutated"):
        engine.query_interval(
            cfg, 1, 11, recipient="Analyst", interactions=df_v3,
            public_user_roster=roster, finalized_up_to_day=11, current_wall_clock_day=13,
        )


def test_progressive_data_growth_does_not_change_epoch_manifest_identity(tmp_path: Path):
    """Epoch manifest identity remains stable and identical as data grows progressively."""
    db_file = tmp_path / "stable_man.db"
    cont_mgr = ContinualStateManager(db_path=db_file)
    engine = ContinualQueryEngine(state_manager=cont_mgr)

    roster = [f"u_{i}" for i in range(5)]
    cfg = ContinualStreamConfig(
        stream_id="stream_stab",
        epoch_id="ep1",
        epoch_start="2026-01-01",
        epoch_days=60,
        epoch_group_l12_bound=20.0,
    )

    records = []
    for d in range(1, 11):
        records.extend(_generate_day_records(roster, d))
    df_10 = pd.DataFrame(records)

    engine.query_interval(
        cfg, 1, 4, recipient="Analyst", interactions=df_10,
        public_user_roster=roster, finalized_up_to_day=10, current_wall_clock_day=12,
    )

    with cont_mgr._connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT manifest_fingerprint FROM continual_epoch_manifests WHERE epoch_id = 'ep1'")
        mf_initial = cursor.fetchone()[0]

    # Data grows to Day 15
    for d in range(11, 16):
        records.extend(_generate_day_records(roster, d))
    df_15 = pd.DataFrame(records)

    engine.query_interval(
        cfg, 1, 8, recipient="Analyst", interactions=df_15,
        public_user_roster=roster, finalized_up_to_day=15, current_wall_clock_day=17,
    )

    with cont_mgr._connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT manifest_fingerprint FROM continual_epoch_manifests WHERE epoch_id = 'ep1'")
        mf_after = cursor.fetchone()[0]

    assert mf_initial == mf_after, "Epoch manifest identity must not change during progressive data growth."


def test_whole_dataset_growth_is_valid_when_new_partition_is_not_previously_finalized(tmp_path: Path):
    """Growing the dataset and advancing frontier is valid when new partitions were not previously finalized."""
    db_file = tmp_path / "growth_valid.db"
    cont_mgr = ContinualStateManager(db_path=db_file)
    engine = ContinualQueryEngine(state_manager=cont_mgr)

    roster = [f"u_{i}" for i in range(5)]
    cfg = ContinualStreamConfig(
        stream_id="stream_grow",
        epoch_id="ep1",
        epoch_start="2026-01-01",
        epoch_days=30,
        epoch_group_l12_bound=20.0,
    )

    records = []
    for d in range(1, 6):
        records.extend(_generate_day_records(roster, d))
    df_5 = pd.DataFrame(records)

    engine.query_interval(
        cfg, 1, 4, recipient="Analyst", interactions=df_5,
        public_user_roster=roster, finalized_up_to_day=5, current_wall_clock_day=7,
    )

    for d in range(6, 9):
        records.extend(_generate_day_records(roster, d))
    df_8 = pd.DataFrame(records)

    res = engine.query_interval(
        cfg, 1, 8, recipient="Analyst", interactions=df_8,
        public_user_roster=roster, finalized_up_to_day=8, current_wall_clock_day=10,
    )
    assert res is not None


def test_configured_external_cannot_claim_trusted(tmp_path: Path):
    """Recipient configured as external cannot claim trusted role to bypass privacy accounting."""
    db_file = tmp_path / "role_sec1.db"
    cont_mgr = ContinualStateManager(db_path=db_file)
    engine = ContinualQueryEngine(
        state_manager=cont_mgr,
        recipient_roles={"ProviderA": "external"},
    )

    with pytest.raises(ProductionSafetyConfigurationError, match="Recipient role override conflicts with authoritative configuration"):
        engine.resolve_scope("ProviderA", recipient_role="trusted")


def test_configured_trusted_role_used_when_role_argument_omitted(tmp_path: Path):
    """When recipient_role is omitted, configured trusted role is authoritatively used."""
    db_file = tmp_path / "role_sec2.db"
    cont_mgr = ContinualStateManager(db_path=db_file)
    engine = ContinualQueryEngine(
        state_manager=cont_mgr,
        recipient_roles={"InternalAuditor": "trusted"},
    )

    scope = engine.resolve_scope("InternalAuditor")
    assert scope == "internal"


def test_configured_external_role_used_when_role_argument_omitted(tmp_path: Path):
    """When recipient_role is omitted, configured external role is authoritatively used."""
    db_file = tmp_path / "role_sec3.db"
    cont_mgr = ContinualStateManager(db_path=db_file)
    engine = ContinualQueryEngine(
        state_manager=cont_mgr,
        composition_mode="per_recipient",
        recipient_roles={"ProviderA": "external"},
    )

    scope = engine.resolve_scope("ProviderA")
    assert scope == "recipient:ProviderA"


def test_recipient_role_override_cannot_change_budget_scope(tmp_path: Path):
    """Adversarial attempt to change budget scope via recipient_role parameter is rejected."""
    db_file = tmp_path / "role_sec4.db"
    cont_mgr = ContinualStateManager(db_path=db_file)
    engine = ContinualQueryEngine(
        state_manager=cont_mgr,
        recipient_roles={"PartnerX": "external"},
    )

    with pytest.raises(ProductionSafetyConfigurationError):
        engine.resolve_scope("PartnerX", recipient_role="trusted")


def test_external_to_internal_scope_bypass_rejected(tmp_path: Path):
    """External to internal scope bypass attempt is rejected before reservation or node disclosure."""
    db_file = tmp_path / "role_sec5.db"
    cont_mgr = ContinualStateManager(db_path=db_file)
    engine = ContinualQueryEngine(
        state_manager=cont_mgr,
        recipient_roles={"ProviderA": "external"},
    )

    roster = ["u_0"]
    df = pd.DataFrame([{"user_id": "u_0", "timestamp": "2026-01-01 10:00:00", "provider": "provider_a", "task": "technical"}])
    cfg = ContinualStreamConfig(stream_id="s_sec", epoch_id="ep1", epoch_days=30, epoch_start="2026-01-01")

    with pytest.raises(ProductionSafetyConfigurationError):
        engine.query_interval(
            cfg, 1, 2, recipient="ProviderA", interactions=df,
            public_user_roster=roster, recipient_role="trusted", finalized_up_to_day=10, current_wall_clock_day=12,
        )

    # Verify zero level exposures occurred
    exposed = cont_mgr.get_exposed_levels("internal", cfg.stream_fingerprint, "ep1")
    assert len(exposed) == 0


def test_real_daily_continual_progression(tmp_path: Path):
    """End-to-end integration test of real daily continual progression across days 1..13."""
    db_file = tmp_path / "real_daily_prog.db"
    cont_mgr = ContinualStateManager(db_path=db_file)
    engine = ContinualQueryEngine(state_manager=cont_mgr, composition_mode="per_recipient")

    roster = [f"u_{i}" for i in range(5)]
    cfg = ContinualStreamConfig(
        stream_id="stream_daily",
        epoch_id="epoch_jan",
        epoch_start="2026-01-01",
        epoch_days=30,
        epoch_group_l12_bound=20.0,
    )

    # Day 1..10: interactions through Day 10, finalized frontier 10
    records = []
    for d in range(1, 11):
        records.extend(_generate_day_records(roster, d))
    df = pd.DataFrame(records)

    res_10 = engine.query_prefix(
        cfg, end_day=10, recipient="AnalystA", interactions=df,
        public_user_roster=roster, finalized_up_to_day=10, current_wall_clock_day=12,
    )
    assert res_10 is not None

    # Day 11: append Day 11 interactions, frontier advances to 11
    records.extend(_generate_day_records(roster, 11))
    df = pd.DataFrame(records)
    res_11 = engine.query_sliding_window(
        cfg, end_day=11, window_days=7, recipient="AnalystA", interactions=df,
        public_user_roster=roster, finalized_up_to_day=11, current_wall_clock_day=13,
    )
    assert res_11 is not None

    # Day 12: append Day 12 interactions, frontier advances to 12
    records.extend(_generate_day_records(roster, 12))
    df = pd.DataFrame(records)
    res_12 = engine.query_interval(
        cfg, start_day=1, end_day=12, recipient="AnalystA", interactions=df,
        public_user_roster=roster, finalized_up_to_day=12, current_wall_clock_day=14,
    )
    assert res_12 is not None

    # Then mutate finalized Day 10: MUST fail closed!
    tampered_records = list(records)
    tampered_records.append({
        "user_id": roster[0],
        "timestamp": "2026-01-10 23:59:59",
        "provider": "provider_a",
        "task": "technical",
    })
    df_tampered_10 = pd.DataFrame(tampered_records)
    with pytest.raises(ProductionSafetyConfigurationError, match="already finalized partition for day 10 cannot be mutated"):
        engine.query_interval(
            cfg, start_day=1, end_day=12, recipient="AnalystA", interactions=df_tampered_10,
            public_user_roster=roster, finalized_up_to_day=12, current_wall_clock_day=14,
        )

    # Mutate not-yet-finalized Day 13 before frontier advances: MUST be allowed
    draft_13_records = list(records) + _generate_day_records(roster, 13, count_per_user=1)
    df_draft_13 = pd.DataFrame(draft_13_records)
    # Query with frontier 12 succeeds
    res_draft = engine.query_interval(
        cfg, start_day=1, end_day=12, recipient="AnalystA", interactions=df_draft_13,
        public_user_roster=roster, finalized_up_to_day=12, current_wall_clock_day=14,
    )
    assert res_draft is not None

    # Finalize Day 13: fingerprint sealed
    real_13_records = list(records) + _generate_day_records(roster, 13, count_per_user=3)
    df_final_13 = pd.DataFrame(real_13_records)
    res_final_13 = engine.query_interval(
        cfg, start_day=1, end_day=13, recipient="AnalystA", interactions=df_final_13,
        public_user_roster=roster, finalized_up_to_day=13, current_wall_clock_day=15,
    )
    assert res_final_13 is not None

    # Mutate Day 13 afterward: MUST fail closed
    tampered_13_records = list(real_13_records) + [{
        "user_id": roster[1],
        "timestamp": "2026-01-13 12:34:56",
        "provider": "provider_a",
        "task": "technical",
    }]
    df_tampered_13 = pd.DataFrame(tampered_13_records)
    with pytest.raises(ProductionSafetyConfigurationError, match="already finalized partition for day 13 cannot be mutated"):
        engine.query_interval(
            cfg, start_day=1, end_day=13, recipient="AnalystA", interactions=df_tampered_13,
            public_user_roster=roster, finalized_up_to_day=13, current_wall_clock_day=15,
        )


# =========================================================================
# SECTION M: ONLINE PREFIX-STABLE TEMPORAL CLIPPING & CROSS-EPOCH UNIFICATION
# =========================================================================

def test_online_temporal_clipping_prefix_stability():
    """Prefix-stability invariant: contributions Z[1..t] are identical regardless of future days."""
    roster = [f"user_{i}" for i in range(10)]

    # 1. Day 10 vs Day 11
    records_10 = []
    for d in range(1, 11):
        records_10.extend(_generate_day_records(roster, d, count_per_user=5))
    df_10 = pd.DataFrame(records_10)

    records_11 = list(records_10)
    records_11.extend(_generate_day_records(roster, 11, count_per_user=5))
    df_11 = pd.DataFrame(records_11)

    Z10, _, _ = build_temporal_user_contributions(
        interactions=df_10,
        public_user_roster=roster,
        epoch_start="2026-01-01",
        epoch_days=30,
        daily_l2_bound=10.0,
        epoch_l12_bound=40.0,
    )
    Z11, _, _ = build_temporal_user_contributions(
        interactions=df_11,
        public_user_roster=roster,
        epoch_start="2026-01-01",
        epoch_days=30,
        daily_l2_bound=10.0,
        epoch_l12_bound=40.0,
    )
    np.testing.assert_allclose(
        Z10[:, :10, :], Z11[:, :10, :], atol=1e-12,
        err_msg="Prefix stability violation: Day 1..10 contributions changed when Day 11 arrived!"
    )

    # 2. Day 20 vs Day 30
    records_20 = list(records_11)
    for d in range(12, 21):
        records_20.extend(_generate_day_records(roster, d, count_per_user=5))
    df_20 = pd.DataFrame(records_20)

    records_30 = list(records_20)
    for d in range(21, 31):
        records_30.extend(_generate_day_records(roster, d, count_per_user=5))
    df_30 = pd.DataFrame(records_30)

    Z20, _, _ = build_temporal_user_contributions(
        interactions=df_20,
        public_user_roster=roster,
        epoch_start="2026-01-01",
        epoch_days=30,
        daily_l2_bound=10.0,
        epoch_l12_bound=40.0,
    )
    Z30, _, _ = build_temporal_user_contributions(
        interactions=df_30,
        public_user_roster=roster,
        epoch_start="2026-01-01",
        epoch_days=30,
        daily_l2_bound=10.0,
        epoch_l12_bound=40.0,
    )
    np.testing.assert_allclose(
        Z20[:, :20, :], Z30[:, :20, :], atol=1e-12,
        err_msg="Prefix stability violation: Day 1..20 contributions changed when Days 21..30 arrived!"
    )

    # 3. Day 30 vs Day 90
    records_90 = list(records_30)
    for d in range(31, 91):
        records_90.extend(_generate_day_records(roster, d, count_per_user=5))
    df_90 = pd.DataFrame(records_90)

    Z30_90, _, _ = build_temporal_user_contributions(
        interactions=df_30,
        public_user_roster=roster,
        epoch_start="2026-01-01",
        epoch_days=90,
        daily_l2_bound=10.0,
        epoch_l12_bound=40.0,
    )
    Z90, _, _ = build_temporal_user_contributions(
        interactions=df_90,
        public_user_roster=roster,
        epoch_start="2026-01-01",
        epoch_days=90,
        daily_l2_bound=10.0,
        epoch_l12_bound=40.0,
    )
    np.testing.assert_allclose(
        Z30_90[:, :30, :], Z90[:, :30, :], atol=1e-12,
        err_msg="Prefix stability violation: Day 1..30 contributions changed when Days 31..90 arrived!"
    )


def test_progressive_level_sensitivity_never_exceeds_epoch_bound():
    """Adversarial regression: 1 user, C_day=10, C_epoch=10, norm=10 on Day1 and Day2."""
    roster = ["u_adv"]
    records = []
    for c in range(10):
        records.append({"user_id": "u_adv", "timestamp": "2026-01-01 10:00:00", "provider": "provider_a", "task": "technical"})
    for c in range(10):
        records.append({"user_id": "u_adv", "timestamp": "2026-01-02 10:00:00", "provider": "provider_a", "task": "technical"})
    df = pd.DataFrame(records)

    Z, _, _ = build_temporal_user_contributions(
        interactions=df,
        public_user_roster=roster,
        epoch_start="2026-01-01",
        epoch_days=30,
        granularity="overall_1_cell",
        daily_l2_bound=10.0,
        epoch_l12_bound=10.0,
    )
    day1_norm = float(np.linalg.norm(Z[0, 0, :]))
    day2_norm = float(np.linalg.norm(Z[0, 1, :]))
    assert np.isclose(day1_norm, 10.0, atol=1e-6)
    assert np.isclose(day2_norm, 0.0, atol=1e-6)

    # Invariant: sum_t ||z_{u, t}||_2 <= C_epoch + tolerance
    total_group_l1_l2 = float(np.sum(np.linalg.norm(Z[0, :, :], axis=1)))
    assert total_group_l1_l2 <= 10.0 + 1e-10

    # For every level in a binary tree, verify level sensitivity <= C_epoch
    tree = BinaryTimeTree(epoch_days=30)
    for lvl in range(tree.num_levels):
        block_size = 2 ** lvl
        num_nodes_at_level = tree.tree_size // block_size
        lvl_vecs = []
        for n_idx in range(num_nodes_at_level):
            u, v = tree.node_interval(lvl, n_idx)
            node_vec = np.sum(Z[0, (u-1):min(v, 30), :], axis=0)
            lvl_vecs.append(node_vec)
        concatenated_lvl = np.concatenate(lvl_vecs)
        lvl_l2_norm = float(np.linalg.norm(concatenated_lvl))
        assert lvl_l2_norm <= 10.0 + 1e-10, f"Level {lvl} L2 sensitivity {lvl_l2_norm} exceeds C_epoch=10.0"


def test_materialized_day10_contribution_unchanged_after_day11_arrives(tmp_path: Path):
    """Exact bounded contribution semantics of Day 10 remains identical after Day 11 arrives."""
    db_file = tmp_path / "mat_d10_d11.db"
    cont_mgr = ContinualStateManager(db_path=db_file)
    engine = ContinualQueryEngine(state_manager=cont_mgr)

    roster = [f"u_{i}" for i in range(5)]
    cfg = ContinualStreamConfig(
        stream_id="stream_mat_test",
        epoch_id="ep1",
        epoch_start="2026-01-01",
        epoch_days=30,
        epoch_group_l12_bound=20.0,
    )

    records_10 = []
    for d in range(1, 11):
        records_10.extend(_generate_day_records(roster, d, count_per_user=3))
    df_10 = pd.DataFrame(records_10)

    # Materialize nodes for interval [1, 10]
    res_10 = engine.query_interval(
        cfg, 1, 10, recipient="Analyst", interactions=df_10,
        public_user_roster=roster, finalized_up_to_day=10, current_wall_clock_day=12,
    )

    # Day 11 arrives
    records_11 = list(records_10) + _generate_day_records(roster, 11, count_per_user=3)
    df_11 = pd.DataFrame(records_11)

    # Query with cached nodes for [1, 10] using new df_11
    res_10_after = engine.query_interval(
        cfg, 1, 10, recipient="Analyst", interactions=df_11,
        public_user_roster=roster, finalized_up_to_day=11, current_wall_clock_day=13,
    )
    np.testing.assert_array_equal(res_10.vector, res_10_after.vector)


def test_progressive_nodes_share_one_prefix_stable_contribution_definition(tmp_path: Path):
    """Progressive nodes share a unified prefix-stable contribution builder across all queries."""
    db_file = tmp_path / "shared_builder.db"
    cont_mgr = ContinualStateManager(db_path=db_file)
    engine = ContinualQueryEngine(state_manager=cont_mgr)

    roster = [f"u_{i}" for i in range(4)]
    cfg = ContinualStreamConfig(
        stream_id="stream_shared",
        epoch_id="ep1",
        epoch_start="2026-01-01",
        epoch_days=16,
        epoch_group_l12_bound=15.0,
    )

    records = []
    for d in range(1, 9):
        records.extend(_generate_day_records(roster, d))
    df = pd.DataFrame(records)

    q1 = engine.query_interval(cfg, 1, 4, recipient="A", interactions=df, public_user_roster=roster, finalized_up_to_day=8, current_wall_clock_day=10)
    q2 = engine.query_interval(cfg, 5, 8, recipient="A", interactions=df, public_user_roster=roster, finalized_up_to_day=8, current_wall_clock_day=10)
    q_all = engine.query_interval(cfg, 1, 8, recipient="A", interactions=df, public_user_roster=roster, finalized_up_to_day=8, current_wall_clock_day=10)

    assert q1 is not None and q2 is not None and q_all is not None


def test_cross_epoch_external_cannot_claim_trusted(tmp_path: Path):
    """query_cross_epoch rejects external recipient claiming trusted role."""
    db_file = tmp_path / "cross_role_sec1.db"
    cont_mgr = ContinualStateManager(db_path=db_file)
    engine = ContinualQueryEngine(state_manager=cont_mgr, recipient_roles={"ProviderA": "external"})

    cfg1 = ContinualStreamConfig(stream_id="s1", epoch_id="ep1", epoch_days=30)
    cfg2 = ContinualStreamConfig(stream_id="s2", epoch_id="ep2", epoch_days=30)
    df, roster = _sample_data()

    with pytest.raises(ProductionSafetyConfigurationError, match="Recipient role override conflicts"):
        engine.query_cross_epoch(
            cfg1, 1, cfg2, 10, recipient="ProviderA", recipient_role="trusted",
            interactions_epoch_1=df, interactions_epoch_2=df, public_user_roster=roster,
        )


def test_cross_epoch_configured_trusted_role_used_when_omitted(tmp_path: Path):
    """query_cross_epoch authoritatively applies configured trusted role when argument is omitted."""
    db_file = tmp_path / "cross_role_sec2.db"
    cont_mgr = ContinualStateManager(db_path=db_file)
    engine = ContinualQueryEngine(state_manager=cont_mgr, recipient_roles={"InternalAuditor": "trusted"})

    scope = engine.resolve_scope("InternalAuditor")
    assert scope == "internal"

    cfg1 = ContinualStreamConfig(stream_id="s1", epoch_id="ep1", epoch_days=30)
    cfg2 = ContinualStreamConfig(stream_id="s2", epoch_id="ep2", epoch_days=30)
    df, roster = _sample_data()

    res = engine.query_cross_epoch(
        cfg1, 20, cfg2, 10, recipient="InternalAuditor",
        interactions_epoch_1=df, interactions_epoch_2=df, public_user_roster=roster,
        finalized_up_to_day_epoch_1=30, finalized_up_to_day_epoch_2=30,
    )
    assert res.budget_scope_id == "internal"


def test_cross_epoch_does_not_use_mutable_whole_dataset_manifest(tmp_path: Path):
    """query_cross_epoch uses stable epoch manifest identities, not mutable interaction data hashes."""
    db_file = tmp_path / "cross_stable_man.db"
    cont_mgr = ContinualStateManager(db_path=db_file)
    engine = ContinualQueryEngine(state_manager=cont_mgr)

    roster = [f"u_{i}" for i in range(5)]
    cfg1 = ContinualStreamConfig(stream_id="s1", epoch_id="ep1", epoch_days=30, epoch_start="2026-01-01")
    cfg2 = ContinualStreamConfig(stream_id="s2", epoch_id="ep2", epoch_days=30, epoch_start="2026-02-01")

    records_1 = []
    for d in range(1, 31):
        records_1.extend(_generate_day_records(roster, d))
    df_1 = pd.DataFrame(records_1)

    records_2_5 = []
    for d in range(1, 6):
        records_2_5.extend(_generate_day_records(roster, d, base_date="2026-02-01"))
    df_2_5 = pd.DataFrame(records_2_5)

    res1 = engine.query_cross_epoch(
        cfg1, 25, cfg2, 4, recipient="Analyst",
        interactions_epoch_1=df_1, interactions_epoch_2=df_2_5, public_user_roster=roster,
        finalized_up_to_day_epoch_1=30, finalized_up_to_day_epoch_2=5,
    )
    assert res1 is not None

    with cont_mgr._connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT manifest_fingerprint FROM continual_epoch_manifests WHERE epoch_id = 'ep2'")
        mf2_initial = cursor.fetchone()[0]

    # Data in epoch 2 grows to Day 10
    records_2_10 = list(records_2_5)
    for d in range(6, 11):
        records_2_10.extend(_generate_day_records(roster, d, base_date="2026-02-01"))
    df_2_10 = pd.DataFrame(records_2_10)

    res2 = engine.query_cross_epoch(
        cfg1, 25, cfg2, 8, recipient="Analyst",
        interactions_epoch_1=df_1, interactions_epoch_2=df_2_10, public_user_roster=roster,
        finalized_up_to_day_epoch_1=30, finalized_up_to_day_epoch_2=10,
    )
    assert res2 is not None

    with cont_mgr._connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT manifest_fingerprint FROM continual_epoch_manifests WHERE epoch_id = 'ep2'")
        mf2_after = cursor.fetchone()[0]

    assert mf2_initial == mf2_after, "Epoch 2 manifest token must remain stable across data growth in cross-epoch query."


def test_cross_epoch_progressive_second_epoch_growth_succeeds(tmp_path: Path):
    """Epoch 1 complete and Epoch 2 growing progressively succeeds across cross-epoch queries."""
    db_file = tmp_path / "cross_prog_growth.db"
    cont_mgr = ContinualStateManager(db_path=db_file)
    engine = ContinualQueryEngine(state_manager=cont_mgr)

    roster = [f"u_{i}" for i in range(4)]
    cfg1 = ContinualStreamConfig(stream_id="s1_prog", epoch_id="ep1", epoch_days=30, epoch_start="2026-01-01")
    cfg2 = ContinualStreamConfig(stream_id="s2_prog", epoch_id="ep2", epoch_days=30, epoch_start="2026-02-01")

    df_1 = pd.DataFrame(_generate_day_records(roster, 30))
    df_2_step1 = pd.DataFrame(_generate_day_records(roster, 1, base_date="2026-02-01"))

    r1 = engine.query_cross_epoch(
        cfg1, 30, cfg2, 1, recipient="Analyst",
        interactions_epoch_1=df_1, interactions_epoch_2=df_2_step1, public_user_roster=roster,
        finalized_up_to_day_epoch_1=30, finalized_up_to_day_epoch_2=1,
    )
    assert r1 is not None

    df_2_step2 = pd.DataFrame(
        _generate_day_records(roster, 1, base_date="2026-02-01") +
        _generate_day_records(roster, 2, base_date="2026-02-01")
    )
    r2 = engine.query_cross_epoch(
        cfg1, 30, cfg2, 2, recipient="Analyst",
        interactions_epoch_1=df_1, interactions_epoch_2=df_2_step2, public_user_roster=roster,
        finalized_up_to_day_epoch_1=30, finalized_up_to_day_epoch_2=2,
    )
    assert r2 is not None

