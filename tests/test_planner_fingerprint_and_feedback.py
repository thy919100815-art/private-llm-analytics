"""Tests for Planner Configuration Fingerprint, History Invalidation, and DP Feedback (v0.3)."""

from decimal import Decimal
import pytest

from private_llm_analytics.config.loader import load_app_config
from private_llm_analytics.planning.planner import (
    SmallNReleasePlanner,
    compute_history_fingerprint,
    compute_planner_config_fingerprint,
)
from private_llm_analytics.privacy.composition import PrivacyView


def test_planner_cache_invalidates_on_prior_change():
    """Modifying the public prior changes the config fingerprint and invalidates plan cache."""
    cfg = load_app_config("configs")
    planner = SmallNReleasePlanner(app_config=cfg)

    fp_initial = compute_planner_config_fingerprint(cfg)
    res1 = planner.plan(metric="interaction_volume", recipient="recipient_internal")
    assert res1.planner_config_fingerprint == fp_initial

    # Modify prior
    cfg.privacy.planner.prior.mean = 120.0
    fp_modified = compute_planner_config_fingerprint(cfg)
    assert fp_initial != fp_modified

    # Re-planning uses fresh configuration
    planner2 = SmallNReleasePlanner(app_config=cfg)
    res2 = planner2.plan(metric="interaction_volume", recipient="recipient_internal")
    assert res2.planner_config_fingerprint == fp_modified


def test_planner_cache_invalidates_on_weight_change():
    """Modifying utility weights changes the config fingerprint."""
    cfg = load_app_config("configs")
    fp1 = compute_planner_config_fingerprint(cfg)

    cfg.privacy.planner.weights.clipping = 5.0
    fp2 = compute_planner_config_fingerprint(cfg)
    assert fp1 != fp2


def test_planner_cache_invalidates_on_candidate_change():
    """Modifying candidate bounds or granularities changes the config fingerprint."""
    cfg = load_app_config("configs")
    fp1 = compute_planner_config_fingerprint(cfg)

    cfg.privacy.planner.candidate_clipping = [10.0, 50.0]
    fp2 = compute_planner_config_fingerprint(cfg)
    assert fp1 != fp2


def test_planner_cache_invalidates_on_history_change():
    """Different historical privacy view states produce distinct history fingerprints."""
    orders = [Decimal("2.0"), Decimal("4.0")]
    rdp_a = {Decimal("2.0"): Decimal("0.1"), Decimal("4.0"): Decimal("0.2")}
    rdp_b = {Decimal("2.0"): Decimal("0.5"), Decimal("4.0"): Decimal("1.0")}

    fp_a = compute_history_fingerprint("global_external", 1, rdp_a)
    fp_b = compute_history_fingerprint("global_external", 2, rdp_b)
    assert fp_a != fp_b


def test_dp_feedback_rejects_raw_context():
    """dp_feedback mode strictly rejects non-DP raw or exact data in feedback context."""
    cfg = load_app_config("configs")
    cfg.privacy.planner.mode = "dp_feedback"
    planner = SmallNReleasePlanner(app_config=cfg)

    # Passing raw exact data in context must raise ValueError
    poisoned_context = {"raw_data": [10.0, 20.0, 30.0]}
    with pytest.raises(ValueError, match="strictly rejects non-DP raw data in feedback context"):
        planner.plan(
            metric="interaction_volume",
            recipient="recipient_internal",
            public_context=poisoned_context,
        )


def test_dp_feedback_rejects_pending_release():
    """dp_feedback rejects releases that are not in COMMITTED status."""
    cfg = load_app_config("configs")
    cfg.privacy.planner.mode = "dp_feedback"

    class MockStateWithPendingRelease:
        def __init__(self):
            pass

        def get_privacy_view(self, budget_scope_id, delta_global, orders, epsilon_cap, accountant_mode):
            return PrivacyView(
                budget_scope_id=budget_scope_id,
                release_count=0,
                total_rdp_vector={o: Decimal("0.0") for o in orders},
                effective_epsilon=Decimal("0.0"),
                best_alpha=orders[0],
                delta_global=Decimal(str(delta_global)),
                epsilon_cap=Decimal(str(epsilon_cap)),
                remaining_epsilon=Decimal(str(epsilon_cap)),
                accountant_mode=accountant_mode,
                orders=orders,
            )

        def _connection(self):
            import sqlite3
            conn = sqlite3.connect(":memory:")
            conn.execute(
                "CREATE TABLE release_store (release_id TEXT, status TEXT, budget_scope_id TEXT, release_key TEXT)"
            )
            # Insert a PENDING release (not COMMITTED)
            conn.execute(
                "INSERT INTO release_store VALUES ('rel_pending_1', 'RESERVED', 'global_external', 'k1')"
            )
            return conn

    mock_state = MockStateWithPendingRelease()
    planner = SmallNReleasePlanner(app_config=cfg, state_manager=mock_state)

    res = planner.plan(
        metric="interaction_volume",
        recipient="recipient_internal",
        public_context={"feedback_release_ids": ["rel_pending_1"]},
    )
    # Pending release cannot be accepted -> falls back safely to static_public
    assert res.metadata.get("fallback") == "static_public"
    assert res.metadata.get("feedback_used") is False


def test_dp_feedback_accepts_committed_dp_release():
    """dp_feedback accepts valid COMMITTED differential privacy release IDs."""
    cfg = load_app_config("configs")
    cfg.privacy.planner.mode = "dp_feedback"

    class MockStateWithCommittedRelease:
        def get_privacy_view(self, budget_scope_id, delta_global, orders, epsilon_cap, accountant_mode):
            return PrivacyView(
                budget_scope_id=budget_scope_id,
                release_count=1,
                total_rdp_vector={o: Decimal("0.1") for o in orders},
                effective_epsilon=Decimal("0.5"),
                best_alpha=orders[0],
                delta_global=Decimal(str(delta_global)),
                epsilon_cap=Decimal(str(epsilon_cap)),
                remaining_epsilon=Decimal(str(epsilon_cap)) - Decimal("0.5"),
                accountant_mode=accountant_mode,
                orders=orders,
            )

        def _connection(self):
            import sqlite3
            conn = sqlite3.connect(":memory:")
            conn.execute(
                "CREATE TABLE release_store (release_id TEXT, status TEXT, budget_scope_id TEXT, release_key TEXT)"
            )
            # Insert a COMMITTED release
            conn.execute(
                "INSERT INTO release_store VALUES ('rel_committed_1', 'COMMITTED', 'global_external', 'k1')"
            )
            return conn

    mock_state = MockStateWithCommittedRelease()
    planner = SmallNReleasePlanner(app_config=cfg, state_manager=mock_state)

    res = planner.plan(
        metric="interaction_volume",
        recipient="recipient_internal",
        public_context={"feedback_release_ids": ["rel_committed_1"]},
    )
    assert res.metadata.get("feedback_used") is True
    assert "DP_RELEASE:rel_committed_1" in res.input_provenance


def test_dp_feedback_fallback_is_explicit():
    """When no feedback context is provided, dp_feedback explicitly notes fallback to static_public."""
    cfg = load_app_config("configs")
    cfg.privacy.planner.mode = "dp_feedback"
    planner = SmallNReleasePlanner(app_config=cfg)

    res = planner.plan(metric="interaction_volume", recipient="recipient_internal")
    assert res.metadata.get("fallback") == "static_public"
    assert res.metadata.get("feedback_used") is False
    assert "DP_FEEDBACK_SAFE_FALLBACK_TO_STATIC_PUBLIC" in res.decision_reason


def test_release_gate_manifest_fingerprint_call():
    """Verify that compute_planner_config_fingerprint accepts either AppConfig or PlannerConfig."""
    cfg = load_app_config("configs")
    fp1 = compute_planner_config_fingerprint(cfg)
    fp2 = compute_planner_config_fingerprint(cfg.privacy.planner)
    assert fp1 == fp2
    assert fp1.startswith("pcfg_")
    assert len(fp1) == 29

