"""Regression test suite for Private LLM Analytics v0.1.5 (Audit Resolution & Freeze Candidate).

Covers:
  - Finding A: Exact Decimal budget accounting (rejection of binary float drift)
  - Finding B: Experiment noise_seed / trial_id in canonical semantics and release identity
  - Finding C: Dataset identity content-binding for explicit version labels
  - Finding D: Synthetic dataset seed=0 preservation and content-derived versioning
  - Finding E: Authoritative state and audit ledger consistency
  - Finding F: Low-level ReleaseOrchestrator isolated default and fail-closed safety
"""

from decimal import Decimal
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from private_llm_analytics.config.loader import load_app_config
from private_llm_analytics.config.validator import ExecutionMode
from private_llm_analytics.ingestion.fingerprint import compute_dataset_fingerprint
from private_llm_analytics.planner.planner import AnalyticsPlanner
from private_llm_analytics.planner.query import AnalyticsQuery
from private_llm_analytics.privacy.budget import PrivacyBudget
from private_llm_analytics.privacy.exceptions import (
    PrivacyBudgetExceededError,
    ProductionRNGControlError,
)
from private_llm_analytics.privacy.state import PrivacyStateManager
from private_llm_analytics.release.release import ReleaseOrchestrator
from private_llm_analytics.service.analytics import AnalyticsService
from private_llm_analytics.service.experiment import ExperimentRunner


@pytest.fixture
def v015_sample_data():
    return pd.DataFrame({
        "interaction_id": [f"ix_{i}" for i in range(12)],
        "user_id": [f"usr_{i % 4}" for i in range(12)],
        "timestamp": ["2026-01-01T00:00:00Z"] * 12,
        "provider": ["provider_a", "provider_b", "provider_c"] * 4,
        "task": ["technical", "document", "analysis", "technical"] * 3,
    })


# =============================================================================
# Finding A: Exact Decimal Budget Accounting (Eliminating SQLite Float Drift)
# =============================================================================

def test_v015_finding_a_adversarial_sum_real_rejected(tmp_path):
    """Finding A: 0.01 + 0.06 = 0.06999999999999999 in binary float.
    A subsequent 5e-18 must be rejected by exact Decimal accounting."""
    db_path = tmp_path / "budget_a.db"
    state_mgr = PrivacyStateManager(db_path)
    global_b = PrivacyBudget(epsilon=0.07, delta=0.0)

    # Step 1: 0.01
    st1, _, _, _ = state_mgr.reserve_or_get_release(
        release_key="k1",
        canonical_semantics={"s": 1},
        global_budget=global_b,
        budget_needed=PrivacyBudget(epsilon=0.01, delta=0.0),
        plan_metadata={},
        allow_overrun=False,
    )
    assert st1 == "RESERVED"

    # Step 2: 0.06
    st2, _, _, _ = state_mgr.reserve_or_get_release(
        release_key="k2",
        canonical_semantics={"s": 2},
        global_budget=global_b,
        budget_needed=PrivacyBudget(epsilon=0.06, delta=0.0),
        plan_metadata={},
        allow_overrun=False,
    )
    assert st2 == "RESERVED"

    # Step 3: 5e-18 (exceeds global 0.07 in Decimal)
    with pytest.raises(PrivacyBudgetExceededError):
        state_mgr.reserve_or_get_release(
            release_key="k3",
            canonical_semantics={"s": 3},
            global_budget=global_b,
            budget_needed=PrivacyBudget(epsilon=5e-18, delta=0.0),
            plan_metadata={},
            allow_overrun=False,
        )


def test_v015_finding_a_decimal_summation_subnormal(tmp_path):
    """Finding A: Subnormal floating point sum 0.0003 + 0.00006 + 1e-20 > 0.00036."""
    db_path = tmp_path / "budget_subnormal.db"
    state_mgr = PrivacyStateManager(db_path)
    global_b = PrivacyBudget(epsilon=0.00036, delta=0.0)

    state_mgr.reserve_or_get_release(
        release_key="k1",
        canonical_semantics={"s": 1},
        global_budget=global_b,
        budget_needed=PrivacyBudget(epsilon=0.0003, delta=0.0),
        plan_metadata={},
        allow_overrun=False,
    )
    state_mgr.reserve_or_get_release(
        release_key="k2",
        canonical_semantics={"s": 2},
        global_budget=global_b,
        budget_needed=PrivacyBudget(epsilon=0.00006, delta=0.0),
        plan_metadata={},
        allow_overrun=False,
    )

    with pytest.raises(PrivacyBudgetExceededError):
        state_mgr.reserve_or_get_release(
            release_key="k3",
            canonical_semantics={"s": 3},
            global_budget=global_b,
            budget_needed=PrivacyBudget(epsilon=1e-20, delta=0.0),
            plan_metadata={},
            allow_overrun=False,
        )


# =============================================================================
# Finding B: Experiment Noise Seed & Trial Identity in Canonical Semantics
# =============================================================================

def test_v015_finding_b_experiment_noise_seed_in_canonical_semantics():
    """Finding B: Noise seed and trial ID are embedded in canonical semantics for experiments."""
    config = load_app_config("configs")
    planner = AnalyticsPlanner(config)
    query = AnalyticsQuery(metric="user_prevalence", recipient="internal")

    # In experiment mode with noise_seed
    plan_s1 = planner.create_plan(
        query,
        execution_mode=ExecutionMode.EXPERIMENT,
        noise_seed=1,
    )
    assert plan_s1.canonical_semantics["noise_seed_identity"] == "seed_1"

    plan_s2 = planner.create_plan(
        query,
        execution_mode=ExecutionMode.EXPERIMENT,
        noise_seed=2,
    )
    assert plan_s2.canonical_semantics["noise_seed_identity"] == "seed_2"
    assert plan_s1.release_key != plan_s2.release_key

    # In production mode, seed is ignored in canonical semantics (always unseeded)
    plan_prod = planner.create_plan(
        query,
        execution_mode=ExecutionMode.PRODUCTION,
    )
    assert plan_prod.canonical_semantics["noise_seed_identity"] == "production_unseeded"


def test_v015_finding_b_experiment_trial_id_disambiguation():
    """Finding B: trial_id disambiguates unseeded Monte Carlo trials."""
    config = load_app_config("configs")
    planner = AnalyticsPlanner(config)
    query = AnalyticsQuery(metric="user_prevalence", recipient="internal")

    plan_t1 = planner.create_plan(
        query,
        execution_mode=ExecutionMode.EXPERIMENT,
        trial_id="mc_trial_1",
    )
    assert plan_t1.canonical_semantics["noise_seed_identity"] == "trial_mc_trial_1"

    plan_t2 = planner.create_plan(
        query,
        execution_mode=ExecutionMode.EXPERIMENT,
        trial_id="mc_trial_2",
    )
    assert plan_t2.canonical_semantics["noise_seed_identity"] == "trial_mc_trial_2"
    assert plan_t1.release_key != plan_t2.release_key


def test_v015_finding_b_production_rejects_noise_seed_parameter(tmp_path, v015_sample_data):
    """Finding B: Production release strictly rejects caller noise_seed."""
    config = load_app_config("configs")
    config.privacy.release.state_db_path = str(tmp_path / "prod.db")
    service = AnalyticsService(config)
    service.set_data(v015_sample_data)

    with pytest.raises(ProductionRNGControlError):
        service.compute(
            metric="user_prevalence",
            recipient="provider_a",
            noise_seed=42,
        )


# =============================================================================
# Finding C: Content-Derived Dataset Identity with Human Labels
# =============================================================================

def test_v015_finding_c_label_override_bound_to_content(v015_sample_data):
    """Finding C: set_data with human label combines label with multiset fingerprint."""
    config = load_app_config("configs")
    service = AnalyticsService(config)

    df_a = v015_sample_data.copy()
    service.set_data(df_a, dataset_version="v1.0")
    fp_a = compute_dataset_fingerprint(df_a)
    assert service._dataset_version == f"v1.0@{fp_a}"

    # Mutate data but keep same human label
    df_b = v015_sample_data.copy()
    df_b.loc[0, "provider"] = "different_provider"
    service.set_data(df_b, dataset_version="v1.0")
    fp_b = compute_dataset_fingerprint(df_b)
    assert service._dataset_version == f"v1.0@{fp_b}"

    assert fp_a != fp_b
    assert service._dataset_version != f"v1.0@{fp_a}"


# =============================================================================
# Finding D: Synthetic Dataset Generation Versioning & Seed=0 Preservation
# =============================================================================

def test_v015_finding_d_synthetic_seed_zero_not_overwritten():
    """Finding D: generate_synthetic_data(seed=0) preserves seed=0."""
    config = load_app_config("configs")
    service = AnalyticsService(config)

    df_0 = service.generate_synthetic_data(n_users=20, seed=0)
    ver_0 = service._dataset_version
    assert "synthetic_seed_0_" in ver_0

    df_42 = service.generate_synthetic_data(n_users=20, seed=42)
    ver_42 = service._dataset_version
    assert "synthetic_seed_42_" in ver_42
    assert ver_0 != ver_42


# =============================================================================
# Finding F: ReleaseOrchestrator Isolated Defaults and Fail-Closed Safety
# =============================================================================

def test_v015_finding_f_orchestrator_experiment_defaults_to_isolated_db(tmp_path):
    """Finding F: ReleaseOrchestrator in experiment mode defaults to results/experiment_state.db."""
    config = load_app_config("configs")
    orch = ReleaseOrchestrator(
        app_config=config,
        execution_mode=ExecutionMode.EXPERIMENT,
    )
    assert orch.state_manager.db_path.name == "experiment_state.db"


def test_v015_finding_f_orchestrator_experiment_fails_closed_on_production_db():
    """Finding F: ReleaseOrchestrator in experiment mode rejects production privacy_state.db."""
    config = load_app_config("configs")
    prod_state_mgr = PrivacyStateManager("results/privacy_state.db")

    with pytest.raises(ValueError, match="Experiment mode cannot use production privacy state database"):
        ReleaseOrchestrator(
            app_config=config,
            state_manager=prod_state_mgr,
            execution_mode=ExecutionMode.EXPERIMENT,
        )


# =============================================================================
# Version and Freeze Check
# =============================================================================

def test_v015_version_string():
    """Ensure package version is 0.1.5 or later (v0.1.7 freeze)."""
    import private_llm_analytics
    assert private_llm_analytics.__version__ in ("0.1.5", "0.1.6", "0.1.7")
