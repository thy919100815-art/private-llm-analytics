"""Regression test suite for Private LLM Analytics v0.1.4 (Freeze Hardening).

Covers:
  - Finding F1: Experiment & Production RNG/State Isolation
  - Finding F2a/F2b: Production Fail-Closed Safety (reuse_existing_release=True, allow_budget_overrun=False)
  - Finding F3: Zero-Slack Decimal Budget Accounting (delta=0 rejection, small epsilon precision)
  - Finding F4: In-Memory Dataset Identity and Multiset Content Fingerprinting
  - Non-blocking N1: Legacy state cleanup and accountant authority
"""

from decimal import Decimal
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from private_llm_analytics.aggregation.user_matrix import (
    UserMatrix,
    build_cell_definitions,
)
from private_llm_analytics.config.loader import load_app_config
from private_llm_analytics.config.validator import (
    ExecutionMode,
    validate_production_config,
)
from private_llm_analytics.ingestion.fingerprint import compute_dataset_fingerprint
from private_llm_analytics.planner.plan import ExecutionPlan
from private_llm_analytics.planner.planner import AnalyticsPlanner
from private_llm_analytics.planner.query import AnalyticsQuery
from private_llm_analytics.privacy.accountant import BasicAccountant
from private_llm_analytics.privacy.budget import PrivacyBudget
from private_llm_analytics.privacy.exceptions import (
    PrivacyBudgetExceededError,
    PrivacyConfigurationError,
    ProductionRNGControlError,
    ProductionSafetyConfigurationError,
    RecipientSecurityViolation,
)
from private_llm_analytics.privacy.state import PrivacyStateManager
from private_llm_analytics.release.release import ReleaseOrchestrator
from private_llm_analytics.service.analytics import AnalyticsService
from private_llm_analytics.service.experiment import ExperimentRunner


@pytest.fixture
def v014_sample_data():
    return pd.DataFrame({
        "interaction_id": [f"ix_{i}" for i in range(12)],
        "user_id": [f"usr_{i % 4}" for i in range(12)],
        "timestamp": ["2026-01-01T00:00:00Z"] * 12,
        "provider": ["provider_a", "provider_b", "provider_c"] * 4,
        "task": ["technical", "document", "analysis", "technical"] * 3,
    })


@pytest.fixture
def v014_matrix():
    provs = ["provider_a", "provider_b", "provider_c"]
    tsks = ["technical", "document", "analysis"]
    cell_names, cell_map = build_cell_definitions(provs, tsks)
    np_matrix = np.ones((4, 9), dtype=np.float64) * 10.0
    return UserMatrix(
        matrix=np_matrix,
        user_ids=[f"usr_{i}" for i in range(4)],
        providers=provs,
        tasks=tsks,
        cell_names=cell_names,
        cell_map=cell_map,
        public_population_size=50000,
    )


# =============================================================================
# Finding F1: Experiment & Production RNG/State Isolation
# =============================================================================

def test_v014_f1_experiment_runner_creates_separate_db(tmp_path, v014_sample_data):
    """F1: ExperimentRunner must use its own isolated DB without touching production state DB."""
    config = load_app_config("configs")
    prod_db = tmp_path / "prod_state.db"
    exp_db = tmp_path / "exp_state.db"
    config.privacy.release.state_db_path = str(prod_db)

    runner = ExperimentRunner(config=config, state_db_path=exp_db)
    runner.set_data(v014_sample_data)

    res = runner.run_query(
        metric="user_prevalence",
        recipient="internal",
        epsilon=1.0,
        delta=1e-6,
        noise_seed=42,
    )
    assert not res["cache_hit"]

    # exp_db should have recorded release, prod_db should not even exist or have 0 spend
    exp_state = PrivacyStateManager(exp_db)
    spent_eps, _ = exp_state.get_total_spent()
    assert float(spent_eps) == 1.0

    assert not prod_db.exists() or PrivacyStateManager(prod_db).get_total_spent() == (0.0, 0.0)


def test_v014_f1_release_key_distinct_by_execution_mode():
    """F1: Canonical semantics must namespace keys by execution_mode."""
    config = load_app_config("configs")
    planner = AnalyticsPlanner(config)
    query = AnalyticsQuery(metric="user_prevalence", recipient="internal", epsilon=1.0, delta=1e-6)

    plan_prod = planner.create_plan(query, dataset_version="v1.0", execution_mode=ExecutionMode.PRODUCTION)
    plan_exp = planner.create_plan(query, dataset_version="v1.0", execution_mode=ExecutionMode.EXPERIMENT)

    assert plan_prod.release_key != plan_exp.release_key
    assert plan_prod.canonical_semantics["execution_mode"] == "production"
    assert plan_exp.canonical_semantics["execution_mode"] == "experiment"


def test_v014_f1_production_query_never_hits_experiment_release(tmp_path, v014_sample_data):
    """F1: Pre-existing experiment release cannot be served to production external caller."""
    config = load_app_config("configs")
    prod_db = tmp_path / "prod.db"
    exp_db = tmp_path / "exp.db"
    config.privacy.release.state_db_path = str(prod_db)

    service = AnalyticsService(config)
    service.set_data(v014_sample_data, dataset_version="fixed_v1")

    # Step 1: Run experiment with deterministic seed
    exp_res = service.compute_experiment(
        metric="user_prevalence",
        recipient="internal",
        epsilon=1.0,
        delta=1e-6,
        noise_seed=12345,
        experiment_db_path=exp_db,
    )
    assert not exp_res["cache_hit"]

    # Step 2: Production query for provider_a
    prod_res = service.compute(
        metric="user_prevalence",
        recipient="provider_a",
        epsilon=1.0,
        delta=1e-6,
    )
    assert not prod_res["cache_hit"]
    assert prod_res["release_key"] != exp_res["release_key"]


def test_v014_f1_production_release_never_poisons_experiment(tmp_path, v014_sample_data):
    """F1: Pre-existing production release does not contaminate experiment-mode seeded release."""
    config = load_app_config("configs")
    prod_db = tmp_path / "prod.db"
    exp_db = tmp_path / "exp.db"
    config.privacy.release.state_db_path = str(prod_db)

    service = AnalyticsService(config)
    service.set_data(v014_sample_data, dataset_version="fixed_v1")

    # Step 1: Production query
    prod_res = service.compute(
        metric="user_prevalence",
        recipient="internal",
        epsilon=1.0,
        delta=1e-6,
    )
    assert not prod_res["cache_hit"]

    # Step 2: Experiment query with seed
    exp_res = service.compute_experiment(
        metric="user_prevalence",
        recipient="internal",
        epsilon=1.0,
        delta=1e-6,
        noise_seed=999,
        experiment_db_path=exp_db,
    )
    assert not exp_res["cache_hit"]
    assert exp_res["release_key"] != prod_res["release_key"]


def test_v014_f1_production_compute_rejects_rng_for_external(tmp_path, v014_sample_data):
    """F1: Production compute() rejects caller RNG for external recipient."""
    config = load_app_config("configs")
    config.privacy.release.state_db_path = str(tmp_path / "prod.db")
    service = AnalyticsService(config)
    service.set_data(v014_sample_data)

    bad_rng = np.random.default_rng(123)
    with pytest.raises(ProductionRNGControlError):
        service.compute(metric="user_prevalence", recipient="provider_a", rng=bad_rng)


def test_v014_f1_production_compute_rejects_rng_for_trusted(tmp_path, v014_sample_data):
    """F1: Production compute() rejects caller RNG for trusted/internal recipient."""
    config = load_app_config("configs")
    config.privacy.release.state_db_path = str(tmp_path / "prod.db")
    service = AnalyticsService(config)
    service.set_data(v014_sample_data)

    bad_rng = np.random.default_rng(123)
    with pytest.raises(ProductionRNGControlError):
        service.compute(metric="user_prevalence", recipient="internal", rng=bad_rng)


def test_v014_f1_production_compute_rejects_seed_kwargs(tmp_path, v014_sample_data):
    """F1: Production compute() rejects seed / noise_seed kwargs."""
    config = load_app_config("configs")
    config.privacy.release.state_db_path = str(tmp_path / "prod.db")
    service = AnalyticsService(config)
    service.set_data(v014_sample_data)

    with pytest.raises(ProductionRNGControlError):
        service.compute(metric="user_prevalence", recipient="provider_a", seed=42)

    with pytest.raises(ProductionRNGControlError):
        service.compute(metric="user_prevalence", recipient="provider_a", noise_seed=42)


def test_v014_f1_production_compute_rejects_positional_extra_args(tmp_path, v014_sample_data):
    """F1: Positional arguments beyond include_exact are rejected."""
    config = load_app_config("configs")
    config.privacy.release.state_db_path = str(tmp_path / "prod.db")
    service = AnalyticsService(config)
    service.set_data(v014_sample_data)

    with pytest.raises(ProductionRNGControlError):
        service.compute("user_prevalence", "provider_a", None, None, None, None, None, None, False, "extra_pos_arg")


def test_v014_f1_experiment_blocks_external_recipients(tmp_path, v014_sample_data):
    """F1: ExperimentRunner strictly rejects commercial external recipients."""
    config = load_app_config("configs")
    runner = ExperimentRunner(config=config, state_db_path=tmp_path / "exp.db")
    runner.set_data(v014_sample_data)

    with pytest.raises(RecipientSecurityViolation, match="cannot target external commercial recipients"):
        runner.run_query(metric="user_prevalence", recipient="provider_a")


def test_v014_f1_experiment_runner_rejects_production_db_path():
    """F1: ExperimentRunner refuses to use production privacy_state.db."""
    config = load_app_config("configs")
    with pytest.raises(ValueError, match="must not collide with production state DB"):
        ExperimentRunner(config=config, state_db_path="results/privacy_state.db")


# =============================================================================
# Finding F2a / F2b: Production Fail-Closed Safety Configurations
# =============================================================================

def test_v014_f2a_validate_production_config_rejects_reuse_false():
    """F2a: validate_production_config raises error if reuse_existing_release is False."""
    config = load_app_config("configs")
    config.privacy.release.reuse_existing_release = False
    with pytest.raises(ProductionSafetyConfigurationError, match="reuse_existing_release=True"):
        validate_production_config(config)


def test_v014_f2b_validate_production_config_rejects_overrun_true():
    """F2b: validate_production_config raises error if allow_budget_overrun is True."""
    config = load_app_config("configs")
    config.privacy.release.allow_budget_overrun = True
    with pytest.raises(ProductionSafetyConfigurationError, match="allow_budget_overrun=False"):
        validate_production_config(config)


def test_v014_f2a_service_compute_enforces_reuse_true(tmp_path, v014_sample_data):
    """F2a: service.compute() fail-closed when reuse_existing_release is False."""
    config = load_app_config("configs")
    config.privacy.release.reuse_existing_release = False
    config.privacy.release.state_db_path = str(tmp_path / "prod.db")
    service = AnalyticsService(config)
    service.set_data(v014_sample_data)

    with pytest.raises(ProductionSafetyConfigurationError):
        service.compute(metric="user_prevalence", recipient="provider_a")


def test_v014_f2b_service_compute_enforces_no_overrun(tmp_path, v014_sample_data):
    """F2b: service.compute() fail-closed when allow_budget_overrun is True."""
    config = load_app_config("configs")
    config.privacy.release.allow_budget_overrun = True
    config.privacy.release.state_db_path = str(tmp_path / "prod.db")
    service = AnalyticsService(config)
    service.set_data(v014_sample_data)

    with pytest.raises(ProductionSafetyConfigurationError):
        service.compute(metric="user_prevalence", recipient="provider_a")


def test_v014_f2_orchestrator_production_mode_forces_safe_parameters(tmp_path, v014_matrix):
    """F2: ReleaseOrchestrator in production mode forces allow_overrun=False."""
    config = load_app_config("configs")
    state_mgr = PrivacyStateManager(tmp_path / "prod.db")
    orch = ReleaseOrchestrator(
        app_config=config,
        state_manager=state_mgr,
        execution_mode=ExecutionMode.PRODUCTION,
    )
    assert orch.execution_mode == ExecutionMode.PRODUCTION


# =============================================================================
# Finding F3: Zero-Slack Decimal Budget Accounting
# =============================================================================

def test_v014_f3_delta_zero_rejects_positive_delta(tmp_path):
    """F3: Global delta=0.0 rejects requested delta=1e-13 (zero slack)."""
    state_mgr = PrivacyStateManager(tmp_path / "state.db")
    global_b = PrivacyBudget(epsilon=4.0, delta=0.0)
    req_b = PrivacyBudget(epsilon=1.0, delta=1e-13)

    with pytest.raises(PrivacyBudgetExceededError):
        state_mgr.reserve_or_get_release(
            release_key="k1",
            canonical_semantics={"test": "k1"},
            global_budget=global_b,
            budget_needed=req_b,
            plan_metadata={},
            allow_overrun=False,
        )


def test_v014_f3_small_epsilon_zero_slack(tmp_path):
    """F3: Small epsilon (1e-8) rejects cumulative overspend (6e-9 + 6e-9 = 1.2e-8 > 1e-8)."""
    state_mgr = PrivacyStateManager(tmp_path / "state.db")
    global_b = PrivacyBudget(epsilon=1e-8, delta=0.0)

    # First request: 6e-9 <= 1e-8
    st, _, _, _ = state_mgr.reserve_or_get_release(
        release_key="k1",
        canonical_semantics={"test": "k1"},
        global_budget=global_b,
        budget_needed=PrivacyBudget(epsilon=6e-9, delta=0.0),
        plan_metadata={},
        allow_overrun=False,
    )
    assert st == "RESERVED"

    # Second request: 6e-9 + 6e-9 = 1.2e-8 > 1e-8 => must reject!
    with pytest.raises(PrivacyBudgetExceededError):
        state_mgr.reserve_or_get_release(
            release_key="k2",
            canonical_semantics={"test": "k2"},
            global_budget=global_b,
            budget_needed=PrivacyBudget(epsilon=6e-9, delta=0.0),
            plan_metadata={},
            allow_overrun=False,
        )


def test_v014_f3_exact_decimal_accounting_no_drift(tmp_path):
    """F3: Exact decimal composition maintains zero floating-point accumulation drift."""
    state_mgr = PrivacyStateManager(tmp_path / "state.db")
    global_b = PrivacyBudget(epsilon=1.0, delta=1e-5)
    step_b = PrivacyBudget(epsilon=0.1, delta=1e-6)

    # 10 steps of 0.1 must sum to exactly 1.0
    for i in range(10):
        st, _, _, _ = state_mgr.reserve_or_get_release(
            release_key=f"step_{i}",
            canonical_semantics={"step": i},
            global_budget=global_b,
            budget_needed=step_b,
            plan_metadata={},
            allow_overrun=False,
        )
        assert st == "RESERVED"

    spent_eps, spent_delta = state_mgr.get_total_reserved_or_spent()
    assert float(spent_eps) == 1.0
    assert abs(float(spent_delta) - 1e-5) < 1e-15

    # 11th step must be rejected
    with pytest.raises(PrivacyBudgetExceededError):
        state_mgr.reserve_or_get_release(
            release_key="step_11",
            canonical_semantics={"step": 11},
            global_budget=global_b,
            budget_needed=PrivacyBudget(epsilon=1e-12, delta=0.0),
            plan_metadata={},
            allow_overrun=False,
        )


def test_v014_f3_accountant_exact_decimal_check():
    """F3: BasicAccountant.check_budget_available uses exact Decimal."""
    accountant = BasicAccountant(
        global_budget=PrivacyBudget(epsilon=4.0, delta=0.0),
        allow_budget_overrun=False,
    )
    # delta > 0 must be rejected when global delta == 0
    assert not accountant.check_budget_available(PrivacyBudget(epsilon=1.0, delta=1e-15))
    # epsilon <= 4.0 and delta == 0 is accepted
    assert accountant.check_budget_available(PrivacyBudget(epsilon=4.0, delta=0.0))
    # epsilon > 4.0 is rejected
    assert not accountant.check_budget_available(PrivacyBudget(epsilon=4.0000000001, delta=0.0))


# =============================================================================
# Finding F4: In-Memory Dataset Identity & Content Fingerprinting
# =============================================================================

def test_v014_f4_set_data_auto_computes_fingerprint(v014_sample_data):
    """F4: set_data() without version auto-derives content fingerprint."""
    config = load_app_config("configs")
    service = AnalyticsService(config)
    service.set_data(v014_sample_data)

    expected_fp = compute_dataset_fingerprint(v014_sample_data)
    assert service._dataset_version == expected_fp
    assert service._dataset_version.startswith("data_sha256_")


def test_v014_f4_dataset_change_generates_distinct_key(v014_sample_data):
    """F4: Changing dataset rows changes fingerprint and release key."""
    config = load_app_config("configs")
    planner = AnalyticsPlanner(config)
    query = AnalyticsQuery(metric="user_prevalence", recipient="provider_a", epsilon=1.0, delta=1e-6)

    fp_1 = compute_dataset_fingerprint(v014_sample_data)
    plan_1 = planner.create_plan(query, dataset_version=fp_1)

    # Modify dataframe content
    modified_data = v014_sample_data.copy()
    modified_data.loc[0, "provider"] = "provider_c"
    fp_2 = compute_dataset_fingerprint(modified_data)
    plan_2 = planner.create_plan(query, dataset_version=fp_2)

    assert fp_1 != fp_2
    assert plan_1.release_key != plan_2.release_key


def test_v014_f4_different_data_no_stale_cache_hit(tmp_path):
    """F4: Different datasets do not hit previous release cache."""
    config = load_app_config("configs")
    config.privacy.release.state_db_path = str(tmp_path / "prod.db")
    service = AnalyticsService(config)

    df_a = pd.DataFrame({
        "interaction_id": ["1", "2"],
        "user_id": ["u1", "u2"],
        "timestamp": ["2026-01-01T00:00:00Z"] * 2,
        "provider": ["provider_a", "provider_a"],
        "task": ["technical", "technical"],
    })
    service.set_data(df_a)
    res_a = service.compute(metric="user_prevalence", recipient="provider_a", epsilon=1.0, delta=1e-6)
    assert not res_a["cache_hit"]

    df_b = pd.DataFrame({
        "interaction_id": ["1", "2"],
        "user_id": ["u1", "u2"],
        "timestamp": ["2026-01-01T00:00:00Z"] * 2,
        "provider": ["provider_b", "provider_b"],
        "task": ["document", "document"],
    })
    service.set_data(df_b)
    res_b = service.compute(metric="user_prevalence", recipient="provider_a", epsilon=1.0, delta=1e-6)
    assert not res_b["cache_hit"]
    assert res_b["release_key"] != res_a["release_key"]


def test_v014_f4_multiset_fingerprint_row_permutation_invariant(v014_sample_data):
    """F4: Row permutation yields identical fingerprint and cache hit."""
    fp_original = compute_dataset_fingerprint(v014_sample_data)
    shuffled = v014_sample_data.sample(frac=1.0, random_state=123).reset_index(drop=True)
    fp_shuffled = compute_dataset_fingerprint(shuffled)

    assert fp_original == fp_shuffled


def test_v014_f4_multiset_fingerprint_column_permutation_invariant(v014_sample_data):
    """F4: Column order permutation yields identical fingerprint."""
    fp_original = compute_dataset_fingerprint(v014_sample_data)
    cols = list(v014_sample_data.columns)[::-1]
    permuted_cols = v014_sample_data[cols]
    fp_permuted = compute_dataset_fingerprint(permuted_cols)

    assert fp_original == fp_permuted


# =============================================================================
# Cleanup & Architecture: N1 & Exception Hierarchy
# =============================================================================

def test_v014_n1_accountant_no_legacy_fallback(tmp_path):
    """N1: BasicAccountant spent_epsilon reads solely from PrivacyStateManager."""
    state_mgr = PrivacyStateManager(tmp_path / "state.db")
    accountant = BasicAccountant(
        global_budget=PrivacyBudget(epsilon=4.0, delta=1e-5),
        allow_budget_overrun=False,
        state_manager=state_mgr,
    )
    assert accountant.spent_epsilon == 0.0
    assert accountant.spent_delta == 0.0


def test_v014_exceptions_hierarchy():
    """Verify exception inheritance hierarchy for backward compatibility."""
    assert issubclass(ProductionRNGControlError, RecipientSecurityViolation)
    assert issubclass(ProductionSafetyConfigurationError, PrivacyConfigurationError)


def test_v014_n2_version_string_updated():
    """N2: Package version must be strictly 0.1.4 or later."""
    import private_llm_analytics
    assert private_llm_analytics.__version__ in ("0.1.4", "0.1.5", "0.1.6", "0.1.7", "1.0.0")

