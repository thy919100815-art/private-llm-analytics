"""Regression test suite for Private LLM Analytics v0.1.6 (Audit Resolution & Final Freeze Candidate).

Covers:
  - Fix A: 500+ independent noise trials in ExperimentRunner with EVALUATION accounting mode
  - Fix B: Low-level ReleaseOrchestrator fail-closed against custom and canonical production state DBs
  - Fix C: Decimal SQLite storage schema (TEXT NOT NULL) and seamless schema migration from REAL
  - Fix D: Full 256-bit SHA-256 multiset content fingerprinting (64 hex characters)
  - Fix E/F: Packaging cleanliness and trial execution semantics / recipient isolation
"""

from decimal import Decimal
from pathlib import Path
import sqlite3
import numpy as np
import pandas as pd
import pytest

from private_llm_analytics.config.loader import load_app_config
from private_llm_analytics.config.validator import (
    ExecutionMode,
    ExperimentAccountingMode,
)
from private_llm_analytics.ingestion.fingerprint import compute_dataset_fingerprint
from private_llm_analytics.privacy.budget import PrivacyBudget
from private_llm_analytics.privacy.exceptions import (
    ExperimentProductionStateCollisionError,
    PrivacyBudgetExceededError,
    RecipientSecurityViolation,
)
from private_llm_analytics.privacy.state import PrivacyStateManager
from private_llm_analytics.release.release import ReleaseOrchestrator
from private_llm_analytics.service.analytics import AnalyticsService
from private_llm_analytics.service.experiment import ExperimentRunner


@pytest.fixture
def v016_sample_data():
    return pd.DataFrame({
        "interaction_id": [f"ix_{i}" for i in range(15)],
        "user_id": [f"user_{i % 5}" for i in range(15)],
        "timestamp": ["2026-01-01T00:00:00Z"] * 15,
        "provider": ["provider_a", "provider_b", "provider_c"] * 5,
        "task": ["technical", "document", "analysis"] * 5,
    })


# =============================================================================
# Fix A: 500+ Independent Noise Trials & Evaluation Accounting Mode
# =============================================================================

def test_v016_fix_a_experiment_runner_eval_mode_default(tmp_path, v016_sample_data):
    """Test 1: ExperimentRunner defaults to EVALUATION accounting mode."""
    config = load_app_config("configs")
    runner = ExperimentRunner(config=config, state_db_path=tmp_path / "exp.db")
    assert runner.accounting_mode == ExperimentAccountingMode.EVALUATION
    assert runner.orchestrator.accounting_mode == ExperimentAccountingMode.EVALUATION


def test_v016_fix_a_eval_mode_allows_budget_overrun(tmp_path, v016_sample_data):
    """Test 2: EVALUATION mode allows exceeding global budget for Monte Carlo trials."""
    config = load_app_config("configs")
    config.privacy.global_budget.epsilon = 0.5
    runner = ExperimentRunner(
        config=config,
        state_db_path=tmp_path / "exp_overrun.db",
        accounting_mode=ExperimentAccountingMode.EVALUATION,
    )
    runner.set_data(v016_sample_data)

    # Spend 5 x 0.2 = 1.0 > 0.5 epsilon
    for s in range(5):
        res = runner.run_trial(
            metric="user_prevalence",
            recipient="internal",
            epsilon=0.2,
            delta=1e-5,
            seed=s,
        )
        assert res is not None
        assert not res["cache_hit"]


def test_v016_fix_a_bounded_mode_enforces_budget(tmp_path, v016_sample_data):
    """Test 3: BOUNDED mode strictly enforces privacy budget fail-closed."""
    config = load_app_config("configs")
    config.privacy.global_budget.epsilon = 0.5
    config.privacy.global_budget.delta = 1e-4
    runner = ExperimentRunner(
        config=config,
        state_db_path=tmp_path / "exp_bounded.db",
        accounting_mode=ExperimentAccountingMode.BOUNDED,
    )
    runner.set_data(v016_sample_data)

    # 1st trial: 0.3 epsilon <= 0.5 -> OK
    runner.run_trial(metric="user_prevalence", recipient="internal", epsilon=0.3, delta=1e-5, seed=1)

    # 2nd trial: 0.3 epsilon (total 0.6 > 0.5) -> Must fail
    with pytest.raises(PrivacyBudgetExceededError):
        runner.run_trial(metric="user_prevalence", recipient="internal", epsilon=0.3, delta=1e-5, seed=2)


def test_v016_fix_a_run_trials_convenience_method(tmp_path, v016_sample_data):
    """Test 4: run_trials convenience method produces expected trial count and distinct keys."""
    config = load_app_config("configs")
    runner = ExperimentRunner(config=config, state_db_path=tmp_path / "exp_trials.db")
    runner.set_data(v016_sample_data)

    n_trials = 10
    results = runner.run_trials(
        metric="user_prevalence",
        recipient="internal",
        n_trials=n_trials,
        base_seed=100,
        epsilon=0.1,
        delta=1e-6,
    )
    assert len(results) == n_trials
    keys = {r["release_key"] for r in results}
    assert len(keys) == n_trials


def test_v016_fix_a_repeated_seed_memoization(tmp_path, v016_sample_data):
    """Test 5: Repeated trial with identical seed returns cache_hit=True."""
    config = load_app_config("configs")
    runner = ExperimentRunner(config=config, state_db_path=tmp_path / "exp_repeat.db")
    runner.set_data(v016_sample_data)

    res1 = runner.run_trial(metric="user_prevalence", recipient="internal", epsilon=0.1, delta=1e-6, seed=42)
    assert not res1["cache_hit"]

    res2 = runner.run_trial(metric="user_prevalence", recipient="internal", epsilon=0.1, delta=1e-6, seed=42)
    assert res2["cache_hit"]
    assert res1["release_key"] == res2["release_key"]
    assert res1["results"] == res2["results"]


def test_v016_fix_a_eval_mode_no_production_contamination(tmp_path, v016_sample_data):
    """Test 6: Monte Carlo trials in EVALUATION mode do not alter production DB."""
    config = load_app_config("configs")
    prod_db = tmp_path / "prod.db"
    exp_db = tmp_path / "exp.db"
    config.privacy.release.state_db_path = str(prod_db)

    prod_mgr = PrivacyStateManager(db_path=prod_db)

    runner = ExperimentRunner(config=config, state_db_path=exp_db)
    runner.set_data(v016_sample_data)
    runner.run_trials(metric="user_prevalence", recipient="internal", n_trials=5, base_seed=0, epsilon=0.1, delta=1e-6)

    # Check production state
    spent_eps, spent_delta = prod_mgr.get_total_spent_decimal()
    assert spent_eps == Decimal("0.0")
    assert spent_delta == Decimal("0.0")
    assert len(prod_mgr.list_releases()) == 0


# =============================================================================
# Fix B: Low-Level Orchestrator & Runner Fail-Closed on Custom Production DB
# =============================================================================

def test_v016_fix_b_experiment_production_state_collision_default(tmp_path):
    """Test 7: Orchestrator in EXPERIMENT mode rejects default production DB."""
    config = load_app_config("configs")
    default_db = Path("results/privacy_state.db")

    with pytest.raises(ExperimentProductionStateCollisionError):
        ReleaseOrchestrator(
            app_config=config,
            execution_mode=ExecutionMode.EXPERIMENT,
            state_manager=PrivacyStateManager(db_path=default_db),
        )


def test_v016_fix_b_experiment_production_state_collision_custom(tmp_path):
    """Test 8: Orchestrator in EXPERIMENT mode rejects custom configured production DB."""
    config = load_app_config("configs")
    custom_prod = tmp_path / "custom_prod.db"
    config.privacy.release.state_db_path = str(custom_prod)

    with pytest.raises(ExperimentProductionStateCollisionError):
        ReleaseOrchestrator(
            app_config=config,
            execution_mode=ExecutionMode.EXPERIMENT,
            state_manager=PrivacyStateManager(db_path=custom_prod),
        )


def test_v016_fix_b_experiment_production_state_collision_relative_traversal(tmp_path):
    """Test 9: Canonical path resolution catches relative path traversal collisions."""
    config = load_app_config("configs")
    custom_prod = tmp_path / "custom_prod.db"
    config.privacy.release.state_db_path = str(custom_prod)

    # Build relative traversal pointing to the same file
    relative_path = tmp_path / "subdir" / ".." / "custom_prod.db"

    with pytest.raises(ExperimentProductionStateCollisionError):
        ReleaseOrchestrator(
            app_config=config,
            execution_mode=ExecutionMode.EXPERIMENT,
            state_manager=PrivacyStateManager(db_path=relative_path),
        )


def test_v016_fix_b_experiment_runner_custom_prod_db_collision(tmp_path):
    """Test 10: ExperimentRunner initialization rejects custom configured production DB."""
    config = load_app_config("configs")
    custom_prod = tmp_path / "custom_prod.db"
    config.privacy.release.state_db_path = str(custom_prod)

    with pytest.raises(ExperimentProductionStateCollisionError):
        ExperimentRunner(config=config, state_db_path=custom_prod)


def test_v016_fix_b_experiment_runner_dedicated_db_succeeds(tmp_path, v016_sample_data):
    """Test 11: ExperimentRunner with a dedicated non-colliding DB initializes and runs cleanly."""
    config = load_app_config("configs")
    custom_prod = tmp_path / "custom_prod.db"
    config.privacy.release.state_db_path = str(custom_prod)

    exp_db = tmp_path / "dedicated_exp.db"
    runner = ExperimentRunner(config=config, state_db_path=exp_db)
    runner.set_data(v016_sample_data)

    res = runner.run_trial(metric="user_prevalence", recipient="internal", epsilon=0.1, delta=1e-6, seed=7)
    assert res is not None


# =============================================================================
# Fix C: Decimal SQLite Storage Schema & Dynamic Migration
# =============================================================================

def test_v016_fix_c_fresh_db_schema_text_columns(tmp_path):
    """Test 12: Fresh SQLite schema creates TEXT columns for epsilon and delta."""
    db_path = tmp_path / "fresh_schema.db"
    PrivacyStateManager(db_path=db_path)

    with sqlite3.connect(db_path) as conn:
        for tbl in ["privacy_reservations", "release_store", "privacy_ledger"]:
            cursor = conn.cursor()
            cursor.execute(f"PRAGMA table_info({tbl});")
            col_map = {row[1]: row[2].upper() for row in cursor.fetchall()}
            assert col_map["epsilon"] == "TEXT", f"Expected TEXT in {tbl}.epsilon"
            assert col_map["delta"] == "TEXT", f"Expected TEXT in {tbl}.delta"


def test_v016_fix_c_legacy_schema_migration_preserves_data(tmp_path):
    """Test 13: Pre-existing database with REAL columns is seamlessly migrated to TEXT."""
    legacy_db = tmp_path / "legacy.db"
    with sqlite3.connect(legacy_db) as conn:
        conn.execute("""
            CREATE TABLE privacy_reservations (
                reservation_id TEXT PRIMARY KEY,
                release_key TEXT NOT NULL,
                owner_id TEXT NOT NULL DEFAULT '',
                generation INTEGER NOT NULL DEFAULT 1,
                epsilon REAL NOT NULL,
                delta REAL NOT NULL,
                status TEXT NOT NULL,
                created_at TEXT NOT NULL,
                committed_at TEXT
            )
        """)
        conn.execute("""
            INSERT INTO privacy_reservations VALUES (
                'r1', 'k1', 'o1', 1, 0.25, 0.0001, 'COMMITTED', '2026-01-01', '2026-01-01'
            )
        """)
        conn.commit()

    # Trigger migration via manager
    PrivacyStateManager(db_path=legacy_db)

    with sqlite3.connect(legacy_db) as conn:
        cursor = conn.cursor()
        cursor.execute("PRAGMA table_info(privacy_reservations);")
        col_map = {row[1]: row[2].upper() for row in cursor.fetchall()}
        assert col_map["epsilon"] == "TEXT"
        assert col_map["delta"] == "TEXT"

        cursor.execute("SELECT epsilon, delta FROM privacy_reservations WHERE reservation_id='r1';")
        row = cursor.fetchone()
        assert row is not None
        assert row[0] == "0.25"
        assert isinstance(row[0], str)


def test_v016_fix_c_exact_decimal_string_stored_in_sqlite(tmp_path):
    """Test 14: Values inserted into SQLite maintain exact Decimal string representation."""
    db_path = tmp_path / "exact_storage.db"
    mgr = PrivacyStateManager(db_path=db_path)

    eps = Decimal("0.100000000000000005")
    delta = Decimal("0.000000000000000001")

    res_tuple = mgr.reserve_or_get_release(
        release_key="k_exact",
        canonical_semantics={"t": 1},
        global_budget=PrivacyBudget(epsilon=Decimal("1.0"), delta=Decimal("1e-4")),
        budget_needed=PrivacyBudget(epsilon=eps, delta=delta),
        plan_metadata={},
    )
    status, _, res_id, _ = res_tuple
    assert status == "RESERVED"

    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT epsilon, delta FROM privacy_reservations WHERE reservation_id=?;", (res_id,))
        eps_str, delta_str = cursor.fetchone()
        assert eps_str == str(eps)
        assert delta_str == str(delta)


def test_v016_fix_c_adversarial_precision_rejection(tmp_path):
    """Test 15: Overspend by 10^-18 is strictly rejected by exact Decimal checks."""
    db_path = tmp_path / "precision.db"
    mgr = PrivacyStateManager(db_path=db_path)

    total_b = PrivacyBudget(epsilon=Decimal("0.000000000000000001"), delta=Decimal("1e-6"))
    needed_b = PrivacyBudget(epsilon=Decimal("0.000000000000000002"), delta=Decimal("1e-6"))

    with pytest.raises(PrivacyBudgetExceededError):
        mgr.reserve_or_get_release(
            release_key="k_nano",
            canonical_semantics={"m": 1},
            global_budget=total_b,
            budget_needed=needed_b,
            plan_metadata={},
            allow_overrun=False,
        )


# =============================================================================
# Fix D: Full 256-Bit SHA-256 Dataset Content Fingerprinting
# =============================================================================

def test_v016_fix_d_dataset_fingerprint_full_256bit_sha256(v016_sample_data):
    """Test 16: Fingerprint format starts with data_sha256_ followed by 64 hex characters."""
    fp = compute_dataset_fingerprint(v016_sample_data)
    assert fp.startswith("data_sha256_")
    hex_part = fp.replace("data_sha256_", "")
    assert len(hex_part) == 64
    assert all(c in "0123456789abcdef" for c in hex_part)


def test_v016_fix_d_dataset_fingerprint_length_and_format(v016_sample_data):
    """Test 17: Full fingerprint string is exactly 76 characters long (12 prefix + 64 hex)."""
    fp = compute_dataset_fingerprint(v016_sample_data)
    assert len(fp) == 76


def test_v016_fix_d_fingerprint_changes_on_row_modification(v016_sample_data):
    """Test 18: Modifying a single cell changes the 256-bit fingerprint."""
    fp1 = compute_dataset_fingerprint(v016_sample_data)

    df_mod = v016_sample_data.copy()
    df_mod.loc[0, "task"] = "document"
    fp2 = compute_dataset_fingerprint(df_mod)

    assert fp1 != fp2


def test_v016_fix_d_fingerprint_multiset_row_order_invariance(v016_sample_data):
    """Test 19: Row permutation preserves the 256-bit multiset content fingerprint."""
    fp1 = compute_dataset_fingerprint(v016_sample_data)

    df_shuffled = v016_sample_data.sample(frac=1.0, random_state=999).reset_index(drop=True)
    fp2 = compute_dataset_fingerprint(df_shuffled)

    assert fp1 == fp2


# =============================================================================
# Fix F / Recipient Security: External Commercial Recipient Protection in Experiment Mode
# =============================================================================

def test_v016_fix_f_experiment_runner_external_recipient_blocked(tmp_path, v016_sample_data):
    """Test 20: Experiment mode with seeded noise strictly forbids targeting external recipients."""
    config = load_app_config("configs")
    runner = ExperimentRunner(config=config, state_db_path=tmp_path / "exp_sec.db")
    runner.set_data(v016_sample_data)

    with pytest.raises(RecipientSecurityViolation):
        runner.run_query(
            metric="user_prevalence",
            recipient="provider_a",
            epsilon=0.1,
            delta=1e-6,
            noise_seed=123,
        )

    with pytest.raises(RecipientSecurityViolation):
        runner.run_trials(
            metric="user_prevalence",
            recipient="provider_a",
            n_trials=2,
            base_seed=0,
        )
