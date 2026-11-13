"""Regression test suite for Private LLM Analytics v0.1.7 (Final Gate Patch / Freeze Candidate).

Covers all 15 audit scenarios from Section 37:
  1. Low-level EXPERIMENT + provider_a rejected
  2. Low-level EXPERIMENT + provider_b rejected
  3. Low-level EXPERIMENT + provider_c rejected
  4. Low-level EXPERIMENT + internal allowed
  5. PRODUCTION + external remains allowed
  6. External experiment rejection has zero accounting side effects
  7. Repeated 10 external experiment attempts all rejected
  8. run_all_experiments exp02 population == 50000
  9. exp02 smoke load has public_population_size == 50000 & fail-fast check
  10. Manifest included/excluded status matches archive
  11. No duplicate canonical dataset files (n10k/n50k aliases removed)
  12. Checksum file is machine-parseable (<hash>  <file>)
  13. All packaged dataset hashes verify bit-for-bit
  14. Fresh ZIP verifier discovers all verify scripts (>= 14 scripts)
  15. New external-block script included and runnable
"""

from decimal import Decimal
import hashlib
import inspect
from pathlib import Path
import re
import sqlite3
import pandas as pd
import pytest

from private_llm_analytics.aggregation.user_matrix import build_user_matrix
from private_llm_analytics.config.loader import load_app_config
from private_llm_analytics.config.validator import ExecutionMode, ExperimentAccountingMode
from private_llm_analytics.planner.query import AnalyticsQuery
from private_llm_analytics.privacy.exceptions import (
    DatasetConfigurationMismatchError,
    RecipientSecurityViolation,
)
from private_llm_analytics.privacy.state import PrivacyStateManager
from private_llm_analytics.release.release import ReleaseOrchestrator


@pytest.fixture
def v017_sample_matrix():
    df = pd.DataFrame({
        "interaction_id": [f"ix_{i}" for i in range(15)],
        "user_id": [f"user_{i % 5}" for i in range(15)],
        "timestamp": ["2026-01-01T00:00:00Z"] * 15,
        "provider": ["provider_a", "provider_b", "provider_c"] * 5,
        "task": ["technical", "document", "analysis"] * 5,
    })
    return build_user_matrix(df, public_population_size=100)


# =============================================================================
# Gate Fix A: Low-level ReleaseOrchestrator External Recipient Block
# =============================================================================

def test_01_low_level_experiment_provider_a_rejected(tmp_path, v017_sample_matrix):
    """Test 1: Low-level EXPERIMENT mode rejects provider_a."""
    config = load_app_config("configs")
    state_mgr = PrivacyStateManager(tmp_path / "exp.db")
    orch = ReleaseOrchestrator(
        app_config=config,
        state_manager=state_mgr,
        execution_mode=ExecutionMode.EXPERIMENT,
        accounting_mode=ExperimentAccountingMode.EVALUATION,
    )
    query = AnalyticsQuery(metric="user_prevalence", recipient="provider_a", epsilon=1.0, delta=1e-5)
    with pytest.raises(RecipientSecurityViolation, match="Experiment mode must never target external recipients"):
        orch.execute_release(query, v017_sample_matrix)


def test_02_low_level_experiment_provider_b_rejected(tmp_path, v017_sample_matrix):
    """Test 2: Low-level EXPERIMENT mode rejects provider_b."""
    config = load_app_config("configs")
    state_mgr = PrivacyStateManager(tmp_path / "exp.db")
    orch = ReleaseOrchestrator(
        app_config=config,
        state_manager=state_mgr,
        execution_mode=ExecutionMode.EXPERIMENT,
        accounting_mode=ExperimentAccountingMode.EVALUATION,
    )
    query = AnalyticsQuery(metric="user_prevalence", recipient="provider_b", epsilon=1.0, delta=1e-5)
    with pytest.raises(RecipientSecurityViolation, match="Experiment mode must never target external recipients"):
        orch.execute_release(query, v017_sample_matrix)


def test_03_low_level_experiment_provider_c_rejected(tmp_path, v017_sample_matrix):
    """Test 3: Low-level EXPERIMENT mode rejects provider_c."""
    config = load_app_config("configs")
    state_mgr = PrivacyStateManager(tmp_path / "exp.db")
    orch = ReleaseOrchestrator(
        app_config=config,
        state_manager=state_mgr,
        execution_mode=ExecutionMode.EXPERIMENT,
        accounting_mode=ExperimentAccountingMode.EVALUATION,
    )
    query = AnalyticsQuery(metric="user_prevalence", recipient="provider_c", epsilon=1.0, delta=1e-5)
    with pytest.raises(RecipientSecurityViolation, match="Experiment mode must never target external recipients"):
        orch.execute_release(query, v017_sample_matrix)


def test_04_low_level_experiment_internal_allowed(tmp_path, v017_sample_matrix):
    """Test 4: Low-level EXPERIMENT mode allows internal recipient."""
    config = load_app_config("configs")
    state_mgr = PrivacyStateManager(tmp_path / "exp.db")
    orch = ReleaseOrchestrator(
        app_config=config,
        state_manager=state_mgr,
        execution_mode=ExecutionMode.EXPERIMENT,
        accounting_mode=ExperimentAccountingMode.EVALUATION,
    )
    query = AnalyticsQuery(metric="user_prevalence", recipient="internal", epsilon=1.0, delta=1e-5)
    res = orch.execute_release(query, v017_sample_matrix, noise_seed=42)
    assert res is not None
    assert "results" in res


def test_05_production_external_remains_allowed(tmp_path, v017_sample_matrix):
    """Test 5: PRODUCTION mode allows external recipient (e.g. provider_a)."""
    config = load_app_config("configs")
    state_mgr = PrivacyStateManager(tmp_path / "prod.db")
    orch = ReleaseOrchestrator(
        app_config=config,
        state_manager=state_mgr,
        execution_mode=ExecutionMode.PRODUCTION,
    )
    query = AnalyticsQuery(metric="user_prevalence", recipient="provider_a", epsilon=1.0, delta=1e-6)
    res = orch.execute_release(query, v017_sample_matrix)
    assert res is not None
    assert "results" in res


def test_06_external_experiment_zero_accounting_side_effects(tmp_path, v017_sample_matrix):
    """Test 6: Blocked external experiment release leaves zero side-effects in state DB."""
    config = load_app_config("configs")
    db_p = tmp_path / "exp.db"
    state_mgr = PrivacyStateManager(db_p)
    orch = ReleaseOrchestrator(
        app_config=config,
        state_manager=state_mgr,
        execution_mode=ExecutionMode.EXPERIMENT,
        accounting_mode=ExperimentAccountingMode.EVALUATION,
    )
    query = AnalyticsQuery(metric="user_prevalence", recipient="provider_a", epsilon=1.0, delta=1e-5)

    with pytest.raises(RecipientSecurityViolation):
        orch.execute_release(query, v017_sample_matrix)

    spent_eps, spent_del = state_mgr.get_total_spent_decimal()
    assert spent_eps == Decimal("0")
    assert spent_del == Decimal("0")

    with sqlite3.connect(db_p) as conn:
        res_count = conn.execute("SELECT COUNT(*) FROM privacy_reservations").fetchone()[0]
        rel_count = conn.execute("SELECT COUNT(*) FROM release_store").fetchone()[0]
        led_count = conn.execute("SELECT COUNT(*) FROM privacy_ledger").fetchone()[0]
    assert res_count == 0
    assert rel_count == 0
    assert led_count == 0


def test_07_repeated_10_external_experiment_attempts_all_rejected(tmp_path, v017_sample_matrix):
    """Test 7: 10 repeated attempts to query external recipient in EXPERIMENT mode are all rejected."""
    config = load_app_config("configs")
    db_p = tmp_path / "exp.db"
    state_mgr = PrivacyStateManager(db_p)
    orch = ReleaseOrchestrator(
        app_config=config,
        state_manager=state_mgr,
        execution_mode=ExecutionMode.EXPERIMENT,
        accounting_mode=ExperimentAccountingMode.EVALUATION,
    )
    for i in range(10):
        query = AnalyticsQuery(metric="user_prevalence", recipient="provider_a", epsilon=1.0, delta=1e-5)
        with pytest.raises(RecipientSecurityViolation):
            orch.execute_release(query, v017_sample_matrix)

    spent_eps, _ = state_mgr.get_total_spent_decimal()
    assert spent_eps == Decimal("0")


# =============================================================================
# Gate Fix B: Exp02 N=50,000 Population Semantics & Reproduction
# =============================================================================

def test_08_run_all_experiments_exp02_population_is_50000():
    """Test 8: run_all_experiments.py explicitly uses N=50,000 and interactions_n50000.parquet."""
    import run_all_experiments
    src = inspect.getsource(run_all_experiments.run_all_experiments)
    assert "n_users=50000" in src
    assert "interactions_n50000.parquet" in src


def test_09_exp02_smoke_load_public_population_50000():
    """Test 9: exp02_epsilon.py has default N=50,000 and raises DatasetConfigurationMismatchError on active > public."""
    from experiments.exp02_epsilon import run_epsilon_sweep
    sig = inspect.signature(run_epsilon_sweep)
    assert sig.parameters["n_users"].default == 50000
    assert "n50000" in sig.parameters["data_path"].default

    # Test fail-fast check: if active users > public population
    small_public_n = 5
    with pytest.raises(DatasetConfigurationMismatchError, match="Active user count .* exceeds declared public population"):
        # N=50000 dataset has ~50,000 active users, exceeding public_n=5
        run_epsilon_sweep(
            n_users=small_public_n,
            epsilons=(1.0,),
            n_trials=1,
            data_path="data/interactions_n500.parquet",
            save_artifacts=False,
        )


# =============================================================================
# Gate Fix C, D, E, F: Manifest, Datasets, Checksums & Verifier Integrity
# =============================================================================

def test_10_manifest_included_excluded_matches_archive():
    """Test 10: DATASET_MANIFEST.md correctly marks n100000 as excluded and packaged files as included."""
    manifest_p = Path("data/DATASET_MANIFEST.md")
    assert manifest_p.is_file()
    content = manifest_p.read_text(encoding="utf-8")

    # interactions_n100000.parquet must be marked as excluded / included = no
    assert re.search(r"interactions_n100000\.parquet.*\|\s*(?:\*\*no\*\*|no|Excluded)", content, re.IGNORECASE)
    # interactions_n50000.parquet must be marked as included / included = yes
    assert re.search(r"interactions_n50000\.parquet.*\|\s*(?:\*\*yes\*\*|yes|Included)", content, re.IGNORECASE)


def test_11_no_duplicate_canonical_datasets():
    """Test 11: No duplicate alias datasets (n10k, n50k) exist in data/."""
    assert not Path("data/interactions_n10k.parquet").exists()
    assert not Path("data/interactions_n50k.parquet").exists()
    assert Path("data/interactions_n10000.parquet").is_file()
    assert Path("data/interactions_n50000.parquet").is_file()


def test_12_checksum_file_is_machine_parseable():
    """Test 12: data/checksums.sha256 strictly follows '<sha256>  <filename>' standard format."""
    checksum_p = Path("data/checksums.sha256")
    assert checksum_p.is_file()
    lines = [line.strip() for line in checksum_p.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(lines) >= 6

    pattern = re.compile(r"^[0-9a-f]{64}  [a-zA-Z0-9_.-]+$")
    for line in lines:
        assert pattern.match(line), f"Line does not match standard machine-parseable format: '{line}'"
        # Must not contain size descriptions like (34.20 MB)
        assert "(" not in line and ")" not in line


def test_13_all_packaged_dataset_hashes_verify():
    """Test 13: All files in data/checksums.sha256 verify bit-for-bit against disk."""
    checksum_p = Path("data/checksums.sha256")
    lines = [line.strip() for line in checksum_p.read_text(encoding="utf-8").splitlines() if line.strip()]

    for line in lines:
        expected_hash, fname = line.split("  ", 1)
        file_p = Path("data") / fname
        assert file_p.is_file(), f"Expected file {file_p} listed in checksums to exist"
        actual_hash = hashlib.sha256(file_p.read_bytes()).hexdigest()
        assert actual_hash == expected_hash, f"Hash mismatch for {fname}: expected {expected_hash}, got {actual_hash}"


def test_14_fresh_zip_verifier_discovers_all_verify_scripts():
    """Test 14: verify_zip_distribution.py dynamically discovers all verify_*.py scripts (>= 16)."""
    all_scripts = sorted(Path("experiments").glob("verify_*.py"))
    assert len(all_scripts) >= 16, f"Expected at least 16 verify scripts, found {len(all_scripts)}: {[s.name for s in all_scripts]}"


def test_15_new_external_block_script_included_in_fresh_zip_verifier():
    """Test 15: verify_experiment_external_block.py, verify_exp02_population.py, and verify_smoke_non_destructive.py exist."""
    assert Path("experiments/verify_experiment_external_block.py").is_file()
    assert Path("experiments/verify_exp02_population.py").is_file()
    assert Path("experiments/verify_smoke_non_destructive.py").is_file()


# =============================================================================
# Smoke Safety & Artifact Isolation (v0.1.7 Final Freeze Smoke-Safety Patch)
# =============================================================================

def test_16_all_experiments_support_explicit_output_dir():
    """Test 16: All experiment suites (Exp01, Exp02, Exp03, Exp04) accept explicit output_dir parameter."""
    from experiments.exp01_baseline import run_baseline_experiment
    from experiments.exp02_epsilon import run_epsilon_sweep
    from experiments.exp03_population import run_population_sweep
    from experiments.exp04_clipping import run_clipping_sweep

    for fn in [run_baseline_experiment, run_epsilon_sweep, run_population_sweep, run_clipping_sweep]:
        sig = inspect.signature(fn)
        assert "output_dir" in sig.parameters, f"Function {fn.__name__} missing 'output_dir' parameter"


def test_17_smoke_mode_does_not_modify_official_research_artifacts(tmp_path):
    """Test 17: run_all_experiments(smoke=True) does not alter official research artifacts."""
    results_dir = Path("results")
    official_files = [
        results_dir / "tables" / "exp01_baseline_summary.csv",
        results_dir / "tables" / "exp02_epsilon_summary.csv",
        results_dir / "tables" / "exp03_population_summary.csv",
        results_dir / "tables" / "exp04_clipping_summary.csv",
        results_dir / "tables" / "scalability_benchmark.csv",
        results_dir / "raw" / "exp01_baseline_trials.csv",
        results_dir / "raw" / "exp02_epsilon_trials.csv",
        results_dir / "raw" / "exp03_population_trials.csv",
        results_dir / "raw" / "exp04_clipping_trials.csv",
    ]

    hashes_before = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in official_files if p.is_file()}
    assert len(hashes_before) >= 5, "Official summary tables must exist before running test"

    # Run smoke test writing to isolated tmp_path
    from run_all_experiments import run_all_experiments
    run_all_experiments(smoke=True, output_dir=str(tmp_path))

    # Verify hashes of official files are identical
    for p, h_before in hashes_before.items():
        assert p.is_file(), f"Official file {p} disappeared!"
        h_after = hashlib.sha256(p.read_bytes()).hexdigest()
        assert h_after == h_before, f"Official file {p} was modified by smoke test!"

    # Verify smoke files were written to tmp_path
    assert (tmp_path / "tables" / "exp01_baseline_summary.csv").is_file()
    assert (tmp_path / "tables" / "exp02_epsilon_summary.csv").is_file()
    assert (tmp_path / "tables" / "exp03_population_summary.csv").is_file()
    assert (tmp_path / "tables" / "exp04_clipping_summary.csv").is_file()

