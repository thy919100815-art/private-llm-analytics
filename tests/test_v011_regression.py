"""Comprehensive regression test suite for Private LLM Analytics v0.1.1.

Validates all 15 audit requirements from Private_LLM_Analytics_v0.1_AUDIT.md.
"""

import numpy as np
import pytest
import sqlite3

from private_llm_analytics.aggregation.user_matrix import UserMatrix, build_cell_definitions
from private_llm_analytics.config.loader import load_app_config
from private_llm_analytics.config.models import (
    AppConfig,
    ContributionConfig,
    GlobalBudgetConfig,
    PrivacyAccountantConfig,
    RecipientPolicy,
)
from private_llm_analytics.config.validator import ConfigValidationError, validate_app_config
from private_llm_analytics.evaluation.privacy_utility import run_repeated_noise_trials
from private_llm_analytics.metrics.interaction_volume import InteractionVolumeMetric
from private_llm_analytics.metrics.normalized_share import UserNormalizedShareMetric
from private_llm_analytics.metrics.prevalence import UserPrevalenceMetric
from private_llm_analytics.planner.query import AnalyticsQuery
from private_llm_analytics.privacy.accountant import BasicAccountant
from private_llm_analytics.privacy.budget import PrivacyBudget
from private_llm_analytics.privacy.ledger import PrivacyLedger
from private_llm_analytics.release.cache import PersistentReleaseStore
from private_llm_analytics.release.recipients import (
    RecipientSecurityViolation,
    filter_release_for_recipient,
)
from private_llm_analytics.release.release import ReleaseOrchestrator


@pytest.fixture
def sample_user_matrix() -> UserMatrix:
    provs = ["provider_a", "provider_b", "provider_c"]
    tsks = ["technical", "document", "analysis"]
    cell_names, cell_map = build_cell_definitions(provs, tsks)
    # 5 users, each with 10 interactions across all 9 cells
    matrix = np.ones((5, 9), dtype=np.float64) * 10.0
    return UserMatrix(
        matrix=matrix,
        user_ids=[f"U00{i}" for i in range(5)],
        providers=provs,
        tasks=tsks,
        cell_names=cell_names,
        cell_map=cell_map,
        public_population_size=50000,
    )


# --- Audit Item C1: No private active user count leaked externally ---
def test_regression_01_external_output_has_no_active_user_count(tmp_path, sample_user_matrix):
    config = load_app_config("configs")
    cache = PersistentReleaseStore(tmp_path / "cache.db")
    ledger = PrivacyLedger(tmp_path / "ledger.db")
    orchestrator = ReleaseOrchestrator(app_config=config, cache=cache, ledger=ledger)

    query_ext = AnalyticsQuery(metric="user_prevalence", recipient="provider_a")
    res_ext = orchestrator.execute_release(query_ext, sample_user_matrix)

    # External recipient MUST NOT receive active user count
    assert "active_user_count" not in res_ext
    assert "n_users" not in res_ext
    # But public population size is allowed if defined
    assert res_ext.get("public_population_size") == 50000

    # Internal recipient receives active user count for administrative audit
    query_int = AnalyticsQuery(metric="user_prevalence", recipient="internal")
    res_int = orchestrator.execute_release(query_int, sample_user_matrix)
    assert res_int.get("active_user_count") == 5


# --- Audit Item C1: Public population denominator used for prevalence ---
def test_regression_02_prevalence_uses_public_population_denominator(sample_user_matrix):
    metric = UserPrevalenceMetric()
    rates = metric.compute_exact_rate(sample_user_matrix)
    # 5 active users out of 50,000 public population -> rate = 5 / 50000 = 0.0001
    assert np.allclose(rates, 5.0 / 50000.0)

    # Post process with DP vector
    dp_counts = np.array([5.0] * 9)
    pp = metric.post_process(dp_counts, sample_user_matrix)
    assert pp["denominator_used"] == 50000
    for rate in pp["prevalence_rates"].values():
        assert abs(rate - 0.0001) < 1e-6


# --- Audit Item C2: Query cannot override clip_norm ---
def test_regression_03_query_cannot_override_clip_norm_sensitivity():
    # AnalyticsQuery in v0.1.1 strictly does NOT support clip_norm override
    with pytest.raises(Exception):
        AnalyticsQuery(
            metric="interaction_volume",
            recipient="internal",
            clip_norm=5.0,  # Invalid query field
        )


# --- Audit Item C3: norm_bound strictly requires 1.0 ---
def test_regression_04_norm_bound_not_one_rejected():
    # Pydantic validator rejection
    with pytest.raises(ValueError, match="strictly requires norm_bound == 1.0"):
        ContributionConfig(mode="l1_normalized", norm_bound=0.5)

    # Metric constructor rejection
    with pytest.raises(ValueError, match="strictly requires norm_bound == 1.0"):
        UserNormalizedShareMetric(norm_bound=0.5)

    # Valid norm_bound == 1.0 works cleanly
    valid_cfg = ContributionConfig(mode="l1_normalized", norm_bound=1.0)
    assert valid_cfg.norm_bound == 1.0
    valid_metric = UserNormalizedShareMetric(norm_bound=1.0)
    assert valid_metric.norm_bound == 1.0


# --- Audit Item C5: Unauthorized exact request consumes zero budget ---
def test_regression_05_unauthorized_exact_consumes_zero_budget(tmp_path, sample_user_matrix):
    config = load_app_config("configs")
    cache = PersistentReleaseStore(tmp_path / "cache.db")
    ledger = PrivacyLedger(tmp_path / "ledger.db")
    orchestrator = ReleaseOrchestrator(app_config=config, cache=cache, ledger=ledger)

    query = AnalyticsQuery(metric="user_prevalence", recipient="provider_a")
    with pytest.raises(RecipientSecurityViolation):
        orchestrator.execute_release(query, sample_user_matrix, include_exact=True)

    assert orchestrator.accountant.spent_epsilon == 0.0
    assert orchestrator.accountant.spent_delta == 0.0


# --- Audit Item C5: Unauthorized exact request writes no cache ---
def test_regression_06_unauthorized_exact_writes_no_cache(tmp_path, sample_user_matrix):
    config = load_app_config("configs")
    cache_path = tmp_path / "cache.db"
    cache = PersistentReleaseStore(cache_path)
    ledger = PrivacyLedger(tmp_path / "ledger.db")
    orchestrator = ReleaseOrchestrator(app_config=config, cache=cache, ledger=ledger)

    query = AnalyticsQuery(metric="user_prevalence", recipient="provider_a")
    with pytest.raises(RecipientSecurityViolation):
        orchestrator.execute_release(query, sample_user_matrix, include_exact=True)

    with sqlite3.connect(cache_path) as conn:
        count = conn.cursor().execute("SELECT count(*) FROM release_store").fetchone()[0]
    assert count == 0


# --- Audit Item C5: Unauthorized exact request writes no ledger ---
def test_regression_07_unauthorized_exact_writes_no_ledger(tmp_path, sample_user_matrix):
    config = load_app_config("configs")
    ledger_path = tmp_path / "ledger.db"
    cache = PersistentReleaseStore(tmp_path / "cache.db")
    ledger = PrivacyLedger(ledger_path)
    orchestrator = ReleaseOrchestrator(app_config=config, cache=cache, ledger=ledger)

    query = AnalyticsQuery(metric="user_prevalence", recipient="provider_a")
    with pytest.raises(RecipientSecurityViolation):
        orchestrator.execute_release(query, sample_user_matrix, include_exact=True)

    with sqlite3.connect(ledger_path) as conn:
        count = conn.cursor().execute("SELECT count(*) FROM privacy_ledger").fetchone()[0]
    assert count == 0


# --- Audit Item C4: Process restart returns identical DP payload ---
def test_regression_08_restart_returns_identical_dp_payload(tmp_path, sample_user_matrix):
    config = load_app_config("configs")
    cache_path = tmp_path / "cache.db"
    ledger_path = tmp_path / "ledger.db"

    # Process 1
    cache1 = PersistentReleaseStore(cache_path)
    ledger1 = PrivacyLedger(ledger_path)
    orch1 = ReleaseOrchestrator(app_config=config, cache=cache1, ledger=ledger1)

    query = AnalyticsQuery(metric="user_normalized_share", recipient="internal")
    res1 = orch1.execute_release(query, sample_user_matrix)

    # Process 2 (simulating fresh restart, memory cleared)
    cache2 = PersistentReleaseStore(cache_path)
    ledger2 = PrivacyLedger(ledger_path)
    orch2 = ReleaseOrchestrator(app_config=config, cache=cache2, ledger=ledger2)

    res2 = orch2.execute_release(query, sample_user_matrix)

    assert res2["cache_hit"] is True
    assert res1["results"]["cell_shares"] == res2["results"]["cell_shares"]


# --- Audit Item C4: Process restart restores accountant spend ---
def test_regression_09_restart_restores_accountant_spend(tmp_path, sample_user_matrix):
    config = load_app_config("configs")
    cache_path = tmp_path / "cache.db"
    ledger_path = tmp_path / "ledger.db"

    # Process 1 consumes budget
    cache1 = PersistentReleaseStore(cache_path)
    ledger1 = PrivacyLedger(ledger_path)
    orch1 = ReleaseOrchestrator(app_config=config, cache=cache1, ledger=ledger1)

    query = AnalyticsQuery(metric="user_prevalence", recipient="internal", epsilon=1.5, delta=1e-6)
    orch1.execute_release(query, sample_user_matrix)
    assert orch1.accountant.spent_epsilon == 1.5

    # Process 2 restores spent budget from SQLite ledger
    ledger2 = PrivacyLedger(ledger_path)
    accountant2 = BasicAccountant(
        global_budget=PrivacyBudget(epsilon=4.0, delta=1e-5),
        ledger=ledger2,
    )
    assert accountant2.spent_epsilon == 1.5
    assert accountant2.spent_delta == 1e-6


# --- Audit Item H2: Provider isolation blocks competitor cells ---
def test_regression_10_provider_isolation_blocks_competitors(tmp_path, sample_user_matrix):
    config = load_app_config("configs")
    cache = PersistentReleaseStore(tmp_path / "cache.db")
    ledger = PrivacyLedger(tmp_path / "ledger.db")
    orchestrator = ReleaseOrchestrator(app_config=config, cache=cache, ledger=ledger)

    query = AnalyticsQuery(metric="user_prevalence", recipient="provider_a")
    res = orchestrator.execute_release(query, sample_user_matrix)

    rates = res["results"]["prevalence_rates"]
    for k in rates.keys():
        assert k.startswith("provider_a:")
        assert "provider_b" not in k
        assert "provider_c" not in k


# --- Audit Item H2: Provider default report excludes cross-provider market share ---
def test_regression_11_provider_default_no_cross_provider_market_share(tmp_path, sample_user_matrix):
    config = load_app_config("configs")
    cache = PersistentReleaseStore(tmp_path / "cache.db")
    ledger = PrivacyLedger(tmp_path / "ledger.db")
    orchestrator = ReleaseOrchestrator(app_config=config, cache=cache, ledger=ledger)

    query = AnalyticsQuery(metric="user_normalized_share", recipient="provider_a")
    res = orchestrator.execute_release(query, sample_user_matrix)

    # Provider distribution across competitors is omitted
    assert "provider_distribution" not in res["results"]

    # Cell shares for provider_a are re-normalized to 1.0 within provider_a
    shares = res["results"]["cell_shares"]
    assert len(shares) == 3
    assert abs(sum(shares.values()) - 1.0) < 1e-6


# --- Audit Item M1: Prevalence evaluation does not report Total Variation distance ---
def test_regression_12_prevalence_evaluation_no_total_variation(sample_user_matrix):
    # Prevalence (overlapping binary indicators) -> No TV distance in trials or summary
    prev_metric = UserPrevalenceMetric()
    prev_df, prev_summary = run_repeated_noise_trials(
        sample_user_matrix,
        prev_metric,
        mechanism_type="gaussian",
        epsilon=1.0,
        n_trials=5,
    )
    assert "total_variation" not in prev_df.columns
    assert "tv_mean" not in prev_summary
    assert "mae" in prev_df.columns
    assert "mae_pp_mean" in prev_summary

    # Normalized share (simplex probability distribution) -> TV distance included
    share_metric = UserNormalizedShareMetric()
    share_df, share_summary = run_repeated_noise_trials(
        sample_user_matrix,
        share_metric,
        mechanism_type="gaussian",
        epsilon=1.0,
        n_trials=5,
    )
    assert "total_variation" in share_df.columns
    assert "tv_mean" in share_summary



# --- Audit Item H4: accountant.type: rdp rejected ---
def test_regression_13_unsupported_accountant_rdp_rejected():
    with pytest.raises(ValueError, match="strictly supports only 'basic'"):
        PrivacyAccountantConfig(type="rdp")


# --- Audit Item C4 / H3: Repeated query cache hit consumes budget only once ---
def test_regression_14_repeated_exact_query_consumes_budget_once(tmp_path, sample_user_matrix):
    config = load_app_config("configs")
    cache = PersistentReleaseStore(tmp_path / "cache.db")
    ledger = PrivacyLedger(tmp_path / "ledger.db")
    orchestrator = ReleaseOrchestrator(app_config=config, cache=cache, ledger=ledger)

    query = AnalyticsQuery(metric="user_prevalence", recipient="internal", epsilon=0.5, delta=1e-6)

    # Query 1: executes and consumes 0.5 epsilon
    res1 = orchestrator.execute_release(query, sample_user_matrix)
    assert res1["cache_hit"] is False
    assert orchestrator.accountant.spent_epsilon == 0.5

    # Query 2 (identical query & dataset): cache hit, spent epsilon remains 0.5!
    res2 = orchestrator.execute_release(query, sample_user_matrix)
    assert res2["cache_hit"] is True
    assert orchestrator.accountant.spent_epsilon == 0.5


# --- Audit Item E2: Cross-config validation runs in load_app_config ---
def test_regression_15_cross_config_validation_runs_in_load_app_config():
    # Valid config loads and passes cross validation
    app_cfg = load_app_config("configs")
    assert app_cfg is not None

    # Tampered config with recipient referencing unknown provider raises ConfigValidationError
    tampered_cfg = app_cfg.model_copy(deep=True)
    tampered_cfg.recipients.recipients["provider_a"].providers = ["nonexistent_provider"]

    with pytest.raises(ConfigValidationError, match="references unknown provider 'nonexistent_provider'"):
        validate_app_config(tampered_cfg)



# --- Audit Item H1: Recipient dp_results=False policy denial ---
def test_regression_16_disallow_dp_results_denied(tmp_path, sample_user_matrix):
    config = load_app_config("configs")
    # Set provider_a dp_results = False
    config.recipients.recipients["provider_a"].dp_results = False

    cache = PersistentReleaseStore(tmp_path / "cache.db")
    ledger = PrivacyLedger(tmp_path / "ledger.db")
    orchestrator = ReleaseOrchestrator(app_config=config, cache=cache, ledger=ledger)

    query = AnalyticsQuery(metric="user_prevalence", recipient="provider_a")
    with pytest.raises(RecipientSecurityViolation, match="dp_results=False"):
        orchestrator.execute_release(query, sample_user_matrix)


# --- Audit Item H1: Recipient exact_results=False denied even for trusted ---
def test_regression_17_disallow_exact_results_denied_for_internal(tmp_path, sample_user_matrix):
    config = load_app_config("configs")
    # Explicitly deny exact_results for internal
    config.recipients.recipients["internal"].exact_results = False

    cache = PersistentReleaseStore(tmp_path / "cache.db")
    ledger = PrivacyLedger(tmp_path / "ledger.db")
    orchestrator = ReleaseOrchestrator(app_config=config, cache=cache, ledger=ledger)

    query = AnalyticsQuery(metric="user_prevalence", recipient="internal")
    with pytest.raises(RecipientSecurityViolation, match="strictly forbidden from receiving exact results"):
        orchestrator.execute_release(query, sample_user_matrix, include_exact=True)



# --- Audit Item M2: Interaction volume error decomposition ---
def test_regression_18_interaction_volume_error_decomposition(sample_user_matrix):
    metric = InteractionVolumeMetric(clip_norm=15.0)
    raw = metric.compute_raw_exact(sample_user_matrix)
    clipped = metric.compute_clipped_exact(sample_user_matrix)

    # 5 users * 10 = 50 in each cell.
    # L2 norm for each user across 9 cells = sqrt(9 * 10^2) = 30.
    # Scaled to C=15 -> each user contribution is halved to 5 in each cell.
    assert np.allclose(clipped, raw * 0.5)

    dp_sim = clipped + 3.0
    decomp = metric.decompose_error(dp_sim, sample_user_matrix)
    assert np.allclose(decomp["total_error"], decomp["clipping_bias"] + decomp["dp_noise_error"])
