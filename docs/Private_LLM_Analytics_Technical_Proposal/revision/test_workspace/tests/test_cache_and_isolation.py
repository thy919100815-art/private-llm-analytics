"""Unit tests for release cache stability and recipient isolation."""

import numpy as np
import pytest

from private_llm_analytics.aggregation.user_matrix import UserMatrix, build_cell_definitions
from private_llm_analytics.config.loader import load_app_config
from private_llm_analytics.privacy.ledger import PrivacyLedger
from private_llm_analytics.release.cache import PersistentReleaseStore
from private_llm_analytics.release.recipients import RecipientSecurityViolation
from private_llm_analytics.release.release import ReleaseOrchestrator
from private_llm_analytics.planner.query import AnalyticsQuery


@pytest.fixture
def dummy_user_matrix() -> UserMatrix:
    provs = ["provider_a", "provider_b", "provider_c"]
    tsks = ["technical", "document", "analysis"]
    cell_names, cell_map = build_cell_definitions(provs, tsks)
    # 5 users, 9 cells
    matrix = np.ones((5, 9), dtype=np.float64) * 10.0
    return UserMatrix(
        matrix=matrix,
        user_ids=[f"U{i}" for i in range(5)],
        providers=provs,
        tasks=tsks,
        cell_names=cell_names,
        cell_map=cell_map,
    )


def test_release_cache_reproducibility(tmp_path, dummy_user_matrix):
    config = load_app_config("configs")
    cache = PersistentReleaseStore(tmp_path / "cache.db")
    ledger = PrivacyLedger(tmp_path / "ledger.db")
    orchestrator = ReleaseOrchestrator(app_config=config, cache=cache, ledger=ledger)

    query = AnalyticsQuery(
        metric="user_normalized_share",
        recipient="internal",
        epsilon=1.0,
        delta=1e-6,
    )

    # First execution: miss
    res1 = orchestrator.execute_release(query, dummy_user_matrix)
    # Second execution: hit
    res2 = orchestrator.execute_release(query, dummy_user_matrix)

    # Values must be identical down to exact floats
    assert res1["results"]["cell_shares"] == res2["results"]["cell_shares"]


def test_external_recipient_exact_denial(tmp_path, dummy_user_matrix):
    config = load_app_config("configs")
    cache = PersistentReleaseStore(tmp_path / "cache.db")
    ledger = PrivacyLedger(tmp_path / "ledger.db")
    orchestrator = ReleaseOrchestrator(app_config=config, cache=cache, ledger=ledger)

    query = AnalyticsQuery(
        metric="user_prevalence",
        recipient="provider_a",
    )

    # Attempting to request exact ground truth for external recipient must raise error
    with pytest.raises(RecipientSecurityViolation, match="strictly forbidden from receiving exact results"):
        orchestrator.execute_release(
            query,
            dummy_user_matrix,
            include_exact=True,
        )


def test_provider_isolation(tmp_path, dummy_user_matrix):
    config = load_app_config("configs")
    cache = PersistentReleaseStore(tmp_path / "cache.db")
    ledger = PrivacyLedger(tmp_path / "ledger.db")
    orchestrator = ReleaseOrchestrator(app_config=config, cache=cache, ledger=ledger)

    query_a = AnalyticsQuery(
        metric="user_prevalence",
        recipient="provider_a",
    )
    res_a = orchestrator.execute_release(query_a, dummy_user_matrix)

    rates = res_a["results"]["prevalence_rates"]

    # Provider A must only see provider_a cells
    for cell_key in rates.keys():
        assert cell_key.startswith("provider_a:"), f"Leaked cell {cell_key} to provider_a!"
        assert not cell_key.startswith("provider_b:")
        assert not cell_key.startswith("provider_c:")

    # Now query user_normalized_share to test provider_distribution isolation
    query_share = AnalyticsQuery(
        metric="user_normalized_share",
        recipient="provider_a",
    )
    res_share = orchestrator.execute_release(query_share, dummy_user_matrix)
    # Default report for external provider must not include cross-provider global market share
    assert "provider_distribution" not in res_share["results"]

