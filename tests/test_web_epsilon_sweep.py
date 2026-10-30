"""Tests for Multi-Trial Epsilon-Utility Sweep Experiment and Isolation."""

import pytest
from starlette.testclient import TestClient

from private_llm_analytics.config import load_app_config
from private_llm_analytics.service.analytics import AnalyticsService
from private_llm_analytics.web.adapters import WebAnalyticsAdapter
from private_llm_analytics.web.app import create_app


@pytest.fixture
def sweep_client(tmp_path):
    config = load_app_config("configs")
    temp_db = tmp_path / "test_sweep_privacy_state.db"
    config.privacy.release.state_db_path = str(temp_db)

    service = AnalyticsService(config=config)
    adapter = WebAnalyticsAdapter(service_or_config_dir=service)
    app = create_app(adapter=adapter)
    return TestClient(app)


def test_01_epsilon_sweep_executes_and_preserves_production_ledger(sweep_client):
    """Verify epsilon sweep runs multiple trials without modifying production ledger or lifetime budget."""
    sweep_client.post("/api/dataset/demo")

    # Incur an initial production release to establish baseline
    init_rel = sweep_client.post("/api/release", json={
        "metric": "interaction_volume",
        "recipient": "provider_a",
        "epsilon": 1.0,
        "delta": 1e-6,
    })
    assert init_rel.status_code == 200

    # Read production privacy status
    status_before = sweep_client.get("/api/privacy/status?recipient=provider_a").json()
    eps_before = status_before["effective_epsilon"]
    releases_before = status_before["release_count"]

    # Execute multi-trial epsilon sweep across 4 candidate epsilons with 3 trials each
    sweep_res = sweep_client.post("/api/evaluation/sweep", json={
        "metric": "interaction_volume",
        "recipient": "internal",
        "candidate_epsilons": [0.25, 0.5, 1.0, 2.0],
        "n_trials": 3,
    })
    assert sweep_res.status_code == 200
    sweep_data = sweep_res.json()

    assert sweep_data["metric"] == "interaction_volume"
    assert len(sweep_data["points"]) == 4

    for pt in sweep_data["points"]:
        assert pt["trials_count"] == 3
        assert pt["mean_error"] >= 0.0
        assert pt["std_error"] >= 0.0

    # Verify production privacy state is UNTOUCHED
    status_after = sweep_client.get("/api/privacy/status?recipient=provider_a").json()
    assert status_after["effective_epsilon"] == eps_before
    assert status_after["release_count"] == releases_before


def test_02_external_recipients_forbidden_from_epsilon_sweep(sweep_client):
    """Verify external callers receive 403 Forbidden when requesting epsilon sweep."""
    sweep_client.post("/api/dataset/demo")

    res = sweep_client.post("/api/evaluation/sweep", json={
        "metric": "interaction_volume",
        "recipient": "provider_a",
        "candidate_epsilons": [0.5, 1.0],
        "n_trials": 2,
    })
    assert res.status_code == 403
