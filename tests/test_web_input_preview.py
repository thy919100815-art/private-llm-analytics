"""Tests for Input Data Preview access permissions and format."""

import pytest
from starlette.testclient import TestClient

from private_llm_analytics.config import load_app_config
from private_llm_analytics.service.analytics import AnalyticsService
from private_llm_analytics.web.adapters import WebAnalyticsAdapter
from private_llm_analytics.web.app import create_app


@pytest.fixture
def web_client(tmp_path):
    config = load_app_config("configs")
    temp_db = tmp_path / "test_preview_privacy_state.db"
    config.privacy.release.state_db_path = str(temp_db)

    service = AnalyticsService(config=config)
    adapter = WebAnalyticsAdapter(service_or_config_dir=service)
    app = create_app(adapter=adapter)
    return TestClient(app)


def test_01_trusted_local_preview_without_recipient_param(web_client):
    """Verify trusted local admin can inspect up to 20 rows of input telemetry without recipient parameter."""
    res_load = web_client.post("/api/dataset/demo")
    assert res_load.status_code == 200

    # Request local preview without recipient parameter (trusted local admin preview)
    res_prev = web_client.get("/api/dataset/preview?limit=20")
    assert res_prev.status_code == 200
    data = res_prev.json()

    assert data["total_rows"] > 0
    assert data["preview_limit"] == 20
    assert len(data["rows"]) == 20
    row_0 = data["rows"][0]
    assert "user_id" in row_0
    assert "timestamp" in row_0
    assert "provider_name" in row_0
    assert "task_name" in row_0
    assert "success" in row_0


def test_02_local_preview_does_not_consume_dp_budget(web_client):
    """Verify data preview does not consume differential privacy budget or create release records."""
    web_client.post("/api/dataset/demo")

    # Check privacy status before preview
    status_before = web_client.get("/api/privacy/status?recipient=provider_a").json()
    eps_before = status_before["effective_epsilon"]
    rel_cnt_before = status_before["release_count"]

    # Call preview multiple times
    for _ in range(5):
        res = web_client.get("/api/dataset/preview?limit=20")
        assert res.status_code == 200

    # Check privacy status after preview: strictly 0 cost
    status_after = web_client.get("/api/privacy/status?recipient=provider_a").json()
    assert status_after["effective_epsilon"] == eps_before
    assert status_after["release_count"] == rel_cnt_before


def test_03_preview_succeeds_even_when_external_recipient_in_ui(web_client):
    """Verify local admin preview is not blocked when UI has external recipient selected."""
    web_client.post("/api/dataset/demo")

    external_recipients = ["provider_a", "provider_b", "provider_c"]
    for rec in external_recipients:
        res = web_client.get(f"/api/dataset/preview?recipient={rec}&limit=20")
        assert res.status_code == 200
        data = res.json()
        assert len(data["rows"]) == 20
