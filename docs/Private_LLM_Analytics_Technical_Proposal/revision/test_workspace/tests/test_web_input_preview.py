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


def test_01_trusted_internal_can_view_input_preview(web_client):
    """Verify trusted internal admin can inspect first 10 rows of input telemetry."""
    # Load demo data first
    res_load = web_client.post("/api/dataset/demo")
    assert res_load.status_code == 200

    # Request preview as internal
    res_prev = web_client.get("/api/dataset/preview?recipient=internal&limit=10")
    assert res_prev.status_code == 200
    data = res_prev.json()

    assert data["total_rows"] > 0
    assert data["preview_limit"] == 10
    assert len(data["rows"]) == 10
    row_0 = data["rows"][0]
    assert "user_id" in row_0
    assert "timestamp" in row_0
    assert "provider_name" in row_0
    assert "task_name" in row_0
    assert "success" in row_0


def test_02_external_recipients_strictly_forbidden_from_input_preview(web_client):
    """Verify external recipients receive 403 Forbidden when attempting to view raw input preview."""
    web_client.post("/api/dataset/demo")

    external_recipients = ["provider_a", "provider_b", "provider_c"]
    for rec in external_recipients:
        res = web_client.get(f"/api/dataset/preview?recipient={rec}&limit=10")
        assert res.status_code == 403
        data = res.json()
        err_lower = data["error"].lower()
        assert "forbidden" in err_lower or "unauthorized" in err_lower or "violation" in err_lower
