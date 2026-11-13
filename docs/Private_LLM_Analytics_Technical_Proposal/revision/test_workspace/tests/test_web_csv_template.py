"""Tests for CSV Sample Template download and re-upload parsing."""

import io
import pytest
from starlette.testclient import TestClient

from private_llm_analytics.config import load_app_config
from private_llm_analytics.service.analytics import AnalyticsService
from private_llm_analytics.web.adapters import WebAnalyticsAdapter
from private_llm_analytics.web.app import create_app


@pytest.fixture
def web_client(tmp_path):
    config = load_app_config("configs")
    temp_db = tmp_path / "test_csv_privacy_state.db"
    config.privacy.release.state_db_path = str(temp_db)

    service = AnalyticsService(config=config)
    adapter = WebAnalyticsAdapter(service_or_config_dir=service)
    app = create_app(adapter=adapter)
    return TestClient(app)


def test_01_download_sample_template_succeeds(web_client):
    """Verify standard CSV template is downloadable with correct filename and headers."""
    res = web_client.get("/api/dataset/sample-template")
    assert res.status_code == 200
    assert "text/csv" in res.headers.get("content-type", "")
    assert "sample_private_llm_analytics.csv" in res.headers.get("content-disposition", "")

    csv_text = res.text
    first_line = csv_text.splitlines()[0]
    expected_headers = ["interaction_id", "user_id", "timestamp", "provider", "task", "success"]
    for h in expected_headers:
        assert h in first_line

    # Verify sample rows exist
    lines = csv_text.strip().splitlines()
    assert len(lines) >= 6


def test_02_reupload_sample_template_parses_successfully(web_client):
    """Verify that the downloaded sample CSV template can be immediately uploaded and parsed."""
    # 1. Download
    res_dl = web_client.get("/api/dataset/sample-template")
    assert res_dl.status_code == 200
    csv_bytes = res_dl.content

    # 2. Upload
    files = {"file": ("test_upload.csv", csv_bytes, "text/csv")}
    res_up = web_client.post("/api/dataset/upload", files=files)
    assert res_up.status_code == 200
    data = res_up.json()

    assert data["loaded"] is True
    assert data["rows"] >= 5
    assert data["user_count"] >= 3
    assert data["has_success_field"] is True
    assert len(data["quality_checks"]) >= 5
    for qc in data["quality_checks"]:
        assert qc["passed"] is True
