"""Tests for DP Utility & Accuracy Evaluation Engine and Permissions."""

import pytest
from starlette.testclient import TestClient

from private_llm_analytics.config import load_app_config
from private_llm_analytics.service.analytics import AnalyticsService
from private_llm_analytics.web.adapters import WebAnalyticsAdapter
from private_llm_analytics.web.app import create_app


@pytest.fixture
def web_client(tmp_path):
    config = load_app_config("configs")
    temp_db = tmp_path / "test_eval_privacy_state.db"
    config.privacy.release.state_db_path = str(temp_db)

    service = AnalyticsService(config=config)
    adapter = WebAnalyticsAdapter(service_or_config_dir=service)
    app = create_app(adapter=adapter)
    return TestClient(app)


def test_01_interaction_volume_accuracy_evaluation(web_client):
    """Verify decomposition (Raw / Clipped / DP) and errors for Interaction Volume."""
    web_client.post("/api/dataset/demo")

    # Execute release
    rel_res = web_client.post("/api/release", json={
        "metric": "interaction_volume",
        "recipient": "internal",
        "epsilon": 1.0,
        "delta": 1e-6,
    })
    assert rel_res.status_code == 200
    release_id = rel_res.json()["release_id"]

    # Request accuracy evaluation
    eval_res = web_client.get(f"/api/evaluation/accuracy?release_id={release_id}&recipient=internal")
    assert eval_res.status_code == 200
    data = eval_res.json()

    assert data["metric"] == "interaction_volume"
    assert data["primary_metric_name"] == "平均相对误差"
    assert data["mae"] >= 0.0
    assert data["rmse"] >= 0.0
    assert data["mean_relative_error_pct"] >= 0.0
    assert len(data["rows"]) == 9

    row = data["rows"][0]
    assert "raw_exact" in row
    assert "clipped_exact" in row
    assert "dp_value" in row
    assert "clipping_bias" in row
    assert "dp_noise_error" in row
    assert "total_error" in row
    assert "relative_error_pct" in row
    # Total error must equal clipping_bias + dp_noise_error
    assert abs(row["total_error"] - (row["clipping_bias"] + row["dp_noise_error"])) < 0.2


def test_02_prevalence_and_success_rate_accuracy_evaluation(web_client):
    """Verify percentage-point error metrics for Prevalence and Success Rate."""
    web_client.post("/api/dataset/demo")

    # Prevalence
    rel_prev = web_client.post("/api/release", json={
        "metric": "user_prevalence",
        "recipient": "internal",
        "epsilon": 1.0,
        "delta": 1e-6,
    })
    rel_id_prev = rel_prev.json()["release_id"]
    eval_prev = web_client.get(f"/api/evaluation/accuracy?release_id={rel_id_prev}&recipient=internal")
    assert eval_prev.status_code == 200
    p_data = eval_prev.json()
    assert p_data["primary_metric_name"] == "平均百分点误差"
    assert p_data["mean_pp_error"] >= 0.0

    # Success rate
    rel_sr = web_client.post("/api/release", json={
        "metric": "success_rate",
        "recipient": "internal",
        "epsilon": 1.0,
        "delta": 1e-6,
    })
    rel_id_sr = rel_sr.json()["release_id"]
    eval_sr = web_client.get(f"/api/evaluation/accuracy?release_id={rel_id_sr}&recipient=internal")
    assert eval_sr.status_code == 200
    sr_data = eval_sr.json()
    assert sr_data["primary_metric_name"] == "平均百分点误差"
    assert sr_data["mean_pp_error"] >= 0.0


def test_03_top_k_tasks_accuracy_evaluation(web_client):
    """Verify Precision@K and Jaccard similarity for Top-K Tasks."""
    web_client.post("/api/dataset/demo")

    rel_tk = web_client.post("/api/release", json={
        "metric": "top_k_tasks",
        "recipient": "internal",
        "epsilon": 1.0,
        "delta": 1e-6,
    })
    rel_id_tk = rel_tk.json()["release_id"]
    eval_tk = web_client.get(f"/api/evaluation/accuracy?release_id={rel_id_tk}&recipient=internal")
    assert eval_tk.status_code == 200
    tk_data = eval_tk.json()

    assert "Top-" in tk_data["primary_metric_name"]
    assert 0.0 <= tk_data["precision_at_k"] <= 100.0
    assert 0.0 <= tk_data["jaccard_similarity"] <= 100.0
    assert "真实热门任务前" in tk_data["explanation"]


def test_04_utility_summary_overview(web_client):
    """Verify all 5 metrics benchmark summary table in isolated evaluation mode."""
    web_client.post("/api/dataset/demo")

    res_sum = web_client.get("/api/evaluation/summary?recipient=internal")
    assert res_sum.status_code == 200
    summary = res_sum.json()
    assert len(summary) == 5

    metric_ids = {item["metric_id"] for item in summary}
    assert metric_ids == {
        "user_prevalence",
        "user_normalized_share",
        "interaction_volume",
        "success_rate",
        "top_k_tasks",
    }
    for item in summary:
        assert item["primary_metric"]
        assert item["value_display"]


def test_05_external_recipients_strictly_forbidden_from_evaluations(web_client):
    """Verify external commercial callers receive 403 Forbidden for all evaluation endpoints."""
    web_client.post("/api/dataset/demo")

    rel = web_client.post("/api/release", json={
        "metric": "interaction_volume",
        "recipient": "provider_a",
        "epsilon": 1.0,
        "delta": 1e-6,
    })
    rel_id = rel.json()["release_id"]

    # 1. Accuracy endpoint
    res_acc = web_client.get(f"/api/evaluation/accuracy?release_id={rel_id}&recipient=provider_a")
    assert res_acc.status_code == 403

    # 2. Utility summary endpoint
    res_sum = web_client.get("/api/evaluation/summary?recipient=provider_a")
    assert res_sum.status_code == 403

    # 3. CSV download endpoint
    res_csv = web_client.get(f"/api/evaluation/download/csv?release_id={rel_id}&recipient=provider_a")
    assert res_csv.status_code == 403

    # 4. JSON download endpoint
    res_json = web_client.get(f"/api/evaluation/download/json?release_id={rel_id}&recipient=provider_a")
    assert res_json.status_code == 403


def test_06_trusted_internal_can_download_evaluation_reports(web_client):
    """Verify trusted internal admin can download CSV and JSON evaluation reports."""
    web_client.post("/api/dataset/demo")
    rel = web_client.post("/api/release", json={
        "metric": "interaction_volume",
        "recipient": "internal",
        "epsilon": 1.0,
        "delta": 1e-6,
    })
    rel_id = rel.json()["release_id"]

    res_csv = web_client.get(f"/api/evaluation/download/csv?release_id={rel_id}&recipient=internal")
    assert res_csv.status_code == 200
    assert "# WARNING:" in res_csv.text
    assert "raw_exact" in res_csv.text

    res_json = web_client.get(f"/api/evaluation/download/json?release_id={rel_id}&recipient=internal")
    assert res_json.status_code == 200
    assert "mae" in res_json.json()
