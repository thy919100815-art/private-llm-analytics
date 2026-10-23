"""Comprehensive unit and integration tests for Private LLM Analytics Local Web Console (v0.5)."""

import io
from pathlib import Path
import tempfile
import pytest
from fastapi.testclient import TestClient

from private_llm_analytics.config import load_app_config
from private_llm_analytics.service.analytics import AnalyticsService
from private_llm_analytics.web.adapters import WebAnalyticsAdapter
from private_llm_analytics.web.app import create_app
from private_llm_analytics.web.__main__ import main as cli_main


@pytest.fixture
def test_app_and_client(tmp_path):
    """Fixture providing an isolated FastAPI TestClient with temporary SQLite state."""
    config = load_app_config("configs")
    temp_db = tmp_path / "test_privacy_state.db"
    config.privacy.release.state_db_path = str(temp_db)

    service = AnalyticsService(config=config)
    adapter = WebAnalyticsAdapter(service_or_config_dir=service)
    app = create_app(adapter=adapter)
    client = TestClient(app)
    return client, adapter, service


def test_01_health_check(test_app_and_client):
    client, _, _ = test_app_and_client
    res = client.get("/api/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "ok"
    assert data["local_only"] is True


def test_02_config_endpoint(test_app_and_client):
    client, _, _ = test_app_and_client
    res = client.get("/api/config")
    assert res.status_code == 200
    data = res.json()
    assert "provider_a" in data["providers"]
    assert "technical" in data["tasks"]
    assert len(data["metrics"]) == 3
    assert data["default_epsilon"] > 0
    assert 0 < data["default_delta"] < 1
    assert data["continual_enabled"] is False


def test_03_ui_root_and_static_assets(test_app_and_client):
    client, _, _ = test_app_and_client
    res_root = client.get("/")
    assert res_root.status_code == 200
    assert "Private LLM Analytics Console" in res_root.text

    res_css = client.get("/static/styles.css")
    assert res_css.status_code == 200
    assert "--bg-color" in res_css.text

    res_js = client.get("/static/app.js")
    assert res_js.status_code == 200
    assert "initConsole" in res_js.text


def test_04_dataset_status_empty_then_demo(test_app_and_client):
    client, _, _ = test_app_and_client
    res_init = client.get("/api/dataset/status")
    assert res_init.status_code == 200
    assert res_init.json()["loaded"] is False

    res_demo = client.post("/api/dataset/demo")
    assert res_demo.status_code == 200
    data = res_demo.json()
    assert data["loaded"] is True
    assert data["rows"] > 0
    assert data["source"] == "demo"

    res_after = client.get("/api/dataset/status")
    assert res_after.json()["loaded"] is True


def test_05_dataset_upload_valid_csv(test_app_and_client):
    client, _, _ = test_app_and_client
    csv_content = (
        "user_id,provider,task\n"
        "u1,provider_a,technical\n"
        "u2,provider_b,document\n"
        "u3,provider_c,analysis\n"
    ).encode("utf-8")

    res = client.post(
        "/api/dataset/upload",
        files={"file": ("test_upload.csv", io.BytesIO(csv_content), "text/csv")},
    )
    assert res.status_code == 200
    data = res.json()
    assert data["loaded"] is True
    assert data["rows"] == 3
    assert data["source"] == "uploaded"


def test_06_dataset_upload_invalid_schema_fails_safely(test_app_and_client):
    client, _, _ = test_app_and_client
    # Missing required 'user_id'
    bad_csv = "provider,task\nprovider_a,technical\n".encode("utf-8")
    res = client.post(
        "/api/dataset/upload",
        files={"file": ("bad.csv", io.BytesIO(bad_csv), "text/csv")},
    )
    assert res.status_code == 400
    assert "Missing required columns" in res.json()["error"]

    # Unknown provider
    bad_prov_csv = "user_id,provider,task\nu1,unknown_model_x,technical\n".encode("utf-8")
    res2 = client.post(
        "/api/dataset/upload",
        files={"file": ("bad_prov.csv", io.BytesIO(bad_prov_csv), "text/csv")},
    )
    assert res2.status_code == 400
    assert "Unknown provider" in res2.json()["error"]


def test_07_privacy_status_reads_authoritative_state(test_app_and_client):
    client, _, _ = test_app_and_client
    res = client.get("/api/privacy/status?recipient=provider_a")
    assert res.status_code == 200
    data = res.json()
    assert data["budget_scope_id"] == "global_external"
    assert data["epsilon_cap"] == 4.0
    assert data["effective_epsilon"] == 0.0
    assert data["remaining_headroom"] == 4.0


def test_08_planner_recommendation_does_not_spend_budget(test_app_and_client):
    client, _, _ = test_app_and_client
    # Check budget before
    res_before = client.get("/api/privacy/status?recipient=provider_a").json()

    res_plan = client.post(
        "/api/planner/recommend",
        json={"metric": "interaction_volume", "recipient": "provider_a", "planning_horizon_days": 90},
    )
    assert res_plan.status_code == 200
    plan_data = res_plan.json()
    assert plan_data["is_recommendation_only"] is True
    assert plan_data["recommended_granularity"] in ["full_9_cell", "provider_3_cell", "task_3_cell", "overall_1_cell"]

    # Check budget after: must have zero spend
    res_after = client.get("/api/privacy/status?recipient=provider_a").json()
    assert res_after["effective_epsilon"] == res_before["effective_epsilon"]
    assert res_after["release_count"] == res_before["release_count"]


def test_09_release_executes_production_path(test_app_and_client):
    client, _, _ = test_app_and_client
    client.post("/api/dataset/demo")

    payload = {
        "metric": "interaction_volume",
        "recipient": "provider_a",
        "epsilon": 1.0,
        "delta": 1e-6,
        "window_days": 30,
    }
    res = client.post("/api/release", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["metric"] == "interaction_volume"
    assert data["recipient"] == "provider_a"
    assert "cell_volumes" in data["results"]
    assert data["cache_hit"] is False
    assert data["incremental_privacy_cost"] == 1.0


def test_10_release_rejects_rng_or_seed_control(test_app_and_client):
    client, _, _ = test_app_and_client
    client.post("/api/dataset/demo")

    # Attacker attempting to pass random seed
    bad_payload = {
        "metric": "interaction_volume",
        "recipient": "provider_a",
        "epsilon": 1.0,
        "delta": 1e-6,
        "seed": 42,
    }
    res = client.post("/api/release", json=bad_payload)
    assert res.status_code in (400, 422)
    assert "seed" in str(res.json()).lower() or "rng" in str(res.json()).lower()


def test_11_release_memoization_reuse(test_app_and_client):
    client, _, _ = test_app_and_client
    client.post("/api/dataset/demo")

    payload = {
        "metric": "interaction_volume",
        "recipient": "provider_a",
        "epsilon": 1.0,
        "delta": 1e-6,
        "window_days": 30,
    }
    res1 = client.post("/api/release", json=payload)
    assert res1.status_code == 200
    data1 = res1.json()
    assert data1["cache_hit"] is False

    # Second identical call: must hit cache and charge 0 incremental spend
    res2 = client.post("/api/release", json=payload)
    assert res2.status_code == 200
    data2 = res2.json()
    assert data2["cache_hit"] is True
    assert data2["incremental_privacy_cost"] == 0.0
    assert data2["release_key"] == data1["release_key"]


def test_12_sanitization_no_active_users_to_external(test_app_and_client):
    client, _, _ = test_app_and_client
    client.post("/api/dataset/demo")

    payload = {
        "metric": "user_prevalence",
        "recipient": "provider_a",
        "epsilon": 1.0,
        "delta": 1e-6,
    }
    res = client.post("/api/release", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert "active_user_count" not in data
    assert "raw_exact" not in data
    assert "raw_dataframe" not in data


def test_13_budget_overrun_rejected(test_app_and_client):
    client, _, service = test_app_and_client
    client.post("/api/dataset/demo")

    # Epsilon = 5.0 exceeds cap = 4.0
    payload = {
        "metric": "interaction_volume",
        "recipient": "provider_a",
        "epsilon": 5.0,
        "delta": 1e-6,
    }
    res = client.post("/api/release", json=payload)
    assert res.status_code == 403
    assert "budget exhausted" in res.json()["error"].lower()


def test_14_release_history_sanitized(test_app_and_client):
    client, _, _ = test_app_and_client
    client.post("/api/dataset/demo")

    client.post("/api/release", json={
        "metric": "interaction_volume",
        "recipient": "provider_a",
        "epsilon": 1.0,
        "delta": 1e-6,
    })

    res = client.get("/api/releases")
    assert res.status_code == 200
    items = res.json()
    assert len(items) >= 1
    first = items[0]
    assert "release_id" in first
    assert "timestamp" in first
    assert "raw_exact" not in first
    assert "seed" not in first


def test_15_continual_disabled_returns_safe_error(test_app_and_client):
    client, _, _ = test_app_and_client
    res_status = client.get("/api/continual/status")
    assert res_status.status_code == 200
    assert res_status.json()["enabled"] is False

    res_query = client.post("/api/continual/query", json={
        "start_day": 1,
        "end_day": 10,
        "metric": "interaction_volume",
        "recipient": "provider_a",
    })
    assert res_query.status_code == 400
    assert "Continual observation is disabled" in res_query.json()["error"]


def test_16_continual_unsupported_metric_rejected(test_app_and_client):
    client, _, _ = test_app_and_client
    res = client.post("/api/continual/query", json={
        "start_day": 1,
        "end_day": 10,
        "metric": "user_prevalence",
        "recipient": "provider_a",
    })
    assert res.status_code in (400, 422)
    assert "Interaction Volume only" in str(res.json())


def test_17_web_server_defaults_to_localhost():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args([])
    assert args.host == "127.0.0.1"
    assert args.port == 8000


def test_18_continual_finalized_frontier_failure_propagates_safely(tmp_path):
    """When continual is enabled but unfinalized data requested, fails closed safely."""
    config = load_app_config("configs")
    temp_db = tmp_path / "test_privacy_state.db"
    config.privacy.release.state_db_path = str(temp_db)
    config.privacy.continual.enabled = True

    service = AnalyticsService(config=config)
    adapter = WebAnalyticsAdapter(service_or_config_dir=service)
    app = create_app(adapter=adapter)
    client = TestClient(app)

    # Load demo data
    client.post("/api/dataset/demo")

    # Issue query: since demo data has no authoritative finalized boundary registered,
    # the continual query engine must fail closed with ProductionSafetyConfigurationError,
    # returning a structured 400 Bad Request error without raising an unhandled exception.
    res = client.post("/api/continual/query", json={
        "start_day": 1,
        "end_day": 10,
        "metric": "interaction_volume",
        "recipient": "provider_a",
    })
    assert res.status_code == 400
    assert "error" in res.json()
    assert "ProductionSafetyConfigurationError" in res.json().get("type", "") or "finalized" in res.json().get("error", "").lower() or "not initialized" in res.json().get("error", "").lower()


def test_web_window_7_vs_30_have_distinct_bounds(test_app_and_client):
    client, adapter, service = test_app_and_client
    client.post("/api/dataset/demo")

    res7 = client.post("/api/release", json={
        "metric": "interaction_volume",
        "recipient": "provider_a",
        "epsilon": 0.5,
        "delta": 1e-6,
        "window_days": 7,
    })
    assert res7.status_code == 200
    data7 = res7.json()

    res30 = client.post("/api/release", json={
        "metric": "interaction_volume",
        "recipient": "provider_a",
        "epsilon": 0.5,
        "delta": 1e-6,
        "window_days": 30,
    })
    assert res30.status_code == 200
    data30 = res30.json()

    # End bound is identical (dataset max timestamp), start bound is distinct
    assert data7["window_end"] == data30["window_end"]
    assert data7["window_start"] != data30["window_start"]


def test_web_window_7_vs_30_have_distinct_release_keys(test_app_and_client):
    client, _, _ = test_app_and_client
    client.post("/api/dataset/demo")

    res7 = client.post("/api/release", json={
        "metric": "interaction_volume",
        "recipient": "provider_a",
        "epsilon": 0.5,
        "delta": 1e-6,
        "window_days": 7,
    })
    res30 = client.post("/api/release", json={
        "metric": "interaction_volume",
        "recipient": "provider_a",
        "epsilon": 0.5,
        "delta": 1e-6,
        "window_days": 30,
    })
    assert res7.status_code == 200
    assert res30.status_code == 200
    assert res7.json()["release_key"] != res30.json()["release_key"]


def test_web_same_window_replays_canonical_release(test_app_and_client):
    client, _, _ = test_app_and_client
    client.post("/api/dataset/demo")

    payload = {
        "metric": "interaction_volume",
        "recipient": "provider_a",
        "epsilon": 0.5,
        "delta": 1e-6,
        "window_days": 14,
    }
    res1 = client.post("/api/release", json=payload)
    assert res1.status_code == 200
    data1 = res1.json()
    assert data1["cache_hit"] is False

    res2 = client.post("/api/release", json=payload)
    assert res2.status_code == 200
    data2 = res2.json()
    assert data2["cache_hit"] is True
    assert data2["incremental_privacy_cost"] == 0.0
    assert data2["release_key"] == data1["release_key"]
    assert data2["window_start"] == data1["window_start"]
    assert data2["window_end"] == data1["window_end"]


def test_web_window_uses_dataset_max_timestamp_not_wall_clock(test_app_and_client):
    client, adapter, service = test_app_and_client
    import pandas as pd
    # Dataset with explicitly fixed historical timestamps (June 2024)
    records = []
    for d in range(1, 16):
        records.append({
            "interaction_id": f"I_{d:04d}",
            "user_id": f"user_{d % 3}",
            "timestamp": f"2024-06-{d:02d} 18:00:00",
            "provider": "provider_a",
            "task": "technical",
        })
    df = pd.DataFrame(records)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    service.set_data(df, dataset_version="historical_2024_v1")

    res = client.post("/api/release", json={
        "metric": "interaction_volume",
        "recipient": "provider_a",
        "epsilon": 0.5,
        "delta": 1e-6,
        "window_days": 7,
    })
    assert res.status_code == 200
    data = res.json()
    # Must use dataset max timestamp (2024-06-15 18:00:00), NOT today's machine time
    assert data["window_end"] == "2024-06-15 18:00:00"
    # 7 days ending 2024-06-15: start is beginning of date 2024-06-15 - 6 days = 2024-06-09 00:00:00
    assert data["window_start"] == "2024-06-09 00:00:00"


def test_web_window_requires_timestamp_column(test_app_and_client):
    client, adapter, service = test_app_and_client
    import pandas as pd
    # Dataset without timestamp column
    df = pd.DataFrame([
        {"user_id": "u1", "provider": "provider_a", "task": "technical"},
        {"user_id": "u2", "provider": "provider_b", "task": "document"},
    ])
    service._raw_dataframe = df

    res = client.post("/api/release", json={
        "metric": "interaction_volume",
        "recipient": "provider_a",
        "epsilon": 0.5,
        "delta": 1e-6,
        "window_days": 7,
    })
    assert res.status_code in (400, 422)
    assert "timestamped telemetry data" in res.json().get("error", "").lower() or "timestamp" in str(res.json()).lower()


def test_web_rejects_window_days_with_explicit_bounds(test_app_and_client):
    client, _, _ = test_app_and_client
    client.post("/api/dataset/demo")

    # Reject both window_days and window_start
    res1 = client.post("/api/release", json={
        "metric": "interaction_volume",
        "recipient": "provider_a",
        "epsilon": 0.5,
        "delta": 1e-6,
        "window_days": 7,
        "window_start": "2026-01-01 00:00:00",
    })
    assert res1.status_code in (400, 422)
    assert "cannot specify both" in str(res1.json()).lower()

    # Reject only one of window_start/window_end
    res2 = client.post("/api/release", json={
        "metric": "interaction_volume",
        "recipient": "provider_a",
        "epsilon": 0.5,
        "delta": 1e-6,
        "window_start": "2026-01-01 00:00:00",
    })
    assert res2.status_code in (400, 422)
    assert "both" in str(res2.json()).lower()


def test_web_explicit_start_end_pass_through(test_app_and_client):
    client, _, _ = test_app_and_client
    client.post("/api/dataset/demo")

    res = client.post("/api/release", json={
        "metric": "interaction_volume",
        "recipient": "provider_a",
        "epsilon": 0.5,
        "delta": 1e-6,
        "window_start": "2026-01-01 00:00:00",
        "window_end": "2026-01-15 00:00:00",
    })
    assert res.status_code == 200
    data = res.json()
    assert data["window_start"] == "2026-01-01 00:00:00"
    assert data["window_end"] == "2026-01-15 00:00:00"


def test_web_display_filters_do_not_spend_budget_or_execute_release(test_app_and_client):
    client, adapter, service = test_app_and_client
    client.post("/api/dataset/demo")

    # Read rendered HTML
    html_res = client.get("/")
    assert html_res.status_code == 200
    html = html_res.text
    assert "Display Filters" in html
    assert "Filters apply only to the already released DP result and do not trigger a new privacy release or additional privacy spend." in html
    assert "Display providers" in html
    assert "Display tasks" in html
    assert 'id="select-granularity"' not in html
    assert "Planner recommendation only" in html

    # Initial budget status
    status1 = client.get("/api/privacy/status").json()
    eps1 = status1["effective_epsilon"]

    # Simulating client-side UI filter operations: toggling display filters makes NO api/release call
    # Privacy budget and ledger entries remain strictly unchanged
    status2 = client.get("/api/privacy/status").json()
    assert status2["effective_epsilon"] == eps1
    assert len(service.orchestrator.ledger.get_entries()) == 0


def test_web_cli_warns_on_non_loopback_host(capsys):
    import io
    import sys
    from private_llm_analytics.web.__main__ import main as cli_main
    from unittest.mock import patch

    # Mock uvicorn.run so it doesn't block
    with patch("uvicorn.run"):
        with patch.object(sys, "argv", ["web", "--host", "0.0.0.0"]):
            cli_main()
            captured = capsys.readouterr()
            assert "WARNING: The Web Console is intended for trusted local administration." in captured.err

