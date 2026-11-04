import os
import pytest
from fastapi.testclient import TestClient

from private_llm_analytics.web.app import create_app


def test_default_no_env_is_fail_closed():
    """Default with no environment variables set must fail-closed (demo_mode=false, 403)."""
    old_env = os.environ.pop("APP_ENV", None)
    old_demo = os.environ.pop("DEMO_MODE", None)
    try:
        app = create_app()
        client = TestClient(app)

        cfg_res = client.get("/api/config")
        assert cfg_res.status_code == 200
        assert cfg_res.json().get("demo_mode") is False

        res_epoch = client.post("/api/demo/reset-epoch")
        assert res_epoch.status_code == 403

        res_all = client.post("/api/demo/reset-all")
        assert res_all.status_code == 403
    finally:
        if old_env is not None:
            os.environ["APP_ENV"] = old_env
        if old_demo is not None:
            os.environ["DEMO_MODE"] = old_demo


def test_only_demo_mode_true_env_is_403():
    """Only DEMO_MODE=true, but APP_ENV=production -> 403."""
    old_env = os.environ.get("APP_ENV")
    old_demo = os.environ.get("DEMO_MODE")
    try:
        os.environ["APP_ENV"] = "production"
        os.environ["DEMO_MODE"] = "true"

        app = create_app()
        client = TestClient(app)

        cfg_res = client.get("/api/config")
        assert cfg_res.status_code == 200
        assert cfg_res.json().get("demo_mode") is False

        res_epoch = client.post("/api/demo/reset-epoch")
        assert res_epoch.status_code == 403

        res_all = client.post("/api/demo/reset-all")
        assert res_all.status_code == 403
    finally:
        if old_env is not None:
            os.environ["APP_ENV"] = old_env
        else:
            os.environ.pop("APP_ENV", None)
        if old_demo is not None:
            os.environ["DEMO_MODE"] = old_demo
        else:
            os.environ.pop("DEMO_MODE", None)


def test_only_app_env_local_demo_is_403():
    """Only APP_ENV=local-demo, but DEMO_MODE=false -> 403."""
    old_env = os.environ.get("APP_ENV")
    old_demo = os.environ.get("DEMO_MODE")
    try:
        os.environ["APP_ENV"] = "local-demo"
        os.environ["DEMO_MODE"] = "false"

        app = create_app()
        client = TestClient(app)

        cfg_res = client.get("/api/config")
        assert cfg_res.status_code == 200
        assert cfg_res.json().get("demo_mode") is False

        res_epoch = client.post("/api/demo/reset-epoch")
        assert res_epoch.status_code == 403

        res_all = client.post("/api/demo/reset-all")
        assert res_all.status_code == 403
    finally:
        if old_env is not None:
            os.environ["APP_ENV"] = old_env
        else:
            os.environ.pop("APP_ENV", None)
        if old_demo is not None:
            os.environ["DEMO_MODE"] = old_demo
        else:
            os.environ.pop("DEMO_MODE", None)


def test_demo_reset_epoch_and_all_in_demo_mode():
    """APP_ENV=local-demo + DEMO_MODE=true -> reset API allowed to execute."""
    old_env = os.environ.get("APP_ENV")
    old_demo = os.environ.get("DEMO_MODE")
    try:
        os.environ["APP_ENV"] = "local-demo"
        os.environ["DEMO_MODE"] = "true"

        app = create_app()
        client = TestClient(app)

        # 1. Config returns demo_mode = True
        cfg_res = client.get("/api/config")
        assert cfg_res.status_code == 200
        assert cfg_res.json().get("demo_mode") is True

        # 2. Reset epoch
        res_epoch = client.post("/api/demo/reset-epoch")
        assert res_epoch.status_code == 200
        data_epoch = res_epoch.json()
        assert data_epoch["success"] is True
        assert data_epoch["effective_epsilon"] == 0.0

        # Privacy status shows 0 epsilon
        priv_res = client.get("/api/privacy/status?recipient=provider_a")
        assert priv_res.status_code == 200
        priv_data = priv_res.json()
        assert priv_data["effective_epsilon"] == 0.0
        assert priv_data["epoch"]["effective_epsilon"] == 0.0
        assert priv_data["epoch"]["workloads"]["interaction_volume"]["spent_percentage"] == 0.0

        # 3. Load demo data and perform a release
        client.post("/api/dataset/demo")
        rel_res = client.post("/api/release", json={
            "metric": "interaction_volume",
            "recipient": "provider_a",
            "window_days": 30,
            "use_epoch_planner": True
        })
        assert rel_res.status_code == 200
        rel_data = rel_res.json()
        assert rel_data["incremental_privacy_cost"] > 0

        priv_res_after = client.get("/api/privacy/status?recipient=provider_a").json()
        assert priv_res_after["epoch"]["workloads"]["interaction_volume"]["spent_percentage"] > 0

        # 4. Reset epoch again, verifying budget quota is cleared
        res_epoch2 = client.post("/api/demo/reset-epoch")
        assert res_epoch2.status_code == 200
        priv_res_reset = client.get("/api/privacy/status?recipient=provider_a").json()
        assert priv_res_reset["epoch"]["effective_epsilon"] == 0.0
        assert priv_res_reset["epoch"]["workloads"]["interaction_volume"]["spent_percentage"] == 0.0

        # 5. Reset all
        res_all = client.post("/api/demo/reset-all")
        assert res_all.status_code == 200
        data_all = res_all.json()
        assert data_all["success"] is True
        assert data_all["epoch_id"] == "epoch_001"
        assert data_all["effective_epsilon"] == 0.0

        # Verify dataset reloaded and privacy status 0
        ds_res = client.get("/api/dataset/status")
        assert ds_res.status_code == 200
        assert ds_res.json()["loaded"] is True
        assert ds_res.json()["rows"] > 0

        priv_res_all = client.get("/api/privacy/status?recipient=provider_a").json()
        assert priv_res_all["effective_epsilon"] == 0.0
        assert priv_res_all["lifetime_exposure"]["lifetime_effective_epsilon"] == 0.0
    finally:
        if old_env is not None:
            os.environ["APP_ENV"] = old_env
        else:
            os.environ.pop("APP_ENV", None)
        if old_demo is not None:
            os.environ["DEMO_MODE"] = old_demo
        else:
            os.environ.pop("DEMO_MODE", None)
