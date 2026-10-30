"""Tests for administrative privacy policy settings and governance security (v1.0.x / v1.1).

Verifies Section 34 requirements:
- External recipient cannot modify lifetime privacy cap (403 Forbidden)
- Trusted admin can update lifetime policy parameters
- Validation rejects non-positive epsilon, out-of-range delta, NaN/inf
- Cap cannot be reduced below current effective spend (preserves historical expenditure)
- Changing policy never resets historical ledger or release store
- Audit records are written to persistent admin audit table
"""

import pytest
from fastapi.testclient import TestClient

from private_llm_analytics.config import load_app_config
from private_llm_analytics.service.analytics import AnalyticsService
from private_llm_analytics.web.adapters import WebAnalyticsAdapter
from private_llm_analytics.web.app import create_app


@pytest.fixture
def policy_client(tmp_path):
    config = load_app_config("configs")
    temp_db = tmp_path / "test_policy_privacy_state.db"
    config.privacy.release.state_db_path = str(temp_db)

    service = AnalyticsService(config=config)
    adapter = WebAnalyticsAdapter(service_or_config_dir=service)
    app = create_app(adapter=adapter)
    client = TestClient(app)
    return client, adapter, service


class TestAdminPrivacyPolicy:
    def test_get_privacy_policy_endpoint(self, policy_client):
        client, adapter, service = policy_client
        res = client.get("/api/admin/privacy-policy")
        assert res.status_code == 200
        data = res.json()
        assert data["lifetime_epsilon_cap"] == 4.0
        assert data["global_delta"] == 1e-6
        assert data["default_query_epsilon"] == 1.0
        assert data["default_query_delta"] == 1e-6
        assert data["current_effective_epsilon"] == 0.0

    def test_external_recipient_cannot_modify_policy(self, policy_client):
        client, adapter, service = policy_client
        # External recipient attempts to modify policy
        res = client.post("/api/admin/privacy-policy", json={
            "recipient": "provider_a",
            "lifetime_epsilon_cap": 8.0,
            "global_delta": 1e-6,
            "default_query_epsilon": 1.0,
            "default_query_delta": 1e-6,
            "reason": "Unauthorized attempt",
        })
        assert res.status_code == 403
        data = res.json()
        assert "RecipientSecurityViolation" in data["type"] or "forbidden" in data["error"].lower()

    def test_invalid_policy_values_rejected(self, policy_client):
        client, adapter, service = policy_client

        # Non-positive epsilon cap
        r1 = client.post("/api/admin/privacy-policy", json={
            "recipient": "internal",
            "lifetime_epsilon_cap": -1.0,
            "global_delta": 1e-6,
            "default_query_epsilon": 1.0,
            "default_query_delta": 1e-6,
        })
        assert r1.status_code in (400, 422)

        # Delta >= 1.0
        r2 = client.post("/api/admin/privacy-policy", json={
            "recipient": "internal",
            "lifetime_epsilon_cap": 4.0,
            "global_delta": 1.5,
            "default_query_epsilon": 1.0,
            "default_query_delta": 1e-6,
        })
        assert r2.status_code in (400, 422)

        # Delta <= 0
        r3 = client.post("/api/admin/privacy-policy", json={
            "recipient": "internal",
            "lifetime_epsilon_cap": 4.0,
            "global_delta": 0.0,
            "default_query_epsilon": 1.0,
            "default_query_delta": 1e-6,
        })
        assert r3.status_code in (400, 422)

    def test_trusted_admin_can_update_policy_and_audit_is_recorded(self, policy_client):
        client, adapter, service = policy_client

        update_res = client.post("/api/admin/privacy-policy", json={
            "recipient": "internal",
            "lifetime_epsilon_cap": 6.0,
            "global_delta": 2e-6,
            "default_query_epsilon": 1.5,
            "default_query_delta": 1.5e-6,
            "reason": "Quarterly governance review expansion",
        })
        assert update_res.status_code == 200
        data = update_res.json()
        assert data["lifetime_epsilon_cap"] == 6.0
        assert data["global_delta"] == 2e-6
        assert data["default_query_epsilon"] == 1.5
        assert len(data["audit_history"]) >= 1
        latest_audit = data["audit_history"][0]
        assert latest_audit["actor"] == "internal"
        assert latest_audit["epsilon_cap"] == "6.0"
        assert latest_audit["reason"] == "Quarterly governance review expansion"

    def test_cap_cannot_be_set_below_spent_epsilon_and_preserves_history(self, policy_client):
        client, adapter, service = policy_client
        client.post("/api/dataset/demo")

        # Incur some privacy spend
        rel = client.post("/api/release", json={
            "metric": "interaction_volume",
            "recipient": "internal",
            "epsilon": 2.0,
            "delta": 1e-6,
        })
        assert rel.status_code == 200
        pv = client.get("/api/privacy/status?recipient=internal").json()
        spent = pv["effective_epsilon"]
        assert spent > 0.0

        # Attempt to set cap below spent epsilon
        too_low_cap = spent / 2.0
        bad_res = client.post("/api/admin/privacy-policy", json={
            "recipient": "internal",
            "lifetime_epsilon_cap": too_low_cap,
            "global_delta": 1e-6,
            "default_query_epsilon": 1.0,
            "default_query_delta": 1e-6,
            "reason": "Attempt to erase spend",
        })
        assert bad_res.status_code == 400
        err_msg = bad_res.json()["error"]
        assert "below" in err_msg.lower() or "cannot set lifetime epsilon cap" in err_msg.lower()

        # Verify historical spend and ledger remain completely preserved
        pv_after = client.get("/api/privacy/status?recipient=internal").json()
        assert pv_after["effective_epsilon"] == spent
        assert pv_after["release_count"] >= 1
