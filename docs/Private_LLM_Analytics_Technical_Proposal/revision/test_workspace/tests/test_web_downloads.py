"""Tests for differential privacy release download and export security (v1.0.x / v1.1).

Verifies Section 33 requirements:
- DP CSV export succeeds with authorized format
- DP JSON export succeeds with rich metadata without internal state leaks
- Download operations incur ZERO incremental privacy spend
- Repeated downloads return identical committed results without executing mechanism
- External recipient cannot download exact internal evaluation comparison (403 Forbidden)
- Trusted internal recipient can download evaluation comparison with warning header
- Provider filtering remains strictly enforced in external downloads
"""

import io
import json
import pytest
from fastapi.testclient import TestClient

from private_llm_analytics.config import load_app_config
from private_llm_analytics.service.analytics import AnalyticsService
from private_llm_analytics.web.adapters import WebAnalyticsAdapter
from private_llm_analytics.web.app import create_app


@pytest.fixture
def web_test_client(tmp_path):
    config = load_app_config("configs")
    temp_db = tmp_path / "test_download_privacy_state.db"
    config.privacy.release.state_db_path = str(temp_db)

    service = AnalyticsService(config=config)
    adapter = WebAnalyticsAdapter(service_or_config_dir=service)
    app = create_app(adapter=adapter)
    client = TestClient(app)
    return client, adapter, service


class TestWebDownloads:
    def test_dp_csv_and_json_downloads_succeed(self, web_test_client):
        client, adapter, service = web_test_client
        client.post("/api/dataset/demo")

        # Execute a production release
        rel_res = client.post("/api/release", json={
            "metric": "interaction_volume",
            "recipient": "provider_a",
            "epsilon": 1.0,
            "delta": 1e-6,
        })
        assert rel_res.status_code == 200
        rel_data = rel_res.json()
        release_id = rel_data["release_id"]

        # 1. Download CSV
        csv_res = client.get(f"/api/release/download/csv?release_id={release_id}&recipient=provider_a")
        assert csv_res.status_code == 200
        assert "text/csv" in csv_res.headers["content-type"]
        assert "attachment" in csv_res.headers["content-disposition"]
        assert f'filename="private_llm_analytics_release_{release_id}.csv"' in csv_res.headers["content-disposition"]
        csv_text = csv_res.text
        assert "provider,provider_display_name" in csv_text or "provider" in csv_text
        assert "OpenAI" in csv_text
        assert "interaction_volume" in csv_text
        # Verify required rich metadata fields in CSV
        assert "recipient_internal_id" in csv_text
        assert "recipient_display_name" in csv_text
        assert "metric_name" in csv_text
        assert "dp_estimate" in csv_text
        assert "statistic_unit" in csv_text
        assert "reporting_window" in csv_text
        assert "disclaimer" in csv_text
        assert "以下结果已加入差分隐私噪声，仅用于隐私保护后的统计共享" in csv_text

        # 2. Download JSON
        json_res = client.get(f"/api/release/download/json?release_id={release_id}&recipient=provider_a")
        assert json_res.status_code == 200
        assert "application/json" in json_res.headers["content-type"]
        assert f'filename="private_llm_analytics_release_{release_id}.json"' in json_res.headers["content-disposition"]
        json_data = json_res.json()
        assert json_data["release_id"] == release_id or json_data.get("release_key") == release_id
        assert json_data["metric"] == "interaction_volume"
        assert json_data["recipient"] == "provider_a"
        assert json_data["recipient_internal_id"] == "provider_a"
        assert json_data["recipient_display_name"] == "OpenAI"
        assert "交互量" in json_data["metric_name"]
        assert "次交互" in json_data["statistic_unit"]
        assert "disclaimer" in json_data
        assert "以下结果已加入差分隐私噪声，仅用于隐私保护后的统计共享" in json_data["disclaimer"]
        assert "results" in json_data
        # Ensure no internal RNG or raw state leaked
        assert "rng" not in json_data
        assert "noise_seed" not in json_data
        assert "raw_dataframe" not in json_data

    def test_download_incurs_zero_incremental_privacy_spend(self, web_test_client):
        client, adapter, service = web_test_client
        client.post("/api/dataset/demo")

        rel_res = client.post("/api/release", json={
            "metric": "user_prevalence",
            "recipient": "provider_a",
            "epsilon": 1.0,
            "delta": 1e-6,
        })
        rel_id = rel_res.json()["release_id"]

        # Budget status after release
        status_before = client.get("/api/privacy/status?recipient=provider_a").json()
        eps_before = status_before["effective_epsilon"]
        count_before = status_before["release_count"]

        # Repeated downloads of both CSV and JSON
        for _ in range(5):
            r_csv = client.get(f"/api/release/download/csv?release_id={rel_id}&recipient=provider_a")
            assert r_csv.status_code == 200
            r_json = client.get(f"/api/release/download/json?release_id={rel_id}&recipient=provider_a")
            assert r_json.status_code == 200

        # Verify budget and release count strictly unchanged
        status_after = client.get("/api/privacy/status?recipient=provider_a").json()
        assert status_after["effective_epsilon"] == eps_before
        assert status_after["release_count"] == count_before

    def test_external_recipient_cannot_download_exact_comparison(self, web_test_client):
        client, adapter, service = web_test_client
        client.post("/api/dataset/demo")

        rel_res = client.post("/api/release", json={
            "metric": "interaction_volume",
            "recipient": "provider_a",
            "epsilon": 1.0,
            "delta": 1e-6,
        })
        rel_id = rel_res.json()["release_id"]

        # External caller attempts to download internal evaluation comparison
        res = client.get(f"/api/release/download/internal-comparison?release_id={rel_id}&recipient=provider_a")
        # Must fail-closed with 403 Forbidden
        assert res.status_code == 403
        data = res.json()
        assert "RecipientSecurityViolation" in data["type"] or "forbidden" in data["error"].lower()

    def test_trusted_recipient_can_download_exact_comparison(self, web_test_client):
        client, adapter, service = web_test_client
        client.post("/api/dataset/demo")

        rel_res = client.post("/api/release", json={
            "metric": "interaction_volume",
            "recipient": "internal",
            "epsilon": 1.0,
            "delta": 1e-6,
        })
        rel_id = rel_res.json()["release_id"]

        # Trusted internal admin requests comparison
        res = client.get(f"/api/release/download/internal-comparison?release_id={rel_id}&recipient=internal")
        assert res.status_code == 200
        text = res.text
        # Verify prominent confidential warning banner present
        assert "该文件包含内部精确统计，仅供可信管理员评估，禁止作为外部发布文件" in text
        assert "raw_exact" in text
        assert "clipped_exact" in text
        assert "dp_released" in text
        assert "clipping_bias" in text
        assert "dp_noise_error" in text
        assert "total_error" in text

    def test_provider_filtering_enforced_in_downloaded_data(self, web_test_client):
        client, adapter, service = web_test_client
        client.post("/api/dataset/demo")

        rel_res = client.post("/api/release", json={
            "metric": "interaction_volume",
            "recipient": "provider_a",
            "epsilon": 1.0,
            "delta": 1e-6,
        })
        rel_id = rel_res.json()["release_id"]

        # Download CSV as provider_a
        csv_res = client.get(f"/api/release/download/csv?release_id={rel_id}&recipient=provider_a")
        csv_text = csv_res.text
        # Must only contain provider_a, never provider_b or provider_c
        assert "provider_a" in csv_text
        assert "provider_b" not in csv_text
        assert "provider_c" not in csv_text

    def test_katex_assets_served(self, web_test_client):
        """Verify KaTeX CSS and JS assets are served locally by FastAPI."""
        client, _, _ = web_test_client
        res_css = client.get("/static/vendor/katex/katex.min.css")
        assert res_css.status_code == 200
        assert len(res_css.content) > 1000

        res_js = client.get("/static/vendor/katex/katex.min.js")
        assert res_js.status_code == 200
        assert len(res_js.content) > 10000

        res_auto = client.get("/static/vendor/katex/contrib/auto-render.min.js")
        assert res_auto.status_code == 200
        assert len(res_auto.content) > 1000

    def test_index_includes_katex_and_dp_notice(self, web_test_client):
        """Verify index.html contains KaTeX tags and privacy sharing notice banner."""
        client, _, _ = web_test_client
        res = client.get("/")
        assert res.status_code == 200
        html = res.text
        assert "katex.min.css" in html
        assert "katex.min.js" in html
        assert "以下结果已加入差分隐私噪声，仅用于隐私保护后的统计共享" in html

    def test_target_formulas_in_cards(self, web_test_client):
        """Verify the 6 required mathematical formulas exist in mechanism cards."""
        from private_llm_analytics.statistics.cards import get_mechanism_card
        config = load_app_config("configs")

        # 1. b_{u,j} = \mathbf{1}[m_{u,j} > 0]
        c_prev = get_mechanism_card("user_prevalence", config)
        assert "b_{u,j} = \\mathbf{1}[m_{u,j} > 0]" in c_prev.user_contribution["formula"]

        # 2. \|b_u\|_0 \le \min(k, D)
        assert "\\|b_u\\|_0 \\le \\min(k, D)" in c_prev.contribution_bound["formula"]

        # 3. \Delta_2 = \sqrt{\min(k, D)} \le \sqrt{k}
        assert "\\Delta_2 = \\sqrt{\\min(k, D)} \\le \\sqrt{k}" in c_prev.sensitivity["formula"]

        # 4. \tilde{S} = S + \mathcal{N}(0, \sigma^2 I)
        assert "\\tilde{S} = S + \\mathcal{N}(0, \\sigma^2 I)" in c_prev.dp_mechanism["equation"]

        # 5. a_u = \min(1, C_{\text{rate}} / \|n_u\|_2), \bar n_u = a_u n_u, \bar s_u = a_u s_u
        c_sr = get_mechanism_card("success_rate", config)
        assert "a_u = \\min" in c_sr.user_contribution["formula"]
        assert "C_{\\text{rate}}" in c_sr.user_contribution["formula"]
        assert "\\|n_u\\|_2" in c_sr.user_contribution["formula"]
        assert "\\bar n_u = a_u n_u" in c_sr.user_contribution["formula"]
        assert "\\bar s_u = a_u s_u" in c_sr.user_contribution["formula"]
        assert "\\Delta_2 \\le \\sqrt{2} C_{\\text{rate}}" in c_sr.sensitivity["formula"]
        assert "\\bar s_u" in c_sr.contribution_bound["formula"] and "\\bar n_u" in c_sr.contribution_bound["formula"]

        # 6. Correct denominator thresholding and ratio without max(T_min)
        sr_steps = " ".join(c_sr.post_processing["steps"])
        assert "\\hat N_j = \\max(\\tilde N_j, 0)" in sr_steps
        assert "\\hat S_j = \\operatorname{clip}(\\tilde S_j, 0, \\hat N_j)" in sr_steps
        assert "SUPPRESSED / null" in sr_steps
        assert "\\hat r_j = \\frac{\\hat S_j}{\\hat N_j}" in sr_steps
        assert "max(\\tilde{T}" not in sr_steps and "T_{\\min}" not in sr_steps

    def test_provider_display_name_cell_formatting(self, web_test_client):
        """Verify CSV export contains friendly provider display names and cell labels."""
        client, _, _ = web_test_client
        client.post("/api/dataset/demo")
        rel_res = client.post("/api/release", json={
            "metric": "interaction_volume",
            "recipient": "internal",
            "epsilon": 1.0,
            "delta": 1e-6,
        })
        rel_id = rel_res.json()["release_id"]
        csv_res = client.get(f"/api/release/download/csv?release_id={rel_id}&recipient=internal")
        csv_text = csv_res.text
        assert "OpenAI × technical" in csv_text
        assert "OpenAI" in csv_text
        assert "provider_display_name" in csv_text
        assert "cell_display" in csv_text
