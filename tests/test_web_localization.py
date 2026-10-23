"""Tests for Private LLM Analytics Web Console Chinese Localization (v0.5.1)."""

import pytest
from fastapi.testclient import TestClient

from private_llm_analytics.config import load_app_config
from private_llm_analytics.service.analytics import AnalyticsService
from private_llm_analytics.web.adapters import WebAnalyticsAdapter
from private_llm_analytics.web.app import create_app


@pytest.fixture
def local_web_client(tmp_path):
    config = load_app_config("configs")
    temp_db = tmp_path / "test_loc_privacy_state.db"
    config.privacy.release.state_db_path = str(temp_db)

    service = AnalyticsService(config=config)
    adapter = WebAnalyticsAdapter(service_or_config_dir=service)
    app = create_app(adapter=adapter)
    return TestClient(app)


def test_01_index_contains_chinese_title(local_web_client):
    """1. 首页包含：大模型隐私分析控制台"""
    res = local_web_client.get("/")
    assert res.status_code == 200
    html = res.text
    assert "大模型隐私分析控制台" in html
    assert "<title>大模型隐私分析控制台</title>" in html


def test_02_index_no_v050_dev(local_web_client):
    """2. 首页不再显示：v0.5.0-dev"""
    res = local_web_client.get("/")
    assert res.status_code == 200
    html = res.text
    assert "v0.5.0-dev" not in html


def test_03_index_shows_v050(local_web_client):
    """3. 首页显示：v0.5.0"""
    res = local_web_client.get("/")
    assert res.status_code == 200
    html = res.text
    assert "v0.5.0" in html


def test_04_index_shows_lifetime_privacy_budget(local_web_client):
    """4. 首页显示：全生命周期隐私预算"""
    res = local_web_client.get("/")
    assert res.status_code == 200
    html = res.text
    assert "全生命周期隐私预算" in html
    assert "已使用有效 ε" in html
    assert "全生命周期 ε 上限" in html
    assert "剩余 ε 空间" in html
    assert "全局 δ" in html
    assert "隐私预算按照历史发布持续累计，不会因重新启动服务而自动重置。" in html


def test_05_index_shows_query_config(local_web_client):
    """5. 首页显示：查询配置"""
    res = local_web_client.get("/")
    assert res.status_code == 200
    html = res.text
    assert "查询配置" in html
    assert "配置差分隐私发布参数并查看规划器建议" in html
    assert "统计指标" in html
    assert "授权接收方" in html
    assert "统计时间窗口" in html
    assert "隐私参数 ε（Epsilon > 0）" in html
    assert "隐私参数 δ（0 < Delta < 1）" in html


def test_06_index_shows_display_filters(local_web_client):
    """6. 首页显示：结果显示筛选"""
    res = local_web_client.get("/")
    assert res.status_code == 200
    html = res.text
    assert "结果显示筛选" in html
    assert "以下筛选仅作用于已经生成的差分隐私结果，不会重新发起发布，也不会额外消耗隐私预算。" in html
    assert "显示模型 / Provider" in html
    assert "显示任务类型" in html


def test_07_index_shows_release_dp_button(local_web_client):
    """7. 首页显示：生成差分隐私结果"""
    res = local_web_client.get("/")
    assert res.status_code == 200
    html = res.text
    assert "生成差分隐私结果" in html
    assert "获取参数建议" in html
    assert "确认差分隐私发布" in html
    assert "本次新查询可能消耗全生命周期隐私预算。" in html


def test_08_js_no_key_english_success_prompts(local_web_client):
    """8. JS 中不再显示关键英文成功提示。"""
    res = local_web_client.get("/static/app.js")
    assert res.status_code == 200
    js_text = res.text

    # English success prompts must NOT be used as display strings
    assert '"Demo synthetic telemetry dataset generated successfully."' not in js_text
    assert '"Existing canonical release reused. No additional privacy spend."' not in js_text
    assert '"DP release executed and committed to persistent ledger.' not in js_text

    # Chinese translated prompts must be present
    assert "演示遥测数据已成功加载。" in js_text
    assert "已复用现有标准发布结果，本次没有产生额外隐私消耗。" in js_text
    assert "差分隐私结果已生成，并写入持久化隐私记账记录。" in js_text
    assert "当前全生命周期隐私预算不足，无法执行本次发布。" in js_text
    assert "尚未加载遥测数据。" in js_text


def test_09_api_json_field_names_not_localized(local_web_client):
    """9. API JSON field names 没有被中文化。"""
    # /api/health
    res_health = local_web_client.get("/api/health").json()
    assert "status" in res_health
    assert "version" in res_health
    assert "service" in res_health
    assert "local_only" in res_health

    # /api/config
    res_cfg = local_web_client.get("/api/config").json()
    assert "providers" in res_cfg
    assert "tasks" in res_cfg
    assert "metrics" in res_cfg
    assert "default_epsilon" in res_cfg
    assert "default_delta" in res_cfg
    assert "candidate_windows" in res_cfg
    assert "recipients" in res_cfg
    assert "continual_enabled" in res_cfg

    # /api/dataset/status
    res_ds = local_web_client.get("/api/dataset/status").json()
    assert "loaded" in res_ds
    assert "rows" in res_ds
    assert "active_users" in res_ds
    assert "source" in res_ds

    # /api/privacy/status
    res_priv = local_web_client.get("/api/privacy/status?recipient=provider_a").json()
    assert "budget_scope_id" in res_priv
    assert "effective_epsilon" in res_priv
    assert "epsilon_cap" in res_priv
    assert "remaining_headroom" in res_priv
    assert "delta_global" in res_priv


def test_10_provider_and_task_internal_ids_preserved(local_web_client):
    """10. provider/task 内部 ID 没有被修改。"""
    res_cfg = local_web_client.get("/api/config").json()
    assert set(res_cfg["providers"]) == {"provider_a", "provider_b", "provider_c"}
    assert set(res_cfg["tasks"]) == {"technical", "document", "analysis"}
    metric_ids = {m["id"] if isinstance(m, dict) else m for m in res_cfg["metrics"]}
    assert metric_ids == {
        "interaction_volume",
        "user_prevalence",
        "user_normalized_share",
    }

    # Verify HTML form radio button values remain untouched
    html = local_web_client.get("/").text
    assert 'value="interaction_volume"' in html
    assert 'value="user_prevalence"' in html
    assert 'value="user_normalized_share"' in html

    # Verify task translation mapping in JS preserves internal keys
    js_text = local_web_client.get("/static/app.js").text
    assert 'technical: "技术类"' in js_text
    assert 'document: "文档类"' in js_text
    assert 'analysis: "分析类"' in js_text
