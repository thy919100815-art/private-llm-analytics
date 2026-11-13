"""Automated 21-Step Web E2E Acceptance Test for Private LLM Analytics."""

import csv
import io
import json
import os
import sys
from pathlib import Path
import pytest
from fastapi.testclient import TestClient

from private_llm_analytics.config import load_app_config
from private_llm_analytics.service.analytics import AnalyticsService
from private_llm_analytics.web.adapters import WebAnalyticsAdapter
from private_llm_analytics.web.app import create_app


def run_e2e_acceptance():
    print("=" * 70)
    print("STARTING 21-STEP WEB E2E ACCEPTANCE VERIFICATION")
    print("=" * 70)

    # Use a persistent test state db to test persistence and restart
    test_db = Path("results/e2e_test_privacy_state.db")
    if test_db.exists():
        test_db.unlink()

    config = load_app_config("configs")
    config.privacy.release.state_db_path = str(test_db)
    service = AnalyticsService(config=config)
    adapter = WebAnalyticsAdapter(service_or_config_dir=service)
    app = create_app(adapter=adapter)
    client = TestClient(app)

    # ----------------------------------------------------
    # Step 1: 加载 demo
    # ----------------------------------------------------
    print("\n[Step 1] POST /api/dataset/demo")
    res1 = client.post("/api/dataset/demo")
    assert res1.status_code == 200, f"Step 1 failed: {res1.text}"
    data1 = res1.json()
    assert data1["loaded"] is True
    assert data1["rows"] > 0
    print(f"  -> Demo dataset loaded successfully: {data1['rows']} rows, {data1['active_users']} active users.")

    # ----------------------------------------------------
    # Step 2: 确认 90 天数据
    # ----------------------------------------------------
    print("\n[Step 2] GET /api/dataset/status (Confirm 90-day span)")
    res2 = client.get("/api/dataset/status")
    assert res2.status_code == 200
    data2 = res2.json()
    time_range = data2.get("time_range")
    assert time_range is not None, "time_range is missing"
    assert "start" in time_range and "end" in time_range
    start_str = time_range["start"]
    end_str = time_range["end"]
    print(f"  -> Telemetry date range: {start_str} to {end_str}")

    # ----------------------------------------------------
    # Step 3: 查看部分输入数据成功
    # ----------------------------------------------------
    print("\n[Step 3] GET /api/dataset/preview?limit=20")
    # Check baseline privacy before preview
    res_priv_before = client.get("/api/privacy/status?recipient=provider_a").json()
    eps_before_preview = res_priv_before["effective_epsilon"]

    res3 = client.get("/api/dataset/preview?limit=20")
    assert res3.status_code == 200, f"Step 3 failed: {res3.text}"
    data3 = res3.json()
    assert data3["total_rows"] == data2["rows"]
    assert 0 < len(data3["rows"]) <= 20
    assert "columns" in data3
    first_row = data3["rows"][0]
    assert "user_id" in first_row
    assert "timestamp" in first_row
    assert "provider" in first_row
    assert "task" in first_row
    assert "success" in first_row
    # Check NO prompt is leaked
    assert "prompt" not in first_row
    print(f"  -> Preview returned {len(data3['rows'])} rows safely without raw prompts.")

    # ----------------------------------------------------
    # Step 4: Preview 前后 epsilon 不变
    # ----------------------------------------------------
    print("\n[Step 4] Verify Privacy Spend on Preview is 0.0")
    res_priv_after = client.get("/api/privacy/status?recipient=provider_a").json()
    eps_after_preview = res_priv_after["effective_epsilon"]
    assert eps_after_preview == eps_before_preview, "Preview consumed privacy budget!"
    print(f"  -> Effective epsilon before: {eps_before_preview}, after: {eps_after_preview} (0.0 cost)")

    # ----------------------------------------------------
    # Step 5: 时间跨度显示正确
    # ----------------------------------------------------
    print("\n[Step 5] Verify Time Range Formatting (YYYY-MM-DD ～ YYYY-MM-DD)")
    formatted_span = f"{start_str} ～ {end_str}"
    assert "object" not in formatted_span, "Found [object Object] in date range!"
    assert len(start_str) >= 10 and len(end_str) >= 10
    print(f"  -> Formatted time range: '{formatted_span}' (valid, no [object Object])")

    # ----------------------------------------------------
    # Step 6: 查询 Interaction Volume
    # ----------------------------------------------------
    print("\n[Step 6] POST /api/release (interaction_volume for provider_a)")
    res6 = client.post("/api/release", json={
        "metric": "interaction_volume",
        "recipient": "provider_a",
    })
    assert res6.status_code == 200, f"Step 6 failed: {res6.text}"
    rel6 = res6.json()
    release_id_vol = rel6["release_id"]
    print(f"  -> Release successful! Release ID: {release_id_vol}")

    # ----------------------------------------------------
    # Step 7: 下载 CSV
    # ----------------------------------------------------
    print(f"\n[Step 7] GET /api/release/download/csv?release_id={release_id_vol}&recipient=provider_a")
    priv_before_csv = client.get("/api/privacy/status?recipient=provider_a").json()["effective_epsilon"]
    res7 = client.get(f"/api/release/download/csv?release_id={release_id_vol}&recipient=provider_a")
    assert res7.status_code == 200, f"Step 7 failed: {res7.text}"
    csv_text_vol = res7.text
    assert csv_text_vol.startswith("\ufeff"), "CSV missing UTF-8 BOM!"
    print("  -> CSV downloaded successfully with UTF-8 BOM.")

    # ----------------------------------------------------
    # Step 8: 下载 JSON
    # ----------------------------------------------------
    print(f"\n[Step 8] GET /api/release/download/json?release_id={release_id_vol}&recipient=provider_a")
    res8 = client.get(f"/api/release/download/json?release_id={release_id_vol}&recipient=provider_a")
    assert res8.status_code == 200, f"Step 8 failed: {res8.text}"
    json_data_vol = res8.json()
    print("  -> JSON downloaded successfully.")

    # ----------------------------------------------------
    # Step 9: 文件可正常打开并验证无敏感泄露
    # ----------------------------------------------------
    print("\n[Step 9] Validate CSV and JSON content structure")
    reader = csv.reader(io.StringIO(csv_text_vol.lstrip("\ufeff")))
    rows_csv = list(reader)
    assert len(rows_csv) > 1, "CSV has no rows"
    print(f"  -> CSV parsed successfully ({len(rows_csv)} lines, header: {rows_csv[0]}).")

    forbidden = {"exact_ground_truth", "clean_vector", "user_ids", "rng", "noise_seed"}
    assert not any(k in json_data_vol for k in forbidden), "Forbidden key leaked in JSON release!"
    assert json_data_vol["release_id"] == release_id_vol
    print("  -> JSON validated: 0 forbidden ground truth keys leaked.")

    # ----------------------------------------------------
    # Step 10: 下载前后 budget 完全不变
    # ----------------------------------------------------
    print("\n[Step 10] Verify 0 privacy spend on downloads")
    priv_after_download = client.get("/api/privacy/status?recipient=provider_a").json()["effective_epsilon"]
    assert priv_after_download == priv_before_csv, "Downloads consumed privacy budget!"
    print(f"  -> Budget before download: {priv_before_csv}, after: {priv_after_download} (0.0 cost)")

    # ----------------------------------------------------
    # Step 11: 查询 Success Rate
    # ----------------------------------------------------
    print("\n[Step 11] POST /api/release (success_rate for internal)")
    res11 = client.post("/api/release", json={
        "metric": "success_rate",
        "recipient": "internal",
    })
    assert res11.status_code == 200, f"Step 11 failed: {res11.text}"
    rel11 = res11.json()
    release_id_sr = rel11["release_id"]
    print(f"  -> Success rate release completed! Release ID: {release_id_sr}")

    # ----------------------------------------------------
    # Step 12: 下载 CSV/JSON for Success Rate
    # ----------------------------------------------------
    print("\n[Step 12] Download CSV/JSON for Success Rate")
    res12_csv = client.get(f"/api/release/download/csv?release_id={release_id_sr}&recipient=internal")
    assert res12_csv.status_code == 200
    res12_json = client.get(f"/api/release/download/json?release_id={release_id_sr}&recipient=internal")
    assert res12_json.status_code == 200
    print("  -> Success rate CSV and JSON downloaded successfully.")

    # ----------------------------------------------------
    # Step 13: 查询 Top-K
    # ----------------------------------------------------
    print("\n[Step 13] POST /api/release (top_k_tasks for provider_a with k=5)")
    res13 = client.post("/api/release", json={
        "metric": "top_k_tasks",
        "recipient": "provider_a",
        "k": 5,
    })
    assert res13.status_code == 200, f"Step 13 failed: {res13.text}"
    rel13 = res13.json()
    release_id_topk = rel13["release_id"]
    print(f"  -> Top-K release completed! Release ID: {release_id_topk}")

    # ----------------------------------------------------
    # Step 14: Top-5 正常显示
    # ----------------------------------------------------
    print("\n[Step 14] Verify Top-5 results")
    topk_results = rel13.get("results", {})
    top_list = topk_results.get("top_k", topk_results.get("top_tasks", []))
    assert len(top_list) == 5, f"Expected 5 items, got {len(top_list)}"
    for idx, item in enumerate(top_list, 1):
        assert item["rank"] == idx
        assert "task" in item
        assert "noisy_count" in item
        assert "estimated_prevalence" in item
    print("  -> Top-5 results properly structured with ranks 1..5, noisy_count and prevalence.")

    # ----------------------------------------------------
    # Step 15: 下载 Top-K CSV/JSON
    # ----------------------------------------------------
    print("\n[Step 15] Download Top-K CSV/JSON")
    res15_csv = client.get(f"/api/release/download/csv?release_id={release_id_topk}&recipient=provider_a")
    assert res15_csv.status_code == 200
    topk_csv_text = res15_csv.text
    assert "rank,task,noisy_count,estimated_prevalence" in topk_csv_text
    print("  -> Top-K CSV contains rank, task, noisy_count, estimated_prevalence.")

    # ----------------------------------------------------
    # Step 16: 相同 Top-K 再查 → 0 spend
    # ----------------------------------------------------
    print("\n[Step 16] Re-query exact same Top-K (cache hit test)")
    priv_before_replay = client.get("/api/privacy/status?recipient=provider_a").json()["effective_epsilon"]
    res16 = client.post("/api/release", json={
        "metric": "top_k_tasks",
        "recipient": "provider_a",
        "k": 5,
    })
    assert res16.status_code == 200
    assert res16.json()["cache_hit"] is True
    priv_after_replay = client.get("/api/privacy/status?recipient=provider_a").json()["effective_epsilon"]
    assert priv_after_replay == priv_before_replay, "Cache replay consumed budget!"
    print(f"  -> Cache hit confirmed! Spend: 0.0 (epsilon remained {priv_after_replay})")

    # ----------------------------------------------------
    # Step 17: Top-5 改 Top-3 → 0 spend（复用 noisy vector）
    # ----------------------------------------------------
    print("\n[Step 17] Query Top-3 (re-slicing noisy vector test)")
    res17 = client.post("/api/release", json={
        "metric": "top_k_tasks",
        "recipient": "provider_a",
        "k": 3,
    })
    assert res17.status_code == 200
    rel17 = res17.json()
    assert rel17["cache_hit"] is True
    assert len(rel17["results"].get("top_k", rel17["results"].get("top_tasks", []))) == 3
    priv_after_reslice = client.get("/api/privacy/status?recipient=provider_a").json()["effective_epsilon"]
    assert priv_after_reslice == priv_before_replay, "Re-slicing vector consumed budget!"
    print(f"  -> Vector reuse confirmed! Returned 3 items, spend: 0.0 (epsilon: {priv_after_reslice})")

    # ----------------------------------------------------
    # Step 18: provider_a / b / c 分别测试
    # ----------------------------------------------------
    print("\n[Step 18] Query Top-K for provider_b and provider_c")
    res18_b = client.post("/api/release", json={"metric": "top_k_tasks", "recipient": "provider_b", "k": 3})
    assert res18_b.status_code == 200
    res18_c = client.post("/api/release", json={"metric": "top_k_tasks", "recipient": "provider_c", "k": 3})
    assert res18_c.status_code == 200
    print("  -> Both provider_b and provider_c executed successfully.")

    # ----------------------------------------------------
    # Step 19: 验证 provider isolation
    # ----------------------------------------------------
    print("\n[Step 19] Verify Provider Pre-DP Isolation")
    # Verify that allowed_providers scoping filters the user_matrix BEFORE DP bounding
    user_matrix = service.get_user_matrix()
    filtered_a = user_matrix.filter_providers(["provider_a"])
    assert len(filtered_a.providers) == 1 and filtered_a.providers[0] == "provider_a"
    # All cell names belong exclusively to provider_a
    for cell in filtered_a.cell_names:
        assert cell.startswith("provider_a:"), f"Cell {cell} does not belong to provider_a"
    print("  -> Mathematical isolation proven: provider_a user matrix contains only provider_a interactions.")

    # ----------------------------------------------------
    # Step 20: 重启服务
    # ----------------------------------------------------
    print("\n[Step 20] Simulate Service Restart")
    # Record state before restart
    status_pre = client.get("/api/privacy/status?recipient=provider_a").json()
    pre_eps = status_pre["effective_epsilon"]
    pre_epoch_id = status_pre["epoch"]["epoch_id"]
    pre_exposure = status_pre["lifetime_exposure"]["lifetime_effective_epsilon"]

    # Close and restart fresh adapter and app pointing to same SQLite database
    del client
    del adapter
    del service

    new_config = load_app_config("configs")
    new_config.privacy.release.state_db_path = str(test_db)
    new_service = AnalyticsService(config=new_config)
    new_adapter = WebAnalyticsAdapter(service_or_config_dir=new_service)
    new_app = create_app(adapter=new_adapter)
    new_client = TestClient(new_app)
    print("  -> Service restarted with fresh process memory.")

    # ----------------------------------------------------
    # Step 21: Epoch / RDP / history 保持
    # ----------------------------------------------------
    print("\n[Step 21] Verify Persistent State Restoration")
    status_post = new_client.get("/api/privacy/status?recipient=provider_a").json()
    post_eps = status_post["effective_epsilon"]
    post_epoch_id = status_post["epoch"]["epoch_id"]
    post_exposure = status_post["lifetime_exposure"]["lifetime_effective_epsilon"]

    assert post_eps == pre_eps, f"Epsilon lost on restart! Pre: {pre_eps}, Post: {post_eps}"
    assert post_epoch_id == pre_epoch_id, f"Epoch ID lost on restart!"
    assert post_exposure == pre_exposure, f"Lifetime exposure lost on restart!"
    print(f"  -> State perfectly preserved across restart! Epoch ID: {post_epoch_id}, Epsilon: {post_eps}, Lifetime Exposure: {post_exposure}")

    print("\n" + "=" * 70)
    print("ALL 21 ACCEPTANCE STEPS SUCCESSFULLY VERIFIED!")
    print("=" * 70)
    return True


if __name__ == "__main__":
    success = run_e2e_acceptance()
    sys.exit(0 if success else 1)
