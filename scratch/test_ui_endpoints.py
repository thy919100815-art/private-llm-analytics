from fastapi.testclient import TestClient
from private_llm_analytics.web.app import create_app

app = create_app()
client = TestClient(app)

res = client.get("/")
assert res.status_code == 200
html = res.text
assert "modal-demo-reset-epoch" in html
assert "modal-demo-reset-all" in html
assert "btn-demo-reset-epoch" in html
assert "btn-demo-reset-all" in html
assert "toast-container" in html
assert "交互记录总数" in html
assert "涉及模型数" in html
assert "任务类型" in html
assert "数据时间跨度" in html
assert "无明文 Prompt" in html
print("HTML verification passed!")

cfg = client.get("/api/config").json()
assert "demo_mode" in cfg
print("Config demo_mode:", cfg["demo_mode"])

ds = client.get("/api/dataset/status").json()
print("Dataset status loaded:", ds["loaded"], "time_range_display:", ds.get("time_range_display"))
print("Dataset quality checks count:", len(ds.get("quality_checks", [])))
for qc in ds.get("quality_checks", []):
    print("  QC:", qc["name"], qc["passed"])

ps = client.get("/api/privacy/status?recipient=provider_a").json()
print("Privacy status effective_epsilon:", ps["effective_epsilon"])
print("Privacy status epoch current_day:", ps["epoch"]["current_day"])
print("Privacy status lifetime exposure:", ps["lifetime_exposure"]["lifetime_effective_epsilon"])
