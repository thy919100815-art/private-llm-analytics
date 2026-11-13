import os
from fastapi.testclient import TestClient
from private_llm_analytics.web.app import create_app

app = create_app()
client = TestClient(app)

print("1. Loading demo data...")
ds = client.post("/api/dataset/demo").json()
print("   Loaded rows:", ds["rows"], "users:", ds["active_users"], "time_range_display:", ds["time_range_display"])
print("   Quality checks:")
for qc in ds["quality_checks"]:
    print(f"     {qc['name']}: {qc['passed']} ({qc['details']})")
assert ds["loaded"] is True
assert any(qc["name"] == "无明文 Prompt" and qc["passed"] for qc in ds["quality_checks"])

print("2. Making a release for provider_a / interaction_volume...")
rel_res = client.post("/api/release", json={
    "metric": "interaction_volume",
    "recipient": "provider_a",
    "window_days": 30,
    "use_epoch_planner": True
})
assert rel_res.status_code == 200
rel = rel_res.json()
print("   Release ID:", rel["release_id"], "effective_eps:", rel["effective_epsilon"])

print("3. Check Privacy Status after release...")
ps = client.get("/api/privacy/status?recipient=provider_a").json()
print("   Epoch eff_eps:", ps["epoch"]["effective_epsilon"])
print("   Lifetime eps:", ps["lifetime_exposure"]["lifetime_effective_epsilon"])
print("   Workload interaction_volume spent:", ps["epoch"]["workloads"]["interaction_volume"]["spent_epsilon"])
assert ps["epoch"]["effective_epsilon"] > 0
assert ps["epoch"]["workloads"]["interaction_volume"]["spent_epsilon"] > 0

print("4. Test Reset Demo Epoch...")
reset_epoch_res = client.post("/api/demo/reset-epoch")
assert reset_epoch_res.status_code == 200
re_data = reset_epoch_res.json()
assert re_data["success"] is True
assert re_data["effective_epsilon"] == 0.0

ps_after_epoch_reset = client.get("/api/privacy/status?recipient=provider_a").json()
print("   After reset epoch, epoch eff_eps:", ps_after_epoch_reset["epoch"]["effective_epsilon"])
print("   After reset epoch, lifetime eps:", ps_after_epoch_reset["lifetime_exposure"]["lifetime_effective_epsilon"])
print("   After reset epoch, interaction_volume spent:", ps_after_epoch_reset["epoch"]["workloads"]["interaction_volume"]["spent_epsilon"])
assert ps_after_epoch_reset["epoch"]["effective_epsilon"] == 0.0
assert ps_after_epoch_reset["epoch"]["workloads"]["interaction_volume"]["spent_epsilon"] == 0.0

print("5. Make another release to spend budget again...")
rel_res2 = client.post("/api/release", json={
    "metric": "top_k_tasks",
    "recipient": "provider_a",
    "window_days": 30,
    "top_k": 3,
    "use_epoch_planner": True
})
assert rel_res2.status_code == 200

print("6. Test Reset All (Restore Demo Initial State)...")
reset_all_res = client.post("/api/demo/reset-all")
assert reset_all_res.status_code == 200
ra_data = reset_all_res.json()
assert ra_data["success"] is True
assert ra_data["epoch_id"] == "epoch_001"
assert ra_data["effective_epsilon"] == 0.0

ps_after_all = client.get("/api/privacy/status?recipient=provider_a").json()
print("   After reset all, epoch eff_eps:", ps_after_all["epoch"]["effective_epsilon"])
print("   After reset all, lifetime eps:", ps_after_all["lifetime_exposure"]["lifetime_effective_epsilon"])
assert ps_after_all["epoch"]["effective_epsilon"] == 0.0
assert ps_after_all["lifetime_exposure"]["lifetime_effective_epsilon"] == 0.0

ds_after_all = client.get("/api/dataset/status").json()
print("   After reset all, dataset loaded:", ds_after_all["loaded"], "rows:", ds_after_all["rows"])
assert ds_after_all["loaded"] is True
assert ds_after_all["rows"] > 0

print("All End-to-End checks passed!")
