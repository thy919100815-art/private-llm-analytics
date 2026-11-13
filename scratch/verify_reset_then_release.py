from fastapi.testclient import TestClient
from private_llm_analytics.web.app import create_app

app = create_app()
client = TestClient(app)

print("Before reset, privacy status:")
ps1 = client.get("/api/privacy/status?recipient=provider_a").json()
print("  Interaction volume spent:", ps1["epoch"]["workloads"]["interaction_volume"]["spent_percentage"], "%")
print("  Reserve spent:", ps1["epoch"]["workloads"]["reserve"]["spent_percentage"], "%")

print("Resetting demo epoch...")
reset_res = client.post("/api/demo/reset-epoch")
assert reset_res.status_code == 200

print("After reset, privacy status:")
ps2 = client.get("/api/privacy/status?recipient=provider_a").json()
print("  Interaction volume spent:", ps2["epoch"]["workloads"]["interaction_volume"]["spent_percentage"], "%")
print("  Reserve spent:", ps2["epoch"]["workloads"]["reserve"]["spent_percentage"], "%")
assert ps2["epoch"]["workloads"]["interaction_volume"]["spent_percentage"] == 0.0

print("Now making release for provider_a / interaction_volume...")
client.post("/api/dataset/demo")
rel_res = client.post("/api/release", json={
    "metric": "interaction_volume",
    "recipient": "provider_a",
    "window_days": 30,
    "use_epoch_planner": True
})
print("Release status code:", rel_res.status_code)
assert rel_res.status_code == 200
rel = rel_res.json()
print("Release successful! release_id:", rel["release_id"], "incremental_privacy_cost:", rel["incremental_privacy_cost"])

print("Check privacy status after release:")
ps3 = client.get("/api/privacy/status?recipient=provider_a").json()
print("  Epoch effective_epsilon:", ps3["epoch"]["effective_epsilon"])
print("  Interaction volume spent:", ps3["epoch"]["workloads"]["interaction_volume"]["spent_percentage"], "%")
assert ps3["epoch"]["workloads"]["interaction_volume"]["spent_percentage"] == 100.0
assert ps3["epoch"]["effective_epsilon"] > 0
print("SUCCESS!")
