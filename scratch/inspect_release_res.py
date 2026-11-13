from fastapi.testclient import TestClient
from private_llm_analytics.web.app import create_app

app = create_app()
client = TestClient(app)
client.post("/api/dataset/demo")
rel_res = client.post("/api/release", json={
    "metric": "interaction_volume",
    "recipient": "provider_a",
    "window_days": 30,
    "use_epoch_planner": True
})
print("Status code:", rel_res.status_code)
rel = rel_res.json()
print("Release keys:", list(rel.keys()))
print("Release sample:", {k: rel[k] for k in ["release_id", "incremental_privacy_cost", "spent_percentage"] if k in rel})
