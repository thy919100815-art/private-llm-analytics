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
print("403 error:", rel_res.json())
