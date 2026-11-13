import urllib.request, urllib.parse, json

BASE = "http://127.0.0.1:8000"

def get(path):
    req = urllib.request.Request(BASE + path)
    with urllib.request.urlopen(req) as resp:
        return resp.status, json.loads(resp.read().decode())

def post(path, body=None):
    data = json.dumps(body).encode() if body is not None else None
    headers = {"Content-Type": "application/json"} if body is not None else {}
    req = urllib.request.Request(BASE + path, data=data, headers=headers, method="POST")
    with urllib.request.urlopen(req) as resp:
        return resp.status, json.loads(resp.read().decode())

print("Testing all features against live server...")

# 1. Health
s, d = get("/api/health")
print(f"1. /api/health: {s} - {d['status']}")

# 2. Config
s, d = get("/api/config")
print(f"2. /api/config: {s} - providers: {len(d['providers'])}, demo_mode: {d.get('demo_mode')}")

# 3. Load Demo Dataset
s, d = post("/api/dataset/demo")
print(f"3. /api/dataset/demo: {s} - rows: {d['rows']}")

# 4. Dataset status
s, d = get("/api/dataset/status")
print(f"4. /api/dataset/status: {s} - active_users: {d['active_users']}, time_range_display: {d.get('time_range_display')}")

# 5. Dataset preview
s, d = get("/api/dataset/preview?limit=10")
print(f"5. /api/dataset/preview: {s} - preview rows: {len(d['rows'])}")

# 6. Planner recommend
s, d = post("/api/planner/recommend", {"metric": "interaction_volume", "recipient": "provider_a", "planning_horizon_days": 90})
print(f"6. /api/planner/recommend: {s} - recommended_window_days: {d['recommended_window_days']}, clipping: {d['recommended_clipping']}")

# 7. Release interaction_volume
s, d = post("/api/release", {"metric": "interaction_volume", "recipient": "provider_a", "window_days": 30, "use_epoch_planner": True})
print(f"7. /api/release (interaction_volume): {s} - release_id: {d['release_id']}, results: {len(d['results'])}")
vol_id = d['release_id']

# 8. Release user_prevalence
s, d = post("/api/release", {"metric": "user_prevalence", "recipient": "provider_a", "window_days": 30, "use_epoch_planner": True})
print(f"8. /api/release (user_prevalence): {s} - release_id: {d['release_id']}, results: {len(d['results'])}")

# 9. Release user_normalized_share
s, d = post("/api/release", {"metric": "user_normalized_share", "recipient": "provider_a", "window_days": 30, "use_epoch_planner": True})
print(f"9. /api/release (user_normalized_share): {s} - release_id: {d['release_id']}, results: {len(d['results'])}")

# 10. Release success_rate (for internal)
s, d = post("/api/release", {"metric": "success_rate", "recipient": "internal", "window_days": 30, "use_epoch_planner": True})
print(f"10. /api/release (success_rate): {s} - release_id: {d['release_id']}, results: {len(d['results'])}")

# 11. Release top_k_tasks
s, d = post("/api/release", {"metric": "top_k_tasks", "recipient": "provider_a", "window_days": 30, "k": 5, "use_epoch_planner": True})
print(f"11. /api/release (top_k_tasks): {s} - release_id: {d['release_id']}, results: {len(d['results'])}")
topk_id = d['release_id']

# 12. List releases
s, d = get("/api/releases")
print(f"12. /api/releases: {s} - count: {len(d)}")

# 13. Download CSV
url_csv = f"{BASE}/api/release/download/csv?release_id={vol_id}&recipient=provider_a"
with urllib.request.urlopen(url_csv) as r:
    csv_bytes = r.read()
    print(f"13. download csv: {r.status} - size: {len(csv_bytes)} bytes")

# 14. Download JSON
url_json = f"{BASE}/api/release/download/json?release_id={vol_id}&recipient=provider_a"
with urllib.request.urlopen(url_json) as r:
    json_bytes = r.read()
    print(f"14. download json: {r.status} - size: {len(json_bytes)} bytes")

# 15. Privacy status
s, d = get("/api/privacy/status?recipient=provider_a")
print(f"15. /api/privacy/status: {s} - effective_eps: {d['epoch']['effective_epsilon']:.4f}")

print("\nALL SERVER ENDPOINTS WORKED PERFECTLY!")
