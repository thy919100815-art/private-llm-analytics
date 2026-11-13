import os
import sys
from fastapi.testclient import TestClient

def run_demo_reset_e2e():
    print("=" * 70)
    print("RUNNING DEMO RESET E2E FREEZE VERIFICATION")
    print("=" * 70)

    # -------------------------------------------------------------
    # Phase 1: Explicit Demo Environment (APP_ENV=local-demo, DEMO_MODE=true)
    # -------------------------------------------------------------
    os.environ["APP_ENV"] = "local-demo"
    os.environ["DEMO_MODE"] = "true"
    print("\n--- PHASE 1: DEMO MODE (APP_ENV=local-demo, DEMO_MODE=true) ---")

    from private_llm_analytics.web.app import create_app
    app = create_app()
    client = TestClient(app)

    # 1. Config returns demo_mode = True
    cfg = client.get("/api/config").json()
    assert cfg.get("demo_mode") is True, f"Expected demo_mode=True, got {cfg.get('demo_mode')}"
    print("✓ [Step 1] /api/config returned demo_mode=True")

    # 2. Load demo dataset
    ds_res = client.post("/api/dataset/demo")
    assert ds_res.status_code == 200
    print(f"✓ [Step 2] Demo dataset loaded: {ds_res.json()['rows']} rows")

    # 3. Perform DP release
    rel_res = client.post("/api/release", json={
        "metric": "interaction_volume",
        "recipient": "provider_a",
        "window_days": 30,
        "use_epoch_planner": True
    })
    assert rel_res.status_code == 200
    rel_id = rel_res.json()["release_id"]
    print(f"✓ [Step 3] DP Release succeeded, release_id: {rel_id}")

    # 4. Check epsilon > 0
    priv1 = client.get("/api/privacy/status?recipient=provider_a").json()
    eps1 = priv1["epoch"]["effective_epsilon"]
    spent_pct1 = priv1["epoch"]["workloads"]["interaction_volume"]["spent_percentage"]
    assert eps1 > 0, f"Expected epsilon > 0, got {eps1}"
    assert spent_pct1 > 0, f"Expected spent_pct > 0, got {spent_pct1}"
    print(f"✓ [Step 4] Epsilon before reset: {eps1:.4f}, interaction_volume spent: {spent_pct1:.1f}%")

    # 5. Reset Epoch
    reset_epoch_res = client.post("/api/demo/reset-epoch")
    assert reset_epoch_res.status_code == 200
    reset_epoch_data = reset_epoch_res.json()
    assert reset_epoch_data["success"] is True
    print(f"✓ [Step 5] Reset Epoch completed: {reset_epoch_data['message']}")

    # 6. Verify current Epoch epsilon = 0
    priv2 = client.get("/api/privacy/status?recipient=provider_a").json()
    eps2 = priv2["epoch"]["effective_epsilon"]
    spent_pct2 = priv2["epoch"]["workloads"]["interaction_volume"]["spent_percentage"]
    assert eps2 == 0.0, f"Expected epsilon == 0, got {eps2}"
    assert spent_pct2 == 0.0, f"Expected spent_pct == 0, got {spent_pct2}"
    print(f"✓ [Step 6] Current Epoch epsilon after reset: {eps2:.4f}, spent: {spent_pct2:.1f}%")

    # 7. Release can be re-run cleanly
    rel_res2 = client.post("/api/release", json={
        "metric": "interaction_volume",
        "recipient": "provider_a",
        "window_days": 30,
        "use_epoch_planner": True
    })
    assert rel_res2.status_code == 200
    rel_id2 = rel_res2.json()["release_id"]
    print(f"✓ [Step 7] Re-running DP Release succeeded after epoch reset, release_id: {rel_id2}")

    # 8. Check Lifetime Exposure > 0
    priv3 = client.get("/api/privacy/status?recipient=provider_a").json()
    lifetime_eps = priv3["lifetime_exposure"]["lifetime_effective_epsilon"]
    assert lifetime_eps > 0, f"Expected lifetime_eps > 0, got {lifetime_eps}"
    print(f"✓ [Step 8] Lifetime Exposure before reset all: {lifetime_eps:.4f}")

    # 9. Reset All
    reset_all_res = client.post("/api/demo/reset-all")
    assert reset_all_res.status_code == 200
    reset_all_data = reset_all_res.json()
    assert reset_all_data["success"] is True
    assert reset_all_data["epoch_id"] == "epoch_001"
    print(f"✓ [Step 9] Reset All completed: {reset_all_data['message']}")

    # 10. Verify Epoch / Lifetime / Release History all restored to initial demo state
    priv4 = client.get("/api/privacy/status?recipient=provider_a").json()
    assert priv4["epoch"]["epoch_id"] == "epoch_001"
    assert priv4["epoch"]["effective_epsilon"] == 0.0
    assert priv4["lifetime_exposure"]["lifetime_effective_epsilon"] == 0.0
    assert priv4["lifetime_exposure"]["total_epochs_count"] == 1
    
    history_res = client.get("/api/releases").json()
    assert len(history_res) == 0, f"Expected 0 releases in history, got {len(history_res)}"
    print("✓ [Step 10] Epoch, Lifetime Exposure, and Release History fully restored to clean Demo state.")

    # -------------------------------------------------------------
    # Phase 2: Production Mode (APP_ENV=production, DEMO_MODE=false)
    # -------------------------------------------------------------
    os.environ["APP_ENV"] = "production"
    os.environ["DEMO_MODE"] = "false"
    print("\n--- PHASE 2: PRODUCTION MODE (APP_ENV=production, DEMO_MODE=false) ---")

    app_prod = create_app()
    client_prod = TestClient(app_prod)

    # 1. Config returns demo_mode = False (controls hidden on frontend)
    cfg_prod = client_prod.get("/api/config").json()
    assert cfg_prod.get("demo_mode") is False, f"Expected demo_mode=False, got {cfg_prod.get('demo_mode')}"
    print("✓ [Step 11] /api/config returned demo_mode=False (Reset controls hidden on UI)")

    # 2. Reset API calls return 403 Forbidden
    res_epoch_prod = client_prod.post("/api/demo/reset-epoch")
    assert res_epoch_prod.status_code == 403, f"Expected 403, got {res_epoch_prod.status_code}"
    print(f"✓ [Step 12] POST /api/demo/reset-epoch strictly rejected with HTTP 403 Forbidden: {res_epoch_prod.json()['detail']}")

    res_all_prod = client_prod.post("/api/demo/reset-all")
    assert res_all_prod.status_code == 403, f"Expected 403, got {res_all_prod.status_code}"
    print(f"✓ [Step 13] POST /api/demo/reset-all strictly rejected with HTTP 403 Forbidden: {res_all_prod.json()['detail']}")

    # 3. Normal DP query can still execute successfully in production
    client_prod.post("/api/dataset/demo")
    rel_prod_res = client_prod.post("/api/release", json={
        "metric": "interaction_volume",
        "recipient": "provider_a",
        "window_days": 30,
        "use_epoch_planner": True
    })
    assert rel_prod_res.status_code == 200, f"Expected 200, got {rel_prod_res.status_code}: {rel_prod_res.text}"
    print(f"✓ [Step 14] Normal DP Release query executed successfully in Production Mode: {rel_prod_res.json()['release_id']}")

    print("\n" + "=" * 70)
    print("ALL DEMO RESET & PRODUCTION ISOLATION TESTS VERIFIED SUCCESSFULLY!")
    print("=" * 70)

if __name__ == "__main__":
    run_demo_reset_e2e()
