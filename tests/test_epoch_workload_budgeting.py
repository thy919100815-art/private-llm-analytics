"""Comprehensive test suite for 90-Day Privacy Epoch and Workload-driven RDP Budgeting."""

from decimal import Decimal
import json
from pathlib import Path
import pytest
from starlette.testclient import TestClient

from private_llm_analytics.config import load_app_config
from private_llm_analytics.privacy.epoch import (
    DEFAULT_WORKLOAD_WEIGHTS,
    EpochBudgetManager,
    EpochBudgetPlanner,
    PrivacyEpoch,
    WorkloadQuota,
)
from private_llm_analytics.privacy.exceptions import PrivacyBudgetExceededError
from private_llm_analytics.privacy.rdp import (
    DEFAULT_RDP_ORDERS,
    compose_rdp_vectors,
    rdp_to_epsilon_delta,
    to_decimal,
)
from private_llm_analytics.privacy.state import PrivacyStateManager
from private_llm_analytics.service.analytics import AnalyticsService
from private_llm_analytics.web.adapters import WebAnalyticsAdapter
from private_llm_analytics.web.app import create_app


@pytest.fixture
def isolated_state_manager(tmp_path):
    db_p = tmp_path / "test_epoch_state.db"
    return PrivacyStateManager(db_p)


@pytest.fixture
def web_test_client(tmp_path):
    config = load_app_config("configs")
    temp_db = tmp_path / "test_web_epoch_state.db"
    config.privacy.release.state_db_path = str(temp_db)

    service = AnalyticsService(config=config)
    adapter = WebAnalyticsAdapter(service_or_config_dir=service)
    app = create_app(adapter=adapter)
    app.state.adapter = adapter
    return TestClient(app)


def test_01_privacy_epoch_lifecycle_and_boundary(isolated_state_manager):
    """Test 90-day epoch initialization, day calculation, and advancing to a new epoch."""
    manager = EpochBudgetManager(state_manager=isolated_state_manager, epoch_days=90)
    epoch1 = manager.get_or_create_active_epoch(default_start_date="2026-10-01")

    assert epoch1.epoch_id == "epoch_001"
    assert epoch1.epoch_start == "2026-10-01"
    assert epoch1.epoch_end == "2026-12-29"
    assert epoch1.epoch_days == 90
    assert 1 <= epoch1.current_day <= 90
    assert epoch1.status == "ACTIVE"
    assert epoch1.epsilon_cap == Decimal("4.0")
    assert epoch1.effective_epsilon == 0.0

    # Advance to new epoch
    epoch2 = manager.advance_to_new_epoch()
    assert epoch2.epoch_id == "epoch_002"
    assert epoch2.epoch_start == "2026-12-30"
    assert epoch2.status == "ACTIVE"

    # Verify old epoch was closed, persisted, and not deleted
    all_epochs = isolated_state_manager.list_all_epochs()
    assert len(all_epochs) == 2
    rec1 = next(e for e in all_epochs if e["epoch_id"] == "epoch_001")
    assert rec1["status"] == "CLOSED"
    assert rec1["closed_at"] is not None


def test_02_workload_allocation_composition_le_4():
    """Verify that all 7 planned workload quotas mathematically compose to <= 4.0 under RDP."""
    planner = EpochBudgetPlanner(epoch_cap=4.0, delta_global=1e-6, continual_enabled=False)
    epoch = planner.create_epoch(epoch_id="epoch_test", start_date="2026-10-01")

    # Weights: volume(25%), success_rate(20%), prevalence(15%), share(10%), top_k(10%), continual(0%->reserve), reserve(20%)
    assert epoch.workloads["interaction_volume"].weight == pytest.approx(0.25)
    assert epoch.workloads["success_rate"].weight == pytest.approx(0.20)
    assert epoch.workloads["user_prevalence"].weight == pytest.approx(0.15)
    assert epoch.workloads["user_normalized_share"].weight == pytest.approx(0.10)
    assert epoch.workloads["top_k_tasks"].weight == pytest.approx(0.10)
    assert not epoch.workloads["continual"].is_supported
    assert epoch.workloads["reserve"].weight == pytest.approx(0.20)

    # Compose all allocated RDP vectors together
    all_vectors = [wl.allocated_rdp for wl in epoch.workloads.values() if wl.weight > 0]
    total_composed = compose_rdp_vectors(all_vectors, orders=epoch.rdp_orders)
    _, total_eps = rdp_to_epsilon_delta(total_composed, epoch.delta_global)

    # Authoritative composed epsilon must strictly not exceed 4.0000
    assert float(total_eps) <= 4.00001
    assert float(total_eps) >= 3.9999


def test_03_workload_noise_calibration():
    """Verify automatic noise sigma calibration for standard workloads."""
    planner = EpochBudgetPlanner(epoch_cap=4.0, delta_global=1e-6)

    # Volume: weight=0.25, sens=20.0
    sigma_vol, eps_vol, rdp_vol = planner.calibrate_workload_noise("interaction_volume", 20.0)
    assert sigma_vol > 0
    assert 1.90 <= eps_vol <= 2.00
    assert all(rdp_vol[alpha] > Decimal("0.0") for alpha in planner.orders)

    # Success rate: weight=0.20, sens=sqrt(2)*20
    sens_rate = (2.0 ** 0.5) * 20.0
    sigma_rate, eps_rate, _ = planner.calibrate_workload_noise("success_rate", sens_rate)
    assert sigma_rate > 0
    assert 1.70 <= eps_rate <= 1.80


def test_04_execution_priority_cache_hit_zero_cost(web_test_client):
    """Priority 1: Repeated requests for identical ReleaseKey return cached result with 0 budget cost."""
    web_test_client.post("/api/dataset/demo")

    # First release execution
    res1 = web_test_client.post(
        "/api/release",
        json={"metric": "interaction_volume", "recipient": "provider_a"},
    )
    assert res1.status_code == 200
    data1 = res1.json()
    assert not data1["cache_hit"]
    assert data1["incremental_privacy_cost"] > 0

    st1 = web_test_client.get("/api/privacy/status?recipient=provider_a").json()
    eps1 = st1["effective_epsilon"]
    rel_cnt1 = st1["release_count"]

    # Second release execution with identical parameters
    res2 = web_test_client.post(
        "/api/release",
        json={"metric": "interaction_volume", "recipient": "provider_a"},
    )
    assert res2.status_code == 200
    data2 = res2.json()
    assert data2["cache_hit"]
    assert data2["incremental_privacy_cost"] == 0.0

    st2 = web_test_client.get("/api/privacy/status?recipient=provider_a").json()
    # Budget spent and release count strictly unchanged
    assert st2["effective_epsilon"] == eps1
    assert st2["release_count"] == rel_cnt1


def test_05_execution_priority_workload_quota_then_reserve(isolated_state_manager):
    """Priority 3 & 4: Workload quota consumed first, then overflow consumes from Reserve."""
    manager = EpochBudgetManager(state_manager=isolated_state_manager, epoch_days=90)
    epoch = manager.get_or_create_active_epoch(default_start_date="2026-10-01")

    vol_quota = epoch.workloads["interaction_volume"]
    cand_rdp = dict(vol_quota.allocated_rdp)

    # First release fits in workload quota (Priority 3)
    ok, source, _ = epoch.can_execute_release("interaction_volume", cand_rdp)
    assert ok
    assert source == "WORKLOAD_QUOTA"
    manager.record_spend("interaction_volume", "rel_001", cand_rdp, source=source)

    # Refresh epoch: workload quota is now 100% spent
    epoch = manager.get_or_create_active_epoch()
    assert epoch.workloads["interaction_volume"].spent_percentage >= 99.9

    # Second release overflows workload quota -> falls back to Reserve (Priority 4)
    # Give a small RDP cost that fits in Reserve
    small_cand = {a: cand_rdp[a] * Decimal("0.5") for a in epoch.rdp_orders}
    ok2, source2, _ = epoch.can_execute_release("interaction_volume", small_cand)
    assert ok2
    assert source2 == "RESERVE"

    manager.record_spend("interaction_volume", "rel_002", small_cand, source=source2)
    epoch = manager.get_or_create_active_epoch()
    assert epoch.workloads["reserve"].release_count == 1
    assert epoch.workloads["reserve"].spent_percentage > 0


def test_06_execution_priority_fail_closed_when_over_cap(isolated_state_manager):
    """Priority 5: Rejects and fails closed when candidate RDP exceeds remaining epoch budget."""
    manager = EpochBudgetManager(state_manager=isolated_state_manager, epoch_days=90)
    epoch = manager.get_or_create_active_epoch()

    # Create candidate RDP that would push total RDP past epsilon=4.0
    huge_cand = {to_decimal(a): Decimal("10.0") * to_decimal(a) for a in epoch.rdp_orders}
    ok, source, reason = epoch.can_execute_release("interaction_volume", huge_cand)
    assert not ok
    assert source == "BLOCKED"
    assert "exceed Epoch ε cap" in reason or "exhausted" in reason


def test_07_lifetime_exposure_monitor_cross_epoch(isolated_state_manager):
    """Lifetime Exposure Monitor accumulates RDP across multiple epochs and displays compliant notice."""
    manager = EpochBudgetManager(state_manager=isolated_state_manager, epoch_days=90)
    epoch1 = manager.get_or_create_active_epoch(default_start_date="2026-10-01")

    # Spend some RDP in epoch 1
    sample_rdp = {to_decimal(a): Decimal("0.05") * to_decimal(a) for a in epoch1.rdp_orders}
    manager.record_spend("interaction_volume", "rel_e1", sample_rdp)

    lt1 = manager.get_lifetime_exposure()
    assert lt1.total_epochs_count == 1
    assert lt1.lifetime_effective_epsilon > 0
    eps1 = lt1.lifetime_effective_epsilon

    # Advance to epoch 2
    epoch2 = manager.advance_to_new_epoch()
    assert epoch2.status == "ACTIVE"
    # New epoch has fresh business ledger (0 effective epsilon in epoch 2)
    assert epoch2.effective_epsilon == 0.0

    # Lifetime exposure does NOT reset to 0! It retains epoch 1's exposure
    lt2 = manager.get_lifetime_exposure()
    assert lt2.total_epochs_count == 2
    assert lt2.lifetime_effective_epsilon == pytest.approx(eps1, rel=1e-3)
    assert "历史隐私损失不会消失" in lt2.notice

    # Spend in epoch 2
    manager.record_spend("success_rate", "rel_e2", sample_rdp)
    lt3 = manager.get_lifetime_exposure()
    # Lifetime exposure increases further
    assert lt3.lifetime_effective_epsilon > eps1


def test_08_sqlite_persistence_survives_restart(isolated_state_manager):
    """Epoch state, workload spend, and lifetime exposure survive process restarts identically."""
    manager1 = EpochBudgetManager(state_manager=isolated_state_manager, epoch_days=90)
    epoch1 = manager1.get_or_create_active_epoch(default_start_date="2026-10-01")

    sample_rdp = {to_decimal(a): Decimal("0.04") * to_decimal(a) for a in epoch1.rdp_orders}
    manager1.record_spend("user_prevalence", "rel_p1", sample_rdp)
    eps_before = manager1.get_or_create_active_epoch().effective_epsilon

    # Simulate restart by instantiating new EpochBudgetManager on the same SQLite state file
    manager2 = EpochBudgetManager(state_manager=isolated_state_manager, epoch_days=90)
    epoch_restored = manager2.get_or_create_active_epoch()

    assert epoch_restored.epoch_id == epoch1.epoch_id
    assert epoch_restored.effective_epsilon == pytest.approx(eps_before, rel=1e-4)
    assert "rel_p1" in epoch_restored.release_ids
    assert epoch_restored.workloads["user_prevalence"].release_count == 1

    lt_restored = manager2.get_lifetime_exposure()
    assert lt_restored.lifetime_effective_epsilon > 0


def test_09_dataset_preview_trusted_local_inspection(web_test_client):
    """Bug 1 fix: Data preview is safe local inspection without recipient RBAC or DP accounting."""
    web_test_client.post("/api/dataset/demo")

    # Preview works without recipient
    res = web_test_client.get("/api/dataset/preview?limit=20")
    assert res.status_code == 200
    data = res.json()
    assert data["preview_limit"] == 20
    assert len(data["rows"]) == 20

    # Privacy status unchanged
    st = web_test_client.get("/api/privacy/status?recipient=provider_a").json()
    assert st["effective_epsilon"] == 0.0
    assert st["release_count"] == 0


def test_10_dataset_status_time_range_formatting(web_test_client):
    """Bug 2 fix: Dataset status returns formatted date span and never [object Object]."""
    web_test_client.post("/api/dataset/demo")

    res = web_test_client.get("/api/dataset/status")
    assert res.status_code == 200
    data = res.json()

    assert data["time_range"] is not None
    assert "start" in data["time_range"]
    assert "end" in data["time_range"]
    assert "～" in data["time_range_display"]
    assert "[object Object]" not in str(data)


def test_11_top_k_tasks_authorization_and_permission_error_handling(web_test_client):
    """Bug fix: top_k_tasks authorized for provider_a and PermissionError maps to HTTP 403."""
    web_test_client.post("/api/dataset/demo")

    # 1. provider_a querying top_k_tasks succeeds
    res = web_test_client.post(
        "/api/release",
        json={
            "metric": "top_k_tasks",
            "recipient": "provider_a",
            "k": 3,
            "window_days": 30,
        },
    )
    assert res.status_code == 200
    data = res.json()
    assert data["metric"] == "top_k_tasks"
    assert "top_k" in data["results"]

    # 2. If a recipient is unauthorized for a metric, returns HTTP 403 (not 500)
    service = web_test_client.app.state.adapter.service
    orig_metrics = list(service.config.recipients.recipients["provider_a"].metrics)
    try:
        service.config.recipients.recipients["provider_a"].metrics = ["user_prevalence"]
        res_bad = web_test_client.post(
            "/api/release",
            json={
                "metric": "success_rate",
                "recipient": "provider_a",
                "window_days": 30,
            },
        )
        assert res_bad.status_code == 403
        err_data = res_bad.json()
        assert err_data["type"] == "PermissionError"
        assert "not authorized" in err_data["error"]
    finally:
        service.config.recipients.recipients["provider_a"].metrics = orig_metrics



def test_12_top_k_provider_isolation_pre_aggregation(isolated_state_manager):
    """Section 2: Top-K is provider-scoped before DP aggregation.
    
    Proof requirement: Fix provider_a data, massively alter provider_b and provider_c data.
    The exact pre-DP bounded contribution vector for provider_a MUST remain 100% bit-for-bit identical!
    """
    import pandas as pd
    import numpy as np
    from private_llm_analytics.aggregation.user_matrix import build_user_matrix
    from private_llm_analytics.metrics.top_k_tasks import TopKTasksMetric

    tasks = ["document", "technical", "analysis", "coding", "translation"]
    providers = ["provider_a", "provider_b", "provider_c"]

    # Dataset 1: Baseline dataset
    rows_base = [
        {"user_id": "u1", "provider": "provider_a", "task": "document"},
        {"user_id": "u1", "provider": "provider_a", "task": "technical"},
        {"user_id": "u2", "provider": "provider_a", "task": "document"},
        {"user_id": "u3", "provider": "provider_a", "task": "analysis"},
        # provider_b and provider_c have modest data
        {"user_id": "u1", "provider": "provider_b", "task": "coding"},
        {"user_id": "u2", "provider": "provider_c", "task": "translation"},
    ]
    df1 = pd.DataFrame(rows_base)
    umatrix1 = build_user_matrix(df1, providers=providers, tasks=tasks)

    # Dataset 2: provider_a data is strictly unchanged, but provider_b & provider_c are MASSIVELY modified
    rows_altered = list(rows_base[:4]) # Keep provider_a interactions identical
    # Add 10,000 interactions for provider_b and provider_c on diverse tasks
    for i in range(1000):
        rows_altered.append({"user_id": f"u_ext_{i}", "provider": "provider_b", "task": "coding"})
        rows_altered.append({"user_id": f"u_ext_{i}", "provider": "provider_c", "task": "translation"})
        rows_altered.append({"user_id": f"u_ext_{i}", "provider": "provider_b", "task": "document"})
    df2 = pd.DataFrame(rows_altered)
    umatrix2 = build_user_matrix(df2, providers=providers, tasks=tasks)

    metric = TopKTasksMetric(k=5, max_tasks_per_user=5)

    # Compute provider_a scoped pre-DP bounded aggregate vector for both datasets
    scoped_mat1 = umatrix1.filter_providers(["provider_a"])
    scoped_mat2 = umatrix2.filter_providers(["provider_a"])

    exact_a_1 = metric.compute_exact(scoped_mat1)
    exact_a_2 = metric.compute_exact(scoped_mat2)

    # Must be 100% bit-for-bit identical!
    np.testing.assert_array_equal(exact_a_1, exact_a_2)

    # Conversely, un-scoped global aggregation is severely polluted by competitor traffic
    global_1 = metric.compute_exact(umatrix1)
    global_2 = metric.compute_exact(umatrix2)
    assert not np.array_equal(global_1, global_2), "Global aggregation should change when competitor data changes"


def test_13_top_k_noisy_vector_reuse_across_k(web_test_client):
    """Section 12: Top-K noisy vector reuse across different k.
    
    Top-5 first -> privacy spend.
    Top-3 next -> 0 additional privacy spend (post-processing reuse).
    Top-5 again -> 0 additional privacy spend (cache hit).
    Also validates effective_k = min(k, task_count).
    """
    web_test_client.post("/api/dataset/demo")

    # 1. First Top-5 query for provider_a
    st_init = web_test_client.get("/api/privacy/status?recipient=provider_a").json()
    eps_init = st_init["effective_epsilon"]
    rel_count_init = st_init["release_count"]

    res_k5_1 = web_test_client.post("/api/release", json={
        "metric": "top_k_tasks",
        "recipient": "provider_a",
        "k": 5,
        "window_days": 30,
    })
    assert res_k5_1.status_code == 200
    d_k5_1 = res_k5_1.json()
    assert d_k5_1["cache_hit"] is False
    assert len(d_k5_1["results"]["top_k"]) == 5
    assert d_k5_1["results"]["effective_k"] == 5

    st_after_k5 = web_test_client.get("/api/privacy/status?recipient=provider_a").json()
    eps_after_k5 = st_after_k5["effective_epsilon"]
    rel_count_after_k5 = st_after_k5["release_count"]
    assert rel_count_after_k5 == rel_count_init + 1

    # 2. Query Top-3 on same dataset & window -> Must be 0 privacy spend!
    res_k3 = web_test_client.post("/api/release", json={
        "metric": "top_k_tasks",
        "recipient": "provider_a",
        "k": 3,
        "window_days": 30,
    })
    assert res_k3.status_code == 200
    d_k3 = res_k3.json()
    assert d_k3["cache_hit"] is True
    assert len(d_k3["results"]["top_k"]) == 3
    assert d_k3["results"]["effective_k"] == 3
    # Top 3 items must match the first 3 items of Top-5
    for rank in range(3):
        assert d_k3["results"]["top_k"][rank]["task"] == d_k5_1["results"]["top_k"][rank]["task"]
        assert d_k3["results"]["top_k"][rank]["noisy_count"] == d_k5_1["results"]["top_k"][rank]["noisy_count"]

    st_after_k3 = web_test_client.get("/api/privacy/status?recipient=provider_a").json()
    assert st_after_k3["effective_epsilon"] == eps_after_k5
    assert st_after_k3["release_count"] == rel_count_after_k5

    # 3. Query Top-5 again -> Cache hit, 0 privacy spend!
    res_k5_2 = web_test_client.post("/api/release", json={
        "metric": "top_k_tasks",
        "recipient": "provider_a",
        "k": 5,
        "window_days": 30,
    })
    assert res_k5_2.status_code == 200
    d_k5_2 = res_k5_2.json()
    assert d_k5_2["cache_hit"] is True
    assert len(d_k5_2["results"]["top_k"]) == 5

    st_after_k5_2 = web_test_client.get("/api/privacy/status?recipient=provider_a").json()
    assert st_after_k5_2["effective_epsilon"] == eps_after_k5
    assert st_after_k5_2["release_count"] == rel_count_after_k5


def test_14_download_csv_and_json_zero_privacy_spend_all_metrics(web_test_client):
    """Section 3 & 4: Download DP CSV & JSON has zero privacy spend for all 5 metrics.
    
    Verifies:
      - UTF-8 BOM on CSV for Excel compatibility
      - Top-K CSV has rank, task, noisy_count, estimated_prevalence
      - JSON does not leak exact ground truth or private internals
      - Before vs After download: exactly 0 budget consumed
    """
    web_test_client.post("/api/dataset/demo")

    metrics_to_test = [
        "interaction_volume",
        "user_prevalence",
        "user_normalized_share",
        "success_rate",
        "top_k_tasks",
    ]

    for m in metrics_to_test:
        # Step A: Perform release
        res = web_test_client.post("/api/release", json={
            "metric": m,
            "recipient": "provider_a",
            "window_days": 30,
        })
        assert res.status_code == 200, f"Release failed for {m}: {res.text}"
        rel_data = res.json()
        rel_id = rel_data.get("release_id") or rel_data.get("release_key")
        assert rel_id is not None

        # Record privacy state before download
        st_before = web_test_client.get("/api/privacy/status?recipient=provider_a").json()
        eps_before = st_before["effective_epsilon"]
        count_before = st_before["release_count"]

        # Step B: Download CSV
        res_csv = web_test_client.get(f"/api/release/download/csv?release_id={rel_id}&recipient=provider_a")
        assert res_csv.status_code == 200, f"CSV download failed for {m}: {res_csv.text}"
        csv_text = res_csv.text
        # Assert UTF-8 BOM
        assert csv_text.startswith("\ufeff"), f"CSV for {m} must start with UTF-8 BOM"

        if m == "top_k_tasks":
            assert "rank" in csv_text
            assert "task" in csv_text
            assert "noisy_count" in csv_text
            assert "estimated_prevalence" in csv_text

        # Step C: Download JSON
        res_json = web_test_client.get(f"/api/release/download/json?release_id={rel_id}&recipient=provider_a")
        assert res_json.status_code == 200, f"JSON download failed for {m}: {res_json.text}"
        json_obj = res_json.json()

        # Security check: must not leak exact ground truth or raw internals
        assert "exact_ground_truth" not in json_obj
        assert "clean_vector" not in json_obj
        assert "user_ids" not in json_obj
        assert "rng" not in json_obj
        assert "noise_seed" not in json_obj
        assert json_obj["recipient"] == "provider_a"

        # Step D: Record privacy state after download and assert 0 privacy spend
        st_after = web_test_client.get("/api/privacy/status?recipient=provider_a").json()
        assert st_after["effective_epsilon"] == eps_before, f"Downloading {m} must not increase effective epsilon"
        assert st_after["release_count"] == count_before, f"Downloading {m} must not increment release count"
