tests_to_append = '''

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
        assert csv_text.startswith("\\ufeff"), f"CSV for {m} must start with UTF-8 BOM"

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
'''

with open('tests/test_epoch_workload_budgeting.py', 'r', encoding='utf-8') as f:
    orig = f.read()

with open('tests/test_epoch_workload_budgeting.py', 'a', encoding='utf-8', newline='\r\n') as f:
    f.write(tests_to_append)

print("Successfully appended test_12, test_13, test_14")
