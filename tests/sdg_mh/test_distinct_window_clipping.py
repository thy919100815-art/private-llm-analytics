"""Unit tests for Distinct Window-Level Contribution Clipping and Sensitivity Verification.

Verifies:
1. Multi-day user with > k distinct tasks is strictly bounded to k tasks across the entire window.
2. In-memory RollingDistinctState and direct compute_distinct_window_histogram agree exactly.
3. Neighboring dataset L2^2 sensitivity is strictly bounded by Delta^2 = m + 2k (replace) and m + k (add/remove).
"""

import numpy as np
import pandas as pd
import pytest

from private_llm_analytics.sdg_mh.clipping import compute_distinct_window_histogram
from private_llm_analytics.sdg_mh.rolling_state import RollingDistinctState
from private_llm_analytics.ingestion.schema import (
    COL_USER_ID,
    COL_PROVIDER,
    COL_TASK,
    COL_TIMESTAMP,
    COL_SUCCESS,
)


@pytest.fixture
def sample_taxonomy():
    services = ["prov_1", "prov_2", "prov_3", "prov_4", "prov_5"]  # m = 5
    tasks = [f"task_{i:02d}" for i in range(12)]  # 12 tasks
    return services, tasks


def test_window_level_task_clipping_counterexample(sample_taxonomy, tmp_path):
    """Counterexample: User u1 has 6 different tasks across 3 days in a 7-day window.

    Day 1: task_00 (2 calls), task_01 (1 call)
    Day 2: task_02 (1 call),  task_03 (1 call)
    Day 3: task_04 (1 call),  task_05 (1 call)
    Total distinct tasks used: 6.
    Requirement: Window-level distinct clipping must restrict user's contribution to at most k=3 tasks.
    """
    services, tasks = sample_taxonomy
    k_tasks_per_user = 3

    rows = [
        {"user_id": "u1", "provider": "prov_1", "task": "task_00", "timestamp": "2025-01-01 10:00:00", "success": True},
        {"user_id": "u1", "provider": "prov_1", "task": "task_00", "timestamp": "2025-01-01 10:05:00", "success": True},
        {"user_id": "u1", "provider": "prov_2", "task": "task_01", "timestamp": "2025-01-01 10:10:00", "success": True},
        {"user_id": "u1", "provider": "prov_2", "task": "task_02", "timestamp": "2025-01-02 10:00:00", "success": True},
        {"user_id": "u1", "provider": "prov_3", "task": "task_03", "timestamp": "2025-01-02 10:05:00", "success": True},
        {"user_id": "u1", "provider": "prov_4", "task": "task_04", "timestamp": "2025-01-03 10:00:00", "success": True},
        {"user_id": "u1", "provider": "prov_5", "task": "task_05", "timestamp": "2025-01-03 10:05:00", "success": True},
    ]
    df_window = pd.DataFrame(rows)
    df_window[COL_TIMESTAMP] = pd.to_datetime(df_window[COL_TIMESTAMP])

    # 1. Test compute_distinct_window_histogram
    s_cnt, t_cnt = compute_distinct_window_histogram(df_window, services, tasks, k_tasks_per_user=k_tasks_per_user)
    assert t_cnt.sum() == k_tasks_per_user, f"Expected {k_tasks_per_user} tasks, got {t_cnt.sum()}"
    assert np.count_nonzero(t_cnt) == k_tasks_per_user
    # Deterministic tie-breaking: task_00 (freq 2), task_01 (freq 1, idx 1), task_02 (freq 1, idx 2)
    assert t_cnt[0] == 1.0
    assert t_cnt[1] == 1.0
    assert t_cnt[2] == 1.0
    assert t_cnt[3] == 0.0
    assert t_cnt[4] == 0.0
    assert t_cnt[5] == 0.0

    # 2. Test RollingDistinctState over days 1..7 (days 4..7 empty)
    rs = RollingDistinctState(storage_dir=tmp_path / "rs_test", w_max=90)
    for d in range(1, 8):
        df_d = df_window[df_window[COL_TIMESTAMP].dt.day == d]
        rs.append_day(d, df_d, services, tasks, persist_disk=False)

    s_cnt_rs, t_cnt_rs = rs.compute_window_histogram(1, 7, services, tasks, k_tasks_per_user=k_tasks_per_user)
    assert t_cnt_rs.sum() == k_tasks_per_user
    np.testing.assert_array_equal(t_cnt, t_cnt_rs)
    np.testing.assert_array_equal(s_cnt, s_cnt_rs)


def test_neighboring_sensitivity_bounds(sample_taxonomy):
    """Verify that user-level L2^2 sensitivity is strictly bounded by Delta^2 = m + 2k."""
    services, tasks = sample_taxonomy
    m = len(services)
    k = 3

    # Dataset D: multiple users active across 7 days
    base_rows = []
    for u in range(10):
        for d in range(1, 8):
            base_rows.append({
                "user_id": f"user_{u:03d}",
                "provider": services[u % m],
                "task": tasks[(u + d) % len(tasks)],
                "timestamp": f"2025-01-{d:02d} 12:00:00",
                "success": True,
            })
    df_base = pd.DataFrame(base_rows)
    df_base[COL_TIMESTAMP] = pd.to_datetime(df_base[COL_TIMESTAMP])

    # Neighbor D1: Add/Remove user_special (uses all 5 services and 8 distinct tasks across the week)
    special_rows = []
    for d in range(1, 8):
        for s_idx in range(m):
            special_rows.append({
                "user_id": "user_special",
                "provider": services[s_idx],
                "task": tasks[(d + s_idx) % len(tasks)],
                "timestamp": f"2025-01-{d:02d} 14:00:00",
                "success": True,
            })
    df_special = pd.DataFrame(special_rows)
    df_special[COL_TIMESTAMP] = pd.to_datetime(df_special[COL_TIMESTAMP])

    df_D = pd.concat([df_base, df_special], ignore_index=True)
    df_D_prime_add_remove = df_base.copy()

    s_D, t_D = compute_distinct_window_histogram(df_D, services, tasks, k_tasks_per_user=k)
    s_D1, t_D1 = compute_distinct_window_histogram(df_D_prime_add_remove, services, tasks, k_tasks_per_user=k)

    diff_s_add = s_D - s_D1
    diff_t_add = t_D - t_D1
    l2_sq_add_remove = float(np.sum(diff_s_add**2) + np.sum(diff_t_add**2))
    # Under Add/Remove, user_special contributes at most m service flags and at most k task flags
    assert l2_sq_add_remove <= m + k, f"Add/remove sensitivity violated: {l2_sq_add_remove} > {m + k}"

    # Neighbor D2: Replace user_special with user_replacement who uses different tasks and services
    replacement_rows = []
    for d in range(1, 8):
        replacement_rows.append({
            "user_id": "user_replacement",
            "provider": services[0],  # only provider 0
            "task": tasks[11],        # only task 11
            "timestamp": f"2025-01-{d:02d} 16:00:00",
            "success": True,
        })
    df_replacement = pd.DataFrame(replacement_rows)
    df_replacement[COL_TIMESTAMP] = pd.to_datetime(df_replacement[COL_TIMESTAMP])

    df_D_prime_replace = pd.concat([df_base, df_replacement], ignore_index=True)
    s_D2, t_D2 = compute_distinct_window_histogram(df_D_prime_replace, services, tasks, k_tasks_per_user=k)

    diff_s_rep = s_D - s_D2
    diff_t_rep = t_D - t_D2
    l2_sq_replace = float(np.sum(diff_s_rep**2) + np.sum(diff_t_rep**2))
    # Under Replace, sensitivity is bounded by m + 2k
    assert l2_sq_replace <= m + 2 * k, f"Replace sensitivity violated: {l2_sq_replace} > {m + 2 * k}"
    assert m + 2 * k == 11
