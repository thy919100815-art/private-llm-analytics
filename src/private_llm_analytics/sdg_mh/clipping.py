"""Contribution Bounding and Truncation for SDG-MH.

Supports:
1. Per-user per-day truncation (default SDG-MH):
   x[u, t] = min(count[u, t], C)
2. Per-user per-window L2 truncation (SDG-MH-WinClip ablation):
   Matches existing system / sliding window practice.
3. Distinct-user window participation:
   Binary indicators for m services + top-k tasks (tie-break by task name).
   Sensitivity Delta^2 = m + 2k.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple
import numpy as np
import pandas as pd

from private_llm_analytics.ingestion.schema import (
    COL_PROVIDER,
    COL_SUCCESS,
    COL_TASK,
    COL_TIMESTAMP,
    COL_USER_ID,
)


@dataclass(frozen=True)
class ClippingDiagnostics:
    """Statistics on user contribution clipping."""

    total_active_users: int
    clipped_users_count: int
    clipping_rate: float        # Fraction of active user-days clipped
    raw_sum: float
    clipped_sum: float
    clipping_bias: float        # raw_sum - clipped_sum


def clip_daily_additive(
    df_day: pd.DataFrame,
    C_r: float = 50.0,
    C_f: float = 10.0,
    providers_list: Optional[Sequence[str]] = None,
    C_p: Optional[float] = None,
) -> Tuple[float, float, Optional[np.ndarray], Dict[str, ClippingDiagnostics]]:
    """Apply data-independent per-user daily clipping to single-day interactions.

    Args:
        df_day: DataFrame containing interactions for a single day.
        C_r: Per-user daily request truncation bound.
        C_f: Per-user daily failure truncation bound.
        providers_list: Optional ordered list of provider identifiers.
        C_p: Optional L2 norm bound on user provider vector.

    Returns:
        (sum_req, sum_fail, sum_prov_vector, diagnostics_dict)
    """
    if df_day.empty:
        empty_diag_r = ClippingDiagnostics(0, 0, 0.0, 0.0, 0.0, 0.0)
        empty_diag_f = ClippingDiagnostics(0, 0, 0.0, 0.0, 0.0, 0.0)
        prov_vec = np.zeros(len(providers_list)) if providers_list else None
        return 0.0, 0.0, prov_vec, {"requests": empty_diag_r, "failures": empty_diag_f}

    users = df_day[COL_USER_ID]
    req_counts = users.value_counts()
    n_users = len(req_counts)

    # 1. Requests clipping
    clipped_req = req_counts.clip(upper=C_r)
    n_clipped_r = int((req_counts > C_r).sum())
    raw_req_sum = float(req_counts.sum())
    clipped_req_sum = float(clipped_req.sum())
    diag_r = ClippingDiagnostics(
        total_active_users=n_users,
        clipped_users_count=n_clipped_r,
        clipping_rate=float(n_clipped_r / n_users) if n_users > 0 else 0.0,
        raw_sum=raw_req_sum,
        clipped_sum=clipped_req_sum,
        clipping_bias=raw_req_sum - clipped_req_sum,
    )

    # 2. Failures clipping
    if COL_SUCCESS in df_day.columns:
        fail_df = df_day[df_day[COL_SUCCESS] == False]  # noqa: E712
        fail_counts = fail_df[COL_USER_ID].value_counts().reindex(req_counts.index, fill_value=0)
    else:
        fail_counts = pd.Series(0, index=req_counts.index)

    clipped_fail = fail_counts.clip(upper=C_f)
    n_clipped_f = int((fail_counts > C_f).sum())
    raw_fail_sum = float(fail_counts.sum())
    clipped_fail_sum = float(clipped_fail.sum())
    diag_f = ClippingDiagnostics(
        total_active_users=n_users,
        clipped_users_count=n_clipped_f,
        clipping_rate=float(n_clipped_f / n_users) if n_users > 0 else 0.0,
        raw_sum=raw_fail_sum,
        clipped_sum=clipped_fail_sum,
        clipping_bias=raw_fail_sum - clipped_fail_sum,
    )

    # 3. Provider vectors (optional)
    prov_sum = None
    if providers_list is not None:
        p_crosstab = pd.crosstab(df_day[COL_USER_ID], df_day[COL_PROVIDER]).reindex(
            index=req_counts.index, columns=providers_list, fill_value=0
        )
        p_mat = p_crosstab.to_numpy(dtype=np.float64)
        if C_p is not None and C_p > 0:
            norms = np.linalg.norm(p_mat, ord=2, axis=1, keepdims=True)
            scale = np.minimum(1.0, C_p / np.maximum(norms, 1e-12))
            p_mat = p_mat * scale
        prov_sum = np.sum(p_mat, axis=0)

    return clipped_req_sum, clipped_fail_sum, prov_sum, {"requests": diag_r, "failures": diag_f}


def clip_window_l2_additive(
    df_window: pd.DataFrame,
    C_r: float = 50.0,
    C_f: float = 10.0,
) -> Tuple[float, float, Dict[str, ClippingDiagnostics]]:
    """Ablation comparator: Per-user per-window L2 clipping across entire query window."""
    if df_window.empty:
        empty_diag = ClippingDiagnostics(0, 0, 0.0, 0.0, 0.0, 0.0)
        return 0.0, 0.0, {"requests": empty_diag, "failures": empty_diag}

    users = df_window[COL_USER_ID]
    req_counts = users.value_counts().astype(float)
    n_users = len(req_counts)

    # Window-level L2 clipping for scalar is just min(val, C_r)
    clipped_req = req_counts.clip(upper=C_r)
    n_clipped_r = int((req_counts > C_r).sum())
    raw_req_sum = float(req_counts.sum())
    clipped_req_sum = float(clipped_req.sum())
    diag_r = ClippingDiagnostics(
        total_active_users=n_users,
        clipped_users_count=n_clipped_r,
        clipping_rate=float(n_clipped_r / n_users) if n_users > 0 else 0.0,
        raw_sum=raw_req_sum,
        clipped_sum=clipped_req_sum,
        clipping_bias=raw_req_sum - clipped_req_sum,
    )

    if COL_SUCCESS in df_window.columns:
        fail_df = df_window[df_window[COL_SUCCESS] == False]  # noqa: E712
        fail_counts = fail_df[COL_USER_ID].value_counts().reindex(req_counts.index, fill_value=0).astype(float)
    else:
        fail_counts = pd.Series(0.0, index=req_counts.index)

    clipped_fail = fail_counts.clip(upper=C_f)
    n_clipped_f = int((fail_counts > C_f).sum())
    raw_fail_sum = float(fail_counts.sum())
    clipped_fail_sum = float(clipped_fail.sum())
    diag_f = ClippingDiagnostics(
        total_active_users=n_users,
        clipped_users_count=n_clipped_f,
        clipping_rate=float(n_clipped_f / n_users) if n_users > 0 else 0.0,
        raw_sum=raw_fail_sum,
        clipped_sum=clipped_fail_sum,
        clipping_bias=raw_fail_sum - clipped_fail_sum,
    )

    return clipped_req_sum, clipped_fail_sum, {"requests": diag_r, "failures": diag_f}


def compute_distinct_window_histogram(
    df_window: pd.DataFrame,
    services_list: Sequence[str],
    tasks_list: Sequence[str],
    k_tasks_per_user: int = 3,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute bounded distinct user histograms for services and tasks within a window.

    - Services: Indicator vector of whether user used service b (m services).
    - Tasks: Top-k tasks per user (by interaction count; tie-broken by task lexicographical index).
    - Sensitivity: Delta^2 = m + 2k.

    Returns:
        (service_counts, task_counts)
    """
    m = len(services_list)
    t_len = len(tasks_list)
    if df_window.empty:
        return np.zeros(m, dtype=np.float64), np.zeros(t_len, dtype=np.float64)

    service_to_idx = {s: i for i, s in enumerate(services_list)}
    task_to_idx = {t: i for i, t in enumerate(tasks_list)}

    # Group by user
    service_user_indicators: Dict[int, Set[str]] = {i: set() for i in range(m)}
    task_user_indicators: Dict[int, Set[str]] = {i: set() for i in range(t_len)}

    for user_id, group in df_window.groupby(COL_USER_ID):
        # 1. Services used by user
        user_provs = set(group[COL_PROVIDER].unique())
        for p in user_provs:
            if p in service_to_idx:
                service_user_indicators[service_to_idx[p]].add(user_id)

        # 2. Tasks used by user (take top k_tasks_per_user)
        task_counts = group[COL_TASK].value_counts()
        # Sort by count desc, then task index asc for deterministic tie-breaking
        sorted_tasks = sorted(
            task_counts.items(),
            key=lambda item: (-item[1], task_to_idx.get(item[0], 999999)),
        )
        selected_tasks = [t for t, _ in sorted_tasks[:k_tasks_per_user]]
        for t in selected_tasks:
            if t in task_to_idx:
                task_user_indicators[task_to_idx[t]].add(user_id)

    service_counts = np.array([len(service_user_indicators[i]) for i in range(m)], dtype=np.float64)
    task_counts = np.array([len(task_user_indicators[i]) for i in range(t_len)], dtype=np.float64)
    return service_counts, task_counts
