"""Granularity transformations, direct contribution builders, and reconstruction (v0.3).

Provides:
- Direct granularity user-level contribution builders (target-space L2 clipping & binary prevalence)
- Deterministic aggregation and reconstruction mappings
- Workload-weighted reconstruction error calculation
"""

import math
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

from private_llm_analytics.aggregation.grouping import DEFAULT_PROVIDERS, DEFAULT_TASKS


def _extract_matrix_and_names(
    raw_user_matrix: Any,
) -> Tuple[np.ndarray, List[str]]:
    """Helper to extract (n_users, 9) numpy matrix and cell_names."""
    if hasattr(raw_user_matrix, "matrix") and hasattr(raw_user_matrix, "cell_names"):
        return np.asarray(raw_user_matrix.matrix, dtype=float), list(raw_user_matrix.cell_names)
    mat = np.asarray(raw_user_matrix, dtype=float)
    if mat.ndim == 1:
        mat = mat.reshape(1, -1)
    if mat.shape[1] == 9:
        default_names = [f"{p}:{t}" for p in DEFAULT_PROVIDERS for t in DEFAULT_TASKS]
        return mat, default_names
    raise ValueError(f"Expected 9-cell user matrix, got shape {mat.shape}")


def get_direct_granularity_sensitivity(
    metric: str,
    granularity: str,
    clip_norm: Optional[float] = None,
) -> float:
    """Return theoretical L2 sensitivity under direct granularity user-level release."""
    metric_norm = metric.lower().strip()
    if "volume" in metric_norm:
        if clip_norm is None or clip_norm <= 0:
            raise ValueError(f"Volume metric requires clip_norm > 0, got {clip_norm}")
        return float(clip_norm)
    elif "prevalence" in metric_norm:
        if granularity == "full_9_cell":
            return 3.0  # sqrt(9)
        elif granularity in ("provider_3_cell", "task_3_cell"):
            return math.sqrt(3.0)  # sqrt(3)
        elif granularity == "overall_1_cell":
            return 1.0  # sqrt(1)
        else:
            raise ValueError(f"Unsupported granularity '{granularity}' for prevalence")
    elif "share" in metric_norm:
        return 1.0  # L2 norm of user-normalized simplex vector is <= 1.0
    else:
        return float(clip_norm or 1.0)


def get_direct_granularity_l1_sensitivity(
    metric: str,
    granularity: str,
    clip_norm: Optional[float] = None,
) -> float:
    """Return theoretical L1 sensitivity under direct granularity user-level release."""
    metric_norm = metric.lower().strip()
    if "volume" in metric_norm:
        if clip_norm is None or clip_norm <= 0:
            raise ValueError(f"Volume metric requires clip_norm > 0, got {clip_norm}")
        c = float(clip_norm)
        if granularity == "full_9_cell":
            return 3.0 * c  # sqrt(9) * C
        elif granularity in ("provider_3_cell", "task_3_cell"):
            return math.sqrt(3.0) * c  # sqrt(3) * C
        elif granularity == "overall_1_cell":
            return 1.0 * c  # sqrt(1) * C
        else:
            raise ValueError(f"Unsupported granularity '{granularity}' for volume L1 sensitivity")
    elif "prevalence" in metric_norm:
        if granularity == "full_9_cell":
            return 9.0
        elif granularity in ("provider_3_cell", "task_3_cell"):
            return 3.0
        elif granularity == "overall_1_cell":
            return 1.0
        else:
            raise ValueError(f"Unsupported granularity '{granularity}' for prevalence L1 sensitivity")
    elif "share" in metric_norm:
        return 1.0
    else:
        return float(clip_norm or 1.0)


def build_volume_contributions_for_granularity(
    raw_user_matrix: Any,
    granularity: str,
    clip_norm: float,
) -> Tuple[np.ndarray, np.ndarray, List[str], float]:
    """Build direct user-level volume contributions clipped in target G-dimensional space.

    Process:
      1. raw 9-cell per-user vector m_u
      2. deterministic per-user linear aggregation to target space: m_u^G = A_G m_u
      3. L2 clip in target G-dimensional space to C: bar_m_u^G = m_u^G * min(1, C / ||m_u^G||_2)
      4. sum: V_G = sum_u bar_m_u^G
      5. sensitivity: Delta_2 = C

    Returns:
      (clipped_user_matrix, exact_aggregate, target_names, l2_sensitivity)
    """
    if clip_norm <= 0:
        raise ValueError(f"clip_norm must be > 0, got {clip_norm}")

    mat_9, cell_names = _extract_matrix_and_names(raw_user_matrix)
    n_users = mat_9.shape[0]

    coords = []
    for name in cell_names:
        if ":" in name:
            parts = name.split(":", 1)
            coords.append((parts[0].strip(), parts[1].strip()))
        else:
            coords.append((name.strip(), "unknown"))

    if granularity == "full_9_cell":
        target_names = list(cell_names)
        target_mat = mat_9.copy()

    elif granularity == "provider_3_cell":
        target_names = sorted(list({p for p, _ in coords}))
        target_mat = np.zeros((n_users, len(target_names)), dtype=float)
        for i, p in enumerate(target_names):
            indices = [idx for idx, (p_c, _) in enumerate(coords) if p_c == p]
            target_mat[:, i] = np.sum(mat_9[:, indices], axis=1)

    elif granularity == "task_3_cell":
        target_names = sorted(list({t for _, t in coords}))
        target_mat = np.zeros((n_users, len(target_names)), dtype=float)
        for i, t in enumerate(target_names):
            indices = [idx for idx, (_, t_c) in enumerate(coords) if t_c == t]
            target_mat[:, i] = np.sum(mat_9[:, indices], axis=1)

    elif granularity == "overall_1_cell":
        target_names = ["overall_total"]
        target_mat = np.sum(mat_9, axis=1, keepdims=True)

    else:
        raise ValueError(f"Unsupported granularity '{granularity}'")

    # L2 clipping in target G-dimensional space
    norms = np.linalg.norm(target_mat, axis=1, keepdims=True)
    scale = np.minimum(1.0, float(clip_norm) / np.maximum(norms, 1e-15))
    clipped_user_matrix = target_mat * scale
    exact_aggregate = np.sum(clipped_user_matrix, axis=0)

    return clipped_user_matrix, exact_aggregate, target_names, float(clip_norm)


def build_prevalence_contributions_for_granularity(
    raw_user_matrix: Any,
    granularity: str,
) -> Tuple[np.ndarray, np.ndarray, List[str], float]:
    """Build direct user-level binary prevalence contributions for target granularity.

    Semantics:
      - full_9_cell: b_{u,p,t} = 1[user used provider p and task t at least once] (Delta_2 = 3)
      - provider_3_cell: b_{u,p} = 1[user used provider p at least once] (Delta_2 = sqrt(3))
      - task_3_cell: b_{u,t} = 1[user used task t at least once] (Delta_2 = sqrt(3))
      - overall_1_cell: b_u = 1[user active in window] (Delta_2 = 1)

    Returns:
      (binary_user_matrix, exact_counts, target_names, l2_sensitivity)
    """
    mat_9, cell_names = _extract_matrix_and_names(raw_user_matrix)
    n_users = mat_9.shape[0]

    coords = []
    for name in cell_names:
        if ":" in name:
            parts = name.split(":", 1)
            coords.append((parts[0].strip(), parts[1].strip()))
        else:
            coords.append((name.strip(), "unknown"))

    if granularity == "full_9_cell":
        target_names = list(cell_names)
        binary_mat = (mat_9 > 0).astype(float)
        sens = 3.0

    elif granularity == "provider_3_cell":
        target_names = sorted(list({p for p, _ in coords}))
        binary_mat = np.zeros((n_users, len(target_names)), dtype=float)
        for i, p in enumerate(target_names):
            indices = [idx for idx, (p_c, _) in enumerate(coords) if p_c == p]
            # User used provider p at least once
            binary_mat[:, i] = (np.sum(mat_9[:, indices], axis=1) > 0).astype(float)
        sens = math.sqrt(3.0)

    elif granularity == "task_3_cell":
        target_names = sorted(list({t for _, t in coords}))
        binary_mat = np.zeros((n_users, len(target_names)), dtype=float)
        for i, t in enumerate(target_names):
            indices = [idx for idx, (_, t_c) in enumerate(coords) if t_c == t]
            # User used task t at least once
            binary_mat[:, i] = (np.sum(mat_9[:, indices], axis=1) > 0).astype(float)
        sens = math.sqrt(3.0)

    elif granularity == "overall_1_cell":
        target_names = ["overall_total"]
        binary_mat = (np.sum(mat_9, axis=1, keepdims=True) > 0).astype(float)
        sens = 1.0

    else:
        raise ValueError(f"Unsupported granularity '{granularity}'")

    exact_counts = np.sum(binary_mat, axis=0)
    return binary_mat, exact_counts, target_names, float(sens)


def build_share_contributions_for_granularity(
    raw_user_matrix: Any,
    granularity: str,
) -> Tuple[np.ndarray, np.ndarray, List[str], float]:
    """Build user-normalized simplex contributions directly in target granularity space.

    Returns:
      (share_user_matrix, exact_share_sum, target_names, l2_sensitivity)
    """
    mat_9, cell_names = _extract_matrix_and_names(raw_user_matrix)
    n_users = mat_9.shape[0]

    coords = []
    for name in cell_names:
        if ":" in name:
            parts = name.split(":", 1)
            coords.append((parts[0].strip(), parts[1].strip()))
        else:
            coords.append((name.strip(), "unknown"))

    if granularity == "full_9_cell":
        target_names = list(cell_names)
        target_mat = mat_9.copy()

    elif granularity == "provider_3_cell":
        target_names = sorted(list({p for p, _ in coords}))
        target_mat = np.zeros((n_users, len(target_names)), dtype=float)
        for i, p in enumerate(target_names):
            indices = [idx for idx, (p_c, _) in enumerate(coords) if p_c == p]
            target_mat[:, i] = np.sum(mat_9[:, indices], axis=1)

    elif granularity == "task_3_cell":
        target_names = sorted(list({t for _, t in coords}))
        target_mat = np.zeros((n_users, len(target_names)), dtype=float)
        for i, t in enumerate(target_names):
            indices = [idx for idx, (_, t_c) in enumerate(coords) if t_c == t]
            target_mat[:, i] = np.sum(mat_9[:, indices], axis=1)

    elif granularity == "overall_1_cell":
        target_names = ["overall_total"]
        target_mat = np.sum(mat_9, axis=1, keepdims=True)

    else:
        raise ValueError(f"Unsupported granularity '{granularity}'")

    # Normalize per user to simplex
    user_totals = np.sum(target_mat, axis=1, keepdims=True)
    mask = (user_totals > 0)
    share_mat = np.zeros_like(target_mat)
    np.divide(target_mat, user_totals, out=share_mat, where=mask)

    exact_sum = np.sum(share_mat, axis=0)
    return share_mat, exact_sum, target_names, 1.0


def build_aggregation_matrix(
    target_granularity: str,
    source_9_cell_names: Optional[List[str]] = None,
) -> Tuple[np.ndarray, List[str]]:
    """Build linear aggregation operator A_G mapping 9-cell vector to target granularity.

    Args:
        target_granularity: One of 'full_9_cell', 'provider_3_cell', 'task_3_cell', 'overall_1_cell'.
        source_9_cell_names: List of 9 cell names (defaults to DEFAULT_PROVIDERS x DEFAULT_TASKS).

    Returns:
        (A_G, target_names):
            A_G: numpy array of shape (dim_target, len(source_9_cell_names)).
            target_names: list of target cell names.
    """
    if source_9_cell_names is None:
        source_9_cell_names, _ = build_cell_definitions(DEFAULT_PROVIDERS, DEFAULT_TASKS)

    n_cells = len(source_9_cell_names)
    coords: List[Tuple[str, str]] = []
    for name in source_9_cell_names:
        if ":" in name:
            parts = name.split(":", 1)
            coords.append((parts[0].strip(), parts[1].strip()))
        else:
            coords.append((name.strip(), "unknown"))

    if target_granularity == "full_9_cell":
        return np.eye(n_cells, dtype=np.float64), list(source_9_cell_names)

    elif target_granularity == "provider_3_cell":
        providers = sorted(list({p for p, _ in coords}))
        A = np.zeros((len(providers), n_cells), dtype=np.float64)
        for i, p in enumerate(providers):
            indices = [idx for idx, (p_c, _) in enumerate(coords) if p_c == p]
            A[i, indices] = 1.0
        return A, providers

    elif target_granularity == "task_3_cell":
        tasks = sorted(list({t for _, t in coords}))
        A = np.zeros((len(tasks), n_cells), dtype=np.float64)
        for i, t in enumerate(tasks):
            indices = [idx for idx, (_, t_c) in enumerate(coords) if t_c == t]
            A[i, indices] = 1.0
        return A, tasks

    elif target_granularity == "overall_1_cell":
        A = np.ones((1, n_cells), dtype=np.float64)
        return A, ["overall_total"]

    else:
        raise ValueError(f"Unsupported granularity '{target_granularity}'")


def build_reconstruction_matrix(
    source_granularity: str,
    target_9_cell_names: Optional[List[str]] = None,
    provider_shares: Optional[Dict[str, float]] = None,
    task_shares: Optional[Dict[str, float]] = None,
    marginal_names: Optional[List[str]] = None,
) -> np.ndarray:
    """Build single source of truth reconstruction matrix R_G mapping source granularity to 9-cell space.

    Args:
        source_granularity: One of 'full_9_cell', 'provider_3_cell', 'task_3_cell', 'overall_1_cell'.
        target_9_cell_names: List of target 9 cell names (defaults to DEFAULT_PROVIDERS x DEFAULT_TASKS).
        provider_shares: Public prior provider distribution shares (used for task_3_cell and overall_1_cell).
        task_shares: Public prior task distribution shares (used for provider_3_cell and overall_1_cell).
        marginal_names: Explicit source column names (defaults to sorted aggregate order).

    Returns:
        R_G: numpy array of shape (len(target_9_cell_names), source_dim).
    """
    if target_9_cell_names is None:
        target_9_cell_names, _ = build_cell_definitions(DEFAULT_PROVIDERS, DEFAULT_TASKS)

    n_cells = len(target_9_cell_names)
    coords: List[Tuple[str, str]] = []
    for name in target_9_cell_names:
        if ":" in name:
            parts = name.split(":", 1)
            coords.append((parts[0].strip(), parts[1].strip()))
        else:
            coords.append((name.strip(), "unknown"))

    if source_granularity == "full_9_cell":
        return np.eye(n_cells, dtype=np.float64)

    elif source_granularity == "provider_3_cell":
        providers = list(marginal_names) if marginal_names is not None else sorted(list({p for p, _ in coords}))
        R_G = np.zeros((n_cells, len(providers)), dtype=np.float64)
        for i, (p, t) in enumerate(coords):
            if p not in providers:
                continue
            j = providers.index(p)
            tasks_under_p = [t_c for p_c, t_c in coords if p_c == p]
            if task_shares is not None:
                w = float(task_shares.get(t, 1.0))
                tot_w = sum(float(task_shares.get(t_c, 1.0)) for t_c in tasks_under_p)
                norm_w = (w / tot_w) if tot_w > 0 else (1.0 / max(1, len(tasks_under_p)))
            else:
                norm_w = 1.0 / max(1, len(tasks_under_p))
            R_G[i, j] = norm_w
        return R_G

    elif source_granularity == "task_3_cell":
        tasks = list(marginal_names) if marginal_names is not None else sorted(list({t for _, t in coords}))
        R_G = np.zeros((n_cells, len(tasks)), dtype=np.float64)
        for i, (p, t) in enumerate(coords):
            if t not in tasks:
                continue
            j = tasks.index(t)
            provs_under_t = [p_c for p_c, t_c in coords if t_c == t]
            if provider_shares is not None:
                w = float(provider_shares.get(p, 1.0))
                tot_w = sum(float(provider_shares.get(p_c, 1.0)) for p_c in provs_under_t)
                norm_w = (w / tot_w) if tot_w > 0 else (1.0 / max(1, len(provs_under_t)))
            else:
                norm_w = 1.0 / max(1, len(provs_under_t))
            R_G[i, j] = norm_w
        return R_G

    elif source_granularity == "overall_1_cell":
        R_G = np.zeros((n_cells, 1), dtype=np.float64)
        raw_w = np.zeros(n_cells, dtype=np.float64)
        for i, (p, t) in enumerate(coords):
            pw = float(provider_shares.get(p, 1.0)) if provider_shares is not None else 1.0
            tw = float(task_shares.get(t, 1.0)) if task_shares is not None else 1.0
            raw_w[i] = pw * tw
        tot_w = float(np.sum(raw_w))
        if tot_w > 0:
            R_G[:, 0] = raw_w / tot_w
        else:
            R_G[:, 0] = 1.0 / max(1, n_cells)
        return R_G

    else:
        raise ValueError(f"Unsupported source_granularity '{source_granularity}'")


def aggregate_to_granularity(
    vector_9_cell: np.ndarray,
    cell_names: List[str],
    target_granularity: str,
) -> Tuple[np.ndarray, List[str]]:
    """Deterministically aggregate a 9-cell DP or ground-truth vector to target granularity."""
    vec = np.asarray(vector_9_cell, dtype=float)
    if vec.ndim == 1 and len(vec) != len(cell_names):
        raise ValueError(f"Vector length ({len(vec)}) does not match cell_names length ({len(cell_names)})")

    A, names = build_aggregation_matrix(target_granularity, cell_names)
    out = A @ vec
    return out, names


def reconstruct_to_9_cell(
    marginal_vector: np.ndarray,
    source_granularity: str,
    marginal_names: Optional[List[str]] = None,
    target_9_cell_names: Optional[List[str]] = None,
    provider_shares: Optional[Dict[str, float]] = None,
    task_shares: Optional[Dict[str, float]] = None,
) -> np.ndarray:
    """Deterministically reconstruct a 9-cell representation from lower granularity."""
    vec = np.asarray(marginal_vector, dtype=float)
    if source_granularity == "full_9_cell":
        return vec.copy()

    R_G = build_reconstruction_matrix(
        source_granularity=source_granularity,
        target_9_cell_names=target_9_cell_names,
        provider_shares=provider_shares,
        task_shares=task_shares,
        marginal_names=marginal_names,
    )
    return R_G @ vec


def compute_workload_reconstruction_error(
    true_9_cell: np.ndarray,
    cell_names: List[str],
    granularity: str,
    workload_weights: Optional[Dict[str, float]] = None,
    provider_shares: Optional[Dict[str, float]] = None,
    task_shares: Optional[Dict[str, float]] = None,
) -> float:
    """Compute normalized workload aggregation error based on public workload weights."""
    if granularity == "full_9_cell":
        return 0.0

    norm_weights = {
        "full_9_cell": 0.4,
        "provider": 0.3,
        "task": 0.2,
        "overall": 0.1,
    }
    if workload_weights is not None:
        norm_weights.update(workload_weights)

    total_w = sum(norm_weights.values())
    if total_w <= 0:
        total_w = 1.0

    # 1. Reconstructed 9-cell via shared R_G
    marginal_vec, marginal_names = aggregate_to_granularity(true_9_cell, cell_names, granularity)
    reconstructed_9 = reconstruct_to_9_cell(
        marginal_vector=marginal_vec,
        source_granularity=granularity,
        marginal_names=marginal_names,
        target_9_cell_names=cell_names,
        provider_shares=provider_shares,
        task_shares=task_shares,
    )

    total_denom = max(1.0, float(np.sum(np.abs(true_9_cell))))

    # Error on 9-cell query
    err_9 = float(np.sum(np.abs(true_9_cell - reconstructed_9)) / total_denom)

    # Error on Provider marginal query
    true_prov, prov_names = aggregate_to_granularity(true_9_cell, cell_names, "provider_3_cell")
    rec_prov, _ = aggregate_to_granularity(reconstructed_9, cell_names, "provider_3_cell")
    denom_prov = max(1.0, float(np.sum(np.abs(true_prov))))
    err_prov = float(np.sum(np.abs(true_prov - rec_prov)) / denom_prov)

    # Error on Task marginal query
    true_task, task_names = aggregate_to_granularity(true_9_cell, cell_names, "task_3_cell")
    rec_task, _ = aggregate_to_granularity(reconstructed_9, cell_names, "task_3_cell")
    denom_task = max(1.0, float(np.sum(np.abs(true_task))))
    err_task = float(np.sum(np.abs(true_task - rec_task)) / denom_task)

    # Error on Overall scalar total
    true_over, _ = aggregate_to_granularity(true_9_cell, cell_names, "overall_1_cell")
    rec_over, _ = aggregate_to_granularity(reconstructed_9, cell_names, "overall_1_cell")
    denom_over = max(1.0, float(np.abs(true_over[0])))
    err_over = float(abs(true_over[0] - rec_over[0]) / denom_over)

    weighted_loss = (
        norm_weights["full_9_cell"] * err_9
        + norm_weights["provider"] * err_prov
        + norm_weights["task"] * err_task
        + norm_weights["overall"] * err_over
    ) / total_w

    return float(weighted_loss)
