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


def aggregate_to_granularity(
    vector_9_cell: np.ndarray,
    cell_names: List[str],
    target_granularity: str,
) -> Tuple[np.ndarray, List[str]]:
    """Deterministically aggregate a 9-cell DP or ground-truth vector to target granularity."""
    vec = np.asarray(vector_9_cell, dtype=float)
    if vec.ndim == 1 and len(vec) != len(cell_names):
        raise ValueError(f"Vector length ({len(vec)}) does not match cell_names length ({len(cell_names)})")

    if target_granularity == "full_9_cell":
        return vec.copy(), list(cell_names)

    coords = []
    for name in cell_names:
        if ":" in name:
            parts = name.split(":", 1)
            coords.append((parts[0].strip(), parts[1].strip()))
        else:
            coords.append((name.strip(), "unknown"))

    if target_granularity == "provider_3_cell":
        providers = sorted(list({p for p, _ in coords}))
        out = np.zeros(len(providers), dtype=float)
        for i, p in enumerate(providers):
            indices = [idx for idx, (p_c, _) in enumerate(coords) if p_c == p]
            out[i] = vec[indices].sum()
        return out, providers

    elif target_granularity == "task_3_cell":
        tasks = sorted(list({t for _, t in coords}))
        out = np.zeros(len(tasks), dtype=float)
        for i, t in enumerate(tasks):
            indices = [idx for idx, (_, t_c) in enumerate(coords) if t_c == t]
            out[i] = vec[indices].sum()
        return out, tasks

    elif target_granularity == "overall_1_cell":
        out = np.array([float(vec.sum())], dtype=float)
        return out, ["overall_total"]

    else:
        raise ValueError(f"Unsupported granularity '{target_granularity}'")


def reconstruct_to_9_cell(
    marginal_vector: np.ndarray,
    source_granularity: str,
    marginal_names: List[str],
    target_9_cell_names: List[str],
) -> np.ndarray:
    """Deterministically reconstruct a 9-cell representation from lower granularity under uniform prior."""
    vec = np.asarray(marginal_vector, dtype=float)
    target_coords = []
    for name in target_9_cell_names:
        if ":" in name:
            parts = name.split(":", 1)
            target_coords.append((parts[0].strip(), parts[1].strip()))
        else:
            target_coords.append((name.strip(), "unknown"))

    reconstructed = np.zeros(len(target_9_cell_names), dtype=float)

    if source_granularity == "full_9_cell":
        return vec.copy()

    elif source_granularity == "provider_3_cell":
        prov_map = {name: vec[i] for i, name in enumerate(marginal_names)}
        for idx, (p, _) in enumerate(target_coords):
            tasks_under_p = sum(1 for p_c, _ in target_coords if p_c == p)
            denom = max(1, tasks_under_p)
            reconstructed[idx] = prov_map.get(p, 0.0) / denom
        return reconstructed

    elif source_granularity == "task_3_cell":
        task_map = {name: vec[i] for i, name in enumerate(marginal_names)}
        for idx, (_, t) in enumerate(target_coords):
            provs_under_t = sum(1 for _, t_c in target_coords if t_c == t)
            denom = max(1, provs_under_t)
            reconstructed[idx] = task_map.get(t, 0.0) / denom
        return reconstructed

    elif source_granularity == "overall_1_cell":
        total_val = float(vec[0])
        n_cells = max(1, len(target_9_cell_names))
        reconstructed[:] = total_val / n_cells
        return reconstructed

    else:
        raise ValueError(f"Unsupported source_granularity '{source_granularity}'")


def compute_workload_reconstruction_error(
    true_9_cell: np.ndarray,
    cell_names: List[str],
    granularity: str,
    workload_weights: Optional[Dict[str, float]] = None,
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

    # 1. Reconstructed 9-cell
    marginal_vec, marginal_names = aggregate_to_granularity(true_9_cell, cell_names, granularity)
    reconstructed_9 = reconstruct_to_9_cell(marginal_vec, granularity, marginal_names, cell_names)

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
