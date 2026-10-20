"""Granularity transformations, deterministic aggregation, and reconstruction (v0.3)."""

from typing import Dict, List, Optional, Tuple
import numpy as np

from private_llm_analytics.aggregation.grouping import DEFAULT_PROVIDERS, DEFAULT_TASKS


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

    # Parse (provider, task) coordinates for each cell
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
            # Equal division among tasks under provider
            tasks_under_p = sum(1 for p_c, _ in target_coords if p_c == p)
            denom = max(1, tasks_under_p)
            reconstructed[idx] = prov_map.get(p, 0.0) / denom
        return reconstructed

    elif source_granularity == "task_3_cell":
        task_map = {name: vec[i] for i, name in enumerate(marginal_names)}
        for idx, (_, t) in enumerate(target_coords):
            # Equal division among providers under task
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
    """Compute empirical L1 reconstruction error of granularity against ground truth."""
    if granularity == "full_9_cell":
        return 0.0

    marginal_vec, marginal_names = aggregate_to_granularity(true_9_cell, cell_names, granularity)
    reconstructed_9 = reconstruct_to_9_cell(marginal_vec, granularity, marginal_names, cell_names)

    # Normalized L1 error against truth
    abs_err = np.sum(np.abs(true_9_cell - reconstructed_9))
    norm_denom = max(1.0, float(np.sum(np.abs(true_9_cell))))
    return float(abs_err / norm_denom)
