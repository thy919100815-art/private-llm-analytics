"""Interaction generation routines."""

from datetime import datetime, timedelta
from typing import List, Tuple
import numpy as np
import pandas as pd

from private_llm_analytics.ingestion.schema import (
    COL_INTERACTION_ID,
    COL_PROVIDER,
    COL_TASK,
    COL_TIMESTAMP,
    COL_USER_ID,
)


def generate_interaction_dataframe(
    user_ids: List[str],
    user_counts: np.ndarray, # shape (n_users, 9)
    providers: List[str],
    tasks: List[str],
    start_date: datetime,
    n_days: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Expand user cell counts into a long-format interaction DataFrame.

    Optimized for high performance on large populations (up to N=100,000).
    """
    n_users, n_cells = user_counts.shape
    assert n_cells == 9

    # Mapping from cell index to (provider, task)
    cell_provider_task: List[Tuple[str, str]] = []
    for p in providers:
        for t in tasks:
            cell_provider_task.append((p, t))

    # Find non-zero entries (user_idx, cell_idx)
    user_idx_nonzero, cell_idx_nonzero = np.nonzero(user_counts)
    repeats = user_counts[user_idx_nonzero, cell_idx_nonzero]
    total_interactions = int(repeats.sum())

    # Repeat user_idx and cell_idx according to the count in each cell
    expanded_user_idx = np.repeat(user_idx_nonzero, repeats)
    expanded_cell_idx = np.repeat(cell_idx_nonzero, repeats)

    # Convert to user_ids, providers, tasks
    user_id_array = np.array(user_ids, dtype=object)
    row_user_ids = user_id_array[expanded_user_idx]

    cell_providers = np.array([cp[0] for cp in cell_provider_task], dtype=object)
    cell_tasks = np.array([cp[1] for cp in cell_provider_task], dtype=object)

    row_providers = cell_providers[expanded_cell_idx]
    row_tasks = cell_tasks[expanded_cell_idx]

    # Generate timestamps across the window:
    # Shuffle or randomize within window
    window_seconds = int(n_days * 86400)
    random_second_offsets = rng.integers(0, window_seconds, size=total_interactions, dtype=np.int64)

    # Sort each user's interactions by timestamp for realism
    # Base timestamp as numpy datetime64
    base_dt64 = np.datetime64(start_date)
    dt64_offsets = random_second_offsets.astype("timedelta64[s]")
    row_timestamps = base_dt64 + dt64_offsets

    # Generate unique interaction IDs
    pad_len = max(7, len(str(total_interactions)))
    # Create sequential IDs
    interaction_ids = [f"I{i+1:0{pad_len}d}" for i in range(total_interactions)]

    df = pd.DataFrame({
        COL_INTERACTION_ID: interaction_ids,
        COL_USER_ID: row_user_ids,
        COL_TIMESTAMP: row_timestamps,
        COL_PROVIDER: row_providers,
        COL_TASK: row_tasks,
    })

    # Sort by timestamp for realistic telemetry arrival
    df.sort_values(by=COL_TIMESTAMP, inplace=True, ignore_index=True)
    # Re-assign sequential interaction_id after temporal sort for clean presentation
    df[COL_INTERACTION_ID] = [f"I{i+1:0{pad_len}d}" for i in range(total_interactions)]

    return df
