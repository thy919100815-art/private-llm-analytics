"""User-level matrix aggregation module in v0.1.1."""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from private_llm_analytics.aggregation.grouping import (
    DEFAULT_PROVIDERS,
    DEFAULT_TASKS,
    build_cell_definitions,
)
from private_llm_analytics.ingestion.schema import (
    COL_PROVIDER,
    COL_SUCCESS,
    COL_TASK,
    COL_USER_ID,
)
from private_llm_analytics.privacy.exceptions import (
    DatasetConfigurationMismatchError,
    PrivacyConfigurationError,
)


@dataclass
class UserMatrix:
    """Represents the pre-aggregated user-by-cell interaction count matrix."""
    matrix: np.ndarray             # shape (n_users, 9), float64 counts
    user_ids: List[str]            # list of length n_users
    providers: List[str]           # 3 providers
    tasks: List[str]               # 3 tasks
    cell_names: List[str]          # 9 cell names
    cell_map: Dict[Tuple[str, str], int]
    public_population_size: Optional[int] = None  # Fixed public eligible population denominator
    success_matrix: Optional[np.ndarray] = None   # shape (n_users, 9), float64 successful interaction counts

    def __post_init__(self):
        if self.public_population_size is not None:
            if self.public_population_size <= 0:
                raise PrivacyConfigurationError(f"public_population_size must be positive, got {self.public_population_size}")
            if self.public_population_size < self.n_users:
                raise DatasetConfigurationMismatchError(
                    f"public_population_size ({self.public_population_size}) cannot be less than active user count ({self.n_users})."
                )

    @property
    def n_users(self) -> int:
        return len(self.user_ids)

    @property
    def n_cells(self) -> int:
        return len(self.cell_names)

    @property
    def total_interactions(self) -> float:
        return float(self.matrix.sum())


def build_user_matrix(
    df: pd.DataFrame,
    providers: Optional[List[str]] = None,
    tasks: Optional[List[str]] = None,
    user_ids: Optional[List[str]] = None,
    public_population_size: Optional[int] = None,
) -> UserMatrix:
    """Aggregate interaction records into a user-level 9-dimensional count matrix.

    Optimized for high performance on millions of rows.
    """
    provs = providers or DEFAULT_PROVIDERS
    tsks = tasks or DEFAULT_TASKS
    cell_names, cell_map = build_cell_definitions(provs, tsks)

    if user_ids is not None:
        target_users = list(user_ids)
    else:
        target_users = sorted(df[COL_USER_ID].unique().tolist())

    n_users = len(target_users)
    if public_population_size is not None:
        if public_population_size <= 0:
            raise PrivacyConfigurationError(f"public_population_size must be positive, got {public_population_size}")
        if public_population_size < n_users:
            raise DatasetConfigurationMismatchError(
                f"public_population_size ({public_population_size}) cannot be less than active user count ({n_users})."
            )

    user_to_idx = {uid: i for i, uid in enumerate(target_users)}

    matrix = np.zeros((n_users, 9), dtype=np.float64)
    success_matrix: Optional[np.ndarray] = None

    if not df.empty:
        # Filter to target users if user_ids was specified
        if user_ids is not None:
            active_mask = df[COL_USER_ID].isin(user_to_idx)
            sub_df = df[active_mask]
        else:
            sub_df = df

        # Map user_id to row index
        u_indices = sub_df[COL_USER_ID].map(user_to_idx).to_numpy(dtype=np.int64)

        # Map (provider, task) to cell index
        p_code = sub_df[COL_PROVIDER].map({p: i for i, p in enumerate(provs)}).to_numpy(dtype=np.int64)
        t_code = sub_df[COL_TASK].map({t: i for i, t in enumerate(tsks)}).to_numpy(dtype=np.int64)
        c_indices = p_code * len(tsks) + t_code

        # Fast 2D bincount via flat index
        flat_indices = u_indices * 9 + c_indices
        flat_counts = np.bincount(flat_indices, minlength=n_users * 9)
        matrix = flat_counts.reshape((n_users, 9)).astype(np.float64)

        if COL_SUCCESS in sub_df.columns:
            s_vals = sub_df[COL_SUCCESS].astype(bool).astype(np.float64).to_numpy()
            flat_success = np.bincount(flat_indices, weights=s_vals, minlength=n_users * 9)
            success_matrix = flat_success.reshape((n_users, 9)).astype(np.float64)
    elif COL_SUCCESS in df.columns:
        success_matrix = np.zeros((n_users, 9), dtype=np.float64)

    return UserMatrix(
        matrix=matrix,
        user_ids=target_users,
        providers=provs,
        tasks=tsks,
        cell_names=cell_names,
        cell_map=cell_map,
        public_population_size=public_population_size,
        success_matrix=success_matrix,
    )
