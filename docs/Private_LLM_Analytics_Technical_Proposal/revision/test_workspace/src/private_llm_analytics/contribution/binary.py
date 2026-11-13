"""User prevalence contribution bounding (binary transformation)."""

from typing import Optional
import numpy as np


def transform_user_prevalence(
    matrix: np.ndarray,
    max_cells_per_user: Optional[int] = None,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Transform user interaction count matrix into binary prevalence vectors.

    b_{u, p, c} = 1[m_{u, p, c} > 0]

    If max_cells_per_user is specified and a user is active in > max_cells_per_user cells,
    a random subset of size max_cells_per_user is sampled without replacement.
    """
    binary_matrix = (matrix > 0).astype(np.float64)
    d = matrix.shape[1] if matrix.ndim > 1 else len(matrix)

    if max_cells_per_user is not None and max_cells_per_user < d:
        k = max_cells_per_user
        generator = rng or np.random.default_rng(42)
        for u in range(binary_matrix.shape[0]):
            active_indices = np.where(binary_matrix[u] > 0)[0]
            if len(active_indices) > k:
                chosen = generator.choice(active_indices, size=k, replace=False)
                new_row = np.zeros(d, dtype=np.float64)
                new_row[chosen] = 1.0
                binary_matrix[u] = new_row

    # Verification: check binary
    unique_vals = np.unique(binary_matrix)
    if not np.all(np.isin(unique_vals, [0.0, 1.0])):
        raise ValueError(f"Prevalence matrix contains non-binary values: {unique_vals}")

    return binary_matrix
