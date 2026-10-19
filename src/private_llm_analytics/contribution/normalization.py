"""User-normalized share contribution transformation."""

import numpy as np


def transform_user_normalized_share(matrix: np.ndarray) -> np.ndarray:
    """Transform user interaction count matrix into user-normalized share vectors.

    x_u = m_u / ||m_u||_1   if ||m_u||_1 > 0
    x_u = 0                 if ||m_u||_1 == 0

    Guarantees:
      - x_{u, j} >= 0
      - ||x_u||_1 = 1 for all active users
      - ||x_u||_2 <= 1 for all active users
    """
    matrix_f = matrix.astype(np.float64)
    l1_norms = np.sum(matrix_f, axis=1, keepdims=True)

    # Avoid division by zero for inactive users
    nonzero_mask = (l1_norms > 0).ravel()
    normalized = np.zeros_like(matrix_f)

    if np.any(nonzero_mask):
        normalized[nonzero_mask] = matrix_f[nonzero_mask] / l1_norms[nonzero_mask]

    # Verify correctness gate conditions
    active_rows = normalized[nonzero_mask]
    if len(active_rows) > 0:
        row_l1s = np.sum(active_rows, axis=1)
        max_l1_err = np.max(np.abs(row_l1s - 1.0))
        if max_l1_err > 1e-10:
            raise ValueError(
                f"User-Normalized Share violation: max |L1 - 1.0| is {max_l1_err:.2e} > 1e-10"
            )

        row_l2s = np.linalg.norm(active_rows, ord=2, axis=1)
        max_l2 = np.max(row_l2s)
        if max_l2 > 1.0 + 1e-10:
            raise ValueError(
                f"User-Normalized Share violation: max L2 norm is {max_l2:.6f} > 1.0"
            )

    return normalized
