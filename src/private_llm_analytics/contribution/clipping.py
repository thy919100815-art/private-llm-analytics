"""Interaction volume L2 clipping contribution transformation."""

import numpy as np


def transform_interaction_volume_l2_clip(
    matrix: np.ndarray,
    clip_norm: float = 20.0,
) -> np.ndarray:
    """Transform user interaction count matrix by clipping user L2 norm to C.

    \bar{m}_u = m_u * min(1, C / ||m_u||_2)

    Guarantees:
      - ||\bar{m}_u||_2 <= C for all users
    """
    if clip_norm <= 0:
        raise ValueError(f"clip_norm C must be strictly positive, got {clip_norm}")

    matrix_f = matrix.astype(np.float64)
    l2_norms = np.linalg.norm(matrix_f, ord=2, axis=1, keepdims=True)

    nonzero_mask = (l2_norms > 0).ravel()
    clipped = np.zeros_like(matrix_f)

    if np.any(nonzero_mask):
        # Scaling factor: min(1, C / ||m_u||_2)
        scale_factors = np.minimum(1.0, clip_norm / np.maximum(l2_norms[nonzero_mask], 1e-15))
        clipped[nonzero_mask] = matrix_f[nonzero_mask] * scale_factors

    # Verify correctness gate conditions
    clipped_l2s = np.linalg.norm(clipped, ord=2, axis=1)
    max_l2 = np.max(clipped_l2s) if len(clipped_l2s) > 0 else 0.0
    if max_l2 > clip_norm + 1e-10:
        raise ValueError(
            f"Interaction Volume clipping violation: max L2 norm {max_l2:.6f} > C ({clip_norm})"
        )

    return clipped
