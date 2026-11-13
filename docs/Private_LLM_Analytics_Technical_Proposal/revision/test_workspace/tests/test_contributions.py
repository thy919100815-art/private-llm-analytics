"""Unit tests for contribution transformations and norm bounding gates."""

import numpy as np
import pytest

from private_llm_analytics.contribution.binary import transform_user_prevalence
from private_llm_analytics.contribution.clipping import transform_interaction_volume_l2_clip
from private_llm_analytics.contribution.normalization import transform_user_normalized_share


def test_user_normalized_share_norms():
    # Matrix with varied counts, including a zero user
    matrix = np.array([
        [10.0, 20.0, 30.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], # inactive user
        [0.0, 0.0, 0.0, 0.0, 100.0, 0.0, 0.0, 0.0, 0.0],
    ])

    norm_matrix = transform_user_normalized_share(matrix)

    # Active rows: index 0, 1, 3
    for u in [0, 1, 3]:
        l1 = float(np.sum(norm_matrix[u]))
        l2 = float(np.linalg.norm(norm_matrix[u], ord=2))
        assert abs(l1 - 1.0) <= 1e-10, f"User {u} L1 norm {l1} != 1.0"
        assert l2 <= 1.0 + 1e-10, f"User {u} L2 norm {l2} > 1.0"

    # Inactive row: index 2 should be all zeros
    assert np.all(norm_matrix[2] == 0.0)


def test_interaction_volume_l2_clipping():
    matrix = np.array([
        [100.0, 100.0, 100.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], # L2 = sqrt(30000) ~ 173.2
        [1.0, 2.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],       # L2 = 3.0
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],       # L2 = 0
    ])
    clip_norm = 20.0

    clipped = transform_interaction_volume_l2_clip(matrix, clip_norm=clip_norm)

    for u in range(len(matrix)):
        l2 = float(np.linalg.norm(clipped[u], ord=2))
        assert l2 <= clip_norm + 1e-10, f"User {u} clipped L2 {l2} exceeds C={clip_norm}"

    # Row 0 was scaled down to exactly C=20.0
    assert abs(np.linalg.norm(clipped[0], ord=2) - 20.0) <= 1e-10

    # Row 1 norm was 3.0 <= 20.0, so it remains unclipped
    assert np.allclose(clipped[1], matrix[1])


def test_user_prevalence_binary_transformation():
    matrix = np.array([
        [5.0, 0.0, 10.0, 0.0, 2.0, 0.0, 0.0, 0.0, 1.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
    ])

    binary = transform_user_prevalence(matrix, max_cells_per_user=9)

    # Must be strictly 0 or 1
    assert set(np.unique(binary)).issubset({0.0, 1.0})

    # Row 0 active cells: 4 cells
    assert np.sum(binary[0]) == 4.0

    # Row 1 active cells: 0 cells
    assert np.sum(binary[1]) == 0.0

    # Bounded to K=2
    binary_k2 = transform_user_prevalence(matrix, max_cells_per_user=2)
    assert np.sum(binary_k2[0]) == 2.0
    assert np.sum(binary_k2[2]) == 2.0
