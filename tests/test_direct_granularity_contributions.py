"""Tests for Direct Granularity User-Level DP Contributions (v0.3).

Verifies target-space clipping, exact sensitivities, and binary prevalence semantics.
"""

import math
import numpy as np
import pytest

from private_llm_analytics.planning.policy import (
    build_prevalence_contributions_for_granularity,
    build_share_contributions_for_granularity,
    build_volume_contributions_for_granularity,
    get_direct_granularity_sensitivity,
)


def test_volume_direct_granularity_l2_bound():
    """Every user's contribution vector in target granularity space strictly respects ||bar_m_u^G||_2 <= C."""
    # Synthetic raw user matrix: 5 users with heterogeneous interaction profiles
    raw_matrix = np.array([
        [10.0, 20.0, 30.0, 0.0, 0.0, 0.0, 5.0, 5.0, 5.0],   # User 1: norm ~38.4
        [100.0, 100.0, 100.0, 50.0, 50.0, 50.0, 0.0, 0.0, 0.0], # User 2: norm ~212.1
        [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],       # User 3: norm 1.0
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],       # User 4: zero
        [50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0], # User 5: norm 150.0
    ])

    granularities = ["full_9_cell", "provider_3_cell", "task_3_cell", "overall_1_cell"]
    c_bounds = [5.0, 10.0, 20.0, 40.0, 80.0]

    for g in granularities:
        for c in c_bounds:
            clipped, exact_sum, names, sens = build_volume_contributions_for_granularity(
                raw_user_matrix=raw_matrix,
                granularity=g,
                clip_norm=c,
            )
            # 1. Verify sensitivity matches C
            assert sens == c

            # 2. Verify dimension matches granularity
            if g == "full_9_cell":
                assert clipped.shape == (5, 9)
            elif g in ("provider_3_cell", "task_3_cell"):
                assert clipped.shape == (5, 3)
            elif g == "overall_1_cell":
                assert clipped.shape == (5, 1)

            # 3. Verify strict L2 bound for every user
            user_norms = np.linalg.norm(clipped, axis=1)
            assert np.all(user_norms <= c + 1e-12), f"User norm exceeded C={c} in granularity {g}: {user_norms}"

            # 4. Sum matches sum of rows
            np.testing.assert_allclose(exact_sum, np.sum(clipped, axis=0))


def test_volume_direct_granularity_sensitivity():
    """L2 sensitivity in target G-space is strictly C under add/remove user adjacency."""
    c = 25.0
    for g in ["full_9_cell", "provider_3_cell", "task_3_cell", "overall_1_cell"]:
        sens = get_direct_granularity_sensitivity("interaction_volume", g, clip_norm=c)
        assert sens == c

        # Adversarial single user with arbitrary extreme magnitude
        extreme_user = np.full((1, 9), 10000.0)
        clipped, _, _, _ = build_volume_contributions_for_granularity(extreme_user, g, clip_norm=c)
        norm_diff = float(np.linalg.norm(clipped[0]))
        assert abs(norm_diff - c) < 1e-12, f"Expected clipped user norm to equal C={c}, got {norm_diff}"


def test_prevalence_provider_binary_semantics():
    """Provider prevalence uses direct binary indicators, NOT the sum of 9-cell binary flags."""
    # User 1 used provider_a across all 3 tasks (technical, document, analysis)
    # User 2 used provider_a for 1 task and provider_b for 1 task
    # User 3 used only provider_c
    raw_matrix = np.array([
        [10.0, 20.0, 30.0,  0.0, 0.0, 0.0,  0.0, 0.0, 0.0],  # User 1
        [ 5.0,  0.0,  0.0,  8.0, 0.0, 0.0,  0.0, 0.0, 0.0],  # User 2
        [ 0.0,  0.0,  0.0,  0.0, 0.0, 0.0,  1.0, 0.0, 0.0],  # User 3
    ])

    binary_mat, counts, names, sens = build_prevalence_contributions_for_granularity(
        raw_user_matrix=raw_matrix,
        granularity="provider_3_cell",
    )

    assert names == ["provider_a", "provider_b", "provider_c"]
    assert sens == pytest.approx(math.sqrt(3.0))

    # User 1 must have count 1 for provider_a (NOT 3!)
    assert binary_mat[0, 0] == 1.0
    assert binary_mat[0, 1] == 0.0
    assert binary_mat[0, 2] == 0.0

    # User 2 has count 1 for provider_a and 1 for provider_b
    assert binary_mat[1, 0] == 1.0
    assert binary_mat[1, 1] == 1.0
    assert binary_mat[1, 2] == 0.0

    # Total counts: provider_a=2, provider_b=1, provider_c=1
    np.testing.assert_array_equal(counts, [2.0, 1.0, 1.0])


def test_prevalence_task_binary_semantics():
    """Task prevalence uses direct binary indicators with sensitivity sqrt(3)."""
    # Columns:
    # 0: prov_a:tech, 1: prov_a:doc, 2: prov_a:analysis
    # 3: prov_b:tech, 4: prov_b:doc, 5: prov_b:analysis
    # 6: prov_c:tech, 7: prov_c:doc, 8: prov_c:analysis
    raw_matrix = np.array([
        [10.0, 0.0, 0.0,  5.0, 0.0, 0.0,  3.0, 0.0, 0.0],  # User 1: used 'technical' across 3 providers
        [ 0.0, 2.0, 0.0,  0.0, 4.0, 0.0,  0.0, 0.0, 6.0],  # User 2: used 'document' and 'analysis'
    ])

    binary_mat, counts, names, sens = build_prevalence_contributions_for_granularity(
        raw_user_matrix=raw_matrix,
        granularity="task_3_cell",
    )

    assert names == ["analysis", "document", "technical"]
    assert sens == pytest.approx(math.sqrt(3.0))

    # User 1 used technical across 3 providers, but task prevalence indicator must be 1.0 (not 3.0!)
    tech_idx = names.index("technical")
    assert binary_mat[0, tech_idx] == 1.0
    # Other tasks 0
    assert binary_mat[0, names.index("analysis")] == 0.0
    assert binary_mat[0, names.index("document")] == 0.0


def test_overall_prevalence_binary_semantics():
    """Overall prevalence is active user indicator with sensitivity 1.0."""
    raw_matrix = np.array([
        [10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], # User 1: active
        [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], # User 2: inactive
        [ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0], # User 3: active
    ])

    binary_mat, counts, names, sens = build_prevalence_contributions_for_granularity(
        raw_user_matrix=raw_matrix,
        granularity="overall_1_cell",
    )

    assert names == ["overall_total"]
    assert sens == 1.0
    np.testing.assert_array_equal(binary_mat.flatten(), [1.0, 0.0, 1.0])
    assert counts[0] == 2.0
