"""Unit tests for global sensitivity bounds under add/remove user adjacency."""

import math
from private_llm_analytics.contribution.sensitivity import (
    get_interaction_volume_sensitivity,
    get_metric_sensitivities,
    get_normalized_share_sensitivity,
    get_prevalence_sensitivity,
)


def test_prevalence_sensitivities():
    l2, l1 = get_prevalence_sensitivity(max_cells_per_user=9)
    assert l2 == 3.0
    assert l1 == 9.0

    l2_k4, l1_k4 = get_prevalence_sensitivity(max_cells_per_user=4)
    assert l2_k4 == 2.0
    assert l1_k4 == 4.0


def test_normalized_share_sensitivities():
    l2, l1 = get_normalized_share_sensitivity(norm_bound=1.0)
    assert l2 == 1.0
    assert l1 == 1.0


def test_interaction_volume_sensitivities():
    for c in [5.0, 10.0, 20.0, 40.0, 80.0]:
        l2, l1 = get_interaction_volume_sensitivity(clip_norm=c, d_cells=9)
        assert l2 == c
        assert l1 == 3.0 * c # sqrt(9) * C = 3 * C


def test_metric_sensitivity_dispatcher():
    l2_prev, l1_prev = get_metric_sensitivities("prevalence", max_cells_per_user=9)
    assert l2_prev == 3.0 and l1_prev == 9.0

    l2_share, l1_share = get_metric_sensitivities("normalized_share", norm_bound=1.0)
    assert l2_share == 1.0 and l1_share == 1.0

    l2_vol, l1_vol = get_metric_sensitivities("volume", clip_norm=20.0)
    assert l2_vol == 20.0 and l1_vol == 60.0
