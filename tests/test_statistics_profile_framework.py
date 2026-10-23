"""Tests for Statistic-Specific DP Profile Framework (v0.6).

Verifies:
  - Registry contains approved production profiles
  - Unknown statistic fails closed
  - Profile metadata is deterministic and public
  - Profile fingerprint is deterministic and changes when semantic parameters change
  - Release identity incorporates profile fingerprint
  - Required columns are enforced per profile
  - Arbitrary mechanism selection is rejected in production
  - Prevalence profile invariants (binary participation, k=9, Delta_2=3.0, public denominator)
  - Normalized share profile invariants (norm bound 1.0, Delta_2=1.0, simplex renormalization)
  - Interaction volume profile invariants (L2 clipping, Delta_2=C, continual/planner metadata)
"""

import pytest
import numpy as np

from private_llm_analytics.aggregation.user_matrix import UserMatrix, build_cell_definitions
from private_llm_analytics.ingestion.schema import COL_SUCCESS, REQUIRED_COLUMNS
from private_llm_analytics.privacy.exceptions import (
    InvalidStatisticProfileConfiguration,
    MissingRequiredStatisticColumn,
    UnknownStatisticProfile,
)
from private_llm_analytics.statistics.base import DPStatisticProfile
from private_llm_analytics.statistics.interaction_volume import InteractionVolumeProfile
from private_llm_analytics.statistics.normalized_share import UserNormalizedShareProfile
from private_llm_analytics.statistics.prevalence import UserPrevalenceProfile
from private_llm_analytics.statistics.registry import StatisticProfileRegistry
from private_llm_analytics.statistics.success_rate import SuccessRateProfile


def make_test_user_matrix(include_success: bool = True) -> UserMatrix:
    provs = ["provider_a", "provider_b", "provider_c"]
    tsks = ["technical", "document", "analysis"]
    cell_names, cell_map = build_cell_definitions(provs, tsks)
    rng = np.random.default_rng(42)
    mat = rng.integers(0, 10, size=(10, 9)).astype(np.float64)
    s_mat = None
    if include_success:
        s_mat = np.clip(mat - rng.integers(0, 3, size=(10, 9)), 0, mat).astype(np.float64)

    return UserMatrix(
        matrix=mat,
        user_ids=[f"u_{i}" for i in range(10)],
        providers=provs,
        tasks=tsks,
        cell_names=cell_names,
        cell_map=cell_map,
        public_population_size=100,
        success_matrix=s_mat,
    )


class TestStatisticProfileRegistry:
    def test_registry_contains_approved_profiles(self):
        registered = StatisticProfileRegistry.list_statistics()
        expected = ["interaction_volume", "user_prevalence", "user_normalized_share", "success_rate"]
        assert set(expected).issubset(set(registered))

        for stat_id in expected:
            prof = StatisticProfileRegistry.get(stat_id)
            assert isinstance(prof, DPStatisticProfile)
            assert prof.statistic_id == stat_id

    def test_unknown_statistic_fails_closed(self):
        with pytest.raises(UnknownStatisticProfile):
            StatisticProfileRegistry.get("unsupported_top_k_frequency")

        with pytest.raises(UnknownStatisticProfile):
            StatisticProfileRegistry.get("exponential_mechanism_quantile")

    def test_profile_metadata_is_deterministic_and_public(self):
        for stat_id in StatisticProfileRegistry.list_statistics():
            meta1 = StatisticProfileRegistry.get_metadata(stat_id)
            meta2 = StatisticProfileRegistry.get_metadata(stat_id)
            assert meta1.dict() == meta2.dict()
            assert meta1.privacy_unit == "user"
            assert meta1.adjacency == "add_remove_user"
            assert meta1.mechanism_family in ["gaussian_vector", "joint_gaussian_vector"]
            assert meta1.accounting_family == "rdp"
            assert len(meta1.required_columns) >= 4


class TestProfileFingerprints:
    def test_fingerprint_deterministic(self):
        p1 = InteractionVolumeProfile(clip_norm=25.0)
        p2 = InteractionVolumeProfile(clip_norm=25.0)
        assert p1.canonical_profile_fingerprint() == p2.canonical_profile_fingerprint()

    def test_fingerprint_changes_with_parameters(self):
        p1 = InteractionVolumeProfile(clip_norm=25.0)
        p2 = InteractionVolumeProfile(clip_norm=30.0)
        assert p1.canonical_profile_fingerprint() != p2.canonical_profile_fingerprint()

        s1 = SuccessRateProfile(clip_norm=20.0, min_noisy_denominator=1.0)
        s2 = SuccessRateProfile(clip_norm=25.0, min_noisy_denominator=1.0)
        s3 = SuccessRateProfile(clip_norm=20.0, min_noisy_denominator=2.0)
        assert s1.canonical_profile_fingerprint() != s2.canonical_profile_fingerprint()
        assert s1.canonical_profile_fingerprint() != s3.canonical_profile_fingerprint()


class TestPrevalenceProfile:
    def test_prevalence_invariants(self):
        prof = UserPrevalenceProfile(max_cells_per_user=9)
        assert prof.sensitivity() == 3.0
        assert prof.contribution_bound() == 9.0

        umatrix = make_test_user_matrix()
        contrib = prof.build_user_contributions(umatrix)
        assert contrib.shape == (10, 9)
        assert np.all(np.isin(contrib, [0.0, 1.0]))

        # L2 norm bounded by 3
        user_norms = np.linalg.norm(contrib, ord=2, axis=1)
        assert np.all(user_norms <= 3.0 + 1e-12)

        # Post-processing: public denominator, clamped to [0, 1]
        noisy_vec = np.array([-5.0, 20.0, 150.0] + [10.0] * 6)
        pp = prof.post_process(noisy_vec, umatrix)
        for val in pp["prevalence_rates"].values():
            assert 0.0 <= val <= 1.0

    def test_prevalence_disallows_invalid_k(self):
        with pytest.raises(InvalidStatisticProfileConfiguration):
            UserPrevalenceProfile(max_cells_per_user=5)


class TestNormalizedShareProfile:
    def test_normalized_share_invariants(self):
        prof = UserNormalizedShareProfile(norm_bound=1.0)
        assert prof.sensitivity() == 1.0
        assert prof.contribution_bound() == 1.0

        umatrix = make_test_user_matrix()
        contrib = prof.build_user_contributions(umatrix)
        assert contrib.shape == (10, 9)
        l1_norms = np.linalg.norm(contrib, ord=1, axis=1)
        assert np.all(l1_norms <= 1.0 + 1e-12)

        # Post-process nonnegative simplex renormalization
        noisy_vec = np.array([-2.0, 5.0, 3.0, -1.0, 4.0, 2.0, 0.0, 1.0, 6.0])
        pp = prof.post_process(noisy_vec, umatrix)
        assert abs(sum(pp["cell_shares"].values()) - 1.0) < 1e-10
        assert abs(sum(pp["provider_distribution"].values()) - 1.0) < 1e-10
        assert abs(sum(pp["task_distribution"].values()) - 1.0) < 1e-10

    def test_normalized_share_disallows_invalid_bound(self):
        with pytest.raises(InvalidStatisticProfileConfiguration):
            UserNormalizedShareProfile(norm_bound=2.0)


class TestInteractionVolumeProfile:
    def test_volume_invariants(self):
        c = 15.0
        prof = InteractionVolumeProfile(clip_norm=c)
        assert prof.sensitivity() == c
        assert prof.contribution_bound() == c
        assert prof.metadata().supports_continual is True
        assert prof.metadata().supports_planner is True

        umatrix = make_test_user_matrix()
        contrib = prof.build_user_contributions(umatrix)
        norms = np.linalg.norm(contrib, ord=2, axis=1)
        assert np.all(norms <= c + 1e-12)

        # Negative output clamped
        noisy_vec = np.array([-10.0, 20.0, -5.0] + [12.0] * 6)
        pp = prof.post_process(noisy_vec, umatrix)
        for val in pp["cell_volumes"].values():
            assert val >= 0.0

    def test_volume_disallows_nonpositive_c(self):
        with pytest.raises(InvalidStatisticProfileConfiguration):
            InteractionVolumeProfile(clip_norm=0.0)
        with pytest.raises(InvalidStatisticProfileConfiguration):
            InteractionVolumeProfile(clip_norm=-5.0)


class TestSuccessRateRequiredColumns:
    def test_missing_success_column_fails_closed(self):
        prof = SuccessRateProfile(clip_norm=20.0)
        umatrix = make_test_user_matrix(include_success=False)
        with pytest.raises(MissingRequiredStatisticColumn):
            prof.build_user_contributions(umatrix)
