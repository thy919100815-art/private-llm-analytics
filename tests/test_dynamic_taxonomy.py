"""Unit and integration tests for Dynamic Taxonomy and Arbitrary Domains in v0.7."""

import numpy as np
import pandas as pd
import pytest

from private_llm_analytics.aggregation.grouping import build_cell_definitions
from private_llm_analytics.aggregation.user_matrix import UserMatrix, build_user_matrix
from private_llm_analytics.config.models import (
    AppConfig,
    ScenarioConfig,
    TaxonomyConfig,
)
from private_llm_analytics.continual.models import (
    ContinualNoiseConfig,
    ContinualStreamConfig,
    ContinualTreeConfig,
)
from private_llm_analytics.planning.policy import (
    aggregate_to_granularity,
    build_volume_contributions_for_granularity,
    get_direct_granularity_sensitivity,
)
from private_llm_analytics.statistics.interaction_volume import InteractionVolumeProfile
from private_llm_analytics.statistics.normalized_share import UserNormalizedShareProfile
from private_llm_analytics.statistics.prevalence import UserPrevalenceProfile
from private_llm_analytics.statistics.success_rate import SuccessRateProfile


class TestTaxonomyConfig:
    def test_taxonomy_fingerprint_deterministic(self):
        tax1 = TaxonomyConfig(
            version=1,
            providers=["provider_b", "provider_a"],
            tasks=["task_y", "task_x"],
            ordering="canonical_lexicographic",
        )
        tax2 = TaxonomyConfig(
            version=1,
            providers=["provider_a", "provider_b"],
            tasks=["task_x", "task_y"],
            ordering="canonical_lexicographic",
        )
        # Lexicographic ordering produces identical fingerprint regardless of input list order
        assert tax1.compute_fingerprint() == tax2.compute_fingerprint()

    def test_taxonomy_fingerprint_changes_on_domain_shift(self):
        tax1 = TaxonomyConfig(version=1, providers=["p1", "p2"], tasks=["t1"])
        tax2 = TaxonomyConfig(version=1, providers=["p1", "p2", "p3"], tasks=["t1"])
        tax3 = TaxonomyConfig(version=2, providers=["p1", "p2"], tasks=["t1"])

        fp1 = tax1.compute_fingerprint()
        fp2 = tax2.compute_fingerprint()
        fp3 = tax3.compute_fingerprint()

        assert fp1 != fp2
        assert fp1 != fp3
        assert fp2 != fp3

    def test_taxonomy_requires_non_empty_providers_and_tasks(self):
        with pytest.raises(ValueError, match="At least 1 provider required"):
            TaxonomyConfig(providers=[], tasks=["t1"])

        with pytest.raises(ValueError, match="At least 1 task required"):
            TaxonomyConfig(providers=["p1"], tasks=[])

        with pytest.raises(ValueError, match="Duplicate providers found"):
            TaxonomyConfig(providers=["p1", "p1"], tasks=["t1"])


class TestDynamicUserMatrix:
    def test_build_user_matrix_arbitrary_dimensions(self):
        providers = [f"p_{i}" for i in range(5)]
        tasks = [f"t_{j}" for j in range(10)]
        expected_cells = 50

        # Create synthetic interactions
        data = {
            "user_id": ["u1", "u1", "u2", "u3", "u1"],
            "timestamp": ["2026-01-01 10:00:00"] * 5,
            "provider": ["p_0", "p_4", "p_2", "p_1", "p_unknown"],  # p_unknown should be ignored
            "task": ["t_1", "t_9", "t_5", "t_3", "t_0"],
            "success": [True, False, True, True, True],
        }
        df = pd.DataFrame(data)

        user_matrix = build_user_matrix(
            df=df,
            providers=providers,
            tasks=tasks,
            public_population_size=10,
        )

        assert user_matrix.n_cells == expected_cells
        assert user_matrix.matrix.shape == (3, expected_cells)
        assert user_matrix.success_matrix is not None
        assert user_matrix.success_matrix.shape == (3, expected_cells)

        # Total interactions should be 4 (1 ignored)
        assert user_matrix.total_interactions == 4.0


class TestDynamicStatisticsProfiles:
    def test_prevalence_profile_dynamic_cells(self):
        providers = [f"prov_{i}" for i in range(4)]
        tasks = [f"task_{j}" for j in range(5)]
        d_cells = 20

        df = pd.DataFrame({
            "user_id": ["u1", "u1", "u2"],
            "timestamp": ["2026-01-01 10:00:00"] * 3,
            "provider": ["prov_0", "prov_1", "prov_3"],
            "task": ["task_0", "task_2", "task_4"],
        })
        user_matrix = build_user_matrix(df, providers=providers, tasks=tasks, public_population_size=10)

        profile = UserPrevalenceProfile(max_cells_per_user=20)
        assert profile.sensitivity() == np.sqrt(20)

        res = profile.execute_release(user_matrix, epsilon=1.0, delta=1e-6)
        assert len(res.dp_vector) == d_cells
        assert "prevalence_rates" in res.post_processed
        assert len(res.post_processed["prevalence_rates"]) == d_cells

    def test_normalized_share_profile_dynamic_cells(self):
        providers = ["p_a", "p_b"]
        tasks = ["t_1", "t_2", "t_3"]
        d_cells = 6

        df = pd.DataFrame({
            "user_id": ["u1", "u2"],
            "timestamp": ["2026-01-01 10:00:00"] * 2,
            "provider": ["p_a", "p_b"],
            "task": ["t_1", "t_3"],
        })
        user_matrix = build_user_matrix(df, providers=providers, tasks=tasks, public_population_size=5)

        profile = UserNormalizedShareProfile()
        assert profile.sensitivity() == 1.0

        res = profile.execute_release(user_matrix, epsilon=1.0, delta=1e-6)
        assert len(res.dp_vector) == d_cells
        assert "grid" in res.post_processed
        assert res.post_processed["grid"].shape == (2, 3)
        assert np.isclose(np.sum(res.post_processed["share_distribution_array"]), 1.0)

    def test_volume_profile_dynamic_cells(self):
        providers = [f"p_{i}" for i in range(5)]
        tasks = [f"t_{j}" for j in range(5)]
        d_cells = 25

        df = pd.DataFrame({
            "user_id": ["u1", "u2"],
            "timestamp": ["2026-01-01 10:00:00"] * 2,
            "provider": ["p_0", "p_4"],
            "task": ["t_0", "t_4"],
        })
        user_matrix = build_user_matrix(df, providers=providers, tasks=tasks, public_population_size=10)

        profile = InteractionVolumeProfile(clip_norm=15.0)
        assert profile.sensitivity() == 15.0

        res = profile.execute_release(user_matrix, epsilon=1.0, delta=1e-6)
        assert len(res.dp_vector) == d_cells
        assert len(res.post_processed["cell_volumes"]) == d_cells

    def test_success_rate_profile_dynamic_cells(self):
        providers = [f"p_{i}" for i in range(3)]
        tasks = [f"t_{j}" for j in range(4)]
        d_cells = 12
        joint_dim = 24

        df = pd.DataFrame({
            "user_id": ["u1", "u2"],
            "timestamp": ["2026-01-01 10:00:00"] * 2,
            "provider": ["p_0", "p_1"],
            "task": ["t_0", "t_2"],
            "success": [True, False],
        })
        user_matrix = build_user_matrix(df, providers=providers, tasks=tasks, public_population_size=10)

        profile = SuccessRateProfile(clip_norm=10.0, min_noisy_denominator=0.0)
        # Mathematical invariant: Delta_2 = sqrt(2) * C_rate, independent of D!
        assert np.isclose(profile.sensitivity(), np.sqrt(2.0) * 10.0)

        contributions = profile.build_user_contributions(user_matrix)
        assert contributions.shape == (2, joint_dim)

        res = profile.execute_release(user_matrix, epsilon=1.0, delta=1e-6)
        assert len(res.dp_vector) == joint_dim
        assert len(res.post_processed["success_rates"]) == d_cells


class TestContinualTaxonomyIntegration:
    def test_continual_stream_incorporates_taxonomy_fingerprint(self):
        tax = TaxonomyConfig(providers=["p1", "p2"], tasks=["t1", "t2"])
        tax_fp = tax.compute_fingerprint()

        cfg1 = ContinualStreamConfig(
            stream_id="test_stream",
            taxonomy_fingerprint=tax_fp,
        )
        cfg2 = ContinualStreamConfig(
            stream_id="test_stream",
            taxonomy_fingerprint=None,
        )
        assert cfg1.stream_fingerprint != cfg2.stream_fingerprint


class TestPlanningConceptualGranularities:
    def test_conceptual_granularity_aliases(self):
        assert get_direct_granularity_sensitivity("volume", "full_cell", clip_norm=20.0) == 20.0
        assert get_direct_granularity_sensitivity("prevalence", "full_cell") == 3.0
        assert get_direct_granularity_sensitivity("prevalence", "provider_marginal") == np.sqrt(3.0)
        assert get_direct_granularity_sensitivity("prevalence", "task_marginal") == np.sqrt(3.0)
        assert get_direct_granularity_sensitivity("prevalence", "overall_cell") == 1.0
