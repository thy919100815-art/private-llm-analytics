"""Validation and regression tests for Audit Fixes in v1.0.x / v1.1-dev.

Covers:
1. Dynamic taxonomy + Laplace sensitivity scaling (D = 4, 9, 12, 50).
2. Fail-closed mechanism enforcement for unsupported combinations.
3. Success Rate dynamic taxonomy helpers and Metric.get_sensitivity().
4. ReleaseKey parameter sensitivity and canonical replay stability.
5. Continual observation Generator / RNG isolation.
6. Top-K tasks end-to-end execution through ReleaseOrchestrator.
"""

import math
import numpy as np
import pandas as pd
import pytest

from private_llm_analytics.aggregation.user_matrix import build_user_matrix
from private_llm_analytics.config import load_app_config
from private_llm_analytics.config.models import AppConfig
from private_llm_analytics.config.validator import ExecutionMode
from private_llm_analytics.contribution.sensitivity import (
    get_interaction_volume_sensitivity,
    get_metric_sensitivities,
    get_success_rate_sensitivity,
)
from private_llm_analytics.metrics.catalog import MetricCatalog
from private_llm_analytics.metrics.success_rate import SuccessRateMetric
from private_llm_analytics.planner.planner import AnalyticsPlanner
from private_llm_analytics.planner.query import AnalyticsQuery
from private_llm_analytics.privacy.exceptions import (
    ProductionRNGControlError,
    UnsupportedMechanismError,
)
from private_llm_analytics.release.release import ReleaseOrchestrator
from private_llm_analytics.statistics.success_rate import SuccessRateProfile


class TestDynamicTaxonomyLaplace:
    """Audit Item 1: Dynamic taxonomy D-cell scaling for L1 sensitivity."""

    @pytest.mark.parametrize("d_cells,expected_mult", [
        (4, 2.0),
        (9, 3.0),
        (12, math.sqrt(12.0)),
        (50, math.sqrt(50.0)),
    ])
    def test_volume_l1_sensitivity_scales_with_d(self, d_cells, expected_mult):
        clip_c = 20.0
        l2_s, l1_s = get_interaction_volume_sensitivity(clip_norm=clip_c, d_cells=d_cells)
        assert l2_s == clip_c
        assert math.isclose(l1_s, expected_mult * clip_c, rel_tol=1e-9)

    @pytest.mark.parametrize("d_cells,expected_mult", [
        (4, math.sqrt(8.0)),
        (9, math.sqrt(18.0)),
        (12, math.sqrt(24.0)),
        (50, 10.0),
    ])
    def test_success_rate_l1_sensitivity_scales_with_d(self, d_cells, expected_mult):
        clip_rate = 10.0
        l2_s, l1_s = get_success_rate_sensitivity(clip_norm_rate=clip_rate, d_cells=d_cells)
        assert math.isclose(l2_s, math.sqrt(2.0) * clip_rate, rel_tol=1e-9)
        assert math.isclose(l1_s, expected_mult * clip_rate, rel_tol=1e-9)

    def test_planner_computes_exact_d_cells_for_laplace(self):
        config = load_app_config("configs")
        config.scenario.providers = [f"p_{i}" for i in range(5)]
        config.scenario.tasks = [f"t_{j}" for j in range(10)]
        # Total D = 50 cells
        planner = AnalyticsPlanner(config)
        query = AnalyticsQuery(
            metric="interaction_volume",
            recipient="internal",
            mechanism="laplace",
            epsilon=1.0,
            delta=0.0,
        )
        plan = planner.create_plan(query, dataset_version="v1.0", execution_mode=ExecutionMode.EXPERIMENT)
        expected_l1 = math.sqrt(50.0) * 20.0
        assert math.isclose(plan.sensitivity, expected_l1, rel_tol=1e-9)
        assert plan.mechanism == "laplace"

    def test_fail_closed_mechanism_enforcement(self):
        config = load_app_config("configs")
        planner = AnalyticsPlanner(config)

        # 1. Success rate rejects Laplace
        q_sr = AnalyticsQuery(metric="success_rate", recipient="internal", mechanism="laplace", epsilon=1.0, delta=0.0)
        with pytest.raises(UnsupportedMechanismError, match="only supports Gaussian mechanism"):
            planner.create_plan(q_sr, dataset_version="v1.0", execution_mode=ExecutionMode.EXPERIMENT)

        # 2. Top-K tasks rejects Laplace
        q_tk = AnalyticsQuery(metric="top_k_tasks", recipient="internal", mechanism="laplace", epsilon=1.0, delta=0.0)
        with pytest.raises(UnsupportedMechanismError, match="only supports Gaussian mechanism"):
            planner.create_plan(q_tk, dataset_version="v1.0", execution_mode=ExecutionMode.EXPERIMENT)

        # 3. Production mode rejects overriding Gaussian metric with Laplace
        q_prod = AnalyticsQuery(metric="interaction_volume", recipient="internal", mechanism="laplace", epsilon=1.0, delta=0.0)
        with pytest.raises(UnsupportedMechanismError, match="Laplace override is rejected in production"):
            planner.create_plan(q_prod, dataset_version="v1.0", execution_mode=ExecutionMode.PRODUCTION)


class TestSuccessRateAuditFixes:
    """Audit Item 2: Dynamic taxonomy support and method existence."""

    def test_success_rate_metric_get_sensitivity(self):
        metric = SuccessRateMetric(name="success_rate", clip_norm_rate=15.0)
        sens = metric.get_sensitivity("gaussian")
        assert math.isclose(sens, math.sqrt(2.0) * 15.0, rel_tol=1e-9)

    def test_success_rate_dynamic_helpers_3x4(self):
        providers = [f"prov_{i}" for i in range(3)]
        tasks = [f"task_{j}" for j in range(4)]
        d_cells = 12

        df = pd.DataFrame({
            "user_id": ["u1", "u1", "u2", "u3", "u2"],
            "timestamp": ["2026-01-01 10:00:00"] * 5,
            "provider": ["prov_0", "prov_2", "prov_1", "prov_0", "prov_2"],
            "task": ["task_0", "task_3", "task_1", "task_2", "task_3"],
            "success": [True, False, True, True, True],
        })
        user_matrix = build_user_matrix(df, providers=providers, tasks=tasks, public_population_size=10)
        profile = SuccessRateProfile(clip_norm=10.0, min_noisy_denominator=0.0)

        s_raw, n_raw, r_raw = profile.compute_raw_exact(user_matrix)
        assert len(s_raw) == d_cells
        assert len(n_raw) == d_cells
        assert len(r_raw) == d_cells

        s_clip, n_clip, r_clip = profile.compute_clipped_exact(user_matrix)
        assert len(s_clip) == d_cells
        assert len(n_clip) == d_cells
        assert len(r_clip) == d_cells

        # evaluate_utility with DP vector of joint dimension 24
        dp_joint = np.ones(d_cells * 2, dtype=np.float64) * 5.0
        util = profile.evaluate_utility(dp_joint, user_matrix)
        assert "mae_vs_raw" in util
        assert "suppression_rate" in util
        assert util["num_valid_cells"] == d_cells


class TestReleaseKeyParameterSensitivity:
    """Audit Item 3: ReleaseKey binds all semantic parameters."""

    def test_release_key_varies_with_min_noisy_denominator(self):
        config1 = load_app_config("configs")
        config1.metrics.metrics["success_rate"].contribution.min_noisy_denominator = 1.0
        p1 = AnalyticsPlanner(config1)

        config2 = load_app_config("configs")
        config2.metrics.metrics["success_rate"].contribution.min_noisy_denominator = 5.0
        p2 = AnalyticsPlanner(config2)

        q = AnalyticsQuery(metric="success_rate", recipient="internal", epsilon=1.0, delta=1e-6)
        plan1 = p1.create_plan(q, dataset_version="v1.0")
        plan2 = p2.create_plan(q, dataset_version="v1.0")

        assert plan1.release_key != plan2.release_key
        assert plan1.canonical_semantics["min_noisy_denominator"] == 1.0
        assert plan2.canonical_semantics["min_noisy_denominator"] == 5.0

    def test_release_key_varies_with_top_k_parameters(self):
        config1 = load_app_config("configs")
        config1.metrics.metrics["top_k_tasks"].contribution.max_tasks_per_user = 3
        p1 = AnalyticsPlanner(config1)

        config2 = load_app_config("configs")
        config2.metrics.metrics["top_k_tasks"].contribution.max_tasks_per_user = 10
        p2 = AnalyticsPlanner(config2)

        q = AnalyticsQuery(metric="top_k_tasks", recipient="internal", epsilon=1.0, delta=1e-6)
        plan1 = p1.create_plan(q, dataset_version="v1.0")
        plan2 = p2.create_plan(q, dataset_version="v1.0")

        assert plan1.release_key != plan2.release_key
        assert plan1.canonical_semantics["max_tasks_per_user"] == 3
        assert plan2.canonical_semantics["max_tasks_per_user"] == 10

    def test_release_key_deterministic_stable_replay(self):
        config = load_app_config("configs")
        planner = AnalyticsPlanner(config)
        q = AnalyticsQuery(metric="interaction_volume", recipient="internal", epsilon=1.0, delta=1e-6)

        plan_a = planner.create_plan(q, dataset_version="v1.0")
        plan_b = planner.create_plan(q, dataset_version="v1.0")

        assert plan_a.release_key == plan_b.release_key
        assert plan_a.canonical_semantics == plan_b.canonical_semantics


class TestTopKTasksIntegration:
    """Audit Item 6: Top-K Tasks end-to-end execution through ReleaseOrchestrator."""

    def test_top_k_orchestrator_execution(self, tmp_path):
        from private_llm_analytics.privacy.state import PrivacyStateManager

        config = load_app_config("configs")
        sm = PrivacyStateManager(tmp_path / "top_k_state.db")
        orch = ReleaseOrchestrator(app_config=config, state_manager=sm, execution_mode=ExecutionMode.EXPERIMENT)

        df = pd.DataFrame({
            "user_id": ["u1", "u1", "u2", "u3", "u4", "u2"],
            "timestamp": ["2026-01-01 10:00:00"] * 6,
            "provider": ["provider_a", "provider_b", "provider_a", "provider_c", "provider_b", "provider_a"],
            "task": ["technical", "analysis", "technical", "document", "technical", "analysis"],
            "success": [True, True, True, True, True, False],
        })
        umatrix = build_user_matrix(
            df,
            providers=config.scenario.providers,
            tasks=config.scenario.tasks,
            public_population_size=10,
        )

        query = AnalyticsQuery(
            metric="top_k_tasks",
            recipient="internal",
            epsilon=1.0,
            delta=1e-6,
        )

        res = orch.execute_release(query, umatrix, dataset_version="v1.0", include_exact=True)
        assert res["metric"] == "top_k_tasks"
        assert "results" in res
        assert "top_k" in res["results"]
        assert len(res["results"]["top_k"]) <= 5
        assert "exact_ground_truth" in res


class TestContinualRNGAuditFixes:
    """Audit Item 5: Continual Observation Generator isolation and caller seed restrictions."""

    def test_continual_production_rejects_caller_rng(self, tmp_path):
        from private_llm_analytics.continual.models import ContinualStreamConfig
        from private_llm_analytics.continual.query import ContinualQueryEngine
        from private_llm_analytics.continual.state import ContinualStateManager

        config = load_app_config("configs")
        config.privacy.continual.enabled = True
        sm = ContinualStateManager(tmp_path / "continual_test.db")
        engine = ContinualQueryEngine.from_app_config(config, state_manager=sm)
        engine.execution_mode = "production"

        stream_cfg = ContinualStreamConfig(stream_id="test_stream", epoch_days=30)
        caller_rng = np.random.default_rng(42)

        with pytest.raises(ProductionRNGControlError, match="External caller cannot supply or control RNG"):
            engine.query_interval(
                stream_config=stream_cfg,
                start_day=1,
                end_day=7,
                recipient="internal",
                rng=caller_rng,
            )

    def test_continual_node_generation_uses_generator_determinism(self, tmp_path):
        from private_llm_analytics.continual.models import ContinualStreamConfig
        from private_llm_analytics.continual.state import ContinualStateManager

        sm1 = ContinualStateManager(tmp_path / "s1.db")
        sm2 = ContinualStateManager(tmp_path / "s2.db")

        stream_cfg = ContinualStreamConfig(stream_id="det_stream", epoch_days=30)

        rng1 = np.random.default_rng(999)
        rng2 = np.random.default_rng(999)

        exact_val = np.array([100.0, 200.0], dtype=np.float64)

        node1 = sm1.get_or_create_noisy_node(
            stream_config=stream_cfg,
            level=0,
            node_index=0,
            sigma=5.0,
            materialize_fn=lambda: exact_val,
            finalized_up_to_day=30,
            manifest_fingerprint="m_fp",
            public_roster_fingerprint="r_fp",
            rng=rng1,
        )

        node2 = sm2.get_or_create_noisy_node(
            stream_config=stream_cfg,
            level=0,
            node_index=0,
            sigma=5.0,
            materialize_fn=lambda: exact_val,
            finalized_up_to_day=30,
            manifest_fingerprint="m_fp",
            public_roster_fingerprint="r_fp",
            rng=rng2,
        )

        assert np.allclose(node1, node2)

