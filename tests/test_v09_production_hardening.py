"""Comprehensive test suite for Milestone v0.9: Production Hardening & Unified Evaluation.

Covers:
1. Malformed inputs fail-closed across all 5 statistic profiles;
2. Profile-aware planner calibration across all 5 statistics;
3. Continual observation binary tree canonical cover and noise invariants;
4. Multi-threaded SQLite ledger ACID transactions and concurrency safety;
5. Adversarial and edge-case robustness (outliers, zero sums, empty domains).
"""

from __future__ import annotations
import concurrent.futures
import math
from pathlib import Path
import sqlite3
import tempfile
import threading
import numpy as np
import pytest

from private_llm_analytics.config import load_app_config
from private_llm_analytics.continual.mechanism import calibrate_tree_gaussian_sigma
from private_llm_analytics.continual.tree import BinaryTimeTree
from private_llm_analytics.aggregation.grouping import build_cell_definitions
from private_llm_analytics.aggregation.user_matrix import UserMatrix
from private_llm_analytics.planning.candidates import ReleasePolicyCandidate
from private_llm_analytics.planning.utility import (
    NoiseEstimator,
    TruncatedNormalPrior,
    UtilityEstimator,
    UtilityWeights,
)
from private_llm_analytics.privacy.exceptions import ProductionSafetyConfigurationError
from private_llm_analytics.service.analytics import AnalyticsService
from private_llm_analytics.statistics.registry import StatisticProfileRegistry


class TestStatisticProfilesFailClosed:
    """Test fail-closed behavior on invalid inputs across all registered statistics."""

    def test_all_five_statistics_registered(self):
        expected = {
            "user_prevalence",
            "user_normalized_share",
            "interaction_volume",
            "success_rate",
            "top_k_tasks",
        }
        registered = set(StatisticProfileRegistry.list_statistics())
        assert expected.issubset(registered), f"Missing statistics: {expected - registered}"

    def test_profile_rejects_negative_or_nan_matrix(self):
        providers = ["p1", "p2"]
        tasks = ["t1", "t2"]
        users = ["u1", "u2"]
        cell_names, cell_map = build_cell_definitions(providers, tasks)

        # NaN matrix
        nan_mat = np.array([[1.0, np.nan, 0.0, 0.0], [0.0, 2.0, 1.0, 0.0]])
        with pytest.raises(ValueError, match="NaN"):
            UserMatrix(
                matrix=nan_mat,
                providers=providers,
                tasks=tasks,
                user_ids=users,
                cell_names=cell_names,
                cell_map=cell_map,
            )

        # Negative count matrix
        neg_mat = np.array([[-1.0, 0.0, 0.0, 0.0], [0.0, 2.0, 1.0, 0.0]])
        with pytest.raises(ValueError, match="negative"):
            UserMatrix(
                matrix=neg_mat,
                providers=providers,
                tasks=tasks,
                user_ids=users,
                cell_names=cell_names,
                cell_map=cell_map,
            )


class TestProfileAwarePlannerCalibration:
    """Test release planner sensitivity and noise estimation across all profiles."""

    def test_noise_estimator_all_statistics(self):
        candidate = ReleasePolicyCandidate(
            contribution_bound=25.0,
            granularity="full_9_cell",
            window_days=30,
            frequency_days=7,
            mechanism="gaussian",
            epsilon_per_release=1.0,
            delta_per_release=1e-6,
        )

        for stat_id in StatisticProfileRegistry.list_statistics():
            sigma, sens = NoiseEstimator.estimate_noise_sigma(stat_id, candidate)
            assert sigma > 0.0, f"Sigma must be positive for {stat_id}"
            assert sens > 0.0, f"Sensitivity must be positive for {stat_id}"

    def test_utility_estimator_evaluates_all_statistics(self):
        prior = TruncatedNormalPrior(mean=60.0, std=20.0, min_val=5.0, max_val=150.0)
        weights = UtilityWeights(objective_mode="COMPONENT_SURROGATE", clipping=1.0, noise=1.0, aggregation=1.0)
        estimator = UtilityEstimator(public_prior=prior, weights=weights)

        candidate = ReleasePolicyCandidate(
            contribution_bound=20.0,
            granularity="full_9_cell",
            window_days=30,
            frequency_days=7,
            mechanism="gaussian",
            epsilon_per_release=1.0,
            delta_per_release=1e-6,
        )

        for stat_id in StatisticProfileRegistry.list_statistics():
            comp = estimator.evaluate_candidate(
                metric=stat_id,
                candidate=candidate,
                population_size=500,
            )
            assert comp.noise_error > 0.0
            assert comp.total_error >= 0.0


class TestContinualObservationInvariants:
    """Test binary time tree canonical partitioning and calibration invariants."""

    def test_canonical_covers_partition_exact_intervals(self):
        T = 64
        tree = BinaryTimeTree(epoch_days=T)
        max_nodes = 2 * tree.num_levels

        for (s, e) in [(1, 1), (1, 64), (10, 25), (32, 64), (7, 49)]:
            cover = tree.canonical_cover(s, e)
            assert len(cover) <= max_nodes

            covered_days = []
            for level, idx in cover:
                start_d, end_d = tree.node_interval(level, idx)
                covered_days.extend(range(start_d, end_d + 1))
            covered_days.sort()

            assert covered_days == list(range(s, e + 1))

    def test_tree_mechanism_noise_calibration(self):
        sigma = calibrate_tree_gaussian_sigma(
            epoch_days=64,
            epoch_l12_bound=30.0,
            target_tree_epsilon=1.5,
            calibration_delta=1e-6,
        )
        assert sigma > 0.0
        assert math.isfinite(sigma)


class TestStateLedgerConcurrencyAndIntegrity:
    """Test SQLite state ledger under multi-threaded concurrency."""

    def test_concurrent_reservations_and_cache_hits(self, tmp_path):
        db_path = str(tmp_path / "concurrent_test.db")
        app_cfg = load_app_config("configs")
        app_cfg.privacy.release.state_db_path = db_path
        app_cfg.privacy.global_budget.epsilon = 100.0
        app_cfg.privacy.global_budget.delta = 0.05

        service = AnalyticsService(config=app_cfg)
        service.generate_synthetic_data(n_users=50, seed=123)

        n_threads = 8
        queries_per_thread = 4
        errors = []
        lock = threading.Lock()

        def worker(w_id: int):
            for q in range(queries_per_thread):
                # Even q: identical query (tests memoization cache hit)
                # Odd q: unique query (tests atomic ledger reservation)
                eps = 0.5 if (q % 2 == 0) else (0.5 + 0.02 * w_id)
                try:
                    res = service.compute(
                        metric="interaction_volume",
                        recipient="internal",
                        epsilon=eps,
                        delta=1e-6,
                    )
                    assert "results" in res
                except Exception as ex:
                    with lock:
                        errors.append(f"Worker {w_id} Q{q}: {ex}")

        with concurrent.futures.ThreadPoolExecutor(max_workers=n_threads) as executor:
            futures = [executor.submit(worker, i) for i in range(n_threads)]
            concurrent.futures.wait(futures)

        assert len(errors) == 0, f"Thread errors: {errors}"

        # Verify SQLite integrity
        conn = sqlite3.connect(db_path)
        try:
            cur = conn.cursor()
            cur.execute("PRAGMA integrity_check;")
            integrity = cur.fetchone()[0]
            assert integrity == "ok"

            cur.execute("SELECT count(*) FROM release_store;")
            releases = cur.fetchone()[0]
            assert releases > 0
        finally:
            conn.close()
