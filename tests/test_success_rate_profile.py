"""Tests for Success Rate DP Profile (v0.6).

Verifies:
  - Mathematical theorem: Delta_2 <= sqrt(2) * C_rate under shared user clipping
  - Shared clipping factor preserves 0 <= sbar <= nbar and ||nbar||_2 <= C_rate
  - Joint contribution vector has dimension 18 and norm <= sqrt(2) * C_rate
  - Empirical sensitivity check across 2,000 synthetic adjacent datasets
  - Single joint Gaussian release with single RDP charge
  - DP-safe ratio post-processing handles:
      * negative noisy denominator
      * negative noisy numerator
      * noisy success > noisy total
      * denominator below threshold suppression
  - Suppression uses only DP output (no inspection of private raw aggregates)
  - Raw counts never exposed in release results
"""

import pytest
import numpy as np

from private_llm_analytics.aggregation.user_matrix import UserMatrix, build_cell_definitions
from private_llm_analytics.mechanisms.gaussian import GaussianMechanism
from private_llm_analytics.statistics.success_rate import SuccessRateProfile


def make_adjacent_pair(c_rate: float, rng: np.random.Generator):
    """Generate two adjacent user contribution states differing by one arbitrary user."""
    provs = ["p1", "p2", "p3"]
    tsks = ["t1", "t2", "t3"]
    cell_names, cell_map = build_cell_definitions(provs, tsks)

    # Base dataset of 5 users
    n_base = rng.integers(0, 50, size=(5, 9)).astype(np.float64)
    s_base = np.clip(n_base - rng.integers(0, 20, size=(5, 9)), 0, n_base).astype(np.float64)

    # Added single user with arbitrary high interaction counts
    n_user = rng.integers(0, 500, size=(1, 9)).astype(np.float64)
    s_user = np.clip(n_user - rng.integers(0, 500, size=(1, 9)), 0, n_user).astype(np.float64)

    umatrix_d = UserMatrix(
        matrix=n_base,
        user_ids=[f"u_{i}" for i in range(5)],
        providers=provs,
        tasks=tsks,
        cell_names=cell_names,
        cell_map=cell_map,
        public_population_size=100,
        success_matrix=s_base,
    )

    umatrix_dp = UserMatrix(
        matrix=np.vstack([n_base, n_user]),
        user_ids=[f"u_{i}" for i in range(6)],
        providers=provs,
        tasks=tsks,
        cell_names=cell_names,
        cell_map=cell_map,
        public_population_size=100,
        success_matrix=np.vstack([s_base, s_user]),
    )

    return umatrix_d, umatrix_dp


class TestSuccessRateMathematics:
    def test_shared_clipping_properties(self):
        c_rate = 20.0
        prof = SuccessRateProfile(clip_norm=c_rate)

        provs = ["p1", "p2", "p3"]
        tsks = ["t1", "t2", "t3"]
        cell_names, cell_map = build_cell_definitions(provs, tsks)

        # Diverse user interactions: low, medium, extreme heavy
        n_mat = np.array([
            [0.0] * 9,                             # Inactive user
            [2.0, 1.0, 0.0] + [0.0] * 6,           # Small user (norm < C)
            [10.0] * 9,                            # Norm = 30 > C
            [100.0, 200.0, 300.0] + [0.0] * 6,     # Massive user
        ], dtype=np.float64)

        s_mat = np.array([
            [0.0] * 9,
            [1.0, 1.0, 0.0] + [0.0] * 6,
            [8.0] * 9,
            [90.0, 150.0, 250.0] + [0.0] * 6,
        ], dtype=np.float64)

        umatrix = UserMatrix(
            matrix=n_mat,
            user_ids=["u0", "u1", "u2", "u3"],
            providers=provs,
            tasks=tsks,
            cell_names=cell_names,
            cell_map=cell_map,
            public_population_size=10,
            success_matrix=s_mat,
        )

        joint = prof.build_user_contributions(umatrix)
        assert joint.shape == (4, 18)

        sbar = joint[:, :9]
        nbar = joint[:, 9:18]

        # 1. 0 <= sbar <= nbar
        assert np.all(sbar >= 0.0)
        assert np.all(sbar <= nbar + 1e-12)

        # 2. ||nbar_u||_2 <= C_rate
        n_norms = np.linalg.norm(nbar, ord=2, axis=1)
        assert np.all(n_norms <= c_rate + 1e-12)

        # 3. ||w_u||_2 <= sqrt(2) * C_rate
        w_norms = np.linalg.norm(joint, ord=2, axis=1)
        expected_max_norm = np.sqrt(2.0) * c_rate
        assert np.all(w_norms <= expected_max_norm + 1e-12)

    def test_empirical_sensitivity_theorem_verification(self):
        """Empirically test 2,000 adjacent datasets:

        ||f(D) - f(D')||_2 <= sqrt(2) * C_rate MUST hold for every single draw.
        """
        c_rate = 18.5
        prof = SuccessRateProfile(clip_norm=c_rate)
        max_theoretical_sens = np.sqrt(2.0) * c_rate

        rng = np.random.default_rng(2026)
        max_observed_diff = 0.0

        for _ in range(2000):
            u_d, u_dp = make_adjacent_pair(c_rate, rng)
            w_d = np.sum(prof.build_user_contributions(u_d), axis=0)
            w_dp = np.sum(prof.build_user_contributions(u_dp), axis=0)

            diff_l2 = np.linalg.norm(w_dp - w_d, ord=2)
            if diff_l2 > max_observed_diff:
                max_observed_diff = diff_l2

            assert diff_l2 <= max_theoretical_sens + 1e-10, (
                f"Sensitivity theorem violated! Diff {diff_l2} > bound {max_theoretical_sens}"
            )

        # Verify we tested cases that get close to the bound
        assert max_observed_diff > 0.8 * max_theoretical_sens


class TestSuccessRateMechanismAndPostProcessing:
    def test_single_joint_gaussian_plan_and_rdp(self):
        c_rate = 20.0
        prof = SuccessRateProfile(clip_norm=c_rate)
        plan = prof.build_mechanism_plan(epsilon=1.0, delta=1e-6)

        assert plan.dimension == 18
        assert abs(plan.sensitivity - np.sqrt(2.0) * c_rate) < 1e-12
        assert plan.noise_scale > 0

        # Privacy cost descriptor must reflect single joint release
        cost = prof.privacy_cost_descriptor(plan)
        assert cost.accounting_family == "rdp"
        assert cost.sensitivity == plan.sensitivity
        assert cost.noise_scale == plan.noise_scale
        assert len(cost.orders) > 0
        assert len(cost.costs) == len(cost.orders)

    def test_dp_safe_post_processing_edge_cases(self):
        prof = SuccessRateProfile(clip_norm=20.0, min_noisy_denominator=5.0)

        provs = ["p1", "p2", "p3"]
        tsks = ["t1", "t2", "t3"]
        cell_names, cell_map = build_cell_definitions(provs, tsks)
        umatrix = UserMatrix(
            matrix=np.ones((1, 9)),
            user_ids=["u1"],
            providers=provs,
            tasks=tsks,
            cell_names=cell_names,
            cell_map=cell_map,
            public_population_size=10,
            success_matrix=np.ones((1, 9)),
        )

        # Cell 0: Noisy total negative (-2.0), noisy success negative (-5.0) -> total clamped to 0 -> suppressed
        # Cell 1: Noisy total positive but below threshold (3.0 < 5.0) -> suppressed
        # Cell 2: Noisy success > noisy total (success=15.0, total=10.0) -> success clamped to total -> rate = 1.0
        # Cell 3: Noisy success negative (-3.0), noisy total (12.0) -> success clamped to 0.0 -> rate = 0.0
        # Cell 4: Normal cell (success=6.0, total=10.0) -> rate = 0.6
        # Cells 5-8: Zero/normal
        noisy_success = np.array([-5.0, 2.0, 15.0, -3.0, 6.0, 0.0, 10.0, 8.0, 4.0])
        noisy_total = np.array([-2.0, 3.0, 10.0, 12.0, 10.0, 0.0, 20.0, 10.0, 5.0])
        dp_vec = np.hstack([noisy_success, noisy_total])

        res = prof.post_process(dp_vec, umatrix)

        # Cell 0: Suppressed
        assert res["success_rates"][cell_names[0]] is None
        assert "below reporting threshold" in res["cell_status"][cell_names[0]]

        # Cell 1: Suppressed (< min_noisy_denominator 5.0)
        assert res["success_rates"][cell_names[1]] is None
        assert "below reporting threshold" in res["cell_status"][cell_names[1]]

        # Cell 2: Rate clamped to 1.0
        assert abs(res["success_rates"][cell_names[2]] - 1.0) < 1e-12

        # Cell 3: Rate clamped to 0.0
        assert abs(res["success_rates"][cell_names[3]] - 0.0) < 1e-12

        # Cell 4: Rate 0.6
        assert abs(res["success_rates"][cell_names[4]] - 0.6) < 1e-12

        # Verify no raw exact private attributes leaked
        assert "raw_success" not in res
        assert "raw_total" not in res
        assert "exact_users" not in res
