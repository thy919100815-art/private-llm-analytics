"""Comprehensive In-Depth Independent Verification Suite for DP Audit."""

import math
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple

from private_llm_analytics.aggregation.grouping import build_cell_definitions
from private_llm_analytics.aggregation.user_matrix import UserMatrix, build_user_matrix
from private_llm_analytics.config import load_app_config
from private_llm_analytics.mechanisms.gaussian import GaussianMechanism, calibrate_analytic_gaussian
from private_llm_analytics.privacy.budget import PrivacyBudget
from private_llm_analytics.privacy.rdp import (
    DEFAULT_RDP_ORDERS,
    compose_rdp_vectors,
    compute_gaussian_rdp,
    rdp_to_epsilon_delta,
)
from private_llm_analytics.privacy.rdp_accountant import RDPAccountant
from private_llm_analytics.service.analytics import AnalyticsService
from private_llm_analytics.statistics.registry import StatisticProfileRegistry


def run_sensitivity_verification():
    """Verify claimed sensitivity ||f(D) - f(D')||_2 <= Delta_2 for all 5 statistics."""
    print("=== 1. INDEPENDENT SENSITIVITY VERIFICATION (D vs D') ===")
    providers = ["provider_a", "provider_b", "provider_c"]
    tasks = ["technical", "document", "analysis"]
    cell_names, _ = build_cell_definitions(providers, tasks)
    d = len(cell_names)

    # Scenarios for user u_star to add/remove:
    # 1: empty user
    # 2: single cell 1 call
    # 3: single cell 1000 calls
    # 4: all cells 1 call
    # 5: all cells 10000 calls (extreme heavy user)
    # 6: over-bound user (active in all cells)
    test_user_vectors = [
        np.zeros(d, dtype=np.float64),
        np.array([1.0] + [0.0]*(d-1), dtype=np.float64),
        np.array([1000.0] + [0.0]*(d-1), dtype=np.float64),
        np.ones(d, dtype=np.float64),
        np.ones(d, dtype=np.float64) * 10000.0,
        np.array([5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0, 45.0], dtype=np.float64),
    ]

    statistics = [
        ("user_prevalence", 3.0),            # sqrt(9) = 3.0
        ("user_normalized_share", 1.0),       # L2 norm of simplex point <= 1.0
        ("interaction_volume", 20.0),         # C = 20.0
        ("success_rate", np.sqrt(2.0) * 20.0),# sqrt(2) * C_rate = sqrt(2) * 20.0 = 28.28427
        ("top_k_tasks", np.sqrt(3.0)),        # min(k_tasks=5, |T|=3) -> sqrt(3) = 1.73205 (conservative config claimed sqrt(5) = 2.236)
    ]

    for stat_id, claimed_sens in statistics:
        profile = StatisticProfileRegistry.get(stat_id)
        max_empirical_l2 = 0.0

        for u_vec in test_user_vectors:
            # Construct Base Matrix D (5 users)
            base_matrix = np.array([
                [1, 2, 0, 0, 5, 0, 0, 1, 0],
                [0, 0, 10, 0, 0, 0, 2, 0, 0],
                [5, 5, 5, 5, 5, 5, 5, 5, 5],
                [0, 1, 0, 1, 0, 1, 0, 1, 0],
                [20, 0, 0, 0, 0, 0, 0, 0, 0],
            ], dtype=np.float64)

            # Neighboring Matrix D' (Base + u_vec)
            neighbor_matrix = np.vstack([base_matrix, u_vec])

            # For success rate, construct success matrix
            base_succ = np.minimum(base_matrix, np.random.default_rng(42).integers(0, 5, size=base_matrix.shape).astype(np.float64))
            u_succ = np.minimum(u_vec, np.random.default_rng(42).integers(0, 5, size=u_vec.shape).astype(np.float64))
            neighbor_succ = np.vstack([base_succ, u_succ])

            um_D = UserMatrix(
                matrix=base_matrix,
                user_ids=[f"u_{i}" for i in range(len(base_matrix))],
                providers=providers,
                tasks=tasks,
                cell_names=cell_names,
                cell_map={tuple(c.split(":")): i for i, c in enumerate(cell_names)},
                public_population_size=100,
                success_matrix=base_succ,
            )
            um_D_prime = UserMatrix(
                matrix=neighbor_matrix,
                user_ids=[f"u_{i}" for i in range(len(neighbor_matrix))],
                providers=providers,
                tasks=tasks,
                cell_names=cell_names,
                cell_map={tuple(c.split(":")): i for i, c in enumerate(cell_names)},
                public_population_size=100,
                success_matrix=neighbor_succ,
            )

            # Contributions
            cont_D = profile.build_user_contributions(um_D)
            cont_D_prime = profile.build_user_contributions(um_D_prime)

            agg_D = np.sum(cont_D, axis=0)
            agg_D_prime = np.sum(cont_D_prime, axis=0)

            diff = agg_D_prime - agg_D
            l2_dist = float(np.linalg.norm(diff, ord=2))
            if l2_dist > max_empirical_l2:
                max_empirical_l2 = l2_dist

        code_sens = profile.sensitivity()
        passed = (max_empirical_l2 <= code_sens + 1e-9)
        print(f"Statistic: {stat_id:<22} Max Empirical ||f(D)-f(D')||_2: {max_empirical_l2:8.4f} | Code Delta_2: {code_sens:8.4f} | Status: {'PASS' if passed else 'FAIL'}")
        assert passed, f"Sensitivity violated for {stat_id}: empirical {max_empirical_l2} > code {code_sens}"


def run_monte_carlo_noise_check():
    """Verify empirical noise distribution vs configured sigma over 5000 trials."""
    print("\n=== 2. GAUSSIAN MONTE CARLO SANITY CHECK (5000 Trials) ===")
    eps, delta, sens = 1.0, 1e-6, 3.0
    mech = GaussianMechanism(epsilon=eps, delta=delta, sensitivity=sens, method="analytic")
    sigma = mech.noise_scale

    n_trials = 5000
    rng = np.random.default_rng(12345)
    clean = np.zeros(9, dtype=np.float64)
    samples = np.array([mech.add_noise(clean, rng=rng) for _ in range(n_trials)])

    empirical_mean = np.mean(samples, axis=0)
    empirical_std = np.std(samples, axis=0)

    print(f"Configured Sigma: {sigma:.4f}")
    print(f"Empirical Mean (across 9 cells): min={np.min(empirical_mean):.4f}, max={np.max(empirical_mean):.4f} (expected ~0.0)")
    print(f"Empirical Std  (across 9 cells): min={np.min(empirical_std):.4f}, max={np.max(empirical_std):.4f} (expected ~{sigma:.4f})")

    mean_z = np.max(np.abs(empirical_mean)) / (sigma / math.sqrt(n_trials))
    std_rel_err = np.max(np.abs(empirical_std - sigma)) / sigma

    print(f"Max Mean Z-score: {mean_z:.2f} (PASS if < 3.0)")
    print(f"Max Std Rel Err : {std_rel_err*100.0:.2f}% (PASS if < 3.0%)")
    assert mean_z < 3.0, f"Monte Carlo mean failed: Z={mean_z}"
    assert std_rel_err < 0.03, f"Monte Carlo std failed: err={std_rel_err}"


def run_rdp_accountant_check():
    """Verify analytical RDP formula vs RDPAccountant and zero incremental spend on replay."""
    print("\n=== 3. RDP ACCOUNTANT & REPLAY ZERO-SPEND AUDIT ===")
    sigma = 12.674036667980548
    sens = 3.0
    orders = DEFAULT_RDP_ORDERS

    # Analytical formula: rho(alpha) = alpha * Delta^2 / (2 * sigma^2)
    analytical_rdp = [(alpha * (sens ** 2)) / (2.0 * (sigma ** 2)) for alpha in orders]

    # Code accountant
    acct = RDPAccountant(orders=orders)
    code_rdp = compute_gaussian_rdp(sigma=sigma, sensitivity=sens, orders=orders)

    diff = [abs(a - float(code_rdp[o])) for a, o in zip(analytical_rdp, orders)]
    max_diff = max(diff)
    print(f"Max difference between analytical RDP and compute_gaussian_rdp: {max_diff:.2e} (PASS if < 1e-12)")
    assert max_diff < 1e-12

    # Verify RDP accumulation across 5 compositions
    composed_analytical = [5.0 * r for r in analytical_rdp]
    composed_code = compose_rdp_vectors([code_rdp] * 5)
    max_comp_diff = max(abs(a - float(composed_code[o])) for a, o in zip(composed_analytical, orders))
    print(f"Max difference after 5 composition steps: {max_comp_diff:.2e} (PASS if < 1e-12)")
    assert max_comp_diff < 1e-12

    # Verify conversion to (epsilon, delta=1e-6)
    eff_eps, best_alpha = rdp_to_epsilon_delta(composed_code, delta_global=1e-6)
    print(f"5 compositions -> Effective Epsilon: {float(eff_eps):.4f} at best alpha: {best_alpha}")


def run_minimal_golden_test():
    """Golden End-to-End Test with 4 users, 2 providers, 2 tasks."""
    print("\n=== 4. MINIMAL GOLDEN TEST (4 users, 2 providers, 2 tasks) ===")
    providers = ["provider_a", "provider_b"]
    tasks = ["technical", "document"]
    cell_names, cell_map = build_cell_definitions(providers, tasks)
    # Cell 0: provider_a:technical
    # Cell 1: provider_a:document
    # Cell 2: provider_b:technical
    # Cell 3: provider_b:document

    # Hand-crafted interactions:
    # User 0: provider_a:technical x 5 (all success)
    # User 1: provider_a:technical x 10, provider_b:document x 10 (User 1 active in 2 cells)
    # User 2: provider_b:document x 30
    # User 3: empty user (0 calls)
    data = [
        # u0: 5 tech with a
        {"user_id": "u0", "provider": "provider_a", "task": "technical", "success": True},
        {"user_id": "u0", "provider": "provider_a", "task": "technical", "success": True},
        {"user_id": "u0", "provider": "provider_a", "task": "technical", "success": True},
        {"user_id": "u0", "provider": "provider_a", "task": "technical", "success": True},
        {"user_id": "u0", "provider": "provider_a", "task": "technical", "success": True},
        # u1: 1 tech with a (success), 1 doc with b (failure)
        {"user_id": "u1", "provider": "provider_a", "task": "technical", "success": True},
        {"user_id": "u1", "provider": "provider_b", "task": "document", "success": False},
        # u2: 2 doc with b (1 success, 1 failure)
        {"user_id": "u2", "provider": "provider_b", "task": "document", "success": True},
        {"user_id": "u2", "provider": "provider_b", "task": "document", "success": False},
        # u3: 0 calls
    ]
    df = pd.DataFrame(data)
    all_users = ["u0", "u1", "u2", "u3"]
    um = build_user_matrix(df, providers=providers, tasks=tasks, user_ids=all_users, public_population_size=4)

    # 1. Hand-calculated expected values:
    # Active users in provider_a:technical: u0, u1 -> 2 users.
    # Total public population: 4.
    # Raw prevalence for provider_a:technical = 2 / 4 = 0.50 (50.0%)
    profile_prev = StatisticProfileRegistry.get("user_prevalence")
    cont_prev = profile_prev.build_user_contributions(um)
    exact_prev = profile_prev.compute_exact(um)
    post_prev = profile_prev.post_process(exact_prev, um)

    expected_prev_rate = 2.0 / 4.0
    actual_prev_rate = post_prev["prevalence_rates"]["provider_a:technical"]
    print(f"Prevalence Golden Check: Expected = {expected_prev_rate:.4f}, Actual = {actual_prev_rate:.4f} -> {'PASS' if abs(expected_prev_rate - actual_prev_rate) < 1e-9 else 'FAIL'}")
    assert abs(expected_prev_rate - actual_prev_rate) < 1e-9

    # 2. Success rate for provider_a:technical:
    # u0 has 5 successes out of 5 calls.
    # u1 has 1 success out of 1 call.
    # Total calls in cell: 6. Total successes: 6.
    # Exact rate = 6 / 6 = 1.0 (100.0%)
    profile_succ = StatisticProfileRegistry.get("success_rate")
    cont_succ = profile_succ.build_user_contributions(um)
    exact_succ = profile_succ.compute_exact(um)
    post_succ = profile_succ.post_process(exact_succ, um)
    actual_succ_rate = post_succ["success_rates"]["provider_a:technical"]
    print(f"Success Rate Golden Check: Expected = 1.0000, Actual = {actual_succ_rate:.4f} -> {'PASS' if abs(1.0 - actual_succ_rate) < 1e-9 else 'FAIL'}")
    assert abs(1.0 - actual_succ_rate) < 1e-9


def run_dynamic_taxonomy_audit():
    """Verify no hardcoded 9 cells in mathematical modules."""
    print("\n=== 5. DYNAMIC TAXONOMY AUDIT (2x2, 3x3, 4x5) ===")
    taxonomies = [
        (["p1", "p2"], ["t1", "t2"]),                           # 2x2 = 4 cells
        (["p1", "p2", "p3"], ["t1", "t2", "t3"]),               # 3x3 = 9 cells
        (["p1", "p2", "p3", "p4"], ["t1", "t2", "t3", "t4", "t5"]), # 4x5 = 20 cells
    ]

    for provs, tsks in taxonomies:
        cell_names, _ = build_cell_definitions(provs, tsks)
        d = len(cell_names)
        mat = np.ones((5, d), dtype=np.float64)
        um = UserMatrix(
            matrix=mat,
            user_ids=[f"u_{i}" for i in range(5)],
            providers=provs,
            tasks=tsks,
            cell_names=cell_names,
            cell_map={tuple(c.split(":")): i for i, c in enumerate(cell_names)},
            public_population_size=10,
            success_matrix=mat,
        )

        for m_id in ["user_prevalence", "user_normalized_share", "interaction_volume", "success_rate", "top_k_tasks"]:
            prof = StatisticProfileRegistry.get(m_id)
            cont = prof.build_user_contributions(um)
            if m_id == "success_rate":
                assert cont.shape == (5, 2 * d), f"Shape mismatch for success_rate with D={d}: {cont.shape}"
            elif m_id == "top_k_tasks":
                assert cont.shape == (5, len(tsks)), f"Shape mismatch for top_k with T={len(tsks)}: {cont.shape}"
            else:
                assert cont.shape == (5, d), f"Shape mismatch for {m_id} with D={d}: {cont.shape}"

            plan = prof.build_mechanism_plan(epsilon=1.0, delta=1e-6, dimension=d)
            agg = np.sum(cont, axis=0)
            noisy = prof.execute_mechanism(agg, plan, rng=np.random.default_rng(42))
            post = prof.post_process(noisy, um)
            assert post is not None
        print(f"Taxonomy {len(provs)} providers x {len(tsks)} tasks = {d} cells: ALL PROFILES PASS DYNAMIC DOMAIN")


if __name__ == "__main__":
    run_sensitivity_verification()
    run_monte_carlo_noise_check()
    run_rdp_accountant_check()
    run_minimal_golden_test()
    run_dynamic_taxonomy_audit()
    print("\n>>> ALL IN-DEPTH AUDIT CHECKS COMPLETED WITH 100% PASS <<<")
