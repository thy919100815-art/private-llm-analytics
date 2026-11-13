"""Exp22: Dynamic Domain Scaling and Utility Across Arbitrary Taxonomies (v0.7).

Evaluates sensitivity, runtime, and empirical utility across scaling taxonomy dimensions:
- 3 x 3 (9 cells)
- 5 x 10 (50 cells)
- 10 x 20 (200 cells)
- 10 x 50 (500 cells)
"""

from __future__ import annotations
import json
from pathlib import Path
import time
from typing import Dict, List, Tuple
import numpy as np

from private_llm_analytics.aggregation.grouping import build_cell_definitions
from private_llm_analytics.aggregation.user_matrix import UserMatrix
from private_llm_analytics.statistics.interaction_volume import InteractionVolumeProfile
from private_llm_analytics.statistics.normalized_share import UserNormalizedShareProfile
from private_llm_analytics.statistics.prevalence import UserPrevalenceProfile
from private_llm_analytics.statistics.success_rate import SuccessRateProfile


def generate_synthetic_user_matrix(
    n_providers: int,
    n_tasks: int,
    n_users: int = 200,
    public_pop: int = 1000,
    seed: int = 42,
) -> UserMatrix:
    providers = [f"provider_{i}" for i in range(n_providers)]
    tasks = [f"task_{j}" for j in range(n_tasks)]
    cell_names, cell_map = build_cell_definitions(providers, tasks)
    d_cells = len(cell_names)

    rng = np.random.default_rng(seed)
    # Poisson distributed interaction counts with sparsity
    counts = rng.poisson(lam=3.0, size=(n_users, d_cells)).astype(np.float64)
    counts[counts < 2.0] = 0.0

    # Success counts bounded by counts
    success = np.zeros_like(counts)
    nonzero = counts > 0
    success[nonzero] = rng.binomial(n=counts[nonzero].astype(int), p=0.85).astype(np.float64)

    return UserMatrix(
        matrix=counts,
        user_ids=[f"u_{i}" for i in range(n_users)],
        providers=providers,
        tasks=tasks,
        cell_names=cell_names,
        cell_map=cell_map,
        public_population_size=public_pop,
        success_matrix=success,
    )


def run_exp22_scaling() -> dict:
    domain_configs: List[Tuple[int, int]] = [
        (3, 3),    # 9 cells
        (5, 10),   # 50 cells
        (10, 20),  # 200 cells
        (10, 50),  # 500 cells
    ]

    results: Dict[str, dict] = {}
    epsilon = 1.0
    delta = 1e-6

    print("============================================================")
    print("Exp22: Dynamic Domain Scaling Across Arbitrary Taxonomies")
    print("============================================================")

    for p_count, t_count in domain_configs:
        d_cells = p_count * t_count
        label = f"{p_count}x{t_count}_{d_cells}cells"
        umatrix = generate_synthetic_user_matrix(p_count, t_count)

        domain_res = {
            "providers": p_count,
            "tasks": t_count,
            "total_cells": d_cells,
            "statistics": {},
        }

        # 1. User Prevalence Profile
        t0 = time.perf_counter()
        prev_profile = UserPrevalenceProfile(max_cells_per_user=d_cells)
        prev_rel = prev_profile.execute_release(umatrix, epsilon=epsilon, delta=delta)
        t_prev = time.perf_counter() - t0
        true_prev = np.sum((umatrix.matrix > 0).astype(float), axis=0) / umatrix.public_population_size
        dp_prev = prev_rel.post_processed["prevalence_rates_array"]
        prev_mae = float(np.mean(np.abs(dp_prev - true_prev)))

        domain_res["statistics"]["user_prevalence"] = {
            "sensitivity": float(prev_profile.sensitivity()),
            "runtime_ms": t_prev * 1000.0,
            "mean_absolute_error": prev_mae,
        }

        # 2. User-Normalized Share Profile
        t0 = time.perf_counter()
        share_profile = UserNormalizedShareProfile()
        share_rel = share_profile.execute_release(umatrix, epsilon=epsilon, delta=delta)
        t_share = time.perf_counter() - t0
        user_sums = np.sum(umatrix.matrix, axis=1, keepdims=True)
        norm_mat = np.divide(umatrix.matrix, user_sums, out=np.zeros_like(umatrix.matrix), where=(user_sums > 0))
        true_share_raw = np.sum(norm_mat, axis=0)
        true_share = true_share_raw / max(1e-12, float(np.sum(true_share_raw)))
        dp_share = share_rel.post_processed["share_distribution_array"]
        share_mae = float(np.mean(np.abs(dp_share - true_share)))

        domain_res["statistics"]["user_normalized_share"] = {
            "sensitivity": float(share_profile.sensitivity()),
            "runtime_ms": t_share * 1000.0,
            "mean_absolute_error": share_mae,
        }

        # 3. Interaction Volume Profile
        t0 = time.perf_counter()
        vol_profile = InteractionVolumeProfile(clip_norm=20.0)
        vol_rel = vol_profile.execute_release(umatrix, epsilon=epsilon, delta=delta)
        t_vol = time.perf_counter() - t0
        true_vol = np.sum(umatrix.matrix, axis=0)
        dp_vol = vol_rel.post_processed["clamped_volume_array"]
        vol_mae = float(np.mean(np.abs(dp_vol - true_vol)))

        domain_res["statistics"]["interaction_volume"] = {
            "sensitivity": float(vol_profile.sensitivity()),
            "runtime_ms": t_vol * 1000.0,
            "mean_absolute_error": vol_mae,
        }

        # 4. Success Rate Profile
        t0 = time.perf_counter()
        sr_profile = SuccessRateProfile(clip_norm=10.0, min_noisy_denominator=1.0)
        sr_rel = sr_profile.execute_release(umatrix, epsilon=epsilon, delta=delta)
        t_sr = time.perf_counter() - t0

        domain_res["statistics"]["success_rate"] = {
            "sensitivity": float(sr_profile.sensitivity()),
            "joint_dimension": 2 * d_cells,
            "runtime_ms": t_sr * 1000.0,
        }

        print(f"Domain {label}: Prev MAE={prev_mae:.5f}, Share MAE={share_mae:.5f}, Vol MAE={vol_mae:.2f}, SR JointDim={2*d_cells}")
        results[label] = domain_res

    # Output directory
    out_dir = Path("results")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "exp22_dynamic_domain_scaling.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\n[PASS] Exp22 completed successfully. Saved to {out_file}")
    return results


if __name__ == "__main__":
    run_exp22_scaling()
