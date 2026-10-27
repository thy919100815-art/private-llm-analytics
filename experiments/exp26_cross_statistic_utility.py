"""Exp26: Cross-Statistic Unified DP Utility Benchmark (v0.9).

Evaluates all 5 approved statistic-specific DP release profiles simultaneously on
identical enterprise interaction telemetry across varying privacy budgets epsilon:
1. User Prevalence (user_prevalence)
2. User-Normalized Share (user_normalized_share)
3. Interaction Volume (interaction_volume)
4. Success Rate (success_rate)
5. Top-K Tasks Discovery (top_k_tasks)
"""

from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np

from private_llm_analytics.aggregation.grouping import build_cell_definitions
from private_llm_analytics.aggregation.user_matrix import UserMatrix
from private_llm_analytics.statistics.interaction_volume import InteractionVolumeProfile
from private_llm_analytics.statistics.normalized_share import UserNormalizedShareProfile
from private_llm_analytics.statistics.prevalence import UserPrevalenceProfile
from private_llm_analytics.statistics.success_rate import SuccessRateProfile
from private_llm_analytics.statistics.top_k_tasks import TopKTasksProfile


def generate_enterprise_benchmark_dataset(
    n_users: int = 500,
    n_providers: int = 4,
    n_tasks: int = 10,
    public_pop: int = 1000,
    seed: int = 20261027,
) -> UserMatrix:
    rng = np.random.default_rng(seed)
    providers = [f"provider_{p}" for p in range(n_providers)]
    tasks = [f"task_{t}" for t in range(n_tasks)]
    cell_names, cell_map = build_cell_definitions(providers, tasks)
    d_cells = len(cell_names)

    counts = rng.poisson(lam=5.0, size=(n_users, d_cells)).astype(np.float64)
    # 60% sparsity
    counts[rng.random(size=counts.shape) < 0.6] = 0.0

    success = np.zeros_like(counts)
    nonzero = counts > 0
    # True success probability ~ 80%
    success[nonzero] = rng.binomial(n=counts[nonzero].astype(int), p=0.80).astype(np.float64)

    return UserMatrix(
        matrix=counts,
        user_ids=[f"user_{u}" for u in range(n_users)],
        providers=providers,
        tasks=tasks,
        cell_names=cell_names,
        cell_map=cell_map,
        public_population_size=public_pop,
        success_matrix=success,
    )


def run_exp26_cross_statistic_utility() -> dict:
    epsilons = [0.25, 0.5, 1.0, 2.0]
    delta = 1e-6
    umatrix = generate_enterprise_benchmark_dataset()

    print("============================================================")
    print("Exp26: Cross-Statistic Unified DP Utility Benchmark")
    print("============================================================")

    # Initialize all 5 profiles
    prev_prof = UserPrevalenceProfile(max_cells_per_user=umatrix.n_cells)
    share_prof = UserNormalizedShareProfile()
    vol_prof = InteractionVolumeProfile(clip_norm=20.0)
    sr_prof = SuccessRateProfile(clip_norm=10.0, min_noisy_denominator=1.0)
    topk_prof = TopKTasksProfile(k=5, max_tasks_per_user=5)

    results: Dict[str, dict] = {}

    for eps in epsilons:
        print(f"\n--- Evaluating Privacy Budget: epsilon={eps:.2f}, delta={delta} ---")
        eps_res = {}

        # 1. User Prevalence
        rel_prev = prev_prof.execute_release(umatrix, epsilon=eps, delta=delta)
        true_prev = np.sum((umatrix.matrix > 0).astype(float), axis=0) / umatrix.public_population_size
        dp_prev = rel_prev.post_processed["prevalence_rates_array"]
        mae_prev = float(np.mean(np.abs(dp_prev - true_prev)))
        eps_res["user_prevalence"] = {
            "sensitivity": float(prev_prof.sensitivity()),
            "noise_scale": float(rel_prev.noise_scale),
            "mae": mae_prev,
        }
        print(f"  Prevalence:       MAE={mae_prev:.5f}, sigma={rel_prev.noise_scale:.3f}")

        # 2. User-Normalized Share
        rel_share = share_prof.execute_release(umatrix, epsilon=eps, delta=delta)
        user_sums = np.sum(umatrix.matrix, axis=1, keepdims=True)
        norm_mat = np.divide(umatrix.matrix, user_sums, out=np.zeros_like(umatrix.matrix), where=(user_sums > 0))
        true_share_raw = np.sum(norm_mat, axis=0)
        true_share = true_share_raw / max(1e-12, float(np.sum(true_share_raw)))
        dp_share = rel_share.post_processed["share_distribution_array"]
        mae_share = float(np.mean(np.abs(dp_share - true_share)))
        eps_res["user_normalized_share"] = {
            "sensitivity": float(share_prof.sensitivity()),
            "noise_scale": float(rel_share.noise_scale),
            "mae": mae_share,
        }
        print(f"  Normalized Share: MAE={mae_share:.5f}, sigma={rel_share.noise_scale:.3f}")

        # 3. Interaction Volume
        rel_vol = vol_prof.execute_release(umatrix, epsilon=eps, delta=delta)
        true_vol = np.sum(umatrix.matrix, axis=0)
        dp_vol = rel_vol.post_processed["clamped_volume_array"]
        mae_vol = float(np.mean(np.abs(dp_vol - true_vol)))
        eps_res["interaction_volume"] = {
            "sensitivity": float(vol_prof.sensitivity()),
            "noise_scale": float(rel_vol.noise_scale),
            "mae": mae_vol,
        }
        print(f"  Volume:           MAE={mae_vol:.2f}, sigma={rel_vol.noise_scale:.3f}")

        # 4. Success Rate
        rel_sr = sr_prof.execute_release(umatrix, epsilon=eps, delta=delta)
        eps_res["success_rate"] = {
            "sensitivity": float(sr_prof.sensitivity()),
            "noise_scale": float(rel_sr.noise_scale),
            "joint_dim": len(rel_sr.dp_vector),
        }
        print(f"  Success Rate:     JointDim={len(rel_sr.dp_vector)}, sigma={rel_sr.noise_scale:.3f}")

        # 5. Top-K Tasks
        rel_topk = topk_prof.execute_release(umatrix, epsilon=eps, delta=delta)
        eps_res["top_k_tasks"] = {
            "sensitivity": float(topk_prof.sensitivity()),
            "noise_scale": float(rel_topk.noise_scale),
            "selected_top_k": rel_topk.post_processed["selected_tasks"],
        }
        print(f"  Top-K Tasks:      Selected={rel_topk.post_processed['selected_tasks']}, sigma={rel_topk.noise_scale:.3f}")

        results[f"epsilon_{eps}"] = eps_res

    out_dir = Path("results")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "exp26_cross_statistic_utility.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\n[PASS] Exp26 completed successfully. Saved to {out_file}")
    return results


if __name__ == "__main__":
    run_exp26_cross_statistic_utility()
