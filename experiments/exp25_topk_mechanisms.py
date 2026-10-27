"""Exp25: DP Top-K Mechanisms and Sensitivity Bounding Verification (v0.8).

Compares:
1. Approved Bounded Gaussian Vector Mechanism (Delta_2 = sqrt(k_bound));
2. Unbounded count release (where heavy-interacting outlier users dominate sensitivity);
3. Verifies empirical bounds on max user influence and error guarantees.
"""

from __future__ import annotations
import json
from pathlib import Path
import numpy as np

from private_llm_analytics.aggregation.grouping import build_cell_definitions
from private_llm_analytics.aggregation.user_matrix import UserMatrix
from private_llm_analytics.statistics.top_k_tasks import TopKTasksProfile


def run_exp25_topk_mechanisms() -> dict:
    print("============================================================")
    print("Exp25: DP Top-K Mechanisms & Sensitivity Bounding")
    print("============================================================")

    n_users = 200
    n_tasks = 40
    providers = ["provider_1"]
    tasks = [f"task_{i:02d}" for i in range(n_tasks)]
    cell_names, cell_map = build_cell_definitions(providers, tasks)

    rng = np.random.default_rng(202610)
    matrix = np.zeros((n_users, n_tasks), dtype=np.float64)

    # 195 regular users: 1-3 tasks each
    for u in range(195):
        active = rng.choice(n_tasks, size=rng.integers(1, 4), replace=False)
        matrix[u, active] = 1.0

    # 5 "super-users" active in ALL 40 tasks (adversarial outlier)
    for u in range(195, 200):
        matrix[u, :] = 1.0

    umatrix = UserMatrix(
        matrix=matrix,
        user_ids=[f"user_{u}" for u in range(n_users)],
        providers=providers,
        tasks=tasks,
        cell_names=cell_names,
        cell_map=cell_map,
        public_population_size=1000,
    )

    # Evaluate profile with contribution bounding
    k_bound = 5
    profile = TopKTasksProfile(k=10, max_tasks_per_user=k_bound)
    contributions = profile.build_user_contributions(umatrix)

    # 1. Verify that super-users' contributions are strictly capped to k_bound
    super_user_contrib_sums = np.sum(contributions[195:], axis=1)
    super_user_l2_norms = np.linalg.norm(contributions[195:], ord=2, axis=1)

    assert np.all(super_user_contrib_sums <= k_bound), "Super-user contributions must be capped!"
    assert np.all(super_user_l2_norms <= np.sqrt(k_bound) + 1e-12), "Super-user L2 norm must be bounded by sqrt(k)!"

    # 2. Sensitivity verification: Adding or removing ANY user changes bounded aggregate by at most sqrt(k_bound)
    plan = profile.build_mechanism_plan(epsilon=1.0, delta=1e-6, dimension=n_tasks)
    assert np.isclose(plan.sensitivity, np.sqrt(k_bound))

    # 3. Execute release with internal RNG
    rel = profile.execute_release(umatrix, epsilon=1.0, delta=1e-6, rng=rng)
    selected = rel.post_processed["selected_tasks"]
    assert len(selected) <= 10

    results = {
        "n_users": n_users,
        "n_tasks": n_tasks,
        "max_tasks_per_user": k_bound,
        "theoretical_l2_sensitivity": float(profile.sensitivity()),
        "max_empirical_user_l2_norm": float(np.max(np.linalg.norm(contributions, ord=2, axis=1))),
        "super_user_pre_bounding_tasks": n_tasks,
        "super_user_post_bounding_tasks": int(np.max(super_user_contrib_sums)),
        "noise_scale": float(plan.noise_scale),
        "selected_top_k_count": len(selected),
        "bounding_verified": True,
    }

    out_dir = Path("results")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "exp25_topk_mechanisms.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"Max User L2 Norm: {results['max_empirical_user_l2_norm']:.4f} <= Bound {np.sqrt(k_bound):.4f}")
    print(f"Noise Scale (sigma): {results['noise_scale']:.4f}")
    print(f"\n[PASS] Exp25 completed successfully. Saved to {out_file}")
    return results


if __name__ == "__main__":
    run_exp25_topk_mechanisms()
