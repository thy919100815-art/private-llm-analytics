"""Exp24: Top-K Tasks DP Utility and Heavy Hitters Evaluation (v0.8).

Evaluates Recall@K, Precision@K, and Spearman rank correlation on long-tail
Zipf distributions across varying privacy budgets epsilon and task dictionary sizes.
"""

from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
from scipy.stats import spearmanr

from private_llm_analytics.aggregation.grouping import build_cell_definitions
from private_llm_analytics.aggregation.user_matrix import UserMatrix
from private_llm_analytics.statistics.top_k_tasks import TopKTasksProfile


def generate_zipf_tasks_dataset(
    n_users: int = 500,
    n_tasks: int = 50,
    zipf_param: float = 1.2,
    seed: int = 42,
) -> UserMatrix:
    rng = np.random.default_rng(seed)
    providers = ["provider_main"]
    tasks = [f"task_{i:03d}" for i in range(n_tasks)]
    cell_names, cell_map = build_cell_definitions(providers, tasks)

    # Probabilities proportional to 1 / (rank + 1)^zipf_param
    ranks = np.arange(1, n_tasks + 1)
    probs = 1.0 / (ranks ** zipf_param)
    probs /= np.sum(probs)

    matrix = np.zeros((n_users, n_tasks), dtype=np.float64)
    for u in range(n_users):
        # User interacts with Poisson(lam=4) tasks sampled according to Zipf distribution
        n_user_tasks = max(1, rng.poisson(lam=4))
        chosen_tasks = rng.choice(n_tasks, size=min(n_user_tasks, n_tasks), replace=False, p=probs)
        for t in chosen_tasks:
            matrix[u, t] = rng.integers(1, 5)

    return UserMatrix(
        matrix=matrix,
        user_ids=[f"u_{i}" for i in range(n_users)],
        providers=providers,
        tasks=tasks,
        cell_names=cell_names,
        cell_map=cell_map,
        public_population_size=1000,
    )


def run_exp24_topk_utility() -> dict:
    epsilons = [0.2, 0.5, 1.0, 2.0]
    k_targets = [5, 10]
    n_trials = 10

    results: Dict[str, dict] = {}
    print("============================================================")
    print("Exp24: Top-K Tasks DP Utility on Long-Tail Distributions")
    print("============================================================")

    umatrix = generate_zipf_tasks_dataset(n_users=500, n_tasks=50, zipf_param=1.2)
    # Ground truth top-K based on true binary user participation
    true_participation = np.sum((umatrix.matrix > 0).astype(float), axis=0)
    true_ranking = sorted(range(len(umatrix.tasks)), key=lambda i: (-true_participation[i], umatrix.tasks[i]))

    for K in k_targets:
        k_results = {}
        true_top_k_set = set([umatrix.tasks[idx] for idx in true_ranking[:K]])
        profile = TopKTasksProfile(k=K, max_tasks_per_user=5)

        for eps in epsilons:
            recalls = []
            precisions = []
            spearman_corrs = []

            for trial in range(n_trials):
                trial_rng = np.random.default_rng(20261000 + int(eps * 100) + trial)
                rel = profile.execute_release(umatrix, epsilon=eps, delta=1e-6, rng=trial_rng)

                selected = set(rel.post_processed["selected_tasks"])
                overlap = len(selected.intersection(true_top_k_set))
                rec = overlap / float(K)
                prec = overlap / float(max(1, len(selected)))

                recalls.append(rec)
                precisions.append(prec)

                # Correlation across all tasks
                noisy_counts = [rel.post_processed["all_noisy_counts"][t] for t in umatrix.tasks]
                corr, _ = spearmanr(true_participation, noisy_counts)
                if not np.isnan(corr):
                    spearman_corrs.append(corr)

            mean_rec = float(np.mean(recalls))
            mean_prec = float(np.mean(precisions))
            mean_corr = float(np.mean(spearman_corrs)) if spearman_corrs else 0.0

            k_results[f"eps_{eps}"] = {
                "epsilon": eps,
                "recall_at_k": mean_rec,
                "precision_at_k": mean_prec,
                "spearman_correlation": mean_corr,
            }
            print(f"K={K:2d}, eps={eps:.1f}: Recall@{K}={mean_rec:.3f}, Precision@{K}={mean_prec:.3f}, RankCorr={mean_corr:.3f}")

        results[f"top_{K}"] = k_results

    out_dir = Path("results")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "exp24_topk_utility.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\n[PASS] Exp24 completed successfully. Saved to {out_file}")
    return results


if __name__ == "__main__":
    run_exp24_topk_utility()
