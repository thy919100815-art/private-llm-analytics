#!/usr/bin/env python3
"""RQ4: Distinct-User Cost and Refresh Frequency Sweep (Calendar A vs Calendar B).

Sweeps:
- Refresh step s in {1, 7, 14, 30} across window lengths w in {7, 30, 90}.
- Noise scale sigma_w in {30, 60, 120}.
- Evaluates Calendar A vs Calendar B.
- Computes:
  - Exact privacy profiles eps_h (h in {1, 7, 30, 90, 365}).
  - Prevalence relative error: |est - true| / max(true, 10).
  - Top-K metrics: Precision@5, Precision@10, NDCG@10.
- Repeats 30 independent trials per configuration with discrete Gaussian noise.
- Generates trade-off plots (Refresh step s vs eps_365 and Top-K NDCG).
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, List, Tuple
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from private_llm_analytics.sdg_mh.calendar_job import (
    DEFAULT_CALENDAR_A,
    DEFAULT_CALENDAR_B,
)
from private_llm_analytics.sdg_mh.clipping import compute_distinct_window_histogram
from private_llm_analytics.sdg_mh.discrete_gaussian import sample_discrete_gaussian
from private_llm_analytics.sdg_mh.ledger import (
    closed_form_distinct_calendar_rho,
    closed_form_sdg_additive_rho,
    rho_to_epsilon,
)
from private_llm_analytics.sdg_mh.synthetic_generator import SyntheticTelemetryGenerator

RESULTS_DIR = Path("results/sdg_mh")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def compute_ndcg_at_k(true_ranking: List[str], noisy_ranking: List[str], k: int = 10) -> float:
    """Compute Normalized Discounted Cumulative Gain at rank k."""
    relevance_map = {t: len(true_ranking) - idx for idx, t in enumerate(true_ranking)}
    ideal_rel = sorted([relevance_map[t] for t in true_ranking[:k]], reverse=True)
    idcg = sum((2**rel - 1) / math.log2(idx + 2) for idx, rel in enumerate(ideal_rel))
    if idcg <= 0:
        return 1.0

    dcg = 0.0
    for idx, t in enumerate(noisy_ranking[:k]):
        rel = relevance_map.get(t, 0)
        dcg += (2**rel - 1) / math.log2(idx + 2)

    return float(min(1.0, max(0.0, dcg / idcg)))


def run_rq4_sweep(n_trials: int = 30):
    print(f"--- [RQ4] Distinct-User Cost and Refresh Frequency Sweep ({n_trials} trials) ---")
    gen = SyntheticTelemetryGenerator(n_initial_users=2000, n_years=0.35, seed=42)

    # Pre-generate 100 days of telemetry
    day_dfs = {}
    print("Generating synthetic 100-day telemetry for RQ4...")
    for day, df_day in gen.stream_daily_telemetry(start_day=1, end_day=100):
        day_dfs[day] = df_day

    def get_window_df(s: int, e: int) -> pd.DataFrame:
        dfs = [day_dfs[d] for d in range(s, e + 1) if d in day_dfs]
        return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

    services_list = gen.services
    tasks_list = gen.tasks
    delta_sq = float(len(services_list) + 2 * 3)  # m=5, k=3 => 11

    # Precompute true distinct histograms for target windows w in {7, 30, 90} at day 90
    print("Precomputing ground-truth distinct histograms...")
    true_data_by_w = {}
    for w in [7, 30, 90]:
        df_w = get_window_df(90 - w + 1, 90)
        true_srv_w, true_tsk_w = compute_distinct_window_histogram(df_w, services_list, tasks_list, k_tasks_per_user=3)
        true_ranking_w = [tasks_list[idx] for idx in np.argsort(-true_tsk_w)]
        true_data_by_w[w] = (true_srv_w, true_tsk_w, true_ranking_w)

    # 1. Evaluate Calendar A and Calendar B full comparison on w=30
    cal_configs = [
        ("Calendar_A", DEFAULT_CALENDAR_A),
        ("Calendar_B", DEFAULT_CALENDAR_B),
    ]

    cal_results = []
    print("Evaluating Calendar A vs Calendar B across 30 runs...")
    true_srv_30, true_tsk_30, true_ranking_30 = true_data_by_w[30]
    r1_add = (50.0**2)/(2 * 2500.0**2) + (10.0**2)/(2 * 700.0**2)

    for cal_name, cal_sched in cal_configs:
        eps_profile = {}
        for h in [1, 7, 30, 90, 365]:
            r_tot = closed_form_sdg_additive_rho(h, r1_add) + closed_form_distinct_calendar_rho(h, cal_sched, delta_sq=delta_sq)
            eps_profile[h] = rho_to_epsilon(r_tot)

        # In both calendars, the w=30 release is available at day 90 (90 % 7 == 6, closest step is 84 or 90)
        # For w=30, sigma is 60.0
        sigma_30 = 60.0
        for trial in range(n_trials):
            rng_noise = np.random.default_rng(500000 + trial * 100)
            noise_srv = sample_discrete_gaussian(sigma_30, size=len(true_srv_30), rng=rng_noise)
            noise_tsk = sample_discrete_gaussian(sigma_30, size=len(true_tsk_30), rng=rng_noise)

            srv_ests = np.maximum(0, true_srv_30 + noise_srv)
            tsk_ests = np.maximum(0, true_tsk_30 + noise_tsk)

            prev_rel_err = float(np.mean(np.abs(srv_ests - true_srv_30) / np.maximum(true_srv_30, 10.0)))
            noisy_ranking = [tasks_list[idx] for idx in np.argsort(-tsk_ests)]

            p5 = len(set(true_ranking_30[:5]).intersection(set(noisy_ranking[:5]))) / 5.0
            p10 = len(set(true_ranking_30[:10]).intersection(set(noisy_ranking[:10]))) / 10.0
            ndcg10 = compute_ndcg_at_k(true_ranking_30, noisy_ranking, k=10)

            cal_results.append({
                "calendar": cal_name,
                "trial": trial,
                "eps_1": round(eps_profile[1], 4),
                "eps_7": round(eps_profile[7], 4),
                "eps_30": round(eps_profile[30], 4),
                "eps_90": round(eps_profile[90], 4),
                "eps_365": round(eps_profile[365], 4),
                "prev_rel_err": round(prev_rel_err, 4),
                "precision_at_5": round(p5, 4),
                "precision_at_10": round(p10, 4),
                "ndcg_at_10": round(ndcg10, 4),
            })

    df_cal = pd.DataFrame(cal_results)
    df_cal.to_csv(RESULTS_DIR / "rq4_calendar_comparison_trials.csv", index=False)

    # 2. Comprehensive Parameter Sweep: s in {1, 7, 14, 30}, w in {7, 30, 90}, sigma_w in {30, 60, 120}
    print("Running parameter grid sweep for s, w, sigma_w...")
    sweep_rows = []
    w_list = [7, 30, 90]
    s_list = [1, 7, 14, 30]
    sigma_list = [30.0, 60.0, 120.0]

    for w in w_list:
        true_srv_w, true_tsk_w, true_ranking_w = true_data_by_w[w]

        for s in s_list:
            for sigma_w in sigma_list:
                sched = [(w, s, sigma_w)]
                eps_h_map = {}
                for h in [1, 7, 30, 90, 365]:
                    r_dist = closed_form_distinct_calendar_rho(h, sched, delta_sq=delta_sq)
                    eps_h_map[h] = rho_to_epsilon(r_dist)

                trial_p5 = []
                trial_p10 = []
                trial_ndcg = []
                trial_prev_err = []

                for trial in range(n_trials):
                    rng_noise = np.random.default_rng(800000 + trial * 100 + w * 10 + s)
                    noise_srv = sample_discrete_gaussian(sigma_w, size=len(true_srv_w), rng=rng_noise)
                    noise_tsk = sample_discrete_gaussian(sigma_w, size=len(true_tsk_w), rng=rng_noise)

                    est_srv = np.maximum(0, true_srv_w + noise_srv)
                    est_tsk = np.maximum(0, true_tsk_w + noise_tsk)

                    p_err = float(np.mean(np.abs(est_srv - true_srv_w) / np.maximum(true_srv_w, 10.0)))
                    noisy_ranking = [tasks_list[idx] for idx in np.argsort(-est_tsk)]

                    p5 = len(set(true_ranking_w[:5]).intersection(set(noisy_ranking[:5]))) / 5.0
                    p10 = len(set(true_ranking_w[:10]).intersection(set(noisy_ranking[:10]))) / 10.0
                    ndcg10 = compute_ndcg_at_k(true_ranking_w, noisy_ranking, k=10)

                    trial_prev_err.append(p_err)
                    trial_p5.append(p5)
                    trial_p10.append(p10)
                    trial_ndcg.append(ndcg10)

                sweep_rows.append({
                    "w": w,
                    "s": s,
                    "sigma_w": sigma_w,
                    "eps_30": round(eps_h_map[30], 4),
                    "eps_365": round(eps_h_map[365], 4),
                    "prev_rel_err_mean": round(float(np.mean(trial_prev_err)), 4),
                    "prev_rel_err_std": round(float(np.std(trial_prev_err)), 4),
                    "precision_at_5_mean": round(float(np.mean(trial_p5)), 4),
                    "precision_at_10_mean": round(float(np.mean(trial_p10)), 4),
                    "ndcg_at_10_mean": round(float(np.mean(trial_ndcg)), 4),
                    "ndcg_at_10_std": round(float(np.std(trial_ndcg)), 4),
                })

    df_sweep = pd.DataFrame(sweep_rows)
    df_sweep.to_csv(RESULTS_DIR / "rq4_distinct_sweep_results.csv", index=False)

    # 3. Generate Trade-off plot
    print("Generating RQ4 trade-off plot...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), dpi=300)

    # Plot 1: Refresh step s vs eps_365 (for w=30 across sigma_w)
    df_w30 = df_sweep[df_sweep["w"] == 30]
    for sigma in sigma_list:
        sub = df_w30[df_w30["sigma_w"] == sigma].sort_values("s")
        ax1.plot(sub["s"], sub["eps_365"], marker="o", label=f"σ_w = {int(sigma)}")
    ax1.set_xlabel("Refresh Step s (Days)", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Annual Privacy Loss ε₃₆₅", fontsize=11, fontweight="bold")
    ax1.set_title("RQ4: Refresh Frequency vs Annual Privacy Loss ε₃₆₅ (w=30)", fontsize=12, pad=10)
    ax1.grid(True, linestyle="--", alpha=0.6)
    ax1.legend()

    # Plot 2: Refresh step s vs Top-10 NDCG
    for sigma in sigma_list:
        sub = df_w30[df_w30["sigma_w"] == sigma].sort_values("s")
        ax2.plot(sub["s"], sub["ndcg_at_10_mean"], marker="s", label=f"σ_w = {int(sigma)}")
    ax2.set_xlabel("Refresh Step s (Days)", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Top-10 NDCG", fontsize=11, fontweight="bold")
    ax2.set_title("RQ4: Refresh Frequency vs Top-10 Task Utility (w=30)", fontsize=12, pad=10)
    ax2.grid(True, linestyle="--", alpha=0.6)
    ax2.legend()

    plt.tight_layout()
    fig.savefig(RESULTS_DIR / "fig_rq4_refresh_tradeoff.png")
    plt.close(fig)
    print(">>> RQ4 complete. Saved results and trade-off figure.")


if __name__ == "__main__":
    run_rq4_sweep(n_trials=30)
