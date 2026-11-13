"""S4: Control Architecture Baseline - Epoch90-Synopsis.

Disentangles the two effects:
1. Privacy accounting semantics: Epoch-based reset vs Continuous stream.
2. Publishing architecture: Query-time noise addition vs Publish-once-and-post-process (Synopsis).

Evaluates Epoch90-Synopsis under:
- (i)  rho_epoch = 0.027204 (Matched to SDG-MH 90-day additive budget)
- (ii) rho_epoch = 0.253295 (Production setting, eps ~ 4.0 per epoch)
"""

from __future__ import annotations

import math
from pathlib import Path
import sys
from typing import Any, Dict, List
import numpy as np
import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).parents[3] / "src"))

from private_llm_analytics.sdg_mh.discrete_gaussian import sample_discrete_gaussian
from private_llm_analytics.sdg_mh.ledger import (
    UnifiedPrivacyLedger,
    closed_form_sdg_additive_rho,
    rho_to_epsilon,
)
from private_llm_analytics.sdg_mh.synthetic_generator import SyntheticTelemetryGenerator

OUTPUT_DIR = Path("results/sdg_mh/supp_v11")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def run_s4_benchmark():
    print("[S4] Benchmarking Epoch90-Synopsis Baseline on Dashboard Workload...")

    # Load Enterprise 365-day dataset (seed 42, N=2000)
    gen = SyntheticTelemetryGenerator(n_initial_users=2000, seed=42)
    daily_raw_requests: Dict[int, float] = {}
    daily_clipped_requests: Dict[int, float] = {}
    C_r = 104.0
    C_f = 7.0

    print("  Streaming 90 days of telemetry...")
    for day, df_day in gen.stream_daily_telemetry(start_day=1, end_day=90):
        daily_raw_requests[day] = float(len(df_day))
        if not df_day.empty:
            u_cnts = df_day["user_id"].value_counts()
            daily_clipped_requests[day] = float(u_cnts.clip(upper=C_r).sum())
        else:
            daily_clipped_requests[day] = 0.0

    # Build Dashboard Workload queries across 90-day epoch (150 queries)
    # 84 rolling 7d, 61 rolling 30d, 1 rolling 90d, 3 monthly, 1 quarterly
    dashboard_queries = []
    for t in range(7, 91):
        dashboard_queries.append(("rolling_7d", t - 6, t, 7))
    for t in range(30, 91):
        dashboard_queries.append(("rolling_30d", t - 29, t, 30))
    dashboard_queries.append(("rolling_90d", 1, 90, 90))
    dashboard_queries.append(("month_1", 1, 30, 30))
    dashboard_queries.append(("month_2", 31, 60, 30))
    dashboard_queries.append(("month_3", 61, 90, 30))
    dashboard_queries.append(("quarter_1", 1, 90, 90))

    configs = [
        {
            "name": "Epoch90-Synopsis (Matched Budget)",
            "rho_epoch": 0.027204,  # Exactly SDG-MH 90d budget = 90 * 0.00030204
            "desc": "rho_epoch = 0.0272 (Matches SDG-MH 90d budget)",
        },
        {
            "name": "Epoch90-Synopsis (Production Setting)",
            "rho_epoch": 0.253295,  # Production budget, eps ~ 4.0 / epoch
            "desc": "rho_epoch = 0.2533 (Production setting, eps ~ 4.0)",
        },
    ]

    workload_rows = []
    profile_rows = []
    horizons = [1, 7, 30, 90, 365]

    for cfg in configs:
        name = cfg["name"]
        rho_epoch = cfg["rho_epoch"]
        daily_rho_add = rho_epoch / 90.0

        # Ratio of requests vs failures in additive channel:
        # rho_1^req = (104^2) / (2 * sigma_r^2) = 0.000200
        # rho_1^fail = (7^2) / (2 * sigma_f^2) = 0.00010204
        # Total rho_1^add = 0.00030204
        # Proportion for requests: 0.000200 / 0.00030204 = 0.66216
        prop_req = 0.000200 / 0.00030204
        daily_rho_req = daily_rho_add * prop_req
        sigma_r = C_r / math.sqrt(2.0 * daily_rho_req)
        sigma_30 = sigma_r * math.sqrt(30.0)

        # 1. Dashboard Workload Simulation (1000 Monte Carlo trials)
        # In Synopsis architecture: each day publishes once!
        # y_t = clipped_t + DiscreteGaussian(sigma_r)
        # Window sum query Q = sum_{t in W} y_t -> pure post-processing, 0 query-time noise!
        n_mc = 1000
        rng_mc = np.random.default_rng(42)

        # Pre-simulate daily noisy values for 1000 trials
        sim_daily_noise = np.array([
            sample_discrete_gaussian(sigma_r, size=90, rng=rng_mc)
            for _ in range(n_mc)
        ])  # shape: (n_mc, 90), index 0..89 corresponds to days 1..90

        trial_30d_pure_errs = []
        trial_30d_e2e_errs = []

        # Evaluate all 30d queries
        queries_30d = [q for q in dashboard_queries if q[3] == 30]
        for _, s, e, length in queries_30d:
            gt_raw = sum(daily_raw_requests[t] for t in range(s, e + 1))
            gt_clip = sum(daily_clipped_requests[t] for t in range(s, e + 1))

            window_noise = np.sum(sim_daily_noise[:, (s - 1):e], axis=1)
            est_vals = np.maximum(0.0, gt_clip + window_noise)

            pure_err = np.mean(np.abs(window_noise) / gt_clip)
            e2e_err = np.mean(np.abs(est_vals - gt_raw) / gt_raw)
            trial_30d_pure_errs.append(pure_err)
            trial_30d_e2e_errs.append(e2e_err)

        mean_pure_mape_30d = float(np.mean(trial_30d_pure_errs))
        mean_e2e_mape_30d = float(np.mean(trial_30d_e2e_errs))

        # Analytical pure noise MAPE for 30d: (sigma_30 * sqrt(2/pi)) / Y_clip
        avg_30d_clip = np.mean([sum(daily_clipped_requests[t] for t in range(s, e + 1)) for _, s, e, l in queries_30d])
        analytical_pure_mape_30d = (sigma_30 * math.sqrt(2.0 / math.pi)) / avg_30d_clip

        workload_rows.append({
            "method": name,
            "setting_desc": cfg["desc"],
            "rho_epoch_cap": round(rho_epoch, 6),
            "ledger_rho_90": round(rho_epoch, 6),
            "availability": "100.0%",
            "sigma_30": round(sigma_30, 1),
            "analytical_noise_mape_30d": round(analytical_pure_mape_30d * 100.0, 2),
            "pure_noise_mape_mean_pct": round(mean_pure_mape_30d * 100.0, 2),
            "e2e_mape_mean_pct": round(mean_e2e_mape_30d * 100.0, 2),
        })

        # 2. Unified Ledger Profile & Cross-boundary Analysis
        # Setup 4 epochs (360 days) to test cross-boundary windows
        ledger = UnifiedPrivacyLedger()
        for d in range(1, 366):
            ledger.record_gaussian_release(
                release_id=f"synopsis_req_{d}",
                days_accessed=[d],
                sensitivity=C_r,
                sigma=sigma_r,
                tag="synopsis_req",
            )

        prof = ledger.get_profile(horizons=horizons)

        # Cross-boundary 90-day window: e.g. spanning days 46 to 135 (45 days in epoch 1, 45 days in epoch 2)
        # Under (u, I)-adjacency, an adversary observing interval I=[46, 135] touches exactly 90 daily releases!
        # Sum of rho is 90 * daily_rho_req = rho_epoch!
        cross_90_rho = 90 * daily_rho_req
        cross_90_eps = rho_to_epsilon(cross_90_rho)

        for h in horizons:
            prof_rho = prof[h]
            prof_eps = rho_to_epsilon(prof_rho)
            profile_rows.append({
                "method": name,
                "horizon_h": h,
                "daily_rho": round(daily_rho_req, 8),
                "rho_h": round(prof_rho, 6),
                "eps_h": round(prof_eps, 4),
                "cross_boundary_90d_rho": round(cross_90_rho, 6) if h == 90 else None,
                "cross_boundary_90d_eps": round(cross_90_eps, 4) if h == 90 else None,
            })

    df_workload = pd.DataFrame(workload_rows)
    df_profile = pd.DataFrame(profile_rows)

    out_workload_csv = OUTPUT_DIR / "s4_epoch90_synopsis_workload.csv"
    out_profile_csv = OUTPUT_DIR / "s4_epoch90_synopsis_profile.csv"

    df_workload.to_csv(out_workload_csv, index=False)
    df_profile.to_csv(out_profile_csv, index=False)

    print(f"\n[S4 Complete] Saved results to {out_workload_csv} and {out_profile_csv}")
    print("\n--- DASHBOARD WORKLOAD RESULTS ---")
    print(df_workload.to_string(index=False))
    print("\n--- PRIVACY PROFILE RESULTS ---")
    print(df_profile.to_string(index=False))


if __name__ == "__main__":
    run_s4_benchmark()
