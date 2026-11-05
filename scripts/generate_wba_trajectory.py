"""Generate 30-day daily trajectory for W-BA under Kellaris et al. (PVLDB 2014) on main experiment data.

Uses the main benchmark telemetry (2,000 synthetic enterprise users, seed 42, C=104.0),
where daily request volume averages ~15,400 requests/day (~462,354 total clipped requests over 30 days).

Rules implemented (Kellaris et al., PVLDB 2014 Section 4.3, Figure 4):
1. Fixed private dissimilarity test M_{i,1} consuming rho_test = rho_w / (2*w).
2. Base publication budget rho_base = rho_w / (2*w).
3. Nullification rule: If a release absorbed budget, to_nullify = round(rho_rel / rho_base) - 1.
   The next to_nullify timestamps are forced null (cannot be absorbed, spends only test budget if evaluated).
4. Capped absorption: to_absorb = day - nullify_until. Max absorption is min(to_absorb, w).
   Effective release budget rho_rel = rho_base * min(to_absorb, w) <= rho_w / 2.
5. If noisy_diff > sigma_candidate, publish; else skip (approximate with last release).

Saves to results/sdg_mh/wba_30day_trajectory.csv.
"""

import math
from pathlib import Path
import numpy as np
import pandas as pd
from private_llm_analytics.sdg_mh.ledger import rho_to_epsilon
from private_llm_analytics.sdg_mh.synthetic_generator import SyntheticTelemetryGenerator

RESULTS_DIR = Path("results/sdg_mh")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
OUT_CSV = RESULTS_DIR / "wba_30day_trajectory.csv"

def run_trajectory():
    # Parameters
    w = 30
    C = 104.0
    delta = 1e-6
    target_rho_w = 0.0090615  # Calibrated 30d additive target rho
    rho_test = target_rho_w / (2.0 * float(w))  # 0.000151025
    rho_base = target_rho_w / (2.0 * float(w))  # 0.000151025
    sigma_test = C / math.sqrt(2.0 * rho_test)  # ~ 5990.7

    # Generate exact 30-day clipped daily request counts from main experiment data
    gen = SyntheticTelemetryGenerator(n_initial_users=2000, seed=42)
    daily_clipped_x = []
    for day, df_day in gen.stream_daily_telemetry(start_day=1, end_day=30):
        u_req = df_day["user_id"].value_counts()
        clipped_r = float(np.clip(u_req, 0, C).sum())
        daily_clipped_x.append(clipped_r)

    print(f"Loaded main experiment data: 30 days, total clipped volume = {sum(daily_clipped_x):,.1f}, mean = {np.mean(daily_clipped_x):,.1f}/day")

    rng_dp = np.random.default_rng(20261105)
    rows = []

    last_published = None
    last_release_day = 0
    last_release_rho = 0.0
    cumulative_rho = 0.0

    for day_idx, true_x in enumerate(daily_clipped_x):
        day = day_idx + 1

        if day == 1:
            # Day 1: initial publication with base publication budget
            rho_rel = rho_base
            sig_rel = C / math.sqrt(2.0 * rho_rel)
            noise_rel = float(rng_dp.normal(0.0, sig_rel))
            pub_val = float(max(0.0, true_x + noise_rel))

            last_published = pub_val
            last_release_day = day
            last_release_rho = rho_rel

            spent_rho = rho_test + rho_rel
            spent_eps = rho_to_epsilon(spent_rho, delta)
            cumulative_rho += spent_rho

            rows.append({
                "day": day,
                "true_x": round(true_x, 1),
                "last_published": None,
                "diff": 0.0,
                "noisy_diff": 0.0,
                "threshold": round(sig_rel, 1),
                "test_passed": False,
                "released": True,
                "status": "initial_release",
                "budget_rho": round(spent_rho, 8),
                "budget_eps": round(spent_eps, 4),
                "pool_state": 0.0,
                "published_val": round(pub_val, 1),
                "staleness_bias": 0.0,
                "pure_noise_err": round(noise_rel, 1),
                "e2e_err": round(pub_val - true_x, 1),
                "cumulative_rho": round(cumulative_rho, 8),
            })
            continue

        # Check nullification from previous release
        to_nullify = max(0, int(round(last_release_rho / rho_base)) - 1)
        nullify_until = last_release_day + to_nullify

        if day <= nullify_until:
            # Forced null due to nullification rule
            # Discard Di, spends test budget
            spent_rho = rho_test
            spent_eps = rho_to_epsilon(spent_rho, delta)
            cumulative_rho += spent_rho
            staleness_bias = last_published - true_x

            rows.append({
                "day": day,
                "true_x": round(true_x, 1),
                "last_published": round(last_published, 1),
                "diff": round(abs(true_x - last_published), 1),
                "noisy_diff": 0.0,
                "threshold": 0.0,
                "test_passed": True,
                "released": False,
                "status": "nullified",
                "budget_rho": round(spent_rho, 8),
                "budget_eps": round(spent_eps, 4),
                "pool_state": 0.0,
                "published_val": round(last_published, 1),
                "staleness_bias": round(staleness_bias, 1),
                "pure_noise_err": 0.0,
                "e2e_err": round(staleness_bias, 1),
                "cumulative_rho": round(cumulative_rho, 8),
            })
            continue

        # Private dissimilarity test
        diff = abs(true_x - last_published)
        noise_test = float(rng_dp.normal(0.0, sigma_test))
        noisy_diff = diff + noise_test

        # Calculate absorbable budget
        to_absorb = day - nullify_until
        num_absorbed = min(to_absorb, w)
        rho_cand = rho_base * float(num_absorbed)
        sig_cand = C / math.sqrt(2.0 * rho_cand)
        threshold = sig_cand

        test_passed = (noisy_diff <= threshold)

        if test_passed:
            # Skip: reuse last published value, save budget for future absorption
            pub_val = last_published
            spent_rho = rho_test
            spent_eps = rho_to_epsilon(spent_rho, delta)
            cumulative_rho += spent_rho
            staleness_bias = pub_val - true_x

            rows.append({
                "day": day,
                "true_x": round(true_x, 1),
                "last_published": round(last_published, 1),
                "diff": round(diff, 1),
                "noisy_diff": round(noisy_diff, 1),
                "threshold": round(threshold, 1),
                "test_passed": True,
                "released": False,
                "status": "absorbed_skip",
                "budget_rho": round(spent_rho, 8),
                "budget_eps": round(spent_eps, 4),
                "pool_state": round(rho_cand, 8),
                "published_val": round(pub_val, 1),
                "staleness_bias": round(staleness_bias, 1),
                "pure_noise_err": 0.0,
                "e2e_err": round(staleness_bias, 1),
                "cumulative_rho": round(cumulative_rho, 8),
            })
        else:
            # Release: fresh release with absorbed budget
            noise_rel = float(rng_dp.normal(0.0, sig_cand))
            pub_val = float(max(0.0, true_x + noise_rel))

            last_published = pub_val
            last_release_day = day
            last_release_rho = rho_cand

            spent_rho = rho_test + rho_cand
            spent_eps = rho_to_epsilon(spent_rho, delta)
            cumulative_rho += spent_rho

            rows.append({
                "day": day,
                "true_x": round(true_x, 1),
                "last_published": round(rows[-1]["published_val"], 1),
                "diff": round(diff, 1),
                "noisy_diff": round(noisy_diff, 1),
                "threshold": round(threshold, 1),
                "test_passed": False,
                "released": True,
                "status": "fresh_release",
                "budget_rho": round(spent_rho, 8),
                "budget_eps": round(spent_eps, 4),
                "pool_state": 0.0,
                "published_val": round(pub_val, 1),
                "staleness_bias": 0.0,
                "pure_noise_err": round(noise_rel, 1),
                "e2e_err": round(pub_val - true_x, 1),
                "cumulative_rho": round(cumulative_rho, 8),
            })

    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False)
    print(f"Saved faithful W-BA trajectory to {OUT_CSV}")
    print("\n--- 30-Day W-BA Trajectory Summary on Main Data ---")
    print(f"Total days: {len(df)}")
    print(f"Releases: {df['released'].sum()} days ({(df['released'].sum()/len(df))*100:.1f}%)")
    print(f"Skips/Nullified: {(~df['released']).sum()} days ({((~df['released']).sum()/len(df))*100:.1f}%)")
    print(f"Total Cumulative rho: {df['cumulative_rho'].iloc[-1]:.6f} (vs target 30d rho: {target_rho_w:.6f})")
    print(f"Average Daily rho: {df['budget_rho'].mean():.8f}")
    mape = (df['e2e_err'].abs() / df['true_x']).mean() * 100
    print(f"Average Daily End-to-End MAPE: {mape:.2f}%")
    print("\nFirst 10 days of trajectory:")
    print(df[["day", "true_x", "diff", "noisy_diff", "threshold", "test_passed", "released", "status", "budget_rho", "published_val"]].head(10).to_string())

if __name__ == "__main__":
    run_trajectory()
