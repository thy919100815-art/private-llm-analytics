"""Generate 30-day daily trajectory for W-BD and W-BA under Kellaris et al. (PVLDB 2014).

Records daily:
- day
- true_x
- last_published
- diff
- noisy_diff
- threshold
- test_passed
- released
- budget_rho
- budget_eps
- pool_state
- published_val
- staleness_bias
- pure_noise_err
- e2e_err

Saves to results/sdg_mh/wba_30day_trajectory.csv.
"""

import math
from pathlib import Path
import numpy as np
import pandas as pd
from private_llm_analytics.sdg_mh.ledger import rho_to_epsilon

RESULTS_DIR = Path("results/sdg_mh")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
OUT_CSV = RESULTS_DIR / "wba_30day_trajectory.csv"

def run_trajectory():
    # Parameters
    w = 30
    C = 104.0
    delta = 1e-6
    target_rho_w = 0.0090611  # Calibrated 30d additive target rho
    rho_daily = target_rho_w / float(w)  # 0.00030204
    alpha = 0.2  # Kellaris et al. 2014 test fraction
    rho_test = alpha * rho_daily  # 0.00006041
    rho_release_base = (1.0 - alpha) * rho_daily  # 0.00024163

    sigma_test = C / math.sqrt(2.0 * rho_test)  # 9459.6
    threshold = 2.0 * sigma_test  # 18919.2

    # Generate realistic 30-day true daily request series (weekday-weekend pattern around ~250k daily requests)
    rng_data = np.random.default_rng(42)
    daily_true_x = []
    for day in range(1, 31):
        is_weekend = (day % 7 in [6, 0])
        base = 160000.0 if is_weekend else 280000.0
        drift = 1000.0 * day
        fluctuation = rng_data.normal(0, 8000.0)
        daily_true_x.append(round(base + drift + fluctuation, 1))

    # Run W-BA trajectory
    rng_dp = np.random.default_rng(20261104)
    rows = []

    last_published = None
    absorption_pool = 0.0

    cumulative_rho = 0.0

    for day_idx, true_x in enumerate(daily_true_x):
        day = day_idx + 1

        if day == 1:
            # First day: mandatory initial release
            sig_rel = C / math.sqrt(2.0 * rho_release_base)
            noise_rel = rng_dp.normal(0.0, sig_rel)
            pub_val = max(0.0, true_x + noise_rel)
            last_published = pub_val
            spent_rho = rho_release_base
            spent_eps = rho_to_epsilon(spent_rho, delta)
            cumulative_rho += spent_rho

            rows.append({
                "day": day,
                "true_x": true_x,
                "last_published": None,
                "diff": 0.0,
                "noisy_diff": 0.0,
                "threshold": round(threshold, 1),
                "test_passed": False,
                "released": True,
                "budget_rho": round(spent_rho, 8),
                "budget_eps": round(spent_eps, 4),
                "pool_state": round(absorption_pool, 8),
                "published_val": round(pub_val, 1),
                "staleness_bias": 0.0,
                "pure_noise_err": round(noise_rel, 1),
                "e2e_err": round(pub_val - true_x, 1),
                "cumulative_rho": round(cumulative_rho, 8),
            })
        else:
            # Day t > 1: Run similarity test
            diff = abs(true_x - last_published)
            noise_test = rng_dp.normal(0.0, sigma_test)
            noisy_diff = diff + noise_test

            test_passed = (noisy_diff < threshold)

            if test_passed:
                # Absorb: reuse last published value, accumulate base release budget into pool
                absorption_pool += rho_release_base
                pub_val = last_published
                spent_rho = rho_test
                spent_eps = rho_to_epsilon(spent_rho, delta)
                cumulative_rho += spent_rho

                staleness_bias = pub_val - true_x
                pure_noise_err = 0.0  # No new noise added

                rows.append({
                    "day": day,
                    "true_x": true_x,
                    "last_published": round(last_published, 1),
                    "diff": round(diff, 1),
                    "noisy_diff": round(noisy_diff, 1),
                    "threshold": round(threshold, 1),
                    "test_passed": True,
                    "released": False,
                    "budget_rho": round(spent_rho, 8),
                    "budget_eps": round(spent_eps, 4),
                    "pool_state": round(absorption_pool, 8),
                    "published_val": round(pub_val, 1),
                    "staleness_bias": round(staleness_bias, 1),
                    "pure_noise_err": 0.0,
                    "e2e_err": round(staleness_bias, 1),
                    "cumulative_rho": round(cumulative_rho, 8),
                })
            else:
                # Release: fresh release with base budget + accumulated pool
                effective_rho_rel = rho_release_base + absorption_pool
                absorption_pool = 0.0  # Reset pool
                sig_rel = C / math.sqrt(2.0 * effective_rho_rel)
                noise_rel = rng_dp.normal(0.0, sig_rel)
                pub_val = max(0.0, true_x + noise_rel)
                last_published = pub_val

                spent_rho = rho_test + effective_rho_rel
                spent_eps = rho_to_epsilon(spent_rho, delta)
                cumulative_rho += spent_rho

                rows.append({
                    "day": day,
                    "true_x": true_x,
                    "last_published": round(rows[-1]["published_val"], 1),
                    "diff": round(diff, 1),
                    "noisy_diff": round(noisy_diff, 1),
                    "threshold": round(threshold, 1),
                    "test_passed": False,
                    "released": True,
                    "budget_rho": round(spent_rho, 8),
                    "budget_eps": round(spent_eps, 4),
                    "pool_state": 0.0,
                    "published_val": round(pub_val, 1),
                    "staleness_bias": 0.0,
                    "pure_noise_err": round(noise_rel, 1),
                    "e2e_err": round(noise_rel, 1),
                    "cumulative_rho": round(cumulative_rho, 8),
                })

    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False)
    print(f"Saved 30-day W-BA trajectory to {OUT_CSV}")
    print("\n--- 30-Day W-BA Trajectory Summary ---")
    print(f"Total days: {len(df)}")
    print(f"Releases: {df['released'].sum()} days ({(df['released'].sum()/len(df))*100:.1f}%)")
    print(f"Absorptions (skipped releases): {(~df['released']).sum()} days ({((~df['released']).sum()/len(df))*100:.1f}%)")
    print(f"Total Cumulative rho: {df['cumulative_rho'].iloc[-1]:.6f} (vs target 30d rho: {target_rho_w:.6f})")
    print(f"Average Daily rho: {df['budget_rho'].mean():.8f}")
    print(f"Average Daily End-to-End MAPE: {(df['e2e_err'].abs() / df['true_x']).mean() * 100:.2f}%")
    print("\nFirst 10 days of trajectory:")
    print(df[["day", "true_x", "diff", "noisy_diff", "test_passed", "released", "budget_rho", "pool_state", "published_val"]].head(10).to_string())

if __name__ == "__main__":
    run_trajectory()
