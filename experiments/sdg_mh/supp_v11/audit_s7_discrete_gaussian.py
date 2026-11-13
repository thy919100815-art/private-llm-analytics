"""S7: Discrete Gaussian Sampler Empirical Variance vs Theoretical Variance.

Evaluates for sigma in {30, 500, 1500, 5200}:
- Empirical variance / sigma^2 with 95% CI (N = 10^6 samples).
- Theoretical Discrete Gaussian variance / sigma^2 (numerical summation of defining series).
- Relative error between empirical and theoretical variance.
- Verification that empirical variance <= sigma^2 within 95% sampling CI.
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

OUTPUT_DIR = Path("results/sdg_mh/supp_v11")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def compute_theoretical_discrete_gaussian_variance(sigma: float) -> float:
    """Compute exact theoretical discrete Gaussian variance by direct numerical summation:

    Var[X] = (2 * sum_{x=1}^K x^2 * exp(-x^2 / (2*sigma^2))) / (1 + 2 * sum_{y=1}^K exp(-y^2 / (2*sigma^2)))
    """
    two_sig_sq = 2.0 * sigma * sigma
    # Truncate at K = ceil(10 * sigma) where tail exp(-50) < 1e-21
    # For large sigma (e.g. 500, 1500, 5200), Poisson summation theorem proves:
    # Var[X] / sigma^2 = 1 - 2*pi * exp(-2*pi^2*sigma^2) + ...
    # which is 1.0 within float64 precision (difference < 1e-16).
    if sigma > 20.0:
        # For sigma >= 20, Poisson summation gives 1.0 to machine precision
        # We can also compute numerical sum for verification
        K = min(int(math.ceil(10.0 * sigma)), 200000)
    else:
        K = int(math.ceil(10.0 * sigma))

    x_arr = np.arange(1, K + 1, dtype=np.float64)
    exp_terms = np.exp(- (x_arr ** 2) / two_sig_sq)
    numerator = 2.0 * np.sum((x_arr ** 2) * exp_terms)
    denominator = 1.0 + 2.0 * np.sum(exp_terms)

    return float(numerator / denominator)


def run_s7_audit():
    print("[S7] Auditing Discrete Gaussian Sampler Variance across 10^6 trials...")
    sigmas = [30.0, 500.0, 1500.0, 5200.0]
    n_samples = 1_000_000
    seed = 42

    rows = []
    # Standard error of s^2 / sigma^2 for sample size N: sqrt(2/N)
    se_ratio = math.sqrt(2.0 / float(n_samples))
    ci95_margin = 1.959964 * se_ratio  # ~0.0027718

    for sig in sigmas:
        print(f"  Sampling 1,000,000 draws for sigma = {sig:.1f}...")
        samples = sample_discrete_gaussian(sigma=sig, size=n_samples, seed=int(seed + int(sig)))

        emp_mean = float(np.mean(samples))
        emp_var = float(np.var(samples, ddof=1))
        var_ratio_emp = emp_var / (sig ** 2)

        ci_lower = var_ratio_emp - ci95_margin
        ci_upper = var_ratio_emp + ci95_margin

        theor_var = compute_theoretical_discrete_gaussian_variance(sig)
        theor_ratio = theor_var / (sig ** 2)

        rel_err = abs(emp_var - theor_var) / theor_var

        # Verification: empirical variance <= sigma^2 within 95% sampling CI (i.e. ci_lower <= 1.0)
        verified = bool(ci_lower <= 1.0)

        rows.append({
            "sigma": sig,
            "sample_size": n_samples,
            "empirical_mean": round(emp_mean, 4),
            "empirical_var": round(emp_var, 4),
            "theoretical_var": round(theor_var, 4),
            "empirical_var_ratio": round(var_ratio_emp, 6),
            "ci95_lower": round(ci_lower, 6),
            "ci95_upper": round(ci_upper, 6),
            "theoretical_var_ratio": round(theor_ratio, 8),
            "relative_error": round(rel_err, 6),
            "empirical_le_sigma_sq": bool(emp_var <= sig ** 2),
            "verified_within_ci": verified,
        })

    df_out = pd.DataFrame(rows)
    out_csv = OUTPUT_DIR / "s7_discrete_gaussian_variance.csv"
    df_out.to_csv(out_csv, index=False)
    print(f"\n[S7 Complete] Saved variance audit table to {out_csv}")
    print(df_out.to_string(index=False))


if __name__ == "__main__":
    run_s7_audit()
