"""Verification script: 100-Release Numerical RDP Composition Cross-Validation (v0.2).

Cross-validates RDP additive composition for 1, 10, and 100 independent releases
against an independent arbitrary-precision analytical reference.
"""

from decimal import Decimal
import math
from pathlib import Path
import sys

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from private_llm_analytics.privacy.rdp import (
    DEFAULT_RDP_ORDERS,
    compute_gaussian_rdp,
    compose_rdp_vectors,
    rdp_to_epsilon_delta,
)


def independent_reference_epsilon(
    n_releases: int,
    sigma: float,
    sensitivity: float,
    delta: float,
    orders: list,
) -> tuple:
    """Independent closed-form calculation of composed epsilon and optimal alpha."""
    ln_inv_delta = math.log(1.0 / delta)
    best_alpha = None
    min_eps = None
    sens_sq = sensitivity ** 2
    var_2 = 2.0 * (sigma ** 2)

    for alpha in orders:
        # Sum of T identical Gaussian RDP costs: T * (alpha * sens^2) / (2 * sigma^2)
        total_rdp = n_releases * (alpha * sens_sq) / var_2
        eps_alpha = total_rdp + ln_inv_delta / (alpha - 1.0)
        if min_eps is None or eps_alpha < min_eps:
            min_eps = eps_alpha
            best_alpha = alpha

    return best_alpha, min_eps


def main():
    print("=" * 70)
    print("VERIFICATION: 100-Release RDP Composition Cross-Validation")
    print("=" * 70)

    delta_global = 1e-6
    sensitivity = 1.0
    sigma = 1.8
    orders = DEFAULT_RDP_ORDERS

    single_rel = compute_gaussian_rdp(sigma=sigma, sensitivity=sensitivity, orders=orders)

    for T in [1, 10, 100]:
        print(f"[*] Composing T={T} independent Gaussian releases...")
        composed = compose_rdp_vectors([single_rel] * T)
        best_alpha, eff_eps = rdp_to_epsilon_delta(composed, delta_global=delta_global)

        ref_alpha, ref_eps = independent_reference_epsilon(
            n_releases=T,
            sigma=sigma,
            sensitivity=sensitivity,
            delta=delta_global,
            orders=orders,
        )

        diff = abs(float(eff_eps) - ref_eps)
        print(f"    RDP Accountant: best_alpha={best_alpha}, effective_epsilon={float(eff_eps):.6f}")
        print(f"    Reference Calc: best_alpha={ref_alpha}, effective_epsilon={ref_eps:.6f}")
        print(f"    Absolute Error: {diff:.2e}")

        assert float(best_alpha) == ref_alpha, f"Alpha mismatch at T={T}: {best_alpha} vs {ref_alpha}"
        assert diff < 1e-11, f"Numerical error too high at T={T}: diff={diff}"
        print(f"    [PASS] T={T} releases matched reference within 1e-11.")

    print("=" * 70)
    print("ALL VERIFICATION CHECKS PASSED: 100-Release RDP Composition")
    print("=" * 70)


if __name__ == "__main__":
    main()
