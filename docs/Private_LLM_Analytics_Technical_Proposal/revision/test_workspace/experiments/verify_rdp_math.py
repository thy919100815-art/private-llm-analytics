"""Verification script: RDP Mathematical Foundations & Conversion (v0.2).

Verifies that:
  1. Gaussian RDP matches exact formula: alpha * Delta^2 / (2 * sigma^2).
  2. Gaussian RDP scales strictly linearly with alpha.
  3. Gaussian RDP decreases monotonically with noise scale sigma.
  4. Order-wise additive composition matches sum_t RDP_t(alpha).
  5. Conversion to (epsilon, delta) correctly finds minimum across candidate orders.
  6. Orders <= 1.0 and delta outside (0, 1) fail closed.
  7. Conservative pure-DP bound for Laplace mechanism holds.
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
    compute_laplace_rdp_bound,
    compose_rdp_vectors,
    rdp_to_epsilon_delta,
    validate_delta,
    validate_orders,
)


def main():
    print("=" * 70)
    print("VERIFICATION: Rényi Differential Privacy (RDP) Mathematical Foundations")
    print("=" * 70)

    # 1. Exact Gaussian formula check
    print("[1] Verifying exact Gaussian RDP formula...")
    orders = [2.0, 4.0, 8.0]
    sensitivity = 1.0
    sigma = 2.0
    rdp_vec = compute_gaussian_rdp(sigma=sigma, sensitivity=sensitivity, orders=orders)
    assert rdp_vec[Decimal("2.0")] == Decimal("0.25"), f"Expected 0.25, got {rdp_vec[Decimal('2.0')]}"
    assert rdp_vec[Decimal("4.0")] == Decimal("0.50"), f"Expected 0.50, got {rdp_vec[Decimal('4.0')]}"
    assert rdp_vec[Decimal("8.0")] == Decimal("1.00"), f"Expected 1.00, got {rdp_vec[Decimal('8.0')]}"
    print("    [PASS] Exact Gaussian RDP formula verified at orders 2, 4, 8.")

    # 2. Linear scaling with alpha
    print("[2] Verifying linear scaling with order alpha...")
    orders_full = DEFAULT_RDP_ORDERS
    rdp_full = compute_gaussian_rdp(sigma=3.0, sensitivity=1.5, orders=orders_full)
    base_cost = rdp_full[Decimal("2.0")] / Decimal("2.0")
    for alpha in orders_full:
        dec_alpha = Decimal(str(alpha))
        expected_cost = base_cost * dec_alpha
        diff = abs(rdp_full[dec_alpha] - expected_cost)
        assert diff < Decimal("1e-20"), f"Mismatch at alpha={alpha}: diff={diff}"
    print("    [PASS] Exact linear order scaling verified across all candidate orders.")

    # 3. Monotonicity with sigma
    print("[3] Verifying strict inverse monotonicity with noise scale sigma...")
    rdp_s1 = compute_gaussian_rdp(sigma=1.0, sensitivity=1.0, orders=orders_full)
    rdp_s2 = compute_gaussian_rdp(sigma=2.0, sensitivity=1.0, orders=orders_full)
    for alpha in orders_full:
        dec_a = Decimal(str(alpha))
        assert rdp_s2[dec_a] < rdp_s1[dec_a]
    print("    [PASS] Strict decrease of RDP with larger sigma verified.")

    # 4. Additive composition
    print("[4] Verifying additive composition across releases...")
    v1 = compute_gaussian_rdp(sigma=2.0, sensitivity=1.0, orders=orders_full)
    v2 = compute_gaussian_rdp(sigma=2.5, sensitivity=1.0, orders=orders_full)
    composed = compose_rdp_vectors([v1, v2])
    for alpha in orders_full:
        dec_a = Decimal(str(alpha))
        assert composed[dec_a] == v1[dec_a] + v2[dec_a]
    print("    [PASS] Additive composition sum_t RDP_t(alpha) verified.")

    # 5. Conversion to (epsilon, delta)
    print("[5] Verifying conversion to (epsilon, delta) and best_alpha selection...")
    delta_global = 1e-6
    best_alpha, eff_eps = rdp_to_epsilon_delta(composed, delta_global=delta_global)
    ln_inv_delta = math.log(1.0 / delta_global)
    for alpha in orders_full:
        dec_a = Decimal(str(alpha))
        manual_eps = float(composed[dec_a]) + ln_inv_delta / (alpha - 1.0)
        assert float(eff_eps) <= manual_eps + 1e-12
    print(f"    [PASS] Optimal conversion verified: best_alpha={best_alpha}, effective_epsilon={float(eff_eps):.4f}")

    # 6. Fail-closed on invalid parameters
    print("[6] Verifying fail-closed behavior on invalid orders and delta...")
    try:
        validate_orders([1.0, 2.0])
        raise AssertionError("Failed to reject alpha=1.0")
    except ValueError:
        pass
    try:
        validate_orders([])
        raise AssertionError("Failed to reject empty orders")
    except ValueError:
        pass
    try:
        validate_delta(0.0)
        raise AssertionError("Failed to reject delta=0.0")
    except ValueError:
        pass
    try:
        validate_delta(1.0)
        raise AssertionError("Failed to reject delta=1.0")
    except ValueError:
        pass
    print("    [PASS] Input parameter validation strictly fails closed.")

    # 7. Conservative pure-DP bound for Laplace
    print("[7] Verifying conservative pure-DP bound for Laplace...")
    laplace_bound = compute_laplace_rdp_bound(epsilon=1.0, orders=orders_full)
    for alpha in orders_full:
        assert laplace_bound[Decimal(str(alpha))] == Decimal("1.0")
    print("    [PASS] Conservative Laplace pure-DP bound verified.")

    print("=" * 70)
    print("ALL VERIFICATION CHECKS PASSED: RDP Mathematical Foundations")
    print("=" * 70)


if __name__ == "__main__":
    main()
