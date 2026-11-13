"""Unit tests for Rényi Differential Privacy (RDP) Math and Composition (v0.2)."""

from decimal import Decimal
import math
import pytest

from private_llm_analytics.config.loader import load_app_config
from private_llm_analytics.config.validator import validate_production_config
from private_llm_analytics.privacy.exceptions import ProductionSafetyConfigurationError
from private_llm_analytics.privacy.rdp import (
    DEFAULT_RDP_ORDERS,
    compute_gaussian_rdp,
    compute_laplace_rdp_bound,
    compose_rdp_vectors,
    rdp_to_epsilon_delta,
    rdp_vector_from_json,
    rdp_vector_to_json,
    to_decimal,
    validate_delta,
    validate_orders,
)
from private_llm_analytics.privacy.rdp_accountant import RDPAccountant


def test_gaussian_rdp_single_order():
    """Verify exact formula: alpha * Delta^2 / (2 * sigma^2)."""
    orders = [2.0]
    sensitivity = 1.0
    sigma = 2.0
    # Expected: 2.0 * 1.0^2 / (2.0 * 2.0^2) = 2.0 / 8.0 = 0.25
    rdp_vec = compute_gaussian_rdp(sigma=sigma, sensitivity=sensitivity, orders=orders)
    assert Decimal("2.0") in rdp_vec
    assert rdp_vec[Decimal("2.0")] == Decimal("0.25")


def test_gaussian_rdp_multiple_orders():
    """Verify Gaussian RDP scales linearly with alpha across default orders."""
    sensitivity = 1.2
    sigma = 3.0
    orders = DEFAULT_RDP_ORDERS
    rdp_vec = compute_gaussian_rdp(sigma=sigma, sensitivity=sensitivity, orders=orders)

    assert len(rdp_vec) == len(orders)
    # Check ratio: RDP(alpha_2) / RDP(alpha_1) == alpha_2 / alpha_1
    val_2 = rdp_vec[Decimal("2.0")]
    val_4 = rdp_vec[Decimal("4.0")]
    assert abs(float(val_4 / val_2) - 2.0) < 1e-12

    val_8 = rdp_vec[Decimal("8.0")]
    assert abs(float(val_8 / val_2) - 4.0) < 1e-12


def test_rdp_additive_composition():
    """Verify additive composition: RDP_total(alpha) = sum_t RDP_t(alpha)."""
    orders = [1.5, 2.0, 4.0]
    v1 = compute_gaussian_rdp(sigma=2.0, sensitivity=1.0, orders=orders)
    v2 = compute_gaussian_rdp(sigma=3.0, sensitivity=1.5, orders=orders)

    composed = compose_rdp_vectors([v1, v2])

    for alpha in [Decimal("1.5"), Decimal("2.0"), Decimal("4.0")]:
        expected = v1[alpha] + v2[alpha]
        assert composed[alpha] == expected


def test_rdp_to_eps_delta():
    """Verify epsilon conversion: epsilon(alpha) = RDP(alpha) + ln(1/delta) / (alpha - 1)."""
    orders = [2.0]
    delta = 1e-6
    # Let RDP(2.0) = 0.5
    rdp_vec = {Decimal("2.0"): Decimal("0.5")}
    best_alpha, eff_eps = rdp_to_epsilon_delta(rdp_vec, delta_global=delta)

    expected_ln = math.log(1e6)  # ln(1 / 1e-6)
    expected_eps = 0.5 + expected_ln / (2.0 - 1.0)
    assert best_alpha == Decimal("2.0")
    assert abs(float(eff_eps) - expected_eps) < 1e-9


def test_best_alpha_selection():
    """Verify best_alpha achieves the minimum epsilon among the candidate orders."""
    sensitivity = 1.0
    sigma = 1.5
    orders = DEFAULT_RDP_ORDERS
    delta = 1e-6
    rdp_vec = compute_gaussian_rdp(sigma=sigma, sensitivity=sensitivity, orders=orders)

    best_alpha, min_eps = rdp_to_epsilon_delta(rdp_vec, delta_global=delta)

    # Re-evaluate all alphas manually to confirm min_eps is the true minimum
    for alpha in orders:
        dec_alpha = Decimal(str(alpha))
        cost = float(rdp_vec[dec_alpha])
        eps_alpha = cost + math.log(1.0 / delta) / (alpha - 1.0)
        assert float(min_eps) <= eps_alpha + 1e-12


def test_rdp_monotonic_with_sigma():
    """Verify larger noise sigma strictly decreases RDP cost for all orders."""
    sensitivity = 1.0
    orders = DEFAULT_RDP_ORDERS
    rdp_small_sigma = compute_gaussian_rdp(sigma=1.0, sensitivity=sensitivity, orders=orders)
    rdp_large_sigma = compute_gaussian_rdp(sigma=2.5, sensitivity=sensitivity, orders=orders)

    for alpha in orders:
        dec_alpha = Decimal(str(alpha))
        assert rdp_large_sigma[dec_alpha] < rdp_small_sigma[dec_alpha]


def test_rdp_composition_monotonic_with_release_count():
    """Verify composing more releases monotonically increases RDP cost and effective epsilon."""
    orders = DEFAULT_RDP_ORDERS
    delta = 1e-6
    single_rel = compute_gaussian_rdp(sigma=2.0, sensitivity=1.0, orders=orders)

    prev_eps = Decimal("0.0")
    for t in [1, 5, 20, 50]:
        composed = compose_rdp_vectors([single_rel] * t)
        _, eff_eps = rdp_to_epsilon_delta(composed, delta_global=delta)
        assert eff_eps > prev_eps
        prev_eps = eff_eps


def test_invalid_alpha_rejected():
    """Verify orders <= 1.0 or empty orders are strictly rejected."""
    with pytest.raises(ValueError, match="strictly greater than 1.0"):
        validate_orders([1.0, 2.0])

    with pytest.raises(ValueError, match="strictly greater than 1.0"):
        validate_orders([0.5, 1.5])

    with pytest.raises(ValueError, match="cannot be empty"):
        validate_orders([])


def test_invalid_delta_rejected():
    """Verify delta outside (0, 1) is strictly rejected."""
    with pytest.raises(ValueError, match="between 0 and 1"):
        validate_delta(0.0)

    with pytest.raises(ValueError, match="between 0 and 1"):
        validate_delta(1.0)

    with pytest.raises(ValueError, match="between 0 and 1"):
        validate_delta(-1e-5)


def test_laplace_rdp_conservative_bound():
    """Verify conservative pure-DP bound for Laplace mechanism."""
    orders = [1.25, 2.0, 5.0, 10.0]
    eps = 0.75
    rdp_bound = compute_laplace_rdp_bound(epsilon=eps, orders=orders)
    for alpha in orders:
        assert rdp_bound[Decimal(str(alpha))] == Decimal("0.75")


def test_rdp_serialization_roundtrip():
    """Verify deterministic JSON serialization and deserialization of RDP vector."""
    orders = [1.5, 2.0, 8.0]
    rdp_vec = compute_gaussian_rdp(sigma=1.8, sensitivity=0.5, orders=orders)
    json_str = rdp_vector_to_json(rdp_vec)
    reconstructed = rdp_vector_from_json(json_str)

    assert set(rdp_vec.keys()) == set(reconstructed.keys())
    for k in rdp_vec:
        assert rdp_vec[k] == reconstructed[k]


def test_100_release_reference_composition():
    """Independent numerical cross-validation for 1, 10, 100 Gaussian releases."""
    delta = 1e-6
    sensitivity = 1.0
    sigma = 1.8
    orders = DEFAULT_RDP_ORDERS

    # Single release RDP
    single_rel = compute_gaussian_rdp(sigma=sigma, sensitivity=sensitivity, orders=orders)

    for T in [1, 10, 100]:
        composed = compose_rdp_vectors([single_rel] * T)
        best_alpha, eff_eps = rdp_to_epsilon_delta(composed, delta_global=delta)

        # Independent high-precision reference calculation
        ref_min_eps = None
        ref_best_alpha = None
        ln_inv_delta = math.log(1.0 / delta)
        for alpha in orders:
            # Independent closed-form formula: T * alpha * Delta^2 / (2 * sigma^2) + ln(1/delta)/(alpha - 1)
            ref_cost = (T * alpha * (sensitivity ** 2)) / (2.0 * (sigma ** 2))
            ref_eps = ref_cost + ln_inv_delta / (alpha - 1.0)
            if ref_min_eps is None or ref_eps < ref_min_eps:
                ref_min_eps = ref_eps
                ref_best_alpha = alpha

        assert float(best_alpha) == pytest.approx(ref_best_alpha, abs=1e-9)
        assert float(eff_eps) == pytest.approx(ref_min_eps, abs=1e-9)


def test_production_fail_closed_rolling_reset():
    """Verify that production rejects rolling privacy reset fail-closed."""
    cfg = load_app_config("configs")
    # Set rolling reset
    cfg.privacy.history.mode = "rolling_reset"
    with pytest.raises(ProductionSafetyConfigurationError, match="strictly forbidden in production"):
        validate_production_config(cfg)
