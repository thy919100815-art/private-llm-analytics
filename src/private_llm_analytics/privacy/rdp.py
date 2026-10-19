"""Rényi Differential Privacy (RDP) Mathematical Foundations & Composition Engine.

Provides exact Decimal-safe RDP calculation for Gaussian mechanisms,
conservative pure-DP bounds for Laplace mechanisms, order-wise additive
composition, and conversion to (epsilon, delta)-DP.
"""

from decimal import Decimal, getcontext
import json
import math
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

# Ensure high precision for Decimal calculations (minimum 50 digits)
getcontext().prec = 60

DEFAULT_RDP_ORDERS: List[float] = [
    1.25, 1.5, 2.0, 3.0, 4.0, 5.0, 8.0, 10.0, 16.0, 32.0, 64.0, 128.0
]


def to_decimal(val: Any) -> Decimal:
    """Helper to convert numeric types to exact Decimal with string mediation."""
    if isinstance(val, Decimal):
        return val
    return Decimal(str(val))


def validate_orders(orders: Sequence[Union[float, int, Decimal]]) -> List[Decimal]:
    """Validate that RDP orders are non-empty and strictly greater than 1.0."""
    if not orders:
        raise ValueError("RDP orders cannot be empty.")
    validated = []
    for alpha in orders:
        dec_alpha = to_decimal(alpha)
        if dec_alpha <= Decimal("1.0"):
            raise ValueError(f"RDP order alpha must be strictly greater than 1.0, got {alpha}")
        validated.append(dec_alpha)
    return validated


def validate_delta(delta: Union[float, Decimal]) -> Decimal:
    """Validate that accountant delta is strictly in (0, 1)."""
    dec_delta = to_decimal(delta)
    if dec_delta <= Decimal("0.0") or dec_delta >= Decimal("1.0"):
        raise ValueError(f"accountant_delta_global must be strictly between 0 and 1, got {delta}")
    return dec_delta


def compute_gaussian_rdp(
    sigma: Union[float, Decimal],
    sensitivity: Union[float, Decimal],
    orders: Sequence[Union[float, Decimal]],
) -> Dict[Decimal, Decimal]:
    """Compute Gaussian mechanism RDP vector across orders alpha.

    For Gaussian mechanism with L2 sensitivity Delta and standard deviation sigma:
        RDP(alpha) = (alpha * Delta^2) / (2 * sigma^2)

    Args:
        sigma: Actual calibrated noise scale / standard deviation used in the release.
        sensitivity: L2 sensitivity Delta of the query.
        orders: Sequence of Rényi orders alpha > 1.

    Returns:
        Dict mapping order Decimal to exact RDP cost Decimal.
    """
    dec_sigma = to_decimal(sigma)
    dec_sens = to_decimal(sensitivity)
    if dec_sigma <= Decimal("0.0"):
        raise ValueError(f"Gaussian noise scale sigma must be > 0, got {sigma}")
    if dec_sens <= Decimal("0.0"):
        raise ValueError(f"L2 sensitivity must be > 0, got {sensitivity}")

    validated_orders = validate_orders(orders)
    variance_term = Decimal("2.0") * (dec_sigma * dec_sigma)
    sens_sq = dec_sens * dec_sens

    rdp_vector: Dict[Decimal, Decimal] = {}
    for alpha in validated_orders:
        cost = (alpha * sens_sq) / variance_term
        rdp_vector[alpha] = cost

    return rdp_vector


def compute_laplace_rdp_bound(
    epsilon: Union[float, Decimal],
    orders: Sequence[Union[float, Decimal]],
) -> Dict[Decimal, Decimal]:
    """Compute conservative pure-DP-to-RDP upper bound for Laplace mechanism.

    For any pure epsilon-DP mechanism (e.g. standard Laplace), Rényi divergence
    of any order alpha > 1 is upper-bounded by epsilon:
        RDP(alpha) <= epsilon  for all alpha > 1

    Note: This is an explicit conservative upper bound, NOT exact Laplace RDP.
    """
    dec_eps = to_decimal(epsilon)
    if dec_eps <= Decimal("0.0"):
        raise ValueError(f"Laplace epsilon must be > 0, got {epsilon}")

    validated_orders = validate_orders(orders)
    return {alpha: dec_eps for alpha in validated_orders}


def compose_rdp_vectors(
    rdp_vectors: Sequence[Dict[Decimal, Decimal]],
) -> Dict[Decimal, Decimal]:
    """Compose multiple RDP vectors by summing RDP costs order-wise.

    In RDP, composition across independent releases is additive for each order alpha:
        RDP_total(alpha) = sum_t RDP_t(alpha)

    Preconditions:
        - All vectors must share identical orders.
        - Summation must be performed independently per alpha before conversion.
    """
    if not rdp_vectors:
        return {}

    orders = list(rdp_vectors[0].keys())
    for idx, v in enumerate(rdp_vectors[1:], start=1):
        if set(v.keys()) != set(orders):
            raise ValueError(
                f"Cannot compose RDP vectors with mismatched orders at index {idx}: "
                f"expected {orders}, got {list(v.keys())}"
            )

    total: Dict[Decimal, Decimal] = {alpha: Decimal("0.0") for alpha in orders}
    for v in rdp_vectors:
        for alpha in orders:
            total[alpha] += to_decimal(v[alpha])

    return total


def rdp_to_epsilon_delta(
    rdp_vector: Dict[Decimal, Decimal],
    delta_global: Union[float, Decimal],
) -> Tuple[Decimal, Decimal]:
    """Convert cumulative RDP vector to effective (epsilon, delta)-DP.

    Standard RDP conversion (Mironov, 2017, Proposition 3):
        epsilon(alpha) = RDP(alpha) + ln(1 / delta_global) / (alpha - 1)
        epsilon_effective = min_alpha epsilon(alpha)

    Args:
        rdp_vector: Map from order alpha to cumulative RDP cost.
        delta_global: Policy-level target delta parameter.

    Returns:
        Tuple of (best_alpha, effective_epsilon) as exact Decimals.
    """
    if not rdp_vector:
        raise ValueError("Cannot convert empty RDP vector.")

    dec_delta = validate_delta(delta_global)
    # Compute ln(1 / delta) using high precision math
    # float(dec_delta) is safe for log since delta is typically 1e-15 to 1e-3
    ln_inv_delta = Decimal(str(math.log(1.0 / float(dec_delta))))

    best_alpha: Optional[Decimal] = None
    min_eps: Optional[Decimal] = None

    for alpha, rdp_cost in rdp_vector.items():
        dec_alpha = to_decimal(alpha)
        dec_cost = to_decimal(rdp_cost)
        if dec_alpha <= Decimal("1.0"):
            raise ValueError(f"Invalid order alpha={alpha} in conversion")

        denom = dec_alpha - Decimal("1.0")
        eps_alpha = dec_cost + (ln_inv_delta / denom)

        if min_eps is None or eps_alpha < min_eps:
            min_eps = eps_alpha
            best_alpha = dec_alpha

    assert min_eps is not None and best_alpha is not None
    return best_alpha, min_eps


def rdp_vector_to_json(rdp_vector: Dict[Decimal, Decimal]) -> str:
    """Serialize RDP vector into deterministic JSON string of stringified Decimals."""
    serializable = {str(k): str(v) for k, v in sorted(rdp_vector.items(), key=lambda x: x[0])}
    return json.dumps(serializable, sort_keys=True)


def rdp_vector_from_json(json_str: str) -> Dict[Decimal, Decimal]:
    """Deserialize JSON string into Dict[Decimal, Decimal]."""
    raw = json.loads(json_str)
    return {to_decimal(k): to_decimal(v) for k, v in raw.items()}
