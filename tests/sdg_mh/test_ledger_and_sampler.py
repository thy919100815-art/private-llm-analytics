"""Unit tests for UnifiedPrivacyLedger, Discrete Gaussian sampler, and Closed-Form Profiles."""

import math
import numpy as np
import pytest

from private_llm_analytics.sdg_mh.discrete_gaussian import sample_discrete_gaussian
from private_llm_analytics.sdg_mh.ledger import (
    UnifiedPrivacyLedger,
    closed_form_distinct_calendar_rho,
    closed_form_sdg_additive_rho,
    combine_inverse_variance,
    epsilon_to_rho,
    rho_to_epsilon,
)


def test_discrete_gaussian_statistical_properties():
    """Verify Canonne-Kamath-Steinke 2020 discrete Gaussian moments and reproducibility."""
    sigma = 30.0
    seed = 42001
    n_samples = 40000

    samples = sample_discrete_gaussian(sigma=sigma, size=n_samples, seed=seed)
    assert isinstance(samples, np.ndarray)
    assert samples.shape == (n_samples,)
    assert np.issubdtype(samples.dtype, np.integer)

    mean = float(np.mean(samples))
    var = float(np.var(samples))
    target_var = sigma ** 2

    # Mean within 3 standard errors of 0: 3 * (sigma / sqrt(N)) = 3 * (30 / 200) = 0.45
    assert abs(mean) < 0.45, f"Discrete Gaussian mean {mean} deviates significantly from 0"
    # Relative variance error < 3%
    rel_var_err = abs(var - target_var) / target_var
    assert rel_var_err < 0.03, f"Discrete Gaussian variance {var} deviates from target {target_var}"


def test_discrete_gaussian_reproducibility():
    """Verify that same seed generates bit-for-bit identical discrete Gaussian samples."""
    samples_a = sample_discrete_gaussian(sigma=50.0, size=500, seed=12345)
    samples_b = sample_discrete_gaussian(sigma=50.0, size=500, seed=12345)
    np.testing.assert_array_equal(samples_a, samples_b)


def test_closed_form_sdg_additive_exact_match():
    """Section 2.4 requirement: Ledger numerical profile for SDG must exactly match h * rho_1_add."""
    C_r, sigma_r = 50.0, 2500.0
    C_f, sigma_f = 10.0, 700.0

    # rho_1^add = C_r^2 / (2 * sigma_r^2) + C_f^2 / (2 * sigma_f^2)
    rho_1_add = (C_r ** 2) / (2.0 * sigma_r ** 2) + (C_f ** 2) / (2.0 * sigma_f ** 2)

    ledger = UnifiedPrivacyLedger(delta=1e-6)
    n_days = 100
    for t in range(1, n_days + 1):
        # Day t release
        ledger.record_gaussian_release(
            release_id=f"sdg_req_day_{t}",
            days_accessed=[t],
            sensitivity=C_r,
            sigma=sigma_r,
            tag="requests",
        )
        ledger.record_gaussian_release(
            release_id=f"sdg_fail_day_{t}",
            days_accessed=[t],
            sensitivity=C_f,
            sigma=sigma_f,
            tag="failures",
        )

    for h in [1, 7, 14, 30, 90]:
        exact_closed = closed_form_sdg_additive_rho(h, rho_1_add)
        ledger_rho = ledger.compute_rho(h)
        assert abs(ledger_rho - exact_closed) < 1e-12, (
            f"Horizon h={h}: ledger rho {ledger_rho} != closed-form {exact_closed}"
        )


def test_closed_form_distinct_calendar_exact_match():
    """Section 2.4 requirement: Ledger numerical profile for Distinct Calendar must exactly match closed-form formula."""
    calendar_b = [(7, 7, 60.0), (30, 7, 60.0), (90, 30, 60.0)]
    delta_sq = 11.0  # m=5, k=3 => delta^2 = 5 + 6 = 11
    sens = math.sqrt(delta_sq)

    ledger = UnifiedPrivacyLedger(delta=1e-6)
    # Simulate a 180-day execution of Calendar B
    for w, s, sigma_w in calendar_b:
        for e in range(s, 181, s):
            window_days = list(range(max(1, e - w + 1), e + 1))
            ledger.record_gaussian_release(
                release_id=f"calendar_{w}_{s}_day_{e}",
                days_accessed=window_days,
                sensitivity=sens,
                sigma=sigma_w,
                tag=f"calendar_{w}_{s}",
            )

    for h in [1, 7, 30, 90]:
        exact_closed = closed_form_distinct_calendar_rho(h, calendar_b, delta_sq=delta_sq)
        ledger_rho = ledger.compute_rho(h)
        assert abs(ledger_rho - exact_closed) < 1e-12, (
            f"Horizon h={h}: ledger rho {ledger_rho} != closed-form {exact_closed}"
        )


def test_revisit_gate_accept_and_reject():
    """Verify that Revisit Gate strictly blocks releases exceeding bound E_bar(h)."""
    ledger = UnifiedPrivacyLedger(delta=1e-6)
    # Day 1-10 committed
    for t in range(1, 11):
        ledger.record_gaussian_release(
            release_id=f"base_day_{t}",
            days_accessed=[t],
            sensitivity=10.0,
            sigma=100.0,
        )

    # Set threshold E_bar
    current_prof = ledger.get_profile(horizons=[1, 7, 30])
    # Set cap at current + 0.05
    max_bound = {h: current_prof[h] + 0.05 for h in [1, 7, 30]}

    # Small candidate release: should pass
    cand_small = [
        ledger.record_gaussian_release(
            release_id="cand_small",
            days_accessed=[5, 6],
            sensitivity=1.0,
            sigma=100.0,
        )
    ]
    # Remove from ledger to test gate
    ledger.records.pop()

    allowed, reason, proj = ledger.check_revisit_gate(cand_small, max_bound)
    assert allowed, f"Should be allowed but got {reason}"

    # Large candidate release with tiny sigma: should be rejected
    cand_large = [
        ledger.record_gaussian_release(
            release_id="cand_large",
            days_accessed=[5],
            sensitivity=10.0,
            sigma=5.0,  # Huge rho = 100 / 50 = 2.0
        )
    ]
    ledger.records.pop()

    allowed_large, reason_large, _ = ledger.check_revisit_gate(cand_large, max_bound)
    assert not allowed_large
    assert "exceeds limit E_bar" in reason_large


def test_inverse_variance_weighting():
    """Verify optimal inverse-variance weighted merge formula."""
    theta1, var1 = 100.0, 4.0
    theta2, var2 = 120.0, 1.0  # Weight ratio is 1:4

    merged_theta, merged_var = combine_inverse_variance(theta1, var1, theta2, var2)
    expected_theta = (100.0 * 0.25 + 120.0 * 1.0) / 1.25  # 116.0
    expected_var = 1.0 / 1.25  # 0.8
    assert abs(merged_theta - expected_theta) < 1e-9
    assert abs(merged_var - expected_var) < 1e-9
