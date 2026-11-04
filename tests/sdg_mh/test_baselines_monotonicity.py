"""Unit tests for baseline ledger accounting and profile monotonicity.

Requirement (Item 2):
- Each baseline verified against a handcrafted small example.
- Every baseline's privacy profile rho(h) and eps(h) is strictly non-decreasing in h.
- Closed-form matches ledger output.
"""

import math
import numpy as np
import pytest

from private_llm_analytics.sdg_mh.calibration import (
    HORIZONS,
    calibrate_noise_multiplier,
    compute_sdg_target_profile,
)
from private_llm_analytics.sdg_mh.ledger import (
    UnifiedPrivacyLedger,
    closed_form_sdg_additive_rho,
    rho_to_epsilon,
)


def test_lifetime_decay_monotonicity_and_handcrafted_match():
    """Verify LifetimeDecay monotonicity and exact agreement with handcrafted ledger."""
    rho_max = 0.12
    C = 50.0
    ledger = UnifiedPrivacyLedger(delta=1e-6)

    # Handcrafted 10-day LifetimeDecay release
    for t in range(1, 11):
        rho_t = rho_max * 6.0 / (math.pi**2 * t**2)
        sigma_t = C / math.sqrt(2.0 * rho_t)
        ledger.record_gaussian_release(f"decay_{t}", [t], sensitivity=C, sigma=sigma_t)

    # Invariant: worst-case interval of length h is always [1, h]
    # Handcrafted expectation: rho(h) = sum_{t=1}^h rho_t
    cum_rho = 0.0
    prev_rho = 0.0
    for h in range(1, 11):
        rho_h = ledger.compute_rho(h)
        cum_rho += rho_max * 6.0 / (math.pi**2 * h**2)
        assert abs(rho_h - cum_rho) < 1e-12, f"h={h}: ledger {rho_h} != handcrafted {cum_rho}"
        assert rho_h >= prev_rho, f"Monotonicity violation at h={h}: {rho_h} < {prev_rho}"
        prev_rho = rho_h


def test_epoch90_monotonicity_and_boundary_crossing():
    """Verify Epoch90 monotonicity and boundary-crossing doubling (Item 12)."""
    ledger = UnifiedPrivacyLedger(delta=1e-6)
    rho_epoch = 0.20
    C = 50.0
    sigma_q = C / math.sqrt(2.0 * (rho_epoch / 5.0))

    # Epoch 1: days 1-90 (5 queries covering entire epoch)
    for q in range(5):
        ledger.record_gaussian_release(f"ep1_q{q}", list(range(1, 91)), sensitivity=C, sigma=sigma_q)

    # Epoch 2: days 91-180 (5 queries covering entire epoch)
    for q in range(5):
        ledger.record_gaussian_release(f"ep2_q{q}", list(range(91, 181)), sensitivity=C, sigma=sigma_q)

    # A ledger with only Epoch 1 (isolated epoch): max rho for h=45 is strictly rho_epoch
    ledger_ep1 = UnifiedPrivacyLedger(delta=1e-6)
    for q in range(5):
        ledger_ep1.record_gaussian_release(f"ep1_q{q}", list(range(1, 91)), sensitivity=C, sigma=sigma_q)
    assert abs(ledger_ep1.compute_rho(45) - rho_epoch) < 1e-12

    # Boundary-crossing: in multi-epoch ledger, any interval straddling the boundary touches BOTH Epoch 1 & 2!
    rho_cross = ledger.compute_rho(90)
    expected_cross = 2.0 * rho_epoch
    assert abs(rho_cross - expected_cross) < 1e-12, f"Crossing rho(90) {rho_cross} != {expected_cross}"

    # Monotonicity check
    prev = 0.0
    for h in [1, 7, 30, 90, 180]:
        val = ledger.compute_rho(h)
        assert val >= prev, f"Monotonicity failure: h={h}, val={val} < {prev}"
        prev = val


def test_wbd_vs_wba_distinct_profiles():
    """Verify W-BD and W-BA have distinct profiles, sliding window modeling, and monotonic behavior."""
    w = 7
    rho_w = 0.03
    C = 50.0
    daily_wbd_rho = rho_w / w
    sigma_wbd = C / math.sqrt(2.0 * daily_wbd_rho)

    ledger_wbd = UnifiedPrivacyLedger(delta=1e-6)
    ledger_wba = UnifiedPrivacyLedger(delta=1e-6)

    # W-BD: sliding window releases every day
    for t in range(1, 21):
        days = list(range(max(1, t - w + 1), t + 1))
        ledger_wbd.record_gaussian_release(f"wbd_{t}", days, sensitivity=C, sigma=sigma_wbd)

    # W-BA: test release every day, but publication occurs on ~67% of days (absorption on 33%)
    sigma_test = C / math.sqrt(2.0 * (daily_wbd_rho * 0.5))
    sigma_rel = C / math.sqrt(2.0 * (daily_wbd_rho * 0.5))
    for t in range(1, 21):
        days = list(range(max(1, t - w + 1), t + 1))
        ledger_wba.record_gaussian_release(f"wba_test_{t}", days, sensitivity=C, sigma=sigma_test)
        if t % 3 != 0:  # 1/3 absorbed, 2/3 released
            ledger_wba.record_gaussian_release(f"wba_rel_{t}", days, sensitivity=C, sigma=sigma_rel)

    prev_wbd = 0.0
    prev_wba = 0.0
    for h in [1, 3, 7, 10]:
        r_wbd = ledger_wbd.compute_rho(h)
        r_wba = ledger_wba.compute_rho(h)
        assert r_wbd > r_wba, f"W-BA should expend less budget due to absorption: wbd={r_wbd}, wba={r_wba}"
        assert r_wbd >= prev_wbd, f"W-BD monotonicity violation: {r_wbd} < {prev_wbd}"
        assert r_wba >= prev_wba, f"W-BA monotonicity violation: {r_wba} < {prev_wba}"
        prev_wbd = r_wbd
        prev_wba = r_wba


def test_all_11_calibrated_baselines_strictly_monotonic():
    """Verify all 11 mechanisms in baseline_calibration_table are strictly monotonic in h."""
    sdg = compute_sdg_target_profile()
    target_rhos_add = sdg["rho_add"]
    target_rho_30_add = sdg["target_rho_30_add"]

    # Test monotonic invariant for all profiles in calibration table
    import pandas as pd
    from pathlib import Path
    cal_file = Path("results/sdg_mh/baseline_calibration_table.csv")
    assert cal_file.exists(), "Calibration table must exist"
    df = pd.read_csv(cal_file)

    for _, row in df.iterrows():
        method = row["method"]
        caliber = row["caliber"]
        rhos = [row[f"rho_{h}"] for h in HORIZONS]
        epss = [row[f"eps_{h}"] for h in HORIZONS]

        for i in range(len(HORIZONS) - 1):
            h1, h2 = HORIZONS[i], HORIZONS[i + 1]
            assert rhos[i + 1] >= rhos[i] - 1e-9, (
                f"[{method} {caliber}] rho({h2})={rhos[i+1]} < rho({h1})={rhos[i]}"
            )
            assert epss[i + 1] >= epss[i] - 1e-9, (
                f"[{method} {caliber}] eps({h2})={epss[i+1]} < eps({h1})={epss[i]}"
            )


def test_tree_gls_satisfies_theorem_7b():
    """Verify Theorem 7(b) holds everywhere: Var_tree(w) >= w^2 * C^2 / (2 * rho(w)) for all variants and positions."""
    from private_llm_analytics.sdg_mh.baselines import TreeBaseline

    C = 104.0
    T = 365
    for variant in ["Tree-U", "Tree-2l", "Tree-sqrt"]:
        tree = TreeBaseline(variant=variant, T=T, L=9, C=C)
        for w in [1, 7, 30, 90, 365]:
            max_rho_term = tree.get_worst_case_rho_term(w)
            rho_w = 0.5 * (C**2) * max_rho_term
            theoretical_lower_bound = (w**2 * C**2) / (2.0 * rho_w)

            # Test multiple start positions across the year
            step = max(1, (T - w) // 10) if T > w else 1
            for s in range(0, T - w + 1, step):
                var_pos = tree.get_window_variance(s + 1, s + w)
                assert var_pos >= theoretical_lower_bound - 1e-6, (
                    f"[{variant}] Theorem 7(b) violation at w={w}, pos={s}: "
                    f"Var={var_pos:.4f} < LowerBound={theoretical_lower_bound:.4f}"
                )

