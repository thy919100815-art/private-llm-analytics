"""v5 Audit Regression Tests: constructed first as failing tests against PR #11 (01ae905).

Covers:
1. WildChat date origin & weekday mapping: 2023-11-20 is Monday (weekday=0), so
   d%7==3 is Thursday (not Sunday), and actual Sunday is d%7==6 (weekday=6).
2. WildChat top-k=3 category truncation: must match RollingDistinctState
   (-in_window_count, category_idx), NOT sorted(cset)[:3].
3. BandedToeplitzBaseline true day-by-day online causal streaming & reconstruction
   using only a b-length ring buffer (no full C_mat @ x or C_inv @ y batch multiply).
4. BandedToeplitzBaseline small-matrix exact sensitivity check & calibration
   perturbation regression (below beta_env must fail multi-horizon check).
"""

from __future__ import annotations

from datetime import date, timedelta
import importlib.util
from pathlib import Path
import sys
import tempfile

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from private_llm_analytics.ingestion.schema import COL_PROVIDER, COL_SUCCESS, COL_TASK, COL_USER_ID  # noqa: E402
from private_llm_analytics.sdg_mh.baselines import BandedToeplitzBaseline  # noqa: E402
from private_llm_analytics.sdg_mh.rolling_state import RollingDistinctState  # noqa: E402


def test_wildchat_v4_weekday_origin_bug_exposed():
    """2023-11-20 is Monday; d % 7 == 3 is Thursday, and actual Sunday is d % 7 == 6."""
    origin = date(2023, 11, 20)
    assert origin.weekday() == 0  # Monday

    # In v4, rel_day=3 was treated as Sunday via `if d % 7 == 3:`
    day_3 = origin + timedelta(days=3)
    assert day_3.weekday() == 3  # Thursday, NOT Sunday!
    day_6 = origin + timedelta(days=6)
    assert day_6.weekday() == 6  # Actual Sunday!

    # Verify phi=0 in [2023-11-20..] is Monday, whereas Sunday is phi=6 (mod 7)
    assert (origin + timedelta(days=0)).weekday() == 0
    assert (origin + timedelta(days=6)).weekday() == 6


def test_v4_topk_truncation_diverges_from_production_rolling_state():
    """Differential test exposing v4's sorted(cset)[:3] bug vs RollingDistinctState (-count, idx)."""
    services = [f"svc_{i}" for i in range(5)]
    tasks = [f"cat_{i}" for i in range(20)]

    # User u0 has 5 distinct categories in a 7-day window with unequal frequencies:
    # cat_0: 1, cat_1: 1, cat_2: 1, cat_5: 10, cat_9: 5.
    # Production RollingDistinctState top-3 MUST pick {5, 9, 0} (counts 10, 5, 1),
    # whereas v4's sorted(cset)[:3] picked {0, 1, 2}!
    rows = (
        [("u0", "svc_0", "cat_0", True)] * 1
        + [("u0", "svc_0", "cat_1", True)] * 1
        + [("u0", "svc_0", "cat_2", True)] * 1
        + [("u0", "svc_0", "cat_5", True)] * 10
        + [("u0", "svc_0", "cat_9", True)] * 5
    )
    df_day = pd.DataFrame(rows, columns=[COL_USER_ID, COL_PROVIDER, COL_TASK, COL_SUCCESS])

    with tempfile.TemporaryDirectory() as tmp:
        rs = RollingDistinctState(storage_dir=Path(tmp), w_max=90)
        rs.append_day(0, df_day, services, tasks, persist_disk=False)
        _, prod_cats = rs.compute_window_histogram(0, 0, services, tasks, k_tasks_per_user=3)

    expected = np.zeros(20, dtype=np.float64)
    expected[[5, 9, 0]] = 1.0
    np.testing.assert_array_equal(prod_cats, expected)

    # Confirm v4's logic fails this check
    cset = {0, 1, 2, 5, 9}
    v4_picked = sorted(cset)[:3]
    assert v4_picked == [0, 1, 2]
    assert set(v4_picked) != {5, 9, 0}


def test_banded_mf_requires_streaming_step_and_exact_small_matrix_sensitivity():
    """BandedToeplitzBaseline must expose causal day-by-day streaming matching batch C_mat/C_inv."""
    bmf = BandedToeplitzBaseline(T=6, bandwidth=3, alpha=0.5, C=2.0)

    # Exact small-matrix brute-force sensitivity check for T=6, b=3, h in 1..6:
    # Worst-case (u, I) perturbation over contiguous interval I of length h sets x_t = +-C for t in I.
    for h in range(1, 7):
        max_l2_sq = 0.0
        for s in range(0, 7 - h):
            vec = np.zeros(6, dtype=np.float64)
            vec[s : s + h] = bmf.C
            y_pert = bmf.C_mat @ vec
            l2_sq = float(np.sum(y_pert**2))
            if l2_sq > max_l2_sq:
                max_l2_sq = l2_sq
        rho_exact = 0.5 * max_l2_sq
        rho_from_method = (bmf.C**2) * bmf.get_worst_case_rho_term(h)
        assert abs(rho_exact - rho_from_method) < 1e-12

    # Require true causal day-by-day streaming method (fails on 01ae905 before implementation)
    assert hasattr(bmf, "reset_stream") and hasattr(bmf, "step_stream"), (
        "BandedToeplitzBaseline must implement day-by-day causal reset_stream() and step_stream()"
    )

    rng = np.random.default_rng(12345)
    x = rng.uniform(10.0, 100.0, size=6)
    z = rng.normal(0.0, 5.0, size=6)

    y_batch = bmf.C_mat @ x + z
    x_hat_batch = bmf.C_inv @ y_batch

    bmf.reset_stream()
    x_hat_stream = np.zeros(6, dtype=np.float64)
    for t in range(6):
        x_hat_stream[t] = bmf.step_stream(float(x[t]), float(z[t]))

    np.testing.assert_allclose(x_hat_stream, x_hat_batch, rtol=1e-11, atol=1e-11)


def test_banded_mf_calibration_perturbation_violates_envelope():
    """Perturbing calibrated beta_env downward by 1% must violate rem_add_envelope at tight_h."""
    T = 420
    rem_env = np.array([0.0005 * h + 0.01 for h in range(1, T + 1)], dtype=np.float64)
    bmf = BandedToeplitzBaseline(T=T, bandwidth=30, alpha=0.50, C=1.0)
    cal = bmf.calibrate_to_multi_horizon_envelope(rem_env, n_days=T)
    beta_env = cal["beta_env"]
    tight_h = cal["tight_h_env"]

    # Exactly calibrated beta_env satisfies every h in [1, T]
    for h in range(1, T + 1):
        assert bmf.get_worst_case_rho_term(h) / (beta_env**2) <= rem_env[h - 1] + 1e-12

    # Perturbing downward by 1% violates at tight_h
    beta_bad = 0.99 * beta_env
    assert bmf.get_worst_case_rho_term(tight_h) / (beta_bad**2) > rem_env[tight_h - 1]
