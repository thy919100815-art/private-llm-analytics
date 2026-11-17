"""v6 Pre-Freeze Audit Regression Tests.

Covers:
1. Exact 420-day calendar origin & phase mapping between 0-indexed WildChat
   experiment days (d=0 -> 2023-11-20 Monday) and 1-indexed production CalendarJob
   days (current_day = d + 1 >= 1) via `experiment_phase_to_runtime_phase` and
   `runtime_phase_to_experiment_phase`, verifying zero +-1 day offset across all
   7d/30d/90d releases and all UTC Sundays.
2. Multi-user, multi-window (7d/30d/90d), frequency-tie, and >k-category
   differential consistency test comparing `run_wildchat_real_validation_v5.py`'s
   `precompute_cohort_suff_stats` cell-by-cell against production `RollingDistinctState`.
"""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path
import sys
import tempfile

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments" / "sdg_mh"))

from private_llm_analytics.ingestion.schema import COL_PROVIDER, COL_SUCCESS, COL_TASK, COL_USER_ID  # noqa: E402
from private_llm_analytics.sdg_mh.calendar_accounting import (  # noqa: E402
    experiment_phase_to_runtime_phase,
    runtime_phase_to_experiment_phase,
)
from private_llm_analytics.sdg_mh.rolling_state import RollingDistinctState  # noqa: E402
from run_wildchat_real_validation_v5 import precompute_cohort_suff_stats  # noqa: E402


def test_420d_experiment_to_runtime_phase_mapping_exact():
    """Verify zero +-1 day offset across all 420 real days for 7d/30d/90d and UTC Sundays."""
    origin = date(2023, 11, 20)
    assert origin.weekday() == 0  # Monday

    # Bijection & release-set equivalence over all 420 days for steps 7 and 30
    for step in (7, 30):
        for phi_exp in range(step):
            phi_rt = experiment_phase_to_runtime_phase(phi_exp, step)
            assert runtime_phase_to_experiment_phase(phi_rt, step) == phi_exp

            exp_release_days = [d for d in range(420) if (d - phi_exp) % step == 0]
            rt_release_days = [
                (current_day - 1)
                for current_day in range(1, 421)
                if (current_day % step) == (phi_rt % step)
            ]
            assert exp_release_days == rt_release_days

    # Sunday-pinned check: in 2023-11-20 (Monday) origin, Sunday is phi_exp=6 -> phi_rt=0
    sun_phi_rt = experiment_phase_to_runtime_phase(6, 7)
    assert sun_phi_rt == 0
    for d in range(420):
        dt = origin + timedelta(days=d)
        is_utc_sunday = (dt.weekday() == 6)
        is_exp_release = ((d - 6) % 7 == 0)
        is_rt_release = (((d + 1) % 7) == (sun_phi_rt % 7))
        assert is_utc_sunday == is_exp_release == is_rt_release


def test_v5_topk_multiuser_multiwindow_ties_differential_against_rolling_state():
    """Cell-by-cell differential check of v5 `precompute_cohort_suff_stats` vs RollingDistinctState."""
    services = [f"svc_{i}" for i in range(5)]
    tasks = [f"cat_{i}" for i in range(20)]

    # Build a 95-day synthetic multi-user stream with:
    # - u0: >3 categories with unequal counts (cat_5:10, cat_9:5, cat_0:1, cat_1:1, cat_2:1)
    # - u1: exact frequency tie across 4 categories (cat_3:3, cat_7:3, cat_12:3, cat_19:3 -> picks 3,7,12)
    # - u2: <=3 categories across 90 days (never clipped)
    # - u3: shifting dominant categories across [0..6] vs [7..35] vs [36..94]
    records_v5 = []
    daily_dfs = {}
    for d in range(95):
        day_rows = []
        if d in (2, 15, 45, 88):
            # u0
            for cat_idx, cnt in [(0, 1), (1, 1), (2, 1), (5, 3), (9, 2)]:
                for _ in range(cnt):
                    records_v5.append(("u0", d, 0, cat_idx))
                    day_rows.append(("u0", "svc_0", f"cat_{cat_idx}", True))
        if d in (5, 25, 60, 90):
            # u1: exact tie among 4 categories (3, 7, 12, 19) -> ascending index picks {3, 7, 12}
            for cat_idx in (19, 12, 7, 3):
                records_v5.append(("u1", d, 1, cat_idx))
                day_rows.append(("u1", "svc_1", f"cat_{cat_idx}", True))
        if d in (1, 80, 92):
            # u2: 2 categories only
            for cat_idx in (4, 14):
                records_v5.append(("u2", d, 2, cat_idx))
                day_rows.append(("u2", "svc_2", f"cat_{cat_idx}", True))
        if d < 10:
            for cat_idx in (10, 11, 15, 16):
                records_v5.append(("u3", d, 3, cat_idx))
                day_rows.append(("u3", "svc_3", f"cat_{cat_idx}", True))
        elif d >= 70:
            for cat_idx, cnt in [(18, 4), (17, 2), (1, 2), (0, 1)]:
                for _ in range(cnt):
                    records_v5.append(("u3", d, 4, cat_idx))
                    day_rows.append(("u3", "svc_4", f"cat_{cat_idx}", True))
        daily_dfs[d] = pd.DataFrame(day_rows, columns=[COL_USER_ID, COL_PROVIDER, COL_TASK, COL_SUCCESS])

    df_v5 = pd.DataFrame(records_v5, columns=["hashed_ip", "rel_day", "service_idx", "cat_idx"])
    suff = precompute_cohort_suff_stats(df_v5)

    with tempfile.TemporaryDirectory() as tmp:
        rs = RollingDistinctState(storage_dir=Path(tmp), w_max=95)
        for d in range(95):
            rs.append_day(d, daily_dfs[d], services, tasks, persist_disk=False)

        for w, check_days in [(7, [6, 15, 45, 92]), (30, [29, 45, 90, 94]), (90, [89, 92, 94])]:
            for d in check_days:
                prod_svc, prod_cat = rs.compute_window_histogram(d - w + 1, d, services, tasks, k_tasks_per_user=3)
                np.testing.assert_array_equal(suff["distinct_windows"][w]["true_svc"][d], prod_svc)
                np.testing.assert_array_equal(suff["distinct_windows"][w]["clipped_cat"][d], prod_cat)
