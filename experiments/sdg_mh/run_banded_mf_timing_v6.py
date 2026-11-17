"""P0 Banded-MF vs SDG-MH Dual-Regime Timing Boundary Audit (v6).

Strictly isolates and measures two apples-to-apples timing regimes for the
420-day ONLINE ADDITIVE STREAMING OPERATOR ONLY (explicitly excluding window
post-processing, UnifiedPrivacyLedger admission checks, and RollingDistinctState
/ CalendarJob distinct-user tracking):

  - Regime (A) Total Online Additive Operator (including per-day noise sampling):
      * SDG-MH: per-day `sample_discrete_gaussian` + additive sealing `x_t + z_t`.
      * Banded-MF: per-day `rng.normal` + causal `step_stream(x_t, z_t)`.
  - Regime (B) Kernel-Only Cost (pre-generated noise excluded from both timers):
      * SDG-MH: `x_t + z_t` per day using pre-sampled Discrete Gaussian array.
      * Banded-MF: `step_stream(x_t, z_t)` per day using pre-sampled Gaussian array.

Also reports explicit 4-way memory taxonomy:
  1. Theoretical minimal ring-buffer state (16 B vs 224 B / 928 B)
  2. Actual Python online `tracemalloc` peak (KB)
  3. 90-day `RollingDistinctState` history cache (separate distinct subsystem, N/A for Banded-MF)
  4. Offline dense matrix storage (`C_inv` 1378.1 KB & calibration matrices 6897.2 KB vs 0.0 KB).

Outputs saved to `results/sdg_mh/banded_mf_timing_audit_v6/` without touching v1-v5.
"""

from __future__ import annotations

import math
from pathlib import Path
import sys
import time
import tracemalloc
from typing import Any, Dict, List

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from private_llm_analytics.ingestion.schema import COL_SUCCESS, COL_USER_ID  # noqa: E402
from private_llm_analytics.sdg_mh.baselines import BandedToeplitzBaseline  # noqa: E402
from private_llm_analytics.sdg_mh.calendar_accounting import WEEK1_V2_CURVE  # noqa: E402
from private_llm_analytics.sdg_mh.discrete_gaussian import sample_discrete_gaussian  # noqa: E402
from private_llm_analytics.sdg_mh.joint_planner import (  # noqa: E402
    JointCalendarPlanner,
    PlannerConfig,
)
from private_llm_analytics.sdg_mh.synthetic_generator import SyntheticTelemetryGenerator  # noqa: E402

OUT_DIR = ROOT / "results" / "sdg_mh" / "banded_mf_timing_audit_v6"
DATA_SEEDS = (42, 1042, 2042, 3042, 4042, 5042, 6042, 7042)
N_MC_TRIALS = 5
N_DAYS = 420
N_USERS = 2000
C_R = 50.0
C_F = 10.0


def derive_crn_seed(data_seed: int, mc_trial: int, day: int, stream_id: int) -> int:
    return (int(data_seed) << 96) | (int(mc_trial) << 64) | (int(day) << 32) | int(stream_id)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    t_plan0 = time.perf_counter()
    planner = JointCalendarPlanner(
        PlannerConfig(commitment=WEEK1_V2_CURVE, workload_weights=(0.40, 0.25, 0.25, 0.10))
    )
    plan = planner.plan_group_d()
    sdg_setup_ms = (time.perf_counter() - t_plan0) * 1000.0

    phi7, phi30, phi90 = plan.phases
    l_dist = np.max(
        planner.n7_tables[phi7] * plan.distinct_costs[0]
        + planner.n30_tables[phi30] * plan.distinct_costs[1]
        + planner.n90_tables[phi90] * plan.distinct_costs[2],
        axis=1,
    )[:N_DAYS]
    rem_add_env = np.maximum(1e-12, planner.rho_bars[:N_DAYS] - l_dist)
    iso_sens_eff = math.sqrt(C_R**2 + C_F**2)
    eta = (C_R**2) / (C_R**2 + C_F**2)
    sigma_sdg_r = C_R / math.sqrt(2.0 * eta * plan.daily_d)
    sigma_sdg_f = C_F / math.sqrt(2.0 * (1.0 - eta) * plan.daily_d)

    variants = [
        ("Banded-MF (b=8, MH-RemEnv)", 8, 0.25, 52),
        ("Banded-MF (b=30, MH-RemEnv)", 30, 0.50, 51),
    ]
    mf_models: Dict[str, Dict[str, Any]] = {}
    for label, bw, alpha, st_id in variants:
        t0 = time.perf_counter()
        mf = BandedToeplitzBaseline(T=N_DAYS, bandwidth=bw, alpha=alpha, C=1.0)
        cal = mf.calibrate_to_multi_horizon_envelope(rem_add_env, n_days=N_DAYS)
        setup_ms = (time.perf_counter() - t0) * 1000.0
        mf_models[label] = {
            "model": mf,
            "cal": cal,
            "setup_ms": setup_ms,
            "mem_bd": mf.get_memory_breakdown_bytes(n_channels=2),
            "stream_id": st_id,
            "sigma_iso": cal["beta_env"] * iso_sens_eff,
        }

    records: List[Dict[str, Any]] = []
    for ds in DATA_SEEDS:
        gen = SyntheticTelemetryGenerator(n_initial_users=N_USERS, n_years=N_DAYS / 365.0, seed=ds)
        x_req = np.zeros(N_DAYS, dtype=np.float64)
        x_fail = np.zeros(N_DAYS, dtype=np.float64)
        for day, df_day in gen.stream_daily_telemetry(start_day=1, end_day=N_DAYS):
            if not df_day.empty:
                rc = df_day[COL_USER_ID].value_counts().astype(float)
                x_req[day - 1] = float(np.minimum(rc, C_R).sum())
                fdf = df_day[~df_day[COL_SUCCESS]]
                if not fdf.empty:
                    fc = fdf[COL_USER_ID].value_counts().reindex(rc.index, fill_value=0.0).astype(float)
                    x_fail[day - 1] = float(np.minimum(fc, C_F).sum())

        for trial in range(N_MC_TRIALS):
            # -----------------------------------------------------------------
            # SDG-MH Regime (A): Including per-day Discrete Gaussian sampling
            # -----------------------------------------------------------------
            tracemalloc.start()
            t0 = time.perf_counter()
            y_a = np.zeros((N_DAYS, 2), dtype=np.float64)
            for t in range(N_DAYS):
                rng_d = np.random.default_rng(derive_crn_seed(ds, trial, t + 1, 1))
                n_r = float(sample_discrete_gaussian(sigma_sdg_r, rng=rng_d))
                n_f = float(sample_discrete_gaussian(sigma_sdg_f, rng=rng_d))
                y_a[t, 0] = x_req[t] + n_r
                y_a[t, 1] = x_fail[t] + n_f
            sdg_reg_a_ms = (time.perf_counter() - t0) * 1000.0
            _, sdg_peak_a = tracemalloc.get_traced_memory()
            tracemalloc.stop()

            # -----------------------------------------------------------------
            # SDG-MH Regime (B): Kernel-Only (pre-generated noise excluded)
            # -----------------------------------------------------------------
            pre_sdg_noise = np.zeros((N_DAYS, 2), dtype=np.float64)
            for t in range(N_DAYS):
                rng_d = np.random.default_rng(derive_crn_seed(ds, trial, t + 1, 1))
                pre_sdg_noise[t, 0] = float(sample_discrete_gaussian(sigma_sdg_r, rng=rng_d))
                pre_sdg_noise[t, 1] = float(sample_discrete_gaussian(sigma_sdg_f, rng=rng_d))

            tracemalloc.start()
            t0 = time.perf_counter()
            y_b = np.zeros((N_DAYS, 2), dtype=np.float64)
            for t in range(N_DAYS):
                y_b[t, 0] = x_req[t] + pre_sdg_noise[t, 0]
                y_b[t, 1] = x_fail[t] + pre_sdg_noise[t, 1]
            sdg_reg_b_ms = (time.perf_counter() - t0) * 1000.0
            _, sdg_peak_b = tracemalloc.get_traced_memory()
            tracemalloc.stop()

            np.testing.assert_array_equal(y_a, y_b)
            records.append({
                "data_seed": ds,
                "mc_trial": trial,
                "method": "SDG-MH (Additive Daily Operator)",
                "regime_a_with_sampling_420d_ms": sdg_reg_a_ms,
                "regime_b_kernel_only_420d_ms": sdg_reg_b_ms,
                "regime_a_per_step_us": (sdg_reg_a_ms * 1000.0) / N_DAYS,
                "regime_b_per_step_us": (sdg_reg_b_ms * 1000.0) / N_DAYS,
                "tracemalloc_peak_kb": sdg_peak_a / 1024.0,
                "theoretical_ring_buffer_bytes": 16,
                "dense_decoder_c_inv_kb": 0.0,
                "offline_calib_matrices_kb": 0.0,
                "offline_setup_ms": sdg_setup_ms,
            })

            # -----------------------------------------------------------------
            # Banded-MF Regime (A) & Regime (B)
            # -----------------------------------------------------------------
            for label, info in mf_models.items():
                mf: BandedToeplitzBaseline = info["model"]
                mem_bd = info["mem_bd"]
                sig_iso = info["sigma_iso"]
                st_id = info["stream_id"]

                # Regime (A): per-day RNG + Gaussian sampling inside 420d loop
                tracemalloc.start()
                t0 = time.perf_counter()
                mf.reset_stream(n_channels=2)
                out_a = np.zeros((N_DAYS, 2), dtype=np.float64)
                for t in range(N_DAYS):
                    rng_t = np.random.default_rng(derive_crn_seed(ds, trial, t + 1, st_id))
                    z_t = rng_t.normal(0.0, sig_iso, size=2)
                    x_t = np.array([x_req[t], x_fail[t]], dtype=np.float64)
                    out_a[t] = mf.step_stream(x_t, z_t)
                mf_reg_a_ms = (time.perf_counter() - t0) * 1000.0
                _, mf_peak_a = tracemalloc.get_traced_memory()
                tracemalloc.stop()

                # Regime (B): pre-generated noise excluded from 420d kernel loop
                pre_mf_noise = np.zeros((N_DAYS, 2), dtype=np.float64)
                for t in range(N_DAYS):
                    rng_t = np.random.default_rng(derive_crn_seed(ds, trial, t + 1, st_id))
                    pre_mf_noise[t] = rng_t.normal(0.0, sig_iso, size=2)

                tracemalloc.start()
                t0 = time.perf_counter()
                mf.reset_stream(n_channels=2)
                out_b = np.zeros((N_DAYS, 2), dtype=np.float64)
                for t in range(N_DAYS):
                    x_t = np.array([x_req[t], x_fail[t]], dtype=np.float64)
                    out_b[t] = mf.step_stream(x_t, pre_mf_noise[t])
                mf_reg_b_ms = (time.perf_counter() - t0) * 1000.0
                _, mf_peak_b = tracemalloc.get_traced_memory()
                tracemalloc.stop()

                np.testing.assert_allclose(out_a, out_b, rtol=1e-12, atol=1e-12)
                records.append({
                    "data_seed": ds,
                    "mc_trial": trial,
                    "method": label,
                    "regime_a_with_sampling_420d_ms": mf_reg_a_ms,
                    "regime_b_kernel_only_420d_ms": mf_reg_b_ms,
                    "regime_a_per_step_us": (mf_reg_a_ms * 1000.0) / N_DAYS,
                    "regime_b_per_step_us": (mf_reg_b_ms * 1000.0) / N_DAYS,
                    "tracemalloc_peak_kb": mf_peak_a / 1024.0,
                    "theoretical_ring_buffer_bytes": mem_bd["theoretical_ring_buffer_bytes"],
                    "dense_decoder_c_inv_kb": mem_bd["dense_decoder_c_inv_bytes"] / 1024.0,
                    "offline_calib_matrices_kb": mem_bd["offline_calibration_all_matrices_bytes"] / 1024.0,
                    "offline_setup_ms": info["setup_ms"],
                })

    df_raw = pd.DataFrame(records)
    df_raw.to_csv(OUT_DIR / "banded_mf_timing_raw_v6.csv", index=False)

    summary = (
        df_raw.groupby("method", sort=False)
        .agg(
            regime_a_with_sampling_420d_ms=("regime_a_with_sampling_420d_ms", "mean"),
            regime_b_kernel_only_420d_ms=("regime_b_kernel_only_420d_ms", "mean"),
            regime_a_per_step_us=("regime_a_per_step_us", "mean"),
            regime_b_per_step_us=("regime_b_per_step_us", "mean"),
            tracemalloc_peak_kb=("tracemalloc_peak_kb", "mean"),
            theoretical_ring_buffer_bytes=("theoretical_ring_buffer_bytes", "first"),
            dense_decoder_c_inv_kb=("dense_decoder_c_inv_kb", "first"),
            offline_calib_matrices_kb=("offline_calib_matrices_kb", "first"),
            offline_setup_ms=("offline_setup_ms", "mean"),
        )
        .reset_index()
    )
    summary["timing_scope"] = "Online additive streaming operator only (excludes post-proc, ledger, and RollingDistinctState)"
    summary["distinct_90d_cache"] = [
        "Separate RollingDistinctState subsystem (required for distinct queries)",
        "N/A (Additive only)",
        "N/A (Additive only)",
    ]
    summary.to_csv(OUT_DIR / "banded_mf_timing_summary_v6.csv", index=False)

    print("=== Dual-Regime Timing Summary (v6) ===")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
