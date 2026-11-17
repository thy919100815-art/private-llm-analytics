"""Banded / Toeplitz Matrix Factorization Competitive Baseline Benchmark (v5).

Addresses all P0 Banded-MF audit items:
1. True day-by-day causal online streaming (`reset_stream` + `step_stream`) using
   O(b) ring buffers instead of batch `C_mat @ x` / `C_inv @ y`.
2. Identical hardware, timing boundary, and data scale measurement for BOTH
   SDG-MH (daily Discrete Gaussian sealing) and Banded-MF (b=30, b=8) — zero
   hardcoded timings. Reports per-step latency (us), 420d cumulative streaming
   time (ms), offline preprocessing time (ms), tracemalloc peak memory (KB),
   theoretical ring-buffer state (B), and dense calibration matrix memory (KB).
3. Evaluates identical [1, 420] remaining additive Multi-Horizon envelope
   rem_add_env(h), identical clipping (C_r=50, C_f=10, eta=25/26), and identical
   window-level post-processing across K=8 data seeds x M=5 CRN trials (df=7),
   with paired t-tests and Holm-Bonferroni correction against SDG-MH.
4. Writes all outputs to `results/sdg_mh/banded_mf_baseline_v5/` without touching
   frozen v1/v2/v3 data or v4 artifacts.
"""

from __future__ import annotations

import ctypes
import ctypes.wintypes
import math
from pathlib import Path
import sys
import time
import tracemalloc
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from scipy import stats

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

OUT_DIR = ROOT / "results" / "sdg_mh" / "banded_mf_baseline_v5"
DATA_SEEDS = (42, 1042, 2042, 3042, 4042, 5042, 6042, 7042)
N_MC_TRIALS = 5
N_DAYS = 420
N_USERS = 2000
C_R = 50.0
C_F = 10.0


def get_process_rss_bytes() -> int:
    """Read current Windows process WorkingSetSize (RSS) via psapi/kernel32."""
    class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
        _fields_ = [
            ("cb", ctypes.wintypes.DWORD),
            ("PageFaultCount", ctypes.wintypes.DWORD),
            ("PeakWorkingSetSize", ctypes.c_size_t),
            ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t),
            ("PeakPagefileUsage", ctypes.c_size_t),
        ]

    pmc = PROCESS_MEMORY_COUNTERS()
    pmc.cb = ctypes.sizeof(PROCESS_MEMORY_COUNTERS)
    handle = ctypes.windll.kernel32.GetCurrentProcess()
    ctypes.windll.psapi.GetProcessMemoryInfo(handle, ctypes.byref(pmc), pmc.cb)
    return int(pmc.WorkingSetSize)


def derive_crn_seed(data_seed: int, mc_trial: int, day: int, stream_id: int) -> int:
    return (int(data_seed) << 96) | (int(mc_trial) << 64) | (int(day) << 32) | int(stream_id)


def _rolling_window_sums(arr: np.ndarray, w: int, start_day: int, end_day: int) -> np.ndarray:
    cs = np.cumsum(arr)
    days = np.arange(start_day, end_day + 1)
    return cs[days] - cs[days - w]


def postprocess_window_pair(win_req: np.ndarray, win_fail: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    hat_r = np.maximum(0.0, win_req)
    hat_f = np.minimum(hat_r, np.maximum(0.0, win_fail))
    return hat_r, hat_f


def holm_bonferroni(p_vals: List[float]) -> List[float]:
    m = len(p_vals)
    order = np.argsort(p_vals)
    adj = [0.0] * m
    running = 0.0
    for r, idx in enumerate(order):
        val = min(1.0, p_vals[idx] * (m - r))
        running = max(running, val)
        adj[idx] = float(running)
    return adj


def run_banded_mf_v5() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Measure SDG-MH offline 7-orbit planning time directly
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

    # 2. Construct and calibrate Banded-MF (b=30, alpha=0.50) and (b=8, alpha=0.25)
    variants = [
        ("Banded-MF (b=30, MH-RemEnv)", 30, 0.50, 51),
        ("Banded-MF (b=8, MH-RemEnv)", 8, 0.25, 52),
    ]
    mf_models: Dict[str, Dict[str, Any]] = {}
    for label, bw, alpha, st_id in variants:
        tracemalloc.start()
        t0 = time.perf_counter()
        mf = BandedToeplitzBaseline(T=N_DAYS, bandwidth=bw, alpha=alpha, C=1.0)
        cal = mf.calibrate_to_multi_horizon_envelope(rem_add_env, n_days=N_DAYS)
        setup_ms = (time.perf_counter() - t0) * 1000.0
        _, setup_peak_bytes = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        mem_bd = mf.get_memory_breakdown_bytes(n_channels=2)
        mf_models[label] = {
            "model": mf,
            "cal": cal,
            "setup_ms": setup_ms,
            "setup_peak_kb": setup_peak_bytes / 1024.0,
            "mem_bd": mem_bd,
            "stream_id": st_id,
            "sigma_iso": cal["beta_env"] * iso_sens_eff,
        }

    # 3. Evaluate SDG-MH and both Banded-MF models day-by-day across K=8 seeds x 5 CRN trials
    raw_rows: List[Dict[str, Any]] = []
    rss_before = get_process_rss_bytes()

    for ds in DATA_SEEDS:
        gen = SyntheticTelemetryGenerator(n_initial_users=N_USERS, n_years=N_DAYS / 365.0, seed=ds)
        daily_raw_req = np.zeros(N_DAYS + 1, dtype=np.float64)
        daily_clip_req = np.zeros(N_DAYS + 1, dtype=np.float64)
        daily_raw_fail = np.zeros(N_DAYS + 1, dtype=np.float64)
        daily_clip_fail = np.zeros(N_DAYS + 1, dtype=np.float64)

        for day, df_day in gen.stream_daily_telemetry(start_day=1, end_day=N_DAYS):
            if not df_day.empty:
                rc = df_day[COL_USER_ID].value_counts().astype(float)
                daily_raw_req[day] = float(rc.sum())
                daily_clip_req[day] = float(np.minimum(rc, C_R).sum())
                fdf = df_day[~df_day[COL_SUCCESS]]
                if not fdf.empty:
                    fc = fdf[COL_USER_ID].value_counts().reindex(rc.index, fill_value=0.0).astype(float)
                    daily_raw_fail[day] = float(fc.sum())
                    daily_clip_fail[day] = float(np.minimum(fc, C_F).sum())

        raw_roll_req = {w: _rolling_window_sums(daily_raw_req, w, w, N_DAYS) for w in (1, 7, 30, 90)}
        raw_roll_fail = {w: _rolling_window_sums(daily_raw_fail, w, w, N_DAYS) for w in (1, 7, 30, 90)}
        x_req = daily_clip_req[1:]
        x_fail = daily_clip_fail[1:]

        for trial in range(N_MC_TRIALS):
            # 3a. SDG-MH true day-by-day Discrete Gaussian sealing loop
            tracemalloc.start()
            t_sdg0 = time.perf_counter()
            est_sdg_r = np.zeros(N_DAYS + 1, dtype=np.float64)
            est_sdg_f = np.zeros(N_DAYS + 1, dtype=np.float64)
            for t in range(N_DAYS):
                rng_d = np.random.default_rng(derive_crn_seed(ds, trial, t + 1, 1))
                n_r = sample_discrete_gaussian(sigma_sdg_r, rng=rng_d)
                n_f = sample_discrete_gaussian(sigma_sdg_f, rng=rng_d)
                est_sdg_r[t + 1] = x_req[t] + float(n_r)
                est_sdg_f[t + 1] = x_fail[t] + float(n_f)
            sdg_exec_ms = (time.perf_counter() - t_sdg0) * 1000.0
            _, sdg_exec_peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()

            m_r, m_f = {}, {}
            for w_add in (1, 7, 30, 90):
                win_r, win_f = postprocess_window_pair(
                    _rolling_window_sums(est_sdg_r, w_add, w_add, N_DAYS),
                    _rolling_window_sums(est_sdg_f, w_add, w_add, N_DAYS),
                )
                m_r[w_add] = float(np.sqrt(np.mean(np.square(win_r - raw_roll_req[w_add]))))
                m_f[w_add] = float(np.sqrt(np.mean(np.square(win_f - raw_roll_fail[w_add]))))

            raw_rows.append({
                "data_seed": ds,
                "mc_trial": trial,
                "method": "SDG-MH (D_Joint, Timed)",
                "privacy_horizon_scope": "Infinite h >= 1 (eval [1,420])",
                "calibration_scope": "Exact Multi-Horizon commitment (h >= 1)",
                "requests_1d_rmse": m_r[1],
                "requests_7d_rmse": m_r[7],
                "requests_30d_rmse": m_r[30],
                "requests_90d_rmse": m_r[90],
                "failures_1d_rmse": m_f[1],
                "failures_7d_rmse": m_f[7],
                "failures_30d_rmse": m_f[30],
                "failures_90d_rmse": m_f[90],
                "setup_ms": sdg_setup_ms,
                "exec_ms": sdg_exec_ms,
                "per_step_us": (sdg_exec_ms * 1000.0) / N_DAYS,
                "online_peak_tracemalloc_kb": sdg_exec_peak / 1024.0,
                "theoretical_ring_buffer_bytes": 16,
                "dense_decoder_c_inv_kb": 0.0,
                "offline_calib_matrices_kb": 0.0,
                "naive_h1_violation_at_365": 1.0,
            })

            # 3b. Banded-MF true day-by-day causal online streaming loop
            for label, info in mf_models.items():
                mf: BandedToeplitzBaseline = info["model"]
                cal = info["cal"]
                mem_bd = info["mem_bd"]
                sig_iso = info["sigma_iso"]
                st_id = info["stream_id"]

                rng = np.random.default_rng(derive_crn_seed(ds, trial, 0, st_id))
                noises = rng.normal(0.0, sig_iso, size=(N_DAYS, 2))

                tracemalloc.start()
                t_exec0 = time.perf_counter()
                mf.reset_stream(n_channels=2)
                est_r_arr = np.zeros(N_DAYS + 1, dtype=np.float64)
                est_f_arr = np.zeros(N_DAYS + 1, dtype=np.float64)
                for t in range(N_DAYS):
                    x_t = np.array([x_req[t], x_fail[t]], dtype=np.float64)
                    xhat_t = mf.step_stream(x_t, noises[t])
                    est_r_arr[t + 1] = xhat_t[0]
                    est_f_arr[t + 1] = xhat_t[1]
                exec_ms = (time.perf_counter() - t_exec0) * 1000.0
                _, exec_peak = tracemalloc.get_traced_memory()
                tracemalloc.stop()

                m_r, m_f = {}, {}
                for w_add in (1, 7, 30, 90):
                    win_r, win_f = postprocess_window_pair(
                        _rolling_window_sums(est_r_arr, w_add, w_add, N_DAYS),
                        _rolling_window_sums(est_f_arr, w_add, w_add, N_DAYS),
                    )
                    m_r[w_add] = float(np.sqrt(np.mean(np.square(win_r - raw_roll_req[w_add]))))
                    m_f[w_add] = float(np.sqrt(np.mean(np.square(win_f - raw_roll_fail[w_add]))))

                raw_rows.append({
                    "data_seed": ds,
                    "mc_trial": trial,
                    "method": label,
                    "privacy_horizon_scope": f"Finite Toeplitz T=420 (b={mf.bandwidth}, alpha={mf.alpha:.2f})",
                    "calibration_scope": f"Exact Multi-Horizon rem_add_envelope (tight_h={cal['tight_h_env']})",
                    "requests_1d_rmse": m_r[1],
                    "requests_7d_rmse": m_r[7],
                    "requests_30d_rmse": m_r[30],
                    "requests_90d_rmse": m_r[90],
                    "failures_1d_rmse": m_f[1],
                    "failures_7d_rmse": m_f[7],
                    "failures_30d_rmse": m_f[30],
                    "failures_90d_rmse": m_f[90],
                    "setup_ms": info["setup_ms"],
                    "exec_ms": exec_ms,
                    "per_step_us": (exec_ms * 1000.0) / N_DAYS,
                    "online_peak_tracemalloc_kb": exec_peak / 1024.0,
                    "theoretical_ring_buffer_bytes": mem_bd["theoretical_ring_buffer_bytes"],
                    "dense_decoder_c_inv_kb": mem_bd["dense_decoder_c_inv_bytes"] / 1024.0,
                    "offline_calib_matrices_kb": mem_bd["offline_calibration_all_matrices_bytes"] / 1024.0,
                    "naive_h1_violation_at_365": cal["naive_h1_violation_ratio_at_365"],
                })

    rss_after = get_process_rss_bytes()
    df_raw = pd.DataFrame(raw_rows)
    df_raw.to_csv(OUT_DIR / "banded_mf_competitive_raw_v5.csv", index=False)

    # 4. Seed-level summary (df=7)
    metric_cols = [
        "requests_1d_rmse",
        "requests_7d_rmse",
        "requests_30d_rmse",
        "requests_90d_rmse",
        "failures_1d_rmse",
        "failures_7d_rmse",
        "failures_30d_rmse",
        "failures_90d_rmse",
    ]
    summary_rows: List[Dict[str, Any]] = []
    t_crit = float(stats.t.ppf(0.975, df=len(DATA_SEEDS) - 1))
    for method_name, sub in df_raw.groupby("method", sort=False):
        seed_means = sub.groupby("data_seed", as_index=False)[metric_cols].mean(numeric_only=True)
        for m in metric_cols:
            vals = seed_means[m].to_numpy(dtype=np.float64)
            mean_val = float(np.mean(vals))
            sd_val = float(np.std(vals, ddof=1))
            ci95_val = float(t_crit * sd_val / math.sqrt(len(vals)))
            summary_rows.append({
                "method": method_name,
                "privacy_horizon_scope": str(sub["privacy_horizon_scope"].iloc[0]),
                "calibration_scope": str(sub["calibration_scope"].iloc[0]),
                "metric": m,
                "K_seeds": len(DATA_SEEDS),
                "M_mc_per_seed": N_MC_TRIALS,
                "mean": round(mean_val, 4),
                "seed_level_sd": round(sd_val, 4),
                "seed_level_ci95": round(ci95_val, 4),
                "setup_ms": round(float(sub["setup_ms"].mean()), 2),
                "online_exec_420d_ms": round(float(sub["exec_ms"].mean()), 3),
                "per_step_us": round(float(sub["per_step_us"].mean()), 2),
                "online_peak_tracemalloc_kb": round(float(sub["online_peak_tracemalloc_kb"].mean()), 2),
                "theoretical_ring_buffer_bytes": int(sub["theoretical_ring_buffer_bytes"].iloc[0]),
                "dense_decoder_c_inv_kb": round(float(sub["dense_decoder_c_inv_kb"].iloc[0]), 1),
                "offline_calib_matrices_kb": round(float(sub["offline_calib_matrices_kb"].iloc[0]), 1),
                "naive_h1_violation_at_365": round(float(sub["naive_h1_violation_at_365"].iloc[0]), 2),
            })

    df_sum = pd.DataFrame(summary_rows)
    df_sum.to_csv(OUT_DIR / "banded_mf_competitive_summary_v5.csv", index=False)

    # 5. Paired tests: SDG-MH (D_Joint, Timed) vs Banded-MF (b=8) and (b=30) across all 8 metrics (df=7)
    sdg_seed = df_raw[df_raw["method"] == "SDG-MH (D_Joint, Timed)"].groupby("data_seed")[metric_cols].mean()
    paired_rows: List[Dict[str, Any]] = []
    for base_label in ("Banded-MF (b=8, MH-RemEnv)", "Banded-MF (b=30, MH-RemEnv)"):
        base_seed = df_raw[df_raw["method"] == base_label].groupby("data_seed")[metric_cols].mean()
        raw_pvals = []
        tmp = []
        for m in metric_cols:
            diffs = sdg_seed[m].to_numpy() - base_seed[m].to_numpy()
            mean_diff = float(np.mean(diffs))
            se_diff = float(np.std(diffs, ddof=1) / math.sqrt(len(diffs)))
            ci_half = t_crit * se_diff
            _, p_t = stats.ttest_rel(sdg_seed[m].to_numpy(), base_seed[m].to_numpy())
            _, p_w = stats.wilcoxon(diffs)
            raw_pvals.append(float(p_t))
            tmp.append({
                "contrast": f"SDG-MH - {base_label}",
                "metric": m,
                "sdg_mean": round(float(sdg_seed[m].mean()), 2),
                "mf_mean": round(float(base_seed[m].mean()), 2),
                "paired_diff": round(mean_diff, 2),
                "ci95_half": round(ci_half, 2),
                "p_t_raw": float(p_t),
                "p_wilcoxon": float(p_w),
                "seeds_sdg_lt_mf": int(np.sum(diffs < 0)),
            })
        holm_p = holm_bonferroni(raw_pvals)
        for i, row in enumerate(tmp):
            row["p_holm"] = holm_p[i]
            paired_rows.append(row)

    df_paired = pd.DataFrame(paired_rows)
    df_paired.to_csv(OUT_DIR / "banded_mf_paired_vs_sdg_v5.csv", index=False)

    # 6. Systems overhead summary (authentic measurements only; process RSS delta logged)
    sys_df = df_sum.drop_duplicates(subset=["method"])[
        [
            "method",
            "privacy_horizon_scope",
            "setup_ms",
            "online_exec_420d_ms",
            "per_step_us",
            "online_peak_tracemalloc_kb",
            "theoretical_ring_buffer_bytes",
            "dense_decoder_c_inv_kb",
            "offline_calib_matrices_kb",
            "naive_h1_violation_at_365",
        ]
    ].copy()
    sys_df["process_rss_delta_mb"] = round((rss_after - rss_before) / (1024.0 * 1024.0), 2)
    sys_df.to_csv(OUT_DIR / "banded_mf_systems_overhead_v5.csv", index=False)

    print("=== Banded-MF v5 Summary (True Streaming + Identical Timing Boundary) ===")
    print(df_sum[["method", "metric", "mean", "seed_level_ci95", "online_exec_420d_ms", "per_step_us"]].to_string(index=False))
    print("\n=== Banded-MF v5 Paired Tests vs SDG-MH (df=7) ===")
    print(df_paired.to_string(index=False))
    print("\n=== Banded-MF v5 Systems Overhead ===")
    print(sys_df.to_string(index=False))


if __name__ == "__main__":
    run_banded_mf_v5()
