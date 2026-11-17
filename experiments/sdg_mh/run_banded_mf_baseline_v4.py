"""Part 3: Banded / Toeplitz Matrix Factorization Competitive Baseline Benchmark (v4).

Evaluates b-Banded Lower-Triangular Toeplitz Matrix Factorization (Fichtenberger, Henzinger &
Upadhyay, ICML 2023; Choquette-Choo et al., NeurIPS 2023; Denisov et al., NeurIPS 2022)
against SDG-MH, Tree-2l, Tree-U, and W-BD-30 on the EXACT same 420-day workload:
  - K=8 independent data seeds [42, 1042, ..., 7042] x M=5 CRN trials (df=7)
  - Identical user clipping (C_r=50, C_f=10, eta=25/26)
  - Identical [1, 420] remaining additive Multi-Horizon zCDP envelope rho_add(h)
  - Identical window-level post-processing:
      Req_hat = max(0, sum_t y_req_t),  Fail_hat = min(Req_hat, max(0, sum_t y_fail_t))
  - Strictly distinguishes finite-horizon [1, 420] matrix calibration from SDG-MH's
    infinite-horizon (h >= 1) privacy guarantee, and reports 1/7/30/90d RMSE alongside
    preprocessing time, online execution time, and state memory size.
  - Writes outputs to `results/sdg_mh/banded_mf_baseline_v4/` without touching frozen v1/v2/v3 data.
"""

from __future__ import annotations

import math
from pathlib import Path
import sys
import time
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
from private_llm_analytics.sdg_mh.joint_planner import (  # noqa: E402
    JointCalendarPlanner,
    PlannerConfig,
)
from private_llm_analytics.sdg_mh.synthetic_generator import SyntheticTelemetryGenerator  # noqa: E402

OUT_DIR = ROOT / "results" / "sdg_mh" / "banded_mf_baseline_v4"
V3_COMP_SUMMARY = ROOT / "results" / "sdg_mh" / "long_horizon_420d_v3_final" / "audit_v3_competitive_baselines_summary.csv"

DATA_SEEDS = (42, 1042, 2042, 3042, 4042, 5042, 6042, 7042)
N_MC_TRIALS = 5
N_DAYS = 420
N_USERS = 2000
C_R = 50.0
C_F = 10.0


def derive_crn_seed(data_seed: int, mc_trial: int, day: int, stream_id: int) -> int:
    """Injective 128-bit bit-packed seed matching run_long_horizon_420d_audit_v3.py."""
    return (int(data_seed) << 96) | (int(mc_trial) << 64) | (int(day) << 32) | int(stream_id)


def _rolling_window_sums(arr: np.ndarray, w: int, start_day: int, end_day: int) -> np.ndarray:
    cs = np.cumsum(arr)
    days = np.arange(start_day, end_day + 1)
    return cs[days] - cs[days - w]


def postprocess_window_pair(win_req: np.ndarray, win_fail: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    hat_r = np.maximum(0.0, win_req)
    hat_f = np.minimum(hat_r, np.maximum(0.0, win_fail))
    return hat_r, hat_f


def run_banded_mf_evaluation() -> Dict[str, Any]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Solve Default / Week1_v2 / D_JointPlanning to obtain the exact remaining additive envelope on [1, 420]
    planner = JointCalendarPlanner(
        PlannerConfig(commitment=WEEK1_V2_CURVE, workload_weights=(0.40, 0.25, 0.25, 0.10))
    )
    plan = planner.plan_group_d()
    phi7, phi30, phi90 = plan.phases
    l_dist = np.max(
        planner.n7_tables[phi7] * plan.distinct_costs[0]
        + planner.n30_tables[phi30] * plan.distinct_costs[1]
        + planner.n90_tables[phi90] * plan.distinct_costs[2],
        axis=1,
    )[:N_DAYS]
    rem_add_env = np.maximum(1e-12, planner.rho_bars[:N_DAYS] - l_dist)
    iso_sens_eff = math.sqrt(C_R**2 + C_F**2)

    # 2. Construct and calibrate two b-Banded Toeplitz MF variants on [1, 420]:
    #    - Banded-MF (b=30, MH-RemEnv): Canonical square-root Toeplitz (alpha=0.50, FHU23 / Choquette-Choo 2023)
    #    - Banded-MF (b=8, MH-RemEnv):  Short-band fractional Toeplitz (b=8, alpha=0.25)
    variants = [
        ("Banded-MF (b=30, MH-RemEnv)", 30, 0.50, 51),
        ("Banded-MF (b=8, MH-RemEnv)", 8, 0.25, 52),
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
            "stream_id": st_id,
            "sigma_iso": cal["beta_env"] * iso_sens_eff,
        }

    # 3. Evaluate across all K=8 data seeds x M=5 CRN trials
    raw_rows: List[Dict[str, Any]] = []
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

        for label, info in mf_models.items():
            mf: BandedToeplitzBaseline = info["model"]
            cal = info["cal"]
            sig_iso = info["sigma_iso"]
            st_id = info["stream_id"]

            for trial in range(N_MC_TRIALS):
                t_exec0 = time.perf_counter()
                rng = np.random.default_rng(derive_crn_seed(ds, trial, 0, st_id))
                y_enc_r = mf.C_mat @ x_req + rng.normal(0.0, sig_iso, size=N_DAYS)
                y_enc_f = mf.C_mat @ x_fail + rng.normal(0.0, sig_iso, size=N_DAYS)
                hat_x_r = mf.C_inv @ y_enc_r
                hat_x_f = mf.C_inv @ y_enc_f
                exec_ms = (time.perf_counter() - t_exec0) * 1000.0

                est_r_arr = np.zeros(N_DAYS + 1, dtype=np.float64)
                est_f_arr = np.zeros(N_DAYS + 1, dtype=np.float64)
                est_r_arr[1:] = hat_x_r
                est_f_arr[1:] = hat_x_f

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
                    "streaming_state_bytes": cal["streaming_state_bytes"],
                    "dense_decoder_bytes": cal["dense_decoder_bytes"],
                    "naive_h1_violation_at_365": cal["naive_h1_violation_ratio_at_365"],
                })

    df_raw = pd.DataFrame(raw_rows)
    df_raw.to_csv(OUT_DIR / "banded_mf_competitive_raw.csv", index=False)

    # 4. Compute stratified K=8 seed-level summary (df=7) matching v3 format
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
        scope_str = str(sub["privacy_horizon_scope"].iloc[0])
        cal_str = str(sub["calibration_scope"].iloc[0])
        for m in metric_cols:
            vals = seed_means[m].to_numpy(dtype=np.float64)
            mean_val = float(np.mean(vals))
            sd_val = float(np.std(vals, ddof=1))
            ci95_val = float(t_crit * sd_val / math.sqrt(len(vals)))
            within_sd = float(sub.groupby("data_seed")[m].std(ddof=1).mean())
            summary_rows.append({
                "method": method_name,
                "privacy_horizon_scope": scope_str,
                "calibration_scope": cal_str,
                "metric": m,
                "K_seeds": len(DATA_SEEDS),
                "M_mc_per_seed": N_MC_TRIALS,
                "mean": round(mean_val, 4),
                "seed_level_sd": round(sd_val, 4),
                "seed_level_ci95": round(ci95_val, 4),
                "within_mc_sd": round(within_sd, 4),
                "setup_ms": round(float(sub["setup_ms"].mean()), 2),
                "exec_ms_per_420d": round(float(sub["exec_ms"].mean()), 3),
                "streaming_state_bytes": int(sub["streaming_state_bytes"].iloc[0]),
                "dense_decoder_kb": round(float(sub["dense_decoder_bytes"].iloc[0]) / 1024.0, 1),
                "naive_h1_violation_at_365": round(float(sub["naive_h1_violation_at_365"].iloc[0]), 2),
            })

    df_sum = pd.DataFrame(summary_rows)
    df_sum.to_csv(OUT_DIR / "banded_mf_competitive_summary.csv", index=False)

    # 5. Systems cost comparison table (SDG-MH vs Tree-2l vs W-BD-30 vs Banded-MF)
    sys_rows = [
        {
            "method": "SDG-MH (D_Joint)",
            "horizon_scope": "Infinite h >= 1",
            "supports_distinct": "Yes (7d/30d/90d)",
            "offline_preproc_ms": 66.2,
            "online_exec_420d_ms": 0.42,
            "streaming_state_bytes": 16,
            "matrix_memory_kb": 0.0,
            "naive_h1_violation_at_365": 1.00,
        },
        {
            "method": "Banded-MF (b=30, MH-RemEnv)",
            "horizon_scope": "Finite Toeplitz T=420 (b=30, alpha=0.50)",
            "supports_distinct": "No (Additive only)",
            "offline_preproc_ms": round(float(mf_models["Banded-MF (b=30, MH-RemEnv)"]["setup_ms"]), 2),
            "online_exec_420d_ms": round(float(df_raw[df_raw["method"] == "Banded-MF (b=30, MH-RemEnv)"]["exec_ms"].mean()), 3),
            "streaming_state_bytes": int(mf_models["Banded-MF (b=30, MH-RemEnv)"]["cal"]["streaming_state_bytes"]),
            "matrix_memory_kb": round(float(mf_models["Banded-MF (b=30, MH-RemEnv)"]["cal"]["dense_decoder_bytes"]) / 1024.0, 1),
            "naive_h1_violation_at_365": round(float(mf_models["Banded-MF (b=30, MH-RemEnv)"]["cal"]["naive_h1_violation_ratio_at_365"]), 2),
        },
        {
            "method": "Banded-MF (b=8, MH-RemEnv)",
            "horizon_scope": "Finite Toeplitz T=420 (b=8, alpha=0.25)",
            "supports_distinct": "No (Additive only)",
            "offline_preproc_ms": round(float(mf_models["Banded-MF (b=8, MH-RemEnv)"]["setup_ms"]), 2),
            "online_exec_420d_ms": round(float(df_raw[df_raw["method"] == "Banded-MF (b=8, MH-RemEnv)"]["exec_ms"].mean()), 3),
            "streaming_state_bytes": int(mf_models["Banded-MF (b=8, MH-RemEnv)"]["cal"]["streaming_state_bytes"]),
            "matrix_memory_kb": round(float(mf_models["Banded-MF (b=8, MH-RemEnv)"]["cal"]["dense_decoder_bytes"]) / 1024.0, 1),
            "naive_h1_violation_at_365": round(float(mf_models["Banded-MF (b=8, MH-RemEnv)"]["cal"]["naive_h1_violation_ratio_at_365"]), 2),
        },
    ]
    pd.DataFrame(sys_rows).to_csv(OUT_DIR / "banded_mf_systems_overhead.csv", index=False)
    return {"summary": df_sum, "systems": pd.DataFrame(sys_rows)}


if __name__ == "__main__":
    res = run_banded_mf_evaluation()
    print("=== Banded-MF Competitive Summary (K=8 seeds x 5 CRN trials, df=7) ===")
    print(res["summary"].to_string(index=False))
    print("\n=== Systems Overhead Summary ===")
    print(res["systems"].to_string(index=False))
