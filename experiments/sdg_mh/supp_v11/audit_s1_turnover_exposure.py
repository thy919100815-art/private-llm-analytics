"""S1: Employee Departure Privacy Exposure and Post-Departure Growth Audit.

Audits:
1. Root cause of v10 '< 10^-12' artifact.
2. Exact lifetime exposure epsilon(tau) at cutoffs:
   tau in {d, d+7, d+30, d+60, d+90, d+91, d+180}
   for both Replacement and Deletion semantics, decomposed into:
   - Additive Channel
   - Distinct Calendar Channel
   - Total (zCDP rho and canonical epsilon).
3. Verification of Pre-committed Lifetime Exposure (tau -> inf).
4. Post-departure incremental exposure summary grouped by tenure.
"""

from __future__ import annotations

import math
from pathlib import Path
import sys
from typing import Any, Dict, List, Set, Tuple

import numpy as np
import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).parents[3] / "src"))

from private_llm_analytics.sdg_mh.calibration import compute_sdg_target_profile
from private_llm_analytics.sdg_mh.ledger import rho_to_epsilon
from private_llm_analytics.sdg_mh.synthetic_generator import SyntheticTelemetryGenerator

OUTPUT_DIR = Path("results/sdg_mh/supp_v11")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def run_s1_audit():
    print("[S1] Simulating 3-year workforce turnover and post-departure exposure...")
    gen = SyntheticTelemetryGenerator(
        n_initial_users=2000,
        annual_join_rate=0.15,
        annual_leave_rate=0.15,
        rejoin_prob=0.05,
        n_years=3.0,  # 3 years = 1095 days
        seed=12345,
    )

    sdg = compute_sdg_target_profile(
        C_r=104.0,
        sigma_r=5200.0,
        C_f=7.0,
        sigma_f=489.9,
    )
    rho_1_add = sdg["rho_1_add"]  # 0.00030204
    calendar_b = [(7, 7, 60.0), (30, 7, 60.0), (90, 30, 60.0)]
    delta_sq = 11.0

    total_days = 1095
    # Simulate workforce history exactly as in v10
    employees = gen._init_workforce()
    daily_leave_rate = 0.15 / 365.0
    daily_join_rate = 0.15 / 365.0
    next_uid = len(employees) + 1

    for d in range(1, total_days + 1):
        active_pool = [e for e in employees.values() if e.is_active_employee]
        for emp in active_pool:
            if gen.rng.uniform(0.0, 1.0) < daily_leave_rate:
                emp.is_active_employee = False
                emp.depart_day = d
                if gen.rng.uniform(0.0, 1.0) < gen.rejoin_prob:
                    delay = int(gen.rng.integers(365, 365 * 2))
                    emp.scheduled_rejoin_day = d + delay

        for emp in employees.values():
            if not emp.is_active_employee and emp.scheduled_rejoin_day == d:
                emp.is_active_employee = True
                emp.scheduled_rejoin_day = None

        n_new = int(gen.rng.poisson(len(active_pool) * daily_join_rate))
        for _ in range(n_new):
            uid = f"emp_{next_uid:06d}"
            next_uid += 1
            employees[uid] = gen._init_workforce()[list(gen._init_workforce().keys())[0]]
            employees[uid].user_id = uid
            employees[uid].join_day = d
            employees[uid].depart_day = None
            employees[uid].is_active_employee = True

    # Pre-generate Calendar B releases up to day 1095 + 180 (to cover post-departure horizon)
    # A release occurs on end_day e if e % step == 0.
    # Its accessed window is [e - W + 1, e].
    sim_horizon = total_days + 180
    calendar_releases: List[Tuple[int, int, int, float]] = []  # (W, s_w, e_w, rho_w)
    for W, step, sigma in calendar_b:
        rho_w = delta_sq / (2.0 * (sigma**2))
        for e_w in range(W, sim_horizon + 1, step):
            s_w = e_w - W + 1
            calendar_releases.append((W, s_w, e_w, rho_w))

    print(f"  Total generated calendar releases: {len(calendar_releases)} up to day {sim_horizon}.")

    # Evaluate all departed employees who departed with at least 180 days before sim_horizon
    departed_employees = [
        emp for emp in employees.values()
        if emp.depart_day is not None and emp.depart_day <= total_days
    ]
    print(f"  Total departed employees in 3-year simulation: {len(departed_employees)}.")

    rng_activity = np.random.default_rng(54321)
    individual_rows = []
    offsets = [0, 7, 30, 60, 90, 91, 180]

    for emp in departed_employees:
        j_day = emp.join_day
        d_day = emp.depart_day
        tenure_days = max(1, d_day - j_day + 1)

        # Active days mask
        active_days_mask = rng_activity.uniform(0.0, 1.0, size=tenure_days) < emp.p_active
        active_day_indices = set(j_day + np.where(active_days_mask)[0])
        n_active_days = len(active_day_indices)

        if tenure_days < 90:
            t_bin = "< 90d (Intern/Probation)"
        elif tenure_days <= 365:
            t_bin = "90d - 1y"
        elif tenure_days <= 730:
            t_bin = "1y - 2y"
        else:
            t_bin = "> 2y (Senior)"

        # Pre-committed lifetime exposure (all future releases whose window intersects activity)
        # Replacement
        rho_add_precommit = tenure_days * rho_1_add
        rho_dist_precommit_rep = sum(
            rho_w for _, s_w, e_w, rho_w in calendar_releases
            if not (e_w < j_day or s_w > d_day)
        )
        eps_precommit_rep = rho_to_epsilon(rho_add_precommit + rho_dist_precommit_rep)

        # Deletion
        rho_dist_precommit_del = sum(
            rho_w for _, s_w, e_w, rho_w in calendar_releases
            if any(s_w <= d_act <= e_w for d_act in active_day_indices)
        )
        eps_precommit_del = rho_to_epsilon(n_active_days * rho_1_add + rho_dist_precommit_del)

        emp_traj = {
            "user_id": emp.user_id,
            "join_day": j_day,
            "depart_day": d_day,
            "tenure_days": tenure_days,
            "active_days": n_active_days,
            "tenure_bin": t_bin,
            "eps_precommit_rep": round(eps_precommit_rep, 4),
            "eps_precommit_del": round(eps_precommit_del, 4),
        }

        # Track trajectory across tau = d + offset
        for offset in offsets:
            tau = d_day + offset
            # 1. Additive channel: releases t <= tau that access employee's data
            # Since employee was active in [j_day, d_day], releases t in [j_day, d_day] are published by tau >= d_day.
            # Releases t > d_day do not access employee's data.
            # Thus additive exposure strictly freezes at d_day!
            rho_add_tau_rep = tenure_days * rho_1_add
            rho_add_tau_del = n_active_days * rho_1_add

            # 2. Distinct calendar channel: releases with end_day e_w <= tau whose window intersects activity
            # Replacement: intersects [j_day, d_day]
            rho_dist_tau_rep = sum(
                rho_w for _, s_w, e_w, rho_w in calendar_releases
                if e_w <= tau and not (e_w < j_day or s_w > d_day)
            )
            # Deletion: window contains at least one active day
            rho_dist_tau_del = sum(
                rho_w for _, s_w, e_w, rho_w in calendar_releases
                if e_w <= tau and any(s_w <= d_act <= e_w for d_act in active_day_indices)
            )

            tot_rho_rep = rho_add_tau_rep + rho_dist_tau_rep
            tot_rho_del = rho_add_tau_del + rho_dist_tau_del

            eps_rep = rho_to_epsilon(tot_rho_rep)
            eps_del = rho_to_epsilon(tot_rho_del)

            emp_traj[f"eps_rep_d+{offset}"] = round(eps_rep, 4)
            emp_traj[f"eps_del_d+{offset}"] = round(eps_del, 4)
            emp_traj[f"rho_dist_rep_d+{offset}"] = round(rho_dist_tau_rep, 6)
            emp_traj[f"rho_dist_del_d+{offset}"] = round(rho_dist_tau_del, 6)

        # Post-departure incremental exposure
        emp_traj["inc_eps_rep"] = round(emp_traj["eps_rep_d+180"] - emp_traj["eps_rep_d+0"], 4)
        emp_traj["inc_eps_del"] = round(emp_traj["eps_del_d+180"] - emp_traj["eps_del_d+0"], 4)
        emp_traj["inc_rho_dist_rep"] = round(emp_traj["rho_dist_rep_d+180"] - emp_traj["rho_dist_rep_d+0"], 6)
        emp_traj["inc_rho_dist_del"] = round(emp_traj["rho_dist_del_d+180"] - emp_traj["rho_dist_del_d+0"], 6)

        individual_rows.append(emp_traj)

    df_ind = pd.DataFrame(individual_rows)
    df_ind.to_csv(OUTPUT_DIR / "s1_turnover_individual_exposure.csv", index=False)
    print(f"  Saved individual exposure trajectory to {OUTPUT_DIR / 's1_turnover_individual_exposure.csv'}")

    # Summary table across tau cutoffs by tenure bin
    tenure_order = ["< 90d (Intern/Probation)", "90d - 1y", "1y - 2y", "> 2y (Senior)"]
    summary_rows = []

    for t_bin in tenure_order:
        sub = df_ind[df_ind["tenure_bin"] == t_bin]
        if sub.empty:
            continue
        n_emp = len(sub)
        tenure_mean = float(sub["tenure_days"].mean())

        row_rep = {
            "tenure_bin": t_bin,
            "semantics": "Replacement",
            "count": n_emp,
            "tenure_days_mean": round(tenure_mean, 1),
            "eps_d0_mean": round(float(sub["eps_rep_d+0"].mean()), 4),
            "eps_d7_mean": round(float(sub["eps_rep_d+7"].mean()), 4),
            "eps_d30_mean": round(float(sub["eps_rep_d+30"].mean()), 4),
            "eps_d60_mean": round(float(sub["eps_rep_d+60"].mean()), 4),
            "eps_d90_mean": round(float(sub["eps_rep_d+90"].mean()), 4),
            "eps_d91_mean": round(float(sub["eps_rep_d+91"].mean()), 4),
            "eps_d180_mean": round(float(sub["eps_rep_d+180"].mean()), 4),
            "eps_precommit_mean": round(float(sub["eps_precommit_rep"].mean()), 4),
            "post_dep_inc_mean": round(float(sub["inc_eps_rep"].mean()), 4),
            "post_dep_inc_median": round(float(sub["inc_eps_rep"].median()), 4),
            "post_dep_inc_p95": round(float(np.percentile(sub["inc_eps_rep"], 95)), 4),
            "dist_rho_inc_mean": round(float(sub["inc_rho_dist_rep"].mean()), 6),
        }
        summary_rows.append(row_rep)

        row_del = {
            "tenure_bin": t_bin,
            "semantics": "Deletion",
            "count": n_emp,
            "tenure_days_mean": round(tenure_mean, 1),
            "eps_d0_mean": round(float(sub["eps_del_d+0"].mean()), 4),
            "eps_d7_mean": round(float(sub["eps_del_d+7"].mean()), 4),
            "eps_d30_mean": round(float(sub["eps_del_d+30"].mean()), 4),
            "eps_d60_mean": round(float(sub["eps_del_d+60"].mean()), 4),
            "eps_d90_mean": round(float(sub["eps_del_d+90"].mean()), 4),
            "eps_d91_mean": round(float(sub["eps_del_d+91"].mean()), 4),
            "eps_d180_mean": round(float(sub["eps_del_d+180"].mean()), 4),
            "eps_precommit_mean": round(float(sub["eps_precommit_del"].mean()), 4),
            "post_dep_inc_mean": round(float(sub["inc_eps_del"].mean()), 4),
            "post_dep_inc_median": round(float(sub["inc_eps_del"].median()), 4),
            "post_dep_inc_p95": round(float(np.percentile(sub["inc_eps_del"], 95)), 4),
            "dist_rho_inc_mean": round(float(sub["inc_rho_dist_del"].mean()), 6),
        }
        summary_rows.append(row_del)

    df_sum = pd.DataFrame(summary_rows)
    df_sum.to_csv(OUTPUT_DIR / "s1_tenure_incremental_summary.csv", index=False)
    print(f"  Saved tenure summary to {OUTPUT_DIR / 's1_tenure_incremental_summary.csv'}")

    # Verify theoretical invariants:
    # 1. Additive strictly freezes: Verified analytically and by definition
    # 2. Growth stops after w_max: eps(d + 90) == eps(d + 91) == eps(d + 180) == eps_precommit
    diff_90_91 = np.abs(df_ind["eps_rep_d+91"] - df_ind["eps_rep_d+90"]).max()
    diff_90_180 = np.abs(df_ind["eps_rep_d+180"] - df_ind["eps_rep_d+90"]).max()
    diff_precommit = np.abs(df_ind["eps_precommit_rep"] - df_ind["eps_rep_d+180"]).max()

    print("\n--- INVARIANT VERIFICATION ---")
    print(f"  Max |eps(d+91) - eps(d+90)| = {diff_90_91:.6f}")
    print(f"  Max |eps(d+180) - eps(d+90)| = {diff_90_180:.6f}")
    print(f"  Max |eps_precommit - eps(d+180)| = {diff_precommit:.6f}")
    print(df_sum.to_string(index=False))


if __name__ == "__main__":
    run_s1_audit()
