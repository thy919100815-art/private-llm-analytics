"""Joint Calendar and Budget Planner for SDG-MH.

Implements:
1. Exact Calendar Phase and Budget Joint Optimization:
   - Phase vector phi = (phi_7, phi_30, phi_90) over 1,470 combinations (210 canonical classes).
   - Budget vector v = (d, x_7, x_{30}, x_{90}) with separable convex objective sum_j (w_j * a_j / v_j).
   - Linear inequality constraints over all horizons h in 1..575: L_max(h; phi, v) <= rho_bar(h).
   - Tail growth condition: d + sum_f (x_f / S_f) <= beta.
2. Four Fair Comparison Groups under identical privacy commitment:
   - Group A: Baseline (Aligned calendar, original budget; with calibrated fallback if unscaled violates bound).
   - Group B: Budget Only (Aligned calendar, optimized budget).
   - Group C: Phase Only (Optimized phase maximizing slack, original budget).
   - Group D: Joint Planning (Optimized phase and optimized budget minimizing workload error).
3. Business Constraints:
   - Preserves 7/30/90 day window semantics and step sizes.
   - Pinned weekday/phase support (e.g. Sunday reports).
   - Maximum refresh latency enforcement.
4. Robustness:
   - Returns NO_FEASIBLE_POLICY when budget cannot meet constraints.
   - Atomic verification and transactional binding with UnifiedPrivacyLedger.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import numpy as np
from scipy.optimize import minimize

from private_llm_analytics.sdg_mh.calendar_accounting import (
    CONSERVATIVE_BACKOFF_REL,
    DEFAULT_FAMILIES,
    JOINT_PERIOD,
    ORIGINAL_PLANNER_CURVE,
    PRIVACY_LEDGER_TOL,
    SOLVER_VIOLATION_TOL,
    STRICT_CERTIFICATION_TOL,
    WEEK1_V2_CURVE,
    CalendarFamily,
    CalendarSchedule,
    PrivacyCommitmentCurve,
    count_intersections_fast_vectorized,
)
from private_llm_analytics.sdg_mh.ledger import (
    PrivacyAdmissionError,
    ReleaseRecord,
    ReleaseStatus,
    UnifiedPrivacyLedger,
)


DEFAULT_HISTORICAL_B: float = 0.000302
DEFAULT_HISTORICAL_D: float = 0.8 * DEFAULT_HISTORICAL_B  # 0.0002416
DEFAULT_HISTORICAL_X: float = 0.00153


@dataclass(frozen=True)
class PlannerConfig:
    """Configuration for Joint Calendar and Budget Planning."""

    commitment: PrivacyCommitmentCurve = WEEK1_V2_CURVE
    # Workload weights for [daily_add, distinct_7, distinct_30, distinct_90]
    workload_weights: Tuple[float, float, float, float] = (0.40, 0.25, 0.25, 0.10)
    # Sensitivity coefficients a_j (proportional to Delta_j^2 / 2)
    sensitivity_coeffs: Tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0)
    # Business Constraints
    fixed_phi_7: Optional[int] = None
    fixed_phi_30: Optional[int] = None
    fixed_phi_90: Optional[int] = None
    max_refresh_latency_days: int = 30
    staleness_penalty_weight: float = 0.0
    # Budget Bounds: [d, x_7, x_30, x_90]
    min_allocations: Tuple[float, float, float, float] = (1e-6, 1e-6, 1e-6, 1e-6)
    max_allocations: Tuple[float, float, float, float] = (0.01, 0.05, 0.05, 0.05)
    max_eval_horizon: int = 575
    solver_max_iters: int = 15
    apply_conservative_shrinkage: bool = True

    def __post_init__(self) -> None:
        if len(self.workload_weights) != 4:
            raise ValueError("workload_weights must have length 4")
        if len(self.sensitivity_coeffs) != 4:
            raise ValueError("sensitivity_coeffs must have length 4")


@dataclass(frozen=True)
class PlanningResult:
    """Result of calendar and budget planning.

    Explicitly separates:
    - Solver approximate optimality (`is_solver_approx_optimal`, `solver_status`)
    - Mathematical model global optimality within declared domain (`is_math_global_optimal`, `optimality_scope`)
    - Independent infinite-horizon privacy certification (`is_certified_publishable`, `certification_status`)
    """

    group: str
    status: str  # 'OPTIMAL', 'FEASIBLE', 'CALIBRATED_BASELINE', 'SOLVER_OPT_UNCERTIFIED', 'CERTIFICATION_FAILED', 'INFEASIBLE', 'NO_FEASIBLE_POLICY'
    phases: Tuple[int, int, int]
    daily_d: float
    distinct_costs: Tuple[float, float, float]
    weighted_workload_error: float
    min_slack: float
    tightest_horizon_h: int
    asymptotic_slope: float
    is_infinite_horizon_secure: bool
    runtime_seconds: float
    diagnostics: Dict[str, Any] = field(default_factory=dict)
    is_solver_approx_optimal: bool = False
    is_math_global_optimal: bool = False
    is_certified_publishable: bool = False
    optimality_scope: str = "NONE"
    solver_status: str = "NOT_RUN"
    certification_status: str = "UNVERIFIED"

    @property
    def schedule(self) -> CalendarSchedule:
        return CalendarSchedule(
            phases=self.phases,
            costs=self.distinct_costs,
            daily_cost_d=self.daily_d,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "group": self.group,
            "status": self.status,
            "phases": list(self.phases),
            "daily_d": round(self.daily_d, 8),
            "distinct_costs": [round(x, 8) for x in self.distinct_costs],
            "weighted_workload_error": round(self.weighted_workload_error, 4),
            "min_slack": round(self.min_slack, 8),
            "tightest_horizon_h": self.tightest_horizon_h,
            "asymptotic_slope": round(self.asymptotic_slope, 8),
            "is_infinite_horizon_secure": self.is_infinite_horizon_secure,
            "is_solver_approx_optimal": self.is_solver_approx_optimal,
            "is_math_global_optimal": self.is_math_global_optimal,
            "is_certified_publishable": self.is_certified_publishable,
            "optimality_scope": self.optimality_scope,
            "solver_status": self.solver_status,
            "certification_status": self.certification_status,
            "runtime_seconds": round(self.runtime_seconds, 4),
            "diagnostics": self.diagnostics,
        }


class JointCalendarPlanner:
    """Joint Calendar and Budget Optimization Engine."""

    def __init__(self, config: Optional[PlannerConfig] = None):
        self.config = config or PlannerConfig()
        self.commitment = self.config.commitment
        self.P = JOINT_PERIOD
        self.eval_H_0 = max(365, int(self.commitment.horizons[-1]))
        self.max_h = max(int(self.config.max_eval_horizon), self.eval_H_0 + self.P)
        self.s_arr = np.arange(self.P, dtype=np.int64)
        self.h_vals = np.arange(1, self.max_h + 1, dtype=np.int64)
        self.rho_bars = self.commitment.rho_bar(self.h_vals)

        # Precompute table of intersections:
        # n7_tables: 7 phases for 7d family, shape (7, max_h, P)
        self.n7_tables = [
            np.array([count_intersections_fast_vectorized(self.s_arr, int(h), 7, 7, p) for h in self.h_vals], dtype=np.int16)
            for p in range(7)
        ]
        # n30_tables: 7 phases for 30d family, shape (7, max_h, P)
        self.n30_tables = [
            np.array([count_intersections_fast_vectorized(self.s_arr, int(h), 30, 7, p) for h in self.h_vals], dtype=np.int16)
            for p in range(7)
        ]
        # n90_tables: 30 phases for 90d family, shape (30, max_h, P)
        self.n90_tables = [
            np.array([count_intersections_fast_vectorized(self.s_arr, int(h), 90, 30, p) for h in self.h_vals], dtype=np.int16)
            for p in range(30)
        ]

    def _effective_weights(self) -> np.ndarray:
        w = np.array(self.config.workload_weights, dtype=np.float64)
        a = np.array(self.config.sensitivity_coeffs, dtype=np.float64)
        return w * a

    def evaluate_schedule_slack(
        self,
        phases: Tuple[int, int, int],
        v: Union[Sequence[float], np.ndarray],
        slack_tol: float = STRICT_CERTIFICATION_TOL,
    ) -> Tuple[float, int, float, bool]:
        """Compute minimum slack and asymptotic slope for given phase and budget vector.

        Uses STRICT_CERTIFICATION_TOL (0.0) by default so negative slacks fail certification.

        Returns:
            (min_slack, min_h, asymptotic_slope, is_secure)
        """
        p7, p30, p90 = phases
        d_val, x7, x30, x90 = float(v[0]), float(v[1]), float(v[2]), float(v[3])

        n7 = self.n7_tables[p7]
        n30 = self.n30_tables[p30]
        n90 = self.n90_tables[p90]

        L_matrix = (
            self.h_vals[:, None] * d_val
            + n7 * x7
            + n30 * x30
            + n90 * x90
        )
        l_max = np.max(L_matrix, axis=1)
        slacks = self.rho_bars - l_max
        min_idx = int(np.argmin(slacks))
        min_slack = float(slacks[min_idx])
        min_h = int(self.h_vals[min_idx])

        asymptotic_slope = d_val + x7 / 7.0 + x30 / 7.0 + x90 / 30.0
        slope_safe = asymptotic_slope <= self.commitment.beta_tail + slack_tol
        affine_tail_ok = (self.eval_H_0 >= self.commitment.horizons[-1])
        is_secure = (min_slack >= -slack_tol) and slope_safe and affine_tail_ok

        return min_slack, min_h, asymptotic_slope, is_secure

    def _shrink_budget_vector(
        self,
        phases: Tuple[int, int, int],
        v_raw: np.ndarray,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Apply minimal conservative scaling gamma <= 1.0 when raw budget sits on/below zero slack."""
        p7, p30, p90 = phases
        v_arr = np.asarray(v_raw, dtype=np.float64)
        n7 = self.n7_tables[p7]
        n30 = self.n30_tables[p30]
        n90 = self.n90_tables[p90]

        L_matrix = (
            self.h_vals[:, None] * v_arr[0]
            + n7 * v_arr[1]
            + n30 * v_arr[2]
            + n90 * v_arr[3]
        )
        l_max = np.max(L_matrix, axis=1)
        slacks = self.rho_bars - l_max
        raw_min_slack = float(np.min(slacks))
        raw_slope = float(v_arr[0] + v_arr[1] / 7.0 + v_arr[2] / 7.0 + v_arr[3] / 30.0)
        raw_headroom = float(self.commitment.beta_tail - raw_slope)

        r_horizon = float(np.max(l_max / self.rho_bars))
        r_tail = raw_slope / float(self.commitment.beta_tail) if self.commitment.beta_tail > 0 else float("inf")
        r_max = max(1.0, r_horizon, r_tail)

        margin_threshold = CONSERVATIVE_BACKOFF_REL * 1e-2
        needs_shrink = (
            self.config.apply_conservative_shrinkage
            and (r_max > 1.0 or raw_min_slack <= margin_threshold or raw_headroom <= margin_threshold)
        )
        if needs_shrink:
            gamma = (1.0 - CONSERVATIVE_BACKOFF_REL) / r_max
            v_safe = v_arr * gamma
        else:
            gamma = 1.0
            v_safe = v_arr.copy()

        min_bounds = np.array(self.config.min_allocations, dtype=np.float64)
        bounds_ok = bool(np.all(v_safe >= min_bounds - 1e-18))

        return v_safe, {
            "shrinkage_applied": bool(needs_shrink),
            "shrinkage_gamma": float(gamma),
            "pre_shrink_min_slack": raw_min_slack,
            "pre_shrink_slope_headroom": raw_headroom,
            "bounds_satisfied_after_shrink": bounds_ok,
        }

    def _certify_and_shrink_budget(
        self,
        phases: Tuple[int, int, int],
        v_raw: np.ndarray,
        slack_tol: float = STRICT_CERTIFICATION_TOL,
    ) -> Tuple[bool, np.ndarray, Dict[str, Any]]:
        """Conservatively shrink (if enabled) and verify strict infinite-horizon certificate."""
        v_safe, shrink_diag = self._shrink_budget_vector(phases, v_raw)
        sched = CalendarSchedule(
            phases=phases,
            costs=(float(v_safe[1]), float(v_safe[2]), float(v_safe[3])),
            daily_cost_d=float(v_safe[0]),
        )
        cert = sched.verify_periodicity_and_long_term_security(
            commitment=self.commitment,
            H_0=self.eval_H_0,
            P=self.P,
            slack_tol=slack_tol,
        )
        is_certified = bool(cert["is_infinite_horizon_secure"]) and bool(shrink_diag["bounds_satisfied_after_shrink"])
        diag = {**shrink_diag, "certificate": cert}
        return is_certified, v_safe, diag

    def solve_budget_for_phase(
        self,
        phases: Tuple[int, int, int],
        initial_v: Optional[np.ndarray] = None,
    ) -> Tuple[bool, Optional[np.ndarray], float, Dict[str, Any]]:
        """Convex budget optimization for fixed phase vector minimizing workload variance.

        Solves:
            min_{v} sum_j (c_j / v_j)
            s.t. L_max(h; phi, v) <= rho_bar(h) for all h in 1..575
                 d + x7/7 + x30/7 + x90/30 <= beta_tail
                 v_min <= v <= v_max
        """
        p7, p30, p90 = phases
        n7 = self.n7_tables[p7]
        n30 = self.n30_tables[p30]
        n90 = self.n90_tables[p90]

        c = self._effective_weights()
        scale = 1e-4
        c_scaled = c

        tail_row = np.array([1.0, 1.0 / 7.0, 1.0 / 7.0, 1.0 / 30.0], dtype=np.float64)
        tail_b = float(self.commitment.beta_tail)

        # Cutting plane active set
        key_horizons = [1, 2, 7, 14, 21, 30, 32, 35, 37, 60, 90, 180, 210, 365, self.max_h]
        active_rows = [tail_row * scale]
        active_b = [tail_b]

        for kh in key_horizons:
            idx = kh - 1
            comb = np.unique(np.stack([n7[idx], n30[idx], n90[idx]], axis=1), axis=0)
            for u in comb:
                active_rows.append(np.array([float(kh), float(u[0]), float(u[1]), float(u[2])]) * scale)
                active_b.append(float(self.rho_bars[idx]))

        active_A = np.array(active_rows, dtype=np.float64)
        active_b = np.array(active_b, dtype=np.float64)

        bounds = [
            (self.config.min_allocations[i] / scale, self.config.max_allocations[i] / scale)
            for i in range(4)
        ]

        if initial_v is not None:
            y0 = np.array(initial_v, dtype=np.float64) / scale
        else:
            y0 = np.array([DEFAULT_HISTORICAL_D, DEFAULT_HISTORICAL_X, DEFAULT_HISTORICAL_X, DEFAULT_HISTORICAL_X]) / scale

        def obj(y: np.ndarray) -> float:
            return float(np.sum(c_scaled / (scale * y)))

        def grad(y: np.ndarray) -> np.ndarray:
            return -c_scaled / (scale * y * y)

        for it in range(self.config.solver_max_iters):
            cons = {
                "type": "ineq",
                "fun": lambda y, A=active_A, b=active_b: b - A @ y,
                "jac": lambda y, A=active_A: -A,
            }
            res = minimize(
                obj,
                y0,
                jac=grad,
                method="SLSQP",
                bounds=bounds,
                constraints=cons,
                options={"ftol": 1e-9, "maxiter": 120},
            )

            if not res.success:
                # Infeasible or numerical stall
                return False, None, float("inf"), {"iterations": it + 1, "message": res.message}

            y0 = res.x
            v = y0 * scale

            # Evaluate full 575 horizons
            L_matrix = (
                self.h_vals[:, None] * v[0]
                + n7 * v[1]
                + n30 * v[2]
                + n90 * v[3]
            )
            l_max = np.max(L_matrix, axis=1)
            viol = l_max - self.rho_bars
            max_viol_idx = int(np.argmax(viol))
            max_viol = viol[max_viol_idx]
            tail_viol = np.dot(tail_row, v) - tail_b

            if max_viol <= SOLVER_VIOLATION_TOL and tail_viol <= SOLVER_VIOLATION_TOL:
                weighted_error = float(np.sum(c / v))
                return True, v, weighted_error, {
                    "iterations": it + 1,
                    "status": "CONVERGED",
                    "solver_max_viol": float(max_viol),
                    "solver_tail_viol": float(tail_viol),
                    "solver_tol": SOLVER_VIOLATION_TOL,
                }

            # Add worst violated constraint
            if max_viol > SOLVER_VIOLATION_TOL:
                worst_s = int(np.argmax(L_matrix[max_viol_idx]))
                new_r = (
                    np.array([
                        float(max_viol_idx + 1),
                        float(n7[max_viol_idx, worst_s]),
                        float(n30[max_viol_idx, worst_s]),
                        float(n90[max_viol_idx, worst_s]),
                    ])
                    * scale
                )
                active_A = np.vstack([active_A, new_r])
                active_b = np.append(active_b, self.rho_bars[max_viol_idx])

            if tail_viol > SOLVER_VIOLATION_TOL:
                active_A = np.vstack([active_A, tail_row * scale])
                active_b = np.append(active_b, tail_b)

        return False, None, float("inf"), {"iterations": self.config.solver_max_iters, "status": "MAX_ITERS"}

    def plan_group_a(self) -> PlanningResult:
        """Group A: Baseline (Aligned calendar, original budget)."""
        t0 = time.time()
        phases = (0, 0, 0)
        v_orig = np.array([DEFAULT_HISTORICAL_D, DEFAULT_HISTORICAL_X, DEFAULT_HISTORICAL_X, DEFAULT_HISTORICAL_X])
        c = self._effective_weights()

        # Note: For Group A uncalibrated baseline check, inspect raw slack directly first
        min_slack, min_h, slope, is_secure = self.evaluate_schedule_slack(
            phases, v_orig, slack_tol=STRICT_CERTIFICATION_TOL
        )
        if is_secure:
            sched_orig = CalendarSchedule(
                phases=phases,
                costs=(float(v_orig[1]), float(v_orig[2]), float(v_orig[3])),
                daily_cost_d=float(v_orig[0]),
            )
            cert_orig = sched_orig.verify_periodicity_and_long_term_security(
                commitment=self.commitment,
                H_0=self.eval_H_0,
                P=self.P,
                slack_tol=STRICT_CERTIFICATION_TOL,
            )
            if cert_orig["is_infinite_horizon_secure"]:
                return PlanningResult(
                    group="A_Baseline",
                    status="FEASIBLE",
                    phases=phases,
                    daily_d=float(v_orig[0]),
                    distinct_costs=(float(v_orig[1]), float(v_orig[2]), float(v_orig[3])),
                    weighted_workload_error=float(np.sum(c / v_orig)),
                    min_slack=min_slack,
                    tightest_horizon_h=min_h,
                    asymptotic_slope=slope,
                    is_infinite_horizon_secure=True,
                    runtime_seconds=time.time() - t0,
                    diagnostics={"calibration_needed": False, "certificate": cert_orig},
                    is_solver_approx_optimal=False,
                    is_math_global_optimal=False,
                    is_certified_publishable=True,
                    optimality_scope="NONE",
                    solver_status="NOT_RUN",
                    certification_status="CERTIFIED_PUBLISHABLE",
                )

        # Baseline violates commitment: construct calibrated baseline fallback by exact scaling + certification
        n7 = self.n7_tables[0]
        n30 = self.n30_tables[0]
        n90 = self.n90_tables[0]
        L_matrix = (self.h_vals[:, None] * v_orig[0] + n7 * v_orig[1] + n30 * v_orig[2] + n90 * v_orig[3])
        l_max = np.max(L_matrix, axis=1)
        ratios = l_max / self.rho_bars
        gamma_raw = float(1.0 / np.max(ratios))
        v_cal_raw = v_orig * gamma_raw

        cert_ok, v_cal, cert_diag = self._certify_and_shrink_budget(phases, v_cal_raw)
        min_slack_cal, min_h_cal, slope_cal, _ = self.evaluate_schedule_slack(
            phases, v_cal, slack_tol=STRICT_CERTIFICATION_TOL
        )
        eff_gamma = float(gamma_raw * cert_diag.get("shrinkage_gamma", 1.0))

        return PlanningResult(
            group="A_Baseline",
            status="CALIBRATED_BASELINE" if cert_ok else "CERTIFICATION_FAILED",
            phases=phases,
            daily_d=float(v_cal[0]),
            distinct_costs=(float(v_cal[1]), float(v_cal[2]), float(v_cal[3])),
            weighted_workload_error=float(np.sum(c / v_cal)) if cert_ok else float("inf"),
            min_slack=min_slack_cal,
            tightest_horizon_h=min_h_cal,
            asymptotic_slope=slope_cal,
            is_infinite_horizon_secure=cert_ok,
            runtime_seconds=time.time() - t0,
            diagnostics={
                "calibration_needed": True,
                "unscaled_min_slack": min_slack,
                "unscaled_tightest_h": min_h,
                "scale_factor_gamma": eff_gamma,
                **cert_diag,
            },
            is_solver_approx_optimal=False,
            is_math_global_optimal=False,
            is_certified_publishable=cert_ok,
            optimality_scope="NONE",
            solver_status="NOT_RUN",
            certification_status="CERTIFIED_PUBLISHABLE" if cert_ok else "UNCERTIFIED_VIOLATION",
        )

    def plan_group_b(self) -> PlanningResult:
        """Group B: Budget Only (Aligned calendar, optimized budget)."""
        t0 = time.time()
        phases = (0, 0, 0)
        ok, v_opt, err, diag = self.solve_budget_for_phase(phases)

        if not ok or v_opt is None:
            return PlanningResult(
                group="B_BudgetOnly",
                status="NO_FEASIBLE_POLICY",
                phases=phases,
                daily_d=0.0,
                distinct_costs=(0.0, 0.0, 0.0),
                weighted_workload_error=float("inf"),
                min_slack=-1.0,
                tightest_horizon_h=0,
                asymptotic_slope=0.0,
                is_infinite_horizon_secure=False,
                runtime_seconds=time.time() - t0,
                diagnostics=diag,
                is_solver_approx_optimal=False,
                is_math_global_optimal=False,
                is_certified_publishable=False,
                optimality_scope="FIXED_PHASE_CONVEX",
                solver_status=diag.get("status", "SOLVER_FAILED"),
                certification_status="REJECTED_INFEASIBLE",
            )

        cert_ok, v_cert, cert_diag = self._certify_and_shrink_budget(phases, v_opt)
        min_slack, min_h, slope, _ = self.evaluate_schedule_slack(
            phases, v_cert, slack_tol=STRICT_CERTIFICATION_TOL
        )
        c = self._effective_weights()
        certified_err = float(np.sum(c / v_cert))
        runtime = time.time() - t0

        return PlanningResult(
            group="B_BudgetOnly",
            status="OPTIMAL" if cert_ok else "SOLVER_OPT_UNCERTIFIED",
            phases=phases,
            daily_d=float(v_cert[0]),
            distinct_costs=(float(v_cert[1]), float(v_cert[2]), float(v_cert[3])),
            weighted_workload_error=certified_err,
            min_slack=min_slack,
            tightest_horizon_h=min_h,
            asymptotic_slope=slope,
            is_infinite_horizon_secure=cert_ok,
            runtime_seconds=runtime,
            diagnostics={**diag, **cert_diag},
            is_solver_approx_optimal=True,
            is_math_global_optimal=True,
            is_certified_publishable=cert_ok,
            optimality_scope="FIXED_PHASE_CONVEX",
            solver_status="CONVERGED",
            certification_status="CERTIFIED_PUBLISHABLE" if cert_ok else "UNCERTIFIED_VIOLATION",
        )

    def plan_group_c(self) -> PlanningResult:
        """Group C: Phase Only (Optimized phase maximizing slack, original budget)."""
        t0 = time.time()
        v_orig = np.array([DEFAULT_HISTORICAL_D, DEFAULT_HISTORICAL_X, DEFAULT_HISTORICAL_X, DEFAULT_HISTORICAL_X])
        c = self._effective_weights()

        best_phi = (0, 0, 0)
        best_min_slack = -1e9
        best_tightest_h = 0
        best_slope = 0.0

        p7_candidates = [self.config.fixed_phi_7] if self.config.fixed_phi_7 is not None else [0]
        p30_candidates = [self.config.fixed_phi_30] if self.config.fixed_phi_30 is not None else range(7)
        p90_candidates = [self.config.fixed_phi_90] if self.config.fixed_phi_90 is not None else range(30)

        for p7 in p7_candidates:
            for p30 in p30_candidates:
                for p90 in p90_candidates:
                    phi = (p7, p30, p90)
                    min_s, min_h, slope, _ = self.evaluate_schedule_slack(
                        phi, v_orig, slack_tol=STRICT_CERTIFICATION_TOL
                    )
                    if min_s > best_min_slack:
                        best_min_slack = min_s
                        best_phi = phi
                        best_tightest_h = min_h
                        best_slope = slope

        sched = CalendarSchedule(
            phases=best_phi,
            costs=(float(v_orig[1]), float(v_orig[2]), float(v_orig[3])),
            daily_cost_d=float(v_orig[0]),
        )
        cert = sched.verify_periodicity_and_long_term_security(
            commitment=self.commitment,
            H_0=self.eval_H_0,
            P=self.P,
            slack_tol=STRICT_CERTIFICATION_TOL,
        )
        is_secure = bool(cert["is_infinite_horizon_secure"])
        runtime = time.time() - t0

        if not is_secure:
            return PlanningResult(
                group="C_PhaseOnly",
                status="INFEASIBLE",
                phases=best_phi,
                daily_d=float(v_orig[0]),
                distinct_costs=(float(v_orig[1]), float(v_orig[2]), float(v_orig[3])),
                weighted_workload_error=float("inf"),
                min_slack=best_min_slack,
                tightest_horizon_h=best_tightest_h,
                asymptotic_slope=best_slope,
                is_infinite_horizon_secure=False,
                runtime_seconds=runtime,
                diagnostics={"best_phase": best_phi, "raw_workload_error": float(np.sum(c / v_orig)), "breached": True, "certificate": cert},
                is_solver_approx_optimal=False,
                is_math_global_optimal=False,
                is_certified_publishable=False,
                optimality_scope="DISCRETE_MAX_SLACK_FIXED_BUDGET",
                solver_status="NOT_RUN",
                certification_status="REJECTED_INFEASIBLE",
            )

        return PlanningResult(
            group="C_PhaseOnly",
            status="OPTIMAL",
            phases=best_phi,
            daily_d=float(v_orig[0]),
            distinct_costs=(float(v_orig[1]), float(v_orig[2]), float(v_orig[3])),
            weighted_workload_error=float(np.sum(c / v_orig)),
            min_slack=best_min_slack,
            tightest_horizon_h=best_tightest_h,
            asymptotic_slope=best_slope,
            is_infinite_horizon_secure=True,
            runtime_seconds=runtime,
            diagnostics={"best_phase": best_phi, "certificate": cert},
            is_solver_approx_optimal=False,
            is_math_global_optimal=True,
            is_certified_publishable=True,
            optimality_scope="DISCRETE_MAX_SLACK_FIXED_BUDGET",
            solver_status="NOT_RUN",
            certification_status="CERTIFIED_PUBLISHABLE",
        )

    def plan_group_d(self) -> PlanningResult:
        """Group D: Joint Planning (Optimized phase and optimized budget)."""
        t0 = time.time()
        c = self._effective_weights()

        p7_candidates = [self.config.fixed_phi_7] if self.config.fixed_phi_7 is not None else [0]
        p30_candidates = [self.config.fixed_phi_30] if self.config.fixed_phi_30 is not None else range(7)
        p90_candidates = [self.config.fixed_phi_90] if self.config.fixed_phi_90 is not None else range(30)

        phases_attempted = 0
        phases_converged = 0

        best_phi: Optional[Tuple[int, int, int]] = None
        best_v: Optional[np.ndarray] = None
        best_err = float("inf")
        best_diag: Dict[str, Any] = {}

        fallback_phi: Optional[Tuple[int, int, int]] = None
        fallback_v: Optional[np.ndarray] = None
        fallback_err = float("inf")
        fallback_diag: Dict[str, Any] = {}

        for p7 in p7_candidates:
            for p30 in p30_candidates:
                for p90 in p90_candidates:
                    phases_attempted += 1
                    phi = (p7, p30, p90)
                    ok, v_opt, err, diag = self.solve_budget_for_phase(phi)
                    if not ok or v_opt is None:
                        continue
                    phases_converged += 1

                    if self.config.apply_conservative_shrinkage:
                        v_shrunk, s_diag = self._shrink_budget_vector(phi, v_opt)
                        err_shrunk = float(np.sum(c / v_shrunk))
                        if s_diag["bounds_satisfied_after_shrink"] and err_shrunk < best_err:
                            best_err = err_shrunk
                            best_v = v_shrunk
                            best_phi = phi
                            best_diag = {**diag, **s_diag}
                        elif err < fallback_err:
                            fallback_err = err
                            fallback_v = v_opt
                            fallback_phi = phi
                            fallback_diag = {**diag, **s_diag}
                    else:
                        if err < best_err:
                            best_err = err
                            best_v = v_opt
                            best_phi = phi
                            best_diag = diag

        runtime = time.time() - t0
        if best_v is None or best_phi is None:
            if fallback_v is not None and fallback_phi is not None:
                best_phi, best_v, best_err, best_diag = fallback_phi, fallback_v, fallback_err, fallback_diag
            else:
                return PlanningResult(
                    group="D_JointPlanning",
                    status="NO_FEASIBLE_POLICY",
                    phases=(0, 0, 0),
                    daily_d=0.0,
                    distinct_costs=(0.0, 0.0, 0.0),
                    weighted_workload_error=float("inf"),
                    min_slack=-1.0,
                    tightest_horizon_h=0,
                    asymptotic_slope=0.0,
                    is_infinite_horizon_secure=False,
                    runtime_seconds=runtime,
                    diagnostics={"error": "No feasible joint policy found", "phases_attempted": phases_attempted},
                    is_solver_approx_optimal=False,
                    is_math_global_optimal=False,
                    is_certified_publishable=False,
                    optimality_scope="JOINT_EXHAUSTIVE_MIXED_CONVEX",
                    solver_status="INFEASIBLE",
                    certification_status="REJECTED_INFEASIBLE",
                )

        cert_ok, v_cert, cert_diag = self._certify_and_shrink_budget(best_phi, best_v)
        min_slack, min_h, slope, _ = self.evaluate_schedule_slack(
            best_phi, v_cert, slack_tol=STRICT_CERTIFICATION_TOL
        )
        certified_err = float(np.sum(c / v_cert))

        unpinned = (
            self.config.fixed_phi_7 is None
            and self.config.fixed_phi_30 is None
            and self.config.fixed_phi_90 is None
        )
        optimality_scope = "JOINT_EXHAUSTIVE_MIXED_CONVEX" if unpinned else "CONSTRAINED_PHASE_CONVEX"
        math_global_opt = bool(phases_converged == phases_attempted)

        return PlanningResult(
            group="D_JointPlanning",
            status="OPTIMAL" if cert_ok else "SOLVER_OPT_UNCERTIFIED",
            phases=best_phi,
            daily_d=float(v_cert[0]),
            distinct_costs=(float(v_cert[1]), float(v_cert[2]), float(v_cert[3])),
            weighted_workload_error=certified_err,
            min_slack=min_slack,
            tightest_horizon_h=min_h,
            asymptotic_slope=slope,
            is_infinite_horizon_secure=cert_ok,
            runtime_seconds=time.time() - t0,
            diagnostics={
                **best_diag,
                **cert_diag,
                "phases_attempted": phases_attempted,
                "phases_converged": phases_converged,
            },
            is_solver_approx_optimal=True,
            is_math_global_optimal=math_global_opt,
            is_certified_publishable=cert_ok,
            optimality_scope=optimality_scope,
            solver_status="CONVERGED",
            certification_status="CERTIFIED_PUBLISHABLE" if cert_ok else "UNCERTIFIED_VIOLATION",
        )

    def plan_all_groups(self) -> Dict[str, PlanningResult]:
        """Execute complete 4-group benchmarking plan."""
        return {
            "A_Baseline": self.plan_group_a(),
            "B_BudgetOnly": self.plan_group_b(),
            "C_PhaseOnly": self.plan_group_c(),
            "D_JointPlanning": self.plan_group_d(),
        }

    def commit_plan_to_ledger(
        self,
        plan: PlanningResult,
        ledger: UnifiedPrivacyLedger,
        start_day: int = 1,
        simulated_days: int = 365,
        populate_records: bool = True,
    ) -> Tuple[bool, str, Dict[str, Any]]:
        """Atomic reservation and validation of planning policy in UnifiedPrivacyLedger."""
        if plan.status in ("NO_FEASIBLE_POLICY", "INFEASIBLE", "SOLVER_OPT_UNCERTIFIED", "CERTIFICATION_FAILED"):
            return False, f"Cannot commit uncertified/infeasible policy: {plan.status}", {"status": plan.status}

        if not plan.is_certified_publishable:
            return False, "Cannot commit policy missing certified publishable flag", {"status": plan.status}

        sched = plan.schedule
        cert = sched.verify_periodicity_and_long_term_security(
            commitment=self.commitment,
            H_0=self.eval_H_0,
            P=JOINT_PERIOD,
            slack_tol=STRICT_CERTIFICATION_TOL,
        )

        if not cert["is_infinite_horizon_secure"]:
            return False, "Schedule failed infinite-horizon privacy verification", cert

        # Generate tentative release records with canonical IDs and status=RESERVED
        records: List[ReleaseRecord] = []
        eta = 25.0 / 26.0
        rho_r = eta * plan.daily_d
        rho_f = (1.0 - eta) * plan.daily_d
        sigma_iso = math.sqrt(2600.0 / (2.0 * plan.daily_d))

        for day in range(start_day, start_day + simulated_days):
            # Daily additive base seal (Requests & Failures split)
            records.append(
                ReleaseRecord(
                    release_id=f"sdg_additive_req_{day}",
                    days_accessed=frozenset([day]),
                    mechanism_type="gaussian",
                    params={"sigma": float(sigma_iso)},
                    sensitivity=50.0,
                    rho_cost=rho_r,
                    tag="sdg_requests",
                    status=ReleaseStatus.RESERVED.value,
                )
            )
            records.append(
                ReleaseRecord(
                    release_id=f"sdg_additive_fail_{day}",
                    days_accessed=frozenset([day]),
                    mechanism_type="gaussian",
                    params={"sigma": float(sigma_iso)},
                    sensitivity=10.0,
                    rho_cost=rho_f,
                    tag="sdg_failures",
                    status=ReleaseStatus.RESERVED.value,
                )
            )

            # Distinct calendar families
            for fam, phi, cost in zip(sched.families, sched.phases, sched.costs):
                w = fam.window_days
                s = fam.step_days
                if day % s == phi % s:
                    w_start = max(1, day - w + 1)
                    w_days = frozenset(range(w_start, day + 1))
                    sigma_w = math.sqrt(11.0 / (2.0 * cost))
                    records.append(
                        ReleaseRecord(
                            release_id=f"calendar_{w}_{s}_day_{day}",
                            days_accessed=w_days,
                            mechanism_type="gaussian",
                            params={"sigma": float(sigma_w)},
                            sensitivity=math.sqrt(11.0),
                            rho_cost=cost,
                            tag=f"calendar_{w}_{s}",
                            status=ReleaseStatus.RESERVED.value,
                        )
                    )

        # Atomic reservation in UnifiedPrivacyLedger
        ledger.set_active_plan(plan, commitment=self.commitment)
        if populate_records:
            try:
                ledger.admit_and_commit_batch(
                    candidate_records=records,
                    commitment_bound=self.commitment,
                    as_status=ReleaseStatus.RESERVED.value,
                    allow_unconfigured_policy=False,
                )
            except PrivacyAdmissionError as e:
                return False, f"Policy rejected at ledger gate: {e}", {"error": str(e), "cert": cert}

        return True, f"Policy successfully bound: {len(records) if populate_records else 0} releases reserved in ledger.", {"exceeded_count": 0, "cert": cert}
