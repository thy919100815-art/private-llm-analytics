"""Comprehensive Synthetic Enterprise Telemetry Generator.

Specification (Section 4):
1. Demographics & Turnover:
   - Initial workforce: 2,000 employees.
   - Annual onboarding & departure rates in {5%, 15%, 30%} (default 15%).
   - 5% of departed employees re-join after 1-3 years.
2. Activity & Volume:
   - User daily activity probability p_u ~ Beta(2, 2).
   - Active day requests ~ LogNormal(mu = ln(10), sigma = 1.0) (median = 10).
   - Weekday to weekend activity ratio = 4:1.
   - Seasonal wave: 1.0 + 0.15 * cos(2 * pi * day_of_year / 365.25).
3. Failure Behavior:
   - Individual user failure probability f_u ~ Beta(1.78, 98.22) (mean ~ 1.78%, matching BurstGPT empirical baseline).
4. Domain Taxonomy:
   - 5 Services, chosen via Zipf(s=1.2).
   - 20 Tasks, chosen via Zipf(s=1.0).
5. Longitudinal Horizons:
   - Default 10 years (3,650 days).
   - Streaming generator yields daily aggregated interactions to preserve memory.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Dict, Generator, List, Optional, Sequence, Tuple
import numpy as np
import pandas as pd


@dataclass
class EmployeeState:
    """Tracks state and trajectory of an employee across time."""

    user_id: str
    p_active: float           # Beta(2, 2)
    failure_rate: float       # Beta(3, 97)
    join_day: int
    depart_day: Optional[int] = None
    scheduled_rejoin_day: Optional[int] = None
    is_active_employee: bool = True


class SyntheticTelemetryGenerator:
    """Enterprise LLM Telemetry Generator."""

    def __init__(
        self,
        n_initial_users: int = 2000,
        annual_join_rate: float = 0.15,
        annual_leave_rate: float = 0.15,
        rejoin_prob: float = 0.05,
        n_years: float = 10.0,
        median_daily_requests: float = 10.0,
        lognorm_sigma: float = 1.0,
        n_services: int = 5,
        zipf_service: float = 1.2,
        n_tasks: int = 20,
        zipf_task: float = 1.0,
        seed: int = 42,
    ):
        self.n_initial_users = int(n_initial_users)
        self.annual_join_rate = float(annual_join_rate)
        self.annual_leave_rate = float(annual_leave_rate)
        self.rejoin_prob = float(rejoin_prob)
        self.total_days = int(math.ceil(n_years * 365))
        self.mu_lognorm = math.log(max(1.0, median_daily_requests))
        self.sigma_lognorm = float(lognorm_sigma)
        self.seed = int(seed)

        self.services = [f"service_{i+1}" for i in range(n_services)]
        self.tasks = [f"task_{i+1:02d}" for i in range(n_tasks)]

        # Precompute Zipf probability distributions
        s_ranks = np.arange(1, n_services + 1)
        s_weights = 1.0 / (s_ranks ** zipf_service)
        self.service_probs = s_weights / np.sum(s_weights)

        t_ranks = np.arange(1, n_tasks + 1)
        t_weights = 1.0 / (t_ranks ** zipf_task)
        self.task_probs = t_weights / np.sum(t_weights)

        self.rng = np.random.default_rng(self.seed)

    def _init_workforce(self) -> Dict[str, EmployeeState]:
        """Initialize demographic pool at day 1."""
        employees: Dict[str, EmployeeState] = {}
        for i in range(1, self.n_initial_users + 1):
            uid = f"emp_{i:06d}"
            p_act = float(self.rng.beta(2.0, 2.0))
            f_rate = float(self.rng.beta(1.78, 98.22))
            employees[uid] = EmployeeState(
                user_id=uid,
                p_active=p_act,
                failure_rate=f_rate,
                join_day=1,
            )
        return employees

    def stream_daily_telemetry(
        self,
        start_day: int = 1,
        end_day: Optional[int] = None,
    ) -> Generator[Tuple[int, pd.DataFrame], None, None]:
        """Stream interactions day by day for scalable longitudinal processing."""
        max_d = self.total_days if end_day is None else min(self.total_days, end_day)
        employees = self._init_workforce()
        next_user_seq = self.n_initial_users + 1

        daily_join_rate = self.annual_join_rate / 365.0
        daily_leave_rate = self.annual_leave_rate / 365.0

        for day in range(start_day, max_d + 1):
            # 1. Turnover update
            active_pool = [e for e in employees.values() if e.is_active_employee]

            # Check departures
            for emp in active_pool:
                if self.rng.uniform(0.0, 1.0) < daily_leave_rate:
                    emp.is_active_employee = False
                    emp.depart_day = day
                    if self.rng.uniform(0.0, 1.0) < self.rejoin_prob:
                        rejoin_delay = int(self.rng.integers(365, 365 * 3 + 1))
                        emp.scheduled_rejoin_day = day + rejoin_delay

            # Check scheduled rejoins
            for emp in employees.values():
                if not emp.is_active_employee and emp.scheduled_rejoin_day == day:
                    emp.is_active_employee = True
                    emp.scheduled_rejoin_day = None

            # New hires
            n_new_hires = int(self.rng.poisson(len(active_pool) * daily_join_rate))
            for _ in range(n_new_hires):
                uid = f"emp_{next_user_seq:06d}"
                next_user_seq += 1
                p_act = float(self.rng.beta(2.0, 2.0))
                f_rate = float(self.rng.beta(1.78, 98.22))
                employees[uid] = EmployeeState(
                    user_id=uid,
                    p_active=p_act,
                    failure_rate=f_rate,
                    join_day=day,
                )

            # 2. Activity & Volume for current day
            # Day of week multiplier (day % 7: 6 and 0 are weekend)
            is_weekend = (day % 7) in (6, 0)
            dow_multiplier = 0.25 if is_weekend else 1.0

            # Seasonal wave
            day_of_year = (day - 1) % 365
            seasonal_multiplier = 1.0 + 0.15 * math.cos(2.0 * math.pi * day_of_year / 365.25)

            current_active = [e for e in employees.values() if e.is_active_employee]
            day_records: List[Dict[str, Any]] = []

            for emp in current_active:
                p_today = emp.p_active * dow_multiplier * seasonal_multiplier
                p_today = max(0.0, min(1.0, p_today))

                if self.rng.uniform(0.0, 1.0) < p_today:
                    # User is active today: sample request count
                    n_req = int(math.ceil(self.rng.lognormal(self.mu_lognorm, self.sigma_lognorm)))
                    n_req = max(1, min(n_req, 2000))

                    # Sample providers and tasks
                    prov_choices = self.rng.choice(self.services, size=n_req, p=self.service_probs)
                    task_choices = self.rng.choice(self.tasks, size=n_req, p=self.task_probs)
                    fail_choices = self.rng.uniform(0.0, 1.0, size=n_req) < emp.failure_rate

                    for r_idx in range(n_req):
                        day_records.append({
                            "interaction_id": f"i_{day}_{emp.user_id}_{r_idx}",
                            "user_id": emp.user_id,
                            "timestamp": f"2026-01-01T{r_idx%24:02d}:00:00Z",
                            "provider": prov_choices[r_idx],
                            "task": task_choices[r_idx],
                            "success": not fail_choices[r_idx],
                        })

            df_day = pd.DataFrame(day_records)
            yield day, df_day
