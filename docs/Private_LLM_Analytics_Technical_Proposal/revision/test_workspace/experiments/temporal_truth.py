"""Rate-consistent independent-increment temporal process simulation.

Uses daily Negative Binomial increments to ensure that for ANY slice window W in [1, H]:
  E[M_W] = mu_ref * (W / W_ref)
  Var[M_W] = Var_ref * (W / W_ref)
  std[M_W] = std_ref * sqrt(W / W_ref)

Completely eliminates the variance attenuation distortion of binomial thinning.
"""

from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from private_llm_analytics.simulator.scenario import SimulationScenario
from private_llm_analytics.simulator.users import sample_user_profiles


def generate_rate_consistent_temporal_truth(
    n_users: int = 1000,
    horizon_days: int = 90,
    reference_days: int = 30,
    reference_mean: float = 60.0,
    reference_std: float = 20.0,
    seed: int = 42,
    start_date: Optional[datetime] = None,
    providers: Optional[List[str]] = None,
    tasks: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Generate longitudinal interaction log with independent daily Negative Binomial increments.

    Preserves both expected mean and variance across any time window slice W:
      mean_day = reference_mean / reference_days
      var_day  = (reference_std ** 2) / reference_days
    """
    provs = providers or ["provider_a", "provider_b", "provider_c"]
    tsk = tasks or ["technical", "document", "analysis"]
    base_date = start_date or datetime(2026, 10, 1, 0, 0, 0)

    # 1. Daily parameters
    mu_day = float(reference_mean) / float(reference_days)
    var_day = float(reference_std ** 2) / float(reference_days)
    if var_day <= mu_day:
        raise ValueError(f"var_day ({var_day}) must be > mu_day ({mu_day}) for Negative Binomial")

    # Var = mu + mu^2 / r  =>  r = mu^2 / (Var - mu)
    r = (mu_day ** 2) / (var_day - mu_day)
    # p = r / (r + mu)
    p = r / (r + mu_day)

    rng = np.random.default_rng(seed)

    # 2. Daily interaction counts for each user across horizon_days
    daily_counts = rng.negative_binomial(r, p, size=(n_users, horizon_days))

    # 3. User profile archetypes
    total_user_counts = np.sum(daily_counts, axis=1)
    user_ids, joint_probs = sample_user_profiles(
        n_users=n_users,
        interaction_counts=np.maximum(1, total_user_counts),
        providers=provs,
        tasks=tsk,
        rng=rng,
    )

    # 4. Generate records for each user and day
    cell_tuples = []
    for p_name in provs:
        for t_name in tsk:
            cell_tuples.append((p_name, t_name))

    records = []
    for u in range(n_users):
        u_id = user_ids[u]
        u_probs = joint_probs[u]
        for day in range(horizon_days):
            c_day = int(daily_counts[u, day])
            if c_day <= 0:
                continue
            cell_counts = rng.multinomial(c_day, u_probs)
            day_start = base_date + timedelta(days=day)
            for c_idx, count in enumerate(cell_counts):
                if count <= 0:
                    continue
                p_name, t_name = cell_tuples[c_idx]
                sec_offsets = rng.uniform(0, 86400, size=count)
                for sec in sec_offsets:
                    ts = day_start + timedelta(seconds=float(sec))
                    records.append({
                        "user_id": u_id,
                        "timestamp": ts,
                        "provider": p_name,
                        "task": t_name,
                    })

    df = pd.DataFrame(records)
    if not df.empty:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.sort_values("timestamp").reset_index(drop=True)
    return df
