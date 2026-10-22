"""Temporal User Contribution Builder with Group-L1-of-L2 Bounding (v0.4).

Implements authoritative user-level temporal clipping:
  1. Target-space projection under direct granularity operator A_G
  2. Daily L2 clipping bounded by C_day
  3. Epoch-wide group-L1-of-L2 clipping bounded by C_epoch
Ensures that for any tree level l with mutually disjoint time intervals,
the concatenated joint vector V_{u, l} satisfies ||V_{u, l}||_2 <= C_epoch.
"""

from datetime import datetime, timezone
import hashlib
import json
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import numpy as np
import pandas as pd

from private_llm_analytics.aggregation.grouping import (
    DEFAULT_PROVIDERS,
    DEFAULT_TASKS,
    build_cell_definitions,
)
from private_llm_analytics.continual.models import (
    SUPPORTED_CONTINUAL_METRICS,
    TemporalContributionDiagnostics,
)
from private_llm_analytics.ingestion.schema import (
    COL_PROVIDER,
    COL_TASK,
    COL_TIMESTAMP,
    COL_USER_ID,
)
from private_llm_analytics.planning.policy import build_aggregation_matrix
from private_llm_analytics.privacy.exceptions import (
    DatasetConfigurationMismatchError,
    PrivacyConfigurationError,
    ProductionSafetyConfigurationError,
)


def build_temporal_user_contributions(
    interactions: Union[pd.DataFrame, Sequence[Dict[str, Any]]],
    public_user_roster: Sequence[str],
    epoch_start: Union[str, datetime, pd.Timestamp],
    epoch_days: int,
    granularity: str = "full_9_cell",
    daily_l2_bound: float = 10.0,
    epoch_l12_bound: float = 40.0,
    metric: str = "interaction_volume",
    timezone_name: str = "UTC",
    providers: Optional[List[str]] = None,
    tasks: Optional[List[str]] = None,
) -> Tuple[np.ndarray, TemporalContributionDiagnostics, List[str]]:
    """Build bounded temporal user contributions in target granularity space.

    Args:
        interactions: Interaction event telemetry containing user_id, timestamp, provider, task.
        public_user_roster: Fixed list of all users in the eligible population.
        epoch_start: Epoch start date or timestamp (inclusive).
        epoch_days: Number of days in the epoch.
        granularity: One of 'full_9_cell', 'provider_3_cell', 'task_3_cell', 'overall_1_cell'.
        daily_l2_bound: C_day daily L2 clipping threshold.
        epoch_l12_bound: C_epoch epoch group-L1-of-L2 clipping threshold.
        metric: Must be 'interaction_volume' in v0.4.
        timezone_name: Timezone for day binning (default 'UTC').
        providers: Optional custom provider names (default 3 providers).
        tasks: Optional custom task names (default 3 tasks).

    Returns:
        (contributions, diagnostics, cell_names):
            contributions: np.ndarray of shape (n_users, epoch_days, dim), dtype float64.
            diagnostics: TemporalContributionDiagnostics with clipping statistics.
            cell_names: List of target cell names.
    """
    norm_metric = str(metric).lower().strip()
    if norm_metric not in SUPPORTED_CONTINUAL_METRICS:
        raise ProductionSafetyConfigurationError(
            f"Continual observation path only supports metrics {SUPPORTED_CONTINUAL_METRICS}, got '{metric}'."
        )

    if epoch_days <= 0:
        raise PrivacyConfigurationError(f"epoch_days must be > 0, got {epoch_days}")
    if daily_l2_bound <= 0:
        raise PrivacyConfigurationError(f"daily_l2_bound must be > 0, got {daily_l2_bound}")
    if epoch_l12_bound <= 0:
        raise PrivacyConfigurationError(f"epoch_l12_bound must be > 0, got {epoch_l12_bound}")
    if not public_user_roster:
        raise PrivacyConfigurationError("public_user_roster cannot be empty.")

    # Prepare DataFrame
    if isinstance(interactions, pd.DataFrame):
        df = interactions
    else:
        df = pd.DataFrame(interactions)

    # Standardize providers and tasks
    provs = list(providers) if providers is not None else DEFAULT_PROVIDERS
    tsks = list(tasks) if tasks is not None else DEFAULT_TASKS
    source_9_names, cell_map = build_cell_definitions(provs, tsks)

    # Build target aggregation operator A_G
    A_G, target_names = build_aggregation_matrix(granularity, source_9_names)
    dim_target = A_G.shape[0]

    # Map public user roster to fixed indices
    user_list = list(public_user_roster)
    n_users = len(user_list)
    user_to_idx = {uid: i for i, uid in enumerate(user_list)}

    # Parse and localize epoch start timestamp
    try:
        dt_start = pd.to_datetime(epoch_start)
        if dt_start.tzinfo is None:
            dt_start = dt_start.tz_localize(timezone_name)
        else:
            dt_start = dt_start.tz_convert(timezone_name)
        dt_start = dt_start.floor("D")
    except Exception as e:
        raise PrivacyConfigurationError(f"Failed to parse epoch_start '{epoch_start}': {e}")

    # Raw 9-cell tensor: (n_users, epoch_days, len(source_9_names))
    n_source_cells = len(source_9_names)
    raw_source_tensor = np.zeros((n_users, epoch_days, n_source_cells), dtype=np.float64)

    if not df.empty and COL_TIMESTAMP in df.columns and COL_USER_ID in df.columns:
        ts_series = pd.to_datetime(df[COL_TIMESTAMP])
        if ts_series.dt.tz is None:
            ts_series = ts_series.dt.tz_localize(timezone_name)
        else:
            ts_series = ts_series.dt.tz_convert(timezone_name)

        day_diffs = (ts_series.dt.floor("D") - dt_start).dt.days.values
        valid_time_mask = (day_diffs >= 0) & (day_diffs < epoch_days)

        valid_df = df[valid_time_mask].copy()
        valid_days = day_diffs[valid_time_mask]

        if not valid_df.empty:
            valid_users = valid_df[COL_USER_ID].values
            valid_provs = valid_df[COL_PROVIDER].values if COL_PROVIDER in valid_df.columns else ["Unknown"] * len(valid_df)
            valid_tasks = valid_df[COL_TASK].values if COL_TASK in valid_df.columns else ["Unknown"] * len(valid_df)

            for i in range(len(valid_df)):
                uid = valid_users[i]
                u_idx = user_to_idx.get(uid)
                if u_idx is None:
                    continue  # Non-roster events do not contribute
                t = valid_days[i]
                c_idx = cell_map.get((valid_provs[i], valid_tasks[i]))
                if c_idx is not None and c_idx < n_source_cells:
                    raw_source_tensor[u_idx, t, c_idx] += 1.0

    # Project raw 9-cell to target granularity: (n_users, epoch_days, dim_target)
    # x_{u, t} = A_G @ raw_source_{u, t}
    raw_target_tensor = np.einsum("gd,utd->utg", A_G, raw_source_tensor)

    # 1. Daily L2 Clipping: y_{u, t} = x_{u, t} * min(1, C_day / ||x_{u, t}||_2)
    daily_norms = np.linalg.norm(raw_target_tensor, axis=2)  # (n_users, epoch_days)
    daily_clip_factors = np.ones_like(daily_norms)
    active_daily = daily_norms > 1e-12
    daily_clip_factors[active_daily] = np.minimum(1.0, daily_l2_bound / daily_norms[active_daily])

    daily_clipped_tensor = raw_target_tensor * daily_clip_factors[:, :, np.newaxis]

    # 2. Epoch Group-L1-of-L2 Clipping:
    # S_u = sum_t ||y_{u, t}||_2
    # z_{u, t} = y_{u, t} * min(1, C_epoch / S_u)
    clipped_daily_norms = np.linalg.norm(daily_clipped_tensor, axis=2)  # (n_users, epoch_days)
    epoch_group_norms = np.sum(clipped_daily_norms, axis=1)  # (n_users,)

    epoch_clip_factors = np.ones_like(epoch_group_norms)
    active_users = epoch_group_norms > 1e-12
    epoch_clip_factors[active_users] = np.minimum(1.0, epoch_l12_bound / epoch_group_norms[active_users])

    final_contributions = daily_clipped_tensor * epoch_clip_factors[:, np.newaxis, np.newaxis]

    # Diagnostics calculation
    raw_total = float(np.sum(raw_target_tensor))
    daily_clipped_total = float(np.sum(daily_clipped_tensor))
    epoch_clipped_total = float(np.sum(final_contributions))

    daily_bias = raw_total - daily_clipped_total
    temporal_bias = daily_clipped_total - epoch_clipped_total

    n_active_daily = int(np.sum(active_daily))
    n_clipped_daily = int(np.sum(active_daily & (daily_norms > daily_l2_bound + 1e-10)))
    frac_daily_clipped = float(n_clipped_daily / n_active_daily) if n_active_daily > 0 else 0.0

    n_active_users = int(np.sum(active_users))
    n_clipped_users = int(np.sum(active_users & (epoch_group_norms > epoch_l12_bound + 1e-10)))
    frac_epoch_clipped = float(n_clipped_users / n_active_users) if n_active_users > 0 else 0.0

    diagnostics = TemporalContributionDiagnostics(
        raw_total=raw_total,
        daily_clipped_total=daily_clipped_total,
        epoch_clipped_total=epoch_clipped_total,
        daily_clipping_bias=daily_bias,
        temporal_clipping_bias=temporal_bias,
        fraction_daily_clipped=frac_daily_clipped,
        fraction_epoch_clipped=frac_epoch_clipped,
        n_users=n_users,
        epoch_days=epoch_days,
        dimension=dim_target,
    )

    return final_contributions, diagnostics, target_names


def compute_partition_fingerprint(
    interactions: Union[pd.DataFrame, Sequence[Dict[str, Any]]],
    epoch_start: Union[str, datetime, pd.Timestamp],
    day: int,
    timezone_name: str = "UTC",
) -> str:
    """Compute deterministic SHA-256 fingerprint for a single day's interaction partition."""
    if isinstance(interactions, pd.DataFrame):
        df = interactions
    else:
        df = pd.DataFrame(interactions)

    if df.empty or COL_TIMESTAMP not in df.columns:
        return hashlib.sha256(f"empty_partition_day_{day}".encode("utf-8")).hexdigest()

    try:
        dt_start = pd.to_datetime(epoch_start)
        if dt_start.tzinfo is None:
            dt_start = dt_start.tz_localize(timezone_name)
        else:
            dt_start = dt_start.tz_convert(timezone_name)
        dt_start = dt_start.floor("D")
    except Exception as e:
        raise PrivacyConfigurationError(f"Failed to parse epoch_start '{epoch_start}': {e}")

    ts_series = pd.to_datetime(df[COL_TIMESTAMP])
    if ts_series.dt.tz is None:
        ts_series = ts_series.dt.tz_localize(timezone_name)
    else:
        ts_series = ts_series.dt.tz_convert(timezone_name)

    target_diff = day - 1
    day_diffs = (ts_series.dt.floor("D") - dt_start).dt.days.values
    mask = day_diffs == target_diff

    day_df = df[mask]
    if day_df.empty:
        return hashlib.sha256(f"empty_partition_day_{day}".encode("utf-8")).hexdigest()

    sort_cols = [c for c in [COL_TIMESTAMP, COL_USER_ID, COL_PROVIDER, COL_TASK] if c in day_df.columns]
    if not sort_cols:
        sort_cols = sorted(list(day_df.columns))

    sorted_df = day_df.sort_values(by=sort_cols).reset_index(drop=True)
    records = sorted_df.to_dict(orient="records")
    canonical_repr = json.dumps(records, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(canonical_repr).hexdigest()


def compute_partition_fingerprints(
    interactions: Union[pd.DataFrame, Sequence[Dict[str, Any]]],
    epoch_start: Union[str, datetime, pd.Timestamp],
    epoch_days: int,
    timezone_name: str = "UTC",
) -> Dict[str, str]:
    """Compute partition fingerprints for all days 1..epoch_days in an epoch."""
    return {
        str(day): compute_partition_fingerprint(interactions, epoch_start, day, timezone_name)
        for day in range(1, epoch_days + 1)
    }

