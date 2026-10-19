"""Time window filtering routines."""

from datetime import datetime
from typing import Optional, Union
import pandas as pd

from private_llm_analytics.ingestion.schema import COL_TIMESTAMP


def filter_by_time_window(
    df: pd.DataFrame,
    start: Optional[Union[str, datetime]] = None,
    end: Optional[Union[str, datetime]] = None,
) -> pd.DataFrame:
    """Filter interaction DataFrame to [start, end) interval."""
    filtered_df = df
    if start is not None:
        start_dt = pd.to_datetime(start)
        filtered_df = filtered_df[filtered_df[COL_TIMESTAMP] >= start_dt]
    if end is not None:
        end_dt = pd.to_datetime(end)
        filtered_df = filtered_df[filtered_df[COL_TIMESTAMP] < end_dt]
    return filtered_df
