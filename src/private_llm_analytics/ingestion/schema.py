"""Schema definition for interaction events."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional

# Required column names for the long-format interaction logs
COL_INTERACTION_ID = "interaction_id"
COL_USER_ID = "user_id"
COL_TIMESTAMP = "timestamp"
COL_PROVIDER = "provider"
COL_TASK = "task"
COL_SUCCESS = "success"

REQUIRED_COLUMNS: List[str] = [
    COL_INTERACTION_ID,
    COL_USER_ID,
    COL_TIMESTAMP,
    COL_PROVIDER,
    COL_TASK,
]


@dataclass(frozen=True)
class InteractionRecord:
    interaction_id: str
    user_id: str
    timestamp: datetime
    provider: str
    task: str
    success: Optional[bool] = None
