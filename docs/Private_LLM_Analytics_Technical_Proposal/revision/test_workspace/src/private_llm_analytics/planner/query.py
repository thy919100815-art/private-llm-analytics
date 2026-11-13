"""Query definitions for analytics requests in v0.1.1."""

from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Union


@dataclass
class AnalyticsQuery:
    """Represents an incoming query for differential privacy analytics."""
    metric: str
    recipient: str
    window_start: Optional[Union[str, datetime]] = None
    window_end: Optional[Union[str, datetime]] = None
    epsilon: Optional[float] = None
    delta: Optional[float] = None
    mechanism: Optional[str] = None
    clip_norm: Optional[float] = None
    seed: Optional[int] = None

    def __post_init__(self):
        if self.clip_norm is not None:
            raise ValueError(
                "Query-level override of contribution bound 'clip_norm' is strictly prohibited in v0.1.1. "
                "Contribution bounds must be defined in metrics configuration."
            )
