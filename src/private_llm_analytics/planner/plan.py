from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Optional, Tuple


@dataclass
class ExecutionPlan:
    """Detailed plan created by planner prior to query execution."""
    metric: str
    time_window: Tuple[Optional[datetime], Optional[datetime]]
    privacy_unit: str
    contribution_method: str
    sensitivity: float
    mechanism: str
    epsilon: float
    delta: float
    action: str
    release_key: str
    canonical_semantics: Dict[str, Any] = field(default_factory=dict)
    public_population_size: Optional[int] = None
    execution_mode: str = "production"
    privacy_scope_id: Optional[str] = None
    statistic_id: Optional[str] = None
    profile_id: Optional[str] = None
    profile_version: Optional[str] = None
    profile_fingerprint: Optional[str] = None

