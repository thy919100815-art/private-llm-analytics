"""Scenario and parameter dataclasses for synthetic data simulation."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import List


@dataclass
class SimulationScenario:
    n_users: int = 10000
    n_days: int = 30
    start_date: datetime = field(default_factory=lambda: datetime(2026, 10, 1, 0, 0, 0))
    providers: List[str] = field(default_factory=lambda: ["provider_a", "provider_b", "provider_c"])
    tasks: List[str] = field(default_factory=lambda: ["technical", "document", "analysis"])
    freq_mean: float = 60.0
    freq_std: float = 20.0
    freq_min: int = 5
    freq_max: int = 150
    random_seed: int = 42
