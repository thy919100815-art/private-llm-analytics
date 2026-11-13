"""Privacy budget definitions."""

from dataclasses import dataclass
from decimal import Decimal
from typing import Union


@dataclass(frozen=True)
class PrivacyBudget:
    epsilon: Union[float, Decimal]
    delta: Union[float, Decimal] = 0.0

    def __post_init__(self):
        eps_d = Decimal(str(self.epsilon)) if not isinstance(self.epsilon, Decimal) else self.epsilon
        delta_d = Decimal(str(self.delta)) if not isinstance(self.delta, Decimal) else self.delta

        if eps_d <= 0:
            raise ValueError(f"epsilon must be > 0, got {self.epsilon}")
        if delta_d < 0 or delta_d >= 1:
            raise ValueError(f"delta must be in [0, 1), got {self.delta}")

    def is_pure_dp(self) -> bool:
        delta_d = Decimal(str(self.delta)) if not isinstance(self.delta, Decimal) else self.delta
        return delta_d == 0


@dataclass(frozen=True)
class PrivacySpent:
    """Represents actual consumed privacy budget (which can be exactly zero)."""
    epsilon: Union[float, Decimal]
    delta: Union[float, Decimal] = 0.0

    def __post_init__(self):
        eps_d = Decimal(str(self.epsilon)) if not isinstance(self.epsilon, Decimal) else self.epsilon
        delta_d = Decimal(str(self.delta)) if not isinstance(self.delta, Decimal) else self.delta

        if eps_d < 0:
            raise ValueError(f"epsilon must be >= 0, got {self.epsilon}")
        if delta_d < 0 or delta_d >= 1:
            raise ValueError(f"delta must be in [0, 1), got {self.delta}")

        object.__setattr__(self, "epsilon", eps_d)
        object.__setattr__(self, "delta", delta_d)

    @property
    def epsilon_float(self) -> float:
        return float(self.epsilon)

    @property
    def delta_float(self) -> float:
        return float(self.delta)

    def is_pure_dp(self) -> bool:
        return self.delta == 0
