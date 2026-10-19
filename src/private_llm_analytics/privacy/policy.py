"""Privacy policy and compliance checker."""

from typing import Literal
from private_llm_analytics.privacy.budget import PrivacyBudget


class PrivacyPolicy:
    """Enforces central user-level DP requirements."""

    def __init__(
        self,
        privacy_unit: Literal["user"] = "user",
        adjacency: Literal["add_remove"] = "add_remove",
    ):
        if privacy_unit != "user":
            raise ValueError(f"v0.1 strictly requires privacy_unit='user', got '{privacy_unit}'")
        if adjacency != "add_remove":
            raise ValueError(f"v0.1 strictly requires adjacency='add_remove', got '{adjacency}'")
        self.privacy_unit = privacy_unit
        self.adjacency = adjacency

    def validate_budget(self, budget: PrivacyBudget) -> None:
        if budget.epsilon <= 0:
            raise ValueError(f"Epsilon must be strictly positive, got {budget.epsilon}")
        if budget.delta < 0 or budget.delta >= 1.0:
            raise ValueError(f"Delta must be in [0, 1), got {budget.delta}")
