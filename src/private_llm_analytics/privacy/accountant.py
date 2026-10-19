"""Privacy Accountant interface and persistent basic accountant in v0.1.4."""

from abc import ABC, abstractmethod
from decimal import Decimal
from pathlib import Path
from typing import Any, List, Optional
from private_llm_analytics.privacy.budget import PrivacyBudget
from private_llm_analytics.privacy.exceptions import PrivacyBudgetExceededError
from private_llm_analytics.privacy.ledger import PrivacyLedger
from private_llm_analytics.privacy.state import PrivacyStateManager


def _to_decimal(val: Any) -> Decimal:
    if isinstance(val, Decimal):
        return val
    return Decimal(str(val))


class PrivacyAccountant(ABC):
    """Abstract interface for privacy accounting (single release, basic, or RDP)."""

    @abstractmethod
    def check_budget_available(self, budget_needed: PrivacyBudget) -> bool:
        pass

    @abstractmethod
    def consume_budget(self, budget: PrivacyBudget, query_id: str) -> None:
        pass

    @abstractmethod
    def get_total_spent(self) -> PrivacyBudget:
        pass


class BasicAccountant(PrivacyAccountant):
    """Persistent basic accountant reading strictly from PrivacyStateManager (single source of truth)."""

    def __init__(
        self,
        global_budget: PrivacyBudget,
        allow_budget_overrun: bool = False,
        ledger: Optional[PrivacyLedger] = None,
        state_manager: Optional[PrivacyStateManager] = None,
    ):
        self.global_budget = global_budget
        self.allow_budget_overrun = allow_budget_overrun
        self.ledger = ledger  # Retained solely for backward compatibility with deprecated legacy fixtures
        self.history: List[dict] = []
        self._manual_spent_epsilon: Optional[float] = None
        self._manual_spent_delta: Optional[float] = None

        if state_manager is not None:
            self.state_manager = state_manager
        elif ledger is not None:
            db_p = getattr(ledger, "db_path", None)
            if db_p is not None:
                state_file = Path(db_p).parent / "privacy_state.db"
                self.state_manager = PrivacyStateManager(state_file)
            else:
                self.state_manager = PrivacyStateManager()
        else:
            self.state_manager = PrivacyStateManager()

    @property
    def spent_epsilon(self) -> float:
        """Cumulative committed epsilon strictly from authoritative PrivacyStateManager."""
        if self._manual_spent_epsilon is not None:
            return self._manual_spent_epsilon
        eps, _ = self.state_manager.get_total_spent()
        return eps

    @spent_epsilon.setter
    def spent_epsilon(self, val: float):
        self._manual_spent_epsilon = val

    @property
    def spent_delta(self) -> float:
        """Cumulative committed delta strictly from authoritative PrivacyStateManager."""
        if self._manual_spent_delta is not None:
            return self._manual_spent_delta
        _, delta = self.state_manager.get_total_spent()
        return delta

    @spent_delta.setter
    def spent_delta(self, val: float):
        self._manual_spent_delta = val

    def check_budget_available(self, budget_needed: PrivacyBudget) -> bool:
        """Exact Decimal comparison preventing tolerance leaks."""
        if self.allow_budget_overrun:
            return True
        spent_eps, spent_del = self.state_manager.get_total_reserved_or_spent()
        eps_ok = (_to_decimal(spent_eps) + _to_decimal(budget_needed.epsilon)) <= _to_decimal(self.global_budget.epsilon)
        delta_ok = (_to_decimal(spent_del) + _to_decimal(budget_needed.delta)) <= _to_decimal(self.global_budget.delta)
        return bool(eps_ok and delta_ok)

    def consume_budget(self, budget: PrivacyBudget, query_id: str) -> None:
        """Atomically record budget consumption in database."""
        status, _, res_id, _ = self.state_manager.reserve_or_get_release(
            release_key=query_id,
            canonical_semantics={"query_id": query_id},
            global_budget=self.global_budget,
            budget_needed=budget,
            plan_metadata={},
            allow_overrun=self.allow_budget_overrun,
        )

        entry = {
            "query_id": query_id,
            "epsilon": budget.epsilon,
            "delta": budget.delta,
            "status": status,
        }
        self.history.append(entry)

    def get_total_spent(self) -> PrivacyBudget:
        return PrivacyBudget(epsilon=self.spent_epsilon, delta=self.spent_delta)
