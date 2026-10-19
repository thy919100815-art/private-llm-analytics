"""Privacy module in v0.1.4."""

from private_llm_analytics.privacy.exceptions import (
    CommitIrreversibleError,
    DatasetConfigurationMismatchError,
    MetricDisabledError,
    PrivacyBudgetExceededError,
    PrivacyConfigurationError,
    ProductionRNGControlError,
    ProductionSafetyConfigurationError,
    RecipientSecurityViolation,
    ReleaseInProgressError,
    ReservationLostError,
    UnsupportedMechanismError,
)
from private_llm_analytics.privacy.budget import PrivacyBudget
from private_llm_analytics.privacy.policy import PrivacyPolicy


def __getattr__(name: str):
    if name in ("BasicAccountant", "PrivacyAccountant"):
        from private_llm_analytics.privacy.accountant import BasicAccountant, PrivacyAccountant
        globals()["BasicAccountant"] = BasicAccountant
        globals()["PrivacyAccountant"] = PrivacyAccountant
        return globals()[name]
    if name in ("PrivacyLedger", "LedgerEntry"):
        from private_llm_analytics.privacy.ledger import LedgerEntry, PrivacyLedger
        globals()["PrivacyLedger"] = PrivacyLedger
        globals()["LedgerEntry"] = LedgerEntry
        return globals()[name]
    if name == "PrivacyStateManager":
        from private_llm_analytics.privacy.state import PrivacyStateManager
        globals()["PrivacyStateManager"] = PrivacyStateManager
        return globals()[name]
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


__all__ = [
    "BasicAccountant",
    "CommitIrreversibleError",
    "DatasetConfigurationMismatchError",
    "LedgerEntry",
    "MetricDisabledError",
    "PrivacyAccountant",
    "PrivacyBudget",
    "PrivacyBudgetExceededError",
    "PrivacyConfigurationError",
    "PrivacyLedger",
    "PrivacyPolicy",
    "PrivacyStateManager",
    "ProductionRNGControlError",
    "ProductionSafetyConfigurationError",
    "RecipientSecurityViolation",
    "ReleaseInProgressError",
    "ReservationLostError",
    "UnsupportedMechanismError",
]
