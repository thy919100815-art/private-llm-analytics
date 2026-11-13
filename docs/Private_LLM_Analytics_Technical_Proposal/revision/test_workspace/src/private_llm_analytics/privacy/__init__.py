"""Privacy module in v0.2."""

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
from private_llm_analytics.privacy.composition import PrivacyView, RDPReleaseCost
from private_llm_analytics.privacy.policy import PrivacyPolicy
from private_llm_analytics.privacy.rdp import (
    DEFAULT_RDP_ORDERS,
    compute_gaussian_rdp,
    compute_laplace_rdp_bound,
    compose_rdp_vectors,
    rdp_to_epsilon_delta,
    rdp_vector_from_json,
    rdp_vector_to_json,
    validate_delta,
    validate_orders,
)


def __getattr__(name: str):
    if name in ("BasicAccountant", "PrivacyAccountant"):
        from private_llm_analytics.privacy.accountant import BasicAccountant, PrivacyAccountant
        globals()["BasicAccountant"] = BasicAccountant
        globals()["PrivacyAccountant"] = PrivacyAccountant
        return globals()[name]
    if name == "RDPAccountant":
        from private_llm_analytics.privacy.rdp_accountant import RDPAccountant
        globals()["RDPAccountant"] = RDPAccountant
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
    "DEFAULT_RDP_ORDERS",
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
    "PrivacyView",
    "ProductionRNGControlError",
    "ProductionSafetyConfigurationError",
    "RDPAccountant",
    "RDPReleaseCost",
    "RecipientSecurityViolation",
    "ReleaseInProgressError",
    "ReservationLostError",
    "UnsupportedMechanismError",
    "compute_gaussian_rdp",
    "compute_laplace_rdp_bound",
    "compose_rdp_vectors",
    "rdp_to_epsilon_delta",
    "rdp_vector_from_json",
    "rdp_vector_to_json",
    "validate_delta",
    "validate_orders",
]
