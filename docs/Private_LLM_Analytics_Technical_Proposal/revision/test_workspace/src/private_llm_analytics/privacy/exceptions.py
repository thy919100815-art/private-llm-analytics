"""Domain exceptions for privacy budgeting, configuration, and security enforcement in v0.1.4."""


class PrivacyBudgetExceededError(Exception):
    """Raised when query budget exceeds remaining global privacy budget."""
    pass


class PrivacyConfigurationError(ValueError):
    """Raised when privacy configuration is invalid (e.g. missing public population)."""
    pass


class ProductionSafetyConfigurationError(PrivacyConfigurationError):
    """Raised when a production configuration violates fail-closed safety invariants (e.g. overrun enabled or reuse disabled)."""
    pass


class ExperimentProductionStateCollisionError(ValueError):
    """Raised when an experiment execution attempts to bind or write to a production privacy state database."""
    pass


class UnsupportedMechanismError(ValueError):
    """Raised when an unsupported differential privacy mechanism is requested."""
    pass


class MetricDisabledError(ValueError):
    """Raised when a query attempts to access a metric configured with enabled: false."""
    pass


class DatasetConfigurationMismatchError(ValueError):
    """Raised when dataset telemetry or manifest contradicts configuration or public roster."""
    pass


class RecipientSecurityViolation(PermissionError):
    """Raised when an unauthorized data access, exact result request, or RNG control is attempted."""
    pass


class ProductionRNGControlError(RecipientSecurityViolation):
    """Raised when caller attempts to supply or control RNG / seed in production mode."""
    pass


class ReservationLostError(RuntimeError):
    """Raised when an in-flight reservation has been superseded or lost ownership; commit is rejected."""
    pass


class ReleaseInProgressError(RuntimeError):
    """Raised when another worker is actively generating a release for the requested key."""
    pass


class CommitIrreversibleError(RuntimeError):
    """Raised when an invalid attempt is made to rollback or abort an already committed release."""
    pass


# v0.6 Statistic-Specific Profile Exceptions
class MissingRequiredStatisticColumn(ValueError):
    """Raised when interaction dataset is missing a column required by a statistic profile."""
    pass


class UnsupportedProductionMechanism(UnsupportedMechanismError):
    """Raised when caller attempts to override an approved production statistic mechanism."""
    pass


class UnknownStatisticProfile(KeyError):
    """Raised when no approved profile exists for a requested statistic."""
    pass


class InvalidStatisticProfileConfiguration(PrivacyConfigurationError):
    """Raised when an approved statistic profile's parameters fail validation."""
    pass


class StatisticPlannerNotSupported(ValueError):
    """Raised when automated utility planning is requested for a statistic that does not support it."""
    pass
