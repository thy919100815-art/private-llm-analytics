"""Configuration validator utilities in v0.1.4."""

from enum import Enum
from typing import List
from private_llm_analytics.config.models import AppConfig
from private_llm_analytics.privacy.exceptions import ProductionSafetyConfigurationError


class ExecutionMode(str, Enum):
    """Explicit execution mode separating production release from evaluation experiments."""
    PRODUCTION = "production"
    EXPERIMENT = "experiment"


class ExperimentAccountingMode(str, Enum):
    """Explicit accounting mode for experiment evaluation versus bounded composition."""
    EVALUATION = "evaluation"
    BOUNDED = "bounded"


class ConfigValidationError(ValueError):
    """Raised when cross-configuration validation fails."""
    pass


def validate_app_config(config: AppConfig) -> List[str]:
    """Perform cross-configuration consistency validation."""
    issues = []
    allowed_providers = set(config.scenario.providers)
    allowed_tasks = set(config.scenario.tasks)

    # Validate recipients against allowed providers and metrics
    for name, policy in config.recipients.recipients.items():
        if policy.role == "external":
            for prov in policy.providers:
                if prov != "*" and prov not in allowed_providers:
                    issues.append(f"Recipient '{name}' references unknown provider '{prov}'")
            for met in policy.metrics:
                if met != "*" and met not in config.metrics.metrics:
                    issues.append(f"Recipient '{name}' references unknown metric '{met}'")

    # Validate collusion groups if configured
    if config.privacy.recipient_composition.mode == "collusion_group":
        for grp_name, members in config.privacy.recipient_composition.groups.items():
            for m in members:
                if m not in allowed_providers and m not in config.recipients.recipients:
                    issues.append(f"Collusion group '{grp_name}' references unknown member '{m}'")

    if issues:
        raise ConfigValidationError("Configuration cross-validation failed:\n" + "\n".join(issues))
    return issues


def validate_production_config(config: AppConfig) -> None:
    """Validate that configuration satisfies strict production safety invariants.

    In production mode:
      1. reuse_existing_release MUST be True (prevents noise-averaging attacks).
      2. allow_budget_overrun MUST be False (strictly enforces global budget invariant).
      3. history.mode MUST be 'lifetime' (forbids rolling privacy resets).
      4. planner.mode MUST NOT be 'oracle_eval_only' (prevents private data snooping).
    """
    if not config.privacy.release.reuse_existing_release:
        raise ProductionSafetyConfigurationError(
            "Production configuration requires privacy.release.reuse_existing_release=True. "
            "Disabling release reuse in production is strictly forbidden as it permits repeated-query noise averaging."
        )
    if config.privacy.release.allow_budget_overrun:
        raise ProductionSafetyConfigurationError(
            "Production configuration requires privacy.release.allow_budget_overrun=False. "
            "Enabling budget overrun in production is strictly forbidden as it violates global privacy guarantees."
        )
    if config.privacy.history.mode == "rolling_reset":
        raise ProductionSafetyConfigurationError(
            "Production configuration requires privacy.history.mode='lifetime'. "
            "Rolling privacy reset is strictly forbidden in production because external recipients permanently retain historical releases."
        )
    if config.privacy.planner.mode == "oracle_eval_only":
        raise ProductionSafetyConfigurationError(
            "Production configuration strictly forbids privacy.planner.mode='oracle_eval_only'. "
            "Oracle evaluation accesses private raw data and is permitted only in isolated experiment benchmarks."
        )
