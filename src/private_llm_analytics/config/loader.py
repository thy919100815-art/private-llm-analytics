"""Configuration loader for YAML files."""

import os
from pathlib import Path
from typing import Optional, Union
import yaml

from private_llm_analytics.config.models import (
    AppConfig,
    MetricsConfig,
    PrivacyConfig,
    RecipientsConfig,
    ScenarioConfig,
)


def load_yaml(file_path: Union[str, Path]) -> dict:
    """Safely load a YAML file into a dictionary."""
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {path.resolve()}")
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if data is None:
        return {}
    return data


def load_scenario_config(config_path: Union[str, Path]) -> ScenarioConfig:
    data = load_yaml(config_path)
    inner = data.get("scenario", data)
    return ScenarioConfig.model_validate(inner)


def load_privacy_config(config_path: Union[str, Path]) -> PrivacyConfig:
    data = load_yaml(config_path)
    inner = data.get("privacy", data)
    return PrivacyConfig.model_validate(inner)


def load_metrics_config(config_path: Union[str, Path]) -> MetricsConfig:
    data = load_yaml(config_path)
    if "metrics" in data and isinstance(data["metrics"], dict) and "metrics" in data["metrics"]:
        inner = data["metrics"]
    else:
        inner = data
    return MetricsConfig.model_validate(inner)


def load_recipients_config(config_path: Union[str, Path]) -> RecipientsConfig:
    data = load_yaml(config_path)
    if "recipients" in data and isinstance(data["recipients"], dict) and "recipients" in data["recipients"]:
        inner = data["recipients"]
    else:
        inner = data
    return RecipientsConfig.model_validate(inner)


def load_app_config(config_dir: Union[str, Path]) -> AppConfig:
    from private_llm_analytics.config.validator import validate_app_config

    cdir = Path(config_dir)
    scenario = load_scenario_config(cdir / "scenario.yaml")
    privacy = load_privacy_config(cdir / "privacy.yaml")
    metrics = load_metrics_config(cdir / "metrics.yaml")
    recipients = load_recipients_config(cdir / "recipients.yaml")
    app_config = AppConfig(
        scenario=scenario,
        privacy=privacy,
        metrics=metrics,
        recipients=recipients,
    )
    validate_app_config(app_config)
    return app_config

