"""Main high-level Analytics Service facade for enterprise deployment and experimentation in v0.1.4."""

import hashlib
from pathlib import Path
from typing import Any, Dict, Optional, Union
import numpy as np
import pandas as pd

from private_llm_analytics.aggregation.user_matrix import UserMatrix, build_user_matrix
from private_llm_analytics.aggregation.windows import filter_by_time_window
from private_llm_analytics.config.loader import load_app_config
from private_llm_analytics.config.models import AppConfig
from private_llm_analytics.config.validator import (
    ExecutionMode,
    ExperimentAccountingMode,
    validate_production_config,
)
from private_llm_analytics.ingestion.csv_loader import load_interactions_csv
from private_llm_analytics.ingestion.fingerprint import compute_dataset_fingerprint
from private_llm_analytics.ingestion.parquet_loader import load_interactions_parquet
from private_llm_analytics.planner.query import AnalyticsQuery
from private_llm_analytics.privacy.composition import PrivacyView
from private_llm_analytics.privacy.exceptions import (
    DatasetConfigurationMismatchError,
    ProductionRNGControlError,
    ProductionSafetyConfigurationError,
    RecipientSecurityViolation,
)

from private_llm_analytics.release.release import ReleaseOrchestrator
from private_llm_analytics.service.experiment import ExperimentRunner
from private_llm_analytics.simulator.generator import SyntheticDataGenerator
from private_llm_analytics.simulator.scenario import SimulationScenario


class AnalyticsService:
    """Enterprise Analytics Layer providing user-level Central DP queries."""

    def __init__(self, config: AppConfig):
        self.config = config
        exec_mode = (
            ExecutionMode.EXPERIMENT
            if (not config.privacy.release.reuse_existing_release) or config.privacy.release.allow_budget_overrun
            else ExecutionMode.PRODUCTION
        )
        self.orchestrator = ReleaseOrchestrator(
            app_config=config,
            execution_mode=exec_mode,
        )
        self._raw_dataframe: Optional[pd.DataFrame] = None
        self._cached_user_matrix: Optional[UserMatrix] = None
        self._dataset_version: Optional[str] = None
        self._experiment_runner: Optional[ExperimentRunner] = None

    @classmethod
    def from_config_dir(cls, config_dir: Union[str, Path] = "configs") -> "AnalyticsService":
        """Factory method instantiating the service from standard YAML configuration directory."""
        config = load_app_config(config_dir)
        return cls(config=config)

    def load_data(
        self,
        file_path: Union[str, Path],
        file_format: Optional[str] = None,
        dataset_version: Optional[str] = None,
    ) -> pd.DataFrame:
        """Load and validate interaction telemetry from CSV or Parquet with full streaming SHA-256 versioning."""
        path = Path(file_path)
        fmt = file_format.lower() if file_format else path.suffix.lstrip(".").lower()
        provs = self.config.scenario.providers
        tsks = self.config.scenario.tasks

        if fmt == "parquet":
            df = load_interactions_parquet(path, allowed_providers=provs, allowed_tasks=tsks)
        elif fmt == "csv":
            df = load_interactions_csv(path, allowed_providers=provs, allowed_tasks=tsks)
        else:
            raise ValueError(f"Unsupported format '{fmt}'. Choose 'parquet' or 'csv'.")

        self._raw_dataframe = df

        # Full-file streaming SHA-256 fingerprinting combined with content fingerprint (Audit E3, Finding C)
        content_fp = compute_dataset_fingerprint(df)
        if dataset_version is None:
            hasher = hashlib.sha256()
            with open(path, "rb") as f:
                while chunk := f.read(1024 * 1024):
                    hasher.update(chunk)
            file_hash = hasher.hexdigest()
            self._dataset_version = f"{path.stem}_{file_hash}@{content_fp}"
        else:
            self._dataset_version = f"{dataset_version}@{content_fp}"

        self._cached_user_matrix = None  # Invalidate cached matrix
        return df

    def set_data(self, df: pd.DataFrame, dataset_version: Optional[str] = None) -> None:
        """Directly set validated DataFrame. Combines explicit dataset_version with content fingerprint."""
        self._raw_dataframe = df
        content_fp = compute_dataset_fingerprint(df)
        if dataset_version is None:
            self._dataset_version = content_fp
        else:
            self._dataset_version = f"{dataset_version}@{content_fp}"
        self._cached_user_matrix = None
        if self._experiment_runner is not None:
            self._experiment_runner.set_data(df, dataset_version=self._dataset_version)

    def generate_synthetic_data(
        self,
        n_users: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> pd.DataFrame:
        """Generate in-memory synthetic telemetry using the configured scenario.

        Audit H8 fix: When n_users is explicitly provided, updates scenario population
        to maintain mathematical consistency with public population size.
        """
        sc_cfg = self.config.scenario
        effective_n_users = n_users if n_users is not None else sc_cfg.population.n_users

        # Maintain population consistency across config and telemetry
        if n_users is not None:
            self.config.scenario.population.n_users = n_users

        effective_seed = seed if seed is not None else sc_cfg.random_seed
        scenario = SimulationScenario(
            n_users=effective_n_users,
            n_days=sc_cfg.population.n_days,
            providers=sc_cfg.providers,
            tasks=sc_cfg.tasks,
            freq_mean=sc_cfg.usage_frequency.mean,
            freq_std=sc_cfg.usage_frequency.std,
            freq_min=sc_cfg.usage_frequency.minimum,
            freq_max=sc_cfg.usage_frequency.maximum,
            random_seed=effective_seed,
        )
        generator = SyntheticDataGenerator(scenario)
        df = generator.generate(seed=effective_seed)
        self.set_data(
            df,
            dataset_version=f"synthetic_seed_{effective_seed}_n_{effective_n_users}",
        )
        return df

    def get_user_matrix(
        self,
        start: Optional[Any] = None,
        end: Optional[Any] = None,
    ) -> UserMatrix:
        """Get or compute user-level matrix with optional time-window filtering and consistency checks."""
        if self._raw_dataframe is None:
            raise RuntimeError("No telemetry data loaded. Call load_data() or generate_synthetic_data() first.")

        df = self._raw_dataframe
        pop_cfg = self.config.scenario.population
        public_n = pop_cfg.n_users if pop_cfg.mode == "fixed_public_population" else None

        # Validate active users <= public population size (Audit E4)
        n_active_users = df["user_id"].nunique()
        if public_n is not None and n_active_users > public_n:
            raise DatasetConfigurationMismatchError(
                f"Active user count ({n_active_users}) exceeds declared public population size ({public_n})."
            )

        if start is not None or end is not None:
            df = filter_by_time_window(df, start=start, end=end)
            return build_user_matrix(
                df,
                providers=self.config.scenario.providers,
                tasks=self.config.scenario.tasks,
                public_population_size=public_n,
            )

        if self._cached_user_matrix is None:
            self._cached_user_matrix = build_user_matrix(
                df,
                providers=self.config.scenario.providers,
                tasks=self.config.scenario.tasks,
                public_population_size=public_n,
            )
        return self._cached_user_matrix

    def compute(
        self,
        metric: str,
        recipient: str,
        start: Optional[Any] = None,
        end: Optional[Any] = None,
        epsilon: Optional[float] = None,
        delta: Optional[float] = None,
        mechanism: Optional[str] = None,
        clip_norm: Optional[float] = None,
        include_exact: bool = False,
        *args: Any,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """High-level query entry point for production releases.

        Production invariants enforced:
          - Validates production safety configuration (fail-closed, Audit F2a/F2b)
          - Prohibits caller-supplied RNG, seed, or noise for ALL recipients (Audit F1)
          - Enforces persistent memoization and strict budget checking
        """
        if args:
            raise ProductionRNGControlError(
                "Positional arguments beyond 'include_exact' are strictly forbidden in production compute()."
            )

        forbidden_rng_keys = {"rng", "seed", "noise_seed", "noise", "random_state"}
        for k in forbidden_rng_keys:
            if k in kwargs:
                raise ProductionRNGControlError(
                    f"Recipient '{recipient}' cannot provide or control RNG / seed / noise ('{k}' provided). "
                    "Production compute() strictly forbids caller-supplied RNG / seed / noise."
                )

        if kwargs:
            raise TypeError(f"Unexpected keyword arguments for compute(): {list(kwargs.keys())}")

        if clip_norm is not None:
            raise ValueError(
                "Query-level override of contribution bound 'clip_norm' is strictly prohibited in v0.1.4. "
                "Contribution bounds must be defined in metrics configuration."
            )

        # Enforce production safety configuration (Audit F2a, F2b)
        validate_production_config(self.config)

        if recipient not in self.config.recipients.recipients:
            raise ValueError(f"Unknown recipient '{recipient}'")

        if self._dataset_version is None and self._raw_dataframe is not None:
            self._dataset_version = compute_dataset_fingerprint(self._raw_dataframe)

        user_matrix = self.get_user_matrix(start=start, end=end)
        query = AnalyticsQuery(
            metric=metric,
            recipient=recipient,
            window_start=start,
            window_end=end,
            epsilon=epsilon,
            delta=delta,
            mechanism=mechanism,
        )
        return self.orchestrator.execute_release(
            query=query,
            user_matrix=user_matrix,
            dataset_version=self._dataset_version or "v1.0",
            rng=None,
            include_exact=include_exact,
        )

    def compute_experiment(
        self,
        metric: str,
        recipient: str = "internal",
        start: Optional[Any] = None,
        end: Optional[Any] = None,
        epsilon: Optional[float] = None,
        delta: Optional[float] = None,
        mechanism: Optional[str] = None,
        include_exact: bool = False,
        noise_seed: Optional[int] = None,
        trial_id: Optional[str] = None,
        rng: Optional[np.random.Generator] = None,
        experiment_db_path: Union[str, Path] = "results/experiment_state.db",
        accounting_mode: Union[str, ExperimentAccountingMode] = ExperimentAccountingMode.EVALUATION,
    ) -> Dict[str, Any]:
        """Dedicated experiment entry point providing isolated state and deterministic noise (Audit F1, v0.1.6 Fix A)."""
        if self._raw_dataframe is None:
            raise RuntimeError("No telemetry data loaded. Call load_data() or generate_synthetic_data() first.")

        acct_mode = (
            ExperimentAccountingMode(accounting_mode)
            if isinstance(accounting_mode, str)
            else accounting_mode
        )
        if (
            self._experiment_runner is None
            or self._experiment_runner.state_db_path.resolve() != Path(experiment_db_path).resolve()
            or getattr(self._experiment_runner, "accounting_mode", None) != acct_mode
        ):
            self._experiment_runner = ExperimentRunner(
                config=self.config,
                state_db_path=experiment_db_path,
                accounting_mode=acct_mode,
            )

        if self._dataset_version is None:
            self._dataset_version = compute_dataset_fingerprint(self._raw_dataframe)

        self._experiment_runner.set_data(self._raw_dataframe, dataset_version=self._dataset_version)
        return self._experiment_runner.run_query(
            metric=metric,
            recipient=recipient,
            start=start,
            end=end,
            epsilon=epsilon,
            delta=delta,
            mechanism=mechanism,
            include_exact=include_exact,
            noise_seed=noise_seed,
            trial_id=trial_id,
            rng=rng,
        )

    def get_privacy_view(self, recipient: str = "global_external") -> PrivacyView:
        """Retrieve authoritative privacy consumption view for a recipient or scope."""
        return self.orchestrator.get_privacy_view(recipient=recipient)

