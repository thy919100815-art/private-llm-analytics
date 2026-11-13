"""Experiment runner providing isolated research/simulation evaluation in v0.1.6."""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import numpy as np
import pandas as pd

from private_llm_analytics.aggregation.user_matrix import UserMatrix, build_user_matrix
from private_llm_analytics.aggregation.windows import filter_by_time_window
from private_llm_analytics.config.models import AppConfig
from private_llm_analytics.config.validator import ExecutionMode, ExperimentAccountingMode
from private_llm_analytics.ingestion.fingerprint import compute_dataset_fingerprint
from private_llm_analytics.planner.query import AnalyticsQuery
from private_llm_analytics.privacy.exceptions import (
    DatasetConfigurationMismatchError,
    ExperimentProductionStateCollisionError,
    RecipientSecurityViolation,
)
from private_llm_analytics.privacy.state import PrivacyStateManager
from private_llm_analytics.release.release import ReleaseOrchestrator


class ExperimentRunner:
    """Isolated runner for research experiments, parameter sweeps, and evaluations.

    Guarantees physical and logical isolation from production:
    1. Operates under ExecutionMode.EXPERIMENT, namespacing all release keys.
    2. Writes only to an isolated experiment state store (default: results/experiment_state.db).
    3. Disallows external recipient targets.
    4. Permits caller-specified deterministic noise seeds for reproducibility.
    5. Supports ExperimentAccountingMode.EVALUATION for 500+ Monte Carlo trials.
    """

    def __init__(
        self,
        config: AppConfig,
        state_db_path: Union[str, Path] = "results/experiment_state.db",
        accounting_mode: Union[str, ExperimentAccountingMode] = ExperimentAccountingMode.EVALUATION,
    ):
        self.config = config
        self.state_db_path = Path(state_db_path)
        self.accounting_mode = (
            ExperimentAccountingMode(accounting_mode)
            if isinstance(accounting_mode, str)
            else accounting_mode
        )

        # Enforce physical database isolation from production (Audit F1, v0.1.6 Fix B)
        prod_default = Path("results/privacy_state.db").resolve()
        configured_prod = (
            Path(config.privacy.release.state_db_path).resolve()
            if getattr(config.privacy.release, "state_db_path", None)
            else prod_default
        )
        canonical_prod_paths = {prod_default, configured_prod}
        if self.state_db_path.resolve() in canonical_prod_paths:
            raise ExperimentProductionStateCollisionError(
                f"ExperimentRunner state DB '{self.state_db_path}' must not collide with production state DB '{configured_prod}'."
            )

        self.state_manager = PrivacyStateManager(self.state_db_path)
        self.orchestrator = ReleaseOrchestrator(
            app_config=config,
            state_manager=self.state_manager,
            execution_mode=ExecutionMode.EXPERIMENT,
            accounting_mode=self.accounting_mode,
        )
        self._raw_dataframe: Optional[pd.DataFrame] = None
        self._cached_user_matrix: Optional[UserMatrix] = None
        self._dataset_version: Optional[str] = None

    def set_data(
        self,
        df: pd.DataFrame,
        dataset_version: Optional[str] = None,
    ) -> None:
        """Set data for experiments with automatic content fingerprinting."""
        self._raw_dataframe = df
        content_fp = compute_dataset_fingerprint(df)
        if dataset_version is None:
            self._dataset_version = content_fp
        else:
            self._dataset_version = f"{dataset_version}@{content_fp}"
        self._cached_user_matrix = None

    def get_user_matrix(
        self,
        start: Optional[Any] = None,
        end: Optional[Any] = None,
    ) -> UserMatrix:
        """Get or compute user matrix for experiment evaluation."""
        if self._raw_dataframe is None:
            raise RuntimeError("No telemetry data loaded in ExperimentRunner. Call set_data() first.")

        df = self._raw_dataframe
        pop_cfg = self.config.scenario.population
        public_n = pop_cfg.n_users if pop_cfg.mode == "fixed_public_population" else None

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

    def run_query(
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
    ) -> Dict[str, Any]:
        """Execute query in isolated experiment mode."""
        if recipient not in self.config.recipients.recipients:
            raise ValueError(f"Unknown recipient '{recipient}'")
        policy = self.config.recipients.recipients[recipient]
        if policy.role == "external":
            raise RecipientSecurityViolation(
                f"Experiment mode with seeded noise cannot target external commercial recipients (attempted for '{recipient}')."
            )

        if noise_seed is not None:
            exec_rng = np.random.default_rng(noise_seed)
        else:
            exec_rng = rng

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
            dataset_version=self._dataset_version or "experiment_v1",
            rng=exec_rng,
            include_exact=include_exact,
            noise_seed=noise_seed,
            trial_id=trial_id,
        )

    def run_trial(
        self,
        metric: str,
        recipient: str = "internal",
        seed: Optional[int] = None,
        start: Optional[Any] = None,
        end: Optional[Any] = None,
        epsilon: Optional[float] = None,
        delta: Optional[float] = None,
        mechanism: Optional[str] = None,
        include_exact: bool = False,
        trial_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Run a single experiment trial with an optional noise seed."""
        return self.run_query(
            metric=metric,
            recipient=recipient,
            start=start,
            end=end,
            epsilon=epsilon,
            delta=delta,
            mechanism=mechanism,
            include_exact=include_exact,
            noise_seed=seed,
            trial_id=trial_id,
        )

    def run_trials(
        self,
        metric: str,
        n_trials: int = 500,
        base_seed: int = 0,
        recipient: str = "internal",
        start: Optional[Any] = None,
        end: Optional[Any] = None,
        epsilon: Optional[float] = None,
        delta: Optional[float] = None,
        mechanism: Optional[str] = None,
        include_exact: bool = False,
    ) -> List[Dict[str, Any]]:
        """Run multiple Monte Carlo evaluation trials with deterministic sequential seeds."""
        results = []
        for i in range(n_trials):
            seed = base_seed + i
            res = self.run_query(
                metric=metric,
                recipient=recipient,
                start=start,
                end=end,
                epsilon=epsilon,
                delta=delta,
                mechanism=mechanism,
                include_exact=include_exact,
                noise_seed=seed,
            )
            results.append(res)
        return results
