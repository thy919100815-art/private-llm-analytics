"""Data validation routines for interaction records."""

from typing import Iterable, Optional, Set
import pandas as pd

from private_llm_analytics.ingestion.schema import (
    COL_INTERACTION_ID,
    COL_PROVIDER,
    COL_TASK,
    COL_TIMESTAMP,
    COL_USER_ID,
    REQUIRED_COLUMNS,
)


class SchemaValidationError(ValueError):
    """Raised when interaction data fails schema or domain validation."""
    pass


def validate_interaction_dataframe(
    df: pd.DataFrame,
    allowed_providers: Optional[Iterable[str]] = None,
    allowed_tasks: Optional[Iterable[str]] = None,
) -> pd.DataFrame:
    """Validate that DataFrame conforms to the interaction schema.

    Returns the validated DataFrame with parsed datetime timestamps.
    Raises SchemaValidationError on any violation.
    """
    if not isinstance(df, pd.DataFrame):
        raise SchemaValidationError(f"Expected pandas DataFrame, got {type(df)}")

    # 1. Check required columns
    missing_cols = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_cols:
        raise SchemaValidationError(f"Missing required columns in dataset: {missing_cols}")

    if df.empty:
        raise SchemaValidationError("Dataset is empty; at least 1 interaction record is required.")

    # 2. Check nulls
    null_counts = df[REQUIRED_COLUMNS].isnull().sum()
    cols_with_nulls = null_counts[null_counts > 0]
    if not cols_with_nulls.empty:
        raise SchemaValidationError(f"Null values detected in columns: {cols_with_nulls.to_dict()}")

    # 3. Check uniqueness of interaction_id
    if not df[COL_INTERACTION_ID].is_unique:
        duplicates_count = df.duplicated(subset=[COL_INTERACTION_ID]).sum()
        raise SchemaValidationError(
            f"interaction_id must be globally unique; found {duplicates_count} duplicate IDs."
        )

    # 4. Check & parse timestamp
    if not pd.api.types.is_datetime64_any_dtype(df[COL_TIMESTAMP]):
        try:
            df = df.copy()
            df[COL_TIMESTAMP] = pd.to_datetime(df[COL_TIMESTAMP])
        except Exception as e:
            raise SchemaValidationError(f"Failed to parse '{COL_TIMESTAMP}' as datetime: {e}") from e

    # 5. Check allowed providers
    if allowed_providers is not None:
        allowed_p_set: Set[str] = set(allowed_providers)
        unique_providers = set(df[COL_PROVIDER].unique())
        invalid_providers = unique_providers - allowed_p_set
        if invalid_providers:
            raise SchemaValidationError(
                f"Disallowed providers found: {sorted(list(invalid_providers))}. "
                f"Allowed providers: {sorted(list(allowed_p_set))}"
            )

    # 6. Check allowed tasks
    if allowed_tasks is not None:
        allowed_t_set: Set[str] = set(allowed_tasks)
        unique_tasks = set(df[COL_TASK].unique())
        invalid_tasks = unique_tasks - allowed_t_set
        if invalid_tasks:
            raise SchemaValidationError(
                f"Disallowed tasks found: {sorted(list(invalid_tasks))}. "
                f"Allowed tasks: {sorted(list(allowed_t_set))}"
            )

    return df
