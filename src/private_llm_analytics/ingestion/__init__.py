"""Ingestion module in v0.1.4."""

from private_llm_analytics.ingestion.csv_loader import load_interactions_csv
from private_llm_analytics.ingestion.fingerprint import compute_dataset_fingerprint
from private_llm_analytics.ingestion.parquet_loader import load_interactions_parquet
from private_llm_analytics.ingestion.schema import (
    COL_INTERACTION_ID,
    COL_PROVIDER,
    COL_TASK,
    COL_TIMESTAMP,
    COL_USER_ID,
    REQUIRED_COLUMNS,
    InteractionRecord,
)
from private_llm_analytics.ingestion.validation import (
    SchemaValidationError,
    validate_interaction_dataframe,
)

__all__ = [
    "COL_INTERACTION_ID",
    "COL_PROVIDER",
    "COL_TASK",
    "COL_TIMESTAMP",
    "COL_USER_ID",
    "InteractionRecord",
    "REQUIRED_COLUMNS",
    "SchemaValidationError",
    "compute_dataset_fingerprint",
    "load_interactions_csv",
    "load_interactions_parquet",
    "validate_interaction_dataframe",
]
