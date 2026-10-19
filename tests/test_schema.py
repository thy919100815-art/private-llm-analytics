"""Unit tests for ingestion schema and validation."""

import pytest
import pandas as pd
from datetime import datetime

from private_llm_analytics.ingestion.schema import (
    COL_INTERACTION_ID,
    COL_USER_ID,
    COL_TIMESTAMP,
    COL_PROVIDER,
    COL_TASK,
)
from private_llm_analytics.ingestion.validation import (
    validate_interaction_dataframe,
    SchemaValidationError,
)


def test_valid_schema():
    df = pd.DataFrame({
        COL_INTERACTION_ID: ["I001", "I002"],
        COL_USER_ID: ["U001", "U001"],
        COL_TIMESTAMP: ["2026-10-01T09:15:21", "2026-10-01T09:18:33"],
        COL_PROVIDER: ["provider_a", "provider_b"],
        COL_TASK: ["technical", "document"],
    })
    validated = validate_interaction_dataframe(
        df,
        allowed_providers=["provider_a", "provider_b", "provider_c"],
        allowed_tasks=["technical", "document", "analysis"],
    )
    assert len(validated) == 2
    assert pd.api.types.is_datetime64_any_dtype(validated[COL_TIMESTAMP])


def test_missing_column():
    df = pd.DataFrame({
        COL_INTERACTION_ID: ["I001"],
        COL_USER_ID: ["U001"],
        COL_TIMESTAMP: ["2026-10-01T09:15:21"],
        # Missing provider & task
    })
    with pytest.raises(SchemaValidationError, match="Missing required columns"):
        validate_interaction_dataframe(df)


def test_duplicate_interaction_id():
    df = pd.DataFrame({
        COL_INTERACTION_ID: ["I001", "I001"],
        COL_USER_ID: ["U001", "U002"],
        COL_TIMESTAMP: ["2026-10-01T09:15:21", "2026-10-01T09:18:33"],
        COL_PROVIDER: ["provider_a", "provider_b"],
        COL_TASK: ["technical", "document"],
    })
    with pytest.raises(SchemaValidationError, match="interaction_id must be globally unique"):
        validate_interaction_dataframe(df)


def test_disallowed_provider():
    df = pd.DataFrame({
        COL_INTERACTION_ID: ["I001"],
        COL_USER_ID: ["U001"],
        COL_TIMESTAMP: ["2026-10-01T09:15:21"],
        COL_PROVIDER: ["unknown_provider"],
        COL_TASK: ["technical"],
    })
    with pytest.raises(SchemaValidationError, match="Disallowed providers"):
        validate_interaction_dataframe(
            df,
            allowed_providers=["provider_a", "provider_b", "provider_c"],
        )


def test_disallowed_task():
    df = pd.DataFrame({
        COL_INTERACTION_ID: ["I001"],
        COL_USER_ID: ["U001"],
        COL_TIMESTAMP: ["2026-10-01T09:15:21"],
        COL_PROVIDER: ["provider_a"],
        COL_TASK: ["unauthorized_task"],
    })
    with pytest.raises(SchemaValidationError, match="Disallowed tasks"):
        validate_interaction_dataframe(
            df,
            allowed_tasks=["technical", "document", "analysis"],
        )
