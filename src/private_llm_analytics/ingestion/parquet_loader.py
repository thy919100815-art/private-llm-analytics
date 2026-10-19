"""Parquet loader for interaction logs."""

from pathlib import Path
from typing import Iterable, Optional, Union
import pandas as pd

from private_llm_analytics.ingestion.validation import validate_interaction_dataframe


def load_interactions_parquet(
    file_path: Union[str, Path],
    allowed_providers: Optional[Iterable[str]] = None,
    allowed_tasks: Optional[Iterable[str]] = None,
) -> pd.DataFrame:
    """Load interactions from Parquet file and validate schema."""
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"Parquet file not found: {path.resolve()}")

    df = pd.read_parquet(path)
    return validate_interaction_dataframe(
        df,
        allowed_providers=allowed_providers,
        allowed_tasks=allowed_tasks,
    )
