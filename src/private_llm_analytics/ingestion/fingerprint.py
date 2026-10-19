"""Deterministic dataset content fingerprinting in v0.1.4."""

import hashlib
import pandas as pd


def compute_dataset_fingerprint(df: pd.DataFrame) -> str:
    """Compute a deterministic SHA-256 fingerprint from DataFrame schema, columns, and content.

    Canonicalizes column order and row multiset order to guarantee deterministic hashing
    regardless of in-memory row order.
    """
    if df is None or df.empty:
        return "empty_dataset"

    sorted_cols = sorted(df.columns.tolist())
    canonical_df = df[sorted_cols].copy()
    try:
        canonical_df = canonical_df.sort_values(by=sorted_cols, kind="mergesort").reset_index(drop=True)
    except Exception:
        pass

    hasher = hashlib.sha256()
    # Schema string: col_name:col_dtype
    schema_str = ";".join(f"{c}:{canonical_df[c].dtype}" for c in sorted_cols)
    hasher.update(schema_str.encode("utf-8"))

    # Hash row contents deterministically
    try:
        import pandas.util
        row_hashes = pandas.util.hash_pandas_object(canonical_df, index=False).values
        hasher.update(row_hashes.tobytes())
    except Exception:
        hasher.update(canonical_df.to_csv(index=False).encode("utf-8"))

    return f"data_sha256_{hasher.hexdigest()}"
