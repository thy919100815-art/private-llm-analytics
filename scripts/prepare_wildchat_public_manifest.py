"""Public-Data Reproducibility & Minimal Aggregate Manifest for allenai/WildChat-4.8M (v6).

Addresses P0 Public-Data Reproducibility:
1. Pins the public Hugging Face Hub dataset `allenai/WildChat-4.8M` at revision
   `c827c6df8fcf008219ffaffa4d1dd77491099367` (2025-08-11T15:12:58.000Z),
   streams ONLY the 4 projected columns (`hashed_ip`, `timestamp`, `model`,
   `country`) across all 86 parquet shards in parallel, and applies deterministic
   schema normalization + sorting.
2. Distinguishes:
   - `input_file_sha256`: binary hash of the local parquet file (can vary across
     parquet writer versions/compression metadata).
   - `logical_records_sha256`: canonical SHA-256 over deterministically sorted
     rows `(timestamp_utc, hashed_ip, model, country)` with `fillna("")` for
     missing `country` (217,369 nulls out of 3,199,860 rows).
3. Supports `--no-git-fallback` to prove reproducibility in a fresh clone with
   no local `.cache` and no local git object fallback.
4. Explicitly documents dataset boundaries:
   - 3,199,860-conversation public filtered release of `allenai/WildChat-4.8M`.
   - `hashed_ip` is an IP pseudonym, not an authenticated account ID.
   - Top-level `timestamp` is the last-turn conversation timestamp, not an HTTP
     request timestamp, and 1 conversation != 1 HTTP API request.
   - The 20 categories (`5 model_families x 4 UTC_6h_quadrants`) are an
     interaction-category proxy, not semantic NLP task topics.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import io
import json
from pathlib import Path
import platform
import subprocess
import sys
import time
import urllib.request

import numpy as np
import pandas as pd
import pyarrow
import scipy

ROOT = Path(__file__).resolve().parents[1]
HF_DATASET_ID = "allenai/WildChat-4.8M"
HF_PINNED_REVISION = "c827c6df8fcf008219ffaffa4d1dd77491099367"
EXPECTED_LOGICAL_RECORDS_SHA256 = "37144d8ddfbbf8b6633331de612e2cdb96d07aea459db166d9ddd6197101bd09"
EXPECTED_LEGACY_4COLS_FILE_SHA256 = "581e68ce482d506a3985fcd4f8cc71c3ba349d5b021164629235d885a3476a2a"
EXPECTED_4COLS_SHA256 = EXPECTED_LEGACY_4COLS_FILE_SHA256
EXPECTED_ROWS = 3199860
EXPECTED_NULL_COUNTRY = 217369
LOCAL_GIT_FALLBACK_SPEC = "d5bb5a771282819cc1c7cbb8546f63c0d39b1fbe:data/wildchat_all_shards_4cols.parquet"


def _fetch_shard(rel_path: str, max_retries: int = 4) -> pd.DataFrame:
    dl_url = f"https://huggingface.co/datasets/{HF_DATASET_ID}/resolve/{HF_PINNED_REVISION}/{rel_path}"
    for attempt in range(max_retries):
        try:
            return pd.read_parquet(dl_url, columns=["hashed_ip", "timestamp", "model", "country"])
        except Exception:
            if attempt == max_retries - 1:
                raise
            time.sleep(1.5 * (2**attempt))
    raise RuntimeError(f"Unreachable retry failure for {rel_path}")


def compute_canonical_logical_hash(df: pd.DataFrame) -> tuple[str, dict[str, int], str, str]:
    """Compute invariant SHA-256 over deterministically sorted, schema-normalized rows."""
    null_counts = {str(k): int(v) for k, v in df.isna().sum().to_dict().items()}
    df_norm = df[["hashed_ip", "timestamp", "model", "country"]].copy()
    ts_utc = pd.to_datetime(df_norm["timestamp"], utc=True)
    min_ts = str(ts_utc.min())
    max_ts = str(ts_utc.max())
    df_norm["timestamp"] = ts_utc.dt.strftime("%Y-%m-%dT%H:%M:%S.%fZ")
    for col in ("hashed_ip", "model", "country"):
        df_norm[col] = df_norm[col].astype("string").fillna("")
    df_norm = df_norm.sort_values(["timestamp", "hashed_ip", "model", "country"], kind="mergesort").reset_index(drop=True)

    h = hashlib.sha256()
    for start in range(0, len(df_norm), 200000):
        chunk = df_norm.iloc[start : start + 200000]
        h.update(chunk.to_csv(index=False, header=(start == 0), lineterminator="\n").encode("utf-8"))
    return h.hexdigest(), null_counts, min_ts, max_ts


def load_or_fetch_4cols_parquet(
    local_cache: Path,
    download_from_hf: bool = False,
    allow_git_fallback: bool = True,
) -> tuple[bytes, str]:
    """Load the 4-column WildChat-4.8M projection from HF Hub or local cache."""
    if download_from_hf:
        api_url = f"https://huggingface.co/api/datasets/{HF_DATASET_ID}/tree/{HF_PINNED_REVISION}/data"
        req = urllib.request.Request(api_url, headers={"User-Agent": "private-llm-analytics-repro/2.0"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            entries = json.loads(resp.read().decode("utf-8"))
        parquet_files = sorted(
            e["path"] for e in entries if isinstance(e, dict) and str(e.get("path", "")).endswith(".parquet")
        )
        with ThreadPoolExecutor(max_workers=4) as pool:
            frames = list(pool.map(_fetch_shard, parquet_files))
        df_all = pd.concat(frames, ignore_index=True)
        local_cache.parent.mkdir(parents=True, exist_ok=True)
        df_all.to_parquet(local_cache, index=False)
        raw = local_cache.read_bytes()
        return raw, f"huggingface:{HF_DATASET_ID}@{HF_PINNED_REVISION} ({len(parquet_files)} shards)"

    if local_cache.exists():
        return local_cache.read_bytes(), f"local_cache:{local_cache}"

    if not allow_git_fallback:
        raise RuntimeError(
            "Local cache missing and --no-git-fallback specified; run with --download-from-hf to fetch from Hugging Face."
        )
    raw = subprocess.check_output(["git", "cat-file", "blob", LOCAL_GIT_FALLBACK_SPEC], cwd=str(ROOT))
    return raw, f"offline_preprojected_cache(sha256={EXPECTED_LEGACY_4COLS_FILE_SHA256[:12]})"


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare reproducible WildChat-4.8M manifest (v6).")
    parser.add_argument(
        "--cache-path",
        type=Path,
        default=ROOT / ".cache" / "wildchat_all_shards_4cols.parquet",
        help="Local scratch path for the 4-column parquet projection.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "results" / "sdg_mh" / "wildchat_repro_audit_v6",
        help="Directory to write manifest JSON.",
    )
    parser.add_argument(
        "--download-from-hf",
        action="store_true",
        help="Fetch directly from pinned Hugging Face Hub revision.",
    )
    parser.add_argument(
        "--no-git-fallback",
        action="store_true",
        help="Forbid falling back to local git object storage.",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw_bytes, source_mode = load_or_fetch_4cols_parquet(
        args.cache_path,
        download_from_hf=args.download_from_hf,
        allow_git_fallback=not args.no_git_fallback,
    )
    file_sha256 = hashlib.sha256(raw_bytes).hexdigest()
    df = pd.read_parquet(io.BytesIO(raw_bytes))
    logical_sha256, null_counts, min_ts, max_ts = compute_canonical_logical_hash(df)

    manifest = {
        "dataset_id": HF_DATASET_ID,
        "pinned_hf_revision": HF_PINNED_REVISION,
        "license": "ODC-By 1.0 / AI2 ImpACT Low-Risk Artifact",
        "source_mode": source_mode,
        "input_file_bytes": len(raw_bytes),
        "input_file_sha256": file_sha256,
        "legacy_blob_file_sha256": EXPECTED_LEGACY_4COLS_FILE_SHA256,
        "logical_records_sha256": logical_sha256,
        "logical_sha256_matches_expected": (logical_sha256 == EXPECTED_LOGICAL_RECORDS_SHA256),
        "row_count": int(len(df)),
        "row_count_matches_expected": (int(len(df)) == EXPECTED_ROWS),
        "columns": list(df.columns),
        "null_counts": null_counts,
        "min_timestamp_utc": min_ts,
        "max_timestamp_utc": max_ts,
        "unique_hashed_ip_count": int(df["hashed_ip"].nunique()),
        "hash_distinction_note": "input_file_sha256 hashes serialized parquet bytes (varies across writer metadata/compression); logical_records_sha256 hashes deterministically sorted canonical rows (timestamp_utc, hashed_ip, model, country) with fillna('').",
        "field_mapping": {
            "service_axis_m5": "5 OpenAI model families (gpt-4o, gpt-4o-mini, gpt-3.5-turbo, gpt-4-turbo, other-openai)",
            "category_axis_A20": "20 interaction-category proxies = 5 model_families x 4 UTC_6h_quadrants (NOT semantic NLP task topics)",
            "user_identifier": "hashed_ip (IP-derived pseudonym; NOT authenticated account ID)",
            "timestamp_semantics": "Top-level WildChat timestamp is the last turn of the conversation (NOT HTTP request timestamp)",
            "unit_of_record": "1 conversation record (NOT 1 raw HTTP API request)",
            "dataset_scope": "3,199,860-conversation public filtered release",
        },
        "dependencies": {
            "python": platform.python_version(),
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "pyarrow": pyarrow.__version__,
            "scipy": scipy.__version__,
        },
    }
    manifest_path = args.output_dir / "wildchat_public_repro_manifest_v6.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(
        f"[Manifest-v6] source={source_mode} | rows={manifest['row_count']:,} | "
        f"logical_sha256_match={manifest['logical_sha256_matches_expected']} | "
        f"file_sha256={file_sha256[:12]}..."
    )
    if not manifest["row_count_matches_expected"] or not manifest["logical_sha256_matches_expected"]:
        print(
            f"[Manifest-v6] FAIL-CLOSED: expected rows={EXPECTED_ROWS}, got {manifest['row_count']}; "
            f"expected logical_sha256={EXPECTED_LOGICAL_RECORDS_SHA256}, got {logical_sha256}",
            file=sys.stderr,
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
