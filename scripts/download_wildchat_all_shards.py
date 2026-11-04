"""Download all 86 shards of allenai/WildChat-4.8M, extracting ONLY 4 columns:
['hashed_ip', 'timestamp', 'model', 'country'].

- Uses local data/wildchat_shard_00000.parquet for shard 0.
- Downloads shards 1 to 85 concurrently with ThreadPoolExecutor.
- Immediately discards raw payload to conserve RAM and disk.
- Saves clean consolidated dataset to data/wildchat_all_shards_4cols.parquet.
- Outputs exact dataset manifest.
"""

import concurrent.futures
import io
import time
import urllib.request
from pathlib import Path
import pandas as pd
import pyarrow.parquet as pq

DATA_DIR = Path("data")
RESULTS_DIR = Path("results/sdg_mh")
DATA_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

OUT_FILE = DATA_DIR / "wildchat_all_shards_4cols.parquet"
MANIFEST_FILE = RESULTS_DIR / "wildchat_manifest.csv"

LOCAL_SHARD_0 = DATA_DIR / "wildchat_shard_00000.parquet"
BASE_URL = "https://huggingface.co/datasets/allenai/WildChat-4.8M/resolve/main/data"
TOTAL_SHARDS = 86
COLUMNS = ["hashed_ip", "timestamp", "model", "country"]

def fetch_shard(shard_idx: int):
    t0 = time.time()
    if shard_idx == 0 and LOCAL_SHARD_0.exists():
        table = pq.read_table(LOCAL_SHARD_0, columns=COLUMNS)
        df_shard = table.to_pandas()
        return {
            "shard": shard_idx,
            "status": "SUCCESS (LOCAL)",
            "rows": len(df_shard),
            "bytes": LOCAL_SHARD_0.stat().st_size,
            "elapsed": time.time() - t0,
            "df": df_shard,
        }

    file_name = f"train-{shard_idx:05d}-of-{TOTAL_SHARDS:05d}.parquet"
    url = f"{BASE_URL}/{file_name}"
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AntigravityResearch/1.0"}
    )
    
    try:
        with urllib.request.urlopen(req, timeout=180) as resp:
            content = resp.read()
        raw_bytes = len(content)
        buf = io.BytesIO(content)
        table = pq.read_table(buf, columns=COLUMNS)
        df_shard = table.to_pandas()
        del content
        del buf
        return {
            "shard": shard_idx,
            "status": "SUCCESS",
            "rows": len(df_shard),
            "bytes": raw_bytes,
            "elapsed": time.time() - t0,
            "df": df_shard,
        }
    except Exception as e:
        return {
            "shard": shard_idx,
            "status": f"ERROR: {e}",
            "rows": 0,
            "bytes": 0,
            "elapsed": time.time() - t0,
            "df": None,
        }

def main():
    start_time = time.time()
    print(f"Starting WildChat full assembly: {TOTAL_SHARDS} shards (columns: {COLUMNS})...")

    manifest_rows = []
    shard_dfs = []

    # Parallelize with 8 workers
    completed = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        future_to_shard = {executor.submit(fetch_shard, i): i for i in range(TOTAL_SHARDS)}
        for future in concurrent.futures.as_completed(future_to_shard):
            res = future.result()
            completed += 1
            if completed % 10 == 0 or completed == TOTAL_SHARDS:
                print(f"  Completed {completed}/{TOTAL_SHARDS} shards (shard {res['shard']}: {res['rows']:,} rows, {res['bytes']/(1024*1024):.1f} MB in {res['elapsed']:.1f}s)...")
            
            manifest_rows.append({
                "shard": res["shard"],
                "status": res["status"],
                "rows": res["rows"],
                "bytes": res["bytes"],
                "elapsed_sec": round(res["elapsed"], 2),
            })
            if res["df"] is not None:
                shard_dfs.append(res["df"])

    print("\nConcatenating all extracted shard dataframes...")
    df_all = pd.concat(shard_dfs, ignore_index=True)
    df_all["timestamp"] = pd.to_datetime(df_all["timestamp"], utc=True)
    df_all = df_all.sort_values("timestamp").reset_index(drop=True)

    print(f"Saving consolidated dataset to {OUT_FILE}...")
    df_all.to_parquet(OUT_FILE, index=False)

    df_manifest = pd.DataFrame(manifest_rows).sort_values("shard").reset_index(drop=True)
    df_manifest.to_csv(MANIFEST_FILE, index=False)

    total_elapsed = time.time() - start_time
    min_date = df_all["timestamp"].min()
    max_date = df_all["timestamp"].max()
    span_days = (max_date - min_date).total_seconds() / 86400.0

    daily_counts = df_all.groupby(df_all["timestamp"].dt.date).size()
    avg_daily_convs = daily_counts.mean()
    median_daily_convs = daily_counts.median()
    total_users = df_all["hashed_ip"].nunique()
    total_models = df_all["model"].nunique()
    total_countries = df_all["country"].nunique()

    print("\n" + "=" * 80)
    print("WildChat Full Dataset Assembly Completed!")
    print(f"Elapsed Time: {total_elapsed:.1f}s")
    print(f"Output File: {OUT_FILE} ({OUT_FILE.stat().st_size / (1024*1024):.2f} MB)")
    print(f"Manifest File: {MANIFEST_FILE}")
    print(f"Total Shards: {len(df_manifest[df_manifest['status'].str.startswith('SUCCESS')])}/{TOTAL_SHARDS}")
    print(f"Total Rows: {len(df_all):,}")
    print(f"Start Timestamp: {min_date}")
    print(f"End Timestamp: {max_date}")
    print(f"Time Span: {span_days:.1f} days ({len(daily_counts)} active days)")
    print(f"Average Daily Conversations: {avg_daily_convs:.1f}")
    print(f"Median Daily Conversations: {median_daily_convs:.1f}")
    print(f"Min Daily Conversations: {daily_counts.min():,}")
    print(f"Max Daily Conversations: {daily_counts.max():,}")
    print(f"Unique Hashed IPs: {total_users:,}")
    print(f"Unique Models: {total_models}")
    print(f"Unique Countries: {total_countries}")
    print("=" * 80)

if __name__ == "__main__":
    main()
