"""Retry failed WildChat shards with retries to reach 86/86 shards."""

import io
import time
import urllib.request
from pathlib import Path
import pandas as pd
import pyarrow.parquet as pq

DATA_DIR = Path("data")
RESULTS_DIR = Path("results/sdg_mh")
OUT_FILE = DATA_DIR / "wildchat_all_shards_4cols.parquet"
MANIFEST_FILE = RESULTS_DIR / "wildchat_manifest.csv"
BASE_URL = "https://huggingface.co/datasets/allenai/WildChat-4.8M/resolve/main/data"
TOTAL_SHARDS = 86
COLUMNS = ["hashed_ip", "timestamp", "model", "country"]

def retry_missing():
    df_manifest = pd.read_csv(MANIFEST_FILE)
    failed_shards = df_manifest[~df_manifest["status"].str.startswith("SUCCESS")]["shard"].tolist()
    print(f"Retrying {len(failed_shards)} failed shards: {failed_shards}...")

    df_existing = pd.read_parquet(OUT_FILE)
    recovered_dfs = []

    for shard_idx in failed_shards:
        file_name = f"train-{shard_idx:05d}-of-{TOTAL_SHARDS:05d}.parquet"
        url = f"{BASE_URL}/{file_name}"
        success = False
        for attempt in range(3):
            try:
                t0 = time.time()
                print(f"Downloading shard {shard_idx} (attempt {attempt+1}/3)...")
                req = urllib.request.Request(
                    url,
                    headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AntigravityResearch/1.0"}
                )
                with urllib.request.urlopen(req, timeout=300) as resp:
                    content = resp.read()
                raw_bytes = len(content)
                table = pq.read_table(io.BytesIO(content), columns=COLUMNS)
                df_shard = table.to_pandas()
                recovered_dfs.append(df_shard)
                elapsed = time.time() - t0
                print(f"  Shard {shard_idx} SUCCESS: {len(df_shard):,} rows in {elapsed:.1f}s.")

                # Update manifest
                mask = df_manifest["shard"] == shard_idx
                df_manifest.loc[mask, "status"] = "SUCCESS"
                df_manifest.loc[mask, "rows"] = len(df_shard)
                df_manifest.loc[mask, "bytes"] = raw_bytes
                df_manifest.loc[mask, "elapsed_sec"] = round(elapsed, 2)
                success = True
                break
            except Exception as e:
                print(f"  Attempt {attempt+1} failed for shard {shard_idx}: {e}")
                time.sleep(2)

        if not success:
            print(f"Shard {shard_idx} permanently failed after 3 attempts.")

    if recovered_dfs:
        print(f"Combining {len(recovered_dfs)} recovered shards with existing data...")
        all_dfs = [df_existing] + recovered_dfs
        df_combined = pd.concat(all_dfs, ignore_index=True)
        df_combined["timestamp"] = pd.to_datetime(df_combined["timestamp"], utc=True)
        df_combined = df_combined.sort_values("timestamp").reset_index(drop=True)
        df_combined.to_parquet(OUT_FILE, index=False)
        df_manifest.to_csv(MANIFEST_FILE, index=False)
        print(f"Updated {OUT_FILE}: {len(df_combined):,} rows total!")
        print(f"Updated {MANIFEST_FILE}: {len(df_manifest[df_manifest['status'].str.startswith('SUCCESS')])}/{TOTAL_SHARDS} shards success!")

if __name__ == "__main__":
    retry_missing()
