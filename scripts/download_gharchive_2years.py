"""Stream-download and aggregate GH Archive daily 12:00 UTC archives across 2 full years:
2012-01-01 to 2013-12-31 (731 days).

Aggregates activity counts by (date, actor) to produce user x day event stream.
Saves:
- data/gharchive_2years_daily.parquet
- results/sdg_mh/gharchive_manifest.csv
"""

import concurrent.futures
import datetime
import gzip
import io
import json
import os
import time
import urllib.request
from pathlib import Path
import pandas as pd

DATA_DIR = Path("data")
RESULTS_DIR = Path("results/sdg_mh")
DATA_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

OUT_FILE = DATA_DIR / "gharchive_2years_daily.parquet"
MANIFEST_FILE = RESULTS_DIR / "gharchive_manifest.csv"

START_DATE = datetime.date(2012, 1, 1)
END_DATE = datetime.date(2013, 12, 31)

def get_actor_login(actor_field):
    if isinstance(actor_field, str):
        return actor_field
    elif isinstance(actor_field, dict):
        return actor_field.get("login") or actor_field.get("name") or "unknown"
    return "unknown"

def process_single_day(d: datetime.date):
    date_str = d.strftime("%Y-%m-%d")
    url = f"https://data.gharchive.org/{date_str}-12.json.gz"
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AntigravityResearch/1.0"}
    )
    
    actor_counts = {}
    file_bytes = 0
    event_count = 0

    try:
        with urllib.request.urlopen(req, timeout=45) as resp:
            gz_bytes = resp.read()
        file_bytes = len(gz_bytes)

        with gzip.open(io.BytesIO(gz_bytes), "rt", encoding="utf-8", errors="ignore") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    obj = json.loads(line)
                    actor = get_actor_login(obj.get("actor"))
                    if actor and actor != "unknown":
                        actor_counts[actor] = actor_counts.get(actor, 0) + 1
                        event_count += 1
                except Exception:
                    continue

        return {
            "date": date_str,
            "status": "SUCCESS",
            "file_bytes": file_bytes,
            "events": event_count,
            "unique_actors": len(actor_counts),
            "actor_counts": actor_counts,
        }
    except Exception as e:
        return {
            "date": date_str,
            "status": f"ERROR: {e}",
            "file_bytes": 0,
            "events": 0,
            "unique_actors": 0,
            "actor_counts": {},
        }

def main():
    start_time = time.time()
    num_days = (END_DATE - START_DATE).days + 1
    dates = [START_DATE + datetime.timedelta(days=i) for i in range(num_days)]
    print(f"Starting GH Archive processing for {num_days} days: {START_DATE} to {END_DATE}...")

    manifest_rows = []
    user_day_records = []

    # Use ThreadPoolExecutor to parallelize download and decompression
    max_workers = 16
    completed = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_date = {executor.submit(process_single_day, d): d for d in dates}
        for future in concurrent.futures.as_completed(future_to_date):
            res = future.result()
            completed += 1
            if completed % 50 == 0 or completed == num_days:
                print(f"  Processed {completed}/{num_days} days ({res['date']}: {res['events']} events, {res['file_bytes'] / 1024:.1f} KB)...")

            manifest_rows.append({
                "date": res["date"],
                "status": res["status"],
                "file_bytes": res["file_bytes"],
                "events": res["events"],
                "unique_actors": res["unique_actors"],
            })

            d_str = res["date"]
            for actor, cnt in res["actor_counts"].items():
                user_day_records.append({
                    "date": d_str,
                    "actor": actor,
                    "event_count": cnt,
                })

    # Save manifest
    df_manifest = pd.DataFrame(manifest_rows).sort_values("date").reset_index(drop=True)
    df_manifest.to_csv(MANIFEST_FILE, index=False)

    # Save user x day activity dataset
    df_users = pd.DataFrame(user_day_records)
    df_users["date"] = pd.to_datetime(df_users["date"])
    df_users = df_users.sort_values(["date", "actor"]).reset_index(drop=True)
    df_users.to_parquet(OUT_FILE, index=False)

    elapsed = time.time() - start_time
    total_events = df_manifest["events"].sum()
    total_files = len(df_manifest[df_manifest["status"] == "SUCCESS"])
    total_mb = df_manifest["file_bytes"].sum() / (1024 * 1024)
    unique_devs = df_users["actor"].nunique()

    print("\n" + "=" * 80)
    print(f"GH Archive 2-Year Dataset Assembly Completed in {elapsed:.1f}s!")
    print(f"Output File: {OUT_FILE} ({OUT_FILE.stat().st_size / (1024*1024):.2f} MB)")
    print(f"Manifest File: {MANIFEST_FILE}")
    print(f"Successful Files: {total_files}/{num_days}")
    print(f"Total Downloaded Size: {total_mb:.2f} MB")
    print(f"Total Events: {total_events:,}")
    print(f"Total Unique Developers: {unique_devs:,}")
    print(f"User x Day Rows: {len(df_users):,}")
    print(f"Average Daily Events: {total_events / num_days:.1f}")
    print(f"Date Range: {df_manifest['date'].min()} to {df_manifest['date'].max()}")
    print("=" * 80)

if __name__ == "__main__":
    main()
