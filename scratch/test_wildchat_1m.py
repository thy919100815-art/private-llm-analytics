import pandas as pd
import urllib.request
import pyarrow.parquet as pq
import io

df_local = pd.read_parquet("data/wildchat_shard_00000.parquet", columns=["conversation_hash", "model", "timestamp", "hashed_ip", "country"])
print("Local shard 0 head:")
print(df_local.head(3))
print("Total rows:", len(df_local))

url = "https://huggingface.co/datasets/allenai/WildChat-1M/resolve/main/data/train-00000-of-00014.parquet"
req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
print("Fetching first part of train-00000-of-00014.parquet...")
# Let's read just the schema or check row count via parquet metadata
# We can download shard 0 or stream it
