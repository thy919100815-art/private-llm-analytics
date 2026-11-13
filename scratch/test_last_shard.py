import urllib.request
import pyarrow.parquet as pq
import io

url = "https://huggingface.co/datasets/allenai/WildChat-4.8M/resolve/main/data/train-00085-of-00086.parquet"
req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
print("Downloading last shard 85...")
with urllib.request.urlopen(req, timeout=120) as resp:
    data = resp.read()

table = pq.read_table(io.BytesIO(data), columns=["hashed_ip", "timestamp", "model", "country"])
df = table.to_pandas()
print("Shard 85 head:")
print(df.head(2))
print("Timestamp range:", df["timestamp"].min(), "to", df["timestamp"].max())
print("Rows:", len(df))
