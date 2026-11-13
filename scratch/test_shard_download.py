import urllib.request
import pyarrow.parquet as pq
import io
import time

url = "https://huggingface.co/datasets/allenai/WildChat-4.8M/resolve/main/data/train-00001-of-00086.parquet"
t0 = time.time()
print(f"Downloading {url}...")
req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
with urllib.request.urlopen(req, timeout=120) as resp:
    data = resp.read()

t_dl = time.time() - t0
print(f"Downloaded {len(data)/(1024*1024):.2f} MB in {t_dl:.2f}s ({len(data)/(1024*1024)/t_dl:.2f} MB/s)")

buf = io.BytesIO(data)
table = pq.read_table(buf, columns=["hashed_ip", "timestamp", "model", "country"])
t_read = time.time() - t0
print(f"Parsed 4 columns: {len(table)} rows in {t_read:.2f}s!")
df = table.to_pandas()
print(df.head(2))
print("Timestamp range:", df["timestamp"].min(), "to", df["timestamp"].max())
