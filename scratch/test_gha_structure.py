import urllib.request
import gzip
import json

url = "https://data.gharchive.org/2012-01-01-12.json.gz"
req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
with urllib.request.urlopen(req, timeout=30) as resp:
    gz_data = resp.read()

with gzip.open(gzip.io.BytesIO(gz_data), "rt", encoding="utf-8", errors="ignore") as f:
    for i in range(5):
        line = f.readline()
        if not line:
            break
        obj = json.loads(line)
        actor = obj.get("actor")
        created_at = obj.get("created_at")
        print(f"Sample {i}: actor={actor}, created_at={created_at}")

url2 = "https://data.gharchive.org/2013-12-31-12.json.gz"
req2 = urllib.request.Request(url2, headers={"User-Agent": "Mozilla/5.0"})
with urllib.request.urlopen(req2, timeout=30) as resp:
    gz_data2 = resp.read()

with gzip.open(gzip.io.BytesIO(gz_data2), "rt", encoding="utf-8", errors="ignore") as f:
    for i in range(5):
        line = f.readline()
        if not line:
            break
        obj = json.loads(line)
        actor = obj.get("actor")
        created_at = obj.get("created_at")
        print(f"Sample 2013 {i}: actor={actor}, created_at={created_at}")
