import urllib.request
import pyarrow.parquet as pq
import io

def inspect_url(url):
    print(f"Inspecting {url}...")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0", "Range": "bytes=0-1000000"})
    try:
        # Just download first 200KB or whole file if needed
        # Parquet footer is at the end, so let's check content-length
        req_head = urllib.request.Request(url, method="HEAD", headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req_head) as resp:
            cl = int(resp.headers.get("Content-Length", 0))
            print(f"  Content-Length: {cl}")
        # Fetch last 64KB for footer
        req_footer = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0", "Range": f"bytes={cl-65536}-{cl-1}"})
        # Wait, pyarrow might want the whole file or a file-like object supporting seek
    except Exception as e:
        print(f"  Error: {e}")

inspect_url("https://huggingface.co/datasets/allenai/WildChat/resolve/main/data/train-00000-of-00006.parquet")
inspect_url("https://huggingface.co/datasets/allenai/WildChat-1M/resolve/main/data/train-00000-of-00014.parquet")
