import urllib.request
import json

for ds in ['allenai/WildChat', 'allenai/WildChat-1M']:
    try:
        url = f'https://huggingface.co/api/datasets/{ds}'
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req) as resp:
            data = json.loads(resp.read().decode())
            print(f'{ds} exists!')
            files = [s['rfilename'] for s in data.get('siblings', []) if s['rfilename'].endswith('.parquet')]
            print(f'  Parquet files in {ds} (total {len(files)}): {files[:8]}')
    except Exception as e:
        print(f'{ds} error: {e}')
