import glob
import pandas as pd

files = sorted(glob.glob('data/*.parquet')) + sorted(glob.glob('data/*.csv')) + ['src/private_llm_analytics/web/static/sample_private_llm_analytics.csv']
for p in files:
    try:
        if p.endswith('.parquet'):
            df = pd.read_parquet(p)
        else:
            df = pd.read_csv(p)
        n_users = df['user_id'].nunique() if 'user_id' in df.columns else None
        print(f"{p}: rows={len(df)}, n_users={n_users}, providers={df['provider'].unique().tolist() if 'provider' in df.columns else None}")
    except Exception as e:
        print(f"Error reading {p}: {e}")
