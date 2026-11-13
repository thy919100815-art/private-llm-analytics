import pandas as pd
df = pd.read_parquet('data/interactions_n50000.parquet')
valid_ts = pd.to_datetime(df['timestamp'], errors='coerce').dropna()
t_max = valid_ts.max()
print('t_max:', t_max)
for days in [7, 14, 30]:
    t_start = t_max.floor('D') - pd.Timedelta(days=days - 1)
    df_w = df[(valid_ts >= t_start) & (valid_ts <= t_max)]
    df_pat = df_w[(df_w['provider'] == 'provider_a') & (df_w['task'] == 'technical')]
    print(f"Window {days} days: rows={len(df_w)}, total_users={df_w['user_id'].nunique()}, pat_users={df_pat['user_id'].nunique()}, pat_rate={df_pat['user_id'].nunique()/50000:.4f}")
