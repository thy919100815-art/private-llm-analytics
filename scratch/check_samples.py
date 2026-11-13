import pandas as pd
from private_llm_analytics.aggregation.user_matrix import build_user_matrix
from private_llm_analytics.statistics.registry import StatisticProfileRegistry

for p in ['src/private_llm_analytics/web/static/sample_private_llm_analytics.csv', 'data/synthetic_sample.csv']:
    df = pd.read_csv(p)
    um = build_user_matrix(df, public_population_size=df['user_id'].nunique())
    print(f"\n=== Dataset: {p} ===")
    print(f"Total rows: {len(df)}, n_users: {um.n_users}")
    for cell in [('provider_a', 'technical'), ('provider_b', 'document')]:
        c_idx = um.cell_map[cell]
        c_name = um.cell_names[c_idx]
        sub = df[(df['provider'] == cell[0]) & (df['task'] == cell[1])]
        raw_u = sub['user_id'].nunique()
        print(f"Cell {c_name}: raw unique users = {raw_u} / {um.n_users} = {raw_u/um.n_users:.4f}")
